"""DeltaMemory: a delta-rule associative memory running BESIDE attention.

WHY (2026-08-20). Every memory mechanism in this repo so far is a *bank* -- a set
of slots that attention reads as extra keys. Banks cost per carried layer and,
for scribes, grow with S. The delta rule is a different answer to the same
question: an associative memory held as a MATRIX state, written and read at O(1)
per token with no slots, no selection and no extra keys in the softmax.

    S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

That recurrence is exactly "write value v_t at key k_t, overwriting whatever was
stored at that key with strength beta_t" -- the delta rule of Widrow-Hoff, which
is why DeltaNet-family models behave like key-value memory rather than the
leaky running average a plain linear-attention state gives you.

This module is deliberately NOT a layer replacement. `use_deltanet` already
swaps whole attention layers for GatedDeltaNet (the 3:1 hybrid in
PRETRAIN_PLAN.md). Here the memory runs as a narrow side branch on top of an
unchanged attention stack, with a zero-init output projection, so the arm starts
bitwise-identical to its backbone and can only earn its way in. That makes it
composable with the bank arms instead of competing with them for the layer slot.

COST. It is sized by `delta_mem_heads` x `delta_mem_head_dim` (default 4 x 64 =
256 wide, against a 896-wide backbone) and runs only every
`delta_mem_interval`-th layer, because our own measurements price a per-layer
mechanism at roughly its layer count -- routing every layer is what made mopr
expensive. Chunked, the cost is O(S * chunk * d), i.e. linear in S: unlike a
bank it does not get more expensive at 32K, and unlike block-sparse attention it
needs no selection.

INTRA-DOC CORRECTNESS. A recurrent state silently carries information across
document boundaries in a packed batch, which is the exact bug class that had to
be fixed for Griffin/Mamba2/Hymba. Handled here at chunk granularity: a token
may read the incoming state only if it belongs to the same document as the end
of the previous chunk, and only tokens in the chunk's final document contribute
to the state carried out. Both masks are exact, not approximations.

NUMERICS. q and k are L2-normalised (standard DeltaNet practice -- it bounds the
state), the UT transform is solved in fp32, and the output is cast back. The
chunked form below is the closed-form solution of the recurrence, not an
approximation of it: `test_delta_memory_chunked_matches_recurrent` checks it
against a literal token-by-token loop.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _chunk_masks(chunk, device, dtype):
    """(strict_tril, incl_tril, eye) as [C,C], built once per call rather than
    once per chunk. They were being reallocated inside the sequential loop."""
    ones = torch.ones(chunk, chunk, device=device, dtype=dtype)
    return torch.tril(ones, -1), torch.tril(ones, 0), torch.eye(chunk, device=device, dtype=dtype)


def _chunk_view(t, nc, chunk):
    return t.view(t.shape[0], t.shape[1], nc, chunk, -1)


def delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 64,
    doc_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Closed-form chunked delta rule. q,k,v: [B,H,S,D]; beta: [B,H,S]. -> [B,H,S,D].

    Derivation (state S in R^{D x D}, o_t = S_t q_t):
        S_t = S_{t-1} + u_t k_t^T,  u_t = beta_t (v_t - S_{t-1} k_t)
    so within a chunk, with L = strict_tril(K K^T),
        (I + diag(beta) L) U = diag(beta) (V - K S_0^T)
        O = Q S_0^T + tril_inclusive(Q K^T) U
        S_C = S_0 + U^T K

    STATE-FACTORED FORM (2026-08-21 optimization pass). The loop above is
    written so that everything except two small matmuls happens OUTSIDE it.
    Because the solve is linear in its right-hand side, U splits into a part
    that knows nothing about the incoming state and a part that is linear in it:

        T  = M^{-1}                       (M unit lower triangular, per chunk)
        W  = T (diag(beta) V)             state-free
        Kb = T (diag(beta * carry) K)     state-free
        U  = W - Kb S_0^T

    substituting into the two consumers of U gives

        O   = (Q*carry - A Kb) S_0^T + A W
        S_C = S_0 * sk - S_0 G + Wk,      G = (Kb*keep)^T K,  Wk = (W*keep)^T K

    so T, W, Kb, A, L, G and Wk are computed for ALL chunks in one batched call
    each, and the sequential chain shrinks to one [C,D]x[D,D] matmul plus one
    [D,D]x[D,D] matmul per chunk. That is 2 kernel launches per chunk where the
    literal form needed ~10, on a module that gate T measured at 4.8x dense
    with a 23% median/min spread -- i.e. host-dispatch bound, not arithmetic
    bound. The arithmetic is unchanged: this is an algebraic regrouping, and
    ``test_delta_memory_chunked_matches_recurrent`` checks it against a literal
    token-by-token loop in fp64.
    """
    b, h, s, d = q.shape
    nc = (s + chunk - 1) // chunk
    pad = nc * chunk - s
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta = F.pad(beta, (0, pad))
        if doc_ids is not None:
            doc_ids = F.pad(doc_ids, (0, pad), value=-1)

    acc = q.dtype if q.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(acc), dim=-1)
    k = F.normalize(k.to(acc), dim=-1)
    v = v.to(acc)
    beta = beta.to(acc)

    Q = q.view(b, h, nc, chunk, d)
    K = k.view(b, h, nc, chunk, d)
    V = v.view(b, h, nc, chunk, d)
    Bt = beta.view(b, h, nc, chunk, 1)
    dc = doc_ids.view(b, nc, chunk) if doc_ids is not None else None

    strict, incl, eye = _chunk_masks(chunk, q.device, q.dtype)

    if dc is not None:
        same = (dc.unsqueeze(-1) == dc.unsqueeze(-2)).unsqueeze(1).to(q.dtype)  # [B,1,nc,C,C]
        # carry[b,c,t]: may token t of chunk c read the state entering chunk c?
        # Only if it is in the same document as the LAST token of chunk c-1.
        prev_last = dc[:, :-1, -1]  # [B,nc-1]
        carry = torch.zeros(b, nc, chunk, device=q.device, dtype=q.dtype)
        carry[:, 1:] = (dc[:, 1:] == prev_last.unsqueeze(-1)).to(q.dtype)
        carry = carry.view(b, 1, nc, chunk, 1)
        # keep[b,c,t]: does token t contribute to the state carried OUT of c?
        keep = (dc == dc[:, :, -1:]).to(q.dtype).view(b, 1, nc, chunk, 1)
        # sk[b,c]: does the state entering chunk c survive it? (Only if the
        # chunk's last document is the one that was already running.)
        sk = torch.zeros(b, nc, device=q.device, dtype=q.dtype)
        sk[:, 1:] = (dc[:, 1:, -1] == prev_last).to(q.dtype)
        sk = sk.view(b, 1, nc, 1, 1)
    else:
        same, carry, keep = None, 1.0, 1.0
        sk = 1.0

    Kt_ = K.transpose(-1, -2)
    L = (K @ Kt_) * strict
    A = (Q @ Kt_) * incl
    if same is not None:
        L = L * same
        A = A * same

    M = eye + Bt * L  # unit lower triangular, [B,H,nc,C,C]
    # One batched inverse instead of nc batched solves. `unitriangular` lets
    # cuBLAS skip the diagonal entirely.
    T = torch.linalg.solve_triangular(
        M, eye.expand(b, h, nc, chunk, chunk), upper=False, unitriangular=True
    )
    W = T @ (Bt * V)
    Kb = T @ ((Bt * carry) * K)
    AW = A @ W
    X = Q * carry - A @ Kb

    Wk = (W * keep).transpose(-1, -2) @ K  # [B,H,nc,Dv,Dk]
    G = (Kb * keep).transpose(-1, -2) @ K  # [B,H,nc,Dk,Dk]

    state = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)  # [B,H,Dv,Dk]
    sk_c = sk if isinstance(sk, float) else sk.unbind(2)
    outs = []
    for c in range(nc):
        outs.append(X[:, :, c] @ state.transpose(-1, -2) + AW[:, :, c])
        s_c = sk if isinstance(sk, float) else sk_c[c]
        state = state * s_c - state @ G[:, :, c] + Wk[:, :, c]

    out = torch.stack(outs, dim=2).reshape(b, h, nc * chunk, d)
    return out[:, :, :s] if pad else out


def delta_rule_recurrent(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Literal token-by-token recurrence. Reference for the chunked form only."""
    b, h, s, d = q.shape
    acc = q.dtype if q.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(acc), dim=-1)
    k = F.normalize(k.to(acc), dim=-1)
    v, beta = v.to(acc), beta.to(acc)
    S = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    out = []
    for t in range(s):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t].unsqueeze(-1)
        u = bt * (vt - torch.einsum("bhvk,bhk->bhv", S, kt))
        S = S + u.unsqueeze(-1) * kt.unsqueeze(-2)
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, :, t]))
    return torch.stack(out, dim=2)


def _decay_matrix(logg, chunk, same=None):
    """D[t,j] = prod_{j<i<=t} gamma_i for j <= t, else 0. Built in LOG space.

    ``logg`` is the INCLUSIVE cumulative log-decay within the chunk, so
    ``logg[t] - logg[j]`` is the product of the decays strictly after j up to t.
    Every gamma is in (0,1], hence the exponent is <= 0 for t >= j and the exp
    cannot overflow -- which is the reason this is not computed as a ratio of
    cumprods (that form underflows to 0/0 for long chunks and small gammas).
    """
    d = logg.unsqueeze(-1) - logg.unsqueeze(-2)  # [..., C, C]
    tri = torch.ones(chunk, chunk, device=logg.device, dtype=torch.bool).tril()
    m = tri if same is None else (tri & same)
    return torch.where(m, d, torch.full_like(d, float("-inf"))).exp()


def gla_chunked(q, k, v, gamma, chunk=64, doc_ids=None):
    """Gated linear attention: S_t = gamma_t * S_{t-1} + v_t k_t^T, o_t = S_t q_t.

    The LIGHT alternative to the delta rule. Same matrix-valued state and the
    same O(S*chunk*D) cost shape, but the write is a plain accumulate instead of
    an overwrite, so there is NO triangular solve -- the whole chunk is two
    matmuls. What it gives up is exactly the delta rule's defining property: it
    cannot erase what is stored at a key, only decay everything uniformly. That
    is the comparison this variant exists to make.

    Same state-factored loop as ``delta_rule_chunked``: the intra-chunk term and
    the state contribution are state-free, so both are computed for all chunks
    at once and the sequential chain is one matmul plus one scaled add.
    """
    b, h, s, d = q.shape
    nc = (s + chunk - 1) // chunk
    pad = nc * chunk - s
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        gamma = F.pad(gamma, (0, pad), value=1.0)
        if doc_ids is not None:
            doc_ids = F.pad(doc_ids, (0, pad), value=-1)
    acc = q.dtype if q.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(acc), dim=-1)
    k = F.normalize(k.to(acc), dim=-1)
    v, gamma = v.to(acc), gamma.to(acc).clamp(min=1e-6, max=1.0)
    Q = q.view(b, h, nc, chunk, d)
    K = k.view(b, h, nc, chunk, d)
    V = v.view(b, h, nc, chunk, d)
    logg = gamma.log().view(b, h, nc, chunk).cumsum(-1)  # inclusive, per chunk
    dc = doc_ids.view(b, nc, chunk) if doc_ids is not None else None

    if dc is not None:
        same = (dc.unsqueeze(-1) == dc.unsqueeze(-2)).unsqueeze(1)  # [B,1,nc,C,C] bool
        prev_last = dc[:, :-1, -1]
        carry = torch.zeros(b, nc, chunk, device=q.device, dtype=q.dtype)
        carry[:, 1:] = (dc[:, 1:] == prev_last.unsqueeze(-1)).to(q.dtype)
        carry = carry.view(b, 1, nc, chunk, 1)
        keep = (dc == dc[:, :, -1:]).to(q.dtype).view(b, 1, nc, chunk, 1)
        sk = torch.zeros(b, nc, device=q.device, dtype=q.dtype)
        sk[:, 1:] = (dc[:, 1:, -1] == prev_last).to(q.dtype)
        sk = sk.view(b, 1, nc, 1, 1)
    else:
        same, carry, keep, sk = None, 1.0, 1.0, 1.0

    D = _decay_matrix(logg, chunk, same)
    intra = ((Q @ K.transpose(-1, -2)) * D) @ V
    Xq = logg.exp().unsqueeze(-1) * Q * carry
    gC = logg[..., -1:]  # [B,H,nc,1]
    w = (gC - logg).exp().unsqueeze(-1)
    contrib = (w * keep * V).transpose(-1, -2) @ K  # [B,H,nc,Dv,Dk]
    decay = gC.exp().view(b, h, nc, 1, 1)

    state = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    outs = []
    for c in range(nc):
        outs.append(Xq[:, :, c] @ state.transpose(-1, -2) + intra[:, :, c])
        s_c = 1.0 if isinstance(sk, float) else sk[:, :, c]
        state = state * decay[:, :, c] * s_c + contrib[:, :, c]
    out = torch.stack(outs, dim=2).reshape(b, h, nc * chunk, d)
    return out[:, :, :s] if pad else out


def ema_chunked(v, gamma, chunk=64, doc_ids=None):
    """Cheapest possible persistent channel: a per-head EMA of values.

    S_t = gamma_t S_{t-1} + (1 - gamma_t) v_t, read as o_t = S_t. The state is a
    VECTOR, not a matrix, so this is O(S*chunk*D) with no outer product and no
    query -- strictly cheaper than either matrix-state variant.

    It is deliberately NOT associative: there is no key, so nothing can be
    recalled BY content, only summarised. That makes it the control that
    separates "the arm needed associative recall" from "the arm needed any
    long-range channel at all" -- a distinction neither delta nor gla can
    establish on its own.

    State-factored like the other two; here the sequential chain is pure
    elementwise -- no matmul at all.
    """
    b, h, s, d = v.shape
    nc = (s + chunk - 1) // chunk
    pad = nc * chunk - s
    if pad:
        v = F.pad(v, (0, 0, 0, pad))
        gamma = F.pad(gamma, (0, pad), value=1.0)
        if doc_ids is not None:
            doc_ids = F.pad(doc_ids, (0, pad), value=-1)
    acc = v.dtype if v.dtype == torch.float64 else torch.float32
    v, gamma = v.to(acc), gamma.to(acc).clamp(min=1e-6, max=1.0)
    V = v.view(b, h, nc, chunk, d)
    gc = gamma.view(b, h, nc, chunk)
    logg = gc.log().cumsum(-1)
    dc = doc_ids.view(b, nc, chunk) if doc_ids is not None else None

    if dc is not None:
        same = (dc.unsqueeze(-1) == dc.unsqueeze(-2)).unsqueeze(1)
        prev_last = dc[:, :-1, -1]
        carry = torch.zeros(b, nc, chunk, device=v.device, dtype=v.dtype)
        carry[:, 1:] = (dc[:, 1:] == prev_last.unsqueeze(-1)).to(v.dtype)
        carry = carry.view(b, 1, nc, chunk, 1)
        keep = (dc == dc[:, :, -1:]).to(v.dtype).view(b, 1, nc, chunk, 1)
        sk = torch.zeros(b, nc, device=v.device, dtype=v.dtype)
        sk[:, 1:] = (dc[:, 1:, -1] == prev_last).to(v.dtype)
        sk = sk.view(b, 1, nc, 1)
    else:
        same, carry, keep, sk = None, 1.0, 1.0, 1.0

    D = _decay_matrix(logg, chunk, same)
    src = (1.0 - gc).unsqueeze(-1) * V
    intra = D @ src
    rowscale = logg.exp().unsqueeze(-1) * carry  # [B,H,nc,C,1]
    gC = logg[..., -1:]
    w = (gC - logg).exp().unsqueeze(-1)
    contrib = (w * keep * src).sum(3)  # [B,H,nc,Dv]
    decay = gC.exp()  # [B,H,nc,1]

    state = torch.zeros(b, h, d, device=v.device, dtype=v.dtype)
    outs = []
    for c in range(nc):
        outs.append(rowscale[:, :, c] * state.unsqueeze(2) + intra[:, :, c])
        s_c = 1.0 if isinstance(sk, float) else sk[:, :, c]
        state = state * decay[:, :, c] * s_c + contrib[:, :, c]
    out = torch.stack(outs, dim=2).reshape(b, h, nc * chunk, d)
    return out[:, :, :s] if pad else out


def gla_recurrent(q, k, v, gamma):
    """Literal recurrence for ``gla_chunked``. Reference only."""
    b, h, s, d = q.shape
    acc = q.dtype if q.dtype == torch.float64 else torch.float32
    q = F.normalize(q.to(acc), dim=-1)
    k = F.normalize(k.to(acc), dim=-1)
    v, gamma = v.to(acc), gamma.to(acc).clamp(min=1e-6, max=1.0)
    S = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    out = []
    for t in range(s):
        S = gamma[:, :, t, None, None] * S + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2)
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, :, t]))
    return torch.stack(out, dim=2)


def ema_recurrent(v, gamma):
    """Literal recurrence for ``ema_chunked``. Reference only."""
    b, h, s, d = v.shape
    acc = v.dtype if v.dtype == torch.float64 else torch.float32
    v, gamma = v.to(acc), gamma.to(acc).clamp(min=1e-6, max=1.0)
    S = torch.zeros(b, h, d, device=v.device, dtype=v.dtype)
    out = []
    for t in range(s):
        g = gamma[:, :, t].unsqueeze(-1)
        S = g * S + (1.0 - g) * v[:, :, t]
        out.append(S)
    return torch.stack(out, dim=2)


class DeltaMemory(nn.Module):
    """Narrow delta-rule associative memory, added to the residual stream."""

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.variant = str(getattr(config, "delta_mem_variant", "delta"))
        self.heads = int(getattr(config, "delta_mem_heads", 4))
        self.hdim = int(getattr(config, "delta_mem_head_dim", 64))
        self.chunk = int(getattr(config, "delta_mem_chunk", 64))
        inner = self.heads * self.hdim
        self.q_proj = nn.Linear(h, inner, bias=False)
        self.k_proj = nn.Linear(h, inner, bias=False)
        self.v_proj = nn.Linear(h, inner, bias=False)
        self.b_proj = nn.Linear(h, self.heads, bias=True)
        self.o_proj = nn.Linear(inner, h, bias=False)
        from .rmsnorm import get_rmsnorm_class

        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.norm = norm_class(h, eps=getattr(config, "rms_norm_eps", 1e-6))
        # Zero-init output: the arm starts identical to its backbone, so any
        # difference in the loss curve is the memory earning its place rather
        # than a different initialisation.
        nn.init.zeros_(self.o_proj.weight)
        # These three inits are MEANINGFUL, and HF re-runs `_init_weights` over
        # every nn.Linear after construction and after every checkpoint load --
        # which silently undid all of them until 2026-08-21 (measured: o_proj
        # absmax 0.087 instead of 0, b_proj.bias 0.0 instead of 2.0, i.e. the
        # arm did NOT start as its backbone and ran at beta=0.5). The markers
        # below are what make them survive; do not remove them.
        self.o_proj._zero_init_weight = True
        # beta near 1 at init = "overwrite fully", the standard DeltaNet start.
        nn.init.zeros_(self.b_proj.weight)
        nn.init.constant_(self.b_proj.bias, 2.0)
        self.b_proj._zero_init_weight = True
        self.b_proj._const_init_bias = 2.0

    def forward(
        self, hidden_states: torch.Tensor, doc_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        x = self.norm(hidden_states)
        shape = (b, s, self.heads, self.hdim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)  # [B,H,S]
        if self.variant == "delta":
            o = delta_rule_chunked(q, k, v, beta, chunk=self.chunk, doc_ids=doc_ids)
        elif self.variant == "gla":
            # beta plays the role of the DECAY here, not the write strength.
            # Reusing the same projection keeps the three variants matched in
            # parameters, so a difference between them is the recurrence and not
            # the parameter budget.
            o = gla_chunked(q, k, v, beta, chunk=self.chunk, doc_ids=doc_ids)
        elif self.variant == "ema":
            o = ema_chunked(v, beta, chunk=self.chunk, doc_ids=doc_ids)
        else:  # pragma: no cover - the config validator rejects anything else
            raise ValueError(f"unknown delta_mem_variant {self.variant!r}")
        o = o.transpose(1, 2).reshape(b, s, self.heads * self.hdim)
        return self.o_proj(o.to(hidden_states.dtype))
