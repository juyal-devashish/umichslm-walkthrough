"""Loop-boundary mixers for router-free recurrent depth (``use_recurrent_depth``).

The plain ``_rec`` loop hands each core iteration the previous iteration's
output and nothing else. Two COLM 2026 papers each fix one weakness of that,
and this module implements both, independently switchable and composable:

**Stable injection** (Parcae, arXiv 2604.12946, ``recurrent_stable_injection``).
The prelude output is normalized once, ``e = RMSNorm(prelude(x))``, and every
iteration's core input is

    x_t = A_bar * h_t + B_bar e,        h_{t+1} = core(x_t)

with ``A = -exp(log_A)`` (negative diagonal), ``A_bar = exp(dt * A)`` (zero-order
hold) and ``B_bar = dt * B`` (Euler), ``dt = softplus(dt_raw) > 0``. Because the
core keeps its own residuals, ``core(x_t) = x_t + R(x_t)`` is exactly the
paper's ``h_{t+1} = A_bar h_t + B_bar e + R(h_t, e)``. ``A_bar`` lies in (0, 1) per
channel, so the linear part is a contraction (spectral radius < 1): this is the
paper's fix for the instability that plain addition (rho = 1) and concatenation
(rho > 1) show when a loop runs past its training depth. ``h_0`` is Gaussian
noise in training (paper Alg. 1; std sqrt(2 / (5 d)), the Geiping "like-init"
scale the paper cites) and zeros at eval, so eval losses are deterministic
(0.04 at d=256, against an injected e of unit RMS scaled by dt >= 0.1).
Not specified by the paper and chosen here: ``log_A = 0`` and ``dt`` log-uniform
in [``dt_min``, ``dt_max``] = [0.1, 1.0] per channel (A_bar in [0.37, 0.90], a
spread of memory timescales as in Mamba), ``B = I``. With B = I and small dt,
``B_bar ~ 1 - A_bar``, so the core input starts as a per-channel blend of the
state and the prompt encoding. ``B`` is dense by default (paper Table 1:
d_h x d_e, "unconstrained"); ``recurrent_inject_dense_b=False`` makes it a
diagonal. The paper's coda projection C only decouples widths and is the
identity here.

**Loop hyper-connections** (Hyperloop, arXiv 2604.21254,
``recurrent_hyper_connections``). The residual stream is widened to ``n``
streams (copies of the core's first input) and, once per iteration ``l``:

    z = RMSNorm(flatten(y))
    H_pre  = sigmoid(a_pre  * W_pre  z + b_pre)          # 1 x n
    H_post = 2 sigmoid(a_post * W_post z + b_post)       # n x 1
    H_res  = diag(sigmoid(a_res * W_res z + b_res))      # n x n, diagonal
    y <- H_res y + H_post (core(H_pre y) + e_l)

with loop-specific weights and a loop position embedding ``e_l`` added AFTER
the core (paper: "after the middle block"); the streams are averaged before the
coda. Iterations past ``recurrent_max_iters`` (the eval depth sweep runs to 2x)
reuse the last loop's parameters. Initialization is unspecified in the paper and
chosen here to START AS THE PLAIN LOOP: H_pre = 1/n (reads the stream mean),
H_post = 1, H_res = sigmoid(-3) ~ 0.047 (the carried stream is nearly dropped,
so the readout ~ the core output), e_l = 0, gate scales a = 0.01. The twin
therefore begins as its ``_rec`` control and departs only where gradients pay,
which keeps the HC-vs-plain comparison about what the mechanism learns.
``recurrent_hc_res_bias`` moves the H_res start (0 gives the paper-ish 0.5).

**Both on** (``_rec_inject_hc``): the hyper-connection read is the state that
gets injected, ``x_t = A_bar * (H_pre y_t) + B_bar e``, and the streams start at
``h_0``. Stability then comes from A_bar on the core input and from H_res < 1 on
the carried streams.

Cost. Everything here runs at model level, once per iteration, outside the
decoder layers' gradient checkpoints. Matmul FLOPs are ``2 d^2`` (dense B) and
``6 n^2 d`` (the three gate projections) per token per iteration -- billed via
``compute_accounting.recurrent_mixer_param_delta`` -- and the rest is
elementwise. Activation memory adds O(n d) per token per iteration, small next
to the logits at screening sizes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import get_rmsnorm_class


class StableLoopInjection(nn.Module):
    """Parcae's negative-diagonal, ZOH-discretized input injection."""

    def __init__(self, config):
        super().__init__()
        d = int(config.hidden_size)
        norm_cls = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        # "prelude norm" (paper App. J): e = LN(P(s)).
        self.prelude_norm = norm_cls(d, eps=config.rms_norm_eps)
        self.log_A = nn.Parameter(torch.zeros(d))
        dt_min = float(getattr(config, "recurrent_inject_dt_min", 0.1))
        dt_max = float(getattr(config, "recurrent_inject_dt_max", 1.0))
        dt = torch.exp(torch.empty(d).uniform_(math.log(dt_min), math.log(dt_max)))
        # Inverse softplus, so softplus(dt_raw) == dt at init.
        self.dt_raw = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dense_b = bool(getattr(config, "recurrent_inject_dense_b", True))
        # Raw Parameters, not nn.Linear: _init_weights re-draws every Linear
        # weight from N(0, initializer_range), which would erase B = I.
        self.B = nn.Parameter(torch.eye(d) if self.dense_b else torch.ones(d))
        self.h0_std = math.sqrt(2.0 / (5.0 * d))

    def encode(self, prelude_out: torch.Tensor) -> torch.Tensor:
        return self.prelude_norm(prelude_out)

    def initial_state(self, like: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.randn_like(like) * self.h0_std
        return torch.zeros_like(like)

    def decay(self) -> torch.Tensor:
        """(A_bar, dt) in fp32; A_bar in (0, 1) by construction."""
        dt = F.softplus(self.dt_raw.float())
        return torch.exp(-dt * torch.exp(self.log_A.float())), dt

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        a_bar, dt = self.decay()
        be = F.linear(e, self.B.to(e.dtype)) if self.dense_b else e * self.B.to(e.dtype)
        return (a_bar * h.float() + dt * be.float()).to(h.dtype)


class LoopHyperConnection(nn.Module):
    """Hyperloop's per-iteration hyper-connection over ``n`` residual streams."""

    def __init__(self, config):
        super().__init__()
        d = int(config.hidden_size)
        n = int(getattr(config, "recurrent_hc_streams", 4))
        self.n = n
        self.eps = float(config.rms_norm_eps)
        self.norm_weight = nn.Parameter(torch.ones(n * d))
        std = float(getattr(config, "initializer_range", 0.02))
        # [pre | post | res] gate projections stacked into one GEMM.
        self.W = nn.Parameter(torch.randn(3 * n, n * d) * std)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        res_bias = float(getattr(config, "recurrent_hc_res_bias", -3.0))
        self.b_pre = nn.Parameter(torch.full((n,), -math.log(n - 1.0) if n > 1 else 10.0))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.full((n,), res_bias))
        self.loop_emb = nn.Parameter(torch.zeros(d))

    def begin(self, y: torch.Tensor):
        """Gates from ``y`` [..., n, d]; returns (core input, H_post, H_res)."""
        z = F.rms_norm(y.flatten(-2).float(), (y.shape[-1] * self.n,), eps=self.eps)
        g = F.linear(z * self.norm_weight.float(), self.W.float())
        g_pre, g_post, g_res = g.split(self.n, dim=-1)
        h_pre = torch.sigmoid(self.alpha[0] * g_pre + self.b_pre)
        h_post = 2.0 * torch.sigmoid(self.alpha[1] * g_post + self.b_post)
        h_res = torch.sigmoid(self.alpha[2] * g_res + self.b_res)
        x = (h_pre.unsqueeze(-1) * y.float()).sum(dim=-2)
        return x.to(y.dtype), h_post, h_res

    def end(self, y, out, h_post, h_res) -> torch.Tensor:
        """y <- H_res y + H_post (core_out + e_l)."""
        upd = (out.float() + self.loop_emb.float()).unsqueeze(-2)
        y_new = h_res.unsqueeze(-1) * y.float() + h_post.unsqueeze(-1) * upd
        return y_new.to(y.dtype)


class RecurrentLoopMixer(nn.Module):
    """Drives the mixers at the loop boundaries of one forward pass.

    ``UMichSLMModel.forward`` calls :meth:`boundary` with ``k = 0..iters`` at the
    schedule positions where iteration ``k`` would begin (``k = iters`` is the
    coda's first layer), passing the hidden state the layer walk holds there.
    Per-forward state lives in the ``state`` dict the caller owns, never on the
    module, so recompute and concurrent forwards cannot see stale state.
    """

    def __init__(self, config):
        super().__init__()
        self.injection = (
            StableLoopInjection(config)
            if getattr(config, "recurrent_stable_injection", False)
            else None
        )
        self.hyper = None
        if getattr(config, "recurrent_hyper_connections", False):
            loops = int(getattr(config, "recurrent_max_iters", 1))
            self.hyper = nn.ModuleList([LoopHyperConnection(config) for _ in range(loops)])

    def _hc(self, k: int) -> LoopHyperConnection:
        assert self.hyper is not None
        return self.hyper[min(k, len(self.hyper) - 1)]

    def _enter(self, state: Dict, k: int) -> torch.Tensor:
        y = state["y"]
        if self.hyper is not None:
            x, state["h_post"], state["h_res"] = self._hc(k).begin(y)
        else:
            x = y
        if self.injection is not None:
            x = self.injection(x, state["e"])
        return x

    def boundary(self, state: Dict, k: int, hidden: torch.Tensor, iters: int) -> torch.Tensor:
        if k == 0:
            # ``hidden`` is the prelude output.
            if self.injection is not None:
                state["e"] = self.injection.encode(hidden)
                h0 = self.injection.initial_state(hidden)
            else:
                h0 = hidden
            if self.hyper is not None:
                n = self.hyper[0].n
                h0 = h0.unsqueeze(-2).expand(*h0.shape[:-1], n, h0.shape[-1])
            state["y"] = h0
            return self._enter(state, 0)
        # ``hidden`` is iteration k-1's core output.
        if self.hyper is not None:
            state["y"] = self._hc(k - 1).end(
                state["y"], hidden, state.pop("h_post"), state.pop("h_res")
            )
        else:
            state["y"] = hidden
        if k < iters:
            return self._enter(state, k)
        y = state["y"]
        return y.mean(dim=-2) if self.hyper is not None else y

    # ------------------------------------------------------------ accounting
    def compute_param_delta(self, expected_iters: float) -> float:
        """Executed-minus-stored params for one forward at E[T] iterations.

        The injection's weights (bar the prelude norm, run once) are SHARED and
        execute every iteration: stored once, executed E[T] times. The
        hyper-connection weights are LOOP-SPECIFIC: ``max_iters`` sets stored,
        and a forward of T iterations executes T of them, so E[T] sets on
        average (exact while T <= max_iters, i.e. in training).
        """
        delta = 0.0
        if self.injection is not None:
            shared = sum(
                p.numel()
                for name, p in self.injection.named_parameters()
                if not name.startswith("prelude_norm")
            )
            delta += (expected_iters - 1.0) * shared
        if self.hyper is not None:
            one = sum(p.numel() for p in self.hyper[0].parameters())
            delta += (expected_iters - len(self.hyper)) * one
        return delta


def build_recurrent_loop_mixer(config) -> Optional[RecurrentLoopMixer]:
    if not getattr(config, "use_recurrent_depth", False):
        return None
    if not (
        getattr(config, "recurrent_stable_injection", False)
        or getattr(config, "recurrent_hyper_connections", False)
    ):
        return None
    return RecurrentLoopMixer(config)
