"""Global Workspace Trunk (GWT) — dynamic-compute chunk 2, Pitch 1.

"Reasoning lives in the memory update, not the token stream."

The SWA+globals design keeps 256 persistent global tokens whose hidden states,
under the causal mask, never ingest content (they sit at positions 0..N-1) —
they are read-only attractors written only by gradient descent. GWT adds a
*live* counterpart: a fixed-size recurrent workspace updated at chunk
boundaries by a deep trunk, while the token stream stays shallow, uniform-cost
and completely unchanged (full cross-chunk attention included).

Mechanism, per forward, at one integration point in the layer stack:

  1. The content sequence is split into chunks of ``trunk_chunk_size``.
  2. Chunk ``c`` READS workspace state ``W_c`` via cross-attention
     (tokens query, workspace is K/V) added to the residual stream. ``W_c``
     was written exclusively from chunks ``< c`` — the load-bearing causality
     invariant (`tests/test_workspace_trunk.py`).
  3. The trunk then WRITES ``W_{c+1} = Trunk(W_c, chunk c token states)``:
     a write cross-attention (workspace queries the completed chunk), trunk
     self-attention over the slots, and an MLP — iterated ``T`` times, with
     ``T`` sampled per micro-batch via Arm B's
     :func:`~umich_slm.modules.recurrent_depth.sample_recursion_count`
     (router-free: no aux loss, no collapse, no train/infer routing mismatch).

Deep compute is O(1) in sequence length — the trunk operates on
``trunk_workspace_slots`` states regardless of context length — and the
iteration count is the native test-time-compute knob (at inference,
``trunk_halt_threshold`` halts on workspace convergence instead of a fixed T).

Safe insertion: the read's output projection is ZERO-initialized (Flamingo-
style), so at init a GWT model is bitwise identical to its static-globals twin
(`swa_globals_dense` backbone). Gradient reaches the trunk once the read
projection grows off zero.

No position encoding inside the trunk: the workspace is an unordered set of
slots, and chunk token states already carry position through the token stream.

KV-cache decode is deferred exactly like Arm B (config forces
``use_cache=False``): bursty workspace refresh at decode-time chunk boundaries
is an inference-integration project, not a screening prerequisite.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_depth import sample_recursion_count
from .rmsnorm import get_rmsnorm_class
from .swiglu import SwiGLUMLP


class TrunkCrossAttention(nn.Module):
    """Bare multi-head cross-attention (MHA, no position encoding).

    Used in three roles: token->workspace read, workspace->chunk write, and
    workspace self-attention (queries == keys/values). ``zero_init_output``
    marks the output projection for zero init in ``_init_weights`` (the flag
    must live on the Linear itself so it survives ``post_init`` re-init).
    """

    def __init__(self, config, zero_init_output: bool = False):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        if self.head_dim * self.num_heads != h:
            raise ValueError(f"hidden_size {h} not divisible by num_attention_heads")
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        if zero_init_output:
            self.o_proj._zero_init_weight = True

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        b, tq, h = q_in.shape
        tk = kv_in.shape[1]
        q = self.q_proj(q_in).view(b, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, tk, self.num_heads, self.head_dim).transpose(1, 2)
        # Default no mask: the write side sees a completed chunk (causally
        # valid) and the read side sees the workspace, which is not part of
        # the sequence. key_mask ([B, Tk] bool, True = attend) restricts the
        # WRITE to same-document chunk tokens under intra-doc masking.
        if top_k is not None and top_k > 0 and top_k < tk:
            # Write-sparse bottleneck: each query may read at most top_k keys
            # (hard top-k on logits; gradients flow through the kept entries).
            # Manual path — SDPA cannot express it.
            scale = self.head_dim**-0.5
            logits = torch.matmul(q, k.transpose(-1, -2)) * scale
            if key_mask is not None:
                logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            kth = logits.topk(top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            out = torch.matmul(F.softmax(logits, dim=-1), v)
        else:
            attn_mask = None
            if key_mask is not None:
                attn_mask = key_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, tq, h))


class TrunkBlock(nn.Module):
    """One trunk update block: write cross-attn -> slot self-attn -> MLP.

    Pre-norm residual throughout, operating on the [B, slots, d] workspace.
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.write_norm_ws = norm_class(h, eps=eps)
        self.write_norm_src = norm_class(h, eps=eps)
        self.write_attn = TrunkCrossAttention(config)
        # Write-sparse bottleneck (0/None = dense; GWT-style write competition).
        self._write_topk = int(getattr(config, "trunk_write_topk", 0)) or None
        self.self_norm = norm_class(h, eps=eps)
        self.self_attn = TrunkCrossAttention(config)
        self.mlp_norm = norm_class(h, eps=eps)
        self.mlp = SwiGLUMLP(
            hidden_size=h,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
        )

    def forward(
        self,
        workspace: torch.Tensor,
        chunk_states: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        workspace = workspace + self.write_attn(
            self.write_norm_ws(workspace),
            self.write_norm_src(chunk_states),
            key_mask=src_mask,
            top_k=self._write_topk,
        )
        normed = self.self_norm(workspace)
        workspace = workspace + self.self_attn(normed, normed)
        workspace = workspace + self.mlp(self.mlp_norm(workspace))
        return workspace


class WorkspaceRead(nn.Module):
    """Token->workspace read: tokens cross-attend to the workspace state.

    Output projection is zero-initialized so inserting the read is a no-op at
    init (the GWT model starts bitwise equal to its static-globals twin).
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.token_norm = norm_class(h, eps=eps)
        self.ws_norm = norm_class(h, eps=eps)
        self.attn = TrunkCrossAttention(config, zero_init_output=True)

    def forward(self, tokens: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
        return self.attn(self.token_norm(tokens), self.ws_norm(workspace))


def trunk_workspace_slots(config) -> int:
    """Resolve the workspace size: explicit, else the globals count, else 256."""
    slots = int(getattr(config, "trunk_workspace_slots", 0) or 0)
    if slots > 0:
        return slots
    globals_count = int(getattr(config, "swa_global_count", 0) or 0)
    return globals_count if globals_count > 0 else 256


class _TrunkIter(nn.Module):
    """One trunk update iteration: a single pass through the block stack.

    Split out so the whole per-iteration kernel chain (write cross-attn +
    slot self-attn + MLP + norms, ~a dozen tiny launches x trunk_layers) can
    be captured and replayed as one CUDA graph. Shares the parent's blocks —
    it owns no parameters and is never registered as a submodule, so
    state_dicts and checkpoints are byte-identical with the flag on or off.
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(
        self,
        workspace: torch.Tensor,
        chunk_states: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            workspace = block(workspace, chunk_states, src_mask=src_mask)
        return workspace


class WorkspaceTrunk(nn.Module):
    """The full GWT mechanism: chunk loop + read + iterated trunk update.

    Called once per forward at the integration point, on the content part of
    the hidden states (prefix globals/meta excluded). Returns the content with
    the workspace read added; workspace states never enter the token sequence.

    Diagnostics stashed per forward (consumed by dyncompute_diagnostics):
      - ``_last_iters``: realized trunk iteration count.
      - ``_last_delta_norms``: per-chunk list of per-iteration relative
        workspace deltas (convergence curves).
      - ``_last_states``: per-chunk workspace states, only when
        ``record_states`` is set (tests / probes; off by default).
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.num_slots = trunk_workspace_slots(config)
        self.chunk_size = int(getattr(config, "trunk_chunk_size", 512))
        self.min_iters = int(getattr(config, "trunk_iters_min", 1))
        self.max_iters = int(getattr(config, "trunk_iters_max", 3))
        self.eval_iters = int(getattr(config, "trunk_eval_iters", 0))
        self.halt_threshold = float(getattr(config, "trunk_halt_threshold", 0.0))
        self.detach_chunks = bool(getattr(config, "trunk_detach_chunks", False))

        # Named ``tokens`` to mirror GlobalTokens: a 2-D learned state table
        # that lands in the same AdamH group as the incumbents' globals.
        init_std = float(getattr(config, "initializer_range", 0.02))
        self.tokens = nn.Parameter(torch.randn(self.num_slots, h) * init_std)

        num_blocks = int(getattr(config, "trunk_layers", 2))
        if bool(getattr(config, "trunk_weight_shared", False)):
            shared = TrunkBlock(config)
            self.blocks = nn.ModuleList([shared] * num_blocks)
        else:
            self.blocks = nn.ModuleList(TrunkBlock(config) for _ in range(num_blocks))
        self.read = WorkspaceRead(config)

        self.record_states = False
        self._last_iters: Optional[int] = None
        self._last_delta_norms: List[List[float]] = []
        self._last_states: List[torch.Tensor] = []

        # CUDA-graph replay of the inner iteration (wall-clock only; the flag
        # is inert on CPU, in eval-with-halting, and under diagnostics).
        self.cuda_graph = bool(getattr(config, "trunk_cuda_graph", False))
        self.cuda_graph_impl = str(getattr(config, "trunk_cuda_graph_impl", "compile"))
        self._graph_fns: Dict[Tuple, Callable] = {}
        self._graph_disabled = False
        self._graph_call_slot = 0

    def initial_state(self, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch, -1, -1).to(dtype=dtype, device=device)

    def _doc_mask_for_chunk(
        self,
        doc_ids: Optional[torch.Tensor],
        start: int,
        end: int,
        ws_doc: Optional[torch.Tensor],
    ):
        """Per-chunk intra-doc controls. Returns (read_gate, write_mask, write_reset, new_ws_doc).

        read_gate  [B, C, 1]: 1 where the token's document matches the document
            the current workspace was written from (``ws_doc``); tokens of any
            LATER document inside the chunk read nothing (their document's
            workspace starts at the learned init when it first gets written).
        write_mask [B, C] bool: True on tokens of the chunk's LAST document —
            the only content written forward (a document sandwiched entirely
            inside one chunk is never written: isolated but lossy, same
            spirit as the chunk-granular design).
        write_reset [B] bool: rows whose chunk contains a boundary (last-token
            document != ws_doc) restart from the learned init BEFORE the write.
        """
        if doc_ids is None:
            return None, None, None, None
        chunk_docs = doc_ids[:, start:end]
        read_gate = (chunk_docs == ws_doc.unsqueeze(1)).unsqueeze(-1)
        last_doc = chunk_docs[:, -1]
        write_mask = chunk_docs == last_doc.unsqueeze(1)
        write_reset = last_doc != ws_doc
        return read_gate, write_mask, write_reset, last_doc

    def _update(
        self,
        workspace: torch.Tensor,
        chunk_states: torch.Tensor,
        iters: int,
        use_halting: bool,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # `float(tensor)` is a device->host sync. Halting genuinely needs the value
        # on the host to decide whether to break, but halting is eval-only
        # (`use_halting = not self.training and ...`), and `record_states` is a
        # diagnostic. In the training path neither applies, so keep the norm on
        # device and skip the sync entirely -- it ran once per trunk iteration per
        # chunk, i.e. 8 more per forward at 4096/512 on top of the reset guards.
        want_host = use_halting or self.record_states
        if (
            self.cuda_graph
            and not self._graph_disabled
            and not want_host  # the host-sync diagnostics below are illegal in a graph
            and self.training
            and workspace.is_cuda
            and torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            graphed = self._graphed_update(workspace, chunk_states, iters, src_mask)
            if graphed is not None:
                # Same list-length contract as the eager path: one (empty,
                # since want_host is False) delta list per write.
                self._last_delta_norms.append([])
                return graphed
        deltas: List[float] = []
        for _ in range(iters):
            new_ws = workspace
            for block in self.blocks:
                new_ws = block(new_ws, chunk_states, src_mask=src_mask)
            if want_host:
                rel = float(
                    (new_ws - workspace).norm().detach()
                    / workspace.norm().detach().clamp_min(1e-6)
                )
                deltas.append(rel)
            workspace = new_ws
            if use_halting and deltas and deltas[-1] < self.halt_threshold:
                break
        self._last_delta_norms.append(deltas)
        return workspace

    def _graphed_update(
        self,
        workspace: torch.Tensor,
        chunk_states: torch.Tensor,
        iters: int,
        src_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Run ``iters`` graph-replayed iterations; None => caller runs eager.

        ``make_graphed_callables`` replays clobber their static buffers, so
        under impl "graph" a captured callable may run at most once per step —
        each call slot within the forward gets its own capture (bounded:
        (chunks-1) * trunk_iters_max). Impl "compile" (cudagraph trees) handles
        the many-calls-per-step pattern natively with one compiled callable.
        """
        # Replays under AMP must not consult the autocast weight cache (the
        # capture ran with cache_enabled=False, per the make_graphed_callables
        # contract); mirror the ambient autocast state without the cache.
        if self.cuda_graph_impl == "graph" and torch.is_autocast_enabled("cuda"):
            amp_ctx = torch.autocast(
                "cuda", dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
            )
        else:
            amp_ctx = contextlib.nullcontext()
        out = workspace
        with amp_ctx:
            for _ in range(iters):
                fn = self._graph_fn_for(out, chunk_states, src_mask)
                if fn is None:
                    return None
                if src_mask is None:
                    out = fn(out.contiguous(), chunk_states.contiguous())
                else:
                    out = fn(out.contiguous(), chunk_states.contiguous(), src_mask)
                self._graph_call_slot += 1
        return out

    def _graph_fn_for(
        self,
        workspace: torch.Tensor,
        chunk_states: torch.Tensor,
        src_mask: Optional[torch.Tensor],
    ) -> Optional[Callable]:
        impl = self.cuda_graph_impl
        key: Tuple = (
            impl,
            tuple(workspace.shape),
            tuple(chunk_states.shape),
            workspace.dtype,
            chunk_states.dtype,
            src_mask is not None,
        )
        if impl == "graph":
            key = key + (self._graph_call_slot,)
        fn = self._graph_fns.get(key)
        if fn is not None:
            return fn
        # A shape change is rare (only B or chunk length); churn past the cap
        # means something is off — degrade to eager for the process.
        cap = 128 if impl == "graph" else 8
        if len(self._graph_fns) >= cap:
            warnings.warn(
                "trunk_cuda_graph: capture-cache cap reached; disabling graphs "
                "for this process (eager fallback, math unchanged)."
            )
            self._graph_disabled = True
            return None
        try:
            iter_mod = _TrunkIter(self.blocks)
            if impl == "compile":
                fn = torch.compile(iter_mod, mode="reduce-overhead", dynamic=False)
            else:
                sample = [
                    workspace.detach().clone().contiguous().requires_grad_(
                        workspace.requires_grad
                    ),
                    chunk_states.detach().clone().contiguous().requires_grad_(
                        chunk_states.requires_grad
                    ),
                ]
                if src_mask is not None:
                    sample.append(src_mask.detach().clone().contiguous())
                if torch.is_autocast_enabled("cuda"):
                    amp_ctx = torch.autocast(
                        "cuda", dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
                    )
                else:
                    amp_ctx = contextlib.nullcontext()
                with amp_ctx:
                    fn = torch.cuda.make_graphed_callables(
                        iter_mod, tuple(sample), allow_unused_input=True
                    )
        except Exception as exc:  # a capture bug must degrade, never kill a run
            warnings.warn(
                f"trunk_cuda_graph capture failed; falling back to eager: {exc!r}"
            )
            self._graph_disabled = True
            return None
        self._graph_fns[key] = fn
        return fn

    def forward(
        self,
        content: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
        force_iters: Optional[int] = None,
    ) -> torch.Tensor:
        b, t, _ = content.shape
        workspace = self.initial_state(b, content.dtype, content.device)

        use_halting = (not self.training) and self.halt_threshold > 0.0
        if use_halting:
            # Halting caps at max_iters; _update breaks on convergence.
            iters = self.max_iters
        else:
            iters = sample_recursion_count(
                self.min_iters,
                self.max_iters,
                "uniform",
                training=self.training,
                force=force_iters,
                eval_iters=self.eval_iters,
            )
        self._last_iters = iters
        self._last_delta_norms = []
        self._last_states = []
        self._graph_call_slot = 0

        # Intra-document correctness: track which document each row's workspace
        # was written from (``ws_doc``). Tokens only READ same-document state;
        # a chunk containing a boundary resets before writing and WRITES only
        # its last document's tokens — so no content ever crosses a document
        # boundary through the workspace, even when boundaries fall mid-chunk
        # (the old chunk-aligned reset leaked exactly there).
        ws_doc = doc_ids[:, 0].clone() if doc_ids is not None else None

        outputs: List[torch.Tensor] = []
        for start in range(0, t, self.chunk_size):
            end = min(start + self.chunk_size, t)
            chunk = content[:, start:end]

            # A chunk whose FIRST token opens a new document restarts from the
            # learned init before reading — a boundary-aligned fresh document
            # behaves exactly like a document at position 0 (content-free bank
            # read). Mid-chunk starters read nothing until their first write.
            if doc_ids is not None and start > 0:
                start_doc = doc_ids[:, start]
                pre_reset = start_doc != ws_doc
                # Unconditional. The `if bool(pre_reset.any())` that used to guard
                # this was a pessimization: it pays a device->host sync (stalling
                # the pipeline) to skip one elementwise torch.where. At 4096/512
                # that is 8 syncs per forward from this site alone, and it is a
                # hard graph break under torch.compile -- which only shows up when
                # intra-doc masking is ON, since doc_ids is None otherwise.
                init = self.initial_state(b, content.dtype, content.device)
                workspace = torch.where(pre_reset.view(b, 1, 1), init, workspace)
                ws_doc = torch.where(pre_reset, start_doc, ws_doc)

            read_gate, write_mask, write_reset, next_ws_doc = self._doc_mask_for_chunk(
                doc_ids, start, end, ws_doc
            )

            if self.record_states:
                self._last_states.append(workspace.detach().clone())

            read = self.read(chunk, workspace)
            if read_gate is not None:
                read = read * read_gate.to(read.dtype)
            chunk_out = chunk + read
            outputs.append(chunk_out)

            # The final chunk's write has no reader — skip it (its gradient
            # path would be dead anyway).
            if end < t:
                if write_reset is not None:
                    # Unconditional for the same reason as the pre-reset above:
                    # torch.where is cheaper than the sync that used to guard it.
                    init = self.initial_state(b, content.dtype, content.device)
                    workspace = torch.where(write_reset.view(b, 1, 1), init, workspace)
                    ws_doc = next_ws_doc
                # Stop-grad across the chunk boundary: detach BOTH update
                # inputs, so the write itself (and hence the trunk parameters,
                # via the next chunk's read) stays grad-connected while no
                # gradient crosses back into earlier chunks — bounded BPTT
                # memory, the mitigation the plan calls for. Detaching only
                # the workspace would still leak gradient through the
                # read -> chunk_out -> next write chain.
                if self.detach_chunks:
                    workspace = self._update(
                        workspace.detach(), chunk_out.detach(), iters, use_halting, write_mask
                    )
                else:
                    workspace = self._update(workspace, chunk_out, iters, use_halting, write_mask)

        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]


class ParallelWorkspaceTrunk(nn.Module):
    """Linearized (all-chunks-batched) twin of :class:`WorkspaceTrunk`.

    The sequential trunk is the last *serial* latency in the scribe arms: 7
    dependent write->read rounds per 4096-token forward, each a chain of tiny
    kernels. This variant removes the recurrence entirely (``trunk_variant=
    "parallel"``): every chunk's write is computed from the SHARED initial
    workspace in ONE batched pass over ``[B*C, slots, H]``, and chunk c reads

        W_c = W0 + budget * tanh( (sum_{j<c, visible} Delta_j) / budget )

    -- a masked cumulative sum over the chunk axis with the delta-bank's
    rms-budget squash (delta_bank.py), so the state can never leave the scale
    the read projections were trained for and the zero-init contract is exact
    (no writes => W_c == W0 bitwise).

    FUNCTIONAL change vs the sequential trunk (screened as its own arm): the
    write is conditioned on W0, not the evolved workspace — the recurrence is
    linearized, chunk writes cannot see earlier chunks' writes. Causality and
    document isolation are preserved exactly:

      * visibility is strictly ``j < c`` (a chunk never reads its own or later
        writes), matching the sequential read-then-write order;
      * with packed documents, reader chunk c's workspace doc is its FIRST
        token's doc; only writer chunks OWNED by that doc contribute
        (documents are contiguous under packing, so this is the sequential
        owner/reset rule without the serial chain), and the token-level read
        gate blocks any token whose doc differs — no content crosses a
        boundary through the workspace.

    Adds exactly one parameter (``dev_scale``) over the sequential trunk.
    """

    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.num_slots = trunk_workspace_slots(config)
        self.chunk_size = int(getattr(config, "trunk_chunk_size", 512))
        init_std = float(getattr(config, "initializer_range", 0.02))
        self.tokens = nn.Parameter(torch.randn(self.num_slots, h) * init_std)
        num_blocks = int(getattr(config, "trunk_layers", 2))
        if bool(getattr(config, "trunk_weight_shared", False)):
            shared = TrunkBlock(config)
            self.blocks = nn.ModuleList([shared] * num_blocks)
        else:
            self.blocks = nn.ModuleList(TrunkBlock(config) for _ in range(num_blocks))
        self.read = WorkspaceRead(config)
        self.dev_scale = nn.Parameter(torch.tensor(1.0))

        # Diagnostics interface parity with WorkspaceTrunk (consumed by
        # dyncompute_diagnostics and the model's _last_trunk_iters passthrough).
        self.record_states = False
        self._last_iters: Optional[int] = None
        self._last_delta_norms: List[List[float]] = []
        self._last_states: List[torch.Tensor] = []

    def forward(
        self,
        content: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
        force_iters: Optional[int] = None,
    ) -> torch.Tensor:
        b, t, h = content.shape
        cs = self.chunk_size
        starts = list(range(0, t, cs))
        n_read = len(starts)
        self._last_iters = 1
        self._last_delta_norms = [[] for _ in range(max(0, n_read - 1))]
        self._last_states = []

        w0 = self.tokens.unsqueeze(0).to(dtype=content.dtype)  # [1, N, h]
        if n_read <= 1:
            # Single (possibly partial) chunk: nothing ever writes, pure read.
            if self.record_states:
                self._last_states.append(w0.expand(b, -1, -1).detach().clone())
            return content + self.read(content, w0.expand(b, -1, -1))

        # Writers: every chunk except the last (its write has no reader) — all
        # full chunk_size by construction, same invariant as the sequential
        # trunk (a partial chunk is always the final one).
        c_w = n_read - 1
        chunks = content[:, : c_w * cs].reshape(b, c_w, cs, h)
        if doc_ids is not None:
            cdocs = doc_ids[:, : c_w * cs].reshape(b, c_w, cs)
            owner = cdocs[:, :, -1]  # last doc owns the chunk (trunk write rule)
            write_mask = (cdocs == owner.unsqueeze(-1)).reshape(b * c_w, cs)
        else:
            owner = None
            write_mask = None

        # ONE batched write pass from the shared initial state.
        src = chunks.reshape(b * c_w, cs, h)
        ws = w0.expand(b * c_w, -1, -1)
        for block in self.blocks:
            ws = block(ws, src, src_mask=write_mask)
        updates = (ws - w0).view(b, c_w, self.num_slots, h)

        if owner is not None:
            first_pos = torch.as_tensor(starts, device=content.device)
            ws_doc = doc_ids[:, first_pos]  # [b, n_read]: reader's doc
        else:
            ws_doc = None

        states = self._reader_states(w0, updates, owner, ws_doc, n_read)  # [b, n_read, N, h]
        if self.record_states:
            for c in range(n_read):
                self._last_states.append(states[:, c].detach().clone())

        # Batched reads: the first n_read-1 readers are full chunks; the last
        # may be partial — two read calls total instead of n_read.
        full = content[:, : c_w * cs].reshape(b * c_w, cs, h)
        read_full = self.read(full, states[:, :c_w].reshape(b * c_w, self.num_slots, h))
        read_full = read_full.view(b, c_w * cs, h)
        tail = content[:, c_w * cs :]
        read_tail = self.read(tail, states[:, c_w])
        read = torch.cat([read_full, read_tail], dim=1)

        if ws_doc is not None:
            # Token-level gate, same rule as the sequential read_gate: a token
            # reads only workspace state attributed to its own document.
            gate_full = (cdocs == ws_doc[:, :c_w].unsqueeze(-1)).reshape(b, c_w * cs)
            gate_tail = doc_ids[:, c_w * cs :] == ws_doc[:, c_w].unsqueeze(-1)
            gate = torch.cat([gate_full, gate_tail], dim=1).unsqueeze(-1)
            read = read * gate.to(read.dtype)
        return content + read

    def _budget(self, dtype: torch.dtype) -> torch.Tensor:
        """Deviation budget in units of the workspace init's rms (delta-bank rule)."""
        rms = self.tokens.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
        return (self.dev_scale.float().abs() * rms).clamp_min(1e-6).to(dtype)

    def _reader_states(
        self,
        w0: torch.Tensor,
        updates: torch.Tensor,
        owner: Optional[torch.Tensor],
        ws_doc: Optional[torch.Tensor],
        n_read: int,
    ) -> torch.Tensor:
        """Per-reader workspace states [b, n_read, N, h] from the batched
        writer updates. Base class: strict-past masked sum over the chunk axis
        (the linearized cumsum) + rms-budget squash. Subclasses override this
        to install a different chunk-axis combine (gated scans)."""
        b, c_w = updates.shape[0], updates.shape[1]
        r_idx = torch.arange(n_read, device=updates.device)
        w_idx = torch.arange(c_w, device=updates.device)
        vis = w_idx.view(1, -1) < r_idx.view(-1, 1)  # [n_read, c_w]
        if owner is not None:
            vis = vis.unsqueeze(0) & (owner.unsqueeze(1) == ws_doc.unsqueeze(-1))
        else:
            vis = vis.unsqueeze(0).expand(b, -1, -1)
        dev = torch.einsum("brc,bcnh->brnh", vis.to(updates.dtype), updates)
        budget = self._budget(updates.dtype)
        return w0.unsqueeze(0) + budget * torch.tanh(dev / budget)


class GatedScanWorkspaceTrunk(ParallelWorkspaceTrunk):
    """State-dependent chunk-axis scans on the batched-trunk skeleton.

    Same speed shape as :class:`ParallelWorkspaceTrunk` (its whole point):
    every chunk's write update is computed in ONE batched pass from the shared
    initial workspace, and only the *combine* over the chunk axis differs — a
    short sequential scan of elementwise ops over C-1 chunk updates (7 at
    4096/512, 63 at 32K; [B, slots, H] each, negligible next to the write).

    What the cumsum combine cannot express — and these scans add:

      * ``trunk_variant="rglru"`` (RG-LRU-gated trunk): input-gated decayed
        accumulation, Griffin's variance-preserving update on the chunk axis

            a_j   = exp(-softplus(c) * exp(log_a) * sigmoid(a_proj(U_j)))
            dev_j = a_j * dev_{j-1} + sqrt(1 - a_j^2) * U_j

        so the workspace can FORGET: stale chunk writes decay at a learned,
        content-dependent rate instead of accumulating forever. At 32K (63
        writes/doc) an un-decayed sum saturates the tanh budget; the gate is
        the mechanism that keeps late-context writes effective.

      * ``trunk_variant="delta_rule"`` (delta-rule trunk): gated
        error-correcting REPLACE, the delta-rule fixed-point update per slot

            beta_j = sigmoid(beta_proj(U_j))
            dev_j  = dev_{j-1} + beta_j * (U_j - dev_{j-1})

        writing the same content twice is idempotent (a true overwrite
        semantics), where the cumsum doubles it — the property motivating
        DeltaNet-style memories.

    Both keep every partrunk contract: zero-init exactness (write output is
    zero-init => U_j == 0 => dev == 0 => state == W0 bitwise; the gates then
    act on zeros), strict ``j < c`` visibility (a reader sees only the scan
    state after writer c-1), doc isolation (scan resets when chunk ownership
    changes doc + the chunk-boundary reader gate + the token-level read gate),
    and the rms-budget squash on the way out. Gate parameters are per-slot
    scalars with per-dim base decay — +O(h) params, matched-N to partrunk to
    within the gate (same tolerance as the register-arm read gates).
    """

    SCAN_VARIANTS = ("rglru", "delta_rule")

    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size
        self.scan = str(getattr(config, "trunk_variant", "rglru"))
        if self.scan not in self.SCAN_VARIANTS:
            raise ValueError(f"GatedScanWorkspaceTrunk got trunk_variant={self.scan!r}")
        if self.scan == "rglru":
            # Per-slot scalar gate from the update content; per-dim learned
            # base decay (log_a, c) exactly as in griffin.RGLRUAttention.
            self.a_proj = nn.Linear(h, 1, bias=True)
            self.log_a = nn.Parameter(torch.zeros(h))
            self.c = nn.Parameter(torch.ones(h))
        else:
            self.beta_proj = nn.Linear(h, 1, bias=True)

    def _scan_update(self, dev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if self.scan == "rglru":
            gate_in = torch.sigmoid(self.a_proj(u))  # [b, N, 1]
            base = F.softplus(self.c) * torch.exp(self.log_a)  # [h]
            a = torch.exp(-base * gate_in)  # [b, N, h]
            return a * dev + torch.sqrt(torch.clamp(1.0 - a * a, min=0.0)) * u
        beta = torch.sigmoid(self.beta_proj(u))  # [b, N, 1]
        return dev + beta * (u - dev)

    def _reader_states(
        self,
        w0: torch.Tensor,
        updates: torch.Tensor,
        owner: Optional[torch.Tensor],
        ws_doc: Optional[torch.Tensor],
        n_read: int,
    ) -> torch.Tensor:
        b, c_w = updates.shape[0], updates.shape[1]
        dev = torch.zeros_like(updates[:, 0])
        devs = [dev]  # reader 0: nothing written yet -> W0 exactly
        state_doc = owner[:, 0] if owner is not None else None
        for j in range(c_w):
            u = updates[:, j]
            if owner is not None:
                # Reset BEFORE writing chunk j: accumulated state never crosses
                # a document boundary (same rule as DeltaBank / the sequential
                # trunk's write_reset). Docs are contiguous under packing, so
                # the surviving run is exactly the reader's-doc writers.
                reset = (owner[:, j] != state_doc).view(b, 1, 1)
                dev = torch.where(reset, torch.zeros_like(dev), dev)
                state_doc = owner[:, j]
            dev = self._scan_update(dev, u)
            if ws_doc is not None:
                # Chunk-boundary reader gate: reader j+1 whose first token
                # opens a NEW document reads W0 (the sequential pre-reset).
                gate = (ws_doc[:, j + 1] == state_doc).view(b, 1, 1)
                devs.append(torch.where(gate, dev, torch.zeros_like(dev)))
            else:
                devs.append(dev)
        dev_all = torch.stack(devs, dim=1)  # [b, n_read, N, h]
        budget = self._budget(updates.dtype)
        return w0.unsqueeze(0) + budget * torch.tanh(dev_all / budget)
