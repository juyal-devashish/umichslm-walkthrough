"""Gated DeltaNet and Kimi Delta Attention implementations.

This module implements linear attention alternatives to standard softmax attention,
combining Mamba2's data-dependent gating with the delta rule memory update.

Key concepts:
- O(n) complexity instead of O(n²) quadratic attention
- Fixed-size recurrent state instead of growing KV cache
- Hybrid architectures supported (mixing with standard attention)

Two variants are implemented:
1. GatedDeltaNet (Qwen3-Next style): Scalar gating per head
2. KimiDeltaAttention (Kimi-K2 style): Channel-wise gating for finer control

References:
- Paper: https://arxiv.org/abs/2412.06464 "Gated Delta Networks"
- FLA Library: https://github.com/fla-org/flash-linear-attention
- NVlabs: https://github.com/NVlabs/GatedDeltaNet
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# Try to import FLA library (optimized Triton kernels)
FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    FLA_AVAILABLE = True
except ImportError:
    pass

# The chunk kernel is the *training* kernel: unlike fused_recurrent_gated_delta_rule
# (inference-only, no backward), chunk_gated_delta_rule implements the true Gated
# DeltaNet with a backward pass, so it can drive the delta-rule training path.
CHUNK_FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    CHUNK_FLA_AVAILABLE = True
except ImportError:
    pass

# Kimi Delta Attention: the same delta rule with a per-KEY-CHANNEL forget gate
# (S <- (I - beta k k^T) Diag(alpha) S + beta k v^T; arXiv:2510.26692).
KDA_FLA_AVAILABLE = False
try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    KDA_FLA_AVAILABLE = True
except ImportError:
    pass

# Track whether we've warned about FLA being unavailable
_FLA_WARNING_ISSUED = False


def _warn_fla_unavailable():
    """Issue a one-time warning that FLA library is not available."""
    global _FLA_WARNING_ISSUED
    if not _FLA_WARNING_ISSUED and not FLA_AVAILABLE:
        logger.warning(
            "flash-linear-attention (FLA) library not found. DeltaNet will use a naive "
            "PyTorch implementation which is significantly slower for long sequences. "
            "For better performance on CUDA, install FLA: pip install flash-linear-attention"
        )
        _FLA_WARNING_ISSUED = True


@dataclass
class DeltaNetCache:
    """Cache for DeltaNet layers during incremental generation.

    DeltaNet requires caching two types of state:
    1. recurrent_state: The delta rule memory [batch, heads, key_dim, value_dim]
    2. conv_state: Last (kernel_size - 1) hidden states for short convolution

    Attributes:
        recurrent_state: The delta rule recurrent memory matrix.
        conv_state: Cached hidden states for short convolution, shape
            [batch, hidden_size, kernel_size - 1]. None if conv is disabled.
        seen_tokens: Total number of tokens represented by this cache. This is
            optional metadata used to round-trip through legacy cache formats.
    """

    recurrent_state: Optional[torch.FloatTensor] = None
    conv_state: Optional[torch.FloatTensor] = None
    seen_tokens: Optional[int] = None


@dataclass
class DeltaNetOutput(ModelOutput):
    """Output type for DeltaNet layers.

    Attributes:
        hidden_states: Output hidden states.
        recurrent_state: The recurrent state for caching during generation.
    """

    hidden_states: torch.FloatTensor = None
    recurrent_state: Optional[torch.FloatTensor] = None


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet attention (Qwen3-Next variant).

    Implements linear attention with scalar gating per head and delta rule
    memory updates. This provides O(n) complexity with fixed-size recurrent
    state instead of O(n²) attention with growing KV cache.

    The delta rule update is:
        S_t = gate * S_{t-1} + beta * (k_t ⊗ v_t)
        o_t = q_t @ S_t

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.

    Attributes:
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        gate_proj: Gate projection (scalar per head, or channel-wise for KDA).
        beta_proj: Beta projection (learning rate for delta rule).
        o_proj: Output projection.
    """

    # Subclasses can override this to change gating behavior
    _use_channel_wise_gating: bool = False

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()

        # Warn once if FLA is not available
        _warn_fla_unavailable()

        self.config = config
        self.layer_idx = layer_idx

        # Intra-document masking (W2): boolean [batch, seq_len] marking each doc's first token,
        # stashed by the model per forward. When set, the recurrent state resets at boundaries
        # so it does not leak across packed documents. None = no intra-doc masking.
        self._intra_doc_start = None
        # Model-level varlen boundaries for the same mask, built ONCE per forward
        # (``build_batched_cu_seqlens``) and handed to every DeltaNet layer. Sharing the
        # one tensor object matters beyond skipping our own rebuild: FLA memoises
        # ``prepare_chunk_indices`` by argument IDENTITY, and that function host-syncs
        # (repeat_interleave over CUDA counts). A fresh tensor per layer missed the
        # cache and synced in every layer, twice under grad-ckpt recompute.
        self._delta_cu_seqlens = None

        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "deltanet_num_heads", None) or config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.expand_k = getattr(config, "deltanet_expand_k", 1)
        self.expand_v = getattr(config, "deltanet_expand_v", 1)

        # Short convolution settings
        self.use_short_conv = getattr(config, "deltanet_use_short_conv", True)
        self.conv_kernel_size = getattr(config, "deltanet_conv_kernel_size", 4)

        # Select the DeltaNet math. False (default) => legacy scalar-gated *linear
        # attention* (see _naive_forward): fast reference but not the delta rule.
        # True => true Gated DeltaNet (Qwen3-Next style: delta correction + qk-l2norm +
        # log-space gate) via FLA's chunk/recurrent kernels. Channel-wise gating (KDA)
        # has no FLA delta kernel, so it always uses the naive path regardless.
        self._use_true_delta = bool(getattr(config, "deltanet_true_delta_rule", False))
        # beta in (0, 2) instead of (0, 1): lets (I - beta k k^T) go negative (see
        # configuration_umichslm.deltanet_allow_neg_eigval).
        self.allow_neg_eigval = bool(getattr(config, "deltanet_allow_neg_eigval", False))
        # True KDA (2026-10-22): channel-wise gating ON the true delta rule. The forget
        # gate is per KEY channel ([H, K], what FLA's chunk_kda takes) and beta stays a
        # per-head scalar -- unlike the legacy channel-wise path below, which gates and
        # writes per VALUE channel on the linear-attention update. Everything else (q/k/v,
        # short conv, sigmoid gate in log space, no output gate/norm) is the GatedDeltaNet
        # harness, so a _kda twin differs from its parent in the decay granularity only.
        self._kda_delta = self._use_channel_wise_gating and self._use_true_delta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size} and num_heads={self.num_heads})"
            )

        # Projections
        self._init_dyn_rank(config)
        # Q and K must have same dimension for delta rule: out = q @ state where state is [D_k, D_v]
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_k,  # Match K dimension
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_k,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_v,
            bias=False,
        )

        # Gating projections - dimension depends on gating type
        gate_dim = self._get_gate_dim()
        self.gate_proj = nn.Linear(self.hidden_size, gate_dim, bias=False)
        beta_dim = self.num_heads if self._kda_delta else gate_dim
        self.beta_proj = nn.Linear(self.hidden_size, beta_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim * self.expand_v,
            self.hidden_size,
            bias=False,
        )

        # Optional short convolution for local context (like in Mamba)
        if self.use_short_conv:
            self.conv = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size - 1,
                groups=self.hidden_size,
            )
        else:
            self.conv = None

        # deltanet_output_gate (2026-10-31): Qwen3-Next / FLA's output norm + gate,
        # o_proj(RMSNorm_head(o) * silu(g_proj(x))), with x the layer input (the pause
        # rows when the pause stream is on, like Q). Off = the harness as before.
        if getattr(config, "deltanet_output_gate", False):
            v_dim = self.head_dim * self.expand_v
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * v_dim, bias=False)
            self.o_norm = nn.RMSNorm(v_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        else:
            self.g_proj = None
            self.o_norm = None


    # ---------------------------------------------------------------- dyn rank
    def _init_dyn_rank(self, config):
        """Per-chunk nested STATE RANK -- dynamic compute for a recurrent layer.

        The state is [d_k, d_v]. Zeroing q[..., r:] and k[..., r:] makes rows r:
        of the state unreachable: they are never written (the update is an outer
        product in k) and never read (the output is S^T q). So the layer runs at
        an EXACT rank-r state, and the tiers are nested by construction the same
        way the mat_ffn widths are.

        WHY THIS CANNOT SAVE FLOPS (with the stock FLA kernel). DeltaNet is
        sequential across chunks -- the state carries in order -- so the
        sort-by-tier trick that makes the nested FFN dispatch work is
        unavailable: chunks cannot be permuted. Masking bills the full kernel and
        uses part of it, exactly like the FFN's static tier windows. That makes
        this a QUALITY-PER-FLOP experiment (does adaptive rank allocation beat
        uniform rank at equal cost?) rather than a speed one, which is the
        lesson the mat_ffn work landed on anyway.

        ROUTING IS CHUNK-GRANULAR AND CAUSAL. A chunk's rank is chosen from the
        hidden state at the LAST POSITION OF THE PREVIOUS CHUNK, never from a
        mean over its own tokens -- pooling within the chunk would let a token at
        position p be influenced by p+1..chunk_end, which is precisely the leak
        the multi-position probe catches.

        Occupancy is STRUCTURAL, via the same threshold controller as
        mat_ffn: rank >= tier k iff tail_k > theta_k, theta_k adapted toward
        realized occupancy. An argmax router would collapse to the cheapest rank
        for the same reason the FFN router did -- no counterfactual gradient --
        and here it would be worse, because a low-rank chunk degrades the state
        handed to EVERY later chunk, so the cost of being cheap is delayed and
        diffuse rather than local.
        """
        # `_forced_rank` pins every position to one rank, bypassing the router.
        # That is the CONTROL: sweep it to measure the rank->loss slope before
        # believing any adaptive result. Skipping this control on the FFN cost a
        # whole run -- an 8x width sweep turned out to sit below the noise bar,
        # which made every routed comparison uninterpretable.
        self._forced_rank = getattr(config, "deltanet_forced_rank", None)
        self.dyn_rank = bool(getattr(config, "deltanet_dyn_rank", False))
        if self._forced_rank is not None:
            self._forced_rank = int(self._forced_rank)
        if not self.dyn_rank and self._forced_rank is None:
            return
        if not self.dyn_rank:
            return
        dk = self.head_dim * self.expand_k
        tiers = getattr(config, "deltanet_rank_tiers", None) or [
            max(8, dk // 4), max(8, dk // 2), dk
        ]
        self.rank_tiers = [max(1, min(int(t), dk)) for t in sorted(set(tiers))]
        pri = getattr(config, "deltanet_rank_prior", None) or [
            1.0 / len(self.rank_tiers)] * len(self.rank_tiers)
        tot = float(sum(pri))
        self.rank_prior = [float(p) / tot for p in pri]
        self.rank_chunk = int(getattr(config, "deltanet_rank_chunk", 128) or 128)
        self.rank_thresh_lr = float(getattr(config, "deltanet_rank_thresh_lr", 0.01))
        self.rank_router = nn.Linear(config.hidden_size, len(self.rank_tiers), bias=False)
        nn.init.normal_(self.rank_router.weight, std=0.02)
        self.register_buffer(
            "rank_thresh",
            torch.tensor([sum(self.rank_prior[j:]) for j in range(1, len(self.rank_tiers))],
                         dtype=torch.float32),
            persistent=True,
        )
        # telemetry: int64 so a .to(bfloat16) on the model cannot saturate it
        self.register_buffer(
            "_rank_counts", torch.zeros(len(self.rank_tiers), dtype=torch.long),
            persistent=False,
        )

    def _apply_dynamic_rank(self, hidden_states, q, k):
        B, S, _ = hidden_states.shape
        C = self.rank_chunk
        n = (S + C - 1) // C
        # CAUSAL router input: the token just BEFORE each chunk starts. Chunk 0
        # uses position 0, which is the only position it could use without
        # looking ahead.
        idx = torch.clamp(torch.arange(n, device=hidden_states.device) * C - 1, min=0)
        ctx = hidden_states.index_select(1, idx)                       # [B, n, H]
        probs = torch.softmax(self.rank_router(ctx).float(), dim=-1)
        tail = probs.flip(-1).cumsum(-1).flip(-1)                      # [B, n, T]
        above = tail[..., 1:] > self.rank_thresh.view(1, 1, -1)
        tier = torch.cumprod(above.long(), dim=-1).sum(dim=-1)         # [B, n]

        if self.training:
            with torch.no_grad():
                # QUANTILE controller, not proportional.
                #
                # A proportional update (theta += lr * (realized - target)) has
                # effectively unbounded gain once the router's distribution
                # concentrates: tail_k lives in [0,1], so near theta -> 1 a tiny
                # step flips a large fraction of chunks. Measured with the router
                # biased +3.0 toward the top rank, the r16/r32 split thrashed
                # between 0.000 and 0.457 while theta_2 saturated at 1.0.
                #
                # Setting theta_k to the (1 - target_k) EMPIRICAL QUANTILE of
                # tail_k instead is self-normalising: it lands on the target
                # occupancy regardless of how the distribution is shaped, and
                # cannot saturate. Cost is a sort over [B * n_chunks] -- a few
                # thousand elements, once per layer per step.
                #
                # EMA'd and applied to the NEXT step, so a chunk's rank still
                # depends only on statistics from strictly earlier steps.
                for j in range(1, len(self.rank_tiers)):
                    tgt = sum(self.rank_prior[j:])
                    qv = torch.quantile(
                        tail[..., j].flatten().float(), max(0.0, min(1.0, 1.0 - tgt)))
                    self.rank_thresh[j - 1] = (
                        (1.0 - self.rank_thresh_lr) * self.rank_thresh[j - 1]
                        + self.rank_thresh_lr * qv)

        ranks = torch.tensor(self.rank_tiers, device=tier.device)
        per_chunk = ranks[tier]                                        # [B, n]
        self._rank_counts.index_add_(
            0, tier.detach().reshape(-1),
            torch.ones(tier.numel(), device=tier.device, dtype=torch.long))
        per_tok = per_chunk.repeat_interleave(C, dim=1)[:, :S]         # [B, S]
        dk = q.shape[-1]
        mask = (torch.arange(dk, device=q.device).view(1, 1, 1, dk)
                < per_tok.view(B, S, 1, 1)).to(q.dtype)
        return q * mask, k * mask

    def _get_gate_dim(self) -> int:
        """Get the output dimension for gate projections.

        Returns:
            num_heads for scalar gating, num_heads * head_dim * expand_k for true KDA
            (per key channel), num_heads * head_dim * expand_v for legacy channel-wise.
        """
        if self._kda_delta:
            return self.num_heads * self.head_dim * self.expand_k
        if self._use_channel_wise_gating:
            return self.num_heads * self.head_dim * self.expand_v
        return self.num_heads

    def _apply_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply short convolution with state caching for incremental decoding.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            conv_state: Cached conv state from previous step, or None.
            use_cache: Whether to compute and return new conv state.

        Returns:
            Tuple of (output hidden states, new conv state or None).
        """
        if self.conv is None:
            return hidden_states, None

        # Intra-document masking (W2): during packed training (no cache), keep the depthwise
        # short conv from reaching across document boundaries — otherwise the kernel bleeds
        # the last (k-1) tokens of one document into the first tokens of the next.
        doc_start = getattr(self, "_intra_doc_start", None)
        if (
            doc_start is not None
            and not use_cache
            and conv_state is None
            and doc_start.shape[-1] == hidden_states.shape[1]
        ):
            return self._short_conv_intra_doc(hidden_states, doc_start), None

        batch_size, seq_len, _ = hidden_states.shape

        # Conv1d expects (batch, channels, seq_len)
        conv_input = hidden_states.transpose(1, 2)  # [B, H, L]

        # For incremental decoding, prepend cached conv state
        if conv_state is not None:
            conv_input = torch.cat([conv_state, conv_input], dim=2)

        # Apply convolution
        conv_out = self.conv(conv_input)

        # Extract output corresponding to current input positions
        if conv_state is not None:
            conv_out = conv_out[
                :, :, self.conv_kernel_size - 1 : self.conv_kernel_size - 1 + seq_len
            ]
        else:
            conv_out = conv_out[:, :, :seq_len]

        output = hidden_states + conv_out.transpose(1, 2)

        # Compute new conv state if caching
        new_conv_state = None
        if use_cache:
            if conv_state is not None:
                new_conv_state = conv_input[:, :, -(self.conv_kernel_size - 1) :].contiguous()
            else:
                cache_len = min(seq_len, self.conv_kernel_size - 1)
                if cache_len < self.conv_kernel_size - 1:
                    padding = torch.zeros(
                        batch_size,
                        self.hidden_size,
                        self.conv_kernel_size - 1 - cache_len,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    new_conv_state = torch.cat([padding, conv_input[:, :, -cache_len:]], dim=2)
                else:
                    new_conv_state = conv_input[:, :, -(self.conv_kernel_size - 1) :]
            new_conv_state = new_conv_state.contiguous()

        return output, new_conv_state

    def _short_conv_intra_doc(
        self, hidden_states: torch.Tensor, doc_start: torch.Tensor
    ) -> torch.Tensor:
        """Depthwise causal short conv that never crosses a document boundary.

        Computes the same causal depthwise convolution as ``self.conv`` (with left zero
        padding), but zeros any kernel tap that would reach into a previous packed document.
        A tap at distance ``i`` (0 = current token) is kept only where the token has at least
        ``i`` prior tokens *within its own document*. Matches the residual form of
        ``_apply_short_conv`` (``output = hidden_states + conv_out``).
        """
        B, L, H = hidden_states.shape
        k = self.conv_kernel_size
        # within-document position (0-based) = tokens of same-doc left context available
        idx = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L)
        run_start = torch.where(doc_start, idx, torch.zeros_like(idx))
        run_start = torch.cummax(run_start, dim=1).values
        within = idx - run_start  # [B, L]

        w = self.conv.weight  # [H, 1, k], depthwise
        out = torch.zeros_like(hidden_states)
        for i in range(k):  # i = distance back; tap uses weight column (k-1-i)
            coef = w[:, 0, k - 1 - i]  # [H]
            if i == 0:
                shifted = hidden_states
            else:
                # functional shift (no in-place slice write) so Inductor fuses all taps
                shifted = F.pad(hidden_states[:, :-i, :], (0, 0, i, 0))
            valid = (within >= i).unsqueeze(-1).to(hidden_states.dtype)  # [B, L, 1]
            out = out + shifted * coef * valid
        if self.conv.bias is not None:
            out = out + self.conv.bias
        return hidden_states + out

    def _compute_gates(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gate and beta values from hidden states.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tuple of (gate, beta) tensors. Shape depends on gating type:
            - Scalar gating: [batch, seq_len, num_heads]
            - Channel-wise: [batch, seq_len, num_heads, value_dim]
        """
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self._kda_delta:
            gate = gate.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_k)
        elif self._use_channel_wise_gating:
            v_dim = self.head_dim * self.expand_v
            gate = gate.view(batch_size, seq_len, self.num_heads, v_dim)
            beta = beta.view(batch_size, seq_len, self.num_heads, v_dim)

        return gate, beta

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DeltaNetCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        pause_query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DeltaNetCache]]:
        """Forward pass with HuggingFace-compatible signature.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Not used by DeltaNet but kept for interface compatibility.
            position_ids: Not used by DeltaNet (no position encoding needed).
            past_key_value: DeltaNetCache from previous step for incremental decoding.
            output_attentions: Not supported for linear attention, always returns None.
            use_cache: Whether to return the cache for incremental generation.
            pause_query_states: the pause stream (modules/pause_stream.py), same shape as
                ``hidden_states``. Q comes from these rows (through the same short conv)
                while k/v/gate/beta come from ``hidden_states``, which the block fills
                with the STATE stream for both halves. So the prediction half reads the
                state stream's recurrent state at each position and writes nothing to it:
                ``o_t = S_t q_pred,t``, the delta-rule form of "attend, write no K/V".

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, hidden_size)
                - None (attention weights not available for linear attention)
                - DeltaNetCache if use_cache=True, else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Extract cached states
        recurrent_state = None
        conv_state = None
        if past_key_value is not None:
            recurrent_state = past_key_value.recurrent_state
            conv_state = past_key_value.conv_state

        if pause_query_states is not None and (
            use_cache or past_key_value is not None or getattr(self, "dyn_rank", False)
        ):
            raise NotImplementedError(
                "pause_query_states supports full-sequence training/eval only (no cache, "
                "no dyn_rank)"
            )

        gate_src = hidden_states if pause_query_states is None else pause_query_states

        # Apply short convolution
        hidden_states, new_conv_state = self._apply_short_conv(hidden_states, conv_state, use_cache)

        # Projections
        if pause_query_states is not None:
            q = self.q_proj(self._apply_short_conv(pause_query_states, None, False)[0])
        else:
            q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to heads: [batch, seq_len, num_heads, dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_k)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_k)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_v)

        _fr = getattr(self, "_forced_rank", None)
        if _fr is not None:
            m = (torch.arange(q.shape[-1], device=q.device) < _fr).to(q.dtype)
            q, k = q * m, k * m
        elif getattr(self, "dyn_rank", False):
            q, k = self._apply_dynamic_rank(hidden_states, q, k)

        # Compute gates
        gate, beta = self._compute_gates(hidden_states, batch_size, seq_len)

        # Choose implementation.
        #
        # True Gated DeltaNet path (deltanet_true_delta_rule=True, scalar gating, CUDA):
        #   - training / full prefill -> chunk_gated_delta_rule (has a backward pass);
        #   - cached incremental decode -> fused_recurrent_gated_delta_rule (no backward,
        #     but decode is under no_grad). Both compute the same delta rule, so train and
        #     eval stay consistent.
        #
        # Legacy linear-attention path (default): FLA's fused_recurrent kernel has no
        # backward, so training falls to the naive scan; eval on CUDA uses the recurrent
        # kernel. Channel-wise gating (KDA) has no FLA delta kernel -> always naive.
        delta_kernel_ok = KDA_FLA_AVAILABLE if self._kda_delta else CHUNK_FLA_AVAILABLE
        delta_layout = self._kda_delta or not self._use_channel_wise_gating
        can_true_delta = (
            self._use_true_delta
            and delta_kernel_ok
            and hidden_states.is_cuda
            and delta_layout
        )
        if self._use_true_delta and hidden_states.is_cuda and delta_layout and not delta_kernel_ok:
            # Never fall through to _naive_forward here: it is a per-timestep Python loop
            # (~37x slower, measured July) AND it computes gated linear attention, not the
            # delta rule -- a run on it would be silently slow and the wrong architecture.
            raise RuntimeError(
                "deltanet_true_delta_rule=True on CUDA needs FLA's "
                + ("chunk_kda" if self._kda_delta else "chunk_gated_delta_rule")
                + ", which failed to import (pip install flash-linear-attention)."
            )
        if self._use_true_delta and delta_layout and not can_true_delta:
            # CPU: the exact recurrence (also the GPU parity reference). A scalar GDN gate
            # is the KDA rule with the gate broadcast over key channels. Before
            # 2026-10-31 scalar GDN fell through to _naive_forward below, i.e. gated
            # LINEAR attention with no delta correction and no q/k L2 norm, so every CPU
            # forward of a true-delta cell ran a different model (18M dn: loss 15.2).
            if not self._kda_delta:
                gate = gate.unsqueeze(-1).expand(*gate.shape, q.shape[-1])
            output, new_recurrent_state = self._naive_kda_forward(
                q, k, v, gate, beta, recurrent_state
            )
        elif can_true_delta:
            if self.training or past_key_value is None:
                output, new_recurrent_state = self._chunk_delta_forward(
                    q, k, v, gate, beta, recurrent_state, use_cache
                )
            else:
                output, new_recurrent_state = self._recurrent_delta_forward(
                    q, k, v, gate, beta, recurrent_state
                )
        else:
            use_fla = (
                FLA_AVAILABLE
                and hidden_states.is_cuda
                and not self._use_channel_wise_gating
                and not self.training  # FLA recurrent kernel has no backward pass
            )
            if use_fla:
                output, new_recurrent_state = self._fla_forward(
                    q, k, v, gate, beta, recurrent_state
                )
            else:
                output, new_recurrent_state = self._naive_forward(
                    q, k, v, gate, beta, recurrent_state
                )

        # Output projection
        if self.g_proj is not None:
            g = self.g_proj(gate_src).view(batch_size, seq_len, self.num_heads, -1)
            output = self.o_norm(output.reshape_as(g)) * F.silu(g)
        output = output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        # Build cache
        present = None
        if use_cache:
            present = DeltaNetCache(
                recurrent_state=new_recurrent_state,
                conv_state=new_conv_state,
            )

        return output, None, present

    def _naive_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Naive PyTorch implementation for reference/CPU fallback.

        This is a sequential implementation that processes one timestep at a time.
        It's slower than the FLA Triton kernels but works on any device.
        Handles both scalar gating (GatedDeltaNet) and channel-wise gating (KDA).

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim].
            k: Key tensor [batch, seq_len, num_heads, key_dim].
            v: Value tensor [batch, seq_len, num_heads, value_dim].
            gate: Gate tensor. Shape is [batch, seq_len, num_heads] for scalar gating,
                or [batch, seq_len, num_heads, value_dim] for channel-wise.
            beta: Beta tensor, same shape as gate.
            state: Previous recurrent state [batch, num_heads, key_dim, value_dim].

        Returns:
            Tuple of (output, new_state).
        """
        B, L, H, D_k = k.shape
        D_v = v.shape[-1]

        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)

        # Intra-document masking (W2): reset the recurrent state at document boundaries so it
        # does not leak across packed documents. This is the training path (FLA has no backward,
        # so training always lands here); the FLA inference path resets via cache instead.
        doc_start = getattr(self, "_intra_doc_start", None)
        if doc_start is not None and doc_start.shape[-1] != L:
            # Length mismatch (e.g. cached generation) -> ignore rather than misalign.
            doc_start = None

        outputs = []
        for t in range(L):
            # Reset state for any sequence whose document begins at this timestep.
            if doc_start is not None:
                reset_t = doc_start[:, t]
                if reset_t.any():
                    state = state * (~reset_t).to(state.dtype).view(B, 1, 1, 1)

            # Get current timestep
            k_t = k[:, t]  # [B, H, D_k]
            v_t = v[:, t]  # [B, H, D_v]
            q_t = q[:, t]  # [B, H, D_q]

            # Outer product k^T @ v -> [B, H, D_k, D_v]
            kv = torch.einsum("bhk,bhv->bhkv", k_t, v_t)

            # Get gates for this timestep and expand for broadcasting
            if self._use_channel_wise_gating:
                # Channel-wise: gate is [B, L, H, D_v]
                g_t = gate[:, t].unsqueeze(2)  # [B, H, 1, D_v]
                b_t = beta[:, t].unsqueeze(2)  # [B, H, 1, D_v]
            else:
                # Scalar: gate is [B, L, H]
                g_t = gate[:, t, :, None, None]  # [B, H, 1, 1]
                b_t = beta[:, t, :, None, None]  # [B, H, 1, 1]

            # Delta rule update: S_t = gate * S_{t-1} + beta * (k ⊗ v)
            state = g_t * state + b_t * kv

            # Query the state: o_t = q_t @ S_t
            out_t = torch.einsum("bhd,bhdv->bhv", q_t, state)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, L, H, D_v]
        return output, state

    def _fla_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FLA library forward with optimized Triton kernels.

        Only supports scalar gating (GatedDeltaNet). For channel-wise gating,
        use _naive_forward instead.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim].
            k: Key tensor [batch, seq_len, num_heads, key_dim].
            v: Value tensor [batch, seq_len, num_heads, value_dim].
            gate: Gate tensor [batch, seq_len, num_heads].
            beta: Beta tensor [batch, seq_len, num_heads].
            state: Previous recurrent state.

        Returns:
            Tuple of (output, new_state).
        """
        # Transpose for FLA: [B, H, L, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        gate = gate.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, H, L, 1]
        beta = beta.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, H, L, 1]

        output, new_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=state,
            output_final_state=True,
        )

        # Transpose back: [B, L, H, D]
        return output.transpose(1, 2), new_state

    # ------------------------------------------------------------------ #
    # True Gated DeltaNet (Qwen3-Next style) via FLA delta-rule kernels.
    #
    # These differ from _naive_forward/_fla_forward: the FLA delta kernels
    # apply the delta correction (S <- S(I - beta k kᵀ) + beta v kᵀ), L2-normalise
    # q/k in-kernel, use scale = 1/sqrt(K), and take the forget gate in *log space*.
    # Layout is the current FLA [B, T, H, D] convention (no head_first transpose).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _delta_log_gate(gate: torch.Tensor) -> torch.Tensor:
        """Convert a (0, 1) scalar gate to a log-space forget gate for FLA.

        ``gate`` is ``sigmoid(gate_proj(x))``; the FLA delta kernels expect ``g`` to be
        the log of the per-step decay. Compute in fp32 with a small floor so a fully
        saturated gate (decay -> 0) maps to a large-negative log rather than ``-inf``.
        """
        return torch.log(gate.float().clamp_min(1e-9))

    def _build_delta_cu_seqlens(self, doc_start: torch.Tensor) -> torch.Tensor:
        """Build ``cu_seqlens`` for a [B, L] doc-start mask flattened to one varlen batch.

        The chunk kernel's varlen mode expects batch size 1 and cumulative segment
        lengths ``[0, ..., B*L]``. Each packed row is an independent sequence, so we force
        a hard boundary at every row start (position ``b*L``) in addition to the per-doc
        starts, so the recurrent state never leaks across documents *or* across rows.
        """
        B, L = doc_start.shape
        ds = doc_start.clone()
        ds[:, 0] = True  # every packed row begins a fresh segment
        starts = torch.nonzero(ds.reshape(-1), as_tuple=False).flatten()
        total = torch.tensor([B * L], device=doc_start.device, dtype=starts.dtype)
        return torch.cat([starts, total]).to(torch.int32)

    def _chunk_delta_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """True Gated DeltaNet training/prefill kernel (chunk_gated_delta_rule).

        q/k/v: [B, L, H, D] (kept as-is). gate/beta: [B, L, H] scalar. Returns
        (output [B, L, H, D_v], final_state or None).
        """
        g = self._delta_log_gate(gate)  # [B, L, H], fp32 log-decay
        beta_in = beta.float()

        doc_start = getattr(self, "_intra_doc_start", None)
        B, L = q.shape[0], q.shape[1]
        if (
            doc_start is not None
            and doc_start.shape[-1] == L
            and state is None
            and not use_cache
        ):
            # Packed training: reset state at doc/row boundaries via varlen cu_seqlens.
            cu_seqlens = self._delta_cu_seqlens
            if cu_seqlens is None:
                cu_seqlens = self._build_delta_cu_seqlens(doc_start)
            elif not torch.compiler.is_compiling():
                # Same invariant MLA/attention varlen enforce: the shared tensor is built
                # for the full sequence. Skipped under compile (int() is a host sync).
                assert int(cu_seqlens[-1]) == B * L, (
                    f"stale DeltaNet cu_seqlens: cu[-1]={int(cu_seqlens[-1])} but buffer "
                    f"has {B * L} rows"
                )
            qf = q.reshape(1, B * L, self.num_heads, -1)
            kf = k.reshape(1, B * L, self.num_heads, -1)
            vf = v.reshape(1, B * L, self.num_heads, -1)
            gf = g.reshape(1, B * L, *g.shape[2:])  # [.., H] or [.., H, K] (KDA)
            bf = beta_in.reshape(1, B * L, self.num_heads)
            output, _ = self._chunk_kernel(
                qf,
                kf,
                vf,
                gf,
                bf,
                scale=None,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            output = output.reshape(B, L, self.num_heads, -1)
            return output.to(q.dtype), None

        output, new_state = self._chunk_kernel(
            q,
            k,
            v,
            g,
            beta_in,
            scale=None,
            initial_state=state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        return output.to(q.dtype), new_state

    @property
    def _chunk_kernel(self):
        return chunk_kda if self._kda_delta else chunk_gated_delta_rule

    def _naive_kda_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact KDA recurrence, per step, fp32 -- FLA's naive_recurrent_kda plus the
        in-kernel q/k L2 norm and the intra-doc reset the chunk path does via cu_seqlens.

        q/k: [B, L, H, K]; v: [B, L, H, V]; gate: [B, L, H, K] in (0, 1); beta: [B, L, H].
        """
        dtype = v.dtype
        B, L, H, K = q.shape
        q = F.normalize(q.float(), dim=-1) * K**-0.5
        k = F.normalize(k.float(), dim=-1)
        v, g, beta = v.float(), self._delta_log_gate(gate), beta.float()
        S = q.new_zeros(B, H, K, v.shape[-1]) if state is None else state.float()
        doc_start = getattr(self, "_intra_doc_start", None)
        reset = doc_start is not None and doc_start.shape[-1] == L and state is None
        out = torch.empty_like(v)
        for t in range(L):
            if reset and t > 0:
                S = S * (~doc_start[:, t]).to(S.dtype).view(B, 1, 1, 1)
            S = S * g[:, t].exp()[..., None]
            k_t = k[:, t]
            u = v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, S)
            S = S + torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k_t, u)
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
        return out.to(dtype), S

    def _recurrent_delta_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """True Gated DeltaNet cached-decode kernel (fused_recurrent_gated_delta_rule).

        Same conventions as :meth:`_chunk_delta_forward` so train and eval agree. No
        backward pass (decode only). Layout [B, L, H, D] — no transpose.
        """
        g = self._delta_log_gate(gate)
        beta_in = beta.float()
        recurrent = fused_recurrent_kda if self._kda_delta else fused_recurrent_gated_delta_rule
        output, new_state = recurrent(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g=g.contiguous(),
            beta=beta_in.contiguous(),
            scale=None,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return output.to(q.dtype), new_state


class KimiDeltaAttention(GatedDeltaNet):
    """Kimi Delta Attention (KDA) - channel-wise gating variant.

    Extends GatedDeltaNet with per-channel gating for finer-grained control.
    Instead of a scalar gate per head, KDA uses a separate gate value for
    each channel in the value dimension.

    This provides more expressiveness at the cost of additional parameters
    and computation. Used in Kimi-K2 (1T parameter MoE model).

    Two paths, chosen by ``deltanet_true_delta_rule``:

    - True (the real KDA, 2026-10-22): the delta rule with a per-KEY-channel forget
      gate and a per-head beta, via FLA's ``chunk_kda`` / ``fused_recurrent_kda``
      (``_naive_kda_forward`` on CPU). This is what the ``_kda`` sweep suffix runs.
    - False (legacy): channel-wise gate AND beta over the VALUE dim on the gated
      linear-attention update, naive PyTorch only. Not KDA despite the name.

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.
    """

    _use_channel_wise_gating: bool = True


# Registry for DeltaNet variants
DELTANET_CLASSES = {
    "gated": GatedDeltaNet,
    "kda": KimiDeltaAttention,
}


def get_deltanet_class(variant: str = "gated"):
    """Get the DeltaNet class for the given variant.

    Args:
        variant: One of "gated" (Qwen3-Next style) or "kda" (Kimi-K2 style).

    Returns:
        The DeltaNet class.
    """
    if variant not in DELTANET_CLASSES:
        raise ValueError(
            f"Unknown DeltaNet variant: {variant}. " f"Available: {list(DELTANET_CLASSES.keys())}"
        )
    return DELTANET_CLASSES[variant]
