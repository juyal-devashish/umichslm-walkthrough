"""Multi-head Latent Attention (MLA) implementation.

This module implements Multi-head Latent Attention as described in DeepSeek V2/V3:
https://arxiv.org/abs/2405.04434

MLA compresses Key-Value pairs into a low-rank latent representation, dramatically
reducing KV cache memory during inference. Key features:
- Low-rank KV compression via learned projections
- Decoupled RoPE for position encoding (separate from compressed KV)
- Optional Q compression for further memory savings
- Compatible with sliding window attention

The core idea is that instead of caching full K and V tensors, we cache:
1. A compressed latent vector c_kv (much smaller)
2. A small RoPE-encoded key component k_rope

During attention, K and V are reconstructed from the latent + RoPE keys.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_sink import extend_kv_with_sink, flash_attn_with_sink
from .kv_cache import MLADynamicCache, _AttentionLayerCacheView
from .msg_struct import msg_attention_forward
from .repo import RePoPositionPredictor, RePoPositionPredictorWithBase
from .rotary import RotaryEmbedding
from .sliding_window import create_sliding_window_mask, is_global_attention_layer

try:  # FlashAttention is GPU/CUDA-only; absent on CPU or non-flash builds.
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - exercised on non-flash hosts
    flash_attn_func = None
    flash_attn_varlen_func = None
    _HAS_FLASH_ATTN = False


class UMichSLMMLAAttention(nn.Module):
    """Multi-head Latent Attention with KV compression.

    This implements MLA as described in DeepSeek V2/V3, which compresses KV
    into a low-rank latent space for memory-efficient inference.

    Architecture:
        1. Compress hidden states to KV latent: c_kv = W_dkv @ x
        2. Decompress to get K (non-RoPE part) and V: k_nope, v = W_uk/uv @ c_kv
        3. Compute RoPE keys separately: k_rope = RoPE(W_kr @ x)
        4. Concatenate K: k = [k_nope, k_rope]
        5. Optionally compress Q similarly
        6. Compute attention with reconstructed K, V

    KV Cache stores only:
        - c_kv: [batch, seq, kv_lora_rank] - compressed latent
        - k_rope: [batch, seq, num_heads, qk_rope_head_dim] - RoPE keys

    Args:
        config: Model configuration with MLA parameters.
        layer_idx: Index of this layer (used for cache and SWA).
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # MLA dimensions from config
        self.kv_lora_rank = getattr(config, "mla_kv_lora_rank", 512)
        self.q_lora_rank = getattr(config, "mla_q_lora_rank", 0)  # 0 = no Q compression
        self.qk_nope_head_dim = getattr(config, "mla_qk_nope_head_dim", 128)
        self.qk_rope_head_dim = getattr(config, "mla_qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "mla_v_head_dim", 128)

        # Total Q/K head dimension
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # mla_shared_rope_key (P11, 2026-11-04): one decoupled RoPE key shared by all
        # heads (DeepSeek-V2 §2.1.2) instead of a per-head one. The per-token cache
        # drops from kv_lora_rank + H*rope to kv_lora_rank + rope elements.
        self.shared_rope_key = bool(getattr(config, "mla_shared_rope_key", False))
        # mla_diff_attn (P4, 2026-11-04): differential attention (arXiv 2410.05258).
        # Each head gets TWO queries against its one K/V; the output is
        # softmax(q1 k) v - lambda_h * softmax(q2 k) v, then a per-head RMSNorm and
        # the (1 - lambda_init) rescale. The 2H queries are laid out as a 2:1 GQA
        # group (query 2h and 2h+1 read KV head h), so flash runs it in one call and
        # the cache is unchanged. Costs one extra Q projection per MLA layer.
        self.diff_attn = bool(getattr(config, "mla_diff_attn", False))
        self.num_q_heads = 2 * self.num_heads if self.diff_attn else self.num_heads

        # Sliding Window Attention configuration
        self.sliding_window = getattr(config, "sliding_window", None)
        self.global_attention_every_n_layers = getattr(config, "global_attention_every_n_layers", 4)

        # Determine if this layer uses sliding window. The per-layer schedule
        # wins when present: in a [swa, swa, swa, mla] hybrid the MLA layer IS
        # the periodic full-context refresh, and deriving that from an
        # independent modulo let the two drift apart (see
        # is_global_attention_layer).
        if self.sliding_window is not None and layer_idx is not None:
            self.use_sliding_window = not is_global_attention_layer(
                layer_idx,
                self.global_attention_every_n_layers,
                getattr(config, "hybrid_attention_ratio", None),
            )
        else:
            self.use_sliding_window = False

        # global_layer_window (2026-08-25): band the FULL-ATTENTION layers.
        #
        # In the swa_globals_mla schedule the periodic "mla"/"attn" layers are the
        # only unbounded ones, so they carry the entire O(S^2) term -- 6 of 23 layers
        # at ar10m, and individually HEAVIER than a dense layer because MLA runs a
        # 192-dim QK against dense's head_dim 64. Capping them at a wide band keeps a
        # long-but-finite reach while removing the quadratic growth; the register bank
        # already supplies the position-free global path, so the unbounded reach may
        # be redundant. None (default) = unbounded, i.e. exactly the previous
        # behaviour, so this flag is bitwise inert unless set.
        _gw = getattr(config, "global_layer_window", None)
        if _gw is not None and layer_idx is not None and not self.use_sliding_window:
            if int(_gw) <= 0:
                raise ValueError(f"global_layer_window must be > 0 or None, got {_gw!r}")
            self.use_sliding_window = True
            self.sliding_window = int(_gw)

        # KV Compression projections
        # Down-projection: hidden_size -> kv_lora_rank (with layernorm)
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False,
        )
        self.kv_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=getattr(config, "rms_norm_eps", 1e-6))

        # Up-projections: kv_lora_rank -> K (nope) and V
        # K nope: kv_lora_rank -> num_heads * qk_nope_head_dim
        self.k_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )
        # V: kv_lora_rank -> num_heads * v_head_dim
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
        )

        # RoPE key projection: hidden_size -> num_heads * qk_rope_head_dim
        self.k_rope_proj = nn.Linear(
            self.hidden_size,
            (1 if self.shared_rope_key else self.num_heads) * self.qk_rope_head_dim,
            bias=False,
        )

        # Q projections (with optional compression)
        if self.q_lora_rank > 0:
            # Compressed Q path
            self.q_down_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
            )
            self.q_layernorm = nn.RMSNorm(
                self.q_lora_rank, eps=getattr(config, "rms_norm_eps", 1e-6)
            )
            # Q nope: q_lora_rank -> num_heads * qk_nope_head_dim
            self.q_up_proj = nn.Linear(
                self.q_lora_rank,
                self.num_q_heads * self.qk_nope_head_dim,
                bias=False,
            )
            # Q rope: q_lora_rank -> num_heads * qk_rope_head_dim
            self.q_rope_proj = nn.Linear(
                self.q_lora_rank,
                self.num_q_heads * self.qk_rope_head_dim,
                bias=False,
            )
        else:
            # Standard Q projection (no compression)
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_q_heads * self.qk_head_dim,
                bias=False,
            )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )

        # RoPE for position encoding (only applied to the rope dimensions).
        #
        # YaRN's temperature goes on the SOFTMAX SCALE here, not on cos/sin
        # (apply_mscale=False; see _yarn_logit_scale). Folded into cos/sin it
        # multiplies only the rope slice of the logit by mscale**2 (1.459 at 8x)
        # and leaves the 128 nope dims alone -- a reweighting of position against
        # content rather than a temperature. deltanet_mla took +0.05..0.17 bpb
        # INSIDE its trained 4K under that read while its GQA twins took
        # +0.01..0.02 (dsgl1022c/d). DeepSeek-V2/V3 apply mscale**2 to the
        # softmax scale for the same reason.
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
            scaling_factor=rope_scaling.get("factor", 1.0),
            rope_type=rope_scaling.get("type", "default"),
            yarn_beta_fast=rope_scaling.get("beta_fast", 32.0),
            yarn_beta_slow=rope_scaling.get("beta_slow", 1.0),
            yarn_mscale=rope_scaling.get("mscale", 1.0),
            original_max_position_embeddings=rope_scaling.get(
                "original_max_position_embeddings"
            ),
            apply_mscale=False,
        )

        # RePo (learned position embeddings) support
        self.use_repo = getattr(config, "use_repo", False)
        repo_start_layer = getattr(config, "repo_start_layer", 5)
        self._repo_active = self.use_repo and (
            layer_idx is not None and layer_idx >= repo_start_layer
        )

        if self._repo_active:
            repo_dim = getattr(config, "repo_dim", None) or (self.hidden_size // 8)
            use_base_positions = getattr(config, "repo_use_base_positions", True)

            if use_base_positions:
                self.repo_predictor = RePoPositionPredictorWithBase(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    repo_dim=repo_dim,
                    position_scale=1.0,
                    bias=False,
                )
            else:
                self.repo_predictor = RePoPositionPredictor(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    repo_dim=repo_dim,
                    bias=False,
                )
        else:
            self.repo_predictor = None

        # attn_logit_sinks (2026-10-27): learned per-head sink logit, one softmax
        # competitor with a zero value (modules/attn_sink.py). It competes with the
        # SCALED logits (YaRN temperature included), identically on all three paths.
        self.attn_logit_sink = (
            nn.Parameter(torch.zeros(self.num_heads))
            if getattr(config, "attn_logit_sinks", False)
            else None
        )

        # Pre-hero layer recipe knobs (configuration_umichslm.py, 2026-10-31).
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.mla_nope = bool(getattr(config, "mla_nope", False))
        self.out_gate = (
            nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)
            if getattr(config, "attn_output_gate", False)
            else None
        )
        qkn = bool(getattr(config, "attn_qk_norm", False))
        self.q_nope_norm = nn.RMSNorm(self.qk_nope_head_dim, eps=eps) if qkn else None
        self.k_nope_norm = nn.RMSNorm(self.qk_nope_head_dim, eps=eps) if qkn else None
        self.q_rope_norm = nn.RMSNorm(self.qk_rope_head_dim, eps=eps) if qkn else None
        self.k_rope_norm = nn.RMSNorm(self.qk_rope_head_dim, eps=eps) if qkn else None

        if self.diff_attn:
            if self.attn_logit_sink is not None:
                raise ValueError("mla_diff_attn does not compose with attn_logit_sinks")
            l = 0 if layer_idx is None else int(layer_idx)
            self.diff_lambda_init = 0.8 - 0.6 * math.exp(-0.3 * l)
            self.diff_lambda = nn.Parameter(torch.zeros(self.num_heads))
            self.diff_subln = nn.RMSNorm(self.v_head_dim, eps=eps)

    def _k_rope_heads(self, k_rope, bsz, q_len):
        """k_rope_proj output -> [B, S, H, rope]; a shared key is broadcast to all heads."""
        if self.shared_rope_key:
            return k_rope.view(bsz, q_len, 1, self.qk_rope_head_dim).expand(
                bsz, q_len, self.num_heads, self.qk_rope_head_dim
            )
        return k_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)

    def _diff_kv(self, k_states, v_states):
        """[B, H, T, D] K/V -> the 2:1 query grouping of mla_diff_attn (no-op if off)."""
        if not self.diff_attn:
            return k_states, v_states
        return k_states.repeat_interleave(2, dim=1), v_states.repeat_interleave(2, dim=1)

    def _diff_combine(self, attn_output):
        """[B, S, Hq, Dv] -> [B, S, H, Dv]: o1 - lambda * o2, per-head RMSNorm, rescale."""
        if not self.diff_attn:
            return attn_output
        b, s = attn_output.shape[:2]
        o = attn_output.view(b, s, self.num_heads, 2, -1)
        lam = (self.diff_lambda + self.diff_lambda_init).to(o.dtype).view(1, 1, -1, 1)
        o = o[..., 0, :] - lam * o[..., 1, :]
        return self.diff_subln(o) * (1.0 - self.diff_lambda_init)

    def _norm_q(self, q_nope, q_rope):
        """attn_qk_norm: per-head RMSNorm on each Q slice, before RoPE."""
        if self.q_nope_norm is None:
            return q_nope, q_rope
        return self.q_nope_norm(q_nope), self.q_rope_norm(q_rope)

    def _norm_k_nope(self, k_nope):
        return k_nope if self.k_nope_norm is None else self.k_nope_norm(k_nope)

    def _norm_k(self, k_nope, k_rope):
        """attn_qk_norm on K. k_nope is re-derived from the cached latent at decode, so
        it is normed again there (``_norm_k_nope``); k_rope is cached post-norm."""
        if self.k_nope_norm is None:
            return k_nope, k_rope
        return self.k_nope_norm(k_nope), self.k_rope_norm(k_rope)

    def _gate_out(self, attn_output, q_src):
        """attn_output_gate (G1): elementwise sigmoid gate from the query-side input."""
        if self.out_gate is None:
            return attn_output
        return attn_output * torch.sigmoid(self.out_gate(q_src))

    def _yarn_logit_scale(self) -> float:
        """YaRN's logit multiplier (``mscale**2``; 1.0 unless the rotary is yarn).

        Read off the live rotary on every call, so an in-place rescope
        (``unscale_windowed_rotary``) is honoured. RePo layers rotate through
        ``forward_continuous``, which has never applied the temperature, so they
        stay at 1.0.
        """
        if self._repo_active or self.mla_nope or self.rotary_emb.rope_type != "yarn":
            return 1.0
        return float(self.rotary_emb.mscale) ** 2

    def _softmax_scale(self) -> Optional[float]:
        """Softmax scale for SDPA / flash: None (the kernel's 1/sqrt(qk_head_dim)
        default, bitwise) unless YaRN is on."""
        m2 = self._yarn_logit_scale()
        return None if m2 == 1.0 else m2 / math.sqrt(self.qk_head_dim)

    # ---- KV cache ---------------------------------------------------------
    # MLA caches (c_kv, k_rope), not (k, v), so it cannot use the plain attention
    # path's cache code. What it is handed is normally this layer's own PAIR (the
    # model slices MLADynamicCache per layer before the call), but it also accepts
    # the cache object itself, which is how a caller outside the model loop -- a
    # probe, a single-layer harness -- can drive it.

    @property
    def _cache_index(self) -> int:
        """Cache slot for this layer. A layer built without an index owns slot 0."""
        return 0 if self.layer_idx is None else self.layer_idx

    def _cache_past_len(self, past_key_value) -> int:
        """How many tokens this layer already has cached.

        Wrong here means wrong RoPE positions for the incoming tokens, so it is
        read from the cache's own accounting rather than from any tensor shape
        the caller happens to hold.
        """
        if past_key_value is None:
            return 0
        if isinstance(past_key_value, MLADynamicCache):
            return int(past_key_value.get_seq_length(self._cache_index))
        if isinstance(past_key_value, _AttentionLayerCacheView):
            return int(past_key_value.get_seq_length())
        if (
            isinstance(past_key_value, tuple)
            and len(past_key_value) == 2
            and torch.is_tensor(past_key_value[0])
        ):
            return int(past_key_value[0].shape[1])
        return 0

    def _cache_update(self, past_key_value, c_kv, k_rope, use_cache):
        """Join this layer's past onto the new states and build its cache output.

        Returns ``(c_kv, k_rope, present_key_value)``, the first two being the
        FULL history this layer must attend over.

        The old code read the past by SUBSCRIPTING the cache object inside a
        `except (IndexError, KeyError, TypeError): pass`. HF hands generation a
        `DynamicCache`, which is not subscriptable in transformers 5.x, so the
        TypeError fired every step and MLA decoded with no past at all -- a
        silent 6.4e-1 max|dlogit| against a full forward, no traceback.
        """
        if isinstance(past_key_value, MLADynamicCache):
            if use_cache:
                c_kv, k_rope = past_key_value.update(c_kv, k_rope, self._cache_index)
                return c_kv, k_rope, past_key_value
            past = past_key_value[self._cache_index]
            if past is not None:
                c_kv = torch.cat([past[0], c_kv], dim=1)
                k_rope = torch.cat([past[1], k_rope], dim=1)
            return c_kv, k_rope, None

        if isinstance(past_key_value, _AttentionLayerCacheView):
            # The DeltaNet hybrids' cache (DeltaNetDynamicCache) hands its attention
            # layers a K/V view that concatenates on dim 2. Store (c_kv, k_rope) as
            # [B, 1, S, r] and [B, H, S, d] so its seq axis and length accounting hold.
            # Before 2026-10-31 this fell through below: nothing was stored and every
            # MLA layer of a deltanet_mla model decoded with no history at all.
            if use_cache:
                ck, kr = past_key_value.update(c_kv.unsqueeze(1), k_rope.transpose(1, 2))
                return ck.squeeze(1), kr.transpose(1, 2), past_key_value
            parent, idx = past_key_value.parent, past_key_value.layer_idx
            if idx in parent.key_cache:
                c_kv = torch.cat([parent.key_cache[idx].squeeze(1), c_kv], dim=1)
                k_rope = torch.cat([parent.value_cache[idx].transpose(1, 2), k_rope], dim=1)
            return c_kv, k_rope, None

        if (
            isinstance(past_key_value, tuple)
            and len(past_key_value) == 2
            and torch.is_tensor(past_key_value[0])
        ):
            c_kv = torch.cat([past_key_value[0], c_kv], dim=1)
            k_rope = torch.cat([past_key_value[1], k_rope], dim=1)

        return c_kv, k_rope, ((c_kv, k_rope) if use_cache else None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        pause_query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for MLA.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Attention mask [batch, 1, q_len, kv_len] or None.
            position_ids: Position indices [batch, seq_len].
            past_key_value: Cached (c_kv, k_rope) tensors for incremental decoding.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.

        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Attention weights (if output_attentions=True, else None)
                - Updated cache tuple (c_kv, k_rope) or None
        """
        bsz, q_len, _ = hidden_states.size()

        # pause_query_states (modules/pause_stream.py): Q from the pause rows, K/V from
        # hidden_states (the block passes the STATE stream for both halves).
        q_src = hidden_states if pause_query_states is None else pause_query_states

        # =============== Compute Q ===============
        if self.q_lora_rank > 0:
            # Compressed Q path
            q_compressed = self.q_down_proj(q_src)
            q_compressed = self.q_layernorm(q_compressed)
            q_nope = self.q_up_proj(q_compressed)  # [batch, seq, num_heads * qk_nope_head_dim]
            q_rope = self.q_rope_proj(q_compressed)  # [batch, seq, num_heads * qk_rope_head_dim]

            # Reshape
            q_nope = q_nope.view(bsz, q_len, self.num_q_heads, self.qk_nope_head_dim)
            q_rope = q_rope.view(bsz, q_len, self.num_q_heads, self.qk_rope_head_dim)
        else:
            # Standard Q (no compression)
            query_states = self.q_proj(q_src)
            query_states = query_states.view(bsz, q_len, self.num_q_heads, self.qk_head_dim)

            # Split into nope and rope parts
            q_nope = query_states[..., : self.qk_nope_head_dim]
            q_rope = query_states[..., self.qk_nope_head_dim :]

        q_nope, q_rope = self._norm_q(q_nope, q_rope)

        # =============== Compute KV ===============
        # Compress to latent
        c_kv = self.kv_down_proj(hidden_states)  # [batch, seq, kv_lora_rank]
        c_kv = self.kv_layernorm(c_kv)

        # Decompress K (non-RoPE part) and V
        k_nope = self.k_up_proj(c_kv)  # [batch, seq, num_heads * qk_nope_head_dim]
        v_states = self.v_up_proj(c_kv)  # [batch, seq, num_heads * v_head_dim]

        # Compute RoPE keys
        k_rope = self.k_rope_proj(hidden_states)  # [batch, seq, num_heads * qk_rope_head_dim]

        # Reshape
        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        v_states = v_states.view(bsz, q_len, self.num_heads, self.v_head_dim)
        k_rope = self._k_rope_heads(k_rope, bsz, q_len)

        k_nope, k_rope = self._norm_k(k_nope, k_rope)

        # =============== Handle KV Cache - Get past length ===============
        past_len = self._cache_past_len(past_key_value)

        # =============== Apply RoPE ===============
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        # Transpose for RoPE: [batch, num_heads, seq, head_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # Apply RoPE to the rope parts only (with RePo if active)
        if self._repo_active and self.repo_predictor is not None:
            # Use learned continuous positions
            if hasattr(self.repo_predictor, "delta_proj"):
                positions = self.repo_predictor(hidden_states, position_ids)
            else:
                positions = self.repo_predictor(hidden_states)
            q_rope, k_rope = self.rotary_emb.forward_continuous(q_rope, k_rope, positions)
        elif not self.mla_nope:
            q_rope, k_rope = self.rotary_emb(q_rope, k_rope, position_ids)

        # Transpose back: [batch, seq, num_heads, head_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # =============== Handle KV Cache - Concatenate ===============
        c_kv, k_rope, present_key_value = self._cache_update(
            past_key_value, c_kv, k_rope, use_cache
        )
        if past_len > 0:
            # Recompute k_nope and v from full c_kv
            k_nope = self.k_up_proj(c_kv)
            v_states = self.v_up_proj(c_kv)
            k_nope = k_nope.view(bsz, -1, self.num_heads, self.qk_nope_head_dim)
            k_nope = self._norm_k_nope(k_nope)
            v_states = v_states.view(bsz, -1, self.num_heads, self.v_head_dim)

        # =============== Concatenate K parts ===============
        # k_full: [batch, kv_len, num_heads, qk_head_dim]
        k_states = torch.cat([k_nope, k_rope], dim=-1)

        # Also concatenate Q parts
        # q_full: [batch, q_len, num_heads, qk_head_dim]
        q_states = torch.cat([q_nope, q_rope], dim=-1)

        # =============== Attention computation ===============
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        kv_len = k_states.shape[2]

        # Message-structure arms (P1/P2, modules/msg_struct.py) own the attention call:
        # pinned-key bias, digest keys and the raw window. No-op unless bound.
        if getattr(self, "_msg_ctx", None) is not None:
            out = msg_attention_forward(self, q_states, k_states, v_states, q_src, attention_mask)
            return out, None, present_key_value

        # Apply sliding window mask if needed
        if (
            self.use_sliding_window
            and self.sliding_window is not None
            and kv_len > self.sliding_window
        ):
            swa_mask = create_sliding_window_mask(
                seq_len=kv_len,
                window_size=self.sliding_window,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if q_len < kv_len:
                swa_mask = swa_mask[:, :, -q_len:, :]

            if attention_mask is not None:
                attention_mask = attention_mask + swa_mask
            else:
                attention_mask = swa_mask

        # Compute attention scores
        # Q: [batch, heads, q_len, qk_head_dim]
        # K: [batch, heads, kv_len, qk_head_dim]
        k_states, v_states = self._diff_kv(k_states, v_states)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(
            self.qk_head_dim
        )
        yarn_scale = self._yarn_logit_scale()
        if yarn_scale != 1.0:
            attn_weights = attn_weights * yarn_scale

        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=q_states.device),
                diagonal=kv_len - q_len + 1,
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

        # Softmax (with the sink: one extra column at logit s_h, dropped after,
        # since its value is zero).
        if self.attn_logit_sink is not None:
            col = self.attn_logit_sink.to(attn_weights.dtype).view(1, -1, 1, 1)
            attn_weights = torch.cat([attn_weights, col.expand(bsz, -1, q_len, 1)], dim=-1)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)[..., :-1]
            attn_weights = attn_weights.to(q_states.dtype)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply to values
        # V: [batch, heads, kv_len, v_head_dim]
        attn_output = torch.matmul(attn_weights, v_states)

        # Reshape output
        attn_output = self._diff_combine(attn_output.transpose(1, 2).contiguous())
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(self._gate_out(attn_output, q_src))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value

    def get_cache_size_per_token(self) -> int:
        """Return the cache size per token in elements.

        For MLA, we cache:
        - c_kv: kv_lora_rank elements
        - k_rope: num_heads * qk_rope_head_dim elements

        This is much smaller than standard attention which caches:
        - K: num_heads * head_dim
        - V: num_heads * head_dim

        Returns:
            Number of elements per token in the cache.
        """
        return self.kv_lora_rank + (1 if self.shared_rope_key else self.num_heads) * self.qk_rope_head_dim

    def get_standard_cache_size_per_token(self) -> int:
        """Return what the cache size would be for standard MHA.

        Returns:
            Number of elements per token for standard K+V cache.
        """
        return 2 * self.num_heads * self.qk_head_dim  # K + V


class UMichSLMMLASdpaAttention(UMichSLMMLAAttention):
    """MLA with SDPA backend.

    Uses PyTorch's scaled_dot_product_attention for potentially better
    performance on supported hardware.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        pause_query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using SDPA."""
        if output_attentions:
            # SDPA doesn't support attention weights
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                pause_query_states=pause_query_states,
            )

        bsz, q_len, _ = hidden_states.size()

        # pause_query_states (modules/pause_stream.py): Q from the pause rows, K/V from
        # hidden_states (the block passes the STATE stream for both halves).
        q_src = hidden_states if pause_query_states is None else pause_query_states

        # =============== Compute Q ===============
        if self.q_lora_rank > 0:
            q_compressed = self.q_down_proj(q_src)
            q_compressed = self.q_layernorm(q_compressed)
            q_nope = self.q_up_proj(q_compressed)
            q_rope = self.q_rope_proj(q_compressed)
            q_nope = q_nope.view(bsz, q_len, self.num_q_heads, self.qk_nope_head_dim)
            q_rope = q_rope.view(bsz, q_len, self.num_q_heads, self.qk_rope_head_dim)
        else:
            query_states = self.q_proj(q_src)
            query_states = query_states.view(bsz, q_len, self.num_q_heads, self.qk_head_dim)
            q_nope = query_states[..., : self.qk_nope_head_dim]
            q_rope = query_states[..., self.qk_nope_head_dim :]

        q_nope, q_rope = self._norm_q(q_nope, q_rope)

        # =============== Compute KV ===============
        c_kv = self.kv_down_proj(hidden_states)
        c_kv = self.kv_layernorm(c_kv)
        k_nope = self.k_up_proj(c_kv)
        v_states = self.v_up_proj(c_kv)
        k_rope = self.k_rope_proj(hidden_states)

        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        v_states = v_states.view(bsz, q_len, self.num_heads, self.v_head_dim)
        k_rope = self._k_rope_heads(k_rope, bsz, q_len)

        k_nope, k_rope = self._norm_k(k_nope, k_rope)

        # =============== Handle KV Cache - Get past length ===============
        past_len = self._cache_past_len(past_key_value)

        # =============== Apply RoPE ===============
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + q_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # Apply RoPE (with RePo if active)
        if self._repo_active and self.repo_predictor is not None:
            # Use learned continuous positions
            if hasattr(self.repo_predictor, "delta_proj"):
                positions = self.repo_predictor(hidden_states, position_ids)
            else:
                positions = self.repo_predictor(hidden_states)
            q_rope, k_rope = self.rotary_emb.forward_continuous(q_rope, k_rope, positions)
        elif not self.mla_nope:
            q_rope, k_rope = self.rotary_emb(q_rope, k_rope, position_ids)

        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # =============== Handle KV Cache - Concatenate ===============
        c_kv, k_rope, present_key_value = self._cache_update(
            past_key_value, c_kv, k_rope, use_cache
        )
        if past_len > 0:
            # Recompute k_nope and v from full c_kv
            k_nope = self.k_up_proj(c_kv)
            v_states = self.v_up_proj(c_kv)
            k_nope = k_nope.view(bsz, -1, self.num_heads, self.qk_nope_head_dim)
            k_nope = self._norm_k_nope(k_nope)
            v_states = v_states.view(bsz, -1, self.num_heads, self.v_head_dim)

        # =============== Concatenate and transpose ===============
        k_states = torch.cat([k_nope, k_rope], dim=-1)
        q_states = torch.cat([q_nope, q_rope], dim=-1)

        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        kv_len = k_states.shape[2]

        # Message-structure arms (P1/P2, modules/msg_struct.py) own the attention call:
        # pinned-key bias, digest keys and the raw window. No-op unless bound.
        if getattr(self, "_msg_ctx", None) is not None:
            out = msg_attention_forward(self, q_states, k_states, v_states, q_src, attention_mask)
            return out, None, present_key_value

        # Apply sliding window mask
        if (
            self.use_sliding_window
            and self.sliding_window is not None
            and kv_len > self.sliding_window
        ):
            swa_mask = create_sliding_window_mask(
                seq_len=kv_len,
                window_size=self.sliding_window,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if q_len < kv_len:
                swa_mask = swa_mask[:, :, -q_len:, :]
            if attention_mask is not None:
                attention_mask = attention_mask + swa_mask
            else:
                attention_mask = swa_mask

        if self.attn_logit_sink is not None:
            k_states, v_states, attention_mask = extend_kv_with_sink(
                k_states, v_states, attention_mask, self.attn_logit_sink, q_len
            )

        # Use SDPA
        k_states, v_states = self._diff_kv(k_states, v_states)
        is_causal = attention_mask is None and q_len > 1
        attn_output = F.scaled_dot_product_attention(
            q_states,
            k_states,
            v_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self._softmax_scale(),
        )

        attn_output = self._diff_combine(attn_output.transpose(1, 2).contiguous())
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(self._gate_out(attn_output, q_src))

        return attn_output, None, present_key_value


class UMichSLMMLAFlashAttention2(UMichSLMMLASdpaAttention):
    """MLA with a FlashAttention-2 backend.

    The eager parent materialises the full ``[B, H, S, S]`` score matrix
    (5 GB bf16 / 10 GB fp32 at 50m/B16/S4096, ~34 GB at 1B/32K) for the periodic
    GLOBAL MLA layers — the SWA layers are windowed-flash, but MLA had no flash
    path, so under ``flash_attention_2`` it silently fell back to eager. This
    class runs the reconstructed per-head Q/K/V through ``flash_attn_func``
    (full-causal or ``sliding_window``) / ``flash_attn_varlen_func`` (packed
    intra-document masking, block-diagonal per doc), so the global layers never
    materialise the score matrix — the whole point at long context.

    Cases flash cannot express — an additive ``attention_mask`` with no
    ``cu_seqlens`` (a shared-prefix arm forced off varlen, ``_intra_doc_force_sdpa``),
    ``output_attentions``, cache decode, CPU, the single-token step — defer to
    ``super().forward``. That parent is now ``UMichSLMMLASdpaAttention`` (NOT the
    eager base): SDPA consumes the additive mask on its mem-efficient O(S) backend
    instead of materialising ``[B, H, S, S]`` eagerly. This matters for the prefix
    arms (``swa_globals``/``swa_hybrid32``/``swa_deltabank``), whose GLOBAL MLA
    layers hit this fallback EVERY step under intra-doc masking — before this they
    silently ran the eager O(S²) path, nullifying the flash win. Same mask, same
    softmax, so the loss curve / sweep comparability are unchanged. (SDPA itself
    still defers to the eager base only for ``output_attentions``, which SDPA
    cannot return.)

    MLA's ``v_head_dim`` may differ from ``qk_head_dim``; FlashAttention wants a
    common head dim, so V is zero-padded up to ``qk_head_dim`` and the output is
    sliced back — exact, since the padded value columns contribute only zeros.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        # Populated per-step by the model (modeling_umichslm.py) when packed
        # intra-document masking is active; None => plain causal within the seq.
        self._intra_doc_cu_seqlens = None
        self._intra_doc_max_seqlen = None
        self._intra_doc_force_sdpa = False
        # Lets the model treat this layer like the SWA varlen layers: no shared
        # additive [B,1,S,S] mask needs to be built on its behalf.
        self._supports_varlen_intra_doc = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
        pause_query_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Defer anything the flash path does not cover to the exact eager parent.
        # The flash path honours only (a) plain causal / sliding-window and (b)
        # packed intra-doc via cu_seqlens. An additive attention_mask with no
        # cu_seqlens (ragged batch, or a shared-prefix arm forced onto SDPA)
        # cannot be expressed by flash here, so it falls back.
        needs_additive_mask = attention_mask is not None and self._intra_doc_cu_seqlens is None
        if (
            not _HAS_FLASH_ATTN
            or output_attentions
            or use_cache
            or past_key_value is not None
            or not hidden_states.is_cuda
            or hidden_states.size(1) == 1
            or self._intra_doc_force_sdpa
            or needs_additive_mask
            or getattr(self, "_msg_ctx", None) is not None
        ):
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                pause_query_states=pause_query_states,
            )

        bsz, q_len, _ = hidden_states.size()

        # pause_query_states (modules/pause_stream.py): Q from the pause rows, K/V from
        # hidden_states (the block passes the STATE stream for both halves).
        q_src = hidden_states if pause_query_states is None else pause_query_states

        # =============== Compute Q (mirrors the eager parent) ===============
        if self.q_lora_rank > 0:
            q_compressed = self.q_layernorm(self.q_down_proj(q_src))
            q_nope = self.q_up_proj(q_compressed).view(
                bsz, q_len, self.num_q_heads, self.qk_nope_head_dim
            )
            q_rope = self.q_rope_proj(q_compressed).view(
                bsz, q_len, self.num_q_heads, self.qk_rope_head_dim
            )
        else:
            query_states = self.q_proj(q_src).view(
                bsz, q_len, self.num_q_heads, self.qk_head_dim
            )
            q_nope = query_states[..., : self.qk_nope_head_dim]
            q_rope = query_states[..., self.qk_nope_head_dim :]

        q_nope, q_rope = self._norm_q(q_nope, q_rope)

        # =============== Compute KV ===============
        c_kv = self.kv_layernorm(self.kv_down_proj(hidden_states))
        k_nope = self.k_up_proj(c_kv).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        v_states = self.v_up_proj(c_kv).view(bsz, q_len, self.num_heads, self.v_head_dim)
        k_rope = self._k_rope_heads(self.k_rope_proj(hidden_states), bsz, q_len)

        k_nope, k_rope = self._norm_k(k_nope, k_rope)

        # =============== Apply RoPE (rope parts only) ===============
        if position_ids is None:
            position_ids = (
                torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            )
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        if self._repo_active and self.repo_predictor is not None:
            if hasattr(self.repo_predictor, "delta_proj"):
                positions = self.repo_predictor(hidden_states, position_ids)
            else:
                positions = self.repo_predictor(hidden_states)
            q_rope, k_rope = self.rotary_emb.forward_continuous(q_rope, k_rope, positions)
        elif not self.mla_nope:
            q_rope, k_rope = self.rotary_emb(q_rope, k_rope, position_ids)
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        # =============== Assemble Q/K/V in flash layout [B, S, H, D] ==========
        q_states = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, H, qk_head_dim]
        k_states = torch.cat([k_nope, k_rope], dim=-1)  # [B, S, H, qk_head_dim]

        # FlashAttention needs a common head dim; pad V up to qk_head_dim.
        pad = self.qk_head_dim - self.v_head_dim
        if pad > 0:
            v_states = F.pad(v_states, (0, pad))

        orig_dtype = q_states.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q_states = q_states.to(torch.bfloat16)
            k_states = k_states.to(torch.bfloat16)
            v_states = v_states.to(torch.bfloat16)

        window = (
            (self.sliding_window, 0)
            if (self.use_sliding_window and self.sliding_window is not None)
            else (-1, -1)
        )

        cu_seqlens = self._intra_doc_cu_seqlens
        if cu_seqlens is not None and not self._intra_doc_force_sdpa:
            # Packed intra-document masking: independent causal attention within
            # each document segment, O(S) memory (block-diagonal, never crossing
            # a boundary). Requires q_len == kv_len (training, no cache).
            hd = q_states.shape[-1]
            cu_seqlens = cu_seqlens.to(q_states.device)
            # Self-enforce the non-local invariant a compact MoD member would break
            # (see UMichSLMModel._run_gathered_mod_block): cu_seqlens is built for the
            # FULL sequence, so its last entry MUST equal this buffer's row count. A
            # stale full-S cu_seqlens against a compact [B*k,...] buffer reads past the
            # end -> CUDA illegal access. A compact member must set
            # _intra_doc_force_sdpa=True to divert here. Skipped under compile:
            # int(cu[-1]) is a host sync that would force a graph break every step.
            if not torch.compiler.is_compiling():
                assert int(cu_seqlens[-1]) == bsz * q_len, (
                    f"stale intra-doc cu_seqlens: cu[-1]={int(cu_seqlens[-1])} but buffer "
                    f"has {bsz * q_len} rows (bsz={bsz}, q_len={q_len}) -- a compact/gathered "
                    "member must set _intra_doc_force_sdpa=True instead of taking varlen"
                )
            if self.attn_logit_sink is not None:
                attn_output = flash_attn_with_sink(
                    q_states.reshape(bsz * q_len, self.num_q_heads, hd),
                    k_states.reshape(bsz * q_len, self.num_heads, hd),
                    v_states.reshape(bsz * q_len, self.num_heads, hd),
                    self.attn_logit_sink,
                    softmax_scale=self._softmax_scale(),
                    window=window,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=int(self._intra_doc_max_seqlen),
                ).reshape(bsz, q_len, self.num_q_heads, hd)
            else:
                attn_output = flash_attn_varlen_func(
                    q_states.reshape(bsz * q_len, self.num_q_heads, hd),
                    k_states.reshape(bsz * q_len, self.num_heads, hd),
                    v_states.reshape(bsz * q_len, self.num_heads, hd),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=int(self._intra_doc_max_seqlen),
                    max_seqlen_k=int(self._intra_doc_max_seqlen),
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    softmax_scale=self._softmax_scale(),
                    causal=True,
                    window_size=window,
                ).reshape(bsz, q_len, self.num_q_heads, hd)
        elif self.attn_logit_sink is not None:
            attn_output = flash_attn_with_sink(
                q_states,
                k_states,
                v_states,
                self.attn_logit_sink,
                softmax_scale=self._softmax_scale(),
                window=window,
            )
        else:
            attn_output = flash_attn_func(
                q_states,
                k_states,
                v_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self._softmax_scale(),
                causal=True,
                window_size=window,
            )

        # Slice off the padded value dims and project.
        #
        # NOTE: this slice-then-reshape is a strided (non-contiguous) gather that
        # Inductor lowers to an unmasked pointwise kernel; its pointwise
        # autotuner can benchmark an over-reading grid config that reads past the
        # padded flash buffer -> illegal memory access / NaN grads on some GPUs
        # (A6000). The fix lives in the titan wrapper (apply_compile sets
        # torch._inductor.config.triton.autotune_pointwise = False), NOT here: a
        # source-level "fold the unpad into a zero-padded o_proj weight" was
        # tried and DEFEATED -- Inductor algebraically drops the zero columns and
        # reconstructs this exact strided gather. A real 192-wide o_proj would
        # avoid it but changes the parameter count (breaks matched-N), and
        # dropping the pad needs FA3/Hopper unequal-V support (unavailable on
        # Ampere/FA2). So the flag is the correct, param-preserving fix.
        attn_output = self._diff_combine(attn_output[..., : self.v_head_dim].to(orig_dtype))
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(self._gate_out(attn_output, q_src))

        present_key_value = None
        return attn_output, None, present_key_value


MLA_ATTENTION_CLASSES = {
    "eager": UMichSLMMLAAttention,
    "sdpa": UMichSLMMLASdpaAttention,
    "flash_attention_2": UMichSLMMLAFlashAttention2,
}


def get_mla_attention_class(attn_implementation: str = "eager"):
    """Get the MLA attention class for the given implementation.

    Args:
        attn_implementation: One of "eager", "sdpa", or "flash_attention_2".

    Returns:
        The MLA attention class.
    """
    if attn_implementation not in MLA_ATTENTION_CLASSES:
        raise ValueError(
            f"Unknown attention implementation for MLA: {attn_implementation}. "
            f"Available: {list(MLA_ATTENTION_CLASSES.keys())}"
        )
    return MLA_ATTENTION_CLASSES[attn_implementation]
