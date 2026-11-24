"""Rotary Position Embeddings (RoPE) and Polar Position Embeddings (PoPE).

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - PoPE: https://arxiv.org/abs/2509.10534
    - YaRN: https://arxiv.org/abs/2309.00071
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# Bounded memo for the largest position seen per position_ids tensor. Every
# layer is handed the same ``position_ids`` object, so without this the
# ``.max().item()`` below runs once per layer -- 12 host syncs per forward on the
# sweep shapes, which was the single largest sync source in the model, larger than
# every bank and router site combined. Keyed on identity + version so a mutated or
# different tensor always misses. A false hit would only ever UNDER-size the cache,
# which surfaces loudly as an index error from the gather rather than silently
# wrong positions.
_MAX_POS_MEMO: dict = {}


def _position_upper_bound(position_ids: torch.Tensor) -> int:
    key = (
        position_ids.data_ptr(),
        tuple(position_ids.shape),
        position_ids.dtype,
        position_ids.device,
        position_ids._version,
    )
    cached = _MAX_POS_MEMO.get(key)
    if cached is None:
        cached = int(position_ids.max().item()) + 1
        # Small bounded memo, not a single entry: the block-atomic MoD runner
        # hands its members a GATHERED position tensor, so a forward alternates
        # between the full position_ids and the compact one. A single slot
        # thrashes between them and re-syncs on every switch (3 per forward
        # instead of 2). Keys are plain scalars, so nothing here keeps a tensor
        # alive.
        if len(_MAX_POS_MEMO) >= 8:
            _MAX_POS_MEMO.clear()
        _MAX_POS_MEMO[key] = cached
    return cached


def _maybe_extend_cache(module: nn.Module, position_ids: torch.Tensor) -> None:
    """Grow ``module``'s cos/sin cache if ``position_ids`` runs past it.

    Reading the largest position requires ``.max().item()``, a host sync that
    is also a hard Dynamo graph break. Under ``torch.compile`` we skip the
    check entirely and rely on the cache built at construction time (sized to
    ``max_position_embeddings``); a position past the end then surfaces as an
    index error from the gather below rather than being silently wrong.

    In eager the check is memoized per position_ids tensor (see above), so it
    costs one sync per forward instead of one per layer.
    """
    if torch.compiler.is_compiling():
        return
    # Fast path: the construction-time cache already covers every position the
    # model is configured for, so there is nothing to grow and no need to look.
    if module.max_seq_len_cached >= getattr(module, "max_position_embeddings", 0):
        max_pos = getattr(module, "max_position_embeddings", 0)
        if max_pos and _position_upper_bound(position_ids) <= max_pos:
            return
    seq_len = _position_upper_bound(position_ids)
    if seq_len > module.max_seq_len_cached:
        module._set_cos_sin_cache(seq_len, device=position_ids.device)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE encodes relative position information through rotation of query/key vectors.
    This provides better length extrapolation than absolute positional embeddings.

    Args:
        dim: Dimension of the rotary embedding (typically head_dim). Passing q/k with a
            wider last dim is partial RoPE: the leading ``dim`` dims are rotated and the
            rest pass through as position-free content (``partial_rotary_factor``).
        max_position_embeddings: Maximum sequence length supported.
        base: Base for the frequency computation.
        scaling_factor: Factor to scale positions (for context extension).
        rope_type: Type of RoPE scaling ("default", "linear", "dynamic", "yarn").
        yarn_beta_fast: High-frequency threshold for YaRN ramp (default 32).
        yarn_beta_slow: Low-frequency threshold for YaRN ramp (default 1).
        yarn_mscale: Multiplier for YaRN attention temperature scaling.
        original_max_position_embeddings: The context length the model was
            TRAINED at, i.e. the length YaRN's correction range is defined over.
            ``None`` (the default, and every checkpoint before 2026-09) falls
            back to ``max_position_embeddings``. Set it when extending: HF
            convention is ``max_position_embeddings`` = the NEW length (which
            sizes the cos/sin cache) and ``rope_scaling.
            original_max_position_embeddings`` = the base length. Without the
            split, a YaRN run must keep ``max_position_embeddings`` at the base
            length, so the cache is too short for the extended sequence -- and
            under torch.compile ``_maybe_extend_cache`` is skipped, so the RoPE
            gather indexes past the end of the cache.
        apply_mscale: Fold the YaRN temperature into cos/sin (default). Squared
            through q.k, that scales the logit by ``mscale**2`` -- a uniform
            temperature ONLY when every q/k dim is rotated. A caller that rotates
            a slice of the head (MLA's decoupled rope dims) passes False and
            applies ``mscale**2`` to its softmax scale instead; ``mscale`` still
            reports the temperature either way. Partial RoPE does not need that: the
            pass-through dims are scaled by ``mscale`` here too, so the temperature
            stays uniform over the whole head.

    Shape:
        - Query/Key Input: (batch, num_heads, seq_len, head_dim)
        - Position IDs: (batch, seq_len)
        - Output: Same as input
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale: float = 1.0,
        original_max_position_embeddings: Optional[int] = None,
        apply_mscale: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.apply_mscale = apply_mscale
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = int(
            original_max_position_embeddings or max_position_embeddings
        )
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_mscale = yarn_mscale

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Attention temperature scale for YaRN (1.0 for all other types)
        self.mscale = self._compute_yarn_mscale() if rope_type == "yarn" else 1.0

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def reset_buffers(self, device: Optional[torch.device] = None) -> None:
        """Rebuild the non-persistent RoPE buffers (inv_freq + cos/sin cache).

        These buffers are ``persistent=False`` so they are absent from saved
        checkpoints; transformers' meta-device fast-init in ``from_pretrained``
        leaves them as uninitialized memory, so they must be recomputed after
        load. Idempotent and cheap — the model's ``_init_weights`` calls this on
        every module during the post-load init pass.
        """
        if device is None:
            device = self.inv_freq.device
        inv_freq = self._compute_inv_freq().to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, device=device)

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for RoPE."""
        if self.rope_type == "yarn":
            return self._compute_yarn_inv_freq()
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        return inv_freq

    # ------------------------------------------------------------------
    # YaRN helpers
    # ------------------------------------------------------------------

    def _yarn_find_correction_dim(self, num_rotations: float) -> float:
        """Map a rotation count to the corresponding frequency dimension index."""
        return (
            self.dim
            * math.log(self.original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(self.base))

    def _yarn_find_correction_range(self) -> Tuple[int, int]:
        """Return the (low, high) dimension indices for the YaRN ramp."""
        low = math.floor(self._yarn_find_correction_dim(self.yarn_beta_fast))
        high = math.ceil(self._yarn_find_correction_dim(self.yarn_beta_slow))
        return max(low, 0), min(high, self.dim // 2 - 1)

    def _yarn_linear_ramp_mask(self, low: int, high: int) -> torch.Tensor:
        """Linear ramp from 0→1 between dim indices low and high."""
        if low == high:
            high += 0.001  # avoid division by zero
        ramp = (torch.arange(self.dim // 2, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(ramp, 0.0, 1.0)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute YaRN-blended inverse frequencies.

        Blends two extremes via a per-dimension ramp:
          - Low-frequency dims  → linear interpolation (inv_freq / scale)
          - High-frequency dims → no scaling           (inv_freq)
        """
        scale = self.scaling_factor
        base_inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        low, high = self._yarn_find_correction_range()
        # The ramp rises 0 -> 1 with the pair INDEX, and a higher index is a LOWER
        # frequency, so the ramp is the interpolation weight. Until 2026-09-18 it was
        # used as the extrapolation weight, which mirrored YaRN: the highest-frequency
        # dims (local position, copying) were squeezed 8x and the lowest were left
        # alone, so nothing was extended and short-range attention broke (dense ar10m
        # at 0-256 tokens: 4.87 -> 6.74 nats). Matches transformers'
        # `_compute_yarn_parameters`, which uses `1 - ramp` for extrapolation.
        interp_weight = self._yarn_linear_ramp_mask(low, high)

        inv_freq_inter = base_inv_freq / scale   # linear interp (low-freq)
        inv_freq_extra = base_inv_freq            # no scaling    (high-freq)
        return inv_freq_inter * interp_weight + inv_freq_extra * (1.0 - interp_weight)

    def _compute_yarn_mscale(self) -> float:
        """Attention temperature scaling factor for YaRN.

        Compensates for the distribution shift caused by context extension.
        mscale = 0.1 * yarn_mscale * ln(scale) + 1  (1.0 when scale <= 1).
        """
        scale = self.scaling_factor
        if scale <= 1.0:
            return 1.0
        return 0.1 * self.yarn_mscale * math.log(scale) + 1.0

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Precompute cos and sin values for the given sequence length."""
        self.max_seq_len_cached = seq_len

        if device is None:
            device = self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Start with base inv_freq
        inv_freq = self.inv_freq

        # Apply scaling if needed
        if self.rope_type == "linear":
            t = t / self.scaling_factor
        elif self.rope_type == "dynamic" and seq_len > self.max_position_embeddings:
            # Dynamic NTK-aware scaling
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))

        freqs = torch.outer(t, inv_freq.to(device))
        # Create [cos, cos] and [sin, sin] for the full dimension
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            position_ids: Position indices of shape (batch, seq_len).

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        _maybe_extend_cache(self, position_ids)

        # Gather cos/sin values for the positions
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # YaRN: scale cos/sin by the attention temperature factor
        scale = 1.0
        if self.rope_type == "yarn" and self.apply_mscale:
            cos = cos * self.mscale
            sin = sin * self.mscale
            scale = self.mscale

        return self._rotate(q, cos, sin, scale), self._rotate(k, cos, sin, scale)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float):
        """Rotate the leading ``self.dim`` dims of x; pass the rest through times ``scale``.

        ``scale`` is the YaRN temperature already folded into cos/sin, applied to the
        pass-through dims as well so that q.k picks up ``mscale**2`` uniformly.
        """
        if x.shape[-1] == self.dim:
            return (x * cos) + (rotate_half(x) * sin)
        rot, rest = x[..., : self.dim], x[..., self.dim :]
        rot = (rot * cos) + (rotate_half(rot) * sin)
        return torch.cat((rot, rest * scale if scale != 1.0 else rest), dim=-1)

    def forward_continuous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings using continuous (non-integer) positions.

        This method computes cos/sin on-the-fly for arbitrary continuous position
        values, enabling learned position assignments (RePo).

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            positions: Continuous position values. Can be:
                - (batch, seq_len): Same positions for all heads
                - (batch, seq_len, 1): Same positions for all heads (broadcastable)
                - (batch, seq_len, num_heads): Per-head positions

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]

        # Handle different position shapes
        # q/k have shape (batch, num_heads, seq_len, head_dim)
        # We need cos/sin to have shape (batch, num_heads_or_1, seq_len, head_dim)
        if positions.dim() == 2:
            # (batch, seq_len) -> (batch, 1, seq_len, 1)
            # Broadcasts to all heads
            positions_q = positions.unsqueeze(1).unsqueeze(-1)
            positions_k = positions_q
        elif positions.dim() == 3:
            # (batch, seq_len, num_heads) -> (batch, num_heads, seq_len, 1)
            positions_q = positions.transpose(1, 2).unsqueeze(-1)

            # Handle GQA: k may have fewer heads than q
            if num_kv_heads == num_q_heads:
                positions_k = positions_q
            else:
                # For GQA, average positions across query heads that share the same KV head
                num_groups = num_q_heads // num_kv_heads
                # positions_q: (batch, num_q_heads, seq_len, 1)
                # Reshape to (batch, num_kv_heads, num_groups, seq_len, 1) and average
                batch_size, _, seq_len, _ = positions_q.shape
                positions_k = positions_q.view(batch_size, num_kv_heads, num_groups, seq_len, 1)
                positions_k = positions_k.mean(dim=2)  # (batch, num_kv_heads, seq_len, 1)
        else:
            raise ValueError(f"positions must have 2 or 3 dimensions, got {positions.dim()}")

        # Compute frequencies: position * inv_freq
        # inv_freq: (dim // 2,)
        inv_freq = self.inv_freq.to(positions_q.dtype).to(positions_q.device)

        # Compute cos/sin for queries
        freqs_q = positions_q * inv_freq  # (batch, heads, seq_len, dim // 2)
        emb_q = torch.cat((freqs_q, freqs_q), dim=-1)  # (batch, heads, seq_len, dim)
        cos_q = emb_q.cos()
        sin_q = emb_q.sin()

        # Compute cos/sin for keys (may differ in GQA)
        if positions_k is positions_q:
            cos_k, sin_k = cos_q, sin_q
        else:
            freqs_k = positions_k * inv_freq
            emb_k = torch.cat((freqs_k, freqs_k), dim=-1)
            cos_k = emb_k.cos()
            sin_k = emb_k.sin()

        return self._rotate(q, cos_q, sin_q, 1.0), self._rotate(k, cos_k, sin_k, 1.0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    This splits the last dimension in half, negates the second half,
    and concatenates them in reverse order.

    Args:
        x: Input tensor of shape (..., dim).

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    This is a standalone function that can be used with precomputed cos/sin values.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine values for positions.
        sin: Sine values for positions.
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PolarPositionEmbedding(nn.Module):
    """Polar Coordinate Position Embedding (PoPE).

    PoPE improves upon RoPE by decoupling the "what" (content) and "where" (position)
    in attention computations. This eliminates the entanglement present in RoPE
    where query/key phases interfere with positional encoding.

    Key differences from RoPE:
    - Uses softplus for non-negative magnitudes (decouples content)
    - Sets phases purely based on position (decouples position)
    - Uses d frequencies instead of d/2 (more expressive)
    - Optional learnable phase bias for flexibility

    Reference: https://arxiv.org/abs/2509.10534

    Args:
        dim: Dimension of the position embedding (typically head_dim).
        max_position_embeddings: Maximum sequence length supported.
        base: Base for the frequency computation.
        learnable_bias: Whether to use learnable phase bias.
        bias_init: Initialization for phase bias ("zero" or "uniform").

    Shape:
        - Query/Key Input: (batch, num_heads, seq_len, head_dim)
        - Position IDs: (batch, seq_len)
        - Output: Same as input
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        learnable_bias: bool = True,
        bias_init: str = "uniform",
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.learnable_bias = learnable_bias

        # Compute frequencies (d frequencies instead of d/2 like RoPE)
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Learnable phase bias (optional)
        if learnable_bias:
            if bias_init == "zero":
                self.phase_bias = nn.Parameter(torch.zeros(dim))
            else:
                # Uniform initialization in [-2π, 0]
                self.phase_bias = nn.Parameter(
                    torch.empty(dim).uniform_(-2 * torch.pi, 0)
                )
        else:
            self.register_buffer("phase_bias", torch.zeros(dim))

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

        # Softplus for magnitude (ensures non-negative)
        self.softplus = nn.Softplus()

    def reset_buffers(self, device: Optional[torch.device] = None) -> None:
        """Rebuild the non-persistent PoPE buffers (inv_freq + cos/sin cache).

        Like :meth:`RotaryEmbedding.reset_buffers`: these are ``persistent=False``
        so they're absent from checkpoints and must be recomputed after
        ``from_pretrained`` (which otherwise leaves them as uninitialized
        memory). The learnable ``phase_bias`` is a real param/persistent buffer
        and is loaded normally, so it is left untouched here.
        """
        if device is None:
            device = self.inv_freq.device
        inv_freq = self._compute_inv_freq().to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, device=device)

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for PoPE (d frequencies)."""
        # Use d frequencies (not d/2 like RoPE)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, dtype=torch.float32) / self.dim)
        )
        return inv_freq

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Precompute cos and sin values for the given sequence length."""
        self.max_seq_len_cached = seq_len

        if device is None:
            device = self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Phase = position * theta
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim]

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply PoPE to query and key tensors.

        The transformation converts q and k to polar coordinates:
        - magnitude = softplus(x) (non-negative, content-dependent)
        - phase = position * theta (position-dependent only)

        Then converts to Cartesian for efficient dot-product attention.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            position_ids: Position indices of shape (batch, seq_len).

        Returns:
            Tuple of (transformed_q, transformed_k) with same shapes as inputs.
        """
        _maybe_extend_cache(self, position_ids)

        # Get cos/sin for positions
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # Compute magnitudes using softplus (ensures non-negative)
        mu_q = self.softplus(q)  # [batch, heads, seq_len, dim]
        mu_k = self.softplus(k)

        # Apply phase bias to keys (learnable offset)
        phase_bias = self.phase_bias.view(1, 1, 1, -1)
        cos_bias = torch.cos(phase_bias)
        sin_bias = torch.sin(phase_bias)

        # Convert to Cartesian coordinates for efficient computation
        # Query: x = mu * cos(pos * theta), y = mu * sin(pos * theta)
        q_x = mu_q * cos
        q_y = mu_q * sin

        # Key: x = mu * cos(pos * theta + bias), y = mu * sin(pos * theta + bias)
        # Using angle addition: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        k_x = mu_k * (cos * cos_bias - sin * sin_bias)
        k_y = mu_k * (sin * cos_bias + cos * sin_bias)

        # Interleave x and y components for attention computation
        # The attention score will be: sum(q_x * k_x + q_y * k_y)
        # which equals: sum(mu_q * mu_k * cos((s-t)*theta + bias))
        q_embed = torch.stack([q_x, q_y], dim=-1).flatten(-2)
        k_embed = torch.stack([k_x, k_y], dim=-1).flatten(-2)

        return q_embed, k_embed

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, "
            f"base={self.base}, learnable_bias={self.learnable_bias}"
        )


def unscale_windowed_rotary(model: nn.Module) -> int:
    """Return every sliding-window layer's RoPE to the unscaled default, in place.

    A YaRN read rescales EVERY layer's RoPE, but a layer banded by ``swa_window``
    only ever sees relative distances inside its window, which the model trained
    on -- it needs no extension. Scaling it anyway changes its low-frequency
    angles and its attention temperature (``mscale``) for no gain. This keeps
    YaRN on the global layers only, so an eval can separate "the global layers
    extend" from "YaRN perturbed the local ones". Windowed layers are identified
    by the ``swa_window`` attribute (``modules/swa_with_globals.py`` and its
    subclasses); dense models have none, so this is a no-op for them.

    Returns the number of rotary modules changed.
    """
    n = 0
    for mod in model.modules():
        rope = getattr(mod, "rotary_emb", None)
        if not isinstance(rope, RotaryEmbedding) or getattr(mod, "swa_window", None) is None:
            continue
        if rope.rope_type == "default" and rope.mscale == 1.0:
            continue
        rope.rope_type = "default"
        rope.scaling_factor = 1.0
        rope.mscale = 1.0
        rope.reset_buffers()
        n += 1
    return n


def get_position_embedding_class(pos_type: str = "rope"):
    """Get the position embedding class by name.

    Args:
        pos_type: Type of position embedding ("rope" or "pope").

    Returns:
        The position embedding class.
    """
    embeddings = {
        "rope": RotaryEmbedding,
        "pope": PolarPositionEmbedding,
    }
    if pos_type not in embeddings:
        raise ValueError(f"Unknown position embedding: {pos_type}. Available: {list(embeddings.keys())}")
    return embeddings[pos_type]
