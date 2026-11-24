"""Griffin RG-LRU (Real-Gated Linear Recurrent Unit) implementation.

Implements the temporal mixing layer from the Griffin architecture, which
replaces attention in most layers with a diagonal linear recurrence.

The recurrence uses a variance-preserving update:
    a_t  = exp(-softplus(c) * exp(log_a) * sigmoid(W_a * x_t))
    h_t  = a_t * h_{t-1} + sqrt(1 - a_t²) * (sigmoid(W_r * x_t) ⊙ W_i * x_t)
    y_t  = h_t ⊙ gelu(W_gate * x_t)
    out  = W_o * y_t

The recurrent state h has fixed size (batch, num_heads, head_dim) regardless
of sequence length, making Griffin memory-efficient for long contexts.

References:
    Griffin: https://arxiv.org/abs/2402.19427
    RecurrentGemma: https://github.com/google-deepmind/recurrentgemma
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class GriffinCache:
    """Recurrent state cache for a single RG-LRU layer.

    Attributes:
        recurrent_state: Hidden state tensor of shape (batch, num_heads, head_dim).
    """

    recurrent_state: Optional[torch.FloatTensor] = None


class GriffinDynamicCache:
    """Per-layer cache for a full Griffin model.

    Stores GriffinCache for RG-LRU layers and standard KV tuples for
    interleaved attention layers, indexed by layer_idx.
    """

    def __init__(self, num_layers: int, griffin_layers: List[bool]):
        self.num_layers = num_layers
        self.griffin_layers = griffin_layers
        self._cache: List[Optional[object]] = [None] * num_layers
        self._seen_tokens: int = 0

    def __getitem__(self, layer_idx: int):
        return self._cache[layer_idx]

    def __setitem__(self, layer_idx: int, value):
        self._cache[layer_idx] = value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def advance_seq_length(self, n: int):
        self._seen_tokens += n

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values,
        num_layers: int,
        griffin_layers: List[bool],
    ) -> "GriffinDynamicCache":
        cache = cls(num_layers, griffin_layers)
        if past_key_values is not None:
            for i, layer_cache in enumerate(past_key_values):
                cache._cache[i] = layer_cache
        return cache


# ---------------------------------------------------------------------------
# RG-LRU
# ---------------------------------------------------------------------------


class RGLRUAttention(nn.Module):
    """Real-Gated Linear Recurrent Unit (RG-LRU).

    Temporal mixing layer used in Griffin's hybrid architecture. Replaces
    attention with an O(L) diagonal linear recurrence that maintains a
    fixed-size hidden state across tokens.

    The hidden state update is variance-preserving:
        h_t = a_t * h_{t-1} + sqrt(1 - a_t²) * (r_t ⊙ i_t)

    where a_t ∈ (0, 1) is the per-timestep recurrence gate and r_t is the
    input gate.

    This module exposes the same forward() signature as UMichSLMAttention
    so it plugs into the existing decoder layer infrastructure unchanged.

    Args:
        config: Model configuration.
        layer_idx: Index of this layer in the stack.

    Note:
        The sequential recurrence loop is O(L) in time and O(1) in memory
        (beyond activations). A parallel associative scan can be swapped in
        for faster training on GPU if needed.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "griffin_num_heads", config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads

        # Input gate and value projections (both map hidden -> hidden)
        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.i_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Recurrence gate input: maps hidden -> num_heads (one scalar per head)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        # Learnable base decay parameters per (head, dim):
        #   base = softplus(c) * exp(log_a)
        #   a_t  = exp(-base * sigmoid(a_proj(x)))
        self.log_a = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.c = nn.Parameter(torch.ones(self.num_heads, self.head_dim))

        # Output gate and projection
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Intra-document masking (W2): boolean [B, L] doc-start mask stashed by the
        # model per forward (True at the first token of each packed document). When
        # set, the recurrent state resets at boundaries so it does not leak across
        # documents. None => single-document / cached-decode / not requested.
        self._intra_doc_start = None

    def _compute_recurrence_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep recurrence gate a_t ∈ (0, 1).

        a_t = exp(-softplus(c) * exp(log_a) * sigmoid(W_a * x_t))

        Values close to 1 → slow decay (long memory).
        Values close to 0 → fast decay (short memory).

        Returns:
            Tensor of shape (batch, seq_len, num_heads, head_dim).
        """
        # (batch, seq_len, num_heads) → (batch, seq_len, num_heads, 1)
        a_input = torch.sigmoid(self.a_proj(x)).unsqueeze(-1)

        # base: (num_heads, head_dim) — positive, learned
        base = F.softplus(self.c) * torch.exp(self.log_a)

        # a_t = exp(-base * a_input): broadcast over head_dim
        return torch.exp(-base * a_input)  # (batch, seq_len, num_heads, head_dim)

    def _rglru_scan(
        self,
        r: torch.Tensor,
        i: torch.Tensor,
        a: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        doc_start: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential RG-LRU scan over the time dimension.

        Args:
            r: Input gate, shape (batch, seq_len, num_heads, head_dim).
            i: Input values, shape (batch, seq_len, num_heads, head_dim).
            a: Recurrence gate, shape (batch, seq_len, num_heads, head_dim).
            h_prev: Previous hidden state (batch, num_heads, head_dim) or None.
            doc_start: Optional boolean (batch, seq_len) marking each packed
                document's first token. When set, the recurrent state is reset to
                zero at those timesteps so one document cannot leak into the next.

        Returns:
            (outputs, h_last):
                outputs: all hidden states, shape (batch, seq_len, num_heads, head_dim).
                h_last: final hidden state, shape (batch, num_heads, head_dim).
        """
        batch, seq_len, num_heads, head_dim = r.shape

        if h_prev is None:
            h_prev = torch.zeros(batch, num_heads, head_dim, device=r.device, dtype=r.dtype)

        outputs = []
        h = h_prev
        for t in range(seq_len):
            # Reset state for any sequence whose document begins at this timestep,
            # BEFORE the update, so a document's first token starts from zero state.
            if doc_start is not None:
                reset_t = doc_start[:, t]
                if reset_t.any():
                    h = h * (~reset_t).to(h.dtype).view(batch, 1, 1)
            a_t = a[:, t]  # (batch, num_heads, head_dim)
            r_t = r[:, t]
            i_t = i[:, t]
            # Variance-preserving update
            noise_scale = torch.sqrt(torch.clamp(1.0 - a_t**2, min=0.0))
            h = a_t * h + noise_scale * (r_t * i_t)
            outputs.append(h)

        return torch.stack(outputs, dim=1), h  # (batch, seq_len, H, D), (batch, H, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[GriffinCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[GriffinCache]]:
        """Forward pass with HuggingFace-compatible signature.

        Args:
            hidden_states: (batch, seq_len, hidden_size).
            attention_mask: Unused — RG-LRU is causal by construction.
            position_ids: Unused — no positional encoding in RG-LRU.
            past_key_value: GriffinCache from the previous generation step.
            output_attentions: Unused — always returns None for weights.
            use_cache: Whether to return updated GriffinCache.
            depth_kv_cache: Unused — MoDA depth cache passthrough.

        Returns:
            (output, None, present):
                output: (batch, seq_len, hidden_size).
                None: no attention weights.
                present: GriffinCache if use_cache=True, else None.
        """
        batch_size, seq_len, _ = hidden_states.shape

        h_prev = past_key_value.recurrent_state if past_key_value is not None else None

        # --- Projections ---
        r = torch.sigmoid(self.r_proj(hidden_states))  # input gate
        i = self.i_proj(hidden_states)  # input values
        a = self._compute_recurrence_gate(hidden_states)  # recurrence gate

        # Reshape to (batch, seq_len, num_heads, head_dim)
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim)
        i = i.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Intra-document masking: reset the recurrent state at packed-document
        # boundaries. Only in the full-sequence training path (no cache, no carried
        # state) and only when the stashed mask matches this sequence length.
        doc_start = getattr(self, "_intra_doc_start", None)
        if doc_start is not None and (
            use_cache or h_prev is not None or doc_start.shape[-1] != seq_len
        ):
            doc_start = None

        # --- Recurrence ---
        h_out, h_last = self._rglru_scan(r, i, a, h_prev, doc_start=doc_start)

        # Flatten back to (batch, seq_len, hidden_size)
        h_out = h_out.reshape(batch_size, seq_len, self.hidden_size)

        # --- Gated output ---
        gate = F.gelu(self.gate_proj(hidden_states))
        output = self.o_proj(h_out * gate)

        # --- Cache ---
        present = GriffinCache(recurrent_state=h_last) if use_cache else None

        return output, None, present


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRIFFIN_CLASSES = {
    "rglru": RGLRUAttention,
}


def get_griffin_class(variant: str = "rglru") -> type:
    """Return the Griffin temporal mixing class for the given variant."""
    if variant not in GRIFFIN_CLASSES:
        raise ValueError(
            f"Unknown Griffin variant: {variant!r}. Available: {list(GRIFFIN_CLASSES)}"
        )
    return GRIFFIN_CLASSES[variant]
