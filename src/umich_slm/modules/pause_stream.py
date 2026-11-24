"""State/prediction separation -- the "free pause token".

Almost-free state-prediction separation (arXiv 2609.03807). A decoder's hidden
state carries two competing jobs: summarize the context for future positions,
and predict the next token right now. SPS splits them into two weight-shared
residual streams:

  * the STATE stream is the ordinary causal self-attention pass, and is the
    only stream that writes keys and values;
  * the PREDICTION stream starts at every position from one learned
    ``e_pause`` vector and, at each layer, forms queries over the state
    stream's K/V while writing no K/V of its own.

``lm_head`` reads the PREDICTION stream. The pause is "free" because it rides
an existing sequence position instead of adding one: no extra context length,
no KV-cache growth, no extra decode step. One new tensor (``e_pause``) and
three scalar gates per layer; everything else is shared, so a standard
checkpoint remains a valid backbone and separation can be phased on partway
through a run.

HOW THIS IS LAID OUT HERE, AND WHY IT NEEDS NO FLEXIBLE MASK. The prediction
query at position i reads state positions <= i, so the prediction attention is
an ORDINARY CAUSAL attention -- just with Q from ``p`` and K/V from ``a``. The
paper's 4x-slower "flexible mask" case is the variant where both streams
occupy slots in one sequence; keeping them in separate residual streams avoids
it, which matters here because our flex path costs 6-8x in compile time.

The two streams are carried CONCATENATED ALONG THE BATCH DIMENSION, ``[2B, S,
D]`` with the state half first. Everything that is per-position and
stream-agnostic -- position ids, document ids, the causal/varlen mask, the FFN,
the norms, gradient checkpointing, per-block compile -- then works unchanged at
batch 2B, and the only thing a layer has to do differently is take K/V from the
state half. That is one FA2 call per layer at batch 2B rather than two calls at
batch B, so it stays kernel-friendly.

COST. Unoptimized this is ~1.9x the baseline FLOPs (the paper's number; the
prediction stream skips nothing here, see the gap below). ``pause_shared_ffn``
evaluates ONE FFN per position on a learned blend of the two streams instead of
one per stream, which is the paper's own cost reducer and takes it to ~1.45x.

KNOWN GAP, stated rather than hidden: the paper's prediction pass skips
``k_proj``/``v_proj``, while routing through the existing ``precomputed_q``
hook recomputes them on the duplicated state half. Costs a few percent over the
paper's optimized form. The fix is an ``external_kv`` kwarg on
``UMichSLMAttention.forward``; not done.
"""

from typing import Optional

import torch
import torch.nn as nn


class PauseStream(nn.Module):
    """Model-level ``e_pause``: the prediction stream's per-position initial state."""

    def __init__(self, config):
        super().__init__()
        d = int(config.hidden_size)
        self.hidden_size = d
        # Initialised at embedding scale. The paper measures e_pause growing
        # ~40x from init to a per-coordinate RMS of ~1, which is what an
        # initializer_range of 0.02 lands on, so this is the init the reported
        # trajectory starts from.
        self.pause_embedding = nn.Parameter(
            torch.randn(d) * float(getattr(config, "initializer_range", 0.02))
        )

    def initial(self, batch_size: int, seq_len: int, dtype, device) -> torch.Tensor:
        """``[B, S, D]`` copies of ``e_pause``, one per position."""
        return self.pause_embedding.to(dtype=dtype, device=device).expand(
            batch_size, seq_len, self.hidden_size
        )


class PauseSharedFFN(nn.Module):
    """The paper's shared gated FFN: one FFN evaluation per position, not two.

        g    = sigmoid(W_in [a; p])            (scalar per position)
        f    = FFN(g * a + (1 - g) * p)
        a_out = sigmoid(W_a a) * f
        p_out = sigmoid(W_p p) * f

    The gates are single-output, so this adds 4*D params per layer. It halves
    the dominant term in the prediction stream's FLOPs and, in the paper, also
    frees enough memory to run a larger micro-batch.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.blend = nn.Linear(2 * hidden_size, 1, bias=False)
        self.state_gate = nn.Linear(hidden_size, 1, bias=False)
        self.pred_gate = nn.Linear(hidden_size, 1, bias=False)

    # pause_decision_only (P12, modules/msg_struct.compute_decision_mask): a [B, S] bool
    # bound per forward by the model. Off the mask the state stream takes the PLAIN
    # FFN (blend g = 1, no gate), so the state stream there is the parent's layer.
    _decision_mask: Optional[torch.Tensor] = None

    def forward(self, mlp: nn.Module, normed: torch.Tensor) -> torch.Tensor:
        """``normed`` is ``[2B, S, D]`` post-norm; returns the same shape."""
        half = normed.shape[0] // 2
        a, p = normed[:half], normed[half:]
        g = torch.sigmoid(self.blend(torch.cat([a, p], dim=-1)))
        dec = self._decision_mask
        if dec is not None:
            dec = dec.unsqueeze(-1)
            g = torch.where(dec, g, torch.ones_like(g))
        f = mlp(g * a + (1.0 - g) * p)
        a_out = torch.sigmoid(self.state_gate(a)) * f
        if dec is not None:
            a_out = torch.where(dec, a_out, f)
        return torch.cat([a_out, torch.sigmoid(self.pred_gate(p)) * f], dim=0)


def split_streams(hidden_states: torch.Tensor):
    """``[2B, S, D]`` -> ``(state, prediction)``."""
    half = hidden_states.shape[0] // 2
    return hidden_states[:half], hidden_states[half:]


def duplicate_for_streams(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Repeat a per-sequence tensor along the batch dim for the second stream."""
    if x is None:
        return None
    return torch.cat([x, x], dim=0)
