"""Persistent learnable global tokens for the SWA + globals attention design.

These tokens are prepended to every input sequence at the embedding layer. Each
SWA layer attends from content tokens to (a) all globals and (b) a fixed
sliding window of previous content tokens; globals attend to all tokens (within
causal order). The MLA layer in the same block uses full attention so globals
are simply part of the attended sequence.

The globals are model-level state — they exist once on the model and are not
duplicated when blocks duplicate at stack transitions. The Mixture-of-Depths
router force-routes the global prefix through every inserted block, and the
MTP loss masks the prefix out so globals never contribute to next-token CE.

Reference: STACKING_IMPLEMENTATION_PLAN_DELTA.md §2.2.

This module mirrors the structure of `MetaTokens` (hymba.py:28) but is kept as
a distinct class so probes can dispatch on `isinstance(m, GlobalTokens)`.
"""

from typing import Optional

import torch
import torch.nn as nn


class GlobalTokens(nn.Module):
    """Learnable global tokens prepended to every input sequence.

    Args:
        num_tokens: Number of global tokens (typical: 256).
        hidden_size: Hidden dimension size.
        init_std: Standard deviation for Gaussian initialization. Default 0.02.
    """

    def __init__(self, num_tokens: int, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.init_std = init_std
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_size) * init_std)

    def forward(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand globals for a batch.

        Returns: tensor [batch_size, num_tokens, hidden_size].
        """
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype, device=device)

    def prepend_to_embeddings(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Concatenate the globals to the front of [B, T, d] -> [B, N+T, d]."""
        if inputs_embeds.dim() != 3:
            raise ValueError(
                f"inputs_embeds must be [B, T, d]; got shape {tuple(inputs_embeds.shape)}."
            )
        b = inputs_embeds.size(0)
        globals_b = self.forward(b, inputs_embeds.dtype, inputs_embeds.device)
        return torch.cat([globals_b, inputs_embeds], dim=1)

    @staticmethod
    def build_attention_mask(
        content_len: int,
        window: int,
        num_globals: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build a boolean attention mask of shape [N+T, N+T].

        Pattern (positions 0..N-1 are globals, N..N+T-1 are content):
          - Global positions can attend to all positions in causal order
            (any global can attend to any earlier global; globals do not need to
            attend forward into content).
          - Content positions attend to:
                * all N globals (always);
                * the previous ``window`` content positions (sliding window).
            Causal: content does not attend to itself's future positions.

        Returns:
          A boolean tensor [seq_len, seq_len] where True = "may attend".
          Suitable for SDPA (after converting to additive mask) or for masking
          attention logits directly.
        """
        seq_len = num_globals + content_len
        idx = torch.arange(seq_len, device=device)
        row = idx.unsqueeze(1)  # [S, 1]
        col = idx.unsqueeze(0)  # [1, S]

        # Causal everywhere
        causal = col <= row

        # Globals: rows 0..N-1
        is_global_row = row < num_globals
        # Targets that are globals: cols 0..N-1
        is_global_col = col < num_globals

        # Content rows: rows >= N. Sliding window over content.
        # For a content row r (=N+t), allowed content cols are [N+t-window+1..N+t]
        # restricted to >= N (no negative content positions).
        content_distance = row - col
        in_window = (content_distance < window) & (content_distance >= 0)

        # A content row is allowed to attend to: any global col, or any content col
        # within the sliding window.
        content_allows = (~is_global_row) & (is_global_col | (in_window & ~is_global_col))

        # A global row is allowed to attend to: any earlier or same position
        # (globals see globals + already-emitted content, but during training causal
        # ordering means globals only see other globals at positions <= row).
        global_allows = is_global_row & causal

        mask = causal & (content_allows | global_allows | (is_global_row & is_global_col))
        return mask.to(dtype=torch.bool)


class RegisterBank(GlobalTokens):
    """Learnable register bank for the swa_registers variant.

    Same parameter shape as :class:`GlobalTokens` (``[num_tokens, hidden]``) so
    the two arch twins stay matched-N, but the tokens are NEVER prepended to the
    sequence — SWA layers read them through a zero-init cross-attention instead
    (see ``modules/swa_registers.py``). Kept as a subclass so probes and the
    optimizer-grouping code can distinguish bank-as-prefix from bank-as-registers
    while sharing the init/expansion helpers.
    """


class SinkBank(GlobalTokens):
    """Tiny out-of-sequence sink/anchor bank (swa_sink_count, swa_scribe arch).

    Same learned [N, hidden] table as GlobalTokens, but NEVER prepended: SWA
    layers join it to the window branch's softmax via an exact LSE merge
    (modules/swa_scribe.py), so it can absorb attention mass (the sink role)
    without forcing the prefix kernel path. Live from step 0 — no gate — just
    like prefix globals. Distinct subclass so probes and param audits can tell
    the roles apart.
    """
