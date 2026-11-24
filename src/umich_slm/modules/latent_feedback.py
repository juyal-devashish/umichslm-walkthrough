"""Latent feedback: the previous token's TOP-layer state re-enters the stack.

Full-bandwidth transformer (arXiv 2608.08888). A standard decoder hands layer 0
nothing but the token embedding, so the shallowest layers see the past only as
processed by the layers beneath them. Here the previous position's FINAL-layer
state is fused with the current token's embedding and fed in at the bottom, so
every layer sees the past as processed by the whole stack, and non-verbalized
state (uncertainty, a partial result, a plan) can re-enter with a renewed depth
budget instead of having to be squeezed through the sampled token.

    e_t (*) h_{t-1} = W_U h_{t-1} * sigmoid(W_G e_t)        then RMSNorm

THE ASYMMETRY IS THE POINT. The hidden state occupies the VALUE path and the
token embedding enters only as a multiplicative gate. A symmetric fusion
(``e_t + W h_{t-1}``, say) lets the model learn to drive the state term to zero
and collapse back to a plain transformer, which is exactly the degenerate
solution this is trying to avoid. Cost is two D x D matrices: 73,728 params at
ar10m, 1.6M at deep80, i.e. under 0.2% of N either way.

WHY THE FIRST TOKEN OF EVERY DOCUMENT KEEPS ITS PLAIN EMBEDDING. The fused
quantity REPLACES the embedding -- it is not added to it -- so a position with
no valid predecessor would otherwise receive ``W_U * 0 = 0`` and lose its token
identity entirely. The paper's own KV-cache definition starts
``C = {e_1, e_2 (*) h_1, ...}``, i.e. position 1 is unfused, and this module
generalises that to every document start. Combined with the prefix mixin below
it is the same code path.

THE NORM APPLIES TO BOTH BRANCHES, AND THAT IS NOT COSMETIC. ``fused`` is an
RMSNorm output, so its scale is the learned gain -- order 1. A raw token
embedding is order 0.09 at these sizes (measured: embed_tokens RMS 0.0905 at
ar10m, identical to the non-feedback parent). Normalising only the fused branch
therefore hands the stack inputs that differ by ~10x, and it does so along TWO
axes at once:

  * WITHIN a fused pass, every document start and every mixin position reverts
    to the plain embedding and lands 10x below its neighbours;
  * BETWEEN passes, because pass 1 has no predecessor states and runs the plain
    embedding for the WHOLE batch. With the default 75/22/3 mixture that is
    three quarters of training at one scale and all of validation at the other.

Measured consequence on the first three ar10m cells (jobs under fbtpause/):
final train loss 2.50-2.61 against val_lm_loss 2.59 / 3.73 / 5.53, and a norm
gain that scattered 0.97 / 1.02 / 4.31 across three runs of the SAME arm on the
same data -- the signature of a parameter nothing in the loss pins down, because
it only ever has to serve the 25% of batches that are multi-pass. Sharing one
norm across both branches makes the two regimes commensurate by construction and
lets layer 0's own RMSNorm absorb what is left.

TWO THINGS THAT HAVE BITTEN THIS CODEBASE BEFORE, both handled here:
  * DOCUMENT RESET. This is a token-level recurrence and intra-doc masking is
    on by default, so without a reset the state of one document's last token
    would seed the next document's first. Griffin/Mamba2/Hymba each shipped
    that bug (project_intra_doc_recurrent_reset).
  * CAUSALITY. ``h_{t-1}`` comes from the PREVIOUS pass, which is causal, so it
    depends only on tokens <= t-1 -- but only because of the shift. Get the
    shift wrong by one and the model reads its own answer. tests/ carries a
    multi-position gradient probe rather than trusting the argument, because
    two arms have passed single-position probes by luck.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .rmsnorm import get_rmsnorm_class

# 75% one pass / 22% two / 3% three, the paper's mixture. The 3% slice is NOT a
# rounding detail: two-pass-only training diverges past its trained recursion
# depth, and the handful of three-pass batches is what makes the feedback map a
# contraction and lets it extrapolate to 1000+ passes. Do not drop it.
DEFAULT_PASS_MIX: Tuple[float, ...] = (0.75, 0.22, 0.03)


def passes_for_index(index: int, mix: Sequence[float], seed: int = 0) -> int:
    """Pass count for forward number ``index``, identical on every rank.

    DDP REQUIRES THIS TO BE DETERMINISTIC. If two ranks picked different pass
    counts they would run different numbers of collectives and the job would
    hang in the NCCL watchdog rather than fail cleanly. A per-rank ``torch.rand``
    would do exactly that, so the draw is a pure function of a counter that
    every rank increments in lockstep (one bump per forward, gradient
    accumulation included).
    """
    if not mix:
        return 1
    # Integer hash rather than random.Random(): no global RNG state to disturb,
    # and the same answer on every rank, device and torch version.
    h = (index * 2654435761 + seed * 40503) & 0xFFFFFFFF
    u = h / 0x100000000
    acc = 0.0
    for k, w in enumerate(mix, start=1):
        acc += float(w)
        if u < acc:
            return k
    return len(mix)


class LatentFeedback(nn.Module):
    """Fuse the previous position's top-layer state into the input embedding."""

    def __init__(self, config):
        super().__init__()
        d = int(config.hidden_size)
        self.hidden_size = d
        # W_U on the state (value path), W_G on the embedding (gate path).
        self.up_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d, bias=False)
        norm_cls = get_rmsnorm_class(getattr(config, "use_zero_centered_rmsnorm", False))
        self.norm = norm_cls(d, eps=float(getattr(config, "rms_norm_eps", 1e-5)))
        # Prefix mixin. Training fuses every position; inference sees a prompt
        # that was prefilled unfused followed by fused generation, so a model
        # trained only on the homogeneous case meets a distribution it has never
        # seen at the switchover. Reverting a random share of positions to the
        # plain embedding during training covers the heterogeneous case.
        self.mixin_p = float(getattr(config, "latent_feedback_mixin_p", 0.1))

    @staticmethod
    def shift_states(
        states: torch.Tensor,
        document_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-shift by one and report which positions have a valid predecessor.

        Returns ``(prev, valid)`` where ``prev[:, t] = states[:, t - 1]`` and
        ``valid[:, t]`` is False at position 0 and at every document start.
        """
        zeros = states.new_zeros((states.shape[0], 1, states.shape[2]))
        # torch.cat, not an in-place slice assignment: `states` is a graph tensor
        # (gradients flow back into the previous pass), and writing into a view
        # of a fresh zeros() would be fine but cat keeps the autograd path
        # obvious and compiles without a copy guard.
        prev = torch.cat([zeros, states[:, :-1, :]], dim=1)

        valid = torch.ones(states.shape[0], states.shape[1], dtype=torch.bool, device=states.device)
        valid[:, 0] = False
        if document_ids is not None:
            same = document_ids[:, 1:] == document_ids[:, :-1]
            first = torch.zeros_like(same[:, :1])
            valid = valid & torch.cat([first, same], dim=1)
        return prev, valid

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        feedback_states: torch.Tensor,
        document_ids: Optional[torch.Tensor] = None,
        shift: bool = True,
    ) -> torch.Tensor:
        """Return the stack's input for this pass.

        Args:
            inputs_embeds: ``[B, S, D]`` plain token embeddings.
            feedback_states: ``[B, S, D]`` top-layer states of the previous
                pass, token-aligned (prefix already stripped). When ``shift`` is
                False these are taken to be pre-aligned ``h_{t-1}`` already --
                the incremental-decode case, where the caller holds one state.
            document_ids: ``[B, S]`` packing ids; drives the reset.
            shift: right-shift ``feedback_states`` by one position.
        """
        if shift:
            prev, valid = self.shift_states(feedback_states, document_ids)
        else:
            prev = feedback_states
            valid = torch.ones(prev.shape[0], prev.shape[1], dtype=torch.bool, device=prev.device)

        fused = self.norm(self.up_proj(prev) * torch.sigmoid(self.gate_proj(inputs_embeds)))

        use_plain = ~valid
        if self.training and self.mixin_p > 0.0:
            revert = torch.rand(prev.shape[0], prev.shape[1], device=prev.device) < self.mixin_p
            use_plain = use_plain | revert
        # self.plain(), not inputs_embeds: the two branches of this where() must
        # arrive at the stack on the same scale. See the module docstring.
        return torch.where(use_plain.unsqueeze(-1), self.plain(inputs_embeds), fused)

    def plain(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """The unfused input, on the same scale the fused one arrives at.

        Called for a position with no valid predecessor, and -- via
        ``UMichSLMModel.forward`` -- for an ENTIRE pass that has no feedback
        states to fuse, which is what pass 1 of every multi-pass step is and
        what 75% of training steps are in full under the default mixture.
        Without it the stack sees two input scales ~10x apart and learns the
        common one, which is not the one validation measures.
        """
        return self.norm(inputs_embeds)
