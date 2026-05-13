"""Aggregate probe outputs into an I/O boundary recommendation.

After running the probe runner on a stage-N checkpoint, this helper turns the
JSON output into a recommendation of how many input/output blocks should be
kept *dense* (not MoD-routed) at the next stage. The recommendation is
advisory — the final ``StageSpec`` may be hand-edited before invoking the
``transition`` CLI.

Heuristic:
  * Find the longest contiguous prefix of blocks with ``block_influence < threshold``
    AND ``layer_deletion_ppl_increase < ppl_threshold`` for *all* layers in
    the block. That defines ``L_in_blocks`` (clamped >= 1).
  * Same on the suffix for ``L_out_blocks``.
  * ``skip_blocks`` lists any block in the middle whose influence is below
    threshold but is sandwiched between higher-influence blocks (these are
    redundant blocks that should also be left dense or could be marked for
    MoD aggressively).
"""

from __future__ import annotations

from typing import Any, Dict, List


def recommend_boundary(
    probe_outputs: Dict[str, Any],
    *,
    n_blocks_total: int,
    bi_threshold: float = 0.05,
    ppl_threshold: float = 0.10,
    block_size: int = 4,
) -> Dict[str, Any]:
    """Return ``{L_in_blocks, L_out_blocks, skip_blocks}`` from probe outputs.

    Args:
        probe_outputs: dict in the shape produced by ``probes.runner``.
            Expected keys (any may be missing — handled defensively):
              * ``block_influence_per_block``: list[float], length == n_blocks_total.
              * ``layer_deletion_ppl_increase``: list[float], length == n_blocks_total * block_size.
        n_blocks_total: total number of blocks in the model being analyzed.
        bi_threshold: a block with influence below this value is "low influence".
        ppl_threshold: per-layer perplexity-increase threshold for layer deletion.
        block_size: layers per block (4 in the canonical 3:1 hybrid).

    Returns:
        dict with keys:
          * ``L_in_blocks`` (int >= 1): how many input blocks to keep dense.
          * ``L_out_blocks`` (int >= 1): how many output blocks to keep dense.
          * ``skip_blocks`` (list[int]): mid-stack candidate skip blocks.
    """
    bi = list(probe_outputs.get("block_influence_per_block", []))
    pdel = list(probe_outputs.get("layer_deletion_ppl_increase", []))

    if len(bi) != n_blocks_total:
        # Probe output stale or shape mismatch — return the conservative default.
        return {"L_in_blocks": 1, "L_out_blocks": 1, "skip_blocks": []}

    def block_layers_below_ppl(block_idx: int) -> bool:
        if not pdel:
            return True
        start = block_idx * block_size
        end = min(start + block_size, len(pdel))
        if end <= start:
            return True
        return all(p < ppl_threshold for p in pdel[start:end])

    def block_low_influence(block_idx: int) -> bool:
        return bi[block_idx] < bi_threshold and block_layers_below_ppl(block_idx)

    # Longest low-influence prefix
    L_in = 0
    for b in range(n_blocks_total):
        if block_low_influence(b):
            L_in += 1
        else:
            break
    # Longest low-influence suffix
    L_out = 0
    for b in range(n_blocks_total - 1, -1, -1):
        if block_low_influence(b):
            L_out += 1
        else:
            break

    L_in = max(1, L_in)
    L_out = max(1, L_out)
    if L_in + L_out >= n_blocks_total:
        # Degenerate: shrink to leave at least one middle block.
        L_in = max(1, min(L_in, n_blocks_total // 2))
        L_out = max(1, n_blocks_total - L_in - 1)

    # Mid-stack skip candidates: blocks BELOW threshold that are surrounded by
    # higher-influence blocks (they're "valleys" worth flagging).
    skip_blocks: List[int] = []
    for b in range(L_in, n_blocks_total - L_out):
        if not block_low_influence(b):
            continue
        # Check the immediate neighbors in the eligible mid-stack range
        left_b = b - 1
        right_b = b + 1
        left_high = left_b < L_in or not block_low_influence(left_b)
        right_high = right_b >= n_blocks_total - L_out or not block_low_influence(right_b)
        if left_high and right_high:
            skip_blocks.append(b)

    return {
        "L_in_blocks": int(L_in),
        "L_out_blocks": int(L_out),
        "skip_blocks": skip_blocks,
    }
