"""Length-weighted long-document pool for long-context training (W4, LONG_CONTEXT_PLAN.md §1.3).

The long-context stages need to *up-weight* the ≥8k / ≥16k document pool: with day-1
intra-doc masking on (W2), a 32K window packed from short docs yields **zero** long-range
gradient (§1.2 of the plan — 81% of corpus tokens live in docs <8k). Masking and long-doc
weighting are inseparable.

This module is **config, not new physical datasets** (per the plan's "no re-pack, no
staggered-by-size datasets" verdict). It expresses, at the granularity the mixture machinery
actually supports (the 11 prepared *buckets*):

  1. which buckets carry the long-context signal for each strategy, and
  2. how much extra probability mass a length-filtered copy of those buckets should get at
     each context stage.

`scripts/prepare_mix_dataset.py` consumes it to add filtered, up-weighted streams alongside
the natural mixture; the same length filter is a cheap char-length proxy so it runs inside a
streaming `.filter()` without a second tokenization pass.

Two strategies (matching the plan's two data pools):
  - **A (YaRN extension probe)**: reasoning/agentic/math only — **no arxiv** — to keep the
    extension test on the target reasoning/agentic/math domain.
  - **B (native long stage)**: the full long pool, adding arxiv (proof-pile) and python
    (taco/starcoder) for maximum ≥32k volume.
"""

from __future__ import annotations

# Buckets that carry long-document mass, per strategy (bucket names = prepared_root subdirs).
# See §1.1 supply measurement: reasoning_traces + agent_traj dominate the ≥8k pool; math
# (proof-pile/textbooks) and arxiv hold most genuine ≥32k material; taco-python is 85% ≥8k.
STRATEGY_A_BUCKETS = ["reasoning_traces", "agent_traj", "math"]
STRATEGY_B_BUCKETS = ["reasoning_traces", "agent_traj", "math", "arxiv", "python"]

STRATEGY_BUCKETS = {"A": STRATEGY_A_BUCKETS, "B": STRATEGY_B_BUCKETS}

# Per context-stage: the token-length threshold defining "long", and the total extra
# probability mass reallocated to the length-filtered long streams. Weights ramp with context
# so far RoPE positions are actually exercised as the window grows.
STAGE_SPECS = {
    "8k": {"threshold_tokens": 4096, "long_weight": 0.25},
    "16k": {"threshold_tokens": 8192, "long_weight": 0.45},
    "32k": {"threshold_tokens": 16384, "long_weight": 0.65},
}

# Manifest heuristic (bytes/3.7 ≈ tokens); reused as the char↔token proxy for length routing.
DEFAULT_CHARS_PER_TOKEN = 3.7


def resolve_stage(strategy: str, stage: str) -> dict:
    """Return {'buckets', 'threshold_tokens', 'long_weight'} for a (strategy, stage) pair.

    strategy: "A" or "B"; stage: one of STAGE_SPECS ("8k" / "16k" / "32k").
    """
    if strategy not in STRATEGY_BUCKETS:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {list(STRATEGY_BUCKETS)}")
    if stage not in STAGE_SPECS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGE_SPECS)}")
    spec = STAGE_SPECS[stage]
    return {
        "buckets": list(STRATEGY_BUCKETS[strategy]),
        "threshold_tokens": spec["threshold_tokens"],
        "long_weight": spec["long_weight"],
    }


def is_long_text(
    text: str, threshold_tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> bool:
    """Cheap char-length proxy for 'this doc has >= threshold_tokens tokens'.

    Avoids a second tokenization pass inside a streaming filter. Errs toward *inclusion*
    (chars/token underestimates token count for code/math), which is the safe bias for a
    long pool — the exact per-position eval (`long_context_ppl.py`) uses the real tokenizer.
    """
    return len(text) >= threshold_tokens * chars_per_token


def reallocate_long_mass(
    mixture: dict, long_buckets: list[str], long_weight: float
) -> tuple[dict, dict]:
    """Split a normalized bucket mixture into (base, long) probability maps summing to 1.

    Base streams keep ``(1 - long_weight)`` of their mass; the freed ``long_weight`` is
    distributed across the length-filtered ``long_buckets`` in proportion to their share of
    the base mixture (a bucket with more natural tokens gets more long mass). Only buckets
    present in ``mixture`` are eligible; if none are, ``long_weight`` collapses to 0 and the
    base mixture is returned unchanged.

    Returns (base_probs, long_probs). ``long_probs`` keys are the eligible long buckets.
    """
    if not 0.0 <= long_weight < 1.0:
        raise ValueError(f"long_weight must be in [0, 1); got {long_weight}")
    eligible = [b for b in long_buckets if b in mixture and mixture[b] > 0]
    if not eligible or long_weight == 0.0:
        return dict(mixture), {}

    base = {b: p * (1.0 - long_weight) for b, p in mixture.items()}
    long_share_total = sum(mixture[b] for b in eligible)
    long_probs = {b: long_weight * (mixture[b] / long_share_total) for b in eligible}
    return base, long_probs
