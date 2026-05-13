"""Per-step instrumentation collectors for layer-stacked pretraining.

The collectors compute the metrics that drive the T1-T5 fallback triggers from
``STACKING_IMPLEMENTATION_PLAN.md`` §5.1:

  * T1 — routing fraction collapse: per-block MoD routing fraction stays in [0.35, 0.65].
  * T2 — aux balancing loss runaway: per-block MoD aux loss stays under 0.1.
  * T3 — loss-spike non-recovery: handled at the train-loop level (transition loss).
  * T4 — reasoning-eval regression: handled at eval time.
  * T5 — DeltaNet gate collapse / SWA global utilization collapse.

The collectors are deliberately cheap (no extra forward passes, no attention
weights). They operate on the model's existing state after a step.
"""

from .metrics import collect_step_metrics, collect_post_transition_metrics

__all__ = ["collect_step_metrics", "collect_post_transition_metrics"]
