"""Detect torch.compile silently degrading to eager mid-run.

WHY THIS EXISTS. ``titan/wrapper.py::apply_compile`` sets
``torch._dynamo.config.suppress_errors = True`` on purpose: compile is lazy, so
backend failures surface hours into a job, and raising there would cost the whole
run. The documented cost is that a failure -- or blowing past
``cache_size_limit`` -- makes Dynamo abandon the frame and run it eager FOREVER,
silently. Its own docstring names the consequence: *"That would read as 'the
causal-router arm is slower', i.e. as an architectural result, in a sweep whose
whole purpose is to compare routing mechanisms."*

That is not hypothetical. Measured 2026-08-22: the ar10m ``mat_ffn`` cell logged
2278.5 ms/step, which reproduces 2 x the EAGER bench (1036.3 ms/micro-batch) plus
the ~163 ms optimizer/loader overhead to within 1.9%, while the COMPILED
prediction is 1467 ms -- 36% off. The same overhead model reproduces
``qwen3_like`` exactly (1269.0 predicted vs 1270.4 observed). So that cell ran
eager for its whole life, the arm was recorded at 1.79x dense when compiled it is
1.18x, and nothing anywhere said so. The number reached a walltime table and an
architecture comparison before anyone checked.

WHAT THIS DOES. Snapshot Dynamo's frame counters once, after compile has had
enough steps to settle, and log them. Warn loudly when the compiled fraction is
low or when a counter names a recompile/cache-size limit. It never raises and
never changes numerics -- a bad reading is a log line, not a dead run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Frames compiled / frames seen, below which the run is treated as degraded.
_OK_FRACTION_FLOOR = 0.90
_LIMIT_HINTS = ("cache_size", "recompile_limit", "cache_limit", "exceeded")


def dynamo_counters() -> Dict[str, Dict[str, Any]]:
    """Dynamo's counter groups, or {} if unavailable. Never raises."""
    try:
        from torch._dynamo.utils import counters

        return {
            group: dict(vals)
            for group, vals in counters.items()
            if group in ("frames", "graph_break", "recompiles", "stats", "unimplemented")
        }
    except Exception:  # pragma: no cover - torch internals move between versions
        return {}


def compile_health(compiled_layers: int) -> Dict[str, Any]:
    """Summarise compile state. ``compiled_layers`` is apply_compile's return."""
    c = dynamo_counters()
    frames = c.get("frames", {})
    total = float(frames.get("total", 0) or 0)
    ok = float(frames.get("ok", 0) or 0)
    frac = (ok / total) if total else None

    limits = []
    for group, vals in c.items():
        for key, n in vals.items():
            k = str(key).lower()
            if n and any(h in k for h in _LIMIT_HINTS):
                limits.append(f"{group}.{key}={n}")

    degraded = bool(
        compiled_layers > 0
        and (limits or (frac is not None and total > 0 and frac < _OK_FRACTION_FLOOR))
    )
    return {
        "compiled_layers": int(compiled_layers),
        "frames_total": int(total),
        "frames_ok": int(ok),
        "frames_ok_frac": frac,
        "limit_hits": limits,
        "degraded": degraded,
        "counters": c,
    }


def log_compile_health(compiled_layers: int, step: int, log_fn=None) -> Optional[Dict[str, Any]]:
    """Log compile health once. Returns the report (or None if compile was off).

    ``log_fn`` lets the torchtitan loop route this through its rank-0 printer;
    it falls back to the module logger.
    """
    emit = log_fn or (lambda m: logger.warning("%s", m))
    if not compiled_layers:
        emit(f"[compile] step {step}: torch.compile is OFF (0 layers compiled)")
        return None
    try:
        rep = compile_health(compiled_layers)
    except Exception as exc:  # never let a diagnostic kill a run
        emit(f"[compile] step {step}: health check failed ({exc!r}); continuing")
        return None

    frac = rep["frames_ok_frac"]
    frac_s = "n/a" if frac is None else f"{frac:.2f}"
    msg = (
        f"[compile] step {step}: layers={rep['compiled_layers']} "
        f"frames ok/total={rep['frames_ok']}/{rep['frames_total']} ({frac_s})"
    )
    if rep["degraded"]:
        emit(
            "[compile] *** DEGRADED: Dynamo appears to be running frames EAGER. "
            "suppress_errors hides this, and an eager arm reads as a slower "
            "ARCHITECTURE (see titan/compile_health.py). "
            + msg
            + (f" limit_hits={rep['limit_hits']}" if rep["limit_hits"] else "")
        )
    else:
        emit(msg)
    return rep
