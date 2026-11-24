"""Forward-pass capability evaluation.

Two modes over one frozen eval set, so the numbers are directly comparable:

  * **reporting** -- the full set, run per checkpoint by
    `scripts/eval_downstream_suite.py`. This is what the sweep analysis reads.
  * **tracking** -- a fixed, manifested subset cheap enough to run inside the
    training loop next to val loss, via `CapabilityTracker`.

The tracking subset is stored as INDICES into the reporting files rather than as
its own copy of the text, which is what guarantees a tracked number and a
reported number describe the same documents.
"""

from umich_slm.eval.bpb import (
    bpb_by_position,
    bpb_of_docs,
    effective_max_length,
    split_docs,
    unigram_bpb,
)
from umich_slm.eval.scoring import load_records, score_records

__all__ = [
    "bpb_by_position",
    "bpb_of_docs",
    "effective_max_length",
    "split_docs",
    "unigram_bpb",
    "load_records",
    "score_records",
]
