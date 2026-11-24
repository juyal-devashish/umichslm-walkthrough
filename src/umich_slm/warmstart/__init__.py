"""Warm-start a UMichSLM model from a pretrained donor's parameters.

See ``docs/parameter_warm_start.md``. The three pieces are vocabulary alignment
(:mod:`.vocab_align`), embedding composition (:mod:`.compose`) and the body
transfer (:mod:`.transfer`), which is shared by every sweep arm because they all
carry the same dense skeleton.
"""

from .compose import RULES, compose_table, fit_positional, summarize  # noqa: F401
from .donor_io import load_embeddings, load_tensor, resolve_snapshot  # noqa: F401
from .vocab_align import Alignment, bpe_pair_decompositions, build_alignment  # noqa: F401
