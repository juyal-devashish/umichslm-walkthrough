"""Canonical project tokenizer.

The production tokenizer is a **SuperBPE tokenizer with vocab size 56000**,
selected by the pipeline in ``TOKENIZER_TRAINING_PLAN.md`` (see the decision in
``outputs/tokenizer/<run>/comparison/decision.json``). Every training / data-prep
stack should obtain its tokenizer through this module rather than hard-coding
SmolLM2 / Meta-Llama-3 or the ``49152`` placeholder that lives in the size presets.

Resolution order for the path (first hit wins):
1. ``$UMICHSLM_TOKENIZER_PATH`` environment variable (set this on Great Lakes,
   where the local prep path is not mounted — point it at the shipped tokenizer).
2. ``DEFAULT_TOKENIZER_PATH`` — the local prep-host (curry) location.

``DEFAULT_VOCAB_SIZE`` is the authoritative vocab to override the preset
placeholder with at model-build time.
"""

from __future__ import annotations

import os

# Authoritative production tokenizer (SuperBPE, vocab 56000).
DEFAULT_VOCAB_SIZE = 56000

# Local prep-host (curry) location of the chosen HF-format tokenizer dir.
DEFAULT_TOKENIZER_PATH = (
    "/shared/6/projects/umich-slm/outputs/tokenizer/"
    "20260501-123305/tokenizers/final_hf_tokenizer"
)

_ENV_VAR = "UMICHSLM_TOKENIZER_PATH"


def get_tokenizer_path() -> str:
    """Return the configured tokenizer path (env override wins)."""
    return os.environ.get(_ENV_VAR, DEFAULT_TOKENIZER_PATH)


def load_default_tokenizer(path: str | None = None):
    """Load the project tokenizer via ``AutoTokenizer``.

    Args:
        path: Explicit path/repo-id; defaults to :func:`get_tokenizer_path`.
    """
    from transformers import AutoTokenizer

    resolved = path or get_tokenizer_path()
    if resolved.startswith("/") and not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Tokenizer dir not found: {resolved}. On a host without the local "
            f"prep path mounted (e.g. Great Lakes), set ${_ENV_VAR} to the shipped "
            f"tokenizer directory."
        )
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
