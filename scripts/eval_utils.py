"""Shared utilities for offline checkpoint evaluation.

Used by compute_val_loss.py and compute_bpb.py. Kept intentionally small
and dependency-free (only torch + transformers) so it works on any
checkpoint that run_eval.py can already load.

Two public helpers:
    - load_model_and_tokenizer(...) : mirrors the loading logic in run_eval.py
                                      (HF dir / pretrained name / raw .pt).
    - score_texts(...)              : single forward pass over a list of raw
                                      strings. Returns the pieces you need to
                                      compute either perplexity or bits-per-byte:
                                        sum_nll_nats, total_pred_tokens, total_bytes_utf8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    ckpt_path: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Load an HF causal LM + tokenizer for offline evaluation.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name (e.g. "Qwen/Qwen2.5-0.5B") or a local
        checkpoint directory containing config.json + weights.
    tokenizer_name_or_path : str | None
        Tokenizer source. Defaults to model_name_or_path (so HF dirs
        with a tokenizer saved alongside — as pretrain_fineweb.py
        produces — work out of the box).
    ckpt_path : str | None
        Optional raw .pt state_dict to overlay on top of the loaded model
        (same behaviour as run_eval.py --ckpt_path).
    dtype : torch.dtype
        Cast weights to this dtype after loading. Default float32 for
        reproducible loss values; use bfloat16/float16 on GPU for speed.
    device : str
        "cpu" or "cuda" (or "cuda:0" etc.).

    Returns
    -------
    (model, tokenizer) — model is in eval() mode, moved to device.
    """
    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
    logger.info("Loading tokenizer: %s", tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    if ckpt_path is not None:
        logger.info("Overlaying raw checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            cleaned[k] = v
        model.load_state_dict(cleaned)

    model = model.to(dtype=dtype, device=device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_texts(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 1024,
    batch_size: int = 4,
    device: str = "cpu",
    insert_eos_between_docs: bool = True,
) -> dict:
    """Concatenate + tokenize `texts`, forward in fixed-size chunks, return loss pieces.

    Chunking policy (MVP):
        1. Tokenize every string without special tokens, concatenate into one
           stream. Optionally insert eos_token_id as a separator between docs.
        2. Split into non-overlapping chunks of length `max_length`. Drop the
           final partial chunk — this keeps every chunk's prediction count equal
           to (max_length - 1), so the math below is exact.
        3. For each chunk: loss = model(input_ids=X, labels=X).loss is already
           averaged over (seq_len - 1) shifted positions, so
               sum_nll_nats += loss.item() * batch_size * (seq_len - 1)

    Returned dict contains everything downstream callers need:
        - sum_nll_nats       : total negative log-likelihood in nats
        - total_pred_tokens  : number of positions the loss was computed over
                               (use this for PPL: exp(sum_nll / total_pred_tokens))
        - total_bytes_utf8   : raw UTF-8 byte count of the *input strings*,
                               tokenizer-independent (use this for BPB)
        - n_chunks           : number of forward passes, useful for logging
    """
    texts = list(texts)
    if not texts:
        raise ValueError("score_texts: `texts` is empty")

    # Accumulate bytes (tokenizer-independent) and tokens (tokenizer-specific).
    all_ids: list[int] = []
    total_bytes = 0
    eos_id = tokenizer.eos_token_id if insert_eos_between_docs else None
    for t in texts:
        total_bytes += len(t.encode("utf-8"))
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(eos_id)

    n_chunks = len(all_ids) // max_length
    if n_chunks == 0:
        raise ValueError(
            f"score_texts: only {len(all_ids)} tokens, need at least {max_length} "
            "for one chunk. Lower --max_length or pass more text."
        )

    # Materialize chunks. For MVP data sizes (~tens of MB) this is fine; if
    # we ever evaluate on GB-scale text we'd switch to a streaming iterator.
    chunks = [all_ids[i * max_length : (i + 1) * max_length] for i in range(n_chunks)]

    sum_nll_nats = 0.0
    total_pred_tokens = 0
    for i in range(0, n_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        # HF averages loss over batch_size * (seq_len - 1) non-ignored positions.
        n_pred = input_ids.shape[0] * (input_ids.shape[1] - 1)
        sum_nll_nats += out.loss.item() * n_pred
        total_pred_tokens += n_pred

    return {
        "sum_nll_nats": sum_nll_nats,
        "total_pred_tokens": total_pred_tokens,
        "total_bytes_utf8": total_bytes,
        "n_chunks": n_chunks,
    }


def resolve_device(requested: str = "auto") -> str:
    """Pick cuda when available unless the user explicitly asked for cpu."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
