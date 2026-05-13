"""Reward functions for RAG-style GRPO training.

Follows the same signature as reward_fns.py:
    fn(completions: list[str], **kwargs) -> list[float]

GRPOTrainer passes all dataset columns as kwargs, so columns written by
prepare_grpo_rag.py are available by name:
  - aliases  : JSON-encoded list[str] of acceptable gold answers
  - context  : raw retrieved passage text (joined)

Four reward functions, each returning 0.0 or 1.0:

  retrieval_format_reward  — schema compliance: exactly one <retrieve> and
                             one <result> block, both properly closed.

  citation_reward          — the <answer> block contains at least one <cite>
                             tag. Trains the model to attribute claims to
                             retrieved passages.

  grounding_reward         — the text inside <cite> tags shares at least 10%
                             token-level Jaccard overlap with the retrieved
                             context. Penalises hallucinated citations.

  rag_accuracy_reward      — the text inside <answer> matches at least one
                             of the gold aliases (exact match after lower-
                             casing and whitespace normalisation).

Combined reward per completion = sum of all four reward functions (range [0, 4]).
Use alongside format_reward from reward_fns.py if <think> compliance is also
needed (range would then be [0, 5]).
"""
from __future__ import annotations

import json
import re

_RETRIEVE_RE = re.compile(r"<retrieve>(.*?)</retrieve>", re.DOTALL)
_RESULT_RE   = re.compile(r"<result>(.*?)</result>",     re.DOTALL)
_ANSWER_RE   = re.compile(r"<answer>(.*?)</answer>",     re.DOTALL)
_CITE_RE     = re.compile(r"<cite>(.*?)</cite>",         re.DOTALL)

_GROUNDING_THRESHOLD = 0.10  # Jaccard ≥ this → grounded citation


def _normalise(s: str) -> str:
    return " ".join(s.strip().lower().split())


def retrieval_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Return 1.0 if completion has exactly one <retrieve> and one <result> block."""
    rewards = []
    for c in completions:
        ok = (
            len(_RETRIEVE_RE.findall(c)) == 1
            and len(_RESULT_RE.findall(c)) == 1
        )
        rewards.append(1.0 if ok else 0.0)
    return rewards


def citation_reward(completions: list[str], **kwargs) -> list[float]:
    """Return 1.0 if the <answer> block contains at least one <cite> tag."""
    rewards = []
    for c in completions:
        answer_m = _ANSWER_RE.search(c)
        if not answer_m:
            rewards.append(0.0)
            continue
        has_cite = bool(_CITE_RE.search(answer_m.group(1)))
        rewards.append(1.0 if has_cite else 0.0)
    return rewards


def grounding_reward(
    completions: list[str], context: list[str], **kwargs
) -> list[float]:
    """Return 1.0 if cited text has Jaccard overlap >= threshold with context.

    Extracts all text inside <cite>...</cite>, tokenises by whitespace, and
    computes the Jaccard similarity against the provided retrieved context.
    Returns 0.0 if there are no citations (encourages citation_reward first).

    Args:
        completions: model output strings.
        context:     list of raw retrieved passage strings, one per completion.
                     Sourced from the `context` column of the GRPO dataset.
    """
    rewards = []
    for c, ctx in zip(completions, context):
        cite_spans = _CITE_RE.findall(c)
        if not cite_spans:
            rewards.append(0.0)
            continue
        cited_tokens = set(_normalise(" ".join(cite_spans)).split())
        ctx_tokens   = set(_normalise(ctx or "").split())
        if not cited_tokens or not ctx_tokens:
            rewards.append(0.0)
            continue
        intersection = cited_tokens & ctx_tokens
        union        = cited_tokens | ctx_tokens
        jaccard      = len(intersection) / len(union)
        rewards.append(1.0 if jaccard >= _GROUNDING_THRESHOLD else 0.0)
    return rewards


def rag_accuracy_reward(
    completions: list[str], aliases: list[str], **kwargs
) -> list[float]:
    """Return 1.0 if extracted <answer> matches any gold alias.

    Args:
        completions: model output strings.
        aliases:     JSON-encoded list[str] of normalised acceptable answers,
                     as written by prepare_grpo_rag.py's `aliases` column.
    """
    rewards = []
    for c, aliases_json in zip(completions, aliases):
        answer_m  = _ANSWER_RE.search(c)
        predicted = _normalise(answer_m.group(1)) if answer_m else ""
        try:
            gold_list: list[str] = json.loads(aliases_json)
        except (json.JSONDecodeError, TypeError):
            gold_list = [aliases_json] if isinstance(aliases_json, str) else []
        matched = any(
            predicted == _normalise(g) for g in gold_list if isinstance(g, str)
        )
        rewards.append(1.0 if matched else 0.0)
    return rewards
