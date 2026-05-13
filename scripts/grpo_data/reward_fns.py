"""Reward functions for GRPO math training.

Two scalar rewards per completion, following the trl GRPOTrainer signature:

    fn(completions: list[str], **kwargs) -> list[float]

format_reward   — +1.0 when the completion contains exactly one properly closed
                  <think>...</think> block and one <answer>...</answer> block.
                  Trains the model to comply with the 6-token schema before
                  optimising for correctness.

accuracy_reward — +1.0 when the text inside <answer>...</answer> matches the
                  gold answer string (light normalisation: strip whitespace and
                  remove thousand-separator commas).  Requires the dataset to
                  have an `answer` column, which GRPOTrainer passes as a kwarg.

Combined reward per completion = format_reward + accuracy_reward (range [0, 2]).
GRPOTrainer sums across reward_funcs before normalising within the group.
"""
from __future__ import annotations

import re

_THINK_RE  = re.compile(r"<think>(.*?)</think>",   re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _normalise(s: str) -> str:
    return s.strip().replace(",", "")


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Return 1.0 if the completion has exactly one <think> and one <answer> block."""
    rewards = []
    for c in completions:
        ok = (len(_THINK_RE.findall(c)) == 1) and (len(_ANSWER_RE.findall(c)) == 1)
        rewards.append(1.0 if ok else 0.0)
    return rewards


def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Return 1.0 if the extracted <answer> content matches the gold answer."""
    rewards = []
    for c, gold in zip(completions, answer):
        m = _ANSWER_RE.search(c)
        predicted = _normalise(m.group(1)) if m else ""
        rewards.append(1.0 if predicted == _normalise(gold) else 0.0)
    return rewards
