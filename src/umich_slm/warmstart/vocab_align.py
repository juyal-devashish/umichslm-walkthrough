"""Align the production SuperBPE vocabulary to a donor model's vocabulary.

Why this exists: the production tokenizer is SuperBPE (vocab 56000), whose tokens
routinely span whitespace (``'yĠtheĠ'``, ``'.ĠTheĠ`'``). No conventional BPE
tokenizer has tokens like that -- their pre-tokenizers split on whitespace first --
so only ~21% of our vocabulary exists verbatim in any donor and ~27% of actual
token *mass*. The other ~73% has to be *composed* from the donor tokens its string
decomposes into (median 3).

The alignment is donor-agnostic: it maps each of our token ids to a list of donor
ids plus weights. :mod:`umich_slm.warmstart.compose` turns that into embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

# GPT-2 byte-level alphabet. Both SuperBPE and every donor we consider (Qwen,
# Llama, SmolLM, Gemma-as-BPE) store tokens in this representation, so our token
# strings and theirs live in the same space -- but we still have to go back
# through raw bytes to re-tokenize, because the donor's pre-tokenizer must get a
# chance to split on whitespace.


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@lru_cache(maxsize=1)
def _unicode_to_bytes() -> Dict[str, int]:
    return {v: k for k, v in _bytes_to_unicode().items()}


def token_to_text(token: str) -> Optional[str]:
    """Decode a byte-level BPE token string back to the text it stands for.

    Returns ``None`` when the token is a partial UTF-8 sequence (common for CJK
    and emoji fragments), which simply means it cannot be re-tokenized as text
    and must fall back to byte-level matching.
    """
    u2b = _unicode_to_bytes()
    try:
        raw = bytes(u2b[ch] for ch in token)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass
class Alignment:
    """Per-token decomposition of our vocabulary into donor token ids.

    ``parts[i]`` are the donor ids our token ``i`` decomposes into and
    ``part_bytes[i]`` their byte lengths (used by the length-weighted rules).
    ``exact[i]`` is True when our token string is itself in the donor vocabulary,
    i.e. a verbatim copy is available.
    """

    parts: List[List[int]]
    part_bytes: List[List[int]]
    exact: List[bool]
    our_size: int
    donor_size: int

    @property
    def n_exact(self) -> int:
        return sum(self.exact)

    @property
    def n_unmapped(self) -> int:
        return sum(1 for p in self.parts if not p)

    def stats(self) -> Dict[str, float]:
        lens = [len(p) for p in self.parts if p]
        return {
            "our_size": self.our_size,
            "donor_size": self.donor_size,
            "exact": self.n_exact,
            "exact_frac": self.n_exact / max(self.our_size, 1),
            "unmapped": self.n_unmapped,
            "mean_parts": sum(lens) / max(len(lens), 1),
            "max_parts": max(lens) if lens else 0,
        }


def build_alignment(our_tok, donor_tok, *, verbose: bool = False) -> Alignment:
    """Decompose every token of ``our_tok`` into ids of ``donor_tok``.

    Three cases, in order of preference:

    1. the token string is in the donor vocabulary verbatim -> single exact part;
    2. the token decodes to text -> re-tokenize that text with the donor;
    3. the token is a partial UTF-8 sequence -> match its byte-level *prefix*
       characters against single-character donor tokens, else leave unmapped.
    """
    our_vocab = our_tok.get_vocab()
    donor_vocab = donor_tok.get_vocab()
    parts: List[List[int]] = [[] for _ in range(len(our_vocab))]
    part_bytes: List[List[int]] = [[] for _ in range(len(our_vocab))]
    exact = [False] * len(our_vocab)
    donor_inv = {v: k for k, v in donor_vocab.items()}

    for tok, idx in our_vocab.items():
        if idx >= len(parts):  # added/special tokens beyond the table
            continue
        hit = donor_vocab.get(tok)
        if hit is not None:
            parts[idx] = [hit]
            part_bytes[idx] = [len(tok)]
            exact[idx] = True
            continue
        text = token_to_text(tok)
        if text is None or text == "":
            # Partial UTF-8: fall back to per-character donor tokens.
            ids = [donor_vocab[ch] for ch in tok if ch in donor_vocab]
            parts[idx] = ids
            part_bytes[idx] = [1] * len(ids)
            continue
        ids = donor_tok(text, add_special_tokens=False)["input_ids"]
        parts[idx] = list(ids)
        part_bytes[idx] = [len(donor_inv.get(i, "")) or 1 for i in ids]

    align = Alignment(parts, part_bytes, exact, len(our_vocab), len(donor_vocab))
    if verbose:
        print(json.dumps(align.stats(), indent=2))
    return align


def bpe_pair_decompositions(
    donor_dir: str, donor_vocab: Dict[str, int]
) -> List[Tuple[int, int, int]]:
    """Recover ``(parent, left, right)`` triples from a donor's BPE merge table.

    This is the supervision for fitting a composition rule: every non-atomic BPE
    token *is* a composite whose true embedding we already know, so "predict the
    parent embedding from its parts" is a held-out-able regression with ~100k
    examples and no extra data. It is a within-word proxy for our cross-word
    case, which is the closest honest supervision available -- conventional BPE
    vocabularies contain no multi-word tokens at all.
    """
    import glob
    import os

    tj = glob.glob(os.path.join(donor_dir, "tokenizer.json"))
    if not tj:
        return []
    with open(tj[0], "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    merges: Sequence = spec.get("model", {}).get("merges", []) or []
    out: List[Tuple[int, int, int]] = []
    for m in merges:
        if isinstance(m, str):
            bits = m.split(" ")
            if len(bits) != 2:
                continue
            a, b = bits
        else:
            if len(m) != 2:
                continue
            a, b = m[0], m[1]
        pid = donor_vocab.get(a + b)
        aid = donor_vocab.get(a)
        bid = donor_vocab.get(b)
        if pid is not None and aid is not None and bid is not None:
            out.append((pid, aid, bid))
    return out
