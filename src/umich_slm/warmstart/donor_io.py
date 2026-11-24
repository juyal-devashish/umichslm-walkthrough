"""Read individual tensors out of a donor checkpoint without loading the model.

A donor is often 4-9B parameters while the piece we need (the embedding table,
or one layer's projections) is a few hundred MB. ``safetensors`` supports
per-tensor reads through the shard index, so the whole analysis runs on a login
node with no GPU and no multi-GB resident set.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from safetensors import safe_open


def resolve_snapshot(repo_id: str, hub_root: str = "/shared/4/models/hub") -> Optional[str]:
    """Locate a cached snapshot directory for ``org/name``."""
    pat = os.path.join(hub_root, f"models--{repo_id.replace('/', '--')}", "snapshots", "*")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


@lru_cache(maxsize=16)
def _shard_map(snapshot: str) -> Dict[str, str]:
    """Map tensor name -> shard path, for both sharded and single-file layouts."""
    idx = glob.glob(os.path.join(snapshot, "*.safetensors.index.json"))
    if idx:
        with open(idx[0], "r", encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        return {k: os.path.join(snapshot, v) for k, v in wm.items()}
    out: Dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                out[k] = f
    return out


def tensor_names(snapshot: str) -> List[str]:
    return sorted(_shard_map(snapshot).keys())


def load_tensor(snapshot: str, name: str) -> torch.Tensor:
    """Read one tensor by name. Raises KeyError with the near misses listed."""
    sm = _shard_map(snapshot)
    path = sm.get(name)
    if path is None:
        stem = name.split(".")[-2] if "." in name else name
        near = [k for k in sm if stem in k][:8]
        raise KeyError(f"{name!r} not in checkpoint; near misses: {near}")
    with safe_open(path, framework="pt") as sf:
        return sf.get_tensor(name)


def find_prefix(snapshot: str) -> str:
    """Return the text-model prefix (``model.`` or ``model.language_model.``).

    Multimodal donors (Qwen3.5, Gemma 4) nest the language model one level
    deeper, so every caller that wants ``layers.0.self_attn.q_proj.weight`` has
    to know which. This resolves it from the embedding table's location.
    """
    names = tensor_names(snapshot)
    for pref in ("model.language_model.", "model.", ""):
        if f"{pref}embed_tokens.weight" in names:
            return pref
    raise KeyError("no embed_tokens.weight found in checkpoint")


def load_embeddings(snapshot: str) -> Dict[str, torch.Tensor]:
    """Return ``{'in': input_embeddings, 'out': output_embeddings}``.

    Donors with tied embeddings have no ``lm_head.weight``; both entries then
    point at the same tensor, which is the correct semantics rather than a
    fallback.
    """
    pref = find_prefix(snapshot)
    emb = load_tensor(snapshot, f"{pref}embed_tokens.weight")
    names = tensor_names(snapshot)
    head = None
    for cand in ("lm_head.weight", "model.lm_head.weight", f"{pref}lm_head.weight"):
        if cand in names:
            head = load_tensor(snapshot, cand)
            break
    return {"in": emb, "out": head if head is not None else emb}
