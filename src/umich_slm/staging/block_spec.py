"""Block- and stage-spec dataclasses for the layer-stacked pretraining pipeline.

A :class:`StageSpec` describes the model at a particular training stage:
  * ``stage_index`` — 1, 2, or 3.
  * ``attention_mode`` — ``"deltanet_mla"`` or ``"swa_globals_mla"``.
  * ``block_specs`` — a list of :class:`BlockSpec`, one per block in the stage.
  * ``parent_stage_index`` / ``parent_checkpoint`` — pointers to where the
    weights came from (``None`` for stage 1, which is trained from random
    initialization).

A :class:`BlockSpec` captures, for a single block in the next stage:
  * ``source`` — the index of the source block in the previous stage from
    which weights are copied (``None`` for a freshly-initialized block, which
    only happens at stage 1).
  * ``is_inserted`` — ``True`` if this block is a stack-time copy that should
    be MoD-routed at ``mod_capacity_inserted`` (default 0.5). Original
    middle blocks and I/O blocks remain dense.
  * ``block_layout`` — per-layer-in-block dispatch entries
    (``["delta","delta","delta","mla"]`` or ``["swa","swa","swa","mla"]``).
  * ``fresh_router`` — when ``is_inserted=True``, controls whether the MoD
    router weights are freshly initialized (``True``) or copied from the
    source block's router (``False``, with optional small Gaussian
    perturbation).
  * ``router_perturb_std`` — Gaussian perturbation std applied to a copied
    router when ``fresh_router=False`` to break symmetry between adjacent
    duplicated blocks at stage 3. Default 0.005.

Both dataclasses serialize to / from plain dicts for storage in
``stage_metadata.json`` next to a checkpoint and for the standalone
``transition`` CLI's ``--config <stage_spec.json>`` argument.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class BlockSpec:
    """Specification for a single block in a stage."""

    source: Optional[int]
    is_inserted: bool
    block_layout: List[str]
    fresh_router: bool = True
    router_perturb_std: float = 0.005

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlockSpec":
        return cls(
            source=d.get("source"),
            is_inserted=bool(d["is_inserted"]),
            block_layout=list(d["block_layout"]),
            fresh_router=bool(d.get("fresh_router", True)),
            router_perturb_std=float(d.get("router_perturb_std", 0.005)),
        )


@dataclass
class StageTSTSpec:
    """Per-stage Token-Superposition Training spec (Peng et al., arXiv:2605.06546).

    When ``enabled``, the stage's training loop runs phase (i) with mean-pooled
    bags of size ``bag_size`` and multi-hot CE, then flips to standard NTP at
    step ``int(stage_steps * recovery_ratio)``. Recovery always completes
    within the stage, so the grown checkpoint handed to the next stage is a
    standard-NTP checkpoint regardless of which stages enabled TST.
    """

    enabled: bool = False
    bag_size: int = 6
    recovery_ratio: float = 0.3
    loss_weighting: str = "uniform"  # "uniform" | "power_law"
    extend_seq_len_for_equal_flops: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageTSTSpec":
        return cls(
            enabled=bool(d.get("enabled", False)),
            bag_size=int(d.get("bag_size", 6)),
            recovery_ratio=float(d.get("recovery_ratio", 0.3)),
            loss_weighting=str(d.get("loss_weighting", "uniform")),
            extend_seq_len_for_equal_flops=bool(
                d.get("extend_seq_len_for_equal_flops", True)
            ),
        )


@dataclass
class StageSpec:
    """Specification for a training stage."""

    stage_index: int
    attention_mode: str  # "deltanet_mla" | "swa_globals_mla"
    block_specs: List[BlockSpec]
    parent_stage_index: Optional[int] = None
    parent_checkpoint: Optional[str] = None
    init_strategy: str = "exact_copy"  # or "llama_pro_identity"
    tst: Optional[StageTSTSpec] = None

    @property
    def num_blocks(self) -> int:
        return len(self.block_specs)

    @property
    def block_size(self) -> int:
        if not self.block_specs:
            return 0
        return len(self.block_specs[0].block_layout)

    @property
    def num_hidden_layers(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def hybrid_attention_ratio(self) -> List[str]:
        """The flat per-layer ratio = concatenation of all block_layouts."""
        out: List[str] = []
        for spec in self.block_specs:
            out.extend(spec.block_layout)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "attention_mode": self.attention_mode,
            "block_specs": [bs.to_dict() for bs in self.block_specs],
            "parent_stage_index": self.parent_stage_index,
            "parent_checkpoint": self.parent_checkpoint,
            "init_strategy": self.init_strategy,
            "tst": self.tst.to_dict() if self.tst is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageSpec":
        if "block_specs" not in d:
            raise ValueError("StageSpec dict missing required key 'block_specs'.")
        tst_raw = d.get("tst")
        tst = StageTSTSpec.from_dict(tst_raw) if isinstance(tst_raw, dict) else None
        return cls(
            stage_index=int(d["stage_index"]),
            attention_mode=str(d["attention_mode"]),
            block_specs=[BlockSpec.from_dict(b) for b in d["block_specs"]],
            parent_stage_index=d.get("parent_stage_index"),
            parent_checkpoint=d.get("parent_checkpoint"),
            init_strategy=str(d.get("init_strategy", "exact_copy")),
            tst=tst,
        )

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StageSpec":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def stage_spec_to_dict(spec: StageSpec) -> Dict[str, Any]:
    """Top-level helper kept for the package public API."""
    return spec.to_dict()


def stage_spec_from_dict(d: Dict[str, Any]) -> StageSpec:
    """Top-level helper kept for the package public API."""
    return StageSpec.from_dict(d)
