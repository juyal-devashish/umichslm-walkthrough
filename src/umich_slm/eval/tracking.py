"""Capability metrics cheap enough to log inside the training loop.

Val loss is one scalar. It tells you a run is improving; it does not tell you
*where*. Held-out bits-per-byte per domain does, at a cost that -- on a fixed
subset -- is comparable to the val-loss pass the trainer already runs. A curve
per bucket makes a data-mixture change or a regression on code legible while the
run is still going, instead of at the next checkpoint analysis.

The design constraint that makes this trustworthy is that the tracking subset is
stored as INDICES into the reporting eval set, never as its own copy of the text.
So a tracked number and a reported number describe the same documents, the two
modes cannot silently drift apart, and the subset is pinned by the same sha256
manifest as everything else.

Two things this deliberately does not do:

  * It does not report accuracy. At 25-100M parameters accuracy sits at chance
    and moves on rounding, so a training curve drawn through it is noise
    (Schaeffer 2304.15004). Continuous byte-normalised likelihoods only.
  * It does not resize itself to a time budget at runtime. The subset is fixed
    at build time, because a metric whose sample changes between steps is not a
    curve.

Usage from a training loop::

    tracker = CapabilityTracker.from_eval_set(EVAL_SET, tokenizer, device="cuda")
    ...
    if step % cfg.checkpoint.eval_every == 0:
        for name, value in tracker.evaluate(model).items():
            metrics[f"capability/{name}"] = value
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch

from umich_slm.eval.bpb import bpb_of_docs, effective_max_length, split_docs
from umich_slm.eval.scoring import load_records, score_records

logger = logging.getLogger(__name__)

TRACKING_SPEC = "tracking/spec.json"

# Which cloze tasks are worth a training curve. Chosen for early signal at small
# scale rather than for recognisability: `mmlu_continuation` is the cloze
# reformulation E2LM (2506.07731) found separates small models where native MMLU
# "is giving noise", and arc_easy / sciq are its other two winners. Tasks whose
# accuracy is pinned to a high floor (boolq at 62%) are absent on purpose.
TRACKING_CLOZE = ["arc_easy", "sciq", "mmlu_continuation", "hellaswag"]


class CapabilityTracker:
    """A fixed subset of the reporting eval set, scored on demand.

    Holds no reference to a model: `evaluate()` takes whatever the trainer has,
    so there is no second copy of the weights and no reload.
    """

    def __init__(
        self,
        domain_docs: dict,
        cloze_records: dict,
        tokenizer,
        *,
        device: str = "cuda",
        max_length: int = 2048,
        bpb_batch_size: int = 64,
        max_batch_tokens: int = 32768,
        spec: dict | None = None,
    ):
        self.domain_docs = domain_docs
        self.cloze_records = cloze_records
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.bpb_batch_size = bpb_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.spec = spec or {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build_spec(
        cls,
        eval_set_dir: Path,
        *,
        n_domain_docs: int = 64,
        n_cloze_docs: int = 200,
        cloze_tasks: list[str] | None = None,
        seed: int = 20260829,
    ) -> dict:
        """Choose the subset once and write down exactly what it is.

        Selection is by index under a recorded seed, so the spec plus the eval-set
        manifest fully determine the numbers. Rebuilding with the same seed on the
        same eval set reproduces them; a new seed is a new metric and gets a new
        spec version.
        """
        eval_set_dir = Path(eval_set_dir)
        rng = random.Random(seed)
        spec = {
            "seed": seed,
            "n_domain_docs": n_domain_docs,
            "n_cloze_docs": n_cloze_docs,
            "domain": {},
            "cloze": {},
        }

        for path in sorted((eval_set_dir / "domain_bpb").glob("*.txt")) + sorted(
            (eval_set_dir / "ood_bpb").glob("*.txt")
        ):
            n_total = len(split_docs(path.read_text(encoding="utf-8")))
            key = f"{path.parent.name}/{path.stem}"
            idx = sorted(rng.sample(range(n_total), min(n_domain_docs, n_total)))
            spec["domain"][key] = {"n_total": n_total, "indices": idx}

        for task in cloze_tasks or TRACKING_CLOZE:
            path = eval_set_dir / "benchmarks" / f"{task}.jsonl"
            if not path.exists():
                logger.warning("tracking: no frozen records for %s -- skipped", task)
                continue
            n_total = sum(1 for line in path.open() if line.strip())
            idx = sorted(rng.sample(range(n_total), min(n_cloze_docs, n_total)))
            spec["cloze"][task] = {"n_total": n_total, "indices": idx}
        return spec

    @classmethod
    def from_eval_set(cls, eval_set_dir, tokenizer, *, device: str = "cuda", **kw):
        """Load the subset named by `tracking/spec.json`, building it if absent."""
        eval_set_dir = Path(eval_set_dir)
        spec_path = eval_set_dir / TRACKING_SPEC
        if spec_path.exists():
            spec = json.loads(spec_path.read_text())
        else:
            spec = cls.build_spec(eval_set_dir)
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec, indent=2))
            logger.info("tracking: wrote a new subset spec to %s", spec_path)

        domain_docs = {}
        for key, entry in spec["domain"].items():
            sub, stem = key.split("/", 1)
            docs = split_docs((eval_set_dir / sub / f"{stem}.txt").read_text(encoding="utf-8"))
            if len(docs) != entry["n_total"]:
                raise RuntimeError(
                    f"tracking spec is stale for {key}: it was built against "
                    f"{entry['n_total']} documents and the file now has {len(docs)}. "
                    f"Rebuild the spec, and treat earlier tracked values as a "
                    f"different metric."
                )
            domain_docs[key] = [docs[i] for i in entry["indices"]]

        cloze_records = {}
        for task, entry in spec["cloze"].items():
            recs = load_records(eval_set_dir / "benchmarks" / f"{task}.jsonl")
            if len(recs) != entry["n_total"]:
                raise RuntimeError(f"tracking spec is stale for cloze task {task}")
            cloze_records[task] = [recs[i] for i in entry["indices"]]

        return cls(domain_docs, cloze_records, tokenizer, device=device, spec=spec, **kw)

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, model, *, families=("domain_bpb", "cloze")) -> dict:
        """Flat {metric_name: value}, ready to merge into a trainer's log dict.

        Restores the model's training mode on the way out: a tracker that leaves
        the model in `eval()` would silently disable dropout for the rest of the
        run.
        """
        was_training = model.training
        model.eval()
        # Read the model's position limit rather than trusting the configured
        # value: a model with learned positions raises an opaque device-side
        # assert on an over-long sequence.
        max_length = effective_max_length(model, self.max_length)
        out: dict[str, float] = {}
        try:
            if "domain_bpb" in families:
                total_nll = total_bytes = 0.0
                for key, docs in self.domain_docs.items():
                    res = bpb_of_docs(
                        model,
                        self.tokenizer,
                        docs,
                        device=self.device,
                        max_length=self.max_length,
                        batch_size=self.bpb_batch_size,
                    )
                    if "bpb" not in res:
                        continue
                    out[f"bpb/{key.split('/')[-1]}"] = res["bpb"]
                    total_nll += res["bpb"] * res["total_bytes"]
                    total_bytes += res["total_bytes"]
                if total_bytes:
                    out["bpb/pooled"] = total_nll / total_bytes

            if "cloze" in families:
                for task, recs in self.cloze_records.items():
                    agg, _ = score_records(
                        model,
                        self.tokenizer,
                        recs,
                        device=self.device,
                        max_length=self.max_length,
                        max_batch_tokens=self.max_batch_tokens,
                        task_name=task,
                    )
                    if "bpb_gold" in agg:
                        out[f"cloze_bpb/{task}"] = agg["bpb_gold"]
                    if "soft" in agg:
                        out[f"cloze_soft/{task}"] = agg["soft"]
        finally:
            if was_training:
                model.train()
        return out

    def describe(self) -> str:
        d = sum(len(v) for v in self.domain_docs.values())
        c = sum(len(v) for v in self.cloze_records.values())
        return (
            f"CapabilityTracker: {d} documents over {len(self.domain_docs)} domains, "
            f"{c} items over {len(self.cloze_records)} cloze tasks "
            f"(seed {self.spec.get('seed')})"
        )
