"""Masked-loss SFT dataset for the torchtitan path.

The pretraining loader (`titan/data.py`) sets ``labels = input_ids.clone()``:
every token is a target. SFT needs the opposite for most of the sequence --
loss on assistant text, assistant reasoning and tool-call JSON only, never on
system prompts, user turns or tool results coming back from the environment.

The mask is not re-derived here. `scripts/sft_prep/record.py` decided it at
prep time and wrote char-offset ``segments`` into each record, so what trains
is auditable in the data. This module's job is to turn those char spans into
token spans without drift, which is why it tokenizes **segment by segment**
rather than tokenizing the whole string and mapping offsets back: a token that
straddles a mask boundary would otherwise have to be assigned to one side, and
under SuperBPE (superword tokens spanning whitespace) that happens often.

Packing keeps whole conversations intact and gives each one its own
``document_id``, so the existing intra-document masking stops packed
conversations from attending across each other. That machinery already exists
for pretraining; this reuses it rather than adding a second mechanism.

Self-test (no external data needed):

    python -m umich_slm.titan.sft_data --selftest
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .harmony_annotate import PAD_VALUES, AnnotateOptions, HarmonyAnnotator

IGNORE_INDEX = -100


def _shuffle_buffered(it: Iterator[dict], size: int, rng: random.Random) -> Iterator[dict]:
    """Reservoir-shuffle a record stream.

    `scripts/sft_prep/build_mix.py` writes the mix source by source, so the raw
    order is blocked: without this, every packed window is single-source and
    the run walks the mixture sequentially instead of sampling it. It also
    separates the copies `source_epochs` emits back to back, which would
    otherwise land in the same window.
    """
    if size <= 1:
        yield from it
        return
    buf: list[dict] = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def encode_record(record: dict, tokenizer) -> tuple[list[int], list[int]]:
    """Tokenize one canonical SFT record into (input_ids, labels).

    Returns labels with IGNORE_INDEX everywhere the record's segments say the
    span is not loss-bearing. Segments are required: a record without them is
    a pretraining row, not an SFT row, and silently training on all of it is
    exactly the failure this module exists to prevent.
    """
    ids, labels, _ = _encode_segments(record, tokenizer)
    return ids, labels


def _encode_segments(record: dict, tokenizer) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """``encode_record`` plus each segment's token range (for annotations)."""
    text = record["text"]
    segments = record.get("segments")
    if not segments:
        raise ValueError(f"record {record.get('doc_id')} has no segments")
    ids: list[int] = []
    labels: list[int] = []
    bounds: list[tuple[int, int]] = []
    for seg in segments:
        chunk = text[seg["start"] : seg["end"]]
        start = len(ids)
        if chunk:
            piece = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            ids.extend(piece)
            if seg.get("loss"):
                labels.extend(piece)
            else:
                labels.extend([IGNORE_INDEX] * len(piece))
        bounds.append((start, len(ids)))
    return ids, labels, bounds


def encode_record_annotated(
    record: dict, tokenizer, annotator: HarmonyAnnotator
) -> tuple[list[int], list[int], dict]:
    """``encode_record`` plus per-token Harmony annotations from the segments.

    Roles come straight from the record's segments (no re-parse of the role);
    the error label / unlikelihood pairing reads each segment's text. Returns
    the annotator's full arrays, including the internal ``unlik_need``.
    """
    ids, labels, bounds = _encode_segments(record, tokenizer)
    ann = annotator.from_segments(ids, record["text"], record["segments"], bounds)
    return ids, labels, ann


class PackedSFTDataset(IterableDataset):
    """Stream canonical SFT records, pack them into fixed-length windows.

    Conversations are never split across windows (`SFT_PLAN.md` section 12:
    "don't fragment trajectories"). A window is emitted once the next
    conversation would not fit, padded out with `pad_id` and IGNORE_INDEX.
    Conversations longer than `seq_len` are handled per `long_policy`:

      "skip"            drop them, and count how many (the default: silently
                        truncating a trajectory throws away the final answer,
                        which is usually the part worth training on)
      "truncate_middle" keep the head and tail, drop the middle -- preserves
                        the system prompt and the outcome
    """

    def __init__(
        self,
        shards: Sequence[Path],
        tokenizer,
        *,
        seq_len: int,
        pad_id: Optional[int] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = False,
        long_policy: str = "skip",
        source_epochs: Optional[dict] = None,
        shuffle_buffer: int = 0,
        annotate: Optional[AnnotateOptions] = None,
    ) -> None:
        self.shards = [Path(s) for s in shards]
        if not self.shards:
            raise ValueError("PackedSFTDataset: empty shard list")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = pad_id if pad_id is not None else _default_pad(tokenizer)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.infinite = infinite
        if long_policy not in ("skip", "truncate_middle"):
            raise ValueError(f"unknown long_policy {long_policy!r}")
        self.long_policy = long_policy
        # Per-source epoch weighting: {"source_name": 5}. A source listed here
        # is yielded that many times per pass over the shard set.
        self.source_epochs = source_epochs or {}
        self.shuffle_buffer = int(shuffle_buffer)
        self.stats = {"records": 0, "skipped_long": 0, "truncated": 0, "windows": 0}
        # Per-token Harmony annotations (P6 role metric/weights, P3 labels).
        self.annotate = annotate if (annotate is not None and annotate.enabled) else None
        self.annotator: Optional[HarmonyAnnotator] = None

    def _reader(self) -> tuple[int, int]:
        """(offset, stride) over the record stream for this (rank, worker).

        Mirrors `titan/data.py`'s rank*num_workers+worker_id pattern. Sharding
        on rank alone -- what this did -- hands every worker in a rank the
        identical stream, so `num_workers=4` trains on four copies of every
        window at 4x the wall-clock and raises nothing.
        """
        w = get_worker_info()
        wid, nw = (0, 1) if w is None else (w.id, w.num_workers)
        return self.rank * nw + wid, self.world_size * nw

    def _iter_records(self, epoch: int) -> Iterator[dict]:
        offset, stride = self._reader()
        rng = random.Random(self.seed + epoch)
        shards = list(self.shards)
        rng.shuffle(shards)

        # Shard-level striding ONLY when there are enough shards to go round.
        # scripts/sft_prep/build_mix.py writes ONE train.jsonl.gz per stage, and
        # with a single shard a modulo over shards gives every record to reader
        # 0 and starves the rest -- which under infinite=True is not an error
        # but a silent hang (__iter__ spins, the trainer blocks in next()).
        if len(shards) >= stride:
            mine, line_off, line_stride = shards[offset::stride], 0, 1
        else:
            # Every reader decompresses every shard but parses only its 1/stride
            # of the lines. Redundant gunzip, but tokenization dominates and it
            # is the only scheme correct for any shard count.
            mine, line_off, line_stride = shards, offset, stride

        idx = 0
        for shard in mine:
            with _open(shard) as fh:
                for line in fh:
                    is_mine = idx % line_stride == line_off
                    idx += 1  # advanced for EVERY physical line, so the index
                    if not is_mine:  # is identical across readers
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    reps = self.source_epochs.get(rec.get("source"), 1)
                    for _ in range(reps):
                        yield rec

    def _fit(
        self, ids: list[int], labels: list[int], ann: Optional[dict] = None
    ) -> Optional[tuple]:
        if len(ids) <= self.seq_len:
            return ids, labels, ann
        if self.long_policy == "skip":
            self.stats["skipped_long"] += 1
            return None
        # truncate_middle: keep the first and last halves of the budget.
        head = self.seq_len // 2
        tail = self.seq_len - head
        self.stats["truncated"] += 1
        if ann is not None:
            cut = len(ids) - tail
            ann = {k: np.concatenate([v[:head], v[cut:]]) for k, v in ann.items()}
            # An unlikelihood target in the tail whose failing call fell before
            # the tail was cut has lost its error context: drop it.
            u, need = ann["unlik_targets"], ann["unlik_need"]
            drop = np.zeros(len(u), dtype=bool)
            drop[head:] = need[head:] < cut
            u[drop & (u != IGNORE_INDEX)] = IGNORE_INDEX
        return ids[:head] + ids[-tail:], labels[:head] + labels[-tail:], ann

    def __iter__(self) -> Iterator[dict]:
        offset, stride = self._reader()
        epoch = 0
        while True:
            rng = random.Random(self.seed + 7919 * epoch + offset)
            produced = 0
            win_ids: list[int] = []
            win_labels: list[int] = []
            win_docs: list[int] = []
            win_ann: list[dict] = []
            doc = 0
            if self.annotate is not None and self.annotator is None:
                self.annotator = HarmonyAnnotator(self.tokenizer, self.annotate)
            for rec in _shuffle_buffered(self._iter_records(epoch), self.shuffle_buffer, rng):
                ann = None
                try:
                    if self.annotator is not None:
                        ids, labels, ann = encode_record_annotated(rec, self.tokenizer, self.annotator)
                    else:
                        ids, labels = encode_record(rec, self.tokenizer)
                except ValueError:
                    continue
                if not ids:
                    continue
                self.stats["records"] += 1
                fitted = self._fit(ids, labels, ann)
                if fitted is None:
                    continue
                ids, labels, ann = fitted

                if len(win_ids) + len(ids) > self.seq_len and win_ids:
                    yield self._finish(win_ids, win_labels, win_docs, win_ann)
                    produced += 1
                    win_ids, win_labels, win_docs, win_ann, doc = [], [], [], [], 0
                doc += 1
                win_ids.extend(ids)
                win_labels.extend(labels)
                win_docs.extend([doc] * len(ids))
                if ann is not None:
                    win_ann.append(ann)
            if win_ids:
                yield self._finish(win_ids, win_labels, win_docs, win_ann)
                produced += 1

            if produced == 0:
                # Loud, not silent. With infinite=True this reader would spin
                # forever and wedge the job with no traceback; with
                # infinite=False the trainer re-iterates once and the second
                # next() raises StopIteration from inside the step loop.
                raise RuntimeError(
                    f"[sft_data] reader (rank={self.rank}, {offset}/{stride}) "
                    f"emitted no window in epoch {epoch} from {len(self.shards)} "
                    f"shard(s). Check the shard glob, world_size*num_workers vs "
                    f"the record count, and long_policy={self.long_policy!r} at "
                    f"seq_len={self.seq_len} "
                    f"(skipped_long={self.stats['skipped_long']})."
                )
            if not self.infinite:
                return
            epoch += 1

    def _finish(
        self, ids: list[int], labels: list[int], docs: list[int], anns: Optional[list] = None
    ) -> dict:
        pad = self.seq_len - len(ids)
        self.stats["windows"] += 1
        n_real = len(ids)
        if pad > 0:
            ids = ids + [self.pad_id] * pad
            labels = labels + [IGNORE_INDEX] * pad
            # Padding gets document id 0, distinct from every conversation, so
            # intra-doc masking never lets real tokens attend into it.
            docs = docs + [0] * pad
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "document_ids": torch.tensor(docs, dtype=torch.long),
        }
        if anns:
            arrays = self.annotator.model_arrays(
                {k: np.concatenate([a[k] for a in anns]) for k in anns[0]}
            )
            for k, v in arrays.items():
                full = np.full(self.seq_len, PAD_VALUES[k], dtype=v.dtype)
                full[:n_real] = v
                out[k] = torch.from_numpy(full.astype(np.float32 if k == "loss_weights" else np.int64))
        return out


def _default_pad(tokenizer) -> int:
    for attr in ("pad_token_id", "eos_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            return int(v)
    return 0


def sft_collate(batch: List[dict]) -> dict:
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    extra = {
        k: torch.stack([b[k] for b in batch], dim=0)
        for k in batch[0]
        if k not in ("input_ids", "labels", "document_ids")
    }
    return {
        **extra,
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": labels,
        "document_ids": torch.stack([b["document_ids"] for b in batch], dim=0),
        # The model's own denominator: it scores labels[:, start+1 : end+1], so
        # position 0 is never a target. Computed in the worker so the training
        # loop can weight microbatches by token count without a device sync.
        "n_loss_tokens": int((labels[:, 1:] != IGNORE_INDEX).sum()),
    }


def accum_weights(
    counts: Sequence[int], world_size: int = 1, total: Optional[int] = None
) -> List[float]:
    """Per-microbatch loss weights for token-weighted gradient accumulation.

    The model returns sum(CE)/count(valid) for its own microbatch and DDP
    averages gradients across ranks, so the weight that makes every
    loss-bearing token count equally is ``w_m = world_size * N_m / total``.
    With all counts equal and ``total`` the local sum this is exactly
    ``1/grad_accum`` -- i.e. the pretraining path is unchanged.

    ``total`` MUST be the GLOBAL loss-token count for the optimizer step (summed
    over every rank) whenever ``world_size > 1``. Derivation: DDP forms
    ``g = (1/W) sum_r g_r`` and we want ``g`` proportional to
    ``sum_r sum_m grad(sum CE_rm) / N_global``. Since each rank contributes
    ``g_r = sum_m w_rm * grad(sum CE_rm) / N_rm``, substituting
    ``w_rm = W * N_rm / N_global`` gives exactly that. Passing a per-rank sum
    instead equalises tokens WITHIN a rank but leaves ranks weighted by their
    own token mass, which is the cross-rank half of the bug and does not vanish
    at grad_accum=1.
    """
    if total is None:
        total = sum(int(c) for c in counts)
    total = max(int(total), 1)
    return [world_size * int(c) / total for c in counts]


def build_sft_dataloader(
    shards: Sequence[Path],
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    num_workers: int = 4,
    infinite: bool = True,
    long_policy: str = "skip",
    source_epochs: Optional[dict] = None,
    shuffle_buffer: int = 0,
    annotate: Optional[AnnotateOptions] = None,
) -> DataLoader:
    dataset = PackedSFTDataset(
        shards,
        tokenizer,
        seq_len=seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        infinite=infinite,
        long_policy=long_policy,
        source_epochs=source_epochs,
        shuffle_buffer=shuffle_buffer,
        annotate=annotate,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=sft_collate,
        drop_last=True,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _selftest() -> int:
    import sys
    import tempfile

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo.parent / "synthetic-agentic" / "src"))
    from scripts.sft_prep.record import build_record  # noqa: E402
    from umich_slm.tokenizer_defaults import load_default_tokenizer  # noqa: E402

    tok = load_default_tokenizer()
    convs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
        [
            {"role": "user", "content": "Read /etc/hosts please."},
            {
                "role": "assistant",
                "reasoning_content": "I should use the file tool.",
                "content": "",
                "tool_calls": [{"name": "file_read", "args": {"path": "/etc/hosts"}}],
            },
            {"role": "tool", "name": "file_read", "content": "127.0.0.1 localhost"},
            {"role": "assistant", "content": "It maps localhost to 127.0.0.1."},
        ],
    ]
    tools = [
        {
            "name": "file_read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        shard = Path(td) / "records.jsonl"
        with open(shard, "w") as fh:
            for i, msgs in enumerate(convs):
                rec = build_record(msgs, tools, source="selftest", index=i)
                fh.write(json.dumps(rec.as_dict()) + "\n")

        ds = PackedSFTDataset([shard], tok, seq_len=512, infinite=False)
        windows = list(ds)
        print(
            f"  {ds.stats['records']} records -> {len(windows)} window(s), "
            f"{ds.stats['skipped_long']} skipped long"
        )
        if not windows:
            print("  FAIL: no windows emitted")
            return 1

        w = windows[0]
        ids, labels, docs = w["input_ids"], w["labels"], w["document_ids"]
        for k, v in w.items():
            if v.shape[0] != 512:
                print(f"  FAIL: {k} length {v.shape[0]} != 512")
                failures += 1

        # The decisive check: every unmasked token must be assistant text.
        kept = tok.decode([int(t) for t, l in zip(ids, labels) if l != IGNORE_INDEX])
        masked_frac = float((labels == IGNORE_INDEX).sum()) / labels.numel()
        print(
            f"  loss-bearing tokens: {int((labels != IGNORE_INDEX).sum())} "
            f"({100 * (1 - masked_frac):.1f}% of window)"
        )
        print(f"  trained text: {kept[:150]!r}")
        for forbidden in ("You are a helpful assistant", "What is 2+2", "127.0.0.1 localhost"):
            if forbidden in kept:
                print(f"  FAIL: masked text leaked into the loss: {forbidden!r}")
                failures += 1
        for required in ("The answer is 4", "file_read"):
            if required not in kept:
                print(f"  FAIL: assistant text missing from the loss: {required!r}")
                failures += 1

        # Padding must be inert and distinguishable.
        n_pad = int((docs == 0).sum())
        if n_pad and int((labels[docs == 0] != IGNORE_INDEX).sum()):
            print("  FAIL: padding carries loss")
            failures += 1
        print(f"  document ids: {sorted(set(int(d) for d in docs))} ({n_pad} pad positions)")

        # THE partition test: the (rank x worker) readers must tile the
        # record stream exactly once. Sharding on rank alone -- what this did
        # -- gave every worker the same stream, i.e. 4x duplicate data at the
        # default num_workers, with nothing raised and no metric to notice it.
        # Needs enough records that every reader gets some: with fewer records
        # than readers, an empty reader is a real misconfiguration and the
        # guard is supposed to fire.
        big = Path(td) / "many.jsonl"
        n_rec = 240
        with open(big, "w") as fh:
            for i in range(n_rec):
                r = build_record(
                    [
                        {"role": "user", "content": f"question number {i}"},
                        {"role": "assistant", "content": f"answer number {i}"},
                    ],
                    [],
                    source="partition",
                    index=i,
                )
                fh.write(json.dumps(r.as_dict()) + "\n")

        class _FakeWorker:
            def __init__(self, wid, nw):
                self.id, self.num_workers = wid, nw

        import umich_slm.titan.sft_data as _mod

        for ws in (1, 2, 4):
            for nw in (1, 2, 4):
                texts: list[str] = []
                real = _mod.get_worker_info
                try:
                    for r in range(ws):
                        for w in range(nw):
                            _mod.get_worker_info = lambda _w=_FakeWorker(w, nw): _w
                            ds = PackedSFTDataset(
                                [big],
                                tok,
                                seq_len=512,
                                infinite=False,
                                rank=r,
                                world_size=ws,
                            )
                            for win in ds:
                                kept = [
                                    int(t)
                                    for t, l in zip(win["input_ids"], win["labels"])
                                    if l != IGNORE_INDEX
                                ]
                                texts.append(tok.decode(kept))
                finally:
                    _mod.get_worker_info = real
                blob = " ".join(texts)
                # every record's answer appears, exactly once, across the union
                missing = [i for i in range(n_rec) if f"answer number {i}" not in blob]
                dupes = [i for i in range(n_rec) if blob.count(f"answer number {i} ") > 1]
                if missing or dupes:
                    print(
                        f"  FAIL ws={ws} nw={nw}: {len(missing)} records missing, "
                        f"{len(dupes)} duplicated"
                    )
                    failures += 1
        print(f"  reader partition: {n_rec} records tile exactly once for " f"ws x nw in (1,2,4)^2")

        # Long-conversation policy. At seq_len=16 every record is skipped, so
        # the reader produces nothing -- which must RAISE rather than return an
        # empty stream. Under infinite=True that state is a silent hang, so the
        # guard is the difference between a clear error and a wedged job.
        tiny = PackedSFTDataset([shard], tok, seq_len=16, infinite=False, long_policy="skip")
        try:
            list(tiny)
            print("  FAIL: a reader that emits nothing must raise, not return empty")
            failures += 1
        except RuntimeError as e:
            if "emitted no window" not in str(e):
                print(f"  FAIL: unexpected RuntimeError: {e}")
                failures += 1
        if tiny.stats["skipped_long"] != 2:
            print(f"  FAIL: expected 2 skips at seq_len=16, got {tiny.stats['skipped_long']}")
            failures += 1
        trunc = PackedSFTDataset(
            [shard], tok, seq_len=16, infinite=False, long_policy="truncate_middle"
        )
        got = list(trunc)
        if trunc.stats["truncated"] != 2 or any(g["input_ids"].shape[0] != 16 for g in got):
            print("  FAIL: truncate_middle did not produce full windows")
            failures += 1
        print(
            f"  long policy: skip={tiny.stats['skipped_long']}, "
            f"truncate={trunc.stats['truncated']}"
        )

    return failures


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("nothing to do; pass --selftest")
    n = _selftest()
    print("all checks passed" if not n else f"FAILURES: {n}")
    raise SystemExit(1 if n else 0)
