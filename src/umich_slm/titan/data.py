"""FineWeb arrow dataset + LM collator + dataloader builder.

Reuses the output of ``scripts/prepare_fineweb_dataset.py``:
``<dataset_dir>/dataset/{train,validation}.arrow`` (single-file Arrow IPC,
text column ``text``). Tokenizes on the fly with the SmolLM2 tokenizer that
ships in the bootstrap checkpoint, packs concatenated documents (separated by
EOS) into ``seq_len`` windows, and shards across DDP ranks via stride
slicing — matching the packing semantics LLaMA Factory uses for ``stage: pt``
runs with ``cutoff_len: 2048``.

The dataset is implemented as ``IterableDataset`` so packing is streamed (no
pre-tokenization pass). Tokenization is the dominant per-step CPU cost on
RTX Pro 6000 / H100, so we set ``num_workers >= 4`` in the dataloader to
hide it behind the GPU step.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .harmony_annotate import AnnotateOptions, HarmonyAnnotator


class _ArrowTextColumn:
    """Lazy, memory-mapped view of one string column of an Arrow IPC file.

    Holds the column as a zero-copy ``ChunkedArray`` backed by the mmap and
    converts exactly one document to a Python ``str`` per ``__getitem__``.
    This replaces ``to_pylist()``, which materializes the *entire* column as
    Python ``str`` objects — tens of GB for the 15B mix, and multiplied across
    dataloader workers (each ``_load_docs`` runs post-fork), the cause of the
    OOM kills on the full train arrow. With mmap-backed access, the OS shares
    one physical copy of the file across workers and RSS stays flat.

    Random access is ``O(log n_chunks)`` via a prefix-sum over chunk lengths;
    ``combine_chunks()`` is deliberately avoided because it would copy all
    string bytes into RSS, reintroducing the very problem we are fixing.
    """

    def __init__(self, path: Path, text_column: str):
        # Keep the mmap handle alive for this object's lifetime — the
        # ChunkedArray buffers point into it (zero-copy).
        self._src = pa.memory_map(str(path), "r")
        reader = ipc.open_file(self._src)
        table = reader.read_all()
        if text_column not in table.column_names:
            self._src.close()
            raise KeyError(
                f"Arrow file {path} has columns {table.column_names}; " f"expected '{text_column}'."
            )
        self._chunks = table.column(text_column).chunks
        # Prefix sums of chunk lengths: global index -> (chunk, local offset).
        self._starts: List[int] = [0]
        for chunk in self._chunks:
            self._starts.append(self._starts[-1] + len(chunk))
        self._n = self._starts[-1]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> str:
        ci = bisect.bisect_right(self._starts, idx) - 1
        value = self._chunks[ci][idx - self._starts[ci]].as_py()
        return value if value is not None else ""


def _read_arrow_text_column(path: Path, text_column: str) -> _ArrowTextColumn:
    """Memory-map a single Arrow IPC file for lazy per-document access.

    Returns a sequence-like ``_ArrowTextColumn`` (supports ``len()`` and
    integer indexing) rather than a fully materialized ``List[str]`` so the
    full-size train arrow can be streamed without OOM.
    """
    return _ArrowTextColumn(path, text_column)


class PackedFineWebDataset(IterableDataset):
    """Stream packed ``seq_len``-token windows from a prepared Arrow file.

    Sharding strategy (matching the plan):
        1. Global shuffle the document indices once, with ``seed``.
        2. Each DDP rank takes ``rank::world_size``.
        3. Within a rank, multiple dataloader workers further stride-shard.
    Each emitted example is a tensor of length ``seq_len + 1`` so the collator
    can split into ``input_ids``/``labels`` if a downstream caller wants
    explicit shifting; for our model ``labels = input_ids.clone()`` and the
    chunked-loss path inside the model handles the shift internally.
    """

    def __init__(
        self,
        arrow_path: Path,
        tokenizer,
        seq_len: int,
        text_column: str = "text",
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = True,
        emit_doc_ids: bool = False,
        annotate: Optional[AnnotateOptions] = None,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self.arrow_path = Path(arrow_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.seed = seed
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.infinite = infinite
        # Intra-document masking (W2): also emit per-token `document_ids` so the model can
        # block cross-document attention and reset RoPE per document. Off by default to keep
        # the existing (bare-tensor) contract for callers that don't need it.
        self.emit_doc_ids = emit_doc_ids
        # Per-token Harmony annotations (role / loss weight / failed-call labels /
        # unlikelihood targets). None or disabled -> the original path, unchanged.
        self.annotate = annotate if (annotate is not None and annotate.enabled) else None

        eos = tokenizer.eos_token_id
        if eos is None:
            # Fall back to the BOS token, then to id 0 — packed pretraining
            # just needs *some* boundary marker.
            eos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        self.eos_id = int(eos)

        self._docs: Optional[_ArrowTextColumn] = None

    # -------------------------------------------------------------- internals
    def _load_docs(self) -> _ArrowTextColumn:
        if self._docs is None:
            self._docs = _read_arrow_text_column(self.arrow_path, self.text_column)
        return self._docs

    def _shuffled_indices(self, epoch: int, rank_offset: int, rank_stride: int) -> List[int]:
        n = len(self._load_docs())
        rng = random.Random(self.seed + 1000 * epoch)
        order = list(range(n))
        rng.shuffle(order)
        return order[rank_offset::rank_stride]

    def _doc_token_iter(self, indices: Iterable[int]) -> Iterator[tuple]:
        """Yield ``(token_id, doc_id)`` for each token, with a trailing EOS per document.

        ``doc_id`` increments per source document (its EOS shares the doc's id), giving the
        contiguous per-document runs that intra-document masking consumes.
        """
        docs = self._load_docs()
        doc_id = 0
        for idx in indices:
            text = docs[idx]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            for tok in ids:
                yield tok, doc_id
            yield self.eos_id, doc_id
            doc_id += 1

    # -------------------------------------------------------- annotated path
    def _iter_annotated(self, global_offset: int, global_stride: int) -> Iterator[dict]:
        """Same packing as ``__iter__`` (EOS-separated docs, ``seq_len`` windows,
        incrementing doc ids), but each window is a dict that also carries the
        per-token Harmony annotations (see ``titan/harmony_annotate.py``).

        Window guard for unlikelihood: a document can be split across windows,
        and an unlikelihood target is only meaningful when the failing call --
        and the error that followed it -- is visible. A target is dropped when
        its failing call starts before this window's fragment of the document.
        """
        annot = HarmonyAnnotator(self.tokenizer, self.annotate)
        self.annotator = annot  # stats live here (per worker)
        opts = self.annotate
        S = self.seq_len
        docs = self._load_docs()
        epoch = 0
        while True:
            indices = self._shuffled_indices(epoch, global_offset, global_stride)
            parts: dict = {k: [] for k in ("ids", "doc", "pos", "role", "err", "unlik", "need")}
            pending = 0
            doc_id = 0
            for idx in indices:
                text = docs[idx]
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                a = annot.from_text(ids, text)
                n = len(ids) + 1  # trailing EOS shares the doc's id and is role 0
                parts["ids"].append(np.asarray(ids + [self.eos_id], dtype=np.int64))
                parts["doc"].append(np.full(n, doc_id, dtype=np.int64))
                parts["pos"].append(np.arange(n, dtype=np.int64))
                parts["role"].append(np.append(a["token_role"], np.int8(0)))
                parts["err"].append(np.append(a["errhead_labels"], np.int8(-1)))
                parts["unlik"].append(np.append(a["unlik_targets"], np.int64(-100)))
                parts["need"].append(np.append(a["unlik_need"], np.int64(-1)))
                pending += n
                doc_id += 1
                while pending >= S:
                    cat = {k: np.concatenate(v) for k, v in parts.items()}
                    win = {k: v[:S] for k, v in cat.items()}
                    parts = {k: [v[S:]] for k, v in cat.items()}
                    pending -= S
                    yield self._annotated_window(win, opts)
            if not self.infinite:
                return
            epoch += 1

    @staticmethod
    def _annotated_window(win: dict, opts: AnnotateOptions) -> dict:
        unlik = win["unlik"].copy()
        hits = np.flatnonzero(unlik != -100)
        if hits.size:
            doc = win["doc"]
            first = np.searchsorted(doc, doc[hits], side="left")  # doc ids ascend
            visible_from = win["pos"][first]
            unlik[hits[win["need"][hits] < visible_from]] = -100
        out = {
            "input_ids": torch.from_numpy(win["ids"].copy()),
            "document_ids": torch.from_numpy(win["doc"].copy()),
            "token_role": torch.from_numpy(win["role"].astype(np.int64)),
        }
        if opts.errhead:
            out["errhead_labels"] = torch.from_numpy(win["err"].astype(np.int64))
        if opts.unlik:
            out["unlik_targets"] = torch.from_numpy(unlik)
        if opts.emit_weights:
            out["loss_weights"] = torch.from_numpy(
                np.where(win["role"] == 2, np.float32(opts.w_tool), np.float32(1.0)).astype(np.float32)
            )
        return out

    # ----------------------------------------------------------------- iter
    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        # Combined rank+worker stride so every (rank, worker) pair sees a
        # disjoint subset of the document index space.
        global_offset = self.rank * num_workers + worker_id
        global_stride = self.world_size * num_workers

        if self.annotate is not None:
            yield from self._iter_annotated(global_offset, global_stride)
            return

        epoch = 0
        while True:
            indices = self._shuffled_indices(epoch, global_offset, global_stride)
            buf: List[int] = []
            dbuf: List[int] = []
            for tok, did in self._doc_token_iter(indices):
                buf.append(tok)
                dbuf.append(did)
                if len(buf) >= self.seq_len:
                    chunk = buf[: self.seq_len]
                    buf = buf[self.seq_len :]
                    ids_t = torch.tensor(chunk, dtype=torch.long)
                    if self.emit_doc_ids:
                        dchunk = dbuf[: self.seq_len]
                        dbuf = dbuf[self.seq_len :]
                        yield ids_t, torch.tensor(dchunk, dtype=torch.long)
                    else:
                        dbuf = dbuf[self.seq_len :]
                        yield ids_t
            if not self.infinite:
                return
            epoch += 1


class ReplayMixDataset(IterableDataset):
    """Interleave whole packed windows from a primary and a replay dataset.

    Window ``i`` of each (rank, worker) stream is replay iff
    ``floor((i + 1) * frac) > floor(i * frac)``: a deterministic Bresenham schedule,
    so the replay share is exact over any prefix rather than right only in
    expectation, and every run with the same ``frac`` sees the same interleave.
    Both inner datasets shard by rank and worker themselves, so each stream stays
    disjoint. Windows never mix pools, which keeps ``document_ids`` intact.
    """

    def __init__(self, primary: PackedFineWebDataset, replay: PackedFineWebDataset, frac: float):
        super().__init__()
        if not 0.0 < frac < 1.0:
            raise ValueError(f"replay frac must be in (0, 1), got {frac}")
        self.primary, self.replay, self.frac = primary, replay, frac

    def __iter__(self):
        a, b = iter(self.primary), iter(self.replay)
        i = 0
        while True:
            take_replay = math.floor((i + 1) * self.frac) > math.floor(i * self.frac)
            try:
                yield next(b) if take_replay else next(a)
            except StopIteration:
                return
            i += 1


def lm_collate(batch: List) -> dict:
    """Stack pre-tokenized windows; labels = input_ids (model handles the shift).

    Accepts either bare ``input_ids`` tensors or ``(input_ids, document_ids)`` pairs (the
    latter when the dataset runs with ``emit_doc_ids=True`` for intra-document masking),
    or dicts from the annotated path (every key stacked; see ``harmony_annotate``).
    """
    if isinstance(batch[0], dict):
        out = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}
        out["labels"] = out["input_ids"].clone()
        return out
    if isinstance(batch[0], (tuple, list)):
        input_ids = torch.stack([b[0] for b in batch], dim=0)
        document_ids = torch.stack([b[1] for b in batch], dim=0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "document_ids": document_ids,
        }
    input_ids = torch.stack(batch, dim=0)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def build_dataloader(
    arrow_path: Path,
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    text_column: str = "text",
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    num_workers: int = 8,
    infinite: bool = True,
    emit_doc_ids: bool = False,
    replay_arrow_path: Optional[Path] = None,
    replay_frac: float = 0.0,
    annotate: Optional[AnnotateOptions] = None,
) -> DataLoader:
    def packed(path: Path) -> PackedFineWebDataset:
        return PackedFineWebDataset(
            arrow_path=Path(path),
            tokenizer=tokenizer,
            seq_len=seq_len,
            text_column=text_column,
            seed=seed,
            rank=rank,
            world_size=world_size,
            infinite=infinite,
            emit_doc_ids=emit_doc_ids,
            annotate=annotate,
        )

    dataset = packed(arrow_path)
    if replay_frac > 0:
        if replay_arrow_path is None:
            raise ValueError("replay_frac > 0 requires replay_arrow_path")
        dataset = ReplayMixDataset(dataset, packed(replay_arrow_path), replay_frac)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=lm_collate,
        drop_last=True,
    )
