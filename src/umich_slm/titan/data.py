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

import itertools
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _read_arrow_text_column(path: Path, text_column: str) -> List[str]:
    """Memory-map a single Arrow IPC file and pull a single string column.

    For the FineWeb-1B prepared file this is ~1-3 GB on disk; mmap keeps it
    out of RSS. Reading the column up front (rather than per-row) is fine
    because ``Dataset.from_list(...)`` already paid the materialization cost
    when the file was written.
    """
    with pa.memory_map(str(path), "r") as src:
        reader = ipc.open_file(src)
        table = reader.read_all()
    if text_column not in table.column_names:
        raise KeyError(
            f"Arrow file {path} has columns {table.column_names}; "
            f"expected '{text_column}'."
        )
    return table.column(text_column).to_pylist()


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

        eos = tokenizer.eos_token_id
        if eos is None:
            # Fall back to the BOS token, then to id 0 — packed pretraining
            # just needs *some* boundary marker.
            eos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        self.eos_id = int(eos)

        self._docs: Optional[List[str]] = None

    # -------------------------------------------------------------- internals
    def _load_docs(self) -> List[str]:
        if self._docs is None:
            self._docs = _read_arrow_text_column(self.arrow_path, self.text_column)
        return self._docs

    def _shuffled_indices(self, epoch: int, rank_offset: int, rank_stride: int) -> List[int]:
        n = len(self._load_docs())
        rng = random.Random(self.seed + 1000 * epoch)
        order = list(range(n))
        rng.shuffle(order)
        return order[rank_offset::rank_stride]

    def _doc_token_iter(self, indices: Iterable[int]) -> Iterator[int]:
        docs = self._load_docs()
        for idx in indices:
            text = docs[idx]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            for tok in ids:
                yield tok
            yield self.eos_id

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

        epoch = 0
        while True:
            indices = self._shuffled_indices(epoch, global_offset, global_stride)
            buf: List[int] = []
            for tok in self._doc_token_iter(indices):
                buf.append(tok)
                if len(buf) >= self.seq_len:
                    chunk = buf[: self.seq_len]
                    buf = buf[self.seq_len :]
                    yield torch.tensor(chunk, dtype=torch.long)
            if not self.infinite:
                return
            epoch += 1


def lm_collate(batch: List[torch.Tensor]) -> dict:
    """Stack pre-tokenized windows; labels = input_ids (model handles the shift)."""
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
) -> DataLoader:
    dataset = PackedFineWebDataset(
        arrow_path=Path(arrow_path),
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_column,
        seed=seed,
        rank=rank,
        world_size=world_size,
        infinite=infinite,
    )
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
