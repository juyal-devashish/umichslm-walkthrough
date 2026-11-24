#!/usr/bin/env python3
"""Prepare a budget-sized pretraining mix from the prepared bucket corpus.

Samples the 11 prepared buckets under
``/shared/6/projects/umich-slm/data/prepared/<bucket>/<source>/shard_*.jsonl.gz``
according to a mixture, takes enough documents to hit a token budget, and saves
train/val Arrow splits plus ``dataset_info.json`` for LLaMA Factory / the scaling
sweep (``scripts/scaling_laws.py``).

The default mixture is the **natural prepared proportions** — each bucket is
weighted by its realized ``est_tokens`` share from
``prepared/manifests/fileset_v1.json``. This is the reasoning-heavy mix described
in ``PRETRAIN_PLAN.md §6.2`` (the prepared corpus was built to those bucket
targets), e.g. roughly web 25% / synthetic 23% / reasoning_traces 13% /
math 12% / nemotron 12% / agent_traj 6% / python 5% / arxiv 3% / (shell,
tool_calling, math_rl) <0.1% each.

All buckets share one schema: ``{"text", "doc_id", "source"}``, gzip-compressed
JSONL. Only ``text`` is retained.

Example usage:
    # Natural-proportion 1B-token mix (defaults: prepared_root + manifest weights)
    python scripts/prepare_mix_dataset.py --train_tokens 1_000_000_000 \
        --output_dir /shared/6/projects/umich-slm/data/scaling_law_datasets/1B

    # Build the full scaling-sweep budget ladder
    for B in 20 40 200 500 1000; do
      python scripts/prepare_mix_dataset.py --train_tokens ${B}_000_000 \
        --output_dir /shared/6/projects/umich-slm/data/scaling_law_datasets/${B}M
    done

    # Override the mixture explicitly (weights are normalized; missing buckets -> 0)
    python scripts/prepare_mix_dataset.py --train_tokens 200_000_000 \
        --output_dir ./data/build/200M \
        --mixture_ratio '{"web":0.4,"reasoning_traces":0.3,"math":0.2,"python":0.1}'

    # Restrict to a subset of buckets (still weighted by natural proportions)
    python scripts/prepare_mix_dataset.py --buckets web,math,reasoning_traces ...

    # Sequence packing: tokenize + pack into fixed-length input_ids Arrow files
    python scripts/prepare_mix_dataset.py --output_dir ./data/build/1B_packed --pack

    # Dry run to see the resolved mixture without building
    python scripts/prepare_mix_dataset.py --dry_run
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, interleave_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_PREPARED_ROOT = "/shared/6/projects/umich-slm/data/prepared"
MANIFEST_REL = "manifests/fileset_v1.json"
# Subdirectories under prepared_root that are not data buckets.
_NON_BUCKET_DIRS = {"manifests"}
# Long-source keys "sft:<stage>/<source>" read the prepared SFT corpus (whole rendered
# Harmony conversations in its `text` field) with its decontamination + dedup drops.
SFT_PREFIX = "sft:"
DEFAULT_SFT_ROOT = "/shared/6/projects/umich-slm/data/sft"

try:
    from umich_slm.tokenizer_defaults import get_tokenizer_path
except Exception:  # package not importable in some ad-hoc invocations

    def get_tokenizer_path() -> str:
        import os

        return os.environ.get(
            "UMICHSLM_TOKENIZER_PATH",
            "/shared/6/projects/umich-slm/outputs/tokenizer/"
            "20260501-123305/tokenizers/final_hf_tokenizer",
        )


# Long-context length-weighting (W4). Kept in-package so both stacks share the config;
# fall back to a local import path when the package isn't installed for an ad-hoc run.
try:
    from umich_slm.data.long_context_mix import (
        DEFAULT_CHARS_PER_TOKEN,
        is_long_text,
        reallocate_long_mass,
        resolve_stage,
    )
except Exception:  # pragma: no cover - exercised only outside an editable install
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from umich_slm.data.long_context_mix import (
        DEFAULT_CHARS_PER_TOKEN,
        is_long_text,
        reallocate_long_mass,
        resolve_stage,
    )


def setup_tokenizer(tokenizer_name=None):
    """Load a tokenizer used purely for token *counting* (to hit the budget).

    Defaults to the project SuperBPE-56000 tokenizer so budgets are in production
    tokens (see umich_slm.tokenizer_defaults / TOKENIZER_TRAINING_PLAN.md).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or get_tokenizer_path(), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Bucket discovery + natural-proportion weights
# ---------------------------------------------------------------------------


def discover_buckets(prepared_root: Path) -> list[str]:
    """Return the bucket names (subdirs of prepared_root that hold shards)."""
    buckets = []
    for child in sorted(prepared_root.iterdir()):
        if not child.is_dir() or child.name in _NON_BUCKET_DIRS:
            continue
        if any(child.rglob("*.jsonl.gz")):
            buckets.append(child.name)
    return buckets


def natural_bucket_weights(prepared_root: Path, buckets: list[str]) -> dict:
    """Weight each bucket by its realized est_tokens share (from the manifest).

    Falls back to total compressed shard bytes if the manifest is missing, so
    the script still runs against a partial/unmanifested tree.
    """
    manifest_path = prepared_root / MANIFEST_REL
    tokens = {b: 0 for b in buckets}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, entry in manifest.get("buckets", {}).items():
            bucket = key.split("/", 1)[0]
            if bucket in tokens:
                tokens[bucket] += int(entry.get("est_tokens", 0))
    if sum(tokens.values()) == 0:
        # Fallback: proxy weight by on-disk shard bytes.
        for b in buckets:
            tokens[b] = sum(p.stat().st_size for p in (prepared_root / b).rglob("*.jsonl.gz"))
    total = sum(tokens.values()) or 1
    return {b: tokens[b] / total for b in buckets}


def resolve_mixture(
    prepared_root: Path,
    buckets: list[str],
    override: dict | None,
) -> dict:
    """Resolve the final {bucket: probability} mixture (normalized, sums to 1)."""
    if override:
        # Keep only buckets that actually exist; warn on unknown keys.
        unknown = [b for b in override if b not in buckets]
        if unknown:
            print(f"Warning: ignoring unknown buckets in mixture_ratio: {unknown}")
        weights = {b: float(override[b]) for b in buckets if b in override and override[b] > 0}
        if not weights:
            raise ValueError("mixture_ratio selects no available bucket.")
    else:
        weights = natural_bucket_weights(prepared_root, buckets)
        weights = {b: w for b, w in weights.items() if w > 0}
    total = sum(weights.values()) or 1.0
    return {b: w / total for b, w in sorted(weights.items())}


_SFT_ROOT = [Path(DEFAULT_SFT_ROOT)]
_SFT_DROPS: dict = {}


def sft_source_dir(key: str) -> Path:
    """``sft:<stage>/<source>`` -> ``<sft_root>/prepared/<stage>/<source>``.

    Several roots are searched in order. The second is the pretraining-only tree
    (``data/sft_pretrain``: the rows SFT filters out -- unresolved, failed, judged
    incorrect), which keeps its own decon/dedup manifests because it is prepped on
    its own schedule; its stage dir (``pretrain``) keeps its doc_ids distinct.
    """
    rel = key[len(SFT_PREFIX):]
    for root in _SFT_ROOT:
        if (root / "prepared" / rel).is_dir():
            return root / "prepared" / rel
    return _SFT_ROOT[0] / "prepared" / rel


def sft_drops() -> set:
    """doc_ids removed by SFT decontamination + dedup, loaded (and staleness-checked
    against the current prep) once. The same list the SFT mix applies: doc_ids are
    ordinals, so ``build_mix.load_drops`` refuses a list computed against an older prep."""
    roots = tuple(_SFT_ROOT)
    if roots not in _SFT_DROPS:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.sft_prep.build_mix import load_drops

        drops: set = set()
        for root in roots:
            got = load_drops(root / "manifests", root / "prepared")
            print(f"  [sft] {root}: {len(got):,} decon/dedup doc_ids will be dropped")
            drops |= got
        _SFT_DROPS[roots] = drops
    return _SFT_DROPS[roots]


def load_sft_stream(key, min_tokens, chars_per_token, shuffle_seed, shuffle_buffer):
    """Stream one prepared SFT source as ``{"text": ...}``, minus decon/dedup drops.

    A plain generator, not ``load_dataset("json")``: SFT records carry nested
    ``messages``/``tools``/``segments``/``meta`` whose inferred Arrow schema differs
    across shards. A raw line shorter than the char threshold cannot hold a long
    ``text`` (the line contains it), so it is skipped before ``json.loads``. A doc_id
    seen twice is kept once (prep's old resume bug duplicated whole passes; see
    ``build_mix.iter_source``).
    """
    import gzip

    src = sft_source_dir(key)
    shards = sorted(src.glob("shard_*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No shards found for {key} under {src}")
    drops = sft_drops()
    min_chars = int(min_tokens * chars_per_token) if min_tokens > 0 else 0
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(shards)

    def gen():
        seen: set = set()
        rng = random.Random(shuffle_seed)
        buf: list = []
        for shard in shards:
            with gzip.open(shard, "rt") as fh:
                for line in fh:
                    if len(line) < min_chars:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    did, text = rec.get("doc_id"), rec.get("text") or ""
                    if did in drops or did in seen or len(text) < min_chars:
                        continue
                    if did is not None:
                        seen.add(did)
                    if shuffle_seed is None:
                        yield {"text": text}
                        continue
                    buf.append({"text": text})
                    if len(buf) >= shuffle_buffer:
                        yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf

    return gen(), len(shards)


def load_bucket_stream(
    prepared_root: Path,
    bucket: str,
    min_tokens: int = 0,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 256,
):
    """Stream a single bucket's shards as a {'text': ...} dataset.

    ``bucket`` may also be a ``bucket/source`` path, which streams one source only.

    When ``min_tokens > 0``, restrict to the long-document pool: keep only docs whose
    char-length proxy implies >= ``min_tokens`` tokens (W4 length routing). The filter uses
    the cheap char proxy (``is_long_text``) so no second tokenization pass is needed.

    ``shuffle_seed`` shuffles the SHARD order (and docs within a small buffer). Without
    it a stream reads its sources/shards in sorted order, so any build smaller than the
    bucket is a slice of the alphabetically-first source's first shards -- fine for a
    budget ladder that wants nested prefixes, wrong for a small pool that should
    represent the whole source.
    """
    if bucket.startswith(SFT_PREFIX):
        return load_sft_stream(bucket, min_tokens, chars_per_token, shuffle_seed, shuffle_buffer)
    shards = sorted(str(p) for p in (prepared_root / bucket).rglob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No shards found for bucket {bucket} under {prepared_root}")
    ds = load_dataset("json", data_files=shards, split="train", streaming=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    # Every prepared shard has a "text" field; drop the rest (doc_id, source).
    ds = ds.map(
        lambda x: {"text": x["text"]},
        remove_columns=list(ds.features.keys()) if ds.features else None,
    )
    if min_tokens > 0:
        ds = ds.filter(lambda x: is_long_text(x["text"], min_tokens, chars_per_token))
    return ds, len(shards)


# ---------------------------------------------------------------------------
# Sequence packing (unchanged machinery, reused for --pack)
# ---------------------------------------------------------------------------


def pack_sequences(
    token_id_lists: list[list[int]],
    eos_token_id: int,
    seq_length: int,
    seed: int = 42,
) -> tuple[list[list[int]], dict]:
    """Pack variable-length token-id lists into fixed-length sequences.

    Documents are shuffled, then concatenated with EOS separators and chunked
    into seq_length blocks. The trailing partial block is discarded so every
    returned sequence has exactly seq_length tokens.
    """
    rng = random.Random(seed)
    order = list(range(len(token_id_lists)))
    rng.shuffle(order)

    buffer: list[int] = []
    packed: list[list[int]] = []
    total_doc_tokens = 0

    for idx in tqdm(order, desc="  packing docs", leave=False):
        ids = token_id_lists[idx]
        total_doc_tokens += len(ids)
        if buffer:
            buffer.append(eos_token_id)
        buffer.extend(ids)
        while len(buffer) >= seq_length:
            packed.append(buffer[:seq_length])
            buffer = buffer[seq_length:]

    stats = {
        "num_packed_sequences": len(packed),
        "num_input_docs": len(token_id_lists),
        "total_doc_tokens": total_doc_tokens,
        "packed_tokens": len(packed) * seq_length,
        "discarded_tokens": len(buffer),
        "utilization_pct": round(len(packed) * seq_length / max(total_doc_tokens, 1) * 100, 1),
    }
    return packed, stats


def _write_packed_arrow(sequences: list[list[int]], path: Path) -> None:
    """Write a list of fixed-length token-id sequences to an Arrow IPC file."""
    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    ids_col = pa.array(sequences, type=pa.list_(pa.int32()))
    table = pa.table({"input_ids": ids_col}, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _pack_and_save(
    full_token_ids: list[list[int]],
    train_split_idx: int,
    eos_token_id: int,
    seq_length: int,
    seed: int,
    output_dir: Path,
    train_tokens: int,
    val_tokens: int,
    mixture: dict,
) -> dict:
    """Pack pre-tokenized documents and save as fixed-length Arrow sequences."""
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot insert document separators.")

    train_ids_list = full_token_ids[:train_split_idx]
    val_ids_list = full_token_ids[train_split_idx:]

    print("\nPacking train sequences...")
    train_seqs, train_stats = pack_sequences(train_ids_list, eos_token_id, seq_length, seed)
    print(
        f"  {train_stats['num_packed_sequences']:,} packed sequences "
        f"({train_stats['utilization_pct']}% utilization, "
        f"{train_stats['discarded_tokens']:,} tokens discarded)"
    )

    print("Packing val sequences...")
    val_seqs, val_stats = pack_sequences(val_ids_list, eos_token_id, seq_length, seed + 1)
    print(
        f"  {val_stats['num_packed_sequences']:,} packed sequences "
        f"({val_stats['utilization_pct']}% utilization, "
        f"{val_stats['discarded_tokens']:,} tokens discarded)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train_packed.arrow"
    val_path = dataset_dir / "validation_packed.arrow"

    print(f"\nSaving packed dataset to {output_dir}...")
    _write_packed_arrow(train_seqs, train_path)
    print(f"Saved packed train to {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")

    _write_packed_arrow(val_seqs, val_path)
    print(f"Saved packed val to   {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")

    stats = {
        "packed": True,
        "seq_length": seq_length,
        "seed": seed,
        "mixture": mixture,
        "target_train_tokens": train_tokens,
        "target_val_tokens": val_tokens,
        "train": train_stats,
        "val": val_stats,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\n" + "=" * 60)
    print("Packed dataset ready!")
    print("=" * 60)
    print("\nTrain with pre-tokenized packed sequences:")
    print("   python scripts/pretrain_packed.py \\")
    print("       --model_path <initialized-model-dir> \\")
    print(f"       --train_file {train_path} \\")
    print(f"       --val_file {val_path} \\")
    print("       --output_dir ./outputs/packed-pretrain")

    return stats


# ---------------------------------------------------------------------------
# Main dataset preparation
# ---------------------------------------------------------------------------


def interleave_drop_exhausted(streams, probs, seed, names=None, events=None):
    """Sample documents by mixture, DROPPING a bucket that runs dry and
    renormalizing over the survivors instead of halting the whole mix.

    WHY NOT ``interleave_datasets``. Its default ``stopping_strategy`` is
    "first_exhausted": the moment ANY bucket runs out the whole stream ends.
    That interacts badly with the fact that the mixture weights are TOKEN shares
    while the draw is per DOCUMENT -- a bucket of unusually long documents drains
    far ahead of the rest. arxiv averages 60 kB/doc against a 4.9 kB corpus mean,
    so it emptied after 647k draws and halted the 60B build at 44.15B tokens with
    91.7% of the corpus never read.

    Dropping the empty bucket and renormalizing keeps the mix at the proportions
    the corpus can actually supply: every bucket contributes its natural share
    for as long as it has documents, and its share simply stops growing once it
    does not. That is the intended semantics -- the mixture reflects what is in
    the data, and a bucket is not silently duplicated to keep a nominal ratio.

    NOTE this does not reproduce HF's draw order, so a build made here is not
    the same document sequence as one made by ``interleave_datasets``. Val
    membership is unaffected: it is a hash of the text (:func:`is_holdout`), not
    a position in the stream.
    """
    rng = random.Random(seed)
    its = [iter(s) for s in streams]
    live = list(range(len(its)))
    drawn = 0
    while live:
        i = rng.choices(live, weights=[probs[j] for j in live], k=1)[0]
        try:
            yield next(its[i])
            drawn += 1
        except StopIteration:
            live.remove(i)
            nm = names[i] if names else str(i)
            print(f"\n  [mix] bucket exhausted: {nm} after {drawn:,} docs drawn; "
                  f"renormalizing over {len(live)} remaining")
            if events is not None:
                events.append({"bucket": nm, "at_docs_drawn": drawn})


class TokenShareInterleaver:
    """Draw documents so each stream's ACCEPTED TOKENS track its target share.

    WHY. ``interleave_drop_exhausted`` treats the weights as per-document pick
    probabilities, so a stream's token share is weight x mean doc length x the
    fraction of picks that survive ``--long_exact_filter``. The first 128K pool
    (2026-10-27) asked for 12% SWE traces and got 0.8% of tokens: SWE traces pass
    the lenient char prefilter at 32-64K real tokens and are then dropped, while
    algstack's survivors are 128K+ apiece.

    HOW. Always draw from the live stream furthest below its target (target share
    of the running total minus its accepted + pending tokens). A drawn document is
    pending at ``len(text) / chars_per_token`` -- the per-stream ratio is learned
    from what has been tokenized -- until :meth:`settle` replaces that estimate with
    the tokens actually accepted into train (0 for an exact-filter drop or a
    holdout doc), so a dropped document is in effect replaced by the next one from
    the same stream. An exhausted stream leaves; the rest renormalize over the
    survivors, as in ``interleave_drop_exhausted``. Deterministic given the streams.
    """

    def __init__(self, streams, shares, names, chars_per_token, events=None):
        self.its = [iter(s) for s in streams]
        self.names = list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        tot = sum(shares) or 1.0
        self.share = [s / tot for s in shares]
        n = len(self.its)
        self.got, self.pend = [0.0] * n, [0.0] * n
        self.chars, self.toks = [0] * n, [0] * n
        self.cpt0 = chars_per_token
        self.fifo = [[] for _ in range(n)]
        self.events = events

    def _cpt(self, i):
        return self.chars[i] / self.toks[i] if self.toks[i] else self.cpt0

    def __iter__(self):
        live = list(range(len(self.its)))
        drawn = 0
        while live:
            tot = sum(self.got[j] + self.pend[j] for j in live)
            z = sum(self.share[j] for j in live)
            i = max(live, key=lambda j: self.share[j] / z * tot - self.got[j] - self.pend[j])
            try:
                item = next(self.its[i])
            except StopIteration:
                live.remove(i)
                print(f"\n  [mix] stream exhausted: {self.names[i]} after {drawn:,} docs "
                      f"drawn; renormalizing over {len(live)} remaining")
                if self.events is not None:
                    self.events.append({"bucket": self.names[i], "at_docs_drawn": drawn})
                continue
            text = item[1]["text"] if isinstance(item, tuple) else item["text"]
            est = len(text) / self._cpt(i)
            self.pend[i] += est
            self.fifo[i].append(est)
            drawn += 1
            yield item

    def settle(self, name, n_chars, n_tokens, accepted_tokens):
        """Book one drawn document of stream ``name`` (in draw order per stream)."""
        i = self.idx[name]
        self.pend[i] -= self.fifo[i].pop(0)
        self.chars[i] += n_chars
        self.toks[i] += n_tokens
        self.got[i] += accepted_tokens

    def realized(self) -> dict:
        tot = sum(self.got) or 1.0
        return {n: round(self.got[i] / tot, 4) for i, n in enumerate(self.names)}


HOLDOUT_MODULUS = 1_000_000

# Token-mass-by-document-length bins reported in stats.json (upper bounds, exclusive).
DOCLEN_BINS = (4096, 8192, 16384, 32768, 65536)


def doclen_bin(n_tokens: int) -> str:
    lo = 0
    for hi in DOCLEN_BINS:
        if n_tokens < hi:
            return f"[{lo},{hi})"
        lo = hi
    return f"[{lo},inf)"


def _tagged(stream, name):
    """Yield ``(stream_name, sample)`` so per-stream token counts can be recorded."""
    for sample in stream:
        yield name, sample


def resolve_long_source_weights(prepared_root: Path, weights: dict) -> dict:
    """Normalize ``{"bucket/source": weight}`` over sources that exist and have shards."""
    out = {}
    for key, w in weights.items():
        if key.startswith("_"):  # "_comment" / provenance keys
            continue
        if float(w) <= 0:
            continue
        src = sft_source_dir(key) if key.startswith(SFT_PREFIX) else prepared_root / key
        if not src.is_dir() or not any(src.rglob("*.jsonl.gz")):
            raise FileNotFoundError(f"long source {key!r}: no shards under {src}")
        out[key] = float(w)
    total = sum(out.values())
    if not total:
        raise ValueError("long_source_weights selects no source")
    return {k: v / total for k, v in sorted(out.items())}


def is_holdout(text: str, holdout_frac: float) -> bool:
    """Deterministic per-DOCUMENT validation membership, from the text alone.

    WHY NOT THE EXISTING TAIL SPLIT. ``prepare_dataset`` takes documents off one
    seeded ``interleave_datasets`` stream until the budget is met, so a bigger
    build is a strict PREFIX SUPERSET of a smaller one -- which is exactly what a
    D-scaling axis wants, and exactly what breaks a positional val. The 15B mix's
    val is the stream just past 15.45B tokens, so it sits INSIDE the train split
    of every larger build: a 60B run would train on the set the 15B cells are
    scored against, and the comparison would silently favour the newer cells.

    Hashing the document instead makes membership independent of position and of
    budget, so one val set is disjoint from every train mix we will ever cut --
    15B, 60B, or the full-corpus production run -- with no coordination between
    builds and no second pass over the corpus.

    blake2b, not ``hash()``: Python salts string hashing per process, so ``hash``
    would put a document in train on one build and val on the next.
    """
    if holdout_frac <= 0.0:
        return False
    h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return (int.from_bytes(h, "big") % HOLDOUT_MODULUS) < holdout_frac * HOLDOUT_MODULUS


def prepare_dataset(
    train_tokens: int,
    val_tokens: int,
    output_dir: Path,
    prepared_root: Path,
    mix_name: str,
    seq_length: int = 2048,
    buckets: list[str] | None = None,
    mixture_override: dict | None = None,
    seed: int = 42,
    tokenizer_name: str | None = None,
    dry_run: bool = False,
    pack: bool = False,
    long_buckets: list[str] | None = None,
    long_threshold_tokens: int = 0,
    long_weight: float = 0.0,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    holdout_frac: float = 0.0,
    long_only: bool = False,
    long_exact_filter: bool = False,
    long_source_weights: dict | None = None,
    shuffle_streams: bool = False,
    weight_by: str = "docs",
    sft_root: str | None = None,
    encode_batch: int | None = None,
) -> dict:
    """Build a budget-sized pretraining mix from the prepared bucket corpus.

    Long-context length weighting (W4): when ``long_threshold_tokens > 0`` and
    ``long_weight > 0``, a length-filtered copy of ``long_buckets`` (docs with >=
    ``long_threshold_tokens`` tokens by char proxy) is interleaved with ``long_weight`` of the
    total probability mass, so long documents are up-weighted as context grows.

    ``long_only`` drops the base streams: the pool is ONLY length-filtered documents (a
    context-extension stage, rather than a mixture that up-weights long docs).
    ``long_exact_filter`` re-checks every long-stream document against
    ``long_threshold_tokens`` on its REAL token count after tokenization (which the
    budget loop does anyway), so the char proxy only has to be a lenient prefilter --
    chars/token runs ~2.5 on code to ~4.5 on prose, so no single proxy ratio is right.
    ``long_source_weights`` (``{"bucket/source": w}``) replaces the per-bucket long
    streams with one stream per named source, so a small pool is not dominated by
    whichever source sorts first in its bucket. Keys ``sft:<stage>/<source>`` read the
    prepared SFT corpus under ``sft_root`` with its decon/dedup drops applied.

    ``weight_by="tokens"`` makes every weight a share of ACCEPTED train tokens
    (:class:`TokenShareInterleaver`); the default ``"docs"`` keeps the historical
    per-document pick probabilities, so existing pools rebuild identically.
    """
    if weight_by not in ("docs", "tokens"):
        raise ValueError(f"weight_by must be 'docs' or 'tokens', not {weight_by!r}")
    if sft_root:
        roots = [sft_root] if isinstance(sft_root, (str, Path)) else list(sft_root)
        _SFT_ROOT[:] = [Path(r) for r in roots]
    output_dir = Path(output_dir)
    prepared_root = Path(prepared_root)
    total_tokens = train_tokens + val_tokens

    available = discover_buckets(prepared_root)
    if not available:
        raise FileNotFoundError(f"No buckets with *.jsonl.gz shards under {prepared_root}")
    # A "bucket/source" key is its own base stream (load_bucket_stream already reads one
    # source), so a pool can weight sources inside a bucket -- e.g. the tool-using
    # trajectories of agent_traj apart from its chat sources. Never list a source AND its
    # bucket: the bucket stream would read that source's documents a second time.
    available += [b for b in (buckets or []) if "/" in b and not b.startswith(SFT_PREFIX)
                  and (prepared_root / b).is_dir() and any((prepared_root / b).rglob("*.jsonl.gz"))]
    selected = [b for b in (buckets or available) if b in available]
    clash = [b for b in selected if "/" in b and b.split("/", 1)[0] in selected]
    if clash:
        raise ValueError(f"sources selected together with their bucket: {clash}")
    if buckets:
        missing = [b for b in buckets if b not in available]
        if missing:
            print(f"Warning: requested buckets not found and skipped: {missing}")
    if not selected:
        raise ValueError("No valid buckets selected.")

    mixture = resolve_mixture(prepared_root, selected, mixture_override)

    # W4 length weighting: reallocate `long_weight` of the mass to filtered long streams.
    if long_only and long_threshold_tokens <= 0:
        raise ValueError("long_only needs long_threshold_tokens > 0")
    long_enabled = long_threshold_tokens > 0 and (long_weight > 0.0 or long_only)
    if long_enabled and long_source_weights:
        long_mass = 1.0 if long_only else long_weight
        src_w = resolve_long_source_weights(prepared_root, long_source_weights)
        long_probs = {k: long_mass * w for k, w in src_w.items()}
        base_probs = {} if long_only else {b: p * (1.0 - long_weight) for b, p in mixture.items()}
    elif long_enabled:
        eligible = [b for b in (long_buckets or []) if b in mixture]
        base_probs, long_probs = reallocate_long_mass(
            mixture, eligible, long_weight if not long_only else 0.5
        )
        if long_only:
            tot = sum(long_probs.values()) or 1.0
            base_probs, long_probs = {}, {b: p / tot for b, p in long_probs.items()}
    else:
        base_probs, long_probs = dict(mixture), {}
    if not base_probs and not long_probs:
        raise ValueError("resolved mixture is empty")

    # Effective mixture for provenance (stats.json): long streams tagged with their threshold.
    effective_mixture = dict(base_probs)
    for b, p in long_probs.items():
        effective_mixture[f"{b}__long>={long_threshold_tokens}"] = p

    print("=" * 60)
    print("Pretraining Mix Dataset Preparation")
    print("=" * 60)
    print(f"Prepared root:            {prepared_root}")
    print(f"Mix name:                 {mix_name}")
    print(f"Training tokens target:   {train_tokens:,} ({train_tokens/1e9:.3f}B)")
    print(f"Validation tokens target: {val_tokens:,} ({val_tokens/1e6:.2f}M)")
    print(f"Sequence packing:         {'enabled' if pack else 'disabled'}")
    print(f"Random seed:              {seed}")
    print(f"Output directory:         {output_dir}")
    print("\nResolved mixture (bucket: probability):")
    for b, p in sorted(base_probs.items(), key=lambda x: -x[1]):
        print(f"  {b:<18} {p*100:6.2f}%")
    if long_probs:
        print(f"\nLong pool (>= {long_threshold_tokens} tokens, char proxy {chars_per_token}):")
        for b, p in sorted(long_probs.items(), key=lambda x: -x[1]):
            print(f"  {b:<18} {p*100:6.2f}%  (long)")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would build dataset with the above mixture.")
        return {"dry_run": True, "mixture": base_probs, "long_mixture": long_probs}

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = setup_tokenizer(tokenizer_name)

    # Open one streaming dataset per selected bucket and interleave by mixture. Base streams
    # first, then any length-filtered long streams (W4), so probabilities align 1:1.
    streams, probs, stream_names = [], [], []
    long_names = set()
    for i, b in enumerate(base_probs):
        ds, n_shards = load_bucket_stream(
            prepared_root, b, shuffle_seed=(seed + i) if shuffle_streams else None
        )
        streams.append(_tagged(ds, b))
        probs.append(base_probs[b])
        stream_names.append(b)
        print(f"  {b:<18} {n_shards} shards, p={base_probs[b]:.4f}")
    for i, b in enumerate(long_probs):
        ds, n_shards = load_bucket_stream(
            prepared_root,
            b,
            min_tokens=long_threshold_tokens,
            chars_per_token=chars_per_token,
            shuffle_seed=(seed + 1000 + i) if shuffle_streams else None,
        )
        name = f"{b}__long"
        streams.append(_tagged(ds, name))
        probs.append(long_probs[b])
        stream_names.append(name)
        long_names.add(name)
        print(
            f"  {b:<18} {n_shards} shards, p={long_probs[b]:.4f}  (long >= {long_threshold_tokens})"
        )

    exhaustion_events: list[dict] = []
    sched = None
    if weight_by == "tokens":
        sched = TokenShareInterleaver(
            streams, probs, stream_names, chars_per_token, events=exhaustion_events
        )
        full_dataset = sched
    else:
        full_dataset = interleave_drop_exhausted(
            streams, probs, seed, names=stream_names, events=exhaustion_events
        )

    # Take documents until we reach the token budget, tracking per-doc counts.
    # Tokenization here is only used to count tokens (and to keep ids when
    # packing), so encode in BATCHES: a HF fast tokenizer runs the batch across
    # all cores with the GIL released, vs. ~one core for per-document encode.
    # This is ~50-100x faster than calling tokenizer.encode() in a Python loop.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    # Token-share draws feed back on what was accepted, so keep the lag short. The pending
    # estimate already books every drawn document, so a bigger batch only delays the
    # correction from estimate to real count; --encode_batch raises it for big pools,
    # where 16-document batches leave the tokenizer's thread pool idle.
    ENCODE_BATCH = encode_batch or (16 if sched is not None else 1000)
    full_samples: list[dict] = []
    token_counts: list[int] = []
    full_token_ids: list[list[int]] = []  # populated only when pack=True
    curr_token_count = 0
    # Without a holdout the val split is carved off the tail, so the stream must
    # run to train+val; with one, val fills from its own bucket in parallel and
    # the stream stops at the train budget.
    stop_tokens = train_tokens if holdout_frac > 0.0 else total_tokens
    pbar = tqdm(full_dataset, desc="Loading samples")

    # Hash-holdout mode keeps train and val in separate lists as the stream is
    # consumed; the positional tail split below is then skipped entirely.
    hold_samples: list[dict] = []
    hold_token_ids: list[list[int]] = []
    hold_token_count = 0
    # Provenance for stats.json: realized tokens per stream and by document length.
    train_by_stream: dict = {}
    val_by_stream: dict = {}
    train_by_doclen: dict = {}
    val_by_doclen: dict = {}
    exact_dropped = {"docs": 0, "tokens": 0}
    holdout_dropped = {"docs": 0, "tokens": 0}

    def _consume(items):
        """Batch-encode `items` ((stream, text) pairs), append results, return True once
        the budget is hit."""
        nonlocal curr_token_count, hold_token_count
        texts = [t for _, t in items]
        id_lists = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for (name, text), ids in zip(items, id_lists):
            n = len(ids)
            if long_exact_filter and name in long_names and n < long_threshold_tokens:
                # Passed the lenient char prefilter, fails on real tokens.
                exact_dropped["docs"] += 1
                exact_dropped["tokens"] += n
                if sched is not None:
                    sched.settle(name, len(text), n, 0)
                continue
            if sched is not None:
                is_val = holdout_frac > 0.0 and is_holdout(text, holdout_frac)
                sched.settle(name, len(text), n, 0 if is_val else n)
            if holdout_frac > 0.0 and is_holdout(text, holdout_frac):
                # Val documents are taken from the SAME stream, so the mixture is
                # the mixture -- no separate draw that could drift from it.
                if hold_token_count < val_tokens:
                    hold_samples.append({"text": text})
                    if pack:
                        hold_token_ids.append(ids)
                    hold_token_count += n
                    val_by_stream[name] = val_by_stream.get(name, 0) + n
                    val_by_doclen[doclen_bin(n)] = val_by_doclen.get(doclen_bin(n), 0) + n
                else:
                    holdout_dropped["docs"] += 1
                    holdout_dropped["tokens"] += n
                # Past the val budget a holdout doc is DROPPED, never demoted into
                # train: membership has to stay a function of the text alone, or
                # the set stops being disjoint from other builds.
                continue
            full_samples.append({"text": text})
            token_counts.append(n)
            if pack:
                full_token_ids.append(ids)
            curr_token_count += n
            train_by_stream[name] = train_by_stream.get(name, 0) + n
            train_by_doclen[doclen_bin(n)] = train_by_doclen.get(doclen_bin(n), 0) + n
            if curr_token_count >= stop_tokens:
                return True
        pbar.set_postfix(
            tokens=f"{curr_token_count/1e6:.1f}M",
            val=f"{hold_token_count/1e6:.1f}M" if holdout_frac > 0 else "tail",
        )
        return False

    batch: list[tuple] = []
    done = False
    for name, sample in pbar:
        batch.append((name, sample["text"]))
        if len(batch) >= ENCODE_BATCH:
            done = _consume(batch)
            batch = []
            if done:
                break
    if not done and batch:
        _consume(batch)

    print(f"Loaded {len(full_samples):,} samples ({curr_token_count/1e6:.1f}M tokens)")

    # Split by cumulative token count (unshuffled), then shuffle within splits.
    train_split_idx = len(full_samples)
    cumulative = 0
    for i, n in enumerate(token_counts):
        cumulative += n
        if cumulative >= train_tokens:
            train_split_idx = i + 1
            break

    if pack:
        return _pack_and_save(
            full_token_ids=full_token_ids,
            train_split_idx=train_split_idx,
            eos_token_id=tokenizer.eos_token_id,
            seq_length=seq_length,
            seed=seed,
            output_dir=output_dir,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            mixture=effective_mixture,
        )

    if holdout_frac > 0.0:
        train_dataset = Dataset.from_list(full_samples).shuffle(seed=seed)
        val_dataset = Dataset.from_list(hold_samples).shuffle(seed=seed)
        print(
            f"\nHash holdout {holdout_frac:.4%}: "
            f"train {curr_token_count/1e9:.2f}B tok / {len(train_dataset):,} docs, "
            f"val {hold_token_count/1e6:.1f}M tok / {len(val_dataset):,} docs"
        )
    else:
        full_dataset = Dataset.from_list(full_samples)
        train_dataset = full_dataset.select(range(train_split_idx)).shuffle(seed=seed)
        val_dataset = full_dataset.select(
            range(train_split_idx, len(full_dataset))
        ).shuffle(seed=seed)
    print(f"\nSplitting dataset (train: {len(train_dataset):,}, val: {len(val_dataset):,})...")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_arrow_path = dataset_dir / "train.arrow"
    val_arrow_path = dataset_dir / "validation.arrow"

    # Single Arrow IPC file per split (zero-copy reads; LLaMA Factory loadable).
    with pa.OSFile(str(train_arrow_path), "wb") as sink:
        with ipc.new_file(sink, train_dataset.data.table.schema) as writer:
            writer.write_table(train_dataset.data.table)
    print(f"Saved train to {train_arrow_path} ({train_arrow_path.stat().st_size/1024/1024:.1f} MB)")

    with pa.OSFile(str(val_arrow_path), "wb") as sink:
        with ipc.new_file(sink, val_dataset.data.table.schema) as writer:
            writer.write_table(val_dataset.data.table)
    print(
        f"Saved validation to {val_arrow_path} ({val_arrow_path.stat().st_size/1024/1024:.1f} MB)"
    )

    train_name = f"{mix_name}_train_{train_tokens/1e6:g}M"
    val_name = f"{mix_name}_val_{val_tokens/1e6:g}M"

    stats = {
        "mix_name": mix_name,
        "prepared_root": str(prepared_root),
        "train_dataset": train_name,
        "val_dataset": val_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "actual_tokens": curr_token_count,
        "target_train_tokens": train_tokens,
        "target_val_tokens": val_tokens,
        "seq_length": seq_length,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "stopping_strategy": "drop_exhausted",
        # Buckets that ran dry before the budget was met. A NON-EMPTY list means
        # the realized mixture drifted from `mixture` after that point -- those
        # buckets stopped contributing while the rest carried on. An empty list
        # means every bucket supplied its full share for the whole build.
        "exhausted_buckets": exhaustion_events,
        "val_source": "hash_holdout" if holdout_frac > 0.0 else "stream_tail",
        "holdout_hash": "blake2b-8/%d" % HOLDOUT_MODULUS if holdout_frac > 0.0 else None,
        "val_tokens_actual": hold_token_count if holdout_frac > 0.0 else None,
        "mixture": effective_mixture,
        "long_pool": (
            {
                "threshold_tokens": long_threshold_tokens,
                "weight": 1.0 if long_only else long_weight,
                "long_only": long_only,
                "exact_token_filter": long_exact_filter,
                "exact_filter_dropped": exact_dropped,
                "buckets": list(long_probs),
                "chars_per_token": chars_per_token,
            }
            if long_probs
            else None
        ),
        "shuffle_streams": shuffle_streams,
        "weight_by": weight_by,
        "encode_batch": ENCODE_BATCH,
        "requested_shares": (
            {n: round(p, 4) for n, p in zip(stream_names, probs)} if weight_by == "tokens" else None
        ),
        "sft_drops_applied": len(sft_drops()) if any(
            b.startswith(SFT_PREFIX) for b in list(base_probs) + list(long_probs)) else None,
        "holdout_over_budget_dropped": holdout_dropped if holdout_frac > 0.0 else None,
        # Realized, not requested: per-doc draws against token-share weights drift
        # toward streams with longer documents.
        "train_tokens_by_stream": dict(sorted(train_by_stream.items())),
        "val_tokens_by_stream": dict(sorted(val_by_stream.items())),
        "train_tokens_by_doclen": {k: train_by_doclen[k] for k in sorted(
            train_by_doclen, key=lambda b: int(b[1:].split(",")[0]))},
        "val_tokens_by_doclen": {k: val_by_doclen[k] for k in sorted(
            val_by_doclen, key=lambda b: int(b[1:].split(",")[0]))},
    }
    stats["harmony_conformance"] = harmony_conformance(train_dataset, seed=seed)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # dataset_info.json with relative paths (portable across machines).
    dataset_info = {
        train_name: {
            "file_name": "dataset/train.arrow",
            "format": "arrow",
            "columns": {"prompt": "text"},
        },
        val_name: {
            "file_name": "dataset/validation.arrow",
            "format": "arrow",
            "columns": {"prompt": "text"},
        },
    }
    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset_info.json to {dataset_info_path}")

    print("\n" + "=" * 60)
    print("Dataset prepared successfully!")
    print("=" * 60)
    print("\nUse with the scaling sweep / LLaMA Factory via:")
    print(f"   --dataset_dir {output_dir}")
    print("\nDataset names:")
    print(f"   - {train_name} (train)")
    print(f"   - {val_name} (val)")

    return stats


def harmony_conformance(dataset, *, seed: int, n: int = 20000, max_rate: float = 0.005) -> dict:
    """Semantic Harmony check on a seeded sample of the built pool, recorded in stats.json.

    2026-11-02: ~15% of 60B_hv1 was chat/reasoning/tool text that was not Harmony or had
    reasoning and calls in the wrong place, and nothing at build or ship time looked.
    greatlakes/verify_pool.py refuses a pool whose record is missing or failing.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.data_prep.check_harmony_conformance import HARD, SOFT, check

    idx = list(range(len(dataset)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    counts = dict.fromkeys(HARD + SOFT, 0)
    for row in dataset.select(idx):
        for rule in check(row["text"]):
            counts[rule] += 1
    rates = {k: round(v / max(1, len(idx)), 5) for k, v in counts.items()}
    fail = any(rates[k] > max_rate for k in HARD)
    print(f"Harmony conformance ({len(idx):,} docs): {'FAIL' if fail else 'pass'} "
          + " ".join(f"{k}={v}" for k, v in rates.items() if v))
    return {"n": len(idx), "max_rate": max_rate, "fail": fail, "rates": rates}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a budget-sized pretraining mix from the prepared bucket corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the prepared dataset (e.g. .../scaling_law_datasets/1B).",
    )
    parser.add_argument(
        "--prepared_root",
        type=str,
        default=DEFAULT_PREPARED_ROOT,
        help="Root of the prepared bucket corpus.",
    )
    parser.add_argument(
        "--mix_name",
        type=str,
        default="pretrain_natural_v1",
        help="Base name for the emitted train/val datasets in dataset_info.json.",
    )
    parser.add_argument(
        "--train_tokens",
        type=int,
        default=1_000_000_000,
        help="Target number of tokens for the training set.",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=None,
        help="Target validation tokens (overrides --val_percent if set).",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=3.0,
        help="Validation size as a percentage of training tokens.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length (used for --pack and sample estimation).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--holdout_frac",
        type=float,
        default=0.0,
        help="Fraction of DOCUMENTS held out for validation by content hash "
        "(e.g. 0.01). Position-independent, so one val set stays disjoint from "
        "every train mix cut at any budget. 0 keeps the legacy tail split, whose "
        "val lands inside the train split of any larger build.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer for token counting; defaults to the project "
        "SuperBPE-56000 (umich_slm.tokenizer_defaults).",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="Comma-separated subset of buckets to include (default: all present).",
    )
    parser.add_argument(
        "--mixture_ratio",
        type=str,
        default=None,
        help="JSON object {bucket: weight} overriding natural proportions (normalized).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to a JSON file with bucket weights (overrides --mixture_ratio).",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Pack tokenized docs into fixed-length input_ids Arrow files "
        "(train with scripts/pretrain_packed.py). NOTE: do NOT use --pack "
        "for long-context work — it bakes a fixed seq_length (LONG_CONTEXT_PLAN.md §1).",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Resolve and print the mixture without building."
    )
    # --- W4 long-context length weighting (LONG_CONTEXT_PLAN.md §1.3) ---
    parser.add_argument(
        "--long_stage",
        type=str,
        default=None,
        choices=["8k", "16k", "32k"],
        help="Context stage: sets --long_threshold_tokens and --long_weight "
        "from umich_slm.data.long_context_mix.STAGE_SPECS.",
    )
    parser.add_argument(
        "--long_strategy",
        type=str,
        default=None,
        choices=["A", "B"],
        help="Long-pool bucket set: A=reasoning/agentic/math (no arxiv, YaRN "
        "probe), B=full (adds arxiv+python, native long stage).",
    )
    parser.add_argument(
        "--long_buckets",
        type=str,
        default=None,
        help="Explicit comma-separated long-pool buckets (overrides --long_strategy).",
    )
    parser.add_argument(
        "--long_threshold_tokens",
        type=int,
        default=None,
        help="Min tokens for the long pool (overrides --long_stage).",
    )
    parser.add_argument(
        "--long_weight",
        type=float,
        default=None,
        help="Total probability mass reallocated to the long pool [0,1) "
        "(overrides --long_stage).",
    )
    parser.add_argument(
        "--chars_per_token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="Char/token proxy for the length filter (manifest heuristic ~3.7).",
    )
    parser.add_argument(
        "--long_only",
        action="store_true",
        help="Build ONLY the length-filtered long streams (no base mixture), e.g. a "
        "context-extension pool. --long_weight is then not needed.",
    )
    parser.add_argument(
        "--long_exact_filter",
        action="store_true",
        help="Re-check long-stream docs on their REAL token count (the char proxy then "
        "only prefilters; pair with a lenient --chars_per_token such as 2.4).",
    )
    parser.add_argument(
        "--long_source_weights",
        type=str,
        default=None,
        help='JSON object or path: {"bucket/source": weight}. One long stream per named '
        "source instead of one per bucket.",
    )
    parser.add_argument(
        "--weight_by",
        choices=("docs", "tokens"),
        default="docs",
        help="docs: weights are per-document pick probabilities (historical). tokens: "
        "weights are shares of accepted train tokens (TokenShareInterleaver).",
    )
    parser.add_argument(
        "--sft_root",
        nargs="+",
        default=[DEFAULT_SFT_ROOT],
        help="Root(s) for long-source keys 'sft:<stage>/<source>' (prepared/ + manifests/), "
        "searched in order; each root's decon/dedup drops apply.",
    )
    parser.add_argument(
        "--shuffle_streams",
        action="store_true",
        help="Shuffle each stream's shard order (seeded). Use for pools smaller than their "
        "sources; leave off for nested budget ladders.",
    )
    parser.add_argument(
        "--encode_batch",
        type=int,
        default=None,
        help="Documents per tokenizer call (default 16 for --weight_by tokens, else 1000).",
    )

    args = parser.parse_args()

    mixture_override = None
    if args.weights:
        raw = json.loads(Path(args.weights).read_text())
        mixture_override = raw.get("optimal_weights", raw)
    elif args.mixture_ratio:
        mixture_override = json.loads(args.mixture_ratio)

    buckets = [b.strip() for b in args.buckets.split(",")] if args.buckets else None

    long_source_weights = None
    if args.long_source_weights:
        raw = args.long_source_weights
        long_source_weights = json.loads(Path(raw).read_text() if Path(raw).is_file() else raw)

    # Resolve the long pool: an explicit flag wins; otherwise derive from --long_stage.
    long_buckets = long_threshold_tokens = long_weight = None
    if (
        args.long_stage
        or args.long_strategy
        or args.long_buckets
        or args.long_threshold_tokens
        or args.long_weight
        or args.long_only
    ):
        if args.long_stage:
            stage = resolve_stage(args.long_strategy or "A", args.long_stage)
            long_buckets = stage["buckets"]
            long_threshold_tokens = stage["threshold_tokens"]
            long_weight = stage["long_weight"]
        elif args.long_strategy:
            from umich_slm.data.long_context_mix import STRATEGY_BUCKETS

            long_buckets = list(STRATEGY_BUCKETS[args.long_strategy])
        if args.long_buckets:
            long_buckets = [b.strip() for b in args.long_buckets.split(",")]
        if args.long_threshold_tokens is not None:
            long_threshold_tokens = args.long_threshold_tokens
        if args.long_weight is not None:
            long_weight = args.long_weight
        if args.long_only and long_weight is None:
            long_weight = 0.0  # unused: long streams carry all the mass
        if args.long_source_weights and not long_buckets:
            long_buckets = ["<per-source>"]
        if not long_buckets or not long_threshold_tokens or not (long_weight or args.long_only):
            parser.error(
                "long pool underspecified: give --long_stage (+optional --long_strategy) or "
                "all of --long_buckets/--long_threshold_tokens/--long_weight."
            )

    if args.val_tokens is None:
        val_tokens = int(args.train_tokens * args.val_percent / 100.0)
    else:
        val_tokens = args.val_tokens
        if args.val_percent != 3.0:
            print(f"Warning: --val_tokens set, ignoring --val_percent={args.val_percent}")

    stats = prepare_dataset(
        train_tokens=args.train_tokens,
        val_tokens=val_tokens,
        output_dir=Path(args.output_dir),
        prepared_root=Path(args.prepared_root),
        mix_name=args.mix_name,
        seq_length=args.seq_length,
        buckets=buckets,
        mixture_override=mixture_override,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
        dry_run=args.dry_run,
        pack=args.pack,
        long_buckets=long_buckets,
        long_threshold_tokens=long_threshold_tokens or 0,
        long_weight=long_weight or 0.0,
        chars_per_token=args.chars_per_token,
        holdout_frac=args.holdout_frac,
        long_only=args.long_only,
        long_exact_filter=args.long_exact_filter,
        long_source_weights=long_source_weights,
        shuffle_streams=args.shuffle_streams,
        weight_by=args.weight_by,
        sft_root=args.sft_root,
        encode_batch=args.encode_batch,
    )

    if not args.dry_run:
        print("\nDataset statistics:")
        for key, value in stats.items():
            if key == "mixture":
                continue
            if isinstance(value, int) and value > 1000:
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
