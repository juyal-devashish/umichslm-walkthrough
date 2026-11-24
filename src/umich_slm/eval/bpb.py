"""Bits-per-byte over documents, without packing.

Lives in the package rather than in `scripts/` so the training loop can import it:
tracking held-out BPB per domain alongside val loss is far more informative than a
single scalar, but only if the trainer can call it directly on the model it already
has in memory.

The denominator is the UTF-8 length of the span the loss actually covers --
`decode(ids[1:])` after truncation -- not of the input string, so numerator and
denominator describe the same text. `eval_utils.score_texts` instead packs
documents into fixed-length chunks, drops the final partial chunk from the
numerator while keeping its bytes in the denominator, and inserts an EOS between
documents that costs nats and contributes no bytes. That ~0.25% bias is fine for a
smoke test and too large for a metric asked to resolve architecture gaps in the
third decimal.
"""

from __future__ import annotations

import math

import torch

LN2 = math.log(2.0)


def _batch_nll(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
) -> tuple[float, int]:
    """(summed NLL in nats, tokens scored) over a padded batch.

    The model returns a MEAN over unmasked label positions, so multiplying by the
    unmasked count recovers the exact batch total -- which is all a corpus-level
    bits-per-byte needs. That is what makes batching possible here at all: a
    per-sequence breakdown would require materialising (B, T, 56000) logits,
    ~7 GB in fp32 at B=8 and T=2048, to read a number the mean already contains.

    Padded positions carry label -100 and contribute to neither term.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = getattr(out, "lm_loss", None)
    if loss is None:
        # Only for architectures with no auxiliary term. Never silently fall back
        # to a FUSED loss on an arm that has one -- nothing here ranks on that.
        loss = out.loss
    n_scored = int((labels[:, 1:] != -100).sum())
    return float(loss) * n_scored, n_scored


def effective_max_length(model, requested: int) -> int:
    """`requested`, clamped to what the model can actually index.

    gpt2 has 1024 learned position embeddings; feeding it a 2048-token sequence
    is an out-of-bounds gather, which surfaces as an opaque
    "device-side assert triggered" long after the fact and poisons the CUDA
    context for the rest of the process. Models with rotary positions do not care,
    but the reference pool mixes both, so the limit is read rather than assumed.
    """
    cfg = getattr(model, "config", None)
    limit = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
    return min(requested, int(limit)) if limit else requested


def _labels_path_materialises_logits(model) -> int:
    """Vocab size if the model returns full logits when `labels` is passed, else 0.

    UMichSLMForCausalLM computes a chunked shifted loss and returns none, which is
    why domain BPB can use large batches at all. A standard HF model returns
    (B, T, vocab) regardless: at Qwen's 151936 that is ~20 GB for 32 sequences of
    2048, before anything else. Probing once is what lets a single default cover
    both without a per-model constant or a long backoff cascade.
    """
    try:
        device = next(model.parameters()).device
        probe = torch.ones(1, 8, dtype=torch.long, device=device)
        out = model(input_ids=probe, attention_mask=torch.ones_like(probe), labels=probe)
        logits = getattr(out, "logits", None)
        return int(logits.shape[-1]) if logits is not None else 0
    except Exception:
        return 0


def _pad_batch(chunk, pad_id: int):
    """(input_ids, attention_mask, labels) for a list of (ids, nbytes) pairs."""
    width = max(len(ids) for ids, _ in chunk)
    input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(chunk), width), dtype=torch.long)
    labels = torch.full((len(chunk), width), -100, dtype=torch.long)
    for r, (ids, _) in enumerate(chunk):
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[r, : len(ids)] = t
        attn[r, : len(ids)] = 1
        labels[r, : len(ids)] = t
    return input_ids, attn, labels


def _nll_with_backoff(model, chunk, pad_id: int, device) -> tuple[float, int]:
    """Score a batch, halving and retrying until it fits.

    The batch size that is right for a 54M model with a 56k vocab is wrong for a
    321M reference model with 65k or a 600M one with 152k, and this suite is meant
    to run over reference models and future architectures without a per-model
    constant. Recursion matters: one halving is often not enough, and stopping
    after one would turn an OOM into a crash anyway. A single sequence that will
    not fit is a real error and is re-raised.
    """
    try:
        input_ids, attn, labels = _pad_batch(chunk, pad_id)
        out = _batch_nll(model, input_ids.to(device), attn.to(device), labels.to(device))
        del input_ids, attn, labels
        return out
    except torch.cuda.OutOfMemoryError:
        if len(chunk) == 1:
            raise
    # Outside the except block on purpose: inside it the live exception holds a
    # traceback holding the failed frame, so every tensor that frame allocated is
    # still reachable and empty_cache() frees nothing.
    torch.cuda.empty_cache()
    mid = len(chunk) // 2
    a_nll, a_tok = _nll_with_backoff(model, chunk[:mid], pad_id, device)
    b_nll, b_tok = _nll_with_backoff(model, chunk[mid:], pad_id, device)
    return a_nll + b_nll, a_tok + b_tok


@torch.no_grad()
def bpb_of_docs(
    model,
    tokenizer,
    texts,
    *,
    device: str = "cuda",
    max_length: int = 2048,
    max_docs: int | None = None,
    batch_size: int = 8,
    max_batch_tokens: int | None = None,
) -> dict:
    """Bits-per-byte over a list of documents, scored without packing.

    The denominator is the UTF-8 length of the span the loss actually covers --
    `decode(ids[1:])` after truncation -- not of the input string, so numerator
    and denominator describe the same text.

    Documents are grouped into padded batches rather than packed into
    fixed-length chunks. Packing is what `eval_utils.score_texts` does, and it
    biases the ratio two ways at once: the final partial chunk is dropped from
    the numerator while its bytes stay in the denominator, and the EOS inserted
    between documents costs nats while contributing no bytes. ~0.25% -- fine for
    a smoke test against Qwen, too large for a metric asked to resolve
    architecture gaps in the third decimal.

    `per_batch_bpb` is the report's resampling unit. Per-document values would
    need the logits this deliberately avoids; with hundreds to thousands of
    documents per bucket, batches resample fine.
    """
    encoded = []
    for i, text in enumerate(texts):
        if max_docs is not None and i >= max_docs:
            break
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        if len(ids) < 2:
            continue
        nbytes = len(tokenizer.decode(ids[1:]).encode("utf-8"))
        if nbytes:
            encoded.append((ids, nbytes))

    if not encoded:
        return {"n_docs": 0, "error": "no scoreable documents"}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Length-sort so no batch is padded out to a single long outlier.
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))

    if max_batch_tokens is None:
        vocab = _labels_path_materialises_logits(model)
        if vocab:
            # ~1 GB of bf16 logits when the labels path materialises them.
            max_batch_tokens = (1 << 29) // vocab
        else:
            # No logits, so activations bind. `batch_size * max_length` is the
            # worst case and overshoots badly: at 32 x 2048 it measured 45.5 GB of
            # a 48 GiB card on swa_sinkdiet, against the 3.6 GB the benchmark saw
            # at its B=32 optimum -- because the benchmark's documents averaged
            # ~1100 tokens, not the 2048 ceiling. 32768 is the measured knee.
            max_batch_tokens = min(batch_size * max_length, 32768)

    sum_nll, total_bytes, total_tokens = 0.0, 0, 0
    per_batch = []
    batch_docs = []

    batches, cur, cur_w = [], [], 0
    for i in order:
        w = len(encoded[i][0])
        if cur and (max(w, cur_w) * (len(cur) + 1) > max_batch_tokens or len(cur) >= batch_size):
            batches.append(cur)
            cur, cur_w = [], 0
        cur.append(i)
        cur_w = max(cur_w, w)
    if cur:
        batches.append(cur)

    for batch in batches:
        chunk = [encoded[i] for i in batch]
        nll, n_tok = _nll_with_backoff(model, chunk, pad_id, device)
        nbytes = sum(b for _, b in chunk)
        sum_nll += nll
        total_bytes += nbytes
        total_tokens += n_tok
        per_batch.append(nll / (LN2 * nbytes))
        batch_docs.append([int(i) for i in batch])

    return {
        "n_docs": len(encoded),
        "n_batches": len(per_batch),
        "bpb": sum_nll / (LN2 * total_bytes),
        "nll_per_token": sum_nll / max(1, total_tokens),
        "bytes_per_token": total_bytes / max(1, total_tokens),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "per_batch_bpb": per_batch,
        # Indices (into the scored documents) behind each per_batch_bpb entry. The
        # batch plan depends on the model -- whether its labels path materialises
        # logits sets the token budget -- so two cells' batches are only pairable
        # where these lists match; a paired comparison must check, not assume.
        "per_batch_docs": batch_docs,
    }


@torch.no_grad()
def bpb_by_position(
    model,
    tokenizer,
    texts,
    *,
    device: str = "cuda",
    seq_len: int = 4096,
    bins=(256, 1024, 2048, 4096),
    max_docs: int = 32,
    items: list | None = None,
) -> dict:
    """Bits-per-byte binned by POSITION in the sequence.

    The point of the whole suite's other families is aggregate quality; this one
    asks where in the context a model's advantage lives. These architectures
    differ in exactly that: a 1024-token sliding window plus a 256-token register
    bank against dense attention. An aggregate scored at 2048, over documents
    whose median length is ~688 tokens, averages that difference away.

    Bin edges are chosen around the window: everything under 256 is inside both
    the window and the bank, 256-1024 is inside the window, and past 1024 a
    windowed arm can only reach earlier context through its bank.

    Scored one document at a time with logits, since a per-position breakdown is
    exactly what the chunked-loss path throws away.

    Past the training context, pass edges out to ``seq_len`` (e.g. 8192, 16384,
    32768). The model is causal, so a bin below the training length reads the
    same number whether the window stops there or runs on -- one run at the
    longest length gives both the in-distribution and the extrapolation bins.
    ``items`` receives one ``{doc, bins: {label: [nll_sum, n_bytes]}}`` per
    document, for paired comparisons across checkpoints.
    """
    edges = list(bins)
    sums = [0.0] * len(edges)
    counts = [0] * len(edges)
    byte_counts = [0] * len(edges)
    n_docs = 0
    for text in texts:
        if n_docs >= max_docs:
            break
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True,
        )
        ids = enc["input_ids"]
        if len(ids) < edges[0] * 2:
            continue
        # Bytes per predicted token, so each bin can report a real bits-per-byte
        # and not just a per-token NLL. Per-token NLL is not tokenizer-invariant,
        # and this family is read next to Monad (8k vocab) and Qwen (151k), where
        # a per-token comparison would be meaningless.  Target of logits row i is
        # token i+1, so the weight for target j is the byte length of token j+1.
        offs = enc["offset_mapping"]
        tok_bytes = [len(text[a:b].encode("utf-8")) for a, b in offs]
        t = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(input_ids=t, attention_mask=torch.ones_like(t))
        logits = getattr(out, "logits", None)
        if logits is None:
            return {"error": "model returned no logits; cannot bin by position"}
        # nll = logsumexp(logits) - logits[target], computed in position chunks.
        # A full log_softmax would materialise a second (4095, 56000) fp32 tensor
        # -- 917 MB -- to read one value per row.
        tgt = t[0, 1:]
        nll = torch.empty(tgt.shape[0], dtype=torch.float32, device=logits.device)
        step = max(64, (1 << 26) // int(logits.shape[-1]))
        for a in range(0, tgt.shape[0], step):
            # cap at len(tgt): logits has one more row than there are targets
            sl = logits[0, a : min(a + step, tgt.shape[0])].float()
            nll[a : a + step] = torch.logsumexp(sl, dim=-1) - sl.gather(
                -1, tgt[a : a + step].unsqueeze(-1)
            ).squeeze(-1)
            del sl
        n_docs += 1
        doc_bins = {}
        # Bin on the GPU: a Python loop over 4095 positions per document, times
        # 32 documents times 4 buckets, is ~500k round-trips to .item().
        lo_edge = 1
        for b, hi in enumerate(edges):
            a0, a1 = lo_edge - 1, min(hi, nll.shape[0] + 1) - 1
            seg = nll[a0:a1]
            if seg.numel():
                seg_sum = float(seg.sum())
                seg_bytes = sum(tok_bytes[a0 + 1 : a1 + 1])
                sums[b] += seg_sum
                counts[b] += seg.numel()
                # targets a0..a1-1 are tokens a0+1..a1
                byte_counts[b] += seg_bytes
                doc_bins[f"{([0] + edges[:-1])[b]}-{hi}"] = [seg_sum, seg_bytes]
            lo_edge = hi
            if lo_edge > nll.shape[0]:
                break
        if items is not None:
            items.append({"doc": n_docs - 1, "bins": doc_bins})
        del out, logits, nll

    lo = [0] + edges[:-1]
    return {
        "n_docs": n_docs,
        "bins": {
            f"{a}-{b}": {
                "bpb": (sums[i] / (LN2 * byte_counts[i])) if byte_counts[i] else None,
                "nll_per_token": (sums[i] / counts[i]) if counts[i] else None,
                "n_tokens": counts[i],
                "n_bytes": byte_counts[i],
            }
            for i, (a, b) in enumerate(zip(lo, edges))
        },
    }


#: Explicit document boundary for eval files whose documents contain blank lines.
#: ``domain_bpb`` predates this and joins on a bare blank line, which is safe there
#: because its documents are short and it is scored as a stream. ``longctx`` cannot
#: use that joiner: an arXiv paper or a Python file is full of blank lines, so
#: splitting on one shatters the long documents this family exists to measure.
DOC_BOUNDARY = "\n\n<|eval_doc_boundary|>\n\n"


def split_docs(blob: str, min_chars: int = 200, sep: str = "\n\n") -> list[str]:
    """Split a frozen bucket file back into documents on its joiner.

    Pass ``sep=DOC_BOUNDARY`` for files written with the explicit marker.
    """
    return [d for d in blob.split(sep) if len(d) >= min_chars]


def unigram_bpb(texts, tokenizer) -> float:
    """The line any model must beat: BPB of the eval text's own unigram model.

    Costs no GPU and is a far more informative floor than a random-init model on
    a BPB task, where random init just reports ln(vocab).
    """
    from collections import Counter

    counts: Counter = Counter()
    total_bytes = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        counts.update(ids[1:])
        total_bytes += len(tokenizer.decode(ids[1:]).encode("utf-8"))
    n = sum(counts.values())
    if not n or not total_bytes:
        return float("nan")
    sum_nll = -sum(c * math.log(c / n) for c in counts.values())
    return sum_nll / (LN2 * total_bytes)
