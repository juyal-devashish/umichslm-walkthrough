"""Score frozen (context, continuation) records with a causal LM.

This is the one scorer behind all three record-based families -- the lm-eval
benchmarks, the gold-solution reasoning continuations, and the agentic MC sets.
It needs nothing but torch + transformers, which is the point: the eval driver
runs in `umich-slm-fa2` on a compute node, and that env has no lm_eval and no
`datasets`.

Metrics per document, and why each is here:

    acc         argmax of the summed continuation log-likelihood.
    acc_norm    argmax of loglik / characters   (lm-eval's `acc_norm`)
    acc_byte    argmax of loglik / UTF-8 bytes  (tokenizer-invariant)
    bpb_gold    -loglik(gold) / (ln2 * bytes(gold))   <- the PRIMARY metric
    soft        loglik(gold)/bytes(gold) - logsumexp_i(loglik_i/bytes_i)

`bpb_gold` is primary because accuracy is floor-bound at 25-70M parameters and
because bits-per-byte is what carries signal there: AI2's Signal-and-Noise
(2508.13144) measures task SNR rising 1.2 -> 7.0 on GSM8K and 2.0 -> 41.8 on
MBPP purely by switching the metric from exact-match to BPB. Byte normalisation
is also what makes a comparison against Monad (8k vocab) or Qwen (151k vocab)
mean anything -- per-token perplexity would just reward the bigger vocabulary.

`soft` is the smooth companion metric consumed by
`scripts/plot_benchmark_scaling.py`, which fits soft(C) = a*C^b + c per
architecture and then maps soft -> hard through a sigmoid anchored on
`evals/reference_models.json`.

Tokenisation follows lm-eval's `_encode_pair` exactly: encode the concatenation
and slice at `len(encode(context))`, rather than encoding the two halves
separately. With a SuperBPE tokenizer the distinction is not cosmetic -- its
superword merges cross the context/continuation boundary, so encoding
separately would score a different token sequence than the model would ever see.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def load_records(path: str | Path) -> list[dict]:
    """Read one frozen JSONL file, expanding the compact shared-context form."""
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "reqs" not in rec:
                rec["reqs"] = [[rec["ctx"], c] for c in rec["conts"]]
            out.append(rec)
    return out


def _encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """(context_ids, continuation_ids), split the way lm-eval splits them.

    A trailing space on the context is moved onto the continuation first; lm-eval
    does this because most tokenizers attach a leading space to the following
    word, so leaving it on the context would put the boundary mid-token.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tokenizer(context, add_special_tokens=False)["input_ids"]
    n_ctx = len(ctx)
    # Guard the pathological case where the merge eats the boundary entirely.
    if n_ctx >= len(whole):
        n_ctx = max(0, len(whole) - 1)
    return whole[:n_ctx], whole[n_ctx:]


def _output_head(model):
    """The model's output projection, if applying it to the final hidden state
    reproduces `logits` exactly.

    Scoring needs log-probabilities at continuation positions only -- typically
    10-20% of a sequence -- but asking the model for `logits` materialises
    (B, T, vocab) for ALL positions. At vocab 56000 that is what forces a small
    batch, and a small batch is what leaves a 54M model launch-bound.

    Projecting only the positions we read removes the constraint, but only if the
    final hidden state is post-final-norm (true for Llama-style models, including
    ours). Rather than assume it, verify once on an 8-token input and fall back to
    the logits path if it does not hold.
    """
    head = None
    try:
        head = model.get_output_embeddings()
    except Exception:
        return None
    if head is None:
        return None
    try:
        probe = torch.ones(1, 8, dtype=torch.long, device=next(model.parameters()).device)
        out = model(
            input_ids=probe, attention_mask=torch.ones_like(probe), output_hidden_states=True
        )
        if getattr(out, "logits", None) is None or not getattr(out, "hidden_states", None):
            return None
        if not torch.allclose(
            head(out.hidden_states[-1]).float(), out.logits.float(), atol=1e-2, rtol=1e-2
        ):
            return None
    except Exception:
        return None
    return head


def _plan_batches(encoded, order, max_tokens: int, max_seqs: int) -> list[list[int]]:
    """Group length-sorted sequences under a PADDED-token budget.

    Batching by sequence count is wrong here because the tasks differ by an order
    of magnitude in length: 8 sequences of ARC is ~1000 tokens and leaves the GPU
    idle, while 8 sequences of a long MMLU context is 30x that. A token budget
    gives short tasks big batches and long tasks small ones, automatically.
    """
    batches, cur = [], []
    for i in order:
        width = len(encoded[i][0])
        # Padded cost, not summed cost: every sequence pads up to the longest.
        if cur and (max(width, cur[1]) * (len(cur[0]) + 1) > max_tokens or len(cur[0]) >= max_seqs):
            batches.append(cur[0])
            cur = None
        if cur is None or not cur:
            cur = ([i], width)
        else:
            cur = (cur[0] + [i], max(cur[1], width))
    if cur and cur[0]:
        batches.append(cur[0])
    return batches


@torch.no_grad()
def _batch_loglikelihood(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: str,
    max_length: int,
    batch_size: int,
    max_batch_tokens: int = 16384,
) -> list[float]:
    """Summed log P(continuation | context) for each pair, right-truncated."""
    encoded = []
    for ctx, cont in pairs:
        ctx_ids, cont_ids = _encode_pair(tokenizer, ctx, cont)
        if not cont_ids:  # empty continuation scores as -inf rather than 0
            encoded.append(None)
            continue
        ids = (ctx_ids + cont_ids)[-max_length:]
        n_cont = min(len(cont_ids), len(ids) - 1)
        encoded.append((ids, n_cont))

    results: list[float] = [float("-inf")] * len(encoded)
    order = [i for i, e in enumerate(encoded) if e is not None]
    # Length-sort so a batch is not padded out to one long outlier.
    order.sort(key=lambda i: len(encoded[i][0]))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    head = _output_head(model)

    def _score_chunk(chunk: list[int], budget: int) -> None:
        """Score one planned batch, halving it and retrying if it will not fit.

        The token budget that is right for a 54M model with a 56k vocab is wrong
        for a 321M model with 65k or a 600M model with 152k, and this suite is
        meant to run over reference models and future architectures without a
        per-model constant. Backing off costs a little time on the batches that
        trip it and nothing on the ones that do not; the alternative is a crash
        that loses the whole cell. Halving changes bf16 accumulation by ~8e-5
        relative, orders below the noise these metrics carry.
        """
        try:
            _forward_chunk(chunk)
            return
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
        # Retry OUTSIDE the except block, on purpose. Inside it, the live
        # exception holds its traceback, which holds the frame of the call that
        # failed, which holds every tensor that frame allocated -- so
        # empty_cache() frees nothing and the retry OOMs on a 2 MB allocation
        # with 47 GB still held. Leaving the block drops the traceback first.
        torch.cuda.empty_cache()
        mid = len(chunk) // 2
        _score_chunk(chunk[:mid], budget)
        _score_chunk(chunk[mid:], budget)

    def _forward_chunk(chunk: list[int]) -> None:
        seqs = [encoded[i][0] for i in chunk]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, : len(s)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        # logits[t] predicts token t+1, so option k's log-probs live at positions
        # [seq_len-n_cont-1, seq_len-1).
        spans = [(len(seqs[r]) - encoded[i][1] - 1, len(seqs[r])) for r, i in enumerate(chunk)]

        if head is not None:
            hidden = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True
            ).hidden_states[-1]
            rows, targets, owners = [], [], []
            for r, (lo, hi) in enumerate(spans):
                rows.append(hidden[r, lo : hi - 1])
                targets.append(input_ids[r, lo + 1 : hi])
                owners.append(chunk[r])
            lengths = [t.shape[0] for t in targets]
            flat_h = torch.cat(rows, dim=0)
            flat_t = torch.cat(targets, dim=0)
            # Chunk the projection so peak memory stays bounded by the slice, not
            # by the batch: this is the whole point of the hidden-state path.
            off = 0
            per_seq = torch.zeros(flat_t.shape[0], dtype=torch.float32, device=flat_h.device)
            # Chunk by LOGIT COUNT, not by row count. A fixed 8192 rows is 8192 x
            # vocab logits: at 56000 that is 1.8 GB in fp32, and the bf16 head
            # output plus log_softmax roughly triple it -- which is what OOM'd
            # both a reference model and our own. Targeting ~2^26 logits keeps a
            # chunk near 256 MB fp32 whatever the vocabulary is.
            vocab = int(getattr(head, "out_features", 0)) or int(head.weight.shape[0])
            step = max(64, (1 << 26) // max(1, vocab))
            for a in range(0, flat_h.shape[0], step):
                lg = head(flat_h[a : a + step]).float()
                lp = F.log_softmax(lg, dim=-1)
                per_seq[a : a + step] = lp.gather(-1, flat_t[a : a + step].unsqueeze(-1)).squeeze(
                    -1
                )
            for i, n in zip(owners, lengths):
                results[i] = float(per_seq[off : off + n].sum())
                off += n
            del hidden, flat_h, flat_t, per_seq, rows, targets
        else:
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError(
                    "model returned no logits and exposes no usable output head; "
                    "pass output_logits=True or use downstream_scoring.bpb_of_docs"
                )
            for r, i in enumerate(chunk):
                lo, hi = spans[r]
                # Slice BEFORE float()/log_softmax -- over the full window this
                # would materialise (B, T, 56000) twice in fp32.
                span = logits[r, lo : hi - 1].float()
                tgt = input_ids[r, lo + 1 : hi]
                lp = F.log_softmax(span, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                results[i] = float(lp.sum())
            del out, logits

    for chunk in _plan_batches(encoded, order, max_batch_tokens, batch_size):
        _score_chunk(chunk, max_batch_tokens)

    return results


def score_records(
    model,
    tokenizer,
    records: Sequence[dict],
    *,
    device: str = "cuda",
    max_length: int = 2048,
    batch_size: int = 64,
    max_batch_tokens: int = 16384,
    task_name: str = "",
) -> tuple[dict, list[dict]]:
    """(aggregate metrics, per-document scores) for one frozen record file."""
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for rec in records:
        spans.append((len(pairs), len(rec["reqs"])))
        pairs.extend((c, k) for c, k in rec["reqs"])

    lls = _batch_loglikelihood(
        model, tokenizer, pairs, device, max_length, batch_size, max_batch_tokens
    )

    n = 0
    hits = {"acc": 0, "acc_norm": 0, "acc_byte": 0}
    sum_nll_gold = 0.0
    sum_bytes_gold = 0
    softs: list[float] = []
    per_doc: list[dict] = []

    for rec, (off, k) in zip(records, spans):
        doc_lls = lls[off : off + k]
        if any(math.isinf(x) for x in doc_lls):
            continue
        gold = rec["gold"]
        g = gold[0]
        nchars = rec.get("nchars") or [max(1, len(c)) for _, c in rec["reqs"]]
        nbytes = rec.get("nbytes") or [max(1, len(c.encode("utf-8"))) for _, c in rec["reqs"]]
        nchars = [max(1, c) for c in nchars]
        nbytes = [max(1, b) for b in nbytes]

        n += 1
        if k > 1:
            hits["acc"] += int(max(range(k), key=lambda i: doc_lls[i]) in gold)
            hits["acc_norm"] += int(max(range(k), key=lambda i: doc_lls[i] / nchars[i]) in gold)
            hits["acc_byte"] += int(max(range(k), key=lambda i: doc_lls[i] / nbytes[i]) in gold)

        sum_nll_gold += -doc_lls[g]
        sum_bytes_gold += nbytes[g]

        if k > 1:
            # Meaningless with a single option: logsumexp of one value is that
            # value, so soft would be identically 0 and would clutter the SNR
            # table with a task that cannot vary.
            per_byte = [doc_lls[i] / nbytes[i] for i in range(k)]
            softs.append(per_byte[g] - torch.logsumexp(torch.tensor(per_byte), dim=0).item())
        per_doc.append({"doc_id": rec["doc_id"], "gold": g, "lls": doc_lls, "nbytes": nbytes})

    if n == 0:
        return {"task": task_name, "n": 0, "error": "no scoreable documents"}, []

    is_mc = any(len(r["reqs"]) > 1 for r in records)
    out = {
        "task": task_name,
        "n": n,
        "n_options": max(len(r["reqs"]) for r in records),
        # PRIMARY. Aggregated over total bytes, not averaged per doc, so a long
        # gold answer weighs proportionally -- same convention as compute_bpb.py.
        "bpb_gold": sum_nll_gold / (LN2 * sum_bytes_gold),
        "nll_gold_per_byte": sum_nll_gold / sum_bytes_gold,
        "total_bytes_gold": sum_bytes_gold,
    }
    if softs:
        out["soft"] = sum(softs) / len(softs)
    if is_mc:
        out.update({k: v / n for k, v in hits.items()})
        out["chance"] = sum(len(r["gold"]) / len(r["reqs"]) for r in records) / len(records)
    return out, per_doc


def bootstrap_std(per_doc: Sequence[dict], metric: str, n_boot: int = 200, seed: int = 0) -> float:
    """Item-resampling std, on the SAME scale as the metric it describes.

    This previously resampled the mean per-document NLL in nats and reported it
    as the noise on `bpb_gold`, which is bits per byte. The scales are unrelated:
    it returned 4.4 for gsm8k against a metric of 1.2, so the comparison to an
    arm spread was meaningless and would have made every task a tie.

    The other noise estimate in this project comes from spread across a cell's own
    checkpoints; that one shares a seed and a data order, so it understates the
    truth. This one is independent of training entirely.
    """
    import random

    if len(per_doc) < 2:
        return float("nan")
    rng = random.Random(seed)
    n = len(per_doc)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        if metric == "acc":
            vals.append(
                sum(
                    int(
                        max(range(len(per_doc[i]["lls"])), key=lambda j: per_doc[i]["lls"][j])
                        == per_doc[i]["gold"]
                    )
                    for i in idx
                )
                / n
            )
        else:
            # Aggregate exactly as the metric does: total nats over total bytes.
            nll = sum(-per_doc[i]["lls"][per_doc[i]["gold"]] for i in idx)
            byt = sum(per_doc[i].get("nbytes", [1])[per_doc[i]["gold"]] for i in idx)
            vals.append(nll / (LN2 * byt) if byt else float("nan"))
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
