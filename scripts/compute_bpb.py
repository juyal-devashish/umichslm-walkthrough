"""Bits-per-byte (BPB) on a frozen eval set — Step 2 MVP.

Why BPB and not PPL:
    Perplexity depends on the tokenizer (a model with a larger vocab produces
    fewer tokens per byte, which shrinks PPL for free). BPB normalises by raw
    UTF-8 bytes instead of tokens, so

        BPB = sum_of_NLL_in_nats / (ln(2) * total_utf8_bytes)

    is comparable across tokenizers, architectures, and papers. This is the
    metric we want on the IsoFLOP plot and in the paper's headline table.

How the eval set stays comparable across runs:
    The raw text under `data/eval/bpb_v1/` is frozen. MANIFEST.json records
    each file's sha256; this script re-hashes on startup and refuses to run
    if anything has drifted. If you intentionally need different text,
    build a v2 set with `prepare_bpb_eval.py --output_dir data/eval/bpb_v2/`
    — never mutate v1 in place, or old BPB numbers become incomparable.

Typical usage:
    # One-time: build the v1 eval set (produces 3 .txt files + MANIFEST.json)
    python scripts/prepare_bpb_eval.py

    # Smoke test on a pretrained model (known good pipeline):
    python scripts/compute_bpb.py --model Qwen/Qwen2.5-0.5B

    # On a local checkpoint from pretrain_fineweb.py:
    python scripts/compute_bpb.py --model ./output/my_run/checkpoint-5000

Expected smoke-test range on Qwen2.5-0.5B: BPB ~0.7-1.3. Values outside
[0.5, 2.0] usually indicate a tokenizer mismatch, a broken checkpoint, or
a corrupted eval set (rerun prepare_bpb_eval.py in that case).
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
from pathlib import Path

import torch

from eval_utils import load_model_and_tokenizer, resolve_device, score_texts

logger = logging.getLogger(__name__)


def _verify_manifest(eval_dir: Path) -> dict:
    """Load MANIFEST.json and re-hash every file listed. Raises on mismatch."""
    manifest_path = eval_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No MANIFEST.json in {eval_dir}. Run `python scripts/prepare_bpb_eval.py` first."
        )
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        p = eval_dir / entry["path"]
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"sha256 mismatch for {p}: manifest says {entry['sha256'][:12]}..., "
                f"file is {actual[:12]}.... Eval set has drifted — BPB numbers "
                f"would not be comparable. Restore the file or bump to a new version."
            )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Compute BPB on the frozen v1 eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--eval_set_dir", type=str, default="data/eval/bpb_v1",
        help="Directory containing MANIFEST.json and the frozen .txt files.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--output_file", type=str, default="./bpb_results.jsonl")
    parser.add_argument(
        "--step", type=int, default=None,
        help="Optional training step, written into the output record.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)

    eval_dir = Path(args.eval_set_dir)
    manifest = _verify_manifest(eval_dir)
    logger.info("Eval set %s verified (%d files)", manifest["eval_set_version"], len(manifest["files"]))

    # Read each frozen file into memory. Small enough that this is fine.
    texts = []
    for entry in manifest["files"]:
        texts.append((eval_dir / entry["path"]).read_text(encoding="utf-8"))

    device = resolve_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        ckpt_path=args.ckpt_path,
        dtype=dtype,
        device=device,
    )

    stats = score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    # Key identity:
    #     BPB = total_nll_nats / (ln(2) * total_bytes_utf8)
    # "Same-tokenizer" PPL is reported too as a sanity check — comparing it
    # across runs that share a tokenizer should agree with BPB's ranking.
    nats = stats["sum_nll_nats"]
    bytes_ = stats["total_bytes_utf8"]
    tokens = stats["total_pred_tokens"]
    bpb = nats / (math.log(2) * bytes_)
    ppl_same_tokenizer = math.exp(nats / tokens)

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model": args.model,
        "ckpt_path": args.ckpt_path,
        "step": args.step,
        "eval_set_version": manifest["eval_set_version"],
        "eval_set_files": [e["path"] for e in manifest["files"]],
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_vocab": tokenizer.vocab_size,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "total_bytes_utf8": bytes_,
        "total_pred_tokens": tokens,
        "bytes_per_token": round(bytes_ / tokens, 3),
        "bpb": round(bpb, 4),
        "ppl_same_tokenizer": round(ppl_same_tokenizer, 3),
    }

    print(json.dumps(record, indent=2))
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Appended result to %s", args.output_file)


if __name__ == "__main__":
    main()
