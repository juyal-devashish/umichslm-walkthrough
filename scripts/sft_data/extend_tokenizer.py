"""Add the 6 SFT special-token pairs to a pretrained model + tokenizer.

The 101M Chinchilla-optimal base was pretrained on FineWeb-Edu without any of
our schema tokens. Before SFT we inject 12 new atomic tokens (6 open + 6 close)
into the tokenizer and resize the model's input/output embeddings accordingly.

Usage:
    python scripts/sft_data/extend_tokenizer.py \\
        --base_model /path/to/data \\
        --output_dir /path/to/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECIAL_TOKENS = [
    "<think>", "</think>",
    "<answer>", "</answer>",
    "<code>", "</code>",
    "<retrieve>", "</retrieve>",
    "<result>", "</result>",
    "<cite>", "</cite>",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend tokenizer with 6 SFT token pairs")
    parser.add_argument("--base_model", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--init_scheme",
        choices=("mean", "random"),
        default="mean",
        help="how to initialize new embedding rows",
    )
    args = parser.parse_args()

    print(f"[extend_tokenizer] loading from {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    )

    pre_vocab = len(tok)
    print(f"[extend_tokenizer] pre-vocab size: {pre_vocab}")

    added = tok.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS},
        replace_additional_special_tokens=False,
    )
    print(f"[extend_tokenizer] newly added: {added} (existing skipped)")

    for t in SPECIAL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{t!r} did not tokenize atomically: {ids}")

    model.resize_token_embeddings(len(tok))
    post_vocab = model.get_input_embeddings().num_embeddings
    print(f"[extend_tokenizer] post-vocab size: {post_vocab}")

    if args.init_scheme == "mean" and added > 0:
        with torch.no_grad():
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            mean_in = in_emb[:pre_vocab].mean(dim=0)
            mean_out = out_emb[:pre_vocab].mean(dim=0)
            for i in range(pre_vocab, post_vocab):
                in_emb[i] = mean_in
                out_emb[i] = mean_out
        print("[extend_tokenizer] new embeddings initialized to vocab mean")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print(f"[extend_tokenizer] saved to {args.output_dir}")

    for t in SPECIAL_TOKENS:
        print(f"  {t!r:14} -> id {tok.convert_tokens_to_ids(t)}")


if __name__ == "__main__":
    main()
