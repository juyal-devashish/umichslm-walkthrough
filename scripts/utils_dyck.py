"""
utils_dyck.py

Utility functions for generating, tokenizing, and serving 
strictly balanced shuffle-Dyck-k datasets for Language Model training.
"""

import argparse
import os
import random
import torch
import unicodedata
from torch.utils.data import Dataset as TorchDataset
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset


def generate_shuff_dyck(
    k: int, 
    max_length: int = 1024, 
    p_open: float = 0.5, 
    min_depth: int = 1, 
    max_depth: int = 8
) -> List[int]:
    """Generate a strictly balanced k-shuffle Dyck sequence of exactly max_length."""
    if max_length % 2 != 0:
        raise ValueError("max_length must be an even number to generate a perfectly balanced sequence.")
    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")

    sequence = []
    counts = [0] * k 

    while len(sequence) < max_length:
        depth = sum(counts)
        remaining_slots = max_length - len(sequence)

        # 1. CRITICAL PHASE
        if depth >= remaining_slots:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # 2. FORCED OPEN
        if depth == 0 or len(sequence) < min_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # 3. FORCED CLOSE
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        # 4. RANDOM PHASE
        if random.random() < p_open:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def get_safe_unicode_mapping(
    tokenizer: PreTrainedTokenizer, 
    required_symbols: int
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Iterates through the Unicode character set to find symbols that tokenize 
    safely into exactly one model Token ID, immune to whitespace merging.
    """
    id_mapping = {}
    char_mapping = {}
    count = 0
    
    for i in range(33, 65536): 
        char = chr(i)
        
        if unicodedata.category(char).startswith(('C', 'Z')):
            continue
            
        isolated_tokens = tokenizer.encode(char, add_special_tokens=False)
        spaced_tokens = tokenizer.encode(f" {char}", add_special_tokens=False)
        
        if len(isolated_tokens) == 1 and len(spaced_tokens) == 1:
            id_mapping[count] = isolated_tokens[0]
            char_mapping[count] = char
            count += 1
            
        if count == required_symbols:
            break
            
    if count < required_symbols:
        raise ValueError(f"Could only find {count} safe single-token characters.")

    return id_mapping, char_mapping


def tokenize_sequence(sequence: List[int], mapping: Dict[int, int]) -> List[int]:
    """Converts a raw integer sequence into model-ready token IDs."""
    return [mapping[symbol] for symbol in sequence]


class ShuffleDyckDataset(TorchDataset):
    """PyTorch Dataset for generating and tokenizing sequences on the fly."""
    def __init__(
        self, tokenizer: PreTrainedTokenizer, k: int = 64, num_samples: int = 10000,
        seq_length: int = 1024, p_open: float = 0.51, min_depth: int = 1, max_depth: int = 8
    ):
        self.tokenizer = tokenizer
        self.k = k
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.p_open = p_open
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.required_symbols = 2 * k
        self.id_mapping, _ = get_safe_unicode_mapping(tokenizer, self.required_symbols)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw_seq = generate_shuff_dyck(self.k, self.seq_length, self.p_open, self.min_depth, self.max_depth)
        token_ids = tokenize_sequence(raw_seq, self.id_mapping)
        return torch.tensor(token_ids, dtype=torch.long)


def generate_shuff_dyck_txt_file(
    file_dir: str, num_symbols: int, n: int, target_length: int, 
    p: float, min_depth: int, max_depth: int, char_mapping: Dict[int, str]
) -> str:
    """Generates a text file containing balanced sequences using safe Unicode chars."""
    os.makedirs(file_dir, exist_ok=True)
    filepath = os.path.join(file_dir, f"dyck_sequences_safe_unicode_{num_symbols}_{p}.txt")
    
    with open(filepath, "w", encoding="utf-8") as f:
        for _ in range(n):
            result = generate_shuff_dyck(num_symbols, target_length, p, min_depth, max_depth)
            dyck_str = " ".join([char_mapping[x] for x in result[:target_length]])
            f.write(f"{dyck_str}\n")
            
    return filepath


def save_as_arrow_dataset(
    output_dir: str, k: int, n: int, seq_length: int, p_open: float, 
    min_depth: int, max_depth: int, id_mapping: Dict[int, int]
) -> str:
    """Generates sequences and saves directly to disk as a Hugging Face Arrow dataset."""
    print(f"Pre-generating and tokenizing {n} sequences for Arrow dataset...")
    
    all_input_ids = []
    
    for i in range(n):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {n} sequences...")
            
        raw_seq = generate_shuff_dyck(k, seq_length, p_open, min_depth, max_depth)
        token_ids = tokenize_sequence(raw_seq, id_mapping)
        all_input_ids.append(token_ids)

    hf_dataset = Dataset.from_dict({"input_ids": all_input_ids})
    
    save_path = os.path.join(output_dir, f"dyck_{k}_arrow_len{seq_length}")
    os.makedirs(save_path, exist_ok=True)
    hf_dataset.save_to_disk(save_path)
    
    return save_path

def register_dataset(save_path):
    dataset_info = {
        "shuffle_dyck_k": {
            "hf_hub_url": save_path,
            "columns": {"prompt": "text"}
        }
    }
    return dataset_info

def main():
    parser = argparse.ArgumentParser(description="Generate safe Unicode Shuffle-Dyck-k sequences.")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer identifier")
    parser.add_argument("-k", "--num_symbols", type=int, default=64, help="k in k-Dyck")
    parser.add_argument("-n", "--num_samples", type=int, default=10000, help="Number of sequences")
    parser.add_argument("-l", "--seq_length", type=int, default=1024, help="Target sequence length")
    parser.add_argument("-p", "--p_open", type=float, default=0.51, help="P(opening a bracket)")
    parser.add_argument("--min_depth", type=int, default=1, help="Minimum nested depth")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum nesting depth allowed")
    
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--save_txt", action="store_true", help="Save sequences to a text file")
    parser.add_argument("--save_arrow", action="store_true", help="Save tokenized sequences as Arrow dataset")
    
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.save_txt or args.save_arrow:
        print("Finding BPE-safe Unicode characters for mapping...")
        id_mapping, char_mapping = get_safe_unicode_mapping(tokenizer, required_symbols=2 * args.num_symbols)

    if args.save_txt:
        print(f"Generating {args.num_samples} balanced sequences to {args.output_dir}...")
        filepath = generate_shuff_dyck_txt_file(
            args.output_dir, args.num_symbols, args.num_samples, args.seq_length,
            args.p_open, args.min_depth, args.max_depth, char_mapping
        )
        print(f"Txt file saved to {filepath}")

    if args.save_arrow:
        arrow_path = save_as_arrow_dataset(
            args.output_dir, args.num_symbols, args.num_samples, args.seq_length,
            args.p_open, args.min_depth, args.max_depth, id_mapping
        )
        print(f"Arrow dataset successfully saved to: {arrow_path}")
        
        # Print dataset registration info
        print("\n=== Dataset Registration Required ===")
        print("Ensure the following is in your LLaMA Factory `dataset_info.json`:")
        dataset_info = register_dataset(arrow_path)
        yaml_str = yaml.dump(dataset_info, default_flow_style=False)
        print(yaml_str)
        with open(os.path.join(args.output_dir, "dataset_info.json"),"w") as f:
            f.write(yaml_str)
            
if __name__ == "__main__":
    main()