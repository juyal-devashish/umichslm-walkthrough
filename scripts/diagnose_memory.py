#!/usr/bin/env python3
"""
Memory diagnostic script for UMichSLM pretraining.

This script measures actual GPU memory usage at different batch sizes and configurations
to help determine safe batch sizes for A100 80GB GPUs.

Usage:
    python scripts/diagnose_memory.py --size small --output memory_results.csv
    python scripts/diagnose_memory.py --size small --config moe --batch_sizes 4,8,12,16
"""

import argparse
import csv
import gc
import sys
from typing import Optional

import torch

from umich_slm import UMICHSLM_CONFIGS, UMichSLMForCausalLM, get_umichslm_config


def clear_memory(force_sync: bool = False):
    """Clear GPU memory and cache."""
    gc.collect()
    if torch.cuda.is_available():
        if force_sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# Track configs with fatal errors (e.g., no backward pass)
FATAL_CONFIGS = set()


def measure_memory(
    batch_size: int,
    seq_len: int = 2048,
    size: str = "small",
    config_name: str = "baseline",
    verbose: bool = True,
) -> tuple[float, float, bool]:
    """
    Measure peak GPU memory for a forward + backward pass.

    Args:
        batch_size: Batch size to test
        seq_len: Sequence length (default 2048)
        size: Model size preset (tiny, small, base, large)
        config_name: Configuration variant to test
        verbose: Whether to print progress

    Returns:
        Tuple of (peak_memory_gb, allocated_memory_gb, success)
        Returns (-2, -2, False) for fatal errors (e.g., no backward pass implemented)
    """
    # Check if this config has a fatal error
    if config_name in FATAL_CONFIGS:
        return -2, -2, False

    clear_memory()

    try:
        # Create config based on variant
        config = get_umichslm_config(size)

        # Apply configuration variant
        if config_name == "gqa":
            config.num_key_value_heads = 4
        elif config_name == "mla":
            config.use_mla = True
            config.mla_kv_lora_rank = 512
            config.mla_q_lora_rank = 0
            config.mla_qk_nope_head_dim = 128
            config.mla_qk_rope_head_dim = 64
            config.mla_v_head_dim = 128
        elif config_name == "moe":
            # Parameter-matched MoE: 8 experts x 256 = 2048 (same as baseline MLP)
            config.use_moe = True
            config.num_experts = 8
            config.num_experts_per_tok = 2
            config.moe_expert_intermediate_size = config.intermediate_size // 8  # 256 for small
        elif config_name == "moe_progressive":
            # Progressive MoE: first half plain MLP, then 2->4->8 experts
            config.use_moe = True
            config.num_experts = 8
            config.num_experts_per_tok = 2
            config.moe_expert_intermediate_size = config.intermediate_size // 8
            config.moe_progressive = True
            config.moe_progressive_start_ratio = 0.5
        elif config_name == "mod":
            config.use_mod = True
            config.mod_capacity = 0.125
        elif config_name == "mod_moe":
            # MoD + parameter-matched MoE
            config.use_mod = True
            config.mod_capacity = 0.125
            config.use_moe = True
            config.num_experts = 8
            config.num_experts_per_tok = 2
            config.moe_expert_intermediate_size = config.intermediate_size // 8
        elif config_name == "mtp":
            config.use_mtp = True
            config.mtp_num_heads = 2
        elif config_name == "deltanet_gated":
            config.use_deltanet = True
            config.deltanet_variant = "gated"
        elif config_name == "deltanet_kda":
            config.use_deltanet = True
            config.deltanet_variant = "kda"
        elif config_name == "pope":
            config.position_embedding_type = "pope"
        elif config_name == "swa512":
            config.sliding_window = 512
        elif config_name == "zcrmsnorm":
            config.use_zero_centered_rmsnorm = True
        # baseline uses default config (MHA)

        if verbose:
            print(f"  Creating model with config: {config_name}")

        # Create model
        model = UMichSLMForCausalLM(config).cuda().bfloat16()
        model.train()

        # Record memory after model creation
        model_memory = torch.cuda.memory_allocated() / 1e9
        if verbose:
            print(f"  Model memory: {model_memory:.2f} GB")

        # Create dummy input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        labels = input_ids.clone()

        # Forward pass
        if verbose:
            print(f"  Running forward pass with batch_size={batch_size}...")
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        forward_memory = torch.cuda.max_memory_allocated() / 1e9
        if verbose:
            print(f"  Memory after forward: {forward_memory:.2f} GB")

        # Backward pass
        if verbose:
            print(f"  Running backward pass...")
        loss.backward()

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        allocated_memory = torch.cuda.memory_allocated() / 1e9

        if verbose:
            print(f"  Peak memory: {peak_memory:.2f} GB")
            print(f"  Allocated after backward: {allocated_memory:.2f} GB")

        # Cleanup
        del model, outputs, loss, input_ids, labels
        clear_memory()

        return peak_memory, allocated_memory, True

    except torch.cuda.OutOfMemoryError as e:
        if verbose:
            print(f"  OOM at batch_size={batch_size}: {e}")
        clear_memory(force_sync=True)
        return -1, -1, False
    except NotImplementedError as e:
        # Fatal error - mark this config as broken
        FATAL_CONFIGS.add(config_name)
        if verbose:
            print(f"  FATAL: {config_name} has no backward pass: {e}")
        # Try to recover CUDA state
        try:
            clear_memory(force_sync=True)
        except Exception:
            pass
        return -2, -2, False
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  Error at batch_size={batch_size}: {e}")
        # Check if this is a backward pass issue
        if "backward" in error_msg.lower() or "not implemented" in error_msg.lower():
            FATAL_CONFIGS.add(config_name)
            if verbose:
                print(f"  Marking {config_name} as FATAL (likely no backward pass)")
            return -2, -2, False
        try:
            clear_memory(force_sync=True)
        except Exception:
            pass
        return -1, -1, False


def find_max_batch_size(
    size: str = "small",
    config_name: str = "baseline",
    seq_len: int = 2048,
    max_memory_gb: float = 75.0,  # Leave some headroom on 80GB
    verbose: bool = True,
) -> int:
    """
    Binary search to find maximum safe batch size.

    Args:
        size: Model size preset
        config_name: Configuration variant
        seq_len: Sequence length
        max_memory_gb: Maximum memory to use (with headroom)
        verbose: Whether to print progress

    Returns:
        Maximum safe batch size
    """
    # Start with powers of 2 to quickly find the range
    batch_sizes_to_try = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

    max_working = 0

    for bs in batch_sizes_to_try:
        if verbose:
            print(f"\nTesting batch_size={bs}...")

        peak_memory, _, success = measure_memory(
            batch_size=bs,
            seq_len=seq_len,
            size=size,
            config_name=config_name,
            verbose=verbose,
        )

        if success and peak_memory < max_memory_gb:
            max_working = bs
            if verbose:
                print(f"  SUCCESS: batch_size={bs} uses {peak_memory:.2f} GB")
        else:
            if verbose:
                if success:
                    print(f"  EXCEEDS LIMIT: batch_size={bs} uses {peak_memory:.2f} GB > {max_memory_gb} GB")
                else:
                    print(f"  FAILED: batch_size={bs} OOM")
            break

    return max_working


def run_full_diagnosis(
    size: str = "small",
    seq_len: int = 2048,
    output_file: Optional[str] = None,
    configs: Optional[list] = None,
    batch_sizes: Optional[list] = None,
):
    """
    Run full memory diagnosis across configurations.

    Args:
        size: Model size preset
        seq_len: Sequence length
        output_file: Optional CSV file to save results
        configs: List of configurations to test (default: all)
        batch_sizes: List of batch sizes to test (default: auto-detect)
    """
    if configs is None:
        configs = [
            "baseline", "gqa", "mla", "mtp", "moe", "moe_progressive", "mod", "mod_moe",
            "deltanet_gated", "deltanet_kda", "pope", "swa512", "zcrmsnorm"
        ]

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]

    results = []

    print(f"\n{'='*70}")
    print(f"Memory Diagnosis for UMichSLM ({size} model, seq_len={seq_len})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*70}\n")

    for config_name in configs:
        print(f"\n{'='*50}")
        print(f"Configuration: {config_name}")
        print(f"{'='*50}")

        # Check if already marked as fatal from a previous run
        if config_name in FATAL_CONFIGS:
            print(f"  SKIPPED: {config_name} has fatal errors (no backward pass)")
            results.append({
                "config": config_name,
                "batch_size": 0,
                "peak_memory_gb": "FATAL",
                "allocated_memory_gb": "FATAL",
                "status": "FATAL",
            })
            continue

        for bs in batch_sizes:
            print(f"\n  Testing batch_size={bs}...")
            peak_memory, allocated_memory, success = measure_memory(
                batch_size=bs,
                seq_len=seq_len,
                size=size,
                config_name=config_name,
                verbose=False,
            )

            # Check for fatal error (-2)
            if peak_memory == -2:
                status = "FATAL"
                print(f"    [{status}] - No backward pass implemented, skipping config")
                results.append({
                    "config": config_name,
                    "batch_size": bs,
                    "peak_memory_gb": "FATAL",
                    "allocated_memory_gb": "FATAL",
                    "status": status,
                })
                break

            status = "OK" if success else "OOM"
            if success:
                print(f"    Peak: {peak_memory:.2f} GB, Allocated: {allocated_memory:.2f} GB [{status}]")
            else:
                print(f"    [{status}]")

            results.append({
                "config": config_name,
                "batch_size": bs,
                "peak_memory_gb": peak_memory if success else "OOM",
                "allocated_memory_gb": allocated_memory if success else "OOM",
                "status": status,
            })

            # Stop testing larger batch sizes if we hit OOM
            if not success:
                break

    # Print summary
    print(f"\n{'='*70}")
    print("Summary: Maximum Safe Batch Sizes")
    print(f"{'='*70}")

    for config_name in configs:
        # Check for fatal errors first
        fatal_results = [r for r in results if r["config"] == config_name and r["status"] == "FATAL"]
        if fatal_results:
            print(f"  {config_name:20s}: FATAL - No backward pass (cannot train)")
            continue

        config_results = [r for r in results if r["config"] == config_name and r["status"] == "OK"]
        if config_results:
            max_bs = max(r["batch_size"] for r in config_results)
            max_result = [r for r in config_results if r["batch_size"] == max_bs][0]
            print(f"  {config_name:20s}: batch_size={max_bs:3d} (peak={max_result['peak_memory_gb']:.1f} GB)")
        else:
            print(f"  {config_name:20s}: NO VALID BATCH SIZE FOUND (try batch_size=1)")

    # Save to CSV if requested
    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["config", "batch_size", "peak_memory_gb", "allocated_memory_gb", "status"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose GPU memory usage for UMichSLM pretraining")
    parser.add_argument("--size", type=str, default="small", choices=list(UMICHSLM_CONFIGS.keys()),
                        help="Model size preset")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for results")
    parser.add_argument("--config", type=str, default=None,
                        help="Single config to test (default: test all)")
    parser.add_argument("--batch_sizes", type=str, default=None,
                        help="Comma-separated batch sizes to test (e.g., '4,8,16,32')")
    parser.add_argument("--find_max", action="store_true",
                        help="Find maximum batch size for a single config")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    # Parse batch sizes if provided
    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Parse configs
    configs = None
    if args.config:
        configs = [args.config]

    if args.find_max:
        if not args.config:
            print("ERROR: --find_max requires --config to be specified")
            sys.exit(1)
        max_bs = find_max_batch_size(
            size=args.size,
            config_name=args.config,
            seq_len=args.seq_len,
        )
        print(f"\nMaximum safe batch size for {args.config}: {max_bs}")
    else:
        run_full_diagnosis(
            size=args.size,
            seq_len=args.seq_len,
            output_file=args.output,
            configs=configs,
            batch_sizes=batch_sizes,
        )


if __name__ == "__main__":
    main()
