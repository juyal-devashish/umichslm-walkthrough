# Memory Requirements and Batch Size Guidelines

This document provides detailed memory calculations and maximum batch size recommendations for training UMichSLM models across different GPU configurations.

## Supported GPUs

| GPU | Memory | Architecture | Notes |
|-----|--------|--------------|-------|
| **NVIDIA H100 SXM** | 80 GB | Hopper | Highest performance, HBM3 |
| **NVIDIA A100** | 80 GB | Ampere | Excellent price/performance |
| **NVIDIA L40** | 48 GB | Ada Lovelace | Good for smaller models |
| **NVIDIA RTX 6000 Ada** | 48 GB | Ada Lovelace | Workstation-class |

## Assumptions

- **Sequence Length**: 2048 tokens
- **Vocabulary Size**: 128,256 (LLaMA 3 tokenizer)
- **Precision**: bf16 for model parameters, fp32 for logits/loss computation
- **Optimizer**: AdamW
- **Gradient Checkpointing**: Enabled
- **Flash Attention 2**: Enabled (where compatible)

## Memory Calculation Framework

Training memory consists of several components:

### Static Memory (Independent of Batch Size)

| Component | Size | Notes |
|-----------|------|-------|
| Parameters | P × 2 bytes | bf16 model weights |
| Gradients | P × 2 bytes | bf16 gradients |
| Optimizer States | P × 8 bytes | AdamW stores momentum + variance in fp32 |
| **Total Static** | **P × 12 bytes** | |

### Dynamic Memory (Scales with Batch Size)

| Component | Size | Notes |
|-----------|------|-------|
| Activations | L × B × S × H × 2 bytes | With gradient checkpointing, stores input to each layer |
| Logits | B × S × V × 4 bytes | fp32 for numerical stability in cross-entropy |
| Backward Overhead | ~2-3× layer activations | Peak memory during backward pass |

> **Important**: With a 128K vocabulary, the logits tensor alone consumes **~1.05 GB per sample** in fp32. This dominates memory usage and is independent of model architecture.

## Model Configurations

### Parameter Counts by Model Size

| Model | Hidden Size | Intermediate | Layers | Attention Heads | KV Heads | Total Params |
|-------|-------------|--------------|--------|-----------------|----------|--------------|
| tiny | 256 | 688 | 6 | 4 | 4 | ~70M |
| small | 768 | 2048 | 12 | 12 | 4 | ~272M |
| base | 1024 | 2752 | 24 | 16 | 4 | ~529M |
| large | 2048 | 5504 | 24 | 16 | 8 | ~1.6B |

### Static Memory by Model Size

| Model | Parameters | Static Memory (P × 12) |
|-------|------------|------------------------|
| tiny | 70M | 0.84 GB |
| small | 272M | 3.3 GB |
| base | 529M | 6.3 GB |
| large | 1.6B | 19.2 GB |

### Per-Sample Dynamic Memory

| Model | Activations | Logits (fp32) | Backward Overhead | **Total/Sample** |
|-------|-------------|---------------|-------------------|------------------|
| tiny | 6 MB | 1.05 GB | ~20 MB | **~1.08 GB** |
| small | 38 MB | 1.05 GB | ~100 MB | **~1.19 GB** |
| base | 100 MB | 1.05 GB | ~200 MB | **~1.35 GB** |
| large | 201 MB | 1.05 GB | ~400 MB | **~1.65 GB** |

---

## Maximum Batch Sizes by GPU

The maximum batch size is calculated as:

```
Max Batch = (GPU Memory - Static Memory - Safety Buffer) / Per-Sample Memory
```

Safety buffer is ~10% of GPU memory for CUDA fragmentation and framework overhead.

---

## NVIDIA H100 80GB

High-performance datacenter GPU with HBM3 memory.

### Baseline (Standard Transformer)

| Model | Static | Available | Per Sample | **Max Batch** | Recommended |
|-------|--------|-----------|------------|---------------|-------------|
| tiny | 0.84 GB | 71.2 GB | 1.08 GB | ~66 | **48** |
| small | 3.3 GB | 68.7 GB | 1.19 GB | ~58 | **42** |
| base | 6.3 GB | 65.7 GB | 1.35 GB | ~49 | **36** |
| large | 19.2 GB | 52.8 GB | 1.65 GB | ~32 | **24** |

### With Multi-Token Prediction (MTP)

| Model | Extra Static | Extra/Sample | **Max Batch** | Recommended |
|-------|--------------|--------------|---------------|-------------|
| tiny | +50 MB | +20 MB | ~64 | **46** |
| small | +130 MB | +50 MB | ~54 | **40** |
| base | +200 MB | +80 MB | ~45 | **32** |
| large | +400 MB | +150 MB | ~29 | **22** |

### With Mixture of Experts (MoE, 8 experts, top-2)

| Model | Total Params | Static Memory | **Max Batch** | Recommended |
|-------|--------------|---------------|---------------|-------------|
| tiny | ~200M | 2.4 GB | ~60 | **44** |
| small | ~670M | 8.0 GB | ~50 | **36** |
| base | ~1.5B | 18 GB | ~38 | **28** |
| large | ~4.5B | 54 GB | ~12 | **8** |

### With Mixture of Depths (MoD, capacity=12.5%)

| Model | Extra Params | Activation Savings | **Max Batch** | Recommended |
|-------|--------------|-------------------|---------------|-------------|
| tiny | ~30K | -25% | ~74 | **56** |
| small | ~90K | -25% | ~66 | **48** |
| base | ~240K | -25% | ~58 | **42** |
| large | ~480K | -25% | ~40 | **30** |

### With DeltaNet (Linear Attention)

| Model | Recurrent State/Sample | **Max Batch** | Recommended |
|-------|------------------------|---------------|-------------|
| tiny | ~0.5 MB | ~68 | **50** |
| small | ~1 MB | ~60 | **44** |
| base | ~2 MB | ~51 | **38** |
| large | ~8 MB | ~34 | **26** |

### H100 Summary Table

| Model | Baseline | +MTP | +MoE | +MoD | +DeltaNet |
|-------|----------|------|------|------|-----------|
| **tiny** | 48 | 46 | 44 | **56** | 50 |
| **small** | 42 | 40 | 36 | **48** | 44 |
| **base** | 36 | 32 | 28 | **42** | 38 |
| **large** | 24 | 22 | 8 | **30** | 26 |

---

## NVIDIA A100 80GB

Excellent datacenter GPU with high memory bandwidth.

### Baseline (Standard Transformer)

| Model | Static | Available | Per Sample | **Max Batch** | Recommended |
|-------|--------|-----------|------------|---------------|-------------|
| tiny | 0.84 GB | 71.2 GB | 1.08 GB | ~66 | **48** |
| small | 3.3 GB | 68.7 GB | 1.19 GB | ~58 | **42** |
| base | 6.3 GB | 65.7 GB | 1.35 GB | ~49 | **36** |
| large | 19.2 GB | 52.8 GB | 1.65 GB | ~32 | **24** |

### With Multi-Token Prediction (MTP)

| Model | Extra Static | Extra/Sample | **Max Batch** | Recommended |
|-------|--------------|--------------|---------------|-------------|
| tiny | +50 MB | +20 MB | ~64 | **46** |
| small | +130 MB | +50 MB | ~54 | **40** |
| base | +200 MB | +80 MB | ~45 | **32** |
| large | +400 MB | +150 MB | ~29 | **22** |

### With Mixture of Experts (MoE, 8 experts, top-2)

| Model | Total Params | Static Memory | **Max Batch** | Recommended |
|-------|--------------|---------------|---------------|-------------|
| tiny | ~200M | 2.4 GB | ~60 | **44** |
| small | ~670M | 8.0 GB | ~50 | **36** |
| base | ~1.5B | 18 GB | ~38 | **28** |
| large | ~4.5B | 54 GB | ~12 | **8** |

### With Mixture of Depths (MoD, capacity=12.5%)

| Model | Extra Params | Activation Savings | **Max Batch** | Recommended |
|-------|--------------|-------------------|---------------|-------------|
| tiny | ~30K | -25% | ~74 | **56** |
| small | ~90K | -25% | ~66 | **48** |
| base | ~240K | -25% | ~58 | **42** |
| large | ~480K | -25% | ~40 | **30** |

### With DeltaNet (Linear Attention)

| Model | Recurrent State/Sample | **Max Batch** | Recommended |
|-------|------------------------|---------------|-------------|
| tiny | ~0.5 MB | ~68 | **50** |
| small | ~1 MB | ~60 | **44** |
| base | ~2 MB | ~51 | **38** |
| large | ~8 MB | ~34 | **26** |

### A100 Summary Table

| Model | Baseline | +MTP | +MoE | +MoD | +DeltaNet |
|-------|----------|------|------|------|-----------|
| **tiny** | 48 | 46 | 44 | **56** | 50 |
| **small** | 42 | 40 | 36 | **48** | 44 |
| **base** | 36 | 32 | 28 | **42** | 38 |
| **large** | 24 | 22 | 8 | **30** | 26 |

---

## NVIDIA L40 48GB

Ada Lovelace datacenter GPU, good balance of performance and cost.

### Baseline (Standard Transformer)

| Model | Static | Available | Per Sample | **Max Batch** | Recommended |
|-------|--------|-----------|------------|---------------|-------------|
| tiny | 0.84 GB | 42.4 GB | 1.08 GB | ~39 | **28** |
| small | 3.3 GB | 39.9 GB | 1.19 GB | ~34 | **24** |
| base | 6.3 GB | 36.9 GB | 1.35 GB | ~27 | **20** |
| large | 19.2 GB | 24.0 GB | 1.65 GB | ~15 | **10** |

### With Multi-Token Prediction (MTP)

| Model | Extra Static | Extra/Sample | **Max Batch** | Recommended |
|-------|--------------|--------------|---------------|-------------|
| tiny | +50 MB | +20 MB | ~37 | **26** |
| small | +130 MB | +50 MB | ~31 | **22** |
| base | +200 MB | +80 MB | ~24 | **18** |
| large | +400 MB | +150 MB | ~13 | **8** |

### With Mixture of Experts (MoE, 8 experts, top-2)

| Model | Total Params | Static Memory | **Max Batch** | Recommended |
|-------|--------------|---------------|---------------|-------------|
| tiny | ~200M | 2.4 GB | ~35 | **26** |
| small | ~670M | 8.0 GB | ~27 | **20** |
| base | ~1.5B | 18 GB | ~17 | **12** |
| large | ~4.5B | 54 GB | OOM | **N/A** |

> **Note**: Large MoE does not fit on L40. Use DeepSpeed ZeRO-3 or multi-GPU training.

### With Mixture of Depths (MoD, capacity=12.5%)

| Model | Extra Params | Activation Savings | **Max Batch** | Recommended |
|-------|--------------|-------------------|---------------|-------------|
| tiny | ~30K | -25% | ~45 | **32** |
| small | ~90K | -25% | ~39 | **28** |
| base | ~240K | -25% | ~33 | **24** |
| large | ~480K | -25% | ~19 | **14** |

### With DeltaNet (Linear Attention)

| Model | Recurrent State/Sample | **Max Batch** | Recommended |
|-------|------------------------|---------------|-------------|
| tiny | ~0.5 MB | ~40 | **30** |
| small | ~1 MB | ~35 | **26** |
| base | ~2 MB | ~28 | **20** |
| large | ~8 MB | ~16 | **12** |

### L40 Summary Table

| Model | Baseline | +MTP | +MoE | +MoD | +DeltaNet |
|-------|----------|------|------|------|-----------|
| **tiny** | 28 | 26 | 26 | **32** | 30 |
| **small** | 24 | 22 | 20 | **28** | 26 |
| **base** | 20 | 18 | 12 | **24** | 20 |
| **large** | 10 | 8 | N/A | **14** | 12 |

---

## NVIDIA RTX 6000 Ada 48GB

Professional workstation GPU with excellent performance.

### Baseline (Standard Transformer)

| Model | Static | Available | Per Sample | **Max Batch** | Recommended |
|-------|--------|-----------|------------|---------------|-------------|
| tiny | 0.84 GB | 42.4 GB | 1.08 GB | ~39 | **28** |
| small | 3.3 GB | 39.9 GB | 1.19 GB | ~34 | **24** |
| base | 6.3 GB | 36.9 GB | 1.35 GB | ~27 | **20** |
| large | 19.2 GB | 24.0 GB | 1.65 GB | ~15 | **10** |

### With Multi-Token Prediction (MTP)

| Model | Extra Static | Extra/Sample | **Max Batch** | Recommended |
|-------|--------------|--------------|---------------|-------------|
| tiny | +50 MB | +20 MB | ~37 | **26** |
| small | +130 MB | +50 MB | ~31 | **22** |
| base | +200 MB | +80 MB | ~24 | **18** |
| large | +400 MB | +150 MB | ~13 | **8** |

### With Mixture of Experts (MoE, 8 experts, top-2)

| Model | Total Params | Static Memory | **Max Batch** | Recommended |
|-------|--------------|---------------|---------------|-------------|
| tiny | ~200M | 2.4 GB | ~35 | **26** |
| small | ~670M | 8.0 GB | ~27 | **20** |
| base | ~1.5B | 18 GB | ~17 | **12** |
| large | ~4.5B | 54 GB | OOM | **N/A** |

> **Note**: Large MoE does not fit on RTX 6000 Ada. Use DeepSpeed ZeRO-3 or multi-GPU training.

### With Mixture of Depths (MoD, capacity=12.5%)

| Model | Extra Params | Activation Savings | **Max Batch** | Recommended |
|-------|--------------|-------------------|---------------|-------------|
| tiny | ~30K | -25% | ~45 | **32** |
| small | ~90K | -25% | ~39 | **28** |
| base | ~240K | -25% | ~33 | **24** |
| large | ~480K | -25% | ~19 | **14** |

### With DeltaNet (Linear Attention)

| Model | Recurrent State/Sample | **Max Batch** | Recommended |
|-------|------------------------|---------------|-------------|
| tiny | ~0.5 MB | ~40 | **30** |
| small | ~1 MB | ~35 | **26** |
| base | ~2 MB | ~28 | **20** |
| large | ~8 MB | ~16 | **12** |

### RTX 6000 Ada Summary Table

| Model | Baseline | +MTP | +MoE | +MoD | +DeltaNet |
|-------|----------|------|------|------|-----------|
| **tiny** | 28 | 26 | 26 | **32** | 30 |
| **small** | 24 | 22 | 20 | **28** | 26 |
| **base** | 20 | 18 | 12 | **24** | 20 |
| **large** | 10 | 8 | N/A | **14** | 12 |

---

## Cross-GPU Comparison

### Recommended Batch Sizes (Baseline Configuration)

| Model | H100 80GB | A100 80GB | L40 48GB | RTX 6000 48GB |
|-------|-----------|-----------|----------|---------------|
| **tiny** | 48 | 48 | 28 | 28 |
| **small** | 42 | 42 | 24 | 24 |
| **base** | 36 | 36 | 20 | 20 |
| **large** | 24 | 24 | 10 | 10 |

### Recommended Batch Sizes (MoD Configuration - Best for Memory)

| Model | H100 80GB | A100 80GB | L40 48GB | RTX 6000 48GB |
|-------|-----------|-----------|----------|---------------|
| **tiny** | 56 | 56 | 32 | 32 |
| **small** | 48 | 48 | 28 | 28 |
| **base** | 42 | 42 | 24 | 24 |
| **large** | 30 | 30 | 14 | 14 |

### Which GPU for Which Model?

| Model Size | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| tiny | Any (even 24GB) | L40/RTX 6000 |
| small | L40/RTX 6000 (24+ batch) | A100/H100 |
| base | L40/RTX 6000 (limited batch) | A100/H100 |
| large | A100/H100 | H100 |
| large + MoE | Multi-GPU required | 2+ H100s with DeepSpeed |

---

## Configuration Examples

### Small Model on L40 (Memory Constrained)
```bash
python scripts/pretrain_fineweb.py \
    --size small \
    --batch_size 24 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16
```

### Small Model on H100 (High Throughput)
```bash
python scripts/pretrain_fineweb.py \
    --size small \
    --batch_size 42 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16
```

### Base Model with MoD on L40 (Memory Efficient)
```bash
python scripts/pretrain_fineweb.py \
    --size base \
    --use_mod \
    --mod_capacity 0.125 \
    --batch_size 24 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16
```

### Large Model on A100 with DeepSpeed
```bash
python scripts/pretrain_fineweb.py \
    --size large \
    --batch_size 24 \
    --gradient_accumulation_steps 8 \
    --deepspeed configs/llama_factory/ds_config_zero2.json \
    --gradient_checkpointing \
    --bf16
```

### Large Model on L40 (Requires DeepSpeed ZeRO-3)
```bash
python scripts/pretrain_fineweb.py \
    --size large \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --deepspeed configs/llama_factory/ds_config_zero3.json \
    --gradient_checkpointing \
    --bf16
```

---

## Memory Optimization Strategies

### By GPU Memory Tier

#### 48 GB GPUs (L40, RTX 6000 Ada)

1. **Use smaller batch sizes** with higher gradient accumulation
2. **Enable MoD** for 25% activation memory savings
3. **Avoid MoE** on large models (won't fit)
4. **Consider DeltaNet** for slight memory improvement
5. **Use DeepSpeed ZeRO-2/3** for large models

```bash
# Recommended for 48GB GPUs
--batch_size 16-24 --gradient_accumulation_steps 8-16
```

#### 80 GB GPUs (A100, H100)

1. **Maximize batch size** for better GPU utilization
2. **MoE is viable** for all but large models
3. **MoD still beneficial** if memory is tight
4. **DeepSpeed ZeRO-2** for large + MoE configurations

```bash
# Recommended for 80GB GPUs
--batch_size 32-48 --gradient_accumulation_steps 4-8
```

### Universal Tips

1. **Start conservative** - Begin at 70% of calculated max batch size
2. **Monitor memory** - Watch for OOM during first few steps
3. **Gradient accumulation** - Maintain effective batch size while reducing memory
4. **Sequence length** - Reducing from 2048 to 1024 nearly halves logits memory

---

## Why is the Vocabulary Size So Impactful?

With the LLaMA 3 tokenizer (128K vocab), the logits tensor for loss computation is enormous:

```
Logits Memory = batch_size × seq_len × vocab_size × 4 bytes
             = B × 2048 × 128,256 × 4
             = 1.05 GB × B
```

This memory cost:
- Is **independent of model architecture** (tiny to large all have the same logits size)
- Uses **fp32** for numerical stability in cross-entropy loss
- Does **not benefit from gradient checkpointing** (computed at the end)

### Vocabulary Size Comparison

| Tokenizer | Vocab Size | Logits/Sample | Impact |
|-----------|------------|---------------|--------|
| LLaMA 2 | 32,000 | 0.26 GB | Baseline |
| LLaMA 3 | 128,256 | 1.05 GB | **4× larger** |

The 4× vocabulary increase adds ~0.8 GB per sample overhead.

---

## Troubleshooting OOM Errors

### Quick Diagnosis

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM at model load | Model too large for GPU | Use DeepSpeed ZeRO or smaller model |
| OOM at first forward | Batch too large | Reduce `--batch_size` |
| OOM during backward | Activation memory | Verify `--gradient_checkpointing` enabled |
| OOM with MoE | Too many experts | Use MoE on fewer layers |
| Gradual OOM | Memory leak | Check for tensor accumulation |

### Debug Commands

```bash
# Test minimum viable configuration
python scripts/pretrain_fineweb.py --size small --batch_size 1 --max_steps 5

# Binary search for max batch size
python scripts/pretrain_fineweb.py --size small --batch_size 16 --max_steps 5
python scripts/pretrain_fineweb.py --size small --batch_size 24 --max_steps 5
python scripts/pretrain_fineweb.py --size small --batch_size 32 --max_steps 5
```

### Memory Monitoring

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

---

## Appendix: Detailed Memory Formulas

### Total Memory Requirement

```
Total = Static + Dynamic

Static = Parameters × 12 bytes (bf16 params + bf16 grads + fp32 optimizer)

Dynamic = Activations + Logits + Backward_Overhead
        = (L × B × S × H × 2) + (B × S × V × 4) + (~0.2 × B × S × H × L × 2)
```

### Per-Configuration Adjustments

| Configuration | Static Δ | Dynamic Δ |
|---------------|----------|-----------|
| +MTP (2 heads, shared) | +4% params | +50 MB/sample |
| +MoE (8 experts) | +700% MLP params | ~same |
| +MoD (12.5% capacity) | +0.01% params | -25% activations |
| +DeltaNet | ~same | -5% (avoids attention) |

### Effective Batch Size

```
Effective Batch = batch_size × gradient_accumulation_steps × num_gpus
```

For consistent training dynamics, maintain similar effective batch sizes across different hardware configurations.

## AdamH memory profile (Delphi recipe)

`AdamH` (`src/umich_slm/optim/adamh.py`) uses the same per-parameter state
as `AdamW` (two moment buffers, one step counter) plus a single scalar
per Hyperball-constrained tensor (the captured `init_norm`). Net VRAM
delta vs `AdamW`: negligible (one float per tensor, not per element).

The Hyperball projection step is in-place, so peak activation memory does
not change either. If you are switching from `AdamW` to `AdamH`, you can
reuse the same `--batch_size` / `--gradient_accumulation_steps`.
