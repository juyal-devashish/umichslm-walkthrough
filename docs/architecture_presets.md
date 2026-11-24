# Architecture Presets

This document records the repository's Qwen-style preset families and the supported ways to create models from them.

## Qwen3-Like Presets

Dense GQA transformer presets with RoPE, RMSNorm, and SwiGLU.

| Size | Hidden | Layers | Heads | KV Heads | Backbone Params | Total Params |
|------|--------|--------|-------|----------|-----------------|--------------|
| tiny | 216 | 10 | 6 | 2 | ~5.0M | ~26.2M |
| xsmall | 432 | 10 | 6 | 2 | ~19.9M | ~62.4M |
| small | 768 | 16 | 12 | 4 | ~100.7M | ~176.2M |

Preset names:
- `qwen3_like_tiny`
- `qwen3_like_xsmall`
- `qwen3_like_small`

## Qwen3.5-Like Presets

Hybrid presets built from gated DeltaNet plus periodic standard attention. The default pattern is `["delta", "delta", "delta", "attn"]`, so every fourth layer uses standard attention and the others use DeltaNet.

| Size | Hidden | Layers | Heads | KV Heads | Backbone Params | Total Params |
|------|--------|--------|-------|----------|-----------------|--------------|
| tiny | 216 | 9 | 6 | 2 | ~4.9M | ~26.2M |
| xsmall | 336 | 15 | 6 | 2 | ~19.9M | ~53.0M |
| small | 552 | 28 | 6 | 2 | ~99.8M | ~154.0M |

Preset names:
- `qwen3_5_like_tiny`
- `qwen3_5_like_xsmall`
- `qwen3_5_like_small`

## Open-ELM Layer-Wise Scaling 
Scales the heads and ffn multiplier linearly with each layer, gradually widening the layers as they approach the output layer. The authors argue this can create a more efficient parameter allocation and more accurate model. 

| Size | Hidden | Layers | Heads | KV Heads | FFN Multiplier | Backbone Params | Total Params |
|------|--------|--------|-------|----------|-----------------|--------------|-----------------|
| tiny | 216 | 10 | 2 to 6 | 2 | 1.5 to 3.0 | ~4.6M | ~25.9M |
| xsmall | 432 | 10 | 2 to 8 | 2 | 1.5 to 3.0 | ~18.1M | ~60.6M |
| small | 768 | 16 | 4 to 12 | 4 | 1.5 to 4.0 | ~107.4M | ~182.9M | 

Preset names: 
- `openelm_like_tiny`
- `openelm_like_xsmall`
- `openelm_like_small`

## Hymba-like KV-cache sharing
| Size | Hidden | Layers | Heads | KV Heads | Backbone Params | Total Params |
|------|--------|--------|-------|----------|--------------|-----------------|
| tiny | 216 | 10 | 6 | 2 | ~4.8M | ~26.1M |
| xsmall | 432 | 10 | 6 | 2 | ~19.3M | ~61.8M |
| small | 768 | 16 | 12 | 4 | ~97.5M | ~173.0M |

Preset names:
- `hymba_like_tiny`
- `hymba_like_xsmall`
- `hymba_like_small`

## Baguettotron-Like Presets

Deep-and-thin presets inspired by Baguettotron's 80-layer architecture. The defining characteristic is an extreme depth-to-width ratio (~0.14–0.16 layers per hidden unit), which trades representational width for sequential reasoning depth. Uses an explicit `head_dim=64` decoupled from `hidden_size / num_heads`, keeping RoPE positional encoding consistent across scales.

| Size | Hidden | Layers | Heads | KV Heads | head_dim | Depth Ratio | Backbone Params | Total Params |
|------|--------|--------|-------|----------|----------|-------------|-----------------|--------------|
| micro | 128 | 20 | 2 | 1 | 64 | 0.156 | ~4M | ~16M |
| tiny | 192 | 30 | 3 | 1 | 64 | 0.156 | ~13M | ~30M |
| small | 288 | 40 | 3 | 1 | 64 | 0.139 | ~36M | ~60M |

Preset names:
- `baguette_micro`
- `baguette_tiny`
- `baguette_small`
- 
## MobileLLM-Style Presets (Deep-and-Thin + Block-Wise Weight Sharing)

Deep-and-thin architecture optimised for sub-100M parameter models. Prioritises depth over width — many layers with small hidden size — following the findings of [MobileLLM (arXiv:2402.14905)](https://arxiv.org/abs/2402.14905). The `_ls` variants add block-wise weight sharing (MobileLLM-LS), where every pair of consecutive layers shares weights, doubling effective depth at no parameter cost.

| Size | Hidden | Layers | Heads | KV Heads | Intermediate | Backbone Params | Total Params |
|------|--------|--------|-------|----------|----------|-------------|-----------------|
| tiny | 128 | 28 | 4 | 1 | 384 | ~5.3M | 9.38M |
| xsmall | 256 | 26 | 4 | 2 | 768 | 19.61M | 27.8M |
| small | 576 | 28 | 9 | 3 | 1536 | ~99.12M | 117.56M |

All presets use tied embeddings (`tie_word_embeddings=True`) and `max_position_embeddings=2048`. `vocab_size=32000` is a placeholder — override to match your tokenizer.

Preset names:
- `mobilellm_tiny` / `mobilellm_tiny_ls`
- `mobilellm_xsmall` / `mobilellm_xsmall_ls`
- `mobilellm_small` / `mobilellm_small_ls`

## MoD (Mixture-of-Depths) Presets

Dense GQA transformers with MoD routing at every other block (12.5% token capacity),
following the paper's optimal hyperparameters. These are intentionally larger than the
qwen3_like presets — they're isoFLOP-equivalent, not larger in effective compute.

| Size   | Hidden | Layers | Heads | KV | Backbone Params | Total Params |
|--------|--------|--------|-------|----|-----------------|--------------|
| tiny   | 216    | 18     | 6     | 2  | ~9M             | ~30M         |
| xsmall | 432    | 18     | 6     | 2  | ~35M            | ~78M         |
| small  | 768    | 28     | 12    | 4  | ~176M           | ~250M        |

Key MoD hyperparameters
- `mod_capacity=0.125`      → 12.5% of tokens processed per routed block (paper's optimal)
- `mod_every_n_layers=2`    → route every other block (crucial for performance)
- `mod_aux_loss_weight=0.01` → router auxiliary loss

Preset names:
- `mod_tiny`
- `mod_xsmall`
- `mod_small`

### Python API

```python
from umich_slm import UMichSLMForCausalLM, get_umichslm_config

# Standard deep-and-thin
config = get_umichslm_config("mobilellm_small")
model = UMichSLMForCausalLM(config)

# With block-wise weight sharing (LS variant)
config = get_umichslm_config("mobilellm_small_ls")
model = UMichSLMForCausalLM(config)
```

Custom config with weight sharing:

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM

config = UMichSLMConfig(
    hidden_size=576,
    intermediate_size=1536,
    num_hidden_layers=28,
    num_attention_heads=9,
    num_key_value_heads=3,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    use_block_weight_sharing=True,
    weight_sharing_block_size=2,    # pairs of layers share weights
)
model = UMichSLMForCausalLM(config)
```

### Training Script

```bash
# Standard MobileLLM-style (deep-and-thin only)
python scripts/pretrain.py --size mobilellm_small --output_dir ./outputs

# MobileLLM-LS (with block-wise weight sharing)
python scripts/pretrain.py --size mobilellm_small_ls --output_dir ./outputs

# Or enable weight sharing on any size via flag
python scripts/pretrain.py \
    --size mobilellm_small \
    --use_block_weight_sharing \
    --weight_sharing_block_size 2
```

### Notes

- Block-wise weight sharing works by aliasing layer `i` to layer `i - (i % block_size)`. With `block_size=2`, layers 0&1 share, 2&3 share, 4&5 share, and so on.
- The parameter count is unchanged by sharing — only unique layers hold parameters. The optimizer deduplicates shared parameters automatically.
- `weight_sharing_block_size` can be set to any value ≥ 2. The paper uses 2 (pairs).
- Compatible with all other features: MoE, MoD, YaRN, GQA, etc.
- Block-wise weight sharing will decrease the parameter count by half. So a model with 10M parameters will have 5M approximately. 

## Griffin Hybrid Recurrent-Attention

Hybrid architecture mixing RG-LRU (Real-Gated Linear Recurrent Unit) layers with standard attention layers. Most layers use the RG-LRU recurrence (O(L), fixed-size state) and a subset use standard attention, following the 2:1 ratio from the paper.

Reference: [Griffin: Mixing Gated Linear Recurrences with Local Notations for Efficient LLMs](https://arxiv.org/abs/2402.19427)

### Key Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_griffin` | `bool` | `False` | Enables Griffin hybrid architecture |
| `griffin_pattern` | `list[str]` | `None` (all RG-LRU) | Repeating per-layer pattern, e.g. `["rglru", "rglru", "attn"]` |
| `griffin_num_heads` | `int` | `num_attention_heads` | Number of RG-LRU heads |


| Size | Hidden | Layers | Heads | KV Heads | Backbone Params | Total Params |
|------|--------|--------|-------|----------|-----------------|--------------|
| tiny | 160 | 12 | 8 | 2 | 4.78M | 20.51M |
| xsmall | 320 | 12 | 8 | 2 | 19.1M | 50.55M |
| small | 512 | 24 | 8 | 2 | 97.65M | 147.98M |

Preset names:
- `Griffin_ting`
- `Griffin_xsmall`
- `Griffin_small`



### Python API

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM

config = UMichSLMConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=1,          # MQA for attention layers (paper default)
    intermediate_size=2048,
    max_position_embeddings=4096,
    use_griffin=True,
    griffin_pattern=["rglru", "rglru", "attn"],  # 2:1 ratio from paper
)
model = UMichSLMForCausalLM(config)
```

### Training Script

```bash
# Paper default: 2 RG-LRU layers for every 1 attention layer
python scripts/pretrain.py \
    --size small \
    --use_griffin \
    --griffin_pattern "rglru,rglru,attn"

# Pure recurrent (no attention layers)
python scripts/pretrain.py \
    --size small \
    --use_griffin \
    --griffin_pattern "rglru"

# Custom head count for RG-LRU
python scripts/pretrain.py \
    --size small \
    --use_griffin \
    --griffin_pattern "rglru,rglru,attn" \
    --griffin_num_heads 8
```

### Notes

- RG-LRU layers maintain a fixed-size hidden state `(batch, num_heads, head_dim)` regardless of sequence length — memory does not grow with context.
- Attention layers in a Griffin model use whatever attention type is configured (`use_mla`, `use_deltanet`, etc.) and are otherwise unchanged.
- Griffin is compatible with MoE, MoD, YaRN, and DRoPE.
- Set `num_key_value_heads=1` for MQA on the attention layers, matching the paper.

## How Models Are Created

### Python API

Use the preset helper to build a config, then instantiate the model:

```python
from umich_slm import UMichSLMForCausalLM, get_umichslm_config

config = get_umichslm_config("qwen3_like_xsmall")
model = UMichSLMForCausalLM(config)
```

You can also create the equivalent configs directly with `UMichSLMConfig`.

Qwen3-like dense example:

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM

config = UMichSLMConfig(
    hidden_size=432,
    num_hidden_layers=10,
    num_attention_heads=6,
    num_key_value_heads=2,   # GQA: 2 KV heads
    intermediate_size=1152,
    max_position_embeddings=4096,
    position_embedding_type="rope",
)
model = UMichSLMForCausalLM(config)
```

Qwen3.5-like hybrid example:

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM

config = UMichSLMConfig(
    hidden_size=336,
    num_hidden_layers=15,
    num_attention_heads=6,
    num_key_value_heads=2,   # GQA: 2 KV heads
    intermediate_size=896,
    max_position_embeddings=4096,
    position_embedding_type="rope",
    use_deltanet=True,
    deltanet_variant="gated",
    hybrid_attention_ratio=["delta", "delta", "delta", "attn"],
)
model = UMichSLMForCausalLM(config)
```

To inspect estimated size:

```python
from umich_slm import estimate_backbone_parameters, estimate_parameters

print(estimate_backbone_parameters("qwen3_5_like_small"))
print(estimate_parameters("qwen3_5_like_small"))
```

### Training Scripts

All preset-driven scripts accept these names through `--size`:

```bash
python scripts/pretrain.py --size qwen3_like_tiny --output_dir ./outputs
python scripts/pretrain_fineweb.py --size qwen3_5_like_xsmall --output_dir ./outputs
python scripts/diagnose_memory.py --size qwen3_5_like_small
```

## Delphi width-scaling presets

Added for the Delphi-style scaling sweep (see `docs/scaling_experiments.md`).
Depth/width follow `depth = round(H / (64 + 4·log₂H − 9))` and
`n_heads = H/128`, with MLP ratio 4 and GQA 4:1. Depths are nudged ±1 from
the strict formula to hit the labelled backbone parameter count within 15 %.

| Preset         | Hidden | Layers | n_heads | n_kv | Intermediate | Backbone | Total (49 k vocab) |
|----------------|--------|--------|---------|------|--------------|----------|--------------------|
| `delphi_10m`   | 384    | 5      | 3       | 1    | 1536         | ~10.8 M  | ~48.6 M            |
| `delphi_25m`   | 512    | 6      | 4       | 1    | 2048         | ~22.8 M  | ~73.1 M            |
| `delphi_50m`   | 640    | 8      | 5       | 1    | 2560         | ~47.2 M  | ~110.1 M           |
| `delphi_100m`  | 896    | 9      | 7       | 2    | 3584         | ~105.3 M | ~193.4 M           |

The runtime arch helper `delphi_arch(H)` in `src/umich_slm/recipe/delphi.py`
returns the strict formula values; the presets above deviate slightly to
hit parameter targets.

## Notes

- `Backbone Params` excludes token embeddings and the LM head.
- `Total Params` includes embeddings and the untied LM head.
- Preset `vocab_size` values are placeholders and should be overridden to match the actual tokenizer during training.
- Preset definitions live in `src/umich_slm/configs/presets.py`.
