# YaRN: Context Length Extension for RoPE

YaRN (Yet Another RoPE extensioN) extends the effective context window of models trained with RoPE without requiring full retraining. It is implemented as a `rope_type` option on `RotaryEmbedding` and is exposed through the `rope_scaling` config field.

Reference: [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)

---

## How It Works

Standard RoPE assigns a fixed frequency to each dimension of the head. When you extend the context beyond the training length, high-frequency dimensions extrapolate well but low-frequency ones break down. YaRN fixes this by treating different frequency dimensions differently:

| Frequency Region | Strategy | Effect |
|------------------|----------|--------|
| High-freq (short wavelength) | No scaling — keep original `inv_freq` | Preserves local position sensitivity |
| Low-freq (long wavelength) | Linear interpolation — divide `inv_freq` by `scale` | Stretches the periodic structure to cover longer contexts |
| Transition region | Smooth linear ramp between the two | Avoids hard discontinuities |

The boundary between high- and low-frequency regions is controlled by `beta_fast` and `beta_slow`, which are mapped to dimension indices via the wavelength formula from the paper.

Additionally, YaRN applies an **attention temperature correction** (`mscale`) to the cos/sin values to compensate for the distribution shift introduced by context extension:

```
mscale = 0.1 * yarn_mscale * ln(scale) + 1.0
```

---

## Configuration

YaRN is configured via the `rope_scaling` dict on `UMichSLMConfig`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `str` | — | Must be `"yarn"` |
| `factor` | `float` | — | Context extension multiplier, must be > 1.0 |
| `beta_fast` | `float` | `32.0` | High-frequency boundary (rotation count) |
| `beta_slow` | `float` | `1.0` | Low-frequency boundary (rotation count) |
| `mscale` | `float` | `1.0` | Attention temperature scaling multiplier |

> `beta_fast` and `beta_slow` correspond to the number of rotations a dimension completes over the original training length. Dimensions completing more than `beta_fast` rotations are treated as high-frequency; fewer than `beta_slow` as low-frequency.

---

## Usage

### Python API

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM

config = UMichSLMConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    max_position_embeddings=4096,   # original training length
    rope_theta=10000.0,
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,              # extend to 4× = 16384 tokens
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
    },
)
model = UMichSLMForCausalLM(config)
```

### From a Preset

```python
from umich_slm import get_umichslm_config, UMichSLMForCausalLM

config = get_umichslm_config("small")
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
}
model = UMichSLMForCausalLM(config)
```

### Pretraining Script

```bash
# 4× context extension with YaRN defaults
python scripts/pretrain.py \
    --size small \
    --rope_type yarn \
    --rope_scaling_factor 4.0

# Custom thresholds and temperature scaling
python scripts/pretrain.py \
    --size small \
    --rope_type yarn \
    --rope_scaling_factor 8.0 \
    --yarn_beta_fast 16.0 \
    --yarn_beta_slow 2.0 \
    --yarn_mscale 0.707
```

---

## Choosing Parameters

### `factor`

The ratio between the target context length and the original training length.

| Target Context | Training Length | `factor` |
|----------------|-----------------|----------|
| 8192 | 2048 | 4.0 |
| 16384 | 2048 | 8.0 |
| 32768 | 4096 | 8.0 |

### `beta_fast` / `beta_slow`

The defaults (`beta_fast=32`, `beta_slow=1`) from the paper work well in most cases. Adjust only if:

- You observe degradation on short sequences → increase `beta_fast` to protect more high-frequency dimensions
- You are extending to very long contexts (>8×) → lower `beta_slow` to interpolate more aggressively

### `mscale`

Controls how strongly the attention temperature correction is applied. Higher values increase the correction.

| Scenario | Recommended `mscale` |
|----------|-----------------------|
| Moderate extension (2–4×) | `1.0` |
| Aggressive extension (8–16×) | `0.707` |
| Fine-tuning on long sequences | `1.0` |

---

## Comparison with Other `rope_type` Options

| `rope_type` | Description | When to Use |
|-------------|-------------|-------------|
| `default` | Standard RoPE, no scaling | Training from scratch within `max_position_embeddings` |
| `linear` | Uniformly divide all frequencies by `factor` | Simple baseline for context extension |
| `dynamic` | NTK-aware scaling, recomputes `inv_freq` at runtime when sequence exceeds training length | Inference without retraining |
| `yarn` | Per-dimension ramp between linear interpolation and no scaling, plus attention temperature correction | Best quality context extension, recommended for fine-tuning |

---

## Implementation Notes

- YaRN modifies `inv_freq` once at initialization — no per-forward overhead beyond a scalar multiply on the cached cos/sin values.
- The `mscale` factor is stored as `rotary_emb.mscale` and applied inside `RotaryEmbedding.forward` when `rope_type="yarn"`.
- Both `UMichSLMAttention` and `MultiHeadLatentAttention` (MLA) read YaRN parameters from `config.rope_scaling` automatically.
- YaRN is compatible with Flash Attention 2, SDPA, and eager attention backends.
- YaRN is not applicable to PoPE (`position_embedding_type="pope"`).
