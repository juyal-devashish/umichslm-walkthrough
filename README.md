# UMichSLM — A Project Walkthrough

UMichSLM is a research-grade transformer codebase built for studying *efficient* small-language-model design. It is a single, modular implementation that lets you swap attention mechanisms, position embeddings, sparsity strategies, recurrent layers, memory modules, and prediction objectives — all behind one `UMichSLMConfig` — so that scaling-law, ablation, and architecture-search experiments stay apples-to-apples.

This README is a walkthrough. It starts with what the project *is*, takes you on a feature-by-feature tour of the architecture, and ends with how to train, evaluate, and extend it.

---

## 1. What this project is

A HuggingFace-compatible decoder-only transformer with:

- **Modern defaults** — SwiGLU, RMSNorm, pre-norm blocks, Flash Attention 2.
- **Pluggable attention** — MHA, GQA, MLA, DeltaNet (Gated / KDA), sliding-window with persistent globals, MoDA, Hymba hybrid attention-SSM, Griffin RG-LRU.
- **Pluggable positions** — RoPE, PoPE, learned RePo offsets, DRoPE training schedule.
- **Pluggable sparsity** — MoE, Progressive MoE, Mixture-of-Depths (MoD), MoDA depth-attention.
- **Pluggable objectives** — DeepSeek-style sequential MTP, Gloeckle parallel MTP, standard CE with chunked logits.
- **Pluggable memory & tokens** — Engram hash n-gram memory, learnable meta tokens / global tokens.
- **Pluggable layer reuse** — MobileLLM-LS block-wise weight sharing, OpenELM-style layer-wise scaling, cross-layer KV sharing.
- **Layer-stacked pretraining** — Progressive depth growth (4 → 8 → 16 blocks) under `attention_mode = "deltanet_mla"` or `"swa_globals_mla"`, with optimizer-state migration between stages.

Every knob lives in one `UMichSLMConfig` object, so you change one thing at a time and rerun a clean experiment. The project is wired into WandB, LLaMA Factory, TorchTitan, and SLURM workflows, and ships with RAG, agentic, and lm-eval-harness evaluation paths.

If you just want to install and run, jump to [§8 Installation](#8-installation) and [§9 Pretraining](#9-pretraining). If you want to understand the architecture, keep reading.

---

## 2. Repository tour

```
umich-slm/
├── src/umich_slm/
│   ├── configuration_umichslm.py   # The one config object that controls everything
│   ├── modeling_umichslm.py        # Top-level model + ForCausalLM head
│   ├── modules/                    # Every architectural component lives here
│   │   ├── attention.py            # MHA / GQA
│   │   ├── mla.py                  # Multi-head Latent Attention (DeepSeek V2/V3)
│   │   ├── deltanet.py             # Gated DeltaNet / Kimi Delta Attention
│   │   ├── sliding_window.py       # Local windowed attention
│   │   ├── swa_with_globals.py     # SWA + persistent global tokens
│   │   ├── global_tokens.py        # Learnable global / meta tokens
│   │   ├── moda.py                 # Mixture-of-Depths Attention
│   │   ├── hymba.py + hymba_cache  # Hymba hybrid attention-SSM head
│   │   ├── mamba2.py               # Mamba2 SSM used by Hymba
│   │   ├── griffin.py              # RG-LRU recurrent block
│   │   ├── engram.py               # Hash-based n-gram conditional memory
│   │   ├── rotary.py               # RoPE / PoPE
│   │   ├── repo.py                 # RePo learned positions
│   │   ├── swiglu.py               # SwiGLU MLP
│   │   ├── moe.py                  # Mixture-of-Experts (incl. progressive)
│   │   ├── mixture_of_depths.py    # MoD token routing
│   │   ├── multi_token_prediction.py  # DeepSeek + Gloeckle MTP
│   │   ├── rmsnorm.py              # Standard + zero-centered
│   │   ├── kv_cache.py             # Cache implementations
│   │   └── transformer_block.py    # Decoder layer assembly
│   ├── configs/presets.py          # 30+ named presets (see §4)
│   ├── callbacks/                  # e.g. DRoPECallback
│   ├── staging/                    # Layer-stacked pretraining (block specs, optim migration)
│   ├── probes/                     # Stage-transition probes (used by dress_rehearsal_8m.py)
│   ├── instrumentation/            # WandB tagging, telemetry hooks
│   ├── titan/                      # TorchTitan integration glue
│   └── utils/                      # Seeding, worker init
├── scripts/                        # Pretraining, SFT, GRPO, eval, dataset prep, SLURM gen
├── torchtitan_train/               # TorchTitan trainer package (launched via scripts/torchtitan_train.py)
├── configs/
│   ├── llama_factory/              # SFT / DPO / LoRA / GRPO / pretrain configs
│   ├── slurm/                      # Great Lakes job templates
│   └── torchtitan/                 # TorchTitan training configs (incl. stacking/)
├── evals/                          # lm-eval-harness style task configs + runners
├── chinchilla_experiments/         # Generated Chinchilla scaling sweep
├── scaling_experiments/            # Generated full scaling grid
├── tests/                          # One test file per module (35+ files)
├── docs/                           # Deeper notes
└── pyproject.toml
```

The two files you'll touch most are `configuration_umichslm.py` (every knob) and `modeling_umichslm.py` (assembly). Each architectural feature is one self-contained module under `src/umich_slm/modules/` with a matching test under `tests/`.

---

## 3. The one config object

Everything is driven from `UMichSLMConfig`. The major knob families are:

| Family                  | Flags                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------|
| Shape                   | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`          |
| Position                | `position_embedding_type`, `rope_theta`, `rope_scaling`, `use_repo`, `use_drope`          |
| Attention variant       | `use_mla`, `use_deltanet`, `use_hymba`, `use_moda`, `use_griffin`, `sliding_window`       |
| Stacking dispatch       | `attention_mode` ∈ `{legacy, deltanet_mla, swa_globals_mla}`, `hybrid_attention_ratio`    |
| Sparsity                | `use_moe`, `moe_progressive`, `use_mod`, `mod_capacity`, `mod_always_used_layers`         |
| Memory & tokens         | `use_engram`, `num_meta_tokens`, `swa_global_count`                                       |
| Layer reuse             | `use_block_weight_sharing`, `weight_sharing_block_size`, `layer_wise_scaling`             |
| Objective               | `use_mtp`, `mtp_variant` ∈ `{deepseek, gloeckle}`, `mtp_predict_offsets`                  |
| Normalization           | `use_zero_centered_rmsnorm`, `norm_placement` ∈ `{pre, post}`                             |
| Loss memory             | `lm_loss_chunk_size`, `lm_loss_chunk_cap_bytes`                                           |

`UMichSLMConfig` runs cross-validation on combinations (e.g. MLA + DeltaNet is only legal in stacking modes, MoDA forbids DeltaNet, Hymba forbids MLA, MoD ∩ MoE layers must be disjoint).

---

## 4. Presets

`get_umichslm_config("<name>")` picks from 30+ named configurations grouped into families:

| Family                   | Members                                                                                       | What's special                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------|
| Generic                  | `tiny`, `small`, `base`, `large`, `xlarge`                                                    | The original size ladder                         |
| Qwen3-like               | `qwen3_like_{tiny,xsmall,small}`                                                              | Dense GQA, RoPE/RMSNorm/SwiGLU                   |
| Qwen3.5-like (hybrid)    | `qwen3_5_like_{tiny,xsmall,small}`                                                            | Gated DeltaNet + standard attn in 3:1 ratio      |
| OpenELM-like             | `openelm_like_{tiny,xsmall,small}`                                                            | Layer-wise scaling of heads & FFN                |
| Hymba-like               | `hymba_like_{tiny,xsmall,small}`                                                              | Attention + Mamba2 SSM in parallel               |
| Baguettotron-like        | `baguette_{micro,tiny,small}`                                                                 | Deep-narrow with explicit `head_dim=64`          |
| Griffin                  | `Griffin_{tiny,xsmall,small}`                                                                 | RG-LRU recurrent + periodic attention            |
| MobileLLM (& -LS)        | `mobilellm_{tiny,xsmall,small}`, `..._ls`                                                     | Tied embeddings; `_ls` adds block weight sharing |
| MoD IsoFLOP              | `mod_{tiny,xsmall,small}`                                                                     | Depth-bumped MoD-equivalents of Qwen3 sizes      |
| Stacking pilots          | `stack_50m_b_dn`, `stack_50m_b_swa`, `stack_50m_c`, `stack_50m_wide`, `stack_100m_b`, `stack_100m_wide` | Staged 4 → 8 → 16 block pretraining       |

```python
from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, get_umichslm_config
config = get_umichslm_config("qwen3_5_like_small")   # or any name above
model = UMichSLMForCausalLM(config)
```

> Vocab sizes in the presets are placeholders. Always override `vocab_size` from the actual tokenizer at training time.

See [`docs/architecture_presets.md`](docs/architecture_presets.md) for the full preset reference with backbone parameter counts.

---

## 5. Architecture deep dive

The heart of the walkthrough. Each subsection introduces one component, explains *why* it exists, and shows the config knobs that control it.

### 5.1 Attention variants

UMichSLM supports many attention regimes. They are dispatched per-layer; you can hybridize via `hybrid_attention_ratio` (a cyclic per-block pattern) or per-feature flags.

#### Multi-Head / Grouped-Query Attention
The default. `num_key_value_heads == num_attention_heads` → MHA; smaller → GQA.

```python
UMichSLMConfig(num_attention_heads=12, num_key_value_heads=4)
```

#### Multi-head Latent Attention (MLA)
DeepSeek-V2/V3 style. Compresses KV via low-rank projections so the cache footprint shrinks dramatically at long context. Optional Q compression.

```python
UMichSLMConfig(
    use_mla=True,
    mla_kv_lora_rank=512, mla_q_lora_rank=0,
    mla_qk_nope_head_dim=128, mla_qk_rope_head_dim=64,
    mla_v_head_dim=128,
)
```

#### DeltaNet (Linear Attention)
Linear attention with delta-rule updates. Two variants:
- `"gated"` — Qwen3-Next scalar gating per head.
- `"kda"` — Kimi-K2 channel-wise gating.

You can apply it everywhere, every Nth layer, or only after the first N standard-attention layers, or explicitly via a per-block pattern.

```python
UMichSLMConfig(
    use_deltanet=True, deltanet_variant="gated",
    hybrid_attention_ratio=["delta", "delta", "delta", "attn"],
    deltanet_use_short_conv=True, deltanet_conv_kernel_size=4,
)
```

#### Sliding-Window Attention (+ global refresh)
Local windowed attention with a periodic global layer.

```python
UMichSLMConfig(sliding_window=512, global_attention_every_n_layers=4)
```

#### SWA with persistent global tokens
A hybrid stacking mode: local SWA layers see a fixed prefix of learnable global tokens, and one layer per block runs MLA over the full sequence.

```python
UMichSLMConfig(
    attention_mode="swa_globals_mla",
    hybrid_attention_ratio=["swa", "swa", "swa", "mla"],
    sliding_window=1024, swa_window=1024,
    swa_global_count=256,
    use_mla=True,
)
```

#### Hymba — parallel attention + Mamba2 SSM
Replaces each attention layer with a head that runs softmax attention and a Mamba2 SSM in parallel and fuses them. Prepends learnable meta tokens as attention sinks. Optional cross-layer KV sharing.

```python
UMichSLMConfig(
    use_hymba=True,
    hymba_num_meta_tokens=128,
    hymba_ssm_state_size=16, hymba_ssm_conv_size=4, hymba_ssm_expand=2,
    hymba_share_kv_layers=False, hymba_kv_share_group_size=2,
)
```

#### Griffin — RG-LRU + periodic attention
Replaces most layers with a real-gated linear recurrent block, keeping a sparse attention pattern.

```python
UMichSLMConfig(
    use_griffin=True,
    griffin_pattern=["rglru", "rglru", "attn"],
    griffin_num_heads=8,
)
```

#### MoDA — Mixture-of-Depths Attention
A softmax-attention variant that conditions KV on depth, optionally using FFN-produced KV. Requires `norm_placement="post"` (auto-set). Cannot be combined with DeltaNet.

```python
UMichSLMConfig(
    use_moda=True,
    moda_use_ffn_kv=True, moda_depth_kv_every_n=1,
    moda_max_depth_cache=-1,
    moda_active_layers=None,
)
```

#### Implementation backends
Each attention layer dispatches to one of three kernels with automatic fallback:

1. `flash_attention_2` — fastest, requires the `flash-attn` package.
2. `sdpa` — PyTorch's `scaled_dot_product_attention`.
3. `eager` — pure PyTorch.

```python
model = UMichSLMForCausalLM.from_pretrained(path, attn_implementation="flash_attention_2")
```

### 5.2 Position embeddings

#### RoPE (default)
Strong length extrapolation; supports linear, dynamic, and YaRN scaling via `rope_scaling={"type": "...", "factor": ...}`. See [`docs/yarn.md`](docs/yarn.md).

```python
UMichSLMConfig(position_embedding_type="rope", rope_theta=10000.0)
```

#### PoPE (Polar)
Decouples content and position in polar coordinates. Better behavior on length-extrapolation benchmarks.

```python
UMichSLMConfig(position_embedding_type="pope")
```

#### RePo — learned positions
Instead of fixed sequential indices, RePo *learns* per-head continuous position offsets from the hidden state via a SwiGLU predictor. Early layers (`< repo_start_layer`) use standard RoPE.

```python
UMichSLMConfig(
    use_repo=True, repo_start_layer=5,
    repo_dim=None,                # default: hidden_size // 8
    repo_use_base_positions=True, # stable training
)
```

#### DRoPE — Dropped RoPE training schedule
Drops position embeddings after a configurable fraction of training to improve length generalization. Wire up with `DRoPECallback` from `umich_slm.callbacks`.

```python
UMichSLMConfig(use_drope=True, drope_drop_fraction=0.875)
```

### 5.3 MLP — SwiGLU and Mixture-of-Experts

Default is SwiGLU. Sparsity scales up from there:

```python
# Standard MoE
UMichSLMConfig(
    use_moe=True,
    num_experts=8, num_experts_per_tok=2,
    num_shared_experts=0,
    moe_expert_intermediate_size=256,
    router_aux_loss_coef=0.01, router_z_loss_coef=0.001,
)

# Progressive MoE (pyramid): dense early, experts grow geometrically later
UMichSLMConfig(
    use_moe=True, moe_progressive=True,
    moe_progressive_start_ratio=0.5,
    num_experts=8,
)
```

### 5.4 Mixture-of-Depths (MoD)

A per-token compute router: some tokens skip blocks. `mod_capacity=0.125` routes only 12.5 % of tokens through the expensive path on MoD layers. You can pin specific layers as "always used" (good for boundary layers).

```python
UMichSLMConfig(
    use_mod=True,
    mod_capacity=0.125, mod_every_n_layers=2,
    mod_aux_loss_weight=0.01, mod_use_aux_predictor=False,
    mod_num_always_used_start=0, mod_num_always_used_end=0,
    # or: mod_always_used_layers=[0, 23],
)
```

### 5.5 Multi-Token Prediction (MTP)

Two variants:

- **DeepSeek-style** (`"deepseek"`) — sequential MTP chain with `mtp_num_heads` extra heads, shared or independent unembedding.
- **Gloeckle parallel** (`"gloeckle"`) — `mtp_num_heads` parallel heads with d×d adapters and independent unembedding, training-only; predicts offsets in `mtp_predict_offsets` (default `[2, 3, 4, 5]`).

```python
UMichSLMConfig(
    use_mtp=True,
    mtp_variant="deepseek",
    mtp_num_heads=2, mtp_loss_weight=0.3,
    mtp_num_layers=1, mtp_share_head=True, mtp_dropout=0.0,
)

UMichSLMConfig(
    use_mtp=True,
    mtp_variant="gloeckle",
    mtp_num_heads=4, mtp_predict_offsets=[2, 3, 4, 5],
    mtp_share_head=False,
)
```

### 5.6 Engram — hash-based n-gram memory

A non-parametric retrieval head that augments selected layers with an FNV-hash table of n-gram embeddings — cheap conditional memory with no extra attention complexity.

```python
UMichSLMConfig(
    use_engram=True,
    engram_layer_ids=[6, 12, 18],
    engram_ngram_sizes=[2, 3],
    engram_num_heads=8, engram_embed_dim=64,
    engram_table_size=49999, engram_seed=42,
    engram_use_conv=False,
)
```

### 5.7 Meta tokens / global tokens

`num_meta_tokens` prepends learnable tokens to every sequence as attention sinks. Works with all architectures; stripped from the output. Distinct from Hymba's `hymba_num_meta_tokens` and SWA's `swa_global_count` (those two are mutually exclusive).

```python
UMichSLMConfig(num_meta_tokens=16)
```

### 5.8 Layer reuse — MobileLLM-LS and OpenELM-LWS

- **Block-wise weight sharing (MobileLLM-LS):** identical block weights reused across `weight_sharing_block_size` consecutive layers, for parameter efficiency on small devices.
- **Layer-wise scaling (OpenELM):** scale `num_attention_heads` and FFN multiplier linearly with depth.
- **Cross-layer KV sharing (Hymba):** `hymba_share_kv_layers`, `hymba_kv_share_group_size`.

```python
UMichSLMConfig(
    use_block_weight_sharing=True,
    weight_sharing_block_size=2,
)
```

### 5.9 Normalization

RMSNorm everywhere. Zero-centered variant (Gemma-2) subtracts the mean. Norm placement is configurable (`"pre"` default; `"post"` is auto-set when `use_moda=True`).

```python
UMichSLMConfig(
    use_zero_centered_rmsnorm=True,
    rms_norm_eps=1e-6,
    norm_placement="pre",
)
```

### 5.10 LM loss memory budget

For large vocabularies, the next-token CE logits dominate peak memory. The forward pass chunks the sequence to keep one chunk's fp32 logits under `lm_loss_chunk_cap_bytes`.

```python
UMichSLMConfig(
    lm_loss_chunk_size=512,
    lm_loss_chunk_cap_bytes=1_200_000_000,
)
```

---

## 6. Layer-stacked pretraining

UMichSLM supports progressive depth growth — train a small model, double the blocks, copy weights, train more. Drives the `stack_50m_*` / `stack_100m_*` presets.

Wiring lives in `src/umich_slm/staging/` (block specs, optimizer-state migration) and `src/umich_slm/probes/` (transition diagnostics). `attention_mode` picks the per-block dispatch contract:

- `"legacy"` — original behavior; MLA xor DeltaNet, ratio entries in `{"delta","attn"}`.
- `"deltanet_mla"` — 3:1 DeltaNet + MLA per block; ratio in `{"delta","mla"}`.
- `"swa_globals_mla"` — 3:1 SWA-with-globals + MLA per block; ratio in `{"swa","mla"}`.

End-to-end sanity check (synthetic data, ~5 min on a laptop):

```bash
python scripts/dress_rehearsal_8m.py --attention-mode deltanet_mla
python scripts/dress_rehearsal_8m.py --attention-mode swa_globals_mla
python scripts/dress_rehearsal_8m.py --quick     # both modes
```

Production stacked runs use the TorchTitan trainer (see `configs/torchtitan/stacking/`).

---

## 7. HuggingFace and LLaMA Factory integration

UMichSLM auto-registers with `AutoConfig` / `AutoModel` / `AutoModelForCausalLM` and uses `GPT2TokenizerFast` (SmolLM2 vocab).

```python
from transformers import AutoModelForCausalLM
model.save_pretrained("./my-model")
model = AutoModelForCausalLM.from_pretrained("./my-model", trust_remote_code=True)
```

LLaMA Factory configs under `configs/llama_factory/`:

| File                                       | Purpose                              |
|--------------------------------------------|--------------------------------------|
| `umichslm_pretrain.yaml`                   | Pretraining                          |
| `umichslm_prepretrain.yaml`                | Pre-pretraining warmup               |
| `umichslm_pretrain_from_checkpoint.yaml`   | Resume from checkpoint               |
| `umichslm_sft.yaml`                        | Supervised fine-tuning               |
| `umichslm_lora.yaml`                       | LoRA fine-tuning                     |
| `umichslm_dpo.yaml` / `umichslm_dpo_80k.yaml` / `umichslm_dpo_webgpt.yaml` | DPO variants |
| `umichslm_grpo.yaml`                       | GRPO with rule-based rewards         |
| `umichslm_reasoning_warmup.yaml`           | OpenMath2 warmup                     |
| `ds_config_zero2.json`                     | DeepSpeed ZeRO-2 optimizer config    |

```bash
pip install -e . && pip install llamafactory

python scripts/llamafactory_train.py train configs/llama_factory/umichslm_sft.yaml
python scripts/llamafactory_train.py train configs/llama_factory/umichslm_lora.yaml
python scripts/llamafactory_train.py train configs/llama_factory/umichslm_dpo.yaml
python scripts/llamafactory_train.py train configs/llama_factory/umichslm_pretrain.yaml
```

---

## 8. Installation

```bash
git clone https://github.com/umich/umich-slm.git
cd umich-slm

pip install -e .                  # base
pip install -e ".[flash-attn]"    # Flash Attention 2 (CUDA)
pip install -e ".[fla]"           # flash-linear-attention (DeltaNet)
pip install -e ".[mamba]"         # mamba-ssm + causal-conv1d (Hymba)
pip install -e ".[dev]"           # tests + tooling
pip install -e ".[all]"           # everything above
```

Conda environment file (`environment.yml`) and Great Lakes setup are documented in [SETUP.md](SETUP.md).

---

## 9. Pretraining

There are several pretraining entry points; pick by data format and trainer:

| Script                              | Trainer        | Data format                                    | Typical use                                  |
|-------------------------------------|----------------|------------------------------------------------|----------------------------------------------|
| `scripts/pretrain_fineweb.py`       | LLaMA Factory  | Streamed / HF dataset (FineWeb-Edu by default) | Most ablations and scaling sweeps            |
| `scripts/pretrain.py`               | LLaMA Factory  | Generic HF dataset                             | Lighter wrapper of the above                 |
| `scripts/pretrain_packed.py`        | HF `Trainer`   | Pre-tokenized & packed Arrow shards            | Maximum throughput, no padding               |
| `scripts/torchtitan_train.py`       | TorchTitan     | TOML-configured                                | FSDP / Compile / FP8 / stacking experiments  |
| `scripts/dress_rehearsal_8m.py`     | Staging        | Synthetic                                      | 5-minute stacking pipeline smoke test        |

### Pretrain on FineWeb-Edu

```bash
# Smallest sane run
python scripts/pretrain_fineweb.py --size small --output_dir ./outputs

# Pre-prepared dataset (recommended)
python scripts/pretrain_fineweb.py --size small \
    --dataset fineweb_edu_1B_train --dataset_dir ./data/fineweb-edu-1B

# With validation
python scripts/pretrain_fineweb.py --size small \
    --do_eval --eval_dataset fineweb_edu_1B_val --eval_steps 500

# Token-target budgeting
python scripts/pretrain_fineweb.py --size small --target_tokens_b 1.0

# Multi-feature stack
python scripts/pretrain_fineweb.py --size small --use_mod --use_mtp --use_moe

# DeltaNet, RePo, Hymba, Griffin
python scripts/pretrain_fineweb.py --size small --use_deltanet --deltanet_variant kda
python scripts/pretrain_fineweb.py --size small --use_repo --repo_start_layer 5
python scripts/pretrain_fineweb.py --size small --use_hymba
python scripts/pretrain_fineweb.py --size small --use_griffin

# Custom shape (scaling experiments)
python scripts/pretrain_fineweb.py --size custom \
    --hidden_size 512 --num_hidden_layers 8 \
    --num_attention_heads 8 --intermediate_size 1376
```

### Dataset prep

```bash
# FineWeb-Edu, train/val split
python scripts/prepare_fineweb_dataset.py \
    --output_dir ./data/fineweb-edu-1B \
    --target_tokens_b 1.0 --val_percent 3.0

# Mixed corpora, optionally pre-packed
python scripts/prepare_mix_dataset.py --pack --output_dir ./data/packed
```

### TorchTitan

Twelve baseline + ablation configs ship under `configs/torchtitan/`:

```bash
# Single GPU
python scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_baseline.toml

# Multi-GPU
torchrun --standalone --nproc_per_node=4 \
    scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_mla.toml

# Stacking
python scripts/torchtitan_train.py configs/torchtitan/stacking/stack_50m_b_dn.toml
```

See [`docs/torchtitan_speed_test.md`](docs/torchtitan_speed_test.md) for throughput comparisons.

### Scaling-law sweeps

```bash
python scripts/scaling_laws.py list_models
python scripts/scaling_laws.py chinchilla --output_dir ./chinchilla_experiments
python scripts/scaling_laws.py generate   --output_dir ./scaling_experiments
python scripts/scaling_laws.py submit     --output_dir ./scaling_experiments \
    --slurm_config configs/slurm/greatlakes.yaml
python scripts/scaling_laws.py status     --output_dir ./scaling_experiments
python scripts/scaling_laws.py optimal    --compute_budgets 1e14,1e15,1e16
```

`scripts/plot_isoflop.py` produces IsoFLOP plots from the resulting `experiments_summary.json`.

### Ablation runner

```bash
python scripts/run_ablations.py --dry_run --wandb_project umichslm-ablations
python scripts/run_ablations.py --ablations position_embedding mod --wandb_project umichslm
```

### WandB tags

Every run is auto-tagged for slicing in WandB:

| Tag           | Meaning                                |
|---------------|----------------------------------------|
| `user:`       | Username                               |
| `size:`       | Preset / shape                         |
| `attention:`  | `mha`, `gqa`, `mla`, `deltanet-*`, `hymba`, `griffin`, `moda` |
| `mlp:`        | `swiglu`, `moe`, `progressive-moe`     |
| `efficiency:` | `mod`, `mtp`, `sliding-window`, `ls`   |
| `pos_emb:`    | `rope`, `pope`                         |
| `position:`   | `repo`, `drope`                        |
| `norm:`       | Normalization variant                  |

---

## 10. Post-training

| Script                  | Purpose                                                        |
|-------------------------|----------------------------------------------------------------|
| `scripts/run_sft.py`    | Supervised fine-tuning (smoltalk, openmath2, …)                |
| `scripts/run_grpo.py`   | GRPO on verifiable rewards (format + accuracy on math)         |
| LLaMA Factory configs   | DPO variants under `configs/llama_factory/umichslm_dpo*.yaml`  |

```bash
python scripts/run_sft.py --dataset smoltalk
python scripts/run_sft.py --dataset openmath2 --max_samples 200000
python scripts/run_grpo.py \
    --model_path ./outputs/umichslm-sft \
    --train_data /path/to/grpo_data/v1/gsm8k/train.jsonl \
    --val_data   /path/to/grpo_data/v1/gsm8k/val.jsonl \
    --output_dir ./outputs/umichslm-grpo-gsm8k
```

SLURM job generators are under `scripts/generate_*_slurm.py` and templates under `configs/slurm/`. See [`docs/slurm_workflows.md`](docs/slurm_workflows.md).

---

## 11. Evaluation

Three complementary harnesses.

### Pretraining-time metrics
```bash
python scripts/compute_val_loss.py --model_path ./outputs/...   # CE + perplexity
python scripts/compute_bpb.py     --model_path ./outputs/...   # bits-per-byte
```
See [`docs/evaluation.md`](docs/evaluation.md) for the metric cheatsheet.

### Downstream benchmarks (lm-eval-harness style)
Tasks under `evals/configs/general/` (MMLU, ARC-Challenge, HellaSwag, GSM8K, TruthfulQA).

```bash
python evals/scripts/run_general.py ...
bash   evals/scripts/run_lm_eval.sh ...
```

### RAG + agentic evals
Verified against `Qwen/Qwen3-0.6B`; swap in a UMichSLM checkpoint via `--ckpt_path`.

```bash
pip install transformers torch datasets sentence-transformers psutil

# Agentic (GSM8K + HotpotQA): Exact Match, Faithfulness, Multi-step Reasoning, Context Grounding
python scripts/eval_agentic.py --model Qwen/Qwen3-0.6B
python scripts/eval_agentic.py --model Qwen/Qwen3-0.6B --ckpt_path ./checkpoints/<ckpt>.pt

# RAG (MS MARCO v1.1): Exact Match, Token F1, Faithfulness, Answer Relevancy
python scripts/eval_rag_basic.py --model Qwen/Qwen3-0.6B
python scripts/eval_rag_basic.py --model Qwen/Qwen3-0.6B --ckpt_path ./checkpoints/<ckpt>.pt
```

> Bump `--n_samples` / `--n_reasoning` / `--n_rdq` above the default 50 for tighter numbers.

[Colab notebook (verified against Qwen3-0.6B)](https://colab.research.google.com/drive/1rofgH6n9Jp_4yX3rg0H6upOJcdTHFuhp?usp=sharing)

---

## 12. Tests

Every module under `src/umich_slm/modules/` has a matching test in `tests/` (35+ files covering attention variants, Hymba, Griffin, Engram, MoDA, DRoPE, stacking, optimizer migration, TorchTitan wrappers, …).

```bash
pip install -e ".[dev]"
pytest                              # everything
pytest --cov=umich_slm              # with coverage
pytest tests/test_attention.py -v   # one file
```

`scripts/smoke_compare_frameworks.py` and `scripts/throughput_compare.py` provide framework-level sanity and speed comparisons.

---

## 13. Where to go next

| Doc                                                                          | Topic                                          |
|------------------------------------------------------------------------------|------------------------------------------------|
| [`docs/architecture_presets.md`](docs/architecture_presets.md)               | Full preset reference with parameter counts    |
| [`docs/scaling_experiments.md`](docs/scaling_experiments.md)                 | Chinchilla-style scaling notes                 |
| [`docs/memory_requirements.md`](docs/memory_requirements.md)                 | GPU memory profiling                           |
| [`docs/slurm_workflows.md`](docs/slurm_workflows.md)                         | Cluster job patterns                           |
| [`docs/evaluation.md`](docs/evaluation.md)                                   | Extended eval setup + metrics                  |
| [`docs/yarn.md`](docs/yarn.md)                                               | Long-context RoPE scaling                      |
| [`docs/torchtitan_speed_test.md`](docs/torchtitan_speed_test.md)             | TorchTitan throughput comparisons              |
| [SETUP.md](SETUP.md)                                                         | Conda / Great Lakes setup                      |

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{umichslm2024,
  title = {UMichSLM: A Modern Transformer Architecture},
  year  = {2024},
  url   = {https://github.com/umich/umich-slm}
}
```

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [PoPE: Polar Coordinate Position Embeddings](https://arxiv.org/abs/2509.10534)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)
- [Multi-token Prediction (Gloeckle et al.)](https://arxiv.org/abs/2404.19737)
- [RePo: Language Models with Context Re-Positioning](https://arxiv.org/abs/2512.14391)
- [Hymba: A Hybrid-Head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676)
- [Griffin: Mixing Gated Linear Recurrences with Local Attention](https://arxiv.org/abs/2402.19427)
- [MobileLLM: Optimizing Sub-billion Parameter Language Models](https://arxiv.org/abs/2402.14905)
- [OpenELM: Efficient Language Model Family with Layer-wise Scaling](https://arxiv.org/abs/2404.14619)
- [Mamba-2: Transformers are SSMs](https://arxiv.org/abs/2405.21060)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
