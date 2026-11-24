# UMichSLM

**A modular transformer codebase for studying efficient small language models.**

<!-- sync:banner -->
> **Snapshot, not the live codebase.** This is a periodically refreshed, reviewed snapshot of the project, taken from upstream commit `59b1552` (2026-11-10). Active development stays in a private repository, so recent work will not appear here.  
> Last synced: 2026-11-24
<!-- /sync:banner -->

UMichSLM is a research-grade implementation of a decoder-only language model family (around 1B parameters at the top end) built so that architecture experiments stay apples-to-apples. Attention mechanisms, position embeddings, sparsity strategies, recurrent layers, memory modules and training objectives are all selected through a single `UMichSLMConfig`. Changing one design choice never means forking the model.

The project was developed by a 15-person team at the University of Michigan.

## Highlights

| Area | What is implemented |
| --- | --- |
| Attention | MHA, GQA, Multi-head Latent Attention (MLA), Gated DeltaNet / Kimi Delta Attention, sliding window with persistent global tokens, MoDA depth attention, Hymba hybrid attention-SSM, Griffin RG-LRU |
| Positions | RoPE, PoPE, learned RePo offsets, DRoPE training schedule |
| Sparsity | Mixture-of-Experts, progressive MoE, Mixture-of-Depths |
| Objectives | Sequential and parallel multi-token prediction, chunked cross-entropy |
| Memory | Engram hash n-gram memory, learnable meta and global tokens |
| Layer reuse | Block-wise weight sharing, layer-wise scaling, cross-layer KV sharing |
| Training | Layer-stacked pretraining with optimizer-state migration, SFT, DPO, GRPO, TorchTitan and SLURM launchers |
| Evaluation | Perplexity and bits-per-byte, lm-eval-harness, RAG and agentic evaluation paths |

The model is HuggingFace-compatible and uses modern defaults: SwiGLU, RMSNorm, pre-norm blocks and Flash Attention 2.

## Start here

- **[Architecture tour](about/architecture-tour.md)**: a feature-by-feature walkthrough of the design, from configuration to training and evaluation.
- **[Setup](SETUP.md)**: environment and installation.
- **[docs/](docs/)**: architecture presets, evaluation, memory requirements, scaling experiments, SLURM workflows, TorchTitan speed tests and YaRN.
- **[Changelog](CHANGELOG.md)**: what changed in each snapshot.

## Repository layout

```
src/umich_slm/        model, configuration and every architectural module
tests/                unit tests for each component
configs/              training, SLURM and TorchTitan configurations
scripts/              pretraining, SFT, evaluation and analysis entry points
evals/                evaluation configurations and runners
torchtitan_train/     TorchTitan-based training loop
chinchilla_experiments/, scaling_experiments/   scaling-law runs and summaries
docs/                 reference documentation
```

## About this repository

This repository is a **reviewed snapshot** of the project, refreshed weekly by an automated job. It is intentionally behind the working codebase: each snapshot is taken from a commit at least two weeks old, and active development stays in a private repository. Expect the newest experiments and results to be absent.

Snapshots are curated rather than mirrored. Only an explicit allowlist of files is published, and the job stops without publishing if any safety check fails. Cluster paths, account names, email addresses and similar identifiers are replaced with placeholders such as `/path/to/data`, so some launch scripts need local paths filled in before they run. The sync tooling lives in [`sync/`](sync/README.md). Every snapshot lands as its own pull request, so any one of them can be reverted.

## License

Released under the MIT License. See [LICENSE](LICENSE).
