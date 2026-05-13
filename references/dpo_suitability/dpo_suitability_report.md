## Run Metadata
- Timestamp (UTC): 2026-04-24T14:10:02.504186+00:00
- Sample size: 24
- Streaming mode: True
- Output directory: references/dpo_suitability
- Command (repo root): .venv/bin/python scripts/dpo_data/evaluate_dpo_suitability.py --sample-size 24 --streaming --output-dir references/dpo_suitability
# DPO Dataset Suitability Report

This report evaluates six class-selected human-preference datasets for DPO readiness.

## Criteria
1. prompt/context present
2. pair constructable
3. chosen/rejected signal
4. relevance to Reasoning-First RAG Hybrid
5. license risk/status
6. implementation effort in current repo
7. LLaMA-Factory prompt/chosen/rejected compatibility

## Summary
| Priority | Dataset | Classification | Prompt | Pair | Signal | RAG | License Risk | License Status | Effort | LLaMA-Factory Compat |
|---:|---|---|:---:|:---:|:---:|---|---|---|---|:---:|
| 1 | openbmb/UltraFeedback | high_priority_convertible | Y | Y | Y | high | requires_manual_review | unverified | low | Y |
| 2 | nvidia/HelpSteer3 | directly_dpo_ready | Y | Y | Y | medium | requires_manual_review | unverified | medium | Y |
| 3 | lmarena-ai/arena-human-preference-55k | directly_dpo_ready | Y | Y | Y | medium | requires_manual_review | unverified | medium | Y |
| 4 | openai/webgpt_comparisons | requires_transformation | Y | Y | Y | high | requires_manual_review | unverified | high | ? |
| 5 | OpenAssistant/oasst1 | weaker_fit | Y | N | N | medium | requires_manual_review | unverified | high | N |
| 6 | nvidia/HelpSteer | weaker_fit | Y | N | N | low | requires_manual_review | unverified | high | N |

## Dataset Notes
### openbmb/UltraFeedback
- Classification: high_priority_convertible
- Checked split: train
- Sample rows inspected: 24
- License: unknown
- License status: unverified
- Notes: UltraFeedback-first: existing repo prep path already supports UltraFeedback binarized conversion.

### nvidia/HelpSteer3
- Classification: directly_dpo_ready
- Checked split: train
- Sample rows inspected: 24
- License: unknown
- License status: unverified
- Notes: Contains paired responses and overall_preference; good DPO candidate after adapter conversion.

### lmarena-ai/arena-human-preference-55k
- Classification: directly_dpo_ready
- Checked split: train
- Sample rows inspected: 24
- License: unknown
- License status: unverified
- Notes: Pairwise winner format with response_a/response_b and winner flags.

### openai/webgpt_comparisons
- Classification: requires_transformation
- Checked split: parquet_fallback
- Sample rows inspected: 24
- License: unavailable
- License status: unverified
- Load issue: RuntimeError: Dataset scripts are no longer supported, but found webgpt_comparisons.py
- Notes: Normal load_dataset('openai/webgpt_comparisons') is blocked by the legacy dataset-script restriction. Parquet sampling works using the recorded path. WebGPT is conceptually strong for Reasoning-First RAG. DPO mapping is feasible with prompt=question.full_text, chosen/rejected from score_0 vs score_1, excluding ties where score_0 == score_1.
- normal load_dataset("openai/webgpt_comparisons") is blocked by the legacy dataset-script restriction
- Parquet sampling works using path: hf://datasets/openai/webgpt_comparisons@refs/convert/parquet/default/train/0000.parquet
- WebGPT is conceptually strong for Reasoning-First RAG
- DPO mapping is feasible as:
  - prompt = question.full_text
  - chosen = answer_0 if score_0 > score_1 else answer_1
  - rejected = answer_1 if score_0 > score_1 else answer_0
  - exclude ties where score_0 == score_1

### OpenAssistant/oasst1
- Classification: weaker_fit
- Checked split: train
- Sample rows inspected: 24
- License: unknown
- License status: unverified
- Notes: Tree-structured chat data with rank metadata; requires pair construction.

### nvidia/HelpSteer
- Classification: weaker_fit
- Checked split: train
- Sample rows inspected: 24
- License: unknown
- License status: unverified
- Notes: Single response with scalar scores; weak direct fit for DPO unless pairs are synthesized.

## Non-DPO Warmup Datasets (Documented, Not Classified as DPO)
- nvidia/OpenMathInstruct-2: SFT warmup for math reasoning
- sahil280114/codealpaca: SFT warmup for code instruction following
- OpenBMB/ToolBench: Tool-use SFT/evaluation warmup

## Output Artifacts
- JSON: dpo_suitability_report.json
