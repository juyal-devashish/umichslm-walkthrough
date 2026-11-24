"""Model preset configurations for UMichSLM.

This module provides preset configurations for different model sizes,
from tiny (for testing) to large (for production use).
"""

import re
from typing import Any, Dict, List


# Import lazily to avoid circular imports
def _get_config_class():
    from ..configuration_umichslm import UMichSLMConfig

    return UMichSLMConfig


# Model size presets
# Parameter counts are approximate and depend on vocab_size
#
# NOTE: vocab_size in these presets defaults to 56000 — the project's SuperBPE
# production tokenizer (umich_slm.tokenizer_defaults.DEFAULT_VOCAB_SIZE; see
# TOKENIZER_TRAINING_PLAN.md). It is still a default: at model-build time you MUST
# override vocab_size with the actual tokenizer's len(tokenizer), since added
# special tokens can push it past the nominal value. Never hard-code SmolLM2/Llama.
UMICHSLM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        # ~15M parameters (for testing and debugging)
        "hidden_size": 256,
        "intermediate_size": 688,  # ~8/3 * 256, rounded to multiple of 16
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,  # MHA (baseline)
        "max_position_embeddings": 2048,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "small": {
        # ~125M parameters
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,  # MHA (baseline) - use --num_key_value_heads 4 for GQA ablation
        "max_position_embeddings": 2048,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "base": {
        # ~350M parameters
        "hidden_size": 1024,
        "intermediate_size": 2752,  # ~8/3 * 1024, rounded
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # MHA (baseline) - use --num_key_value_heads 4 for GQA ablation
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "large": {
        # ~1.3B parameters
        "hidden_size": 2048,
        "intermediate_size": 5504,  # ~8/3 * 2048, rounded
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # MHA (baseline) - use --num_key_value_heads 8 for GQA ablation
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "xlarge": {
        # ~3B parameters
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 24,  # MHA (baseline) - use --num_key_value_heads 8 for GQA ablation
        "max_position_embeddings": 8192,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_tiny": {
        # ~5.0M backbone params / ~26.2M total with default 49k untied vocab
        "hidden_size": 216,
        "intermediate_size": 576,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_xsmall": {
        # ~19.9M backbone params / ~62.4M total with default 49k untied vocab
        "hidden_size": 432,
        "intermediate_size": 1152,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_small": {
        # ~100.7M backbone params / ~176M total with default 49k untied vocab
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_5_like_tiny": {
        # ~5.0M hybrid backbone params / ~26.2M total with default 49k untied vocab
        "hidden_size": 216,
        "intermediate_size": 576,
        "num_hidden_layers": 9,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "use_deltanet": True,
        "deltanet_variant": "gated",
        "deltanet_true_delta_rule": True,
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_5_like_xsmall": {
        # ~19.9M hybrid backbone params / ~53.0M total with default 49k untied vocab
        "hidden_size": 336,
        "intermediate_size": 896,
        "num_hidden_layers": 15,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "use_deltanet": True,
        "deltanet_variant": "gated",
        "deltanet_true_delta_rule": True,
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_5_like_small": {
        # ~99.8M hybrid backbone params / ~154.0M total with default 49k untied vocab
        "hidden_size": 552,
        "intermediate_size": 1472,
        "num_hidden_layers": 28,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "use_deltanet": True,
        "deltanet_variant": "gated",
        "deltanet_true_delta_rule": True,
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "openelm_like_tiny": {
        # ~4.6M backbone params / ~25.9M total with default 49k untied vocab
        "hidden_size": 216,
        "num_hidden_layers": 10,
        "num_key_value_heads": 2,
        "layer_wise_scaling": True,
        "head_scaling_range": [2, 6],
        "ffn_multiplier_range": [1.5, 3.0],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "openelm_like_xsmall": {
        # ~18.1M backbone params / ~60.6M total with default 49k untied vocab
        "hidden_size": 432,
        "num_hidden_layers": 10,
        "num_key_value_heads": 2,
        "layer_wise_scaling": True,
        "head_scaling_range": [2, 8],
        "ffn_multiplier_range": [1.5, 3.0],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "openelm_like_small": {
        # ~107.4M backbone params / ~182.9M total with default 49k untied vocab
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "num_key_value_heads": 4,
        "layer_wise_scaling": True,
        "head_scaling_range": [4, 12],
        "ffn_multiplier_range": [1.5, 4.0],
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "hymba_like_tiny": {
        # ~4.8M backbone params / ~26.1M total
        "hidden_size": 216,
        "intermediate_size": 576,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "share_kv_adjacent_layers": True,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "hymba_like_xsmall": {
        # ~19.3M backbone params / ~61.8M total
        "hidden_size": 432,
        "intermediate_size": 1152,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "share_kv_adjacent_layers": True,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "hymba_like_small": {
        # ~97.5M backbone params / ~173.0M total
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "share_kv_adjacent_layers": True,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # placeholder
    },
    "baguette_micro": {
        # ~4M backbone / ~47M total with 65k vocab (untied)
        "hidden_size": 128,
        "intermediate_size": 352,  # ~8/3 * 128, rounded to mult of 16
        "num_hidden_layers": 20,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,  # GQA 2:1
        "head_dim": 64,  # explicit, decoupled from hidden/heads
        "max_position_embeddings": 2048,
        "vocab_size": 56000,  # PLACEHOLDER - Baguettotron tokenizer vocab
    },
    "baguette_tiny": {
        # ~13M backbone / ~112M total with 65k vocab (untied)
        "hidden_size": 192,
        "intermediate_size": 512,  # ~8/3 * 192, rounded
        "num_hidden_layers": 30,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,  # GQA 3:1
        "head_dim": 64,
        "max_position_embeddings": 2048,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "baguette_small": {
        # ~36M backbone / ~188M total with 65k vocab (untied)
        "hidden_size": 288,
        "intermediate_size": 768,  # ~8/3 * 288, exact
        "num_hidden_layers": 40,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,  # GQA 3:1
        "head_dim": 64,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "Griffin_tiny": {
        # ~14M backbone / ~26M total with 49k vocab (untied)
        "hidden_size": 160,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 640,
        "max_position_embeddings": 512,
        "use_griffin": True,
        "griffin_pattern": ["rglru", "rglru", "attn"],
        "griffin_num_heads": 8,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "Griffin_xsmall": {
        "hidden_size": 320,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 1280,
        "max_position_embeddings": 1024,
        "use_griffin": True,
        "griffin_pattern": ["rglru", "rglru", "attn"],
        "griffin_num_heads": 8,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "Griffin_small": {
        "hidden_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 2048,
        "max_position_embeddings": 2048,
        "use_griffin": True,
        "griffin_pattern": ["rglru", "rglru", "attn"],
        "griffin_num_heads": 8,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "mobilellm_tiny": {
        # ~5.3M backbone params
        "hidden_size": 128,
        "intermediate_size": 384,
        "num_hidden_layers": 28,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,  # GQA 4:1
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "vocab_size": 32000,
    },
    "mobilellm_tiny_ls": {
        # ~5.3M backbone params, block-wise weight sharing (MobileLLM-LS)
        "hidden_size": 128,
        "intermediate_size": 384,
        "num_hidden_layers": 28,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "use_block_weight_sharing": True,
        "weight_sharing_block_size": 2,
        "vocab_size": 32000,
    },
    "mobilellm_xsmall": {
        # ~20.4M backbone params
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,  # GQA 2:1
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "vocab_size": 32000,
    },
    "mobilellm_xsmall_ls": {
        # ~20.4M backbone params, block-wise weight sharing (MobileLLM-LS)
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "use_block_weight_sharing": True,
        "weight_sharing_block_size": 2,
        "vocab_size": 32000,
    },
    "mobilellm_small": {
        # ~99M backbone params
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,  # GQA 3:1
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "vocab_size": 32000,
    },
    "mobilellm_small_ls": {
        # ~99M backbone params, block-wise weight sharing (MobileLLM-LS)
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": True,
        "use_block_weight_sharing": True,
        "weight_sharing_block_size": 2,
        "vocab_size": 32000,
    },
    "mod_tiny": {
        # IsoFLOP-equivalent to qwen3_like_tiny (~5M backbone)
        # 10 layers → 18 layers at same width ≈ 9M backbone
        # Effective FLOPs at 12.5% cap, every-other-block ≈ 5M vanilla FLOPs
        "hidden_size": 216,
        "intermediate_size": 576,
        "num_hidden_layers": 18,  # +80% depth vs vanilla (paper: add depth not width)
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "use_mod": True,
        "mod_capacity": 0.125,  # paper's optimal: 12.5% tokens per routed block
        "mod_every_n_layers": 2,  # paper's optimal: route every other block
        "mod_aux_loss_weight": 0.01,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "mod_xsmall": {
        # IsoFLOP-equivalent to qwen3_like_xsmall (~20M backbone)
        # 10 layers → 18 layers at same width ≈ 35M backbone
        "hidden_size": 432,
        "intermediate_size": 1152,
        "num_hidden_layers": 18,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "use_mod": True,
        "mod_capacity": 0.125,
        "mod_every_n_layers": 2,
        "mod_aux_loss_weight": 0.01,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "mod_small": {
        # IsoFLOP-equivalent to qwen3_like_small (~100M backbone)
        # 16 layers → 28 layers at same width ≈ 176M backbone
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "use_mod": True,
        "mod_capacity": 0.125,
        "mod_every_n_layers": 2,
        "mod_aux_loss_weight": 0.01,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    # ====================================================================
    # Delphi-style width-scaling presets (openathena.ai/blog/delphi).
    # Depth follows delphi_arch(H) = round( H / (64 + 4·log₂H − 9) ),
    # n_heads = H/128, head_dim = 128, GQA 4:1 (n_kv = max(1, n_heads/4)),
    # MLP ratio 4. Depths are nudged ±1 from the strict formula where needed
    # to hit the labeled total-backbone-param target within 10 %.
    # ====================================================================
    "delphi_10m": {
        # ~10.8M backbone params / ~30M total with default 49k untied vocab
        "hidden_size": 384,
        "intermediate_size": 1536,  # MLP ratio 4
        "num_hidden_layers": 5,  # strict formula gives 4; bumped to hit 10M
        "num_attention_heads": 3,
        "num_key_value_heads": 1,  # GQA 3:1 (head_dim=128)
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER
    },
    "delphi_25m": {
        # ~22.8M backbone params / ~48M total with default 49k untied vocab
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
    },
    "delphi_50m": {
        # ~47.4M backbone params / ~80M total with default 49k untied vocab
        "hidden_size": 640,
        "intermediate_size": 2560,
        "num_hidden_layers": 8,  # strict formula ≈ 7
        "num_attention_heads": 5,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
    },
    "delphi_100m": {
        # ~104.8M backbone params / ~150M total with default 49k untied vocab
        "hidden_size": 896,
        "intermediate_size": 3584,
        "num_hidden_layers": 9,  # strict formula ≈ 9-10
        "num_attention_heads": 7,
        "num_key_value_heads": 2,  # GQA ~4:1
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
    },
    # ====================================================================
    # Layer-stacked pretraining pilots (see STACKING_IMPLEMENTATION_PLAN.md
    # and STACKING_IMPLEMENTATION_PLAN_DELTA.md). All target TPR ~350.
    # Stage 1/2/3 token allocation is 5/25/70 of total budget.
    # tie_word_embeddings=False; Gloeckle parallel MTP with independent heads.
    # ====================================================================
    "stack_50m_b_dn": {
        # 50M Config B (deep, DeltaNet+MLA). Stages 4 -> 8 -> 16 blocks
        # = 16 -> 32 -> 64 layers. ~52M total params at d=256.
        "hidden_size": 256,
        "intermediate_size": 688,  # ~8/3 d, mult of 16
        "num_hidden_layers": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,  # PLACEHOLDER
        "tie_word_embeddings": False,
        "use_mla": True,
        "use_deltanet": True,
        "deltanet_variant": "gated",
        "attention_mode": "deltanet_mla",
        "hybrid_attention_ratio": ["delta", "delta", "delta", "mla"],
        "mla_kv_lora_rank": 128,  # d/2 at d<768 per plan §2.1
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mod": True,
        "mod_capacity": 0.5,
        "mod_capacity_inserted": 0.5,
        "mod_every_n_layers": 4,  # MoD on the MLA layer of each block
        "mod_aux_loss_weight": 0.01,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
    "stack_50m_b_swa": {
        # 50M Config B (deep, SWA+globals+MLA). Same shape as stack_50m_b_dn.
        "hidden_size": 256,
        "intermediate_size": 688,
        "num_hidden_layers": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,  # 2:1 GQA
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
        "tie_word_embeddings": False,
        "use_mla": True,
        "attention_mode": "swa_globals_mla",
        "hybrid_attention_ratio": ["swa", "swa", "swa", "mla"],
        "sliding_window": 1024,
        "swa_window": 1024,
        "swa_global_count": 256,
        "mla_kv_lora_rank": 128,
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mod": True,
        "mod_capacity": 0.5,
        "mod_capacity_inserted": 0.5,
        "mod_atomic_block": True,  # block-atomic gather/scatter MoD on inserted blocks
        "mod_block_size": 4,
        "mod_aux_loss_weight": 0.01,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
    "stack_50m_c": {
        # 50M Config C (balanced, SWA+globals+MLA). Stages 3 -> 6 -> 12
        # blocks = 12 -> 24 -> 48 layers (full doubling per stage).
        # ~62M trunk params at d=320.
        "hidden_size": 320,
        "intermediate_size": 864,
        "num_hidden_layers": 48,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,  # GQA 5:1 (head_dim 64 -> 320/5)
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
        "tie_word_embeddings": False,
        "use_mla": True,
        "attention_mode": "swa_globals_mla",
        "hybrid_attention_ratio": ["swa", "swa", "swa", "mla"],
        "sliding_window": 1024,
        "swa_window": 1024,
        "swa_global_count": 256,
        "mla_kv_lora_rank": 160,
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mod": True,
        "mod_capacity": 0.5,
        "mod_capacity_inserted": 0.5,
        "mod_atomic_block": True,  # block-atomic gather/scatter MoD on inserted blocks
        "mod_block_size": 4,
        "mod_aux_loss_weight": 0.01,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
    "stack_50m_wide": {
        # 50M wide control (flat training, SWA+globals+MLA).
        # 10 layers at d=512. ~50M params. No stacking.
        "hidden_size": 512,
        "intermediate_size": 1376,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
        "tie_word_embeddings": False,
        "use_mla": True,
        "attention_mode": "swa_globals_mla",
        "hybrid_attention_ratio": ["swa", "swa", "swa", "mla"],
        "sliding_window": 1024,
        "swa_window": 1024,
        "swa_global_count": 256,
        "mla_kv_lora_rank": 256,  # d/2
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
    "stack_100m_b": {
        # 100M Config B (production decision pilot). Attention mode (DeltaNet vs SWA)
        # is decided by the 50M comparison; presets default to deltanet_mla, override
        # via TOML/YAML to switch to swa_globals_mla.
        # Stages 5 -> 10 -> 20 blocks = 20 -> 40 -> 80 layers. ~95M params at d=320.
        "hidden_size": 320,
        "intermediate_size": 864,
        "num_hidden_layers": 80,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
        "tie_word_embeddings": False,
        "use_mla": True,
        "use_deltanet": True,
        "deltanet_variant": "gated",
        "attention_mode": "deltanet_mla",
        "hybrid_attention_ratio": ["delta", "delta", "delta", "mla"],
        "mla_kv_lora_rank": 160,
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mod": True,
        "mod_capacity": 0.5,
        "mod_capacity_inserted": 0.5,
        # deltanet_mla default: per-layer MoD (gather/scatter can't skip the
        # recurrent delta layers). Switch mod_atomic_block on only if this preset
        # is overridden to swa_globals_mla via TOML/YAML.
        "mod_every_n_layers": 4,
        "mod_aux_loss_weight": 0.01,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
    "stack_100m_wide": {
        # 100M wide control. 16 layers at d=640. ~93M params. No stacking.
        "hidden_size": 640,
        "intermediate_size": 1728,
        "num_hidden_layers": 16,
        "num_attention_heads": 10,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "vocab_size": 56000,
        "tie_word_embeddings": False,
        "use_mla": True,
        "attention_mode": "swa_globals_mla",
        "hybrid_attention_ratio": ["swa", "swa", "swa", "mla"],
        "sliding_window": 1024,
        "swa_window": 1024,
        "swa_global_count": 256,
        "mla_kv_lora_rank": 320,  # d/2
        "mla_q_lora_rank": 0,
        "mla_qk_nope_head_dim": 32,
        "mla_qk_rope_head_dim": 32,
        "mla_v_head_dim": 64,
        "use_mtp": True,
        "mtp_variant": "gloeckle",
        "mtp_num_heads": 4,
        "mtp_loss_weight": 0.3,
        "mtp_predict_offsets": [2, 3, 4, 5],
        "mtp_share_head": False,
    },
}


# ========================================================================
# Cross-architecture scaling-sweep presets (flat / constant-depth).
#
# Every architecture shares the SAME backbone shape at a given size (the Delphi
# width-scaling ladder), differing ONLY in architecture-specific flags. This
# makes the cross-architecture L(N, D) comparison clean: matched N per size, with
# the architecture as the only variable. Names are ``<arch>_<size>`` for
# size in {10m, 25m, 50m, 200m}; the Delphi sweep maps to them directly
# (scripts/scaling_laws.py::_delphi_preset_name). vocab_size = 56000 (SuperBPE).
#
# MTP is intentionally excluded here (it is a training-signal add-on applied at
# production, not part of the backbone param count being scaled).
# ========================================================================

_SWEEP_BASE_SHAPES: Dict[str, Dict[str, Any]] = {
    "10m": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 5,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 640,
        "intermediate_size": 2560,
        "num_hidden_layers": 8,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,
    },
    # RENAMED 2026-08-20: this rung was called "100m" until it was measured.
    # The old label came from delphi_100m, which named ~104.8M *backbone* params
    # at the then-default 49k vocab. Under the sweep's actual settings -- vocab
    # 56000, UNTIED -- the embeddings alone are 2 x 56000 x 896 = 100.4M, so the
    # dense anchor is 203.6M total and the label was off by 2x. It is now "200m".
    # "<arch>_100m" still resolves (deprecated alias registered below) so that
    # existing result dirs under models/arch_mfu/100m stay readable.
    #
    # NOTE this is exactly deep80's width (h=896, ffn=3584, 7 heads, kv=1,
    # head_dim=128) at 9 layers instead of 80, which is why it is the cheap
    # proxy rung for the deep bench -- and why anything that scales with DEPTH
    # (per-layer routing overhead, per-carried-layer bank reads) hides here.
    "200m": {
        "hidden_size": 896,
        "intermediate_size": 3584,
        "num_hidden_layers": 9,
        "num_attention_heads": 7,
        # kv=1: 7 heads is prime, so kv=2 cannot divide it. The old kv=2 built
        # fine on CPU and crashed at the first GPU attention call (repeat_kv
        # yields 6 heads vs Hq=7) — the 100m rung had simply never been run.
        # See test_gqa_heads_divide.
        "num_key_value_heads": 1,
    },
    # Matched-EFFECTIVE-depth tiers for the deep (72-128 layer) 1B efficiency
    # study (2026-08-19). The size label names the layer-equivalents/token
    # target, NOT params: at width 1024 every arch in a tier spends ~the same
    # per-token block FLOPs and only the mechanism providing the depth differs.
    #   eff54: dense L=54 == mopd L=48 + 3 stages x budget 2.0 == scribe-MoD
    #          L=96 @ capacity 0.5 (eff 8 + 0.5*4*22 = 52; exact 54 via
    #          --set mod_capacity=0.523).
    #   eff78: dense L=78 == mopd L=72+6 == scribe-MoD L=128 @ capacity 0.583
    #          (design 0.5 gives eff 68).
    # Per-arch depths live in _SWEEP_ARCH_SHAPES; this base entry is the dense
    # control's shape. Width 1024/heads 8 keeps head_dim at the 128 fast-kernel
    # limit (sm80/sm86 rule; qwen3_like sits at 128 everywhere).
    "eff54": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 54,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "eff78": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 78,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },  # ASPECT-RATIO TIERS (ar10m/ar25m/ar50m/ar100m, 2026-08-20). The screening
    # tiers above are shallow-and-wide relative to the deep target: at 25m the
    # bank family is L=12/h=336 (L/h = 0.036) and the dense anchor L=6/h=512
    # (0.012), against deep80's 0.089 and deep128's 0.167. A sweep run on those
    # shapes measures the right ARMS on the wrong aspect ratio, and depth-vs-
    # width is not neutral for this project -- the whole premise is that depth
    # buys reasoning quality, so a 12-layer proxy cannot show what an 80-layer
    # model does.
    #
    # These tiers hold NON-EMBEDDING params to the same band as the existing
    # rungs (within 1%) while putting L/h at 0.106-0.133, i.e. bracketing the
    # deep80/deep128 target range, at a clean head_dim of 64. Use them for loss
    # curves that have to extrapolate to the deep target; the original tiers
    # stay for continuity with cells already on disk, and the pair of them is
    # itself the wide-vs-deep experiment.
    # ar18m extends the N lever DOWNWARD, added 2026-10-03 to identify alpha.
    # The grid spanned N_active over only 2.62x (33.9-88.9M), which cannot pin
    # alpha: forcing E across 0.0-1.4 moves alpha 0.18-0.51 with RMSE flat to
    # 0.0007 nats (5% of the seed noise floor). Cost scales as N^2, so a rung
    # BELOW ar10m buys the same log-range as one above it for ~1% of the compute.
    #
    # 18.3M is the floor worth having, not a free choice: embeddings are untied,
    # so the two tables cost 2*56000*h and are already 63% of N_active at ar10m.
    # At h=96 they are 85% and the body is 1.9M -- a model that would sit off the
    # common curve and BIAS alpha rather than sharpen it. h=128 keeps head_dim at
    # the family's 64 (heads = h/64) and L/h at 0.125, inside the 0.106-0.133
    # band the other tiers occupy; L=18 would be 0.141, i.e. off the shape trend.
    "ar18m": {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_hidden_layers": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    },
    "ar10m": {
        "hidden_size": 192,
        "intermediate_size": 768,
        "num_hidden_layers": 23,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
    },
    "ar25m": {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 27,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "ar50m": {
        "hidden_size": 320,
        "intermediate_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,
    },
    "ar100m": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 51,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    # DEEP TIERS (2026-08-20, user requirement): the target model is deep --
    # ~80 layers in the Baguettotron mould, with a per-token compute scalar so
    # most tokens traverse thin/cheap layers and hard tokens go deep. Both are
    # sized to ~0.9-1.0B total at vocab 56000 and keep head_dim = 128.
    #   deep80  L=80  h=896  -> 0.871B   (the ~80-layer reference point)
    #   deep128 L=128 h=768  -> 0.992B   (very deep + thin; the mat_ffn shape,
    #                                     where "thin for most tokens" comes
    #                                     from the nested FFN width, not depth)
    "deep80": {
        "hidden_size": 896,
        "intermediate_size": 3584,
        "num_hidden_layers": 80,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
    },
    "deep128": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 128,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    # deep128's per-layer shape (h=768, head_dim 128, kv=2) at 48 and 104 layers:
    # 0.50B and 0.99B dense at vocab 56000 (deep128 itself measures 1.19B there).
    # The 500M / 1B production-scale bench points (2026-10-26); same kernels at
    # both, so the pair isolates depth.
    "deep48": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 48,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    "deep104": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 104,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    # Deep-narrow tier for the count-vs-share ratio screen (PRETRAIN_PLAN.md §0.4
    # P2, 2026-10-26): ar25m's width (h=256, head_dim 64) at 64 layers, L/h = 0.25,
    # i.e. Config B's aspect ratio (176/768 = 0.23). ~90M dense at vocab 56000.
    "w256l64": {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    # PRETRAIN_PLAN.md §2.4 / §2.5 production shapes, for the Phase 2 throughput bench
    # (2026-10-31): Config B d=768 x 176 (12 heads, FFN 2048) and Config C d=896 x 128
    # (14 heads, FFN 2400), head_dim 64 as the plan specifies. MLA's latent stays at
    # each arm's own rule (h/2), not the plan's d/4.
    "cfgB": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 176,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    },
    "cfgC": {
        "hidden_size": 896,
        "intermediate_size": 2400,
        "num_hidden_layers": 128,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    },
    # P9 SHAPE PILOT (FABLE review 2026-11-04): three deltanet_mla shapes at matched
    # TOTAL N (~266-271M at vocab 56000 untied, within 1.8% of each other), head_dim 64,
    # kv=1, FFN 4h. p270b is Config B's aspect (L/h 0.23), p270c Config C's (0.14),
    # p270w a conventional shallow-wide model (0.023). Meta-device N, deltanet_mla /
    # qwen3_like: p270b 265,727,232 / 229,067,136; p270c 270,617,536 / 233,734,592;
    # p270w 269,476,224 / 236,449,536 (p270w at L=16 / 24 would be 247.9M / 328.9M).
    "p270b": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 88,
        "num_attention_heads": 6,
        "num_key_value_heads": 1,
    },
    "p270c": {
        "hidden_size": 448,
        "intermediate_size": 1792,
        "num_hidden_layers": 64,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
    },
    "p270w": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 18,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
    },
}


# Parents that take a "<parent>_rec" recurrent-depth twin (see _sweep_arch_flags).
_REC_PARENTS = ("qwen3_like", "swa_sinkdiet_gqa")
# Loop-mixer suffixes on top of "_rec" (modules/loop_mixers.py): Parcae stable
# injection, Hyperloop loop hyper-connections, and both.
_REC_MIXERS = {
    "": {},
    "_inject": {"recurrent_stable_injection": True},
    "_hc": {"recurrent_hyper_connections": True},
    "_inject_hc": {"recurrent_stable_injection": True, "recurrent_hyper_connections": True},
}


def _split_rec_arch(arch: str):
    """'<parent>_rec<mixer>' -> (parent, mixer flags), else None."""
    for _mix, _flags in _REC_MIXERS.items():
        _tail = "_rec" + _mix
        if arch.endswith(_tail) and arch[: -len(_tail)] in _REC_PARENTS:
            return arch[: -len(_tail)], _flags
    return None


def _ratio_final_global(pattern: List[str], num_layers: int, global_kind: str) -> List[str]:
    """Expand a repeating layer pattern to ``num_layers``, forcing the last global.

    A bare 4-cycle like ``["swa","swa","swa","mla"]`` lands a *local* layer last
    whenever ``num_layers % 4 != 0`` (the sweep depths 5, 6 and 9 all do). That
    layer feeds the LM head, so with a 1024-token window over a 2048-token
    sequence the final representation could never see beyond the window. Gemma 4
    states the invariant explicitly — "the final layer is always global" — so we
    expand the pattern to full depth and overwrite the tail.
    """
    out = [pattern[i % len(pattern)] for i in range(num_layers)]
    if out:
        out[-1] = global_kind
    return out


# --------------------------------------------------------- mat_ffn nest arms
# W1 nesting round (2026-08-22, docs/mat_ffn_nesting.md). Each arm is a
# (menu ratios, tier prior) pair solved so EXPECTED ACTIVE params match
# `qwen3_like` at the same size -- total params run far over, by design: the
# whole premise is a big nest whose typical token touches a small prefix of it.
#
# Read the table as an experiment, not four variations. All four hold the
# expected routed width E[w] within ~1% of each other (ar10m: 656-662), so they
# cost the same per token and carry the same active params; what varies is HOW
# DEEP THE NEST IS (d_ff 1056 -> 2656 at ar10m, a 2.5x span) and whether the
# prior is skewed toward the cheap tiers. That is the question the round exists
# to answer: does a rarely-used wide tier buy anything at fixed expected cost?
#
#   mat_ffn_flat4   uniform prior over quarters. The CONTROL: it carries the
#                   fixed-width decoupling, the warmup and the KL aux, so a
#                   flat4-vs-skew comparison isolates the SKEW itself rather
#                   than the new machinery.
#   mat_ffn_skew3   3 tiers {1/4, 1/2, 1}, prior 0.50/0.30/0.20.
#   mat_ffn_skew4   4 halvings {1/8 ... 1}, prior 0.60/0.22/0.12/0.06. The
#                   deepest nest; d_ff is 3.5x the dense anchor's.
#   mat_ffn_skew4_nowarm  skew4 with the curriculum OFF. Prices the warmup.
#   mat_ffn_skew4_explore  skew4 plus a PERMANENT exploration floor
#                   (`mat_ffn_explore_frac=0.1`). Identical geometry, prior,
#                   warmup and aux to skew4 -- a single-variable change.
#
#                   WHY IT EXISTS (2026-08-22). Every arm above collapses to the
#                   narrowest tier and cannot recover: tier k's only LM gradient
#                   reaches the router through the straight-through gate
#                   `tail[:, k]`, which both dispatch paths multiply by the hard
#                   mask `(choice >= k)`, so a tier nobody selects contributes
#                   exactly zero gradient and zero occupancy is an ABSORBING
#                   STATE. Measured on the live skew4 ar10m cell at step 6246:
#                   the widest tier is at EXACTLY 0.0000 occupancy in all five
#                   routed layers, trailing the argmax by 3.0-8.0 logits, and
#                   E[w] is 390-447 against a target of 768 -- i.e. the arm pays
#                   3072-wide GEMMs (the tier slices are sized from the PRIOR,
#                   not from demand) and delivers ~52-58% of the width it is
#                   priced at. The KL aux, the only escape, contributes 2.5e-4
#                   to 3.8e-3 nats against a 0.036-nat seed-noise bar.
#                   `scripts/diagnose_mat_ffn_router_collapse.py` measures this.
#
# `mat_ffn` itself is NOT in this family and is untouched -- it has a cell in
# flight (55097) and a landed bench series.
#
# WHY dyn_start/dyn_end MOVE. `mat_ffn` uses an absolute 8/8, which at deep80
# is 10% of depth but at ar10m (L=23, clamped to 7) is 61% of all layers -- so
# the screening proxy would spend most of its depth on fixed-compute layers and
# could not show what the arm does at the target shape. These arms use 10% of
# depth at every size, which reproduces deep80's 20/20/60 routed/fixed/budget
# split. At deep80 the rule returns 8, i.e. the same value `mat_ffn` bakes.
#
# SOLVED d_ff PER SIZE. Only tabulated sizes are matched-N (verified against
# the live accounting path, `num_active_params`, not a closed form). Other sizes
# fall back to a proportional derivation so the arch stays constructible for
# tests and benches -- but do NOT quote a matched-N claim for them.
_MAT_FFN_NEST: Dict[str, Dict[str, Any]] = {
    "mat_ffn_flat4": {
        "ratios": [0.25, 0.50, 0.75, 1.0],
        "prior": [0.25, 0.25, 0.25, 0.25],
        "warmup": 0.15,
    },
    "mat_ffn_skew3": {
        "ratios": [0.25, 0.50, 1.0],
        "prior": [0.50, 0.30, 0.20],
        "warmup": 0.15,
    },
    "mat_ffn_skew4": {
        "ratios": [0.125, 0.25, 0.50, 1.0],
        "prior": [0.60, 0.22, 0.12, 0.06],
        "warmup": 0.15,
    },
    "mat_ffn_skew4_nowarm": {
        "ratios": [0.125, 0.25, 0.50, 1.0],
        "prior": [0.60, 0.22, 0.12, 0.06],
        "warmup": 0.0,
    },
    "mat_ffn_flat4_explore": {
        "ratios": [0.25, 0.50, 0.75, 1.0],
        "prior": [0.25, 0.25, 0.25, 0.25],
        "warmup": 0.15,
        "explore": 0.10,
    },
    "mat_ffn_skew3_explore": {
        "ratios": [0.25, 0.50, 1.0],
        "prior": [0.50, 0.30, 0.20],
        "warmup": 0.15,
        "explore": 0.10,
    },
    "mat_ffn_skew4_explore": {
        "ratios": [0.125, 0.25, 0.50, 1.0],
        "prior": [0.60, 0.22, 0.12, 0.06],
        "warmup": 0.15,
        # eps=0.1 floors tier 3 at 0.1*0.06 = 0.6% of tokens (~200/layer/step at
        # ar10m's 32768-token batch) -- ample gradient, and small enough that
        # occupancy ABOVE the floor is readable as genuine router preference.
        "explore": 0.10,
    },
    # THRESHOLD ROUTING (2026-08-24). Identical geometry to mat_ffn_skew4_explore
    # -- same ratios, same prior, same budget -- so this is a pure mechanism A/B.
    #
    # Two changes. `route_mode="threshold"` replaces argmax with
    # `tier >= k iff tail_k > theta_k`, theta adapted by a controller toward
    # REALIZED (post-clip) occupancy. `_causal_budget_clip` only ever DEMOTES, so
    # nothing walked a collapsed router back up; the controller supplies the
    # missing lower bound and collapse becomes structurally impossible.
    #
    # And explore=0.0: the eps floor is redundant here. It never fixed anything
    # anyway -- eval mean width was 394 at eps=0.4 vs 388 at eps=0.1 vs 388 at
    # baseline. It manufactures TRAIN occupancy the router has not learned.
    #
    # Measured at the tiny testbed (L=4, TPR ~91, 12k steps), eval mean width
    # against a 768 budget: threshold 767 (occupancy within 0.0004 of target on
    # every tier, held flat from step 3000), mixture 639, baseline 394.
    "mat_ffn_thresh": {
        "ratios": [0.125, 0.25, 0.50, 1.0],
        "prior": [0.60, 0.22, 0.12, 0.06],
        "warmup": 0.15,
        "explore": 0.0,
        "route_mode": "threshold",
        "route_thresh_lr": 0.01,
    },
    # NEST GEOMETRY RE-TEST (2026-10-19). `mat_ffn_thresh` inherited skew4's
    # geometry because skew4 was the arm being fixed, not because it won: the
    # flat4 / skew3 / skew4 round was read under the argmax router, which
    # collapsed to tier 0 in every arm, so it measured three dead routers and
    # says nothing about which nest a WORKING router should get. This is flat4's
    # geometry and uniform prior under the threshold router + "wish" controller,
    # everything else identical to mat_ffn_thresh -- read `_gqa` against
    # `mat_ffn_thresh_gqa` (same E[w] = 768 and active params at ar10m; d_ff
    # 1216 vs 3072, so total N is ~5.7M smaller).
    "mat_ffn_thresh_flat4": {
        "ratios": [0.25, 0.50, 0.75, 1.0],
        "prior": [0.25, 0.25, 0.25, 0.25],
        "warmup": 0.15,
        "explore": 0.0,
        "route_mode": "threshold",
        "route_thresh_lr": 0.01,
    },
}

# {arch: {L: (d_ff, [widths], fixed_width)}} -- keyed on depth because that is
# what identifies the size tier inside `_sweep_arch_flags` (it receives the
# shape, not the size name).
_MAT_FFN_NEST_SOLVED: Dict[str, Dict[int, Any]] = {
    # ar10m (L=23, h=192). ANCHORED ON `swa_sinkdiet`, active = 35,411,622.
    #
    # WHY NOT qwen3_like (changed 2026-08-22). These arms ARE swa_sinkdiet plus
    # the nested FFN -- the config diff is nothing but `mat_ffn_*` knobs, while
    # against qwen3_like there are NINE architectural differences before the FFN
    # (legacy GQA vs SWA+MLA, no logit sinks, no registers, no flex attention,
    # ...). Anchoring on qwen3_like also cannot be done honestly: the sinkdiet
    # backbone costs +1,463,142 active params over qwen3_like at the SAME shape,
    # so matching active params to qwen3_like forces E[w] down to ~656 -- 14%
    # BELOW the anchor's own d_ff of 768. The nest arm would then be doing less
    # FFN work per token and paying for it in bank params, and a win could be
    # either. Anchored on sinkdiet, E[w] lands at 752-766 against its d_ff of
    # 768 and `fixed_width` comes out at exactly 768, so the fixed prefix/suffix
    # layers are IDENTICAL to sinkdiet's: same backbone, same expected FFN work,
    # fixed width vs routed nest, and nothing else.
    #
    # deep80 (L=80, h=896, anchor d_ff 3584): swa_sinkdiet active = 1,056,890,184.
    # This is the production shape and the one the wall-clock gate runs on.
    # Note skew4 there: d_ff = 14336, FOUR TIMES the anchor's FFN, with the
    # widest tier alone as wide as 4 dense FFNs -- 1.52B total params for 1.06B
    # active. That is the design at full strength, and every increment
    # (1792/1792/3584/7168) sits well above the measured A100 GEMM plateau
    # (~896-1024), which is finding 5a of docs/mat_ffn_nesting.md holding at the
    # production shape: skewing the prior pushes d_ff up faster than the tier
    # fraction comes down, so the thin-GEMM problem a 12.5% first tier would
    # otherwise cause simply does not arise.
    #
    #                       d_ff  widths                          fixed_w  E[w]   active
    "mat_ffn_flat4": {
        23: (1216, [320, 640, 896, 1216], 768),  #    768  +0.01%
        80: (5760, [1408, 2880, 4352, 5760], 3584),  #   3600  +0.27%
    },
    "mat_ffn_skew3": {
        23: (1664, [384, 832, 1664], 768),  #    774  +0.24%
        80: (7552, [1920, 3776, 7552], 3584),  #   3603  +0.28%
    },
    "mat_ffn_skew4": {
        23: (3072, [384, 768, 1536, 3072], 768),  #    768  +0.01%
        80: (14336, [1792, 3584, 7168, 14336], 3584),  #   3584  +0.01%
    },
    "mat_ffn_skew4_nowarm": {
        23: (3072, [384, 768, 1536, 3072], 768),
        80: (14336, [1792, 3584, 7168, 14336], 3584),
    },
    # Each *_explore entry is byte-identical to its un-floored twin: the ONLY
    # difference is `mat_ffn_explore_frac`, so N, E[w] and wall-clock all match.
    "mat_ffn_flat4_explore": {
        23: (1216, [320, 640, 896, 1216], 768),
        80: (5760, [1408, 2880, 4352, 5760], 3584),
    },
    "mat_ffn_skew3_explore": {
        23: (1664, [384, 832, 1664], 768),
        80: (7552, [1920, 3776, 7552], 3584),
    },
    "mat_ffn_skew4_explore": {
        23: (3072, [384, 768, 1536, 3072], 768),
        80: (14336, [1792, 3584, 7168, 14336], 3584),
    },
    # Same solved geometry as skew4_explore -- matched N, so the A/B isolates
    # the routing MECHANISM and nothing else.
    "mat_ffn_thresh": {
        23: (3072, [384, 768, 1536, 3072], 768),
        80: (14336, [1792, 3584, 7168, 14336], 3584),
    },
    # flat4's solved geometry, so it is matched-N against mat_ffn_thresh exactly
    # as mat_ffn_flat4 was against mat_ffn_skew4.
    "mat_ffn_thresh_flat4": {
        23: (1216, [320, 640, 896, 1216], 768),
        80: (5760, [1408, 2880, 4352, 5760], 3584),
    },
}


def _mat_ffn_nest_flags(arch: str, base: Dict[str, Any]) -> Dict[str, Any]:
    # A "_gqa" suffix is a single-knob delta on ANY nest arm: identical nest
    # geometry, but the 6 periodic global layers become GQA instead of MLA.
    # The parity bench (2026-08-26, job 55748) decomposed the sinkdiet-family
    # wall-clock gap and found it is MLA PROJECTION cost, not attention span:
    # gwin8 (band the globals to 8192) moved nothing at 32K, while gqa (same
    # span, no MLA) took swa_sinkdiet from 1.202x dense to 1.037x. Every
    # mat_ffn arm carries the same 17 SWA + 6 MLA stack, so the lever ports
    # verbatim. N drops ~3.6% (GQA is cheaper than MLA), so these are NOT
    # matched-N against their parents -- they need their own BUDGET_N entry.
    gqa = arch.endswith("_gqa") and arch[: -len("_gqa")] in _MAT_FFN_NEST
    if gqa:
        arch = arch[: -len("_gqa")]
    # "_delta" (2026-10-20): the same nest on the 3:1 Gated DeltaNet / GQA hybrid
    # (qwen3_5_like's layout) instead of the SWA+sink+register substrate. Read it
    # against qwen3_5_like for the nest and against mat_ffn_thresh_gqa for the
    # sequence mixer. Not matched-N with either: needs its own BUDGET_N entry.
    delta = arch.endswith("_delta") and arch[: -len("_delta")] in _MAT_FFN_NEST
    if delta:
        arch = arch[: -len("_delta")]
    # "_delta_mla" (2026-10-22): the same nest on deltanet_mla's 3:1 Gated DeltaNet /
    # MLA stack. Read it against deltanet_mla for the nest and against
    # mat_ffn_thresh_delta for the attention kind (a 2x2 at ar18m, where every
    # budget is the dense anchor).
    delta_mla = arch.endswith("_delta_mla") and arch[: -len("_delta_mla")] in _MAT_FFN_NEST
    if delta_mla:
        arch = arch[: -len("_delta_mla")]
    # "_delta_gqa" (2026-10-27): swap the NON-recurrent layer only. The 3:1 Gated
    # DeltaNet / GQA hybrid (qwen3_5_like layout) whose 1-in-4 attention slots become
    # mat_ffn_thresh_gqa's kind of layer: full-attention GQA + the routed nest + a
    # learned logit sink + a read of a 128-slot register bank (swa_sinkdiet's bank
    # size). The DeltaNet layers keep a plain backbone-width SwiGLU
    # (mat_ffn_layer_kinds=["attn"]). Read against deltanet_mla / qwen3_5_like at
    # matched D (gen_a40_grid anchors it on dense, like the plain hybrids).
    delta_gqa = arch.endswith("_delta_gqa") and arch[: -len("_delta_gqa")] in _MAT_FFN_NEST
    if delta_gqa:
        arch = arch[: -len("_delta_gqa")]
    spec = _MAT_FFN_NEST[arch]
    L = base["num_hidden_layers"]
    dyn = max(1, round(0.10 * L))
    solved = _MAT_FFN_NEST_SOLVED.get(arch, {}).get(L)
    if solved is not None:
        dff, widths, fixed_w = solved
    else:
        # Fallback: hold the expected width at the backbone's d_ff and let the
        # nest grow around it. Constructible, NOT matched-N -- solve and
        # tabulate before running a cell at a new size.
        m = sum(p * r for p, r in zip(spec["prior"], spec["ratios"]))
        dff = max(64, int(round(base["intermediate_size"] / m / 64)) * 64)
        widths = [max(64, int(round(r * dff / 64)) * 64) for r in spec["ratios"]]
        widths[-1] = dff
        for i in range(1, len(widths)):
            if widths[i] <= widths[i - 1]:
                widths[i] = widths[i - 1] + 64
        dff = widths[-1] = max(dff, widths[-1])
        fixed_w = max(64, int(round(base["intermediate_size"] / 64)) * 64)
    if delta or delta_mla or delta_gqa:
        # mat_ffn's own flags minus the swa_sinkdiet substrate it layers them on.
        flags = dict(_sweep_arch_flags("deltanet_mla" if delta_mla else "qwen3_5_like", base))
        flags.update(use_mat_ffn=True, mat_ffn_budget=0.0, mat_ffn_aux_weight=0.01)
        if delta_gqa:
            flags.update(
                mat_ffn_layer_kinds=["attn"],
                attn_logit_sinks=True,
                attn_register_count=128,
            )
    else:
        flags = dict(_sweep_arch_flags("mat_ffn", base))
    flags.update(
        intermediate_size=dff,
        mat_ffn_widths=list(widths),
        mat_ffn_tier_fracs=list(spec["prior"]),
        mat_ffn_fixed_width=int(fixed_w),
        mat_ffn_dyn_start=dyn,
        mat_ffn_dyn_end=dyn,
        # KL to the prior, not the legacy mean-only term: the mean is satisfied
        # by a bimodal split that never touches the middle tiers, which on a
        # 4-tier nest is precisely the failure to guard against. Weight stays at
        # mat_ffn's 0.01 -- the capacity clip is the hard constraint, this only
        # shapes the router's preferences so it does not learn a policy the clip
        # spends every step overriding.
        mat_ffn_aux_mode="kl",
        mat_ffn_warmup_frac=float(spec["warmup"]),
        mat_ffn_warmup_stages=4,
        mat_ffn_explore_frac=float(spec.get("explore", 0.0)),
        mat_ffn_route_mode=str(spec.get("route_mode", "argmax")),
        mat_ffn_route_thresh_lr=float(spec.get("route_thresh_lr", 0.01)),
        mat_ffn_route_ctrl=str(spec.get("route_ctrl", "wish")),
        mat_ffn_cap_headroom=float(spec.get("cap_headroom", 1.0)),
    )
    if gqa:
        flags.update(
            hybrid_attention_ratio=_ratio_final_global(
                ["swa", "swa", "swa", "attn"], int(L), "attn"
            ),
        )
    if delta or delta_mla:
        # PLACEMENT (2026-10-27). The nest routes every mat_ffn_route_interval-th
        # (4th) layer from mat_ffn_dyn_start, and the 3:1 cycle puts the attention
        # slot at index 3 mod 4. With dyn = round(0.1 L) that landed on DeltaNet
        # layers at L=16/23 (dyn 2: layers 2, 6, 10, ...) but on the ATTENTION
        # layers at L=27 (dyn 3: 3, 7, 11, ...), so the arm changed identity with
        # size. Every banked _delta/_delta_mla cell (ar18m, ar10m) routed DeltaNet
        # layers; pin that at every size: start at the nearest index <= dyn that is
        # 2 mod 4. The attention-slot placement is mat_ffn_thresh_delta_gqa.
        flags.update(mat_ffn_dyn_start=dyn - ((dyn - 2) % 4))

    return flags


# Pre-hero layer-recipe suffix twins (2026-10-31; configuration_umichslm.py documents
# each knob). No suffix here is a suffix of another.
_RECIPE_TWINS: Dict[str, Dict[str, Any]] = {
    "_attngate": {"attn_output_gate": True},
    "_dngate": {"deltanet_output_gate": True},
    "_gates": {"attn_output_gate": True, "deltanet_output_gate": True},
    "_qknorm": {"attn_qk_norm": True},
    # max_position_embeddings covers the stretched ids: under compile the rotary
    # cache cannot grow, so a position past it would fail.
    "_posjit": {"mla_position_jitter": 8, "max_position_embeddings": 65536},
    "_cp27": {"depth_scale_base_layers": 27},
    "_lns": {"layernorm_scaling": True},
    # Agentic program twins (2026-11-04; roadmap "Agentic innovation program").
    "_sharedrope": {"mla_shared_rope_key": True},  # P11: one RoPE key for all MLA heads
    "_diff": {"mla_diff_attn": True},  # P4: differential attention on the MLA layers
    # P7: a bigger DeltaNet state. expand_k=2 doubles each head's key dim (state
    # 2x, +params in q/k); _dnwide uses one DeltaNet head of width h (state h^2 per
    # layer instead of h^2/H: 2x at ar18m, 3x at ar10m), same projection params.
    "_dnexpk": {"deltanet_expand_k": 2},
    "_dnwide": {"deltanet_num_heads": 1},
    # P1 / P2 (FABLE review 2026-11-04, modules/msg_struct.py). _turnpos: the MLA
    # rope also turns with the Harmony message index (theta 1K, lowest-frequency half
    # of the pairs) and o resets per message; pinned keys (system tool block, first
    # user message) get a learned per-head logit bias. +H params per MLA layer;
    # exactly the parent on text with no <|start|>. _digest: k learned queries
    # (8 tool / 4 other) write per-message digests after the first MLA layer; MLA
    # layers see the last 2048 raw tokens plus the digests of earlier complete
    # messages. W=2048 so the mechanism is exercised inside the 4K screen.
    # deltanet_mla_turnpos_digest composes them (digests carry their message index).
    "_turnpos": {"msg_rope": True, "msg_pin_bias": True},
    "_digest": {"msg_digest": True, "digest_window": 2048},
    # P10 follow-up: the generic _mtp twin (1 head, shared lm_head, weight 0.3) trained
    # as Soft-MTP (arXiv 2610.38331): KL to the model's own NTP distribution reaches
    # the head and trunk, CE reaches lm_head. Read against a same-commit <arch>_mtp.
    "_softmtp": {"use_mtp": True, "mtp_num_heads": 1, "mtp_share_head": True,
                 "mtp_loss_weight": 0.3, "mtp_soft": True},
}

# qwen3_like_nmatch FFN width per (hidden, layers): deltanet_mla's N at that tier.
_NMATCH_FFN = {(128, 16): 608, (192, 23): 928, (256, 27): 1248, (320, 36): 1568, (256, 64): 1248}


def _graded_ratio(L: int) -> List[str]:
    """deltanet_mla_graded's layer kinds: ~L/8 MLA layers, 3:1 near both ends."""
    first, last = round(0.15 * L), round(0.25 * L)
    mla = {i for i in range(3, first, 4)} | {i for i in range(L - 1, L - 1 - last, -4)}
    n_mid = max(0, round(L / 8) - len(mla))
    lo, hi = first, L - last
    mla |= {lo + (k + 1) * (hi - lo) // (n_mid + 1) for k in range(n_mid)}
    return ["mla" if i in mla else "delta" for i in range(L)]


def _think_suffix_pathway_flags(L: int) -> Dict[str, Any]:
    """think_suffix's recursed top block (see the ``think_suffix`` arm): one weight-tied
    Pathways lane block after layer L-2, iterated 0/1/2/4 times per token at a 1.75
    budget, reading the register bank through the stage export."""
    return dict(
        use_pathways=True,
        pathway_stages=1,
        pathway_variant="recursive",
        pathway_depths=[0, 1, 2, 4],
        pathway_budget=1.75,
        pathway_shared_bottom=max(1, L - 2),
        pathway_aux_loss_weight=0.01,
        pathway_entropy_weight=0.001,
        pathway_floor_frac=0.1,
        pathway_floor_steps=2000,
        pathway_share_kv=True,
        pathway_bank_export=True,
    )


def _sweep_arch_flags(arch: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Architecture-specific config flags layered onto a matched backbone shape."""
    h = base["hidden_size"]
    f = base["intermediate_size"]
    L = base["num_hidden_layers"]
    if arch.endswith("_bag"):
        # Bag-of-next-K lookahead twin: "<arch>_bag" is <arch> plus the
        # order-free multi-label objective (modules/bag_lookahead.py). Head is
        # weight-tied to lm_head, so N moves only by the d^2 transform and the
        # pair stays a near-matched-N single-variable ablation. Like MTP this is
        # a TRAINING-TIME bet — the head is discarded at inference, so rank the
        # pair on val_lm_loss, which excludes the aux term.
        flags = dict(_sweep_arch_flags(arch[: -len("_bag")], base))
        flags.update(
            use_bag_lookahead=True,
            bag_lookahead_k=16,
            bag_lookahead_weight=0.1,
            bag_share_head=True,
        )
        return flags
    _rec = _split_rec_arch(arch)
    if _rec is not None:
        # Router-free recurrent depth twin (2026-10-08): "<parent>_rec" is the
        # parent with its middle block looped 1-5 times (uniform, drawn per
        # micro-batch, rank-synced), mean 3 -- the Hyperloop shape (arXiv
        # 2604.21254: middle x3). The GQA successor to swa_globals_rec, whose MLA
        # backbone is off the table. PLAIN loop: no input re-injection, no
        # hyper-connections, i.e. Hyperloop's "Looped" baseline and the control
        # for any Parcae/Hyperloop variant built on top.
        #
        # FLOP-MATCHED, not param-matched, at the ar* tiers: _REC_SHAPES picks
        # (L, block) so the unrolled depth at the mean, 2*block + 3*(L - 2*block),
        # equals the parent's L AND, for the 3:1 SWA:global parent, the unrolled
        # schedule carries the same number of global layers. The parent's flags
        # are rebuilt at the new L, so the pair differs only by the loop. Unique N
        # is ~half the body. val_lm_loss is scored at the mean depth (eval_iters
        # 0 -> round of the mean = 3); summary.json's val_lm_loss_by_iters holds
        # T = 1..10. Sizes without a _REC_SHAPES entry keep the parent's L with a
        # quarter-depth block (so the legacy shallow tiers still build) and are
        # NOT FLOP-matched.
        #
        # "<parent>_rec_inject" / "_rec_hc" / "_rec_inject_hc" (2026-10-09) add
        # the loop-boundary mixers of modules/loop_mixers.py on the SAME shape:
        # Parcae stable injection (the fix if val_lm_loss_by_iters rises past
        # the training depth) and Hyperloop per-loop hyper-connections (the fix
        # if the plain loop loses to its parent at matched FLOPs). Each one's
        # control is "<parent>_rec", not the parent. They add 2*d^2 and 6*n^2*d
        # matmul FLOPs per token per iteration, under 1% of the tier totals.
        parent, mixer = _rec
        flags = dict(_sweep_arch_flags(parent, base))
        flags.update(
            use_recurrent_depth=True,
            recurrent_min_iters=1,
            recurrent_max_iters=5,
            recurrent_block_size=base.get(
                "recurrent_block_size", max(1, base["num_hidden_layers"] // 4)
            ),
            **mixer,
        )
        return flags
    if arch.endswith("_fbt"):
        # Latent-feedback twin (modules/latent_feedback.py, arXiv 2608.08888):
        # "<arch>_fbt" is <arch> plus the previous token's TOP-layer state fused
        # into the next token's input. Adds 2*h^2 params (73,728 at ar10m,
        # 1.6M at deep80, under 0.2% of N), so it is near-matched-N -- but it is
        # NOT matched-COMPUTE: the default pass mixture costs ~1.28x per token.
        #
        # So this pair is only readable as loss-per-GPU-hour. At our measured
        # ~0.055 nats/doubling the same compute spent on tokens buys 0.020 nats,
        # which is the bar this arm has to clear. Rank on val_lm_loss, which is
        # the FINAL pass's next-token loss, never the pass-averaged value.
        flags = dict(_sweep_arch_flags(arch[: -len("_fbt")], base))
        flags.update(
            use_latent_feedback=True,
            latent_feedback_pass_mix=[0.75, 0.22, 0.03],
            latent_feedback_mixin_p=0.1,
            latent_feedback_eval_passes=2,
        )
        return flags
    if arch.endswith("_kda"):
        # Kimi Delta Attention twin (2026-10-22; arXiv:2510.26692): "<arch>_kda" swaps
        # the per-head scalar forget gate for a per-KEY-channel one on the same true
        # delta rule (FLA chunk_kda). Adds h*(H*K - H) gate params per DeltaNet layer
        # (~1-2% of N at ar18m/ar10m), FLOPs ~unchanged: near-matched, read on
        # val_lm_loss with the N gap stated. Kimi Linear's own layer also adds an
        # output gate + norm and an A_log/dt_bias gate parameterisation; those are left
        # out so the pair differs in decay granularity only.
        flags = dict(_sweep_arch_flags(arch[: -len("_kda")], base))
        if not (flags.get("use_deltanet") and flags.get("deltanet_true_delta_rule")):
            raise ValueError(f"{arch}: _kda needs a true-delta-rule DeltaNet parent")
        flags.update(deltanet_variant="kda")
        return flags
    if arch.endswith(("_prope", "_nope")):
        # Position-free content dims on the GQA global layers (2026-10-23). deltanet_mla
        # beats qwen3_5_like on induction in every cell, and its MLA heads rotate 64 of
        # 192 q/k dims. "<arch>_prope" rotates a quarter of each GQA head (Qwen3-Next's
        # partial_rotary_factor); "<arch>_nope" rotates none (Kimi Linear's global
        # layers), leaving position to the DeltaNet layers. Matched-N and ~matched-FLOPs
        # with the parent. On an MLA parent, "_nope" sets mla_nope (2026-10-31): the MLA
        # layers skip RoPE on their rope slice (Kimi Linear's NoPE MLA), matched-N.
        suffix = "_prope" if arch.endswith("_prope") else "_nope"
        flags = dict(_sweep_arch_flags(arch[: -len(suffix)], base))
        if suffix == "_nope" and flags.get("use_deltanet") and flags.get("use_mla"):
            flags.update(mla_nope=True)
            return flags
        if not flags.get("use_deltanet") or flags.get("use_mla"):
            raise ValueError(f"{arch}: {suffix} needs a DeltaNet/GQA hybrid parent")
        flags.update(partial_rotary_factor=0.25 if suffix == "_prope" else 0.0)
        return flags
    m = re.search(r"_r(7|15)$", arch)
    if m:
        # Sparser hybrid ratio (2026-10-26, PRETRAIN_PLAN.md §0.4 P2): "<arch>_r7" /
        # "<arch>_r15" keep the parent's layer types but repeat k DeltaNet layers per
        # global layer instead of 3, interleaved, final layer global. MLA count at
        # 16/23/27/64 layers: 3:1 4/6/7/16, 7:1 2/3/4/8, 15:1 1/2/2/4.
        k = int(m.group(1))
        flags = dict(_sweep_arch_flags(arch[: m.start()], base))
        ratio = flags.get("hybrid_attention_ratio") or []
        glob = sorted({e for e in ratio if e != "delta"})
        if not flags.get("use_deltanet") or "delta" not in ratio or len(glob) != 1:
            raise ValueError(f"{arch}: _r{k} needs a DeltaNet hybrid with one global layer type")
        flags["hybrid_attention_ratio"] = _ratio_final_global(["delta"] * k + glob, L, glob[0])
        return flags
    if arch.endswith("_sink"):
        # Logit-sink twin (2026-10-27): "<arch>_sink" adds a learned per-head sink
        # logit to every FULL-attention layer ("attn" GQA or "mla") of a hybrid
        # (modules/attn_sink.py, exact FA2 gradient). +H params per such layer, so
        # matched-N with the parent to within rounding; read the pair directly.
        flags = dict(_sweep_arch_flags(arch[: -len("_sink")], base))
        ratio = flags.get("hybrid_attention_ratio") or []
        if flags.get("attention_mode") == "swa_globals_mla" or not {"attn", "mla"} & set(ratio):
            raise ValueError(f"{arch}: _sink needs a hybrid with attn/mla layers (not swa_globals)")
        flags.update(attn_logit_sinks=True)
        return flags
    if arch.endswith("_engram"):
        # Engram twin (P8, 2026-11-04; arXiv 2601.07372, modules/engram.py): hashed
        # 2/3-gram embedding tables added to the residual after the first DeltaNet
        # triplet (layer 3, the first MLA layer's input), gated by the hidden state.
        # Tables sized to ~25% of the tier's dense N (a lookup: ~0 FLOPs), so report
        # total and active N separately; the comparison is at matched FLOPs.
        flags = dict(_sweep_arch_flags(arch[: -len("_engram")], base))
        dense_n = 2 * 56000 * h + L * 12 * h * h
        heads, dim, ngrams = 4, 64, [2, 3]
        flags.update(
            use_engram=True,
            engram_layer_ids=[min(3, L - 1)],
            engram_ngram_sizes=ngrams,
            engram_num_heads=heads,
            engram_embed_dim=dim,
            engram_table_size=int(0.25 * dense_n / (heads * len(ngrams) * dim)) | 1,
        )
        return flags
    for _sfx, _upd in _RECIPE_TWINS.items():
        if arch.endswith(_sfx):
            # Pre-hero layer-recipe twins (2026-10-31, roadmap Phase 1/2): one knob each
            # on any parent (see _RECIPE_TWINS); the config validator rejects MLA-only
            # knobs on a parent without MLA layers.
            flags = dict(_sweep_arch_flags(arch[: -len(_sfx)], base))
            flags.update(_upd)
            return flags
    if arch == "deltanet_mla_graded":
        # Depth-graded hybrid (N2, 2026-10-31): 7:1's MLA count (~L/8), but 3:1 in the
        # first ~15% and last ~25% of layers and the remainder spread evenly through the
        # middle; final layer MLA. Only distinct from _r7 at depth (w256l64: 8 MLA as
        # 2 early + 2 middle + 4 late, vs _r7's 8 evenly spaced).
        flags = dict(_sweep_arch_flags("deltanet_mla", base))
        flags.update(hybrid_attention_ratio=_graded_ratio(L))
        return flags
    if arch == "qwen3_like_nmatch":
        # Parameter-matched dense control (2026-10-31): qwen3_like with its FFN widened
        # so N matches deltanet_mla at the same tier (within 0.25%), because the grid
        # matches arms on SHAPE and deltanet_mla carries +3-11% N. Read against
        # deltanet_mla at the same D (gen_a40_grid DELTA budget).
        # Registration builds every tier at import, so an unlisted shape cannot raise
        # here; -1 makes the FFN Linear fail at model build instead of training a
        # silent copy of qwen3_like.
        return {"intermediate_size": _NMATCH_FFN.get((h, L), -1)}
    if arch.endswith("_think"):
        # think_suffix twin (2026-10-31): "<arch>_think" ports think_suffix's mechanisms
        # onto a DeltaNet hybrid with full-attention (MLA or GQA) layers -- a learned
        # per-head logit sink on those layers (as _sink), a 128-slot register bank, and
        # ONE weight-tied recursed top block (0/1/2/4 passes, budget 1.75) that reads the
        # bank through its export. As on think_suffix, no spine layer reads the bank.
        # The lane block keeps think_suffix's 1024-token band, so the recursion is local
        # attention + bank; long range stays with the DeltaNet state and the MLA layers.
        # Read against <arch>_sink (recursion + bank) and <arch> (everything).
        flags = dict(_sweep_arch_flags(arch[: -len("_think")], base))
        ratio = flags.get("hybrid_attention_ratio") or []
        if not flags.get("use_deltanet") or not {"attn", "mla"} & set(ratio):
            raise ValueError(f"{arch}: _think needs a DeltaNet hybrid with attn/mla layers")
        flags.update(_think_suffix_pathway_flags(L))
        flags.update(attn_logit_sinks=True, attn_register_count=128, pathway_lane_window=1024)
        return flags
    if arch.endswith("_rwkv7"):
        # RWKV-7 twin (2026-10-31; arXiv 2503.14456, modules/rwkv7.py): the DeltaNet
        # slots run RWKV-7's time-mix (generalized delta rule: vector decay, vector
        # in-context LR, decoupled remove/replace keys, value residual) through FLA's
        # chunk_rwkv7; token shift replaces the short conv; our SwiGLU FFN stays.
        # ~+3% N over GatedDeltaNet at these widths (LoRA minimum rank 32).
        flags = dict(_sweep_arch_flags(arch[: -len("_rwkv7")], base))
        if not flags.get("use_deltanet"):
            raise ValueError(f"{arch}: _rwkv7 needs a DeltaNet hybrid parent")
        flags.update(deltanet_variant="rwkv7")
        return flags
    if arch.endswith("_negeig"):
        # Negative-eigenvalue twin (2026-10-22): "<arch>_negeig" is a true-delta-rule
        # DeltaNet hybrid with beta in (0, 2), so (I - beta k k^T) can go negative
        # (arXiv:2411.12537; Olmo Hybrid, arXiv:2604.03444). Matched-N and
        # matched-FLOPs with its parent: read the pair on val_lm_loss directly.
        flags = dict(_sweep_arch_flags(arch[: -len("_negeig")], base))
        if not (flags.get("use_deltanet") and flags.get("deltanet_true_delta_rule")):
            raise ValueError(f"{arch}: _negeig needs a true-delta-rule DeltaNet parent")
        flags.update(deltanet_allow_neg_eigval=True)
        return flags
    if arch.endswith("_pause_call"):
        # P12 (FABLE review 2026-11-04, modules/msg_struct.compute_decision_mask):
        # "<arch>_pause_call" is <arch>_pause with the prediction stream read -- and the
        # shared FFN blended -- only at Harmony decision tokens (an assistant
        # <|channel|>, the rest of a commentary/tool-call message, the first 16 tokens
        # of a final message); the state stream and readout are the parent's
        # elsewhere. MASKED, not compacted: costs what _pause costs, so it tests the
        # kill condition (does the gain survive confinement), not the price.
        # Matched-N with <arch>_pause. Read against <arch>_pause and <arch>.
        flags = dict(_sweep_arch_flags(arch[: -len("_call")], base))
        flags.update(pause_decision_only=True, pause_final_tokens=16)
        return flags
    if arch.endswith("_pause"):
        # Free-pause-token twin (modules/pause_stream.py, arXiv 2609.03807):
        # "<arch>_pause" is <arch> plus a second weight-shared prediction stream
        # that writes no K/V. Adds e_pause (h params) plus 4h of scalar gates per
        # layer -- 4,608 params at ar10m, i.e. matched-N to within rounding.
        # Cost is ~1.45x per step with the shared gated FFN on (the paper's own
        # reducer, hence the default), ~1.9x without.
        #
        # The prediction query forks off q_proj on GQA/SWA layers and arrives as
        # pause_query_states on MLA and Gated DeltaNet (2026-10-31): there the
        # prediction stream reads the state stream's recurrent state, o_t = S_t q_t,
        # and writes none of it. The block raises for any other mixer, post-norm,
        # or a meta/global token prefix.
        flags = dict(_sweep_arch_flags(arch[: -len("_pause")], base))
        flags.update(use_pause_stream=True, pause_shared_ffn=True)
        return flags
    if arch.endswith("_recon"):
        # Bank-reconstruction twin: "<arch>_recon" is <arch> plus the loss that
        # forces earlier-chunk content to be recoverable from the bank state
        # alone (modules/bank_reconstruction.py). ONLY valid on arms with a
        # WRITTEN bank (use_delta_bank); the model raises on a static bank,
        # where the target is unpredictable by construction. Pair it strictly
        # against its own base arm, never against a static-bank arch.
        flags = dict(_sweep_arch_flags(arch[: -len("_recon")], base))
        flags.update(
            use_bank_reconstruction=True,
            bank_recon_samples=32,
            bank_recon_weight=0.1,
        )
        return flags
    if arch.endswith("_mtp") and arch != "swa_globals_mtp":
        # Generic Multi-Token Prediction twin of any arm: "<arch>_mtp" is <arch>
        # plus MTP heads. DeepSeek-V3 settings (1 extra token, shared LM head,
        # loss weight 0.3), matching the swa_globals_mtp precedent below.
        #
        # MTP is a TRAINING-TIME bet: the heads are discarded at inference, so it
        # must repay its extra FLOPs/params within the same budget. It is NOT
        # matched-N with its base twin (the extra transformer block(s) add
        # params; mtp_share_head=True keeps the vocab projection shared, which is
        # most of what it could have cost) — report unique-N.
        #
        # Rank the pair on val_lm_loss, which is the NEXT-TOKEN loss only, never
        # the fused loss (CLAUDE.md): the question is whether MTP training
        # improves ordinary language modelling, so the metric must exclude the
        # MTP auxiliary term.
        flags = dict(_sweep_arch_flags(arch[: -len("_mtp")], base))
        flags.update(
            use_mtp=True,
            mtp_num_heads=1,
            mtp_share_head=True,
            mtp_loss_weight=0.3,
        )
        return flags
    if arch == "vanilla":
        # Vanilla dense transformer baseline: full multi-head attention (no GQA),
        # otherwise the shared matched backbone (RoPE + RMSNorm + SwiGLU). Differs
        # from qwen3_like only by MHA vs GQA.
        return {"num_key_value_heads": base["num_attention_heads"]}
    if arch == "qwen3_like":
        # Dense GQA softmax attention (baseline).
        return {}
    if arch == "qwen3_5_like":
        # 3:1 Gated DeltaNet / attention hybrid. Uses the *true* delta rule (Qwen3-Next
        # style: delta correction + qk-l2norm + log-space gate) via FLA's chunk kernel,
        # not the legacy scalar-gated linear-attention fallback. The last layer is forced
        # to attention, as in deltanet_mla, so the GQA/MLA twins differ only in the
        # global layer's kind (a bare 4-cycle ends on DeltaNet at L=23/27). Changed
        # 2026-10-20; the July delphi screens ran the bare cycle.
        return {
            "use_deltanet": True,
            "deltanet_variant": "gated",
            "deltanet_true_delta_rule": True,
            "hybrid_attention_ratio": _ratio_final_global(
                ["delta", "delta", "delta", "attn"], L, "attn"
            ),
        }
    if arch == "mod":
        # Plain Mixture-of-Depths on a dense backbone.
        return {
            "use_mod": True,
            "mod_capacity": 0.125,
            "mod_every_n_layers": 2,
            "mod_aux_loss_weight": 0.01,
        }
    if arch == "moda":
        # MoDA (Mixture-of-Depths Attention, Arm C): heads attend over sequence
        # KV *and* per-position depth KV from preceding layers under one softmax.
        # Honest framing: a QUALITY arm (dynamic attention over depth), not a
        # FLOP-savings arm — the algorithmic extra cost is ~num_depth_entries
        # extra attendable positions per query, negligible next to seq_len.
        # Rides the shared dense-GQA backbone. NOTE: use_moda auto-switches the
        # backbone to post-norm (the MoDA reference design); that is part of
        # this arm's identity, not an accident. Depth-KV memory is bounded via
        # every-other-layer contributions and a 4-entry FIFO.
        return {
            "use_moda": True,
            "moda_use_ffn_kv": True,
            "moda_depth_kv_every_n": 2,
            "moda_max_depth_cache": 4,
        }
    if arch == "mod_mlp":
        # Arm A (MoD-MLP): route ONLY the FFN; attention runs dense for every
        # token. Mirrors the `mod` incumbent (same stride, same capacity) so the
        # comparison isolates what routing attention away actually buys/costs.
        # gate_fn=sigmoid keeps this arm clean of the raw-gate instability the
        # parallel MoD-tuning workstream is addressing (bounded gain, BCE and LM
        # loss agree on the gate direction).
        return {
            "use_mod": True,
            "mod_mlp_only": True,
            "mod_capacity": 0.125,
            "mod_every_n_layers": 2,
            "mod_aux_loss_weight": 0.01,
            "mod_gate_fn": "sigmoid",
        }
    if arch == "deltanet_mla_mlp":
        # Arm A on the recurrent hybrid: FFN-only routing composes with
        # DeltaNet because the FFN is position-wise (no cross-token state to
        # corrupt) — the config full-block MoD falls back to dense on.
        return {
            "use_mla": True,
            "use_deltanet": True,
            "deltanet_variant": "gated",
            "deltanet_true_delta_rule": True,
            "attention_mode": "deltanet_mla",
            "hybrid_attention_ratio": _ratio_final_global(
                ["delta", "delta", "delta", "mla"], L, "mla"
            ),
            "mla_kv_lora_rank": max(64, h // 2),
            "use_mod": True,
            "mod_mlp_only": True,
            "mod_capacity": 0.125,
            "mod_every_n_layers": 2,
            "mod_aux_loss_weight": 0.01,
            "mod_gate_fn": "sigmoid",
        }
    if arch == "swa_globals_mlp":
        # Arm A on the primary backbone: block-atomic decision (one router per
        # 4-layer block) but gating ONLY each member layer's FFN delta.
        # Attention is never routed, so total bypass is impossible and this arm
        # rides the shared 5-9 layer sweep shapes — no deep-shape tax, no
        # custom _SWEEP_ARCH_SHAPES entry, matched-N with the dense controls.
        # io_blocks_dense=False is safe here for the same reason.
        return {
            "use_mla": True,
            "attention_mode": "swa_globals_mla",
            "hybrid_attention_ratio": _ratio_final_global(["swa", "swa", "swa", "mla"], L, "mla"),
            "sliding_window": 1024,
            "swa_window": 1024,
            "swa_global_count": 256,
            "mla_kv_lora_rank": max(64, h // 2),
            "use_mod": True,
            "mod_mlp_only": True,
            "mod_atomic_block": True,
            "mod_block_size": 4,
            "mod_io_blocks_dense": False,
            "mod_capacity": 0.5,
            "mod_aux_loss_weight": 0.01,
            "mod_gate_fn": "sigmoid",
        }
    if arch == "swa_globals_dense":
        # Control: the SWA+globals+MLA backbone with NO dynamic compute at all.
        # Same L=12 shape as swa_globals -> isolates what block-atomic MoD
        # costs/buys on the primary backbone, and serves as Arm B's
        # "dense-stack parity at matched unique-N" bar (swa_globals_rec is this
        # exact model plus core-block recursion, zero extra params).
        return {
            "use_mla": True,
            "attention_mode": "swa_globals_mla",
            "hybrid_attention_ratio": _ratio_final_global(["swa", "swa", "swa", "mla"], L, "mla"),
            "sliding_window": 1024,
            "swa_window": 1024,
            "swa_global_count": 256,
            "mla_kv_lora_rank": max(64, h // 2),
        }
    if arch == "swa_globals_rec":
        # Arm B: router-free recurrent depth (Geiping-style latent reasoning)
        # on the softmax SWA-globals/MLA backbone — NO MoD. The middle block
        # (layers 4-7 at L=12) is looped 1-3 times (sampled per micro-batch);
        # adaptive depth at inference arrives later via halting. No router, no
        # aux loss, no train/infer routing mismatch by construction; routing
        # entropy/collapse diagnostics are N/A for this arm (a selling point).
        # Rides the swa_globals L=12 shape (matched-N to the incumbent); the
        # L(N,D) fit gets unique-N (shared params counted once) while the
        # token horizon uses compute-N (core billed at E[iters]=2).
        return {
            "use_mla": True,
            "attention_mode": "swa_globals_mla",
            "hybrid_attention_ratio": _ratio_final_global(["swa", "swa", "swa", "mla"], L, "mla"),
            "sliding_window": 1024,
            "swa_window": 1024,
            "swa_global_count": 256,
            "mla_kv_lora_rank": max(64, h // 2),
            "use_recurrent_depth": True,
            "recurrent_block_size": 4,
            "recurrent_min_iters": 1,
            "recurrent_max_iters": 3,
        }
    if arch == "gwt":
        # Pitch 1 (GWT, chunk 2): Global Workspace Trunk. Shallow SWA+globals+
        # MLA token stream (L=8, uniform per-token cost, NO routing) plus a
        # deep trunk that updates a 256-slot live workspace at 512-token chunk
        # boundaries — write cross-attn + slot self-attn + MLP, iterated 1-3x
        # (sampled per micro-batch via Arm B's machinery; halting decode
        # later). Deep compute is O(1) in sequence length. The workspace READ
        # is zero-initialized, so at init this model IS its static-globals
        # twin. Twins: swa_globals_dense / swa_globals (static globals).
        # Reports unique-N; the trunk bills per execution in compute-N.
        return {
            "use_mla": True,
            "attention_mode": "swa_globals_mla",
            "hybrid_attention_ratio": _ratio_final_global(["swa", "swa", "swa", "mla"], L, "mla"),
            "sliding_window": 1024,
            "swa_window": 1024,
            "swa_global_count": 256,
            "mla_kv_lora_rank": max(64, h // 2),
            "use_workspace_trunk": True,
            "trunk_layers": 2,
            "trunk_chunk_size": 512,
            "trunk_iters_min": 1,
            "trunk_iters_max": 3,
            "trunk_insert_layer": L // 2,
            # CUDA-graph the iterated trunk inner loop: -37% median step time at
            # 25m (512->323 ms, p90 724->371; A/B 2026-08-18, A5000, intra-doc
            # ON). Parity + doc-reset + weight-shared gates in
            # tests/test_workspace_trunk.py::TestTrunkCudaGraph. Inert on CPU
            # and outside training; degrades to eager on capture failure.
            "trunk_cuda_graph": True,
        }
    if arch == "mop":
        # Pitch 2 (MoP, chunk 2): Mixture of Pathways on the dense-GQA
        # backbone. One causal lane commitment per token among express(0) /
        # standard(2) / deep(4) block lanes; lanes cross-attend to the shared
        # export only (no lane-token mixing -> dense compute == gathered,
        # bitwise). Cost-aware aux balances expected FLOPs to budget=2.0
        # lane-blocks/token (standard parity); deep-lane floor for the first
        # 2k steps prevents early starvation. Zero train/infer routing
        # mismatch by construction. Twin: mop_dense (single standard lane).
        return {
            "use_pathways": True,
            "pathway_depths": [0, 2, 4],
            "pathway_shared_bottom": max(1, (2 * L) // 3),
            "pathway_budget": 2.0,
            "pathway_aux_loss_weight": 0.01,
            "pathway_entropy_weight": 0.001,
            "pathway_floor_frac": 0.1,
            "pathway_floor_steps": 2000,
        }
    if arch == "mop_kv":
        # MoP sharing ladder step 1: identical to `mop` EXCEPT lane-block K/V
        # are projected once from the shared export and reused by every block
        # in every lane — a common learned "read of the shared stream" (and a
        # small param/FLOP saving). Isolates: does sharing the read help?
        return {
            "use_pathways": True,
            "pathway_depths": [0, 2, 4],
            "pathway_shared_bottom": max(1, (2 * L) // 3),
            "pathway_budget": 2.0,
            "pathway_aux_loss_weight": 0.01,
            "pathway_entropy_weight": 0.001,
            "pathway_floor_frac": 0.1,
            "pathway_floor_steps": 2000,
            "pathway_share_kv": True,
        }
    if arch == "mop_nested":
        # Sharing ladder step 2: identical to `mop_kv` EXCEPT lanes are
        # PREFIXES of one shared 4-block stack (standard exits after block 2).
        # The early blocks are common low/intermediate processing trained by
        # standard + deep tokens together; specialization lives in the tail.
        # Isolates: does prefix-sharing the lane stack help? Fewer lane params
        # than disjoint -> wider matched-N shape (see _SWEEP_ARCH_SHAPES).
        return {
            "use_pathways": True,
            "pathway_depths": [0, 2, 4],
            "pathway_shared_bottom": max(1, (2 * L) // 3),
            "pathway_budget": 2.0,
            "pathway_aux_loss_weight": 0.01,
            "pathway_entropy_weight": 0.001,
            "pathway_floor_frac": 0.1,
            "pathway_floor_steps": 2000,
            "pathway_share_kv": True,
            "pathway_variant": "nested",
        }
    if arch == "mop_rec":
        # Sharing ladder step 3 (MoR-flavored end of the spectrum): identical
        # to `mop_nested` EXCEPT the lane stack is ONE weight-tied block —
        # lane depth is the iteration count (0/2/4 passes). Maximal signal
        # sharing: every routed token trains the same parameters; depth is
        # pure compute. Compute-N > unique-N here (passes billed per
        # execution), the mirror image of the disjoint discount.
        return {
            "use_pathways": True,
            "pathway_depths": [0, 2, 4],
            "pathway_shared_bottom": max(1, (2 * L) // 3),
            "pathway_budget": 2.0,
            "pathway_aux_loss_weight": 0.01,
            "pathway_entropy_weight": 0.001,
            "pathway_floor_frac": 0.1,
            "pathway_floor_steps": 2000,
            "pathway_share_kv": True,
            "pathway_variant": "recursive",
        }
    if arch in ("mop_stack", "mop_stack_shared"):
        # Stacked MoP x blocked-MoD hybrid (the deep-target topology): dense
        # bottom stack (input representation, never skipped) | S stages of
        # [pathway lanes + one dense spine layer] | dense top stack (output
        # shaping, never skipped). Params grow LINEARLY in stages while the
        # per-token depth menu grows exponentially (a lane choice per stage,
        # 3^S combinations); ONE global budget across stages lets a token
        # spend all its depth in a few stages and coast elsewhere. Lanes are
        # nested within each stage (Wave 2 ladder rung), share_kv on.
        # mop_stack_shared reuses ONE lane set across every stage (sub-linear
        # params — what makes the 72-128 layer target affordable); routers
        # and merge scales stay per-stage. Screen axis: S (2-3), mirroring
        # how swa_globals_deep/deeper screened routed-block count R.
        stages = 2 if L <= 6 else 3
        bottom = {5: 2, 6: 3, 8: 3, 9: 4}.get(L, max(2, L // 3))
        flags = {
            "use_pathways": True,
            "pathway_depths": [0, 2, 4],
            "pathway_shared_bottom": bottom,
            "pathway_budget": 2.0,
            "pathway_aux_loss_weight": 0.01,
            "pathway_entropy_weight": 0.001,
            "pathway_floor_frac": 0.1,
            "pathway_floor_steps": 2000,
            "pathway_share_kv": True,
            "pathway_variant": "nested",
            "pathway_stages": stages,
        }
        if arch == "mop_stack_shared":
            flags["pathway_stage_share_lanes"] = True
        return flags
    if arch == "mop_dense":
        # MoP's matched-N twin: identical machinery, ONE standard-depth lane,
        # no routing choice (router skipped, aux inert). Isolates what the
        # commit-once routing itself buys/costs at matched N and matched
        # per-token compute shape.
        return {
            "use_pathways": True,
            "pathway_depths": [2],
            "pathway_shared_bottom": max(1, (2 * L) // 3),
        }
    if arch == "swa_registers":
        # Kernel-friendly twin of swa_globals: the SAME 256-token bank, read as
        # REGISTERS via zero-init cross-attention instead of being prepended.
        # Self-attention becomes a pure FA2-native sliding window (varlen
        # intra-doc at O(S) memory — no [B,1,S,S] additive masks, no SDPA
        # fallback). Everything else (MoD block-atomic, MLA 3:1, shapes) is
        # inherited from swa_globals so the twin isolates ONE question: does
        # moving the globals out of the softmax (two softmaxes + gate instead
        # of one joint prefix softmax) cost quality, and what does it buy in
        # MFU / memory? Matched-N: the only param delta is the per-head gates.
        flags = dict(_sweep_arch_flags("swa_globals", base))
        flags["swa_use_registers"] = True
        return flags
    if arch == "swa_hybrid":
        # Half-and-half split of the 256-token bank: 128 stay a prepended
        # global prefix (joint-softmax sink + visibility to the MLA
        # full-attention layers — the mechanisms the 4096 intra-doc pair
        # suggests drive the prefix arch's quality edge) and 128 become a
        # register bank read via the zero-init gated cross-attention on SWA
        # layers (cheap per-layer capacity + a clean post-training attachment
        # point for agentic/tool memory). Matched-N with both twins: total
        # bank is still 256 x d; only the per-head gates are extra. Stays on
        # the prefix kernel path (doc-aware flex), so this arm isolates the
        # FUNCTIONAL split, not kernel efficiency.
        flags = dict(_sweep_arch_flags("swa_globals", base))
        flags["swa_use_registers"] = True
        flags["swa_register_count"] = int(flags.get("swa_global_count", 256)) // 2
        return flags
    if arch == "swa_hybrid32":
        # Sink-minimal hybrid (2026-07-20): 32 prefix + 128 registers. The
        # probe results showed the registers deficit is UNIFORM (missing-sink
        # mechanism, StreamingLLM), not recall — and the sink literature says a
        # handful of sink tokens carries that function. So shrink the expensive
        # prefix 4x and keep the full register bank. NOTE bank total is 160,
        # not 256 — N drops by 96 x d vs the 256-bank arms; report unique-N.
        flags = dict(_sweep_arch_flags("swa_globals", base))
        flags["swa_use_registers"] = True
        flags["swa_global_count"] = 160
        flags["swa_register_count"] = 128
        return flags
    if arch == "swa_deltabank":
        # swa_hybrid32 + exactly ONE mechanism: cumulative chunk-writes into
        # the register bank (modules/delta_bank.py). The out-of-band-carry
        # candidate: every arch probed 2026-07-20 was at floor/chance beyond
        # window+bank reach; this is the only design whose mechanism CAN move
        # content forward across chunks. Zero-init => at init bitwise
        # swa_hybrid32, so the pair is a single-variable ablation. Write layer
        # must be MoD-block-aligned (dead-trunk post-mortem 2026-07-19). The
        # write module adds ~4 d^2 params — unique-N is reported.
        #
        # Write layer: the FIRST routable block boundary (L//3 rounded to the
        # block), NOT the L//2 the scribe/trunk arms use. The delta bank's
        # whole purpose is carrying content forward, so it needs layers ABOVE
        # the write to read it: at L=12, writing at 8 leaves 2 SWA readers,
        # writing at 4 leaves 5. Shallower write states are the cost.
        _safe_insert = max(4, ((L // 3 + 3) // 4) * 4)
        flags = dict(_sweep_arch_flags("swa_hybrid32", base))
        flags.update(
            use_delta_bank=True,
            delta_chunk_size=512,
            delta_write_layer=_safe_insert,
            delta_write_topk=16,
        )
        return flags
    if arch in ("swa_ws", "swa_globals_ws", "swa_hybrid_ws"):
        # Writable-workspace bank arms (2026-07-18). The prefix globals are
        # CAUSAL at positions 0..G-1 — they can never read content; likewise
        # the register bank. These arms add the third bank kind: a live
        # workspace written from content at chunk boundaries (WorkspaceTrunk,
        # reused from gwt but pinned to ONE deterministic iteration and ONE
        # 1-block trunk — this is a BANK comparison, not a test-time-compute
        # arm). Zero-init read => each arm starts bitwise as its bank-less
        # twin. Trunk adds ~4-5% params (unique-N reported; IsoFLOP recomputes
        # D from measured N).
        #   swa_ws:        NO prefix, 256-slot workspace. Twin of swa_registers
        #                  (static 256 bank, gated read) — isolates "does the
        #                  bank being WRITABLE matter" at matched read style.
        #   swa_globals_ws: full 256 prefix + 128-slot workspace — does a
        #                  writable bank add anything ON TOP of production?
        #   swa_hybrid_ws: tri-bank. 128 prefix (sink + MLA visibility) +
        #                  128 registers (free capacity, growable post-hoc) +
        #                  128-slot workspace (in-context state).
        flags = dict(_sweep_arch_flags("swa_globals", base))
        # Insert MUST be block-aligned: the MoD gather consumes non-head members
        # of routed blocks BEFORE the insert hook runs (modeling ~1430), so an
        # insert at a consumed index silently never fires. L//2 landed inside
        # the routed block and trained a DEAD trunk (2026-07-19 post-mortem).
        _safe_insert = ((L // 2 + 3) // 4) * 4
        flags.update(
            use_workspace_trunk=True,
            trunk_layers=1,
            trunk_chunk_size=512,
            trunk_iters_min=1,
            trunk_iters_max=1,
            trunk_insert_layer=_safe_insert,
        )
        if arch == "swa_ws":
            flags["swa_global_count"] = 0
            flags["trunk_workspace_slots"] = 256
        elif arch == "swa_globals_ws":
            flags["trunk_workspace_slots"] = 128
        else:  # swa_hybrid_ws
            flags["swa_use_registers"] = True
            flags["swa_register_count"] = int(flags.get("swa_global_count", 256)) // 2
            flags["trunk_workspace_slots"] = 128
        return flags
    if arch in ("swa_scribe_mopr", "swa_scribe_moprec"):
        # Shallow-spine stacked MoP (2026-08-19, deep-study follow-up). The
        # deep study showed effective depth should come from STACKING, but the
        # benched mopd point (spine 72 + 3 stages) has almost no dynamic range
        # (min 72 / max 84). These arms move most of the depth into the routed
        # menu: min cost == the shallow spine, max ~= spine + stages*6 — for a
        # hard/easy token mix, easy tokens really are cheap.
        # Depth menu stays [0,2,4]: its UNIFORM mean (2.0) equals the budget,
        # so a random-init router realizes the tier's effective depth at bench
        # time (an asymmetric menu like [0,2,6] would silently bench at mean
        # 2.67/stage — the budget only acts through the aux loss). Dynamic
        # range comes from STAGE COUNT instead:
        #   eff54 (L=32): 11 stages x 2.0 -> min 32 / mean 54 / max 76.
        #   eff78 (L=36): 21 stages x 2.0 -> min 36 / mean 78 / max 120.
        #   mopr   — nested shared lanes (4 unique lane blocks, prefix exits).
        #   moprec — pathway_variant="recursive": ONE weight-tied lane block,
        #            depth = iteration count (the MoR end; depth is pure
        #            compute, params ~1 block). Same geometry as mopr so the
        #            pair isolates weight reuse's wall-clock-per-FLOP.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        for k in (
            "use_mod",
            "mod_capacity",
            "mod_atomic_block",
            "mod_block_size",
            "mod_io_blocks_dense",
            "mod_aux_loss_weight",
        ):
            flags.pop(k, None)
        # (stages, shared_bottom) keyed on the spine depth L. The SMALL-TIER
        # entry (L=20) is the scaled-down eff54 recipe: eff = L + stages*budget
        # = 34, i.e. 1.70x the spine against eff54's 54/32 = 1.69. Without it
        # the 10m/25m/50m presets fell through to the L=12 fallback (3 stages,
        # bottom 4) and were GEOMETRICALLY IDENTICAL to swa_scribe_mopd —
        # the shallow-spine thesis was not under test at any screenable size.
        # L=20 is the shallowest spine that still leaves the scribe write layer
        # (bottom+stages = 12) SWA readers above it.
        # (stages, bottom) per spine depth. deep80/deep128 keep the arm's
        # thesis at the deep tiers: spine 48 + 16 stages -> eff 80, spine 64 +
        # 32 stages -> eff 128, so effective depth matches the tier's dense L
        # while the SPINE (the cost every token pays) stays far shorter.
        # Without these entries mopr fell through to the 3-stage default and
        # was no longer a shallow-spine stacker at exactly the depths the
        # target model cares about.
        stages, bottom = {
            32: (11, 8),
            36: (21, 8),
            20: (7, 5),
            48: (16, 12),
            64: (32, 16),
            # aspect-ratio tiers: spine ~0.6 x the tier depth, stages make up
            # the rest, so eff depth matches the tier while the spine every
            # token pays stays short.
            14: (5, 4),
            16: (6, 5),
            22: (7, 6),
            30: (11, 8),
        }.get(L, (2 if L <= 6 else 3, max(2, L // 3)))
        budget = 2.0
        flags.update(
            use_pathways=True,
            pathway_depths=[0, 2, 4],
            pathway_shared_bottom=bottom,
            pathway_budget=budget,
            pathway_aux_loss_weight=0.01,
            pathway_entropy_weight=0.001,
            pathway_floor_frac=0.1,
            pathway_floor_steps=2000,
            pathway_share_kv=True,
            pathway_variant="nested" if arch == "swa_scribe_mopr" else "recursive",
            pathway_stages=stages,
            pathway_stage_share_lanes=True,
        )
        last_stage = flags["pathway_shared_bottom"] + stages - 1
        insert = min(L - 1, max(flags["scribe_write_layer"], last_stage + 1))
        flags["scribe_write_layer"] = insert
        flags["trunk_insert_layer"] = insert
        return flags
    if arch in ("swa_scribe_mopr_be", "swa_scribe_mopr_b1"):
        # Two single-knob twins of swa_scribe_mopr (2026-08-20, wall-clock
        # scaling sweep). Both keep mopr's geometry EXACTLY, so each pairs with
        # it as a one-variable comparison.
        #   _be — pathway_bank_export: the register bank joins the stage KV
        #         export and the spine reads it inside attention it already
        #         runs, instead of a per-SWA-layer zero-init cross-attention.
        #         Worth -98 ms/step at 4096 on the eff54 bench (the first
        #         sub-1s mopr54 step) but it MOVES WHERE THE BANK IS READ, so
        #         it is a functional change and is screened as its own arm
        #         rather than folded into mopr.
        #   _b1 — pathway_budget 1.0 instead of 2.0 on the SAME [0,2,4] menu:
        #         the aux loss targets one lane-block/token, so converged
        #         effective depth is ~L + stages = 27 rather than 34. Prices
        #         the low end of the budget curriculum (train the 4K bulk
        #         cheap, anneal the budget up later). NOTE the menu is
        #         deliberately left asymmetric to the budget (mean 2.0 vs
        #         budget 1.0): a random-init BENCH of this arm therefore still
        #         routes at ~2.0/stage and will not show the saving -- only a
        #         trained cell will. Bench it with the aux loss converged or
        #         read the realized depth from the router counters.
        flags = dict(_sweep_arch_flags("swa_scribe_mopr", base))
        if arch == "swa_scribe_mopr_be":
            flags.update(pathway_bank_export=True)
        else:
            flags.update(pathway_budget=1.0)
        return flags
    if arch in ("swa_scribe_mop", "swa_scribe_mopd", "swa_scribe_mopd_delta"):
        # Backbone variants of swa_scribe: the FOUR BANKS are held fixed and
        # only the dynamic-compute mechanism changes, so the comparison
        # swa_scribe (block-atomic MoD) / _mop / _mopd isolates the backbone.
        #   _mop  — plain Mixture of Pathways: one causal lane commitment per
        #           token (express 0 / standard 2 / deep 4 blocks) after a
        #           shared bottom, cost-aware aux to budget 2.0.
        #   _mopd — stacked block-MoP with SHARED lanes: S stages of
        #           [lanes + dense spine], ONE lane set reused across stages.
        #           Params grow SUB-LINEARLY in stages while the per-token
        #           depth menu grows as 3^S — the property that makes the deep
        #           (72-128 layer) 1B target affordable, which is why this is
        #           the variant carried forward rather than unshared stacking.
        # Shapes alias swa_globals (same width/depth as every other bank arm),
        # NOT the tuned mop shapes: holding the bank machinery and the shape
        # fixed is what makes the backbone the only moving part. Lane params
        # therefore push N above the MoD twin — unique-N is reported and the
        # IsoFLOP grid recomputes D from measured N.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        for k in (
            "use_mod",
            "mod_capacity",
            "mod_atomic_block",
            "mod_block_size",
            "mod_io_blocks_dense",
            "mod_aux_loss_weight",
        ):
            flags.pop(k, None)
        flags.update(
            use_pathways=True,
            pathway_depths=[0, 2, 4],
            pathway_budget=2.0,
            pathway_aux_loss_weight=0.01,
            pathway_entropy_weight=0.001,
            pathway_floor_frac=0.1,
            pathway_floor_steps=2000,
            pathway_share_kv=True,
        )
        if arch == "swa_scribe_mop":
            flags["pathway_shared_bottom"] = max(1, (2 * L) // 3)
        else:
            stages = 2 if L <= 6 else 3
            flags.update(
                pathway_shared_bottom={5: 2, 6: 3, 8: 3, 9: 4}.get(L, max(2, L // 3)),
                pathway_variant="nested",
                pathway_stages=stages,
                pathway_stage_share_lanes=True,
            )
        # Bank inserts sit ABOVE every pathway stage insert so scribes/workspace
        # summarize post-routing states (no MoD gather here, so the block-
        # alignment rule does not apply; the ordering rule does).
        last_stage = flags["pathway_shared_bottom"] + int(flags.get("pathway_stages", 1)) - 1
        insert = min(L - 1, max(flags["scribe_write_layer"], last_stage + 1))
        flags["scribe_write_layer"] = insert
        flags["trunk_insert_layer"] = insert
        if arch == "swa_scribe_mopd_delta":
            # Fifth candidate arm (2026-07-20): the deltabank mechanism on the
            # sub-linear mopd backbone — cumulative chunk-writes into scribe's
            # 128-register bank (swa_use_registers is already on for scribe).
            # This composes the out-of-band-carry mechanism with the deep-target
            # backbone, so it confounds backbone + write-mechanism vs plain
            # scribe_mopd; read it only AFTER the clean hybrid32/deltabank pair
            # isolates whether the mechanism does anything. The write shares the
            # scribe/trunk insert layer (above every pathway stage; no MoD
            # gather here, so only the ordering rule applies). The delta read
            # replaces the static register-read branch on layers above it and
            # composes with scribe's register refinement (the write scans from
            # the refined bank via the model's _register_base).
            flags.update(
                use_delta_bank=True,
                delta_chunk_size=512,
                delta_write_layer=insert,
                delta_write_topk=16,
            )
        return flags
    if arch == "swa_scribe_cmod":
        # SIXTH candidate arm (2026-07-21): swa_scribe with a CAUSAL MoD router.
        # Identical banks, shape and backbone; the ONLY change is the selection
        # rule, so swa_scribe / swa_scribe_cmod / swa_scribe_mopd is a clean
        # three-way on the dynamic-compute mechanism at matched N.
        #
        # Why it exists: block-atomic MoD selects with a top-k over the WHOLE
        # sequence, which is an order statistic -- token t's fate depends on
        # tokens AFTER t, and under packed documents on its neighbours' documents
        # (measured 5.7e-2 doc-B sensitivity vs 0.0 with routing pinned). The
        # threshold rule "score + bias > 0" is per-token, hence strictly causal
        # and free of any cross-document channel; it is also the rule inference
        # ALREADY uses (mixture_of_depths.py, seq_len==1 branch), so train and
        # generation finally agree.
        #
        # Capacity holds only in expectation, so the gather pads to
        # mod_gather_bucket and bills realized fractions. Adds ONE scalar buffer
        # (route_bias) -- N is matched with swa_scribe to the parameter.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        flags.update(
            mod_router_mode="threshold",
            mod_route_bias_lr=1e-3,
            mod_gather_bucket=128,
        )
        return flags
    if arch == "swa_scribe_mopd_k":
        # Kernel-friendly twin of swa_scribe_mopd (2026-08-18, dyncompute perf
        # work): every lane tier gathers at a STATIC width — ceil(0.65*T) for
        # depth>=2, ceil(0.35*T) for depth>=4, budget-matched to 2.0 expected
        # blocks/token — with causal positional clipping (earliest routed
        # tokens keep their lane; overflow tokens skip the deep blocks). Buys
        # ONE gather shape per tier for the whole run (the argmax router's
        # per-step width churn is the leading recompile-pressure suspect on
        # mop-stack) and zero routing host syncs. The aux budget loss stays —
        # it keeps assignment rates near the static widths, minimizing
        # clip/pad waste. Functionally different — screened as its own arm.
        flags = dict(_sweep_arch_flags("swa_scribe_mopd", base))
        flags.update(pathway_fixed_tier_fracs=[0.65, 0.35])
        return flags
    if arch == "swa_scribe_partrunk":
        # Linearized-trunk twin of swa_scribe (2026-08-18, dyncompute perf
        # work): the 64-slot workspace is updated by ONE batched write from the
        # shared initial state + a doc-gated chunk-axis cumsum with the
        # delta-bank rms-budget squash, instead of the 7-round sequential
        # write->read recurrence. Removes the trunk's serial latency and its
        # data dependency (composes with per-layer compile). Functional change:
        # writes are conditioned on W0, not the evolved workspace. +1 param
        # (dev_scale). See ParallelWorkspaceTrunk.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        flags.update(trunk_variant="parallel")
        return flags
    if arch in ("swa_scribe_rglrutrunk", "swa_scribe_deltatrunk"):
        # Third/fourth points on the trunk-linearization design line
        # (2026-08-18, user-requested): partrunk's batched-write skeleton with
        # a GATED chunk-axis scan instead of the plain cumsum
        # (GatedScanWorkspaceTrunk). rglrutrunk = Griffin RG-LRU-gated decay —
        # the workspace can FORGET at a learned content-dependent rate (the
        # mechanism that matters at 32K, where 63 writes/doc saturate an
        # un-decayed sum's tanh budget). deltatrunk = delta-rule gated replace
        # — writing the same content twice is idempotent (true overwrite
        # semantics) where the cumsum doubles it. Speed shape identical to
        # partrunk (one batched write + a C-step elementwise scan); +O(h) gate
        # params (matched-N to partrunk within the gate, same tolerance as the
        # register-arm read gates). Functional changes — screened as own arms.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        flags.update(trunk_variant="rglru" if arch == "swa_scribe_rglrutrunk" else "delta_rule")
        return flags
    if arch == "swa_scribe_kmod":
        # Kernel-friendly twin of swa_scribe_cmod (2026-08-18, dyncompute perf
        # work): same causal threshold router, but the gather runs at a STATIC
        # width (prefix + ceil(content*capacity*1.1)) with causal positional
        # clipping — a row selecting more than the width keeps its EARLIEST
        # selections (inclusion at position p depends only on decisions <= p, so
        # the selection channel stays strictly causal). Buys: one gather shape
        # for the whole run (no recompile churn), zero host syncs on the routing
        # path, bounded padding (<= slack instead of per-batch row max). Cost:
        # rare overflow clipping (the bias controller keeps E[count] at
        # capacity, so clipping touches the tail of above-average rows only).
        # Functionally different from cmod — screened as its own arm.
        flags = dict(_sweep_arch_flags("swa_scribe_cmod", base))
        flags.update(
            mod_fixed_gather=True,
            mod_fixed_gather_slack=1.1,
        )
        return flags
    if arch == "dual_stream":
        # P1 `dual_stream` (2026-08-20, redesign round): the PARAMETER axis.
        # A narrow fast stream runs every token; every 4th layer a WIDE block
        # (2x width) runs over ~20% of tokens, self-attending among themselves
        # by original position and cross-attending into the register bank.
        # Registers are the bus: the fast stream reads them on every SWA layer,
        # the slow stream writes through them, so capacity routing and the
        # memory system are one module rather than two costs.
        # Zero-init write-back means step 0 is bitwise the fast stream alone.
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        flags.update(
            use_dual_stream=True,
            dual_stream_every=4,
            dual_stream_width_mult=2.0,
            dual_stream_frac=0.2,
            # 2026-08-25: WITHOUT this the router is dead. `route_bias` was a zero
            # buffer nothing updated, and the ar10m cell selected ZERO tokens from
            # step 2,485 to the end -- the slow stream's weights decayed to ~1e-4
            # and the arm computed nothing while paying its full +22% wall-clock.
            # Its 2.5681 was the +15% token budget its inflated N bought, not the
            # mechanism. Cells before this date are parked as `.deadrouter_*` and
            # their numbers must NOT be compared with anything produced after.
            dual_stream_route_bias_lr=1e-2,  # mat_ffn's rate, not MoD's 1e-3
        )
        return flags
    if arch == "dual_stream_kmax":
        # dual_stream with the slow stream's self-attention BANDED to the
        # previous 1024 selected tokens (2026-08-25). Same mechanism, same
        # params, same selection rule -- only the k x k mask narrows.
        #
        # At ar10m (S=4096, frac=0.2) k is 819 < 1024, so no band is built and
        # this arm is BITWISE `dual_stream` -- there is no 4K loss question to
        # ask, and no 4K cell is needed. It exists for 32K, where k reaches 6554
        # and the band cuts the slow stream's attention work 6.4x.
        #
        # 1024 is chosen to match `swa_window`: the slow stream gets the same
        # band budget as the fast stream, measured in selected slots rather than
        # raw positions.
        flags = dict(_sweep_arch_flags("dual_stream", base))
        flags.update(dual_stream_kmax=1024)
        return flags
    if arch == "dual_stream_pm":
        # PARAM-MATCHED dual_stream (2026-08-23). The ar10m round ran
        # dual_stream at TPR=100 on its OWN N, so it took 40.7M active params
        # and 4.07B tokens against swa_sinkdiet's 35.4M / 3.54B and won by
        # 0.038 nats -- a win bought with +15% params AND +15% tokens AND +22%
        # wall-clock, so it settled nothing.
        #
        # The mechanism is left exactly as designed (every=4, mult=2.0, frac=0.2
        # -> still 3 SlowStreamBlocks at layers 11/15/19 pre-shrink); the TRUNK
        # is shrunk instead, which is the honest form of the question: at a
        # fixed parameter budget, is a sparse wide slow stream a better use of
        # the params than more trunk? Shape override below puts active N at
        # 35.93M vs sinkdiet's 35.41M (+1.5%), and because the submitter derives
        # steps as TPR*N, matching N matches TOKENS to the same 1.5% for free.
        return dict(_sweep_arch_flags("dual_stream", base))
    if arch == "hourglass":
        # S1 `hourglass` (2026-08-20, redesign round): the sequence-resolution
        # axis. Thin full-length outer layers; the deep middle stack (where the
        # banks live) runs on S/4 learned segments. No token gather and no
        # scatter -- pooling is one segment-mean, unpooling one gather -- so it
        # sidesteps the bandwidth cost that dominates the stacker's backward,
        # and both attention (O(S*W) under the window) and the FFN fall
        # linearly with the pooling ratio.
        # Dynamism: the segment COUNT is fixed (static shapes, compile-friendly)
        # but the boundary POSITIONS are learned, so dense regions are
        # represented finely and boilerplate is compressed hard -- fixed total
        # compute, learned allocation, the same bargain MoE makes on params.
        # num_hidden_layers here counts ONLY the outer layers; the middle stack
        # is owned by HourglassMiddle and sized to keep total depth comparable.
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        outer = max(2, L // 3)
        flags.update(
            use_hourglass=True,
            hourglass_middle_layers=max(1, L - outer),
            hourglass_pool_ratio=4,
            hourglass_insert_layer=max(1, outer // 2),
        )
        for k in (
            "use_mod",
            "mod_capacity",
            "mod_atomic_block",
            "mod_block_size",
            "mod_io_blocks_dense",
            "mod_aux_loss_weight",
        ):
            flags.pop(k, None)
        # Flags are merged AFTER the shape in the registration loop, so this
        # wins: the model's own layer list holds only the OUTER layers. Total
        # depth (outer + middle) still equals the tier's L, so the arm is
        # depth-matched to its siblings -- only the resolution differs.
        flags["num_hidden_layers"] = outer
        return flags
    if arch == "chunk_mop":
        # D2 `chunk_mop` (2026-08-20, redesign round): mopr's shallow-spine
        # stack, routed once per 128-token CHUNK instead of once per token.
        # Amortizes the router ~128x and makes each tier's token set near-
        # contiguous, so the permute dispatch slices instead of gathering.
        #
        # EXPECT WORSE NATS THAN mopr. Measured against the trained
        # swa_scribe_mopd router before this arm was written: only 16% of the
        # per-token depth variance is between-chunk at 128 (13% at 256) against
        # a 30% bar set in advance, so a chunk router cannot express most of the
        # allocation the token router makes. Kept because it is a dispatch
        # change rather than a new architecture, and because it is the right
        # rescue if wall-clock turns out to dominate quality.
        # See docs/dyncompute_redesign.md sketch D2 for the table.
        flags = dict(_sweep_arch_flags("swa_scribe_mopr", base))
        flags.update(pathway_route_chunk=128)
        return flags
    if arch in ("think_suffix", "think_suffix_gqa"):
        # D1 `think_suffix` (2026-08-20, redesign round). The measured lesson of
        # rounds 1-7 is that ROUTING POINTS are the cost, not routing: ~30 ms of
        # fixed machinery per stage, so mopr's 11 stages spend ~330 ms before
        # doing any useful work. This arm keeps the dynamic depth but pays that
        # cost ONCE -- a dense backbone with no routing inside it, and a single
        # weight-tied block at the top that routed tokens iterate 0/1/2/4 times.
        #
        # NOT new machinery: it is Pathways with stages=1,
        # variant="recursive" (one weight-tied lane block, depth = iteration
        # count) inserted near the top, plus pathway_bank_export so each pass
        # reads the register bank inside attention the spine already runs --
        # "thinking longer" IS "consulting memory again". Reusing the stacker's
        # code means the permute dispatch, the compiled prologue/merge and the
        # shared-KV export all apply unchanged.
        #
        # Menu [0,1,2,4] with budget 1.75 == the menu mean: the menu-symmetry
        # rule (docs/arch_naming.md) says a random-init bench routes ~uniformly,
        # so an asymmetric budget would make the bench report a cost the trained
        # model never pays.
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        flags.update(_think_suffix_pathway_flags(L))
        if arch == "think_suffix_gqa":
            # Parity round's GQA lever (job 55748): the sinkdiet-family wall-clock
            # gap is MLA PROJECTION cost, not attention span -- banding the globals
            # to 8192 moved nothing at 32K while swapping MLA for GQA took
            # swa_sinkdiet from 1.202x dense to 1.037x. think_suffix sits on the
            # same 17 SWA + 6 MLA backbone, so the lever ports verbatim. It matters
            # more here than elsewhere: this arm ties the round lead on nats
            # (2.6069 vs swa_sinkdiet 2.6059) but costs ~1.21x dense in wall-clock,
            # so the loss is already paid for and only the clock is in the way.
            # N drops ~1.29M, so it is budgeted on think_suffix's N, not its own.
            flags.update(
                hybrid_attention_ratio=_ratio_final_global(
                    ["swa", "swa", "swa", "attn"], L, "attn"
                ),
            )
        for k in (
            "use_mod",
            "mod_capacity",
            "mod_atomic_block",
            "mod_block_size",
            "mod_io_blocks_dense",
            "mod_aux_loss_weight",
        ):
            flags.pop(k, None)
        return flags
    if arch == "mat_ffn":
        # W1 `mat_ffn` (2026-08-20, redesign round): per-TOKEN nested FFN width
        # on the minimal-bank backbone. Attention is UNTOUCHED, so this arm
        # keeps sinks + registers exactly as `swa_sinkdiet` has them and pays no
        # KV export, no merge and no depth-router machinery -- the dynamic range
        # (4x on ~2/3 of block FLOPs) comes entirely from the FFN.
        # Nested widths are prefixes of intermediate_size, so smaller tiers are
        # literally the head of the big FFN and params equal the dense model's.
        # Menu {1/4, 1/2, 1} with budget = the menu mean keeps a random-init
        # bench honest (same rule as the MoP depth menu).
        # mat_ffn_widths is deliberately left None: the config derives
        # {1/4, 1/2, 1} x intermediate_size at build time. Baking absolute
        # widths here would hard-fail the moment anything overrides the shape
        # (a bench sweep, a size probe) -- which the validator caught.
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        flags.update(
            use_mat_ffn=True,
            mat_ffn_budget=0.0,  # 0 -> mean of the menu (menu symmetry)
            mat_ffn_aux_weight=0.01,
        )
        return flags
    if arch in _MAT_FFN_NEST or any(
        arch.endswith(sfx) and arch[: -len(sfx)] in _MAT_FFN_NEST
        for sfx in ("_gqa", "_delta", "_delta_mla", "_delta_gqa")
    ):
        return _mat_ffn_nest_flags(arch, base)
    if arch in ("mat_ffn_deltamem", "mat_ffn_gla", "mat_ffn_ema"):
        # W1 + the delta-rule side memory (2026-08-21). These two arms sit on the
        # SAME `swa_sinkdiet` backbone and change orthogonal things -- `mat_ffn`
        # varies FFN WIDTH per token (capacity per token, inside the block),
        # `swa_deltamem` adds an associative memory BESIDE attention (long-range
        # recall, on the residual). Neither touches the other's tensors, so the
        # composition is the two flags together and nothing else; that is exactly
        # why it is worth a cell. mat_ffn is currently tied with dense on loss at
        # ar10m while spending ~62% of FFN width, i.e. it has bought headroom and
        # not yet spent it on anything; deltamem is the cheapest mechanism we have
        # for the axis mat_ffn cannot touch.
        #
        # Both parents are exact backbone controls: mat_ffn_deltamem vs mat_ffn
        # isolates the memory, vs swa_deltamem isolates the width router, and both
        # reduce to swa_sinkdiet. DeltaMemory's o_proj is zero-init, so at step 0
        # this arm is BITWISE mat_ffn.
        #
        # THREE VARIANTS OF THE SAME SLOT, so the comparison is the recurrence
        # and nothing else -- identical projections, identical head count, width
        # and interval, identical zero-init read, same backbone:
        #   mat_ffn_deltamem  "delta"  overwrite at a key (Widrow-Hoff). Most
        #                              expressive, most expensive: a triangular
        #                              solve per chunk.
        #   mat_ffn_gla       "gla"    decay + accumulate. Same matrix state, no
        #                              solve (two matmuls), but it can only fade
        #                              what it holds, never overwrite one key.
        #   mat_ffn_ema       "ema"    per-head EMA of values. VECTOR state, no
        #                              key at all, so it is not associative --
        #                              it is the control that says whether the
        #                              win (if any) comes from RECALL or merely
        #                              from having some long-range channel.
        # Reading them together is the point: gla vs delta prices erasure, ema
        # vs both prices associativity itself.
        flags = dict(_sweep_arch_flags("mat_ffn", base))
        flags.update(
            use_delta_memory=True,
            delta_mem_variant={
                "mat_ffn_deltamem": "delta",
                "mat_ffn_gla": "gla",
                "mat_ffn_ema": "ema",
            }[arch],
            delta_mem_heads=4,
            delta_mem_head_dim=64,
            delta_mem_chunk=64,
            delta_mem_interval=4,
        )
        return flags
    if arch in (
        "swa_moba",
        "swa_hils",
        "swa_bsa_switch",
        "swa_deltamem",
        "swa_hils_deltamem",
    ):
        # Round 11. Every one of these is `swa_sinkdiet` (logit sinks +
        # registers, no scribes, no MoD) PLUS one long-range mechanism, so the
        # sinkdiet cell is the exact backbone control for all of them and any
        # gap is attributable to the mechanism rather than to the memory system.
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        if arch in ("swa_moba", "swa_hils", "swa_bsa_switch", "swa_hils_deltamem"):
            flags.update(
                use_block_sparse_attn=True,
                bsa_mode=("moba" if arch == "swa_moba" else "hils"),
                bsa_block_size=128,
                bsa_top_k=4,
                # Selection is a routing decision, and our own numbers price a
                # routing point at ~30 ms/stage with a large fixed term. Sharing
                # one selection across a query BLOCK already amortises it over
                # 128 tokens; the layer interval amortises it again over depth.
                bsa_layer_interval=2,
            )
        if arch == "swa_bsa_switch":
            # InfLLM-V2's point: trainable sparsity is a net loss at short
            # context and a win at long. Below 8K this arm IS its backbone.
            flags["bsa_min_seq"] = 8192
        if arch in ("swa_deltamem", "swa_hils_deltamem"):
            flags.update(
                use_delta_memory=True,
                delta_mem_variant="delta",
                delta_mem_heads=4,
                delta_mem_head_dim=64,
                delta_mem_chunk=64,
                delta_mem_interval=4,
            )
        return flags
    if arch == "swa_sinkdiet":
        # P2 `sink_diet` (2026-08-20, redesign round) -- the MINIMAL-BANK
        # control, and the baseline every dynamic-compute candidate must beat.
        # Keeps exactly the two mechanisms the project requires and drops the
        # rest:
        #   sinks     -> learned PER-HEAD LOGITS (use_logit_sinks): zero K/V,
        #                zero sequence positions, no prefix, so every SWA layer
        #                stays FA2-varlen-native at O(S). Replaces the 16-token
        #                sink bank.
        #   registers -> the 128-slot bank, kept.
        #   scribes / workspace trunk -> OFF. Their joint keys grow with S, so
        #                they earn their cost at long context, not in the 4K
        #                phase that dominates wall-clock. Turn them back on for
        #                a long-context variant rather than paying for them here.
        # No dynamic compute by design: this arm prices the BANKS so that any
        # candidate's gap can be split into "banks" vs "dynamism".
        # MoD is dropped here, for two reasons that happen to coincide.
        # (1) SPEC: P2 is meant to have NO dynamic compute -- it is the control
        #     that prices the BANKS so every other candidate's gap can be split
        #     into "banks" vs "dynamism". Inheriting swa_scribe's block-atomic
        #     MoD would have made the control itself a dynamic-compute arm.
        # (2) CAUSALITY: MoD routes by top-k, a whole-sequence order statistic,
        #     so a token's routing depends on its future. Measured with a
        #     multi-position perturbation probe, every use_mod arm leaks
        #     (swa_scribe itself 1/6 probe positions, sinkdiet 5/6 before this
        #     change) while the MoD-free arms are 0/6. That leak is pre-existing
        #     and accepted for the incumbents -- cmod/kmod are the causal twins
        #     -- but there is no reason to inherit it into new arms that carry
        #     their own dynamic-compute mechanism. It also matches the rule the
        #     config validator already enforces for pathways: one dynamic-compute
        #     mechanism per arm keeps attribution clean.
        flags = dict(_sweep_arch_flags("swa_scribe", base))
        flags.update(
            swa_sink_count=0,
            use_logit_sinks=True,
            use_scribe_bank=False,
            use_workspace_trunk=False,
        )
        for k in (
            "use_mod",
            "mod_capacity",
            "mod_atomic_block",
            "mod_block_size",
            "mod_io_blocks_dense",
            "mod_aux_loss_weight",
        ):
            flags.pop(k, None)
        for k in (
            "scribe_slots_per_chunk",
            "scribe_chunk_size",
            "scribe_write_layer",
            "scribe_write_topk",
            "trunk_layers",
            "trunk_chunk_size",
            "trunk_iters_min",
            "trunk_iters_max",
            "trunk_insert_layer",
            "trunk_workspace_slots",
            "trunk_write_topk",
        ):
            flags.pop(k, None)
        return flags

    # ---------------------------------------------------------------- parity arms
    # PARITY ROUND (2026-08-25). swa_sinkdiet trails dense by ~11-14% at 4K. A
    # decomposition of that gap says it is NOT a kernel-path penalty -- sinkdiet's
    # SWA layers are FA2-varlen-native (registers, not a prefix; sinks via the free
    # LSE merge). The cost is in two places, and each arm below removes one so the
    # split can be measured rather than argued:
    #
    #   (a) the REGISTER READ, which runs on every SWA layer (an extra q_proj plus
    #       an S x 128 attention, per token and independent of key count)
    #           -> swa_sinkdiet_bri2 / _bri4 thin it to every 2nd / 4th layer.
    #   (b) the 6 FULL-ATTENTION MLA layers, which carry the whole O(S^2) term AND
    #       are individually heavier than a dense layer (MLA runs a 192-dim QK
    #       against dense's head_dim 64)
    #           -> swa_sinkdiet_g8    keeps MLA but halves how many layers are global
    #           -> swa_sinkdiet_gwin8 keeps 6 globals but bands them at 8192
    #           -> swa_sinkdiet_gqa   keeps 6 unbounded globals but makes them GQA
    #
    # Each is a single-knob delta from swa_sinkdiet, so the arms are additive and a
    # winner can be composed. All keep sinks and registers (project requirement).
    if arch in (
        "swa_sinkdiet_bri2",
        "swa_sinkdiet_bri4",
        "swa_sinkdiet_g8",
        "swa_sinkdiet_gwin8",
        "swa_sinkdiet_gqa",
    ):
        flags = dict(_sweep_arch_flags("swa_sinkdiet", base))
        L = int(base["num_hidden_layers"])
        if arch == "swa_sinkdiet_bri2":
            flags.update(swa_bank_read_interval=2)
        elif arch == "swa_sinkdiet_bri4":
            flags.update(swa_bank_read_interval=4)
        elif arch == "swa_sinkdiet_g8":
            # 1-in-8 instead of 1-in-4: 3 global layers of 23 rather than 6.
            flags.update(
                global_attention_every_n_layers=8,
                hybrid_attention_ratio=_ratio_final_global(["swa"] * 7 + ["mla"], L, "mla"),
            )
        elif arch == "swa_sinkdiet_gwin8":
            # Same 6 global layers, but banded at 8192 instead of unbounded. At the
            # 4K bench this is INERT by construction (8192 > 4096, so no key is
            # excluded) -- it exists to move the 32K number, and the 4K cell is its
            # own control.
            flags.update(global_layer_window=8192)
        elif arch == "swa_sinkdiet_gqa":
            flags.update(
                hybrid_attention_ratio=_ratio_final_global(
                    ["swa", "swa", "swa", "attn"], L, "attn"
                ),
            )
        return flags

    if arch == "swa_scribe":
        # Four-bank FA2-native arm (2026-07-19, user-designed): 16-token sink
        # bank joined to the FA2 window softmax by an EXACT LSE merge (sink +
        # joint softmax without the prefix kernel path), 128 registers with
        # one bidirectional zero-init refinement per forward, 4 scribe slots
        # per 512-token chunk (content-written at ONE block-aligned layer,
        # top-16-sparse write), and a 64-slot write-sparse workspace trunk.
        # Budget: 16+128+4+64 = 212 x d static bank tokens vs incumbents' 256;
        # extra attention ~ O(S*(16+C*4)) joint keys + O(S*128) register read
        # ~= 17% of the window's score FLOPs. Insert layers MUST be multiples
        # of mod_block_size (dead-trunk post-mortem 2026-07-19).
        _safe_insert = ((L // 2 + 3) // 4) * 4
        flags = dict(_sweep_arch_flags("swa_globals", base))
        flags.update(
            swa_use_registers=True,
            swa_global_count=128,  # ALL registers — no prepended prefix
            swa_register_refine=True,
            swa_sink_count=16,
            use_scribe_bank=True,
            scribe_slots_per_chunk=4,
            scribe_chunk_size=512,
            scribe_write_layer=_safe_insert,
            scribe_write_topk=16,
            use_workspace_trunk=True,
            trunk_layers=1,
            trunk_chunk_size=512,
            trunk_iters_min=1,
            trunk_iters_max=1,
            trunk_insert_layer=_safe_insert,
            trunk_workspace_slots=64,
            trunk_write_topk=16,
        )
        return flags
    if arch.startswith("swa_globals"):
        # MoD + sliding-window attention + persistent globals + MLA (3:1).
        # MoD is block-atomic: the 4-layer block (3 SWA + 1 MLA) is routed as a
        # unit (one decision, one residual gate across the block), matching the
        # stacking-plan design.
        #
        # io_blocks_dense MUST stay True. It was briefly set False so that MoD
        # would fire on the old 5-6 layer sweep shapes (with block_size=4 those
        # give <=2 blocks, and dense I/O blocks would leave none to route). That
        # is a trap: with no dense block, EVERY layer sits in a routable block, so
        # a token the router declines in every block traverses *zero* transformer
        # layers — embedding straight to the final norm and the LM head. Measured
        # on a trained 25m checkpoint: 24.2% of tokens saw no layer at all, and
        # mean CE by blocks-visited was 6.38 (0 blocks) / 6.31 (1) / 5.08 (2).
        # That is the whole ~1.4-nat deficit swa_globals showed in the screen.
        #
        # The fix is depth, not routing: these archs get their own shapes (see
        # _SWEEP_ARCH_SHAPES) with L = 4*(R+2), giving R routed blocks between a
        # dense first and last block. Guarded in UMichSLMModel.__init__.
        flags = {
            "use_mla": True,
            "attention_mode": "swa_globals_mla",
            "hybrid_attention_ratio": _ratio_final_global(["swa", "swa", "swa", "mla"], L, "mla"),
            "sliding_window": 1024,
            "swa_window": 1024,
            "swa_global_count": 256,
            "mla_kv_lora_rank": max(64, h // 2),
            "use_mod": True,
            "mod_capacity": 0.5,
            "mod_atomic_block": True,
            "mod_block_size": 4,
            "mod_io_blocks_dense": True,
            "mod_aux_loss_weight": 0.01,
            # FlexAttention BlockMask for the compact MoD gather (SWA members;
            # MLA member keeps SDPA+causal mask). Verified parity + leakage
            # (tests/test_mod_intra_doc.py::TestFlexCompactEndToEnd) and A/B
            # 2026-08-18 (A5000, intra-doc ON, compile layers): cmod 330->239 ms
            # (-27%, peak 4.99->4.15 GB), scribe 347->293 ms (-16%), kmod
            # 315->283 ms. Inherited by every arm that copies these flags; inert
            # off-CUDA or when flex is unavailable.
            "mod_flex_attention": True,
        }
        if arch == "swa_globals_mtp":
            # Multi-Token Prediction on the primary arch. MTP heads are training-
            # only (discarded at inference), so this is a pure training-time bet:
            # it must pay for its extra FLOPs within the same IsoFLOP budget.
            flags.update({"use_mtp": True, "mtp_num_heads": 1, "mtp_share_head": True})
        if arch == "swa_globals_ple":
            # Gemma-style Per-Layer Embeddings on the primary arch. PLE is a sparse
            # lookup (no matmul), so it adds capacity at ~zero FLOP cost — the same
            # trade that is currently carrying `gemma` in the screen. Applied to the
            # gathered compact sequence inside each MoD block (see
            # _run_gathered_mod_block); without that it would receive no gradient.
            flags.update({"use_ple": True, "ple_dim": 256})
        return flags
    if arch == "deltanet_mla":
        # PRETRAIN_PLAN's core architecture: 3:1 Gated DeltaNet / MLA hybrid, true
        # delta rule. Distinct from qwen3_5_like, whose 4th layer is plain GQA
        # rather than MLA. Note: block-atomic MoD is NOT enabled — gather/scatter
        # cannot skip tokens in a recurrent delta layer without corrupting its state.
        return {
            "use_mla": True,
            "use_deltanet": True,
            "deltanet_variant": "gated",
            "deltanet_true_delta_rule": True,
            "attention_mode": "deltanet_mla",
            "hybrid_attention_ratio": _ratio_final_global(
                ["delta", "delta", "delta", "mla"], L, "mla"
            ),
            "mla_kv_lora_rank": max(64, h // 2),
        }
    if arch == "qwen3_like_mtp":
        # MTP on the dense GQA baseline. This is the CONTROL for swa_globals_mtp:
        # without it, an MTP gain on swa_globals can't be separated from an
        # interaction with block-atomic MoD + SWA.
        return {"use_mtp": True, "mtp_num_heads": 1, "mtp_share_head": True}
    if arch == "smollm3_like":
        # SmolLM3: dense GQA + NoPE (position encoding stripped from every 4th
        # layer) + tied embeddings. Tying matters a lot here — at vocab 56000 the
        # untied lm_head is ~28.7M params, a third of the whole model.
        return {"nope_every_n_layers": 4, "tie_word_embeddings": True}
    if arch == "gemma":
        # Gemma-3n/4 style: alternating local/global SWA (4:1), zero-centered
        # RMSNorm, Per-Layer Embeddings + MatFormer nested FFNs.
        return {
            "sliding_window": 512,
            "global_attention_every_n_layers": 4,
            "use_zero_centered_rmsnorm": True,
            "use_ple": True,
            "ple_dim": 256,
            "use_matformer": True,
            "matformer_granularities": [f // 2, f],
        }
    raise ValueError(f"Unknown sweep arch: {arch}")


# --------------------------------------------------------- retired sweep arms
# Arms that must NOT be given new sweep cells. They stay in SWEEP_ARCHS and stay
# constructible on purpose: the banked cells have to remain reproducible, and
# the collapse diagnostics build these very configs. What is retired is spending
# NEW compute on them.
#
# THE NEST ROUTER COLLAPSE (2026-08-26 decision; measured 2026-08-22 onward).
# Tier k's only LM gradient reaches the router through the straight-through gate
# `tail[:, k]`, which both dispatch paths multiply by the hard mask
# `(choice >= k)`. A tier nobody selects therefore contributes exactly zero
# gradient, so ZERO OCCUPANCY IS AN ABSORBING STATE -- a router that stops
# choosing a tier can never learn to start. Measured on the live skew4 ar10m
# cell at step 6246: the widest tier sat at exactly 0.0000 occupancy in all five
# routed layers, trailing the argmax by 3.0-8.0 logits, with E[w] 390-447
# against a 768 target -- the arm pays 3072-wide GEMMs (tier slices are sized
# from the PRIOR, not from demand) and delivers ~52-58% of the width it is
# priced at. The KL aux, the only escape, contributes 2.5e-4 to 3.8e-3 nats
# against a 0.036-nat seed-noise bar.
#
# The eps exploration floor did NOT fix it: eval mean width was 394 at eps=0.4
# vs 388 at eps=0.1 vs 388 at baseline. It manufactures TRAIN occupancy the
# router has not learned, so the `*_explore` arms are retired too.
#
# `mat_ffn_thresh` replaces argmax with an adaptive threshold. Its FIRST
# controller (post-clip, every cell before 2026-09-18) hit occupancy 767.9/768
# but ratcheted theta to 0, so the clip granted width by POSITION, not content
# (see RoutedMatFormerSwiGLUMLP `route_ctrl`). The "wish" controller -- feedback
# on the routed pre-clip share -- replaces it; the interim "quantile" controller
# fixed the positional routing but under-filled the budget (toy: 643/768 granted). It is the only mat_ffn arm that may take new cells.
RETIRED_ARCHS: frozenset = frozenset(
    {
        "mat_ffn",
        "mat_ffn_flat4",
        "mat_ffn_skew3",
        "mat_ffn_skew4",
        "mat_ffn_skew4_nowarm",
        "mat_ffn_flat4_explore",
        "mat_ffn_skew3_explore",
        "mat_ffn_skew4_explore",
        # Side-memory arms sit on the collapsed `mat_ffn` base, so they inherit a
        # router that under-uses the nest. They were to be rebased on
        # mat_ffn_thresh; that is now moot, see below.
        "mat_ffn_deltamem",
        "mat_ffn_gla",
        "mat_ffn_ema",
        # NAMING, because this set was once widened by reading one: on
        # 2026-08-28 the instruction "we never want to run mat_ffn" was applied
        # to `mat_ffn*` as a family and added `mat_ffn_thresh` and
        # `mat_ffn_thresh_gqa` here (318a5f7). That was wrong -- `mat_ffn` is a
        # specific arm, not a prefix -- and both were removed again the same day.
        # A slug in this set must be a slug someone actually named.
        #
        # The threshold arms are NOT retired. Threshold routing fixed the
        # occupancy collapse that retired the arms above (mean width 767.9
        # against a 768 budget, eval == train) -- but, as shipped until
        # 2026-09-18, by granting width by position rather than content; every
        # result below measured that positional schedule. They are unattractive on the
        # evidence so far -- at ar10m mat_ffn_thresh landed val_lm_loss 2.6381
        # against dense 2.6321 at 1.21x dense per step, and at ar25m
        # mat_ffn_thresh_gqa landed 2.4058 against swa_sinkdiet 2.3603, 0.045
        # behind and past the 0.036 bar -- but "unattractive" is a ranking
        # question, not a ban, and the wall-clock half of that case was measured
        # at 4096, which does not transfer to 32K (the arm runs 1.083x dense at
        # 4096 against 0.933x at 32768). Gate them on results, not on this set.
        #
        # `mat_ffn_thresh_gqa` carries an unfixed data-dependent grad-ckpt
        # CheckpointError at S=8192 (job 56494); 4096/16384/32768 all pass.
    }
)


SWEEP_ARCHS: tuple = (
    "vanilla",
    "qwen3_like",
    "qwen3_5_like",
    "mod",
    "swa_globals",
    "gemma",
    # --- added arms ---
    "deltanet_mla",  # PRETRAIN_PLAN's core 3:1 GatedDeltaNet/MLA hybrid
    "smollm3_like",  # dense GQA + NoPE every 4th layer + tied embeddings
    "swa_globals_mtp",  # primary arch + Multi-Token Prediction
    "qwen3_like_mtp",  # MTP control on the dense baseline
    # --- training-time compute bets (2026-10-07) ---
    # Both suffixes are orthogonal to the backbone, so they are registered on
    # all four live screening arms rather than only on the dense baseline.
    # `_pause` needs an attention module that accepts a caller-supplied query
    # (dense GQA and the registers/scribe SWA family do; MLA and the recurrent
    # variants do not and the block raises).
    "qwen3_like_fbt",  # latent feedback: prev token's top-layer state -> next input
    "qwen3_like_rec",  # router-free looped middle block, FLOP-matched to qwen3_like (ar*)
    "swa_sinkdiet_gqa_rec",  # same loop on the sink+register SWA arm, FLOP-matched to it
    # loop-boundary mixers on the _rec twins (control: <parent>_rec)
    "qwen3_like_rec_inject",  # + Parcae stable injection (contractive A_bar, ZOH)
    "qwen3_like_rec_hc",  # + Hyperloop per-loop hyper-connections, 4 streams
    "qwen3_like_rec_inject_hc",  # + both
    "swa_sinkdiet_gqa_rec_inject",
    "swa_sinkdiet_gqa_rec_hc",
    "swa_sinkdiet_gqa_rec_inject_hc",
    "qwen3_like_pause",  # state/prediction separation, "free pause token"
    "swa_sinkdiet_gqa_fbt",
    "swa_sinkdiet_gqa_pause",
    "think_suffix_gqa_fbt",
    "think_suffix_gqa_pause",
    "mat_ffn_thresh_gqa_fbt",
    "mat_ffn_thresh_gqa_pause",
    "swa_globals_ple",  # primary arch + Gemma-style Per-Layer Embeddings
    # --- depth axis on the primary arch (see _SWEEP_ARCH_SHAPES) ---
    "swa_globals_deep",  # L=16 -> 2 routed blocks
    "swa_globals_deeper",  # L=20 -> 3 routed blocks
    # --- dynamic-compute arms (dyncompute chunk 1) ---
    "moda",  # Arm C: Mixture-of-Depths Attention (quality arm, dense-GQA backbone)
    "mod_mlp",  # Arm A: FFN-only routing on the dense-GQA backbone
    "swa_globals_mlp",  # Arm A on the primary backbone (block-atomic decision, FFN-only gate)
    "deltanet_mla_mlp",  # Arm A on the recurrent hybrid (F3: FFN routing is recurrent-safe)
    "swa_globals_rec",  # Arm B: router-free recurrent depth (no MoD; unique-N reporting)
    "swa_globals_dense",  # control: primary backbone with no dynamic compute (twin for A/B)
    # --- novel dynamic-compute arms (dyncompute chunk 2) ---
    "gwt",  # Pitch 1: Global Workspace Trunk (live workspace + deep chunk-recurrent trunk)
    "mop",  # Pitch 2: Mixture of Pathways (depth-specialized lanes, one causal commitment)
    "mop_dense",  # MoP twin: single standard lane, no routing choice
    # --- MoP parameter-sharing ladder (each step changes exactly one thing) ---
    "mop_kv",  # + shared K/V read of the export (common read, lanes still disjoint)
    "mop_nested",  # + lanes as prefixes of one stack (shared low/intermediate blocks)
    "mop_rec",  # + weight-tied block, depth = iteration count (max sharing, MoR-flavored)
    # --- stacked MoP (MoP x blocked-MoD, the deep-target topology) ---
    "mop_stack",  # S stages of nested lanes between dense I/O stacks, global budget
    "mop_stack_shared",  # + one lane set shared across all stages (sub-linear params)
    # --- kernel-friendly attention twin ---
    "swa_registers",  # primary arch with the globals read as registers (FA2-native SWA)
    "swa_hybrid",  # half prefix globals (sink + MLA visibility) + half register bank
    # --- writable-workspace bank arms ---
    "swa_ws",  # no prefix; 256-slot WRITABLE workspace (twin of swa_registers' static bank)
    "swa_globals_ws",  # full 256 prefix + 128-slot writable workspace
    "swa_hybrid_ws",  # tri-bank: 128 prefix + 128 registers + 128-slot workspace
    "swa_scribe",  # 16-sink LSE merge + refined registers + scribes + sparse workspace
    "swa_scribe_mop",  # same four banks on a plain MoP (pathways) backbone
    "swa_scribe_mopd",  # same four banks on stacked block-MoP w/ shared lanes
    # --- larger-size candidate arms (2026-07-20 down-select) ---
    "swa_hybrid32",  # sink-minimal hybrid: 32 prefix + 128 registers
    "swa_deltabank",  # hybrid32 + cumulative chunk-writes into the register bank
    "swa_scribe_mopd_delta",  # deltabank mechanism on the sub-linear mopd backbone
    "swa_scribe_cmod",  # swa_scribe + causal (threshold) MoD router -- sixth arm
    "swa_scribe_kmod",  # cmod w/ static-width causal-clip gather (kernel-friendly twin)
    "swa_scribe_partrunk",  # swa_scribe w/ linearized (batched) workspace trunk
    "swa_scribe_rglrutrunk",  # partrunk skeleton + RG-LRU-gated chunk scan (forgetting)
    "swa_scribe_deltatrunk",  # partrunk skeleton + delta-rule gated replace (idempotent)
    "swa_scribe_mopd_k",  # mop-stack w/ static per-tier lane widths (causal clip)
    "swa_scribe_mopr",  # shallow-spine stacked MoP: depth menu carries the depth (min=spine)
    "swa_scribe_moprec",  # mopr geometry w/ ONE weight-tied lane block (MoR: depth=iterations)
    "swa_scribe_mopr_be",  # mopr + bank-export register read (functional; -98ms/step at eff54)
    "swa_scribe_mopr_b1",  # mopr at pathway_budget 1.0 (budget-curriculum low end)
    "swa_sinkdiet",  # P2: logit sinks + registers only (minimal-bank control)
    # --- parity round (2026-08-25): single-knob deltas from swa_sinkdiet that
    # each remove one component of its ~11-14% deficit vs dense at 4K.
    "swa_sinkdiet_bri2",  # register read every 2nd SWA layer
    "swa_sinkdiet_bri4",  # register read every 4th SWA layer
    "swa_sinkdiet_g8",  # 1-in-8 global layers instead of 1-in-4
    "swa_sinkdiet_gwin8",  # 6 globals kept, banded at 8192 (inert at S<=8192)
    "swa_sinkdiet_gqa",  # 6 globals kept unbounded, but GQA instead of MLA
    "dual_stream_kmax",  # slow stream self-attn banded to 1024 selected slots
    # --- Round 11 (2026-08-20): SUBSTITUTIVE long-range + non-bank memory. ---
    # Every arm above answers "what happened outside my window?" with a BANK
    # whose cost is ADDITIVE and, for scribes, grows with S (3.60x dense at
    # 32K). These four answer it with mechanisms whose far-field cost is flat
    # or linear in S. All keep the required sinks + registers.
    "swa_moba",  # top-k block selection, parameter-less gate (2502.13189)
    "swa_hils",  # learned chunk keys, retrieval score inside the softmax (2607.02980)
    "swa_bsa_switch",  # hils, but far branch OFF below bsa_min_seq (2509.24663)
    "swa_deltamem",  # delta-rule associative memory as a side branch (no bank)
    "swa_hils_deltamem",  # both: block-sparse far field + associative memory
    "mat_ffn",  # W1: per-token nested FFN width (matryoshka) on the sinkdiet backbone
    # W1 nesting round (2026-08-22, docs/mat_ffn_nesting.md): matched EXPECTED
    # width, varying nest depth + tier prior. flat4 is the uniform control.
    "mat_ffn_flat4",
    "mat_ffn_skew3",
    "mat_ffn_skew4",
    "mat_ffn_skew4_nowarm",
    "mat_ffn_skew4_explore",
    "mat_ffn_thresh",
    # The GQA lever (parity job 55748) on the ONE nest arm whose router actually
    # allocates. Every other nest arm collapses to the narrowest tier and cannot
    # recover -- zero occupancy is an absorbing state, and the eps floor never
    # fixed it (eval mean width 388 at eps=0.1 vs 388 at baseline) -- so a GQA
    # variant of one would only make a dead router faster. 2026-08-26.
    "mat_ffn_thresh_gqa",
    "mat_ffn_thresh_delta",  # same nest on the 3:1 DeltaNet/GQA hybrid (qwen3_5_like layout)
    "mat_ffn_thresh_delta_mla",  # same nest on the 3:1 DeltaNet/MLA hybrid (deltanet_mla layout)
    # 3:1 DeltaNet/GQA with only the attention slots swapped to mat_ffn_thresh_gqa's kind:
    # GQA + routed nest + logit sink + register read; DeltaNet layers plain (2026-10-27)
    "mat_ffn_thresh_delta_gqa",
    "deltanet_mla_sink",  # deltanet_mla with a learned per-head logit sink on its MLA layers
    "deltanet_mla_negeig",  # deltanet_mla with beta in (0, 2): negative-eigenvalue delta rule
    "deltanet_mla_kda",  # deltanet_mla with Kimi Delta Attention (per-key-channel decay)
    "deltanet_mla_think",  # deltanet_mla + think_suffix's sink, bank and recursed top block
    "deltanet_mla_pause",  # deltanet_mla + the free pause token (prediction stream)
    # Pre-hero layer recipe and depth twins (2026-10-31, roadmap Phases 1-2)
    "deltanet_mla_attngate",  # + G1 output gate on the MLA layers
    "deltanet_mla_dngate",  # + Gated DeltaNet output norm and gate
    "deltanet_mla_gates",  # + both output gates
    "deltanet_mla_qknorm",  # + per-head QK RMSNorm on the MLA layers
    "deltanet_mla_nope",  # NoPE MLA layers (Kimi Linear); position from DeltaNet only
    "deltanet_mla_posjit",  # trained with randomized position gaps (x8) on half the rows
    "deltanet_mla_cp27",  # CompleteP residual branch scale 27/L
    "deltanet_mla_lns",  # LayerNorm scaling 1/sqrt(layer_idx + 1)
    "deltanet_mla_graded",  # 7:1's MLA count, 3:1 near the input and output
    "qwen3_like_nmatch",  # qwen3_like with FFN widened to deltanet_mla's N
    "deltanet_mla_mtp",  # + one MTP head (render late with training.mtp_phase_start_frac)
    "deltanet_mla_softmtp",  # deltanet_mla_mtp trained as Soft-MTP (KL-routed gradient)
    "qwen3_like_softmtp",  # qwen3_like_mtp trained as Soft-MTP: the dense twin of the P10 pair
    "deltanet_mla_rwkv7",  # RWKV-7 time-mix in the DeltaNet slots (FLA chunk_rwkv7)
    # Agentic innovation program (2026-11-04, roadmap): P11, P4, P7, P8
    "deltanet_mla_sharedrope",  # one decoupled RoPE key shared by all MLA heads
    "deltanet_mla_diff",  # differential attention on the MLA layers
    "deltanet_mla_dnexpk",  # DeltaNet key dim x2 (bigger recurrent state)
    "deltanet_mla_dnwide",  # one wide DeltaNet head (state h^2 per layer)
    "deltanet_mla_engram",  # + Engram n-gram lookup tables (~25% of N, ~0 FLOPs)
    # Message structure on the MLA layers (2026-11-04, FABLE review P1/P2)
    "deltanet_mla_turnpos",  # MLA rope on (offset-in-message, message index) + pinned keys
    "deltanet_mla_digest",  # per-message digest slots; MLA = 2048 raw window + digests
    "deltanet_mla_turnpos_digest",  # both
    "deltanet_mla_pause_call",  # free pause stream read only at Harmony decision tokens (P12)
    "qwen3_5_like_prope",  # qwen3_5_like with partial RoPE (1/4 of head_dim) on its GQA layers
    "qwen3_5_like_nope",  # qwen3_5_like with NoPE GQA layers (position from DeltaNet only)
    "qwen3_5_like_kda_prope",  # qwen3_5_like_prope with Kimi Delta Attention in its DeltaNet layers
    "deltanet_mla_r7",  # deltanet_mla at 7:1 DeltaNet:MLA (final layer MLA)
    "deltanet_mla_r15",  # deltanet_mla at 15:1 DeltaNet:MLA (final layer MLA)
    # flat4 geometry under the same working router: the nest-shape question the
    # 2026-08-22 round could not answer because every arm in it collapsed.
    "mat_ffn_thresh_flat4_gqa",
    "mat_ffn_flat4_explore",
    "mat_ffn_skew3_explore",
    "mat_ffn_deltamem",  # W1 + delta-rule side memory (orthogonal axes, same backbone)
    "mat_ffn_gla",  # W1 + gated-linear-attention side memory (lighter: no solve)
    "mat_ffn_ema",  # W1 + per-head value EMA (lightest: vector state, not associative)
    "think_suffix",  # D1: dense backbone + ONE recursed top block (single routing point)
    "think_suffix_gqa",  # D1 + the parity round's GQA lever on the 6 global layers
    "chunk_mop",  # D2: mopr routed once per 128-token chunk (locality bet; see redesign doc)
    "hourglass",  # S1: learned dynamic pooling; deep middle stack runs at S/4
    "dual_stream",  # P1: narrow fast stream + wide sparse slow stream (registers as bus)
    "dual_stream_pm",  # P2: dual_stream with the trunk shrunk to sinkdiet's N (and D)
    # --- MTP twins of the five candidates (training-time bet; see _sweep_arch_flags) ---
    "swa_hybrid32_mtp",
    "swa_deltabank_mtp",
    "swa_scribe_mtp",
    "swa_scribe_mopd_mtp",
    "swa_scribe_mopd_delta_mtp",
    # --- bag-of-next-K lookahead twins (order-free planning aux) ---
    "swa_hybrid32_bag",
    "swa_deltabank_bag",
    "swa_scribe_bag",
    "swa_scribe_mopd_bag",
    "swa_scribe_mopd_delta_bag",
    # --- bank-reconstruction twins. WRITTEN-bank arms only: on a static bank
    # the state is the same for every chunk, so the target is unpredictable by
    # construction and the model raises at build time. ---
    "swa_deltabank_recon",
    "swa_scribe_mopd_delta_recon",
)

# Per-arch shape overrides, layered over _SWEEP_BASE_SHAPES[size].
#
# The swa_globals family cannot use the shared backbone shape. Block-atomic MoD
# with dense I/O blocks needs at least three 4-layer blocks (dense | routed |
# dense), i.e. L >= 12, but the shared shapes are 5-9 layers. Forcing MoD to fire
# at those depths meant dropping the dense I/O blocks, which let tokens bypass the
# entire network (see _sweep_arch_flags).
#
# So depth is a property of the arch here, and it is also the axis we screen:
# with block_size=4 and dense first/last blocks, L = 4*(R+2) gives exactly R
# routed blocks — R is "how many blocks a token can skip", which is the quantity
# that actually scales into the deep 1B. Widths were chosen (meta-device param
# count, not a formula — MLA/MoD/globals defeat the formula) to land N within a
# few % of the qwen3_like anchor at each rung. Exact N-matching is not required:
# the sweep is IsoFLOP, so D = C/(6N) is recomputed from each arch's measured N.
_SWEEP_ARCH_SHAPES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # R=1 routed block (L=12).
    "swa_globals": {
        "10m": {
            "hidden_size": 224,
            "intermediate_size": 896,
            "num_hidden_layers": 12,
            "num_attention_heads": 7,
            "num_key_value_heads": 1,
        },
        "25m": {
            "hidden_size": 336,
            "intermediate_size": 1344,
            "num_hidden_layers": 12,
            "num_attention_heads": 7,
            "num_key_value_heads": 1,
        },
        "50m": {
            "hidden_size": 480,
            "intermediate_size": 1920,
            "num_hidden_layers": 12,
            "num_attention_heads": 10,
            "num_key_value_heads": 2,
        },
        # NOTE kv=1: 11 heads is prime, so kv must be 1 (kv=2 built on CPU but
        # crashed at runtime — Hq=11 vs Hkv=10 after repeat_kv. Caught 2026-07-18
        # by the first 100m GPU run; see test_gqa_heads_divide.)
        "200m": {
            "hidden_size": 704,
            "intermediate_size": 2816,
            "num_hidden_layers": 12,
            "num_attention_heads": 11,
            "num_key_value_heads": 1,
        },
        # Deep-study tiers (see _SWEEP_BASE_SHAPES eff54/eff78): the MoD
        # skippers reach the tier's effective depth by carrying ~2x the layers
        # at ~half capacity (L = 4*(R+2); eff = 8 + 4*c*R). The mopd family
        # overrides these again below (dense spine L=48/72 + lane stages).
        "eff54": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 96,  # R=22; eff 52 @ c=0.5, 54 @ c=0.523
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
        "eff78": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 128,  # R=30; eff 68 @ c=0.5, 78 @ c=0.583
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
    },
    # R=2 routed blocks (L=16): deeper + narrower at matched N.
    "swa_globals_deep": {
        "10m": {
            "hidden_size": 192,
            "intermediate_size": 768,
            "num_hidden_layers": 16,
            "num_attention_heads": 6,
            "num_key_value_heads": 1,
        },
        "25m": {
            "hidden_size": 288,
            "intermediate_size": 1152,
            "num_hidden_layers": 16,
            "num_attention_heads": 6,
            "num_key_value_heads": 1,
        },
        # kv=1: 9 % 2 != 0 (same latent GQA-divisibility bug; test_gqa_heads_divide).
        "50m": {
            "hidden_size": 432,
            "intermediate_size": 1728,
            "num_hidden_layers": 16,
            "num_attention_heads": 9,
            "num_key_value_heads": 1,
        },
        "200m": {
            "hidden_size": 640,
            "intermediate_size": 2560,
            "num_hidden_layers": 16,
            "num_attention_heads": 10,
            "num_key_value_heads": 2,
        },
    },
    # R=3 routed blocks (L=20): the skinniest arm — closest in shape to the deep
    # 1B target, where most blocks are skippable.
    "swa_globals_deeper": {
        "10m": {
            "hidden_size": 192,
            "intermediate_size": 768,
            "num_hidden_layers": 20,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
        },
        "25m": {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 20,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
        "50m": {
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_hidden_layers": 20,
            "num_attention_heads": 6,
            "num_key_value_heads": 1,
        },
        # kv=1: 9 % 2 != 0 (same latent 100m bug as swa_globals above).
        "200m": {
            "hidden_size": 576,
            "intermediate_size": 2304,
            "num_hidden_layers": 20,
            "num_attention_heads": 9,
            "num_key_value_heads": 1,
        },
    },
}
# The MTP / PLE arms ride on the primary arch, so they inherit its shape.
_SWEEP_ARCH_SHAPES["swa_globals_mtp"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_globals_ple"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# The registers twin rides the primary shape too: bank size is identical and the
# only param delta is the per-head read gates (a few hundred params).
_SWEEP_ARCH_SHAPES["swa_registers"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_hybrid"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_ws"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_globals_ws"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_hybrid_ws"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# Redesign arms (2026-08-20) at the DEEP tiers. At 10m/25m/50m they ride the
# swa_globals deep-narrow shape, which is what they were benched on. At eff54/
# eff78 that alias would hand them L=96/128 -- the SKIPPER geometry, which
# carries 2x layers at half capacity and exists specifically for the MoD arms.
# None of these are skippers: sinkdiet/mat_ffn/dual_stream are dense-depth
# backbones with a different mechanism bolted on, think_suffix is a dense
# backbone plus one recursed block, and hourglass splits the tier's depth into
# outer + pooled-middle. They therefore take the tier's own dense shape
# (L=54/78 at h=1024), so a comparison against qwen3_like at that tier differs
# by the MECHANISM rather than by a 2x depth handicap.
_REDESIGN_DEEP = {
    # deep80 / deep128 are taken straight from the base table -- the redesign
    # arms are dense-depth backbones, so at these tiers they should carry the
    # tier's own shape rather than the L=96/128 skipper alias.
    "deep80": {
        "hidden_size": 896,
        "intermediate_size": 3584,
        "num_hidden_layers": 80,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
    },
    "deep128": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 128,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    "eff54": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 54,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "eff78": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 78,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
}
_SWEEP_ARCH_SHAPES["swa_sinkdiet"] = {**_SWEEP_ARCH_SHAPES["swa_globals"], **_REDESIGN_DEEP}
_SWEEP_ARCH_SHAPES["mat_ffn"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
for _nest in (
    "mat_ffn_flat4",
    "mat_ffn_skew3",
    "mat_ffn_skew4",
    "mat_ffn_skew4_nowarm",
    "mat_ffn_skew4_explore",
    "mat_ffn_thresh",
    "mat_ffn_flat4_explore",
    "mat_ffn_skew3_explore",
):
    # Same backbone shape as mat_ffn; intermediate_size is REPLACED per arm by
    # the matched-active-N solve in _mat_ffn_nest_flags, so the entry here only
    # supplies depth/width/heads.
    _SWEEP_ARCH_SHAPES[_nest] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["mat_ffn_deltamem"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["mat_ffn_gla"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["mat_ffn_ema"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["think_suffix"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["think_suffix_gqa"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["hourglass"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["dual_stream"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
# Trunk shrunk ONLY at ar10m, where the matched-N cell is being run. L=20 keeps
# all 3 SlowStreamBlocks (the (i+1)%4 placement with dual_stream_start_layer);
# L=19 drops one and would change the mechanism, not just the budget.
_SWEEP_ARCH_SHAPES["dual_stream_pm"] = {
    **_SWEEP_ARCH_SHAPES["dual_stream"],
    "ar10m": {
        "hidden_size": 192,
        "intermediate_size": 512,
        "num_hidden_layers": 20,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
    },
}
_SWEEP_ARCH_SHAPES["swa_moba"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
# <parent>_rec: (L, block) solving 2*block + 3*(L - 2*block) = the tier's L, so
# the unrolled depth at the mean of iters U[1, 5] matches the parent. ONE table
# for every parent, so the loop is identical across arms at a tier. Where two
# solutions exist the pick also matches the 3:1 SWA:global parent's GLOBAL-layer
# count once unrolled (ar25m: (13, 3) would give 8 globals against the parent's
# 7; ar50m: (16, 3) gives 10 against 9):
#   ar18m 2+4x3+2=16 (4 globals), ar10m 4+5x3+4=23 (6), ar25m 6+5x3+6=27 (7),
#   ar50m 6+8x3+6=36 (9).
_REC_SHAPES = {
    "ar18m": {"num_hidden_layers": 8, "recurrent_block_size": 2},
    "ar10m": {"num_hidden_layers": 13, "recurrent_block_size": 4},
    "ar25m": {"num_hidden_layers": 17, "recurrent_block_size": 6},
    "ar50m": {"num_hidden_layers": 20, "recurrent_block_size": 6},
}
for _p in _REC_PARENTS:
    _SWEEP_ARCH_SHAPES[f"{_p}_rec"] = {
        **_SWEEP_ARCH_SHAPES.get(_p, {}),
        **{
            _s: {**_SWEEP_ARCH_SHAPES.get(_p, {}).get(_s, {}), **_shape}
            for _s, _shape in _REC_SHAPES.items()
        },
    }
    for _mix in _REC_MIXERS:
        _SWEEP_ARCH_SHAPES[f"{_p}_rec{_mix}"] = _SWEEP_ARCH_SHAPES[f"{_p}_rec"]
_SWEEP_ARCH_SHAPES["swa_hils"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["swa_bsa_switch"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["swa_deltamem"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["swa_hils_deltamem"] = _SWEEP_ARCH_SHAPES["swa_sinkdiet"]
_SWEEP_ARCH_SHAPES["swa_scribe_mop"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopd"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_hybrid32"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_deltabank"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopd_delta"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_cmod"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_kmod"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_partrunk"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_rglrutrunk"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_deltatrunk"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopd_k"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# Deep-study tiers, mopd family: the label's effective depth comes from a
# SHALLOWER dense spine plus the 3 lane stages (eff = L + stages * budget =
# L + 6), so the spine is L=48/72 where the MoD skippers carry 96/128 layers.
# At width 1024 this makes mopd matched with the DENSE control in both params
# and effective FLOPs — the skippers pay ~1.75x params for the same eff depth.
_SWEEP_ARCH_SHAPES["swa_scribe_mopd"] = {
    **_SWEEP_ARCH_SHAPES["swa_globals"],
    "eff54": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 48,  # + 3 stages x budget 2.0 = eff 54
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "eff78": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 72,  # + 6 = eff 78
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
}
_SWEEP_ARCH_SHAPES["swa_scribe_mopd_k"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopd"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopd_delta"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopd"]


# Shallow-spine stackers: nearly ALL effective depth lives in the routed menu,
# so the spine is 32/36 layers where mopd carries 48/72. Same width as the
# tier (matched per-application FLOPs); params are LOWER than the tier anchors
# by design (that is the point) — report unique-N.
def _ar_spine(h, ffn, L, heads, kv):
    return {
        "hidden_size": h,
        "intermediate_size": ffn,
        "num_hidden_layers": L,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
    }


_SWEEP_ARCH_SHAPES["swa_scribe_mopr"] = {
    **_SWEEP_ARCH_SHAPES["swa_globals"],
    # Shallow spine at the aspect-ratio tiers (eff depth = spine + stages*2
    # matches the tier's L; only the spine is paid by every token).
    "ar10m": _ar_spine(192, 768, 14, 3, 1),  # 14 + 5*2 = 24 ~ tier 23
    "ar25m": _ar_spine(256, 1024, 16, 4, 1),  # 16 + 6*2 = 28 ~ tier 27
    "ar50m": _ar_spine(320, 1280, 22, 5, 1),  # 22 + 7*2 = 36 == tier 36
    "ar100m": _ar_spine(384, 1536, 30, 6, 2),  # 30 + 11*2 = 52 ~ tier 51
    # Deep tiers: the SPINE is shorter than the tier's dense depth on purpose --
    # effective depth (spine + stages x budget) matches the tier, but only the
    # spine is paid by every token. deep80 -> 48 + 16*2 = 80;
    # deep128 -> 64 + 32*2 = 128.
    "deep80": {
        "hidden_size": 896,
        "intermediate_size": 3584,
        "num_hidden_layers": 48,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
    },
    "deep128": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 64,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    # SCREENABLE TIERS (2026-08-20): the shallow-spine geometry needs a spine
    # deeper than the bank family's L=12 to fit its stage stack, so mopr gets
    # its OWN deep-narrow shape at 10m/25m/50m/100m instead of aliasing
    # swa_globals. Held MATCHED on NON-EMBEDDING params to swa_scribe (the
    # banks-only control) within 1% -- 12.53 vs 12.40M / 25.88 vs 26.06M /
    # 53.84 vs 53.36M / 110.94 vs 109.77M -- by trimming the FFN multiplier
    # (3.00 / 3.75 / 3.33 / 4.13) rather than the width, so head_dim stays a
    # clean 64 at every rung (kernel-friendly; the width-matched alternative
    # landed on head_dim 23/35, which the FA2/SDPA paths run badly).
    # L=20, bottom 5, 7 stages, budget 2.0 -> effective depth 34 = 1.70x spine.
    "10m": {
        "hidden_size": 192,
        "intermediate_size": 576,
        "num_hidden_layers": 20,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 256,
        "intermediate_size": 960,
        "num_hidden_layers": 20,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 384,
        "intermediate_size": 1280,
        "num_hidden_layers": 20,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
    },
    "200m": {
        "hidden_size": 512,
        "intermediate_size": 2112,
        "num_hidden_layers": 20,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "eff54": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 32,  # + 11 stages x budget 2.0 = eff 54 (min 32/max 76)
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "eff78": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 36,  # + 21 stages x budget 2.0 = eff 78 (min 36/max 120)
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
}
_SWEEP_ARCH_SHAPES["swa_scribe_moprec"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopr"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopr_be"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopr"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopr_b1"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopr"]
# D2 shares mopr's shallow spine exactly; only the routing GRANULARITY differs.
_SWEEP_ARCH_SHAPES["chunk_mop"] = _SWEEP_ARCH_SHAPES["swa_scribe_mopr"]
# Bank-reconstruction twins: only the WRITTEN-bank arms qualify.
_SWEEP_ARCH_SHAPES["swa_deltabank_recon"] = _SWEEP_ARCH_SHAPES["swa_globals"]
_SWEEP_ARCH_SHAPES["swa_scribe_mopd_delta_recon"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# MTP twins share their base arm's shape, so the pair differs ONLY by the MTP heads.
for _base in (
    "swa_hybrid32",
    "swa_deltabank",
    "swa_scribe",
    "swa_scribe_mopd",
    "swa_scribe_mopd_delta",
):
    _SWEEP_ARCH_SHAPES[f"{_base}_mtp"] = _SWEEP_ARCH_SHAPES["swa_globals"]
    _SWEEP_ARCH_SHAPES[f"{_base}_bag"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# Arm B needs L >= 2 * recurrent_block_size + 1 (prelude | looped core | coda);
# the swa_globals L=12 shape provides exactly that and keeps it matched-N
# (unique params) with the MoD incumbent it is compared against.
_SWEEP_ARCH_SHAPES["swa_globals_rec"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# The dense control shares the same shape so all three (incumbent MoD,
# recurrent, dense) differ only in the dynamic-compute mechanism.
_SWEEP_ARCH_SHAPES["swa_globals_dense"] = _SWEEP_ARCH_SHAPES["swa_globals"]
# GWT: shallow token stream (L=8, the pitch) + 2-block trunk + workspace read.
# Widths hand-tuned by meta-device param count to land unique-N within a few %
# of the qwen3_like anchor at each rung (trunk params included in N).
_SWEEP_ARCH_SHAPES["gwt"] = {
    "10m": {
        "hidden_size": 240,
        "intermediate_size": 960,
        "num_hidden_layers": 8,
        "num_attention_heads": 6,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 352,
        "intermediate_size": 1408,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 528,
        "intermediate_size": 2112,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    "200m": {
        "hidden_size": 800,
        "intermediate_size": 3200,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
}
# MoP: shared bottom+top layers (num_hidden_layers) + 6 lane blocks (0+2+4)
# at the same width; narrower than the dense base so total N (lanes included)
# matches the anchor. mop_dense carries only 2 lane blocks, so it stays wider.
_SWEEP_ARCH_SHAPES["mop"] = {
    "10m": {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 368,
        "intermediate_size": 1472,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 480,
        "intermediate_size": 1920,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 704,
        "intermediate_size": 2816,
        "num_hidden_layers": 9,
        "num_attention_heads": 11,
        "num_key_value_heads": 1,
    },
}
# mop_kv drops only the per-block K/V projections (small) — rides mop's shape.
_SWEEP_ARCH_SHAPES["mop_kv"] = _SWEEP_ARCH_SHAPES["mop"]
# Nested lanes carry 4 lane blocks (one shared stack) instead of 6 disjoint
# ones; recursive carries 1 weight-tied block. Fewer unique lane params ->
# wider shared width at matched N.
_SWEEP_ARCH_SHAPES["mop_nested"] = {
    "10m": {
        "hidden_size": 288,
        "intermediate_size": 1152,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 400,
        "intermediate_size": 1600,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 9,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    },
}
_SWEEP_ARCH_SHAPES["mop_rec"] = {
    "10m": {
        "hidden_size": 352,
        "intermediate_size": 1408,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 480,
        "intermediate_size": 1920,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 608,
        "intermediate_size": 2432,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 864,
        "intermediate_size": 3456,
        "num_hidden_layers": 9,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    },
}
# Stacked MoP: mop_stack carries S stages x 4 nested lane blocks (S=2 at
# 10m/25m, S=3 at 50m/100m) -> narrower than single-stage; mop_stack_shared
# carries ONE 4-block lane set regardless of S -> wider.
_SWEEP_ARCH_SHAPES["mop_stack"] = {
    "10m": {
        "hidden_size": 240,
        "intermediate_size": 960,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 336,
        "intermediate_size": 1344,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 400,
        "intermediate_size": 1600,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 592,
        "intermediate_size": 2368,
        "num_hidden_layers": 9,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
}
_SWEEP_ARCH_SHAPES["mop_stack_shared"] = {
    "10m": {
        "hidden_size": 288,
        "intermediate_size": 1152,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 400,
        "intermediate_size": 1600,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "50m": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 9,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    },
}
_SWEEP_ARCH_SHAPES["mop_dense"] = {
    "10m": {
        "hidden_size": 320,
        "intermediate_size": 1280,
        "num_hidden_layers": 5,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,
    },
    "25m": {
        "hidden_size": 448,
        "intermediate_size": 1792,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
    },
    # 50m/100m head counts chosen to keep head_dim <= 128: the wider hidden
    # sizes here (mop_dense compensates for carrying only 2 lane blocks) gave
    # hd=144/160 with the original 4/5 heads, which knocks SDPA/flex off the
    # fast attention kernels on sm86 — measured 3.4x step-time blowup at 50m
    # (311 ms vs 153 ms for mop3, 2026-08-18). Head count does not change N;
    # neither size has ever been trained, so no trained cell is invalidated.
    "50m": {
        "hidden_size": 576,
        "intermediate_size": 2304,
        "num_hidden_layers": 8,
        "num_attention_heads": 6,  # hd=96 (was 4 -> hd=144)
        "num_key_value_heads": 1,
    },
    "200m": {
        "hidden_size": 800,
        "intermediate_size": 3200,
        "num_hidden_layers": 9,
        "num_attention_heads": 8,  # hd=100 (was 5 -> hd=160)
        "num_key_value_heads": 1,
    },
}

for _arch in SWEEP_ARCHS:
    for _size, _base in _SWEEP_BASE_SHAPES.items():
        _name = f"{_arch}_{_size}"
        # Don't clobber an existing hand-tuned preset of the same name.
        if _name in UMICHSLM_CONFIGS:
            continue
        _base = {**_base, **_SWEEP_ARCH_SHAPES.get(_arch, {}).get(_size, {})}
        UMICHSLM_CONFIGS[_name] = {
            **_base,
            "max_position_embeddings": 4096,
            "vocab_size": 56000,
            "tie_word_embeddings": False,
            **_sweep_arch_flags(_arch, _base),
        }

# Deprecated size alias: the "200m" rung was named "100m" until 2026-08-20 (see
# the comment on _SWEEP_BASE_SHAPES["200m"]). Result dirs, older gate scripts and
# saved bench JSON all carry the old name, so "<arch>_100m" must keep resolving.
# New code should use "<arch>_200m"; this alias is for reading the past, not for
# writing the future.
for _arch in SWEEP_ARCHS:
    _new, _old = f"{_arch}_200m", f"{_arch}_100m"
    if _new in UMICHSLM_CONFIGS and _old not in UMICHSLM_CONFIGS:
        UMICHSLM_CONFIGS[_old] = UMICHSLM_CONFIGS[_new]


def _get_preset_dict(size: str) -> Dict[str, Any]:
    """Return a preset config dict or raise for unknown preset."""
    if size not in UMICHSLM_CONFIGS:
        available = list(UMICHSLM_CONFIGS.keys())
        raise ValueError(f"Unknown size: {size}. Available sizes: {available}")

    return UMICHSLM_CONFIGS[size]


def _apply_lws_to_layer(config: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """Applies Layer-Wise Scaling math to a specific layer's config."""
    if not config.get("layer_wise_scaling", False):
        return config

    layer_config = config.copy()
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    fraction = layer_idx / (num_layers - 1) if num_layers > 1 else 0
    head_range = config.get("head_scaling_range")

    # scale attention heads (ensure it's a whole number, closest to the next number of possible valid heads)
    if head_range:
        min_heads, max_heads = head_range
        raw_heads = min_heads + (max_heads - min_heads) * fraction
        valid_heads = [h for h in range(1, hidden_size + 1) if hidden_size % h == 0]
        snapped_heads = min(valid_heads, key=lambda x: abs(x - raw_heads))
        layer_config["num_attention_heads"] = snapped_heads

    # scale FFN Multiplier (linearly)
    ffn_range = config.get("ffn_multiplier_range")
    if ffn_range:
        min_ffn, max_ffn = ffn_range
        current_ffn = min_ffn + (max_ffn - min_ffn) * fraction
        layer_config["intermediate_size"] = int(hidden_size * current_ffn)

    return layer_config


def _estimate_backbone_from_config(config: Dict[str, Any]) -> int:
    """Estimate non-embedding backbone parameters for a preset config."""
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]

    # Layer norms (2 per layer)
    norm_params = 2 * hidden_size

    total = 0
    for layer_idx in range(num_layers):
        layer_config = _apply_lws_to_layer(config, layer_idx)
        current_intermediate = layer_config.get(
            "intermediate_size", config.get("intermediate_size", 0)
        )
        mlp_params = 3 * hidden_size * current_intermediate
        if _should_use_deltanet_for_count(layer_config, layer_idx):
            layer_attn_params = _estimate_deltanet_params(layer_config)
        else:
            layer_attn_params = _estimate_attention_params(layer_config, layer_idx)
        total += layer_attn_params + mlp_params + norm_params

    return int(total)


def _estimate_attention_params(config: Dict[str, Any], layer_idx: int = 0) -> int:
    """Estimate standard attention projection parameters for one layer."""
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden_size // num_heads)

    q_proj = hidden_size * num_heads * head_dim
    o_proj = num_heads * head_dim * hidden_size

    if config.get("share_kv_adjacent_layers", False) and layer_idx % 2 != 0:
        k_proj = 0
        v_proj = 0
    else:
        k_proj = hidden_size * num_kv_heads * head_dim
        v_proj = hidden_size * num_kv_heads * head_dim

    return q_proj + k_proj + v_proj + o_proj


def _estimate_deltanet_params(config: Dict[str, Any]) -> int:
    """Estimate DeltaNet projection parameters for one layer."""
    hidden_size = config["hidden_size"]
    num_heads = config.get("deltanet_num_heads") or config["num_attention_heads"]
    head_dim = hidden_size // num_heads
    expand_k = config.get("deltanet_expand_k", 1)
    expand_v = config.get("deltanet_expand_v", 1)
    variant = config.get("deltanet_variant", "gated")

    q_proj = hidden_size * num_heads * head_dim * expand_k
    k_proj = hidden_size * num_heads * head_dim * expand_k
    v_proj = hidden_size * num_heads * head_dim * expand_v
    o_proj = num_heads * head_dim * expand_v * hidden_size

    beta_dim = None
    if variant == "kda" and config.get("deltanet_true_delta_rule", False):
        # True KDA: per-key-channel gate, per-head beta (modules/deltanet.py).
        gate_dim, beta_dim = num_heads * head_dim * expand_k, num_heads
    elif variant == "kda":
        gate_dim = num_heads * head_dim * expand_v
    else:
        gate_dim = num_heads
    gate_proj = hidden_size * gate_dim
    beta_proj = hidden_size * (beta_dim or gate_dim)

    conv_params = 0
    if config.get("deltanet_use_short_conv", True):
        kernel_size = config.get("deltanet_conv_kernel_size", 4)
        conv_params = hidden_size * kernel_size + hidden_size

    return q_proj + k_proj + v_proj + o_proj + gate_proj + beta_proj + conv_params


def _should_use_deltanet_for_count(config: Dict[str, Any], layer_idx: int) -> bool:
    """Mirror DeltaNet layer selection logic for preset parameter counting."""
    if not config.get("use_deltanet", False):
        return False

    pattern = config.get("hybrid_attention_ratio")
    if pattern:
        return pattern[layer_idx % len(pattern)] == "delta"

    first_n = config.get("deltanet_first_n_layers", 0)
    if layer_idx < first_n:
        return False

    every_n = config.get("deltanet_every_n_layers", 0)
    if every_n <= 0:
        return True

    adjusted_idx = layer_idx - first_n
    return (adjusted_idx + 1) % every_n != 0


def get_umichslm_config(size: str = "base", **overrides) -> "UMichSLMConfig":
    """Get a preset configuration with optional overrides.

    This function returns a configuration for a specific model size,
    optionally modified with custom parameters.

    Args:
        size: Model size preset ("tiny", "small", "base", "large", "xlarge").
        **overrides: Parameters to override in the preset.

    Returns:
        UMichSLMConfig instance.

    Raises:
        ValueError: If the size is not recognized.

    Example:
        >>> config = get_umichslm_config("small")
        >>> print(config.hidden_size)
        768

        >>> config = get_umichslm_config("base", num_hidden_layers=12)
        >>> print(config.num_hidden_layers)
        12
    """
    config_dict = _get_preset_dict(size).copy()
    config_dict.update(overrides)

    UMichSLMConfig = _get_config_class()
    return UMichSLMConfig(**config_dict)


def list_available_sizes() -> list:
    """List all available model size presets.

    Returns:
        List of available size names.
    """
    return list(UMICHSLM_CONFIGS.keys())


def get_preset_info(size: str) -> Dict[str, Any]:
    """Get information about a specific preset.

    Args:
        size: Model size preset name.

    Returns:
        Dictionary with preset configuration.

    Raises:
        ValueError: If the size is not recognized.
    """
    return _get_preset_dict(size).copy()


def estimate_backbone_parameters(size: str) -> int:
    """Estimate backbone-only parameters for a given size.

    This excludes token embeddings and LM head parameters so preset families can
    be compared by transformer stack size independent of tokenizer vocabulary.

    Args:
        size: Model size preset name.

    Returns:
        Estimated backbone parameter count.

    Raises:
        ValueError: If the size is not recognized.
    """
    config = _get_preset_dict(size)
    return _estimate_backbone_from_config(config)


def estimate_parameters(size: str) -> int:
    """Estimate the number of parameters for a given size.

    This provides a rough estimate based on the configuration.
    Actual parameter count depends on vocabulary size and other factors.

    Args:
        size: Model size preset name.

    Returns:
        Estimated parameter count.

    Raises:
        ValueError: If the size is not recognized.
    """
    config = _get_preset_dict(size)
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    total_layer_params = _estimate_backbone_from_config(config)

    # Embedding parameters
    embedding_params = vocab_size * hidden_size

    # Final norm and LM head
    final_params = hidden_size + vocab_size * hidden_size

    total = embedding_params + total_layer_params + final_params
    return int(total)
