"""Model preset configurations for UMichSLM.

This module provides preset configurations for different model sizes,
from tiny (for testing) to large (for production use).
"""

from typing import Any, Dict

# Import lazily to avoid circular imports
def _get_config_class():
    from ..configuration_umichslm import UMichSLMConfig
    return UMichSLMConfig


# Model size presets
# Parameter counts are approximate and depend on vocab_size
#
# NOTE: The vocab_size in these presets (49152) matches SmolLM2 tokenizer.
# When using these presets with a tokenizer, you MUST override vocab_size with the
# actual tokenizer's vocab_size. The preset value should never be used directly in
# training scripts - always load vocab_size from the tokenizer being used.
UMICHSLM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        # ~15M parameters (for testing and debugging)
        "hidden_size": 256,
        "intermediate_size": 688,  # ~8/3 * 256, rounded to multiple of 16
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,  # MHA (baseline)
        "max_position_embeddings": 2048,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "small": {
        # ~125M parameters
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,  # MHA (baseline) - use --num_key_value_heads 4 for GQA ablation
        "max_position_embeddings": 2048,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "base": {
        # ~350M parameters
        "hidden_size": 1024,
        "intermediate_size": 2752,  # ~8/3 * 1024, rounded
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # MHA (baseline) - use --num_key_value_heads 4 for GQA ablation
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "large": {
        # ~1.3B parameters
        "hidden_size": 2048,
        "intermediate_size": 5504,  # ~8/3 * 2048, rounded
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # MHA (baseline) - use --num_key_value_heads 8 for GQA ablation
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "xlarge": {
        # ~3B parameters
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 24,  # MHA (baseline) - use --num_key_value_heads 8 for GQA ablation
        "max_position_embeddings": 8192,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_tiny": {
        # ~5.0M backbone params / ~26.2M total with default 49k untied vocab
        "hidden_size": 216,
        "intermediate_size": 576,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_xsmall": {
        # ~19.9M backbone params / ~62.4M total with default 49k untied vocab
        "hidden_size": 432,
        "intermediate_size": 1152,
        "num_hidden_layers": 10,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
    },
    "qwen3_like_small": {
        # ~100.7M backbone params / ~176M total with default 49k untied vocab
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,  # GQA 3:1 ratio
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
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
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
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
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
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
        "hybrid_attention_ratio": ["delta", "delta", "delta", "attn"],
        "max_position_embeddings": 4096,
        "vocab_size": 49152,  # PLACEHOLDER - must be overridden with actual tokenizer vocab_size
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
        "vocab_size": 49152, # placeholder 
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
        "vocab_size": 49152, # placeholder 
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
        "vocab_size": 49152, # placeholder 
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
        "vocab_size": 49152, # placeholder 
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
        "vocab_size": 49152, # placeholder 
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
        "vocab_size": 49152, # placeholder 
    },

    "baguette_micro": {
        # ~4M backbone / ~47M total with 65k vocab (untied)
        "hidden_size": 128,
        "intermediate_size": 352,      # ~8/3 * 128, rounded to mult of 16
        "num_hidden_layers": 20,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,      # GQA 2:1
        "head_dim": 64,                # explicit, decoupled from hidden/heads
        "max_position_embeddings": 2048,
        "vocab_size": 49152,           # PLACEHOLDER - Baguettotron tokenizer vocab
    },

    "baguette_tiny": {
        # ~13M backbone / ~112M total with 65k vocab (untied)
        "hidden_size": 192,
        "intermediate_size": 512,      # ~8/3 * 192, rounded
        "num_hidden_layers": 30,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,      # GQA 3:1
        "head_dim": 64,
        "max_position_embeddings": 2048,
        "vocab_size": 49152,           # PLACEHOLDER
    },

    "baguette_small": {
            # ~36M backbone / ~188M total with 65k vocab (untied)
            "hidden_size": 288,
            "intermediate_size": 768,      # ~8/3 * 288, exact
            "num_hidden_layers": 40,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,      # GQA 3:1
            "head_dim": 64,
            "max_position_embeddings": 4096,
            "vocab_size": 49152,           # PLACEHOLDER
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
            "vocab_size": 49152,  # PLACEHOLDER
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
            "vocab_size": 49152,  # PLACEHOLDER
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
            "vocab_size": 49152,  # PLACEHOLDER
        },

    "mobilellm_tiny": {
        # ~5.3M backbone params
        "hidden_size": 128,
        "intermediate_size": 384,
        "num_hidden_layers": 28,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,       # GQA 4:1
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
        "num_key_value_heads": 2,       # GQA 2:1
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
        "num_key_value_heads": 3,       # GQA 3:1
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
        "num_hidden_layers": 18,       # +80% depth vs vanilla (paper: add depth not width)
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "use_mod": True,
        "mod_capacity": 0.125,         # paper's optimal: 12.5% tokens per routed block
        "mod_every_n_layers": 2,       # paper's optimal: route every other block
        "mod_aux_loss_weight": 0.01,
        "vocab_size": 49152, # PLACEHOLDER
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
        "vocab_size": 49152, # PLACEHOLDER
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
        "vocab_size": 49152, # PLACEHOLDER
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
        "vocab_size": 49152,  # PLACEHOLDER
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
        "vocab_size": 49152,
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
        "mod_every_n_layers": 4,
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
        "vocab_size": 49152,
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
        "mod_every_n_layers": 4,
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
        "vocab_size": 49152,
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
        "vocab_size": 49152,
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
        "vocab_size": 49152,
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
        current_intermediate = layer_config.get("intermediate_size", config.get("intermediate_size", 0))
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

    if variant == "kda":
        gate_dim = num_heads * head_dim * expand_v
    else:
        gate_dim = num_heads
    gate_proj = hidden_size * gate_dim
    beta_proj = hidden_size * gate_dim

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
