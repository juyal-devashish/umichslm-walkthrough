"""Configuration class for UMichSLM models.

This module defines the configuration class that specifies all hyperparameters
for the UMichSLM transformer architecture. The configuration is compatible
with HuggingFace's PretrainedConfig for seamless integration with the
transformers library.
"""

from typing import Dict, List, Optional

from transformers import PretrainedConfig


class UMichSLMConfig(PretrainedConfig):
    """Configuration class for UMichSLM models.

    This class stores all hyperparameters needed to instantiate a UMichSLM model.
    It inherits from PretrainedConfig for HuggingFace compatibility.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of hidden representations.
        intermediate_size: Dimension of the MLP hidden layer.
            For SwiGLU, this should be approximately 8/3 * hidden_size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads for grouped-query attention.
            If None, defaults to num_attention_heads (standard MHA).
        hidden_act: Activation function in the MLP ("silu" for SwiGLU).
        max_position_embeddings: Maximum sequence length the model supports.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        use_cache: Whether to use KV cache for generation.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for RoPE.
        rope_scaling: Configuration for RoPE scaling (for context extension).
        position_embedding_type: Type of position embedding ("rope" or "pope").
        attention_bias: Whether to use bias in attention projections.
        attention_dropout: Dropout rate for attention weights.
        mlp_bias: Whether to use bias in MLP projections.
        parallel_mlp: Whether to use parallel MLP architecture.
        use_moe: Whether to use Mixture of Experts instead of standard MLP.
        num_experts: Number of expert networks in MoE layers (max for progressive MoE).
        num_experts_per_tok: Number of experts each token is routed to (top-k).
        num_shared_experts: Number of shared experts that process all tokens.
        moe_expert_intermediate_size: Intermediate size for each expert MLP. If None,
            uses `intermediate_size` (8x params with 8 experts). Set to
            `intermediate_size // num_experts` for parameter-matched MoE.
        moe_progressive: Whether to use progressive MoE where first half of layers
            use plain MLP, then experts increase geometrically (2→4→8→...).
        moe_progressive_start_ratio: Fraction of layers that use plain MLP before
            MoE begins. Default 0.5 (first half uses plain MLP).
        router_aux_loss_coef: Coefficient for MoE load balancing auxiliary loss.
        router_z_loss_coef: Coefficient for MoE router z-loss.
        router_jitter: Noise added to router during training for exploration.
        sliding_window: Size of the sliding window for local attention. If None,
            uses full attention. Typical values: 4096, 8192.
        global_attention_every_n_layers: For sliding window attention, use full
            global attention every N layers. This "global refresh" pattern allows
            information to propagate beyond the local window. Set to 0 to disable.
        use_sliding_window_cache: Whether to use a rolling cache that only keeps
            the most recent `sliding_window` tokens during generation.
        use_mtp: Whether to use Multi-Token Prediction for training. MTP predicts
            multiple future tokens simultaneously, providing denser training signals.
        mtp_num_heads: Number of additional tokens to predict beyond the next token.
            DeepSeek V3 uses 1 (predicts next 2 tokens total). Set higher for more
            aggressive multi-token prediction.
        mtp_loss_weight: Weight for the MTP auxiliary loss. DeepSeek V3 uses 0.3
            initially, reducing to 0.1 during training.
        mtp_num_layers: Number of transformer layers in each MTP module. Default 1.
        mtp_share_head: Whether MTP modules share the main model's LM head (True)
            or use independent heads (False). Shared is more memory efficient.
        mtp_dropout: Dropout probability for MTP modules. Applied after MLP in
            each MTP transformer block. Default 0.0 (no dropout).
        use_mod: Whether to use Mixture of Depths (MoD) for dynamic compute allocation.
            MoD allows tokens to skip layers based on learned routing, reducing FLOPs.
        mod_capacity: Fraction of tokens to process at each MoD layer (0.0-1.0).
            Default 0.125 (12.5%) as recommended in the paper. Lower = more efficient.
        mod_every_n_layers: Apply MoD routing every N layers. Default 2 (alternating).
            Set to 1 for every layer, 0 to disable.
        mod_aux_loss_weight: Weight for the auxiliary loss that trains router for
            autoregressive inference. Default 0.01.
        mod_use_aux_predictor: Use auxiliary predictor MLP for inference routing
            instead of binary sigmoid. More accurate but adds parameters.
        use_zero_centered_rmsnorm: Whether to use zero-centered RMSNorm instead of
            standard RMSNorm. Zero-centered RMSNorm subtracts the mean before
            normalizing, similar to LayerNorm but without the bias term. Used in
            architectures like Gemma 2.
        use_mla: Whether to use Multi-head Latent Attention (MLA) as in DeepSeek V2/V3.
            MLA compresses KV into a low-rank latent space, dramatically reducing
            KV cache memory during inference.
        mla_kv_lora_rank: Rank of the low-rank KV compression. Controls the size of
            the latent vector. Smaller = more compression but potential quality loss.
            DeepSeek V2 uses 512 for a 236B model. Typical: hidden_size // 8.
        mla_q_lora_rank: Rank of the low-rank Q compression. If 0, Q is not compressed.
            DeepSeek V2 uses 1536. Set to 0 to disable Q compression.
        mla_qk_nope_head_dim: Dimension of the non-position-encoded part of Q/K heads.
            This part uses the compressed KV. DeepSeek V2 uses 128.
        mla_qk_rope_head_dim: Dimension of the RoPE-encoded part of Q/K heads.
            This uses decoupled RoPE keys for position encoding. DeepSeek V2 uses 64.
        mla_v_head_dim: Dimension of value heads. DeepSeek V2 uses 128.
        use_deltanet: Whether to use Gated DeltaNet linear attention layers. When enabled,
            some or all attention layers are replaced with DeltaNet, providing O(n)
            complexity with fixed-size recurrent state instead of growing KV cache.
            Default False - must be explicitly enabled.
        deltanet_variant: Type of DeltaNet to use. "gated" for Qwen3-Next style scalar
            gating per head, "kda" for Kimi-K2 style channel-wise gating. Default "gated".
        deltanet_expand_k: Key dimension expansion factor for DeltaNet. Default 1.
        deltanet_expand_v: Value dimension expansion factor for DeltaNet. Default 1.
        deltanet_num_heads: Number of heads for DeltaNet layers. If None, uses
            num_attention_heads. Useful for hybrid architectures.
        deltanet_every_n_layers: For hybrid architectures, every Nth layer uses standard
            attention instead of DeltaNet (acting as a "global refresh" layer).
            - 0: All layers use DeltaNet (default when use_deltanet=True)
            - 4: Layers 3, 7, 11, ... use attention (every 4th layer)
            - 2: Layers 1, 3, 5, ... use attention (alternating)
        deltanet_first_n_layers: Force first N layers to use standard attention even when
            use_deltanet=True. Helps with initial feature extraction. Default 0.
        hybrid_attention_ratio: Explicit per-layer pattern as list of "delta" or "attn".
            E.g., ["delta", "delta", "delta", "attn"] for Qwen3-Next 3:1 ratio.
            Repeats cyclically for all layers. Overrides deltanet_every_n_layers.
        deltanet_use_short_conv: Whether to use short convolution for local context in
            DeltaNet layers (like Mamba). Default True.
        deltanet_conv_kernel_size: Kernel size for the short convolution. Default 4.
        use_drope: Whether to enable DRoPE (Dropped Rotary Position Embeddings) training.
            When enabled, position embeddings can be dropped after a certain point in
            training to improve length generalization. Reference: https://arxiv.org/abs/2512.12167
        drope_drop_fraction: Fraction of training (0.0-1.0) after which position embeddings
            are dropped. For example, 0.875 means RoPE is used for 87.5% of training, then
            dropped for the final 12.5% "recalibration" phase. Default 0.875.
        use_repo: Whether to enable RePo (Re-Positioning) for learned position embeddings.
            RePo replaces fixed sequential positions with learned continuous positions
            predicted from hidden states. Reference: https://arxiv.org/abs/2512.14391
        repo_dim: Dimension of the position representation in RePo. If None, defaults to
            hidden_size // 8. Smaller values reduce parameters but may limit expressiveness.
        repo_start_layer: First layer to use RePo (0-indexed). Layers before this use
            standard RoPE. Default 5 (following the paper's recommendation to keep
            lower layers with standard RoPE for initial feature extraction).
        repo_use_base_positions: If True, RePo predicts position deltas that are added
            to base sequential positions. If False, predicts absolute positions directly.
            Default True for training stability.
        use_hymba: Whether to use Hymba hybrid attention-SSM heads. When enabled, each
            attention layer is replaced with a HymbaHybridHead that runs attention and
            Mamba2 SSM in parallel with learned fusion. Reference: https://arxiv.org/abs/2411.13676
        hymba_num_meta_tokens: Number of learnable meta tokens prepended to input for Hymba.
            These act as attention sinks. Default 128. Only used when use_hymba=True.
        hymba_ssm_state_size: SSM state dimension (d_state) for Mamba2 in Hymba. Default 16.
        hymba_ssm_conv_size: Conv1d kernel size in Mamba2 SSM. Default 4.
        hymba_ssm_expand: SSM inner dimension multiplier. d_inner = hidden_size * expand.
            Default 2.
        hymba_share_kv_layers: Whether to enable cross-layer KV cache sharing in Hymba.
            Adjacent layers share the same KV cache to reduce memory. Default False.
        hymba_kv_share_group_size: Number of consecutive layers per KV sharing group.
            Only the first layer computes fresh K, V; the rest reuse. Default 2.
        num_meta_tokens: Number of learnable meta tokens prepended to input sequences.
            Available for all architectures (not just Hymba). Meta tokens serve as
            attention sinks and are stripped from the output. Set to 0 to disable.
            When use_hymba=True, hymba_num_meta_tokens takes priority if num_meta_tokens=0.

    Example:
        >>> from umich_slm import UMichSLMConfig
        >>> config = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     num_hidden_layers=24,
        ...     num_attention_heads=16,
        ... )
        >>> print(config.hidden_size)
        1024

        # MoE configuration example
        >>> config_moe = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_moe=True,
        ...     num_experts=8,
        ...     num_experts_per_tok=2,
        ... )

        # Sliding Window Attention with Global Refresh
        >>> config_swa = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     sliding_window=4096,
        ...     global_attention_every_n_layers=4,  # Every 4th layer is global
        ... )

        # Multi-Token Prediction (MTP) configuration
        >>> config_mtp = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_mtp=True,
        ...     mtp_num_heads=1,  # Predict 1 additional token (like DeepSeek V3)
        ...     mtp_loss_weight=0.3,
        ... )

        # Mixture of Depths (MoD) configuration
        >>> config_mod = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_mod=True,
        ...     mod_capacity=0.125,  # Process 12.5% of tokens per MoD layer
        ...     mod_every_n_layers=2,  # Every other layer uses MoD
        ... )

        # Multi-head Latent Attention (MLA) configuration
        >>> config_mla = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_mla=True,
        ...     mla_kv_lora_rank=128,  # Compress KV to 128-dim latent
        ...     mla_q_lora_rank=256,   # Compress Q (0 to disable)
        ...     mla_qk_nope_head_dim=64,
        ...     mla_qk_rope_head_dim=32,
        ...     mla_v_head_dim=64,
        ... )

        # Gated DeltaNet (Qwen3-Next style with 3:1 hybrid ratio)
        >>> config_deltanet = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_deltanet=True,  # Must explicitly enable
        ...     deltanet_variant="gated",
        ...     hybrid_attention_ratio=["delta", "delta", "delta", "attn"],
        ... )

        # Kimi Delta Attention (KDA) with channel-wise gating
        >>> config_kda = UMichSLMConfig(
        ...     hidden_size=1024,
        ...     use_deltanet=True,
        ...     deltanet_variant="kda",
        ...     deltanet_every_n_layers=4,  # Every 4th layer is attention
        ... )
    """

    model_type = "umichslm"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Specify tokenizer class - UMichSLM uses GPT2TokenizerFast (SmolLM2 tokenizer)
    # This allows AutoTokenizer to automatically find the right tokenizer
    try:
        from transformers.models.gpt2 import GPT2TokenizerFast

        tokenizer_class = "GPT2TokenizerFast"
        tokenizer_class_fast = "GPT2TokenizerFast"
    except ImportError:
        tokenizer_class = None
        tokenizer_class_fast = None

    def __init__(
        self,
        vocab_size: int = 49152,  # SmolLM2 tokenizer
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
        position_embedding_type: str = "rope",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        parallel_mlp: bool = False,
        # MoE configuration
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 0,
        moe_expert_intermediate_size: Optional[int] = None,
        moe_progressive: bool = False,
        moe_progressive_start_ratio: float = 0.5,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        router_jitter: float = 0.0,
        # Sliding Window Attention configuration
        sliding_window: Optional[int] = None,
        global_attention_every_n_layers: int = 4,
        use_sliding_window_cache: bool = True,
        # Multi-Token Prediction configuration
        use_mtp: bool = False,
        mtp_num_heads: int = 1,
        mtp_loss_weight: float = 0.3,
        mtp_num_layers: int = 1,
        mtp_share_head: bool = True,
        mtp_dropout: float = 0.0,
        # Mixture of Depths configuration
        use_mod: bool = False,
        mod_capacity: float = 0.125,
        mod_every_n_layers: int = 2,
        mod_aux_loss_weight: float = 0.01,
        mod_use_aux_predictor: bool = False,
        # Normalization configuration
        use_zero_centered_rmsnorm: bool = False,
        # Multi-head Latent Attention (MLA) configuration
        use_mla: bool = False,
        mla_kv_lora_rank: int = 512,
        mla_q_lora_rank: int = 0,
        mla_qk_nope_head_dim: int = 128,
        mla_qk_rope_head_dim: int = 64,
        mla_v_head_dim: int = 128,
        # Gated DeltaNet / Linear Attention configuration
        # DEFAULT: Disabled - standard attention used unless explicitly enabled
        use_deltanet: bool = False,
        deltanet_variant: str = "gated",
        deltanet_expand_k: int = 1,
        deltanet_expand_v: int = 1,
        deltanet_num_heads: Optional[int] = None,
        deltanet_every_n_layers: int = 0,
        deltanet_first_n_layers: int = 0,
        hybrid_attention_ratio: Optional[List[str]] = None,
        deltanet_use_short_conv: bool = True,
        deltanet_conv_kernel_size: int = 4,
        # DRoPE (Dropped Rotary Position Embeddings) configuration
        use_drope: bool = False,
        drope_drop_fraction: float = 0.875,
        # RePo (Re-Positioning) configuration
        use_repo: bool = False,
        repo_dim: Optional[int] = None,
        repo_start_layer: int = 5,
        repo_use_base_positions: bool = True,
        # Hymba (Hybrid Mamba-Attention) configuration
        use_hymba: bool = False,
        hymba_num_meta_tokens: int = 128,
        hymba_ssm_state_size: int = 16,
        hymba_ssm_conv_size: int = 4,
        hymba_ssm_expand: int = 2,
        hymba_share_kv_layers: bool = False,
        hymba_kv_share_group_size: int = 2,
        # Meta tokens configuration (available for all architectures)
        num_meta_tokens: int = 0,
        # MoDA (Mixture-of-Depths Attention) configuration
        use_moda: bool = False,
        moda_use_ffn_kv: bool = True,
        moda_depth_kv_every_n: int = 1,
        moda_max_depth_cache: int = -1,
        moda_active_layers: Optional[List[int]] = None,
        # MoD always-used layers configuration
        mod_always_used_layers: Optional[List[int]] = None,
        mod_num_always_used_start: int = 0,
        mod_num_always_used_end: int = 0,
        # Norm placement configuration
        norm_placement: str = "pre",
        # MobileLLM block-wise weight sharing (MobileLLM-LS) configuration
        use_block_weight_sharing: bool = False,
        weight_sharing_block_size: int = 2,
        # Griffin (RG-LRU hybrid recurrent-attention) configuration
        use_griffin: bool = False,
        griffin_pattern: Optional[List[str]] = None,
        griffin_num_heads: Optional[int] = None,
        # Engram (Hash-Based Conditional Memory) configuration
        use_engram: bool = False,
        engram_layer_ids: Optional[List[int]] = None,
        engram_ngram_sizes: Optional[List[int]] = None,
        engram_num_heads: int = 8,
        engram_embed_dim: int = 64,
        engram_table_size: int = 49999,
        engram_seed: int = 42,
        engram_use_conv: bool = False,
        engram_conv_kernel_size: int = 3,
        # Next-token CE is computed in sequence chunks to bound peak logits memory
        lm_loss_chunk_size: int = 512,
        # Max bytes budget for one CE chunk's fp32 logits (batch_chunk * vocab * 4);
        # forward auto-shrinks sequence chunk when microbatch × vocab is large.
        lm_loss_chunk_cap_bytes: int = 1_200_000_000,
        # Stacking pretraining configuration
        # attention_mode selects the per-layer dispatch contract:
        #   "legacy"            — original behavior (MLA xor DeltaNet, ratio in {delta, attn}).
        #   "deltanet_mla"      — 3:1 hybrid via hybrid_attention_ratio entries in {delta, mla}.
        #                         Lifts the legacy MLA-XOR-DeltaNet validation.
        #   "swa_globals_mla"   — 3:1 hybrid with sliding-window-attention + persistent globals
        #                         on local layers and MLA on the global layer. Entries in {swa, mla}.
        attention_mode: str = "legacy",
        # SWA + persistent globals (used only when attention_mode == "swa_globals_mla")
        swa_global_count: int = 0,  # Number of learnable globals prepended to every batch.
        swa_window: int = 1024,  # Sliding-window length on local SWA layers.
        # MTP variant ("deepseek" = sequential chain — current behavior; "gloeckle" = 4 parallel
        # heads with d×d adapters and INDEPENDENT unembedding heads, training-only).
        mtp_variant: str = "deepseek",
        mtp_predict_offsets: Optional[List[int]] = None,  # Used only by gloeckle; default [2,3,4,5].
        # Stack-time MoD capacity: original blocks use mod_capacity (existing); inserted blocks
        # (marked via stage_block_spec[i].is_inserted) use mod_capacity_inserted.
        mod_capacity_inserted: float = 0.5,
        # Optional per-block spec consumed by the staging package. When set, MoD routing decisions
        # are taken from the spec rather than from is_mod_layer(). List of dicts with keys:
        #   {"source": int|None, "is_inserted": bool, "block_layout": List[str], "fresh_router": bool}.
        stage_block_spec: Optional[List[Dict]] = None,
        # Toggle FLA's L2Warp on inserted DeltaNet blocks for bf16 stability under exact-copy
        # initialization at stack transitions.
        deltanet_l2warp_inserted: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # Default to MHA if num_key_value_heads not specified
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.position_embedding_type = position_embedding_type
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.parallel_mlp = parallel_mlp

        # MoE configuration
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        # Expert intermediate size: None = use intermediate_size (8x params with 8 experts)
        # Set to intermediate_size // num_experts for parameter-matched MoE
        self.moe_expert_intermediate_size = moe_expert_intermediate_size
        # Progressive MoE: first half plain MLP, then 2→4→8→... experts
        self.moe_progressive = moe_progressive
        self.moe_progressive_start_ratio = moe_progressive_start_ratio
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router_jitter = router_jitter

        # Sliding Window Attention configuration
        self.sliding_window = sliding_window
        self.global_attention_every_n_layers = global_attention_every_n_layers
        self.use_sliding_window_cache = use_sliding_window_cache

        # Multi-Token Prediction configuration
        self.use_mtp = use_mtp
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_num_layers = mtp_num_layers
        self.mtp_share_head = mtp_share_head
        self.mtp_dropout = mtp_dropout

        # Mixture of Depths configuration
        self.use_mod = use_mod
        self.mod_capacity = mod_capacity
        self.mod_every_n_layers = mod_every_n_layers
        self.mod_aux_loss_weight = mod_aux_loss_weight
        self.mod_use_aux_predictor = mod_use_aux_predictor

        # Normalization configuration
        self.use_zero_centered_rmsnorm = use_zero_centered_rmsnorm

        # Multi-head Latent Attention (MLA) configuration
        self.use_mla = use_mla
        self.mla_kv_lora_rank = mla_kv_lora_rank
        self.mla_q_lora_rank = mla_q_lora_rank
        self.mla_qk_nope_head_dim = mla_qk_nope_head_dim
        self.mla_qk_rope_head_dim = mla_qk_rope_head_dim
        self.mla_v_head_dim = mla_v_head_dim

        # Gated DeltaNet / Linear Attention configuration
        self.use_deltanet = use_deltanet
        self.deltanet_variant = deltanet_variant
        self.deltanet_expand_k = deltanet_expand_k
        self.deltanet_expand_v = deltanet_expand_v
        self.deltanet_num_heads = deltanet_num_heads
        self.deltanet_every_n_layers = deltanet_every_n_layers
        self.deltanet_first_n_layers = deltanet_first_n_layers
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.deltanet_use_short_conv = deltanet_use_short_conv
        self.deltanet_conv_kernel_size = deltanet_conv_kernel_size

        # DRoPE (Dropped Rotary Position Embeddings) configuration
        self.use_drope = use_drope
        self.drope_drop_fraction = drope_drop_fraction

        # RePo (Re-Positioning) configuration
        self.use_repo = use_repo
        self.repo_dim = repo_dim  # Will default to hidden_size // 8 if None
        self.repo_start_layer = repo_start_layer
        self.repo_use_base_positions = repo_use_base_positions

        # Hymba (Hybrid Mamba-Attention) configuration
        self.use_hymba = use_hymba
        self.hymba_num_meta_tokens = hymba_num_meta_tokens
        self.hymba_ssm_state_size = hymba_ssm_state_size
        self.hymba_ssm_conv_size = hymba_ssm_conv_size
        self.hymba_ssm_expand = hymba_ssm_expand
        self.hymba_share_kv_layers = hymba_share_kv_layers
        self.hymba_kv_share_group_size = hymba_kv_share_group_size

        # Meta tokens (available for all architectures)
        self.num_meta_tokens = num_meta_tokens

        # MoDA (Mixture-of-Depths Attention) configuration
        self.use_moda = use_moda
        self.moda_use_ffn_kv = moda_use_ffn_kv
        self.moda_depth_kv_every_n = moda_depth_kv_every_n
        self.moda_max_depth_cache = moda_max_depth_cache
        self.moda_active_layers = moda_active_layers

        # MoD always-used layers
        self.mod_always_used_layers = mod_always_used_layers
        self.mod_num_always_used_start = mod_num_always_used_start
        self.mod_num_always_used_end = mod_num_always_used_end

        # Norm placement
        self.norm_placement = norm_placement

        # Auto-set norm_placement to "post" when use_moda=True unless explicitly overridden
        if self.use_moda and norm_placement == "pre" and "norm_placement" not in kwargs:
            self.norm_placement = "post"

        # MobileLLM block-wise weight sharing configuration
        self.use_block_weight_sharing = use_block_weight_sharing
        self.weight_sharing_block_size = weight_sharing_block_size

        # Griffin (RG-LRU hybrid) configuration
        self.use_griffin = use_griffin
        self.griffin_pattern = griffin_pattern
        self.griffin_num_heads = griffin_num_heads if griffin_num_heads is not None else num_attention_heads

        # Engram (Hash-Based Conditional Memory) configuration
        self.use_engram = use_engram
        self.engram_layer_ids = engram_layer_ids
        self.engram_ngram_sizes = engram_ngram_sizes
        self.engram_num_heads = engram_num_heads
        self.engram_embed_dim = engram_embed_dim
        self.engram_table_size = engram_table_size
        self.engram_seed = engram_seed
        self.engram_use_conv = engram_use_conv
        self.engram_conv_kernel_size = engram_conv_kernel_size

        self.lm_loss_chunk_size = max(1, int(lm_loss_chunk_size))
        self.lm_loss_chunk_cap_bytes = max(64 * 1024 * 1024, int(lm_loss_chunk_cap_bytes))

        # Stacking pretraining configuration
        self.attention_mode = attention_mode
        self.swa_global_count = int(swa_global_count)
        self.swa_window = int(swa_window)
        self.mtp_variant = mtp_variant
        self.mtp_predict_offsets = (
            list(mtp_predict_offsets) if mtp_predict_offsets is not None else [2, 3, 4, 5]
        )
        self.mod_capacity_inserted = float(mod_capacity_inserted)
        self.stage_block_spec = stage_block_spec
        self.deltanet_l2warp_inserted = bool(deltanet_l2warp_inserted)

        # Validate rope_scaling if provided
        self._validate_rope_scaling()

        # Validate stacking attention_mode if non-legacy
        self._validate_attention_mode()

        # Validate stage_block_spec shape (if provided)
        self._validate_stage_block_spec()

        # Validate DeltaNet configuration if enabled
        self._validate_deltanet_config()

        # Validate Hymba configuration if enabled
        self._validate_hymba_config()

        # Validate MoD + MoE layer overlap
        self._validate_mod_moe_config()

        # Validate Engram configuration if enabled
        self._validate_engram_config()

        # Validate Griffin configuration if enabled
        self._validate_griffin_config()

        # Validate MoDA configuration if enabled
        self._validate_moda_config()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_rope_scaling(self):
        """Validate RoPE scaling configuration."""
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(f"rope_scaling must be a dict, got {type(self.rope_scaling)}")

        required_keys = {"type", "factor"}
        if not required_keys.issubset(set(self.rope_scaling.keys())):
            raise ValueError(
                f"rope_scaling must contain keys {required_keys}, "
                f"got {set(self.rope_scaling.keys())}"
            )

        valid_types = ["linear", "dynamic", "yarn"]
        if self.rope_scaling["type"] not in valid_types:
            raise ValueError(
                f"rope_scaling type must be one of {valid_types}, "
                f"got {self.rope_scaling['type']}"
            )

        if self.rope_scaling["factor"] <= 1.0:
            raise ValueError(
                f"rope_scaling factor must be > 1.0, got {self.rope_scaling['factor']}"
            )

    def _validate_deltanet_config(self):
        """Validate DeltaNet configuration."""
        if not self.use_deltanet:
            return

        # Validate variant
        valid_variants = ["gated", "kda"]
        if self.deltanet_variant not in valid_variants:
            raise ValueError(
                f"deltanet_variant must be one of {valid_variants}, " f"got {self.deltanet_variant}"
            )

        # Validate hybrid_attention_ratio if provided. Stacking modes ("deltanet_mla",
        # "swa_globals_mla") allow "mla" / "swa" entries; "legacy" sticks to {"delta","attn"}.
        if self.hybrid_attention_ratio is not None:
            if self.attention_mode == "legacy":
                valid_layer_types = ["delta", "attn"]
            else:
                valid_layer_types = ["delta", "attn", "mla", "swa"]
            for i, layer_type in enumerate(self.hybrid_attention_ratio):
                if layer_type not in valid_layer_types:
                    raise ValueError(
                        f"hybrid_attention_ratio[{i}] must be one of {valid_layer_types}, "
                        f"got {layer_type}"
                    )

        # MLA + DeltaNet co-existence is only forbidden in legacy mode. In stacking modes
        # (e.g. "deltanet_mla"), use_mla and use_deltanet must both be enabled and the
        # per-layer ratio drives dispatch.
        if self.use_mla and self.attention_mode == "legacy":
            raise ValueError(
                "Cannot use both MLA and DeltaNet simultaneously in attention_mode='legacy'. "
                "Set attention_mode='deltanet_mla' to enable the 3:1 hybrid via "
                "hybrid_attention_ratio entries in {'delta','mla'}."
            )

    def _validate_attention_mode(self):
        """Validate stacking attention_mode and its co-flag invariants."""
        valid_modes = ["legacy", "deltanet_mla", "swa_globals_mla"]
        if self.attention_mode not in valid_modes:
            raise ValueError(
                f"attention_mode must be one of {valid_modes}, got {self.attention_mode!r}."
            )

        if self.attention_mode == "legacy":
            return

        if self.hybrid_attention_ratio is None:
            raise ValueError(
                f"attention_mode={self.attention_mode!r} requires hybrid_attention_ratio "
                "to express the per-layer-in-block pattern (e.g. ['delta','delta','delta','mla'])."
            )

        if self.attention_mode == "deltanet_mla":
            if not self.use_deltanet:
                raise ValueError("attention_mode='deltanet_mla' requires use_deltanet=True.")
            if not self.use_mla:
                raise ValueError("attention_mode='deltanet_mla' requires use_mla=True.")
            allowed = {"delta", "mla"}
            for i, e in enumerate(self.hybrid_attention_ratio):
                if e not in allowed:
                    raise ValueError(
                        f"attention_mode='deltanet_mla' requires hybrid_attention_ratio entries "
                        f"in {sorted(allowed)}; got entry {i!r}={e!r}."
                    )

        elif self.attention_mode == "swa_globals_mla":
            if not self.use_mla:
                raise ValueError("attention_mode='swa_globals_mla' requires use_mla=True.")
            if self.use_deltanet:
                raise ValueError(
                    "attention_mode='swa_globals_mla' cannot be combined with use_deltanet=True."
                )
            if self.sliding_window is None:
                raise ValueError(
                    "attention_mode='swa_globals_mla' requires sliding_window to be set "
                    "(typical: 1024). swa_window separately controls the SWA layer's window."
                )
            allowed = {"swa", "mla"}
            for i, e in enumerate(self.hybrid_attention_ratio):
                if e not in allowed:
                    raise ValueError(
                        f"attention_mode='swa_globals_mla' requires hybrid_attention_ratio entries "
                        f"in {sorted(allowed)}; got entry {i!r}={e!r}."
                    )
            if self.swa_global_count > 0 and self.num_meta_tokens != 0:
                raise ValueError(
                    "swa_global_count > 0 conflicts with num_meta_tokens > 0 — only one "
                    "prepended-token mechanism may be active at a time."
                )

    def _validate_stage_block_spec(self):
        """Validate stage_block_spec shape (lightweight; deeper checks live in staging.block_spec)."""
        if self.stage_block_spec is None:
            return
        if not isinstance(self.stage_block_spec, list):
            raise ValueError("stage_block_spec must be a list of per-block dicts (or None).")
        required_keys = {"source", "is_inserted", "block_layout"}
        block_size = None
        total_layers = 0
        for i, entry in enumerate(self.stage_block_spec):
            if not isinstance(entry, dict):
                raise ValueError(f"stage_block_spec[{i}] must be a dict, got {type(entry)}.")
            missing = required_keys - set(entry.keys())
            if missing:
                raise ValueError(f"stage_block_spec[{i}] missing keys: {sorted(missing)}.")
            layout = entry["block_layout"]
            if not isinstance(layout, list) or not layout:
                raise ValueError(f"stage_block_spec[{i}].block_layout must be a non-empty list.")
            if block_size is None:
                block_size = len(layout)
            elif len(layout) != block_size:
                raise ValueError(
                    f"stage_block_spec[{i}].block_layout has length {len(layout)}, "
                    f"expected {block_size} (all blocks must share the same layout length)."
                )
            total_layers += len(layout)
        if total_layers != self.num_hidden_layers:
            raise ValueError(
                f"stage_block_spec accounts for {total_layers} layers but "
                f"num_hidden_layers={self.num_hidden_layers}."
            )

    def _validate_hymba_config(self):
        """Validate Hymba configuration."""
        if not self.use_hymba:
            return

        # Cannot combine with DeltaNet (both fundamentally modify attention)
        if self.use_deltanet:
            raise ValueError(
                "Cannot use both Hymba and DeltaNet simultaneously. "
                "Both modify the attention mechanism."
            )

        # Cannot combine with MLA (both replace the attention mechanism)
        if self.use_mla:
            raise ValueError(
                "Cannot use both Hymba and MLA simultaneously. "
                "Hymba replaces attention with a hybrid attention-SSM head, "
                "while MLA replaces attention with compressed latent attention."
            )

        # Validate SSM parameters
        if self.hymba_ssm_state_size <= 0:
            raise ValueError(f"hymba_ssm_state_size must be > 0, got {self.hymba_ssm_state_size}")
        if self.hymba_ssm_expand < 1:
            raise ValueError(f"hymba_ssm_expand must be >= 1, got {self.hymba_ssm_expand}")
        if self.hymba_num_meta_tokens < 0:
            raise ValueError(
                f"hymba_num_meta_tokens must be >= 0, got {self.hymba_num_meta_tokens}"
            )

    def _validate_mod_moe_config(self):
        """Validate MoE and MoD layers don't overlap."""
        if not self.use_mod or not self.use_moe:
            return
        for layer_idx in range(self.num_hidden_layers):
            # Inline is_mod_layer logic to avoid circular import
            layer_is_mod = self.mod_every_n_layers > 0 and layer_idx % self.mod_every_n_layers == 1
            if not layer_is_mod:
                continue
            if self.moe_progressive:
                layer_is_moe = self.get_num_experts_for_layer(layer_idx) > 1
            else:
                layer_is_moe = True
            if layer_is_moe:
                raise ValueError(
                    f"Layer {layer_idx} is both MoE and MoD. "
                    f"Adjust mod_every_n_layers or use moe_progressive to avoid overlap."
                )

    def _validate_engram_config(self):
        """Validate Engram configuration."""
        if not self.use_engram:
            return

        # Validate layer IDs are within range
        if self.engram_layer_ids is not None:
            for lid in self.engram_layer_ids:
                if lid < 0 or lid >= self.num_hidden_layers:
                    raise ValueError(
                        f"engram_layer_ids contains {lid}, but num_hidden_layers is "
                        f"{self.num_hidden_layers}. Layer IDs must be in "
                        f"[0, {self.num_hidden_layers})."
                    )

        # Validate n-gram sizes
        ngram_sizes = self.engram_ngram_sizes or [2, 3]
        for ng in ngram_sizes:
            if ng < 2:
                raise ValueError(
                    f"engram_ngram_sizes contains {ng}, but n-gram sizes must be >= 2."
                )

    def _validate_griffin_config(self):
        """Validate Griffin configuration."""
        if not self.use_griffin:
            return

        valid_tokens = {"rglru", "attn"}
        if self.griffin_pattern is not None:
            for i, token in enumerate(self.griffin_pattern):
                if token not in valid_tokens:
                    raise ValueError(
                        f"griffin_pattern[{i}] = {token!r} is invalid. "
                        f"Must be one of {sorted(valid_tokens)}."
                    )

        if self.griffin_num_heads <= 0:
            raise ValueError(
                f"griffin_num_heads must be > 0, got {self.griffin_num_heads}."
            )

        if self.hidden_size % self.griffin_num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"griffin_num_heads ({self.griffin_num_heads})."
            )

    def _validate_moda_config(self):
        """Validate MoDA configuration."""
        # Validate norm_placement
        if self.norm_placement not in ("pre", "post"):
            raise ValueError(f"norm_placement must be 'pre' or 'post', got '{self.norm_placement}'")

        if not self.use_moda:
            # Still validate mod_always_used_layers if set
            if self.mod_always_used_layers is not None:
                for lid in self.mod_always_used_layers:
                    if lid < 0 or lid >= self.num_hidden_layers:
                        raise ValueError(
                            f"mod_always_used_layers contains {lid}, but num_hidden_layers is "
                            f"{self.num_hidden_layers}. Layer IDs must be in "
                            f"[0, {self.num_hidden_layers})."
                        )
            if (
                self.mod_num_always_used_start + self.mod_num_always_used_end
                >= self.num_hidden_layers
            ):
                raise ValueError(
                    f"mod_num_always_used_start ({self.mod_num_always_used_start}) + "
                    f"mod_num_always_used_end ({self.mod_num_always_used_end}) must be < "
                    f"num_hidden_layers ({self.num_hidden_layers})."
                )
            return

        # MoDA + DeltaNet incompatibility
        if self.use_deltanet:
            raise ValueError(
                "Cannot use both MoDA and DeltaNet simultaneously. "
                "MoDA requires softmax attention to unify depth KV under a single softmax, "
                "but DeltaNet uses linear attention without softmax."
            )

        # Validate moda_active_layers
        if self.moda_active_layers is not None:
            for lid in self.moda_active_layers:
                if lid < 0 or lid >= self.num_hidden_layers:
                    raise ValueError(
                        f"moda_active_layers contains {lid}, but num_hidden_layers is "
                        f"{self.num_hidden_layers}. Layer IDs must be in "
                        f"[0, {self.num_hidden_layers})."
                    )

        # Validate mod_always_used_layers
        if self.mod_always_used_layers is not None:
            for lid in self.mod_always_used_layers:
                if lid < 0 or lid >= self.num_hidden_layers:
                    raise ValueError(
                        f"mod_always_used_layers contains {lid}, but num_hidden_layers is "
                        f"{self.num_hidden_layers}. Layer IDs must be in "
                        f"[0, {self.num_hidden_layers})."
                    )

        if self.mod_num_always_used_start + self.mod_num_always_used_end >= self.num_hidden_layers:
            raise ValueError(
                f"mod_num_always_used_start ({self.mod_num_always_used_start}) + "
                f"mod_num_always_used_end ({self.mod_num_always_used_end}) must be < "
                f"num_hidden_layers ({self.num_hidden_layers})."
            )

    def get_always_used_layers(self) -> set:
        """Compute the set of layer indices that always process all tokens.

        These layers use the standard decoder layer even when use_mod=True.
        Explicit list takes precedence over start/end convenience params.

        Returns:
            Set of layer indices.
        """
        if self.mod_always_used_layers is not None:
            return set(self.mod_always_used_layers)
        start_layers = list(range(self.mod_num_always_used_start))
        end_layers = list(
            range(
                self.num_hidden_layers - self.mod_num_always_used_end,
                self.num_hidden_layers,
            )
        )
        return set(start_layers + end_layers)

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    def get_num_experts_for_layer(self, layer_idx: int) -> int:
        """Get the number of experts for a specific layer.

        For progressive MoE, the pattern is:
        - First half of layers (controlled by moe_progressive_start_ratio): 1 expert (plain MLP)
        - Remaining layers: progressive increase from 2 to num_experts

        The expert count doubles in a pyramid pattern where each level gets
        progressively fewer layers: 1/2 with 2 experts, 1/4 with 4 experts,
        1/8 with 8 experts, etc.

        Args:
            layer_idx: Index of the layer (0-indexed).

        Returns:
            Number of experts for this layer (1 for plain MLP).
        """
        if not self.use_moe:
            return 1

        if not self.moe_progressive:
            # Standard MoE: all layers use num_experts
            return self.num_experts

        # Progressive MoE
        num_layers = self.num_hidden_layers
        start_layer = int(num_layers * self.moe_progressive_start_ratio)

        # First portion uses plain MLP
        if layer_idx < start_layer:
            return 1

        # Remaining layers use progressive MoE
        moe_layers = num_layers - start_layer
        relative_idx = layer_idx - start_layer  # 0-indexed within MoE section

        # Compute the expert schedule using recursive subdivision
        # For max_experts=8: 1/2 (2 exp), 1/4 (4 exp), 1/8 (8 exp), 1/8 (8 exp)
        # For max_experts=16: 1/2 (2 exp), 1/4 (4 exp), 1/8 (8 exp), 1/16 (16 exp), 1/16 (16 exp)
        max_experts = self.num_experts

        # Build cumulative boundaries for each expert level
        # Each level gets half the remaining fraction
        import math

        num_levels = int(math.log2(max_experts))  # e.g., 3 for 8 experts

        # Calculate layer boundaries for each expert count (2, 4, 8, ...)
        cumulative_fraction = 0.0
        for level in range(num_levels):
            experts_at_level = 2 ** (level + 1)  # 2, 4, 8, 16, ...

            if level == num_levels - 1:
                # Last level(s) get the remaining fraction
                fraction = 1.0 - cumulative_fraction
            else:
                # Each level gets half of what's left, but minimum 1/2^(num_levels)
                fraction = 0.5 ** (level + 1)

            layer_boundary = int(cumulative_fraction * moe_layers)
            next_boundary = int((cumulative_fraction + fraction) * moe_layers)

            if layer_boundary <= relative_idx < next_boundary:
                return min(experts_at_level, max_experts)

            cumulative_fraction += fraction

        # Fallback: return max experts for any remaining layers
        return max_experts

    def get_progressive_moe_schedule(self) -> List[int]:
        """Get the full expert schedule for all layers.

        Returns:
            List of expert counts, one per layer.
        """
        return [self.get_num_experts_for_layer(i) for i in range(self.num_hidden_layers)]

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "UMichSLMConfig":
        """Create config from dictionary.

        This overrides the parent method to handle any custom preprocessing.

        Args:
            config_dict: Dictionary of config parameters.
            **kwargs: Additional parameters.

        Returns:
            UMichSLMConfig instance.
        """
        return super().from_dict(config_dict, **kwargs)
