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
        rope_scaling_scope: Which layers ``rope_scaling`` applies to. ``"all"`` (default)
            scales every rotary; ``"global"`` returns every sliding-window layer
            (``swa_window``) to unscaled RoPE after construction, since a banded layer
            only sees in-window distances it trained on. A no-op on models with no
            windowed layers.
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
        mtp_soft: Soft-MTP (arXiv 2610.38331). The MTP heads and trunk get the
            gradient of KL(sg(p_ntp at t+k) || p_mtp at t) instead of the MTP CE
            gradient; the shared lm_head still gets CE. Needs the DeepSeek variant
            with a shared head. Loss values are unchanged.
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
        deltanet_true_delta_rule: Select the DeltaNet math. When False (default) the layer
            uses the legacy scalar-gated *linear attention* update
            (S = gate*S + beta*(k⊗v); no delta correction) — fast reference but NOT the delta
            rule. When True the layer computes the true Gated DeltaNet (Qwen3-Next style:
            delta correction + qk-l2norm + log-space forget gate) via FLA's
            chunk_gated_delta_rule (training/prefill) and fused_recurrent_gated_delta_rule
            (cached decode); intra-document boundaries are honored via cu_seqlens. Requires
            the FLA library + CUDA; on CPU / without FLA it falls back to the naive path.
            Default False.
        deltanet_allow_neg_eigval: Scale the delta-rule write strength beta from (0, 1)
            to (0, 2), so the state transition (I - beta k k^T) can take NEGATIVE
            eigenvalues (Grazzi et al. 2025, arXiv:2411.12537; Olmo Hybrid's choice,
            arXiv:2604.03444). Needed for state tracking (e.g. parity), ~free in FLOPs.
            Same placement as FLA's own GatedDeltaNet layer (beta * 2). Requires
            deltanet_true_delta_rule. Default False.
        use_drope: Whether to enable DRoPE (Dropped Rotary Position Embeddings) training.
            When enabled, position embeddings can be dropped after a certain point in
            training to improve length generalization. Reference: https://arxiv.org/abs/2512.12167
        partial_rotary_factor: Fraction of each head's q/k dims that RoPE rotates in the
            GQA/MHA attention layers (``modules/attention.py``); the rest carry content
            with no position, as in GPT-NeoX and Qwen3-Next (0.25). The rotated slice is
            the leading ``int(head_dim * f)`` dims, rounded down to even. 0.0 is NoPE on
            every such layer. MLA layers keep their own decoupled rope split and are not
            affected. Default 1.0 (full RoPE).
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
        rope_scaling_scope: str = "all",
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
        # Intra-document masking: when packed windows carry a per-token `document_ids`
        # signal, tokens attend only within their own document and RoPE positions reset at
        # each boundary (LONG_CONTEXT_PLAN.md §4). Correctness fix + long-context prerequisite.
        # Supported under sdpa/eager; FlashAttention-2 varlen is a separate (GPU) follow-up.
        use_intra_doc_masking: bool = False,
        # Multi-Token Prediction configuration
        use_mtp: bool = False,
        mtp_num_heads: int = 1,
        mtp_loss_weight: float = 0.3,
        # Tokens per slice of the chunked MTP loss. The full [B*S, vocab] fp32
        # logits are 13.7 GiB at B=16/S=4096/V=56000 and OOMed every MTP cell.
        mtp_loss_chunk_tokens: int = 4096,
        # Soft-MTP (arXiv 2610.38331): route the NTP loss's -W^T(q - y) into each MTP
        # hidden state so the heads and trunk train on KL(sg(q_ntp) || p_mtp) while the
        # shared lm_head keeps its CE gradient. Forward values are unchanged.
        mtp_soft: bool = False,
        # --- bag-of-next-K lookahead (modules/bag_lookahead.py) ---
        use_bag_lookahead: bool = False,
        bag_lookahead_k: int = 16,
        bag_lookahead_weight: float = 0.1,
        bag_vocab: int = 0,  # 0 = full vocab; else collapse to top-V + OTHER
        bag_share_head: bool = True,  # tie to lm_head (full-vocab case only)
        bag_loss_chunk_tokens: int = 4096,
        # --- bank reconstruction (modules/bank_reconstruction.py) ---
        use_bank_reconstruction: bool = False,
        bank_recon_samples: int = 32,
        bank_recon_weight: float = 0.1,
        # --- latent feedback (modules/latent_feedback.py, arXiv 2608.08888) ---
        # The previous token's TOP-layer state fused into the next token's input.
        # TRAINING-TIME cost only: extra forward passes, ~1.28x at the default
        # mixture. `pass_mix` is the probability of 1/2/3... passes for a batch
        # and MUST keep its three-pass tail -- two-pass-only training diverges
        # past its trained recursion depth (see the module docstring).
        use_latent_feedback: bool = False,
        latent_feedback_pass_mix: Optional[List[float]] = None,
        latent_feedback_mixin_p: float = 0.1,
        latent_feedback_seed: int = 1234,
        # Passes used when not training (teacher-forced validation). 2 is the
        # paper's "fused" decoding regime, i.e. the model's real predictive
        # distribution, and is what val_lm_loss must be measured under.
        latent_feedback_eval_passes: int = 2,
        # --- state/prediction separation (modules/pause_stream.py, arXiv 2609.03807) ---
        use_pause_stream: bool = False,
        pause_shared_ffn: bool = True,
        mtp_num_layers: int = 1,
        mtp_share_head: bool = True,
        mtp_dropout: float = 0.0,
        # Mixture of Depths configuration
        use_mod: bool = False,
        mod_capacity: float = 0.125,
        mod_every_n_layers: int = 2,
        # Block-atomic MoD: route whole blocks of `mod_block_size` layers as a
        # unit (one router decision + one residual gate spanning the block), per
        # the stacking-plan design where the block is the atomic routing unit.
        # When False, MoD falls back to the legacy per-layer routing (a single
        # layer per stride is gated). See modeling_umichslm for the wiring.
        mod_atomic_block: bool = False,
        mod_block_size: int = 0,  # 0 -> infer (len(hybrid ratio) | mod_every_n_layers | 4)
        mod_io_blocks_dense: bool = True,  # first & last block stay dense (I/O blocks)
        # Escape hatch for the "every layer is inside a routable block" config, in
        # which a token declined by every router traverses ZERO transformer layers.
        # That is a bug in any real model (it silently costs ~1.4 nats) and the
        # model refuses to build it, but the unit tests for the gather/scatter
        # machinery legitimately need it. Say the name to get it.
        mod_allow_total_bypass: bool = False,
        mod_aux_loss_weight: float = 0.01,
        mod_use_aux_predictor: bool = False,
        # Gate gain shaping. Defaults reproduce the original behaviour exactly:
        # "raw" leaves the unbounded logit as the multiplicative gain.
        mod_gate_fn: str = "raw",
        mod_tanh_alpha: float = 0.2,
        mod_capacity_warmup_steps: int = 0,
        mod_symmetric_reweight: bool = False,
        # MoD-MLP ("attention-always"): the router gates ONLY the FFN sublayer;
        # attention (and any KV / recurrent state) runs dense for every token.
        # Composes with per-layer MoD (each routed layer owns a router) and with
        # mod_atomic_block (one decision at the block head gates every member
        # layer's FFN). Structurally immune to total bypass, so the deep-shape
        # requirement (L = block_size*(R+2)) does not apply.
        mod_mlp_only: bool = False,
        # Router-free recurrent depth (Geiping-style latent reasoning): the core
        # block (layers between the first and last block of `recurrent_block_size`
        # layers) is looped a sampled number of times per training forward.
        # No router, no aux loss, no train/infer routing mismatch. Softmax
        # attention only — recursion desyncs DeltaNet/Mamba/Griffin/Hymba state.
        use_recurrent_depth: bool = False,
        recurrent_min_iters: int = 1,
        recurrent_max_iters: int = 3,
        recurrent_sample_dist: str = "uniform",
        recurrent_block_size: int = 0,  # 0 -> infer (len(hybrid ratio) | 4)
        recurrent_eval_iters: int = 0,  # 0 -> round of the sampling mean
        # Loop-boundary mixers (modules/loop_mixers.py), both off = plain loop.
        # Parcae stable injection: core input = A_bar*h + B_bar*RMSNorm(prelude).
        recurrent_stable_injection: bool = False,
        recurrent_inject_dense_b: bool = True,
        recurrent_inject_dt_min: float = 0.1,
        recurrent_inject_dt_max: float = 1.0,
        # Hyperloop per-iteration hyper-connections over n residual streams.
        recurrent_hyper_connections: bool = False,
        recurrent_hc_streams: int = 4,
        recurrent_hc_res_bias: float = -3.0,  # sigmoid(-3)=0.047: starts as the plain loop
        # Global Workspace Trunk (GWT, dyncompute chunk 2 Pitch 1): a fixed-size
        # recurrent workspace updated at chunk boundaries by a deep trunk while
        # the token stream stays shallow and uniform-cost. Router-free; the
        # trunk iteration count is sampled per micro-batch (Arm B machinery).
        # See modules/workspace_trunk.py.
        use_workspace_trunk: bool = False,
        trunk_layers: int = 2,  # physical trunk blocks per iteration
        trunk_weight_shared: bool = False,  # share one block across trunk_layers slots
        trunk_chunk_size: int = 512,  # workspace update period (content tokens)
        trunk_workspace_slots: int = 0,  # 0 -> swa_global_count, else 256
        trunk_iters_min: int = 1,
        trunk_iters_max: int = 3,
        trunk_eval_iters: int = 0,  # 0 -> round of the sampling mean
        trunk_halt_threshold: float = 0.0,  # >0: inference halts on ||dW||/||W|| < thr
        trunk_detach_chunks: bool = False,  # truncated BPTT across chunk boundaries
        trunk_insert_layer: int = -1,  # -1 -> num_hidden_layers // 2
        # CUDA-graph the trunk's inner iteration (wall-clock only; math is
        # unchanged and the flag is inert on CPU / eval / diagnostics paths).
        # impl "compile" = torch.compile(mode="reduce-overhead") cudagraph
        # trees (handles the many-calls-per-step pattern natively); "graph" =
        # torch.cuda.make_graphed_callables, one capture per call slot.
        trunk_cuda_graph: bool = False,
        trunk_cuda_graph_impl: str = "compile",
        # "sequential" = the recurrent chunk loop (WorkspaceTrunk);
        # "parallel" = the linearized all-chunks-batched twin
        # (ParallelWorkspaceTrunk) — writes conditioned on the INITIAL
        # workspace, masked chunk-axis cumsum + rms-budget squash. A
        # functional change, screened as its own arm (swa_scribe_partrunk).
        # "rglru" / "delta_rule" = the same batched-write skeleton with a
        # gated chunk-axis scan instead of the cumsum (GatedScanWorkspaceTrunk):
        # rglru = Griffin-style input-gated decay (the workspace can FORGET —
        # the 32K mechanism), delta_rule = gated error-correcting replace
        # (idempotent overwrite). Arms swa_scribe_rglrutrunk / _deltatrunk.
        trunk_variant: str = "sequential",
        # Mixture of Pathways (MoP, dyncompute chunk 2 Pitch 2): one causal
        # commitment per token among parallel lanes of different depth. Lane
        # blocks cross-attend to the shared-stream export (never lane tokens),
        # aux loss balances expected FLOPs against a budget (in lane-blocks
        # per token). See modules/pathways.py.
        use_pathways: bool = False,
        pathway_depths: Optional[List[int]] = None,  # default [0, 2, 4]
        pathway_shared_bottom: int = 0,  # 0 -> max(1, 2 * L // 3); top = L - bottom
        pathway_budget: float = 0.0,  # expected lane-blocks/token; 0 -> mean(depths)
        pathway_aux_loss_weight: float = 0.01,
        pathway_entropy_weight: float = 0.001,
        pathway_floor_frac: float = 0.0,  # deep-lane floor (starvation guard)
        pathway_floor_steps: int = 0,  # steps the floor is enforced, then released
        # MoP parameter-sharing ladder (each sweep arm changes exactly one thing):
        # share_kv projects lane-block K/V once from the export (shared read);
        # variant "nested" makes lanes prefixes of one shared stack; "recursive"
        # iterates one weight-tied block (depths = pass counts, MoR-flavored).
        pathway_share_kv: bool = False,
        pathway_variant: str = "disjoint",  # disjoint | nested | recursive
        # Stacked pathway stages (deep-network MoP x blocked-MoD hybrid): the
        # middle region holds `pathway_stages` MoP units at consecutive layer
        # boundaries (one dense spine layer between stages); the dense bottom
        # (pathway_shared_bottom) and top stacks are never skipped. Parameters
        # grow LINEARLY in stages while the per-token depth menu grows
        # exponentially (lane choice per stage). stage_share_lanes reuses one
        # lane set across every stage (sub-linear params — the variant that
        # makes 72-128 layer targets cheap).
        pathway_stages: int = 1,
        pathway_stage_share_lanes: bool = False,
        # Lane execution efficiency. Both are PURE PERFORMANCE switches -- the
        # fast paths are mathematically identical to the dense/SDPA originals,
        # because lane blocks cross-attend only to the shared-stream export
        # (never to other lane tokens) and their MLP is position-wise, so
        # dropping a query row cannot change any other row's output.
        #   gather_lanes  -- run each nested lane block only over the tokens
        #     routed at least that deep (pathway_budget/max(depths) of the dense
        #     cost) instead of running every lane on every token and discarding
        #     all but the routed fraction at the merge.
        #   flex_attention -- express causal (∧ intra-doc) as a FlexAttention
        #     BlockMask instead of reading the dense [B,1,S,S] additive mask once
        #     per lane block. Falls back to SDPA off-CUDA or on older torch.
        pathway_gather_lanes: bool = True,
        pathway_flex_attention: bool = True,
        #   interval_mask -- build the lane BlockMask from per-row contiguous
        #     kv intervals [max(doc_start, pos-w+1), pos] (exact under packed
        #     docs) instead of the general causal∧doc∧window predicate: fewer
        #     mask_mod gathers inside the fwd/bwd flex kernels. Math-identical;
        #     kept toggleable for A/B timing.
        pathway_interval_mask: bool = True,
        pathway_gather_bucket: int = 512,  # lane gather-width rounding (padding vs recompiles)
        # mop-stack-k: gather each lane tier at a STATIC width
        # ceil(frac * seq_len) (one entry per tier, aligned with the nested
        # tier plan) with causal positional clipping — tokens routed deep
        # beyond the width keep their lane label but skip the deep blocks
        # (earliest tokens win; inclusion depends only on earlier routing, so
        # the channel stays causal). One shape per tier for the whole run,
        # zero routing host syncs, zero recompile churn. Functional change vs
        # argmax-width MoP — screened as its own arm.
        pathway_fixed_tier_fracs: Optional[list] = None,
        mod_flex_attention: bool = False,
        mod_router_mode: str = "topk",
        mod_router_chunk: int = 128,
        mod_route_bias_lr: float = 1e-3,
        mod_gather_bucket: int = 128,
        # kmod: gather the threshold selection at a STATIC width (prefix +
        # ceil(content*capacity*slack)) with causal positional clipping instead
        # of padding to the per-batch row max — one shape for the whole run,
        # zero host syncs, zero recompile churn. Functional change vs cmod
        # (overflow rows drop their latest selections), so it is a separate arm.
        mod_fixed_gather: bool = False,
        mod_fixed_gather_slack: float = 1.1,
        #   checkpoint -- whether MoP stages participate in gradient
        #     checkpointing. Checkpointing makes a component cost ~4 units
        #     (forward + recompute + 2x backward) instead of 3, so exempting the
        #     stages returns ~25% of their cost; it is affordable because
        #     gather_lanes + the compact merge cut lane activations to roughly
        #     the routed fraction. This is SELECTIVE: the backbone decoder layers
        #     keep checkpointing regardless. Set True to trade the speed back for
        #     memory if a large cell runs tight.
        pathway_checkpoint: bool = False,
        # With pathway_stage_share_lanes: number of independent lane sets,
        # shared within CONTIGUOUS groups of stages (depth bands). 1 = one set
        # for all stages (the sub-linear default); num_stages = fully unshared
        # (equivalent to share_lanes=False). Rationale: early/mid/late stages
        # read qualitatively different streams — banded sharing lets lane
        # weights specialize by depth while params stay sub-linear in stages.
        pathway_stage_lane_groups: int = 1,
        # Banks-in-export (deep-study follow-up, 2026-08-19): append the
        # register bank rows to each stage's shared-KV export so lane blocks
        # read [S + N_reg] keys, and DROP the per-spine-layer register read.
        # Bank access piggybacks on reads that already happen (+~3% keys) and
        # its frequency becomes proportional to routed depth. Registers stay
        # position-free and visible to every query (the current read
        # semantics); sink stays per-layer (softmax stabilizer). FUNCTIONAL
        # change — screen as its own arm.
        pathway_bank_export: bool = False,
        # Memory lane (requires pathway_bank_export): one extra routed lane
        # whose single block cross-attends ONLY to the bank rows — "consult
        # memory" becomes a per-token routed choice instead of an ambient
        # per-layer service. Lane depth counts as pathway_memory_lane_depth
        # (default 1) in the budget. FUNCTIONAL change — own arm.
        pathway_memory_lane: bool = False,
        pathway_memory_lane_depth: int = 1,
        # Permute-not-gather dispatch (math-identical): sort the sequence by
        # lane once per stage and run tiers as contiguous suffix slices (two
        # full-seq permutations replace the per-tier gather + per-lane
        # scatter_add row traffic — the bandwidth-bound residual the round-4/6
        # profiling identified). Requires depths monotone in lane index;
        # incompatible with pathway_memory_lane (its lane breaks the
        # nested-suffix structure).
        pathway_permute_dispatch: bool = False,
        # Stages recompute the shared-stream export (norm + KV projection)
        # every forward; with interval k > 1 only every k-th stage recomputes
        # and the stages in between reuse the previous export (STALE by the
        # intervening spine layer(s) — a functional change, screen as such).
        # Incompatible with pathway_checkpoint: a cache tensor escaping the
        # checkpointed frame would silently carry no gradient.
        pathway_export_interval: int = 1,
        #   lane_window -- band for the lane blocks' cross-attention into the
        #     shared stream. None inherits swa_window, matching the backbone;
        #     0 gives the lanes full context. It defaulted to full context by
        #     omission, which made mopd/mopd_delta the only arms in the sweep
        #     with any attention reaching past the SWA band -- a ~2.3x attention
        #     cost AND a confound against their swa_scribe twin.
        pathway_lane_window: Optional[int] = None,
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
        global_layer_window: Optional[int] = None,
        dual_stream_kmax: Optional[int] = None,
        dual_stream_route_bias_lr: float = 0.0,  # >0 enables mat_ffn-style bounded threshold control
        deltanet_use_short_conv: bool = True,
        deltanet_conv_kernel_size: int = 4,
        deltanet_true_delta_rule: bool = False,
        deltanet_allow_neg_eigval: bool = False,
        # DRoPE (Dropped Rotary Position Embeddings) configuration
        use_drope: bool = False,
        # NoPE (SmolLM3; Yang et al. 2025 "RoPE to NoRoPE and Back Again"): strip
        # position encoding from every Nth layer for the whole run. 0 disables.
        # Layers with (layer_idx + 1) % nope_every_n_layers == 0 get no RoPE.
        nope_every_n_layers: int = 0,
        # Partial RoPE on the GQA/MHA layers: rotate this fraction of head_dim. 0.0 = NoPE.
        partial_rotary_factor: float = 1.0,
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
        # Registers twin (swa_registers arch): keep the swa_global_count-sized bank but
        # read it via zero-init cross-attention instead of prepending it — self-attn
        # stays a pure FA2-native window (see modules/swa_registers.py).
        swa_use_registers: bool = False,
        # Hybrid split (swa_hybrid arch): of the swa_global_count-sized bank, this many
        # tokens are read as registers; the REMAINDER stays a prepended global prefix
        # (visible to the MLA full-attention layers, joint-softmax sink). None => all
        # tokens are registers (pure swa_registers behavior). Requires swa_use_registers.
        swa_register_count: Optional[int] = None,
        # swa_scribe arch: out-of-sequence banks merged into the SWA joint softmax
        # via LSE renormalization (FA2 window kernel kept). See modules/swa_scribe.py.
        swa_sink_count: int = 0,  # tiny sink bank, live from step 0, exact joint softmax
        swa_register_refine: bool = False,  # bidirectional refinement of the register bank
        # Bank reads (register read + scribe joint keys) run only on layers with
        # layer_idx % interval == 0; the sink stays on every layer (16 keys, it
        # is the softmax stabilizer). 1 = every layer (the historical behavior).
        # Deep-study motivation: per-layer bank reads scale with CARRIED layers
        # and dominate deep models at 32K (docs/arch_naming.md 2026-08-19).
        swa_bank_read_interval: int = 1,
        swa_fuse_register_proj: bool = True,
        # P2 `sink_diet` (2026-08-20): learned PER-HEAD sink logit instead of a
        # sink bank. Adds one softmax competitor with an all-zero value vector,
        # so it drains attention mass like a sink token at zero K/V cost and
        # without a prefix (which is what forces the SDPA [B,1,S,S] path and
        # makes 32K infeasible for the prefix family). See swa_scribe.py.
        use_logit_sinks: bool = False,
        # attn_logit_sinks (2026-10-27): the same per-head sink logit on the
        # FULL-attention layers ("attn" GQA and "mla"), modules/attn_sink.py. A
        # separate name on purpose: use_logit_sinks never reached the globals, so
        # widening it would change every banked swa_sinkdiet-family arm.
        attn_logit_sinks: bool = False,
        # attn_register_count (2026-10-27): a model-level register bank of this
        # many slots, read (zero-init gated) by every "attn" layer of a
        # hybrid_attention_ratio schedule. The DeltaNet hybrids' counterpart of the
        # SWA register read; 0 = off. Not for attention_mode="swa_globals_mla",
        # which owns its bank through swa_global_count.
        attn_register_count: int = 0,
        # Pre-hero layer recipe knobs (2026-10-31; tests/test_layer_recipe.py). All
        # default off, so every banked cell is unchanged.
        # attn_output_gate: Qwen gated attention "G1" (arXiv 2505.06708) on MLA layers,
        #   o_proj(attn * sigmoid(W_g x)), elementwise per head channel.
        # attn_qk_norm: per-head RMSNorm on MLA's q/k (nope and rope slices normed
        #   separately, before RoPE), the logit-growth guard Qwen3 uses.
        # mla_nope: MLA layers skip RoPE on their rope slice (Kimi Linear's NoPE MLA);
        #   position then comes from the DeltaNet layers only.
        # deltanet_output_gate: Gated DeltaNet's output norm + gate (Qwen3-Next / FLA),
        #   o_proj(RMSNorm_head(o) * silu(W_g x)).
        attn_output_gate: bool = False,
        attn_qk_norm: bool = False,
        mla_nope: bool = False,
        deltanet_output_gate: bool = False,
        # Agentic program knobs (2026-11-04, roadmap "Agentic innovation program").
        # mla_shared_rope_key (P11): one decoupled RoPE key shared by all MLA heads
        #   (DeepSeek-V2), cache kv_lora_rank + rope instead of kv_lora_rank + H*rope.
        # mla_diff_attn (P4): differential attention on the MLA layers (arXiv
        #   2410.05258), two queries per head, o1 - lambda*o2. One extra Q projection.
        mla_shared_rope_key: bool = False,
        mla_diff_attn: bool = False,
        # mla_position_jitter (2026-10-31, randomized positions, arXiv 2606.23687 as a
        # pretraining prior): during TRAINING, half the rows take position ids built
        # from random gaps in [1, k] (restarting at each document), so global layers
        # see positions up to ~k x the training length. 0/1 = off; eval is untouched.
        mla_position_jitter: int = 0,
        # Depth parameterization (2026-10-31, Phase 2 depth-rule test):
        # depth_scale_base_layers: CompleteP-style residual branch multiplier
        #   base / num_hidden_layers on every block (arXiv 2505.01618, alpha = 1), so a
        #   model at the base depth is unchanged. 0 = off.
        # layernorm_scaling: scale each pre-norm output by 1/sqrt(layer_idx + 1)
        #   (arXiv 2502.05795).
        depth_scale_base_layers: int = 0,
        layernorm_scaling: bool = False,
        # Message structure on the MLA layers (2026-11-04, FABLE review P1/P2;
        # modules/msg_struct.py, tests/test_msg_struct.py). All default off.
        # msg_rope (P1): MLA rope dims rotated by offset-in-message o (stock theta) and,
        #   on the lowest-frequency msg_rope_frac of the pairs, additionally by the
        #   Harmony message index m at base msg_rope_theta. m = 0 / o = pos on text with
        #   no <|start|>, which is the parent exactly.
        # msg_index_jitter: training-only random message-index gaps in [1, k] on half
        #   the rows (the mla_position_jitter idea on the m axis). 0/1 = off.
        # msg_pin_bias (P1): learned per-head logit bias on pinned keys (first system
        #   message's tool block, first user message), folded into q.k (exact on FA2).
        # msg_digest (P2): per-message digest slots written after digest_source_layer
        #   (-1 = the first MLA layer); MLA layers attend over the last digest_window raw
        #   tokens plus the digests of earlier complete messages (layers above the source).
        #   digest_slots_tool / _other slots per tool / other message, at most
        #   digest_max_messages messages per row.
        # pause_decision_only (P12): with use_pause_stream, the prediction stream is read
        #   (and the shared FFN blends) only at Harmony decision tokens: an assistant
        #   <|channel|>, the rest of a commentary (tool-call) message, and the header +
        #   first pause_final_tokens content tokens of a final message. Masked, not
        #   compacted: the stream still runs everywhere (no compute saved).
        # msg_token_ids: Harmony ids (None = production SuperBPE vocab 56000).
        msg_rope: bool = False,
        msg_rope_theta: float = 1000.0,
        msg_rope_frac: float = 0.5,
        msg_index_jitter: int = 0,
        msg_pin_bias: bool = False,
        msg_digest: bool = False,
        digest_window: int = 2048,
        digest_slots_tool: int = 8,
        digest_slots_other: int = 4,
        digest_max_messages: int = 128,
        digest_source_layer: int = -1,
        pause_decision_only: bool = False,
        pause_final_tokens: int = 16,
        msg_token_ids: Optional[Dict[str, List[int]]] = None,
        # swa_lse_grad_exact (2026-10-27): window-LSE arms (swa_scribe merge) take
        # the exact FA2 LSE gradient. False = the pre-7c28e10 dropped-gradient
        # training, kept ONLY for same-commit control cells.
        swa_lse_grad_exact: bool = True,
        # W1 `mat_ffn` (2026-08-20): per-TOKEN nested FFN width. `mat_ffn_widths`
        # are absolute intermediate sizes (nested prefixes of intermediate_size);
        # `mat_ffn_budget` is the target MEAN routed width (0 -> mean of the
        # menu, which is what keeps a random-init bench honest).
        # D2 `chunk_mop` (2026-08-20): one lane commitment per CONTIGUOUS CHUNK
        # of this many tokens (1 = per token, the default). Decided from the
        # chunk's FIRST token so the rule stays causal.
        # S1 `hourglass` (2026-08-20): full-resolution outer layers + a deep
        # middle stack that runs on a POOLED sequence (S / hourglass_pool_ratio
        # segments). num_hidden_layers counts only the outer layers; the middle
        # stack is owned by the HourglassMiddle module.
        # P1 `dual_stream` (2026-08-20): a WIDE slow stream over a causally
        # selected subset of tokens, every `dual_stream_every` layers. Separates
        # parameter capacity from compute per token.
        use_dual_stream: bool = False,
        dual_stream_every: int = 4,
        dual_stream_width_mult: float = 2.0,
        dual_stream_frac: float = 0.2,
        dual_stream_start_layer: int = 8,  # fixed-compute prefix (design docs)
        use_hourglass: bool = False,
        hourglass_middle_layers: int = 0,
        hourglass_pool_ratio: int = 4,
        hourglass_insert_layer: Optional[int] = None,
        # MEASURED 2026-08-20: varlen is NOT faster here and is sometimes worse.
        # deep80: 722.6 (varlen) vs 720.5 (sdpa) -- a wash. deep128: 1275.2 vs
        # 1105.8 -- varlen is 15% WORSE. The segment-level mask is only
        # [B,1,K,K] with K = S/pool_ratio, so materialising it is cheap, while
        # the varlen path pays index/unpad bookkeeping per layer that grows with
        # depth. Default is therefore "sdpa"; "varlen" is kept because it may
        # still win at 32K, where K is 8x larger and the mask is not cheap.
        hourglass_attn_mode: str = "sdpa",  # "sdpa" (default) | "varlen"
        pathway_route_chunk: int = 1,
        use_mat_ffn: bool = False,
        mat_ffn_widths: Optional[List[int]] = None,
        mat_ffn_budget: float = 0.0,
        mat_ffn_aux_weight: float = 0.01,
        # "sorted" = contiguous-prefix dispatch at static tier widths (fast,
        # one dynamo graph). "gather" = the original index_select/index_add_
        # path, kept so the optimization pass can A/B them in a single job.
        mat_ffn_dispatch: str = "sorted",
        # Fixed-compute prefix/suffix (design docs): the first `dyn_start` and
        # last `dyn_end` layers run a FULL-width SwiGLU for every token; only the
        # middle reasoning layers route. Mirrors mod_num_always_used_start/_end.
        mat_ffn_dyn_start: int = 8,
        mat_ffn_dyn_end: int = 8,
        # Route every Nth layer inside the dynamic band; the layers between run
        # a plain SwiGLU at the BUDGET width (no router, no gather, no sync).
        mat_ffn_route_interval: int = 4,
        # ---- 2026-08-22 nesting round (docs/mat_ffn_nesting.md) ----
        # SKEWED tier occupancy. `mat_ffn_tier_fracs` is the target share of
        # tokens per tier, smallest first; it must sum to 1 and match
        # `mat_ffn_widths` in length. None keeps the legacy behaviour, where
        # `utils.forced_routing.budget_fractions` derives the CLOSEST-TO-UNIFORM
        # vector hitting `mat_ffn_budget` -- the opposite of "small tiers used
        # more often". Setting this replaces both the capacity clip's caps and
        # the expected width used for accounting (budget := sum(pi_k * w_k)).
        mat_ffn_tier_fracs: Optional[List[float]] = None,
        # Width of the FIXED-COMPUTE prefix/suffix layers. None -> full
        # intermediate_size (legacy). This matters as soon as the prior is
        # skewed: at matched ACTIVE params a skewed prior pushes d_ff up, and
        # the dyn_start+dyn_end layers run FULL width for every token, so their
        # share of the active budget grows with the skew (29% of the deep80 FFN
        # budget at quarters, ~50% at d_ff=8513). Decoupling them is what lets
        # the nest get deep without the fixed layers eating the saving.
        mat_ffn_fixed_width: Optional[int] = None,
        # mat_ffn_layer_kinds (2026-10-27): route the nest only on layers whose
        # hybrid_attention_ratio entry is in this list (e.g. ["attn"]); every other
        # layer runs a plain SwiGLU at mat_ffn_fixed_width (the backbone d_ff).
        # None = every layer in the dynamic band, the historical behaviour.
        mat_ffn_layer_kinds: Optional[List[str]] = None,
        # Nested-dropout / sandwich WARMUP. For the first `warmup_frac` of
        # training the tier assignment is SAMPLED from a prior instead of taken
        # from the router argmax, so every width is trained on its share of
        # tokens from step 0 and the widest tier cannot be starved by an early
        # router. The prior anneals uniform -> `mat_ffn_tier_fracs` over
        # `warmup_stages` discrete stages (discrete because the per-tier cap
        # sizes a STATIC window in the sorted dispatch: a continuously varying
        # cap would recompile every step). The router still receives gradient
        # throughout, via the straight-through tail gate.
        mat_ffn_warmup_frac: float = 0.0,
        mat_ffn_warmup_stages: int = 4,
        # LARGEST-GRANULARITY UPWEIGHT (MatFormer 2310.07707, appendix: they
        # tune the per-granularity sampling probabilities and report that
        # "upweighting the loss for the largest granularity gives a boost to
        # the performance of bigger models while modest degradation to the
        # performance of the smaller submodels").
        #
        # Our per-token analogue. During the warmup the tier assignment is
        # SAMPLED from a prior, and a token's tier decides which weights it
        # exercises -- so raising the widest tier's sampling share is exactly
        # MatFormer's knob. It matters because the nested prefix is SHARED:
        # tier 0's columns take gradient from every token, and this changes the
        # mixture on them toward the widest configuration. (A gradient scale on
        # the widest tier's EXCLUSIVE columns would be a near no-op under Adam,
        # which is scale-invariant per parameter -- the shared prefix is the
        # only place the reweighting can act.)
        #
        # lambda re-shapes only the warmup's STARTING prior: base = normalize
        # ([1, ..., 1, lambda]) instead of uniform. The anneal endpoint is still
        # exactly `mat_ffn_tier_fracs`, so the steady state, E[w] and the
        # matched-N solve are all bitwise unchanged, and the extra warmup FLOPs
        # are already priced by the submitter's warm0 ms/step term. 1.0 = off.
        #
        # LIMITATION, state it when quoting a result: this acts during
        # `mat_ffn_warmup_frac` of training only. It does NOT address the
        # widest tier's steady-state occupancy (6% for skew4).
        mat_ffn_warmup_top_weight: float = 1.0,
        # "mean" = legacy ((mean_w - budget)/d_ff)^2, which constrains only the
        # MEAN and is blind to a bimodal split that never uses the middle tiers.
        # "kl" = KL(batch-marginal router distribution || tier_fracs), which
        # pins the whole shape. "none" = no aux (the capacity clip is already a
        # hard guarantee; the aux is a backstop, not the mechanism).
        # PERMANENT EXPLORATION FLOOR (2026-08-22). Fraction of tokens whose
        # tier is DRAWN from the target prior instead of taken from the router's
        # argmax, for the whole run rather than just warmup. 0.0 = off (the
        # pre-2026-08-22 behaviour, bitwise).
        #
        # WHY THIS EXISTS. Zero occupancy is an ABSORBING STATE of the LM
        # objective, verified numerically: tier k's only LM gradient reaches the
        # router through the straight-through gate `tail[:, k]`, and that gate is
        # multiplied by the hard mask `(choice >= k)` in BOTH dispatch paths
        # (`_forward_sorted`, `_forward_grouped`). A token that did not choose
        # tier >= k contributes exactly zero gradient to tail_k, so the LM loss
        # can only ever REDUCE a tier's occupancy -- it carries no counterfactual
        # signal about whether a tier the router has stopped using would have
        # helped. With every token at tier 0 the router gradient is identically
        # zero (tier 0 is ungated), and the router is frozen with whatever logit
        # margins it had. Measured on `mat_ffn_skew4` at ar10m step 6246: the
        # widest tier sits at EXACTLY 0.0000 occupancy in all five routed layers
        # and trails the argmax by 3.0-8.0 logits, while the KL aux -- the only
        # remaining escape -- contributes 2.5e-4 to 3.8e-3 nats against a
        # 0.036-nat seed-noise bar.
        #
        # A floor of eps guarantees occupancy >= eps * tier_fracs[k] > 0 forever,
        # which makes the absorbing state unreachable and keeps every tier's
        # gradient alive. The draw is per-token i.i.d. and content-independent,
        # so it moves no information between positions (causally safe), and it
        # costs zero FLOPs: `_forward_sorted`/`_forward_grouped` size their tier
        # slices from the PRIOR, not from demand, so those columns are computed
        # every step regardless of whether any token asked for them.
        #
        # Read the arm's final occupancy against `eps * tier_fracs` to separate
        # the floor from genuine router preference.
        mat_ffn_explore_frac: float = 0.0,
        mat_ffn_gate_mode: str = "ste",
        mat_ffn_gate_anneal_frac: float = 0.0,
        mat_ffn_mixture_frac: float = 1.0,
        deltanet_dyn_rank: bool = False,
        deltanet_forced_rank=None,
        deltanet_rank_tiers=None,
        deltanet_rank_prior=None,
        deltanet_rank_chunk: int = 128,
        deltanet_rank_thresh_lr: float = 0.01,
        mat_ffn_route_mode: str = "argmax",
        mat_ffn_route_thresh_lr: float = 0.01,
        mat_ffn_route_ctrl: str = "wish",
        mat_ffn_cap_headroom: float = 1.0,
        mat_ffn_explore_from: str = "target",
        mat_ffn_tier_scale_init=None,
        mat_ffn_prior_bias_init: bool = False,
        mat_ffn_track_occupancy: bool = True,
        mat_ffn_aux_mode: str = "mean",
        # Block-sparse long-range attention (2026-08-20): the SUBSTITUTIVE
        # alternative to scribes. bsa_mode "moba" = parameter-less top-k block
        # gate (2502.13189); "hils" = learned chunk keys whose retrieval score
        # enters the softmax so the LM loss trains the selector (2607.02980).
        # bsa_min_seq implements the InfLLM-V2 observation (2509.24663) that
        # trainable sparsity is a loss at short context and a win at long: below
        # this length the far branch is skipped entirely and the arm is exactly
        # its backbone. 0 = always on.
        use_block_sparse_attn: bool = False,
        bsa_mode: str = "moba",
        bsa_block_size: int = 128,
        bsa_top_k: int = 4,
        bsa_min_seq: int = 0,
        bsa_layer_interval: int = 1,
        # DeltaMemory (2026-08-20): delta-rule associative memory as a narrow
        # SIDE branch on the residual stream, not a layer replacement. Zero-init
        # output, so the arm starts identical to its backbone.
        # delta_mem_variant: "delta" = Widrow-Hoff overwrite (associative,
        # triangular solve); "gla" = decay + accumulate (associative, no solve,
        # cannot erase); "ema" = per-head value EMA (vector state, no key, not
        # associative -- the control that separates recall from any-long-range).
        use_delta_memory: bool = False,
        delta_mem_variant: str = "delta",
        delta_mem_heads: int = 4,
        delta_mem_head_dim: int = 64,
        delta_mem_chunk: int = 64,
        delta_mem_interval: int = 4,
        delta_mem_band_only: bool = True,
        delta_mem_checkpoint: bool = True,
        use_scribe_bank: bool = False,  # chunk-committed content-written scribe slots
        scribe_slots_per_chunk: int = 4,
        scribe_chunk_size: int = 512,
        scribe_write_layer: int = -1,  # -1 -> num_hidden_layers // 2 (must be MoD-block-aligned)
        scribe_write_topk: int = 0,  # 0 = dense write; >0 = hard top-k bottleneck
        trunk_write_topk: int = 0,  # same knob for the workspace trunk's write
        # swa_deltabank arch: the register bank becomes STATEFUL — cumulative
        # chunk-writes at one write layer; a token in chunk c reads the bank as
        # written by chunks < c (doc-reset + doc-gated under intra-doc masking).
        # Zero-init: at init identical to the same config without it. The only
        # bank design whose mechanism can carry content beyond window+bank reach
        # (the capability every probed arch lacked, 2026-07-20). See
        # modules/delta_bank.py. Requires swa_use_registers.
        use_delta_bank: bool = False,
        delta_chunk_size: int = 512,
        delta_write_layer: int = -1,  # -1 -> num_hidden_layers // 2 (must be MoD-block-aligned)
        delta_write_topk: int = 0,  # 0 = dense write; >0 = hard top-k bottleneck
        # Init of the learnable deviation budget, in units of ||B0||_rms. The
        # unbounded scan blew the state to 3090x the bank scale and cost 0.92
        # nats (see modules/delta_bank.py).
        delta_state_clamp: float = 1.0,
        # Per-Layer Embeddings (Gemma-3n/4 style). Each decoder layer gets its own
        # small token embedding (vocab x ple_dim), projected to hidden and added to
        # that layer's input. The PLE tables are large (vocab x ple_dim x layers) but
        # are sparse lookups streamable from cheap memory -> they inflate TOTAL params
        # without inflating ACTIVE compute/fast-memory footprint.
        use_ple: bool = False,
        ple_dim: int = 256,
        # MatFormer (Matryoshka) nested FFNs. A smaller FFN is the top-left
        # sub-matrix of the larger one; matformer_active_size selects how much of the
        # intermediate dim is active per forward (None -> full). The full intermediate
        # counts toward TOTAL params; the active slice is the ACTIVE compute.
        use_matformer: bool = False,
        matformer_granularities: Optional[List[int]] = None,  # active sizes to sample in training
        matformer_active_size: Optional[int] = None,  # fixed active size (inference / eval)
        # MTP variant ("deepseek" = sequential chain — current behavior; "gloeckle" = 4 parallel
        # heads with d×d adapters and INDEPENDENT unembedding heads, training-only).
        mtp_variant: str = "deepseek",
        mtp_predict_offsets: Optional[
            List[int]
        ] = None,  # Used only by gloeckle; default [2,3,4,5].
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
        # Token-Superposition Training (TST). Two-phase pretraining:
        # phase (i) bag-of-token-embeddings + multi-hot CE on next bag-of-tokens;
        # phase (ii) standard NTP. `tst_enabled` only declares the model *may* use TST.
        # Runtime phase is controlled by `tst_active` (toggled by TSTController/callback).
        tst_enabled: bool = False,
        tst_active: bool = False,
        tst_bag_size: int = 1,
        tst_recovery_ratio: float = 0.3,
        tst_loss_weighting: str = "uniform",  # "uniform" | "power_law"
        # Delphi-style recipe (openathena.ai/blog/delphi). When `delphi_recipe`
        # is True the training stack swaps in AdamH + WSD + token-horizon LR
        # via the patch in scripts/llamafactory_train.py (and the equivalent
        # torchtitan branch). Constants here are the recipe's reference
        # values; per-run (tokens, batch) come from training-config fields.
        delphi_recipe: bool = False,
        delphi_optimizer: str = "adamh",  # "adamh" | "adamw"
        delphi_schedule: str = "wsd",  # "wsd" | "cosine" | "linear"
        delphi_warmup_frac: float = 0.10,
        delphi_decay_frac: float = 0.20,
        delphi_eta0_proj: float = 6.30e-3,
        delphi_eta0_adam: float = 6.56e-4,
        delphi_batch_ref: int = 64,
        delphi_token_ref: float = 2.5e9,
        delphi_horizon_exp: float = 0.30,
        delphi_epsilon0: float = 1.85e-8,
        delphi_beta2_max: float = 0.9999,
        delphi_beta2_min: float = 0.9,
        delphi_grad_clip: float = 0.1,
        # Run-context fields populated by the training script (not by presets).
        # These are needed by the Hyperball / horizon-LR formulas at Trainer
        # construction time. The model itself never reads them.
        delphi_run_tokens: float = 0.0,
        delphi_run_batch: int = 0,
        delphi_run_total_steps: int = 0,
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
        self.rope_scaling_scope = rope_scaling_scope
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
        self.use_intra_doc_masking = use_intra_doc_masking

        # Multi-Token Prediction configuration
        self.use_mtp = use_mtp
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_loss_chunk_tokens = int(mtp_loss_chunk_tokens)
        self.mtp_soft = bool(mtp_soft)
        self.use_bag_lookahead = bool(use_bag_lookahead)
        self.bag_lookahead_k = int(bag_lookahead_k)
        self.bag_lookahead_weight = float(bag_lookahead_weight)
        self.bag_vocab = int(bag_vocab)
        self.bag_share_head = bool(bag_share_head)
        self.bag_loss_chunk_tokens = int(bag_loss_chunk_tokens)
        self.use_bank_reconstruction = bool(use_bank_reconstruction)
        self.bank_recon_samples = int(bank_recon_samples)
        self.bank_recon_weight = float(bank_recon_weight)
        self.use_latent_feedback = bool(use_latent_feedback)
        # Normalised to a list so it round-trips through JSON unchanged; a tuple
        # default would come back from a saved config as a list and compare
        # unequal in any test that checks the config survived a save/load.
        self.latent_feedback_pass_mix = (
            [0.75, 0.22, 0.03]
            if latent_feedback_pass_mix is None
            else [float(x) for x in latent_feedback_pass_mix]
        )
        self.latent_feedback_mixin_p = float(latent_feedback_mixin_p)
        self.latent_feedback_seed = int(latent_feedback_seed)
        self.latent_feedback_eval_passes = int(latent_feedback_eval_passes)
        self.use_pause_stream = bool(use_pause_stream)
        self.pause_shared_ffn = bool(pause_shared_ffn)
        self.mtp_num_layers = mtp_num_layers
        self.mtp_share_head = mtp_share_head
        self.mtp_dropout = mtp_dropout

        # Mixture of Depths configuration
        self.use_mod = use_mod
        self.mod_capacity = mod_capacity
        self.mod_every_n_layers = mod_every_n_layers
        self.mod_atomic_block = bool(mod_atomic_block)
        self.mod_block_size = int(mod_block_size)
        self.mod_io_blocks_dense = bool(mod_io_blocks_dense)
        self.mod_allow_total_bypass = bool(mod_allow_total_bypass)
        self.mod_aux_loss_weight = mod_aux_loss_weight
        self.mod_use_aux_predictor = mod_use_aux_predictor
        self.mod_gate_fn = str(mod_gate_fn)
        self.mod_tanh_alpha = float(mod_tanh_alpha)
        self.mod_capacity_warmup_steps = int(mod_capacity_warmup_steps)
        self.mod_symmetric_reweight = bool(mod_symmetric_reweight)
        self.mod_mlp_only = bool(mod_mlp_only)

        # Router-free recurrent depth
        self.use_recurrent_depth = bool(use_recurrent_depth)
        self.recurrent_min_iters = int(recurrent_min_iters)
        self.recurrent_max_iters = int(recurrent_max_iters)
        self.recurrent_sample_dist = str(recurrent_sample_dist)
        self.recurrent_block_size = int(recurrent_block_size)
        self.recurrent_eval_iters = int(recurrent_eval_iters)
        self.recurrent_stable_injection = bool(recurrent_stable_injection)
        self.recurrent_inject_dense_b = bool(recurrent_inject_dense_b)
        self.recurrent_inject_dt_min = float(recurrent_inject_dt_min)
        self.recurrent_inject_dt_max = float(recurrent_inject_dt_max)
        self.recurrent_hyper_connections = bool(recurrent_hyper_connections)
        self.recurrent_hc_streams = int(recurrent_hc_streams)
        self.recurrent_hc_res_bias = float(recurrent_hc_res_bias)

        # Global Workspace Trunk (GWT)
        self.use_workspace_trunk = bool(use_workspace_trunk)
        self.trunk_layers = int(trunk_layers)
        self.trunk_weight_shared = bool(trunk_weight_shared)
        self.trunk_chunk_size = int(trunk_chunk_size)
        self.trunk_workspace_slots = int(trunk_workspace_slots)
        self.trunk_iters_min = int(trunk_iters_min)
        self.trunk_iters_max = int(trunk_iters_max)
        self.trunk_eval_iters = int(trunk_eval_iters)
        self.trunk_halt_threshold = float(trunk_halt_threshold)
        self.trunk_detach_chunks = bool(trunk_detach_chunks)
        self.trunk_insert_layer = int(trunk_insert_layer)
        self.trunk_cuda_graph = bool(trunk_cuda_graph)
        self.trunk_cuda_graph_impl = str(trunk_cuda_graph_impl)
        self.trunk_variant = str(trunk_variant)

        # Mixture of Pathways (MoP)
        self.use_pathways = bool(use_pathways)
        self.pathway_depths = list(pathway_depths) if pathway_depths is not None else None
        self.pathway_shared_bottom = int(pathway_shared_bottom)
        self.pathway_budget = float(pathway_budget)
        self.pathway_aux_loss_weight = float(pathway_aux_loss_weight)
        self.pathway_entropy_weight = float(pathway_entropy_weight)
        self.pathway_floor_frac = float(pathway_floor_frac)
        self.pathway_floor_steps = int(pathway_floor_steps)
        self.pathway_share_kv = bool(pathway_share_kv)
        self.pathway_variant = str(pathway_variant)
        self.pathway_stages = int(pathway_stages)
        self.pathway_stage_share_lanes = bool(pathway_stage_share_lanes)
        self.pathway_gather_lanes = bool(pathway_gather_lanes)
        self.pathway_flex_attention = bool(pathway_flex_attention)
        self.pathway_interval_mask = bool(pathway_interval_mask)
        self.pathway_gather_bucket = int(pathway_gather_bucket)
        self.pathway_fixed_tier_fracs = (
            None
            if pathway_fixed_tier_fracs is None
            else [float(f) for f in pathway_fixed_tier_fracs]
        )
        # Block-sparse BlockMask for the block-atomic MoD compact path instead of
        # two dense [B,1,k,k] additive masks (268 MB at B16/k2080/bf16, and an
        # additive mask also disqualifies SDPA's flash backend). CUDA-only.
        #
        # DEFAULT OFF pending a GPU parity check: the flex mask_mod is a
        # REIMPLEMENTATION of build_mod_compact_masks' visibility rule, and the
        # two can only be compared where flex runs, i.e. on CUDA. The document
        # fix in build_mod_compact_masks is independent of this flag and is
        # active either way. Flip to True once
        # tests/test_mod_intra_doc.py::TestFlexParity has run green on a GPU.
        self.mod_flex_attention = bool(mod_flex_attention)
        # How the block router turns per-token scores into a routed set. The SCORE
        # is causal in every mode; the SELECTION RULE is what differs.
        #   "topk"          MoD as published: an order statistic over the WHOLE
        #                   sequence, so a token's fate depends on its future and
        #                   (when documents are packed) on its neighbours'
        #                   documents. The default, because it is what every
        #                   existing checkpoint trained with.
        #   "threshold"     score + bias > 0. Strictly causal, no cross-document
        #                   channel, and identical to the rule inference already
        #                   uses -- so train and generation finally agree. Capacity
        #                   holds only in expectation, hence `mod_gather_bucket`.
        if mod_router_mode not in {"topk", "threshold"}:
            raise ValueError(f"mod_router_mode must be topk|threshold, got {mod_router_mode!r}")
        self.mod_router_mode = str(mod_router_mode)
        self.mod_router_chunk = int(mod_router_chunk)
        self.mod_route_bias_lr = float(mod_route_bias_lr)
        self.mod_gather_bucket = int(mod_gather_bucket)
        self.mod_fixed_gather = bool(mod_fixed_gather)
        self.mod_fixed_gather_slack = float(mod_fixed_gather_slack)
        self.pathway_checkpoint = bool(pathway_checkpoint)
        self.pathway_stage_lane_groups = int(pathway_stage_lane_groups)
        if self.pathway_stage_lane_groups < 1:
            raise ValueError(
                f"pathway_stage_lane_groups must be >= 1, got {self.pathway_stage_lane_groups}"
            )
        self.pathway_bank_export = bool(pathway_bank_export)
        self.pathway_memory_lane = bool(pathway_memory_lane)
        self.pathway_memory_lane_depth = int(pathway_memory_lane_depth)
        self.pathway_permute_dispatch = bool(pathway_permute_dispatch)
        if self.pathway_memory_lane and not self.pathway_bank_export:
            raise ValueError("pathway_memory_lane requires pathway_bank_export=True")
        if self.pathway_memory_lane_depth < 1:
            raise ValueError(
                f"pathway_memory_lane_depth must be >= 1, got {self.pathway_memory_lane_depth}"
            )
        if self.pathway_permute_dispatch and self.pathway_memory_lane:
            raise ValueError(
                "pathway_permute_dispatch is incompatible with pathway_memory_lane: "
                "the memory lane's token set is not a suffix of the depth tiers."
            )
        self.pathway_export_interval = int(pathway_export_interval)
        if self.pathway_export_interval < 1:
            raise ValueError(
                f"pathway_export_interval must be >= 1, got {self.pathway_export_interval}"
            )
        if self.pathway_export_interval > 1 and self.pathway_checkpoint:
            raise ValueError(
                "pathway_export_interval > 1 is incompatible with pathway_checkpoint: "
                "the cached export would escape the checkpointed frame without gradient."
            )
        self.pathway_lane_window = None if pathway_lane_window is None else int(pathway_lane_window)

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
        self.dual_stream_route_bias_lr = float(dual_stream_route_bias_lr or 0.0)
        self.dual_stream_kmax = int(dual_stream_kmax) if dual_stream_kmax else None
        self.global_layer_window = (
            int(global_layer_window) if global_layer_window is not None else None
        )
        self.deltanet_use_short_conv = deltanet_use_short_conv
        self.deltanet_conv_kernel_size = deltanet_conv_kernel_size
        self.deltanet_true_delta_rule = deltanet_true_delta_rule
        self.deltanet_allow_neg_eigval = deltanet_allow_neg_eigval

        # DRoPE (Dropped Rotary Position Embeddings) configuration
        self.use_drope = use_drope
        self.nope_every_n_layers = nope_every_n_layers
        self.partial_rotary_factor = partial_rotary_factor
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
        self.griffin_num_heads = (
            griffin_num_heads if griffin_num_heads is not None else num_attention_heads
        )

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
        self.swa_use_registers = bool(swa_use_registers)
        self.swa_register_count = None if swa_register_count is None else int(swa_register_count)
        self.swa_sink_count = int(swa_sink_count)
        self.swa_register_refine = bool(swa_register_refine)
        self.swa_bank_read_interval = int(swa_bank_read_interval)
        # Share q_proj/o_proj between a register layer's window branch and its
        # bank read. Exactly equivalent arithmetic (o_proj is bias-free, and the
        # read has always used the un-rotated q_proj output), so this is a pure
        # optimisation -- it removes one full q_proj and one full o_proj GEMM per
        # bank-reading layer.
        #
        # DEFAULT ON since 2026-08-27, when the A/B landed on two GPUs (jobs
        # 56294 A100 / 56295 A6000, deep80 ~1018M active, S=4096, vocab 56000,
        # adamw_fused, grad-ckpt, compile). It removes ~309 aten::mm calls per
        # step -- the 60 duplicate q_proj/o_proj pairs plus their backward and
        # grad-ckpt recompute -- taking mm self-CUDA from 8.535s to 8.040s
        # against dense qwen3_like's 8.015s, i.e. back to GEMM parity:
        #     arm                  before -> after (x dense, A100 / A6000 @ B=8)
        #     swa_sinkdiet_gqa     1.081 -> 1.025  /  1.083 -> 1.022
        #     mat_ffn_thresh_gqa   1.161 -> 1.105  /  1.131 -> 1.070
        #     think_suffix_gqa     1.026 -> 1.027  /  1.016 -> 1.016   (no-op)
        # think_suffix is a no-op BY CONSTRUCTION and that is the tell that the
        # accounting is right: it reads the bank through pathway_bank_export
        # (~5 flex dispatches), not through 60 per-layer register reads, so it
        # has no duplicate projection to remove. SWAHybridAttention pins the
        # flag off (it adds the read AFTER o_proj -- see that class).
        #
        # Equivalent, not bitwise identical: the fused path sums before o_proj,
        # so bf16 rounding differs at dtype epsilon. Cells landed before this
        # date ran unfused; the difference is far below the 0.036-nat bar.
        self.swa_fuse_register_proj = bool(swa_fuse_register_proj)
        self.use_logit_sinks = bool(use_logit_sinks)
        self.attn_logit_sinks = bool(attn_logit_sinks)
        self.attn_register_count = int(attn_register_count or 0)
        self.attn_output_gate = bool(attn_output_gate)
        self.attn_qk_norm = bool(attn_qk_norm)
        self.mla_nope = bool(mla_nope)
        self.deltanet_output_gate = bool(deltanet_output_gate)
        self.mla_shared_rope_key = bool(mla_shared_rope_key)
        self.mla_diff_attn = bool(mla_diff_attn)
        if (self.mla_shared_rope_key or self.mla_diff_attn) and not getattr(self, "use_mla", False):
            raise ValueError("mla_shared_rope_key / mla_diff_attn act on MLA layers only")
        self.mla_position_jitter = int(mla_position_jitter or 0)
        self.depth_scale_base_layers = int(depth_scale_base_layers or 0)
        self.layernorm_scaling = bool(layernorm_scaling)
        if (self.attn_output_gate or self.attn_qk_norm or self.mla_nope) and not getattr(
            self, "use_mla", False
        ):
            raise ValueError("attn_output_gate / attn_qk_norm / mla_nope act on MLA layers only")
        if self.mla_position_jitter < 0 or self.depth_scale_base_layers < 0:
            raise ValueError("mla_position_jitter and depth_scale_base_layers must be >= 0")
        self.swa_lse_grad_exact = bool(swa_lse_grad_exact)
        if self.attn_register_count < 0:
            raise ValueError(f"attn_register_count must be >= 0, got {attn_register_count}")
        self.use_dual_stream = bool(use_dual_stream)
        self.dual_stream_every = int(dual_stream_every)
        self.dual_stream_width_mult = float(dual_stream_width_mult)
        self.dual_stream_frac = float(dual_stream_frac)
        self.dual_stream_start_layer = int(dual_stream_start_layer)
        if self.use_dual_stream:
            if self.dual_stream_every < 1:
                raise ValueError("dual_stream_every must be >= 1")
            if not (0.0 < self.dual_stream_frac <= 1.0):
                raise ValueError(f"dual_stream_frac must be in (0,1], got {self.dual_stream_frac}")
        self.use_hourglass = bool(use_hourglass)
        self.hourglass_middle_layers = int(hourglass_middle_layers)
        self.hourglass_pool_ratio = int(hourglass_pool_ratio)
        self.hourglass_attn_mode = str(hourglass_attn_mode)
        self.hourglass_insert_layer = (
            None if hourglass_insert_layer is None else int(hourglass_insert_layer)
        )
        if self.use_hourglass:
            if self.hourglass_middle_layers < 1:
                raise ValueError("use_hourglass requires hourglass_middle_layers >= 1")
            if self.hourglass_pool_ratio < 2:
                raise ValueError(
                    f"hourglass_pool_ratio must be >= 2 (1 would pool nothing), "
                    f"got {self.hourglass_pool_ratio}"
                )
        self.pathway_route_chunk = int(pathway_route_chunk)
        if self.pathway_route_chunk < 1:
            raise ValueError(f"pathway_route_chunk must be >= 1, got {self.pathway_route_chunk}")
        self.use_mat_ffn = bool(use_mat_ffn)
        self.mat_ffn_widths = list(mat_ffn_widths) if mat_ffn_widths else None
        self.mat_ffn_budget = float(mat_ffn_budget)
        self.mat_ffn_aux_weight = float(mat_ffn_aux_weight)
        self.mat_ffn_dispatch = str(mat_ffn_dispatch)
        self.mat_ffn_dyn_start = int(mat_ffn_dyn_start)
        self.mat_ffn_dyn_end = int(mat_ffn_dyn_end)
        self.mat_ffn_route_interval = int(mat_ffn_route_interval)
        self.mat_ffn_tier_fracs = (
            list(mat_ffn_tier_fracs) if mat_ffn_tier_fracs is not None else None
        )
        self.mat_ffn_fixed_width = (
            int(mat_ffn_fixed_width) if mat_ffn_fixed_width is not None else None
        )
        self.mat_ffn_layer_kinds = (
            list(mat_ffn_layer_kinds) if mat_ffn_layer_kinds is not None else None
        )
        self.mat_ffn_warmup_frac = float(mat_ffn_warmup_frac)
        self.mat_ffn_warmup_stages = int(mat_ffn_warmup_stages)
        self.mat_ffn_warmup_top_weight = float(mat_ffn_warmup_top_weight)
        if self.mat_ffn_warmup_top_weight <= 0:
            raise ValueError(
                "mat_ffn_warmup_top_weight must be > 0, got " f"{self.mat_ffn_warmup_top_weight}"
            )
        self.mat_ffn_explore_frac = float(mat_ffn_explore_frac)
        # Always-on, sync-free tier-occupancy counters; the trainer drains them
        # every mat_ffn_stats_interval steps. Off only for bitwise-legacy repro.
        # Router gradient mode. "ste" is the shipped straight-through path
        # (forward gate == 1, members only). "mixture" trains the exact soft
        # mixture over tiers so the router sees a per-token counterfactual --
        # full-width compute, so it buys router quality rather than FLOPs.
        # Zero-init per-tier output scale (LayerScale/ReZero): 0.0 makes adding a
        # tier an exact no-op at init, so an untrained wide slice cannot inject
        # noise the router then learns to avoid. None = parameter absent.
        self.mat_ffn_tier_scale_init = (
            None if mat_ffn_tier_scale_init is None else float(mat_ffn_tier_scale_init)
        )
        # Init the router bias to log(prior) so step 0's marginal IS the target.
        self.mat_ffn_prior_bias_init = bool(mat_ffn_prior_bias_init)
        # Distribution the exploration floor samples from: "target" (the skewed
        # prior) or "uniform" (4x more mass on the widest tier at k=4).
        self.mat_ffn_explore_from = str(mat_ffn_explore_from)
        if self.mat_ffn_explore_from not in ("target", "uniform"):
            raise ValueError(
                "mat_ffn_explore_from must be 'target' or 'uniform', got "
                f"{self.mat_ffn_explore_from!r}"
            )
        self.mat_ffn_gate_mode = str(mat_ffn_gate_mode)
        # Fraction of training over which gate_anneal goes 0 -> 1, i.e. from the
        # counterfactual mixture to the exact hard staircase. 0 = no annealing
        # (mixture all the way, which trains a different function than it
        # evaluates). The trainer drives the per-step value.
        self.mat_ffn_gate_anneal_frac = float(mat_ffn_gate_anneal_frac)
        # Fraction of tokens given the FULL counterfactual mixture gate; the rest
        # get the hard gate. 1.0 = every token (the original full mixture),
        # 0.0 = none (exactly the hard gate). See mixture_frac in swiglu.py.
        self.mat_ffn_mixture_frac = float(mat_ffn_mixture_frac)
        # "argmax" (default) or "threshold". Threshold routing makes occupancy
        # STRUCTURAL via a controller-adapted threshold, so collapse is
        # impossible rather than merely discouraged. See route_mode in swiglu.py.
        # ---- DeltaNet dynamic state RANK (2026-08-24) ----
        # The state is [d_k, d_v]; zeroing q[..., r:] and k[..., r:] makes rows
        # r: unreachable (never written -- the update is an outer product in k;
        # never read -- the output is S^T q), so the layer runs at an exact
        # rank-r state, nested the same way the mat_ffn widths are.
        #
        # This cannot save FLOPs with the stock FLA kernel: DeltaNet is
        # SEQUENTIAL across chunks, so chunks cannot be sorted by tier the way
        # the nested FFN rows are. It bills full and uses part, which makes it a
        # quality-per-FLOP experiment, not a speed one.
        #
        # `deltanet_forced_rank` pins every position to one rank and bypasses
        # the router -- that is the CONTROL that measures the rank->loss slope.
        # Run it BEFORE trusting any adaptive result.
        self.deltanet_dyn_rank = bool(deltanet_dyn_rank)
        self.deltanet_forced_rank = deltanet_forced_rank
        self.deltanet_rank_tiers = deltanet_rank_tiers
        self.deltanet_rank_prior = deltanet_rank_prior
        self.deltanet_rank_chunk = int(deltanet_rank_chunk)
        self.deltanet_rank_thresh_lr = float(deltanet_rank_thresh_lr)
        self.mat_ffn_route_mode = str(mat_ffn_route_mode)
        if self.mat_ffn_route_mode not in ("argmax", "threshold"):
            raise ValueError(
                f"mat_ffn_route_mode must be 'argmax' or 'threshold', got "
                f"{self.mat_ffn_route_mode!r}"
            )
        self.mat_ffn_route_thresh_lr = float(mat_ffn_route_thresh_lr)
        # Threshold controller + per-sequence cap headroom; see
        # RoutedMatFormerSwiGLUMLP.__init__. "post_clip" ratcheted theta to 0 in
        # every cell trained with it (routing became a position schedule).
        # Headroom defaults to 1.0 so every existing checkpoint evaluates
        # bit-identically; the controller only acts in training.
        self.mat_ffn_route_ctrl = str(mat_ffn_route_ctrl)
        if self.mat_ffn_route_ctrl not in ("quantile", "quantile_nested", "wish", "post_clip"):
            raise ValueError(
                f"mat_ffn_route_ctrl must be 'quantile', 'quantile_nested', 'wish' or "
                f"'post_clip', got {self.mat_ffn_route_ctrl!r}"
            )
        self.mat_ffn_cap_headroom = float(mat_ffn_cap_headroom)
        if self.mat_ffn_gate_mode not in ("ste", "mixture"):
            raise ValueError(
                "mat_ffn_gate_mode must be 'ste' or 'mixture', got " f"{self.mat_ffn_gate_mode!r}"
            )
        self.mat_ffn_track_occupancy = bool(mat_ffn_track_occupancy)
        if not 0.0 <= self.mat_ffn_explore_frac < 1.0:
            raise ValueError(
                "mat_ffn_explore_frac must be in [0, 1), got " f"{self.mat_ffn_explore_frac}"
            )
        self.mat_ffn_aux_mode = str(mat_ffn_aux_mode)
        if self.mat_ffn_aux_mode not in ("mean", "kl", "none"):
            raise ValueError(
                f"mat_ffn_aux_mode must be 'mean', 'kl' or 'none', got {self.mat_ffn_aux_mode!r}"
            )
        if self.mat_ffn_dispatch not in ("sorted", "gather", "grouped"):
            raise ValueError(
                f"mat_ffn_dispatch must be 'sorted', 'gather' or 'grouped', got "
                f"{self.mat_ffn_dispatch!r}"
            )
        if self.use_mat_ffn:
            # UNIFORM increments by default (quarters), not {1/4, 1/2, 1}.
            # torch._grouped_mm needs every tier's GEMM to share K and N and
            # differ only in row count, which holds iff the increments are
            # equal. Uniform quarters keep the menu mean at 0.625 x
            # intermediate_size instead of 0.583 -- slightly less aggressive,
            # and the price of being able to fuse the tiers into one kernel.
            q = max(1, self.intermediate_size // 4)
            w = self.mat_ffn_widths or [q, 2 * q, 3 * q, self.intermediate_size]
            if sorted(w) != list(w) or len(set(w)) != len(w):
                raise ValueError(
                    f"mat_ffn_widths must be strictly increasing (nested prefixes), got {w}"
                )
            if max(w) > self.intermediate_size:
                raise ValueError(
                    f"mat_ffn_widths max {max(w)} exceeds intermediate_size "
                    f"{self.intermediate_size}"
                )
            self.mat_ffn_widths = list(w)
            if self.mat_ffn_tier_fracs is not None:
                pi = [float(x) for x in self.mat_ffn_tier_fracs]
                if len(pi) != len(w):
                    raise ValueError(
                        f"mat_ffn_tier_fracs has {len(pi)} entries but there are "
                        f"{len(w)} tiers ({w})"
                    )
                if min(pi) < 0.0:
                    raise ValueError(f"mat_ffn_tier_fracs must be non-negative, got {pi}")
                tot = sum(pi)
                if tot <= 0:
                    raise ValueError("mat_ffn_tier_fracs must sum to a positive value")
                if abs(tot - 1.0) > 1e-6:
                    raise ValueError(f"mat_ffn_tier_fracs must sum to 1.0, got {tot}")
                self.mat_ffn_tier_fracs = pi
                # ONE definition of the expected width. Everything downstream --
                # the between-routed-layers SwiGLU in transformer_block, the
                # active-param accounting, the bench's forced routing -- reads
                # `mat_ffn_budget`, so deriving it here keeps them consistent
                # instead of each re-deriving it from the prior and drifting.
                self.mat_ffn_budget = float(sum(p * x for p, x in zip(pi, w)))
            if self.mat_ffn_fixed_width is not None:
                fw = int(self.mat_ffn_fixed_width)
                if not (0 < fw <= self.intermediate_size):
                    raise ValueError(
                        f"mat_ffn_fixed_width must be in (0, intermediate_size="
                        f"{self.intermediate_size}], got {fw}"
                    )
                self.mat_ffn_fixed_width = fw
            if not 0.0 <= self.mat_ffn_warmup_frac < 1.0:
                raise ValueError(
                    f"mat_ffn_warmup_frac must be in [0, 1), got {self.mat_ffn_warmup_frac}"
                )
            if self.mat_ffn_warmup_frac > 0 and self.mat_ffn_warmup_stages < 1:
                raise ValueError(
                    f"mat_ffn_warmup_stages must be >= 1, got {self.mat_ffn_warmup_stages}"
                )
            if self.mat_ffn_aux_mode == "kl" and self.mat_ffn_tier_fracs is None:
                raise ValueError(
                    "mat_ffn_aux_mode='kl' needs mat_ffn_tier_fracs (the target prior "
                    "it measures KL against)"
                )
        if self.swa_bank_read_interval < 1:
            raise ValueError(
                f"swa_bank_read_interval must be >= 1, got {self.swa_bank_read_interval}"
            )
        self.use_block_sparse_attn = bool(use_block_sparse_attn)
        self.bsa_mode = str(bsa_mode)
        self.bsa_block_size = int(bsa_block_size)
        self.bsa_top_k = int(bsa_top_k)
        self.bsa_min_seq = int(bsa_min_seq)
        self.bsa_layer_interval = int(bsa_layer_interval)
        self.use_delta_memory = bool(use_delta_memory)
        self.delta_mem_variant = str(delta_mem_variant)
        self.delta_mem_heads = int(delta_mem_heads)
        self.delta_mem_head_dim = int(delta_mem_head_dim)
        self.delta_mem_chunk = int(delta_mem_chunk)
        self.delta_mem_interval = int(delta_mem_interval)
        self.delta_mem_band_only = bool(delta_mem_band_only)
        self.delta_mem_checkpoint = bool(delta_mem_checkpoint)
        if self.use_block_sparse_attn:
            if self.bsa_mode not in ("moba", "hils"):
                raise ValueError(f"bsa_mode must be 'moba' or 'hils', got {self.bsa_mode!r}")
            if self.bsa_block_size < 1 or self.bsa_top_k < 1:
                raise ValueError("bsa_block_size and bsa_top_k must be >= 1.")
            # The far branch draws only from keys that end before the SWA window
            # begins, so with no window there is nothing for it to be disjoint
            # from and the LSE merge would double-count.
            if not self.sliding_window and not getattr(self, "swa_window", 0):
                raise ValueError("use_block_sparse_attn requires a sliding window.")
        # "titans" was accepted here but NEVER read by modules/delta_memory.py --
        # a dead flag that silently did nothing (found 2026-08-21). Replaced with
        # the variants that are actually implemented; adding one means adding a
        # branch in DeltaMemory.forward, not just a string here.
        if self.use_delta_memory and self.delta_mem_variant not in ("delta", "gla", "ema"):
            raise ValueError(
                "delta_mem_variant must be one of 'delta' (Widrow-Hoff overwrite), "
                f"'gla' (decay+accumulate), 'ema' (vector summary), got {self.delta_mem_variant!r}"
            )
        self.use_scribe_bank = bool(use_scribe_bank)
        self.scribe_slots_per_chunk = int(scribe_slots_per_chunk)
        self.scribe_chunk_size = int(scribe_chunk_size)
        self.scribe_write_layer = int(scribe_write_layer)
        self.scribe_write_topk = int(scribe_write_topk)
        self.trunk_write_topk = int(trunk_write_topk)
        self.use_delta_bank = bool(use_delta_bank)
        self.delta_chunk_size = int(delta_chunk_size)
        self.delta_write_layer = int(delta_write_layer)
        self.delta_write_topk = int(delta_write_topk)
        self.delta_state_clamp = float(delta_state_clamp)
        # Per-Layer Embeddings + MatFormer (Gemma-3n/4 style)
        self.use_ple = bool(use_ple)
        self.ple_dim = int(ple_dim)
        self.use_matformer = bool(use_matformer)
        self.matformer_granularities = matformer_granularities
        self.matformer_active_size = matformer_active_size
        self.mtp_variant = mtp_variant
        self.mtp_predict_offsets = (
            list(mtp_predict_offsets) if mtp_predict_offsets is not None else [2, 3, 4, 5]
        )
        # The KL recovery needs the NTP and MTP distributions to share one output head:
        # only then is the NTP gradient at h_{t+k} the residual W^T(q - y) the MTP CE
        # gradient at h^k_t carries. Gloeckle heads own their unembeddings.
        if self.mtp_soft and (mtp_variant != "deepseek" or not self.mtp_share_head):
            raise ValueError("mtp_soft needs mtp_variant='deepseek' and mtp_share_head=True")
        self.mod_capacity_inserted = float(mod_capacity_inserted)
        self.stage_block_spec = stage_block_spec
        self.deltanet_l2warp_inserted = bool(deltanet_l2warp_inserted)

        # Token-Superposition Training (TST)
        self.tst_enabled = bool(tst_enabled)
        self.tst_active = bool(tst_active)
        self.tst_bag_size = max(1, int(tst_bag_size))
        self.tst_recovery_ratio = float(tst_recovery_ratio)
        if tst_loss_weighting not in ("uniform", "power_law"):
            raise ValueError(
                f"tst_loss_weighting must be 'uniform' or 'power_law', got {tst_loss_weighting!r}"
            )
        self.tst_loss_weighting = tst_loss_weighting

        # Delphi recipe — see openathena.ai/blog/delphi. Recipe is inert
        # unless `delphi_recipe=True`; constants are stored for downstream
        # consumers (llamafactory_train.py patch, torchtitan loop).
        self.delphi_recipe = bool(delphi_recipe)
        self.delphi_optimizer = str(delphi_optimizer)
        self.delphi_schedule = str(delphi_schedule)
        self.delphi_warmup_frac = float(delphi_warmup_frac)
        self.delphi_decay_frac = float(delphi_decay_frac)
        self.delphi_eta0_proj = float(delphi_eta0_proj)
        self.delphi_eta0_adam = float(delphi_eta0_adam)
        self.delphi_batch_ref = int(delphi_batch_ref)
        self.delphi_token_ref = float(delphi_token_ref)
        self.delphi_horizon_exp = float(delphi_horizon_exp)
        self.delphi_epsilon0 = float(delphi_epsilon0)
        self.delphi_beta2_max = float(delphi_beta2_max)
        self.delphi_beta2_min = float(delphi_beta2_min)
        self.delphi_grad_clip = float(delphi_grad_clip)
        self.delphi_run_tokens = float(delphi_run_tokens)
        self.delphi_run_batch = int(delphi_run_batch)
        self.delphi_run_total_steps = int(delphi_run_total_steps)

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

        # Validate recurrent-depth configuration if enabled
        self._validate_recurrent_depth_config()

        # Validate workspace-trunk (GWT) configuration if enabled
        self._validate_workspace_trunk_config()

        # Validate pathways (MoP) configuration if enabled
        self._validate_pathways_config()

        # Message structure (P1/P2, modules/msg_struct.py); see the signature block.
        self.msg_rope = bool(msg_rope)
        self.msg_rope_theta = float(msg_rope_theta)
        self.msg_rope_frac = float(msg_rope_frac)
        self.msg_index_jitter = int(msg_index_jitter or 0)
        self.msg_pin_bias = bool(msg_pin_bias)
        self.msg_digest = bool(msg_digest)
        self.digest_window = int(digest_window)
        self.digest_slots_tool = int(digest_slots_tool)
        self.digest_slots_other = int(digest_slots_other)
        self.digest_max_messages = int(digest_max_messages)
        self.digest_source_layer = int(digest_source_layer)
        self.pause_decision_only = bool(pause_decision_only)
        self.pause_final_tokens = int(pause_final_tokens)
        self.msg_token_ids = dict(msg_token_ids) if msg_token_ids else None
        self._validate_msg_struct_config()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_msg_struct_config(self):
        """P1/P2 (modules/msg_struct.py) act on MLA layers and replace their attention
        call, so they refuse the features that path does not carry."""
        if self.pause_decision_only and not getattr(self, "use_pause_stream", False):
            raise ValueError("pause_decision_only needs use_pause_stream=True")
        if self.pause_final_tokens < 0:
            raise ValueError("pause_final_tokens must be >= 0")
        if not (self.msg_rope or self.msg_pin_bias or self.msg_digest):
            return
        if not getattr(self, "use_mla", False):
            raise ValueError("msg_rope / msg_pin_bias / msg_digest act on MLA layers only")
        clash = {
            "mla_diff_attn": getattr(self, "mla_diff_attn", False),
            "use_repo": getattr(self, "use_repo", False),
            "use_mod": getattr(self, "use_mod", False),
            "use_pause_stream": getattr(self, "use_pause_stream", False),
            "use_latent_feedback": getattr(self, "use_latent_feedback", False),
            "num_meta_tokens": getattr(self, "num_meta_tokens", 0),
            "swa_globals_mla": getattr(self, "attention_mode", None) == "swa_globals_mla",
        }
        bad = [k for k, v in clash.items() if v]
        if bad:
            raise ValueError(f"msg_rope / msg_pin_bias / msg_digest do not compose with {bad}")
        if not 0.0 < self.msg_rope_frac <= 1.0 or self.msg_rope_theta <= 1.0:
            raise ValueError("msg_rope_frac must be in (0, 1] and msg_rope_theta > 1")
        if self.msg_index_jitter < 0:
            raise ValueError("msg_index_jitter must be >= 0")
        if self.msg_digest and (
            self.digest_window < 1
            or min(self.digest_slots_tool, self.digest_slots_other) < 1
            or self.digest_max_messages < 1
        ):
            raise ValueError("digest_window, digest_slots_* and digest_max_messages must be >= 1")

    def _validate_recurrent_depth_config(self):
        """Validate router-free recurrent depth (Arm B) invariants."""
        if not self.use_recurrent_depth:
            if self.recurrent_stable_injection or self.recurrent_hyper_connections:
                raise ValueError(
                    "recurrent_stable_injection / recurrent_hyper_connections mix the "
                    "loop boundaries and need use_recurrent_depth=True."
                )
            return
        if self.recurrent_hyper_connections and self.recurrent_hc_streams < 1:
            raise ValueError(f"recurrent_hc_streams must be >= 1, got {self.recurrent_hc_streams}.")
        if self.recurrent_stable_injection and not (
            0.0 < self.recurrent_inject_dt_min <= self.recurrent_inject_dt_max
        ):
            raise ValueError(
                "recurrent injection step sizes must satisfy 0 < recurrent_inject_dt_min "
                f"<= recurrent_inject_dt_max, got [{self.recurrent_inject_dt_min}, "
                f"{self.recurrent_inject_dt_max}]."
            )
        if self.recurrent_sample_dist != "uniform":
            raise ValueError(
                "recurrent_sample_dist='uniform' is the only implemented sampling "
                f"distribution, got {self.recurrent_sample_dist!r}. New distributions must "
                "also extend umich_slm.compute_accounting.expected_recurrent_iters."
            )
        if self.recurrent_min_iters < 1 or self.recurrent_max_iters < self.recurrent_min_iters:
            raise ValueError(
                "recurrent iteration bounds must satisfy 1 <= recurrent_min_iters <= "
                f"recurrent_max_iters, got [{self.recurrent_min_iters}, {self.recurrent_max_iters}]."
            )
        if self.use_mod:
            raise ValueError(
                "use_recurrent_depth is router-free by design and does not compose with "
                "use_mod (routing + recursion would both rewrite the layer schedule)."
            )
        if self.use_deltanet or self.use_hymba or self.use_griffin:
            raise ValueError(
                "use_recurrent_depth requires softmax attention only: looping a block "
                "desynchronizes DeltaNet/Mamba/Griffin/Hymba recurrent state."
            )
        bs = self.recurrent_block_size
        if bs <= 0:
            bs = len(self.hybrid_attention_ratio) if self.hybrid_attention_ratio else 4
        # A core of ONE layer is enough: the schedule is prelude, core x iters,
        # coda. (Was 3 * bs until 2026-10-08, which also forced core >= block and
        # ruled out the FLOP-matched ar25m _rec shape, 6 + 5x3 + 6.)
        if self.num_hidden_layers < 2 * bs + 1:
            raise ValueError(
                f"use_recurrent_depth needs num_hidden_layers >= {2 * bs + 1} (prelude block "
                f"+ a looped core of at least one layer + coda block at "
                f"recurrent_block_size={bs}); got {self.num_hidden_layers}."
            )
        # A looped core cannot KV-cache (each iteration would append to the
        # same layers' caches), so the model refuses use_cache=True at forward
        # time. Flip the config default so plain forwards/evals just work;
        # explicit use_cache=True still raises with a clear message.
        self.use_cache = False

    def _validate_workspace_trunk_config(self):
        """Validate Global Workspace Trunk (GWT) invariants."""
        if not self.use_workspace_trunk:
            return
        if self.trunk_layers < 1:
            raise ValueError(f"trunk_layers must be >= 1, got {self.trunk_layers}.")
        if self.trunk_chunk_size < 1:
            raise ValueError(f"trunk_chunk_size must be >= 1, got {self.trunk_chunk_size}.")
        if self.trunk_iters_min < 1 or self.trunk_iters_max < self.trunk_iters_min:
            raise ValueError(
                "trunk iteration bounds must satisfy 1 <= trunk_iters_min <= "
                f"trunk_iters_max, got [{self.trunk_iters_min}, {self.trunk_iters_max}]."
            )
        if self.use_recurrent_depth:
            raise ValueError(
                "use_workspace_trunk does not compose with use_recurrent_depth "
                "(one dynamic-compute mechanism per sweep arm keeps attribution clean)."
            )
        if (self.use_mod or self.use_pathways) and self.trunk_iters_max != self.trunk_iters_min:
            # MoD composition is allowed ONLY for the bank-role trunk (the
            # swa_*_ws arms: iters pinned, deterministic — a writable MEMORY
            # BANK on the production MoD backbone, not a test-time-compute
            # mechanism). Sampled iteration counts on top of MoD routing would
            # re-tangle dynamic-compute attribution, so keep that forbidden.
            raise ValueError(
                "use_workspace_trunk composes with a MoD/MoP backbone only when the "
                "trunk is deterministic (trunk_iters_min == trunk_iters_max); sampled "
                "iteration counts would confound dynamic-compute attribution."
            )
        if self.trunk_cuda_graph_impl not in ("compile", "graph"):
            raise ValueError(
                f"trunk_cuda_graph_impl must be 'compile' or 'graph', got "
                f"{self.trunk_cuda_graph_impl!r}."
            )
        if self.trunk_variant not in ("sequential", "parallel", "rglru", "delta_rule"):
            raise ValueError(
                f"trunk_variant must be 'sequential', 'parallel', 'rglru' or "
                f"'delta_rule', got {self.trunk_variant!r}."
            )
        if self.trunk_variant != "sequential" and self.trunk_cuda_graph:
            raise ValueError(
                "trunk_cuda_graph targets the sequential chunk loop; the batched "
                "trunk variants have no per-chunk kernel chain to graph — unset one."
            )
        insert = self.trunk_insert_layer
        if insert >= 0 and not (1 <= insert <= self.num_hidden_layers - 1):
            raise ValueError(
                f"trunk_insert_layer must be in [1, num_hidden_layers-1] (at least one "
                f"token-stream layer on each side), got {insert} with "
                f"num_hidden_layers={self.num_hidden_layers}. Use -1 for the midpoint."
            )
        # Decode-time workspace refresh (bursty updates at chunk boundaries)
        # is deferred, exactly like Arm B's halting decode: force the cache
        # off so plain forwards/evals just work.
        self.use_cache = False

    def _validate_pathways_config(self):
        """Validate Mixture-of-Pathways (MoP) invariants."""
        if not self.use_pathways:
            return
        depths = self.pathway_depths if self.pathway_depths is not None else [0, 2, 4]
        if not depths or any(int(d) < 0 for d in depths) or max(depths) == 0:
            raise ValueError(
                f"pathway_depths must be a non-empty list of depths >= 0 with at least "
                f"one positive lane, got {depths}."
            )
        if self.use_mod or self.use_recurrent_depth:
            raise ValueError(
                "use_pathways does not compose with use_mod / use_recurrent_depth "
                "(one dynamic-compute mechanism per sweep arm keeps attribution clean; "
                "MoP also shares the mod_aux_loss channel)."
            )
        bottom = self.pathway_shared_bottom
        if bottom == 0:
            bottom = max(1, (2 * self.num_hidden_layers) // 3)
        if not (1 <= bottom <= self.num_hidden_layers - 1):
            raise ValueError(
                f"pathway_shared_bottom must leave >= 1 shared bottom and >= 1 shared "
                f"top layer: got {self.pathway_shared_bottom} with "
                f"num_hidden_layers={self.num_hidden_layers}."
            )
        if self.pathway_budget < 0 or self.pathway_budget > max(depths):
            raise ValueError(
                f"pathway_budget must be in [0, max(pathway_depths)] lane-blocks/token "
                f"(0 -> mean of depths), got {self.pathway_budget} with depths {depths}."
            )
        if self.pathway_variant not in ("disjoint", "nested", "recursive"):
            raise ValueError(
                f"pathway_variant must be 'disjoint', 'nested' or 'recursive', "
                f"got {self.pathway_variant!r}."
            )
        if self.pathway_stages < 1:
            raise ValueError(f"pathway_stages must be >= 1, got {self.pathway_stages}.")
        # Stage s sits before layer index bottom+s (one spine layer between
        # consecutive stages); at least one dense top layer must remain after
        # the last stage: bottom + stages - 1 <= L - 1.
        if self.num_hidden_layers < bottom + self.pathway_stages:
            raise ValueError(
                f"pathway_stages={self.pathway_stages} with pathway_shared_bottom="
                f"{bottom} needs num_hidden_layers >= {bottom + self.pathway_stages} "
                f"(spine layer per stage + >= 1 dense top layer); "
                f"got {self.num_hidden_layers}."
            )
        # Lane cross-attention over cached shared-stream K/V needs its own
        # cache plumbing (deferred inference integration) — force cache off.
        self.use_cache = False

    def _validate_rope_scaling(self):
        """Validate RoPE scaling configuration."""
        if getattr(self, "rope_scaling_scope", "all") not in ("all", "global"):
            raise ValueError(
                f"rope_scaling_scope must be 'all' or 'global', got {self.rope_scaling_scope!r}"
            )

        if not 0.0 <= float(self.partial_rotary_factor) <= 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in [0, 1], got {self.partial_rotary_factor!r}"
            )
        # The split lives in RotaryEmbedding.forward / forward_continuous, i.e. the GQA/MHA
        # path. PoPE and RePo build their own encodings, and MLA has its own rope slice.
        if float(self.partial_rotary_factor) < 1.0 and (
            getattr(self, "position_embedding_type", "rope") == "pope"
            or self.use_repo
            or self.use_mla
        ):
            raise ValueError("partial_rotary_factor < 1 supports plain RoPE on GQA/MHA layers only")
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

        # YaRN's correction range is defined over the TRAINED length. When it is
        # carried separately (HF convention), it cannot exceed the cache length.
        orig = self.rope_scaling.get("original_max_position_embeddings")
        if orig is not None:
            if int(orig) <= 0 or int(orig) > int(self.max_position_embeddings):
                raise ValueError(
                    "rope_scaling.original_max_position_embeddings must be in "
                    f"(0, max_position_embeddings={self.max_position_embeddings}], got {orig}"
                )

    def _validate_deltanet_config(self):
        """Validate DeltaNet configuration."""
        if not self.use_deltanet:
            return

        # Validate variant
        valid_variants = ["gated", "kda", "rwkv7"]
        if self.deltanet_variant not in valid_variants:
            raise ValueError(
                f"deltanet_variant must be one of {valid_variants}, " f"got {self.deltanet_variant}"
            )
        # Negative eigenvalues are a property of the delta correction (I - beta k k^T);
        # on the legacy linear-attention update beta*2 would only double the write.
        if self.deltanet_allow_neg_eigval and not self.deltanet_true_delta_rule:
            raise ValueError("deltanet_allow_neg_eigval requires deltanet_true_delta_rule=True")

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

        # The full-attention bank/sink knobs live on hybrid "attn"/"mla" entries.
        # On swa_globals_mla they would collide with the SWA bank and the SWA sink.
        if (getattr(self, "attn_register_count", 0) or getattr(self, "attn_logit_sinks", False)) and (
            self.attention_mode == "swa_globals_mla"
        ):
            raise ValueError(
                "attn_register_count / attn_logit_sinks are for DeltaNet/dense hybrids; "
                "attention_mode='swa_globals_mla' carries its own bank (swa_global_count) "
                "and sinks (use_logit_sinks)."
            )
        # pathway_bank_export reads the bank through the Pathways stage instead (the
        # deltanet_mla_think arm), so it needs no 'attn' layer.
        bank_via_export = getattr(self, "use_pathways", False) and getattr(
            self, "pathway_bank_export", False
        )
        if (
            getattr(self, "attn_register_count", 0)
            and "attn" not in (self.hybrid_attention_ratio or [])
            and not bank_via_export
        ):
            raise ValueError(
                "attn_register_count needs a hybrid_attention_ratio with 'attn' layers (or "
                "pathway_bank_export) to read the bank; with neither it would train nothing"
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
            # "attn" = a plain GQA full-attention layer in the global slot instead of
            # MLA. MLA's win is KV-cache compression at inference; in pretraining it
            # just makes the one unbounded layer type more expensive (192-dim QK vs
            # head_dim 64), so the GQA variant is a real arm, not a degenerate config.
            allowed = {"swa", "mla", "attn"}
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
            if self.swa_use_registers and self.swa_global_count <= 0:
                raise ValueError(
                    "swa_use_registers=True needs a bank size: set swa_global_count > 0 "
                    "(the bank reuses that knob; the tokens are read as registers, "
                    "never prepended)."
                )
            if self.swa_sink_count > 0 or self.use_scribe_bank:
                # swa_scribe contract: out-of-sequence banks joined to the FA2
                # window branch by an exact LSE merge. Requirements:
                if float(getattr(self, "attention_dropout", 0.0) or 0.0) != 0.0:
                    raise ValueError(
                        "swa_sink_count/use_scribe_bank require attention_dropout=0.0 "
                        "(FA2's returned LSE is pre-dropout; the merge would be wrong)."
                    )
                if self.swa_global_count > 0 and not (
                    self.swa_use_registers
                    and (self.swa_register_count in (None, self.swa_global_count))
                ):
                    raise ValueError(
                        "swa_sink_count/use_scribe_bank cannot combine with a prepended "
                        "prefix (the sink bank replaces it); use swa_use_registers with "
                        "the full bank, or swa_global_count=0."
                    )
                if self.use_scribe_bank and (
                    self.scribe_slots_per_chunk < 1 or self.scribe_chunk_size < 1
                ):
                    raise ValueError("scribe_slots_per_chunk and scribe_chunk_size must be >= 1.")
                if self.use_scribe_bank:
                    # Only SWA layers read scribes. If every layer above the
                    # write point is an MLA (or otherwise non-SWA) layer, the
                    # bank is written and never read: dead parameters, zero
                    # gradient, and a silently inert arm — the same failure
                    # class as the MoD dead-trunk bug (2026-07-19).
                    write = (
                        self.scribe_write_layer
                        if self.scribe_write_layer >= 0
                        else self.num_hidden_layers // 2
                    )
                    ratio = self.hybrid_attention_ratio or []
                    readers = [
                        i
                        for i in range(write + 1, self.num_hidden_layers)
                        if not ratio or ratio[i % len(ratio)] == "swa"
                    ]
                    if not readers:
                        raise ValueError(
                            f"scribe_write_layer={write} leaves no SWA layer above it "
                            f"(layers {write + 1}..{self.num_hidden_layers - 1} are not 'swa'); "
                            "the scribe bank would be written but never read."
                        )
                self.use_cache = False  # decode-time scribe/sink refresh not implemented
            if self.swa_register_refine and not self.swa_use_registers:
                raise ValueError("swa_register_refine requires swa_use_registers=True.")
            if self.swa_register_count is not None:
                if not self.swa_use_registers:
                    raise ValueError(
                        "swa_register_count requires swa_use_registers=True (it splits the "
                        "swa_global_count bank between a prepended prefix and a register read)."
                    )
                if not (0 < self.swa_register_count <= self.swa_global_count):
                    raise ValueError(
                        f"swa_register_count must be in (0, swa_global_count]; got "
                        f"{self.swa_register_count} with swa_global_count={self.swa_global_count}."
                    )
            if self.use_delta_bank:
                if not self.swa_use_registers:
                    raise ValueError(
                        "use_delta_bank requires swa_use_registers=True (the chunk-writes "
                        "accumulate into the register bank; there is nothing to write into "
                        "on a pure-prefix arch)."
                    )
                if self.delta_chunk_size < 1:
                    raise ValueError("delta_chunk_size must be >= 1.")
                if self.delta_state_clamp <= 0:
                    raise ValueError("delta_state_clamp must be > 0.")

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
            raise ValueError(f"griffin_num_heads must be > 0, got {self.griffin_num_heads}.")

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

    @property
    def layer_types(self) -> Optional[List[str]]:
        """Per-layer cache kind, read by `transformers.cache_utils.DynamicCache`.

        HF builds a generation cache from this attribute, and when it is ABSENT
        but `sliding_window` is set it falls back to marking EVERY layer
        "sliding_attention" -- so it evicts on the periodic FULL-attention layers
        too, which are the only layers that exist to see the whole history. The
        symptom is silent: the cache reports the true token count while holding
        `sliding_window - 1` keys, and decoding quietly loses the long range.

        The split here is the one `attention.py` itself applies, via the same
        `is_global_attention_layer` predicate, so the two cannot drift.

        Returned as a property, not an `__init__` attribute, so it stays out of
        the serialized config -- it is derived, and a stale copy in a saved
        config.json would be another way for the two to drift. A config.json
        that does carry one (written by other tooling) is still accepted, via
        the setter below, because `PretrainedConfig.__init__` assigns every key
        it finds and a read-only property would turn that into a load failure.
        """
        override = self.__dict__.get("_layer_types_override")
        if override is not None:
            return list(override)

        if getattr(self, "sliding_window", None) is None:
            return None  # No windowing: HF's own default (all full) is right.

        n_layers = int(self.num_hidden_layers)

        # The swa_globals family bands its local layers by `swa_window`, NOT by
        # `sliding_window` (which it uses only for its global MLA layers) -- see
        # the CLAUDE.md note on the two being easy to confuse. Reporting
        # "sliding_attention" would make HF evict to the wrong length, so those
        # models keep a complete cache and let the attention modules do the
        # windowing themselves. They are also `use_mla=True`, hence carried by
        # MLADynamicCache rather than by HF's, so this is belt and braces.
        if getattr(self, "attention_mode", "legacy") != "legacy":
            return ["full_attention"] * n_layers

        from .modules.sliding_window import is_global_attention_layer

        every_n = getattr(self, "global_attention_every_n_layers", 4)
        ratio = getattr(self, "hybrid_attention_ratio", None)
        return [
            (
                "full_attention"
                if is_global_attention_layer(i, every_n, ratio)
                else "sliding_attention"
            )
            for i in range(n_layers)
        ]

    @layer_types.setter
    def layer_types(self, value) -> None:
        """Accept an explicit per-layer schedule; None restores the derived one."""
        self.__dict__["_layer_types_override"] = None if value is None else list(value)

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
