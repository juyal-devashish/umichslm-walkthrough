"""UMichSLM model implementation.

This module contains the main model classes for UMichSLM, a modern
transformer architecture with SwiGLU, RoPE/PoPE, and RMSNorm.

The implementation is compatible with HuggingFace's transformers library,
supporting AutoModel loading and all standard PreTrainedModel features.

Supports Mixture of Experts (MoE) with load balancing auxiliary losses.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.utils import logging

from .configuration_umichslm import UMichSLMConfig
from .modules.rmsnorm import get_rmsnorm_class
from .modules.transformer_block import (
    UMichSLMDecoderLayer,
    get_decoder_layer_class,
    _should_use_deltanet,
    _should_use_griffin,
)
from .modules.griffin import GriffinDynamicCache, GriffinCache
from .modules.multi_token_prediction import (
    GloeckleMultiTokenPrediction,
    MTPOutput,
    MultiTokenPrediction,
    build_mtp_module,
)
from .modules.mixture_of_depths import MoDOutput, is_mod_layer, compute_mod_aux_loss
from .modules.kv_cache import DeltaNetDynamicCache, MLADynamicCache
from .modules.hymba_cache import HymbaDynamicCache
from .modules.engram import EngramModule
from .modules.moda import MoDADepthKVCache


@dataclass
class MoEModelOutputWithPast(ModelOutput):
    """Output type for MoE models.

    Extends BaseModelOutputWithPast with MoE-specific outputs.

    Attributes:
        last_hidden_state: Last layer hidden states.
        past_key_values: Cached key/value states.
        hidden_states: All hidden states (if output_hidden_states=True).
        attentions: All attention weights (if output_attentions=True).
        router_logits: Router logits from all MoE layers.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    """Output type for MoE causal language models.

    Extends CausalLMOutputWithPast with MoE-specific outputs.

    Attributes:
        loss: Language modeling loss (includes aux loss if compute_moe_loss=True).
        aux_loss: MoE auxiliary loss (load balancing + z-loss).
        logits: Prediction scores.
        past_key_values: Cached key/value states.
        hidden_states: All hidden states (if output_hidden_states=True).
        attentions: All attention weights (if output_attentions=True).
        router_logits: Router logits from all MoE layers.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithMTP(ModelOutput):
    """Output type for causal language models with Multi-Token Prediction.

    Extends CausalLMOutputWithPast with MTP-specific outputs.

    Attributes:
        loss: Total loss (LM loss + MTP loss + optional aux loss).
        lm_loss: Standard language modeling loss.
        mtp_loss: Multi-Token Prediction auxiliary loss.
        aux_loss: MoE auxiliary loss (if using MoE).
        logits: Prediction scores for next token.
        past_key_values: Cached key/value states.
        hidden_states: All hidden states (if output_hidden_states=True).
        attentions: All attention weights (if output_attentions=True).
        router_logits: Router logits from MoE layers (if using MoE).
        mtp_logits: Logits from each MTP prediction depth.
    """

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    mtp_logits: Optional[List[torch.FloatTensor]] = None


@dataclass
class MoDModelOutputWithPast(ModelOutput):
    """Output type for Mixture of Depths models.

    Extends BaseModelOutputWithPast with MoD-specific outputs.

    Attributes:
        last_hidden_state: Last layer hidden states.
        past_key_values: Cached key/value states.
        hidden_states: All hidden states (if output_hidden_states=True).
        attentions: All attention weights (if output_attentions=True).
        mod_aux_loss: MoD auxiliary loss for router training.
        mod_infos: MoDOutput from each MoD layer with routing info.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mod_aux_loss: Optional[torch.FloatTensor] = None
    mod_infos: Optional[Tuple[MoDOutput, ...]] = None


logger = logging.get_logger(__name__)


class UMichSLMPreTrainedModel(PreTrainedModel):
    """Base class for all UMichSLM models.

    This class handles weight initialization and provides common interfaces
    required by the HuggingFace ecosystem.

    Attributes:
        config_class: The configuration class for this model.
        base_model_prefix: Prefix for the base model.
        supports_gradient_checkpointing: Whether gradient checkpointing is supported.
        _no_split_modules: Modules that should not be split across devices.
        _skip_keys_device_placement: Keys to skip during device placement.
        _supports_flash_attn_2: Whether Flash Attention 2 is supported.
        _supports_sdpa: Whether PyTorch SDPA is supported.
        _supports_cache_class: Whether the model supports Cache classes.
    """

    config_class = UMichSLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "UMichSLMDecoderLayer",
        "GatedDeltaNet",
        "KimiDeltaAttention",
        "HymbaHybridHead",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        """Initialize model weights.

        Uses standard normal initialization with configurable standard deviation.

        Args:
            module: The module to initialize.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class UMichSLMModel(UMichSLMPreTrainedModel):
    """The bare UMichSLM model outputting raw hidden states.

    This model outputs the raw hidden states without any head on top.
    It's useful as a base for other models or for extracting embeddings.

    Args:
        config: The model configuration.

    Example:
        >>> from umich_slm import UMichSLMConfig, UMichSLMModel
        >>> config = UMichSLMConfig(hidden_size=256, num_hidden_layers=6)
        >>> model = UMichSLMModel(config)
        >>> inputs = torch.randint(0, config.vocab_size, (1, 10))
        >>> outputs = model(inputs)
        >>> hidden_states = outputs.last_hidden_state
    """

    def __init__(self, config: UMichSLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        # Track feature flags
        self.use_moe = getattr(config, "use_moe", False)
        self.use_mod = getattr(config, "use_mod", False)
        self.mod_every_n_layers = getattr(config, "mod_every_n_layers", 2)
        self.use_deltanet = getattr(config, "use_deltanet", False)
        self.use_griffin = getattr(config, "use_griffin", False)
        self.use_hymba = getattr(config, "use_hymba", False)
        self.use_moda = getattr(config, "use_moda", False)

        # MoD always-used layers and MoDA active layers
        self.always_used_layers = (
            config.get_always_used_layers() if hasattr(config, "get_always_used_layers") else set()
        )
        self.moda_active_layers = (
            set(config.moda_active_layers)
            if getattr(config, "moda_active_layers", None) is not None
            else None
        )

        # Meta tokens (available for all architectures)
        # Determine effective number of meta tokens:
        # 1. num_meta_tokens is the universal config (works with any architecture)
        # 2. hymba_num_meta_tokens is Hymba-specific and takes priority when use_hymba=True
        num_meta = getattr(config, "num_meta_tokens", 0)
        if self.use_hymba and num_meta == 0:
            num_meta = getattr(config, "hymba_num_meta_tokens", 0)
        self.num_meta_tokens = num_meta

        if self.num_meta_tokens > 0:
            from .modules.hymba import MetaTokens

            self.meta_tokens = MetaTokens(self.num_meta_tokens, config.hidden_size)
        else:
            self.meta_tokens = None

        # Persistent learnable global tokens (used by attention_mode="swa_globals_mla").
        # Globals are prepended to every batch's input embeddings and remain present
        # across all stages — they are NOT duplicated when blocks duplicate at stack
        # transitions. Globals attend to all positions; content tokens attend to all
        # globals + a sliding window of previous content. The MoD router on inserted
        # blocks force-routes the prefix and the MTP loss masks the prefix out.
        self.attention_mode = getattr(config, "attention_mode", "legacy")
        swa_global_count = int(getattr(config, "swa_global_count", 0))
        if self.attention_mode == "swa_globals_mla" and swa_global_count > 0:
            from .modules.global_tokens import GlobalTokens

            self.global_tokens = GlobalTokens(swa_global_count, config.hidden_size)
            self.num_global_tokens = swa_global_count
        else:
            self.global_tokens = None
            self.num_global_tokens = 0

        # Engram (Hash-Based Conditional Memory) modules
        self.use_engram = getattr(config, "use_engram", False)
        if self.use_engram:
            engram_layer_ids = config.engram_layer_ids or []
            self.engram_modules = nn.ModuleDict(
                {str(lid): EngramModule(config, lid) for lid in engram_layer_ids}
            )
        else:
            self.engram_modules = nn.ModuleDict()

        # Transformer layers
        # MoD uses alternating pattern: some layers are MoD, others are regular
        # Progressive MoE: first half uses plain MLP, then experts increase (2→4→8→...)
        self.moe_progressive = getattr(config, "moe_progressive", False)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            # Determine if this layer should use MoD
            # Layers in always_used_layers are excluded from MoD routing
            layer_uses_mod = (
                self.use_mod
                and is_mod_layer(layer_idx, self.mod_every_n_layers)
                and layer_idx not in self.always_used_layers
            )

            # For progressive MoE, get per-layer expert count
            if self.use_moe and self.moe_progressive:
                num_experts = config.get_num_experts_for_layer(layer_idx)
                # If only 1 expert, use regular decoder (no routing overhead)
                layer_uses_moe = num_experts > 1
            else:
                num_experts = config.num_experts if self.use_moe else 1
                layer_uses_moe = self.use_moe

            decoder_layer_class = get_decoder_layer_class(
                parallel_mlp=getattr(config, "parallel_mlp", False),
                use_moe=layer_uses_moe,
                use_mod=layer_uses_mod,
            )

            # Pass num_experts to MoE layers for progressive MoE
            if layer_uses_moe:
                self.layers.append(decoder_layer_class(config, layer_idx, num_experts=num_experts))
            else:
                self.layers.append(decoder_layer_class(config, layer_idx))

        # Block-wise weight sharing (MobileLLM-LS)
        # Every block of `weight_sharing_block_size` consecutive layers shares weights.
        # Layers 1..block_size-1 within each block are replaced with a reference to
        # layer 0 of that block, so they share identical parameters.
        # Example with block_size=2: layers[1]=layers[0], layers[3]=layers[2], ...
        if getattr(config, "use_block_weight_sharing", False):
            block_size = getattr(config, "weight_sharing_block_size", 2)
            for i in range(config.num_hidden_layers):
                block_start = (i // block_size) * block_size
                if i != block_start:
                    self.layers[i] = self.layers[block_start]

        # Final layer norm
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # DRoPE (Dropped Rotary Position Embeddings) state
        # When True, position embeddings are skipped in attention layers
        self._drope_enabled = False

        # Initialize weights and apply final processing
        self.post_init()

    def set_drope_enabled(self, enabled: bool):
        """Enable or disable DRoPE (dropped position embeddings).

        When DRoPE is enabled, position embeddings are skipped in all attention
        layers. This is used during the "recalibration" phase of DRoPE training
        where the model learns to operate without explicit position information.

        Reference: https://arxiv.org/abs/2512.12167

        Args:
            enabled: If True, drop position embeddings. If False, use normal RoPE.
        """
        self._drope_enabled = enabled

        # Propagate to all attention layers
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.drope_enabled = enabled
            # Handle parallel MLP layers which may have different structure
            elif hasattr(layer, "attention"):
                layer.attention.drope_enabled = enabled

    def get_drope_enabled(self) -> bool:
        """Check if DRoPE is currently enabled.

        Returns:
            True if position embeddings are currently dropped.
        """
        return self._drope_enabled

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings module."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings module."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast, MoEModelOutputWithPast]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
            position_ids: Position IDs of shape (batch, seq_len).
            past_key_values: Cached key/value states for incremental decoding.
            inputs_embeds: Pre-computed input embeddings.
            use_cache: Whether to return updated cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            return_dict: Whether to return a ModelOutput instead of tuple.

        Returns:
            BaseModelOutputWithPast (or MoEModelOutputWithPast if MoE) or tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Note: Memory-intensive outputs (use_cache, output_hidden_states, output_attentions)
        # are controlled by the caller. During actual training with the Trainer, these
        # are typically disabled by default. The CausalLM forward applies additional
        # optimizations when labels are provided.

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # Handle past key values
        past_length = 0
        use_mla = getattr(self.config, "use_mla", False)

        # Convert legacy cache to HymbaDynamicCache for Hymba models
        if (
            self.use_hymba
            and past_key_values is not None
            and not isinstance(past_key_values, HymbaDynamicCache)
        ):
            if isinstance(past_key_values, (tuple, list)):
                past_key_values = HymbaDynamicCache.from_legacy_cache(
                    past_key_values, num_layers=self.config.num_hidden_layers
                )

        # Convert legacy tuple cache to MLADynamicCache for MLA models
        if (
            use_mla
            and past_key_values is not None
            and not isinstance(past_key_values, MLADynamicCache)
        ):
            if isinstance(past_key_values, tuple):
                past_key_values = MLADynamicCache.from_legacy_cache(past_key_values)

        if past_key_values is not None:
            if self.use_hymba and isinstance(past_key_values, HymbaDynamicCache):
                past_length = past_key_values.get_seq_length()
            elif use_mla:
                # MLA cache is now MLADynamicCache with get_seq_length() method
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                else:
                    # Legacy tuple format for MLA (should not happen after conversion above)
                    first_layer = past_key_values[0]
                    if isinstance(first_layer, (list, tuple)) and len(first_layer) >= 1:
                        c_kv = first_layer[0]
                        if hasattr(c_kv, "shape"):
                            past_length = c_kv.shape[1]  # c_kv is [batch, seq, rank]
            elif self.use_deltanet:
                # DeltaNet cache format: recurrent state per layer
                # DeltaNet stores a single tensor [B, H, D_k, D_v] as recurrent state
                # For pure DeltaNet models, past_length doesn't affect DeltaNet (no position encoding)
                # For hybrid models, find an attention layer's cache to determine past_length
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                else:
                    # Legacy tuple format - find first attention layer's cache
                    for layer_idx, layer_cache in enumerate(past_key_values):
                        if layer_cache is None:
                            continue
                        # Check if this is an attention layer (KV cache = tuple of 2 tensors)
                        # vs DeltaNet layer (recurrent state = single tensor)
                        if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                            # This is likely a KV cache from attention layer
                            key_tensor = layer_cache[0]
                            if hasattr(key_tensor, "shape") and key_tensor.dim() == 4:
                                past_length = key_tensor.shape[2]  # [batch, heads, seq, dim]
                                break
                        # DeltaNet recurrent state is a single tensor, skip it
            else:
                # Standard attention cache format
                # Modern Cache API (transformers 4.36+)
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                else:
                    # Legacy tuple format: ((k1, v1), (k2, v2), ...)
                    first_layer = past_key_values[0]
                    if isinstance(first_layer, (list, tuple)) and len(first_layer) >= 1:
                        key_tensor = first_layer[0]
                        if hasattr(key_tensor, "shape"):
                            past_length = key_tensor.shape[2]

        # Create position IDs if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length,
                seq_length + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get input embeddings
        if inputs_embeds is None:
            # Validate token IDs are within vocabulary range
            if input_ids is not None:
                max_token_id = input_ids.max().item()
                if max_token_id >= self.vocab_size:
                    raise ValueError(
                        f"Token ID {max_token_id} is out of vocabulary range [0, {self.vocab_size}). "
                        f"This indicates a vocabulary size mismatch between the tokenizer and model. "
                        f"Model vocab_size: {self.vocab_size}, but tokenizer produced token ID {max_token_id}. "
                        f"Please ensure the model's vocab_size matches the tokenizer's vocab_size."
                    )
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepend meta tokens (available for all architectures)
        # Meta tokens are learnable tokens that serve as attention sinks
        # They are prepended before the layer loop and stripped from output
        num_meta = 0
        if self.meta_tokens is not None and past_length == 0:
            # Only prepend on initial forward (not during cached generation)
            meta = self.meta_tokens(batch_size, inputs_embeds.dtype, inputs_embeds.device)
            num_meta = meta.shape[1]
            inputs_embeds = torch.cat([meta, inputs_embeds], dim=1)

            # Extend attention mask to cover meta tokens (always attend to meta)
            if attention_mask is not None:
                meta_mask = torch.ones(
                    batch_size, num_meta, device=attention_mask.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

            # Extend position_ids (meta tokens get positions 0..num_meta-1,
            # real tokens continue after)
            if position_ids is not None:
                meta_pos = (
                    torch.arange(num_meta, device=position_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                position_ids = torch.cat([meta_pos, position_ids + num_meta], dim=1)

            seq_length += num_meta

        # Prepend persistent global tokens (SWA + globals path).
        # Globals occupy positions 0..N-1; content shifts to N..N+T-1. Globals are
        # only prepended on the initial forward (past_length == 0) so cached
        # autoregressive generation operates on raw content positions.
        num_globals = 0
        if (
            self.global_tokens is not None
            and past_length == 0
            and self.num_global_tokens > 0
        ):
            globs = self.global_tokens(batch_size, inputs_embeds.dtype, inputs_embeds.device)
            num_globals = globs.shape[1]
            inputs_embeds = torch.cat([globs, inputs_embeds], dim=1)

            if attention_mask is not None:
                gmask = torch.ones(
                    batch_size, num_globals,
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([gmask, attention_mask], dim=1)

            if position_ids is not None:
                gpos = (
                    torch.arange(num_globals, device=position_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                position_ids = torch.cat([gpos, position_ids + num_globals], dim=1)

            seq_length += num_globals

        # Stash the protected-prefix count on every layer that needs it, so the
        # SWA layer can build its globals+window mask and the MoD router can
        # force-route the prefix. Setting attributes on layers is the simplest
        # zero-allocation channel that doesn't disturb HF's forward signatures.
        if num_globals > 0:
            for layer in self.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_protected_prefix"):
                    layer.self_attn.num_protected_prefix = num_globals
                if hasattr(layer, "num_protected_prefix"):
                    # MoD decoder layer
                    layer.num_protected_prefix = num_globals
        else:
            # Make sure leftover values from a previous forward don't leak in
            # (this matters for the SWA layer when later batches use no globals).
            for layer in self.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_protected_prefix"):
                    layer.self_attn.num_protected_prefix = 0
                if hasattr(layer, "num_protected_prefix"):
                    layer.num_protected_prefix = 0

        # Create attention mask
        attn_implementation = getattr(self.config, "_attn_implementation", "eager")

        # PoPE (Polar Position Embeddings) doubles the Q/K head dimensions, which is
        # incompatible with Flash Attention and SDPA. The attention layers will fall
        # back to eager implementation, so we must prepare the mask accordingly.
        if getattr(self.config, "position_embedding_type", "rope") == "pope":
            if attn_implementation in ("flash_attention_2", "sdpa"):
                attn_implementation = "eager"

        if attn_implementation == "flash_attention_2":
            # Flash Attention 2 handles masking internally
            attention_mask_4d = attention_mask if attention_mask is not None else None
        elif attn_implementation == "sdpa":
            attention_mask_4d = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_length,
            )
        else:
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_length,
            )

        hidden_states = inputs_embeds

        # Engram: store input_ids and manage token buffer for generation
        if self.use_engram and len(self.engram_modules) > 0:
            if input_ids is not None:
                if past_key_values is None:
                    # Prefill: reset buffers and store full input_ids
                    for mod in self.engram_modules.values():
                        mod.reset_token_buffer()
                        mod.update_token_buffer(input_ids)
                    engram_input_ids = input_ids
                else:
                    # Generation: append new tokens to buffer
                    for mod in self.engram_modules.values():
                        mod.update_token_buffer(input_ids)
                    # Use the full buffer from the first module
                    first_mod = next(iter(self.engram_modules.values()))
                    engram_input_ids = first_mod.get_token_ids(input_ids, past_key_values)
            else:
                engram_input_ids = None
        else:
            engram_input_ids = None

        # Gradient checkpointing
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "Gradient checkpointing is not compatible with caching. " "Disabling caching."
            )
            use_cache = False

        # For DeltaNet: normalize incoming cache objects to a DeltaNet-aware container.
        if self.use_deltanet and use_cache:
            deltanet_layers = [
                _should_use_deltanet(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
            if past_key_values is None:
                past_key_values = DeltaNetDynamicCache(
                    num_layers=self.config.num_hidden_layers,
                    deltanet_layers=deltanet_layers,
                )
            elif isinstance(past_key_values, tuple):
                past_key_values = DeltaNetDynamicCache.from_legacy_cache(
                    past_key_values,
                    num_layers=self.config.num_hidden_layers,
                    deltanet_layers=deltanet_layers,
                )
            elif isinstance(past_key_values, Cache) and not isinstance(
                past_key_values, DeltaNetDynamicCache
            ):
                past_key_values = DeltaNetDynamicCache(
                    num_layers=self.config.num_hidden_layers,
                    deltanet_layers=deltanet_layers,
                )

        # For Griffin: normalize incoming cache to a GriffinDynamicCache.
        if self.use_griffin and use_cache:
            griffin_layers = [
                _should_use_griffin(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
            if past_key_values is None:
                past_key_values = GriffinDynamicCache(
                    num_layers=self.config.num_hidden_layers,
                    griffin_layers=griffin_layers,
                )
            elif isinstance(past_key_values, tuple):
                past_key_values = GriffinDynamicCache.from_legacy_cache(
                    past_key_values,
                    num_layers=self.config.num_hidden_layers,
                    griffin_layers=griffin_layers,
                )

        # For Hymba: create a shared HymbaDynamicCache if none exists
        if self.use_hymba and use_cache and past_key_values is None:
            past_key_values = HymbaDynamicCache(
                num_layers=self.config.num_hidden_layers,
                share_kv=getattr(self.config, "hymba_share_kv_layers", False),
                kv_share_group_size=getattr(self.config, "hymba_kv_share_group_size", 2),
            )

        if use_cache and isinstance(past_key_values, DeltaNetDynamicCache):
            past_key_values.advance_seq_length(seq_length)

        if use_cache and isinstance(past_key_values, GriffinDynamicCache):
            past_key_values.advance_seq_length(seq_length)

        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        _is_recurrent_cache = isinstance(past_key_values, (DeltaNetDynamicCache, GriffinDynamicCache))
        next_decoder_cache = None if _is_recurrent_cache else (() if use_cache else None)

        # MoE router logits (only for MoE models)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.use_moe
        )
        all_router_logits = () if output_router_logits and self.use_moe else None

        # MoD info (only for MoD models during training)
        output_mod_info = self.use_mod and self.training
        all_mod_infos = () if output_mod_info else None

        # MoDA: create depth KV cache for this forward pass
        depth_kv_cache = None
        if self.use_moda:
            depth_kv_cache = MoDADepthKVCache(
                max_entries=getattr(self.config, "moda_max_depth_cache", -1),
                depth_kv_every_n=getattr(self.config, "moda_depth_kv_every_n", 1),
                active_layers=self.moda_active_layers,
            )

        # Forward through layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get past key value for this layer
            if isinstance(past_key_values, (DeltaNetDynamicCache, GriffinDynamicCache)):
                past_key_value = past_key_values[idx]
            else:
                past_key_value = (
                    past_key_values[idx]
                    if past_key_values is not None
                    and not isinstance(past_key_values, Cache)
                    and not isinstance(past_key_values, HymbaDynamicCache)
                    else past_key_values
                )

            # Check if this is a MoD layer
            layer_is_mod = self.use_mod and is_mod_layer(idx, self.mod_every_n_layers)

            # Determine per-layer MoE status (supports progressive MoE where some
            # layers are plain MLP despite global use_moe=True)
            layer_is_moe = hasattr(decoder_layer, "moe")

            # Determine depth_kv_cache for this layer (only if MoDA is active
            # and this layer is in the active set)
            layer_depth_kv = None
            if depth_kv_cache is not None:
                if self.moda_active_layers is None or idx in self.moda_active_layers:
                    layer_depth_kv = depth_kv_cache

            if self.gradient_checkpointing and self.training:
                # Bind MoE/MoD/MoDA kwargs so checkpoint() passes them through
                ckpt_kwargs = {}
                if layer_is_moe:
                    ckpt_kwargs["output_router_logits"] = output_router_logits
                if layer_is_mod:
                    ckpt_kwargs["output_mod_info"] = output_mod_info
                if layer_depth_kv is not None:
                    ckpt_kwargs["depth_kv_cache"] = layer_depth_kv

                ckpt_fn = (
                    partial(decoder_layer.__call__, **ckpt_kwargs)
                    if ckpt_kwargs
                    else decoder_layer.__call__
                )

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    ckpt_fn,
                    hidden_states,
                    attention_mask_4d,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                # MoE layers accept output_router_logits parameter
                if layer_is_moe:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask_4d,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        output_router_logits=output_router_logits,
                        depth_kv_cache=layer_depth_kv,
                    )
                elif layer_is_mod:
                    # MoD layers accept output_mod_info parameter
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask_4d,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        output_mod_info=output_mod_info,
                        depth_kv_cache=layer_depth_kv,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask_4d,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        depth_kv_cache=layer_depth_kv,
                    )

            hidden_states = layer_outputs[0]

            # Apply Engram after designated layers
            if self.use_engram and engram_input_ids is not None and str(idx) in self.engram_modules:
                hidden_states = hidden_states + self.engram_modules[str(idx)](
                    hidden_states, engram_input_ids
                )

            if use_cache and isinstance(past_key_values, DeltaNetDynamicCache):
                layer_cache = layer_outputs[2 if output_attentions else 1]
                if _should_use_deltanet(self.config, idx):
                    past_key_values.set_deltanet_cache(idx, layer_cache)
            elif use_cache and isinstance(past_key_values, GriffinDynamicCache):
                layer_cache = layer_outputs[2 if output_attentions else 1]
                past_key_values[idx] = layer_cache
            elif use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Collect router logits for MoE layers
            if all_router_logits is not None and layer_is_moe:
                # MoE output is the last element when output_router_logits=True
                router_idx = 1 + (1 if output_attentions else 0) + (1 if use_cache else 0)
                if len(layer_outputs) > router_idx:
                    all_router_logits += (layer_outputs[router_idx],)

            # Collect MoD info
            if all_mod_infos is not None and layer_is_mod:
                # MoD output is the last element when output_mod_info=True
                mod_idx = 1 + (1 if output_attentions else 0) + (1 if use_cache else 0)
                if len(layer_outputs) > mod_idx:
                    all_mod_infos += (layer_outputs[mod_idx],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Strip meta tokens from output
        if num_meta > 0:
            hidden_states = hidden_states[:, num_meta:, :]

        # Strip persistent globals from output (mirror of the prepend above).
        if num_globals > 0:
            hidden_states = hidden_states[:, num_globals:, :]

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if use_cache and isinstance(past_key_values, DeltaNetDynamicCache):
            next_cache = past_key_values

        if use_cache and isinstance(past_key_values, GriffinDynamicCache):
            next_cache = past_key_values

        # For Hymba: the shared HymbaDynamicCache was updated in-place by all layers
        if use_cache and self.use_hymba:
            if isinstance(past_key_values, HymbaDynamicCache):
                next_cache = past_key_values
            elif isinstance(next_cache, tuple) and len(next_cache) > 0:
                for nc in next_cache:
                    if isinstance(nc, HymbaDynamicCache):
                        next_cache = nc
                        break

        # Convert MLA cache tuple to MLADynamicCache for proper HuggingFace integration
        if use_cache and use_mla and next_cache is not None and isinstance(next_cache, tuple):
            next_cache = MLADynamicCache.from_legacy_cache(next_cache)

        # Compute MoD auxiliary loss
        mod_aux_loss = None
        if all_mod_infos is not None and len(all_mod_infos) > 0:
            mod_aux_loss = compute_mod_aux_loss(all_mod_infos)

        if not return_dict:
            outputs = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if all_router_logits is not None:
                outputs.append(all_router_logits)
            if mod_aux_loss is not None:
                outputs.append(mod_aux_loss)
            return tuple(v for v in outputs if v is not None)

        # Return MoD-specific output if using MoD
        if self.use_mod:
            return MoDModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                mod_aux_loss=mod_aux_loss,
                mod_infos=all_mod_infos,
            )

        # Return MoE-specific output if using MoE
        if self.use_moe:
            return MoEModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class UMichSLMForCausalLM(UMichSLMPreTrainedModel, GenerationMixin):
    """UMichSLM model with a causal language modeling head.

    This model extends UMichSLMModel with a linear layer for next-token
    prediction, making it suitable for text generation tasks.

    Args:
        config: The model configuration.

    Example:
        >>> from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
        >>> config = UMichSLMConfig(hidden_size=256, num_hidden_layers=6)
        >>> model = UMichSLMForCausalLM(config)
        >>> inputs = torch.randint(0, config.vocab_size, (1, 10))
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
    """

    # ``_tied_weights_keys`` is the *intent declaration* HF uses during
    # save/load: keys named here are treated as aliases of another tensor and
    # are skipped during weight-loading (assumed to be tied).
    #
    # Newer transformers (>= 4.49) expect a dict mapping alias-name -> source-name
    # (it calls ``tied.keys()`` in ``_get_tied_weight_keys``); older versions
    # accepted a list. We use an *empty* dict at the class level so HF never
    # treats any of our parameters as tied during save/load — this preserves
    # the ``tie_word_embeddings=False`` invariant required by the stacking
    # presets (input embedding and output lm_head are independent matrices).
    #
    # Models that enable ``tie_word_embeddings=True`` should call
    # ``self.tie_weights()`` after construction; HF's ``PreTrainedModel.tie_weights``
    # handles the tying without needing this class attribute.
    _tied_weights_keys: dict = {}
    _dynamic_tied_weights_keys: dict = {}

    def __init__(self, config: UMichSLMConfig):
        super().__init__(config)
        self.model = UMichSLMModel(config)
        self.vocab_size = config.vocab_size

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Multi-Token Prediction (training only). Supports both the existing
        # sequential DeepSeek-V3 chain (mtp_variant="deepseek", default) and the
        # new parallel Gloeckle variant (mtp_variant="gloeckle") which uses
        # independent unembedding heads (does NOT share the main lm_head, even
        # when mtp_share_head=True is set).
        self.use_mtp = getattr(config, "use_mtp", False)
        if self.use_mtp:
            self.mtp = build_mtp_module(
                config=config,
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head if getattr(config, "mtp_share_head", True) else None,
            )
        else:
            self.mtp = None

        # Mixture of Depths
        self.use_mod = getattr(config, "use_mod", False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings module."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings module."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get the output embeddings (LM head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set the output embeddings (LM head)."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: UMichSLMModel):
        """Set the decoder (base model)."""
        self.model = decoder

    def get_decoder(self) -> UMichSLMModel:
        """Get the decoder (base model)."""
        return self.model

    def set_drope_enabled(self, enabled: bool):
        """Enable or disable DRoPE (dropped position embeddings).

        Convenience method that delegates to the base model.
        See UMichSLMModel.set_drope_enabled() for details.

        Args:
            enabled: If True, drop position embeddings. If False, use normal RoPE.
        """
        self.model.set_drope_enabled(enabled)

    def get_drope_enabled(self) -> bool:
        """Check if DRoPE is currently enabled.

        Returns:
            True if position embeddings are currently dropped.
        """
        return self.model.get_drope_enabled()

    def _compute_shifted_lm_loss_chunked(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mean next-token cross-entropy, chunked over sequence to cap logits memory.

        Avoids materializing ``(batch, seq_len, vocab_size)`` logits at once. Each
        chunk runs ``lm_head`` on a slice of hidden states and accumulates loss
        with the same global mean as a single full-sequence cross-entropy over
        non-ignored positions (``ignore_index=-100``).
        """
        vocab_size = self.config.vocab_size
        bsz, seq_len, _ = hidden_states.shape
        if seq_len < 2:
            return hidden_states.sum() * 0.0

        chunk = getattr(self.config, "lm_loss_chunk_size", 512)
        chunk = max(1, min(chunk, seq_len - 1))
        # Further cap chunk from microbatch × vocab so fp32 CE logits (~flat_logits)
        # stay bounded even with large per-device batch (avoids multi-GB spikes).
        cap_bytes = int(getattr(self.config, "lm_loss_chunk_cap_bytes", 1_200_000_000))
        bytes_per_position = bsz * vocab_size * 4  # float32 logits row
        if bytes_per_position > 0:
            max_pos = max(32, cap_bytes // bytes_per_position)
            chunk = min(chunk, max_pos)
        chunk = max(1, min(chunk, seq_len - 1))

        total_loss = hidden_states.new_zeros(())
        total_count = hidden_states.new_zeros((), dtype=torch.float32)

        for start in range(0, seq_len - 1, chunk):
            end = min(start + chunk, seq_len - 1)
            chunk_h = hidden_states[:, start:end, :]
            chunk_logits = self.lm_head(chunk_h)
            chunk_labels = labels[:, start + 1 : end + 1]
            flat_logits = chunk_logits.reshape(-1, vocab_size)
            flat_labels = chunk_labels.reshape(-1).to(flat_logits.device)
            chunk_sum = F.cross_entropy(
                flat_logits.float(),
                flat_labels,
                ignore_index=-100,
                reduction="sum",
            )
            valid = (flat_labels != -100).to(dtype=torch.float32).sum()
            total_loss = total_loss + chunk_sum
            total_count = total_count + valid

        if total_count.item() == 0:
            return hidden_states.sum() * 0.0
        return total_loss / total_count

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast, MoECausalLMOutputWithPast, CausalLMOutputWithMTP]:
        """Forward pass with optional language modeling loss.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
            position_ids: Position IDs of shape (batch, seq_len).
            past_key_values: Cached key/value states.
            inputs_embeds: Pre-computed input embeddings.
            labels: Labels for language modeling loss. Shape (batch, seq_len).
                Tokens with label -100 are ignored in loss computation.
            use_cache: Whether to return updated cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            return_dict: Whether to return a ModelOutput.

        Returns:
            CausalLMOutputWithPast (or MoECausalLMOutputWithPast if MoE,
            CausalLMOutputWithMTP if using MTP) containing loss (if labels
            provided), logits, cache, and optional hidden states/attentions.
            When ``labels`` is set, ``logits`` is omitted (``None``) so the full
            ``(batch, seq, vocab)`` tensor is never materialized; call without
            ``labels`` for logits or use ``generate`` for decoding.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Memory optimization: When computing loss (labels provided), disable expensive outputs
        # that aren't needed for training. This saves significant memory.
        if labels is not None and self.training:
            use_cache = False  # KV caches are not needed during training
            output_hidden_states = False  # Accumulating all hidden states is memory-intensive
            output_attentions = False  # Attention weights are very memory-intensive

        # For MoE models, output router logits by default during training
        use_moe = getattr(self.config, "use_moe", False)
        if output_router_logits is None:
            output_router_logits = use_moe and self.training

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits: Optional[torch.Tensor] = None

        # Compute loss if labels provided
        loss = None
        lm_loss = None
        mtp_loss = None
        aux_loss = None
        mtp_logits = None

        if labels is not None:
            lm_loss = self._compute_shifted_lm_loss_chunked(hidden_states, labels)
            loss = lm_loss

            # Add Multi-Token Prediction loss (training only)
            # MTP requires input_ids for embedding lookup; skip if using inputs_embeds
            if self.use_mtp and self.mtp is not None and self.training and input_ids is not None:
                mtp_output = self.mtp(
                    hidden_states=hidden_states,
                    input_ids_or_labels=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_logits=False,  # Don't store logits during training to save memory
                )
                mtp_loss = mtp_output.loss
                mtp_logits = None  # Not stored during training for memory efficiency
                if mtp_loss is not None:
                    loss = loss + mtp_loss

            # Add MoE auxiliary losses
            if use_moe and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                aux_loss = self._compute_moe_aux_loss(outputs.router_logits)
                if aux_loss is not None:
                    loss = loss + aux_loss

            # Add MoD auxiliary loss (training only)
            if (
                self.use_mod
                and hasattr(outputs, "mod_aux_loss")
                and outputs.mod_aux_loss is not None
            ):
                mod_aux_loss = outputs.mod_aux_loss
                loss = loss + mod_aux_loss
                # Store MoD aux loss in aux_loss if not already set
                if aux_loss is None:
                    aux_loss = mod_aux_loss
                else:
                    aux_loss = aux_loss + mod_aux_loss

            # No full (B, T, V) logits when labels are present (train or eval); use
            # chunked loss above. Call without labels for logits / generation.

        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Return MTP-specific output if using MTP
        if self.use_mtp:
            router_logits = getattr(outputs, "router_logits", None) if use_moe else None
            return CausalLMOutputWithMTP(
                loss=loss,
                lm_loss=lm_loss,
                mtp_loss=mtp_loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=router_logits,
                mtp_logits=mtp_logits,
            )

        # Return MoE-specific output if using MoE
        if use_moe:
            router_logits = getattr(outputs, "router_logits", None)
            return MoECausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=router_logits,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_moe_aux_loss(
        self,
        router_logits: Tuple,
    ) -> Optional[torch.Tensor]:
        """Compute MoE auxiliary loss from router outputs.

        The auxiliary loss includes:
        1. Load balancing loss - encourages uniform expert utilization
        2. Router z-loss - prevents large router logits

        Args:
            router_logits: Tuple of MoEOutput from each layer.

        Returns:
            Total auxiliary loss tensor, or None if no valid outputs.
        """
        if router_logits is None or len(router_logits) == 0:
            return None

        total_aux_loss = torch.tensor(0.0, device=router_logits[0].aux_loss.device)
        for moe_output in router_logits:
            if moe_output is not None:
                total_aux_loss = total_aux_loss + moe_output.aux_loss + moe_output.router_z_loss

        return total_aux_loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation.

        This method is called by the generate() method to prepare the inputs
        for each generation step.

        Args:
            input_ids: Current token IDs.
            past_key_values: Cached key/value states.
            attention_mask: Current attention mask.
            inputs_embeds: Pre-computed embeddings (used in first step only).
            **kwargs: Additional arguments.

        Returns:
            Dictionary of model inputs.
        """
        # Handle past key values
        if past_key_values is not None:
            # Modern Cache API (transformers 4.36+)
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
                cache_length = getattr(past_key_values, "get_max_length", lambda: None)()
            else:
                # Legacy tuple format: ((k1, v1), (k2, v2), ...)
                # Each layer has (key_states, value_states) with shape [batch, heads, seq, dim]
                first_layer = past_key_values[0]
                if isinstance(first_layer, (list, tuple)) and len(first_layer) >= 1:
                    key_tensor = first_layer[0]
                    if hasattr(key_tensor, "shape"):
                        past_length = key_tensor.shape[2]
                    else:
                        past_length = 0
                else:
                    past_length = 0
                cache_length = None

            # Keep only tokens that haven't been processed
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif past_length >= input_ids.shape[1]:
                # This can happen when meta tokens inflate the cache beyond the
                # number of real tokens. Only the last token needs processing.
                input_ids = input_ids[:, -1:]

            # Handle static cache (if max length exceeded)
            if cache_length is not None and attention_mask is not None:
                if attention_mask.shape[1] > cache_length:
                    attention_mask = attention_mask[:, -cache_length:]

        # Create position IDs
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        elif position_ids is not None and past_key_values is not None:
            # Generation may pass forward a full-length position_ids tensor from
            # previous steps. Keep only the positions corresponding to the new
            # tokens that still need processing.
            position_ids = position_ids[:, -input_ids.shape[1] :]

        # Handle inputs_embeds (only used in first step)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Account for meta tokens in attention mask during generation
        # Meta tokens are prepended on the first forward pass, so the KV cache
        # includes them. The attention mask needs to cover those extra positions.
        num_meta = getattr(self.model, "num_meta_tokens", 0)
        if num_meta > 0 and past_key_values is not None and attention_mask is not None:
            # Only extend once (check if we already extended by comparing lengths)
            if hasattr(past_key_values, "get_seq_length"):
                cache_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                first = past_key_values[0]
                if isinstance(first, (list, tuple)) and len(first) >= 1:
                    cache_len = first[0].shape[2]
                else:
                    cache_len = 0
            else:
                cache_len = 0

            # If cache is longer than attention_mask, we need to extend the mask
            # to cover the meta token positions in the cache
            mask_len = attention_mask.shape[1]
            expected_mask_len = cache_len + input_ids.shape[1]
            if expected_mask_len > mask_len:
                # Prepend 1s for meta token positions
                extra = expected_mask_len - mask_len
                meta_mask = torch.ones(
                    attention_mask.shape[0],
                    extra,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: Union[Cache, Tuple[Tuple[torch.Tensor]]],
        beam_idx: torch.LongTensor,
    ) -> Union[Cache, Tuple[Tuple[torch.Tensor]]]:
        """Reorder cache for beam search.

        Args:
            past_key_values: The cached key/value states.
            beam_idx: Indices to select beams.

        Returns:
            Reordered cache.
        """
        if isinstance(past_key_values, DeltaNetDynamicCache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
