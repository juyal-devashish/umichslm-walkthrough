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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from .modules.doc_masking import (
    PREFIX_ID,
    build_batched_cu_seqlens,
    build_doc_start_mask,
    build_intra_doc_additive_mask,
    build_intra_doc_position_ids,
)
from .modules.engram import EngramModule
from .modules.griffin import GriffinCache, GriffinDynamicCache
from .modules.hymba_cache import HymbaDynamicCache
from .modules.kv_cache import DeltaNetDynamicCache, MLADynamicCache
from .modules.mixture_of_depths import (
    MoDGate,
    MoDOutput,
    MoDRouter,
    build_mod_compact_masks,
    compute_mod_aux_loss,
    compute_mod_block_plan,
    is_mod_layer,
    mod_gather_index,
)
from .modules.moda import MoDADepthKVCache
from .modules.multi_token_prediction import (
    GloeckleMultiTokenPrediction,
    MTPOutput,
    MultiTokenPrediction,
    build_mtp_module,
)
from .modules.rmsnorm import get_rmsnorm_class
from .modules.rotary import unscale_windowed_rotary
from .modules.token_superposition import (
    bag_embeddings,
    multihot_ce,
    shift_labels_for_tst,
)
from .modules.transformer_block import (
    UMichSLMDecoderLayer,
    UMichSLMMoDMLPDecoderLayer,
    _should_use_deltanet,
    _should_use_griffin,
    get_decoder_layer_class,
)


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
        lm_loss: Standard language modeling loss, before any aux loss is folded in.
        aux_loss: MoE auxiliary loss (load balancing + z-loss).
        logits: Prediction scores.
        past_key_values: Cached key/value states.
        hidden_states: All hidden states (if output_hidden_states=True).
        attentions: All attention weights (if output_attentions=True).
        router_logits: Router logits from all MoE layers.
    """

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
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


def pathway_attention_mask(attention_mask_4d, attention_mask, shape, inputs_embeds, past_length):
    """The mask to hand the MoP lane blocks, which run their own attention.

    Under flash_attention_2 the model leaves ``attention_mask_4d`` as the caller's raw
    ``[B, S]`` padding mask, because FA2 unpads for itself. The lane blocks read it as an
    additive ``[B, 1, S, S]`` mask, so a 2D mask either failed a shape check (ragged
    batches) or was expanded into a ``[B, S, k, S]`` gather (batch 1, 184,320 GiB at 32K).
    Hand them what the sdpa branch builds instead: nothing when no token is padded, which
    keeps the lanes on the flex causal/doc path, else the 4D causal + padding mask.
    """
    if attention_mask_4d is None or attention_mask_4d.dim() != 2:
        return attention_mask_4d
    if bool(attention_mask_4d.all()):
        return None
    return _prepare_4d_causal_attention_mask(attention_mask, shape, inputs_embeds, past_length)


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
        "RWKV7Mixer",
        "HymbaHybridHead",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True  # transformers <5.x flag name
    _supports_flash_attn = True  # transformers >=5.x renamed the flag (both kept for compat)
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        """Initialize model weights.

        Uses standard normal initialization with configurable standard deviation.

        Must go through ``torch.nn.init.*`` (not raw ``tensor.data.normal_()``):
        transformers >= 5 re-runs ``_init_weights`` on every module *after*
        loading a checkpoint and relies on the ``init`` functions being guarded
        by the ``_is_hf_initialized`` flag to leave already-loaded params alone.
        Raw in-place tensor methods bypass that guard and would silently
        overwrite the loaded checkpoint with fresh random weights. ``nn.init.*``
        is functionally identical on older transformers, so this is safe across
        versions.

        Args:
            module: The module to initialize.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            # Flamingo-style safe insertion (GWT workspace read, and any
            # future zero-init merge projections): modules mark the Linear
            # itself so the zero survives every post_init / post-load re-init.
            if getattr(module, "_zero_init_weight", False):
                nn.init.zeros_(module.weight)
            # Same problem, bias edition: a module whose init MEANS something
            # (DeltaMemory's beta bias of 2.0 == "overwrite fully", the standard
            # DeltaNet start) loses it to the nn.init.zeros_ above unless it is
            # marked. Found 2026-08-21: DeltaMemory shipped with BOTH its
            # zero-init o_proj and its beta bias silently overwritten here, so
            # `swa_deltamem` never actually started as its backbone and ran at
            # beta=sigmoid(0)=0.5 instead of 0.88.
            _cb = getattr(module, "_const_init_bias", None)
            if _cb is not None and module.bias is not None:
                nn.init.constant_(module.bias, float(_cb))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            # Zero the padding row only when this weight wasn't loaded from a
            # checkpoint (the flag is set by the loader on restored params).
            if module.padding_idx is not None and not getattr(
                module.weight, "_is_hf_initialized", False
            ):
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

        # Rebuild any module that owns computed ``persistent=False`` buffers
        # (RoPE/PoPE inv_freq + cos/sin caches). transformers runs
        # ``_init_weights`` on every module during the post-load init pass, which
        # is the only reliable hook to recompute these — meta-device fast-init in
        # ``from_pretrained`` leaves non-persistent computed buffers as
        # uninitialized memory (they're absent from the checkpoint by design).
        if hasattr(module, "reset_buffers"):
            module.reset_buffers()


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

        # Per-Layer Embeddings (Gemma-3n/4): one small token table per decoder
        # layer, added to that layer's input. No-op unless config.use_ple.
        self.per_layer_embeddings = None
        if getattr(config, "use_ple", False):
            from .modules.per_layer_embeddings import PerLayerEmbeddings

            self.per_layer_embeddings = PerLayerEmbeddings(
                vocab_size=config.vocab_size,
                num_layers=config.num_hidden_layers,
                ple_dim=getattr(config, "ple_dim", 256),
                hidden_size=config.hidden_size,
            )

        # Track feature flags
        self.use_moe = getattr(config, "use_moe", False)
        self.use_mod = getattr(config, "use_mod", False)
        self.mod_every_n_layers = getattr(config, "mod_every_n_layers", 2)
        self.mod_atomic_block = getattr(config, "mod_atomic_block", False)
        self.mod_mlp_only = getattr(config, "mod_mlp_only", False)
        self.use_deltanet = getattr(config, "use_deltanet", False)
        self.use_griffin = getattr(config, "use_griffin", False)
        self.use_hymba = getattr(config, "use_hymba", False)
        self.use_moda = getattr(config, "use_moda", False)

        # Router-free recurrent depth (Arm B): the core block (between the
        # first and last block) is looped a sampled number of times per
        # training forward. `_recurrent_force_iters` overrides sampling for
        # tests/diagnostics; `_last_recurrent_iters` records the realized count
        # for the per-token-compute diagnostics.
        self.use_recurrent_depth = getattr(config, "use_recurrent_depth", False)
        self._recurrent_force_iters: Optional[int] = None
        self._last_recurrent_iters: Optional[int] = None
        # Loop-boundary mixers on top of the plain loop: Parcae stable
        # injection and/or Hyperloop hyper-connections (modules/loop_mixers.py).
        # None for the plain loop and for every non-recurrent arch.
        from .modules.loop_mixers import build_recurrent_loop_mixer

        self.recurrent_mixer = build_recurrent_loop_mixer(config)

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
        swa_use_registers = bool(getattr(config, "swa_use_registers", False))
        if self.attention_mode == "swa_globals_mla" and swa_global_count > 0 and swa_use_registers:
            # Registers twin: same-sized bank, but NEVER prepended. SWA layers
            # read it via zero-init cross-attention (bound in forward, mirroring
            # the num_protected_prefix channel so stacking-inserted layers get
            # bound too). num_global_tokens stays 0 so every prefix mechanism
            # (mask building, force-routing, loss slicing) is naturally inert.
            #
            # Hybrid split (swa_register_count < swa_global_count): that many
            # tokens form the register bank and the REMAINDER stays a prepended
            # global prefix — full prefix machinery active at the smaller count
            # (sinks + MLA-layer visibility), register read on the SWA layers.
            from .modules.global_tokens import RegisterBank

            reg_count = int(getattr(config, "swa_register_count", None) or swa_global_count)
            prefix_count = swa_global_count - reg_count
            self.register_tokens = RegisterBank(reg_count, config.hidden_size)
            if prefix_count > 0:
                from .modules.global_tokens import GlobalTokens

                self.global_tokens = GlobalTokens(prefix_count, config.hidden_size)
                self.num_global_tokens = prefix_count
            else:
                self.global_tokens = None
                self.num_global_tokens = 0
        elif self.attention_mode == "swa_globals_mla" and swa_global_count > 0:
            from .modules.global_tokens import GlobalTokens

            self.global_tokens = GlobalTokens(swa_global_count, config.hidden_size)
            self.num_global_tokens = swa_global_count
            self.register_tokens = None
        elif int(getattr(config, "attn_register_count", 0) or 0) > 0:
            # Register bank for the full-attention layers of a hybrid schedule
            # (modules/swa_registers.py FullRegisterReadMixin); bound in forward
            # exactly like the SWA registers twin, never prepended.
            from .modules.global_tokens import RegisterBank

            self.global_tokens = None
            self.num_global_tokens = 0
            self.register_tokens = RegisterBank(int(config.attn_register_count), config.hidden_size)
        else:
            self.global_tokens = None
            self.num_global_tokens = 0
            self.register_tokens = None

        # Latent feedback (modules/latent_feedback.py): the previous position's
        # top-layer state fused into this position's input. Model-level, like the
        # SWA globals, so it survives a progressive-stacking depth doubling.
        self.latent_feedback = None
        if getattr(config, "use_latent_feedback", False):
            from .modules.latent_feedback import LatentFeedback

            self.latent_feedback = LatentFeedback(config)
        # Counts forwards to drive the pass-count mixture. A PLAIN INT, not a
        # buffer: it must never enter a checkpoint (a resumed run would then
        # disagree with a fresh one about the schedule for no benefit), and
        # every rank bumps it in lockstep so DDP ranks agree on the pass count.
        self._feedback_fwd_index = 0

        # State/prediction separation (modules/pause_stream.py).
        self.pause_stream = None
        self.pause_enabled = False
        if getattr(config, "use_pause_stream", False):
            from .modules.pause_stream import PauseStream

            self.pause_stream = PauseStream(config)
            self.pause_enabled = True

        # swa_scribe banks: out-of-sequence sink bank + chunk-committed scribes
        # + optional per-forward register refinement (modules/swa_scribe.py).
        self.sink_tokens = None
        self.scribe_bank = None
        self.scribe_write_idx = 0
        self.register_refiner = None
        if int(getattr(config, "swa_sink_count", 0)) > 0:
            from .modules.global_tokens import SinkBank

            self.sink_tokens = SinkBank(int(config.swa_sink_count), config.hidden_size)
        if getattr(config, "use_scribe_bank", False):
            from .modules.swa_scribe import ScribeBank

            self.scribe_bank = ScribeBank(config)
            w_idx = int(getattr(config, "scribe_write_layer", -1))
            self.scribe_write_idx = w_idx if w_idx >= 0 else config.num_hidden_layers // 2
            if getattr(config, "use_mod", False) and getattr(config, "mod_atomic_block", False):
                bs = int(getattr(config, "mod_block_size", 4))
                if self.scribe_write_idx % bs != 0:
                    raise ValueError(
                        f"scribe_write_layer={self.scribe_write_idx} is not block-aligned "
                        f"(mod_block_size={bs}); a gathered MoD block consumes it and the "
                        f"scribe write never runs. Use a multiple of {bs}."
                    )
        if getattr(config, "swa_register_refine", False):
            from .modules.swa_scribe import BankRefiner

            self.register_refiner = BankRefiner(config)

        # swa_deltabank: cumulative chunk-writes into the register bank at one
        # write layer (modules/delta_bank.py). Same block-alignment contract as
        # the scribe write (dead-trunk post-mortem 2026-07-19).
        self.delta_bank = None
        self.delta_write_idx = 0
        if getattr(config, "use_delta_bank", False):
            from .modules.delta_bank import DeltaBank

            self.delta_bank = DeltaBank(config)
            w_idx = int(getattr(config, "delta_write_layer", -1))
            self.delta_write_idx = w_idx if w_idx >= 0 else config.num_hidden_layers // 2
            if getattr(config, "use_mod", False) and getattr(config, "mod_atomic_block", False):
                bs = int(getattr(config, "mod_block_size", 4))
                if self.delta_write_idx % bs != 0:
                    raise ValueError(
                        f"delta_write_layer={self.delta_write_idx} is not block-aligned "
                        f"(mod_block_size={bs}); a gathered MoD block consumes it and the "
                        f"delta write never runs. Use a multiple of {bs}."
                    )

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

        # MoD-MLP (Arm A): FFN-only routing. The routed-layer set mirrors the
        # standard MoD plan (per-layer stride, or block-atomic block members).
        # Members become UMichSLMMoDMLPDecoderLayer; under mod_atomic_block they
        # carry no router of their own — one decision at the block head (built
        # below) gates every member layer's FFN.
        _mod_mlp_members: set = set()
        if self.use_mod and self.mod_mlp_only:
            from .compute_accounting import mod_routed_capacities

            _mod_mlp_members = set(mod_routed_capacities(config).keys())
            _mod_mlp_members -= self.always_used_layers

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            # Determine if this layer should use MoD
            # Layers in always_used_layers are excluded from MoD routing
            # Block-atomic MoD keeps every decoder layer plain (no per-layer MoD
            # wrapper); routing is applied around whole blocks in forward() via
            # self.mod_block_routers / self.mod_block_gate built below. The legacy
            # per-layer path is used only when mod_atomic_block is off.
            layer_uses_mod = (
                self.use_mod
                and not self.mod_atomic_block
                and not self.mod_mlp_only
                and is_mod_layer(layer_idx, self.mod_every_n_layers)
                and layer_idx not in self.always_used_layers
            )
            layer_uses_mod_mlp = layer_idx in _mod_mlp_members

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
                use_mod_mlp=layer_uses_mod_mlp,
            )

            # Pass num_experts to MoE layers for progressive MoE
            if layer_uses_moe:
                self.layers.append(decoder_layer_class(config, layer_idx, num_experts=num_experts))
            elif layer_uses_mod_mlp:
                # Block-atomic MLP routing: the decision is made once at the
                # block head (model-level router); member layers carry none.
                self.layers.append(
                    decoder_layer_class(config, layer_idx, own_router=not self.mod_atomic_block)
                )
            else:
                self.layers.append(decoder_layer_class(config, layer_idx))

        # NoPE (SmolLM3): statically strip position encoding from every Nth layer.
        # Unlike DRoPE this is fixed for the whole run, so it is set once here
        # rather than toggled by a training callback.
        nope_n = int(getattr(config, "nope_every_n_layers", 0) or 0)
        if nope_n > 0:
            for layer_idx, layer in enumerate(self.layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None and (layer_idx + 1) % nope_n == 0:
                    attn.nope = True

        # Block-atomic MoD routers. One MoDRouter per MoD block (keyed by the
        # block-head layer index) plus one shared gate; the block is routed as a
        # unit in forward(). Built here so the plain decoder layers above are
        # left untouched, keeping KV-cache / staging / titan per-layer indexing
        # intact.
        self.mod_block_routers = nn.ModuleDict()
        self.mod_block_gate = None
        self._mod_head_to_members: Dict[int, list] = {}
        self._mod_tail_to_head: Dict[int, int] = {}
        if self.use_mod and self.mod_atomic_block:
            block_size = getattr(config, "mod_block_size", 0) or 0
            if block_size <= 0:
                hybrid = getattr(config, "hybrid_attention_ratio", None)
                if hybrid:
                    block_size = len(hybrid)
                elif self.mod_every_n_layers > 0:
                    block_size = self.mod_every_n_layers
                else:
                    block_size = 4
            self._mod_block_size = block_size

            # Which blocks are MoD-routed, and at what capacity. Two regimes:
            #  - Staging (config.stage_block_spec present): route exactly the
            #    blocks marked is_inserted, at mod_capacity_inserted — the
            #    stacking-plan design where inserted (duplicated) blocks are MoD
            #    and original + I/O blocks stay dense.
            #  - Flat (no stage spec): the compute_mod_block_plan rule (all blocks
            #    minus I/O when io_blocks_dense), all at mod_capacity.
            stage_spec = getattr(config, "stage_block_spec", None)
            mod_blocks: list = []  # list of (members, capacity)
            if stage_spec:
                inserted_cap = float(getattr(config, "mod_capacity_inserted", 0.5))
                for b, entry in enumerate(stage_spec):
                    if not bool(entry.get("is_inserted", False)):
                        continue
                    members = list(range(b * block_size, (b + 1) * block_size))
                    mod_blocks.append((members, inserted_cap))
            else:
                cap = float(getattr(config, "mod_capacity", 0.125))
                _, plan = compute_mod_block_plan(
                    config.num_hidden_layers,
                    block_size,
                    io_blocks_dense=getattr(config, "mod_io_blocks_dense", True),
                )
                mod_blocks = [(members, cap) for members in plan]

            # Two ways a block-atomic MoD plan can be silently useless, both of
            # which have bitten this sweep:
            #
            #  1. No routed block at all (io_blocks_dense=True with <=2 blocks):
            #     MoD is configured but never fires, so the arch is not the arch.
            #  2. No DENSE layer at all (io_blocks_dense=False, every layer in a
            #     routable block): a token the router declines in every block
            #     traverses zero transformer layers — embeddings straight to the
            #     final norm and the LM head. It trains, converges, and floors ~1.4
            #     nats short. Measured: 24.2% of tokens saw no layer.
            #
            # Neither raises anywhere else, so refuse to build the model.
            # Exception: MoD-MLP (FFN-only routing) is structurally immune to
            # total bypass — attention runs dense for every token in every
            # layer — so routable-everywhere is legal there (that is exactly
            # what lets swa_globals_mlp ride the shared 5-9 layer sweep shapes).
            allow_bypass = bool(getattr(config, "mod_allow_total_bypass", False))
            allow_bypass = allow_bypass or self.mod_mlp_only
            if mod_blocks:
                routed = {m for members, _ in mod_blocks for m in members}
                if len(routed) >= config.num_hidden_layers and not allow_bypass:
                    raise ValueError(
                        f"Block-atomic MoD would route ALL {config.num_hidden_layers} layers "
                        f"(block_size={block_size}, mod_io_blocks_dense="
                        f"{getattr(config, 'mod_io_blocks_dense', True)}), leaving no dense "
                        "layer. Tokens declined by every router would bypass the whole "
                        "network. Set mod_io_blocks_dense=True and use num_hidden_layers >= "
                        f"{3 * block_size} (L = block_size*(R+2) gives R routed blocks). "
                        "If you are unit-testing the gather/scatter machinery and want "
                        "this on purpose, set mod_allow_total_bypass=True."
                    )
            elif getattr(config, "mod_atomic_block", False):
                raise ValueError(
                    f"Block-atomic MoD is enabled but no block is routed: "
                    f"num_hidden_layers={config.num_hidden_layers} with block_size="
                    f"{block_size} yields <=2 blocks, and mod_io_blocks_dense keeps the "
                    f"first and last dense. Use num_hidden_layers >= {3 * block_size}."
                )

            for members, capacity in mod_blocks:
                head = members[0]
                self.mod_block_routers[str(head)] = MoDRouter(
                    hidden_size=config.hidden_size,
                    capacity=capacity,
                    aux_loss_weight=getattr(config, "mod_aux_loss_weight", 0.01),
                    use_aux_predictor=getattr(config, "mod_use_aux_predictor", False),
                    gate_fn=getattr(config, "mod_gate_fn", "raw"),
                    tanh_alpha=getattr(config, "mod_tanh_alpha", 0.2),
                    capacity_warmup_steps=getattr(config, "mod_capacity_warmup_steps", 0),
                    router_mode=getattr(config, "mod_router_mode", "topk"),
                    router_chunk=getattr(config, "mod_router_chunk", 128),
                    route_bias_lr=getattr(config, "mod_route_bias_lr", 1e-3),
                )
                self._mod_head_to_members[head] = list(members)
                self._mod_tail_to_head[members[-1]] = head
            # MoD-MLP: no block-boundary gate — each member layer gates its own
            # FFN delta with the head decision, so the block gate stays None
            # (which also keeps the attention gather/scatter path disabled).
            if self._mod_head_to_members and not self.mod_mlp_only:
                self.mod_block_gate = MoDGate(config.hidden_size)
        # Member layer -> block head, for handing the head decision to MoD-MLP
        # member layers during forward.
        self._mod_layer_to_head: Dict[int, int] = {
            m: head for head, members in self._mod_head_to_members.items() for m in members
        }

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

        # Global Workspace Trunk (GWT, chunk 2 Pitch 1): a live recurrent
        # workspace updated by a deep trunk at chunk boundaries, read by the
        # content tokens at ONE integration point between token-stream layers.
        # The token stream itself (all decoder layers, masks, cross-chunk
        # attention) is completely unchanged.
        # DeltaMemory: delta-rule associative memory as a SIDE branch on the
        # residual stream (modules/delta_memory.py). Placed on every
        # delta_mem_interval-th layer rather than all of them -- a per-layer
        # mechanism costs roughly its layer count, and at 80 layers that is the
        # difference between free and unaffordable. Zero-init output, so the arm
        # starts bitwise-identical to its backbone.
        self.delta_memories = None
        self._delta_mem_at = {}
        if getattr(config, "use_delta_memory", False):
            from .modules.delta_memory import DeltaMemory

            iv = max(1, int(getattr(config, "delta_mem_interval", 4)))
            lo, hi = 0, int(config.num_hidden_layers)
            if getattr(config, "delta_mem_band_only", True) and getattr(
                config, "use_mat_ffn", False
            ):
                # STAY IN THE ADAPTIVE BAND. mat_ffn deliberately keeps a
                # fixed-compute, full-width prefix and suffix -- "early layers
                # build the representation everything else reads; a token
                # starved there cannot recover" (modules/transformer_block.py)
                # -- and only the middle band routes per token. A persistent
                # memory belongs with the reasoning steps, not bolted onto the
                # layers that were deliberately left non-adaptive. The shipped
                # placement (`i % iv == 0` over the whole stack) put 4 of 20
                # modules at layers 0, 4, 72 and 76, i.e. inside both fixed
                # zones. Band-only also drops the module count 20 -> 16, which
                # is a straight 20% cut in the memory the arm carries.
                #
                # Only applies when there IS a band. swa_deltamem rides the
                # sinkdiet backbone, which has no mat_ffn band and does have
                # trained 10m/25m cells, so its placement is left alone rather
                # than silently changed under existing results.
                L = int(config.num_hidden_layers)
                cap = max(0, (L - 2) // 3)
                lo = min(int(getattr(config, "mat_ffn_dyn_start", 0) or 0), cap)
                hi = L - min(int(getattr(config, "mat_ffn_dyn_end", 0) or 0), cap)
            at = [i for i in range(lo, hi) if (i - lo) % iv == 0]
            self.delta_memories = nn.ModuleList([DeltaMemory(config) for _ in at])
            self._delta_mem_at = {layer: n for n, layer in enumerate(at)}
        self.workspace_trunk = None
        self.trunk_insert_idx = 0
        if getattr(config, "use_workspace_trunk", False):
            from .modules.workspace_trunk import (
                GatedScanWorkspaceTrunk,
                ParallelWorkspaceTrunk,
                WorkspaceTrunk,
            )

            variant = str(getattr(config, "trunk_variant", "sequential"))
            if variant == "parallel":
                self.workspace_trunk = ParallelWorkspaceTrunk(config)
            elif variant in GatedScanWorkspaceTrunk.SCAN_VARIANTS:
                self.workspace_trunk = GatedScanWorkspaceTrunk(config)
            else:
                self.workspace_trunk = WorkspaceTrunk(config)
            insert = int(getattr(config, "trunk_insert_layer", -1))
            self.trunk_insert_idx = insert if insert >= 0 else config.num_hidden_layers // 2
            # Block-atomic MoD consumes non-head members of routed blocks
            # before the insert hook is reached, silently skipping the trunk.
            if getattr(config, "use_mod", False) and getattr(config, "mod_atomic_block", False):
                bs = int(getattr(config, "mod_block_size", 4))
                if self.trunk_insert_idx % bs != 0:
                    raise ValueError(
                        f"trunk_insert_layer={self.trunk_insert_idx} is not block-aligned "
                        f"(mod_block_size={bs}); a gathered MoD block consumes it and the "
                        f"trunk never runs. Use a multiple of {bs}."
                    )
        self._trunk_force_iters: Optional[int] = None
        self._last_trunk_iters: Optional[int] = None

        # Mixture of Pathways (MoP, chunk 2 Pitch 2): parallel depth lanes
        # between the shared bottom layers and shared top layers, one causal
        # lane commitment per token per stage. With pathway_stages > 1 the
        # middle region holds several stages at consecutive layer boundaries
        # (the decoder layer between two stages is that stage's dense spine),
        # combining MoP's committed depth choice with blocked-MoD's linear
        # parameter growth in depth; `pathway_stage_share_lanes` reuses one
        # lane set across all stages (sub-linear — the deep-target variant).
        self.pathways = None
        self.pathway_stages = None
        self.pathway_insert_idx = 0
        self._pathway_insert: Dict[int, nn.Module] = {}
        if getattr(config, "use_pathways", False):
            from .modules.pathways import Pathways

            bottom = int(getattr(config, "pathway_shared_bottom", 0) or 0)
            if bottom == 0:
                bottom = max(1, (2 * config.num_hidden_layers) // 3)
            self.pathway_insert_idx = bottom
            num_stages = int(getattr(config, "pathway_stages", 1) or 1)
            if num_stages == 1:
                self.pathways = Pathways(config)
                self._pathway_insert = {bottom: self.pathways}
            else:
                stages = [Pathways(config) for _ in range(num_stages)]
                if getattr(config, "pathway_stage_share_lanes", False):
                    # Lane sets (and shared-K/V heads) reused across stages;
                    # routers and merge norms/scales stay per-stage. With
                    # pathway_stage_lane_groups > 1 the sharing is banded:
                    # contiguous stage groups each own an independent lane set
                    # (early/mid/late depth specialization at sub-linear
                    # params); groups == num_stages degenerates to unshared.
                    groups = int(getattr(config, "pathway_stage_lane_groups", 1) or 1)
                    leaders: dict = {}
                    for s, stage in enumerate(stages):
                        g = (s * groups) // num_stages
                        lead = leaders.setdefault(g, stage)
                        if lead is stage:
                            continue
                        if lead.lanes is not None:
                            stage.lanes = lead.lanes
                        else:
                            stage.lane_stack = lead.lane_stack
                        if lead.shared_kv_proj is not None:
                            stage.shared_kv_proj = lead.shared_kv_proj
                        if lead.memory_block is not None:
                            stage.memory_block = lead.memory_block
                for s, stage in enumerate(stages):
                    stage._stage_index = s  # pathway_export_interval phase
                self.pathway_stages = nn.ModuleList(stages)
                self._pathway_insert = {bottom + s: stages[s] for s in range(num_stages)}

        # P1 `dual_stream`: wide sparse blocks every dual_stream_every layers.
        self._slow_stream = None
        if getattr(config, "use_dual_stream", False):
            from .modules.dual_stream import SlowStreamBlock

            every = max(1, int(getattr(config, "dual_stream_every", 4)))
            # Fixed-compute prefix: no slow-stream block before
            # dual_stream_start_layer, so the early layers process every token
            # identically (design docs -- only the middle reasoning layers are
            # dynamic).
            _ds_lo = int(getattr(config, "dual_stream_start_layer", 0) or 0)
            self._slow_at = {
                i for i in range(config.num_hidden_layers) if (i + 1) % every == 0 and i >= _ds_lo
            }
            self._slow_stream = nn.ModuleDict(
                {
                    str(i): SlowStreamBlock(
                        config,
                        float(getattr(config, "dual_stream_width_mult", 2.0)),
                        float(getattr(config, "dual_stream_frac", 0.2)),
                    )
                    for i in sorted(self._slow_at)
                }
            )

        # S1 `hourglass`: the expensive middle stack lives INSIDE this module and
        # runs on a pooled sequence, so `self.layers` above holds only the
        # full-resolution outer layers. See modules/hourglass.py.
        self._hourglass = None
        self._hourglass_at = -1
        if getattr(config, "use_hourglass", False):
            from .modules.hourglass import HourglassMiddle

            _hg_layers = int(getattr(config, "hourglass_middle_layers", 0) or 0)
            if _hg_layers <= 0:
                raise ValueError("use_hourglass requires hourglass_middle_layers >= 1")
            _hg_cls = get_decoder_layer_class(config)
            self._hourglass = HourglassMiddle(
                config,
                _hg_cls,
                _hg_layers,
                int(getattr(config, "hourglass_pool_ratio", 4)),
                base_idx=config.num_hidden_layers,
            )
            at = getattr(config, "hourglass_insert_layer", None)
            self._hourglass_at = config.num_hidden_layers // 2 if at is None else int(at)

        # Final layer norm
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Latch for the one-shot vocab-range check in forward(). Plain attribute,
        # NOT a buffer: it is a debug guard, and persisting it would change every
        # existing checkpoint's state_dict.
        self._vocab_range_checked = False

        # DRoPE (Dropped Rotary Position Embeddings) state
        # When True, position embeddings are skipped in attention layers
        self._drope_enabled = False

        # Message structure on the MLA layers (P1 turnpos / P2 digests,
        # modules/msg_struct.py): None unless msg_rope / msg_pin_bias / msg_digest.
        from .modules.msg_struct import build_message_structure

        self.msg_struct = build_message_structure(config, self)

        # Initialize weights and apply final processing
        if getattr(config, "deltanet_variant", None) == "rwkv7":
            # RWKV-7's value residual: one shared per-forward store across the mixers.
            from .modules.rwkv7 import bind_value_residual

            bind_value_residual(self)

        self.post_init()

        # YaRN on the global layers only (config.rope_scaling_scope="global"). Runs
        # after post_init() and changes the rotaries' ATTRIBUTES, so the post-load
        # reset_buffers() pass in _init_weights rebuilds them unscaled as well.
        # Before the presize below, whose cache length it would otherwise reset.
        if self.config.rope_scaling and getattr(self.config, "rope_scaling_scope", "all") == "global":
            unscale_windowed_rotary(self)

        # Pre-size the RoPE caches to cover the prepended prefix so torch.compile
        # never needs a dynamic cache extend. A prefix of P tokens shifts content
        # positions up by P (see forward: position_ids + num_globals/num_meta), so
        # the largest position is (max_position_embeddings - 1) + P, one past the
        # P=0 cache. In eager _maybe_extend_cache grows the cache at that first
        # step; under compile it is a no-op (a dynamic extend would be a graph
        # break / device-side "index out of bounds" in the compiled RoPE gather —
        # exactly the crash the prefix arms hit at 4096 with torch_compile=true,
        # 2026-07-20). Pre-sizing here reproduces the eager cache bit-for-bit,
        # just ahead of the first forward. MUST run AFTER post_init(), which
        # re-registers the (non-persistent) cos/sin buffers at the P=0 size.
        self._presize_rotary_caches()

    def _presize_rotary_caches(self) -> None:
        prefix = int(getattr(self, "num_global_tokens", 0)) + int(
            getattr(self, "num_meta_tokens", 0)
        )
        if prefix <= 0:
            return
        target = int(self.config.max_position_embeddings) + prefix
        for mod in self.modules():
            if hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "max_seq_len_cached"):
                if int(getattr(mod, "max_seq_len_cached", 0)) < target:
                    mod._set_cos_sin_cache(target)

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

    def set_pause_enabled(self, enabled: bool) -> bool:
        """Turn state/prediction separation on or off mid-run (phasing).

        The paper's cheap variants (1.09-1.33x instead of 1.57x) run a standard
        schedule for a fraction of the steps and switch separation on for the
        rest; ``e_pause`` is the only tensor that has to initialise fresh, so
        every other piece of training state carries straight over. Returns the
        flag actually set (False when the model has no pause stream at all), so
        a caller can log what happened rather than assume.
        """
        if self.pause_stream is None:
            return False
        enabled = bool(enabled)
        self.pause_enabled = enabled
        # Each layer reads its own plain bool, so the switch has to be pushed
        # down. Under torch.compile this costs one recompile at the switchover.
        for layer in self.layers:
            if getattr(layer, "pause_two_stream", False):
                layer.pause_enabled = enabled
        return enabled

    def _run_gathered_mod_block(
        self,
        members,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        num_globals: int,
        input_ids: Optional[torch.LongTensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        flex_cache: Optional[dict] = None,
    ):
        """Run one MoD block over ONLY the selected tokens (real FLOP savings).

        Gathers the router-selected tokens into a compact ``[B, k, H]`` sequence,
        runs the block's layers on it (so attention is O(k^2) and MLP is O(k),
        not O(T)), then scatters the result back — unselected tokens bypass the
        whole block. Selected tokens attend only to other selected tokens (true
        MoD semantics), with the sliding window honoured in original-position
        space via a precomputed mask. Returns ``(new_hidden, MoDOutput)``.

        Only valid without a KV cache (training / loss eval), and only for
        softmax-attention blocks (SWA / MLA / GQA) whose visibility is expressible
        as an additive mask. Returns ``(None, None)`` if the per-row selection
        counts disagree or a member layer uses an unsupported attention (e.g.
        DeltaNet / Mamba / Griffin recurrences), so the caller falls back to the
        dense gate path.
        """
        from .modules.flex_swa import flex_available
        from .modules.swa_registers import RegisterReadMixin
        from .modules.swa_with_globals import SWAWithGlobalsAttention

        # Gather/scatter over a compact sequence only makes sense for attention
        # that consumes an additive mask; linear/recurrent variants (DeltaNet,
        # Mamba2, Griffin, Hymba) carry cross-token state that a token subset
        # would corrupt. Fall back to the dense gate path for those.
        _unsupported = ("DeltaNet", "Mamba", "Griffin", "Hymba")
        for _m in members:
            _attn_name = type(getattr(self.layers[_m], "self_attn", None)).__name__
            if any(u in _attn_name for u in _unsupported):
                return None, None

        head = members[0]
        router = self.mod_block_routers[str(head)]
        router_weights, selected_mask, aux_loss = router(
            hidden_states, use_cache=False, force_route_prefix=num_globals
        )
        # Sync-free gather hints. `expected` comes from the router (topk selects
        # exactly prefix+k per row by construction) and removes the last
        # device->host sync on the hot path. `fixed` (mod_fixed_gather, the kmod
        # arm) gathers a STATIC width with causal positional clipping — see
        # mod_gather_index; only consulted when counts are ragged.
        expected = getattr(router, "_last_uniform_count", None)
        fixed = None
        if expected is None and bool(getattr(self.config, "mod_fixed_gather", False)):
            slack = float(getattr(self.config, "mod_fixed_gather_slack", 1.1))
            content = hidden_states.shape[1] - num_globals
            cap = float(getattr(router, "_last_capacity", 0.0))
            fixed = num_globals + max(1, int(math.ceil(content * cap * slack)))
        _gathered = mod_gather_index(
            selected_mask,
            bucket=int(getattr(self.config, "mod_gather_bucket", 128)),
            expected_count=expected,
            fixed_width=fixed,
        )
        if _gathered is None:
            return None, None
        gather_idx, gather_valid = _gathered  # [B, k]; valid is None when uniform

        B, T, H = hidden_states.shape
        k = gather_idx.shape[1]
        idx_h = gather_idx.unsqueeze(-1).expand(B, k, H)
        compact_in = torch.gather(hidden_states, 1, idx_h)  # [B, k, H]
        compact_pos = torch.gather(position_ids, 1, gather_idx)  # [B, k]
        compact = compact_in

        window = int(
            getattr(self.config, "swa_window", getattr(self.config, "sliding_window", 0)) or 0
        )
        # Document ids of the gathered tokens. Without these the compact mask
        # tests causality on positions that build_intra_doc_position_ids has
        # already reset per document, and content leaks across packed-document
        # boundaries for every layer of this block (see build_mod_compact_masks).
        compact_docs = torch.gather(doc_ids, 1, gather_idx) if doc_ids is not None else None

        # FlexAttention block-sparse path: same visibility, but a mask_mod plus a
        # coarse map of skippable 128x128 tiles instead of two dense [B,1,k,k]
        # additive masks (268 MB at B16/k2080/bf16) that also disqualify SDPA's
        # flash backend. Fixed capacity + forced prefix make k identical on every
        # step (see mod_gather_index), so the dynamic=False compile does not
        # thrash. Only the SWA members can take it -- MLA is a different class
        # with no _flex_block_mask hook -- so the causal mask is still built.
        use_flex = (
            getattr(self.config, "mod_flex_attention", True)
            and flex_available()
            and compact_in.is_cuda
        )
        flex_masks: Dict[int, object] = {}

        def _flex_mask_for(n_extra: int):
            # A scribe-class member appends its sink bank as ``n_extra`` extra
            # K/V columns inside the attention (always-visible, content-free
            # learned tokens); the BlockMask must cover k + n_extra keys or the
            # kernel rejects the shape. Built lazily per distinct extension —
            # in practice one variant per arch — through the step cache so
            # checkpoint recompute reuses the identical object.
            bm = flex_masks.get(n_extra)
            if bm is None:
                from .modules.flex_swa import get_mod_compact_block_mask

                bm = get_mod_compact_block_mask(
                    compact_pos,
                    compact_docs,
                    num_globals,
                    window if window > 0 else T,
                    step_cache=flex_cache,
                    cache_key=(id(self), "mod", head, k, "swa", n_extra),
                    valid=gather_valid,
                    extra_kv=n_extra,
                )
                flex_masks[n_extra] = bm
            return bm

        # When flex covers the SWA members, skip materialising the windowed
        # [B,1,k,k] additive mask entirely: the BlockMask takes unconditional
        # dispatch priority (attention.py:382-386), so the dense tensor would
        # be built, carried, and never read. The causal mask is still needed
        # (the MLA member has no flex hook), and it doubles as the non-None
        # ``_mod_external_mask`` compact-contract signal for SWA members below.
        causal_mask, windowed_mask = build_mod_compact_masks(
            compact_pos,
            num_globals,
            window if window > 0 else T,
            compact_in.dtype,
            doc_ids=compact_docs,
            valid=gather_valid,
            want_windowed=not use_flex,
        )

        # Per-Layer Embeddings must be applied to the COMPACT sequence: the main
        # forward loop's PLE step is skipped for the layers this block consumes, so
        # without this a PLE + block-atomic-MoD model would silently train with its
        # per-layer tables receiving no gradient at all. The router force-routes the
        # global prefix, so compact positions [0, num_globals) are exactly the
        # globals and the remainder index content tokens (offset by num_globals).
        compact_ids = None
        if self.per_layer_embeddings is not None and input_ids is not None:
            content_idx = gather_idx[:, num_globals:] - num_globals
            compact_ids = torch.gather(input_ids, 1, content_idx.clamp_min(0))

        def _run_member(layer, attn, is_swa, hs, mask, pos, block_mask=None):
            # ``_mod_external_mask`` must be installed *inside* this callable, not
            # around the checkpoint() call. With use_reentrant=False the body is
            # re-executed during backward, and an attribute set (and cleared) by
            # the caller is already None by then — SWAWithGlobalsAttention would
            # silently take its flex/globals-window branch on recompute instead of
            # the compact-mask branch, producing a different op sequence and
            # tripping CheckpointError ("recomputed values have different
            # metadata"). Setting it here makes the side effect replay identically.
            # ``_flex_block_mask`` rides along for the same reason: it is read by
            # UMichSLMAttention.forward and must be present on recompute too. The
            # step cache hands back the identical BlockMask object, so the replay
            # is bitwise.
            #
            # ``_intra_doc_force_sdpa`` is forced on for EVERY member: this compact
            # sequence has length k (< S), but the model set each layer's
            # ``_intra_doc_cu_seqlens`` for the FULL S. An MLA/GQA member left on
            # its varlen path would hand flash_attn_varlen_func the full-S
            # cu_seqlens (last value B*S) against a compact [B*k, ...] buffer and
            # read B*(S-k) rows past the end -> CUDA illegal memory access (real
            # packed data only; synthetic equal-length docs happen to route around
            # it). The compact ``mask`` already encodes intra-doc causality in
            # original-position space, so route every member through SDPA that
            # consumes it. Set inside the callable for checkpoint-replay parity,
            # same as the masks above; restore the prior value in ``finally``.
            prev_force_sdpa = getattr(attn, "_intra_doc_force_sdpa", None)
            has_force = attn is not None and hasattr(attn, "_intra_doc_force_sdpa")
            if has_force:
                attn._intra_doc_force_sdpa = True
            if is_swa:
                attn._mod_external_mask = mask
                if block_mask is not None:
                    attn._flex_block_mask = block_mask
            try:
                return layer(hs, attention_mask=mask, position_ids=pos, use_cache=False)
            finally:
                if has_force:
                    attn._intra_doc_force_sdpa = prev_force_sdpa
                if is_swa:
                    attn._mod_external_mask = None
                    attn._flex_block_mask = None

        ckpt = self.gradient_checkpointing and self.training
        for m in members:
            layer = self.layers[m]
            attn = getattr(layer, "self_attn", None)
            # Registers-SWA layers share the compact-path contract: window over
            # ORIGINAL positions via _mod_external_mask (the register read is
            # position-free, so gather/scatter commutes with it).
            is_swa = isinstance(attn, (SWAWithGlobalsAttention, RegisterReadMixin))
            if is_swa and use_flex:
                # The BlockMask carries the real visibility; the additive mask is
                # never read under flex (dispatch priority), but it must be
                # non-None: _mod_external_mask doubles as the "compact mode"
                # signal in SWAWithGlobalsAttention/RegisterReadMixin. Reuse the
                # already-built causal mask rather than building the windowed one.
                sink = getattr(attn, "_sink_bank", None)
                n_extra = int(sink.tokens.shape[0]) if sink is not None else 0
                bmask = _flex_mask_for(n_extra)
                mask = causal_mask
            else:
                mask = windowed_mask if is_swa else causal_mask
                bmask = None
            if compact_ids is not None and m < self.per_layer_embeddings.num_layers:
                compact = compact + self.per_layer_embeddings.forward_layer(
                    compact_ids, m, prefix_len=num_globals
                )
            if ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    _run_member,
                    layer,
                    attn,
                    is_swa,
                    compact,
                    mask,
                    compact_pos,
                    bmask,
                    use_reentrant=False,
                )
            else:
                out = _run_member(layer, attn, is_swa, compact, mask, compact_pos, bmask)
            compact = out[0]

        # Router-weighted block delta on selected tokens; scatter back over the
        # (bypassed) block input.
        w = torch.gather(router_weights, 1, gather_idx).unsqueeze(-1)  # [B, k, 1]
        if gather_valid is not None:
            # A padded slot points at a real but UNSELECTED position (padding has to
            # index something). Zeroing its gate makes `gated` collapse to
            # compact_in there, so the scatter writes the value that was already at
            # that position — a no-op — instead of a bogus block output.
            w = w * gather_valid.unsqueeze(-1).to(w.dtype)
        gated = compact_in + w * (compact - compact_in)
        new_hidden = hidden_states.clone()
        new_hidden.scatter_(1, idx_h, gated)

        mod_out = MoDOutput(
            hidden_states=new_hidden,
            router_weights=router_weights,
            selected_mask=selected_mask,
            aux_loss=aux_loss,
        )
        return new_hidden, mod_out

    def _has_non_fa2_layers(self) -> bool:
        """Does any attention layer run a kernel other than FlashAttention-2?

        MLA has no FA2 path, and the swa_globals family is SDPA-backed even though
        it registers under the "flash_attention_2" key, so a model can select FA2
        and still hold layers that need a real SDPA mask.
        """
        from .modules.attention import UMichSLMFlashAttention2

        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and not isinstance(attn, UMichSLMFlashAttention2):
                return True
        return False

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
        document_ids: Optional[torch.LongTensor] = None,
        feedback_states: Optional[torch.FloatTensor] = None,
        feedback_shift: bool = True,
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

        # Adopt whatever cache we were handed into an MLADynamicCache for MLA models.
        #
        # This covers the HF `Cache` that `.generate()` always builds, not just a
        # legacy tuple. An MLA layer caches (c_kv, k_rope), which a `DynamicCache`
        # cannot hold correctly -- and in a MIXED model (the swa_globals family is
        # MLA only on its periodic global layers) the plain layers would write
        # their (k, v) into the HF cache while the MLA layers returned tuples,
        # leaving `forward` with an L-tuple mixing cache objects and pairs that
        # `from_legacy_cache` then died unpacking. One cache holds both kinds.
        # Not for the DeltaNet hybrids: their MLA layers live inside the
        # DeltaNetDynamicCache built below (one cache for both layer kinds), and adopting
        # here first crashed .generate() on a legacy DeltaNetCache entry.
        if (
            use_mla
            and not self.use_deltanet
            and past_key_values is not None
            and not isinstance(past_key_values, MLADynamicCache)
        ):
            if isinstance(past_key_values, tuple):
                past_key_values = MLADynamicCache.from_legacy_cache(past_key_values)
            elif isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
                # Empty: generation's freshly built cache, nothing to carry over.
                past_key_values = MLADynamicCache()

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

        # Intra-document masking (W2, LONG_CONTEXT_PLAN.md §4): when packed windows carry a
        # per-token `document_ids` signal, reset RoPE positions per document and (below) block
        # cross-document attention. Only on the initial forward — cached generation operates on
        # single-document continuations.
        intra_doc_active = (
            getattr(self.config, "use_intra_doc_masking", False)
            and document_ids is not None
            and past_length == 0
        )
        if intra_doc_active:
            if document_ids.shape[:2] != (batch_size, seq_length):
                raise ValueError(
                    f"document_ids shape {tuple(document_ids.shape)} does not match "
                    f"(batch={batch_size}, seq_len={seq_length})"
                )
            position_ids = build_intra_doc_position_ids(document_ids)

        # mla_position_jitter (training only): stretch the RoPE distances half the rows
        # see, so global layers meet positions far past the training length.
        _jitter = int(getattr(self.config, "mla_position_jitter", 0) or 0)
        if _jitter > 1 and self.training and past_length == 0:
            from .modules.doc_masking import jitter_position_ids

            position_ids = jitter_position_ids(position_ids, _jitter)

        # Message structure (modules/msg_struct.py): one per-forward context built from
        # input_ids and the final position ids, bound to every MLA layer.
        if self.msg_struct is not None:
            self.msg_struct.begin(
                input_ids, position_ids, document_ids if intra_doc_active else None, past_length
            )

        # Get input embeddings
        if inputs_embeds is None:
            # Validate token IDs are within vocabulary range.
            #
            # ONCE, not every step. `.item()` is a device->host sync that stalls the
            # pipeline right before the embedding, and it fired on every forward of
            # every arch. What it checks is a STATIC property of the tokenizer/model
            # pair, so a run that passes the first batch cannot start failing later;
            # an out-of-range id would also trip a device-side assert in the
            # embedding kernel anyway. Latching after the first successful check
            # keeps the diagnostic (a clear message instead of a CUDA assert) and
            # drops the recurring sync.
            if input_ids is not None and not self._vocab_range_checked:
                max_token_id = input_ids.max().item()
                self._vocab_range_checked = True
                if max_token_id >= self.vocab_size:
                    raise ValueError(
                        f"Token ID {max_token_id} is out of vocabulary range [0, {self.vocab_size}). "
                        f"This indicates a vocabulary size mismatch between the tokenizer and model. "
                        f"Model vocab_size: {self.vocab_size}, but tokenizer produced token ID {max_token_id}. "
                        f"Please ensure the model's vocab_size matches the tokenizer's vocab_size."
                    )
            inputs_embeds = self.embed_tokens(input_ids)

        # Latent feedback: replace the plain embedding with the fusion of this
        # token's embedding and the PREVIOUS position's top-layer state from the
        # previous pass. Positions with no valid predecessor -- position 0 and
        # every document start -- keep the plain embedding; see the module
        # docstring for why that is not optional.
        if self.latent_feedback is not None:
            if feedback_states is not None:
                inputs_embeds = self.latent_feedback(
                    inputs_embeds,
                    feedback_states,
                    document_ids=document_ids if feedback_shift else None,
                    shift=feedback_shift,
                )
            else:
                # Pass 1 has no predecessor states, so there is nothing to fuse
                # -- but it must still hand the stack the SAME SCALE a fused
                # pass does. The fused value is an RMSNorm output (order 1) and
                # a raw embedding is order 0.09, so skipping this made three
                # quarters of training a different input distribution from all
                # of validation. That is what the first three _fbt cells
                # measured: train 2.50-2.61, val 2.59/3.73/5.53.
                inputs_embeds = self.latent_feedback.plain(inputs_embeds)

        # State/prediction separation: carry both streams concatenated along the
        # BATCH dim, state half first. Everything downstream that is per-position
        # and stream-agnostic (position ids, document ids, the causal/varlen
        # mask, the FFN, the norms, checkpointing, per-block compile) then works
        # unchanged at batch 2B.
        pause_active = self.pause_stream is not None and self.pause_enabled and past_length == 0
        if pause_active:
            if self.num_meta_tokens or self.num_global_tokens:
                raise NotImplementedError(
                    "use_pause_stream does not support a meta/global token prefix: "
                    "the prefix is prepended per batch row and would be duplicated "
                    "into the prediction stream. Supported on the dense GQA path "
                    "(qwen3_like) only."
                )
            from .modules.pause_stream import duplicate_for_streams

            # pause_decision_only (P12, modules/msg_struct.py): the prediction stream
            # is read, and the shared FFN blends, only at Harmony decision tokens.
            pause_dec = None
            if getattr(self.config, "pause_decision_only", False):
                from .modules.msg_struct import compute_decision_mask, msg_token_ids

                if input_ids is None:
                    raise ValueError("pause_decision_only needs input_ids")
                pause_dec = compute_decision_mask(
                    input_ids,
                    document_ids if intra_doc_active else None,
                    msg_token_ids(self.config),
                    int(getattr(self.config, "pause_final_tokens", 16)),
                )
            for _layer in self.layers:
                if getattr(_layer, "pause_ffn", None) is not None:
                    _layer.pause_ffn._decision_mask = pause_dec

            # A cache built at batch 2B is unusable (pause has no decode path, see
            # below), and the DeltaNet/MLA pause read refuses one outright.
            use_cache = False
            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    self.pause_stream.initial(
                        batch_size, seq_length, inputs_embeds.dtype, inputs_embeds.device
                    ),
                ],
                dim=0,
            )
            attention_mask = duplicate_for_streams(attention_mask)
            position_ids = duplicate_for_streams(position_ids)
            document_ids = duplicate_for_streams(document_ids)
            batch_size *= 2
        elif self.pause_stream is not None and self.pause_enabled:
            raise NotImplementedError(
                "use_pause_stream has no incremental-decode path: a second attention "
                "call with a live cache would double-append, because "
                "past_key_value.update() runs whenever a cache is passed regardless "
                "of use_cache (attention.py). Training and teacher-forced evaluation "
                "(including the forward-pass-only capability suite) do not need one."
            )

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
        if self.global_tokens is not None and past_length == 0 and self.num_global_tokens > 0:
            globs = self.global_tokens(batch_size, inputs_embeds.dtype, inputs_embeds.device)
            num_globals = globs.shape[1]
            inputs_embeds = torch.cat([globs, inputs_embeds], dim=1)

            if attention_mask is not None:
                gmask = torch.ones(
                    batch_size,
                    num_globals,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
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
        #
        # On a CACHED DECODE step nothing is prepended (num_globals == 0) but the
        # globals are still sitting at the front of the KV cache, so the count
        # the layers need is what the cache holds, not what this step prepended.
        # Clearing it to 0 there made every SWA layer window the globals away
        # from token one of generation onward -- the arch's entire long-range
        # mechanism, gone, with correct-looking logits.
        protected_prefix = num_globals
        if (
            protected_prefix == 0
            and past_length > 0
            and self.global_tokens is not None
            and self.num_global_tokens > 0
        ):
            protected_prefix = int(self.num_global_tokens)

        if protected_prefix > 0:
            for layer in self.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_protected_prefix"):
                    layer.self_attn.num_protected_prefix = protected_prefix
                if hasattr(layer, "num_protected_prefix"):
                    # MoD decoder layer
                    layer.num_protected_prefix = protected_prefix
        else:
            # Make sure leftover values from a previous forward don't leak in
            # (this matters for the SWA layer when later batches use no globals).
            for layer in self.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_protected_prefix"):
                    layer.self_attn.num_protected_prefix = 0
                if hasattr(layer, "num_protected_prefix"):
                    layer.num_protected_prefix = 0

        # Bind the register bank (swa_use_registers) to any registers-SWA layer
        # that doesn't hold it yet. Done at forward time — mirroring the
        # num_protected_prefix channel — so layers inserted at stack transitions
        # get bound too. The bank's parameters live on the model exactly once;
        # layers keep a plain (non-module) reference.
        _bank_export = bool(getattr(self.config, "pathway_bank_export", False))
        if getattr(self, "register_tokens", None) is not None:
            refined = None
            if getattr(self, "register_refiner", None) is not None:
                refined = self.register_refiner(self.register_tokens.tokens)
            # Base bank tensor for this forward — the delta write hook scans
            # from the same tokens the static read uses.
            self._register_base = refined
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None or not hasattr(attn, "bind_register_bank"):
                    continue
                # pathway_bank_export: the bank is read through the stage
                # export instead — per-spine-layer register reads stay unbound.
                if attn._register_bank is None and not _bank_export:
                    attn.bind_register_bank(self.register_tokens)
                # Refresh per-forward stashes (cleared here, not post-forward,
                # so gradient-checkpoint recompute sees the same tensors).
                attn._register_tokens_override = refined
                attn._delta_states = None
                attn._delta_state_docs = None
                attn._delta_token_docs = None

        # swa_scribe: bind the sink bank and clear stale scribe stashes at
        # forward START (checkpoint-recompute-safe, like _flex_doc_ids).
        if getattr(self, "sink_tokens", None) is not None or self.scribe_bank is not None:
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None or not hasattr(attn, "bind_sink_bank"):
                    continue
                if self.sink_tokens is not None and attn._sink_bank is None:
                    attn.bind_sink_bank(self.sink_tokens)
                attn._scribe_states = None
                attn._scribe_doc = None
                attn._scribe_token_docs = None

        # Intra-document masking: extend document_ids over the prepended prefix (globals/meta
        # are per-sequence -> PREFIX_ID, visible across all packed docs) and stash the per-doc
        # reset signal on any DeltaNet layer so its recurrent state resets at boundaries.
        full_doc_ids = None
        if intra_doc_active:
            num_prefix = num_meta + num_globals
            if num_prefix > 0:
                prefix = torch.full(
                    (batch_size, num_prefix),
                    PREFIX_ID,
                    dtype=document_ids.dtype,
                    device=document_ids.device,
                )
                full_doc_ids = torch.cat([prefix, document_ids], dim=1)
            else:
                full_doc_ids = document_ids
            doc_start = build_doc_start_mask(full_doc_ids)
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "_intra_doc_start"):
                    attn._intra_doc_start = doc_start
        else:
            # Clear any leftover intra-doc signals so a masked forward can't leak into a later
            # unmasked one (matters when the same layers are reused across batches).
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                if hasattr(attn, "_intra_doc_start"):
                    attn._intra_doc_start = None
                if hasattr(attn, "_delta_cu_seqlens"):
                    attn._delta_cu_seqlens = None
                if hasattr(attn, "_intra_doc_cu_seqlens"):
                    attn._intra_doc_cu_seqlens = None
                    attn._intra_doc_max_seqlen = None
                    attn._intra_doc_force_sdpa = False
                if hasattr(attn, "_flex_doc_ids"):
                    attn._flex_doc_ids = None
                    attn._flex_doc_cache = None

        # FlexAttention gating for the SWA-with-globals fast path. The BlockMask
        # encodes causal + window + globals (+ docs, below) but CANNOT encode
        # padding, so any real padding in the 2D mask must push those layers back
        # onto the SDPA additive path. One .all() sync per forward, and only when
        # a mask is supplied — the packed training path passes None.
        flex_disable = attention_mask is not None and not bool(attention_mask.all())
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_flex_disable"):
                attn._flex_disable = flex_disable

        # Create attention mask
        attn_implementation = getattr(self.config, "_attn_implementation", "eager")

        # PoPE (Polar Position Embeddings) doubles the Q/K head dimensions, which is
        # incompatible with Flash Attention and SDPA. The attention layers will fall
        # back to eager implementation, so we must prepare the mask accordingly.
        if getattr(self.config, "position_embedding_type", "rope") == "pope":
            if attn_implementation in ("flash_attention_2", "sdpa"):
                attn_implementation = "eager"

        if attn_implementation == "flash_attention_2":
            # Flash Attention 2 handles masking internally, so its own layers never
            # read this mask -- but the layers that CANNOT run FA2 do. MLA and the
            # whole swa_globals family fall back to SDPA internally and hand
            # whatever they are given straight to `F.scaled_dot_product_attention`,
            # which rejects the raw 2-D padding mask outright:
            #   Expected attn_mask dtype to be bool or float ..., got long int
            # That is a PREFILL failure, not a cache one, and it stayed hidden
            # because the A40-grid arms all train with intra-doc masking, which
            # materialises a 4-D additive mask further down and papers over it.
            # Build the SDPA mask exactly when such a layer exists; building it
            # unconditionally would cost the O(S^2) memory that the varlen path on
            # the dense arms exists to avoid.
            if attention_mask is not None and self._has_non_fa2_layers():
                attention_mask_4d = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_length,
                )
            else:
                attention_mask_4d = attention_mask
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

        # Intra-document masking: block cross-document attention. Two mechanisms, chosen so we
        # never pay O(S^2) memory when FlashAttention varlen can avoid it:
        #   - sdpa/eager/MLA/SWA layers consume a block-diagonal ADDITIVE mask folded into the
        #     4D causal mask (composes with the SWA+globals and MLA masks automatically);
        #   - plain FlashAttention-2 layers use the VARLEN path (cu_seqlens) — but only when
        #     there is no per-sequence prefix (globals/meta), which varlen can't share across
        #     documents. With a prefix, those FA2 layers fall back to the sdpa additive path.
        # Shared per-forward BlockMask cache. Declared here (not inside the
        # intra-doc branch) because the MoP lane blocks use it whether or not
        # documents are packed, and gradient-checkpoint recompute must hit the
        # same dict so it reuses the identical mask object.
        step_flex_cache: dict = {}
        # pathway_export_interval: per-forward export/kv reuse across stages.
        step_export_cache: dict = {}

        if intra_doc_active and full_doc_ids is not None:
            from .modules.attention import UMichSLMFlashAttention2
            from .modules.deltanet import GatedDeltaNet
            from .modules.rwkv7 import RWKV7Mixer

            _recurrent = (GatedDeltaNet, RWKV7Mixer)
            prefix_present = (num_meta + num_globals) > 0
            is_fa2 = attn_implementation == "flash_attention_2"

            # Does any layer consume the additive 4D mask? DeltaNet (state reset) and pure-FA2
            # varlen layers (no shared prefix) do not; everything else does.
            need_additive = False
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None or isinstance(attn, _recurrent):
                    continue
                if (
                    is_fa2
                    and not prefix_present
                    and (
                        isinstance(attn, UMichSLMFlashAttention2)
                        or getattr(attn, "_supports_varlen_intra_doc", False)
                    )
                ):
                    # SWA varlen layers AND MLA-flash layers carry intra-doc
                    # masking via cu_seqlens, not a shared additive mask.
                    continue
                need_additive = True

            cu_seqlens = max_seqlen = None
            force_sdpa = False
            if is_fa2:
                if prefix_present:
                    force_sdpa = True
                else:
                    cu_seqlens, max_seqlen = build_batched_cu_seqlens(full_doc_ids)

            # DeltaNet state resets: one shared cu_seqlens for every DeltaNet layer (see
            # GatedDeltaNet._delta_cu_seqlens). The FA2 tensor is reused when it exists;
            # it splits at the same doc/row boundaries as build_doc_start_mask. A prefix
            # run is just one more segment for a recurrent layer, so build it otherwise.
            delta_cu = cu_seqlens
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if isinstance(attn, _recurrent):
                    if delta_cu is None:
                        delta_cu, _ = build_batched_cu_seqlens(full_doc_ids)
                    attn._delta_cu_seqlens = delta_cu

            if need_additive:
                if attention_mask_4d is None or attention_mask_4d.dim() != 4:
                    # sdpa may skip the 4D mask (is_causal fast path); materialize it so the
                    # block-diagonal mask has something to compose with.
                    attention_mask_4d = _prepare_4d_causal_attention_mask(
                        torch.ones(
                            (batch_size, seq_length),
                            dtype=torch.long,
                            device=inputs_embeds.device,
                        ),
                        (batch_size, seq_length),
                        inputs_embeds,
                        past_length,
                    )
                attention_mask_4d = attention_mask_4d + build_intra_doc_additive_mask(
                    full_doc_ids, attention_mask_4d.dtype
                )

            # Doc-aware FlexAttention: hand the SWA layers the full doc-id map and a
            # per-forward cache so the flex fast path can build ONE doc-aware
            # BlockMask per step (shared across layers; checkpoint recompute hits
            # the same dict). Without doc ids the flex branch would drop the doc
            # mask and leak cross-document attention.
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "_intra_doc_cu_seqlens"):
                    attn._intra_doc_cu_seqlens = cu_seqlens
                    attn._intra_doc_max_seqlen = max_seqlen
                    attn._intra_doc_force_sdpa = force_sdpa
                if attn is not None and hasattr(attn, "_flex_doc_ids"):
                    attn._flex_doc_ids = full_doc_ids
                    attn._flex_doc_cache = step_flex_cache

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
        _is_recurrent_cache = isinstance(
            past_key_values, (DeltaNetDynamicCache, GriffinDynamicCache)
        )
        next_decoder_cache = None if _is_recurrent_cache else (() if use_cache else None)

        # MoE router logits (only for MoE models)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.use_moe
        )
        all_router_logits = () if output_router_logits and self.use_moe else None

        # MoD info (only for MoD models during training). MoP emits its
        # cost-aware aux loss through the same channel (F4: one aux path).
        uses_pathways = bool(self._pathway_insert)
        output_mod_info = (self.use_mod or uses_pathways) and self.training
        all_mod_infos = () if output_mod_info else None

        # GWT / MoP: decode-time integration (workspace refresh at chunk
        # boundaries; lane cross-attention over cached shared-stream K/V) is
        # deferred — screening runs use_cache=False, mirroring recurrent depth.
        if use_cache and (self.workspace_trunk is not None or uses_pathways):
            raise RuntimeError(
                "use_workspace_trunk / use_pathways do not support KV caching yet; "
                "run with use_cache=False (screening semantics)."
            )

        # MoP floor-release step counter. Incremented here rather than inside
        # Pathways.forward so gradient-checkpoint recompute can't double-count.
        if uses_pathways and self.training:
            for _stage in self._pathway_insert.values():
                _stage._train_steps += 1

        # Stacked MoP: per-stage (expected_cost, entropy_penalty) parts,
        # combined into ONE global-budget aux after the layer loop.
        pathway_parts: list = []

        # Block-atomic MoD: per-block routing state captured at each block head
        # (block input snapshot + router decision) and consumed at the block tail.
        mod_block_state: Dict[int, tuple] = {}

        # Block-atomic MoD with real FLOP savings runs each block over only its
        # selected tokens (gather/scatter). Enabled for the no-cache path
        # (training / loss eval); the KV-cache/generation path keeps the dense
        # gate fallback below (per-token routing there is a separate problem).
        mod_gather_active = (
            self.use_mod
            and self.mod_atomic_block
            and not use_cache
            and self.mod_block_gate is not None
        )
        mod_consumed_layers: set = set()

        # MoDA: create depth KV cache for this forward pass
        depth_kv_cache = None
        if self.use_moda:
            depth_kv_cache = MoDADepthKVCache(
                max_entries=getattr(self.config, "moda_max_depth_cache", -1),
                depth_kv_every_n=getattr(self.config, "moda_depth_kv_every_n", 1),
                active_layers=self.moda_active_layers,
            )

        # Layer execution schedule. Normally the identity walk; with recurrent
        # depth the core block's indices repeat a sampled number of times
        # (prelude and coda run once). Everything keyed by layer index (PLE,
        # engram, protected prefix) naturally follows the physical layer.
        layer_schedule = range(len(self.layers))
        # Schedule position -> loop boundary k (k = 0..iters; k = iters is the
        # coda's first layer). Only populated when a loop mixer is present.
        loop_bounds: Dict[int, int] = {}
        loop_state: Dict[str, Any] = {}
        if self.use_recurrent_depth:
            if use_cache:
                raise RuntimeError(
                    "use_recurrent_depth does not support KV caching yet: looping the "
                    "core block would append to the same layers' caches every "
                    "iteration. Screening runs use_cache=False (adaptive-depth "
                    "decoding via halting is deferred)."
                )
            from .compute_accounting import recurrent_block_size
            from .modules.recurrent_depth import build_recurrent_schedule, sample_recursion_count

            iters = sample_recursion_count(
                int(getattr(self.config, "recurrent_min_iters", 1)),
                int(getattr(self.config, "recurrent_max_iters", 1)),
                str(getattr(self.config, "recurrent_sample_dist", "uniform")),
                training=self.training,
                force=self._recurrent_force_iters,
                eval_iters=int(getattr(self.config, "recurrent_eval_iters", 0)),
            )
            self._last_recurrent_iters = iters
            _bs = recurrent_block_size(self.config)
            layer_schedule = build_recurrent_schedule(len(self.layers), _bs, iters)
            if self.recurrent_mixer is not None:
                _core = len(self.layers) - 2 * _bs
                loop_bounds = {_bs + k * _core: k for k in range(iters + 1)}

        # Forward through layers
        for pos, idx in enumerate(layer_schedule):
            # Loop mixers run first at a boundary, so nothing below (and no
            # `continue`) can skip one. The prefix rows, if any, are mixed with
            # the content: they loop through the same core.
            if pos in loop_bounds:
                hidden_states = self.recurrent_mixer.boundary(
                    loop_state, loop_bounds[pos], hidden_states, iters
                )
            decoder_layer = self.layers[idx]
            # Layers already run as part of a gathered MoD block are skipped
            # (the block head consumed the whole block in one gather/scatter).
            if idx in mod_consumed_layers:
                continue

            # GWT: between layers idx-1 and idx, chunks read the live
            # workspace and the trunk writes it forward. Content tokens only;
            # the static globals/meta prefix passes through untouched. Runs
            # outside per-layer gradient checkpointing (the workspace is
            # [B, slots, d] — small; `trunk_detach_chunks` bounds chunk BPTT).
            # swa_scribe: at the write boundary, commit per-chunk scribe states
            # from the current hidden states (content only — this arch has no
            # prefix) and stash them on every SWA layer ABOVE the boundary.
            # Runs OUTSIDE per-layer gradient checkpointing (trunk precedent).
            if self.scribe_bank is not None and idx == self.scribe_write_idx:
                scribe_states, scribe_doc = self.scribe_bank(
                    hidden_states, document_ids if intra_doc_active else None
                )
                for later_idx in range(idx + 1, len(self.layers)):
                    attn = getattr(self.layers[later_idx], "self_attn", None)
                    if attn is not None and hasattr(attn, "bind_sink_bank"):
                        attn._scribe_states = scribe_states
                        attn._scribe_doc = scribe_doc
                        attn._scribe_token_docs = document_ids if intra_doc_active else None

            # swa_deltabank: at the write boundary, scan content chunks into
            # cumulative register-bank states and stash them on every register-
            # reading layer ABOVE the boundary (layers below keep the static
            # read). Content only — the prefix passes through untouched. Runs
            # OUTSIDE per-layer gradient checkpointing (trunk/scribe precedent).
            if self.delta_bank is not None and idx == self.delta_write_idx:
                prefix_len = num_meta + num_globals
                content = hidden_states[:, prefix_len:] if prefix_len else hidden_states
                base0 = getattr(self, "_register_base", None)
                if base0 is None:
                    base0 = self.register_tokens.tokens
                delta_states, delta_docs = self.delta_bank(
                    base0, content, document_ids if intra_doc_active else None
                )
                for later_idx in range(idx + 1, len(self.layers)):
                    attn = getattr(self.layers[later_idx], "self_attn", None)
                    if attn is not None and hasattr(attn, "bind_register_bank"):
                        attn._delta_states = delta_states
                        attn._delta_state_docs = delta_docs
                        attn._delta_token_docs = document_ids if intra_doc_active else None
                        attn._delta_prefix_len = prefix_len
                # Stash for the bank-reconstruction aux loss, which is computed
                # by the ForCausalLM wrapper (it needs input_ids and the head).
                self._recon_states = delta_states
                self._recon_state_docs = delta_docs
                self._recon_token_docs = document_ids if intra_doc_active else None
                self._recon_prefix_len = prefix_len

            if self.delta_memories is not None and idx in self._delta_mem_at:
                prefix_len = num_meta + num_globals
                content = hidden_states[:, prefix_len:] if prefix_len else hidden_states
                _mem_mod = self.delta_memories[self._delta_mem_at[idx]]
                _mem_docs = document_ids if intra_doc_active else None
                if (
                    self.gradient_checkpointing
                    and self.training
                    and getattr(self.config, "delta_mem_checkpoint", True)
                ):
                    # CHECKPOINT THE SIDE BRANCH TOO (2026-08-21). This module
                    # runs at MODEL level between decoder layers, and
                    # gradient_checkpointing_enable wraps only the decoder-layer
                    # calls -- so every intermediate here was retained for
                    # backward. Measured at B=16/S=4096/deep80: 3366.8 MB saved
                    # per module against 234.9 MB when checkpointed, and 234.9 MB
                    # is exactly the [16,4096,896] input, i.e. checkpointing
                    # retains the input and nothing else. That is the whole of
                    # the arm's memory premium (gate V: mat_ffn 41.1 GB ->
                    # +delta 63.7 GB at b16, which capped it at batch 16 while
                    # mat_ffn alone reaches 24+). The recompute is cheap now that
                    # the module is compiled and state-factored. Costs one extra
                    # forward of THIS module per step; `delta_mem_checkpoint=false`
                    # turns it off so the memory/time trade can be A/B'd.
                    mem = torch.utils.checkpoint.checkpoint(
                        _mem_mod, content, _mem_docs, use_reentrant=False
                    )
                else:
                    mem = _mem_mod(content, doc_ids=_mem_docs)
                content = content + mem
                if prefix_len:
                    hidden_states = torch.cat([hidden_states[:, :prefix_len], content], dim=1)
                else:
                    hidden_states = content

            if self.workspace_trunk is not None and idx == self.trunk_insert_idx:
                prefix_len = num_meta + num_globals
                content = hidden_states[:, prefix_len:] if prefix_len else hidden_states
                content = self.workspace_trunk(
                    content,
                    doc_ids=document_ids if intra_doc_active else None,
                    force_iters=self._trunk_force_iters,
                )
                if prefix_len:
                    hidden_states = torch.cat([hidden_states[:, :prefix_len], content], dim=1)
                else:
                    hidden_states = content
                self._last_trunk_iters = self.workspace_trunk._last_iters

            # MoP: parallel depth lanes between shared decoder layers. Single
            # stage emits its own aux (aux_mode="local", unchanged); stacked
            # stages emit (cost, entropy) parts for one global budget.
            _stage = self._pathway_insert.get(idx)
            if _stage is not None:
                prefix_len = num_meta + num_globals
                _aux_mode = "local" if self.pathway_stages is None else "parts"
                # doc_ids/flex_cache let the lane blocks rebuild the causal ∧
                # intra-doc predicate as a block-sparse BlockMask instead of
                # reading the dense [B,1,S,S] additive mask once per lane block.
                # mask_is_causal_doc certifies that attention_mask_4d carries
                # nothing but causality (+ the intra-doc blocks), which holds iff
                # the caller supplied no per-token padding mask. Packed training
                # never does; a ragged batch would, and must stay on SDPA.
                _pw_mask = pathway_attention_mask(
                    attention_mask_4d,
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_length,
                )
                _pw_kw = dict(
                    num_prefix=prefix_len,
                    aux_mode=_aux_mode,
                    doc_ids=full_doc_ids if intra_doc_active else None,
                    flex_cache=step_flex_cache,
                    mask_is_causal_doc=attention_mask is None or _pw_mask is None,
                    export_cache=step_export_cache,
                )
                if _bank_export and getattr(self, "register_tokens", None) is not None:
                    # Banks-in-export: the (refined) register rows join the
                    # stage's shared-KV export; spine layers no longer read.
                    _base = getattr(self, "_register_base", None)
                    _pw_kw["bank_rows"] = (
                        _base if _base is not None else self.register_tokens.tokens
                    )
                # Selective checkpointing: the backbone layers below still
                # checkpoint, but MoP stages are exempt by default because
                # gather_lanes + the compact merge shrank their activations to
                # roughly the routed fraction, making the recompute a poor trade.
                if (
                    self.gradient_checkpointing
                    and self.training
                    and getattr(self.config, "pathway_checkpoint", False)
                ):
                    pw_fn = partial(_stage.__call__, **_pw_kw)
                    hidden_states, pathway_aux = torch.utils.checkpoint.checkpoint(
                        pw_fn, hidden_states, _pw_mask, use_reentrant=False
                    )
                else:
                    hidden_states, pathway_aux = _stage(hidden_states, _pw_mask, **_pw_kw)
                if pathway_aux is not None:
                    if _aux_mode == "parts":
                        pathway_parts.append(pathway_aux)
                    elif all_mod_infos is not None:
                        all_mod_infos += (MoDOutput(aux_loss=pathway_aux),)

            # P1 dual_stream: wide slow block over a causally selected subset.
            if self._slow_stream is not None and str(idx) in self._slow_stream:
                _bank = None
                _rt = getattr(self, "register_tokens", None)
                if _rt is not None:
                    _bank = getattr(_rt, "tokens", None)
                hidden_states = self._slow_stream[str(idx)](
                    hidden_states,
                    bank=_bank,
                    doc_ids=full_doc_ids if intra_doc_active else None,
                )

            # S1 hourglass: pool -> deep middle stack -> causally-shifted unpool.
            if self._hourglass is not None and idx == self._hourglass_at:
                hidden_states = self._hourglass(
                    hidden_states,
                    position_ids=position_ids,
                    doc_ids=full_doc_ids if intra_doc_active else None,
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Block-atomic MoD (gather/scatter): at a block head, run the entire
            # block over only its selected tokens, then skip its member layers.
            if mod_gather_active and idx in self._mod_head_to_members:
                members = self._mod_head_to_members[idx]
                new_hidden, mod_out = self._run_gathered_mod_block(
                    members,
                    hidden_states,
                    position_ids,
                    num_globals,
                    input_ids=input_ids,
                    doc_ids=full_doc_ids if intra_doc_active else None,
                    flex_cache=step_flex_cache,
                )
                if new_hidden is not None:
                    hidden_states = new_hidden
                    if all_mod_infos is not None and mod_out is not None:
                        all_mod_infos += (mod_out,)
                    mod_consumed_layers.update(members)
                    continue
                # Ragged selection -> fall through and run the head densely; the
                # gate head/tail path below handles it.

            # Per-Layer Embeddings: add this layer's token-specific signal to its
            # input. Aligns to content positions (zero-padded over any prepended
            # prefix, e.g. SWA globals / meta tokens). Skipped when only
            # inputs_embeds is provided (no token ids to look up).
            if (
                self.per_layer_embeddings is not None
                and input_ids is not None
                and idx < self.per_layer_embeddings.num_layers
            ):
                prefix_len = hidden_states.shape[1] - input_ids.shape[1]
                hidden_states = hidden_states + self.per_layer_embeddings.forward_layer(
                    input_ids, idx, prefix_len=max(0, prefix_len)
                )

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

            # Check if this is a MoD layer (legacy per-layer path only)
            layer_is_mod = (
                self.use_mod
                and not self.mod_atomic_block
                and not self.mod_mlp_only
                and is_mod_layer(idx, self.mod_every_n_layers)
            )
            # MoD-MLP layers: per-layer variant owns its router and emits its
            # own MoDOutput; block-atomic members consume the head decision.
            layer_is_mod_mlp = isinstance(decoder_layer, UMichSLMMoDMLPDecoderLayer)
            layer_emits_mod_info = layer_is_mod or (layer_is_mod_mlp and not self.mod_atomic_block)

            # Block-atomic MoD: at a block head, compute the routing decision
            # from the block's input (which every layer in the block will share)
            # and snapshot the input so unselected tokens can bypass the whole
            # block when it is gated at the tail. Done outside the per-layer
            # gradient checkpoint so the router weights/mask are plain graph
            # inputs to the block's layers.
            if self.mod_atomic_block and idx in self._mod_head_to_members:
                _blk_router = self.mod_block_routers[str(idx)]
                _blk_w, _blk_sel, _blk_aux = _blk_router(
                    hidden_states,
                    use_cache=use_cache,
                    force_route_prefix=num_globals,
                )
                mod_block_state[idx] = (hidden_states, _blk_w, _blk_sel, _blk_aux)

            # Block-atomic MoD-MLP: hand the block head's routing decision to
            # this member layer (which owns no router of its own). Must come
            # after the head-state store above so the head layer itself sees it.
            mod_routing = None
            if layer_is_mod_mlp and self.mod_atomic_block:
                _mlp_head = self._mod_layer_to_head.get(idx)
                if _mlp_head is not None and _mlp_head in mod_block_state:
                    _, _rw, _rsel, _ = mod_block_state[_mlp_head]
                    mod_routing = (_rw, _rsel)

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
                if layer_emits_mod_info:
                    ckpt_kwargs["output_mod_info"] = output_mod_info
                if mod_routing is not None:
                    # Router ran outside the checkpoint, so the decision tensors
                    # are plain graph inputs that replay identically on recompute.
                    ckpt_kwargs["mod_routing"] = mod_routing
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
                elif layer_is_mod or layer_emits_mod_info:
                    # MoD / per-layer MoD-MLP layers accept output_mod_info
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
                elif mod_routing is not None:
                    # Block-atomic MoD-MLP member: consume the head decision.
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask_4d,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mod_routing=mod_routing,
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
            if self.msg_struct is not None and self.msg_struct.writer is not None and (
                idx == self.msg_struct.source_layer
            ):
                self.msg_struct.write(
                    hidden_states, checkpoint=self.gradient_checkpointing and self.training
                )

            # Block-atomic MoD gate: at the block tail, unselected tokens are
            # reset to the block input (they bypassed all block_size layers) and
            # selected tokens keep the router-weighted block delta. This makes the
            # block the atomic routing unit, matching the stacking-plan design.
            if self.mod_atomic_block and idx in self._mod_tail_to_head:
                _head = self._mod_tail_to_head[idx]
                _blk_in, _blk_w, _blk_sel, _blk_aux = mod_block_state.pop(_head)
                # MoD-MLP: FFN gating already happened inside each member layer
                # (attention deltas must survive for declined tokens), so no
                # block-boundary gate — only the head router's aux loss and
                # routing info are emitted here.
                if not self.mod_mlp_only:
                    hidden_states = self.mod_block_gate(
                        _blk_in, hidden_states - _blk_in, _blk_w, _blk_sel
                    )
                if all_mod_infos is not None:
                    all_mod_infos += (
                        MoDOutput(
                            hidden_states=hidden_states,
                            router_weights=_blk_w,
                            selected_mask=_blk_sel,
                            aux_loss=_blk_aux,
                        ),
                    )

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
            if all_mod_infos is not None and layer_emits_mod_info:
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

        if pause_active:
            # lm_head reads the PREDICTION stream. The state stream's job was to
            # summarize context into the K/V the prediction stream read, and it is
            # dropped here. Earlier entries of all_hidden_states still carry both
            # halves at batch 2B -- a diagnostic, not a path anything ranks on.
            from .modules.pause_stream import split_streams

            _state, hidden_states = split_streams(hidden_states)
            if pause_dec is not None:
                # P12: the parent's (state) readout off the decision tokens.
                hidden_states = torch.where(pause_dec.unsqueeze(-1), hidden_states, _state)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # A plain HF `Cache` is updated IN PLACE by every layer, so each layer
        # appends the same object and `next_decoder_cache` comes out as an
        # L-tuple of identical caches. `prepare_inputs_for_generation` then reads
        # `past_key_values[0]`, finds a Cache where it expects a (k, v) pair,
        # falls through to `past_length = 0`, and re-feeds the WHOLE sequence
        # every step into a cache that already holds it -- generation that does
        # not raise but silently attends over duplicated keys.
        #
        # MLA is excluded: its layers cache (c_kv, k_rope), which a plain HF Cache
        # cannot hold. An MLA model is adopted into an MLADynamicCache on the way
        # in (see above); that is deliberately NOT a `Cache` subclass, so the
        # decoder loop hands each layer its own pair and every layer -- MLA and
        # plain alike -- returns one, which the conversion below reassembles.
        if use_cache and not use_mla and isinstance(past_key_values, Cache):
            next_cache = past_key_values

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

        # Reassemble the per-layer pairs into the cache HF generation carries.
        # In a MIXED model those pairs are not all the same KIND -- the global
        # layers hand back MLA's (c_kv, k_rope) and the rest an ordinary (k, v)
        # -- which is what used to raise `too many values to unpack` here.
        if use_cache and use_mla and next_cache is not None and isinstance(next_cache, tuple):
            next_cache = MLADynamicCache.from_legacy_cache(next_cache)

        # Stacked MoP: one GLOBAL budget across stages — the mean per-stage
        # expected cost balances to pathway_budget, so stages may specialize
        # (one mostly express, another mostly deep) while the total holds.
        if pathway_parts and all_mod_infos is not None:
            _ref = next(iter(self._pathway_insert.values()))
            _mean_cost = sum(c for c, _ in pathway_parts) / len(pathway_parts)
            _max_depth = float(max(_ref.depths))
            _stack_aux = _ref.aux_weight * ((_mean_cost - _ref.budget) / _max_depth) ** 2
            if _ref.entropy_weight > 0:
                _stack_aux = _stack_aux + _ref.entropy_weight * (
                    sum(e for _, e in pathway_parts) / len(pathway_parts)
                )
            all_mod_infos += (MoDOutput(aux_loss=_stack_aux.to(hidden_states.dtype)),)

        # W1 `mat_ffn`: the per-token nested-width routers publish their
        # cost-aware aux on the SAME channel MoD/MoP use, so the width budget is
        # actually enforced by the optimizer instead of merely being reported.
        # Without this the router is free to send every token to the widest tier
        # and the arm silently becomes a dense model with extra indexing.
        if getattr(self.config, "use_mat_ffn", False):
            _mat_mlps = getattr(self, "_mat_ffn_mlps", None)
            if _mat_mlps is None:
                from .modules.swiglu import RoutedMatFormerSwiGLUMLP

                _mat_mlps = [m for m in self.modules() if isinstance(m, RoutedMatFormerSwiGLUMLP)]
                object.__setattr__(self, "_mat_ffn_mlps", _mat_mlps)
            _auxes = [m._last_aux for m in _mat_mlps if m._last_aux is not None]
            if _auxes:
                _ma = torch.stack([a.reshape(()) for a in _auxes]).mean()
                if all_mod_infos is None:
                    all_mod_infos = ()
                all_mod_infos += (MoDOutput(aux_loss=_ma.to(hidden_states.dtype)),)

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

        # Return MoD-specific output if using MoD (or MoP, which shares the
        # mod_aux_loss channel — F4).
        if self.use_mod or self._pathway_insert:
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
        # Phased MTP (training.mtp_phase_start_frac): the MTP loss is only computed
        # while this is True. A plain bool flipped by train.py, like pause phasing.
        self.mtp_enabled = True
        if self.use_mtp:
            self.mtp = build_mtp_module(
                config=config,
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head if getattr(config, "mtp_share_head", True) else None,
            )
        else:
            self.mtp = None

        # Auxiliary training objectives (both training-only, discarded at
        # inference). Neither changes the model's forward computation — they
        # attach extra losses to the SAME final hidden states / bank states.
        self.bag_lookahead = None
        if getattr(config, "use_bag_lookahead", False):
            from .modules.bag_lookahead import BagLookaheadHead

            self.bag_lookahead = BagLookaheadHead(
                config, lm_head=self.lm_head if getattr(config, "bag_share_head", True) else None
            )
        self.bank_recon = None
        if getattr(config, "use_bank_reconstruction", False):
            from .modules.bank_reconstruction import BankReconstructionHead

            if not getattr(config, "use_delta_bank", False):
                raise ValueError(
                    "use_bank_reconstruction requires a WRITTEN bank (use_delta_bank=True). "
                    "With a static register bank the state is identical for every chunk, so "
                    "the reconstruction target is unpredictable by construction."
                )
            self.bank_recon = BankReconstructionHead(config, lm_head=self.lm_head)
        # P3 failed-call predictor (modules/agentic_heads.py): a scalar head on the
        # final hidden state at each <|call|> token. Trained only through
        # ``token_targets`` (torchtitan path); inert otherwise.
        self.errhead = None
        if getattr(config, "use_errhead", False):
            from .modules.agentic_heads import ErrorHead

            self.errhead = ErrorHead(config.hidden_size)

        # MTP shares the input embedding (and optionally the LM head) with the
        # base model by holding references to the same modules. Declare those
        # aliases as tied weights so ``save_pretrained``'s safetensors writer
        # knows the shared storage is intentional — it otherwise raises
        # ("shared tensors ... not properly defined"). Detected structurally so
        # it tracks whatever ``build_mtp_module`` actually shared (the Gloeckle
        # variant keeps independent heads even when ``mtp_share_head=True``).
        # Instance-level so the empty class attr (used by non-MTP / stacking
        # presets) is left untouched.
        if self.mtp is not None:
            tied: dict = {}
            if getattr(self.mtp, "embed_tokens", None) is self.model.embed_tokens:
                tied["mtp.embed_tokens.weight"] = "model.embed_tokens.weight"
            if getattr(self.mtp, "_shared_lm_head", None) is self.lm_head:
                tied["mtp._shared_lm_head.weight"] = "lm_head.weight"
            if tied:
                self._tied_weights_keys = tied

        # Mixture of Depths
        self.use_mod = getattr(config, "use_mod", False)

        # Catch-all for *structural* weight sharing introduced by a module (not
        # by an explicit tie_weights call). MUST run *before* post_init:
        # transformers v5 snapshots the expanded mapping into
        # ``self.all_tied_weights_keys`` during post_init, and that snapshot is
        # what from_pretrained consults to decide which keys may be absent from
        # the checkpoint. Declaring afterwards saves fine but loads WRONG — the
        # aliases count as missing, get re-initialized, and because they share
        # storage with the source that re-init silently overwrites the weights
        # that were correctly loaded.
        self._declare_shared_storage_as_tied()

        # Initialize weights and apply final processing
        self.post_init()

    def _declare_shared_storage_as_tied(self) -> None:
        """Declare every parameter that aliases another one's storage.

        ``save_pretrained(safe_serialization=True)`` aborts with "shared tensors
        ... which are not properly defined" whenever two parameter names point at
        one storage without an entry in ``_tied_weights_keys``. Modules that share
        weights *structurally* — rather than via an explicit ``tie_weights()`` —
        therefore cannot checkpoint at all. That killed all four ``mop_rec`` /
        ``mop_stack_shared`` cells on 2026-07-19: they trained fine and then died
        on their first checkpoint write, producing zero results.

        Rather than hand-maintaining a list per arch, detect it: walk the
        parameters keeping duplicates (``named_parameters`` de-duplicates shared
        storage by default, which would hide exactly what we are looking for) and
        alias every repeat occurrence to the first name seen for that storage.
        Merged into any existing declaration, so the MTP and
        ``tie_word_embeddings`` aliases above are preserved.
        """
        first_for_storage: dict = {}
        aliases: dict = {}
        for name, param in self.named_parameters(remove_duplicate=False):
            key = param.data_ptr()
            if key in first_for_storage:
                aliases[name] = first_for_storage[key]
            else:
                first_for_storage[key] = name
        # Kept separately from ``_tied_weights_keys`` because these aliases must
        # survive ``tie_word_embeddings=False`` — see get_expanded_tied_weights_keys.
        self._structural_tied_weights_keys = aliases
        if aliases:
            self._tied_weights_keys = {**dict(self._tied_weights_keys or {}), **aliases}

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        """Expanded ``{target: source}`` tied mapping, including structural aliases.

        transformers v5 returns an EMPTY mapping whenever
        ``config.tie_word_embeddings`` is False — it assumes the only reason to tie
        anything is the embedding/lm_head pair. Every sweep preset sets that flag
        False (the stacking presets require independent matrices), so weight sharing
        that comes from the module graph rather than from the config — the recursive
        ``mop_rec`` lane stack, the stage-shared ``mop_stack_shared`` lanes — would be
        dropped from ``all_tied_weights_keys`` and thus invisible to from_pretrained.
        Re-add them unconditionally so the aliases are never treated as missing (and
        therefore never re-initialized over the loaded source tensor).
        """
        mapping = dict(super().get_expanded_tied_weights_keys(all_submodels=all_submodels) or {})
        mapping.update(getattr(self, "_structural_tied_weights_keys", None) or {})
        return mapping

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

    def tie_weights(self, *args, **kwargs):
        """Tie the LM head to the input embeddings iff requested by config.

        The class-level ``_tied_weights_keys`` is empty so HF never *implicitly*
        ties during save/load (the stacking presets require independent input
        and output matrices). But when a config explicitly sets
        ``tie_word_embeddings=True`` (e.g. the MobileLLM presets), we honor it by
        sharing the weight here. Called by ``post_init()`` and ``from_pretrained``.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            output = self.get_output_embeddings()
            inp = self.get_input_embeddings()
            if output is not None and inp is not None:
                # Share the parameter directly (version-independent; some
                # transformers releases drop ``_tie_or_clone_weights``). Input
                # embedding (vocab, hidden) and lm_head weight (vocab, hidden)
                # have matching shape.
                output.weight = inp.weight
                # Declare the alias instance-level. Sharing the storage without
                # telling HF makes ``save_pretrained`` abort on the safetensors
                # write ("shared tensors ... which are not properly defined"),
                # because ``remove_tied_weights_from_state_dict`` finds aliased
                # storage but no declared target. Merge rather than overwrite so
                # an MTP model that also ties keeps its own aliases.
                self._tied_weights_keys = {
                    **dict(self._tied_weights_keys or {}),
                    "lm_head.weight": "model.embed_tokens.weight",
                }

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

        # `total_count.item()` here was a device->host SYNC on every forward, at the
        # end of the graph where the backward has not been queued yet -- so the whole
        # pipeline drains to answer a question about a degenerate batch that a
        # pretraining run never has (every position ignored). `clamp` answers it on
        # device: with no valid targets `cross_entropy(reduction="sum")` is exactly
        # 0.0, so 0/1 reproduces the old return value, and the grad path is if
        # anything better (it runs through lm_head instead of only hidden_states).
        return total_loss / total_count.clamp(min=1.0)

    def _soft_mtp_route(self, hidden_states, labels, mtp_output) -> torch.Tensor:
        """Soft-MTP gradient routing (arXiv 2610.38331, Algorithm 1).

        MTP depth d at position i predicts labels[i+d+2], the same target as the
        NTP head at position j = i+d+1, through the same lm_head W. The MTP CE
        gradient at h^d_i is W^T(p - y) = W^T(p - q) + W^T(q - y): a KL term toward
        the model's own NTP distribution q, plus the residual W^T(q - y), which is
        the NTP gradient at h_j. Returning h_j + s * (sg(h^d_i) - h^d_i) leaves
        every value unchanged but sends -s * dL_ntp/dh_j into h^d_i, cancelling the
        residual, so the MTP heads and trunk train on KL(sg(q) || p) alone. lm_head
        sees h_j and h^d_i unchanged and keeps both CE gradients.

        s undoes lm_loss's 1/N_0 mean and applies depth d's weight in mtp_loss
        (loss_weight / D, mean over its N_d targets). Both losses mask the same
        label positions, so the counts differ only by the shorter MTP range.
        """
        seq_len = hidden_states.shape[1]
        n_ntp = (labels[:, 1:] != -100).sum().clamp(min=1).float()
        depths = len(mtp_output.hidden_per_depth)
        routed = hidden_states
        for d, (h_mtp, n_mtp) in enumerate(
            zip(mtp_output.hidden_per_depth, mtp_output.targets_per_depth)
        ):
            scale = self.mtp.loss_weight * n_ntp / (depths * n_mtp.clamp(min=1).float())
            # Exactly zero forward; the scale is applied in fp32 on the way back.
            zero = ((h_mtp.detach().float() - h_mtp.float()) * scale).to(hidden_states.dtype)
            start = d + 1
            routed = routed + F.pad(zero, (0, 0, start, seq_len - start - h_mtp.shape[1]))
        return routed

    def set_mtp_enabled(self, enabled: bool) -> bool:
        """Turn the MTP auxiliary loss on or off mid-run (``mtp_phase_start_frac``).

        The MTP heads simply receive no gradient while it is off, so the switch
        carries every other piece of training state over. Returns the flag set
        (False when the model has no MTP module).
        """
        if self.mtp is None:
            return False
        self.mtp_enabled = bool(enabled)
        return self.mtp_enabled

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
        document_ids: Optional[torch.LongTensor] = None,
        output_logits: Optional[bool] = None,
        logits_to_keep: int = 0,
        token_targets: Optional[Dict[str, Any]] = None,
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

            ``logits`` policy: when ``labels`` is set the loss is computed with a
            chunked cross-entropy and the full ``(batch, seq, vocab)`` tensor is
            never materialized, so ``logits`` comes back ``None``. That is a real
            memory constraint, not a quirk — at the sweep's training shape the
            tensor is 32 x 2048 x 56000 x 4B = 14.7 GB in fp32, which OOMs an
            A6000 on its own. Pass ``output_logits=True`` to get logits *and* the
            loss in one pass when you can afford it (evaluation, small models,
            analysis); the default stays memory-safe. Calling without ``labels``
            always returns logits. ``logits_to_keep=k > 0`` (HF convention)
            projects only the last ``k`` positions when ``labels`` is unset: a
            last-position probe at 32K otherwise builds 15 GiB of fp32 logits.
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

        # Token-Superposition Training (TST) phase (i): if active, mean-pool
        # contiguous bags of `s` token embeddings BEFORE the transformer stack,
        # so the model processes L/s "s-tokens". Recovery (phase ii) flips
        # `tst_active` off via TSTController, and this branch is a no-op.
        tst_active = bool(getattr(self.config, "tst_active", False))
        tst_s = int(getattr(self.config, "tst_bag_size", 1))
        tst_in_use = tst_active and tst_s > 1 and labels is not None
        if tst_in_use:
            if use_cache:
                raise RuntimeError("TST does not support KV cache / generation (phase i).")
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("TST requires input_ids or inputs_embeds.")
                inputs_embeds = self.model.embed_tokens(input_ids)
            if inputs_embeds.shape[1] % tst_s != 0:
                raise ValueError(
                    f"TST requires sequence length divisible by bag size {tst_s}; "
                    f"got {inputs_embeds.shape[1]}."
                )
            inputs_embeds = bag_embeddings(inputs_embeds, tst_s)
            input_ids = None  # passing embeds instead
            attention_mask = None  # full attention over compressed length
            position_ids = None

        # Latent feedback (arXiv 2608.08888): run the warm-up passes, then fall
        # through to the ordinary body for the FINAL pass, so MTP, the MoE aux
        # loss, the return types and logits are all untouched.
        #
        # Pass 1 sees plain embeddings; pass k sees pass k-1's top-layer states
        # shifted one position right and fused in. Gradients are NOT detached --
        # a later pass's loss backpropagates into the earlier pass, which is what
        # makes the earlier passes an auxiliary objective rather than a stale
        # input. Teacher forcing keeps every pass parallel over positions.
        feedback_states = None
        feedback_pass_losses: List[torch.Tensor] = []
        if self.model.latent_feedback is not None and past_key_values is None:
            from .modules.latent_feedback import passes_for_index

            if self.training:
                # Pass count from the MIXTURE, drawn deterministically from a
                # counter so every DDP rank agrees (a per-rank draw would have
                # ranks running different collective counts and hang the job).
                passes = passes_for_index(
                    self.model._feedback_fwd_index,
                    getattr(self.config, "latent_feedback_pass_mix", None) or (1.0,),
                    int(getattr(self.config, "latent_feedback_seed", 0)),
                )
                self.model._feedback_fwd_index += 1
            else:
                passes = max(1, int(getattr(self.config, "latent_feedback_eval_passes", 2)))
            for _ in range(passes - 1):
                warm = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    output_router_logits=False,
                    return_dict=True,
                    document_ids=document_ids,
                    feedback_states=feedback_states,
                )
                feedback_states = warm[0]
                if labels is not None:
                    feedback_pass_losses.append(
                        self._compute_shifted_lm_loss_chunked(feedback_states, labels)
                    )

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
            document_ids=document_ids,
            feedback_states=feedback_states,
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
            if tst_in_use:
                # Multi-hot CE on next bag-of-tokens. hidden_states has length L/s;
                # produce one logit row per s-token, score against the next bag.
                shifted = shift_labels_for_tst(labels, tst_s, pad_id=-100)
                logits_tst = self.lm_head(hidden_states)
                weighting = getattr(self.config, "tst_loss_weighting", "uniform")
                chunk = max(1, int(getattr(self.config, "lm_loss_chunk_size", 512)))
                lm_loss = multihot_ce(
                    logits_tst.float(),
                    shifted,
                    s=tst_s,
                    weighting=weighting,
                    ignore_index=-100,
                    chunk_size=chunk,
                )
                loss = lm_loss
                # MTP/MoD/MoE aux losses are orthogonal to TST and out of scope for
                # phase (i). MoE router aux loss is still added below if router_logits
                # are present, since it has no dependency on labels alignment.
                if (
                    use_moe
                    and hasattr(outputs, "router_logits")
                    and outputs.router_logits is not None
                ):
                    aux_loss = self._compute_moe_aux_loss(outputs.router_logits)
                    if aux_loss is not None:
                        loss = loss + aux_loss
                # Skip the rest of the standard-NTP loss path.
                if not return_dict:
                    output = (None,) + outputs[1:]
                    return (loss,) + output
                if self.use_mtp:
                    router_logits = getattr(outputs, "router_logits", None) if use_moe else None
                    return CausalLMOutputWithMTP(
                        loss=loss,
                        lm_loss=lm_loss,
                        mtp_loss=None,
                        aux_loss=aux_loss,
                        logits=None,
                        past_key_values=outputs.past_key_values,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                        router_logits=router_logits,
                        mtp_logits=None,
                    )
                if use_moe:
                    router_logits = getattr(outputs, "router_logits", None)
                    return MoECausalLMOutputWithPast(
                        loss=loss,
                        aux_loss=aux_loss,
                        logits=None,
                        past_key_values=outputs.past_key_values,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                        router_logits=router_logits,
                    )
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=None,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            # Multi-Token Prediction (training only). MTP requires input_ids for the
            # embedding lookup; skip if using inputs_embeds. It runs BEFORE the NTP
            # loss because Soft-MTP routes the NTP gradient into the MTP hidden states.
            mtp_output = None
            if (
                self.use_mtp
                and self.mtp is not None
                and self.mtp_enabled
                and self.training
                and input_ids is not None
            ):
                soft = bool(getattr(self.config, "mtp_soft", False))
                if soft and (token_targets or feedback_pass_losses):
                    # The routing scale assumes lm_loss is the plain mean CE with
                    # coefficient 1 in the total; role weights and pass averaging break it.
                    raise ValueError(
                        "mtp_soft needs the plain mean NTP loss; disable [token_loss] / "
                        "latent feedback, or set model mtp_soft=False"
                    )
                mtp_output = self.mtp(
                    hidden_states=hidden_states,
                    input_ids_or_labels=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_logits=False,  # Don't store logits during training to save memory
                    return_hidden=soft,
                )
            ntp_hidden = hidden_states
            if mtp_output is not None and mtp_output.hidden_per_depth:
                ntp_hidden = self._soft_mtp_route(hidden_states, labels, mtp_output)

            # Per-token annotations (titan/harmony_annotate.py: P6 role weights,
            # P3 failed-call head + unlikelihood) take the token-level path.
            # ``lm_loss`` stays the UNWEIGHTED mean CE either way; the stats ride
            # on ``self._token_loss_stats`` for the titan wrapper to read.
            if token_targets:
                from .modules.agentic_heads import token_level_lm_losses

                lm_loss, loss, self._token_loss_stats = token_level_lm_losses(
                    self, hidden_states, labels, token_targets
                )
            else:
                lm_loss = self._compute_shifted_lm_loss_chunked(ntp_hidden, labels)
                loss = lm_loss

            # Latent feedback: the paper applies the NTP loss to every pass. MEAN,
            # not sum, so the gradient scale -- and therefore the LR and beta2
            # rules -- carry over from the baseline unchanged.
            #
            # `lm_loss` deliberately stays the FINAL pass's number. That is the
            # model's actual predictive distribution under the paper's "fused"
            # decoding, it is what val_lm_loss reports, and it is the only thing
            # comparable to a baseline cell. Ranking on the pass-averaged value
            # would mix in the deliberately-handicapped early passes.
            if feedback_pass_losses:
                loss = (loss + sum(feedback_pass_losses)) / (len(feedback_pass_losses) + 1)

            # Add Multi-Token Prediction loss (training only; computed above)
            if mtp_output is not None:
                mtp_loss = mtp_output.loss
                mtp_logits = None  # Not stored during training for memory efficiency
                if mtp_loss is not None:
                    loss = loss + mtp_loss

            # Bag-of-next-K lookahead (training only). Order-free multi-label
            # "which tokens appear in the next K positions", weighted into the
            # total. Reported through aux_loss, NEVER folded into lm_loss — the
            # sweep ranks on val_lm_loss and must stay a pure next-token number.
            if self.bag_lookahead is not None and self.training and input_ids is not None:
                bag_loss = self.bag_lookahead(
                    hidden_states,
                    input_ids,
                    doc_ids=getattr(self.model, "_flex_doc_ids", None),
                )
                loss = loss + self.bag_lookahead.weight * bag_loss

            # Bank reconstruction (training only): recover earlier-chunk tokens
            # from the bank state alone. Needs the states the delta write stashed
            # on this forward; absent (e.g. cached generation) it is skipped.
            if self.bank_recon is not None and self.training and input_ids is not None:
                states = getattr(self.model, "_recon_states", None)
                if states is not None:
                    recon_loss = self.bank_recon(
                        states,
                        input_ids,
                        state_docs=getattr(self.model, "_recon_state_docs", None),
                        doc_ids=getattr(self.model, "_recon_token_docs", None),
                        prefix_len=int(getattr(self.model, "_recon_prefix_len", 0)),
                    )
                    loss = loss + self.bank_recon.weight * recon_loss

            # Add MoE auxiliary losses
            if use_moe and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                aux_loss = self._compute_moe_aux_loss(outputs.router_logits)
                if aux_loss is not None:
                    loss = loss + aux_loss

            # Add MoD auxiliary loss (training only). MoP's cost-aware aux
            # arrives through the same channel (F4: one aux path).
            if (
                (self.use_mod or getattr(self.config, "use_pathways", False))
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

            # No full (B, T, V) logits when labels are present (train or eval); the
            # chunked loss above exists precisely to avoid materializing them. Only
            # pay for them when the caller explicitly asks.
            if output_logits:
                logits = self.lm_head(hidden_states).float()

        else:
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
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
                lm_loss=lm_loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=router_logits,
            )

        # ``loss`` here may already have MoD's aux loss folded in (see above), so
        # report ``lm_loss``/``aux_loss`` separately — the scaling sweep ranks on
        # the pure LM loss and cannot recover it from the fused scalar.
        return CausalLMOutputWithMTP(
            loss=loss,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            aux_loss=aux_loss,
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
        if bool(getattr(self.config, "tst_active", False)):
            raise RuntimeError(
                "Generation is not supported while TST phase (i) is active "
                "(set config.tst_active=False after the recovery cutover)."
            )
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

        # Account for any PREFIX the model prepends on the first forward: meta
        # tokens, and the persistent globals of the swa_globals family. Both land
        # in the KV cache at prefill and stay there, so from the second step on
        # the cache is longer than generate's attention mask by exactly the
        # prefix and the [B, 1, q, kv] mask no longer lines up with the keys.
        #
        # Stated as "whatever the cache holds beyond the mask", not as a count of
        # meta tokens: the old version asked `num_meta_tokens` alone and so did
        # nothing for the globals prefix, leaving the swa_globals arms to die on
        # a 17-vs-13 mask mismatch in the first MLA layer of the first decode
        # step. A prefix is visible to every token, hence ones.
        prefix_added = 0
        if past_key_values is not None and attention_mask is not None:
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

            mask_len = attention_mask.shape[1]
            expected_mask_len = cache_len + input_ids.shape[1]
            if expected_mask_len > mask_len:
                extra = expected_mask_len - mask_len
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    extra,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                prefix_added = extra

        # Create position IDs.
        #
        # Recomputed from the EXTENDED mask whenever a prefix was added, even
        # when generate supplied its own: generate builds position_ids from its
        # own prefix-free mask, so the next token would be numbered `prefix`
        # positions below where the prefill actually put its predecessors (16 ->
        # 12 with four globals). RoPE then rotates it to the wrong place.
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and (position_ids is None or prefix_added):
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
