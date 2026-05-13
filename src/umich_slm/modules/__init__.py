"""Core modules for UMichSLM transformer architecture."""

from .rmsnorm import RMSNorm, ZeroCenteredRMSNorm, get_rmsnorm_class
from .rotary import (
    RotaryEmbedding,
    PolarPositionEmbedding,
    apply_rotary_pos_emb,
    get_position_embedding_class,
)
from .swiglu import SwiGLUMLP, compute_intermediate_size
from .attention import UMichSLMAttention
from .transformer_block import UMichSLMDecoderLayer
from .kv_cache import DynamicCache, StaticCache, MLADynamicCache
from .moe import MoELayer, MoERouter, MoEOutput, SparseMoE, compute_moe_loss
from .sliding_window import (
    create_sliding_window_mask,
    create_sliding_window_mask_for_sdpa,
    is_global_attention_layer,
    SlidingWindowCache,
    get_attention_pattern_for_layer,
)
from .multi_token_prediction import (
    MultiTokenPrediction,
    MTPModule,
    MTPOutput,
    compute_mtp_loss,
)
from .mixture_of_depths import (
    MoDRouter,
    MoDGate,
    MoDOutput,
    is_mod_layer,
    compute_mod_aux_loss,
)
from .mla import (
    UMichSLMMLAAttention,
    UMichSLMMLASdpaAttention,
    MLA_ATTENTION_CLASSES,
    get_mla_attention_class,
)
from .deltanet import (
    GatedDeltaNet,
    KimiDeltaAttention,
    DELTANET_CLASSES,
    DeltaNetCache,
    DeltaNetOutput,
    get_deltanet_class,
)
from .griffin import (
    RGLRUAttention,
    GriffinCache,
    GriffinDynamicCache,
    GRIFFIN_CLASSES,
    get_griffin_class,
)
from .repo import (
    RePoPositionPredictor,
    RePoPositionPredictorWithBase,
)
from .mamba2 import Mamba2SSM, Mamba2Cache
from .hymba import HymbaHybridHead, MetaTokens
from .hymba_cache import HymbaDynamicCache
from .engram import EngramModule, EngramHash, EngramMemory, EngramGate
from .moda import MoDADepthKVCache, MoDAFFNProjection, build_moda_combined_mask
from .transformer_block import UMichSLMMoDDecoderLayer, _should_use_deltanet, _should_use_griffin

__all__ = [
    "RMSNorm",
    "ZeroCenteredRMSNorm",
    "get_rmsnorm_class",
    "RotaryEmbedding",
    "PolarPositionEmbedding",
    "apply_rotary_pos_emb",
    "get_position_embedding_class",
    "SwiGLUMLP",
    "compute_intermediate_size",
    "UMichSLMAttention",
    "UMichSLMDecoderLayer",
    "DynamicCache",
    "StaticCache",
    "MLADynamicCache",
    "MoELayer",
    "MoERouter",
    "MoEOutput",
    "SparseMoE",
    "compute_moe_loss",
    "create_sliding_window_mask",
    "create_sliding_window_mask_for_sdpa",
    "is_global_attention_layer",
    "SlidingWindowCache",
    "get_attention_pattern_for_layer",
    "MultiTokenPrediction",
    "MTPModule",
    "MTPOutput",
    "compute_mtp_loss",
    "MoDRouter",
    "MoDGate",
    "MoDOutput",
    "UMichSLMMoDDecoderLayer",
    "is_mod_layer",
    "compute_mod_aux_loss",
    "UMichSLMMLAAttention",
    "UMichSLMMLASdpaAttention",
    "MLA_ATTENTION_CLASSES",
    "get_mla_attention_class",
    "GatedDeltaNet",
    "KimiDeltaAttention",
    "DELTANET_CLASSES",
    "DeltaNetCache",
    "DeltaNetOutput",
    "get_deltanet_class",
    "_should_use_deltanet",
    "RGLRUAttention",
    "GriffinCache",
    "GriffinDynamicCache",
    "GRIFFIN_CLASSES",
    "get_griffin_class",
    "_should_use_griffin",
    "RePoPositionPredictor",
    "RePoPositionPredictorWithBase",
    "Mamba2SSM",
    "Mamba2Cache",
    "HymbaHybridHead",
    "MetaTokens",
    "HymbaDynamicCache",
    "EngramModule",
    "EngramHash",
    "EngramMemory",
    "EngramGate",
    "MoDADepthKVCache",
    "MoDAFFNProjection",
    "build_moda_combined_mask",
]
