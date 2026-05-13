"""UMichSLM - A Modern Transformer Architecture.

This package provides a modular implementation of a transformer architecture
featuring SwiGLU activation, RoPE/PoPE position embeddings, RMSNorm, and
Flash Attention 2 support.

The implementation is fully compatible with HuggingFace's transformers library.

Example:
    >>> from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
    >>> config = UMichSLMConfig(hidden_size=256, num_hidden_layers=6)
    >>> model = UMichSLMForCausalLM(config)
    >>> inputs = torch.randint(0, config.vocab_size, (1, 10))
    >>> outputs = model(inputs)
"""

__version__ = "0.1.0"

# Import configuration
from .configuration_umichslm import UMichSLMConfig

# Import models
from .modeling_umichslm import (
    UMichSLMPreTrainedModel,
    UMichSLMModel,
    UMichSLMForCausalLM,
)

# Import presets
from .configs.presets import (
    UMICHSLM_CONFIGS,
    get_umichslm_config,
    list_available_sizes,
    estimate_parameters,
    estimate_backbone_parameters,
)

# Import core modules for advanced usage
from .modules import (
    RMSNorm,
    RotaryEmbedding,
    PolarPositionEmbedding,
    SwiGLUMLP,
    UMichSLMAttention,
    UMichSLMDecoderLayer,
    DynamicCache,
    StaticCache,
    MultiTokenPrediction,
    MTPOutput,
    HymbaDynamicCache,
)

# Import utilities
from .utils import (
    set_seed,
    get_worker_init_fn,
    get_generator,
)

# Import callbacks
from .callbacks import DRoPECallback

# Register with HuggingFace Auto classes
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.models.gpt2 import GPT2TokenizerFast
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    AutoConfig.register("umichslm", UMichSLMConfig)
    AutoModel.register(UMichSLMConfig, UMichSLMModel)
    AutoModelForCausalLM.register(UMichSLMConfig, UMichSLMForCausalLM)

    # Register tokenizer mapping - UMichSLM uses GPT2TokenizerFast (SmolLM2 tokenizer)
    # Use the register method which properly updates the forward mapping
    TOKENIZER_MAPPING.register(UMichSLMConfig, (GPT2TokenizerFast, GPT2TokenizerFast))

    # Also manually update the reverse mapping so AutoTokenizer can look up by config class name
    # The reverse mapping maps config class names (strings) to model_type (strings)
    if hasattr(TOKENIZER_MAPPING, "_reverse_config_mapping"):
        TOKENIZER_MAPPING._reverse_config_mapping["UMichSLMConfig"] = "umichslm"
except ImportError:
    pass  # transformers not installed

__all__ = [
    # Version
    "__version__",
    # Configuration
    "UMichSLMConfig",
    # Models
    "UMichSLMPreTrainedModel",
    "UMichSLMModel",
    "UMichSLMForCausalLM",
    # Presets
    "UMICHSLM_CONFIGS",
    "get_umichslm_config",
    "list_available_sizes",
    "estimate_parameters",
    "estimate_backbone_parameters",
    # Modules
    "RMSNorm",
    "RotaryEmbedding",
    "PolarPositionEmbedding",
    "SwiGLUMLP",
    "UMichSLMAttention",
    "UMichSLMDecoderLayer",
    "DynamicCache",
    "StaticCache",
    "MultiTokenPrediction",
    "MTPOutput",
    "HymbaDynamicCache",
    # Utilities
    "set_seed",
    "get_worker_init_fn",
    "get_generator",
    # Callbacks
    "DRoPECallback",
]
