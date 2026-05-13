"""Torchtitan-style pretraining adapter for UMichSLM.

Provides a thin wrapper, DDP setup, FineWeb dataloader, throughput
instrumentation, and HF-compatible checkpointing so that the existing
``UMichSLMForCausalLM`` model can be trained from ``torchtitan_train/train.py``
without any modifications to model code.

The adapter lives under ``src/umich_slm/titan/`` so it ships with the package
(and benefits from the same AutoConfig/AutoModel registration) while the
runner (``torchtitan_train/``) and TOML configs (``configs/torchtitan/``) sit
at the repo root, mirroring how ``scripts/llamafactory_train.py`` and
``configs/llama_factory/`` are organized.
"""

from .config_schema import TitanConfig, load_toml_config
from .wrapper import TitanModelOutput, UMichSLMTitanWrapper, build_wrapper_from_config

__all__ = [
    "TitanConfig",
    "TitanModelOutput",
    "UMichSLMTitanWrapper",
    "build_wrapper_from_config",
    "load_toml_config",
]
