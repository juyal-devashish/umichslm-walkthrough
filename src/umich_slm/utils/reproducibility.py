"""Reproducibility utilities for UMichSLM training and testing.

This module provides functions to ensure reproducible training runs by properly
seeding all random number generators and optionally enabling deterministic algorithms.

Example:
    >>> from umich_slm.utils import set_seed, get_worker_init_fn
    >>> set_seed(42)  # Set all seeds (no performance impact)
    >>> set_seed(42, deterministic=True)  # Also enable deterministic algorithms
    >>>
    >>> # For DataLoader
    >>> dataloader = DataLoader(dataset, worker_init_fn=get_worker_init_fn(42))
"""

import logging
import os
import random
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(
    seed: int,
    deterministic: bool = False,
    warn_only: bool = True,
) -> None:
    """Set random seeds for reproducibility across all libraries.

    By default, this function sets all RNG seeds and configures cuDNN for
    deterministic behavior with no performance impact. With deterministic=True,
    it also enables torch.use_deterministic_algorithms() which may slow down
    training by 5-15%.

    Args:
        seed: The random seed to use.
        deterministic: If True, enables PyTorch deterministic algorithms and
            sets CUBLAS workspace config. This ensures bitwise reproducibility
            but may impact performance. Default False.
        warn_only: If True and deterministic=True, only warn on non-deterministic
            operations instead of raising errors. Default True.

    Note:
        Default behavior (deterministic=False) provides:
        - Seeding of Python random, NumPy, PyTorch CPU/CUDA, and HuggingFace
        - cuDNN deterministic mode (no performance impact)

        With deterministic=True:
        - Some operations may be 5-15% slower
        - Some operations may not have deterministic implementations
        - A warning is logged to inform the user of potential slowdown

    Example:
        >>> set_seed(42)  # Basic seeding, no performance impact
        >>> set_seed(42, deterministic=True)  # Full determinism, may be slower
    """
    # Set all RNG seeds (no performance impact)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # HuggingFace Transformers (if available)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass

    # Disable cuDNN benchmark for deterministic conv algorithm selection
    # This has minimal performance impact but ensures same algorithm each run
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Full deterministic mode (has performance impact)
    if deterministic:
        logger.warning(
            "Deterministic mode enabled. This may slow down training by 5-15%% "
            "due to use of deterministic CUDA algorithms. Use only when bitwise "
            "reproducibility is required."
        )
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Get a worker_init_fn for DataLoader that ensures reproducible data loading.

    When using multiple DataLoader workers, each worker needs to be seeded
    differently but deterministically to ensure reproducibility.

    Args:
        seed: Base seed. Each worker will use seed + worker_id.

    Returns:
        A function suitable for DataLoader's worker_init_fn parameter.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> worker_init = get_worker_init_fn(42)
        >>> dataloader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init)
    """

    def worker_init_fn(worker_id: int) -> None:
        """Initialize worker with deterministic seed."""
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def get_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    """Create a seeded PyTorch Generator for reproducible random operations.

    Useful for operations that accept a generator parameter, like
    torch.randperm or DataLoader shuffling.

    Args:
        seed: The random seed.
        device: Device for the generator ("cpu" or "cuda"). Default None uses CPU.

    Returns:
        A seeded torch.Generator instance.

    Example:
        >>> g = get_generator(42)
        >>> indices = torch.randperm(100, generator=g)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
