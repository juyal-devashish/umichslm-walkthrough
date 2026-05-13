"""Pytest fixtures for UMichSLM tests."""

import pytest
import torch

# Import the package (this also registers with AutoModel)
import sys
sys.path.insert(0, str(__file__).replace("/tests/conftest.py", "/src"))

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, UMichSLMModel
from umich_slm.utils import set_seed, get_generator


# Global test seed for reproducibility
TEST_SEED = 42


@pytest.fixture(autouse=True)
def seed_everything():
    """Automatically seed all RNGs before each test for reproducibility.

    This fixture runs automatically before every test to ensure
    consistent random state.
    """
    set_seed(TEST_SEED)
    yield


@pytest.fixture
def reproducible_generator():
    """Provide a seeded generator for tests that need explicit control."""
    return get_generator(TEST_SEED)


@pytest.fixture
def tiny_config():
    """Create a minimal configuration for fast testing."""
    return UMichSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=172,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Create a minimal model for testing."""
    return UMichSLMForCausalLM(tiny_config)


@pytest.fixture
def tiny_base_model(tiny_config):
    """Create a minimal base model (without LM head) for testing."""
    return UMichSLMModel(tiny_config)


@pytest.fixture
def sample_input(tiny_config, reproducible_generator):
    """Create sample input tensors with reproducible randomness."""
    batch_size = 2
    seq_len = 16
    return {
        "input_ids": torch.randint(
            0, tiny_config.vocab_size, (batch_size, seq_len),
            generator=reproducible_generator
        ),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


@pytest.fixture
def sample_input_with_labels(tiny_config, reproducible_generator):
    """Create sample input tensors with labels for training."""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        0, tiny_config.vocab_size, (batch_size, seq_len),
        generator=reproducible_generator
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": input_ids.clone(),
    }


@pytest.fixture(params=["cpu"])
def device(request):
    """Device fixture that handles CUDA availability."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture
def hymba_config():
    """Create a configuration with Hymba enabled for testing."""
    return UMichSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_hymba=True,
        hymba_num_meta_tokens=4,
        hymba_ssm_state_size=8,
        hymba_ssm_expand=2,
    )


@pytest.fixture
def pope_config():
    """Create a configuration with PoPE position embeddings."""
    return UMichSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=172,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        position_embedding_type="pope",
    )
