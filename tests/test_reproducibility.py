"""Tests for reproducibility utilities and guarantees."""

import random

import numpy as np
import pytest
import torch

from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
from umich_slm.utils import get_generator, get_worker_init_fn, set_seed


class TestSetSeed:
    """Tests for the set_seed function."""

    def test_python_random_reproducible(self):
        """Python random should be reproducible."""
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        set_seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_numpy_reproducible(self):
        """NumPy random should be reproducible."""
        set_seed(42)
        arr1 = np.random.randn(10)

        set_seed(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_reproducible(self):
        """PyTorch random should be reproducible."""
        set_seed(42)
        t1 = torch.randn(10)

        set_seed(42)
        t2 = torch.randn(10)

        torch.testing.assert_close(t1, t2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        set_seed(42)
        val1 = random.random()

        set_seed(123)
        val2 = random.random()

        assert val1 != val2

    def test_model_init_reproducible(self, tiny_config):
        """Model initialization should be reproducible with same seed."""
        set_seed(42)
        model1 = UMichSLMForCausalLM(tiny_config)

        set_seed(42)
        model2 = UMichSLMForCausalLM(tiny_config)

        # Compare all parameters
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, msg=f"Parameter {n1} differs")

    def test_model_forward_reproducible(self, tiny_config):
        """Model forward pass should be reproducible."""
        set_seed(42)
        model = UMichSLMForCausalLM(tiny_config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            out1 = model(input_ids).logits

        with torch.no_grad():
            out2 = model(input_ids).logits

        torch.testing.assert_close(out1, out2)

    def test_model_init_different_seeds_different_weights(self, tiny_config):
        """Different seeds should produce different model weights."""
        set_seed(42)
        model1 = UMichSLMForCausalLM(tiny_config)

        set_seed(123)
        model2 = UMichSLMForCausalLM(tiny_config)

        # At least some parameters should differ
        any_different = False
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if not torch.allclose(p1, p2):
                any_different = True
                break

        assert any_different, "Different seeds should produce different weights"


class TestWorkerInitFn:
    """Tests for DataLoader worker initialization."""

    def test_worker_init_creates_function(self):
        """get_worker_init_fn should return a callable."""
        fn = get_worker_init_fn(42)
        assert callable(fn)

    def test_different_workers_different_seeds(self):
        """Different workers should get different but deterministic seeds."""
        fn = get_worker_init_fn(42)

        # Simulate worker 0
        fn(0)
        random_0 = random.random()

        # Simulate worker 1
        fn(1)
        random_1 = random.random()

        # Should be different
        assert random_0 != random_1

    def test_worker_seeds_reproducible(self):
        """Worker seeds should be reproducible across calls."""
        fn = get_worker_init_fn(42)

        # First run of worker 0
        fn(0)
        random_0_first = random.random()

        # Second run of worker 0
        fn(0)
        random_0_second = random.random()

        assert random_0_first == random_0_second

    def test_different_base_seeds_different_workers(self):
        """Different base seeds should produce different worker seeds."""
        fn1 = get_worker_init_fn(42)
        fn2 = get_worker_init_fn(123)

        fn1(0)
        val1 = random.random()

        fn2(0)
        val2 = random.random()

        assert val1 != val2


class TestGenerator:
    """Tests for seeded generator creation."""

    def test_generator_reproducible(self):
        """Generator should produce reproducible results."""
        g1 = get_generator(42)
        vals1 = torch.randn(10, generator=g1)

        g2 = get_generator(42)
        vals2 = torch.randn(10, generator=g2)

        torch.testing.assert_close(vals1, vals2)

    def test_different_seeds_different_values(self):
        """Different seeds should produce different values."""
        g1 = get_generator(42)
        vals1 = torch.randn(10, generator=g1)

        g2 = get_generator(123)
        vals2 = torch.randn(10, generator=g2)

        assert not torch.allclose(vals1, vals2)

    def test_generator_for_randperm(self):
        """Generator should work with randperm."""
        g1 = get_generator(42)
        perm1 = torch.randperm(100, generator=g1)

        g2 = get_generator(42)
        perm2 = torch.randperm(100, generator=g2)

        torch.testing.assert_close(perm1, perm2)

    def test_generator_for_randint(self):
        """Generator should work with randint."""
        g1 = get_generator(42)
        ints1 = torch.randint(0, 1000, (50,), generator=g1)

        g2 = get_generator(42)
        ints2 = torch.randint(0, 1000, (50,), generator=g2)

        torch.testing.assert_close(ints1, ints2)


class TestCuDNNSettings:
    """Tests for cuDNN determinism settings."""

    def test_cudnn_benchmark_disabled(self):
        """cuDNN benchmark should be disabled after set_seed."""
        set_seed(42)
        if torch.backends.cudnn.is_available():
            assert torch.backends.cudnn.benchmark is False

    def test_cudnn_deterministic_enabled(self):
        """cuDNN deterministic should be enabled after set_seed."""
        set_seed(42)
        if torch.backends.cudnn.is_available():
            assert torch.backends.cudnn.deterministic is True
