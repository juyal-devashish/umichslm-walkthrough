"""Tests for Rotary Position Embeddings (RoPE) and PoPE."""

import pytest
import torch

from umich_slm.modules.rotary import (
    RotaryEmbedding,
    PolarPositionEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    get_position_embedding_class,
)


class TestRotaryEmbedding:
    """Tests for RoPE implementation."""

    def test_output_shape(self):
        """Output should have the same shape as input."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 64)  # [batch, heads, seq, dim]
        k = torch.randn(2, 4, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_different_output(self):
        """Different positions should produce different encodings."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(1, 1, 2, 64)  # Two positions
        k = q.clone()

        pos_0 = torch.tensor([[0, 1]])
        pos_1 = torch.tensor([[10, 11]])

        q0, _ = rope(q, k, pos_0)
        q1, _ = rope(q, k, pos_1)

        assert not torch.allclose(q0, q1)

    def test_same_input_same_position_same_output(self):
        """Same input and position should produce same output."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(1, 1, 4, 64)
        k = q.clone()
        position_ids = torch.tensor([[0, 1, 2, 3]])

        q1, k1 = rope(q, k, position_ids)
        q2, k2 = rope(q, k, position_ids)

        assert torch.allclose(q1, q2)
        assert torch.allclose(k1, k2)

    def test_cache_extension(self):
        """Cache should extend when sequence exceeds initial max."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=32)
        q = torch.randn(1, 1, 64, 64)  # Longer than initial max
        k = q.clone()
        position_ids = torch.arange(64).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert rope.max_seq_len_cached >= 64

    def test_linear_scaling(self):
        """Linear scaling should modify the position frequencies."""
        rope = RotaryEmbedding(
            dim=64,
            max_position_embeddings=512,
            scaling_factor=2.0,
            rope_type="linear",
        )
        q = torch.randn(1, 1, 16, 64)
        k = q.clone()
        position_ids = torch.arange(16).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)
        assert q_rot.shape == q.shape

    def test_gradient_flow(self):
        """Gradients should flow through RoPE."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 64, requires_grad=True)
        k = torch.randn(2, 4, 16, 64, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = rope(q, k, position_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None


class TestYarnOriginalLength:
    """YaRN's correction range must be defined over the TRAINED length, while the
    cos/sin cache is sized to the extended length (HF convention)."""

    def _yarn(self, **kw):
        return RotaryEmbedding(
            dim=64, base=10000.0, scaling_factor=8.0, rope_type="yarn", **kw
        )

    def test_default_falls_back_to_max_position_embeddings(self):
        # Every pre-existing config: no original length given -> old behaviour.
        a = self._yarn(max_position_embeddings=4096)
        b = self._yarn(max_position_embeddings=4096, original_max_position_embeddings=4096)
        assert a.original_max_position_embeddings == 4096
        assert torch.equal(a.inv_freq, b.inv_freq)

    def test_extended_cache_keeps_the_base_correction_range(self):
        base = self._yarn(max_position_embeddings=4096)
        ext = self._yarn(max_position_embeddings=32768, original_max_position_embeddings=4096)
        # Same frequencies as the base-length YaRN ...
        assert torch.equal(ext.inv_freq, base.inv_freq)
        # ... but a cache that covers the extended sequence, which a compiled run
        # needs because it never grows the cache.
        assert ext.cos_cached.shape[0] == 32768 and base.cos_cached.shape[0] == 4096
        # Positions both caches cover are identical.
        assert torch.equal(ext.cos_cached[:4096], base.cos_cached)

    def test_misusing_the_extended_length_shifts_the_ramp(self):
        # The bug the split prevents: max_position_embeddings=32768 WITHOUT the
        # original length silently computes the ramp over 32768.
        wrong = self._yarn(max_position_embeddings=32768)
        right = self._yarn(max_position_embeddings=32768, original_max_position_embeddings=4096)
        assert not torch.equal(wrong.inv_freq, right.inv_freq)

    def test_high_frequencies_kept_low_frequencies_interpolated(self):
        # YaRN leaves the fast dims alone (they resolve local position and never
        # wrap past the trained length) and divides the slow ones by the factor.
        # The ramp was once applied the other way round, which squeezed the fast
        # dims 8x, extended nothing, and wrecked short-range attention.
        rope = self._yarn(max_position_embeddings=32768, original_max_position_embeddings=4096)
        base = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        ratio = rope.inv_freq / base
        assert ratio[0] == pytest.approx(1.0)
        assert ratio[-1] == pytest.approx(1.0 / 8.0)
        assert torch.all(ratio[1:] <= ratio[:-1] + 1e-6)  # monotone: fast kept, slow squeezed

    def test_matches_transformers_reference(self):
        rope_utils = pytest.importorskip("transformers.modeling_rope_utils")
        from transformers import PretrainedConfig

        cfg = PretrainedConfig()
        cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim = 192, 3, 64
        cfg.max_position_embeddings, cfg.rope_theta = 4096, 10000.0
        cfg.rope_scaling = {
            "rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096,
            "beta_fast": 32, "beta_slow": 1, "rope_theta": 10000.0,
        }
        cfg.rope_parameters = cfg.rope_scaling
        ref, attn_factor = rope_utils.ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
        rope = self._yarn(max_position_embeddings=32768, original_max_position_embeddings=4096)
        torch.testing.assert_close(rope.inv_freq, ref.float(), rtol=1e-5, atol=0)
        assert rope.mscale == pytest.approx(attn_factor)

    def test_mscale_is_the_yarn_temperature(self):
        import math

        rope = self._yarn(max_position_embeddings=32768, original_max_position_embeddings=4096)
        assert rope.mscale == pytest.approx(0.1 * math.log(8.0) + 1.0)

    def test_attention_modules_read_it_from_config(self):
        from umich_slm.configuration_umichslm import UMichSLMConfig
        from umich_slm.modeling_umichslm import UMichSLMForCausalLM

        cfg = UMichSLMConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 256},
        )
        model = UMichSLMForCausalLM(cfg)
        ropes = [m for m in model.modules() if isinstance(m, RotaryEmbedding)]
        assert len(ropes) == 2
        assert {(r.original_max_position_embeddings, r.cos_cached.shape[0]) for r in ropes} == {
            (256, 1024)
        }

    def test_config_rejects_original_longer_than_cache(self):
        from umich_slm.configuration_umichslm import UMichSLMConfig

        with pytest.raises(ValueError, match="original_max_position_embeddings"):
            UMichSLMConfig(
                max_position_embeddings=1024,
                rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
            )


class TestPolarPositionEmbedding:
    """Tests for PoPE implementation."""

    def test_output_shape(self):
        """Output dimensions double due to interleaved x/y components."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)

        # Output dimension doubles due to x/y interleaving
        assert q_rot.shape == (2, 4, 16, 64)
        assert k_rot.shape == (2, 4, 16, 64)

    def test_learnable_bias(self):
        """Phase bias should be learnable when enabled."""
        pope = PolarPositionEmbedding(dim=64, learnable_bias=True)
        assert hasattr(pope, "phase_bias")
        assert pope.phase_bias.requires_grad

    def test_no_learnable_bias(self):
        """Phase bias should be a buffer when disabled."""
        pope = PolarPositionEmbedding(dim=64, learnable_bias=False)
        assert hasattr(pope, "phase_bias")
        assert not pope.phase_bias.requires_grad

    def test_different_positions_different_output(self):
        """Different positions should produce different encodings."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(1, 1, 2, 32)
        k = q.clone()

        pos_0 = torch.tensor([[0, 1]])
        pos_1 = torch.tensor([[10, 11]])

        q0, _ = pope(q, k, pos_0)
        q1, _ = pope(q, k, pos_1)

        assert not torch.allclose(q0, q1)

    def test_non_negative_magnitudes(self):
        """Softplus should ensure non-negative magnitudes."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        # Use negative input to test softplus
        q = torch.randn(2, 4, 16, 32) - 5  # Shift to negative
        k = q.clone()
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)

        # No NaN should appear
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    def test_gradient_flow(self):
        """Gradients should flow through PoPE."""
        pope = PolarPositionEmbedding(dim=32, max_position_embeddings=512)
        q = torch.randn(2, 4, 16, 32, requires_grad=True)
        k = torch.randn(2, 4, 16, 32, requires_grad=True)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = pope(q, k, position_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        if pope.learnable_bias:
            assert pope.phase_bias.grad is not None


class TestRotateHalf:
    """Tests for rotate_half helper function."""

    def test_rotate_half_shape(self):
        """Output should have same shape as input."""
        x = torch.randn(2, 4, 16, 64)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_values(self):
        """Verify rotation operation is correct."""
        x = torch.tensor([[1, 2, 3, 4]])  # [1, 4]
        out = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2]])
        assert torch.allclose(out, expected)


class TestGetPositionEmbeddingClass:
    """Tests for get_position_embedding_class helper."""

    def test_get_rope(self):
        """Should return RotaryEmbedding for 'rope'."""
        cls = get_position_embedding_class("rope")
        assert cls == RotaryEmbedding

    def test_get_pope(self):
        """Should return PolarPositionEmbedding for 'pope'."""
        cls = get_position_embedding_class("pope")
        assert cls == PolarPositionEmbedding

    def test_invalid_type(self):
        """Should raise error for unknown type."""
        with pytest.raises(ValueError):
            get_position_embedding_class("unknown")


class TestUnscaleWindowedRotary:
    """YaRN on global layers only: windowed layers go back to unscaled RoPE."""

    def _yarn_model(self, arch):
        from umich_slm.configs.presets import get_umichslm_config
        from umich_slm.modeling_umichslm import UMichSLMForCausalLM

        cfg = get_umichslm_config(f"{arch}_ar18m")
        cfg.rope_scaling = {"type": "yarn", "factor": 8.0,
                            "original_max_position_embeddings": 4096,
                            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0}
        return UMichSLMForCausalLM(cfg)

    def test_windowed_layers_unscaled_globals_keep_yarn(self):
        from umich_slm.modules.rotary import unscale_windowed_rotary

        m = self._yarn_model("swa_sinkdiet_gqa")
        n = unscale_windowed_rotary(m)
        local = [x.rotary_emb for x in m.modules()
                 if isinstance(getattr(x, "rotary_emb", None), RotaryEmbedding)
                 and getattr(x, "swa_window", None) is not None]
        glob = [x.rotary_emb for x in m.modules()
                if isinstance(getattr(x, "rotary_emb", None), RotaryEmbedding)
                and getattr(x, "swa_window", None) is None]
        assert n == len(local) > 0 and glob
        ref = RotaryEmbedding(local[0].dim, max_position_embeddings=local[0].max_position_embeddings,
                              base=local[0].base)
        for r in local:
            assert r.rope_type == "default" and r.mscale == 1.0
            assert torch.equal(r.inv_freq, ref.inv_freq)
            assert torch.equal(r.cos_cached, ref.cos_cached)
        for r in glob:
            assert r.rope_type == "yarn" and r.mscale > 1.0
        assert unscale_windowed_rotary(m) == 0  # idempotent

    def test_dense_is_noop(self):
        from umich_slm.modules.rotary import unscale_windowed_rotary

        assert unscale_windowed_rotary(self._yarn_model("qwen3_like")) == 0

    @staticmethod
    def _sides(m):
        rot = [x for x in m.modules() if isinstance(getattr(x, "rotary_emb", None), RotaryEmbedding)]
        local = [x.rotary_emb for x in rot if getattr(x, "swa_window", None) is not None]
        glob = [x.rotary_emb for x in rot if getattr(x, "swa_window", None) is None]
        return local, glob

    def _scoped_model(self, arch):
        from umich_slm.configs.presets import get_umichslm_config
        from umich_slm.modeling_umichslm import UMichSLMForCausalLM

        cfg = get_umichslm_config(f"{arch}_ar18m")
        cfg.rope_scaling = {"type": "yarn", "factor": 8.0,
                            "original_max_position_embeddings": 4096,
                            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0}
        cfg.rope_scaling_scope = "global"
        return UMichSLMForCausalLM(cfg)

    def test_config_scope_global_applies_at_build(self):
        """config.rope_scaling_scope='global' is the training path: no eval-side call."""
        local, glob = self._sides(self._scoped_model("swa_sinkdiet_gqa"))
        assert local and glob
        ref = RotaryEmbedding(local[0].dim, max_position_embeddings=local[0].max_position_embeddings,
                              base=local[0].base)
        for r in local:
            assert r.rope_type == "default" and r.mscale == 1.0
            assert torch.equal(r.cos_cached, ref.cos_cached)
        for r in glob:
            assert r.rope_type == "yarn" and r.mscale > 1.0

    def test_config_scope_survives_save_and_load(self, tmp_path):
        """from_pretrained rebuilds the non-persistent buffers; they must stay unscaled."""
        from umich_slm.modeling_umichslm import UMichSLMForCausalLM

        m = self._scoped_model("swa_sinkdiet_gqa")
        m.save_pretrained(tmp_path)
        loaded = UMichSLMForCausalLM.from_pretrained(tmp_path)
        assert loaded.config.rope_scaling_scope == "global"
        (l0, g0), (l1, g1) = self._sides(m), self._sides(loaded)
        for a, b in zip(l0 + g0, l1 + g1):
            assert (a.rope_type, a.mscale) == (b.rope_type, b.mscale)
            assert torch.equal(a.inv_freq, b.inv_freq) and torch.equal(a.cos_cached, b.cos_cached)

    def test_invalid_scope_rejected(self):
        from umich_slm.configuration_umichslm import UMichSLMConfig

        with pytest.raises(ValueError, match="rope_scaling_scope"):
            UMichSLMConfig(rope_scaling_scope="local")

    def test_verifier_audits_both_sides(self):
        """verify_context_extension passes global-only and still fails YaRN-everywhere."""
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "verify_context_extension.py"
        spec = importlib.util.spec_from_file_location("verify_context_extension", path)
        vce = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vce)
        m = self._scoped_model("swa_sinkdiet_gqa")
        rs, theta = m.config.rope_scaling, m.config.rope_theta
        _, fails = vce.audit_rotary(m, 4096, rs, theta, scope="global")
        assert fails == []
        _, fails = vce.audit_rotary(m, 4096, rs, theta, scope="all")
        assert any("rope_type='default' expected 'yarn'" in f for f in fails)
        # and a YaRN-everywhere model fails the global-only audit
        _, fails = vce.audit_rotary(self._yarn_model("swa_sinkdiet_gqa"), 4096, rs, theta,
                                    scope="global")
        assert any("windowed layer has rope_type='yarn'" in f for f in fails)

    def test_eval_labels_the_scope_the_model_runs(self):
        """A plain yarn8 read of a global-scope checkpoint is a yarn8g read."""
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "eval_downstream_suite.py"
        spec = importlib.util.spec_from_file_location("eval_downstream_suite", path)
        eds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eds)
        rs = {"type": "yarn", "factor": 8.0}
        assert eds.effective_rope_scope(self._scoped_model("swa_sinkdiet_gqa"), "all") == "global"
        assert eds.effective_rope_scope(self._yarn_model("swa_sinkdiet_gqa"), "all") == "all"
        assert eds.effective_rope_scope(self._yarn_model("qwen3_like"), "all") == "all"
        # must equal downstream_job.sh's explicit label, or the relabel never fires
        assert (eds.rope_label(rs, "all"), eds.rope_label(rs, "global")) == ("yarn8", "yarn8g")
        assert eds.rope_label(None, "global") == "native"
