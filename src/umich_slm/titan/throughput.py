"""Throughput instrumentation: step timing, FLOPs/token, MFU.

The headline deliverable is an apples-to-apples speed comparison vs LLaMA
Factory, so the FLOPs estimator and MFU calculation must be transparent and
configurable. Defaults follow Chinchilla-style accounting (``6 * N``) plus
attention quadratic, MoE active-experts adjustment, MTP head cost, and a
DeltaNet linear-attention term.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


# ---------------------------------------------------------------- HW peaks
# Hardware peak bf16 dense TFLOP/s (vendor-published). Float8 doubles the H100
# figure. RTX Pro 6000 Blackwell number is a placeholder until the spec sheet
# can be confirmed in Phase 4 — flagged in the plan's risk list.
HW_PEAK_TFLOPS: Dict[str, float] = {
    "h100_80gb_bf16": 989.0,
    "h100_80gb_fp8": 1979.0,
    "rtx_pro_6000_blackwell_bf16": 280.0,  # placeholder — confirm at Phase 4
    "a100_80gb_bf16": 312.0,
    "rtx_6000_ada_bf16": 91.1,
}


def detect_hardware_key(precision: str = "bf16") -> str:
    """Best-effort GPU identifier. Falls back to a generic key."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0).lower()
    if "h100" in name:
        return "h100_80gb_fp8" if precision == "fp8" else "h100_80gb_bf16"
    if "rtx pro 6000" in name or "rtx 6000 blackwell" in name:
        return "rtx_pro_6000_blackwell_bf16"
    if "a100" in name:
        return "a100_80gb_bf16"
    if "rtx 6000" in name and "ada" in name:
        return "rtx_6000_ada_bf16"
    return "unknown"


# -------------------------------------------------------------- step timer
class StepTimer:
    """CUDA-event-based step timer with peak-memory tracking.

    Use as ``timer.start()`` / ``timer.end()`` per training step. Times are
    measured on the default stream; results are float milliseconds.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._use_cuda = device.type == "cuda"
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        self._start_wall: Optional[float] = None

    def start(self) -> None:
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._start.record()
        self._start_wall = time.perf_counter()

    def end(self) -> tuple[float, int]:
        wall_end = time.perf_counter()
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = float(self._start.elapsed_time(self._end))
            peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        else:
            elapsed_ms = (wall_end - (self._start_wall or wall_end)) * 1000.0
            peak_bytes = 0
        return elapsed_ms, peak_bytes


# ------------------------------------------------------------- flops counter
@dataclass
class FlopsBreakdown:
    base: float = 0.0
    attention: float = 0.0
    moe: float = 0.0
    mtp: float = 0.0
    deltanet: float = 0.0

    @property
    def total(self) -> float:
        return self.base + self.attention + self.moe + self.mtp + self.deltanet


class FlopsCounter:
    """Analytical per-token forward FLOPs estimator.

    Backward is approximated as 2x forward (Kaplan et al. convention), so the
    per-step training FLOPs reported is ``3 * forward_flops_per_token *
    tokens_per_step``.

    Notes:
        - "Active params" excludes idle MoE experts (see
          ``UMichSLMTitanWrapper.num_active_params``). Plug that in via
          ``set_active_params``.
        - Attention quadratic term scales with seq_len^2 and is reported
          separately from the dense ``6*N*seq_len`` baseline so we can sanity
          check the breakdown in logs.
    """

    def __init__(self, model_config, seq_len: int):
        self.config = model_config
        self.seq_len = seq_len
        self._active_params: Optional[int] = None

    def set_active_params(self, n: int) -> None:
        self._active_params = int(n)

    def per_token_forward_flops(self) -> FlopsBreakdown:
        cfg = self.config
        n_active = self._active_params or 0

        # Dense backbone term — ``2 * N`` FLOPs/token forward (Kaplan ``6N``
        # is forward+backward; we keep forward separate).
        base = 2.0 * n_active

        # Attention quadratic per token: ``2 * num_heads * head_dim * seq_len``
        # per layer (the seq_len^2 cost spread across tokens).
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        attn = (
            2.0
            * cfg.num_hidden_layers
            * cfg.num_attention_heads
            * head_dim
            * self.seq_len
        )

        # MoE term: ``2 * top_k * per_expert_FFN_FLOPs`` is already in ``base``
        # via active-params accounting. Nothing to add (kept here for clarity).
        moe = 0.0

        # MTP heads: ``mtp_num_heads * 2 * hidden * vocab`` per token (one
        # extra LM-head matmul per head). Hidden->vocab is the bulk of the
        # cost; the MTP transformer block per head is folded into ``base``.
        mtp = 0.0
        if getattr(cfg, "use_mtp", False):
            num_heads = int(getattr(cfg, "mtp_num_heads", 1) or 1)
            mtp = 2.0 * num_heads * cfg.hidden_size * cfg.vocab_size

        # DeltaNet recurrent state update: ``2 * num_layers * num_heads *
        # head_dim^2`` per token. Modest term — included for transparency.
        deltanet = 0.0
        if getattr(cfg, "use_deltanet", False):
            dn_heads = int(
                getattr(cfg, "deltanet_num_heads", None) or cfg.num_attention_heads
            )
            dn_head_dim = cfg.hidden_size // max(dn_heads, 1)
            deltanet = 2.0 * cfg.num_hidden_layers * dn_heads * (dn_head_dim ** 2)

        return FlopsBreakdown(
            base=base, attention=attn, moe=moe, mtp=mtp, deltanet=deltanet
        )

    def per_step_flops(self, *, tokens_per_step: int) -> float:
        forward = self.per_token_forward_flops().total * tokens_per_step
        # 1x forward + 2x backward (Kaplan/Chinchilla convention).
        return 3.0 * forward


# ------------------------------------------------------------------ MFU
class MFUTracker:
    def __init__(self, peak_tflops: float):
        self.peak_flops = peak_tflops * 1e12

    def mfu(self, *, total_flops: float, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0 or self.peak_flops <= 0:
            return 0.0
        return total_flops / (elapsed_seconds * self.peak_flops)


def gpu_inventory_snapshot() -> Dict[str, str]:
    """Capture ``nvidia-smi`` output once at run start for the comparison
    script's apples-to-apples assertion. Failures are non-fatal — the field
    is just informational metadata.
    """
    info: Dict[str, str] = {}
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        return info
    info["device_name"] = torch.cuda.get_device_name(0)
    info["device_count"] = str(torch.cuda.device_count())
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8").strip()
        info["nvidia_smi"] = out
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return info
