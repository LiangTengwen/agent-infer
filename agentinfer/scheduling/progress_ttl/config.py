# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated configuration for the Progress-TTL scheduling policy.

The policy combines workload-derived continuous-growth protection, capacity-aware pause/resume, and an acting TTL
derived from estimated cold-prefill cost. Transport, backend polling, and vLLM-specific fields are excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ProgressTTLMode(str, Enum):
    """Operator-selected control mode for Progress-TTL state transitions."""

    ON = "on"
    OFF = "off"
    AUTO = "auto"


class ProgressTTLResumeOrder(str, Enum):
    """Ordering applied to ordinary paused reasoning Programs."""

    MRU = "mru"
    FCFS = "fcfs"


@dataclass(frozen=True)
class ProgressTTLConfig:
    """Policy-owned Progress-TTL parameters."""

    target_max_segment_rounds: int = 14
    mode: ProgressTTLMode = ProgressTTLMode.ON
    ttl_min_seconds: float = 0.05
    ttl_max_seconds: float = 32.0
    ttl_max_cache_miss_impact_ratio: float = 1.0
    auto_enable_utility_seconds: float = 20.0
    auto_disable_utility_seconds: float = 5.0
    ttl_prefill_model_intercept_seconds: float = 0.042935
    ttl_prefill_model_linear_seconds_per_1k_tokens: float = 0.080027
    ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared: float = 0.00220962
    ttl_decode_throughput_alpha: float = 0.15
    shared_prefix_freshness_warmup_seconds: float = 100.0
    shared_prefix_freshness_kv_turnovers: float = 2.0
    resume_capacity_ratio: float = 1.0
    resume_order: ProgressTTLResumeOrder = ProgressTTLResumeOrder.MRU
    resume_reclaim_acting_programs: bool = True
    pause_capacity_ratio: float = 1.0
    privileged_max_context_tokens: int = 262_144
    privileged_ttl_seconds: float = 5.0
    use_fixed_input_token_growth: bool = False
    fixed_input_token_growth_per_round: int = 1024
    enable_batch_gain_admission: bool = True
    decode_step_fixed_seconds: float = 0.012112
    decode_step_seconds_per_request: float = 0.0006939
    decode_step_seconds_per_context_token: float = 2.2516e-7
    decode_buffer_tokens: int = 100
    force_resume_timeout_scale: float = 3.0
    force_resume_timeout_min_seconds: float = 30.0
    force_resume_timeout_max_seconds: float = 300.0
    paused_program_ttl_seconds: float = 1800.0
    # Bounded retention: a request retained before native admission waits at most this long; the runtime then admits
    # it directly, so a saturated backend can never turn retention into an unbounded client wait. ``0`` disables the
    # bound and restores unbounded retention, which is the behavior measured on 2026-11-09 (mean +17.7 s per request).
    queue_timeout_seconds: float = 120.0
    # Optional backend-waiting gate. The embedded adapter reports the native vLLM waiting-request count; when it
    # reaches this threshold, ordinary admission and resume pause until the backend queue drains. ``0`` disables the
    # gate, which is the default: a closed-loop agent workload cannot convert blocked admission into a larger batch,
    # so the gate only inserts latency in front of a queue the engine already owns.
    admission_waiting_request_threshold: int = 0
    # Rank-local active Program count guard, mirroring the router reference ``max_active_programs_per_target``.
    # ``0`` disables the guard.
    max_active_programs_per_target: int = 64
    # Prefer observed engine usage over the router ledger in capacity repair when the adapter reports it. Pausing an
    # idle Program frees no physical KV, so repair must follow the engine's own pressure signal.
    capacity_repair_uses_observed_usage: bool = True

    def __post_init__(self) -> None:
        """Reject values that make deadlines or capacity projections invalid."""
        if not isinstance(self.mode, ProgressTTLMode):
            raise ValueError("mode must be a ProgressTTLMode")
        if not isinstance(self.resume_order, ProgressTTLResumeOrder):
            raise ValueError("resume_order must be a ProgressTTLResumeOrder")
        if self.target_max_segment_rounds <= 0:
            raise ValueError("target_max_segment_rounds must be positive")
        finite_non_negative = (
            self.ttl_min_seconds,
            self.ttl_max_seconds,
            self.ttl_max_cache_miss_impact_ratio,
            self.auto_enable_utility_seconds,
            self.auto_disable_utility_seconds,
            self.ttl_prefill_model_intercept_seconds,
            self.ttl_prefill_model_linear_seconds_per_1k_tokens,
            self.ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared,
            self.shared_prefix_freshness_warmup_seconds,
            self.shared_prefix_freshness_kv_turnovers,
            self.force_resume_timeout_scale,
            self.force_resume_timeout_min_seconds,
            self.force_resume_timeout_max_seconds,
            self.paused_program_ttl_seconds,
            self.privileged_ttl_seconds,
            self.decode_step_fixed_seconds,
            self.decode_step_seconds_per_request,
            self.decode_step_seconds_per_context_token,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite_non_negative):
            raise ValueError("Progress-TTL time, cost, and weight values must be finite and non-negative")
        if not math.isfinite(self.queue_timeout_seconds) or self.queue_timeout_seconds < 0:
            raise ValueError("queue_timeout_seconds must be finite and non-negative")
        if (
            isinstance(self.admission_waiting_request_threshold, bool)
            or not isinstance(self.admission_waiting_request_threshold, int)
            or self.admission_waiting_request_threshold < 0
        ):
            raise ValueError("admission_waiting_request_threshold must be a non-negative integer")
        if (
            isinstance(self.max_active_programs_per_target, bool)
            or not isinstance(self.max_active_programs_per_target, int)
            or self.max_active_programs_per_target < 0
        ):
            raise ValueError("max_active_programs_per_target must be a non-negative integer")
        if not isinstance(self.capacity_repair_uses_observed_usage, bool):
            raise ValueError("capacity_repair_uses_observed_usage must be a boolean")
        if self.ttl_max_seconds < self.ttl_min_seconds:
            raise ValueError("ttl_max_seconds must be >= ttl_min_seconds")
        if self.shared_prefix_freshness_kv_turnovers <= 0:
            raise ValueError("shared_prefix_freshness_kv_turnovers must be positive")
        if self.ttl_max_cache_miss_impact_ratio > 1:
            raise ValueError("ttl_max_cache_miss_impact_ratio must be in [0, 1]")
        if self.auto_disable_utility_seconds > self.auto_enable_utility_seconds:
            raise ValueError("auto_disable_utility_seconds must be <= auto_enable_utility_seconds")
        if self.force_resume_timeout_scale <= 0:
            raise ValueError("force_resume_timeout_scale must be positive")
        if self.force_resume_timeout_min_seconds > self.force_resume_timeout_max_seconds:
            raise ValueError("force_resume_timeout_min_seconds must be <= force_resume_timeout_max_seconds")
        if not math.isfinite(self.ttl_decode_throughput_alpha) or not 0 <= self.ttl_decode_throughput_alpha <= 1:
            raise ValueError("ttl_decode_throughput_alpha must be between 0 and 1")
        if not math.isfinite(self.resume_capacity_ratio) or not 0 < self.resume_capacity_ratio <= 1:
            raise ValueError("resume_capacity_ratio must be in (0, 1]")
        if not isinstance(self.resume_reclaim_acting_programs, bool):
            raise ValueError("resume_reclaim_acting_programs must be a boolean")
        if not math.isfinite(self.pause_capacity_ratio) or not 0 < self.pause_capacity_ratio <= 1:
            raise ValueError("pause_capacity_ratio must be in (0, 1]")
        if self.decode_buffer_tokens < 0:
            raise ValueError("decode_buffer_tokens must be non-negative")
        if self.privileged_max_context_tokens <= 0:
            raise ValueError("privileged_max_context_tokens must be positive")
        if not isinstance(self.use_fixed_input_token_growth, bool):
            raise ValueError("use_fixed_input_token_growth must be a boolean")
        if not isinstance(self.enable_batch_gain_admission, bool):
            raise ValueError("enable_batch_gain_admission must be a boolean")
        if self.decode_step_fixed_seconds <= 0:
            raise ValueError("decode_step_fixed_seconds must be positive")
        if (
            isinstance(self.fixed_input_token_growth_per_round, bool)
            or not isinstance(self.fixed_input_token_growth_per_round, int)
            or self.fixed_input_token_growth_per_round < 0
        ):
            raise ValueError("fixed_input_token_growth_per_round must be a non-negative integer")
