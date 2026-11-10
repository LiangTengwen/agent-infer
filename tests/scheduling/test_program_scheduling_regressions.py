# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Regression tests for the 2026-11-09 Program-scheduling degradation.

The SWE-bench A/B pair (router#17 vs router#18) showed per-request client latency growing by 17.7 s while the engine's
own service time stayed flat. The gaps covered here are the ones that turned admission from an optimization into an
unbounded second queue:

* native waiting demand charged into the capacity ledger;
* global queue precedence blocking every new arrival;
* unbounded retention without a fail-open path;
* poll-only admission (no event-driven wakeup);
* missing active-Program count guard;
* capacity repair pausing on a ledger that credits physical KV to a bookkeeping transition.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import (
    AdmissionDisposition,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    SchedulingEvent,
    SchedulingEventKind,
    SchedulingSnapshot,
    TransitionController,
    TransitionRequest,
    TransitionResult,
)
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.progress_ttl import (
    ProgressTTLConfig,
    ProgressTTLGlobalFactors,
    ProgressTTLStrategy,
)
from agentinfer.scheduling.runtime import ProgramScheduler

pytestmark = pytest.mark.cpu_test


def strategy(config: ProgressTTLConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=ProgressTTLStrategy(config or ProgressTTLConfig(decode_buffer_tokens=0)),
        initial_factors=StrategyFactors(global_factors=ProgressTTLGlobalFactors()),
    )


def view(
    program_id: str,
    *,
    state: ProgramState,
    status: ProgramStatus,
    tokens: int = 100,
    backend_id: str | None = "backend",
) -> ProgramView:
    return ProgramView(
        ref=ProgramRef(program_id, 0),
        state=state,
        status=status,
        tokens=ProgramTokenObservation(estimated_context_tokens=tokens),
        backend_id=backend_id,
    )


def snapshot(
    *programs: ProgramView, capacity: int = 100_000, waiting: tuple[ProgramRef, ...] = ()
) -> SchedulingSnapshot:
    return SchedulingSnapshot(
        observed_at_monotonic_s=100.0,
        backend_id="backend",
        total_kv_tokens=capacity,
        programs=programs,
        waiting_programs=waiting,
    )


EVENT_KIND = {
    "admit": SchedulingEventKind.REQUEST_ADMITTED,
    "queue": SchedulingEventKind.REQUEST_PENDING,
    "pause": SchedulingEventKind.PROGRAM_PAUSED,
    "mark_for_pause": SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
    "resume": SchedulingEventKind.PROGRAM_RESUMED,
    "release": SchedulingEventKind.PROGRAM_RELEASED,
}


def controller(calls: list[TransitionRequest]) -> TransitionController:
    def apply(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        return TransitionResult(
            request.kind,
            request.program,
            True,
            request.reason,
            SchedulingEvent(
                event_id=f"event-{len(calls)}",
                sequence=len(calls),
                kind=EVENT_KIND[request.kind.value],
                occurred_at_monotonic_s=100.0,
                reason=request.reason,
                program=request.program,
            ),
        )

    return TransitionController(apply)


def backend(capacity: int, *, used: int | None = None, waiting: int | None = None) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                backend_id="vllm-local",
                backend_url="embedded://vllm",
                healthy=True,
                dp_ranks=(
                    DpRankInfo(
                        0,
                        True,
                        True,
                        waiting_requests=waiting,
                        total_hbm_kv_tokens=capacity,
                        used_hbm_kv_tokens=used,
                    ),
                ),
            ),
        )
    )


# --------------------------------------------------------------------------- capacity ledger


def test_native_waiting_tokens_are_not_charged_against_capacity() -> None:
    """A queued backend must not look like consumed capacity; the waiting gate is a separate precondition."""
    components = strategy()
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    acting = view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=200)
    base = snapshot(candidate, acting, capacity=10_000)

    assert components.strategy._remaining_tokens(base, 1.0) == 9_800
    assert components.strategy._remaining_tokens(replace(base, native_used_kv_tokens=2_000), 1.0) == 7_800
    # Waiting demand is reported for diagnostics but never subtracted from usable capacity.
    assert (
        components.strategy._remaining_tokens(
            replace(base, native_used_kv_tokens=2_000, native_waiting_kv_tokens=6_000),
            1.0,
        )
        == 7_800
    )


def test_backend_waiting_gate_queues_only_when_enabled_and_reached() -> None:
    disabled = strategy(ProgressTTLConfig(decode_buffer_tokens=0, admission_waiting_request_threshold=0))
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    calls: list[TransitionRequest] = []

    outcome = disabled.strategy.handle_admission(
        replace(snapshot(candidate), native_waiting_requests=5),
        disabled.initial_factors,
        controller(calls),
        candidate.ref,
    )
    assert outcome.disposition is AdmissionDisposition.ADMITTED

    enabled = strategy(ProgressTTLConfig(decode_buffer_tokens=0, admission_waiting_request_threshold=1))
    enabled.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    calls.clear()
    outcome = enabled.strategy.handle_admission(
        replace(snapshot(candidate), native_waiting_requests=1),
        enabled.initial_factors,
        controller(calls),
        candidate.ref,
    )
    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert outcome.reason == "backend_waiting_gate"


# --------------------------------------------------------------------------- queue structure


def test_another_program_waiting_does_not_block_a_new_program() -> None:
    """Per-Program FIFO: another Program's pending request must not force a global convoy."""
    components = strategy()
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    other = view("other", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(candidate, other, waiting=(other.ref,)),
        components.initial_factors,
        controller(calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.ADMITTED
    assert outcome.reason in {"direct_capacity", "batch_gain_over_recovery_cost"}


def test_same_program_second_request_keeps_per_program_fifo_precedence() -> None:
    components = strategy()
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        replace(
            snapshot(candidate, waiting=(candidate.ref,)),
            request_pool_waiting_requests=2,
        ),
        components.initial_factors,
        controller(calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert outcome.reason == "already_waiting"


def test_active_program_count_guard_queues_new_program() -> None:
    components = strategy(ProgressTTLConfig(decode_buffer_tokens=0, max_active_programs_per_target=1))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    active = view("active", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(active, candidate),
        components.initial_factors,
        controller(calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert outcome.reason == "active_program_limit"


# --------------------------------------------------------------------------- bounded retention


def test_queue_timeout_fails_open_and_activates_the_program() -> None:
    components = strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            ttl_min_seconds=1000,
            ttl_max_seconds=1000,
            queue_timeout_seconds=5,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
        queue_timeout_seconds=5,
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=100.0):
        assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1") is True
        assert scheduler.on_request_completion("r1", 600) is None
        # r2 does not fit next to r1's retained context, so the policy retains it.
        assert scheduler.on_request_arrival("r2", AgentIdentity("p2"), 600, backend(1000), "native-r2") is False
        assert scheduler.retained_request_ids == ("r2",)

        assert scheduler.fail_open_expired_requests(now_monotonic_s=104.0) == 0
        assert scheduler.retained_request_ids == ("r2",)

        assert scheduler.fail_open_expired_requests(now_monotonic_s=105.0) == 1
        # The admitted attempt is already committed, so a second pass must not re-admit or double count it.
        assert scheduler.fail_open_expired_requests(now_monotonic_s=105.0) == 0

    admitted = scheduler.consume_admitted_requests()
    assert [(entry.request_id, entry.retained_request) for entry, _ in admitted] == [("r2", "native-r2")]
    p2_ref = scheduler.registry.current_ref("p2")
    assert p2_ref is not None
    assert scheduler.registry.require(p2_ref).state is ProgramState.ACTIVE


def test_queue_timeout_disabled_keeps_retention_unbounded() -> None:
    components = strategy(ProgressTTLConfig(decode_buffer_tokens=0, ttl_min_seconds=1000, ttl_max_seconds=1000))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=100.0):
        scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1")
        scheduler.on_request_completion("r1", 600)
        assert scheduler.on_request_arrival("r2", AgentIdentity("p2"), 600, backend(1000), "native-r2") is False
        assert scheduler.fail_open_expired_requests(now_monotonic_s=1_000_000.0) == 0

    assert scheduler.retained_request_ids == ("r2",)


# --------------------------------------------------------------------------- event-driven wakeup


def test_retained_arrival_wakes_the_next_cycle_immediately() -> None:
    components = strategy(ProgressTTLConfig(decode_buffer_tokens=0, ttl_min_seconds=1000, ttl_max_seconds=1000))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
        schedule_interval_seconds=60,
        queue_timeout_seconds=0,
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0):
        scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1")
        scheduler.schedule_cycle(backend(1000))
    assert scheduler.needs_schedule_cycle(30.0) is False

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=30.0):
        assert scheduler.on_request_arrival("r2", AgentIdentity("p2"), 600, backend(1000), "native-r2") is False
    # The periodic deadline was 60 s away; a retained arrival must not wait for it.
    assert scheduler._next_periodic_check_at_monotonic_s == 0.0
    assert scheduler.needs_schedule_cycle(30.0) is True


def test_completion_wakes_the_next_cycle_immediately() -> None:
    components = strategy(ProgressTTLConfig(decode_buffer_tokens=0, ttl_min_seconds=1000, ttl_max_seconds=1000))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
        schedule_interval_seconds=60,
        queue_timeout_seconds=0,
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0):
        scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1")
        scheduler.schedule_cycle(backend(1000))
        scheduler.on_request_completion("r1", 600)

    assert scheduler._next_periodic_check_at_monotonic_s == 0.0


# --------------------------------------------------------------------------- capacity repair


def test_capacity_repair_follows_observed_usage_instead_of_the_ledger() -> None:
    """Idle Program context in the ledger must not itself trigger a pause when the engine has room."""

    def run(uses_observed: bool) -> list[TransitionRequest]:
        components = strategy(
            ProgressTTLConfig(
                decode_buffer_tokens=0,
                pause_capacity_ratio=1.0,
                capacity_repair_uses_observed_usage=uses_observed,
            )
        )
        components.initial_factors.global_factors.avg_completion_tokens = 100
        active_reasoning = view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
        acting = view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=800)
        calls: list[TransitionRequest] = []
        components.strategy.repair_capacity(
            replace(
                snapshot(active_reasoning, acting, capacity=1000),
                native_used_kv_tokens=200,
            ),
            components.initial_factors,
            controller(calls),
        )
        return calls

    # Ledger view: 200 observed + 800 acting leaves nothing for the 100-token decode headroom -> pause.
    assert [call.kind.value for call in run(uses_observed=False)] == ["pause"]
    # Observed view: only 200 tokens are physically in use, so repair must not churn Program state.
    assert run(uses_observed=True) == []


# --------------------------------------------------------------------------- configuration


def test_config_rejects_invalid_retention_and_gate_values() -> None:
    with pytest.raises(ValueError, match="queue_timeout_seconds"):
        ProgressTTLConfig(queue_timeout_seconds=-1)
    with pytest.raises(ValueError, match="admission_waiting_request_threshold"):
        ProgressTTLConfig(admission_waiting_request_threshold=-1)
    with pytest.raises(ValueError, match="max_active_programs_per_target"):
        ProgressTTLConfig(max_active_programs_per_target=-1)
    with pytest.raises(ValueError, match="capacity_repair_uses_observed_usage"):
        ProgressTTLConfig(capacity_repair_uses_observed_usage="yes")  # type: ignore[arg-type]


def test_program_scheduler_rejects_negative_queue_timeout() -> None:
    components = strategy()
    with pytest.raises(ValueError, match="queue_timeout_seconds"):
        ProgramScheduler[str, object, object](
            components.strategy,
            components.initial_factors,
            backend(1000),
            queue_timeout_seconds=-0.5,
        )
