"""Model-free acceptance harness for the asynchronous runtime components."""

from __future__ import annotations

import threading
import time
import os

import numpy as np

from armbench.vla.async_dispatch import (
    AsyncChunkDispatcher,
    AsyncCommandDecision,
    AsyncDispatchConfig,
    DispatchUpdate,
)
from armbench.vla.async_worker import LatestPolicyWorker, PolicyOutcome
from armbench.vla.process_worker import (
    DelayedProcessPolicyFactory,
    ProcessPolicyWorker,
)
from armbench.vla.types import ActionChunk, DROID_ACTION_DIM, VLAObservation


class _DelayedSmokePolicy:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started = time.monotonic()
        time.sleep(self.latency_s)
        received = time.monotonic()
        actions = np.zeros((15, DROID_ACTION_DIM), dtype=float)
        actions[:, 0] = np.arange(15, dtype=float) * 0.01
        actions[:, 7] = float(observation.gripper_position[0])
        return ActionChunk(
            actions=actions,
            source="delayed_scripted_smoke",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received - started) * 1000.0,
            received_at_s=received,
        )


def run_async_runtime_smoke(
    *,
    policy_latency_ms: float = 160.0,
    control_period_ms: float = 10.0,
    action_period_ms: float = 1000.0 / 15.0,
    deadline_ms: float = 200.0,
) -> dict[str, object]:
    """Exercise the threaded runtime locally without MuJoCo or model inference."""

    values = np.asarray(
        [policy_latency_ms, control_period_ms, action_period_ms, deadline_ms],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or policy_latency_ms < 0.0
        or control_period_ms <= 0.0
        or action_period_ms <= 0.0
        or deadline_ms < 0.0
    ):
        raise ValueError("smoke timing values are invalid")
    captured_at = time.monotonic()
    observation = VLAObservation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7, dtype=float),
        gripper_position=np.array([1.0]),
        prompt="local asynchronous runtime smoke test",
        sequence_id=0,
        captured_at_s=captured_at,
    )
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(
            action_period_s=action_period_ms / 1000.0,
            deadline_s=deadline_ms / 1000.0,
        )
    )
    hold_action = np.zeros(DROID_ACTION_DIM, dtype=float)
    hold_action[7] = 1.0
    tick_times: list[float] = []
    update: DispatchUpdate | None = None
    decision: AsyncCommandDecision | None = None
    outcome: PolicyOutcome | None = None
    control_thread_id = threading.get_ident()
    policy = _DelayedSmokePolicy(policy_latency_ms / 1000.0)
    worker = LatestPolicyWorker(policy)
    try:
        submit_started = time.monotonic()
        submission = worker.submit(observation)
        submit_elapsed_ms = (time.monotonic() - submit_started) * 1000.0
        next_tick = time.monotonic()
        timeout_at = next_tick + max(2.0, policy_latency_ms / 1000.0 + 1.0)
        while outcome is None:
            now = time.monotonic()
            if now >= timeout_at:
                raise TimeoutError("asynchronous policy smoke test timed out")
            if now < next_tick:
                time.sleep(min(next_tick - now, control_period_ms / 1000.0))
                continue
            tick_times.append(now)
            for completed in worker.drain():
                outcome = completed
                update = dispatcher.publish(completed, now_s=now)
            decision = dispatcher.select(hold_action, now_s=now)
            next_tick += control_period_ms / 1000.0
    finally:
        worker_stopped = worker.close(timeout_s=2.0)
    assert outcome is not None and update is not None and decision is not None
    gaps_ms = np.diff(np.asarray(tick_times, dtype=float)) * 1000.0
    ticks_during_inference = sum(
        tick <= outcome.finished_at_s for tick in tick_times
    )
    passed = bool(
        worker_stopped
        and outcome.succeeded
        and outcome.worker_thread_id != control_thread_id
        and ticks_during_inference >= 2
        and update.status in {"accepted", "rejected"}
        and decision.status
        == ("execute" if update.status == "accepted" else "hold")
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "scope": "scripted_threading_and_dispatch_only",
        "policy_latency_requested_ms": policy_latency_ms,
        "control_period_ms": control_period_ms,
        "action_period_ms": action_period_ms,
        "deadline_ms": deadline_ms,
        "submission": {
            "request_id": submission.request_id,
            "submit_call_ms": submit_elapsed_ms,
        },
        "control": {
            "thread_id": control_thread_id,
            "ticks": len(tick_times),
            "ticks_during_inference": ticks_during_inference,
            "max_tick_gap_ms": float(np.max(gaps_ms)) if len(gaps_ms) else 0.0,
        },
        "policy": outcome.metrics(),
        "dispatch_update": {
            "status": update.status,
            "reason": update.reason,
            "observation_age_ms": update.observation_age_ms,
            "action_offset": update.action_offset,
        },
        "command": decision.metrics(),
        "worker": worker.metrics(),
        "dispatcher": dispatcher.metrics(),
    }


def run_process_runtime_smoke(
    *,
    policy_latency_ms: float = 160.0,
    control_period_ms: float = 10.0,
    action_period_ms: float = 1000.0 / 15.0,
    deadline_ms: float = 200.0,
) -> dict[str, object]:
    """Exercise spawned inference while the parent control clock keeps ticking."""

    values = np.asarray(
        [policy_latency_ms, control_period_ms, action_period_ms, deadline_ms],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or policy_latency_ms < 0.0
        or control_period_ms <= 0.0
        or action_period_ms <= 0.0
        or deadline_ms < 0.0
    ):
        raise ValueError("smoke timing values are invalid")
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(
            action_period_s=action_period_ms / 1000.0,
            deadline_s=deadline_ms / 1000.0,
        )
    )
    hold_action = np.zeros(DROID_ACTION_DIM, dtype=float)
    hold_action[7] = 1.0
    tick_times: list[float] = []
    update: DispatchUpdate | None = None
    decision: AsyncCommandDecision | None = None
    outcome: PolicyOutcome | None = None
    parent_process_id = os.getpid()
    worker = ProcessPolicyWorker(
        DelayedProcessPolicyFactory(policy_latency_ms / 1000.0)
    )
    captured_at_s = time.monotonic()
    observation = VLAObservation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7, dtype=float),
        gripper_position=np.array([1.0]),
        prompt="spawned asynchronous runtime smoke test",
        sequence_id=0,
        captured_at_s=captured_at_s,
    )
    try:
        submit_started = time.monotonic()
        submission = worker.submit(observation)
        submit_elapsed_ms = (time.monotonic() - submit_started) * 1000.0
        next_tick = time.monotonic()
        timeout_at = next_tick + max(3.0, policy_latency_ms / 1000.0 + 2.0)
        while outcome is None:
            now_s = time.monotonic()
            if now_s >= timeout_at:
                raise TimeoutError("process policy smoke test timed out")
            if now_s < next_tick:
                time.sleep(min(next_tick - now_s, control_period_ms / 1000.0))
                continue
            tick_times.append(now_s)
            for completed in worker.drain():
                outcome = completed
                update = dispatcher.publish(completed, now_s=now_s)
            decision = dispatcher.select(hold_action, now_s=now_s)
            next_tick += control_period_ms / 1000.0
    finally:
        worker_stopped = worker.close(timeout_s=2.0)
    assert outcome is not None and update is not None and decision is not None
    assert outcome.worker_process_id is not None
    gaps_ms = np.diff(np.asarray(tick_times, dtype=float)) * 1000.0
    ticks_during_inference = sum(
        tick <= outcome.finished_at_s for tick in tick_times
    )
    passed = bool(
        worker_stopped
        and outcome.succeeded
        and outcome.worker_process_id != parent_process_id
        and ticks_during_inference >= 2
        and update.status in {"accepted", "rejected"}
        and decision.status
        == ("execute" if update.status == "accepted" else "hold")
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "scope": "scripted_process_isolation_and_dispatch_only",
        "policy_latency_requested_ms": policy_latency_ms,
        "control_period_ms": control_period_ms,
        "action_period_ms": action_period_ms,
        "deadline_ms": deadline_ms,
        "submission": {
            "request_id": submission.request_id,
            "submit_call_ms": submit_elapsed_ms,
        },
        "control": {
            "process_id": parent_process_id,
            "ticks": len(tick_times),
            "ticks_during_inference": ticks_during_inference,
            "max_tick_gap_ms": float(np.max(gaps_ms)) if len(gaps_ms) else 0.0,
        },
        "policy": outcome.metrics(),
        "dispatch_update": {
            "status": update.status,
            "reason": update.reason,
            "observation_age_ms": update.observation_age_ms,
            "action_offset": update.action_offset,
        },
        "command": decision.metrics(),
        "worker": worker.metrics(),
        "dispatcher": dispatcher.metrics(),
    }
