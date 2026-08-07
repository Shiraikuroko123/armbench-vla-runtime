"""Best-effort periodic Panda control with asynchronous action-chunk inference.

The policy used by the local benchmark is deliberately scripted and
non-learned.  The module exercises the runtime boundary around a VLA policy:
camera/state capture, blocking inference on a worker thread, observation-age
alignment, bounded action repair, torque-controlled MuJoCo execution, and a
verified terminal stop.  It does not claim hard real-time scheduling or
physical-robot safety certification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.execution import DEFAULT_KD, DEFAULT_KP
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.async_dispatch import AsyncChunkDispatcher, AsyncDispatchConfig
from armbench.vla.async_worker import LatestPolicyWorker, PolicyOutcome
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.observation import MuJoCoDroidObservationBuilder
from armbench.vla.online import _OnlineVideoRecorder
from armbench.vla.trajectory_repair import (
    BrakingRepairConfig,
    BrakingTrajectoryGuard,
)
from armbench.vla.types import ActionChunk, DROID_ACTION_DIM, VLAObservation


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ASYNC_PANDA_MODES = ("unguarded", "legacy_greedy", "braking_invariant")


@dataclass(frozen=True)
class AsyncPandaConfig:
    """Timing, safety, and execution limits for one wall-clock episode."""

    action_period_s: float = 1.0 / 15.0
    control_period_s: float = 0.01
    response_deadline_s: float = 0.2
    warmup_s: float = 0.1
    settle_s: float = 0.3
    max_action_steps: int = 45
    action_horizon: int = 15
    goal_tolerance_rad: float = 0.05
    clearance_m: float = 0.02
    collision_resolution_rad: float = 0.02
    max_state_mismatch_rad: float = 0.05
    joint_velocity_limit_rad_s: float = 1.0
    joint_acceleration_limit_rad_s2: float = 15.0
    terminal_feedback_gain_s_inv: float = 2.0
    repair_selection_deadline_ms: float = 20.0
    kp: tuple[float, ...] = tuple(float(value) for value in DEFAULT_KP)
    kd: tuple[float, ...] = tuple(float(value) for value in DEFAULT_KD)

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.action_period_s,
                self.control_period_s,
                self.response_deadline_s,
                self.warmup_s,
                self.settle_s,
                self.goal_tolerance_rad,
                self.clearance_m,
                self.collision_resolution_rad,
                self.max_state_mismatch_rad,
                self.joint_velocity_limit_rad_s,
                self.joint_acceleration_limit_rad_s2,
                self.terminal_feedback_gain_s_inv,
                self.repair_selection_deadline_ms,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(values))
            or self.action_period_s <= 0.0
            or self.control_period_s <= 0.0
            or self.response_deadline_s < 0.0
            or self.warmup_s < 0.0
            or self.settle_s < 0.0
            or self.goal_tolerance_rad < 0.0
            or self.clearance_m < 0.0
            or self.collision_resolution_rad <= 0.0
            or self.max_state_mismatch_rad < 0.0
            or self.joint_velocity_limit_rad_s <= 0.0
            or self.joint_acceleration_limit_rad_s2 <= 0.0
            or self.terminal_feedback_gain_s_inv <= 0.0
            or self.repair_selection_deadline_ms <= 0.0
            or type(self.max_action_steps) is not int
            or self.max_action_steps <= 0
            or type(self.action_horizon) is not int
            or self.action_horizon <= 0
        ):
            raise ValueError("asynchronous Panda configuration is invalid")
        if self.response_deadline_s < self.action_period_s:
            raise ValueError("response deadline must allow at least one action period")
        for label, gains in (("kp", self.kp), ("kd", self.kd)):
            values = np.asarray(gains, dtype=float)
            if (
                values.shape != (7,)
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise ValueError(f"{label} must contain seven nonnegative gains")


@dataclass(frozen=True)
class ScriptedPolicyFaults:
    """Deterministic response-time and action faults keyed by sequence id."""

    latency_schedule_ms: tuple[float, ...] = (0.0,)
    latency_jitter_ms: float = 0.0
    drop_probability: float = 0.0
    seed: int = 20260807
    spike_sequence_id: int | None = None
    spike_joint: int = 0
    spike_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        schedule = np.asarray(self.latency_schedule_ms, dtype=float)
        if (
            schedule.ndim != 1
            or len(schedule) == 0
            or not np.all(np.isfinite(schedule))
            or np.any(schedule < 0.0)
            or not np.isfinite(self.latency_jitter_ms)
            or self.latency_jitter_ms < 0.0
            or not np.isfinite(self.drop_probability)
            or not 0.0 <= self.drop_probability <= 1.0
            or type(self.seed) is not int
            or self.seed < 0
            or self.spike_joint not in range(7)
            or not np.isfinite(self.spike_velocity_rad_s)
            or (
                self.spike_sequence_id is not None
                and self.spike_sequence_id < 0
            )
        ):
            raise ValueError("scripted policy fault configuration is invalid")

    def sample(self, sequence_id: int) -> tuple[float, bool]:
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(sequence_id)])
        )
        jitter = (
            float(generator.normal(0.0, self.latency_jitter_ms))
            if self.latency_jitter_ms > 0.0
            else 0.0
        )
        base = self.latency_schedule_ms[
            sequence_id % len(self.latency_schedule_ms)
        ]
        latency_ms = max(0.0, float(base) + jitter)
        dropped = bool(generator.random() < self.drop_probability)
        return latency_ms, dropped


class SleepingReferenceActionChunkPolicy:
    """A real blocking policy call used to test asynchronous scheduling."""

    def __init__(
        self,
        reference_positions: ArrayLike,
        config: AsyncPandaConfig,
        faults: ScriptedPolicyFaults = ScriptedPolicyFaults(),
    ) -> None:
        reference = np.asarray(reference_positions, dtype=float)
        if (
            reference.ndim != 2
            or reference.shape[1] != 7
            or len(reference) < 2
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("reference_positions must be a finite Nx7 path")
        self.reference_positions = reference.copy()
        self.config = config
        self.faults = faults

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started_at = time.monotonic()
        latency_ms, dropped = self.faults.sample(observation.sequence_id)
        if latency_ms > 0.0:
            time.sleep(latency_ms / 1000.0)
        if dropped:
            raise TimeoutError("scripted policy response dropped")

        # Sequence ids identify timing, not task progress. Re-anchor the
        # scripted policy to measured state so holds do not silently skip the
        # geometric reference and turn this into an open-loop clock follower.
        distances = np.max(
            np.abs(
                self.reference_positions
                - observation.joint_position[None, :]
            ),
            axis=1,
        )
        cursor = int(np.argmin(distances))
        predicted_q = observation.joint_position.copy()
        actions = np.zeros(
            (self.config.action_horizon, DROID_ACTION_DIM), dtype=float
        )
        for index in range(self.config.action_horizon):
            target_index = min(
                cursor + index + 1, len(self.reference_positions) - 1
            )
            target = self.reference_positions[target_index]
            velocity = np.clip(
                (target - predicted_q) / self.config.action_period_s,
                -self.config.joint_velocity_limit_rad_s,
                self.config.joint_velocity_limit_rad_s,
            )
            actions[index, :7] = velocity
            actions[index, 7] = float(observation.gripper_position[0])
            predicted_q += self.config.action_period_s * velocity
        if observation.sequence_id >= len(self.reference_positions) - 1:
            terminal_velocity = np.clip(
                self.config.terminal_feedback_gain_s_inv
                * (
                    self.reference_positions[-1]
                    - observation.joint_position
                ),
                -self.config.joint_velocity_limit_rad_s,
                self.config.joint_velocity_limit_rad_s,
            )
            actions[:, :7] = terminal_velocity
        inject_spike = (
            self.faults.spike_velocity_rad_s != 0.0
            and (
                self.faults.spike_sequence_id is None
                or observation.sequence_id == self.faults.spike_sequence_id
            )
        )
        if inject_spike:
            actions[:, self.faults.spike_joint] = (
                self.faults.spike_velocity_rad_s
            )
        received_at = time.monotonic()
        return ActionChunk(
            actions=actions,
            source="scripted_non_learned_async_reference",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received_at - started_at) * 1000.0,
            received_at_s=received_at,
            server_timing={"requested_sleep_ms": latency_ms},
        )


@dataclass(frozen=True)
class ObservationSubmission:
    request_id: int
    sequence_id: int
    replaced_request_id: int | None


@dataclass(frozen=True)
class ObservationOutcome:
    request_id: int
    sequence_id: int
    captured_at_s: float
    started_at_s: float
    finished_at_s: float
    worker_thread_id: int
    observation: VLAObservation | None
    failure_type: str | None = None
    failure_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.observation is not None

    @property
    def queue_wait_ms(self) -> float:
        return (self.started_at_s - self.captured_at_s) * 1000.0

    @property
    def capture_latency_ms(self) -> float:
        return (self.finished_at_s - self.started_at_s) * 1000.0


@dataclass(frozen=True)
class _ObservationRequest:
    request_id: int
    sequence_id: int
    captured_at_s: float
    joint_position: FloatArray
    gripper_position: float
    prompt: str


class LatestObservationWorker:
    """Render the latest MuJoCo camera/state snapshot off the control loop."""

    def __init__(self, robot: MuJoCoPanda, *, max_outcomes: int = 64) -> None:
        if max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        self.robot = robot
        self._max_outcomes = int(max_outcomes)
        self._condition = threading.Condition()
        self._pending: _ObservationRequest | None = None
        self._outcomes: deque[ObservationOutcome] = deque()
        self._closed = False
        self._next_request_id = 0
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._superseded_pending = 0
        self._cancelled_pending = 0
        self._dropped_outcomes = 0
        self._worker_thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="armbench-observation-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        sequence_id: int,
        captured_at_s: float,
        joint_position: ArrayLike,
        gripper_position: float,
        prompt: str,
    ) -> ObservationSubmission:
        q = self.robot.validate_configuration(joint_position).copy()
        if (
            sequence_id < 0
            or not np.isfinite(captured_at_s)
            or not np.isfinite(gripper_position)
            or not 0.0 <= gripper_position <= 1.0
            or not prompt.strip()
        ):
            raise ValueError("observation request is invalid")
        with self._condition:
            if self._closed:
                raise RuntimeError("observation worker is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced = self._pending.request_id if self._pending else None
            if replaced is not None:
                self._superseded_pending += 1
            self._pending = _ObservationRequest(
                request_id=request_id,
                sequence_id=sequence_id,
                captured_at_s=float(captured_at_s),
                joint_position=q,
                gripper_position=float(gripper_position),
                prompt=prompt,
            )
            self._submitted += 1
            self._condition.notify_all()
        return ObservationSubmission(request_id, sequence_id, replaced)

    def drain(self) -> tuple[ObservationOutcome, ...]:
        with self._condition:
            outcomes = tuple(self._outcomes)
            self._outcomes.clear()
        return outcomes

    def metrics(self) -> dict[str, object]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "failed": self._failed,
                "superseded_pending": self._superseded_pending,
                "cancelled_pending": self._cancelled_pending,
                "dropped_outcomes": self._dropped_outcomes,
                "pending": self._pending is not None,
                "worker_thread_id": self._worker_thread_id,
                "worker_alive": self._thread.is_alive(),
                "closed": self._closed,
            }

    def close(self, *, timeout_s: float = 2.0) -> bool:
        with self._condition:
            self._closed = True
            if self._pending is not None:
                self._cancelled_pending += 1
                self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _run(self) -> None:
        self._worker_thread_id = threading.get_ident()
        data = mujoco.MjData(self.robot.model)
        try:
            with MuJoCoDroidObservationBuilder(self.robot) as builder:
                while True:
                    with self._condition:
                        while self._pending is None and not self._closed:
                            self._condition.wait()
                        if self._closed and self._pending is None:
                            return
                        request = self._pending
                        self._pending = None
                        self._started += 1
                    assert request is not None
                    started_at = time.monotonic()
                    observation: VLAObservation | None = None
                    failure_type: str | None = None
                    failure_message: str | None = None
                    try:
                        self.robot.set_configuration(
                            data,
                            request.joint_position,
                            finger_position=0.04 * request.gripper_position,
                        )
                        observation = builder.capture(
                            data,
                            prompt=request.prompt,
                            sequence_id=request.sequence_id,
                            captured_at_s=request.captured_at_s,
                        )
                    except Exception as error:
                        failure_type = type(error).__name__
                        failure_message = (str(error).strip() or "no error message")[
                            :500
                        ]
                    finished_at = time.monotonic()
                    outcome = ObservationOutcome(
                        request_id=request.request_id,
                        sequence_id=request.sequence_id,
                        captured_at_s=request.captured_at_s,
                        started_at_s=started_at,
                        finished_at_s=finished_at,
                        worker_thread_id=threading.get_ident(),
                        observation=observation,
                        failure_type=failure_type,
                        failure_message=failure_message,
                    )
                    with self._condition:
                        if len(self._outcomes) >= self._max_outcomes:
                            self._outcomes.popleft()
                            self._dropped_outcomes += 1
                        self._outcomes.append(outcome)
                        self._completed += 1
                        self._failed += int(not outcome.succeeded)
                        self._condition.notify_all()
        except Exception:
            # Renderer initialization failures remain observable in metrics and
            # cannot silently turn into a blank-image policy input.
            with self._condition:
                self._failed += 1
                self._closed = True
                self._condition.notify_all()


@dataclass(frozen=True)
class _PreparedPlan:
    request_id: int
    observation_sequence_id: int
    base_action_index: int
    actions: FloatArray
    predicted_positions: FloatArray
    safe_after_guard: bool
    selected_scale: float
    intervention_steps: int
    repair_latency_ms: float
    selection_deadline_exceeded: bool
    state_mismatch_rad: float
    response_age_ms: float
    terminal_brake_steps: int
    fallback_reason: str | None


@dataclass(frozen=True)
class AsyncPandaEpisodeResult:
    scenario: str
    mode: str
    policy_source: str
    target_is_scenario_goal: bool
    target_reached: bool
    physical_safe: bool
    final_target_error_rad: float
    tracking_rmse_rad: float
    max_tracking_error_rad: float
    control_thread_id: int
    policy_worker_thread_id: int | None
    observation_worker_thread_id: int | None
    control_ticks_during_inference: int
    observation_frames_completed: int
    observation_frames_superseded: int
    accepted_responses: int
    rejected_responses: int
    deadline_rejections: int
    policy_failures: int
    superseded_pending_requests: int
    action_boundaries: int
    hold_boundaries: int
    scaled_plan_count: int
    planned_intervention_steps: int
    braking_boundaries: int
    abrupt_stop_violations: int
    repair_selection_deadline_exceedances: int
    unsafe_prepared_plans: int
    torque_saturation_count: int
    obstacle_contact_steps: int
    self_contact_steps: int
    joint_limit_violation_steps: int
    policy_latencies_ms: FloatArray
    observation_latencies_ms: FloatArray
    repair_latencies_ms: FloatArray
    scheduled_wall_times_s: FloatArray
    actual_wall_times_s: FloatArray
    simulated_times_s: FloatArray
    desired_positions: FloatArray
    actual_positions: FloatArray
    command_velocities: FloatArray
    command_statuses: NDArray[np.str_]
    request_ids: IntArray
    action_indices: IntArray
    observation_ages_ms: FloatArray
    policy_inflight: BoolArray
    obstacle_contacts: BoolArray
    self_contacts: BoolArray
    joint_limit_violations: BoolArray
    events: tuple[dict[str, Any], ...]
    worker_metrics: dict[str, object]
    observation_worker_metrics: dict[str, object]
    dispatcher_metrics: dict[str, object]
    video_path: str | None

    def metrics(self) -> dict[str, object]:
        lateness_ms = (
            self.actual_wall_times_s - self.scheduled_wall_times_s
        ) * 1000.0
        gaps_ms = np.diff(self.actual_wall_times_s) * 1000.0
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "policy_source": self.policy_source,
            "policy_checkpoint_executed": False,
            "scripted_policy": True,
            "physics_executed": True,
            "hard_realtime_claim": False,
            "physical_safety_claim": False,
            "target_is_scenario_goal": self.target_is_scenario_goal,
            "target_reached": self.target_reached,
            "physical_safe": self.physical_safe,
            "safe_target_reached": self.target_reached and self.physical_safe,
            "final_target_error_rad": self.final_target_error_rad,
            "tracking_rmse_rad": self.tracking_rmse_rad,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "control_thread_id": self.control_thread_id,
            "policy_worker_thread_id": self.policy_worker_thread_id,
            "observation_worker_thread_id": (
                self.observation_worker_thread_id
            ),
            "separate_policy_thread": (
                self.policy_worker_thread_id is not None
                and self.policy_worker_thread_id != self.control_thread_id
            ),
            "control_ticks": len(self.actual_wall_times_s),
            "control_ticks_during_inference": (
                self.control_ticks_during_inference
            ),
            "observation_frames_completed": (
                self.observation_frames_completed
            ),
            "observation_frames_superseded": (
                self.observation_frames_superseded
            ),
            "p95_control_tick_lateness_ms": _percentile(lateness_ms, 95),
            "max_control_tick_lateness_ms": _maximum(lateness_ms),
            "p95_control_tick_gap_ms": _percentile(gaps_ms, 95),
            "max_control_tick_gap_ms": _maximum(gaps_ms),
            "accepted_responses": self.accepted_responses,
            "rejected_responses": self.rejected_responses,
            "deadline_rejections": self.deadline_rejections,
            "policy_failures": self.policy_failures,
            "superseded_pending_requests": self.superseded_pending_requests,
            "action_boundaries": self.action_boundaries,
            "hold_boundaries": self.hold_boundaries,
            "hold_rate": (
                self.hold_boundaries / self.action_boundaries
                if self.action_boundaries
                else 0.0
            ),
            "scaled_plan_count": self.scaled_plan_count,
            "planned_intervention_steps": self.planned_intervention_steps,
            "braking_boundaries": self.braking_boundaries,
            "abrupt_stop_violations": self.abrupt_stop_violations,
            "repair_selection_deadline_exceedances": (
                self.repair_selection_deadline_exceedances
            ),
            "unsafe_prepared_plans": self.unsafe_prepared_plans,
            "mean_policy_latency_ms": _mean(self.policy_latencies_ms),
            "p95_policy_latency_ms": _percentile(
                self.policy_latencies_ms, 95
            ),
            "max_policy_latency_ms": _maximum(self.policy_latencies_ms),
            "mean_observation_latency_ms": _mean(
                self.observation_latencies_ms
            ),
            "p95_observation_latency_ms": _percentile(
                self.observation_latencies_ms, 95
            ),
            "max_observation_latency_ms": _maximum(
                self.observation_latencies_ms
            ),
            "mean_repair_latency_ms": _mean(self.repair_latencies_ms),
            "p95_repair_latency_ms": _percentile(
                self.repair_latencies_ms, 95
            ),
            "max_repair_latency_ms": _maximum(self.repair_latencies_ms),
            "max_observation_age_ms": _maximum(
                self.observation_ages_ms[
                    np.isfinite(self.observation_ages_ms)
                ]
            ),
            "torque_saturation_count": self.torque_saturation_count,
            "obstacle_contact_steps": self.obstacle_contact_steps,
            "self_contact_steps": self.self_contact_steps,
            "joint_limit_violation_steps": (
                self.joint_limit_violation_steps
            ),
            "video_path": self.video_path,
        }


def _mean(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)) if len(array) else 0.0


def _maximum(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(array)) if len(array) else 0.0


def _percentile(values: ArrayLike, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, percentile)) if len(array) else 0.0


def _guard_config(config: AsyncPandaConfig) -> GuardConfig:
    return GuardConfig(
        control_dt_s=config.action_period_s,
        deadline_ms=config.response_deadline_s * 1000.0,
        max_state_mismatch_rad=config.max_state_mismatch_rad,
        joint_velocity_clip_rad_s=config.joint_velocity_limit_rad_s,
        joint_acceleration_clip_rad_s2=(
            config.joint_acceleration_limit_rad_s2
        ),
        latch_on_deadline=False,
        latch_on_state_mismatch=False,
    )


def _last_committable_action_index(
    config: AsyncPandaConfig, horizon: int
) -> int:
    # The final index may execute for only the wall-clock time remaining before
    # the deadline. Its complete edge is still checked, which is conservative
    # for position validity; the stop is rebuilt from measured state at expiry.
    index = int(
        math.ceil(
            config.response_deadline_s / config.action_period_s - 1e-12
        )
    )
    return min(horizon - 1, index)


def _prefix_chunk(
    outcome: PolicyOutcome,
    begin: int,
    end: int,
) -> ActionChunk:
    assert outcome.chunk is not None
    return ActionChunk(
        actions=outcome.chunk.actions[begin:end],
        source=outcome.chunk.source,
        observation_sequence_id=outcome.observation.sequence_id,
        inference_latency_ms=outcome.worker_latency_ms,
        received_at_s=outcome.finished_at_s,
        server_timing=outcome.chunk.server_timing,
    )


def _integrate(q_start: FloatArray, actions: FloatArray, dt: float) -> FloatArray:
    positions = [q_start.copy()]
    for action in actions:
        positions.append(positions[-1] + dt * action[:7])
    return np.asarray(positions)


def _prepare_plan(
    mode: str,
    outcome: PolicyOutcome,
    action_index: int,
    q_start: FloatArray,
    gripper_position: float,
    previous_velocity: FloatArray,
    checker: MuJoCoCollisionChecker,
    config: AsyncPandaConfig,
) -> _PreparedPlan | None:
    assert outcome.chunk is not None
    final_index = _last_committable_action_index(
        config, outcome.chunk.horizon
    )
    if action_index > final_index:
        return None
    chunk = _prefix_chunk(outcome, action_index, final_index + 1)
    response_age_ms = (
        outcome.finished_at_s - outcome.observation.captured_at_s
    ) * 1000.0
    mismatch = float(
        np.max(np.abs(q_start - outcome.observation.joint_position))
    )
    if mode == "unguarded":
        actions = chunk.actions.copy()
        positions = _integrate(q_start, actions, config.action_period_s)
        return _PreparedPlan(
            request_id=outcome.request_id,
            observation_sequence_id=outcome.observation.sequence_id,
            base_action_index=action_index,
            actions=actions,
            predicted_positions=positions,
            safe_after_guard=checker.path_is_valid(positions),
            selected_scale=1.0,
            intervention_steps=0,
            repair_latency_ms=0.0,
            selection_deadline_exceeded=False,
            state_mismatch_rad=mismatch,
            response_age_ms=response_age_ms,
            terminal_brake_steps=0,
            fallback_reason=None,
        )
    if mode == "legacy_greedy":
        guard = ActionChunkGuard(checker, _guard_config(config))
        guard.reset(previous_velocity)
        result = guard.guard(
            q_start,
            gripper_position,
            outcome.observation,
            chunk,
        )
        scales = [step.scale for step in result.steps]
        return _PreparedPlan(
            request_id=outcome.request_id,
            observation_sequence_id=outcome.observation.sequence_id,
            base_action_index=action_index,
            actions=result.guarded_actions.copy(),
            predicted_positions=result.predicted_positions.copy(),
            safe_after_guard=(
                result.safe_after_guard
                and checker.path_is_valid(result.predicted_positions)
            ),
            selected_scale=min(scales) if scales else 0.0,
            intervention_steps=result.intervention_steps,
            repair_latency_ms=result.guard_latency_ms,
            selection_deadline_exceeded=False,
            state_mismatch_rad=result.state_mismatch_rad,
            response_age_ms=response_age_ms,
            terminal_brake_steps=0,
            fallback_reason=result.fallback_reason,
        )

    guard = BrakingTrajectoryGuard(
        checker,
        _guard_config(config),
        BrakingRepairConfig(
            selection_deadline_ms=config.repair_selection_deadline_ms
        ),
    )
    guard.reset(previous_velocity)
    result = guard.repair(
        q_start,
        gripper_position,
        outcome.observation,
        chunk,
    )
    return _PreparedPlan(
        request_id=outcome.request_id,
        observation_sequence_id=outcome.observation.sequence_id,
        base_action_index=action_index,
        actions=result.repaired_actions.copy(),
        predicted_positions=result.predicted_positions.copy(),
        safe_after_guard=(
            result.safe_after_repair
            and checker.path_is_valid(result.predicted_positions)
            and checker.path_is_valid(result.terminal_braking_positions)
        ),
        selected_scale=result.selected_scale,
        intervention_steps=result.intervention_steps,
        repair_latency_ms=result.repair_latency_ms,
        selection_deadline_exceeded=result.selection_deadline_exceeded,
        state_mismatch_rad=result.state_mismatch_rad,
        response_age_ms=response_age_ms,
        terminal_brake_steps=result.terminal_brake_steps,
        fallback_reason=result.fallback_reason,
    )


def _verified_braking_actions(
    q_start: FloatArray,
    gripper_position: float,
    previous_velocity: FloatArray,
    checker: MuJoCoCollisionChecker,
    config: AsyncPandaConfig,
    *,
    step_s: float | None = None,
) -> tuple[FloatArray, bool, float]:
    """Build and validate a complete acceleration-limited emergency stop."""

    started = time.perf_counter()
    step = config.action_period_s if step_s is None else float(step_s)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("braking step must be finite and positive")
    delta = config.joint_acceleration_limit_rad_s2 * step
    q = q_start.copy()
    velocity = previous_velocity.copy()
    actions: list[FloatArray] = []
    safe = checker.configuration_is_valid(q)
    maximum_steps = int(
        math.ceil(
            config.joint_velocity_limit_rad_s / delta
        )
    ) + 1
    for _ in range(maximum_steps):
        if np.max(np.abs(velocity)) <= 1e-12:
            break
        next_velocity = np.sign(velocity) * np.maximum(
            np.abs(velocity) - delta, 0.0
        )
        q_next = q + step * next_velocity
        if not checker.edge_is_valid(q, q_next):
            safe = False
            break
        action = np.zeros(DROID_ACTION_DIM, dtype=float)
        action[:7] = next_velocity
        action[7] = gripper_position
        actions.append(action)
        q = q_next
        velocity = next_velocity
    if np.max(np.abs(velocity)) > 1e-12:
        safe = False
    return (
        np.asarray(actions, dtype=float).reshape(-1, DROID_ACTION_DIM),
        safe,
        (time.perf_counter() - started) * 1000.0,
    )


def _write_posthoc_video(
    robot: MuJoCoPanda,
    times_s: FloatArray,
    positions: FloatArray,
    path: Path | None,
    *,
    fps: int,
    render_size: tuple[int, int],
) -> None:
    """Render a measured trace after execution so recording cannot add jitter."""

    if path is None:
        return
    if fps <= 0 or len(render_size) != 2 or any(value <= 0 for value in render_size):
        raise ValueError("video settings must be positive")
    data = mujoco.MjData(robot.model)
    with _OnlineVideoRecorder(
        robot.model,
        path,
        fps=fps,
        render_size=render_size,
    ) as recorder:
        origin = float(times_s[0])
        for trace_time, q in zip(times_s, positions):
            robot.set_configuration(data, q)
            data.time = float(trace_time) - origin
            mujoco.mj_forward(robot.model, data)
            recorder.capture(data)


def run_async_panda_episode(
    scenario_name: str,
    mode: str,
    reference_positions: ArrayLike,
    *,
    policy_faults: ScriptedPolicyFaults = ScriptedPolicyFaults(),
    config: AsyncPandaConfig = AsyncPandaConfig(),
    payload_mass: float = 0.0,
    prompt: str | None = None,
    video_path: Path | None = None,
    video_fps: int = 30,
    render_size: tuple[int, int] = (640, 480),
) -> AsyncPandaEpisodeResult:
    """Run one asynchronous policy/repair/physics episode in wall-clock time."""

    if mode not in ASYNC_PANDA_MODES:
        raise ValueError(f"unknown asynchronous Panda mode: {mode}")
    if not np.isfinite(payload_mass) or payload_mass < 0.0:
        raise ValueError("payload_mass must be finite and nonnegative")
    reference = np.asarray(reference_positions, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1] != 7
        or len(reference) < 2
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("reference_positions must be a finite Nx7 path")
    scenarios = mujoco_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"unknown MuJoCo scenario: {scenario_name}")
    scenario = scenarios[scenario_name]
    task_prompt = prompt or f"move the gripper to the {scenario_name} goal"

    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, config.clearance_m)
    )
    checker = MuJoCoCollisionChecker(
        guard_robot, resolution=config.collision_resolution_rad
    )
    robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        payload_mass=payload_mass,
        torque_control=True,
        vla_cameras=True,
        goal_marker=guard_robot.hand_position(reference[-1]),
    )
    observation_robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        payload_mass=payload_mass,
        vla_cameras=True,
        goal_marker=guard_robot.hand_position(reference[-1]),
    )
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, reference[0])
    data.ctrl[7] = 255.0
    arm_qpos = robot.arm_qpos_addresses
    arm_dofs = robot.arm_dof_addresses
    kp = np.asarray(config.kp, dtype=float)
    kd = np.asarray(config.kd, dtype=float)
    physics_dt = float(robot.model.opt.timestep)

    policy = SleepingReferenceActionChunkPolicy(reference, config, policy_faults)
    worker = LatestPolicyWorker(policy, max_outcomes=128)
    observation_worker = LatestObservationWorker(
        observation_robot, max_outcomes=128
    )
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(
            action_period_s=config.action_period_s,
            deadline_s=config.response_deadline_s,
        )
    )
    hold_action = np.zeros(DROID_ACTION_DIM, dtype=float)
    hold_action[7] = 1.0
    outcome_by_request: dict[int, PolicyOutcome] = {}
    active_plan: _PreparedPlan | None = None
    braking_actions = np.empty((0, DROID_ACTION_DIM), dtype=float)
    braking_cursor = 0
    command = hold_action.copy()
    desired_q = reference[0].copy()
    gripper_position = 1.0
    events: list[dict[str, Any]] = []
    policy_latencies: list[float] = []
    observation_latencies: list[float] = []
    repair_latencies: list[float] = []
    scheduled_times: list[float] = []
    actual_times: list[float] = []
    simulated_times: list[float] = []
    desired_trace: list[FloatArray] = []
    actual_trace: list[FloatArray] = []
    velocity_trace: list[FloatArray] = []
    status_trace: list[str] = []
    request_trace: list[int] = []
    action_index_trace: list[int] = []
    age_trace: list[float] = []
    inflight_trace: list[bool] = []
    obstacle_trace: list[bool] = []
    self_trace: list[bool] = []
    joint_limit_trace: list[bool] = []
    torque_saturation_count = 0
    accepted_responses = 0
    rejected_responses = 0
    deadline_rejections = 0
    policy_failures = 0
    action_boundaries = 0
    hold_boundaries = 0
    scaled_plan_count = 0
    planned_intervention_steps = 0
    braking_boundaries = 0
    abrupt_stop_violations = 0
    repair_selection_deadline_exceedances = 0
    unsafe_prepared_plans = 0
    previous_boundary_velocity = np.zeros(7, dtype=float)
    next_action_boundary = 0.0
    action_sequence_id = 0
    current_status = "hold:no_policy_response"
    current_request_id = -1
    current_action_index = -1
    current_observation_age_ms = float("nan")
    current_command_identity: tuple[str, int, int] = ("hold", -1, -1)
    last_command_switch_s = 0.0
    brake_generation = 0
    worker_closed = False
    observation_worker_closed = False

    def apply_control() -> None:
        nonlocal torque_saturation_count
        current_q = data.qpos[arm_qpos].copy()
        current_dq = data.qvel[arm_dofs].copy()
        requested = (
            kp * (desired_q - current_q)
            + kd * (command[:7] - current_dq)
            + data.qfrc_bias[arm_dofs]
        )
        applied = np.clip(requested, -robot.force_limits, robot.force_limits)
        torque_saturation_count += int(
            np.count_nonzero(np.abs(requested - applied) > 1e-10)
        )
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[arm_dofs] = applied
        data.ctrl[7] = 255.0 * float(command[7])

    def advance_physics(target_time: float) -> None:
        while data.time + 0.5 * physics_dt < target_time:
            mujoco.mj_step(robot.model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("MuJoCo state became non-finite")

    def record_plan(plan: _PreparedPlan, elapsed_s: float) -> None:
        nonlocal scaled_plan_count
        nonlocal planned_intervention_steps
        nonlocal repair_selection_deadline_exceedances
        nonlocal unsafe_prepared_plans
        repair_latencies.append(plan.repair_latency_ms)
        scaled_plan_count += int(plan.selected_scale < 1.0)
        planned_intervention_steps += plan.intervention_steps
        repair_selection_deadline_exceedances += int(
            plan.selection_deadline_exceeded
        )
        unsafe_prepared_plans += int(not plan.safe_after_guard)
        events.append(
            {
                "event": "plan_prepared",
                "wall_time_s": elapsed_s,
                "request_id": plan.request_id,
                "observation_sequence_id": plan.observation_sequence_id,
                "base_action_index": plan.base_action_index,
                "horizon": len(plan.actions),
                "mode": mode,
                "safe_after_guard": plan.safe_after_guard,
                "selected_scale": plan.selected_scale,
                "intervention_steps": plan.intervention_steps,
                "repair_latency_ms": plan.repair_latency_ms,
                "selection_deadline_exceeded": (
                    plan.selection_deadline_exceeded
                ),
                "state_mismatch_rad": plan.state_mismatch_rad,
                "response_age_ms": plan.response_age_ms,
                "terminal_brake_steps": plan.terminal_brake_steps,
                "fallback_reason": plan.fallback_reason,
            }
        )

    def switch_command(
        selected: FloatArray,
        *,
        status: str,
        reason: str,
        request_id: int,
        action_index: int,
        observation_age_ms: float | None,
        identity: tuple[str, int, int],
        elapsed_s: float,
        q_now: FloatArray,
    ) -> None:
        nonlocal command
        nonlocal previous_boundary_velocity
        nonlocal desired_q
        nonlocal current_status
        nonlocal current_request_id
        nonlocal current_action_index
        nonlocal current_observation_age_ms
        nonlocal current_command_identity
        nonlocal last_command_switch_s
        nonlocal abrupt_stop_violations
        if identity == current_command_identity and np.array_equal(selected, command):
            return
        interval_s = max(
            config.control_period_s, elapsed_s - last_command_switch_s
        )
        selected_value = selected.copy()
        if mode == "braking_invariant" and status == "execute":
            maximum_delta = (
                config.joint_acceleration_limit_rad_s2 * interval_s
            )
            selected_value[:7] = np.clip(
                selected_value[:7],
                previous_boundary_velocity - maximum_delta,
                previous_boundary_velocity + maximum_delta,
            )
            age_s = (
                0.0
                if observation_age_ms is None
                else observation_age_ms / 1000.0
            )
            remaining_s = max(
                config.control_period_s,
                min(
                    config.action_period_s,
                    config.response_deadline_s - age_s,
                ),
            )
            q_edge = q_now + remaining_s * selected_value[:7]
            if not checker.edge_is_valid(q_now, q_edge):
                selected_value[:7] = previous_boundary_velocity
                status = "hold"
                reason = "activation_edge_invalid"
                identity = ("hold", request_id, action_index)
        acceleration = float(
            np.max(
                np.abs(
                    selected_value[:7] - previous_boundary_velocity
                )
            )
            / interval_s
        )
        abrupt_stop_violations += int(
            acceleration
            > config.joint_acceleration_limit_rad_s2 + 1e-9
        )
        command = selected_value
        previous_boundary_velocity = command[:7].copy()
        desired_q = q_now.copy()
        current_status = f"{status}:{reason}"
        current_request_id = request_id
        current_action_index = action_index
        current_observation_age_ms = (
            float(observation_age_ms)
            if observation_age_ms is not None
            else float("nan")
        )
        current_command_identity = identity
        last_command_switch_s = elapsed_s
        events.append(
            {
                "event": "command_switch",
                "wall_time_s": elapsed_s,
                "status": status,
                "reason": reason,
                "request_id": request_id,
                "action_index": action_index,
                "observation_age_ms": observation_age_ms,
                "max_acceleration_rad_s2": acceleration,
                "evaluation_boundary": elapsed_s < action_duration,
            }
        )

    # Warm up dynamics and both render paths outside the measured scheduler.
    warmup_end = data.time + config.warmup_s
    while data.time + 0.5 * physics_dt < warmup_end:
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
        mujoco.mj_step(robot.model, data)

    # Renderer/context initialization is intentionally outside the measured
    # interval. The observation itself is discarded; measured observations
    # still include their full snapshot-to-render age.
    try:
        observation_worker.submit(
            sequence_id=0,
            captured_at_s=time.monotonic(),
            joint_position=reference[0],
            gripper_position=1.0,
            prompt=task_prompt,
        )
        prewarm_deadline = time.monotonic() + 3.0
        prewarm_complete = False
        while time.monotonic() < prewarm_deadline and not prewarm_complete:
            for sensor_outcome in observation_worker.drain():
                if not sensor_outcome.succeeded:
                    raise RuntimeError(
                        "observation renderer prewarm failed: "
                        f"{sensor_outcome.failure_type}: "
                        f"{sensor_outcome.failure_message}"
                    )
                prewarm_complete = True
            if not prewarm_complete:
                time.sleep(0.005)
        if not prewarm_complete:
            raise TimeoutError("observation renderer prewarm timed out")
    except Exception:
        observation_worker.close(timeout_s=2.0)
        worker.close(timeout_s=2.0)
        raise

    try:
        control_thread_id = threading.get_ident()
        wall_start = time.monotonic()
        next_control_tick = wall_start
        action_duration = config.max_action_steps * config.action_period_s
        total_duration = action_duration + config.settle_s
        tick_index = 0

        while True:
            now = time.monotonic()
            if now < next_control_tick:
                time.sleep(next_control_tick - now)
            tick_started = time.monotonic()
            elapsed = tick_started - wall_start
            if elapsed > total_duration + config.control_period_s:
                break

            for sensor_outcome in observation_worker.drain():
                observation_latencies.append(
                    sensor_outcome.capture_latency_ms
                )
                events.append(
                    {
                        "event": "observation_outcome",
                        "wall_time_s": elapsed,
                        "request_id": sensor_outcome.request_id,
                        "observation_sequence_id": (
                            sensor_outcome.sequence_id
                        ),
                        "succeeded": sensor_outcome.succeeded,
                        "queue_wait_ms": sensor_outcome.queue_wait_ms,
                        "capture_latency_ms": (
                            sensor_outcome.capture_latency_ms
                        ),
                        "failure_type": sensor_outcome.failure_type,
                        "failure_message": sensor_outcome.failure_message,
                    }
                )
                if sensor_outcome.observation is not None:
                    submission = worker.submit(
                        sensor_outcome.observation
                    )
                    events.append(
                        {
                            "event": "policy_submission",
                            "wall_time_s": elapsed,
                            "request_id": submission.request_id,
                            "observation_sequence_id": (
                                sensor_outcome.sequence_id
                            ),
                            "replaced_request_id": (
                                submission.replaced_request_id
                            ),
                            "sensor_request_id": (
                                sensor_outcome.request_id
                            ),
                        }
                    )

            for outcome in worker.drain():
                outcome_by_request[outcome.request_id] = outcome
                update = dispatcher.publish(outcome, now_s=tick_started)
                policy_latencies.append(outcome.worker_latency_ms)
                policy_failures += int(not outcome.succeeded)
                accepted_responses += int(update.status == "accepted")
                rejected_responses += int(update.status == "rejected")
                deadline_rejections += int(
                    update.reason == "deadline_exceeded"
                )
                events.append(
                    {
                        "event": "policy_outcome",
                        "wall_time_s": elapsed,
                        **outcome.metrics(),
                        "dispatch_status": update.status,
                        "dispatch_reason": update.reason,
                        "observation_age_ms": update.observation_age_ms,
                        "action_offset": update.action_offset,
                    }
                )

            control_now = time.monotonic()
            decision = dispatcher.select(hold_action, now_s=control_now)
            runtime_active = elapsed < action_duration
            q_control = data.qpos[arm_qpos].copy()
            gripper_position = float(
                np.clip(
                    np.mean(data.qpos[robot.finger_qpos_addresses]) / 0.04,
                    0.0,
                    1.0,
                )
            )
            selected_command: FloatArray | None = None
            selected_status = "hold"
            selected_reason = (
                decision.reason if runtime_active else "episode_complete"
            )
            if (
                runtime_active
                and decision.status == "execute"
                and decision.request_id is not None
                and decision.action_index is not None
            ):
                outcome = outcome_by_request.get(decision.request_id)
                if outcome is not None and outcome.succeeded:
                    if (
                        active_plan is None
                        or active_plan.request_id != decision.request_id
                    ):
                        active_plan = _prepare_plan(
                            mode,
                            outcome,
                            decision.action_index,
                            q_control,
                            gripper_position,
                            previous_boundary_velocity,
                            checker,
                            config,
                        )
                        if active_plan is not None:
                            record_plan(active_plan, elapsed)
                    refreshed = dispatcher.select(
                        hold_action, now_s=time.monotonic()
                    )
                    decision = refreshed
                    selected_reason = refreshed.reason
                    if (
                        active_plan is not None
                        and refreshed.status == "execute"
                        and refreshed.request_id == active_plan.request_id
                        and refreshed.action_index is not None
                    ):
                        local_index = (
                            refreshed.action_index
                            - active_plan.base_action_index
                        )
                        if (
                            0 <= local_index < len(active_plan.actions)
                            and (
                                mode != "braking_invariant"
                                or active_plan.safe_after_guard
                            )
                        ):
                            selected_command = (
                                active_plan.actions[local_index].copy()
                            )
                            selected_status = "execute"
                            selected_reason = refreshed.reason

            if selected_command is not None:
                braking_actions = np.empty(
                    (0, DROID_ACTION_DIM), dtype=float
                )
                braking_cursor = 0
                switch_command(
                    selected_command,
                    status=selected_status,
                    reason=selected_reason,
                    request_id=int(decision.request_id),
                    action_index=int(decision.action_index),
                    observation_age_ms=decision.observation_age_ms,
                    identity=(
                        "execute",
                        int(decision.request_id),
                        int(decision.action_index),
                    ),
                    elapsed_s=elapsed,
                    q_now=q_control,
                )
            else:
                active_plan = None
                hold_request_id = (
                    int(decision.request_id)
                    if decision.request_id is not None
                    else -1
                )
                hold_action_index = (
                    int(decision.action_index)
                    if decision.action_index is not None
                    else -1
                )
                if mode == "braking_invariant":
                    moving = bool(
                        np.max(np.abs(previous_boundary_velocity)) > 1e-12
                    )
                    if moving and not current_status.startswith("brake:"):
                        (
                            braking_actions,
                            brake_safe,
                            brake_latency_ms,
                        ) = _verified_braking_actions(
                            q_control,
                            gripper_position,
                            previous_boundary_velocity,
                            checker,
                            config,
                            step_s=config.control_period_s,
                        )
                        braking_cursor = 0
                        brake_generation += 1
                        repair_latencies.append(brake_latency_ms)
                        unsafe_prepared_plans += int(not brake_safe)
                        events.append(
                            {
                                "event": "terminal_brake_prepared",
                                "wall_time_s": elapsed,
                                "reason": selected_reason,
                                "steps": len(braking_actions),
                                "safe": brake_safe,
                                "repair_latency_ms": brake_latency_ms,
                                "step_s": config.control_period_s,
                            }
                        )
                    if braking_cursor < len(braking_actions):
                        selected_command = braking_actions[
                            braking_cursor
                        ].copy()
                        braking_cursor += 1
                        braking_boundaries += 1
                        switch_command(
                            selected_command,
                            status="brake",
                            reason=selected_reason,
                            request_id=hold_request_id,
                            action_index=hold_action_index,
                            observation_age_ms=decision.observation_age_ms,
                            identity=(
                                "brake",
                                brake_generation,
                                braking_cursor,
                            ),
                            elapsed_s=elapsed,
                            q_now=q_control,
                        )
                    else:
                        selected_command = hold_action.copy()
                        selected_command[7] = gripper_position
                        switch_command(
                            selected_command,
                            status="hold",
                            reason=selected_reason,
                            request_id=hold_request_id,
                            action_index=hold_action_index,
                            observation_age_ms=decision.observation_age_ms,
                            identity=("hold", hold_request_id, -1),
                            elapsed_s=elapsed,
                            q_now=q_control,
                        )
                else:
                    braking_actions = np.empty(
                        (0, DROID_ACTION_DIM), dtype=float
                    )
                    braking_cursor = 0
                    selected_command = hold_action.copy()
                    selected_command[7] = gripper_position
                    switch_command(
                        selected_command,
                        status="hold",
                        reason=selected_reason,
                        request_id=hold_request_id,
                        action_index=hold_action_index,
                        observation_age_ms=decision.observation_age_ms,
                        identity=("hold", hold_request_id, -1),
                        elapsed_s=elapsed,
                        q_now=q_control,
                    )

            while (
                elapsed + 1e-9 >= next_action_boundary
                and next_action_boundary < total_duration + 1e-9
            ):
                evaluation_boundary = (
                    next_action_boundary < action_duration - 1e-9
                )
                boundary_status = current_status.split(":", 1)[0]
                if evaluation_boundary:
                    action_boundaries += 1
                    hold_boundaries += int(boundary_status != "execute")
                events.append(
                    {
                        "event": "command_boundary",
                        "wall_time_s": elapsed,
                        "sequence_id": action_sequence_id,
                        "status": boundary_status,
                        "reason": current_status.split(":", 1)[-1],
                        "request_id": current_request_id,
                        "action_index": current_action_index,
                        "observation_age_ms": (
                            None
                            if not np.isfinite(
                                current_observation_age_ms
                            )
                            else current_observation_age_ms
                        ),
                        "max_acceleration_rad_s2": 0.0,
                        "evaluation_boundary": evaluation_boundary,
                    }
                )
                if evaluation_boundary:
                    capture_started = time.monotonic()
                    submission = observation_worker.submit(
                        sequence_id=action_sequence_id,
                        captured_at_s=capture_started,
                        joint_position=q_control,
                        gripper_position=gripper_position,
                        prompt=task_prompt,
                    )
                    events.append(
                        {
                            "event": "observation_submission",
                            "wall_time_s": elapsed,
                            "request_id": submission.request_id,
                            "observation_sequence_id": (
                                action_sequence_id
                            ),
                            "replaced_request_id": (
                                submission.replaced_request_id
                            ),
                            "submit_latency_ms": (
                                time.monotonic() - capture_started
                            ) * 1000.0,
                        }
                    )
                action_sequence_id += 1
                next_action_boundary += config.action_period_s

            desired_q = np.clip(
                desired_q + config.control_period_s * command[:7],
                robot.lower_limits,
                robot.upper_limits,
            )
            apply_control()
            target_sim_time = data.time + config.control_period_s
            advance_physics(target_sim_time)
            q_actual = data.qpos[arm_qpos].copy()
            obstacle = bool(robot.obstacle_contacts(data))
            self_contact = bool(robot.self_contacts(data))
            joint_violation = bool(
                np.any(q_actual < robot.lower_limits - 1e-6)
                or np.any(q_actual > robot.upper_limits + 1e-6)
            )
            worker_state = worker.metrics()
            inflight = int(worker_state["started"]) > int(
                worker_state["completed"]
            )
            scheduled_times.append(next_control_tick - wall_start)
            actual_times.append(tick_started - wall_start)
            simulated_times.append(float(data.time))
            desired_trace.append(desired_q.copy())
            actual_trace.append(q_actual)
            velocity_trace.append(command[:7].copy())
            status_trace.append(current_status)
            request_trace.append(current_request_id)
            action_index_trace.append(current_action_index)
            age_trace.append(current_observation_age_ms)
            inflight_trace.append(inflight)
            obstacle_trace.append(obstacle)
            self_trace.append(self_contact)
            joint_limit_trace.append(joint_violation)

            tick_index += 1
            next_control_tick = wall_start + tick_index * config.control_period_s
    finally:
        observation_worker_closed = observation_worker.close(timeout_s=2.0)
        worker_closed = worker.close(
            timeout_s=max(2.0, max(policy_faults.latency_schedule_ms) / 1000.0 + 1.0)
        )

    if not observation_worker_closed:
        raise RuntimeError("observation worker did not stop after the episode")
    if not worker_closed:
        raise RuntimeError("policy worker did not stop after the episode")
    actual_array = np.asarray(actual_trace, dtype=float)
    desired_array = np.asarray(desired_trace, dtype=float)
    if len(actual_array) == 0:
        raise RuntimeError("asynchronous Panda episode produced no control ticks")
    errors = actual_array - desired_array
    final_error = float(np.max(np.abs(actual_array[-1] - reference[-1])))
    target_is_goal = bool(np.allclose(reference[-1], scenario.goal, atol=1e-12))
    worker_metrics = worker.metrics()
    observation_worker_metrics = observation_worker.metrics()
    dispatcher_metrics = dispatcher.metrics()
    simulated_array = np.asarray(simulated_times, dtype=float)
    _write_posthoc_video(
        robot,
        simulated_array,
        actual_array,
        video_path,
        fps=video_fps,
        render_size=render_size,
    )
    return AsyncPandaEpisodeResult(
        scenario=scenario_name,
        mode=mode,
        policy_source="scripted_non_learned_async_reference",
        target_is_scenario_goal=target_is_goal,
        target_reached=final_error <= config.goal_tolerance_rad,
        physical_safe=(
            not any(obstacle_trace)
            and not any(self_trace)
            and not any(joint_limit_trace)
        ),
        final_target_error_rad=final_error,
        tracking_rmse_rad=float(np.sqrt(np.mean(errors**2))),
        max_tracking_error_rad=float(np.max(np.abs(errors))),
        control_thread_id=control_thread_id,
        policy_worker_thread_id=(
            int(worker_metrics["worker_thread_id"])
            if worker_metrics["worker_thread_id"] is not None
            else None
        ),
        observation_worker_thread_id=(
            int(observation_worker_metrics["worker_thread_id"])
            if observation_worker_metrics["worker_thread_id"] is not None
            else None
        ),
        control_ticks_during_inference=sum(inflight_trace),
        observation_frames_completed=len(observation_latencies),
        observation_frames_superseded=int(
            observation_worker_metrics["superseded_pending"]
        ),
        accepted_responses=accepted_responses,
        rejected_responses=rejected_responses,
        deadline_rejections=deadline_rejections,
        policy_failures=policy_failures,
        superseded_pending_requests=int(
            worker_metrics["superseded_pending"]
        ),
        action_boundaries=action_boundaries,
        hold_boundaries=hold_boundaries,
        scaled_plan_count=scaled_plan_count,
        planned_intervention_steps=planned_intervention_steps,
        braking_boundaries=braking_boundaries,
        abrupt_stop_violations=abrupt_stop_violations,
        repair_selection_deadline_exceedances=(
            repair_selection_deadline_exceedances
        ),
        unsafe_prepared_plans=unsafe_prepared_plans,
        torque_saturation_count=torque_saturation_count,
        obstacle_contact_steps=sum(obstacle_trace),
        self_contact_steps=sum(self_trace),
        joint_limit_violation_steps=sum(joint_limit_trace),
        policy_latencies_ms=np.asarray(policy_latencies, dtype=float),
        observation_latencies_ms=np.asarray(
            observation_latencies, dtype=float
        ),
        repair_latencies_ms=np.asarray(repair_latencies, dtype=float),
        scheduled_wall_times_s=np.asarray(scheduled_times, dtype=float),
        actual_wall_times_s=np.asarray(actual_times, dtype=float),
        simulated_times_s=simulated_array,
        desired_positions=desired_array,
        actual_positions=actual_array,
        command_velocities=np.asarray(velocity_trace, dtype=float),
        command_statuses=np.asarray(status_trace),
        request_ids=np.asarray(request_trace, dtype=np.int64),
        action_indices=np.asarray(action_index_trace, dtype=np.int64),
        observation_ages_ms=np.asarray(age_trace, dtype=float),
        policy_inflight=np.asarray(inflight_trace, dtype=bool),
        obstacle_contacts=np.asarray(obstacle_trace, dtype=bool),
        self_contacts=np.asarray(self_trace, dtype=bool),
        joint_limit_violations=np.asarray(joint_limit_trace, dtype=bool),
        events=tuple(events),
        worker_metrics=worker_metrics,
        observation_worker_metrics=observation_worker_metrics,
        dispatcher_metrics=dispatcher_metrics,
        video_path=str(video_path.resolve()) if video_path is not None else None,
    )
