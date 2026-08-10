"""Paired CPU replay from attested pi0.5 responses to Panda assurance.

The runner uses previously captured, hash-verified responses. It does not load
or execute a checkpoint. Every source response/scenario pair is adapted once
and then evaluated by direct dispatch, kinematic QP projection, and the full
asynchronous Panda supervisor. Candidate plans and atomically published plans
are stored separately so a rejected prefix cannot be mistaken for execution.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shutil
from statistics import fmean
import threading
import time
from time import perf_counter
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import osqp

from armbench.mujoco_sim.dynamics_braking import (
    generate_dynamics_validated_brake,
)
from armbench.mujoco_sim.model import (
    MENAGERIE_COMMIT,
    MuJoCoPanda,
    default_panda_scene_path,
)
from armbench.mujoco_sim.scenarios import (
    MUJOCO_SCENARIO_VERSION,
    mujoco_scenarios,
)
from armbench.vla.cartesian_adapter import (
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
)
from armbench.vla.integrated_panda_async import (
    AtomicPandaPlanGate,
    LatestIntegratedPandaWorker,
)
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
)
from armbench.vla.integrated_panda_task import make_integrated_task_checker
from armbench.vla.pi05_archive_replay import (
    PI05_CHECKPOINT,
    PI05_CHECKPOINT_SHA256,
    PI05_POLICY_CONFIG,
    PI05_POLICY_FAMILY,
    ValidatedPi05Archive,
    _load_json,
    _require,
    _sha256_file,
    _validate_root_manifest,
    _write_json,
    _write_root_manifest,
    select_stratified_chunks,
    validate_pi05_source_archive,
)
from armbench.vla.qp_projection import (
    QPActionProjector,
    QPProjectionConfig,
    QPProjectionResult,
)
from armbench.vla.types import ActionChunk


INTEGRATED_REPLAY_SCHEMA = "armbench.pi05_integrated_panda_cpu_replay.v1"
INTEGRATED_REPLAY_SUMMARY_SCHEMA = (
    "armbench.pi05_integrated_panda_cpu_replay_summary.v1"
)
INTEGRATED_REPLAY_TRAJECTORY_SCHEMA = (
    "armbench.pi05_integrated_panda_cpu_trajectories.v1"
)
INTEGRATED_REPLAY_SCOPE = "frozen_pi05_paired_integrated_panda_cpu_replay"
PROTOCOL_SCHEMA = "armbench.pi05_integrated_panda_cpu_protocol.v1"
MODES = ("direct_dispatch", "qp_projection", "full_assurance")
SCENARIOS = ("free_space", "single_block", "narrow_gate")
PANDA_DOF = 7
ACTION_DIM = 8
_PROTOCOL_RELATIVE = Path(
    "docs/research/pi05_integrated_panda_cpu_protocol_20260810.json"
)
_IMPLEMENTATION_RELATIVE_PATHS = (
    "armbench/vla/pi05_integrated_replay.py",
    "armbench/vla/pi05_archive_replay.py",
    "armbench/vla/cartesian_adapter.py",
    "armbench/vla/integrated_panda_async.py",
    "armbench/vla/integrated_panda_guard.py",
    "armbench/vla/integrated_panda_task.py",
    "armbench/vla/qp_projection.py",
    "armbench/mujoco_sim/continuous_collision.py",
    "armbench/mujoco_sim/dynamics_braking.py",
)

CSV_FIELDS = (
    "schema_version",
    "case_id",
    "pair_id",
    "selection_index",
    "source_row_index",
    "task_suite",
    "task_id",
    "method",
    "episode_id",
    "episode_index",
    "query_index",
    "scenario",
    "mode",
    "response_action_sha256",
    "source_inference_latency_ms",
    "adapter_latency_ms",
    "mode_latency_ms",
    "audit_latency_ms",
    "activation_response_age_ms",
    "status",
    "reason",
    "supervisor_status",
    "failure_stage",
    "candidate_complete",
    "candidate_action_count",
    "published_action_count",
    "policy_actions_executable",
    "partial_prefix_exposed",
    "all_or_none_publication",
    "integrated_atomic_gate_used",
    "assurance_worker_thread_separate",
    "fallback_validated",
    "qp_feasible",
    "qp_intervention_steps",
    "software_budget_exceeded",
    "response_deadline_exceeded",
    "joint_position_violation_steps",
    "joint_velocity_violation_steps",
    "joint_acceleration_violation_steps",
    "continuous_edges_checked",
    "continuous_unsafe_edges",
    "continuous_indeterminate_edges",
    "braking_boundaries_checked",
    "braking_invalid_boundaries",
    "all_registered_constraints_satisfied",
    "unsafe_plan_published",
    "raw_hand_displacement_m",
    "candidate_hand_displacement_m",
    "published_hand_displacement_m",
    "published_motion_retention_ratio",
)

_INTEGER_FIELDS = frozenset(
    {
        "selection_index",
        "source_row_index",
        "task_id",
        "episode_index",
        "query_index",
        "candidate_action_count",
        "published_action_count",
        "qp_intervention_steps",
        "joint_position_violation_steps",
        "joint_velocity_violation_steps",
        "joint_acceleration_violation_steps",
        "continuous_edges_checked",
        "continuous_unsafe_edges",
        "continuous_indeterminate_edges",
        "braking_boundaries_checked",
        "braking_invalid_boundaries",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "source_inference_latency_ms",
        "adapter_latency_ms",
        "mode_latency_ms",
        "audit_latency_ms",
        "activation_response_age_ms",
        "raw_hand_displacement_m",
        "candidate_hand_displacement_m",
        "published_hand_displacement_m",
        "published_motion_retention_ratio",
    }
)
_BOOLEAN_FIELDS = frozenset(
    {
        "candidate_complete",
        "policy_actions_executable",
        "partial_prefix_exposed",
        "all_or_none_publication",
        "integrated_atomic_gate_used",
        "assurance_worker_thread_separate",
        "fallback_validated",
        "qp_feasible",
        "software_budget_exceeded",
        "response_deadline_exceeded",
        "all_registered_constraints_satisfied",
        "unsafe_plan_published",
    }
)
TRAJECTORY_KEYS = frozenset(
    {
        "schema_version",
        "case_ids",
        "source_row_indices",
        "scenario_codes",
        "mode_codes",
        "candidate_lengths",
        "published_lengths",
        "candidate_actions",
        "candidate_positions",
        "published_actions",
        "published_positions",
    }
)


@dataclass(frozen=True)
class Pi05IntegratedReplayConfig:
    """Registered controls for one frozen-response CPU matrix."""

    chunk_count: int = 30
    selection_seed: int = 20260810
    scenarios: tuple[str, ...] = SCENARIOS
    modes: tuple[str, ...] = MODES
    response_deadline_ms: float = 200.0
    software_budget_ms: float = 20.0
    qp_step_budget_ms: float = 5.0
    worker_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if type(self.chunk_count) is not int or self.chunk_count <= 0:
            raise ValueError("chunk_count must be a positive integer")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("selection_seed must be nonnegative")
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or any(name not in SCENARIOS for name in self.scenarios)
        ):
            raise ValueError("scenarios must be unique registered scenarios")
        if (
            not self.modes
            or len(set(self.modes)) != len(self.modes)
            or any(mode not in MODES for mode in self.modes)
        ):
            raise ValueError("modes must be unique registered modes")
        timings = (
            self.response_deadline_ms,
            self.software_budget_ms,
            self.qp_step_budget_ms,
            self.worker_timeout_s,
        )
        if any(
            type(value) not in {int, float}
            or not np.isfinite(value)
            or value <= 0.0
            for value in timings
        ):
            raise ValueError("replay timing values must be finite and positive")

    @property
    def protocol_conformant(self) -> bool:
        return (
            self.chunk_count == 30
            and self.selection_seed == 20260810
            and self.scenarios == SCENARIOS
            and self.modes == MODES
            and self.response_deadline_ms == 200.0
            and self.software_budget_ms == 20.0
            and self.qp_step_budget_ms == 5.0
        )


@dataclass(frozen=True)
class _CandidateAudit:
    latency_ms: float
    position_violations: int
    velocity_violations: int
    acceleration_violations: int
    edges_checked: int
    unsafe_edges: int
    indeterminate_edges: int
    braking_boundaries: int
    invalid_braking_boundaries: int

    @property
    def all_satisfied(self) -> bool:
        return (
            self.position_violations == 0
            and self.velocity_violations == 0
            and self.acceleration_violations == 0
            and self.unsafe_edges == 0
            and self.indeterminate_edges == 0
            and self.braking_boundaries > 0
            and self.invalid_braking_boundaries == 0
        )


@dataclass(frozen=True)
class _ModeOutcome:
    status: str
    reason: str
    supervisor_status: str
    failure_stage: str
    mode_latency_ms: float
    activation_response_age_ms: float
    candidate_action_count: int
    candidate_actions: np.ndarray
    candidate_positions: np.ndarray
    published_actions: np.ndarray
    published_positions: np.ndarray
    qp_feasible: bool
    qp_intervention_steps: int
    all_or_none_publication: bool
    integrated_atomic_gate_used: bool
    worker_thread_separate: bool
    fallback_validated: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return _project_root() / _PROTOCOL_RELATIVE


def _implementation_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    return {
        relative: _sha256_file(source_root / relative)
        for relative in _IMPLEMENTATION_RELATIVE_PATHS
    }


def _guard_config(config: Pi05IntegratedReplayConfig) -> IntegratedPandaGuardConfig:
    return IntegratedPandaGuardConfig(
        response_deadline_ms=config.response_deadline_ms,
        supervision_budget_ms=config.software_budget_ms,
        qp_step_budget_ms=config.qp_step_budget_ms,
    )


def _qp_projector(
    robot: MuJoCoPanda, config: Pi05IntegratedReplayConfig
) -> QPActionProjector:
    guard = _guard_config(config)
    return QPActionProjector(
        robot,
        None,
        QPProjectionConfig(
            control_dt_s=guard.control_dt_s,
            joint_velocity_limit_scale=1.0,
            absolute_joint_velocity_limits_rad_s=(
                guard.joint_velocity_limits_rad_s
            ),
            joint_acceleration_limit_rad_s2=(
                guard.joint_acceleration_limit_rad_s2
            ),
            joint_limit_margin_rad=guard.joint_limit_margin_rad,
            step_budget_ms=guard.qp_step_budget_ms,
        ),
    )


def _pad_candidate(
    actions: np.ndarray,
    positions: np.ndarray,
    *,
    horizon: int,
    q_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_actions = np.asarray(actions, dtype=float)
    raw_positions = np.asarray(positions, dtype=float)
    length = min(len(raw_actions), max(0, len(raw_positions) - 1), horizon)
    padded_actions = np.zeros((horizon, ACTION_DIM), dtype=float)
    padded_positions = np.repeat(q_start[None, :], horizon + 1, axis=0)
    if length:
        padded_actions[:length] = raw_actions[:length]
        padded_positions[: length + 1] = raw_positions[: length + 1]
        padded_positions[length + 1 :] = raw_positions[length]
    return padded_actions, padded_positions, length


def _published_positions(
    q_start: np.ndarray, actions: np.ndarray, length: int, control_dt_s: float
) -> np.ndarray:
    horizon = len(actions)
    positions = np.repeat(q_start[None, :], horizon + 1, axis=0)
    for index in range(length):
        positions[index + 1] = (
            positions[index] + control_dt_s * actions[index, :PANDA_DOF]
        )
    if length:
        positions[length + 1 :] = positions[length]
    return positions


def _wait_for_outcome(
    worker: LatestIntegratedPandaWorker, timeout_s: float
) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        outcomes = worker.drain()
        if outcomes:
            return outcomes[0]
        time.sleep(0.0005)
    raise TimeoutError("integrated assurance worker timed out")


def _run_direct(
    chunk: ActionChunk,
    positions: np.ndarray,
    *,
    q_start: np.ndarray,
    base_response_age_ms: float,
    config: Pi05IntegratedReplayConfig,
) -> _ModeOutcome:
    started = perf_counter()
    actions = np.asarray(chunk.actions, dtype=float).copy()
    candidate_positions = np.asarray(positions, dtype=float).copy()
    mode_latency_ms = (perf_counter() - started) * 1000.0
    activation_age = base_response_age_ms + mode_latency_ms
    if activation_age > config.response_deadline_ms:
        status = "hold"
        reason = "response_deadline_exceeded_before_activation"
    elif mode_latency_ms > config.software_budget_ms:
        status = "hold"
        reason = "software_budget_exceeded"
    else:
        status = "execute"
        reason = "direct_candidate_atomically_published"
    published = actions.copy() if status == "execute" else np.zeros_like(actions)
    published_positions = _published_positions(
        q_start,
        published,
        len(published) if status == "execute" else 0,
        CartesianAdapterConfig().control_dt_s,
    )
    return _ModeOutcome(
        status=status,
        reason=reason,
        supervisor_status="not_applicable",
        failure_stage=("" if status == "execute" else "activation"),
        mode_latency_ms=mode_latency_ms,
        activation_response_age_ms=activation_age,
        candidate_action_count=len(actions),
        candidate_actions=actions,
        candidate_positions=candidate_positions,
        published_actions=published,
        published_positions=published_positions,
        qp_feasible=False,
        qp_intervention_steps=0,
        all_or_none_publication=True,
        integrated_atomic_gate_used=False,
        worker_thread_separate=False,
        fallback_validated=False,
    )


def _run_qp(
    projector: QPActionProjector,
    chunk: ActionChunk,
    *,
    q_start: np.ndarray,
    base_response_age_ms: float,
    config: Pi05IntegratedReplayConfig,
) -> _ModeOutcome:
    started = perf_counter()
    projection = projector.project_chunk(
        q_start, chunk, previous_velocity=np.zeros(PANDA_DOF)
    )
    mode_latency_ms = (perf_counter() - started) * 1000.0
    activation_age = base_response_age_ms + mode_latency_ms
    actions, positions, length = _pad_candidate(
        projection.projected_actions,
        projection.predicted_positions,
        horizon=len(chunk.actions),
        q_start=q_start,
    )
    step_budget_exceeded = any(step.budget_exceeded for step in projection.steps)
    if activation_age > config.response_deadline_ms:
        status = "hold"
        reason = "response_deadline_exceeded_before_activation"
    elif mode_latency_ms > config.software_budget_ms or step_budget_exceeded:
        status = "hold"
        reason = "software_budget_exceeded"
    elif not projection.feasible or length != len(chunk.actions):
        status = "hold"
        reason = f"qp_projection_failed:{projection.failure_reason}"
    else:
        status = "execute"
        reason = "kinematic_qp_candidate_atomically_published"
    published = actions.copy() if status == "execute" else np.zeros_like(actions)
    published_positions = _published_positions(
        q_start,
        published,
        len(published) if status == "execute" else 0,
        projector.config.control_dt_s,
    )
    return _ModeOutcome(
        status=status,
        reason=reason,
        supervisor_status="not_applicable",
        failure_stage=("" if status == "execute" else "qp_or_activation"),
        mode_latency_ms=mode_latency_ms,
        activation_response_age_ms=activation_age,
        candidate_action_count=length,
        candidate_actions=actions,
        candidate_positions=positions,
        published_actions=published,
        published_positions=published_positions,
        qp_feasible=projection.feasible,
        qp_intervention_steps=projection.intervention_steps,
        all_or_none_publication=True,
        integrated_atomic_gate_used=False,
        worker_thread_separate=False,
        fallback_validated=False,
    )


def _run_full_assurance(
    supervisor: IntegratedPandaSupervisor,
    chunk: ActionChunk,
    *,
    q_start: np.ndarray,
    base_response_age_ms: float,
    config: Pi05IntegratedReplayConfig,
) -> _ModeOutcome:
    gate = AtomicPandaPlanGate(supervisor)
    control_thread_id = threading.get_ident()
    started = perf_counter()
    with LatestIntegratedPandaWorker(supervisor) as worker:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=chunk.observation_sequence_id,
            q=q_start,
            qvel=np.zeros(PANDA_DOF),
            observed_q=q_start,
            response_age_ms=base_response_age_ms,
            chunk=chunk,
        )
        outcome = _wait_for_outcome(worker, config.worker_timeout_s)
        atomic = gate.commit(outcome, q_now=q_start)
    mode_latency_ms = (perf_counter() - started) * 1000.0
    decision = outcome.decision
    projection: QPProjectionResult | None = (
        None if decision is None else decision.qp_result
    )
    if projection is None:
        actions = np.zeros_like(chunk.actions)
        positions = np.repeat(
            q_start[None, :], len(chunk.actions) + 1, axis=0
        )
        candidate_length = 0
    else:
        actions, positions, candidate_length = _pad_candidate(
            projection.projected_actions,
            projection.predicted_positions,
            horizon=len(chunk.actions),
            q_start=q_start,
        )
    published = np.zeros_like(chunk.actions)
    published_count = len(atomic.policy_actions)
    if published_count:
        published[:published_count] = atomic.policy_actions
    published_positions = _published_positions(
        q_start,
        published,
        published_count,
        supervisor.config.control_dt_s,
    )
    return _ModeOutcome(
        status=atomic.status,
        reason=atomic.reason,
        supervisor_status=(
            "" if atomic.supervisor_status is None else atomic.supervisor_status
        ),
        failure_stage=(
            "" if decision is None or decision.failure_stage is None
            else decision.failure_stage
        ),
        mode_latency_ms=mode_latency_ms,
        activation_response_age_ms=atomic.response_age_ms,
        candidate_action_count=candidate_length,
        candidate_actions=actions,
        candidate_positions=positions,
        published_actions=published,
        published_positions=published_positions,
        qp_feasible=bool(projection is not None and projection.feasible),
        qp_intervention_steps=(
            0 if projection is None else projection.intervention_steps
        ),
        all_or_none_publication=True,
        integrated_atomic_gate_used=True,
        worker_thread_separate=(outcome.worker_thread_id != control_thread_id),
        fallback_validated=atomic.fallback_validated,
    )


def _audit_candidate(
    robot: MuJoCoPanda,
    checker: object,
    actions: np.ndarray,
    positions: np.ndarray,
    length: int,
    config: Pi05IntegratedReplayConfig,
) -> _CandidateAudit:
    started = perf_counter()
    guard = _guard_config(config)
    candidate_actions = np.asarray(actions[:length], dtype=float)
    candidate_positions = np.asarray(positions[: length + 1], dtype=float)
    margin = guard.joint_limit_margin_rad
    position_violations = sum(
        bool(
            np.any(q < robot.lower_limits + margin - 1e-9)
            or np.any(q > robot.upper_limits - margin + 1e-9)
        )
        for q in candidate_positions
    )
    velocity_limits = np.asarray(guard.joint_velocity_limits_rad_s)
    velocity_violations = sum(
        bool(np.any(np.abs(action[:PANDA_DOF]) > velocity_limits + 1e-9))
        for action in candidate_actions
    )
    previous = np.zeros(PANDA_DOF)
    acceleration_violations = 0
    for action in candidate_actions:
        velocity = action[:PANDA_DOF]
        acceleration = np.abs(velocity - previous) / guard.control_dt_s
        acceleration_violations += int(
            np.any(
                acceleration
                > guard.joint_acceleration_limit_rad_s2 + 1e-9
            )
        )
        previous = velocity

    unsafe_edges = 0
    indeterminate_edges = 0
    for q_before, q_after in zip(
        candidate_positions[:-1], candidate_positions[1:]
    ):
        certificate = checker.edge_certificate(q_before, q_after)
        unsafe_edges += int(not certificate.certified_safe)
        indeterminate_edges += int(certificate.status == "indeterminate")

    invalid_brakes = 0
    for q_after, action in zip(candidate_positions[1:], candidate_actions):
        brake = generate_dynamics_validated_brake(
            robot,
            checker,
            q_after,
            action[:PANDA_DOF],
            guard.braking,
        )
        invalid_brakes += int(not brake.validated)
    return _CandidateAudit(
        latency_ms=(perf_counter() - started) * 1000.0,
        position_violations=position_violations,
        velocity_violations=velocity_violations,
        acceleration_violations=acceleration_violations,
        edges_checked=length,
        unsafe_edges=unsafe_edges,
        indeterminate_edges=indeterminate_edges,
        braking_boundaries=length,
        invalid_braking_boundaries=invalid_brakes,
    )


def _mode_outcome(
    mode: str,
    *,
    supervisor: IntegratedPandaSupervisor,
    projector: QPActionProjector,
    chunk: ActionChunk,
    adapted_positions: np.ndarray,
    q_start: np.ndarray,
    base_response_age_ms: float,
    config: Pi05IntegratedReplayConfig,
) -> _ModeOutcome:
    if mode == "direct_dispatch":
        return _run_direct(
            chunk,
            adapted_positions,
            q_start=q_start,
            base_response_age_ms=base_response_age_ms,
            config=config,
        )
    if mode == "qp_projection":
        return _run_qp(
            projector,
            chunk,
            q_start=q_start,
            base_response_age_ms=base_response_age_ms,
            config=config,
        )
    if mode == "full_assurance":
        return _run_full_assurance(
            supervisor,
            chunk,
            q_start=q_start,
            base_response_age_ms=base_response_age_ms,
            config=config,
        )
    raise ValueError(f"unknown replay mode: {mode}")


def _run_case(
    archive: ValidatedPi05Archive,
    *,
    source_index: int,
    selection_index: int,
    scenario_name: str,
    mode: str,
    robot: MuJoCoPanda,
    checker: object,
    supervisor: IntegratedPandaSupervisor,
    projector: QPActionProjector,
    adapted_chunk: ActionChunk,
    adapted_positions: np.ndarray,
    adapter_latency_ms: float,
    config: Pi05IntegratedReplayConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    query = archive.transition_queries[source_index]
    scenario = mujoco_scenarios()[scenario_name]
    source_latency = float(query["inference_latency_ms"])
    base_response_age = source_latency + adapter_latency_ms
    result = _mode_outcome(
        mode,
        supervisor=supervisor,
        projector=projector,
        chunk=adapted_chunk,
        adapted_positions=adapted_positions,
        q_start=scenario.start,
        base_response_age_ms=base_response_age,
        config=config,
    )
    horizon = len(adapted_chunk.actions)
    candidate_length = result.candidate_action_count
    published_length = (
        horizon if result.status == "execute" else 0
    )
    audit = _audit_candidate(
        robot,
        checker,
        result.candidate_actions,
        result.candidate_positions,
        candidate_length,
        config,
    )
    start_hand = robot.hand_position(scenario.start)
    raw_end_hand = robot.hand_position(adapted_positions[-1])
    candidate_end = result.candidate_positions[candidate_length]
    candidate_end_hand = robot.hand_position(candidate_end)
    published_end = result.published_positions[published_length]
    published_end_hand = robot.hand_position(published_end)
    raw_displacement = float(np.linalg.norm(raw_end_hand - start_hand))
    candidate_displacement = float(
        np.linalg.norm(candidate_end_hand - start_hand)
    )
    published_displacement = float(
        np.linalg.norm(published_end_hand - start_hand)
    )
    retention = (
        0.0
        if raw_displacement <= 1e-12
        else published_displacement / raw_displacement
    )
    candidate_complete = candidate_length == horizon
    partial_prefix = published_length not in {0, horizon}
    software_exceeded = bool(
        result.mode_latency_ms > config.software_budget_ms
        or result.failure_stage == "supervision_budget"
    )
    response_exceeded = bool(
        result.activation_response_age_ms > config.response_deadline_ms
        or result.failure_stage == "deadline"
    )
    all_constraints = bool(candidate_complete and audit.all_satisfied)
    unsafe_published = bool(result.status == "execute" and not all_constraints)
    case_id = f"row_{source_index:05d}__{scenario_name}__{mode}"
    record = {
        "schema_version": INTEGRATED_REPLAY_SCHEMA,
        "case_id": case_id,
        "pair_id": str(query["pair_id"]),
        "selection_index": selection_index,
        "source_row_index": source_index,
        "task_suite": str(query["task_suite"]),
        "task_id": int(query["task_id"]),
        "method": str(query["method"]),
        "episode_id": str(query["episode_id"]),
        "episode_index": int(query["episode_index"]),
        "query_index": int(query["query_index"]),
        "scenario": scenario_name,
        "mode": mode,
        "response_action_sha256": archive.response_hashes[source_index],
        "source_inference_latency_ms": source_latency,
        "adapter_latency_ms": adapter_latency_ms,
        "mode_latency_ms": result.mode_latency_ms,
        "audit_latency_ms": audit.latency_ms,
        "activation_response_age_ms": result.activation_response_age_ms,
        "status": result.status,
        "reason": result.reason,
        "supervisor_status": result.supervisor_status,
        "failure_stage": result.failure_stage,
        "candidate_complete": candidate_complete,
        "candidate_action_count": candidate_length,
        "published_action_count": published_length,
        "policy_actions_executable": result.status == "execute",
        "partial_prefix_exposed": partial_prefix,
        "all_or_none_publication": result.all_or_none_publication,
        "integrated_atomic_gate_used": result.integrated_atomic_gate_used,
        "assurance_worker_thread_separate": (
            result.worker_thread_separate
        ),
        "fallback_validated": result.fallback_validated,
        "qp_feasible": result.qp_feasible,
        "qp_intervention_steps": result.qp_intervention_steps,
        "software_budget_exceeded": software_exceeded,
        "response_deadline_exceeded": response_exceeded,
        "joint_position_violation_steps": audit.position_violations,
        "joint_velocity_violation_steps": audit.velocity_violations,
        "joint_acceleration_violation_steps": (
            audit.acceleration_violations
        ),
        "continuous_edges_checked": audit.edges_checked,
        "continuous_unsafe_edges": audit.unsafe_edges,
        "continuous_indeterminate_edges": audit.indeterminate_edges,
        "braking_boundaries_checked": audit.braking_boundaries,
        "braking_invalid_boundaries": audit.invalid_braking_boundaries,
        "all_registered_constraints_satisfied": all_constraints,
        "unsafe_plan_published": unsafe_published,
        "raw_hand_displacement_m": raw_displacement,
        "candidate_hand_displacement_m": candidate_displacement,
        "published_hand_displacement_m": published_displacement,
        "published_motion_retention_ratio": retention,
    }
    trajectories = {
        "case_id": np.asarray(case_id),
        "source_row_index": np.asarray(source_index, dtype="<i4"),
        "scenario_code": np.asarray(SCENARIOS.index(scenario_name), dtype="u1"),
        "mode_code": np.asarray(MODES.index(mode), dtype="u1"),
        "candidate_length": np.asarray(candidate_length, dtype="<i4"),
        "published_length": np.asarray(published_length, dtype="<i4"),
        "candidate_actions": result.candidate_actions,
        "candidate_positions": result.candidate_positions,
        "published_actions": result.published_actions,
        "published_positions": result.published_positions,
    }
    return record, trajectories


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _config_document(config: Pi05IntegratedReplayConfig) -> dict[str, Any]:
    document = asdict(config)
    document["scenarios"] = list(config.scenarios)
    document["modes"] = list(config.modes)
    return document


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty replay matrix")
    latencies = [float(row["mode_latency_ms"]) for row in rows]
    return {
        "cases": len(rows),
        "execute": sum(row["status"] == "execute" for row in rows),
        "hold": sum(row["status"] == "hold" for row in rows),
        "verified_brake": sum(
            row["status"] == "verified_brake" for row in rows
        ),
        "unrecoverable_stop": sum(
            row["status"] == "unrecoverable_stop" for row in rows
        ),
        "complete_candidates": sum(
            bool(row["candidate_complete"]) for row in rows
        ),
        "constraint_satisfied_candidates": sum(
            bool(row["all_registered_constraints_satisfied"])
            for row in rows
        ),
        "unsafe_plans_published": sum(
            bool(row["unsafe_plan_published"]) for row in rows
        ),
        "partial_prefixes_exposed": sum(
            bool(row["partial_prefix_exposed"]) for row in rows
        ),
        "software_budget_exceeded": sum(
            bool(row["software_budget_exceeded"]) for row in rows
        ),
        "response_deadline_exceeded": sum(
            bool(row["response_deadline_exceeded"]) for row in rows
        ),
        "qp_intervention_steps": sum(
            int(row["qp_intervention_steps"]) for row in rows
        ),
        "continuous_unsafe_edges": sum(
            int(row["continuous_unsafe_edges"]) for row in rows
        ),
        "braking_invalid_boundaries": sum(
            int(row["braking_invalid_boundaries"]) for row in rows
        ),
        "mean_published_motion_retention_ratio": fmean(
            float(row["published_motion_retention_ratio"])
            for row in rows
        ),
        "p50_mode_latency_ms": _percentile(latencies, 50),
        "p95_mode_latency_ms": _percentile(latencies, 95),
        "maximum_mode_latency_ms": max(latencies),
        "p95_audit_latency_ms": _percentile(
            [float(row["audit_latency_ms"]) for row in rows], 95
        ),
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    config: Pi05IntegratedReplayConfig,
    archive: ValidatedPi05Archive,
) -> dict[str, Any]:
    by_mode = {
        mode: _aggregate([row for row in rows if row["mode"] == mode])
        for mode in config.modes
    }
    full = by_mode.get("full_assurance")
    go = bool(
        full is not None
        and full["execute"] > 0
        and full["unsafe_plans_published"] == 0
        and full["partial_prefixes_exposed"] == 0
    )
    return {
        "schema_version": INTEGRATED_REPLAY_SUMMARY_SCHEMA,
        "scope": INTEGRATED_REPLAY_SCOPE,
        "source_policy_checkpoint_attested": True,
        "policy_checkpoint_executed_in_replay": False,
        "panda_closed_loop_executed": False,
        "task_success_evaluated": False,
        "protocol_conformant": config.protocol_conformant,
        "configuration": _config_document(config),
        "source_validation": {
            "transition_count": archive.transition_count,
            "response_action_hashes_verified": len(archive.response_hashes),
        },
        "overall": _aggregate(rows),
        "by_mode": by_mode,
        "go_no_go": {
            "decision": "go" if go else "no_go",
            "full_assurance_has_executable_plan": bool(
                full is not None and full["execute"] > 0
            ),
            "full_assurance_unsafe_plans_published": (
                0 if full is None else full["unsafe_plans_published"]
            ),
            "partial_prefixes_exposed": _aggregate(rows)[
                "partial_prefixes_exposed"
            ],
            "thresholds_changed_after_freeze": False,
        },
        "claim_boundary": [
            "Frozen official responses are replayed; the checkpoint is not executed.",
            "LIBERO actions are cross-controller inputs, not native Panda outputs.",
            "Hand displacement measures command retention, not task progress.",
            "Cases reset independently and provide no policy feedback loop.",
            "Measured Python timing is not a hard-real-time guarantee.",
            "MuJoCo predicates are not physical-robot safety certification.",
        ],
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen pi0.5 to integrated Panda CPU replay",
        "",
        (
            "The official checkpoint was not executed in this run. This report "
            "replays attested frozen responses and does not measure task success."
        ),
        "",
        "| Mode | Cases | Execute | Hold | Constraint-safe candidates | "
        "Unsafe published | Budget misses | P95 mode latency | Motion retained |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        if mode not in summary["by_mode"]:
            continue
        values = summary["by_mode"][mode]
        lines.append(
            f"| `{mode}` | {values['cases']} | {values['execute']} | "
            f"{values['hold']} | {values['constraint_satisfied_candidates']} | "
            f"{values['unsafe_plans_published']} | "
            f"{values['software_budget_exceeded']} | "
            f"{values['p95_mode_latency_ms']:.3f} ms | "
            f"{values['mean_published_motion_retention_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Registered go/no-go result: **{summary['go_no_go']['decision']}**.",
            "",
            "Hand displacement is only a command-retention proxy. It is not "
            "LIBERO or Panda task progress. Timing is measured best-effort Python "
            "CPU timing, not a worst-case real-time guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectories(
    path: Path, trajectories: Sequence[Mapping[str, np.ndarray]]
) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(INTEGRATED_REPLAY_TRAJECTORY_SCHEMA),
        case_ids=np.asarray([str(item["case_id"]) for item in trajectories]),
        source_row_indices=np.asarray(
            [item["source_row_index"] for item in trajectories], dtype="<i4"
        ),
        scenario_codes=np.asarray(
            [item["scenario_code"] for item in trajectories], dtype="u1"
        ),
        mode_codes=np.asarray(
            [item["mode_code"] for item in trajectories], dtype="u1"
        ),
        candidate_lengths=np.asarray(
            [item["candidate_length"] for item in trajectories], dtype="<i4"
        ),
        published_lengths=np.asarray(
            [item["published_length"] for item in trajectories], dtype="<i4"
        ),
        candidate_actions=np.stack(
            [np.asarray(item["candidate_actions"], dtype="<f8")
             for item in trajectories]
        ),
        candidate_positions=np.stack(
            [np.asarray(item["candidate_positions"], dtype="<f8")
             for item in trajectories]
        ),
        published_actions=np.stack(
            [np.asarray(item["published_actions"], dtype="<f8")
             for item in trajectories]
        ),
        published_positions=np.stack(
            [np.asarray(item["published_positions"], dtype="<f8")
             for item in trajectories]
        ),
    )


def _provenance(
    archive: ValidatedPi05Archive,
    config: Pi05IntegratedReplayConfig,
    selected: Sequence[int],
) -> dict[str, Any]:
    protocol = _protocol_path()
    scene = default_panda_scene_path()
    return {
        "schema_version": INTEGRATED_REPLAY_SCHEMA,
        "scope": INTEGRATED_REPLAY_SCOPE,
        "policy": {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
            "checkpoint_executed_this_run": False,
        },
        "source": {
            "directory": str(archive.root),
            "root_manifest_sha256": archive.root_manifest_sha256,
            "transition_count": archive.transition_count,
            "response_action_hashes_verified": len(archive.response_hashes),
        },
        "protocol": {
            "relative_path": _PROTOCOL_RELATIVE.as_posix(),
            "schema_version": PROTOCOL_SCHEMA,
            "sha256": _sha256_file(protocol),
            "conformant": config.protocol_conformant,
        },
        "selection": {
            "algorithm": "sha256_rank_equal_per_task_method_v1",
            "seed": config.selection_seed,
            "chunk_count": len(selected),
            "source_row_indices": list(selected),
            "scenarios": list(config.scenarios),
            "modes": list(config.modes),
        },
        "configuration": _config_document(config),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "mujoco_version": mujoco.__version__,
            "osqp_version": osqp.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
            "mujoco_scenario_version": MUJOCO_SCENARIO_VERSION,
            "panda_scene_sha256": _sha256_file(scene),
        },
        "implementation_sha256": _implementation_hashes(),
        "claim_flags": {
            "source_policy_checkpoint_attested": True,
            "policy_checkpoint_executed_in_replay": False,
            "panda_closed_loop_executed": False,
            "task_success_evaluated": False,
            "hard_realtime_claim": False,
            "physical_safety_claim": False,
        },
    }


def execute_pi05_integrated_cpu_replay(
    source_directory: Path,
    output_directory: Path,
    config: Pi05IntegratedReplayConfig = Pi05IntegratedReplayConfig(),
) -> Path:
    """Create a paired, hash-bound CPU replay artifact."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    archive = validate_pi05_source_archive(source_directory)
    selected = select_stratified_chunks(
        archive, config.chunk_count, config.selection_seed
    )
    scenarios = mujoco_scenarios()
    robots = {
        name: MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        for name in config.scenarios
    }
    checkers = {
        name: make_integrated_task_checker(robots[name])[0]
        for name in config.scenarios
    }
    supervisors = {
        name: IntegratedPandaSupervisor(
            robots[name], checkers[name], _guard_config(config)
        )
        for name in config.scenarios
    }
    projectors = {
        name: _qp_projector(robots[name], config) for name in config.scenarios
    }
    adapters = {
        name: PandaCartesianActionAdapter(robots[name])
        for name in config.scenarios
    }

    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, np.ndarray]] = []
    for selection_index, source_index in enumerate(selected):
        query = archive.transition_queries[source_index]
        for scenario_name in config.scenarios:
            scenario = scenarios[scenario_name]
            adapter_started = perf_counter()
            adapted = adapters[scenario_name].adapt(
                archive.arrays["response_actions"][source_index],
                scenario.start,
                source=(
                    "official_pi05_attested_frozen_response:"
                    f"{archive.response_hashes[source_index]}"
                ),
                observation_sequence_id=source_index,
                inference_latency_ms=float(query["inference_latency_ms"]),
                received_at_s=(
                    1000.0 + float(query["inference_latency_ms"]) / 1000.0
                ),
            )
            adapter_latency_ms = (perf_counter() - adapter_started) * 1000.0
            for mode in config.modes:
                row, trace = _run_case(
                    archive,
                    source_index=source_index,
                    selection_index=selection_index,
                    scenario_name=scenario_name,
                    mode=mode,
                    robot=robots[scenario_name],
                    checker=checkers[scenario_name],
                    supervisor=supervisors[scenario_name],
                    projector=projectors[scenario_name],
                    adapted_chunk=adapted.chunk,
                    adapted_positions=adapted.predicted_positions,
                    adapter_latency_ms=adapter_latency_ms,
                    config=config,
                )
                rows.append(row)
                trajectories.append(trace)

    summary = _build_summary(rows, config, archive)
    output.mkdir(parents=True)
    shutil.copyfile(_protocol_path(), output / "protocol.json")
    _write_json(output / "provenance.json", _provenance(archive, config, selected))
    _write_csv(output / "per_case.csv", rows)
    _write_trajectories(output / "trajectories.npz", trajectories)
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_root_manifest(output)
    validate_pi05_integrated_cpu_replay(output, source_directory)
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == CSV_FIELDS,
            "integrated replay CSV fields do not match the schema",
        )
        for raw in reader:
            _require(set(raw) == set(CSV_FIELDS), "integrated replay CSV row is invalid")
            row: dict[str, Any] = dict(raw)
            try:
                for field in _INTEGER_FIELDS:
                    row[field] = int(raw[field])
                for field in _FLOAT_FIELDS:
                    row[field] = float(raw[field])
                for field in _BOOLEAN_FIELDS:
                    _require(
                        raw[field] in {"True", "False"},
                        f"invalid replay boolean: {field}",
                    )
                    row[field] = raw[field] == "True"
            except (TypeError, ValueError) as error:
                raise ValueError("integrated replay CSV value is invalid") from error
            numeric = [float(row[field]) for field in _INTEGER_FIELDS | _FLOAT_FIELDS]
            _require(
                bool(np.all(np.isfinite(numeric))) and all(value >= 0.0 for value in numeric),
                "integrated replay CSV numeric value is invalid",
            )
            _require(
                row["schema_version"] == INTEGRATED_REPLAY_SCHEMA
                and row["scenario"] in SCENARIOS
                and row["mode"] in MODES
                and row["status"]
                in {"execute", "hold", "verified_brake", "unrecoverable_stop"},
                "integrated replay CSV identity or status is invalid",
            )
            digest = str(row["response_action_sha256"])
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "integrated replay response hash is invalid",
            )
            rows.append(row)
    _require(bool(rows), "integrated replay CSV contains no rows")
    return rows


def _config_from_json(value: object) -> Pi05IntegratedReplayConfig:
    _require(isinstance(value, Mapping), "integrated replay configuration is missing")
    expected = {
        "chunk_count",
        "selection_seed",
        "scenarios",
        "modes",
        "response_deadline_ms",
        "software_budget_ms",
        "qp_step_budget_ms",
        "worker_timeout_s",
    }
    _require(set(value) == expected, "integrated replay configuration fields are invalid")
    try:
        return Pi05IntegratedReplayConfig(
            chunk_count=int(value["chunk_count"]),
            selection_seed=int(value["selection_seed"]),
            scenarios=tuple(str(item) for item in value["scenarios"]),
            modes=tuple(str(item) for item in value["modes"]),
            response_deadline_ms=float(value["response_deadline_ms"]),
            software_budget_ms=float(value["software_budget_ms"]),
            qp_step_budget_ms=float(value["qp_step_budget_ms"]),
            worker_timeout_s=float(value["worker_timeout_s"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("integrated replay configuration is invalid") from error


def _load_trajectories(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            _require(
                set(loaded.files) == TRAJECTORY_KEYS,
                "integrated replay trajectory fields do not match the schema",
            )
            arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("cannot load integrated replay trajectories") from error
    _require(
        arrays["schema_version"].shape == ()
        and str(arrays["schema_version"].item())
        == INTEGRATED_REPLAY_TRAJECTORY_SCHEMA,
        "integrated replay trajectory schema mismatch",
    )
    count = len(rows)
    _require(
        arrays["case_ids"].shape == (count,)
        and arrays["source_row_indices"].shape == (count,)
        and arrays["scenario_codes"].shape == (count,)
        and arrays["mode_codes"].shape == (count,)
        and arrays["candidate_lengths"].shape == (count,)
        and arrays["published_lengths"].shape == (count,),
        "integrated replay trajectory identity shapes are invalid",
    )
    horizon = arrays["candidate_actions"].shape[1]
    _require(
        arrays["candidate_actions"].shape == (count, horizon, ACTION_DIM)
        and arrays["published_actions"].shape == (count, horizon, ACTION_DIM)
        and arrays["candidate_positions"].shape
        == (count, horizon + 1, PANDA_DOF)
        and arrays["published_positions"].shape
        == (count, horizon + 1, PANDA_DOF),
        "integrated replay trajectory action/position shapes are invalid",
    )
    _require(
        arrays["source_row_indices"].dtype == np.dtype("<i4")
        and arrays["candidate_lengths"].dtype == np.dtype("<i4")
        and arrays["published_lengths"].dtype == np.dtype("<i4")
        and arrays["scenario_codes"].dtype == np.uint8
        and arrays["mode_codes"].dtype == np.uint8,
        "integrated replay trajectory integer dtypes are invalid",
    )
    for key in (
        "candidate_actions",
        "candidate_positions",
        "published_actions",
        "published_positions",
    ):
        _require(
            arrays[key].dtype == np.dtype("<f8")
            and bool(np.all(np.isfinite(arrays[key]))),
            f"integrated replay trajectory array is invalid: {key}",
        )
    return arrays


def _validate_provenance(
    provenance: Mapping[str, Any],
    config: Pi05IntegratedReplayConfig,
    archive: ValidatedPi05Archive | None,
) -> tuple[int, ...]:
    expected_fields = {
        "schema_version",
        "scope",
        "policy",
        "source",
        "protocol",
        "selection",
        "configuration",
        "environment",
        "implementation_sha256",
        "claim_flags",
    }
    _require(
        set(provenance) == expected_fields
        and provenance.get("schema_version") == INTEGRATED_REPLAY_SCHEMA
        and provenance.get("scope") == INTEGRATED_REPLAY_SCOPE,
        "integrated replay provenance schema mismatch",
    )
    _require(
        provenance.get("policy")
        == {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
            "checkpoint_executed_this_run": False,
        },
        "integrated replay policy provenance mismatch",
    )
    flags = provenance.get("claim_flags")
    _require(
        flags
        == {
            "source_policy_checkpoint_attested": True,
            "policy_checkpoint_executed_in_replay": False,
            "panda_closed_loop_executed": False,
            "task_success_evaluated": False,
            "hard_realtime_claim": False,
            "physical_safety_claim": False,
        },
        "integrated replay claim flags are invalid",
    )
    protocol = provenance.get("protocol")
    protocol_path = _protocol_path()
    _require(
        isinstance(protocol, Mapping)
        and protocol
        == {
            "relative_path": _PROTOCOL_RELATIVE.as_posix(),
            "schema_version": PROTOCOL_SCHEMA,
            "sha256": _sha256_file(protocol_path),
            "conformant": config.protocol_conformant,
        },
        "integrated replay protocol provenance mismatch",
    )
    implementation = provenance.get("implementation_sha256")
    _require(
        isinstance(implementation, Mapping)
        and dict(implementation) == _implementation_hashes(),
        "integrated replay implementation hash mismatch",
    )
    selection = provenance.get("selection")
    _require(isinstance(selection, Mapping), "integrated replay selection is missing")
    selected = selection.get("source_row_indices")
    _require(
        selection.get("algorithm") == "sha256_rank_equal_per_task_method_v1"
        and selection.get("seed") == config.selection_seed
        and selection.get("chunk_count") == config.chunk_count
        and selection.get("scenarios") == list(config.scenarios)
        and selection.get("modes") == list(config.modes)
        and isinstance(selected, list)
        and all(type(index) is int and index >= 0 for index in selected)
        and len(selected) == config.chunk_count
        and len(set(selected)) == len(selected),
        "integrated replay source selection is invalid",
    )
    source = provenance.get("source")
    _require(isinstance(source, Mapping), "integrated replay source provenance is missing")
    if archive is not None:
        _require(
            source.get("root_manifest_sha256") == archive.root_manifest_sha256
            and source.get("transition_count") == archive.transition_count
            and source.get("response_action_hashes_verified")
            == len(archive.response_hashes),
            "integrated replay source binding mismatch",
        )
        expected_selected = select_stratified_chunks(
            archive, config.chunk_count, config.selection_seed
        )
        _require(
            tuple(selected) == expected_selected,
            "integrated replay source selection cannot be reproduced",
        )
    environment = provenance.get("environment")
    _require(
        isinstance(environment, Mapping)
        and environment.get("numpy_version") == np.__version__
        and environment.get("mujoco_version") == mujoco.__version__
        and environment.get("osqp_version") == osqp.__version__
        and environment.get("menagerie_commit") == MENAGERIE_COMMIT
        and environment.get("mujoco_scenario_version")
        == MUJOCO_SCENARIO_VERSION
        and environment.get("panda_scene_sha256")
        == _sha256_file(default_panda_scene_path()),
        "integrated replay environment provenance mismatch",
    )
    return tuple(int(index) for index in selected)


def _recompute_kinematic_counts(
    robot: MuJoCoPanda,
    actions: np.ndarray,
    positions: np.ndarray,
    length: int,
    config: Pi05IntegratedReplayConfig,
) -> tuple[int, int, int]:
    guard = _guard_config(config)
    margin = guard.joint_limit_margin_rad
    candidate_positions = positions[: length + 1]
    position_violations = sum(
        bool(
            np.any(q < robot.lower_limits + margin - 1e-9)
            or np.any(q > robot.upper_limits - margin + 1e-9)
        )
        for q in candidate_positions
    )
    limits = np.asarray(guard.joint_velocity_limits_rad_s)
    velocity_violations = sum(
        bool(np.any(np.abs(action[:PANDA_DOF]) > limits + 1e-9))
        for action in actions[:length]
    )
    previous = np.zeros(PANDA_DOF)
    acceleration_violations = 0
    for action in actions[:length]:
        velocity = action[:PANDA_DOF]
        acceleration_violations += int(
            np.any(
                np.abs(velocity - previous) / guard.control_dt_s
                > guard.joint_acceleration_limit_rad_s2 + 1e-9
            )
        )
        previous = velocity
    return position_violations, velocity_violations, acceleration_violations


def _validate_rows_and_trajectories(
    rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, np.ndarray],
    config: Pi05IntegratedReplayConfig,
    selected: Sequence[int],
    archive: ValidatedPi05Archive | None,
) -> None:
    expected_cases = [
        (source_index, scenario, mode)
        for source_index in selected
        for scenario in config.scenarios
        for mode in config.modes
    ]
    actual_cases = [
        (int(row["source_row_index"]), str(row["scenario"]), str(row["mode"]))
        for row in rows
    ]
    _require(
        actual_cases == expected_cases
        and len({str(row["case_id"]) for row in rows}) == len(rows),
        "integrated replay paired matrix is incomplete or out of order",
    )
    _require(
        trajectories["case_ids"].tolist()
        == [str(row["case_id"]) for row in rows],
        "integrated replay row/trajectory identities disagree",
    )
    horizon = trajectories["candidate_actions"].shape[1]
    robots = {
        name: MuJoCoPanda.create(obstacles=mujoco_scenarios()[name].obstacles)
        for name in config.scenarios
    }
    checkers = {
        name: make_integrated_task_checker(robots[name])[0]
        for name in config.scenarios
    }
    raw_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]] = {}
    selection_order = {
        source_index: index for index, source_index in enumerate(selected)
    }
    for index, row in enumerate(rows):
        source_index = int(row["source_row_index"])
        scenario_name = str(row["scenario"])
        mode = str(row["mode"])
        scenario = mujoco_scenarios()[scenario_name]
        robot = robots[scenario_name]
        checker = checkers[scenario_name]
        candidate_length = int(trajectories["candidate_lengths"][index])
        published_length = int(trajectories["published_lengths"][index])
        actions = trajectories["candidate_actions"][index]
        positions = trajectories["candidate_positions"][index]
        published_actions = trajectories["published_actions"][index]
        published_positions = trajectories["published_positions"][index]
        _require(
            int(trajectories["source_row_indices"][index]) == source_index
            and int(trajectories["scenario_codes"][index])
            == SCENARIOS.index(scenario_name)
            and int(trajectories["mode_codes"][index]) == MODES.index(mode)
            and int(row["selection_index"]) == selection_order[source_index]
            and row["case_id"]
            == f"row_{source_index:05d}__{scenario_name}__{mode}",
            "integrated replay case identity mismatch",
        )
        _require(
            0 <= candidate_length <= horizon
            and published_length in {0, horizon}
            and candidate_length == int(row["candidate_action_count"])
            and published_length == int(row["published_action_count"])
            and bool(row["candidate_complete"])
            == (candidate_length == horizon),
            "integrated replay trajectory lengths are invalid",
        )
        if candidate_length:
            integrated = positions[0] + np.cumsum(
                actions[:candidate_length, :PANDA_DOF]
                * CartesianAdapterConfig().control_dt_s,
                axis=0,
            )
            _require(
                np.allclose(
                    positions[1 : candidate_length + 1],
                    integrated,
                    atol=1e-10,
                    rtol=0.0,
                )
                and np.allclose(
                    positions[candidate_length + 1 :],
                    positions[candidate_length],
                    atol=0.0,
                    rtol=0.0,
                ),
                "integrated replay candidate trajectory does not integrate",
            )
        _require(
            np.allclose(positions[0], scenario.start, atol=1e-12, rtol=0.0),
            "integrated replay candidate start state mismatch",
        )
        if published_length == 0:
            _require(
                not np.any(published_actions)
                and np.allclose(
                    published_positions,
                    scenario.start[None, :],
                    atol=0.0,
                    rtol=0.0,
                ),
                "integrated replay hold exposed policy motion",
            )
        else:
            _require(
                candidate_length == horizon
                and np.array_equal(published_actions, actions)
                and np.array_equal(published_positions, positions),
                "integrated replay execute did not publish one complete candidate",
            )
        executable = row["status"] == "execute"
        _require(
            bool(row["policy_actions_executable"]) == executable
            and (published_length == horizon) == executable
            and bool(row["all_or_none_publication"])
            and not bool(row["partial_prefix_exposed"])
            and bool(row["integrated_atomic_gate_used"])
            == (mode == "full_assurance")
            and bool(row["assurance_worker_thread_separate"])
            == (mode == "full_assurance"),
            "integrated replay publication contract is invalid",
        )
        _require(
            bool(row["software_budget_exceeded"])
            == (
                float(row["mode_latency_ms"]) > config.software_budget_ms
                or row["failure_stage"] == "supervision_budget"
            )
            and bool(row["response_deadline_exceeded"])
            == (
                float(row["activation_response_age_ms"])
                > config.response_deadline_ms
                or row["failure_stage"] == "deadline"
            ),
            "integrated replay timing flags are inconsistent",
        )
        counts = _recompute_kinematic_counts(
            robot, actions, positions, candidate_length, config
        )
        _require(
            counts
            == (
                int(row["joint_position_violation_steps"]),
                int(row["joint_velocity_violation_steps"]),
                int(row["joint_acceleration_violation_steps"]),
            ),
            "integrated replay kinematic predicates do not recompute",
        )
        audit = _audit_candidate(
            robot, checker, actions, positions, candidate_length, config
        )
        _require(
            int(row["continuous_edges_checked"]) == audit.edges_checked
            and int(row["continuous_unsafe_edges"]) == audit.unsafe_edges
            and int(row["continuous_indeterminate_edges"])
            == audit.indeterminate_edges
            and int(row["braking_boundaries_checked"])
            == audit.braking_boundaries
            and int(row["braking_invalid_boundaries"])
            == audit.invalid_braking_boundaries,
            "integrated replay collision/braking predicates do not recompute",
        )
        all_satisfied = bool(candidate_length == horizon and audit.all_satisfied)
        _require(
            bool(row["all_registered_constraints_satisfied"])
            == all_satisfied
            and bool(row["unsafe_plan_published"])
            == (executable and not all_satisfied),
            "integrated replay safety aggregate is inconsistent",
        )
        start_hand = robot.hand_position(scenario.start)
        candidate_hand = robot.hand_position(positions[candidate_length])
        published_hand = robot.hand_position(published_positions[published_length])
        candidate_displacement = float(np.linalg.norm(candidate_hand - start_hand))
        published_displacement = float(np.linalg.norm(published_hand - start_hand))
        _require(
            np.isclose(
                float(row["candidate_hand_displacement_m"]),
                candidate_displacement,
                atol=1e-12,
                rtol=0.0,
            )
            and np.isclose(
                float(row["published_hand_displacement_m"]),
                published_displacement,
                atol=1e-12,
                rtol=0.0,
            ),
            "integrated replay hand displacement does not recompute",
        )
        if archive is not None:
            query = archive.transition_queries[source_index]
            _require(
                row["response_action_sha256"]
                == archive.response_hashes[source_index]
                and row["task_suite"] == query["task_suite"]
                and int(row["task_id"]) == int(query["task_id"])
                and row["method"] == query["method"]
                and row["pair_id"] == query["pair_id"]
                and row["episode_id"] == query["episode_id"]
                and int(row["episode_index"]) == int(query["episode_index"])
                and int(row["query_index"]) == int(query["query_index"])
                and np.isclose(
                    float(row["source_inference_latency_ms"]),
                    float(query["inference_latency_ms"]),
                    atol=0.0,
                    rtol=0.0,
                ),
                "integrated replay row does not bind to the source response",
            )
            cache_key = (source_index, scenario_name)
            if cache_key not in raw_cache:
                adapted = PandaCartesianActionAdapter(robot).adapt(
                    archive.arrays["response_actions"][source_index],
                    scenario.start,
                    source="validator_recomputed_attested_frozen_response",
                    observation_sequence_id=source_index,
                    inference_latency_ms=float(query["inference_latency_ms"]),
                )
                raw_end = robot.hand_position(adapted.predicted_positions[-1])
                raw_cache[cache_key] = (
                    adapted.chunk.actions,
                    adapted.predicted_positions,
                    float(np.linalg.norm(raw_end - start_hand)),
                )
            raw_actions, raw_positions, raw_displacement = raw_cache[cache_key]
            _require(
                np.isclose(
                    float(row["raw_hand_displacement_m"]),
                    raw_displacement,
                    atol=1e-12,
                    rtol=0.0,
                ),
                "integrated replay raw command-retention proxy mismatch",
            )
            if mode == "direct_dispatch":
                _require(
                    np.array_equal(actions, raw_actions)
                    and np.array_equal(positions, raw_positions),
                    "integrated replay direct baseline changed the adapted input",
                )
        raw_displacement = float(row["raw_hand_displacement_m"])
        expected_retention = (
            0.0
            if raw_displacement <= 1e-12
            else published_displacement / raw_displacement
        )
        _require(
            np.isclose(
                float(row["published_motion_retention_ratio"]),
                expected_retention,
                atol=1e-12,
                rtol=0.0,
            ),
            "integrated replay command-retention ratio mismatch",
        )


def validate_pi05_integrated_cpu_replay(
    directory: Path,
    source_directory: Path | None = None,
) -> dict[str, Any]:
    """Recompute a saved CPU matrix and optionally bind it to source bytes."""

    root = directory.resolve()
    _require(root.is_dir(), f"integrated replay directory not found: {root}")
    manifest = _validate_root_manifest(root)
    provenance_raw = _load_json(root / "provenance.json")
    summary = _load_json(root / "summary.json")
    protocol = _load_json(root / "protocol.json")
    _require(isinstance(provenance_raw, Mapping), "integrated replay provenance is invalid")
    _require(isinstance(summary, Mapping), "integrated replay summary is invalid")
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("schema_version") == PROTOCOL_SCHEMA
        and protocol.get("status")
        == "protocol_frozen_before_runner_implementation_and_scored_replay"
        and _sha256_file(root / "protocol.json") == _sha256_file(_protocol_path()),
        "integrated replay frozen protocol mismatch",
    )
    config = _config_from_json(provenance_raw.get("configuration"))
    _require(
        summary.get("configuration") == provenance_raw.get("configuration"),
        "integrated replay configuration documents disagree",
    )
    archive = (
        None
        if source_directory is None
        else validate_pi05_source_archive(source_directory)
    )
    selected = _validate_provenance(provenance_raw, config, archive)
    rows = _read_csv(root / "per_case.csv")
    trajectories = _load_trajectories(root / "trajectories.npz", rows)
    _require(
        len(rows)
        == config.chunk_count * len(config.scenarios) * len(config.modes),
        "integrated replay matrix row count is invalid",
    )
    _validate_rows_and_trajectories(
        rows, trajectories, config, selected, archive
    )
    if archive is None:
        source = provenance_raw["source"]
        source_stub = type(
            "SourceStub",
            (),
            {
                "transition_count": int(source["transition_count"]),
                "response_hashes": (None,)
                * int(source["response_action_hashes_verified"]),
            },
        )()
        expected_summary = _build_summary(rows, config, source_stub)
    else:
        expected_summary = _build_summary(rows, config, archive)
    _require(
        summary == expected_summary,
        "integrated replay summary cannot be reproduced from CSV",
    )
    _require(
        (root / "summary.md").read_text(encoding="utf-8")
        == _summary_markdown(summary),
        "integrated replay Markdown summary is not reproducible",
    )
    checks = [
        "recursive_manifest_inventory_sizes_and_hashes",
        "frozen_protocol_identity",
        "implementation_and_scene_hashes",
        "complete_paired_response_scenario_mode_matrix",
        "all_or_none_published_trajectory_contract",
        "kinematic_predicates_recomputed",
        "continuous_collision_predicates_recomputed",
        "dynamics_braking_predicates_recomputed",
        "aggregates_and_markdown_reproduced",
    ]
    if archive is not None:
        checks.extend(
            [
                "source_archive_reverified",
                "response_hashes_and_metadata_rebound",
                "direct_adapter_output_recomputed",
            ]
        )
    return {
        "valid": True,
        "scope": INTEGRATED_REPLAY_SCOPE,
        "cases": len(rows),
        "selected_responses": config.chunk_count,
        "protocol_conformant": config.protocol_conformant,
        "go_no_go": summary["go_no_go"]["decision"],
        "manifest_files_sha256": manifest["files_sha256"],
        "source_reverified": archive is not None,
        "checks": checks,
    }


__all__ = [
    "CSV_FIELDS",
    "INTEGRATED_REPLAY_SCHEMA",
    "MODES",
    "Pi05IntegratedReplayConfig",
    "execute_pi05_integrated_cpu_replay",
    "validate_pi05_integrated_cpu_replay",
]
