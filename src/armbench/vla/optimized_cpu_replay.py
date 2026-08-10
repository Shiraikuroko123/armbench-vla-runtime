"""Manifest-bound CPU audit for the optimized pi0.5-to-Panda runtime."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import platform
import shutil
import threading
import time
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import osqp

from armbench.mujoco_sim.broad_phase_continuous_collision import (
    BroadPhaseContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MENAGERIE_COMMIT, MuJoCoPanda
from armbench.mujoco_sim.persistent_dynamics_braking import (
    PersistentDynamicsBrakingValidator,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_async import (
    AtomicPandaPlanGate,
    LatestIntegratedPandaWorker,
)
from armbench.vla.integrated_panda_guard import (
    PANDA_DOF,
    IntegratedPandaGuardConfig,
)
from armbench.vla.integrated_panda_task import (
    FIXED_GRIPPER_ALLOWED_BODY_PAIR,
)
from armbench.vla.optimized_integrated_panda_guard import (
    OptimizedIntegratedPandaSupervisor,
)
from armbench.vla.pi05_integrated_replay import (
    _load_trajectories,
    _pad_candidate,
    _published_positions,
    _read_csv,
)
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    write_json,
)
from armbench.vla.types import ActionChunk


SCHEMA = "armbench.pi05_optimized_cpu_replay.v1"
SUMMARY_SCHEMA = "armbench.pi05_optimized_cpu_replay_summary.v1"
TRAJECTORY_SCHEMA = "armbench.pi05_optimized_cpu_replay_trajectory.v1"
MANIFEST_SCHEMA = "armbench.pi05_optimized_cpu_replay_manifest.v1"
SCOPE = "frozen_pi05_panda_optimized_atomic_cpu_engineering_audit"
PROTOCOL_RELATIVE = Path(
    "docs/research/pi05_optimized_cpu_replay_protocol_20260810.json"
)
PROTOCOL_SCHEMA = "armbench.pi05_optimized_cpu_replay_protocol.v1"
PROFILE_IDS = ("operational_20ms", "diagnostic_100ms")
ACTION_DIM = 8
HORIZON = 10

CSV_FIELDS = (
    "schema_version",
    "case_id",
    "source_case_id",
    "selection_index",
    "source_row_index",
    "scenario",
    "profile",
    "response_action_sha256",
    "input_action_sha256",
    "supervision_budget_ms",
    "base_response_age_ms",
    "activation_response_age_ms",
    "supervisor_latency_ms",
    "assurance_worker_latency_ms",
    "qp_latency_ms",
    "continuous_collision_latency_ms",
    "dynamics_braking_latency_ms",
    "fallback_brake_latency_ms",
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
    "assurance_worker_thread_separate",
    "fallback_validated",
    "qp_feasible",
    "qp_intervention_steps",
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
    "software_budget_exceeded",
    "response_deadline_exceeded",
    "broad_phase_pair_tests",
    "broad_phase_pruned_pairs",
    "broad_phase_exact_pair_evaluations",
    "broad_phase_prune_rate",
    "safe_configuration_cache_hits",
    "persistent_qp_solve_calls",
    "persistent_brake_validation_calls",
)

_BOOLEAN_FIELDS = {
    "candidate_complete",
    "policy_actions_executable",
    "partial_prefix_exposed",
    "all_or_none_publication",
    "assurance_worker_thread_separate",
    "fallback_validated",
    "qp_feasible",
    "all_registered_constraints_satisfied",
    "unsafe_plan_published",
    "software_budget_exceeded",
    "response_deadline_exceeded",
}
_INTEGER_FIELDS = {
    "selection_index",
    "source_row_index",
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
    "broad_phase_pair_tests",
    "broad_phase_pruned_pairs",
    "broad_phase_exact_pair_evaluations",
    "safe_configuration_cache_hits",
    "persistent_qp_solve_calls",
    "persistent_brake_validation_calls",
}
_FLOAT_FIELDS = {
    "supervision_budget_ms",
    "base_response_age_ms",
    "activation_response_age_ms",
    "supervisor_latency_ms",
    "assurance_worker_latency_ms",
    "qp_latency_ms",
    "continuous_collision_latency_ms",
    "dynamics_braking_latency_ms",
    "fallback_brake_latency_ms",
    "broad_phase_prune_rate",
}
_ARTIFACT_FILES = {
    "per_case.csv",
    "protocol.json",
    "provenance.json",
    "summary.json",
    "summary.md",
    "trajectories.npz",
}
_CLAIM_BOUNDARY = [
    "This is an engineering audit fixed after optimization profiling, not a preregistered inferential study.",
    "Frozen official responses are replayed; the pi0.5 checkpoint is not executed.",
    "LIBERO actions are cross-controller Panda inputs, not native Panda outputs.",
    "Independent cases do not provide closed-loop policy feedback or task success.",
    "Measured Python timing is not a hard-real-time guarantee.",
    "MuJoCo checks are not physical-robot safety certification.",
]


@dataclass(frozen=True)
class OptimizedCPUReplayConfig:
    chunk_count: int = 30
    operational_budget_ms: float = 20.0
    diagnostic_budget_ms: float = 100.0
    response_deadline_ms: float = 200.0
    qp_step_budget_ms: float = 5.0
    worker_timeout_s: float = 30.0
    poll_interval_s: float = 0.0005

    def __post_init__(self) -> None:
        if type(self.chunk_count) is not int or self.chunk_count <= 0:
            raise ValueError("chunk_count must be a positive integer")
        for value, label in (
            (self.operational_budget_ms, "operational_budget_ms"),
            (self.diagnostic_budget_ms, "diagnostic_budget_ms"),
            (self.response_deadline_ms, "response_deadline_ms"),
            (self.qp_step_budget_ms, "qp_step_budget_ms"),
            (self.worker_timeout_s, "worker_timeout_s"),
            (self.poll_interval_s, "poll_interval_s"),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")
        if self.diagnostic_budget_ms <= self.operational_budget_ms:
            raise ValueError("diagnostic budget must exceed operational budget")

    @property
    def protocol_conformant(self) -> bool:
        return self == OptimizedCPUReplayConfig()


@dataclass(frozen=True)
class _Audit:
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
            self.edges_checked == HORIZON
            and self.braking_boundaries == HORIZON
            and self.position_violations == 0
            and self.velocity_violations == 0
            and self.acceleration_violations == 0
            and self.unsafe_edges == 0
            and self.indeterminate_edges == 0
            and self.invalid_braking_boundaries == 0
        )


@dataclass
class _System:
    robot: MuJoCoPanda
    checker: BroadPhaseContinuousMuJoCoCollisionChecker
    supervisor: OptimizedIntegratedPandaSupervisor
    worker: LatestIntegratedPandaWorker
    gate: AtomicPandaPlanGate
    worker_startup_ms: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return _project_root() / PROTOCOL_RELATIVE


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return sha256(array.tobytes(order="C")).hexdigest()


def _input_inventory(root: Path) -> tuple[str, str]:
    manifest_path = root / "manifest.json"
    manifest = strict_json_load(manifest_path)
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, list):
        raise ValueError("input artifact manifest is invalid")
    legacy = set(manifest) == {"schema_version", "files", "files_sha256"}
    current = set(manifest) == {"schema_version", "files", "inventory_sha256"}
    aggregate_field = "files_sha256" if legacy else "inventory_sha256"
    size_field = "bytes" if legacy else "size_bytes"
    if not (legacy or current) or not is_sha256(manifest.get(aggregate_field)):
        raise ValueError("input artifact manifest is invalid")
    declared: set[str] = set()
    for item in files:
        if not (
            isinstance(item, Mapping)
            and set(item) == {"path", size_field, "sha256"}
            and isinstance(item["path"], str)
            and type(item[size_field]) is int
            and item[size_field] >= 0
            and is_sha256(item["sha256"])
        ):
            raise ValueError("input artifact manifest entry is invalid")
        relative = Path(item["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != item[size_field]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"input artifact manifest mismatch: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if declared != actual or manifest[aggregate_field] != sha256_bytes(
        canonical_json(files)
    ):
        raise ValueError("input artifact inventory mismatch")
    return str(manifest[aggregate_field]), sha256_file(manifest_path)


def _input_rows_and_arrays(
    root: Path, chunk_count: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = _read_csv(root / "per_case.csv")
    arrays = _load_trajectories(root / "trajectories.npz", rows)
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if row["mode"] == "direct_dispatch"
        and int(row["selection_index"]) < chunk_count
    ]
    if len(selected) != chunk_count * 3:
        raise ValueError("input artifact lacks the requested direct-dispatch rows")
    indices = np.asarray([index for index, _ in selected], dtype=int)
    direct_arrays = {key: value[indices] for key, value in arrays.items() if value.ndim}
    return [row for _, row in selected], direct_arrays


def _checker(robot: MuJoCoPanda) -> BroadPhaseContinuousMuJoCoCollisionChecker:
    checker = BroadPhaseContinuousMuJoCoCollisionChecker(robot)
    retained = []
    for pair in checker.pairs:
        bodies = frozenset(
            (
                robot.body_name_for_geom(pair.geom1),
                robot.body_name_for_geom(pair.geom2),
            )
        )
        if pair.kind == "self_collision" and bodies == FIXED_GRIPPER_ALLOWED_BODY_PAIR:
            continue
        retained.append(pair)
    checker.set_pairs(tuple(retained))
    return checker


def _guard_config(config: OptimizedCPUReplayConfig, budget_ms: float) -> IntegratedPandaGuardConfig:
    return IntegratedPandaGuardConfig(
        response_deadline_ms=config.response_deadline_ms,
        supervision_budget_ms=budget_ms,
        qp_step_budget_ms=config.qp_step_budget_ms,
    )


def _make_system(
    scenario_name: str,
    budget_ms: float,
    config: OptimizedCPUReplayConfig,
) -> _System:
    scenario = mujoco_scenarios()[scenario_name]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = _checker(robot)
    supervisor = OptimizedIntegratedPandaSupervisor(
        robot, checker, _guard_config(config, budget_ms)
    )
    started = time.perf_counter()
    worker = LatestIntegratedPandaWorker(supervisor)
    deadline = time.monotonic() + 5.0
    while worker.metrics()["worker_thread_id"] is None:
        if time.monotonic() >= deadline:
            worker.close()
            raise TimeoutError("optimized assurance worker did not start")
        time.sleep(config.poll_interval_s)
    startup_ms = (time.perf_counter() - started) * 1000.0
    return _System(
        robot=robot,
        checker=checker,
        supervisor=supervisor,
        worker=worker,
        gate=AtomicPandaPlanGate(supervisor),
        worker_startup_ms=startup_ms,
    )


def _wait_outcome(system: _System, config: OptimizedCPUReplayConfig) -> object:
    deadline = time.monotonic() + config.worker_timeout_s
    while time.monotonic() < deadline:
        outcomes = system.worker.drain()
        if outcomes:
            return outcomes[0]
        time.sleep(config.poll_interval_s)
    raise TimeoutError("optimized assurance worker timed out")


def _audit_candidate(
    robot: MuJoCoPanda,
    checker: BroadPhaseContinuousMuJoCoCollisionChecker,
    braking: PersistentDynamicsBrakingValidator,
    actions: np.ndarray,
    positions: np.ndarray,
    length: int,
    guard: IntegratedPandaGuardConfig,
) -> _Audit:
    checker.clear_safe_configuration_cache()
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
        acceleration_violations += int(
            np.any(
                np.abs(velocity - previous) / guard.control_dt_s
                > guard.joint_acceleration_limit_rad_s2 + 1e-9
            )
        )
        previous = velocity
    unsafe_edges = 0
    indeterminate = 0
    for q_before, q_after in zip(
        candidate_positions[:-1], candidate_positions[1:]
    ):
        certificate = checker.edge_certificate(q_before, q_after)
        unsafe_edges += int(not certificate.certified_safe)
        indeterminate += int(certificate.status == "indeterminate")
    invalid_brakes = sum(
        not braking.validate(q_after, action[:PANDA_DOF]).validated
        for q_after, action in zip(candidate_positions[1:], candidate_actions)
    )
    return _Audit(
        position_violations=position_violations,
        velocity_violations=velocity_violations,
        acceleration_violations=acceleration_violations,
        edges_checked=length,
        unsafe_edges=unsafe_edges,
        indeterminate_edges=indeterminate,
        braking_boundaries=length,
        invalid_braking_boundaries=int(invalid_brakes),
    )


def _profile_specs(config: OptimizedCPUReplayConfig) -> tuple[tuple[str, float], ...]:
    return (
        ("operational_20ms", config.operational_budget_ms),
        ("diagnostic_100ms", config.diagnostic_budget_ms),
    )


def _run_case(
    source_row: Mapping[str, Any],
    source_actions: np.ndarray,
    source_positions: np.ndarray,
    *,
    profile: str,
    budget_ms: float,
    system: _System,
    config: OptimizedCPUReplayConfig,
    audit_system: tuple[
        MuJoCoPanda,
        BroadPhaseContinuousMuJoCoCollisionChecker,
        PersistentDynamicsBrakingValidator,
    ],
    audit_cache: dict[bytes, _Audit],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    system.checker.clear_safe_configuration_cache()
    system.checker.reset_metrics()
    before = system.supervisor.optimization_metrics()
    generation = system.gate.reset()
    source_index = int(source_row["source_row_index"])
    chunk = ActionChunk(
        actions=source_actions,
        source="manifest_bound_frozen_pi05_direct_candidate",
        observation_sequence_id=source_index,
        inference_latency_ms=0.0,
    )
    base_age = float(source_row["source_inference_latency_ms"]) + float(
        source_row["adapter_latency_ms"]
    )
    system.worker.submit(
        generation=generation,
        observation_sequence_id=source_index,
        q=source_positions[0],
        qvel=np.zeros(PANDA_DOF),
        observed_q=source_positions[0],
        response_age_ms=base_age,
        chunk=chunk,
    )
    outcome = _wait_outcome(system, config)
    atomic = system.gate.commit(outcome, q_now=source_positions[0])
    if outcome.decision is None or outcome.decision.qp_result is None:
        candidate_actions = np.zeros((HORIZON, ACTION_DIM), dtype=float)
        candidate_positions = np.repeat(
            source_positions[0][None, :], HORIZON + 1, axis=0
        )
        candidate_length = 0
        qp_feasible = False
        qp_interventions = 0
        supervisor_latency = outcome.worker_latency_ms
        stages: Mapping[str, float] = {}
        failure_stage = "worker" if outcome.decision is None else (
            outcome.decision.failure_stage or ""
        )
    else:
        decision = outcome.decision
        projection = decision.qp_result
        candidate_actions, candidate_positions, candidate_length = _pad_candidate(
            projection.projected_actions,
            projection.predicted_positions,
            horizon=HORIZON,
            q_start=source_positions[0],
        )
        qp_feasible = projection.feasible
        qp_interventions = projection.intervention_steps
        supervisor_latency = decision.total_latency_ms
        stages = decision.stage_latencies_ms
        failure_stage = decision.failure_stage or ""
    published_length = len(atomic.policy_actions)
    published_actions = np.zeros((HORIZON, ACTION_DIM), dtype=float)
    if published_length:
        published_actions[:published_length] = atomic.policy_actions
    published_positions = _published_positions(
        source_positions[0],
        published_actions,
        published_length,
        system.supervisor.config.control_dt_s,
    )
    audit_key = canonical_json(
        {
            "scenario": source_row["scenario"],
            "length": candidate_length,
            "actions": _array_sha256(candidate_actions),
            "positions": _array_sha256(candidate_positions),
        }
    )
    if audit_key not in audit_cache:
        audit_robot, audit_checker, audit_braking = audit_system
        audit_cache[audit_key] = _audit_candidate(
            audit_robot,
            audit_checker,
            audit_braking,
            candidate_actions,
            candidate_positions,
            candidate_length,
            _guard_config(config, config.diagnostic_budget_ms),
        )
    audit = audit_cache[audit_key]
    after = system.supervisor.optimization_metrics()
    candidate_complete = candidate_length == HORIZON
    partial = published_length not in {0, HORIZON}
    all_constraints = bool(candidate_complete and audit.all_satisfied)
    unsafe_published = bool(atomic.status == "execute" and not all_constraints)
    pair_tests = int(after["broad_phase_pair_tests"])
    pruned = int(after["broad_phase_pruned_pairs"])
    case_id = f"{source_row['case_id']}__{profile}"
    row = {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "source_case_id": source_row["case_id"],
        "selection_index": int(source_row["selection_index"]),
        "source_row_index": source_index,
        "scenario": source_row["scenario"],
        "profile": profile,
        "response_action_sha256": source_row["response_action_sha256"],
        "input_action_sha256": _array_sha256(source_actions),
        "supervision_budget_ms": budget_ms,
        "base_response_age_ms": base_age,
        "activation_response_age_ms": atomic.response_age_ms,
        "supervisor_latency_ms": supervisor_latency,
        "assurance_worker_latency_ms": outcome.worker_latency_ms,
        "qp_latency_ms": float(stages.get("qp_projection", 0.0)),
        "continuous_collision_latency_ms": float(
            stages.get("continuous_collision", 0.0)
        ),
        "dynamics_braking_latency_ms": float(
            stages.get("dynamics_braking", 0.0)
        ),
        "fallback_brake_latency_ms": float(stages.get("fallback_brake", 0.0)),
        "status": atomic.status,
        "reason": atomic.reason,
        "supervisor_status": atomic.supervisor_status or "",
        "failure_stage": failure_stage,
        "candidate_complete": candidate_complete,
        "candidate_action_count": candidate_length,
        "published_action_count": published_length,
        "policy_actions_executable": atomic.policy_actions_executable,
        "partial_prefix_exposed": partial,
        "all_or_none_publication": not partial,
        "assurance_worker_thread_separate": (
            outcome.worker_thread_id != threading.get_ident()
        ),
        "fallback_validated": atomic.fallback_validated,
        "qp_feasible": qp_feasible,
        "qp_intervention_steps": qp_interventions,
        "joint_position_violation_steps": audit.position_violations,
        "joint_velocity_violation_steps": audit.velocity_violations,
        "joint_acceleration_violation_steps": audit.acceleration_violations,
        "continuous_edges_checked": audit.edges_checked,
        "continuous_unsafe_edges": audit.unsafe_edges,
        "continuous_indeterminate_edges": audit.indeterminate_edges,
        "braking_boundaries_checked": audit.braking_boundaries,
        "braking_invalid_boundaries": audit.invalid_braking_boundaries,
        "all_registered_constraints_satisfied": all_constraints,
        "unsafe_plan_published": unsafe_published,
        "software_budget_exceeded": bool(
            failure_stage == "supervision_budget"
            or supervisor_latency > budget_ms
        ),
        "response_deadline_exceeded": bool(
            failure_stage == "deadline"
            or atomic.response_age_ms > config.response_deadline_ms
        ),
        "broad_phase_pair_tests": pair_tests,
        "broad_phase_pruned_pairs": pruned,
        "broad_phase_exact_pair_evaluations": int(
            after["broad_phase_exact_pair_evaluations"]
        ),
        "broad_phase_prune_rate": 0.0 if pair_tests == 0 else pruned / pair_tests,
        "safe_configuration_cache_hits": int(
            after["safe_configuration_cache_hits"]
        ),
        "persistent_qp_solve_calls": int(
            after["persistent_qp_solve_calls"]
            - before["persistent_qp_solve_calls"]
        ),
        "persistent_brake_validation_calls": int(
            after["persistent_brake_validation_calls"]
            - before["persistent_brake_validation_calls"]
        ),
    }
    trace = {
        "case_id": np.asarray(case_id),
        "source_row_index": np.asarray(source_index, dtype="<i4"),
        "scenario_code": np.asarray(
            ("free_space", "single_block", "narrow_gate").index(
                str(source_row["scenario"])
            ),
            dtype="u1",
        ),
        "profile_code": np.asarray(PROFILE_IDS.index(profile), dtype="u1"),
        "candidate_length": np.asarray(candidate_length, dtype="<i4"),
        "published_length": np.asarray(published_length, dtype="<i4"),
        "candidate_actions": candidate_actions,
        "candidate_positions": candidate_positions,
        "published_actions": published_actions,
        "published_positions": published_positions,
    }
    return row, trace


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty optimized replay rows")
    supervisor = [float(row["supervisor_latency_ms"]) for row in rows]
    worker = [float(row["assurance_worker_latency_ms"]) for row in rows]
    pair_tests = sum(int(row["broad_phase_pair_tests"]) for row in rows)
    pruned = sum(int(row["broad_phase_pruned_pairs"]) for row in rows)
    return {
        "cases": len(rows),
        "execute": sum(row["status"] == "execute" for row in rows),
        "hold": sum(row["status"] == "hold" for row in rows),
        "verified_brake": sum(row["status"] == "verified_brake" for row in rows),
        "unrecoverable_stop": sum(
            row["status"] == "unrecoverable_stop" for row in rows
        ),
        "constraint_satisfied_candidates": sum(
            bool(row["all_registered_constraints_satisfied"]) for row in rows
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
        "p50_supervisor_latency_ms": float(np.percentile(supervisor, 50)),
        "p95_supervisor_latency_ms": float(np.percentile(supervisor, 95)),
        "maximum_supervisor_latency_ms": max(supervisor),
        "p50_worker_latency_ms": float(np.percentile(worker, 50)),
        "p95_worker_latency_ms": float(np.percentile(worker, 95)),
        "maximum_worker_latency_ms": max(worker),
        "broad_phase_pair_tests": pair_tests,
        "broad_phase_exact_pair_evaluations": sum(
            int(row["broad_phase_exact_pair_evaluations"]) for row in rows
        ),
        "broad_phase_prune_rate": 0.0 if pair_tests == 0 else pruned / pair_tests,
        "safe_configuration_cache_hits": sum(
            int(row["safe_configuration_cache_hits"]) for row in rows
        ),
    }


def _summary(
    rows: Sequence[Mapping[str, Any]], config: OptimizedCPUReplayConfig
) -> dict[str, Any]:
    by_profile = {
        profile: _aggregate([row for row in rows if row["profile"] == profile])
        for profile in PROFILE_IDS
    }
    operational = by_profile["operational_20ms"]
    go = bool(
        operational["execute"] > 0
        and operational["unsafe_plans_published"] == 0
        and operational["partial_prefixes_exposed"] == 0
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "scope": SCOPE,
        "configuration": asdict(config),
        "protocol_conformant": config.protocol_conformant,
        "overall": _aggregate(rows),
        "by_profile": by_profile,
        "go_no_go": {
            "decision": "go" if go else "no_go",
            "operational_complete_plans": operational["execute"],
            "operational_unsafe_plans_published": operational[
                "unsafe_plans_published"
            ],
            "partial_prefixes_exposed": operational[
                "partial_prefixes_exposed"
            ],
            "diagnostic_cannot_override_operational_decision": True,
        },
        "claim_boundary": list(_CLAIM_BOUNDARY),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Optimized pi0.5-to-Panda CPU replay",
        "",
        "Frozen responses are replayed through the optimized atomic runtime; the checkpoint is not executed.",
        "",
        "| Profile | Cases | Execute | Constraint-safe | Unsafe published | Budget misses | P95 supervisor | P95 worker |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in PROFILE_IDS:
        values = summary["by_profile"][profile]
        lines.append(
            f"| `{profile}` | {values['cases']} | {values['execute']} | "
            f"{values['constraint_satisfied_candidates']} | "
            f"{values['unsafe_plans_published']} | "
            f"{values['software_budget_exceeded']} | "
            f"{values['p95_supervisor_latency_ms']:.3f} ms | "
            f"{values['p95_worker_latency_ms']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            f"Operational go/no-go: **{summary['go_no_go']['decision']}**.",
            "",
            "The 100 ms profile is diagnostic only and cannot change the 20 ms decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_trajectories(path: Path, traces: Sequence[Mapping[str, np.ndarray]]) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(TRAJECTORY_SCHEMA),
        case_ids=np.asarray([str(trace["case_id"]) for trace in traces]),
        source_row_indices=np.asarray(
            [trace["source_row_index"] for trace in traces], dtype="<i4"
        ),
        scenario_codes=np.asarray(
            [trace["scenario_code"] for trace in traces], dtype="u1"
        ),
        profile_codes=np.asarray(
            [trace["profile_code"] for trace in traces], dtype="u1"
        ),
        candidate_lengths=np.asarray(
            [trace["candidate_length"] for trace in traces], dtype="<i4"
        ),
        published_lengths=np.asarray(
            [trace["published_length"] for trace in traces], dtype="<i4"
        ),
        candidate_actions=np.stack([trace["candidate_actions"] for trace in traces]),
        candidate_positions=np.stack(
            [trace["candidate_positions"] for trace in traces]
        ),
        published_actions=np.stack([trace["published_actions"] for trace in traces]),
        published_positions=np.stack(
            [trace["published_positions"] for trace in traces]
        ),
    )


def _implementation_paths() -> dict[str, Path]:
    root = _project_root()
    paths = (
        "src/armbench/vla/optimized_cpu_replay.py",
        "src/armbench/vla/optimized_integrated_panda_guard.py",
        "src/armbench/vla/persistent_qp_projection.py",
        "src/armbench/vla/integrated_panda_async.py",
        "src/armbench/mujoco_sim/broad_phase_continuous_collision.py",
        "src/armbench/mujoco_sim/persistent_dynamics_braking.py",
        "src/armbench/mujoco_sim/continuous_collision.py",
        "src/armbench/mujoco_sim/dynamics_braking.py",
        PROTOCOL_RELATIVE.as_posix(),
    )
    return {relative: root / relative for relative in paths}


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _write_manifest(root: Path) -> None:
    files = _inventory(root)
    write_json(
        root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA,
            "files": files,
            "inventory_sha256": sha256_bytes(canonical_json(files)),
        },
    )


def execute_optimized_cpu_replay(
    input_directory: Path,
    output_directory: Path,
    config: OptimizedCPUReplayConfig = OptimizedCPUReplayConfig(),
) -> Path:
    """Run the cold-case, persistent-worker optimized CPU audit."""

    source = input_directory.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"optimized replay output already exists: {output}")
    input_inventory_sha, input_manifest_sha = _input_inventory(source)
    source_rows, source_arrays = _input_rows_and_arrays(source, config.chunk_count)
    scenarios = mujoco_scenarios()
    audit_systems = {}
    for name in ("free_space", "single_block", "narrow_gate"):
        robot = MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        checker = _checker(robot)
        audit_systems[name] = (
            robot,
            checker,
            PersistentDynamicsBrakingValidator(
                robot,
                checker,
                _guard_config(config, config.diagnostic_budget_ms).braking,
            ),
        )
    systems: dict[tuple[str, str], _System] = {}
    for profile, budget in _profile_specs(config):
        for name in ("free_space", "single_block", "narrow_gate"):
            systems[(profile, name)] = _make_system(name, budget, config)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, np.ndarray]] = []
    audit_cache: dict[bytes, _Audit] = {}
    try:
        for profile, budget in _profile_specs(config):
            for index, source_row in enumerate(source_rows):
                scenario_name = str(source_row["scenario"])
                row, trace = _run_case(
                    source_row,
                    source_arrays["candidate_actions"][index],
                    source_arrays["candidate_positions"][index],
                    profile=profile,
                    budget_ms=budget,
                    system=systems[(profile, scenario_name)],
                    config=config,
                    audit_system=audit_systems[scenario_name],
                    audit_cache=audit_cache,
                )
                rows.append(row)
                traces.append(trace)
    finally:
        for system in systems.values():
            system.worker.close(timeout_s=config.worker_timeout_s)
    summary = _summary(rows, config)
    output.mkdir(parents=True)
    shutil.copyfile(_protocol_path(), output / "protocol.json")
    with (output / "per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_trajectories(output / "trajectories.npz", traces)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    input_provenance = strict_json_load(source / "provenance.json")
    write_json(
        output / "provenance.json",
        {
            "schema_version": SCHEMA,
            "scope": SCOPE,
            "input": {
                "directory": str(input_directory),
                "manifest_sha256": input_manifest_sha,
                "inventory_sha256": input_inventory_sha,
                "summary_sha256": sha256_file(source / "summary.json"),
                "source_root_manifest_sha256": input_provenance["source"][
                    "root_manifest_sha256"
                ],
            },
            "protocol": {
                "schema_version": PROTOCOL_SCHEMA,
                "sha256": sha256_file(_protocol_path()),
                "conformant": config.protocol_conformant,
            },
            "worker_setup": {
                f"{profile}:{scenario}": {
                    "startup_ms": system.worker_startup_ms,
                    "excluded_from_request_latency": True,
                }
                for (profile, scenario), system in systems.items()
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "mujoco_version": mujoco.__version__,
                "osqp_version": osqp.__version__,
                "menagerie_commit": MENAGERIE_COMMIT,
            },
            "implementation_sha256": {
                label: sha256_file(path)
                for label, path in _implementation_paths().items()
            },
            "claim_boundary": list(_CLAIM_BOUNDARY),
        },
    )
    _write_manifest(output)
    validate_optimized_cpu_replay(output, source)
    return output


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("optimized replay CSV fields are invalid")
        for raw in reader:
            if set(raw) != set(CSV_FIELDS):
                raise ValueError("optimized replay CSV row is invalid")
            row: dict[str, Any] = dict(raw)
            try:
                for field in _BOOLEAN_FIELDS:
                    if raw[field] not in {"True", "False"}:
                        raise ValueError("invalid optimized replay boolean")
                    row[field] = raw[field] == "True"
                for field in _INTEGER_FIELDS:
                    row[field] = int(raw[field])
                for field in _FLOAT_FIELDS:
                    row[field] = float(raw[field])
            except (TypeError, ValueError) as error:
                raise ValueError("optimized replay CSV value is invalid") from error
            numeric = [float(row[field]) for field in _INTEGER_FIELDS | _FLOAT_FIELDS]
            if not np.all(np.isfinite(numeric)) or any(value < 0 for value in numeric):
                raise ValueError("optimized replay numeric contract is invalid")
            if (
                row["schema_version"] != SCHEMA
                or row["profile"] not in PROFILE_IDS
                or row["status"]
                not in {"execute", "hold", "verified_brake", "unrecoverable_stop"}
                or not is_sha256(row["response_action_sha256"])
                or not is_sha256(row["input_action_sha256"])
            ):
                raise ValueError("optimized replay row identity is invalid")
            rows.append(row)
    if not rows:
        raise ValueError("optimized replay CSV is empty")
    return rows


def _load_output_trajectories(
    path: Path, count: int
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("optimized replay trajectories are unreadable") from error
    expected = {
        "schema_version",
        "case_ids",
        "source_row_indices",
        "scenario_codes",
        "profile_codes",
        "candidate_lengths",
        "published_lengths",
        "candidate_actions",
        "candidate_positions",
        "published_actions",
        "published_positions",
    }
    if set(arrays) != expected or str(arrays["schema_version"].item()) != TRAJECTORY_SCHEMA:
        raise ValueError("optimized replay trajectory schema is invalid")
    shapes = {
        "case_ids": (count,),
        "source_row_indices": (count,),
        "scenario_codes": (count,),
        "profile_codes": (count,),
        "candidate_lengths": (count,),
        "published_lengths": (count,),
        "candidate_actions": (count, HORIZON, ACTION_DIM),
        "candidate_positions": (count, HORIZON + 1, PANDA_DOF),
        "published_actions": (count, HORIZON, ACTION_DIM),
        "published_positions": (count, HORIZON + 1, PANDA_DOF),
    }
    if any(arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("optimized replay trajectory shape is invalid")
    for key in (
        "candidate_actions",
        "candidate_positions",
        "published_actions",
        "published_positions",
    ):
        if arrays[key].dtype.kind != "f" or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"optimized replay trajectory values are invalid: {key}")
    return arrays


def _validate_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    expected = _inventory(root)
    if not (
        actual == _ARTIFACT_FILES
        and has_exact_fields(manifest, {"schema_version", "files", "inventory_sha256"})
        and manifest["schema_version"] == MANIFEST_SCHEMA
        and files == expected
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"] == sha256_bytes(canonical_json(expected))
    ):
        raise ValueError("optimized replay manifest mismatch")
    return str(manifest["inventory_sha256"])


def _validate_provenance(root: Path, source: Path, config: OptimizedCPUReplayConfig) -> None:
    provenance = strict_json_load(root / "provenance.json")
    input_inventory_sha, input_manifest_sha = _input_inventory(source)
    if not (
        provenance.get("schema_version") == SCHEMA
        and provenance.get("scope") == SCOPE
        and provenance.get("input", {}).get("manifest_sha256") == input_manifest_sha
        and provenance.get("input", {}).get("inventory_sha256") == input_inventory_sha
        and provenance.get("input", {}).get("summary_sha256")
        == sha256_file(source / "summary.json")
        and provenance.get("protocol", {}).get("schema_version") == PROTOCOL_SCHEMA
        and provenance.get("protocol", {}).get("sha256") == sha256_file(_protocol_path())
        and provenance.get("protocol", {}).get("conformant")
        == config.protocol_conformant
        and provenance.get("implementation_sha256")
        == {label: sha256_file(path) for label, path in _implementation_paths().items()}
        and canonical_json(provenance.get("claim_boundary"))
        == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("optimized replay provenance mismatch")


def _config_from_json(value: object) -> OptimizedCPUReplayConfig:
    if not isinstance(value, Mapping) or set(value) != set(asdict(OptimizedCPUReplayConfig())):
        raise ValueError("optimized replay configuration is invalid")
    try:
        config = OptimizedCPUReplayConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("optimized replay configuration is invalid") from error
    if canonical_json(value) != canonical_json(asdict(config)):
        raise ValueError("optimized replay configuration is not canonical")
    return config


def validate_optimized_cpu_replay(
    directory: Path, input_directory: Path
) -> dict[str, Any]:
    """Verify hashes, all-or-none publication, and every candidate constraint."""

    root = directory.resolve()
    source = input_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"optimized replay directory not found: {root}")
    inventory_sha = _validate_manifest(root)
    if (root / "protocol.json").read_bytes() != _protocol_path().read_bytes():
        raise ValueError("optimized replay protocol copy mismatch")
    protocol = strict_json_load(root / "protocol.json")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("optimized replay protocol schema mismatch")
    summary = strict_json_load(root / "summary.json")
    if not (
        summary.get("schema_version") == SUMMARY_SCHEMA
        and summary.get("scope") == SCOPE
        and canonical_json(summary.get("claim_boundary"))
        == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("optimized replay summary is invalid")
    config = _config_from_json(summary.get("configuration"))
    if summary.get("protocol_conformant") != config.protocol_conformant:
        raise ValueError("optimized replay protocol conformance mismatch")
    _validate_provenance(root, source, config)
    rows = _parse_csv(root / "per_case.csv")
    expected_count = config.chunk_count * 3 * len(PROFILE_IDS)
    if len(rows) != expected_count or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("optimized replay row count or identity is invalid")
    arrays = _load_output_trajectories(root / "trajectories.npz", len(rows))
    source_rows, source_arrays = _input_rows_and_arrays(source, config.chunk_count)
    source_by_id = {
        str(row["case_id"]): (index, row) for index, row in enumerate(source_rows)
    }
    scenarios = mujoco_scenarios()
    audit_systems = {}
    for name in ("free_space", "single_block", "narrow_gate"):
        robot = MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        checker = _checker(robot)
        audit_systems[name] = (
            robot,
            checker,
            PersistentDynamicsBrakingValidator(
                robot,
                checker,
                _guard_config(config, config.diagnostic_budget_ms).braking,
            ),
        )
    audit_cache: dict[bytes, _Audit] = {}
    for index, row in enumerate(rows):
        if str(arrays["case_ids"][index]) != row["case_id"]:
            raise ValueError("optimized replay row/trajectory identity mismatch")
        source_item = source_by_id.get(str(row["source_case_id"]))
        if source_item is None:
            raise ValueError("optimized replay source case is unknown")
        source_index, source_row = source_item
        if not (
            int(arrays["source_row_indices"][index]) == row["source_row_index"]
            == int(source_row["source_row_index"])
            and row["scenario"] == source_row["scenario"]
            and row["response_action_sha256"]
            == source_row["response_action_sha256"]
            and row["input_action_sha256"]
            == _array_sha256(source_arrays["candidate_actions"][source_index])
        ):
            raise ValueError("optimized replay source binding mismatch")
        length = int(arrays["candidate_lengths"][index])
        published_length = int(arrays["published_lengths"][index])
        actions = arrays["candidate_actions"][index]
        positions = arrays["candidate_positions"][index]
        published_actions = arrays["published_actions"][index]
        published_positions = arrays["published_positions"][index]
        if not (
            length == int(row["candidate_action_count"])
            and published_length == int(row["published_action_count"])
            and 0 <= length <= HORIZON
            and published_length in {0, HORIZON}
            and bool(row["candidate_complete"]) == (length == HORIZON)
        ):
            raise ValueError("optimized replay trajectory length contract failed")
        if length:
            integrated = positions[0] + np.cumsum(
                actions[:length, :PANDA_DOF] * 0.05, axis=0
            )
            if not np.allclose(
                integrated, positions[1 : length + 1], atol=1e-12, rtol=0.0
            ):
                raise ValueError("optimized replay candidate does not integrate")
        executable = row["status"] == "execute"
        if executable:
            if not (
                published_length == HORIZON
                and np.array_equal(published_actions, actions)
                and np.array_equal(published_positions, positions)
            ):
                raise ValueError("optimized replay execute publication mismatch")
        elif not (
            published_length == 0
            and not np.any(published_actions)
            and np.allclose(
                published_positions,
                positions[0][None, :],
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise ValueError("optimized replay hold exposed policy motion")
        if (
            bool(row["policy_actions_executable"]) != executable
            or bool(row["partial_prefix_exposed"])
            or not bool(row["all_or_none_publication"])
            or not bool(row["assurance_worker_thread_separate"])
        ):
            raise ValueError("optimized replay atomic publication contract failed")
        audit_key = canonical_json(
            {
                "scenario": row["scenario"],
                "length": length,
                "actions": _array_sha256(actions),
                "positions": _array_sha256(positions),
            }
        )
        if audit_key not in audit_cache:
            audit_robot, audit_checker, audit_braking = audit_systems[
                str(row["scenario"])
            ]
            audit_cache[audit_key] = _audit_candidate(
                audit_robot,
                audit_checker,
                audit_braking,
                actions,
                positions,
                length,
                _guard_config(config, config.diagnostic_budget_ms),
            )
        audit = audit_cache[audit_key]
        expected_audit = {
            "joint_position_violation_steps": audit.position_violations,
            "joint_velocity_violation_steps": audit.velocity_violations,
            "joint_acceleration_violation_steps": audit.acceleration_violations,
            "continuous_edges_checked": audit.edges_checked,
            "continuous_unsafe_edges": audit.unsafe_edges,
            "continuous_indeterminate_edges": audit.indeterminate_edges,
            "braking_boundaries_checked": audit.braking_boundaries,
            "braking_invalid_boundaries": audit.invalid_braking_boundaries,
            "all_registered_constraints_satisfied": bool(
                length == HORIZON and audit.all_satisfied
            ),
        }
        if any(row[key] != value for key, value in expected_audit.items()):
            raise ValueError(f"optimized replay audit mismatch: {row['case_id']}")
        unsafe = bool(executable and not audit.all_satisfied)
        if bool(row["unsafe_plan_published"]) != unsafe or unsafe:
            raise ValueError("optimized replay published an unsafe plan")
    recomputed = _summary(rows, config)
    if canonical_json(summary) != canonical_json(recomputed):
        raise ValueError("optimized replay summary aggregate mismatch")
    if (root / "summary.md").read_text("utf-8") != _summary_markdown(summary):
        raise ValueError("optimized replay Markdown summary mismatch")
    return {
        "valid": True,
        "scope": SCOPE,
        "cases": len(rows),
        "operational_execute": summary["by_profile"]["operational_20ms"][
            "execute"
        ],
        "diagnostic_execute": summary["by_profile"]["diagnostic_100ms"][
            "execute"
        ],
        "unsafe_plans_published": summary["overall"]["unsafe_plans_published"],
        "partial_prefixes_exposed": summary["overall"][
            "partial_prefixes_exposed"
        ],
        "go_no_go": summary["go_no_go"]["decision"],
        "manifest_inventory_sha256": inventory_sha,
        "checks": [
            "recursive_manifest_and_exact_file_set",
            "frozen_input_manifest_and_response_bindings",
            "implementation_and_protocol_hashes",
            "all_or_none_trajectory_publication",
            "candidate_kinematics_recomputed",
            "continuous_collision_and_braking_audit_recomputed",
            "summary_and_markdown_recomputed",
        ],
    }


__all__ = [
    "CSV_FIELDS",
    "OptimizedCPUReplayConfig",
    "execute_optimized_cpu_replay",
    "validate_optimized_cpu_replay",
]
