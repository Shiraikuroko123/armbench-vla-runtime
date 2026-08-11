"""Paired CPU audit for full-chunk and short-window Panda assurance.

The optimized supervisor can certify a complete ten-action VLA chunk, but the
registered 20 ms software budget is too small for that work on the reference
host.  This module evaluates a different publication contract: certify and
atomically publish only the next action window, then re-enter the supervisor
before any later window is exposed.

The source chunk remains visible in the artifact.  A one-action publication is
therefore recorded as a progressive source-chunk prefix, not mislabeled as
full-chunk atomicity.  Every published window still passes QP projection,
continuous collision checks, a braking invariant, activation-age validation,
and an independent post-run audit.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shutil
import threading
import time
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
import osqp

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
from armbench.vla.optimized_cpu_replay import (
    ACTION_DIM,
    HORIZON,
    _array_sha256,
    _audit_candidate,
    _checker,
    _input_inventory,
    _input_rows_and_arrays,
)
from armbench.vla.optimized_integrated_panda_guard import (
    OptimizedIntegratedPandaSupervisor,
)
from armbench.vla.pi05_integrated_replay import (
    _pad_candidate,
    _published_positions,
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


SCHEMA = "armbench.pi05_windowed_cpu_replay.v1"
SUMMARY_SCHEMA = "armbench.pi05_windowed_cpu_replay_summary.v1"
TRAJECTORY_SCHEMA = "armbench.pi05_windowed_cpu_replay_trajectory.v1"
MANIFEST_SCHEMA = "armbench.pi05_windowed_cpu_replay_manifest.v1"
PROTOCOL_SCHEMA = "armbench.pi05_windowed_cpu_replay_protocol.v1"
SCOPE = "frozen_pi05_panda_paired_window_publication_cpu_audit"
PROTOCOL_RELATIVE = Path(
    "docs/research/pi05_windowed_cpu_replay_protocol_20260811.json"
)
PROFILE_SPECS = (
    ("full_chunk_h10", HORIZON),
    ("certified_window_h1", 1),
)
PROFILE_IDS = tuple(name for name, _ in PROFILE_SPECS)
SCENARIOS = ("free_space", "single_block", "narrow_gate")

CSV_FIELDS = (
    "schema_version",
    "case_id",
    "source_case_id",
    "selection_index",
    "source_row_index",
    "scenario",
    "profile",
    "source_chunk_horizon",
    "certification_horizon",
    "response_action_sha256",
    "input_window_sha256",
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
    "partial_window_exposed",
    "all_or_none_window_publication",
    "source_chunk_prefix_published",
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
    "unsafe_window_published",
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
    "partial_window_exposed",
    "all_or_none_window_publication",
    "source_chunk_prefix_published",
    "assurance_worker_thread_separate",
    "fallback_validated",
    "qp_feasible",
    "all_registered_constraints_satisfied",
    "unsafe_window_published",
    "software_budget_exceeded",
    "response_deadline_exceeded",
}
_INTEGER_FIELDS = {
    "selection_index",
    "source_row_index",
    "source_chunk_horizon",
    "certification_horizon",
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
_TRACE_FIELDS = {
    "schema_version",
    "case_ids",
    "source_row_indices",
    "scenario_codes",
    "profile_codes",
    "certification_horizons",
    "candidate_lengths",
    "published_lengths",
    "input_actions",
    "candidate_actions",
    "candidate_positions",
    "published_actions",
    "published_positions",
}
_CLAIM_BOUNDARY = [
    (
        "The one-action horizon was selected after exploratory profiling; this "
        "is a descriptive engineering audit, not a preregistered inference."
    ),
    (
        "Frozen official pi0.5 responses are replayed; the checkpoint is not "
        "executed and no closed-loop task-success observation is created."
    ),
    (
        "Window publication is atomic, while the original ten-action source "
        "chunk is progressive and may stop after an already certified window."
    ),
    (
        "Measured best-effort Python timing is not an operating-system hard-"
        "real-time guarantee or physical-robot safety certification."
    ),
]


@dataclass(frozen=True)
class WindowedCPUReplayConfig:
    """Configuration fixed for the paired H=10 versus H=1 CPU audit."""

    chunk_count: int = 30
    supervision_budget_ms: float = 20.0
    response_deadline_ms: float = 200.0
    qp_step_budget_ms: float = 5.0
    worker_timeout_s: float = 30.0
    poll_interval_s: float = 0.0005

    def __post_init__(self) -> None:
        if type(self.chunk_count) is not int or self.chunk_count <= 0:
            raise ValueError("chunk_count must be a positive integer")
        for value, label in (
            (self.supervision_budget_ms, "supervision_budget_ms"),
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

    @property
    def protocol_conformant(self) -> bool:
        return self == WindowedCPUReplayConfig()


@dataclass
class _System:
    robot: MuJoCoPanda
    supervisor: OptimizedIntegratedPandaSupervisor
    worker: LatestIntegratedPandaWorker
    gate: AtomicPandaPlanGate
    worker_startup_ms: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return _project_root() / PROTOCOL_RELATIVE


def _guard_config(config: WindowedCPUReplayConfig) -> IntegratedPandaGuardConfig:
    return IntegratedPandaGuardConfig(
        response_deadline_ms=config.response_deadline_ms,
        supervision_budget_ms=config.supervision_budget_ms,
        qp_step_budget_ms=config.qp_step_budget_ms,
    )


def _make_system(
    scenario_name: str, config: WindowedCPUReplayConfig
) -> _System:
    scenario = mujoco_scenarios()[scenario_name]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = _checker(robot)
    supervisor = OptimizedIntegratedPandaSupervisor(
        robot, checker, _guard_config(config)
    )
    started = time.perf_counter()
    worker = LatestIntegratedPandaWorker(supervisor)
    deadline = time.monotonic() + 5.0
    while worker.metrics()["worker_thread_id"] is None:
        if time.monotonic() >= deadline:
            worker.close()
            raise TimeoutError("windowed assurance worker did not start")
        time.sleep(config.poll_interval_s)
    return _System(
        robot=robot,
        supervisor=supervisor,
        worker=worker,
        gate=AtomicPandaPlanGate(supervisor),
        worker_startup_ms=(time.perf_counter() - started) * 1000.0,
    )


def _wait_outcome(system: _System, config: WindowedCPUReplayConfig) -> object:
    deadline = time.monotonic() + config.worker_timeout_s
    while time.monotonic() < deadline:
        outcomes = system.worker.drain()
        if outcomes:
            return outcomes[0]
        time.sleep(config.poll_interval_s)
    raise TimeoutError("windowed assurance worker timed out")


def _audit_satisfied(audit: object, horizon: int) -> bool:
    return bool(
        audit.edges_checked == horizon
        and audit.braking_boundaries == horizon
        and audit.position_violations == 0
        and audit.velocity_violations == 0
        and audit.acceleration_violations == 0
        and audit.unsafe_edges == 0
        and audit.indeterminate_edges == 0
        and audit.invalid_braking_boundaries == 0
    )


def _run_case(
    source_row: Mapping[str, Any],
    source_actions: np.ndarray,
    source_positions: np.ndarray,
    *,
    profile: str,
    horizon: int,
    system: _System,
    config: WindowedCPUReplayConfig,
    audit_system: tuple[
        MuJoCoPanda,
        object,
        PersistentDynamicsBrakingValidator,
    ],
    audit_cache: dict[bytes, object],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    checker = system.supervisor.checker
    checker.clear_safe_configuration_cache()
    checker.reset_metrics()
    before = system.supervisor.optimization_metrics()
    generation = system.gate.reset()
    source_index = int(source_row["source_row_index"])
    input_window = np.asarray(source_actions[:horizon], dtype=float)
    chunk = ActionChunk(
        actions=input_window,
        source="manifest_bound_frozen_pi05_certification_window",
        observation_sequence_id=source_index,
        inference_latency_ms=0.0,
    )
    base_age = float(source_row["source_inference_latency_ms"]) + float(
        source_row["adapter_latency_ms"]
    )
    q_start = np.asarray(source_positions[0], dtype=float)
    system.worker.submit(
        generation=generation,
        observation_sequence_id=source_index,
        q=q_start,
        qvel=np.zeros(PANDA_DOF),
        observed_q=q_start,
        response_age_ms=base_age,
        chunk=chunk,
    )
    outcome = _wait_outcome(system, config)
    atomic = system.gate.commit(outcome, q_now=q_start)
    decision = outcome.decision
    if decision is None or decision.qp_result is None:
        candidate_actions = np.zeros((HORIZON, ACTION_DIM), dtype=float)
        candidate_positions = np.repeat(
            q_start[None, :], HORIZON + 1, axis=0
        )
        candidate_length = 0
        qp_feasible = False
        qp_interventions = 0
        supervisor_latency = outcome.worker_latency_ms
        stages: Mapping[str, float] = {}
        failure_stage = "worker" if decision is None else (
            decision.failure_stage or ""
        )
    else:
        projection = decision.qp_result
        candidate_actions, candidate_positions, candidate_length = _pad_candidate(
            projection.projected_actions,
            projection.predicted_positions,
            horizon=HORIZON,
            q_start=q_start,
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
        q_start,
        published_actions,
        published_length,
        system.supervisor.config.control_dt_s,
    )

    audit_key = canonical_json(
        {
            "scenario": source_row["scenario"],
            "horizon": horizon,
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
            _guard_config(config),
        )
    audit = audit_cache[audit_key]
    after = system.supervisor.optimization_metrics()
    candidate_complete = candidate_length == horizon
    window_atomic = published_length in {0, horizon}
    constraints_satisfied = bool(
        candidate_complete and _audit_satisfied(audit, horizon)
    )
    unsafe_published = bool(
        atomic.status == "execute" and not constraints_satisfied
    )
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
        "source_chunk_horizon": HORIZON,
        "certification_horizon": horizon,
        "response_action_sha256": source_row["response_action_sha256"],
        "input_window_sha256": _array_sha256(input_window),
        "supervision_budget_ms": config.supervision_budget_ms,
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
        "fallback_brake_latency_ms": float(
            stages.get("fallback_brake", 0.0)
        ),
        "status": atomic.status,
        "reason": atomic.reason,
        "supervisor_status": atomic.supervisor_status or "",
        "failure_stage": failure_stage,
        "candidate_complete": candidate_complete,
        "candidate_action_count": candidate_length,
        "published_action_count": published_length,
        "policy_actions_executable": atomic.policy_actions_executable,
        "partial_window_exposed": not window_atomic,
        "all_or_none_window_publication": window_atomic,
        "source_chunk_prefix_published": bool(
            published_length > 0 and horizon < HORIZON
        ),
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
        "all_registered_constraints_satisfied": constraints_satisfied,
        "unsafe_window_published": unsafe_published,
        "software_budget_exceeded": bool(
            failure_stage == "supervision_budget"
            or supervisor_latency > config.supervision_budget_ms
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
        "broad_phase_prune_rate": (
            0.0 if pair_tests == 0 else pruned / pair_tests
        ),
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
    input_actions = np.zeros((HORIZON, ACTION_DIM), dtype=float)
    input_actions[:horizon] = input_window
    trace = {
        "case_id": np.asarray(case_id),
        "source_row_index": np.asarray(source_index, dtype="<i4"),
        "scenario_code": np.asarray(SCENARIOS.index(str(row["scenario"])), dtype="u1"),
        "profile_code": np.asarray(PROFILE_IDS.index(profile), dtype="u1"),
        "certification_horizon": np.asarray(horizon, dtype="u1"),
        "candidate_length": np.asarray(candidate_length, dtype="<i4"),
        "published_length": np.asarray(published_length, dtype="<i4"),
        "input_actions": input_actions,
        "candidate_actions": candidate_actions,
        "candidate_positions": candidate_positions,
        "published_actions": published_actions,
        "published_positions": published_positions,
    }
    return row, trace


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty windowed replay rows")
    supervisor = [float(row["supervisor_latency_ms"]) for row in rows]
    worker = [float(row["assurance_worker_latency_ms"]) for row in rows]
    pair_tests = sum(int(row["broad_phase_pair_tests"]) for row in rows)
    pruned = sum(int(row["broad_phase_pruned_pairs"]) for row in rows)
    failures: dict[str, int] = {}
    for row in rows:
        stage = str(row["failure_stage"] or "accepted")
        failures[stage] = failures.get(stage, 0) + 1
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
        "constraint_satisfied_candidates": sum(
            bool(row["all_registered_constraints_satisfied"])
            for row in rows
        ),
        "unsafe_windows_published": sum(
            bool(row["unsafe_window_published"]) for row in rows
        ),
        "partial_windows_exposed": sum(
            bool(row["partial_window_exposed"]) for row in rows
        ),
        "source_chunk_prefix_publications": sum(
            bool(row["source_chunk_prefix_published"]) for row in rows
        ),
        "software_budget_exceeded": sum(
            bool(row["software_budget_exceeded"]) for row in rows
        ),
        "response_deadline_exceeded": sum(
            bool(row["response_deadline_exceeded"]) for row in rows
        ),
        "failure_stages": dict(sorted(failures.items())),
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
        "broad_phase_prune_rate": (
            0.0 if pair_tests == 0 else pruned / pair_tests
        ),
        "safe_configuration_cache_hits": sum(
            int(row["safe_configuration_cache_hits"]) for row in rows
        ),
    }


def _summary(
    rows: Sequence[Mapping[str, Any]], config: WindowedCPUReplayConfig
) -> dict[str, Any]:
    by_profile = {
        profile: _aggregate([row for row in rows if row["profile"] == profile])
        for profile in PROFILE_IDS
    }
    full = by_profile["full_chunk_h10"]
    window = by_profile["certified_window_h1"]
    candidate = bool(
        window["execute"] > full["execute"]
        and window["unsafe_windows_published"] == 0
        and window["partial_windows_exposed"] == 0
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "scope": SCOPE,
        "configuration": asdict(config),
        "protocol_conformant": config.protocol_conformant,
        "overall": _aggregate(rows),
        "by_profile": by_profile,
        "paired_comparison": {
            "window_minus_full_execute": window["execute"] - full["execute"],
            "window_minus_full_constraint_candidates": (
                window["constraint_satisfied_candidates"]
                - full["constraint_satisfied_candidates"]
            ),
            "window_is_repeatability_candidate": candidate,
            "source_chunk_atomicity_changed": True,
            "window_atomicity_preserved": True,
        },
        "claim_boundary": list(_CLAIM_BOUNDARY),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Windowed Panda CPU assurance audit",
        "",
        (
            "The same frozen pi0.5-to-Panda inputs are evaluated as a complete "
            "H=10 publication and as an H=1 certified execution window."
        ),
        "",
        (
            "| Profile | Cases | Execute | Constraint-safe | Unsafe windows | "
            "Partial windows | Budget misses | P95 supervisor |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in PROFILE_IDS:
        values = summary["by_profile"][profile]
        lines.append(
            f"| `{profile}` | {values['cases']} | {values['execute']} | "
            f"{values['constraint_satisfied_candidates']} | "
            f"{values['unsafe_windows_published']} | "
            f"{values['partial_windows_exposed']} | "
            f"{values['software_budget_exceeded']} | "
            f"{values['p95_supervisor_latency_ms']:.3f} ms |"
        )
    comparison = summary["paired_comparison"]
    lines.extend(
        [
            "",
            (
                "Window minus full-chunk execute count: "
                f"**{comparison['window_minus_full_execute']:+d}**."
            ),
            "",
            (
                "The H=1 result changes publication granularity. It does not "
                "claim full-source-chunk atomicity, task success, hard real-time "
                "execution, or physical safety."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_trajectories(
    path: Path, traces: Sequence[Mapping[str, np.ndarray]]
) -> None:
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
        certification_horizons=np.asarray(
            [trace["certification_horizon"] for trace in traces], dtype="u1"
        ),
        candidate_lengths=np.asarray(
            [trace["candidate_length"] for trace in traces], dtype="<i4"
        ),
        published_lengths=np.asarray(
            [trace["published_length"] for trace in traces], dtype="<i4"
        ),
        input_actions=np.stack([trace["input_actions"] for trace in traces]),
        candidate_actions=np.stack(
            [trace["candidate_actions"] for trace in traces]
        ),
        candidate_positions=np.stack(
            [trace["candidate_positions"] for trace in traces]
        ),
        published_actions=np.stack(
            [trace["published_actions"] for trace in traces]
        ),
        published_positions=np.stack(
            [trace["published_positions"] for trace in traces]
        ),
    )


def _implementation_paths() -> dict[str, Path]:
    root = _project_root()
    relatives = (
        "src/armbench/vla/windowed_cpu_replay.py",
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
    return {relative: root / relative for relative in relatives}


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


def execute_windowed_cpu_replay(
    input_directory: Path,
    output_directory: Path,
    config: WindowedCPUReplayConfig = WindowedCPUReplayConfig(),
) -> Path:
    """Run the paired full-chunk and certified-window CPU audit."""

    source = input_directory.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"windowed replay output already exists: {output}")
    input_inventory_sha, input_manifest_sha = _input_inventory(source)
    source_rows, source_arrays = _input_rows_and_arrays(
        source, config.chunk_count
    )
    scenarios = mujoco_scenarios()
    audit_systems = {}
    for name in SCENARIOS:
        robot = MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        checker = _checker(robot)
        audit_systems[name] = (
            robot,
            checker,
            PersistentDynamicsBrakingValidator(
                robot, checker, _guard_config(config).braking
            ),
        )
    systems = {
        (profile, scenario): _make_system(scenario, config)
        for profile, _ in PROFILE_SPECS
        for scenario in SCENARIOS
    }
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, np.ndarray]] = []
    audit_cache: dict[bytes, object] = {}
    try:
        for profile, horizon in PROFILE_SPECS:
            for index, source_row in enumerate(source_rows):
                scenario = str(source_row["scenario"])
                row, trace = _run_case(
                    source_row,
                    source_arrays["candidate_actions"][index],
                    source_arrays["candidate_positions"][index],
                    profile=profile,
                    horizon=horizon,
                    system=systems[(profile, scenario)],
                    config=config,
                    audit_system=audit_systems[scenario],
                    audit_cache=audit_cache,
                )
                rows.append(row)
                traces.append(trace)
    finally:
        for system in systems.values():
            system.worker.close(timeout_s=config.worker_timeout_s)

    output.mkdir(parents=True)
    shutil.copyfile(_protocol_path(), output / "protocol.json")
    with (output / "per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_trajectories(output / "trajectories.npz", traces)
    summary = _summary(rows, config)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
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
    validate_windowed_cpu_replay(output, source)
    return output


def _validate_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    expected = _inventory(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if not (
        has_exact_fields(
            manifest, {"schema_version", "files", "inventory_sha256"}
        )
        and manifest["schema_version"] == MANIFEST_SCHEMA
        and manifest["files"] == expected
        and actual == _ARTIFACT_FILES
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"]
        == sha256_bytes(canonical_json(expected))
    ):
        raise ValueError("windowed replay manifest mismatch")
    return str(manifest["inventory_sha256"])


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("windowed replay CSV fields are invalid")
        for raw in reader:
            if set(raw) != set(CSV_FIELDS):
                raise ValueError("windowed replay CSV row is invalid")
            row: dict[str, Any] = dict(raw)
            try:
                for field in _BOOLEAN_FIELDS:
                    if raw[field] not in {"True", "False"}:
                        raise ValueError("invalid windowed replay boolean")
                    row[field] = raw[field] == "True"
                for field in _INTEGER_FIELDS:
                    row[field] = int(raw[field])
                for field in _FLOAT_FIELDS:
                    row[field] = float(raw[field])
            except (TypeError, ValueError) as error:
                raise ValueError("windowed replay CSV value is invalid") from error
            numeric = [
                float(row[field]) for field in _INTEGER_FIELDS | _FLOAT_FIELDS
            ]
            if not np.all(np.isfinite(numeric)) or any(
                value < 0.0 for value in numeric
            ):
                raise ValueError("windowed replay numeric contract is invalid")
            rows.append(row)
    if not rows:
        raise ValueError("windowed replay CSV is empty")
    return rows


def _load_trajectories(path: Path, count: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _TRACE_FIELDS:
                raise ValueError("windowed trajectory fields are invalid")
            arrays = {
                key: np.asarray(archive[key]).copy() for key in archive.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("windowed trajectories are unreadable") from error
    if str(arrays["schema_version"].item()) != TRAJECTORY_SCHEMA:
        raise ValueError("windowed trajectory schema is invalid")
    expected_shapes = {
        "case_ids": (count,),
        "source_row_indices": (count,),
        "scenario_codes": (count,),
        "profile_codes": (count,),
        "certification_horizons": (count,),
        "candidate_lengths": (count,),
        "published_lengths": (count,),
        "input_actions": (count, HORIZON, ACTION_DIM),
        "candidate_actions": (count, HORIZON, ACTION_DIM),
        "candidate_positions": (count, HORIZON + 1, PANDA_DOF),
        "published_actions": (count, HORIZON, ACTION_DIM),
        "published_positions": (count, HORIZON + 1, PANDA_DOF),
    }
    for field, shape in expected_shapes.items():
        if arrays[field].shape != shape:
            raise ValueError(f"windowed trajectory shape is invalid: {field}")
        if field != "case_ids" and not np.all(np.isfinite(arrays[field])):
            raise ValueError(f"windowed trajectory values are invalid: {field}")
    return arrays


def _config_from_json(value: object) -> WindowedCPUReplayConfig:
    expected = set(asdict(WindowedCPUReplayConfig()))
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("windowed replay configuration is invalid")
    try:
        config = WindowedCPUReplayConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("windowed replay configuration is invalid") from error
    if asdict(config) != dict(value):
        raise ValueError("windowed replay configuration is not canonical")
    return config


def _validate_provenance(
    root: Path, source: Path, config: WindowedCPUReplayConfig
) -> None:
    provenance = strict_json_load(root / "provenance.json")
    input_inventory, input_manifest = _input_inventory(source)
    source_provenance = strict_json_load(source / "provenance.json")
    implementation = {
        label: sha256_file(path)
        for label, path in _implementation_paths().items()
    }
    if not (
        isinstance(provenance, Mapping)
        and provenance.get("schema_version") == SCHEMA
        and provenance.get("scope") == SCOPE
        and provenance.get("claim_boundary") == _CLAIM_BOUNDARY
        and provenance.get("implementation_sha256") == implementation
        and provenance.get("protocol")
        == {
            "schema_version": PROTOCOL_SCHEMA,
            "sha256": sha256_file(_protocol_path()),
            "conformant": config.protocol_conformant,
        }
        and isinstance(provenance.get("input"), Mapping)
        and provenance["input"].get("manifest_sha256") == input_manifest
        and provenance["input"].get("inventory_sha256") == input_inventory
        and provenance["input"].get("summary_sha256")
        == sha256_file(source / "summary.json")
        and provenance["input"].get("source_root_manifest_sha256")
        == source_provenance["source"]["root_manifest_sha256"]
        and isinstance(provenance.get("worker_setup"), Mapping)
        and set(provenance["worker_setup"])
        == {
            f"{profile}:{scenario}"
            for profile in PROFILE_IDS
            for scenario in SCENARIOS
        }
    ):
        raise ValueError("windowed replay provenance mismatch")


def _row_audit_fields(audit: object) -> dict[str, int]:
    return {
        "joint_position_violation_steps": audit.position_violations,
        "joint_velocity_violation_steps": audit.velocity_violations,
        "joint_acceleration_violation_steps": audit.acceleration_violations,
        "continuous_edges_checked": audit.edges_checked,
        "continuous_unsafe_edges": audit.unsafe_edges,
        "continuous_indeterminate_edges": audit.indeterminate_edges,
        "braking_boundaries_checked": audit.braking_boundaries,
        "braking_invalid_boundaries": audit.invalid_braking_boundaries,
    }


def validate_windowed_cpu_replay(
    directory: Path, input_directory: Path
) -> dict[str, Any]:
    """Validate identity, publication, trajectory, and safety contracts."""

    root = directory.resolve()
    source = input_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"windowed replay directory not found: {root}")
    inventory_sha = _validate_manifest(root)
    if (root / "protocol.json").read_bytes() != _protocol_path().read_bytes():
        raise ValueError("windowed replay protocol copy mismatch")
    protocol = strict_json_load(root / "protocol.json")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("windowed replay protocol schema mismatch")
    summary = strict_json_load(root / "summary.json")
    if not isinstance(summary, Mapping):
        raise ValueError("windowed replay summary is invalid")
    config = _config_from_json(summary.get("configuration"))
    _validate_provenance(root, source, config)
    rows = _parse_csv(root / "per_case.csv")
    trajectories = _load_trajectories(root / "trajectories.npz", len(rows))
    source_rows, source_arrays = _input_rows_and_arrays(
        source, config.chunk_count
    )
    source_lookup = {
        (int(row["source_row_index"]), str(row["scenario"])): (index, row)
        for index, row in enumerate(source_rows)
    }
    expected_identities = {
        (int(row["source_row_index"]), str(row["scenario"]), profile)
        for row in source_rows
        for profile in PROFILE_IDS
    }
    actual_identities = {
        (
            int(row["source_row_index"]),
            str(row["scenario"]),
            str(row["profile"]),
        )
        for row in rows
    }
    if actual_identities != expected_identities or len(rows) != len(
        expected_identities
    ):
        raise ValueError("windowed replay row identity is invalid")

    scenarios = mujoco_scenarios()
    audit_systems = {}
    for name in SCENARIOS:
        robot = MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        checker = _checker(robot)
        audit_systems[name] = (
            robot,
            checker,
            PersistentDynamicsBrakingValidator(
                robot, checker, _guard_config(config).braking
            ),
        )
    audit_cache: dict[bytes, object] = {}
    profile_horizons = dict(PROFILE_SPECS)
    dt = _guard_config(config).control_dt_s
    for row_index, row in enumerate(rows):
        profile = str(row["profile"])
        if profile not in profile_horizons:
            raise ValueError("windowed replay profile is invalid")
        horizon = profile_horizons[profile]
        source_index = int(row["source_row_index"])
        source_key = (source_index, str(row["scenario"]))
        if source_key not in source_lookup:
            raise ValueError("windowed replay source row is unknown")
        source_offset, source_row = source_lookup[source_key]
        source_actions = source_arrays["candidate_actions"][source_offset]
        q_start = source_arrays["candidate_positions"][source_offset][0]
        expected_case_id = f"{source_row['case_id']}__{profile}"
        expected_base_age = float(
            source_row["source_inference_latency_ms"]
        ) + float(source_row["adapter_latency_ms"])
        if not (
            row["schema_version"] == SCHEMA
            and row["case_id"] == expected_case_id
            and row["source_case_id"] == source_row["case_id"]
            and row["selection_index"] == source_row["selection_index"]
            and row["scenario"] == source_row["scenario"]
            and row["source_chunk_horizon"] == HORIZON
            and row["certification_horizon"] == horizon
            and row["response_action_sha256"]
            == source_row["response_action_sha256"]
            and row["input_window_sha256"]
            == _array_sha256(source_actions[:horizon])
            and np.isclose(row["base_response_age_ms"], expected_base_age)
            and row["supervision_budget_ms"]
            == config.supervision_budget_ms
            and str(trajectories["case_ids"][row_index]) == expected_case_id
            and trajectories["source_row_indices"][row_index] == source_index
            and trajectories["scenario_codes"][row_index]
            == SCENARIOS.index(str(row["scenario"]))
            and trajectories["profile_codes"][row_index]
            == PROFILE_IDS.index(profile)
            and trajectories["certification_horizons"][row_index] == horizon
        ):
            raise ValueError("windowed replay source binding mismatch")

        input_actions = trajectories["input_actions"][row_index]
        expected_input = np.zeros((HORIZON, ACTION_DIM), dtype=float)
        expected_input[:horizon] = source_actions[:horizon]
        if not np.array_equal(input_actions, expected_input):
            raise ValueError("windowed replay input actions mismatch")
        candidate_length = int(trajectories["candidate_lengths"][row_index])
        published_length = int(trajectories["published_lengths"][row_index])
        candidate_actions = trajectories["candidate_actions"][row_index]
        candidate_positions = trajectories["candidate_positions"][row_index]
        published_actions = trajectories["published_actions"][row_index]
        published_positions = trajectories["published_positions"][row_index]
        if not (
            candidate_length == int(row["candidate_action_count"])
            and published_length == int(row["published_action_count"])
            and 0 <= candidate_length <= horizon
            and published_length in {0, horizon}
            and bool(row["candidate_complete"])
            == (candidate_length == horizon)
            and bool(row["partial_window_exposed"]) is False
            and bool(row["all_or_none_window_publication"]) is True
        ):
            raise ValueError("windowed publication length contract failed")
        expected_positions = _published_positions(
            q_start, candidate_actions, candidate_length, dt
        )
        if not np.allclose(
            candidate_positions, expected_positions, atol=1e-12, rtol=0.0
        ):
            raise ValueError("windowed candidate does not integrate")
        expected_published_positions = _published_positions(
            q_start, published_actions, published_length, dt
        )
        if not np.allclose(
            published_positions,
            expected_published_positions,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("windowed publication does not integrate")
        if published_length:
            if not (
                row["status"] == "execute"
                and candidate_length == horizon
                and np.array_equal(
                    published_actions[:horizon], candidate_actions[:horizon]
                )
            ):
                raise ValueError("windowed execute publication mismatch")
        elif np.any(published_actions):
            raise ValueError("windowed hold exposed policy motion")

        source_prefix = bool(published_length and horizon < HORIZON)
        if bool(row["source_chunk_prefix_published"]) != source_prefix:
            raise ValueError("windowed source-prefix accounting mismatch")
        audit_key = canonical_json(
            {
                "scenario": row["scenario"],
                "horizon": horizon,
                "length": candidate_length,
                "actions": _array_sha256(candidate_actions),
                "positions": _array_sha256(candidate_positions),
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
                candidate_actions,
                candidate_positions,
                candidate_length,
                _guard_config(config),
            )
        audit = audit_cache[audit_key]
        if any(row[field] != value for field, value in _row_audit_fields(audit).items()):
            raise ValueError("windowed candidate audit mismatch")
        constraints = bool(
            candidate_length == horizon and _audit_satisfied(audit, horizon)
        )
        unsafe = bool(row["status"] == "execute" and not constraints)
        if not (
            bool(row["all_registered_constraints_satisfied"]) == constraints
            and bool(row["unsafe_window_published"]) == unsafe
            and not unsafe
        ):
            raise ValueError("windowed safety publication contract failed")

    expected_summary = _summary(rows, config)
    if summary != expected_summary:
        raise ValueError("windowed replay summary aggregate mismatch")
    expected_markdown = _summary_markdown(expected_summary)
    if (root / "summary.md").read_text(encoding="utf-8") != expected_markdown:
        raise ValueError("windowed replay Markdown summary mismatch")
    return {
        "valid": True,
        "cases": len(rows),
        "inventory_sha256": inventory_sha,
        "full_chunk_execute": summary["by_profile"]["full_chunk_h10"][
            "execute"
        ],
        "window_execute": summary["by_profile"]["certified_window_h1"][
            "execute"
        ],
        "unsafe_windows_published": summary["overall"][
            "unsafe_windows_published"
        ],
        "partial_windows_exposed": summary["overall"][
            "partial_windows_exposed"
        ],
        "checks": [
            "recursive_manifest_and_exact_file_set",
            "frozen_input_manifest_and_response_bindings",
            "implementation_and_protocol_hashes",
            "window_level_all_or_none_publication",
            "candidate_kinematics_recomputed",
            "continuous_collision_and_braking_audit_recomputed",
            "source_chunk_prefix_semantics_checked",
            "summary_and_markdown_recomputed",
        ],
    }


__all__ = [
    "WindowedCPUReplayConfig",
    "execute_windowed_cpu_replay",
    "validate_windowed_cpu_replay",
]
