"""Paired offline evaluation of greedy and braking-invariant Panda guards."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import platform
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.model import MENAGERIE_COMMIT, default_panda_scene_path
from armbench.mujoco_sim.scenarios import (
    MUJOCO_SCENARIO_VERSION,
    mujoco_scenarios,
)
from armbench.vla.cartesian_adapter import (
    LIBERO_CONTROLLER_SEMANTICS_ID,
    PANDA_KINEMATIC_CONTROL_POINT_ID,
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
)
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.pi05_archive_replay import (
    PI05_ACTION_ADAPTER,
    PI05_ACTION_ADAPTER_SOURCE,
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
from armbench.vla.trajectory_repair import (
    BrakingRepairConfig,
    BrakingTrajectoryGuard,
)
from armbench.vla.types import DROID_IMAGE_SHAPE, VLAObservation


COMPARISON_SCOPE = "offline_paired_braking_invariant_repair"
COMPARISON_PROVENANCE_SCHEMA = "armbench.pi05_panda_braking_repair.v1"
COMPARISON_SUMMARY_SCHEMA = "armbench.pi05_panda_braking_repair_summary.v1"
TRAJECTORY_SCHEMA = "armbench.pi05_panda_braking_trajectories.v1"
TRAJECTORY_KEYS = frozenset(
    {
        "schema_version",
        "case_ids",
        "source_row_indices",
        "scenarios",
        "times_s",
        "raw_positions",
        "legacy_positions",
        "repair_positions",
        "terminal_braking_positions",
        "terminal_braking_lengths",
    }
)

CSV_FIELDS = (
    "case_id",
    "trajectory_index",
    "selection_index",
    "source_row_index",
    "task_suite",
    "task_id",
    "method",
    "pair_id",
    "episode_id",
    "episode_index",
    "query_index",
    "scenario",
    "response_action_sha256",
    "inference_latency_ms",
    "response_deadline_exceeded",
    "input_clipped_steps",
    "raw_path_safe",
    "legacy_safe_after_guard",
    "legacy_path_safe",
    "legacy_unsafe_raw_steps",
    "legacy_intervention_steps",
    "legacy_hold_steps",
    "legacy_acceleration_override_steps",
    "legacy_max_acceleration_rad_s2",
    "legacy_guard_latency_ms",
    "repair_safe_after_guard",
    "repair_path_safe",
    "repair_terminal_path_safe",
    "repair_selected_scale",
    "repair_terminal_brake_steps",
    "repair_intervention_steps",
    "repair_max_acceleration_rad_s2",
    "repair_selection_deadline_exceeded",
    "repair_evaluated_scale_count",
    "repair_collision_edge_checks",
    "repair_latency_ms",
    "repair_fallback_reason",
    "legacy_conflict_resolved",
    "repair_regression",
    "raw_hand_displacement_m",
    "legacy_hand_displacement_m",
    "repair_hand_displacement_m",
)

_INTEGER_FIELDS = frozenset(
    {
        "trajectory_index",
        "selection_index",
        "source_row_index",
        "task_id",
        "episode_index",
        "query_index",
        "input_clipped_steps",
        "legacy_unsafe_raw_steps",
        "legacy_intervention_steps",
        "legacy_hold_steps",
        "legacy_acceleration_override_steps",
        "repair_terminal_brake_steps",
        "repair_intervention_steps",
        "repair_evaluated_scale_count",
        "repair_collision_edge_checks",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "inference_latency_ms",
        "legacy_max_acceleration_rad_s2",
        "legacy_guard_latency_ms",
        "repair_selected_scale",
        "repair_max_acceleration_rad_s2",
        "repair_latency_ms",
        "raw_hand_displacement_m",
        "legacy_hand_displacement_m",
        "repair_hand_displacement_m",
    }
)
_BOOLEAN_FIELDS = frozenset(
    {
        "response_deadline_exceeded",
        "raw_path_safe",
        "legacy_safe_after_guard",
        "legacy_path_safe",
        "repair_safe_after_guard",
        "repair_path_safe",
        "repair_terminal_path_safe",
        "repair_selection_deadline_exceeded",
        "legacy_conflict_resolved",
        "repair_regression",
    }
)


@dataclass(frozen=True)
class Pi05BrakingComparisonConfig:
    chunk_count: int = 90
    selection_seed: int = 20260807
    scenarios: tuple[str, ...] = (
        "free_space",
        "single_block",
        "narrow_gate",
    )
    response_deadline_ms: float = 200.0
    repair_selection_deadline_ms: float = 20.0
    collision_resolution_rad: float = 0.02

    def __post_init__(self) -> None:
        if type(self.chunk_count) is not int or self.chunk_count <= 0:
            raise ValueError("chunk_count must be a positive integer")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("selection_seed must be nonnegative")
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or any(name not in mujoco_scenarios() for name in self.scenarios)
        ):
            raise ValueError("scenarios must be unique known MuJoCo scenarios")
        for label, value in (
            ("response_deadline_ms", self.response_deadline_ms),
            ("repair_selection_deadline_ms", self.repair_selection_deadline_ms),
            ("collision_resolution_rad", self.collision_resolution_rad),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")


def _guard_config(robot: MuJoCoPanda, config: Pi05BrakingComparisonConfig) -> GuardConfig:
    return GuardConfig(
        control_dt_s=CartesianAdapterConfig().control_dt_s,
        deadline_ms=config.response_deadline_ms,
        joint_velocity_clip_rad_s=float(np.max(robot.velocity_limits)),
    )


def _run_case(
    archive: ValidatedPi05Archive,
    source_index: int,
    selection_index: int,
    trajectory_index: int,
    scenario_name: str,
    robot: MuJoCoPanda,
    config: Pi05BrakingComparisonConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    query = archive.transition_queries[source_index]
    scenario = mujoco_scenarios()[scenario_name]
    source_actions = archive.arrays["response_actions"][source_index]
    latency_ms = float(query["inference_latency_ms"])
    captured_at_s = 1000.0
    adapter = PandaCartesianActionAdapter(robot)
    adapted = adapter.adapt(
        source_actions,
        scenario.start,
        source=(
            f"official_pi05_frozen_response:{archive.response_hashes[source_index]}"
        ),
        observation_sequence_id=source_index,
        inference_latency_ms=latency_ms,
        received_at_s=captured_at_s + latency_ms / 1000.0,
    )
    image = np.zeros(DROID_IMAGE_SHAPE, dtype=np.uint8)
    observation = VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=scenario.start,
        gripper_position=np.array([1.0]),
        prompt="offline paired braking repair",
        sequence_id=source_index,
        captured_at_s=captured_at_s,
    )
    guard_config = _guard_config(robot, config)

    raw_checker = MuJoCoCollisionChecker(
        robot, resolution=config.collision_resolution_rad
    )
    raw_path_safe = raw_checker.path_is_valid(adapted.predicted_positions)
    legacy_checker = MuJoCoCollisionChecker(
        robot, resolution=config.collision_resolution_rad
    )
    legacy = ActionChunkGuard(legacy_checker, guard_config).guard(
        scenario.start,
        1.0,
        observation,
        adapted.chunk,
    )
    legacy_path_safe = legacy_checker.path_is_valid(legacy.predicted_positions)
    repair_checker = MuJoCoCollisionChecker(
        robot, resolution=config.collision_resolution_rad
    )
    repair = BrakingTrajectoryGuard(
        repair_checker,
        guard_config,
        BrakingRepairConfig(
            selection_deadline_ms=config.repair_selection_deadline_ms
        ),
    ).repair(
        scenario.start,
        1.0,
        observation,
        adapted.chunk,
    )
    repair_path_safe = repair_checker.path_is_valid(repair.predicted_positions)
    terminal_path_safe = repair_checker.path_is_valid(
        repair.terminal_braking_positions
    )

    start_hand = robot.hand_position(scenario.start)
    raw_end = robot.hand_position(adapted.predicted_positions[-1])
    legacy_end = robot.hand_position(legacy.predicted_positions[-1])
    repair_end = robot.hand_position(repair.predicted_positions[-1])
    legacy_conflict_resolved = bool(
        not legacy.safe_after_guard and repair.safe_after_repair
    )
    repair_regression = bool(
        legacy.safe_after_guard and not repair.safe_after_repair
    )
    record = {
        "case_id": f"row_{source_index:05d}__{scenario_name}",
        "trajectory_index": trajectory_index,
        "selection_index": selection_index,
        "source_row_index": source_index,
        "task_suite": str(query["task_suite"]),
        "task_id": int(query["task_id"]),
        "method": str(query["method"]),
        "pair_id": str(query["pair_id"]),
        "episode_id": str(query["episode_id"]),
        "episode_index": int(query["episode_index"]),
        "query_index": int(query["query_index"]),
        "scenario": scenario_name,
        "response_action_sha256": archive.response_hashes[source_index],
        "inference_latency_ms": latency_ms,
        "response_deadline_exceeded": repair.response_deadline_exceeded,
        "input_clipped_steps": adapted.clipped_input_steps,
        "raw_path_safe": bool(raw_path_safe),
        "legacy_safe_after_guard": legacy.safe_after_guard,
        "legacy_path_safe": bool(legacy_path_safe),
        "legacy_unsafe_raw_steps": legacy.unsafe_raw_steps,
        "legacy_intervention_steps": legacy.intervention_steps,
        "legacy_hold_steps": legacy.hold_steps,
        "legacy_acceleration_override_steps": legacy.acceleration_override_steps,
        "legacy_max_acceleration_rad_s2": legacy.max_guarded_acceleration_rad_s2,
        "legacy_guard_latency_ms": legacy.guard_latency_ms,
        "repair_safe_after_guard": repair.safe_after_repair,
        "repair_path_safe": bool(repair_path_safe),
        "repair_terminal_path_safe": bool(terminal_path_safe),
        "repair_selected_scale": repair.selected_scale,
        "repair_terminal_brake_steps": repair.terminal_brake_steps,
        "repair_intervention_steps": repair.intervention_steps,
        "repair_max_acceleration_rad_s2": repair.max_repaired_acceleration_rad_s2,
        "repair_selection_deadline_exceeded": (
            repair.selection_deadline_exceeded
        ),
        "repair_evaluated_scale_count": len(repair.evaluated_scales),
        "repair_collision_edge_checks": repair.collision_edge_checks,
        "repair_latency_ms": repair.repair_latency_ms,
        "repair_fallback_reason": repair.fallback_reason,
        "legacy_conflict_resolved": legacy_conflict_resolved,
        "repair_regression": repair_regression,
        "raw_hand_displacement_m": float(np.linalg.norm(raw_end - start_hand)),
        "legacy_hand_displacement_m": float(
            np.linalg.norm(legacy_end - start_hand)
        ),
        "repair_hand_displacement_m": float(
            np.linalg.norm(repair_end - start_hand)
        ),
    }
    trajectories = {
        "raw": adapted.predicted_positions,
        "legacy": legacy.predicted_positions,
        "repair": repair.predicted_positions,
        "terminal": repair.terminal_braking_positions,
    }
    return record, trajectories


def _p95(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def _scale_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot summarize an empty repair comparison")
    cases = len(rows)
    scales: dict[str, int] = {}
    for row in rows:
        label = _scale_label(float(row["repair_selected_scale"]))
        scales[label] = scales.get(label, 0) + 1
    return {
        "cases": cases,
        "unique_chunks": len({int(row["source_row_index"]) for row in rows}),
        "response_deadline_exceeded_cases": sum(
            bool(row["response_deadline_exceeded"]) for row in rows
        ),
        "raw_path_invalid_cases": sum(not bool(row["raw_path_safe"]) for row in rows),
        "legacy_constraint_satisfied_cases": sum(
            bool(row["legacy_safe_after_guard"]) for row in rows
        ),
        "legacy_path_valid_cases": sum(bool(row["legacy_path_safe"]) for row in rows),
        "legacy_acceleration_conflict_cases": sum(
            int(row["legacy_acceleration_override_steps"]) > 0 for row in rows
        ),
        "legacy_intervention_cases": sum(
            int(row["legacy_intervention_steps"]) > 0 for row in rows
        ),
        "repair_constraint_satisfied_cases": sum(
            bool(row["repair_safe_after_guard"]) for row in rows
        ),
        "repair_path_valid_cases": sum(bool(row["repair_path_safe"]) for row in rows),
        "repair_terminal_path_valid_cases": sum(
            bool(row["repair_terminal_path_safe"]) for row in rows
        ),
        "repair_selection_deadline_exceeded_cases": sum(
            bool(row["repair_selection_deadline_exceeded"]) for row in rows
        ),
        "repair_intervention_cases": sum(
            int(row["repair_intervention_steps"]) > 0 for row in rows
        ),
        "legacy_conflicts_resolved": sum(
            bool(row["legacy_conflict_resolved"]) for row in rows
        ),
        "repair_regressions": sum(bool(row["repair_regression"]) for row in rows),
        "selected_scale_counts": dict(
            sorted(scales.items(), key=lambda item: -float(item[0]))
        ),
        "max_terminal_brake_steps": max(
            int(row["repair_terminal_brake_steps"]) for row in rows
        ),
        "mean_legacy_guard_latency_ms": fmean(
            float(row["legacy_guard_latency_ms"]) for row in rows
        ),
        "p95_legacy_guard_latency_ms": _p95(
            [float(row["legacy_guard_latency_ms"]) for row in rows]
        ),
        "mean_repair_latency_ms": fmean(
            float(row["repair_latency_ms"]) for row in rows
        ),
        "p95_repair_latency_ms": _p95(
            [float(row["repair_latency_ms"]) for row in rows]
        ),
        "max_repair_latency_ms": max(float(row["repair_latency_ms"]) for row in rows),
        "mean_raw_hand_displacement_m": fmean(
            float(row["raw_hand_displacement_m"]) for row in rows
        ),
        "mean_legacy_hand_displacement_m": fmean(
            float(row["legacy_hand_displacement_m"]) for row in rows
        ),
        "mean_repair_hand_displacement_m": fmean(
            float(row["repair_hand_displacement_m"]) for row in rows
        ),
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_transition_count: int,
    source_hashes_verified: int,
    selection_seed: int,
) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    return {
        "schema_version": COMPARISON_SUMMARY_SCHEMA,
        "scope": COMPARISON_SCOPE,
        "source_policy_checkpoint_attested": True,
        "policy_checkpoint_executed_in_replay": False,
        "task_success_evaluated": False,
        "panda_closed_loop_executed": False,
        "source_validation": {
            "transition_count": source_transition_count,
            "response_action_hashes_verified": source_hashes_verified,
        },
        "selection": {
            "algorithm": "sha256_rank_equal_per_task_method_v1",
            "seed": selection_seed,
            "chunk_count": len(
                {int(row["source_row_index"]) for row in rows}
            ),
        },
        "overall": _aggregate(rows),
        "by_method": {
            method: _aggregate([row for row in rows if row["method"] == method])
            for method in methods
        },
        "by_scenario": {
            scenario: _aggregate(
                [row for row in rows if row["scenario"] == scenario]
            )
            for scenario in scenarios
        },
        "interpretation": {
            "registered_effect": "legacy_constraint_conflicts_resolved",
            "method_comparison_registered": False,
            "task_efficacy_claim": False,
            "hard_realtime_claim": False,
            "physical_safety_claim": False,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "CSV schema mismatch")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for field in CSV_FIELDS:
            value = raw[field]
            if field in _INTEGER_FIELDS:
                try:
                    row[field] = int(value)
                except ValueError as error:
                    raise ValueError(f"invalid CSV integer: {field}") from error
            elif field in _FLOAT_FIELDS:
                try:
                    row[field] = float(value)
                except ValueError as error:
                    raise ValueError(f"invalid CSV float: {field}") from error
                _require(np.isfinite(row[field]), f"nonfinite CSV float: {field}")
            elif field in _BOOLEAN_FIELDS:
                _require(value in {"True", "False"}, f"invalid CSV boolean: {field}")
                row[field] = value == "True"
            elif field == "repair_fallback_reason":
                row[field] = value or None
            else:
                _require(bool(value), f"empty CSV field: {field}")
                row[field] = value
        rows.append(row)
    return rows


def _write_trajectories(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, np.ndarray]],
    config: Pi05BrakingComparisonConfig,
) -> None:
    count = len(rows)
    terminal_size = BrakingRepairConfig().max_terminal_brake_steps + 1
    terminal = np.zeros((count, terminal_size, 7), dtype=float)
    terminal_lengths = np.zeros(count, dtype=np.int32)
    for index, values in enumerate(trajectories):
        braking = np.asarray(values["terminal"], dtype=float)
        length = len(braking)
        _require(length <= terminal_size, "terminal braking trace is too long")
        terminal[index, :length] = braking
        terminal[index, length:] = braking[-1]
        terminal_lengths[index] = length
    np.savez_compressed(
        path,
        schema_version=np.asarray(TRAJECTORY_SCHEMA),
        case_ids=np.asarray([row["case_id"] for row in rows]),
        source_row_indices=np.asarray(
            [row["source_row_index"] for row in rows], dtype=np.int32
        ),
        scenarios=np.asarray([row["scenario"] for row in rows]),
        times_s=(
            np.arange(11, dtype=float) * CartesianAdapterConfig().control_dt_s
        ),
        raw_positions=np.stack([values["raw"] for values in trajectories]),
        legacy_positions=np.stack(
            [values["legacy"] for values in trajectories]
        ),
        repair_positions=np.stack(
            [values["repair"] for values in trajectories]
        ),
        terminal_braking_positions=terminal,
        terminal_braking_lengths=terminal_lengths,
    )


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    scales = ", ".join(
        f"{scale}: {count}"
        for scale, count in overall["selected_scale_counts"].items()
    )
    return "\n".join(
        (
            "# Braking-invariant repair on frozen pi0.5 responses",
            "",
            "This paired offline diagnostic compares the existing greedy guard",
            "with a trajectory-scale search that preserves a collision-valid",
            "terminal braking path. The checkpoint was not executed in this run.",
            "",
            "## Paired outcome",
            "",
            "| Metric | Greedy guard | Braking-invariant repair |",
            "| --- | ---: | ---: |",
            f"| All registered constraints satisfied | {overall['legacy_constraint_satisfied_cases']} / {overall['cases']} | {overall['repair_constraint_satisfied_cases']} / {overall['cases']} |",
            f"| Position path valid | {overall['legacy_path_valid_cases']} / {overall['cases']} | {overall['repair_path_valid_cases']} / {overall['cases']} |",
            f"| Acceleration-conflict cases | {overall['legacy_acceleration_conflict_cases']} | {overall['cases'] - overall['repair_constraint_satisfied_cases']} |",
            f"| P95 software latency | {overall['p95_legacy_guard_latency_ms']:.3f} ms | {overall['p95_repair_latency_ms']:.3f} ms |",
            "",
            f"Resolved legacy conflicts: {overall['legacy_conflicts_resolved']}",
            f"Repair regressions: {overall['repair_regressions']}",
            f"Selected trajectory scales: {scales}",
            f"Selection-deadline exceedances: {overall['repair_selection_deadline_exceeded_cases']}",
            "",
            "## Claim boundary",
            "",
            "No task success, Panda closed loop, hard-real-time guarantee,",
            "continuous-collision certificate, or physical-safety claim is made.",
            "",
        )
    )


def _provenance(
    archive: ValidatedPi05Archive,
    config: Pi05BrakingComparisonConfig,
    selected: Sequence[int],
) -> dict[str, Any]:
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("trajectory_repair.py"),
        Path(__file__).with_name("guard.py"),
        Path(__file__).with_name("cartesian_adapter.py"),
    )
    return {
        "schema_version": COMPARISON_PROVENANCE_SCHEMA,
        "scope": COMPARISON_SCOPE,
        "source_policy_checkpoint_attested": True,
        "policy_checkpoint_executed_in_replay": False,
        "task_success_evaluated": False,
        "panda_closed_loop_executed": False,
        "source": {
            "artifact_id": (
                archive.root.parent.name
                if archive.root.name == "evaluation"
                else archive.root.name
            ),
            "root_manifest_sha256": archive.root_manifest_sha256,
            "root_manifest_files_sha256": archive.root_manifest["files_sha256"],
            "transition_archive_sha256": archive.descriptor["archive"]["sha256"],
            "transition_count": archive.transition_count,
            "response_action_hashes_verified": len(archive.response_hashes),
        },
        "policy": dict(archive.descriptor["policy"]),
        "source_action_adapter": dict(archive.descriptor["action_adapter"]),
        "comparison": {
            "baseline": "greedy_per_step_backtracking_v1",
            "repair": "whole_chunk_scale_search_terminal_braking_v1",
            "response_deadline_ms": config.response_deadline_ms,
            "repair_selection_deadline_ms": config.repair_selection_deadline_ms,
            "collision_resolution_rad": config.collision_resolution_rad,
            "trajectory_scales": list(BrakingRepairConfig().trajectory_scales),
            "max_scale_evaluations": (
                BrakingRepairConfig().max_scale_evaluations
            ),
            "max_terminal_brake_steps": (
                BrakingRepairConfig().max_terminal_brake_steps
            ),
            "deadline_is_os_hard_realtime": False,
        },
        "selection": {
            "algorithm": "sha256_rank_equal_per_task_method_v1",
            "seed": config.selection_seed,
            "source_row_indices": list(selected),
            "scenarios": list(config.scenarios),
            "mujoco_scenario_version": MUJOCO_SCENARIO_VERSION,
        },
        "action_contract": {
            "source_adapter": PI05_ACTION_ADAPTER,
            "source_adapter_path": PI05_ACTION_ADAPTER_SOURCE,
            "controller_semantics_id": LIBERO_CONTROLLER_SEMANTICS_ID,
            "kinematic_control_point_id": PANDA_KINEMATIC_CONTROL_POINT_ID,
        },
        "local_runtime": {
            "python": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
            "panda_scene_sha256": _sha256_file(default_panda_scene_path()),
            "implementation_sha256": {
                f"armbench/vla/{path.name}": _sha256_file(path)
                for path in implementation_paths
            },
        },
        "limitations": [
            "Frozen responses are replayed without policy inference or feedback.",
            "Whole-chunk scaling is conservative and does not optimize task progress.",
            "The wall-clock selection deadline is not an OS scheduling guarantee.",
            "Collision validity uses resolution-bounded joint-space edge sampling.",
        ],
    }


def execute_pi05_braking_comparison(
    source_directory: Path,
    output_directory: Path,
    config: Pi05BrakingComparisonConfig = Pi05BrakingComparisonConfig(),
) -> Path:
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    archive = validate_pi05_source_archive(source_directory)
    selected = select_stratified_chunks(
        archive,
        config.chunk_count,
        config.selection_seed,
    )
    scenarios = mujoco_scenarios()
    robots = {
        name: MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        for name in config.scenarios
    }
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, np.ndarray]] = []
    for selection_index, source_index in enumerate(selected):
        for scenario_name in config.scenarios:
            record, trace = _run_case(
                archive,
                source_index,
                selection_index,
                len(rows),
                scenario_name,
                robots[scenario_name],
                config,
            )
            rows.append(record)
            trajectories.append(trace)

    provenance = _provenance(archive, config, selected)
    summary = _build_summary(
        rows,
        source_transition_count=archive.transition_count,
        source_hashes_verified=len(archive.response_hashes),
        selection_seed=config.selection_seed,
    )
    output.mkdir(parents=True)
    _write_json(output / "provenance.json", provenance)
    _write_csv(output / "per_case.csv", rows)
    _write_trajectories(output / "trajectories.npz", rows, trajectories, config)
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_root_manifest(output)
    validate_pi05_braking_comparison(output, source_directory=archive.root)
    return output


def validate_pi05_braking_comparison(
    directory: Path,
    source_directory: Path | None = None,
) -> dict[str, Any]:
    root = directory.resolve()
    _require(root.is_dir(), f"comparison directory not found: {root}")
    manifest = _validate_root_manifest(root)
    expected_paths = {
        "per_case.csv",
        "provenance.json",
        "summary.json",
        "summary.md",
        "trajectories.npz",
    }
    _require(
        {str(item["path"]) for item in manifest["files"]} == expected_paths,
        "comparison artifact file set mismatch",
    )
    provenance = _load_json(root / "provenance.json")
    summary = _load_json(root / "summary.json")
    rows = _read_csv(root / "per_case.csv")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("schema_version") == COMPARISON_PROVENANCE_SCHEMA,
        "comparison provenance schema mismatch",
    )
    _require(
        isinstance(summary, Mapping)
        and summary.get("schema_version") == COMPARISON_SUMMARY_SCHEMA,
        "comparison summary schema mismatch",
    )
    for document in (provenance, summary):
        _require(document.get("scope") == COMPARISON_SCOPE, "scope mismatch")
        _require(
            document.get("source_policy_checkpoint_attested") is True,
            "checkpoint attestation flag missing",
        )
        for field in (
            "policy_checkpoint_executed_in_replay",
            "task_success_evaluated",
            "panda_closed_loop_executed",
        ):
            _require(document.get(field) is False, f"invalid claim flag: {field}")
    _require(
        provenance.get("policy")
        == {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
        },
        "comparison policy provenance mismatch",
    )
    source = provenance.get("source")
    selection = provenance.get("selection")
    _require(
        isinstance(source, Mapping) and isinstance(selection, Mapping),
        "comparison provenance sections missing",
    )
    expected_summary = _build_summary(
        rows,
        source_transition_count=int(source["transition_count"]),
        source_hashes_verified=int(source["response_action_hashes_verified"]),
        selection_seed=int(selection["seed"]),
    )
    _require(summary == expected_summary, "summary is not reproducible from CSV")
    _require(
        [int(row["trajectory_index"]) for row in rows] == list(range(len(rows))),
        "trajectory indices are not contiguous",
    )
    expected_cases = [
        (int(source_index), str(scenario))
        for source_index in selection["source_row_indices"]
        for scenario in selection["scenarios"]
    ]
    _require(
        [
            (int(row["source_row_index"]), str(row["scenario"]))
            for row in rows
        ]
        == expected_cases,
        "CSV does not match the selected paired matrix",
    )

    try:
        with np.load(root / "trajectories.npz", allow_pickle=False) as loaded:
            _require(set(loaded.files) == TRAJECTORY_KEYS, "trajectory fields mismatch")
            trajectory_values = {
                key: np.array(loaded[key], copy=True) for key in loaded.files
            }
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("cannot read trajectory archive") from error
    _require(
        str(trajectory_values["schema_version"].item()) == TRAJECTORY_SCHEMA,
        "trajectory schema mismatch",
    )
    count = len(rows)
    for key in ("raw_positions", "legacy_positions", "repair_positions"):
        value = trajectory_values[key]
        _require(
            value.shape == (count, 11, 7) and bool(np.all(np.isfinite(value))),
            f"trajectory shape or values invalid: {key}",
        )
    terminal = trajectory_values["terminal_braking_positions"]
    lengths = trajectory_values["terminal_braking_lengths"]
    _require(
        terminal.shape == (count, 9, 7)
        and bool(np.all(np.isfinite(terminal)))
        and lengths.shape == (count,)
        and lengths.dtype == np.dtype("<i4")
        and bool(np.all((lengths >= 1) & (lengths <= 9))),
        "terminal braking trajectory contract mismatch",
    )
    _require(
        trajectory_values["case_ids"].tolist()
        == [str(row["case_id"]) for row in rows]
        and trajectory_values["source_row_indices"].tolist()
        == [int(row["source_row_index"]) for row in rows]
        and trajectory_values["scenarios"].tolist()
        == [str(row["scenario"]) for row in rows],
        "trajectory identities do not match CSV",
    )
    _require(
        trajectory_values["times_s"].shape == (11,)
        and bool(np.all(np.diff(trajectory_values["times_s"]) > 0.0)),
        "trajectory time axis mismatch",
    )
    for index, row in enumerate(rows):
        start = mujoco_scenarios()[str(row["scenario"])].start
        for key in ("raw_positions", "legacy_positions", "repair_positions"):
            _require(
                np.array_equal(trajectory_values[key][index, 0], start),
                f"trajectory start state mismatch: {key}",
            )

    checks = [
        "manifest_inventory_sizes_and_hashes",
        "claim_boundary_flags",
        "paired_matrix_identity",
        "summary_recomputed_from_csv",
        "trajectory_archive_shapes_and_identities",
    ]
    if source_directory is not None:
        archive = validate_pi05_source_archive(source_directory)
        _require(
            archive.root_manifest_sha256 == source["root_manifest_sha256"]
            and archive.root_manifest["files_sha256"]
            == source["root_manifest_files_sha256"]
            and archive.descriptor["archive"]["sha256"]
            == source["transition_archive_sha256"]
            and archive.transition_count == source["transition_count"]
            and len(archive.response_hashes)
            == source["response_action_hashes_verified"],
            "comparison does not match supplied source artifact",
        )
        for row in rows:
            source_index = int(row["source_row_index"])
            query = archive.transition_queries[source_index]
            _require(
                row["response_action_sha256"]
                == archive.response_hashes[source_index]
                and row["task_suite"] == query["task_suite"]
                and row["task_id"] == query["task_id"]
                and row["method"] == query["method"]
                and row["pair_id"] == query["pair_id"]
                and row["episode_id"] == query["episode_id"]
                and row["episode_index"] == query["episode_index"]
                and row["query_index"] == query["query_index"]
                and row["inference_latency_ms"] == query["inference_latency_ms"],
                "comparison CSV row does not match supplied source",
            )
        checks.append("source_archive_reverified")
    markdown = (root / "summary.md").read_text(encoding="utf-8")
    _require(
        "No task success" in markdown
        and "hard-real-time" in markdown,
        "human-readable claim boundary missing",
    )
    return {
        "valid": True,
        "scope": COMPARISON_SCOPE,
        "chunks": int(summary["selection"]["chunk_count"]),
        "cases": int(summary["overall"]["cases"]),
        "legacy_conflicts_resolved": int(
            summary["overall"]["legacy_conflicts_resolved"]
        ),
        "repair_regressions": int(summary["overall"]["repair_regressions"]),
        "manifest_files_sha256": manifest["files_sha256"],
        "checks": checks,
    }
