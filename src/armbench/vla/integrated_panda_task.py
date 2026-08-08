"""Closed-loop MuJoCo task evidence for integrated Panda action assurance."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import mujoco
import numpy as np
import osqp

from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.execution import execute_trajectory
from armbench.mujoco_sim.model import (
    MENAGERIE_COMMIT,
    MuJoCoPanda,
    default_panda_scene_path,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.planners import RRTConnect
from armbench.postprocess import Trajectory, shortcut_path, time_parameterize
from armbench.result import path_length
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
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


TASK_SCHEMA = "armbench.integrated_panda_task.v1"
TASK_SCOPE = "offline_assured_joint_waypoint_task_with_mujoco_torque_execution"
FIXED_GRIPPER_ALLOWED_BODY_PAIR = frozenset(("left_finger", "right_finger"))
REGISTERED_PROFILES: Mapping[str, Mapping[str, object]] = {
    "free_space_short_smoke": {
        "scenario": "free_space",
        "payload_mass_kg": 0.0,
        "speed_scale": 0.10,
        "delay_ms": 0,
        "feedback_mode": "delayed",
        "target_fraction": 0.02,
        "planning_seed": 0,
        "kp": [120.0, 120.0, 100.0, 100.0, 60.0, 40.0, 30.0],
        "kd": [22.0, 22.0, 18.0, 18.0, 12.0, 8.0, 6.0],
    },
    "single_block_goal": {
        "scenario": "single_block",
        "payload_mass_kg": 0.0,
        "speed_scale": 0.15,
        "delay_ms": 0,
        "feedback_mode": "delayed",
        "target_fraction": 1.0,
        "planning_seed": 1,
        "kp": [120.0, 120.0, 100.0, 100.0, 60.0, 40.0, 30.0],
        "kd": [22.0, 22.0, 18.0, 18.0, 12.0, 8.0, 6.0],
    },
    "narrow_gate_payload_delay_goal": {
        "scenario": "narrow_gate",
        "payload_mass_kg": 0.5,
        "speed_scale": 0.10,
        "delay_ms": 80,
        "feedback_mode": "velocity_prediction",
        "target_fraction": 1.0,
        "planning_seed": 0,
        "kp": 5.0,
        "kd": 2.0,
    },
}
DEFAULT_TASK_PROFILES = (
    "single_block_goal",
    "narrow_gate_payload_delay_goal",
)
CSV_FIELDS = (
    "schema_version",
    "case_id",
    "scenario",
    "target_is_scenario_goal",
    "payload_mass_kg",
    "delay_ms",
    "feedback_mode",
    "speed_scale",
    "clearance_m",
    "planning_waypoints",
    "planned_path_length_rad",
    "planning_latency_ms",
    "action_horizon",
    "supervisor_status",
    "supervisor_reason",
    "supervision_latency_ms",
    "qp_intervention_steps",
    "continuous_edges_checked",
    "braking_boundaries_checked",
    "excluded_fixed_gripper_pairs",
    "guard_target_error_rad",
    "target_reached",
    "physical_safe",
    "safe_task_success",
    "tracking_rmse_rad",
    "max_tracking_error_rad",
    "final_target_error_rad",
    "torque_saturation_count",
    "joint_limit_violation_steps",
    "obstacle_contact_steps",
    "self_contact_steps",
    "trace_path",
)
_SUMMARY_FIELDS = {
    "schema_version",
    "scope",
    "configuration",
    "environment",
    "panda_scene_sha256",
    "implementation_sha256",
    "overall",
    "cases",
    "allowed_collision_matrix",
    "claim_boundary",
}
_ENVIRONMENT_FIELDS = {
    "mujoco_version",
    "numpy_version",
    "osqp_version",
    "menagerie_commit",
}
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256"}
_CASE_DOCUMENT_FIELDS = {"schema_version", "cases"}
_CASE_FIELDS = {
    "case_id",
    "scenario",
    "payload_mass_kg",
    "speed_scale",
    "delay_ms",
    "feedback_mode",
    "target_fraction",
    "target_is_scenario_goal",
    "planning_seed",
    "kp",
    "kd",
}
_AGGREGATE_FIELDS = {
    "cases",
    "supervisor_accepted",
    "target_reached",
    "physical_safe",
    "safe_task_success",
    "qp_intervention_steps",
    "continuous_edges_checked",
    "braking_boundaries_checked",
    "p95_supervision_latency_ms",
    "maximum_supervision_latency_ms",
    "mean_tracking_rmse_rad",
    "maximum_tracking_error_rad",
    "maximum_final_target_error_rad",
    "obstacle_contact_steps",
    "self_contact_steps",
    "joint_limit_violation_steps",
}
_TRACE_FIELDS = {
    "schema_version",
    "certified_times_s",
    "certified_positions",
    "certified_velocities",
    "times",
    "desired_positions",
    "actual_positions",
    "applied_torques",
}
_CLAIM_BOUNDARY = [
    "The policy chunk is scripted from an RRT-Connect reference, not a learned VLA output.",
    "Assurance is computed offline before execution and is not a hard-real-time loop.",
    "The task is arm joint-waypoint reaching; grasp success and object manipulation are out of scope.",
    "Execution uses MuJoCo torque control, not a physical robot.",
    "The fixed-open gripper body pair is explicitly excluded; all other registered self pairs remain.",
]


@dataclass(frozen=True)
class IntegratedPandaTaskConfig:
    """Registered planning, assurance, and execution settings."""

    profiles: tuple[str, ...] = DEFAULT_TASK_PROFILES
    clearance_m: float = 0.02
    collision_resolution_rad: float = 0.05
    planner_step_size_rad: float = 0.35
    planner_max_iterations: int = 2000
    planner_timeout_s: float = 2.0
    planner_goal_sample_rate: float = 0.05
    shortcut_attempts: int = 150
    action_dt_s: float = 0.05
    settle_action_steps: int = 10
    response_deadline_ms: float = 120_000.0
    supervision_budget_ms: float = 120_000.0
    qp_step_budget_ms: float = 500.0
    execution_control_dt_s: float = 0.01
    execution_warmup_s: float = 0.1
    execution_hold_s: float = 0.5
    goal_tolerance_rad: float = 0.05

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profiles, tuple)
            or not self.profiles
            or any(type(name) is not str or name not in REGISTERED_PROFILES for name in self.profiles)
            or len(set(self.profiles)) != len(self.profiles)
        ):
            raise ValueError("integrated task profiles are invalid")
        numeric = np.asarray(
            [
                self.clearance_m,
                self.collision_resolution_rad,
                self.planner_step_size_rad,
                self.planner_timeout_s,
                self.planner_goal_sample_rate,
                self.action_dt_s,
                self.response_deadline_ms,
                self.supervision_budget_ms,
                self.qp_step_budget_ms,
                self.execution_control_dt_s,
                self.execution_warmup_s,
                self.execution_hold_s,
                self.goal_tolerance_rad,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(numeric))
            or self.clearance_m < 0.0
            or np.any(numeric[1:9] <= 0.0)
            or self.execution_control_dt_s <= 0.0
            or self.execution_warmup_s < 0.0
            or self.execution_hold_s < 0.0
            or self.goal_tolerance_rad < 0.0
            or not 0.0 < self.planner_goal_sample_rate <= 1.0
            or type(self.planner_max_iterations) is not int
            or self.planner_max_iterations <= 0
            or type(self.shortcut_attempts) is not int
            or self.shortcut_attempts < 0
            or type(self.settle_action_steps) is not int
            or self.settle_action_steps <= 0
        ):
            raise ValueError("integrated task timing and planning limits are invalid")
        object.__setattr__(self, "profiles", tuple(self.profiles))


def _case_specs(config: IntegratedPandaTaskConfig) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for name in config.profiles:
        profile = REGISTERED_PROFILES[name]
        fraction = float(profile["target_fraction"])
        cases.append(
            {
                "case_id": name,
                "scenario": profile["scenario"],
                "payload_mass_kg": profile["payload_mass_kg"],
                "speed_scale": profile["speed_scale"],
                "delay_ms": profile["delay_ms"],
                "feedback_mode": profile["feedback_mode"],
                "target_fraction": fraction,
                "target_is_scenario_goal": bool(np.isclose(fraction, 1.0)),
                "planning_seed": profile["planning_seed"],
                "kp": profile["kp"],
                "kd": profile["kd"],
            }
        )
    return cases


def make_integrated_task_checker(
    robot: MuJoCoPanda,
) -> tuple[ContinuousMuJoCoCollisionChecker, int]:
    """Build the arm-only checker with one explicit fixed-gripper exclusion."""

    checker = ContinuousMuJoCoCollisionChecker(robot)
    retained = []
    excluded = 0
    for pair in checker.pairs:
        bodies = frozenset(
            (
                robot.body_name_for_geom(pair.geom1),
                robot.body_name_for_geom(pair.geom2),
            )
        )
        if pair.kind == "self_collision" and bodies == FIXED_GRIPPER_ALLOWED_BODY_PAIR:
            excluded += 1
        else:
            retained.append(pair)
    if excluded == 0:
        raise RuntimeError("fixed gripper collision pair was not registered")
    checker.pairs = tuple(retained)
    return checker, excluded


def _plan_reference(
    spec: Mapping[str, object],
    config: IntegratedPandaTaskConfig,
    robot: MuJoCoPanda,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    scenario = mujoco_scenarios()[str(spec["scenario"])]
    fraction = float(spec["target_fraction"])
    target = scenario.start + fraction * (scenario.goal - scenario.start)
    if fraction < 1.0:
        path = [scenario.start.copy(), target.copy()]
        checker = MuJoCoCollisionChecker(
            robot, resolution=config.collision_resolution_rad
        )
        if not checker.path_is_valid(path):
            raise RuntimeError("registered short task path is not collision valid")
        return path, target, 0.0
    checker = MuJoCoCollisionChecker(
        robot, resolution=config.collision_resolution_rad
    )
    planning_seed = int(spec["planning_seed"])
    planner = RRTConnect(
        checker,
        np.random.default_rng(planning_seed),
        step_size=config.planner_step_size_rad,
        max_iterations=config.planner_max_iterations,
        timeout_s=config.planner_timeout_s,
        goal_sample_rate=config.planner_goal_sample_rate,
    )
    planned = planner.plan(scenario.start, target)
    if not planned.success:
        raise RuntimeError(f"integrated task planning failed: {planned.status.value}")
    smoothed = shortcut_path(
        planned.path,
        checker,
        np.random.default_rng(planning_seed + 200_000),
        attempts=config.shortcut_attempts,
    )
    return smoothed.path, target, planned.elapsed_s * 1000.0


def _scripted_chunk(
    path: list[np.ndarray],
    robot: MuJoCoPanda,
    spec: Mapping[str, object],
    config: IntegratedPandaTaskConfig,
) -> tuple[ActionChunk, Trajectory]:
    reference = time_parameterize(
        path,
        robot.velocity_limits,
        control_dt=config.action_dt_s,
        speed_scale=float(spec["speed_scale"]),
    )
    motion_steps = len(reference.positions) - 1
    actions = np.zeros((motion_steps + config.settle_action_steps, 8), dtype=float)
    actions[:motion_steps, :7] = (
        np.diff(reference.positions, axis=0) / config.action_dt_s
    )
    actions[:, 7] = 1.0
    return (
        ActionChunk(
            actions=actions,
            source="scripted_nonlearned_rrt_reference",
            observation_sequence_id=0,
            inference_latency_ms=0.0,
            received_at_s=1.0,
        ),
        reference,
    )


def _certified_trajectory(
    positions: np.ndarray, actions: np.ndarray, dt: float
) -> Trajectory:
    times = np.arange(len(positions), dtype=float) * dt
    velocities = np.vstack((np.zeros((1, 7)), actions[:, :7]))
    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        segment_durations=np.diff(times),
    )


def _trace_document(
    certified: Trajectory,
    execution: object,
) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(TASK_SCHEMA),
        "certified_times_s": certified.times,
        "certified_positions": certified.positions,
        "certified_velocities": certified.velocities,
        "times": execution.times,
        "desired_positions": execution.desired_positions,
        "actual_positions": execution.actual_positions,
        "applied_torques": execution.applied_torques,
    }


def _evaluate_case(
    spec: Mapping[str, object],
    config: IntegratedPandaTaskConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scenario = mujoco_scenarios()[str(spec["scenario"])]
    payload = float(spec["payload_mass_kg"])
    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, config.clearance_m),
        payload_mass=payload,
    )
    path, target, planning_latency_ms = _plan_reference(spec, config, guard_robot)
    chunk, _ = _scripted_chunk(path, guard_robot, spec, config)
    checker, excluded_pairs = make_integrated_task_checker(guard_robot)
    supervisor = IntegratedPandaSupervisor(
        guard_robot,
        checker,
        IntegratedPandaGuardConfig(
            control_dt_s=config.action_dt_s,
            response_deadline_ms=config.response_deadline_ms,
            supervision_budget_ms=config.supervision_budget_ms,
            qp_step_budget_ms=config.qp_step_budget_ms,
        ),
    )
    supervised_started = perf_counter()
    decision = supervisor.supervise(
        scenario.start,
        np.zeros(7),
        chunk,
        observed_q=scenario.start,
        response_age_ms=0.0,
    )
    supervision_latency_ms = (perf_counter() - supervised_started) * 1000.0
    if decision.status != "accepted":
        raise RuntimeError(
            f"registered integrated task rejected: {spec['case_id']}: "
            f"{decision.failure_stage}/{decision.reason}"
        )
    certified = _certified_trajectory(
        decision.predicted_positions,
        decision.executable_actions,
        config.action_dt_s,
    )
    execution_robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        payload_mass=payload,
        torque_control=True,
    )
    execution = execute_trajectory(
        execution_robot,
        certified,
        delay_ms=int(spec["delay_ms"]),
        control_dt=config.execution_control_dt_s,
        warmup_s=config.execution_warmup_s,
        hold_s=config.execution_hold_s,
        goal_tolerance=config.goal_tolerance_rad,
        feedback_mode=str(spec["feedback_mode"]),
        kp=spec["kp"],
        kd=spec["kd"],
    )
    final_target_error = float(
        np.max(np.abs(execution.actual_positions[-1] - target))
    )
    target_reached = final_target_error <= config.goal_tolerance_rad
    physical_safe = bool(
        execution.obstacle_contact_steps == 0
        and execution.self_contact_steps == 0
        and execution.joint_limit_violation_steps == 0
    )
    safe_task_success = bool(target_reached and physical_safe)
    trace_path = f"traces/{spec['case_id']}.npz"
    row = {
        "schema_version": TASK_SCHEMA,
        "case_id": spec["case_id"],
        "scenario": spec["scenario"],
        "target_is_scenario_goal": spec["target_is_scenario_goal"],
        "payload_mass_kg": payload,
        "delay_ms": spec["delay_ms"],
        "feedback_mode": spec["feedback_mode"],
        "speed_scale": spec["speed_scale"],
        "clearance_m": config.clearance_m,
        "planning_waypoints": len(path),
        "planned_path_length_rad": path_length(path),
        "planning_latency_ms": planning_latency_ms,
        "action_horizon": len(decision.executable_actions),
        "supervisor_status": decision.status,
        "supervisor_reason": decision.reason,
        "supervision_latency_ms": supervision_latency_ms,
        "qp_intervention_steps": decision.intervention_steps,
        "continuous_edges_checked": len(decision.edge_certificates),
        "braking_boundaries_checked": len(decision.braking_certificates),
        "excluded_fixed_gripper_pairs": excluded_pairs,
        "guard_target_error_rad": float(
            np.max(np.abs(decision.predicted_positions[-1] - target))
        ),
        "target_reached": target_reached,
        "physical_safe": physical_safe,
        "safe_task_success": safe_task_success,
        "tracking_rmse_rad": execution.rmse,
        "max_tracking_error_rad": execution.max_tracking_error,
        "final_target_error_rad": final_target_error,
        "torque_saturation_count": execution.torque_saturation_count,
        "joint_limit_violation_steps": execution.joint_limit_violation_steps,
        "obstacle_contact_steps": execution.obstacle_contact_steps,
        "self_contact_steps": execution.self_contact_steps,
        "trace_path": trace_path,
    }
    return row, _trace_document(certified, execution)


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate empty integrated task results")
    latencies = np.asarray([float(row["supervision_latency_ms"]) for row in rows])
    return {
        "cases": len(rows),
        "supervisor_accepted": sum(
            row["supervisor_status"] == "accepted" for row in rows
        ),
        "target_reached": sum(bool(row["target_reached"]) for row in rows),
        "physical_safe": sum(bool(row["physical_safe"]) for row in rows),
        "safe_task_success": sum(bool(row["safe_task_success"]) for row in rows),
        "qp_intervention_steps": sum(
            int(row["qp_intervention_steps"]) for row in rows
        ),
        "continuous_edges_checked": sum(
            int(row["continuous_edges_checked"]) for row in rows
        ),
        "braking_boundaries_checked": sum(
            int(row["braking_boundaries_checked"]) for row in rows
        ),
        "p95_supervision_latency_ms": float(np.percentile(latencies, 95)),
        "maximum_supervision_latency_ms": float(np.max(latencies)),
        "mean_tracking_rmse_rad": float(
            np.mean([float(row["tracking_rmse_rad"]) for row in rows])
        ),
        "maximum_tracking_error_rad": max(
            float(row["max_tracking_error_rad"]) for row in rows
        ),
        "maximum_final_target_error_rad": max(
            float(row["final_target_error_rad"]) for row in rows
        ),
        "obstacle_contact_steps": sum(
            int(row["obstacle_contact_steps"]) for row in rows
        ),
        "self_contact_steps": sum(int(row["self_contact_steps"]) for row in rows),
        "joint_limit_violation_steps": sum(
            int(row["joint_limit_violation_steps"]) for row in rows
        ),
    }


def _summary_markdown(
    summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
    overall = summary["overall"]
    if not isinstance(overall, Mapping):
        raise ValueError("integrated task summary is invalid")
    lines = [
        "# Integrated Panda task execution",
        "",
        f"Safe task successes: {overall['safe_task_success']}/{overall['cases']}",
        "",
        "| Case | Scenario | Payload | Delay | Guard | Target | Physical safety |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['scenario']} | "
            f"{float(row['payload_mass_kg']):.1f} kg | {row['delay_ms']} ms | "
            f"{row['supervisor_status']} | {row['target_reached']} | "
            f"{row['physical_safe']} |"
        )
    lines.extend(
        [
            "",
            "The action source is a scripted RRT-Connect reference. Assurance is ",
            "computed offline before MuJoCo torque execution; this is not a learned ",
            "VLA, hard-real-time, or physical-robot safety result.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_manifest(root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": f"{TASK_SCHEMA}.manifest",
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json(files)),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _implementation_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    return {
        "armbench/vla/integrated_panda_task.py": Path(__file__),
        "armbench/vla/integrated_panda_guard.py": Path(__file__).with_name(
            "integrated_panda_guard.py"
        ),
        "armbench/vla/qp_projection.py": Path(__file__).with_name(
            "qp_projection.py"
        ),
        "armbench/mujoco_sim/continuous_collision.py": package
        / "mujoco_sim"
        / "continuous_collision.py",
        "armbench/mujoco_sim/dynamics_braking.py": package
        / "mujoco_sim"
        / "dynamics_braking.py",
        "armbench/mujoco_sim/execution.py": package / "mujoco_sim" / "execution.py",
        "armbench/planners/rrt_connect.py": package
        / "planners"
        / "rrt_connect.py",
        "armbench/postprocess/shortcut.py": package
        / "postprocess"
        / "shortcut.py",
        "armbench/postprocess/time_parameterization.py": package
        / "postprocess"
        / "time_parameterization.py",
    }


def run_integrated_panda_tasks(
    output_directory: Path,
    config: IntegratedPandaTaskConfig = IntegratedPandaTaskConfig(),
) -> Path:
    """Plan, assure, execute, and preserve registered local Panda tasks."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"integrated task output already exists: {output}")
    output.mkdir(parents=True)
    (output / "traces").mkdir()
    specs = _case_specs(config)
    rows: list[dict[str, object]] = []
    for spec in specs:
        row, trace = _evaluate_case(spec, config)
        rows.append(row)
        np.savez_compressed(output / str(row["trace_path"]), **trace)
    with (output / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "cases.json",
        {"schema_version": TASK_SCHEMA, "cases": specs},
    )
    summary = {
        "schema_version": TASK_SCHEMA,
        "scope": TASK_SCOPE,
        "configuration": asdict(config),
        "environment": {
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "osqp_version": osqp.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "panda_scene_sha256": sha256_file(default_panda_scene_path()),
        "implementation_sha256": {
            label: sha256_file(path) for label, path in _implementation_paths().items()
        },
        "overall": _aggregate(rows),
        "cases": {str(row["case_id"]): dict(row) for row in rows},
        "allowed_collision_matrix": {
            "fixed_open_gripper_body_pair": sorted(
                FIXED_GRIPPER_ALLOWED_BODY_PAIR
            ),
            "scope": "arm_joint_motion_with_fixed_gripper_configuration",
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary, rows), encoding="utf-8"
    )
    _write_manifest(output)
    validate_integrated_panda_tasks(output)
    return output


def _validate_manifest(root: Path, trace_paths: set[str]) -> str:
    manifest = strict_json_load(root / "manifest.json")
    expected = {
        "cases.json",
        "per_case.csv",
        "summary.json",
        "summary.md",
        *trace_paths,
    }
    if not (
        has_exact_fields(manifest, _MANIFEST_FIELDS)
        and manifest["schema_version"] == f"{TASK_SCHEMA}.manifest"
        and isinstance(manifest["files"], list)
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"]
        == sha256_bytes(canonical_json(manifest["files"]))
    ):
        raise ValueError("integrated task manifest is invalid")
    declared = set()
    for item in manifest["files"]:
        if not (
            has_exact_fields(item, _MANIFEST_ENTRY_FIELDS)
            and isinstance(item["path"], str)
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0
            and is_sha256(item["sha256"])
        ):
            raise ValueError("integrated task manifest entry is invalid")
        relative = Path(item["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"integrated task manifest mismatch: {relative}")
        declared.add(item["path"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if declared != expected or actual != expected:
        raise ValueError("integrated task artifact file set is invalid")
    return str(manifest["inventory_sha256"])


def _load_config(value: object) -> IntegratedPandaTaskConfig:
    if not isinstance(value, Mapping):
        raise ValueError("integrated task configuration is invalid")
    try:
        config = IntegratedPandaTaskConfig(
            profiles=tuple(value["profiles"]),
            clearance_m=value["clearance_m"],
            collision_resolution_rad=value["collision_resolution_rad"],
            planner_step_size_rad=value["planner_step_size_rad"],
            planner_max_iterations=value["planner_max_iterations"],
            planner_timeout_s=value["planner_timeout_s"],
            planner_goal_sample_rate=value["planner_goal_sample_rate"],
            shortcut_attempts=value["shortcut_attempts"],
            action_dt_s=value["action_dt_s"],
            settle_action_steps=value["settle_action_steps"],
            response_deadline_ms=value["response_deadline_ms"],
            supervision_budget_ms=value["supervision_budget_ms"],
            qp_step_budget_ms=value["qp_step_budget_ms"],
            execution_control_dt_s=value["execution_control_dt_s"],
            execution_warmup_s=value["execution_warmup_s"],
            execution_hold_s=value["execution_hold_s"],
            goal_tolerance_rad=value["goal_tolerance_rad"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("integrated task configuration is invalid") from error
    if canonical_json(value) != canonical_json(asdict(config)):
        raise ValueError("integrated task configuration is not canonical")
    return config


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{label} is not a canonical boolean")


def _parse_row(raw: Mapping[str, str]) -> dict[str, object]:
    if set(raw) != set(CSV_FIELDS) or raw["schema_version"] != TASK_SCHEMA:
        raise ValueError("integrated task CSV fields are invalid")
    bool_fields = (
        "target_is_scenario_goal",
        "target_reached",
        "physical_safe",
        "safe_task_success",
    )
    int_fields = (
        "delay_ms",
        "planning_waypoints",
        "action_horizon",
        "qp_intervention_steps",
        "continuous_edges_checked",
        "braking_boundaries_checked",
        "excluded_fixed_gripper_pairs",
        "torque_saturation_count",
        "joint_limit_violation_steps",
        "obstacle_contact_steps",
        "self_contact_steps",
    )
    float_fields = (
        "payload_mass_kg",
        "speed_scale",
        "clearance_m",
        "planned_path_length_rad",
        "planning_latency_ms",
        "supervision_latency_ms",
        "guard_target_error_rad",
        "tracking_rmse_rad",
        "max_tracking_error_rad",
        "final_target_error_rad",
    )
    try:
        row: dict[str, object] = dict(raw)
        for field in bool_fields:
            row[field] = _parse_bool(raw[field], field)
        for field in int_fields:
            row[field] = int(raw[field])
        for field in float_fields:
            row[field] = float(raw[field])
    except (TypeError, ValueError) as error:
        raise ValueError("integrated task CSV value is invalid") from error
    numeric = [float(row[field]) for field in (*int_fields, *float_fields)]
    if not np.all(np.isfinite(numeric)) or any(value < 0.0 for value in numeric):
        raise ValueError("integrated task CSV numeric contract is invalid")
    return row


def _normalize_generated_row(row: Mapping[str, object]) -> dict[str, object]:
    return _parse_row({field: str(row[field]) for field in CSV_FIELDS})


def _same_deterministic_row(
    stored: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    ignored = {"planning_latency_ms", "supervision_latency_ms"}
    exact_fields = set(CSV_FIELDS) - ignored - {
        "planned_path_length_rad",
        "guard_target_error_rad",
        "tracking_rmse_rad",
        "max_tracking_error_rad",
        "final_target_error_rad",
    }
    if any(stored[field] != expected[field] for field in exact_fields):
        return False
    return all(
        np.isclose(
            float(stored[field]), float(expected[field]), rtol=1e-10, atol=1e-12
        )
        for field in (
            "planned_path_length_rad",
            "guard_target_error_rad",
            "tracking_rmse_rad",
            "max_tracking_error_rad",
            "final_target_error_rad",
        )
    )


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _TRACE_FIELDS:
                raise ValueError("integrated task trace fields are invalid")
            values = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read integrated task trace: {path}") from error
    if str(values["schema_version"].item()) != TASK_SCHEMA:
        raise ValueError("integrated task trace schema is invalid")
    for key, value in values.items():
        if key == "schema_version":
            continue
        if value.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(value)):
            raise ValueError(f"integrated task trace array is invalid: {key}")
    return values


def _same_trace(
    stored: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> bool:
    if set(stored) != set(expected):
        return False
    for key in stored:
        if key == "schema_version":
            if str(stored[key].item()) != str(expected[key].item()):
                return False
        elif stored[key].shape != expected[key].shape or not np.allclose(
            stored[key], expected[key], rtol=1e-10, atol=1e-12
        ):
            return False
    return True


def _validate_aggregate(actual: object, expected: Mapping[str, object]) -> None:
    if not has_exact_fields(actual, _AGGREGATE_FIELDS):
        raise ValueError("integrated task aggregate fields are invalid")
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("integrated task aggregate mismatch")


def validate_integrated_panda_tasks(directory: Path) -> dict[str, object]:
    """Rerun planning, assurance, physics, traces, and summary aggregation."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"integrated task directory not found: {root}")
    summary = strict_json_load(root / "summary.json")
    if not (
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == TASK_SCHEMA
        and summary["scope"] == TASK_SCOPE
        and summary["panda_scene_sha256"] == sha256_file(default_panda_scene_path())
        and canonical_json(summary["claim_boundary"]) == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("integrated task summary is invalid")
    config = _load_config(summary["configuration"])
    specs = _case_specs(config)
    trace_paths = {f"traces/{spec['case_id']}.npz" for spec in specs}
    inventory_hash = _validate_manifest(root, trace_paths)
    environment = summary["environment"]
    if not (
        has_exact_fields(environment, _ENVIRONMENT_FIELDS)
        and environment["mujoco_version"] == mujoco.__version__
        and environment["numpy_version"] == np.__version__
        and environment["osqp_version"] == osqp.__version__
        and environment["menagerie_commit"] == MENAGERIE_COMMIT
    ):
        raise ValueError("integrated task environment mismatch")
    expected_paths = _implementation_paths()
    hashes = summary["implementation_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(expected_paths):
        raise ValueError("integrated task implementation hashes are invalid")
    if any(hashes[label] != sha256_file(path) for label, path in expected_paths.items()):
        raise ValueError("integrated task implementation hash mismatch")
    expected_allowed = {
        "fixed_open_gripper_body_pair": sorted(FIXED_GRIPPER_ALLOWED_BODY_PAIR),
        "scope": "arm_joint_motion_with_fixed_gripper_configuration",
    }
    if canonical_json(summary["allowed_collision_matrix"]) != canonical_json(
        expected_allowed
    ):
        raise ValueError("integrated task allowed-collision matrix is invalid")

    case_document = strict_json_load(root / "cases.json")
    if not (
        has_exact_fields(case_document, _CASE_DOCUMENT_FIELDS)
        and case_document["schema_version"] == TASK_SCHEMA
        and isinstance(case_document["cases"], list)
        and all(has_exact_fields(item, _CASE_FIELDS) for item in case_document["cases"])
        and canonical_json(case_document["cases"]) == canonical_json(specs)
    ):
        raise ValueError("integrated task registered cases are invalid")
    with (root / "per_case.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("integrated task CSV fields are invalid")
        stored_rows = [_parse_row(row) for row in reader]
    if len(stored_rows) != len(specs):
        raise ValueError("integrated task case count is invalid")

    replayed_rows: list[dict[str, object]] = []
    for spec, stored in zip(specs, stored_rows):
        expected_raw, expected_trace = _evaluate_case(spec, config)
        expected = _normalize_generated_row(expected_raw)
        replayed_rows.append(expected)
        if not _same_deterministic_row(stored, expected):
            raise ValueError(f"integrated task recomputation failed: {stored['case_id']}")
        stored_trace = _load_trace(root / str(stored["trace_path"]))
        if not _same_trace(stored_trace, expected_trace):
            raise ValueError(f"integrated task trace mismatch: {stored['case_id']}")
    if not all(bool(row["safe_task_success"]) for row in stored_rows):
        raise ValueError("integrated task registered acceptance failed")
    overall = _aggregate(stored_rows)
    _validate_aggregate(summary["overall"], overall)
    cases = summary["cases"]
    expected_case_ids = {str(row["case_id"]) for row in stored_rows}
    if not isinstance(cases, Mapping) or set(cases) != expected_case_ids:
        raise ValueError("integrated task summary case mapping is invalid")
    for row in stored_rows:
        if canonical_json(cases[str(row["case_id"])]) != canonical_json(row):
            raise ValueError("integrated task summary case row mismatch")
    if (root / "summary.md").read_text("utf-8") != _summary_markdown(
        summary, stored_rows
    ):
        raise ValueError("integrated task Markdown summary is not reproducible")
    return {
        "valid": True,
        "scope": TASK_SCOPE,
        "cases": len(stored_rows),
        "safe_task_success": overall["safe_task_success"],
        "target_reached": overall["target_reached"],
        "physical_safe": overall["physical_safe"],
        "continuous_edges_checked": overall["continuous_edges_checked"],
        "braking_boundaries_checked": overall["braking_boundaries_checked"],
        "manifest_inventory_sha256": inventory_hash,
        "checks": [
            "recursive_manifest",
            "registered_planning_and_supervision_rerun",
            "closed_loop_mujoco_physics_rerun",
            "trajectory_arrays_recomputed",
            "task_and_contact_metrics_recomputed",
        ],
    }


__all__ = [
    "IntegratedPandaTaskConfig",
    "make_integrated_task_checker",
    "run_integrated_panda_tasks",
    "validate_integrated_panda_tasks",
]
