"""Auditable fault matrix for the asynchronous MuJoCo Panda runtime."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from armbench.mujoco_sim.model import (
    MENAGERIE_COMMIT,
    MuJoCoPanda,
    default_panda_scene_path,
)
from armbench.mujoco_sim.scenarios import MUJOCO_SCENARIO_VERSION, mujoco_scenarios
from armbench.vla.async_panda import (
    ASYNC_PANDA_MODES,
    AsyncPandaConfig,
    AsyncPandaEpisodeResult,
    ScriptedPolicyFaults,
    run_async_panda_episode,
)
from armbench.vla.benchmark import (
    _guard_config as configured_guard,
    _integrate_actions,
    _safe_stream,
    load_vla_config,
)
from armbench.vla.pi05_archive_replay import (
    _load_json,
    _sha256_file,
    _validate_root_manifest,
    _write_json,
    _write_root_manifest,
)


ARTIFACT_SCHEMA = "armbench.async_panda_closed_loop.v1"
SUMMARY_SCHEMA = "armbench.async_panda_closed_loop_summary.v1"
PROVENANCE_SCHEMA = "armbench.async_panda_closed_loop_provenance.v1"
TRACE_SCHEMA = "armbench.async_panda_trace.v1"
SCOPE = "scripted_async_vla_runtime_mujoco_closed_loop"


@dataclass(frozen=True)
class AsyncPandaCondition:
    name: str
    latency_schedule_ms: tuple[float, ...]
    latency_jitter_ms: float = 0.0
    drop_probability: float = 0.0
    payload_mass_kg: float = 0.0
    spike_joint: int = 0
    spike_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip() or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in self.name
        ):
            raise ValueError("condition name must be lowercase snake case")
        if not np.isfinite(self.payload_mass_kg) or self.payload_mass_kg < 0.0:
            raise ValueError("condition payload must be finite and nonnegative")
        ScriptedPolicyFaults(
            latency_schedule_ms=self.latency_schedule_ms,
            latency_jitter_ms=self.latency_jitter_ms,
            drop_probability=self.drop_probability,
            spike_joint=self.spike_joint,
            spike_velocity_rad_s=self.spike_velocity_rad_s,
        )

    def policy_faults(self, seed: int) -> ScriptedPolicyFaults:
        return ScriptedPolicyFaults(
            latency_schedule_ms=self.latency_schedule_ms,
            latency_jitter_ms=self.latency_jitter_ms,
            drop_probability=self.drop_probability,
            seed=seed,
            spike_joint=self.spike_joint,
            spike_velocity_rad_s=self.spike_velocity_rad_s,
        )


def default_async_panda_conditions() -> tuple[AsyncPandaCondition, ...]:
    fixed = tuple(
        AsyncPandaCondition(
            name=f"fixed_{latency_ms:03d}ms",
            latency_schedule_ms=(float(latency_ms),),
        )
        for latency_ms in (0, 40, 80, 160, 240)
    )
    return fixed + (
        AsyncPandaCondition(
            name="jitter_080ms_025ms",
            latency_schedule_ms=(80.0,),
            latency_jitter_ms=25.0,
        ),
        AsyncPandaCondition(
            name="drop_080ms_010pct",
            latency_schedule_ms=(80.0,),
            drop_probability=0.1,
        ),
        AsyncPandaCondition(
            name="payload_080ms_0500g",
            latency_schedule_ms=(80.0,),
            payload_mass_kg=0.5,
        ),
        AsyncPandaCondition(
            name="action_fault_040ms",
            latency_schedule_ms=(40.0,),
            spike_joint=0,
            spike_velocity_rad_s=2.5,
        ),
    )


CSV_FIELDS = (
    "artifact_schema",
    "case_id",
    "condition",
    "mode",
    "scenario",
    "trace",
    "video_path",
    "payload_mass_kg",
    "latency_schedule_ms",
    "latency_jitter_ms",
    "drop_probability",
    "spike_joint",
    "spike_velocity_rad_s",
    "reference_action_steps",
    "runtime_action_steps",
    "target_is_scenario_goal",
    "target_reached",
    "physical_safe",
    "safe_target_reached",
    "final_target_error_rad",
    "tracking_rmse_rad",
    "max_tracking_error_rad",
    "control_thread_id",
    "policy_worker_thread_id",
    "observation_worker_thread_id",
    "control_ticks",
    "control_ticks_during_inference",
    "p95_control_tick_lateness_ms",
    "max_control_tick_lateness_ms",
    "p95_control_tick_gap_ms",
    "max_control_tick_gap_ms",
    "accepted_responses",
    "rejected_responses",
    "deadline_rejections",
    "policy_failures",
    "observation_frames_completed",
    "observation_frames_superseded",
    "superseded_pending_requests",
    "action_boundaries",
    "hold_boundaries",
    "hold_rate",
    "stale_action_commands",
    "index_zero_action_commands",
    "scaled_plan_count",
    "planned_intervention_steps",
    "braking_boundaries",
    "abrupt_stop_violations",
    "unsafe_prepared_plans",
    "p95_policy_latency_ms",
    "max_policy_latency_ms",
    "p95_observation_latency_ms",
    "max_observation_latency_ms",
    "p95_repair_latency_ms",
    "max_repair_latency_ms",
    "max_observation_age_ms",
    "torque_saturation_count",
    "obstacle_contact_steps",
    "self_contact_steps",
    "joint_limit_violation_steps",
)

INTEGER_FIELDS = frozenset(
    {
        "spike_joint",
        "reference_action_steps",
        "runtime_action_steps",
        "control_thread_id",
        "policy_worker_thread_id",
        "observation_worker_thread_id",
        "control_ticks",
        "control_ticks_during_inference",
        "accepted_responses",
        "rejected_responses",
        "deadline_rejections",
        "policy_failures",
        "observation_frames_completed",
        "observation_frames_superseded",
        "superseded_pending_requests",
        "action_boundaries",
        "hold_boundaries",
        "stale_action_commands",
        "index_zero_action_commands",
        "scaled_plan_count",
        "planned_intervention_steps",
        "braking_boundaries",
        "abrupt_stop_violations",
        "unsafe_prepared_plans",
        "torque_saturation_count",
        "obstacle_contact_steps",
        "self_contact_steps",
        "joint_limit_violation_steps",
    }
)
BOOLEAN_FIELDS = frozenset(
    {
        "target_is_scenario_goal",
        "target_reached",
        "physical_safe",
        "safe_target_reached",
    }
)
FLOAT_FIELDS = frozenset(CSV_FIELDS).difference(
    INTEGER_FIELDS
    | BOOLEAN_FIELDS
    | {
        "artifact_schema",
        "case_id",
        "condition",
        "mode",
        "scenario",
        "trace",
        "video_path",
        "latency_schedule_ms",
    }
)

TRACE_KEYS = frozenset(
    {
        "schema_version",
        "case_id",
        "condition",
        "mode",
        "scenario",
        "reference_positions",
        "policy_latencies_ms",
        "observation_latencies_ms",
        "repair_latencies_ms",
        "scheduled_wall_times_s",
        "actual_wall_times_s",
        "simulated_times_s",
        "desired_positions",
        "actual_positions",
        "command_velocities",
        "command_statuses",
        "request_ids",
        "action_indices",
        "observation_ages_ms",
        "policy_inflight",
        "obstacle_contacts",
        "self_contacts",
        "joint_limit_violations",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty asynchronous Panda matrix")
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
            if field in INTEGER_FIELDS:
                try:
                    row[field] = int(value)
                except ValueError as error:
                    raise ValueError(f"invalid CSV integer: {field}") from error
            elif field in FLOAT_FIELDS:
                try:
                    row[field] = float(value)
                except ValueError as error:
                    raise ValueError(f"invalid CSV float: {field}") from error
                _require(np.isfinite(row[field]), f"nonfinite CSV float: {field}")
            elif field in BOOLEAN_FIELDS:
                _require(value in {"True", "False"}, f"invalid CSV boolean: {field}")
                row[field] = value == "True"
            elif field == "latency_schedule_ms":
                try:
                    schedule = json.loads(value)
                except json.JSONDecodeError as error:
                    raise ValueError("invalid latency schedule JSON") from error
                _require(
                    isinstance(schedule, list)
                    and bool(schedule)
                    and all(
                        type(item) in {int, float} and np.isfinite(item)
                        for item in schedule
                    ),
                    "invalid latency schedule values",
                )
                row[field] = [float(item) for item in schedule]
            else:
                row[field] = value
        rows.append(row)
    return rows


def _write_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        try:
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON token: {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid event JSON at line {line_number}") from error
        _require(isinstance(value, dict), "event JSONL rows must be objects")
        events.append(value)
    return events


def _write_trace(
    path: Path,
    case_id: str,
    condition: str,
    result: AsyncPandaEpisodeResult,
    reference_positions: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(TRACE_SCHEMA),
        case_id=np.asarray(case_id),
        condition=np.asarray(condition),
        mode=np.asarray(result.mode),
        scenario=np.asarray(result.scenario),
        reference_positions=reference_positions,
        policy_latencies_ms=result.policy_latencies_ms,
        observation_latencies_ms=result.observation_latencies_ms,
        repair_latencies_ms=result.repair_latencies_ms,
        scheduled_wall_times_s=result.scheduled_wall_times_s,
        actual_wall_times_s=result.actual_wall_times_s,
        simulated_times_s=result.simulated_times_s,
        desired_positions=result.desired_positions,
        actual_positions=result.actual_positions,
        command_velocities=result.command_velocities,
        command_statuses=result.command_statuses,
        request_ids=result.request_ids,
        action_indices=result.action_indices,
        observation_ages_ms=result.observation_ages_ms,
        policy_inflight=result.policy_inflight,
        obstacle_contacts=result.obstacle_contacts,
        self_contacts=result.self_contacts,
        joint_limit_violations=result.joint_limit_violations,
    )


def _case_row(
    case_id: str,
    condition: AsyncPandaCondition,
    result: AsyncPandaEpisodeResult,
    *,
    reference_action_steps: int,
    runtime_action_steps: int,
    trace_path: str,
    video_path: str,
) -> dict[str, Any]:
    metrics = result.metrics()
    command_events = [
        event
        for event in result.events
        if event["event"] == "command_switch"
        and event["evaluation_boundary"]
        and event["status"] == "execute"
    ]
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "case_id": case_id,
        "condition": condition.name,
        "mode": result.mode,
        "scenario": result.scenario,
        "trace": trace_path,
        "video_path": video_path,
        "payload_mass_kg": condition.payload_mass_kg,
        "latency_schedule_ms": json.dumps(list(condition.latency_schedule_ms)),
        "latency_jitter_ms": condition.latency_jitter_ms,
        "drop_probability": condition.drop_probability,
        "spike_joint": condition.spike_joint,
        "spike_velocity_rad_s": condition.spike_velocity_rad_s,
        "reference_action_steps": reference_action_steps,
        "runtime_action_steps": runtime_action_steps,
        "target_is_scenario_goal": metrics["target_is_scenario_goal"],
        "target_reached": metrics["target_reached"],
        "physical_safe": metrics["physical_safe"],
        "safe_target_reached": metrics["safe_target_reached"],
        "final_target_error_rad": metrics["final_target_error_rad"],
        "tracking_rmse_rad": metrics["tracking_rmse_rad"],
        "max_tracking_error_rad": metrics["max_tracking_error_rad"],
        "control_thread_id": metrics["control_thread_id"],
        "policy_worker_thread_id": metrics["policy_worker_thread_id"] or -1,
        "observation_worker_thread_id": (
            metrics["observation_worker_thread_id"] or -1
        ),
        "control_ticks": metrics["control_ticks"],
        "control_ticks_during_inference": metrics[
            "control_ticks_during_inference"
        ],
        "p95_control_tick_lateness_ms": metrics[
            "p95_control_tick_lateness_ms"
        ],
        "max_control_tick_lateness_ms": metrics[
            "max_control_tick_lateness_ms"
        ],
        "p95_control_tick_gap_ms": metrics["p95_control_tick_gap_ms"],
        "max_control_tick_gap_ms": metrics["max_control_tick_gap_ms"],
        "accepted_responses": metrics["accepted_responses"],
        "rejected_responses": metrics["rejected_responses"],
        "deadline_rejections": metrics["deadline_rejections"],
        "policy_failures": metrics["policy_failures"],
        "observation_frames_completed": metrics[
            "observation_frames_completed"
        ],
        "observation_frames_superseded": metrics[
            "observation_frames_superseded"
        ],
        "superseded_pending_requests": metrics[
            "superseded_pending_requests"
        ],
        "action_boundaries": metrics["action_boundaries"],
        "hold_boundaries": metrics["hold_boundaries"],
        "hold_rate": metrics["hold_rate"],
        "stale_action_commands": sum(
            int(event["action_index"]) > 0 for event in command_events
        ),
        "index_zero_action_commands": sum(
            int(event["action_index"]) == 0 for event in command_events
        ),
        "scaled_plan_count": metrics["scaled_plan_count"],
        "planned_intervention_steps": metrics["planned_intervention_steps"],
        "braking_boundaries": metrics["braking_boundaries"],
        "abrupt_stop_violations": metrics["abrupt_stop_violations"],
        "unsafe_prepared_plans": metrics["unsafe_prepared_plans"],
        "p95_policy_latency_ms": metrics["p95_policy_latency_ms"],
        "max_policy_latency_ms": metrics["max_policy_latency_ms"],
        "p95_observation_latency_ms": metrics[
            "p95_observation_latency_ms"
        ],
        "max_observation_latency_ms": metrics[
            "max_observation_latency_ms"
        ],
        "p95_repair_latency_ms": metrics["p95_repair_latency_ms"],
        "max_repair_latency_ms": metrics["max_repair_latency_ms"],
        "max_observation_age_ms": metrics["max_observation_age_ms"],
        "torque_saturation_count": metrics["torque_saturation_count"],
        "obstacle_contact_steps": metrics["obstacle_contact_steps"],
        "self_contact_steps": metrics["self_contact_steps"],
        "joint_limit_violation_steps": metrics[
            "joint_limit_violation_steps"
        ],
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot aggregate an empty case set")
    return {
        "cases": len(rows),
        "target_reached_cases": sum(bool(row["target_reached"]) for row in rows),
        "physical_safe_cases": sum(bool(row["physical_safe"]) for row in rows),
        "safe_target_reached_cases": sum(
            bool(row["safe_target_reached"]) for row in rows
        ),
        "cases_with_accepted_response": sum(
            int(row["accepted_responses"]) > 0 for row in rows
        ),
        "deadline_rejections": sum(int(row["deadline_rejections"]) for row in rows),
        "policy_failures": sum(int(row["policy_failures"]) for row in rows),
        "stale_action_commands": sum(
            int(row["stale_action_commands"]) for row in rows
        ),
        "index_zero_action_commands": sum(
            int(row["index_zero_action_commands"]) for row in rows
        ),
        "abrupt_stop_violations": sum(
            int(row["abrupt_stop_violations"]) for row in rows
        ),
        "obstacle_contact_steps": sum(
            int(row["obstacle_contact_steps"]) for row in rows
        ),
        "mean_hold_rate": fmean(float(row["hold_rate"]) for row in rows),
        "mean_final_target_error_rad": fmean(
            float(row["final_target_error_rad"]) for row in rows
        ),
        "p95_control_tick_lateness_ms": float(
            np.percentile(
                [float(row["p95_control_tick_lateness_ms"]) for row in rows],
                95,
            )
        ),
        "max_control_tick_lateness_ms": max(
            float(row["max_control_tick_lateness_ms"]) for row in rows
        ),
        "p95_repair_latency_ms": float(
            np.percentile(
                [float(row["p95_repair_latency_ms"]) for row in rows], 95
            )
        ),
        "max_repair_latency_ms": max(
            float(row["max_repair_latency_ms"]) for row in rows
        ),
    }


def _build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(row["mode"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    return {
        "schema_version": SUMMARY_SCHEMA,
        "scope": SCOPE,
        "policy_checkpoint_executed": False,
        "scripted_policy": True,
        "panda_closed_loop_executed": True,
        "hard_realtime_claim": False,
        "physical_safety_claim": False,
        "overall": _aggregate(rows),
        "by_mode": {
            mode: _aggregate([row for row in rows if row["mode"] == mode])
            for mode in modes
        },
        "by_condition": {
            condition: _aggregate(
                [row for row in rows if row["condition"] == condition]
            )
            for condition in conditions
        },
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Asynchronous Panda closed-loop runtime",
        "",
        "A blocking scripted policy runs on a worker thread while a best-effort",
        "periodic control loop continues torque-controlled MuJoCo execution.",
        "Camera/state acquisition uses a latest-only worker; response age includes",
        "sensor acquisition and policy inference.",
        "",
        "| Mode | Cases | Target reached | Physically safe | Hold rate | Abrupt stops | Contacts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in summary["by_mode"].items():
        lines.append(
            f"| {mode} | {values['cases']} | {values['target_reached_cases']} | "
            f"{values['physical_safe_cases']} | {values['mean_hold_rate']:.3f} | "
            f"{values['abrupt_stop_violations']} | "
            f"{values['obstacle_contact_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The policy is scripted and non-learned; no pi0/pi0.5 checkpoint was",
            "executed. Deadlines are measured software budgets, not an OS hard-real-",
            "time guarantee. MuJoCo contact outcomes are not physical-robot safety",
            "certification.",
            "",
        ]
    )
    return "\n".join(lines)


def _runtime_config(
    config: Mapping[str, Any],
    *,
    max_action_steps: int,
    runtime_clearance_m: float,
    response_deadline_ms: float | None,
) -> AsyncPandaConfig:
    guard = dict(config["guard"])
    online = dict(config["online"])
    contract = dict(config["openpi_contract"])
    return AsyncPandaConfig(
        action_period_s=1.0 / float(contract["control_hz"]),
        control_period_s=float(online["controller_dt_s"]),
        response_deadline_s=(
            float(guard["deadline_ms"])
            if response_deadline_ms is None
            else float(response_deadline_ms)
        ) / 1000.0,
        warmup_s=float(online["warmup_s"]),
        settle_s=float(online["hold_s"]),
        max_action_steps=max_action_steps,
        action_horizon=int(contract["action_horizon"]),
        goal_tolerance_rad=float(online["goal_tolerance_rad"]),
        clearance_m=runtime_clearance_m,
        collision_resolution_rad=float(guard["collision_resolution_rad"]),
        max_state_mismatch_rad=float(guard["max_state_mismatch_rad"]),
        joint_velocity_limit_rad_s=float(guard["joint_velocity_clip_rad_s"]),
        joint_acceleration_limit_rad_s2=float(
            guard["joint_acceleration_clip_rad_s2"]
        ),
        kp=tuple(float(value) for value in online["kp"]),
        kd=tuple(float(value) for value in online["kd"]),
    )


def execute_async_panda_benchmark(
    config_path: Path,
    output_directory: Path,
    *,
    scenario_name: str = "single_block",
    modes: Sequence[str] = ASYNC_PANDA_MODES,
    conditions: Sequence[AsyncPandaCondition] | None = None,
    max_reference_steps: int | None = None,
    extra_action_steps: int = 15,
    seed: int = 20260807,
    make_videos: bool = False,
    runtime_clearance_m: float | None = None,
    response_deadline_ms: float | None = None,
) -> Path:
    """Execute a paired mode-by-fault matrix and write immutable evidence."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    if scenario_name not in mujoco_scenarios():
        raise ValueError(f"unknown MuJoCo scenario: {scenario_name}")
    selected_modes = tuple(str(mode) for mode in modes)
    if (
        not selected_modes
        or len(set(selected_modes)) != len(selected_modes)
        or any(mode not in ASYNC_PANDA_MODES for mode in selected_modes)
    ):
        raise ValueError("modes must be unique asynchronous Panda modes")
    selected_conditions = tuple(conditions or default_async_panda_conditions())
    if (
        not selected_conditions
        or len({condition.name for condition in selected_conditions})
        != len(selected_conditions)
    ):
        raise ValueError("conditions must be nonempty and uniquely named")
    if (
        max_reference_steps is not None and max_reference_steps <= 0
    ) or extra_action_steps < 0 or (
        runtime_clearance_m is not None
        and (
            not np.isfinite(runtime_clearance_m)
            or runtime_clearance_m < 0.0
        )
    ) or (
        response_deadline_ms is not None
        and (
            not np.isfinite(response_deadline_ms)
            or response_deadline_ms <= 0.0
        )
    ):
        raise ValueError("runtime timing/reference limits are invalid")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    raw_config = load_vla_config(config_path)
    planning_clearance_m = float(dict(raw_config["guard"])["clearance_m"])
    resolved_runtime_clearance_m = (
        planning_clearance_m
        if runtime_clearance_m is None
        else float(runtime_clearance_m)
    )
    guard_config = configured_guard(raw_config)
    actions = _safe_stream(scenario_name, raw_config, guard_config)
    if max_reference_steps is not None:
        actions = actions[:max_reference_steps]
    scenario = mujoco_scenarios()[scenario_name]
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    reference = _integrate_actions(
        reference_robot,
        scenario.start,
        actions,
        guard_config,
    )
    if max_reference_steps is None:
        reference[-1] = scenario.goal
    runtime_steps = len(actions) + extra_action_steps
    runtime_config = _runtime_config(
        raw_config,
        max_action_steps=runtime_steps,
        runtime_clearance_m=resolved_runtime_clearance_m,
        response_deadline_ms=response_deadline_ms,
    )

    output.mkdir(parents=True)
    trace_directory = output / "traces"
    trace_directory.mkdir()
    if make_videos:
        (output / "videos").mkdir()
    rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    case_index = 0
    for condition in selected_conditions:
        for mode in selected_modes:
            case_id = f"case_{case_index:03d}__{condition.name}__{mode}"
            trace_relative = f"traces/{case_id}.npz"
            video_relative = f"videos/{case_id}.mp4" if make_videos else ""
            result = run_async_panda_episode(
                scenario_name,
                mode,
                reference,
                policy_faults=condition.policy_faults(seed),
                config=runtime_config,
                payload_mass=condition.payload_mass_kg,
                prompt=str(
                    dict(raw_config["prompts"]).get(
                        scenario_name,
                        f"move the gripper to the {scenario_name} goal",
                    )
                ),
                video_path=(output / video_relative if video_relative else None),
            )
            _write_trace(
                output / trace_relative,
                case_id,
                condition.name,
                result,
                reference,
            )
            rows.append(
                _case_row(
                    case_id,
                    condition,
                    result,
                    reference_action_steps=len(actions),
                    runtime_action_steps=runtime_steps,
                    trace_path=trace_relative,
                    video_path=video_relative,
                )
            )
            all_events.extend(
                {"case_id": case_id, "condition": condition.name, **event}
                for event in result.events
            )
            case_index += 1

    summary = _build_summary(rows)
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("async_panda.py"),
        Path(__file__).with_name("async_dispatch.py"),
        Path(__file__).with_name("async_worker.py"),
        Path(__file__).with_name("trajectory_repair.py"),
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "scope": SCOPE,
        "policy": {
            "source": "scripted_non_learned_async_reference",
            "checkpoint_executed": False,
            "pi0_or_pi05_executed": False,
        },
        "runtime": {
            "control_scheduler": "best_effort_wall_clock_periodic",
            "policy_execution": "blocking_call_on_latest_only_worker_thread",
            "observation_execution": "latest_only_background_renderer",
            "response_alignment": "observation_age_action_suffix",
            "deadline_fallback": "hold_or_verified_terminal_braking",
            "hard_realtime_claim": False,
            "physical_safety_claim": False,
        },
        "matrix": {
            "scenario": scenario_name,
            "modes": list(selected_modes),
            "conditions": [asdict(condition) for condition in selected_conditions],
            "seed": seed,
            "reference_action_steps": len(actions),
            "runtime_action_steps": runtime_steps,
            "target_is_scenario_goal": bool(
                np.allclose(reference[-1], scenario.goal, atol=1e-12)
            ),
            "planning_clearance_m": planning_clearance_m,
            "runtime_clearance_m": resolved_runtime_clearance_m,
            "runtime_clearance_source": (
                "planning_clearance"
                if runtime_clearance_m is None
                else "explicit_override"
            ),
        },
        "configuration": asdict(runtime_config),
        "local_runtime": {
            "python": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_scenario_version": MUJOCO_SCENARIO_VERSION,
            "menagerie_commit": MENAGERIE_COMMIT,
            "panda_scene_sha256": _sha256_file(default_panda_scene_path()),
            "implementation_sha256": {
                f"armbench/vla/{path.name}": _sha256_file(path)
                for path in implementation_paths
            },
        },
        "limitations": [
            "The policy is scripted and non-learned.",
            "No pi0 or pi0.5 checkpoint is executed by this benchmark.",
            "Wall-clock deadlines are not an OS hard-real-time guarantee.",
            "MuJoCo contacts are not physical-robot safety certification.",
            "Camera rendering and Python scheduling are machine dependent.",
            "Runtime collision checks use resolution-bounded joint-space edges.",
        ],
    }
    _write_csv(output / "per_case.csv", rows)
    _write_events(output / "events.jsonl", all_events)
    _write_json(output / "summary.json", summary)
    _write_json(output / "provenance.json", provenance)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_root_manifest(output)
    validate_async_panda_artifact(output)
    return output


def _trace_values(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            _require(set(loaded.files) == TRACE_KEYS, "trace field set mismatch")
            return {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"cannot read trace archive: {path}") from error


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if len(values) else 0.0


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _assert_close(actual: float, expected: float, label: str) -> None:
    _require(
        bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-10)),
        f"trace-derived metric mismatch: {label}",
    )


def _validate_case_trace(
    root: Path,
    row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    goal_tolerance_rad: float,
    acceleration_limit_rad_s2: float,
) -> None:
    trace = _trace_values(root / str(row["trace"]))
    for key in ("schema_version", "case_id", "condition", "mode", "scenario"):
        expected = TRACE_SCHEMA if key == "schema_version" else str(row[key])
        _require(str(trace[key].item()) == expected, f"trace identity mismatch: {key}")
    ticks = len(trace["actual_wall_times_s"])
    _require(ticks > 0 and int(row["control_ticks"]) == ticks, "control tick mismatch")
    for key in (
        "scheduled_wall_times_s",
        "simulated_times_s",
        "desired_positions",
        "actual_positions",
        "command_velocities",
        "command_statuses",
        "request_ids",
        "action_indices",
        "observation_ages_ms",
        "policy_inflight",
        "obstacle_contacts",
        "self_contacts",
        "joint_limit_violations",
    ):
        _require(len(trace[key]) == ticks, f"trace tick length mismatch: {key}")
    for key in ("desired_positions", "actual_positions", "command_velocities"):
        _require(
            trace[key].shape == (ticks, 7)
            and bool(np.all(np.isfinite(trace[key]))),
            f"trace vector shape/value mismatch: {key}",
        )
    reference = trace["reference_positions"]
    _require(
        reference.ndim == 2
        and reference.shape[1] == 7
        and len(reference) == int(row["reference_action_steps"]) + 1,
        "reference trace shape mismatch",
    )
    actual = trace["actual_positions"]
    desired = trace["desired_positions"]
    error = actual - desired
    lateness = (
        trace["actual_wall_times_s"] - trace["scheduled_wall_times_s"]
    ) * 1000.0
    gaps = np.diff(trace["actual_wall_times_s"]) * 1000.0
    finite_ages = trace["observation_ages_ms"][
        np.isfinite(trace["observation_ages_ms"])
    ]
    derived = {
        "final_target_error_rad": float(
            np.max(np.abs(actual[-1] - reference[-1]))
        ),
        "tracking_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "max_tracking_error_rad": float(np.max(np.abs(error))),
        "p95_control_tick_lateness_ms": _p95(lateness),
        "max_control_tick_lateness_ms": _max(lateness),
        "p95_control_tick_gap_ms": _p95(gaps),
        "max_control_tick_gap_ms": _max(gaps),
        "p95_policy_latency_ms": _p95(trace["policy_latencies_ms"]),
        "max_policy_latency_ms": _max(trace["policy_latencies_ms"]),
        "p95_observation_latency_ms": _p95(trace["observation_latencies_ms"]),
        "max_observation_latency_ms": _max(trace["observation_latencies_ms"]),
        "p95_repair_latency_ms": _p95(trace["repair_latencies_ms"]),
        "max_repair_latency_ms": _max(trace["repair_latencies_ms"]),
        "max_observation_age_ms": _max(finite_ages),
    }
    for field, value in derived.items():
        _assert_close(float(row[field]), value, field)
    _require(
        bool(row["target_reached"])
        == (derived["final_target_error_rad"] <= goal_tolerance_rad),
        "target-reached flag is not trace-derived",
    )
    integer_derived = {
        "control_ticks_during_inference": int(np.sum(trace["policy_inflight"])),
        "observation_frames_completed": len(trace["observation_latencies_ms"]),
        "obstacle_contact_steps": int(np.sum(trace["obstacle_contacts"])),
        "self_contact_steps": int(np.sum(trace["self_contacts"])),
        "joint_limit_violation_steps": int(
            np.sum(trace["joint_limit_violations"])
        ),
    }
    for field, value in integer_derived.items():
        _require(int(row[field]) == value, f"trace count mismatch: {field}")
    expected_physical_safe = bool(
        integer_derived["obstacle_contact_steps"] == 0
        and integer_derived["self_contact_steps"] == 0
        and integer_derived["joint_limit_violation_steps"] == 0
    )
    _require(
        bool(row["physical_safe"]) == expected_physical_safe,
        "physical-safe flag mismatch",
    )
    _require(
        bool(row["safe_target_reached"])
        == (bool(row["target_reached"]) and expected_physical_safe),
        "safe-target-reached flag mismatch",
    )

    command_events = [
        event
        for event in events
        if event.get("event") == "command_boundary"
    ]
    command_switches = [
        event
        for event in events
        if event.get("event") == "command_switch"
    ]
    evaluation = [event for event in command_events if event["evaluation_boundary"]]
    executions = [
        event
        for event in command_switches
        if event["evaluation_boundary"] and event["status"] == "execute"
    ]
    outcomes = [event for event in events if event.get("event") == "policy_outcome"]
    plans = [event for event in events if event.get("event") == "plan_prepared"]
    sensor_submissions = [
        event for event in events if event.get("event") == "observation_submission"
    ]
    policy_submissions = [
        event for event in events if event.get("event") == "policy_submission"
    ]
    event_counts = {
        "action_boundaries": len(evaluation),
        "hold_boundaries": sum(event["status"] != "execute" for event in evaluation),
        "braking_boundaries": sum(
            event["status"] == "brake" for event in command_switches
        ),
        "abrupt_stop_violations": sum(
            float(event["max_acceleration_rad_s2"])
            > acceleration_limit_rad_s2 + 1e-9
            for event in command_switches
        ),
        "accepted_responses": sum(
            event["dispatch_status"] == "accepted" for event in outcomes
        ),
        "rejected_responses": sum(
            event["dispatch_status"] == "rejected" for event in outcomes
        ),
        "deadline_rejections": sum(
            event["dispatch_reason"] == "deadline_exceeded" for event in outcomes
        ),
        "policy_failures": sum(not event["succeeded"] for event in outcomes),
        "observation_frames_superseded": sum(
            event["replaced_request_id"] is not None for event in sensor_submissions
        ),
        "superseded_pending_requests": sum(
            event["replaced_request_id"] is not None for event in policy_submissions
        ),
        "stale_action_commands": sum(int(event["action_index"]) > 0 for event in executions),
        "index_zero_action_commands": sum(int(event["action_index"]) == 0 for event in executions),
        "scaled_plan_count": sum(float(event["selected_scale"]) < 1.0 for event in plans),
        "planned_intervention_steps": sum(
            int(event["intervention_steps"]) for event in plans
        ),
        "unsafe_prepared_plans": sum(not event["safe_after_guard"] for event in plans)
        + sum(
            not event["safe"]
            for event in events
            if event.get("event") == "terminal_brake_prepared"
        ),
    }
    for field, value in event_counts.items():
        _require(int(row[field]) == value, f"event count mismatch: {field}")
    _assert_close(
        float(row["hold_rate"]),
        event_counts["hold_boundaries"] / max(1, len(evaluation)),
        "hold_rate",
    )


def validate_async_panda_artifact(directory: Path) -> dict[str, Any]:
    root = directory.resolve()
    _require(root.is_dir(), f"artifact directory not found: {root}")
    manifest = _validate_root_manifest(root)
    rows = _read_csv(root / "per_case.csv")
    events = _read_events(root / "events.jsonl")
    summary = _load_json(root / "summary.json")
    provenance = _load_json(root / "provenance.json")
    _require(bool(rows), "artifact contains no cases")
    _require(
        len({str(row["case_id"]) for row in rows}) == len(rows),
        "case ids are not unique",
    )
    _require(
        all(row["artifact_schema"] == ARTIFACT_SCHEMA for row in rows),
        "case artifact schema mismatch",
    )
    _require(
        isinstance(summary, Mapping)
        and summary.get("schema_version") == SUMMARY_SCHEMA
        and summary.get("scope") == SCOPE,
        "summary schema/scope mismatch",
    )
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("schema_version") == PROVENANCE_SCHEMA
        and provenance.get("scope") == SCOPE,
        "provenance schema/scope mismatch",
    )
    for document in (summary, provenance["runtime"]):
        _require(document.get("hard_realtime_claim") is False, "invalid hard-real-time claim")
        _require(document.get("physical_safety_claim") is False, "invalid physical-safety claim")
    _require(
        summary.get("policy_checkpoint_executed") is False
        and summary.get("scripted_policy") is True
        and provenance["policy"]["pi0_or_pi05_executed"] is False,
        "policy claim boundary mismatch",
    )
    matrix = provenance.get("matrix")
    configuration = provenance.get("configuration")
    _require(
        isinstance(matrix, Mapping) and isinstance(configuration, Mapping),
        "provenance matrix/configuration missing",
    )
    modes = matrix.get("modes")
    conditions = matrix.get("conditions")
    _require(
        isinstance(modes, list)
        and isinstance(conditions, list)
        and all(isinstance(condition, Mapping) for condition in conditions),
        "provenance matrix is invalid",
    )
    expected_pairs = [
        (str(condition["name"]), str(mode))
        for condition in conditions
        for mode in modes
    ]
    _require(
        [(str(row["condition"]), str(row["mode"])) for row in rows]
        == expected_pairs,
        "CSV does not match the declared paired matrix",
    )
    condition_by_name = {
        str(condition["name"]): condition for condition in conditions
    }
    for row in rows:
        condition = condition_by_name[str(row["condition"])]
        _require(
            row["latency_schedule_ms"]
            == [float(value) for value in condition["latency_schedule_ms"]]
            and float(row["latency_jitter_ms"])
            == float(condition["latency_jitter_ms"])
            and float(row["drop_probability"])
            == float(condition["drop_probability"])
            and float(row["payload_mass_kg"])
            == float(condition["payload_mass_kg"])
            and int(row["spike_joint"]) == int(condition["spike_joint"])
            and float(row["spike_velocity_rad_s"])
            == float(condition["spike_velocity_rad_s"]),
            "CSV condition does not match provenance",
        )
        _require(
            int(row["control_thread_id"]) > 0
            and int(row["policy_worker_thread_id"]) > 0
            and int(row["observation_worker_thread_id"]) > 0
            and int(row["control_thread_id"])
            != int(row["policy_worker_thread_id"])
            and int(row["control_thread_id"])
            != int(row["observation_worker_thread_id"]),
            "worker/control execution identities are invalid",
        )
    events_by_case: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        case_id = event.get("case_id")
        _require(isinstance(case_id, str), "event case id missing")
        events_by_case.setdefault(case_id, []).append(event)
    _require(
        set(events_by_case) == {str(row["case_id"]) for row in rows},
        "event/case identities do not match",
    )
    expected_files = {
        "events.jsonl",
        "per_case.csv",
        "provenance.json",
        "summary.json",
        "summary.md",
        *(str(row["trace"]) for row in rows),
        *(str(row["video_path"]) for row in rows if str(row["video_path"])),
    }
    _require(
        {str(item["path"]) for item in manifest["files"]} == expected_files,
        "artifact file set does not match declared cases",
    )
    goal_tolerance = float(configuration["goal_tolerance_rad"])
    acceleration_limit = float(
        configuration["joint_acceleration_limit_rad_s2"]
    )
    _require(
        np.isfinite(goal_tolerance)
        and goal_tolerance >= 0.0
        and np.isfinite(acceleration_limit)
        and acceleration_limit > 0.0,
        "provenance safety thresholds are invalid",
    )
    _require(
        float(configuration["clearance_m"])
        == float(matrix["runtime_clearance_m"])
        and float(matrix["planning_clearance_m"]) >= 0.0,
        "planning/runtime clearance provenance mismatch",
    )
    clearance_source = matrix.get("runtime_clearance_source")
    _require(
        clearance_source in {"planning_clearance", "explicit_override"},
        "runtime clearance source is invalid",
    )
    if clearance_source == "planning_clearance":
        _require(
            float(matrix["runtime_clearance_m"])
            == float(matrix["planning_clearance_m"]),
            "inherited runtime clearance does not match planning clearance",
        )
    for row in rows:
        _validate_case_trace(
            root,
            row,
            events_by_case[str(row["case_id"])],
            goal_tolerance_rad=goal_tolerance,
            acceleration_limit_rad_s2=acceleration_limit,
        )
        video = str(row["video_path"])
        if video:
            _require((root / video).is_file(), f"recorded video missing: {video}")
    expected_summary = _build_summary(rows)
    _require(summary == expected_summary, "summary is not reproducible from CSV")
    markdown = (root / "summary.md").read_text("utf-8")
    _require(
        "scripted and non-learned" in markdown
        and "hard-real" in markdown
        and "not physical-robot safety" in markdown,
        "human-readable claim boundary missing",
    )
    return {
        "valid": True,
        "scope": SCOPE,
        "cases": len(rows),
        "modes": sorted({str(row["mode"]) for row in rows}),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "manifest_files_sha256": manifest["files_sha256"],
        "checks": [
            "manifest_inventory_sizes_and_hashes",
            "claim_boundary_flags",
            "case_event_identity",
            "trace_shapes_and_metrics_recomputed",
            "event_counts_recomputed",
            "summary_recomputed_from_csv",
        ],
    }
