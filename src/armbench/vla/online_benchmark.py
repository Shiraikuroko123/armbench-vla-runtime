"""Reproducible receding-horizon MuJoCo VLA runtime benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np

from armbench.benchmark import environment_metadata
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.benchmark import (
    OPENPI_COMMIT,
    _guard_config,
    _integrate_actions,
    _package_version,
    _safe_stream,
    _write_csv,
    _write_json,
    load_vla_config,
)
from armbench.vla.online import (
    OnlineEpisodeResult,
    OnlineExecutionConfig,
    OnlineFaultConfig,
    ReferenceActionChunkPolicy,
    run_online_episode,
)
from armbench.vla.policy import OpenPIPolicyClient

ONLINE_ARTIFACT_SCHEMA_VERSION = 4


def _online_config(config: dict[str, object]) -> dict[str, object]:
    if "online" not in config:
        raise ValueError("VLA config is missing the online benchmark section")
    online = dict(config["online"])
    required = {
        "execution_horizons",
        "payload_masses_kg",
        "policy_latency_ms",
        "controller_dt_s",
        "warmup_s",
        "hold_s",
        "goal_tolerance_rad",
        "max_extra_actions",
        "kp",
        "kd",
    }
    missing = required.difference(online)
    if missing:
        raise ValueError(f"online config is missing keys: {sorted(missing)}")
    horizons = [int(value) for value in online["execution_horizons"]]
    if not horizons or len(horizons) != len(set(horizons)):
        raise ValueError("online execution horizons must be unique and nonempty")
    if any(value <= 0 or value > 15 for value in horizons):
        raise ValueError("online execution horizons must be within [1, 15]")
    payloads = [float(value) for value in online["payload_masses_kg"]]
    if not payloads or any(
        not np.isfinite(value) or value < 0.0 for value in payloads
    ):
        raise ValueError("online payload masses must be nonnegative and nonempty")
    latency_ms = float(online["policy_latency_ms"])
    if not np.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("online policy latency must be nonnegative")
    return online


def _write_overview(
    path: Path,
    rows: list[dict[str, object]],
    exterior_path: Path,
    wrist_path: Path,
    *,
    title: str = "ArmBench receding-horizon VLA runtime",
    footer: str = (
        "Policy: scripted non-learned reference; no pi0/pi0.5 checkpoint used"
    ),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exterior = imageio.imread(exterior_path)
    wrist = imageio.imread(wrist_path)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    axes[0, 0].imshow(exterior)
    axes[0, 0].set_title("Live exterior observation")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(wrist)
    axes[0, 1].set_title("Live wrist observation")
    axes[0, 1].axis("off")

    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), float(row["payload_mass"]))
        groups.setdefault(key, []).append(row)
    for (scenario, payload), group in groups.items():
        ordered = sorted(group, key=lambda item: int(item["execution_horizon"]))
        label = f"{scenario}, {payload:g} kg"
        axes[1, 0].plot(
            [int(item["execution_horizon"]) for item in ordered],
            [float(item["final_goal_error_rad"]) for item in ordered],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            [int(item["execution_horizon"]) for item in ordered],
            [int(item["policy_queries"]) for item in ordered],
            marker="o",
            label=label,
        )
    horizons = sorted({int(row["execution_horizon"]) for row in rows})
    axes[1, 0].set_xticks(horizons)
    axes[1, 0].set_xlabel("Executed actions before re-observation")
    axes[1, 0].set_ylabel("Final max joint error (rad)")
    axes[1, 0].set_title("Closed-loop horizon vs. goal error")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].set_xticks(horizons)
    axes[1, 1].set_xlabel("Executed actions before re-observation")
    axes[1, 1].set_ylabel("Policy queries / camera recaptures")
    axes[1, 1].set_title("Responsiveness cost")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(title, fontsize=16)
    figure.text(
        0.5,
        0.015,
        footer,
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Receding-horizon MuJoCo VLA runtime benchmark",
        "",
        "This benchmark executes only a prefix of each 15x8 action chunk, then "
        "recaptures both 224x224 cameras and actual MuJoCo joint state before "
        "the next policy query.",
        "",
        "The policy is `scripted_non_learned_reference`. No pi0/pi0.5 "
        "checkpoint or learned-policy inference was used.",
        "",
        "Repeating synthetic latency profile (ms): "
        f"`{json.dumps(rows[0]['policy_latency_schedule_ms'])}`.",
        "",
        "| Scenario | Payload kg | Horizon | Queries | Termination | Task | Safe | Faults | Deadlines | State mismatches | Goal error rad | RMSE rad |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {float(row['payload_mass']):g} | "
            f"{row['execution_horizon']} | {row['policy_queries']} | "
            f"{row['termination_reason']} | "
            f"{row['task_success']} | {row['physical_safe']} | "
            f"{row['fault_injections']} | {row['deadline_chunks']} | "
            f"{row['state_mismatch_chunks']} | "
            f"{float(row['final_goal_error_rad']):.5f} | "
            f"{float(row['rmse_rad']):.5f} |"
        )
    lines.extend(
        [
            "",
            "The latency profile is synthetic in the reference-policy benchmark. "
            "MuJoCo advances under a pose-hold controller for each delay before "
            "the response is guarded; it is not measured model latency.",
            "",
            "The comparison isolates runtime feedback frequency and physics "
            "tracking. It is not evidence of VLA task competence.",
            "",
        ]
    )
    if any(bool(row["synthetic_state_jump"]) for row in rows):
        lines.extend(
            [
                "The optional state jump is a deterministic fault injected "
                "directly into MuJoCo joint state after observation capture. "
                "It tests dispatch-state consistency handling; it is not a "
                "modeled contact impulse.",
                "",
            ]
        )
    return "\n".join(lines)


def _write_episode_artifacts(
    run_directory: Path,
    case_name: str,
    result: OnlineEpisodeResult,
    reference_positions: np.ndarray,
    *,
    extra_metrics: dict[str, object],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    Path,
    Path,
]:
    image_directory = run_directory / "observations"
    image_directory.mkdir(parents=True, exist_ok=True)
    first_external_path = image_directory / f"{case_name}__first_external.png"
    first_wrist_path = image_directory / f"{case_name}__first_wrist.png"
    last_external_path = image_directory / f"{case_name}__last_external.png"
    last_wrist_path = image_directory / f"{case_name}__last_wrist.png"
    imageio.imwrite(first_external_path, result.first_exterior_image)
    imageio.imwrite(first_wrist_path, result.first_wrist_image)
    imageio.imwrite(last_external_path, result.last_exterior_image)
    imageio.imwrite(last_wrist_path, result.last_wrist_image)

    raw_action_chunks = np.asarray(
        [
            record.raw_actions
            if record.raw_actions is not None
            else np.full_like(record.guarded_actions, np.nan)
            for record in result.chunks
        ]
    )
    guarded_action_chunks = np.asarray(
        [record.guarded_actions for record in result.chunks]
    )
    trace_name = f"{case_name}.npz"
    np.savez_compressed(
        run_directory / trace_name,
        reference_positions=reference_positions,
        times=result.times,
        desired_positions=result.desired_positions,
        actual_positions=result.actual_positions,
        observation_positions=np.asarray(
            [record.observation_q for record in result.chunks]
        ),
        dispatch_positions=np.asarray(
            [record.dispatch_q for record in result.chunks]
        ),
        post_chunk_positions=np.asarray(
            [record.actual_q_after for record in result.chunks]
        ),
        query_indices=np.asarray(
            [record.query_index for record in result.chunks], dtype=int
        ),
        action_offsets=np.asarray(
            [record.action_offset for record in result.chunks], dtype=int
        ),
        executed_horizons=np.asarray(
            [record.executed_horizon for record in result.chunks], dtype=int
        ),
        policy_latencies_ms=np.asarray(
            [record.policy_latency_ms for record in result.chunks]
        ),
        client_inference_latencies_ms=np.asarray(
            [record.client_inference_latency_ms for record in result.chunks]
        ),
        exterior_image_sha256=np.asarray(
            [record.exterior_image_sha256 for record in result.chunks]
        ),
        wrist_image_sha256=np.asarray(
            [record.wrist_image_sha256 for record in result.chunks]
        ),
        exterior_frame_delta_mean_abs=np.asarray(
            [
                np.nan
                if record.exterior_frame_delta_mean_abs is None
                else record.exterior_frame_delta_mean_abs
                for record in result.chunks
            ]
        ),
        wrist_frame_delta_mean_abs=np.asarray(
            [
                np.nan
                if record.wrist_frame_delta_mean_abs is None
                else record.wrist_frame_delta_mean_abs
                for record in result.chunks
            ]
        ),
        exterior_image_thumbnails=np.asarray(
            [record.exterior_thumbnail for record in result.chunks]
        ),
        wrist_image_thumbnails=np.asarray(
            [record.wrist_thumbnail for record in result.chunks]
        ),
        raw_action_chunks=raw_action_chunks,
        guarded_action_chunks=guarded_action_chunks,
        predicted_position_chunks=np.asarray(
            [record.predicted_positions for record in result.chunks]
        ),
    )
    row = {
        **result.metrics(),
        **extra_metrics,
        "video_path": (
            str(
                Path(result.video_path).relative_to(
                    run_directory.resolve()
                )
            )
            if result.video_path is not None
            else None
        ),
        "external_image": str(first_external_path.relative_to(run_directory)),
        "wrist_image": str(first_wrist_path.relative_to(run_directory)),
        "last_external_image": str(
            last_external_path.relative_to(run_directory)
        ),
        "last_wrist_image": str(last_wrist_path.relative_to(run_directory)),
        "trace": trace_name,
    }
    chunk_rows = [
        {
            "scenario": result.scenario,
            "payload_mass": result.payload_mass,
            "execution_horizon": result.execution_horizon,
            **record.metrics(),
        }
        for record in result.chunks
    ]
    action_rows: list[dict[str, object]] = []
    for record in result.chunks:
        for action_index, guarded_action in enumerate(record.guarded_actions):
            raw_action = (
                record.raw_actions[action_index]
                if record.raw_actions is not None
                else None
            )
            action_rows.append(
                {
                    "scenario": result.scenario,
                    "payload_mass": result.payload_mass,
                    "execution_horizon": result.execution_horizon,
                    "query_index": record.query_index,
                    "sequence_id": record.sequence_id,
                    "action_offset": record.action_offset,
                    "action_index": action_index,
                    "executed": action_index < record.executed_horizon,
                    "decision_status": record.decision_status,
                    "policy_source": record.policy_source,
                    "reason": record.action_reasons[action_index],
                    "intervened": record.action_interventions[action_index],
                    "scale": float(record.action_scales[action_index]),
                    "raw_action": (
                        json.dumps(raw_action.tolist())
                        if raw_action is not None
                        else None
                    ),
                    "guarded_action": json.dumps(guarded_action.tolist()),
                    "q_before": json.dumps(
                        record.predicted_positions[action_index].tolist()
                    ),
                    "q_after": json.dumps(
                        record.predicted_positions[action_index + 1].tolist()
                    ),
                }
            )
    return (
        row,
        chunk_rows,
        action_rows,
        first_external_path,
        first_wrist_path,
    )


def execute_vla_online_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    scenarios: Sequence[str] | None = None,
    execution_horizons: Sequence[int] | None = None,
    payload_masses: Sequence[float] | None = None,
    policy_latency_ms: float | None = None,
    policy_latency_schedule_ms: Sequence[float] | None = None,
    state_jump_query: int | None = None,
    state_jump_joint: int | None = None,
    state_jump_rad: float | None = None,
    make_videos: bool = False,
) -> Path:
    config = load_vla_config(config_path)
    online = _online_config(config)
    selected_scenarios = [
        str(value) for value in (scenarios or config["scenarios"])
    ]
    unknown = set(selected_scenarios).difference(mujoco_scenarios())
    if unknown:
        raise ValueError(f"unknown online scenarios: {sorted(unknown)}")
    horizons = [
        int(value)
        for value in (
            execution_horizons or list(online["execution_horizons"])
        )
    ]
    if not horizons or any(value <= 0 or value > 15 for value in horizons):
        raise ValueError("online execution horizons must be within [1, 15]")
    payloads = [
        float(value)
        for value in (payload_masses or list(online["payload_masses_kg"]))
    ]
    if not payloads or any(
        not np.isfinite(value) or value < 0.0 for value in payloads
    ):
        raise ValueError("online payload masses must be nonnegative and nonempty")
    if policy_latency_ms is not None and policy_latency_schedule_ms is not None:
        raise ValueError("use either policy latency or a latency schedule")
    if policy_latency_schedule_ms is None:
        selected_policy_latency_ms = (
            float(online["policy_latency_ms"])
            if policy_latency_ms is None
            else float(policy_latency_ms)
        )
        selected_latency_schedule = [selected_policy_latency_ms]
    else:
        selected_policy_latency_ms = None
        selected_latency_schedule = [
            float(value) for value in policy_latency_schedule_ms
        ]
    if not selected_latency_schedule or any(
        not np.isfinite(value) or value < 0.0
        for value in selected_latency_schedule
    ):
        raise ValueError(
            "online policy latency profile must be finite, nonnegative, and nonempty"
        )
    jump_arguments = (state_jump_joint, state_jump_rad)
    if (jump_arguments[0] is None) != (jump_arguments[1] is None):
        raise ValueError("state jump requires both joint and magnitude")
    selected_state_jump_query: int | None = None
    selected_state_jump = np.zeros(7, dtype=float)
    if state_jump_joint is not None and state_jump_rad is not None:
        if state_jump_joint < 1 or state_jump_joint > 7:
            raise ValueError("state jump joint must be within [1, 7]")
        if not np.isfinite(state_jump_rad) or state_jump_rad == 0.0:
            raise ValueError("state jump magnitude must be finite and nonzero")
        selected_state_jump_query = (
            0 if state_jump_query is None else int(state_jump_query)
        )
        selected_state_jump[state_jump_joint - 1] = float(state_jump_rad)
    elif state_jump_query is not None:
        raise ValueError("state jump query requires a configured state jump")
    fault_config = OnlineFaultConfig(
        state_jump_query=selected_state_jump_query,
        state_jump_rad=tuple(float(value) for value in selected_state_jump),
    )

    resolved_id = run_id or datetime.now(timezone.utc).strftime(
        "vla_online_%Y%m%dT%H%M%SZ"
    )
    run_directory = output_root / resolved_id
    run_directory.mkdir(parents=True, exist_ok=False)
    config_snapshot = dict(config)
    config_snapshot["online_selected"] = {
        "artifact_schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "scenarios": selected_scenarios,
        "execution_horizons": horizons,
        "payload_masses_kg": payloads,
        "policy_latency_ms": selected_policy_latency_ms,
        "policy_latency_schedule_ms": selected_latency_schedule,
        "state_jump_query": selected_state_jump_query,
        "state_jump_rad": selected_state_jump.tolist(),
        "make_videos": bool(make_videos),
    }
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["packages"].update(
        {
            "imageio": _package_version("imageio"),
            "msgpack": _package_version("msgpack"),
            "mujoco": _package_version("mujoco"),
            "openpi-client": _package_version("openpi-client"),
            "websockets": _package_version("websockets"),
        }
    )
    metadata["vla_online"] = {
        "online_physics_feedback": True,
        "artifact_schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "camera_recapture_per_query": True,
        "camera_observation_audit": {
            "full_frame_hash": "sha256",
            "frame_delta": "mean_abs_uint8",
            "thumbnail_shape": [16, 16, 3],
        },
        "policy_provenance": "scripted_non_learned_reference",
        "remote_policy_response_validated": False,
        "checkpoint_identity_verified": False,
        "synthetic_state_jump": fault_config.enabled,
        "online_video_recording": bool(make_videos),
        "openpi_contract": dict(config["openpi_contract"]),
    }
    _write_json(run_directory / "config.json", config_snapshot)
    _write_json(run_directory / "environment.json", metadata)
    guard_config = _guard_config(config)
    execution_config = OnlineExecutionConfig(
        action_dt_s=guard_config.control_dt_s,
        controller_dt_s=float(online["controller_dt_s"]),
        warmup_s=float(online["warmup_s"]),
        hold_s=float(online["hold_s"]),
        goal_tolerance_rad=float(online["goal_tolerance_rad"]),
        max_extra_actions=int(online["max_extra_actions"]),
        kp=tuple(float(value) for value in online["kp"]),
        kd=tuple(float(value) for value in online["kd"]),
    )
    raw_guard = dict(config["guard"])
    rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    representative_external: Path | None = None
    representative_wrist: Path | None = None
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        log_lines.append(line)
        print(message, flush=True)

    try:
        for scenario_name in selected_scenarios:
            scenario = mujoco_scenarios()[scenario_name]
            reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
            actions = _safe_stream(scenario_name, config, guard_config)
            references = _integrate_actions(
                reference_robot, scenario.start, actions, guard_config
            )
            for payload in payloads:
                for horizon in horizons:
                    case_name = (
                        f"{scenario_name}__payload_{payload:g}kg__horizon_{horizon:02d}"
                    )
                    video_path = (
                        run_directory / "videos" / f"{case_name}.mp4"
                        if make_videos
                        else None
                    )
                    policy = ReferenceActionChunkPolicy(
                        references,
                        action_dt_s=guard_config.control_dt_s,
                        action_horizon=int(
                            dict(config["openpi_contract"])["action_horizon"]
                        ),
                        velocity_limit_rad_s=(
                            guard_config.joint_velocity_clip_rad_s
                        ),
                        latency_ms=(
                            float(selected_policy_latency_ms)
                            if selected_policy_latency_ms is not None
                            else 0.0
                        ),
                        latency_schedule_ms=(
                            selected_latency_schedule
                            if selected_policy_latency_ms is None
                            else None
                        ),
                    )
                    result = run_online_episode(
                        scenario_name,
                        policy,
                        references,
                        execution_horizon=horizon,
                        payload_mass=payload,
                        clearance_m=float(raw_guard["clearance_m"]),
                        collision_resolution_rad=float(
                            raw_guard["collision_resolution_rad"]
                        ),
                        guard_config=guard_config,
                        execution_config=execution_config,
                        fault_config=fault_config,
                        prompt=str(dict(config["prompts"])[scenario_name]),
                        video_path=video_path,
                        video_fps=int(dict(config["execution"])["video_fps"]),
                    )
                    (
                        row,
                        case_chunks,
                        case_actions,
                        first_external_path,
                        first_wrist_path,
                    ) = _write_episode_artifacts(
                        run_directory,
                        case_name,
                        result,
                        references,
                        extra_metrics={
                            "online_physics_feedback": True,
                            "artifact_schema_version": (
                                ONLINE_ARTIFACT_SCHEMA_VERSION
                            ),
                            "camera_recapture_per_query": True,
                            "remote_policy_response_validated": False,
                            "checkpoint_identity_verified": False,
                            "policy_latency_ms": selected_policy_latency_ms,
                            "policy_latency_schedule_ms": (
                                selected_latency_schedule
                            ),
                            "synthetic_state_jump": fault_config.enabled,
                            "state_jump_query": selected_state_jump_query,
                            "state_jump_rad": selected_state_jump.tolist(),
                        },
                    )
                    if representative_external is None:
                        representative_external = first_external_path
                        representative_wrist = first_wrist_path
                    rows.append(row)
                    chunk_rows.extend(case_chunks)
                    action_rows.extend(case_actions)
                    log(
                        f"{case_name}: task={result.task_success} "
                        f"safe={result.physical_safe} queries={result.policy_queries} "
                        f"error={result.final_goal_error_rad:.5f}"
                    )
        _write_csv(run_directory / "per_episode.csv", rows)
        _write_csv(run_directory / "per_chunk.csv", chunk_rows)
        _write_csv(run_directory / "per_action.csv", action_rows)
        _write_json(run_directory / "aggregate.json", rows)
        if representative_external is None or representative_wrist is None:
            raise RuntimeError("online benchmark produced no camera evidence")
        _write_overview(
            run_directory / "overview.png",
            rows,
            representative_external,
            representative_wrist,
        )
        (run_directory / "summary.md").write_text(
            _summary(rows), encoding="utf-8", newline="\n"
        )
        log("Online VLA benchmark complete")
    finally:
        (run_directory / "run.log").write_text(
            "\n".join(log_lines) + ("\n" if log_lines else ""),
            encoding="utf-8",
            newline="\n",
        )
    return run_directory


def _remote_summary(
    row: dict[str, object], server: str, server_metadata: dict[str, object]
) -> str:
    remote_validated = bool(row["remote_policy_response_validated"])
    validated = int(row["validated_remote_chunks"])
    metadata_text = json.dumps(
        server_metadata,
        sort_keys=True,
        default=lambda item: item.tolist()
        if isinstance(item, np.ndarray)
        else str(item),
    )
    return "\n".join(
        [
            "# OpenPI-compatible remote closed-loop MuJoCo run",
            "",
            f"- Server: `{server}`",
            f"- Server metadata: `{metadata_text}`",
            f"- Validated remote 15x8 chunks: `{validated}`",
            "- Remote policy response validated: "
            f"`{str(remote_validated).lower()}`",
            "- Checkpoint identity verified by protocol: `false`",
            f"- Termination: `{row['termination_reason']}`",
            "",
            "| Scenario | Horizon | Queries | Valid replies | Task | Safe | P95 end-to-end ms | Interventions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| {row['scenario']} | {row['execution_horizon']} | "
            f"{row['policy_queries']} | {validated} | {row['task_success']} | "
            f"{row['physical_safe']} | "
            f"{float(row['p95_policy_latency_ms']):.2f} | "
            f"{row['executed_interventions']} |",
            "",
            "This command uses the real bounded OpenPI WebSocket transport and "
            "recaptures live MuJoCo state and both cameras between action "
            "prefixes. A connected server without a validated reply is not "
            "counted as a validated remote policy response.",
            "",
            "The WebSocket protocol does not attest checkpoint identity. "
            "Preserve the GPU server launch command/log alongside this "
            "artifact when making a pi0/pi0.5-specific claim.",
            "",
            "The synthetic workcell is outside the evidence used to train the "
            "checkpoint. Task success or failure here is an integration result, "
            "not a general policy-quality benchmark.",
            "",
        ]
    )


def execute_openpi_online_run(
    config_path: Path,
    output_directory: Path,
    *,
    host: str,
    port: int,
    scenario_name: str,
    execution_horizon: int = 5,
    payload_mass: float = 0.0,
    max_policy_queries: int = 10,
    prompt: str | None = None,
    api_key: str | None = None,
    connect_timeout_s: float = 3.0,
    inference_timeout_s: float = 1.0,
    make_video: bool = False,
) -> Path:
    """Run bounded remote OpenPI inference in the live MuJoCo feedback loop."""

    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    config = load_vla_config(config_path)
    online = _online_config(config)
    scenarios = mujoco_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"unknown online scenario: {scenario_name}")
    if execution_horizon <= 0 or execution_horizon > 15:
        raise ValueError("execution_horizon must be within [1, 15]")
    if not np.isfinite(payload_mass) or payload_mass < 0.0:
        raise ValueError("payload_mass must be finite and nonnegative")
    if max_policy_queries <= 0:
        raise ValueError("max_policy_queries must be positive")

    scenario = scenarios[scenario_name]
    task_prompt = prompt or str(dict(config["prompts"])[scenario_name])
    guard_config = _guard_config(config)
    raw_guard = dict(config["guard"])
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    actions = _safe_stream(scenario_name, config, guard_config)
    references = _integrate_actions(
        reference_robot, scenario.start, actions, guard_config
    )
    execution_config = OnlineExecutionConfig(
        action_dt_s=guard_config.control_dt_s,
        controller_dt_s=float(online["controller_dt_s"]),
        warmup_s=float(online["warmup_s"]),
        hold_s=float(online["hold_s"]),
        goal_tolerance_rad=float(online["goal_tolerance_rad"]),
        max_extra_actions=int(online["max_extra_actions"]),
        max_policy_queries=int(max_policy_queries),
        kp=tuple(float(value) for value in online["kp"]),
        kd=tuple(float(value) for value in online["kd"]),
    )
    expected_horizon = int(dict(config["openpi_contract"])["action_horizon"])
    server = f"{host}:{port}"
    case_name = (
        f"{scenario_name}__openpi_remote__horizon_{execution_horizon:02d}"
    )
    with OpenPIPolicyClient(
        host=host,
        port=port,
        api_key=api_key,
        expected_horizon=expected_horizon,
        connect_timeout_s=connect_timeout_s,
        inference_timeout_s=inference_timeout_s,
    ) as client:
        server_metadata = client.server_metadata
        output_directory.mkdir(parents=True, exist_ok=False)
        result = run_online_episode(
            scenario_name,
            client,
            references,
            execution_horizon=execution_horizon,
            payload_mass=payload_mass,
            clearance_m=float(raw_guard["clearance_m"]),
            collision_resolution_rad=float(
                raw_guard["collision_resolution_rad"]
            ),
            guard_config=guard_config,
            execution_config=execution_config,
            prompt=task_prompt,
            video_path=(
                output_directory / "videos" / f"{case_name}.mp4"
                if make_video
                else None
            ),
            video_fps=int(dict(config["execution"])["video_fps"]),
        )

    validated_remote_chunks = sum(
        record.validated_policy_response
        and record.policy_source == "openpi_remote"
        and record.raw_actions is not None
        for record in result.chunks
    )
    remote_policy_response_validated = validated_remote_chunks > 0
    config_snapshot = dict(config)
    config_snapshot["openpi_online_selected"] = {
        "artifact_schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "server": server,
        "scenario": scenario_name,
        "execution_horizon": execution_horizon,
        "payload_mass_kg": payload_mass,
        "max_policy_queries": max_policy_queries,
        "prompt": task_prompt,
        "connect_timeout_s": connect_timeout_s,
        "inference_timeout_s": inference_timeout_s,
        "api_key_configured": api_key is not None,
        "make_video": bool(make_video),
    }
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["packages"].update(
        {
            "imageio": _package_version("imageio"),
            "msgpack": _package_version("msgpack"),
            "mujoco": _package_version("mujoco"),
            "openpi-client": _package_version("openpi-client"),
            "websockets": _package_version("websockets"),
        }
    )
    metadata["vla_online"] = {
        "online_physics_feedback": True,
        "artifact_schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "camera_recapture_per_query": True,
        "camera_observation_audit": {
            "full_frame_hash": "sha256",
            "frame_delta": "mean_abs_uint8",
            "thumbnail_shape": [16, 16, 3],
        },
        "remote_openpi_transport": True,
        "remote_policy_response_validated": remote_policy_response_validated,
        "checkpoint_identity_verified": False,
        "validated_remote_chunks": validated_remote_chunks,
        "server": server,
        "server_metadata": server_metadata,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_contract": dict(config["openpi_contract"]),
        "online_video_recording": bool(make_video),
    }
    _write_json(output_directory / "config.json", config_snapshot)
    _write_json(output_directory / "environment.json", metadata)
    row, chunks, action_rows, exterior_path, wrist_path = (
        _write_episode_artifacts(
            output_directory,
            case_name,
            result,
            references,
            extra_metrics={
                "online_physics_feedback": True,
                "artifact_schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
                "camera_recapture_per_query": True,
                "remote_inference_attempted": True,
                "remote_policy_response_validated": (
                    remote_policy_response_validated
                ),
                "checkpoint_identity_verified": False,
                "validated_remote_chunks": validated_remote_chunks,
                "server": server,
                "openpi_commit": OPENPI_COMMIT,
                "synthetic_state_jump": False,
            },
        )
    )
    _write_csv(output_directory / "per_episode.csv", [row])
    _write_csv(output_directory / "per_chunk.csv", chunks)
    _write_csv(output_directory / "per_action.csv", action_rows)
    _write_json(output_directory / "aggregate.json", [row])
    _write_overview(
        output_directory / "overview.png",
        [row],
        exterior_path,
        wrist_path,
        title="ArmBench remote OpenPI closed loop",
        footer=(
            f"Policy: openpi_remote; validated replies: "
            f"{validated_remote_chunks}/{result.policy_queries}"
        ),
    )
    (output_directory / "summary.md").write_text(
        _remote_summary(row, server, server_metadata),
        encoding="utf-8",
        newline="\n",
    )
    (output_directory / "run.log").write_text(
        f"remote_openpi server={server} scenario={scenario_name} "
        f"queries={result.policy_queries} "
        f"validated={validated_remote_chunks} "
        f"termination={result.termination_reason} "
        f"task={result.task_success} safe={result.physical_safe}\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_directory
