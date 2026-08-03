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
    _guard_config,
    _integrate_actions,
    _safe_stream,
    _write_csv,
    _write_json,
    load_vla_config,
)
from armbench.vla.online import (
    OnlineExecutionConfig,
    ReferenceActionChunkPolicy,
    run_online_episode,
)


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
    figure.suptitle("ArmBench receding-horizon VLA runtime", fontsize=16)
    figure.text(
        0.5,
        0.015,
        "Policy: scripted non-learned reference; no pi0/pi0.5 checkpoint used",
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
        "| Scenario | Payload kg | Horizon | Queries | Task | Safe | Goal error rad | RMSE rad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {float(row['payload_mass']):g} | "
            f"{row['execution_horizon']} | {row['policy_queries']} | "
            f"{row['task_success']} | {row['physical_safe']} | "
            f"{float(row['final_goal_error_rad']):.5f} | "
            f"{float(row['rmse_rad']):.5f} |"
        )
    lines.extend(
        [
            "",
            "`policy_latency_ms` is synthetic in the reference-policy benchmark. "
            "MuJoCo advances under a pose-hold controller for that duration "
            "before the response is guarded; it is not measured model latency.",
            "",
            "The comparison isolates runtime feedback frequency and physics "
            "tracking. It is not evidence of VLA task competence.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_vla_online_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    scenarios: Sequence[str] | None = None,
    execution_horizons: Sequence[int] | None = None,
    payload_masses: Sequence[float] | None = None,
    policy_latency_ms: float | None = None,
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
    selected_policy_latency_ms = (
        float(online["policy_latency_ms"])
        if policy_latency_ms is None
        else float(policy_latency_ms)
    )
    if (
        not np.isfinite(selected_policy_latency_ms)
        or selected_policy_latency_ms < 0.0
    ):
        raise ValueError("online policy latency must be finite and nonnegative")

    resolved_id = run_id or datetime.now(timezone.utc).strftime(
        "vla_online_%Y%m%dT%H%M%SZ"
    )
    run_directory = output_root / resolved_id
    run_directory.mkdir(parents=True, exist_ok=False)
    config_snapshot = dict(config)
    config_snapshot["online_selected"] = {
        "scenarios": selected_scenarios,
        "execution_horizons": horizons,
        "payload_masses_kg": payloads,
        "policy_latency_ms": selected_policy_latency_ms,
    }
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["vla_online"] = {
        "online_physics_feedback": True,
        "camera_recapture_per_query": True,
        "policy_provenance": "scripted_non_learned_reference",
        "actual_openpi_inference": False,
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
                    policy = ReferenceActionChunkPolicy(
                        references,
                        action_dt_s=guard_config.control_dt_s,
                        action_horizon=int(
                            dict(config["openpi_contract"])["action_horizon"]
                        ),
                        velocity_limit_rad_s=(
                            guard_config.joint_velocity_clip_rad_s
                        ),
                        latency_ms=selected_policy_latency_ms,
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
                        prompt=str(dict(config["prompts"])[scenario_name]),
                    )
                    case_name = (
                        f"{scenario_name}__payload_{payload:g}kg__horizon_{horizon:02d}"
                    )
                    image_directory = run_directory / "observations"
                    image_directory.mkdir(parents=True, exist_ok=True)
                    first_external_path = (
                        image_directory / f"{case_name}__first_external.png"
                    )
                    first_wrist_path = (
                        image_directory / f"{case_name}__first_wrist.png"
                    )
                    last_external_path = (
                        image_directory / f"{case_name}__last_external.png"
                    )
                    last_wrist_path = (
                        image_directory / f"{case_name}__last_wrist.png"
                    )
                    imageio.imwrite(
                        first_external_path, result.first_exterior_image
                    )
                    imageio.imwrite(first_wrist_path, result.first_wrist_image)
                    imageio.imwrite(last_external_path, result.last_exterior_image)
                    imageio.imwrite(last_wrist_path, result.last_wrist_image)
                    if representative_external is None:
                        representative_external = first_external_path
                        representative_wrist = first_wrist_path
                    np.savez_compressed(
                        run_directory / f"{case_name}.npz",
                        reference_positions=references,
                        times=result.times,
                        desired_positions=result.desired_positions,
                        actual_positions=result.actual_positions,
                        observation_positions=np.asarray(
                            [record.observation_q for record in result.chunks]
                        ),
                        action_offsets=np.asarray(
                            [record.action_offset for record in result.chunks]
                        ),
                    )
                    row = {
                        **result.metrics(),
                        "online_physics_feedback": True,
                        "camera_recapture_per_query": True,
                        "actual_openpi_inference": False,
                        "policy_latency_ms": selected_policy_latency_ms,
                        "external_image": str(
                            first_external_path.relative_to(run_directory)
                        ),
                        "wrist_image": str(
                            first_wrist_path.relative_to(run_directory)
                        ),
                        "last_external_image": str(
                            last_external_path.relative_to(run_directory)
                        ),
                        "last_wrist_image": str(
                            last_wrist_path.relative_to(run_directory)
                        ),
                        "trace": f"{case_name}.npz",
                    }
                    rows.append(row)
                    for record in result.chunks:
                        chunk_rows.append(
                            {
                                "scenario": scenario_name,
                                "payload_mass": payload,
                                "execution_horizon": horizon,
                                **record.metrics(),
                            }
                        )
                    log(
                        f"{case_name}: task={result.task_success} "
                        f"safe={result.physical_safe} queries={result.policy_queries} "
                        f"error={result.final_goal_error_rad:.5f}"
                    )
        _write_csv(run_directory / "per_episode.csv", rows)
        _write_csv(run_directory / "per_chunk.csv", chunk_rows)
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
