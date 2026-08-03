"""Non-interactive figures for paths and benchmark summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from armbench.control.simulation import TrackingResult
from armbench.model import RobotModel
from armbench.scenario import Scenario


def _plot_sphere(ax: plt.Axes, center: np.ndarray, radius: float) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 18j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color="#d95f59", alpha=0.45, linewidth=0)


def plot_scene_path(
    robot: RobotModel,
    scenario: Scenario,
    path: Sequence[np.ndarray],
    output: Path,
    *,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for obstacle in scenario.obstacles:
        _plot_sphere(ax, obstacle.center, obstacle.radius)
        ax.text(*obstacle.center, obstacle.label, fontsize=7)

    sample_count = min(12, len(path))
    indices = np.unique(np.linspace(0, len(path) - 1, sample_count).astype(int))
    for index in indices:
        points = robot.forward_points(path[index])
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            "o-",
            color="#67717a",
            alpha=0.22,
            linewidth=1.0,
            markersize=2.5,
        )
    for configuration, color, label in (
        (path[0], "#16855b", "start"),
        (path[-1], "#bd3e37", "goal"),
    ):
        points = robot.forward_points(configuration)
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            "o-",
            color=color,
            linewidth=2.2,
            markersize=4,
            label=label,
        )
    end_effector = np.asarray([robot.forward_points(q)[-1] for q in path])
    ax.plot(
        end_effector[:, 0],
        end_effector[:, 1],
        end_effector[:, 2],
        color="#276fbf",
        linewidth=1.6,
        label="end-effector path",
    )
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
    ax.set_xlim(-0.65, 0.75)
    ax.set_ylim(-0.65, 0.75)
    ax.set_zlim(0.0, 1.15)
    ax.set_box_aspect((1.4, 1.4, 1.15))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_planning_summary(aggregate: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in aggregate))
    planners = list(dict.fromkeys(str(row["planner"]) for row in aggregate))
    lookup = {(str(row["scenario"]), str(row["planner"])): row for row in aggregate}
    x = np.arange(len(scenarios))
    width = 0.36 if len(planners) == 2 else 0.8 / max(len(planners), 1)
    colors = ["#276fbf", "#d05a44", "#16855b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for index, planner in enumerate(planners):
        offset = (index - (len(planners) - 1) / 2.0) * width
        success = [float(lookup[(scenario, planner)]["success_rate"]) for scenario in scenarios]
        latency = [float(lookup[(scenario, planner)]["latency_p95_ms"]) for scenario in scenarios]
        axes[0].bar(x + offset, success, width, label=planner, color=colors[index])
        axes[1].bar(x + offset, latency, width, label=planner, color=colors[index])
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("P95 planning latency (ms)")
    for ax in axes:
        ax.set_xticks(x, scenarios, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_control_summary(aggregate: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    controllers = list(dict.fromkeys(str(row["controller"]) for row in aggregate))
    inertias = sorted({float(row["inertia_scale"]) for row in aggregate})
    colors = {"pd": "#d05a44", "lqr": "#276fbf"}
    for controller in controllers:
        for inertia in inertias:
            rows = sorted(
                (
                    row
                    for row in aggregate
                    if row["controller"] == controller
                    and float(row["inertia_scale"]) == inertia
                ),
                key=lambda row: int(row["delay_ms"]),
            )
            label = f"{controller}, load x{inertia:g}"
            style = "-" if inertia == min(inertias) else "--"
            axes[0].plot(
                [row["delay_ms"] for row in rows],
                [row["rmse_mean"] for row in rows],
                marker="o",
                linestyle=style,
                color=colors.get(controller),
                label=label,
            )
            axes[1].plot(
                [row["delay_ms"] for row in rows],
                [row["invalid_state_samples_mean"] for row in rows],
                marker="o",
                linestyle=style,
                color=colors.get(controller),
                label=label,
            )
    axes[0].set(xlabel="observation delay (ms)", ylabel="joint RMSE (rad)")
    axes[1].set(
        xlabel="observation delay (ms)",
        ylabel="invalid state samples (mean)",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_tracking_trace(
    result: TrackingResult, trajectory_duration: float, output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    error = np.max(
        np.abs(result.actual_positions - result.desired_positions), axis=1
    )
    command = np.max(np.abs(result.commands), axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True)
    axes[0].plot(result.times, error, color="#276fbf")
    axes[0].set_ylabel("max joint error (rad)")
    axes[1].plot(result.times, command, color="#d05a44")
    axes[1].set(xlabel="time (s)", ylabel="max |command| (rad/s^2)")
    for ax in axes:
        ax.axvline(trajectory_duration, color="#333333", linestyle="--", linewidth=1)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{result.controller.upper()} | delay {result.delay_ms} ms | "
        f"load x{result.inertia_scale:g}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)

