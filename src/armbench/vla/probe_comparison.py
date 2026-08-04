"""Paired comparison of two validated fixed-request OpenPI probes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from armbench.benchmark import environment_metadata
from armbench.vla.benchmark import _write_csv, _write_json
from armbench.vla.replay_probe import (
    RecordedProbeValidationResult,
    validate_recorded_openpi_probe,
)


PROBE_COMPARISON_ARTIFACT_TYPE = "armbench_recorded_openpi_probe_comparison_v1"


@dataclass(frozen=True)
class _ProbeData:
    directory: Path
    validation: RecordedProbeValidationResult
    response: dict[str, object]
    raw_actions: np.ndarray
    guarded_actions: np.ndarray
    predicted_positions: np.ndarray


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if not 0 <= value <= 15:
        raise ValueError(f"{label} must be in [0, 15]")
    return value


def _load_probe(directory: Path) -> _ProbeData:
    root = directory.resolve()
    validation = validate_recorded_openpi_probe(root)
    response_value = json.loads(
        (root / "response.json").read_text(encoding="utf-8")
    )
    response = _mapping(response_value, "probe response")
    with np.load(root / "response.npz", allow_pickle=False) as trace:
        raw_actions = np.asarray(trace["raw_actions"]).copy()
        guarded_actions = np.asarray(trace["guarded_actions"]).copy()
        predicted_positions = np.asarray(trace["predicted_positions"]).copy()
    return _ProbeData(
        directory=root,
        validation=validation,
        response=response,
        raw_actions=raw_actions,
        guarded_actions=guarded_actions,
        predicted_positions=predicted_positions,
    )


def _probe_metrics(probe: _ProbeData, label: str) -> dict[str, object]:
    guard = _mapping(probe.response.get("guard"), f"{label} guard")
    return {
        "label": label,
        "directory": str(probe.directory),
        "server": str(probe.response.get("server", "")),
        "policy_provenance": probe.validation.policy_provenance,
        "action_sha256": probe.validation.action_sha256,
        "client_inference_latency_ms": _number(
            probe.response.get("client_inference_latency_ms"),
            f"{label} client inference latency",
        ),
        "guard_safe_after": probe.validation.guard_safe_after,
        "guard_intervention_steps": _count(
            guard.get("intervention_steps"),
            f"{label} guard intervention steps",
        ),
        "guard_hold_steps": _count(
            guard.get("hold_steps"), f"{label} guard hold steps"
        ),
        "checkpoint_identity_verified": False,
    }


def _difference_metrics(
    left: _ProbeData, right: _ProbeData
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    raw_delta = right.raw_actions - left.raw_actions
    guarded_delta = right.guarded_actions - left.guarded_actions
    position_delta = right.predicted_positions - left.predicted_positions
    raw_step_l2 = np.linalg.norm(raw_delta, axis=1)
    guarded_step_l2 = np.linalg.norm(guarded_delta, axis=1)
    per_step = [
        {
            "step": step,
            "raw_action_l2_difference": float(raw_step_l2[step]),
            "raw_velocity_l2_difference": float(
                np.linalg.norm(raw_delta[step, :7])
            ),
            "raw_gripper_abs_difference": float(abs(raw_delta[step, 7])),
            "guarded_action_l2_difference": float(guarded_step_l2[step]),
            "left_raw_action_l2": float(
                np.linalg.norm(left.raw_actions[step])
            ),
            "right_raw_action_l2": float(
                np.linalg.norm(right.raw_actions[step])
            ),
        }
        for step in range(15)
    ]
    dimension_metrics = [
        {
            "dimension": dimension,
            "name": f"joint_velocity_{dimension + 1}"
            if dimension < 7
            else "gripper",
            "rmse": float(np.sqrt(np.mean(raw_delta[:, dimension] ** 2))),
            "max_abs_difference": float(
                np.max(np.abs(raw_delta[:, dimension]))
            ),
        }
        for dimension in range(8)
    ]
    metrics: dict[str, object] = {
        "raw_actions_identical": bool(
            left.validation.action_sha256 == right.validation.action_sha256
        ),
        "raw_action_rmse": float(np.sqrt(np.mean(raw_delta**2))),
        "raw_action_mean_abs_difference": float(np.mean(np.abs(raw_delta))),
        "raw_action_max_abs_difference": float(np.max(np.abs(raw_delta))),
        "guarded_action_rmse": float(np.sqrt(np.mean(guarded_delta**2))),
        "guarded_action_max_abs_difference": float(
            np.max(np.abs(guarded_delta))
        ),
        "predicted_position_rmse_rad": float(
            np.sqrt(np.mean(position_delta**2))
        ),
        "predicted_position_max_abs_difference_rad": float(
            np.max(np.abs(position_delta))
        ),
        "per_dimension": dimension_metrics,
    }
    return metrics, per_step, raw_step_l2, guarded_step_l2


def _write_plot(
    path: Path,
    left_label: str,
    right_label: str,
    raw_step_l2: np.ndarray,
    guarded_step_l2: np.ndarray,
    dimension_metrics: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(15)
    dimensions = np.arange(8)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(
        steps,
        raw_step_l2,
        marker="o",
        linewidth=1.8,
        label="Raw action",
        color="#b84a3a",
    )
    axes[0].plot(
        steps,
        guarded_step_l2,
        marker="s",
        linewidth=1.8,
        label="After guard",
        color="#267a62",
    )
    axes[0].set_xlabel("Action step")
    axes[0].set_ylabel("L2 difference")
    axes[0].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axes[0].set_title("Chunk difference by step")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        dimensions,
        [float(row["rmse"]) for row in dimension_metrics],
        color=["#3677a8"] * 7 + ["#d29b31"],
    )
    axes[1].set_xticks(
        dimensions,
        ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Grip"],
    )
    axes[1].set_xlabel("Action dimension")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Raw difference by dimension")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        f"Fixed-request OpenPI probe: {left_label} vs {right_label}"
    )
    figure.text(
        0.5,
        0.015,
        "Same serialized input required | labels are user-supplied, not "
        "checkpoint attestation",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary(comparison: dict[str, object]) -> str:
    left = _mapping(comparison["left"], "left comparison")
    right = _mapping(comparison["right"], "right comparison")
    metrics = _mapping(comparison["metrics"], "comparison metrics")
    return "\n".join(
        [
            "# Paired recorded OpenPI probe comparison",
            "",
            f"- Left label: `{left['label']}`",
            f"- Right label: `{right['label']}`",
            "- Matched request payload SHA-256: "
            f"`{comparison['request_payload_sha256']}`",
            f"- Raw action RMSE: `{float(metrics['raw_action_rmse']):.9f}`",
            "- Raw maximum absolute difference: "
            f"`{float(metrics['raw_action_max_abs_difference']):.9f}`",
            "- Guarded action RMSE: "
            f"`{float(metrics['guarded_action_rmse']):.9f}`",
            "- Left/right guard interventions: "
            f"`{left['guard_intervention_steps']}/"
            f"{right['guard_intervention_steps']}`",
            "- Left/right client latency ms: "
            f"`{float(left['client_inference_latency_ms']):.3f}/"
            f"{float(right['client_inference_latency_ms']):.3f}`",
            "- Physics executed: `false`",
            "- Checkpoint identity verified by protocol: `false`",
            "",
            "Both artifacts passed independent validation and contain the same "
            "serialized input payload. The labels are user-supplied; preserve "
            "server launch logs to establish checkpoint provenance.",
            "",
            "This is a paired inference-output comparison, not a task rollout, "
            "physical-safety result, or statistical model-quality estimate.",
            "",
        ]
    )


def execute_recorded_probe_comparison(
    left_directory: Path,
    right_directory: Path,
    output_directory: Path,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> Path:
    """Compare two validated responses to one byte-identical OpenPI request."""

    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists: {output_directory}"
        )
    left_label = left_label.strip()
    right_label = right_label.strip()
    if not left_label or not right_label:
        raise ValueError("comparison labels must be nonempty")
    left = _load_probe(left_directory)
    right = _load_probe(right_directory)
    if (
        left.validation.request_payload_sha256
        != right.validation.request_payload_sha256
    ):
        raise ValueError(
            "probe requests differ; paired comparison requires the same "
            "serialized input payload SHA-256"
        )

    left_metrics = _probe_metrics(left, left_label)
    right_metrics = _probe_metrics(right, right_label)
    metrics, per_step, raw_step_l2, guarded_step_l2 = _difference_metrics(
        left, right
    )
    comparison: dict[str, object] = {
        "artifact_type": PROBE_COMPARISON_ARTIFACT_TYPE,
        "comparison_validated": True,
        "same_request_payload": True,
        "request_payload_sha256": left.validation.request_payload_sha256,
        "labels_user_supplied": True,
        "left": left_metrics,
        "right": right_metrics,
        "metrics": metrics,
        "physics_executed": False,
        "physical_safe": None,
        "checkpoint_identity_verified": False,
    }

    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json(output_directory / "comparison.json", comparison)
    _write_csv(output_directory / "per_step.csv", per_step)
    dimension_metrics = metrics["per_dimension"]
    if not isinstance(dimension_metrics, list):
        raise RuntimeError("internal per-dimension metrics are invalid")
    _write_plot(
        output_directory / "comparison.png",
        left_label,
        right_label,
        raw_step_l2,
        guarded_step_l2,
        dimension_metrics,
    )
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["recorded_probe_comparison"] = {
        "artifact_type": PROBE_COMPARISON_ARTIFACT_TYPE,
        "request_payload_sha256": left.validation.request_payload_sha256,
        "left_action_sha256": left.validation.action_sha256,
        "right_action_sha256": right.validation.action_sha256,
        "physics_executed": False,
        "checkpoint_identity_verified": False,
    }
    _write_json(output_directory / "environment.json", metadata)
    (output_directory / "summary.md").write_text(
        _summary(comparison), encoding="utf-8", newline="\n"
    )
    return output_directory
