"""Paired comparison of two validated fixed-request OpenPI probes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from armbench.benchmark import environment_metadata
from armbench.vla.benchmark import _write_csv, _write_json
from armbench.vla.replay_probe import (
    RecordedProbeValidationResult,
    validate_recorded_openpi_probe,
)


PROBE_COMPARISON_ARTIFACT_TYPE = "armbench_recorded_openpi_probe_comparison_v1"


class ProbeComparisonValidationError(ValueError):
    """Raised when a paired probe comparison artifact is inconsistent."""


@dataclass(frozen=True)
class ProbeComparisonValidationResult:
    directory: str
    request_payload_sha256: str
    left_action_sha256: str
    right_action_sha256: str
    raw_action_rmse: float
    action_rows: int
    artifact_sha256: str
    checks: tuple[str, ...]

    def metrics(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "request_payload_sha256": self.request_payload_sha256,
            "left_action_sha256": self.left_action_sha256,
            "right_action_sha256": self.right_action_sha256,
            "raw_action_rmse": self.raw_action_rmse,
            "action_rows": self.action_rows,
            "artifact_sha256": self.artifact_sha256,
            "checks": list(self.checks),
            "valid": True,
        }


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProbeComparisonValidationError(f"invalid {label}")
    return value


def _validation_require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeComparisonValidationError(message)


def _validation_mapping(value: object, label: str) -> dict[str, object]:
    _validation_require(isinstance(value, dict), f"{label} must be a mapping")
    return dict(value)


def _validation_json(path: Path) -> dict[str, object]:
    _validation_require(
        path.is_file() and path.stat().st_size > 0, f"missing file: {path}"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeComparisonValidationError(f"invalid JSON: {path}") from error
    return _validation_mapping(value, str(path))


def _validation_array(
    value: np.ndarray, shape: tuple[int, ...], label: str
) -> None:
    _validation_require(value.shape == shape, f"unexpected {label} shape")
    _validation_require(
        np.issubdtype(value.dtype, np.number),
        f"{label} must have a numeric dtype",
    )
    _validation_require(
        bool(np.all(np.isfinite(value))), f"{label} contains nonfinite values"
    )


def _metric_matches(value: object, expected: float, label: str) -> None:
    try:
        actual = _number(value, label)
    except ValueError as error:
        raise ProbeComparisonValidationError(str(error)) from error
    _validation_require(
        bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-12)),
        f"{label} mismatch",
    )


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


def validate_recorded_probe_comparison(
    directory: Path,
) -> ProbeComparisonValidationResult:
    """Recompute a self-contained paired-probe comparison artifact."""

    root = directory.resolve()
    _validation_require(
        root.is_dir(), f"artifact directory does not exist: {root}"
    )
    comparison_path = root / "comparison.json"
    csv_path = root / "per_step.csv"
    arrays_path = root / "paired_responses.npz"
    plot_path = root / "comparison.png"
    environment_path = root / "environment.json"
    summary_path = root / "summary.md"
    comparison = _validation_json(comparison_path)
    environment = _validation_json(environment_path)
    for path in (csv_path, arrays_path, plot_path, summary_path):
        _validation_require(
            path.is_file() and path.stat().st_size > 0,
            f"missing file: {path}",
        )

    _validation_require(
        comparison.get("artifact_type") == PROBE_COMPARISON_ARTIFACT_TYPE,
        "unexpected probe comparison artifact type",
    )
    _validation_require(
        comparison.get("comparison_validated") is True,
        "comparison is not marked validated",
    )
    _validation_require(
        comparison.get("same_request_payload") is True,
        "comparison does not use the same request payload",
    )
    _validation_require(
        comparison.get("labels_user_supplied") is True,
        "comparison label provenance is invalid",
    )
    _validation_require(
        comparison.get("physics_executed") is False,
        "comparison must not claim physics execution",
    )
    _validation_require(
        comparison.get("physical_safe") is None,
        "comparison must not claim a physical-safety outcome",
    )
    _validation_require(
        comparison.get("checkpoint_identity_verified") is False,
        "comparison must not claim checkpoint attestation",
    )
    request_hash = _valid_sha256(
        comparison.get("request_payload_sha256"),
        "comparison request payload SHA-256",
    )
    left = _validation_mapping(comparison.get("left"), "left comparison")
    right = _validation_mapping(comparison.get("right"), "right comparison")
    for side, value in (("left", left), ("right", right)):
        _validation_require(
            isinstance(value.get("label"), str)
            and bool(str(value["label"]).strip()),
            f"{side} label must be nonempty",
        )
        _validation_require(
            value.get("checkpoint_identity_verified") is False,
            f"{side} checkpoint identity must remain unverified",
        )
        _validation_require(
            isinstance(value.get("guard_safe_after"), bool),
            f"{side} guard safety must be boolean",
        )
    left_action_hash = _valid_sha256(
        left.get("action_sha256"), "left action SHA-256"
    )
    right_action_hash = _valid_sha256(
        right.get("action_sha256"), "right action SHA-256"
    )

    expected_array_names = {
        "left_raw_actions",
        "right_raw_actions",
        "left_guarded_actions",
        "right_guarded_actions",
        "left_predicted_positions",
        "right_predicted_positions",
    }
    try:
        with np.load(arrays_path, allow_pickle=False) as trace:
            _validation_require(
                set(trace.files) == expected_array_names,
                "paired_responses.npz has unexpected arrays",
            )
            arrays = {name: np.asarray(trace[name]) for name in trace.files}
    except ProbeComparisonValidationError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise ProbeComparisonValidationError(
            f"cannot load paired response arrays: {arrays_path}"
        ) from error
    for name in (
        "left_raw_actions",
        "right_raw_actions",
        "left_guarded_actions",
        "right_guarded_actions",
    ):
        _validation_array(arrays[name], (15, 8), name.replace("_", " "))
    for name in ("left_predicted_positions", "right_predicted_positions"):
        _validation_array(arrays[name], (16, 7), name.replace("_", " "))
    recomputed_left_hash = hashlib.sha256(
        arrays["left_raw_actions"].tobytes(order="C")
    ).hexdigest()
    recomputed_right_hash = hashlib.sha256(
        arrays["right_raw_actions"].tobytes(order="C")
    ).hexdigest()
    _validation_require(
        left_action_hash == recomputed_left_hash,
        "left raw action SHA-256 mismatch",
    )
    _validation_require(
        right_action_hash == recomputed_right_hash,
        "right raw action SHA-256 mismatch",
    )

    raw_delta = arrays["right_raw_actions"] - arrays["left_raw_actions"]
    guarded_delta = (
        arrays["right_guarded_actions"] - arrays["left_guarded_actions"]
    )
    position_delta = (
        arrays["right_predicted_positions"]
        - arrays["left_predicted_positions"]
    )
    metrics = _validation_mapping(
        comparison.get("metrics"), "comparison metrics"
    )
    expected_metrics = {
        "raw_action_rmse": float(np.sqrt(np.mean(raw_delta**2))),
        "raw_action_mean_abs_difference": float(
            np.mean(np.abs(raw_delta))
        ),
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
    }
    _validation_require(
        metrics.get("raw_actions_identical")
        is (left_action_hash == right_action_hash),
        "raw action identity flag mismatch",
    )
    for name, expected in expected_metrics.items():
        _metric_matches(metrics.get(name), expected, name.replace("_", " "))

    dimensions = metrics.get("per_dimension")
    _validation_require(
        isinstance(dimensions, list) and len(dimensions) == 8,
        "per-dimension metrics must contain eight rows",
    )
    for index, value in enumerate(dimensions):
        row = _validation_mapping(value, f"dimension {index}")
        expected_name = (
            f"joint_velocity_{index + 1}" if index < 7 else "gripper"
        )
        _validation_require(
            row.get("dimension") == index and row.get("name") == expected_name,
            f"dimension {index} identity mismatch",
        )
        _metric_matches(
            row.get("rmse"),
            float(np.sqrt(np.mean(raw_delta[:, index] ** 2))),
            f"dimension {index} RMSE",
        )
        _metric_matches(
            row.get("max_abs_difference"),
            float(np.max(np.abs(raw_delta[:, index]))),
            f"dimension {index} maximum difference",
        )

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            step_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ProbeComparisonValidationError(
            f"invalid per-step CSV: {csv_path}"
        ) from error
    expected_fields = {
        "step",
        "raw_action_l2_difference",
        "raw_velocity_l2_difference",
        "raw_gripper_abs_difference",
        "guarded_action_l2_difference",
        "left_raw_action_l2",
        "right_raw_action_l2",
    }
    _validation_require(len(step_rows) == 15, "per-step CSV row count mismatch")
    _validation_require(
        bool(step_rows) and set(step_rows[0]) == expected_fields,
        "per-step CSV fields mismatch",
    )
    for index, row in enumerate(step_rows):
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProbeComparisonValidationError(
                f"invalid per-step index at row {index}"
            ) from error
        _validation_require(step == index, f"per-step index mismatch at row {index}")
        expected_step = {
            "raw_action_l2_difference": float(
                np.linalg.norm(raw_delta[index])
            ),
            "raw_velocity_l2_difference": float(
                np.linalg.norm(raw_delta[index, :7])
            ),
            "raw_gripper_abs_difference": float(abs(raw_delta[index, 7])),
            "guarded_action_l2_difference": float(
                np.linalg.norm(guarded_delta[index])
            ),
            "left_raw_action_l2": float(
                np.linalg.norm(arrays["left_raw_actions"][index])
            ),
            "right_raw_action_l2": float(
                np.linalg.norm(arrays["right_raw_actions"][index])
            ),
        }
        for name, expected in expected_step.items():
            try:
                csv_value = float(row[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ProbeComparisonValidationError(
                    f"invalid per-step {index} {name}"
                ) from error
            _metric_matches(
                csv_value, expected, f"per-step {index} {name}"
            )

    probe_environment = _validation_mapping(
        environment.get("recorded_probe_comparison"),
        "comparison environment metadata",
    )
    expected_environment: dict[str, object] = {
        "artifact_type": PROBE_COMPARISON_ARTIFACT_TYPE,
        "request_payload_sha256": request_hash,
        "left_action_sha256": left_action_hash,
        "right_action_sha256": right_action_hash,
        "comparison_json_sha256": _file_sha256(comparison_path),
        "per_step_csv_sha256": _file_sha256(csv_path),
        "paired_responses_sha256": _file_sha256(arrays_path),
        "comparison_png_sha256": _file_sha256(plot_path),
        "physics_executed": False,
        "checkpoint_identity_verified": False,
    }
    for name, expected in expected_environment.items():
        _validation_require(
            probe_environment.get(name) == expected,
            f"comparison environment mismatch: {name}",
        )

    try:
        plot = imageio.imread(plot_path)
    except Exception as error:
        raise ProbeComparisonValidationError(
            f"comparison plot cannot be decoded: {plot_path}"
        ) from error
    _validation_require(
        plot.ndim == 3 and float(plot.std()) > 1.0,
        "comparison plot is blank or malformed",
    )
    summary = summary_path.read_text(encoding="utf-8")
    _validation_require(request_hash in summary, "summary request hash mismatch")
    _validation_require(
        "Physics executed: `false`" in summary,
        "summary physics claim mismatch",
    )
    _validation_require(
        "Checkpoint identity verified by protocol: `false`" in summary,
        "summary checkpoint claim mismatch",
    )
    aggregate = hashlib.sha256()
    for path in (
        comparison_path,
        csv_path,
        arrays_path,
        plot_path,
        environment_path,
        summary_path,
    ):
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(path.read_bytes())
    return ProbeComparisonValidationResult(
        directory=str(root),
        request_payload_sha256=request_hash,
        left_action_sha256=left_action_hash,
        right_action_sha256=right_action_hash,
        raw_action_rmse=expected_metrics["raw_action_rmse"],
        action_rows=len(step_rows),
        artifact_sha256=aggregate.hexdigest(),
        checks=(
            "claim_boundaries",
            "paired_response_arrays",
            "action_hashes",
            "aggregate_metrics",
            "per_dimension_metrics",
            "per_step_metrics",
            "environment_hashes",
            "plot_decode",
            "summary_claims",
        ),
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
    np.savez_compressed(
        output_directory / "paired_responses.npz",
        left_raw_actions=left.raw_actions,
        right_raw_actions=right.raw_actions,
        left_guarded_actions=left.guarded_actions,
        right_guarded_actions=right.guarded_actions,
        left_predicted_positions=left.predicted_positions,
        right_predicted_positions=right.predicted_positions,
    )
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
        "comparison_json_sha256": _file_sha256(
            output_directory / "comparison.json"
        ),
        "per_step_csv_sha256": _file_sha256(
            output_directory / "per_step.csv"
        ),
        "paired_responses_sha256": _file_sha256(
            output_directory / "paired_responses.npz"
        ),
        "comparison_png_sha256": _file_sha256(
            output_directory / "comparison.png"
        ),
        "physics_executed": False,
        "checkpoint_identity_verified": False,
    }
    _write_json(output_directory / "environment.json", metadata)
    (output_directory / "summary.md").write_text(
        _summary(comparison), encoding="utf-8", newline="\n"
    )
    return output_directory
