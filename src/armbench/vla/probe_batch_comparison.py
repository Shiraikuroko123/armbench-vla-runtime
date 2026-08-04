"""Batch paired comparison for cohorts of recorded OpenPI probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from armbench.benchmark import environment_metadata
from armbench.vla.benchmark import _write_csv, _write_json
from armbench.vla.probe_comparison import (
    execute_recorded_probe_comparison,
    validate_recorded_probe_comparison,
)
from armbench.vla.replay_probe import validate_recorded_openpi_probe


PROBE_BATCH_COMPARISON_ARTIFACT_TYPE = (
    "armbench_recorded_openpi_probe_batch_comparison_v1"
)
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_RESAMPLES = 10_000


def _json_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return dict(value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _probe_directories(root: Path) -> list[Path]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"probe root does not exist: {resolved}")
    if (resolved / "response.json").is_file():
        return [resolved]
    return sorted(
        {path.parent for path in resolved.rglob("response.json")},
        key=lambda path: str(path).lower(),
    )


def _probe_index(root: Path, side: str) -> dict[str, Path]:
    directories = _probe_directories(root)
    if not directories:
        raise ValueError(f"{side} probe root contains no probe artifacts")
    result: dict[str, Path] = {}
    for directory in directories:
        validation = validate_recorded_openpi_probe(directory)
        request_hash = validation.request_payload_sha256
        if request_hash in result:
            raise ValueError(
                f"{side} probe root contains duplicate request payload "
                f"SHA-256: {request_hash}"
            )
        result[request_hash] = directory
    return result


def _bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size)
    )
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    lower, upper = _bootstrap_mean_interval(array)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "bootstrap_mean_ci95_lower": lower,
        "bootstrap_mean_ci95_upper": upper,
    }


def _write_plot(
    path: Path,
    rows: list[dict[str, object]],
    left_label: str,
    right_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(rows), dtype=float)
    width = 0.36
    short_labels = [str(row["request_payload_sha256"])[:8] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    axes[0].bar(
        positions - width / 2,
        [float(row["raw_action_rmse"]) for row in rows],
        width,
        label="Raw action",
        color="#b84a3a",
    )
    axes[0].bar(
        positions + width / 2,
        [float(row["guarded_action_rmse"]) for row in rows],
        width,
        label="After guard",
        color="#267a62",
    )
    axes[0].set_xticks(positions, short_labels, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Paired action differences")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        positions,
        [float(row["left_latency_ms"]) for row in rows],
        marker="o",
        label=left_label,
        color="#3677a8",
    )
    axes[1].plot(
        positions,
        [float(row["right_latency_ms"]) for row in rows],
        marker="s",
        label=right_label,
        color="#d29b31",
    )
    axes[1].set_xticks(positions, short_labels, rotation=35, ha="right")
    axes[1].set_ylabel("Client inference latency (ms)")
    axes[1].set_title("Observed paired latency")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    figure.suptitle(
        f"Recorded OpenPI probe cohort: {left_label} vs {right_label}"
    )
    figure.text(
        0.5,
        0.012,
        "Exact request-hash pairing | descriptive output differences, not "
        "task performance",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary(batch: dict[str, object]) -> str:
    aggregate = _mapping(batch["aggregate"], "batch aggregate")
    raw = _mapping(aggregate["raw_action_rmse"], "raw RMSE statistics")
    return "\n".join(
        [
            "# Recorded OpenPI probe batch comparison",
            "",
            f"- Left label: `{batch['left_label']}`",
            f"- Right label: `{batch['right_label']}`",
            f"- Exact matched requests: `{batch['pair_count']}`",
            f"- Mean raw action RMSE: `{float(raw['mean']):.9f}`",
            f"- Median raw action RMSE: `{float(raw['median']):.9f}`",
            f"- P95 raw action RMSE: `{float(raw['p95']):.9f}`",
            "- Bootstrap mean 95% interval: "
            f"`[{float(raw['bootstrap_mean_ci95_lower']):.9f}, "
            f"{float(raw['bootstrap_mean_ci95_upper']):.9f}]`",
            f"- Bootstrap seed/resamples: `{BOOTSTRAP_SEED}/{BOOTSTRAP_RESAMPLES}`",
            "- Physics executed: `false`",
            "- Checkpoint identity verified by protocol: `false`",
            "",
            "Every pair passed independent probe validation, used the same "
            "serialized request SHA, and produced a validated child comparison.",
            "",
            "The interval summarizes this fixed request cohort. Requests may be "
            "correlated, labels are user-supplied, and output difference is not "
            "task success, model quality, or a physical-safety result.",
            "",
        ]
    )


def execute_recorded_probe_batch_comparison(
    left_root: Path,
    right_root: Path,
    output_directory: Path,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> Path:
    """Pair validated probe cohorts by exact request SHA and summarize them."""

    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists: {output_directory}"
        )
    left_label = left_label.strip()
    right_label = right_label.strip()
    if not left_label or not right_label:
        raise ValueError("batch comparison labels must be nonempty")
    left_index = _probe_index(left_root, "left")
    right_index = _probe_index(right_root, "right")
    left_hashes = set(left_index)
    right_hashes = set(right_index)
    if left_hashes != right_hashes:
        missing_left = sorted(right_hashes - left_hashes)
        missing_right = sorted(left_hashes - right_hashes)
        raise ValueError(
            "probe cohorts do not contain the same request payloads; "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )

    output_directory.mkdir(parents=True, exist_ok=False)
    pair_root = output_directory / "pairs"
    pair_root.mkdir()
    rows: list[dict[str, object]] = []
    child_hashes: list[str] = []
    for index, request_hash in enumerate(sorted(left_hashes)):
        child_name = f"pair_{index:04d}_{request_hash[:12]}"
        child_directory = pair_root / child_name
        execute_recorded_probe_comparison(
            left_index[request_hash],
            right_index[request_hash],
            child_directory,
            left_label=left_label,
            right_label=right_label,
        )
        validation = validate_recorded_probe_comparison(child_directory)
        child_hashes.append(validation.artifact_sha256)
        comparison = _json_mapping(child_directory / "comparison.json")
        left = _mapping(comparison["left"], "left pair")
        right = _mapping(comparison["right"], "right pair")
        metrics = _mapping(comparison["metrics"], "pair metrics")
        rows.append(
            {
                "pair_index": index,
                "request_payload_sha256": request_hash,
                "left_artifact": str(left_index[request_hash]),
                "right_artifact": str(right_index[request_hash]),
                "child_comparison": f"pairs/{child_name}",
                "child_artifact_sha256": validation.artifact_sha256,
                "left_action_sha256": str(left["action_sha256"]),
                "right_action_sha256": str(right["action_sha256"]),
                "raw_actions_identical": bool(
                    metrics["raw_actions_identical"]
                ),
                "raw_action_rmse": float(metrics["raw_action_rmse"]),
                "raw_action_max_abs_difference": float(
                    metrics["raw_action_max_abs_difference"]
                ),
                "guarded_action_rmse": float(
                    metrics["guarded_action_rmse"]
                ),
                "predicted_position_rmse_rad": float(
                    metrics["predicted_position_rmse_rad"]
                ),
                "left_latency_ms": float(
                    left["client_inference_latency_ms"]
                ),
                "right_latency_ms": float(
                    right["client_inference_latency_ms"]
                ),
                "latency_delta_ms": float(
                    right["client_inference_latency_ms"]
                )
                - float(left["client_inference_latency_ms"]),
                "left_guard_intervention_steps": int(
                    left["guard_intervention_steps"]
                ),
                "right_guard_intervention_steps": int(
                    right["guard_intervention_steps"]
                ),
            }
        )

    raw_statistics = _statistics(
        [float(row["raw_action_rmse"]) for row in rows]
    )
    guarded_statistics = _statistics(
        [float(row["guarded_action_rmse"]) for row in rows]
    )
    latency_differences = [float(row["latency_delta_ms"]) for row in rows]
    batch: dict[str, object] = {
        "artifact_type": PROBE_BATCH_COMPARISON_ARTIFACT_TYPE,
        "batch_validated": True,
        "left_root": str(left_root.resolve()),
        "right_root": str(right_root.resolve()),
        "left_label": left_label,
        "right_label": right_label,
        "labels_user_supplied": True,
        "pair_count": len(rows),
        "request_payload_sha256": sorted(left_hashes),
        "aggregate": {
            "raw_action_rmse": raw_statistics,
            "guarded_action_rmse": guarded_statistics,
            "identical_action_pairs": sum(
                bool(row["raw_actions_identical"]) for row in rows
            ),
            "mean_latency_delta_ms": float(np.mean(latency_differences)),
            "left_total_guard_intervention_steps": sum(
                int(row["left_guard_intervention_steps"]) for row in rows
            ),
            "right_total_guard_intervention_steps": sum(
                int(row["right_guard_intervention_steps"]) for row in rows
            ),
        },
        "bootstrap": {
            "method": "percentile paired-request bootstrap of the mean",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        },
        "child_artifact_sha256": child_hashes,
        "physics_executed": False,
        "physical_safe": None,
        "checkpoint_identity_verified": False,
    }
    _write_json(output_directory / "batch.json", batch)
    _write_csv(output_directory / "per_pair.csv", rows)
    _write_plot(
        output_directory / "overview.png", rows, left_label, right_label
    )
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["recorded_probe_batch_comparison"] = {
        "artifact_type": PROBE_BATCH_COMPARISON_ARTIFACT_TYPE,
        "pair_count": len(rows),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "batch_json_sha256": hashlib.sha256(
            (output_directory / "batch.json").read_bytes()
        ).hexdigest(),
        "per_pair_csv_sha256": hashlib.sha256(
            (output_directory / "per_pair.csv").read_bytes()
        ).hexdigest(),
        "physics_executed": False,
        "checkpoint_identity_verified": False,
    }
    _write_json(output_directory / "environment.json", metadata)
    (output_directory / "summary.md").write_text(
        _summary(batch), encoding="utf-8", newline="\n"
    )
    return output_directory
