from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from armbench.vla.probe_comparison import (
    ProbeComparisonValidationError,
    execute_recorded_probe_comparison,
    validate_recorded_probe_comparison,
)
from armbench.vla.replay_probe import validate_recorded_openpi_probe


def _write_probe(
    directory: Path,
    raw_actions: np.ndarray,
    *,
    request_hash: str,
    latency_ms: float,
    interventions: int,
) -> None:
    guarded_actions = raw_actions * 0.5
    predicted_positions = np.zeros((16, 7), dtype=float)
    predicted_positions[1:] = np.cumsum(guarded_actions[:, :7], axis=0)
    action_hash = hashlib.sha256(
        raw_actions.tobytes(order="C")
    ).hexdigest()
    response = {
        "artifact_type": "armbench_recorded_openpi_probe_v1",
        "source_request": {
            "packed_payload_sha256": request_hash,
            "server_payload_sha256": request_hash,
            "server_payload_matches": True,
            "replayable": True,
        },
        "server": "127.0.0.1:8000",
        "policy_provenance": "test_server_user_asserted",
        "remote_policy_response_validated": True,
        "checkpoint_identity_verified": False,
        "action_shape": [15, 8],
        "action_dtype": str(raw_actions.dtype),
        "action_sha256": action_hash,
        "action_min": float(np.min(raw_actions)),
        "action_max": float(np.max(raw_actions)),
        "client_inference_latency_ms": latency_ms,
        "guard_safe_after": True,
        "guard": {
            "safe_after_guard": True,
            "intervention_steps": interventions,
            "hold_steps": 0,
        },
        "physics_executed": False,
        "physical_safe": None,
    }
    environment = {
        "recorded_openpi_probe": {
            "remote_policy_response_validated": True,
            "checkpoint_identity_verified": False,
            "policy_provenance": "test_server_user_asserted",
            "server": "127.0.0.1:8000",
            "source_request_payload_sha256": request_hash,
            "response_action_sha256": action_hash,
            "physics_executed": False,
        }
    }
    directory.mkdir()
    (directory / "response.json").write_text(
        json.dumps(response), encoding="utf-8"
    )
    (directory / "environment.json").write_text(
        json.dumps(environment), encoding="utf-8"
    )
    np.savez_compressed(
        directory / "response.npz",
        raw_actions=raw_actions,
        guarded_actions=guarded_actions,
        predicted_positions=predicted_positions,
    )
    (directory / "summary.md").write_text(
        "\n".join(
            [
                "# Test probe",
                f"Request: `{request_hash}`",
                f"Action: `{action_hash}`",
                "Checkpoint identity verified by protocol: `false`",
                "Physics executed: `false`",
            ]
        ),
        encoding="utf-8",
    )


def test_compare_validated_probes_with_same_request(tmp_path: Path) -> None:
    request_hash = "a" * 64
    left_actions = np.zeros((15, 8), dtype=np.float64)
    right_actions = np.full((15, 8), 0.1, dtype=np.float64)
    left = tmp_path / "left_probe"
    right = tmp_path / "right_probe"
    _write_probe(
        left,
        left_actions,
        request_hash=request_hash,
        latency_ms=12.0,
        interventions=0,
    )
    _write_probe(
        right,
        right_actions,
        request_hash=request_hash,
        latency_ms=18.0,
        interventions=15,
    )
    validate_recorded_openpi_probe(left)
    validate_recorded_openpi_probe(right)

    output = tmp_path / "comparison"
    execute_recorded_probe_comparison(
        left,
        right,
        output,
        left_label="pi0 server",
        right_label="pi0.5 server",
    )

    comparison = json.loads(
        (output / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["same_request_payload"] is True
    assert comparison["request_payload_sha256"] == request_hash
    assert comparison["metrics"]["raw_action_rmse"] == pytest.approx(0.1)
    assert comparison["metrics"]["guarded_action_rmse"] == pytest.approx(0.05)
    assert comparison["metrics"]["raw_actions_identical"] is False
    assert comparison["left"]["guard_intervention_steps"] == 0
    assert comparison["right"]["guard_intervention_steps"] == 15
    assert comparison["physics_executed"] is False
    assert comparison["physical_safe"] is None
    assert comparison["checkpoint_identity_verified"] is False
    with (output / "per_step.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert float(rows[0]["raw_action_l2_difference"]) == pytest.approx(
        np.sqrt(8.0) * 0.1
    )
    plot = imageio.imread(output / "comparison.png")
    assert plot.ndim == 3
    assert float(plot.std()) > 1.0
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "labels are user-supplied" in summary
    assert "Physics executed: `false`" in summary
    validation = validate_recorded_probe_comparison(output)
    assert validation.request_payload_sha256 == request_hash
    assert validation.raw_action_rmse == pytest.approx(0.1)
    assert validation.action_rows == 15
    assert len(validation.artifact_sha256) == 64


def test_compare_rejects_different_request_payloads(tmp_path: Path) -> None:
    actions = np.zeros((15, 8), dtype=np.float64)
    left = tmp_path / "left_probe"
    right = tmp_path / "right_probe"
    _write_probe(
        left,
        actions,
        request_hash="a" * 64,
        latency_ms=12.0,
        interventions=0,
    )
    _write_probe(
        right,
        actions,
        request_hash="b" * 64,
        latency_ms=12.0,
        interventions=0,
    )
    output = tmp_path / "comparison"

    with pytest.raises(ValueError, match="probe requests differ"):
        execute_recorded_probe_comparison(left, right, output)
    assert not output.exists()


def test_comparison_validator_rejects_tampered_actions(
    tmp_path: Path,
) -> None:
    request_hash = "c" * 64
    left = tmp_path / "left_probe"
    right = tmp_path / "right_probe"
    _write_probe(
        left,
        np.zeros((15, 8), dtype=np.float64),
        request_hash=request_hash,
        latency_ms=10.0,
        interventions=0,
    )
    _write_probe(
        right,
        np.full((15, 8), 0.2, dtype=np.float64),
        request_hash=request_hash,
        latency_ms=11.0,
        interventions=0,
    )
    output = tmp_path / "comparison"
    execute_recorded_probe_comparison(left, right, output)
    arrays_path = output / "paired_responses.npz"
    with np.load(arrays_path, allow_pickle=False) as trace:
        arrays = {name: trace[name] for name in trace.files}
    arrays["left_raw_actions"] = arrays["left_raw_actions"].copy()
    arrays["left_raw_actions"][0, 0] = 0.25
    np.savez_compressed(arrays_path, **arrays)

    with pytest.raises(
        ProbeComparisonValidationError,
        match="left raw action SHA-256 mismatch",
    ):
        validate_recorded_probe_comparison(output)
