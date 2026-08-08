from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import mujoco
import numpy as np

from armbench.mujoco_sim.dynamics_braking_audit import (
    DynamicsBrakingAuditConfig,
    _write_manifest,
    run_dynamics_braking_audit,
    validate_dynamics_braking_audit,
)
from armbench.mujoco_sim.model import MENAGERIE_COMMIT


def _small_config() -> DynamicsBrakingAuditConfig:
    return DynamicsBrakingAuditConfig(
        payload_masses_kg=(0.0, 0.5),
        joint_damping_scales=(1.0,),
        velocity_profiles=("stationary", "low_forward"),
    )


def test_dynamics_audit_roundtrip_reruns_registered_matrix(tmp_path: Path) -> None:
    output = run_dynamics_braking_audit(tmp_path / "audit", _small_config())

    result = validate_dynamics_braking_audit(output)

    assert result["valid"] is True
    assert result["cases"] == 4
    assert result["validated_stops"] == 4
    assert result["fail_closed_rejections"] == 0
    assert result["checks"] == [
        "recursive_manifest",
        "all_cases_rerun_with_mujoco_inverse_dynamics",
        "continuous_collision_edges_rerun",
        "payload_and_damping_aggregates_recomputed",
    ]
    summary = __import__("json").loads(
        (output / "summary.json").read_text("utf-8")
    )
    assert summary["environment"] == {
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "menagerie_commit": MENAGERIE_COMMIT,
    }
    assert (output / "summary.md").read_text("utf-8").startswith(
        "# Panda dynamics braking audit"
    )


def test_dynamics_audit_rejects_tampered_case_even_with_original_manifest(
    tmp_path: Path,
) -> None:
    output = run_dynamics_braking_audit(tmp_path / "audit", _small_config())
    csv_path = output / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["validated"] = "False"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_dynamics_braking_audit(output)


def test_resigned_string_velocity_vector_is_rejected(tmp_path: Path) -> None:
    output = run_dynamics_braking_audit(tmp_path / "audit", _small_config())
    csv_path = output / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["initial_velocity_rad_s"] = json.dumps(["0"] * 7)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(output)

    with pytest.raises(ValueError, match="velocity vector"):
        validate_dynamics_braking_audit(output)


def test_resigned_claim_boundary_tamper_is_rejected(tmp_path: Path) -> None:
    output = run_dynamics_braking_audit(tmp_path / "audit", _small_config())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["claim_boundary"].pop()
    summary_path.write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    _write_manifest(output)

    with pytest.raises(ValueError, match="summary"):
        validate_dynamics_braking_audit(output)


def test_manifest_unknown_field_is_rejected(tmp_path: Path) -> None:
    output = run_dynamics_braking_audit(tmp_path / "audit", _small_config())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["trusted"] = True
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest"):
        validate_dynamics_braking_audit(output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload_masses_kg": ()},
        {"payload_masses_kg": (-0.1,)},
        {"payload_masses_kg": ("0.5",)},
        {"joint_damping_scales": (0.0,)},
        {"joint_damping_scales": (True,)},
        {"velocity_profiles": ("unknown",)},
        {"sample_dt_s": "0.01"},
        {"max_stop_time_s": False},
        {"actuator_force_limit_scale": 1.1},
    ],
)
def test_dynamics_audit_configuration_fails_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DynamicsBrakingAuditConfig(**kwargs)
