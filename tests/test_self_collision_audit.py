from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from armbench.mujoco_sim.self_collision_audit import (
    SelfCollisionAuditConfig,
    _write_manifest,
    run_self_collision_audit,
    validate_self_collision_audit,
)


def _config() -> SelfCollisionAuditConfig:
    return SelfCollisionAuditConfig(
        strata=("known_intermediate", "local"),
        samples_per_stratum=1,
        dense_resolution_rad=0.01,
    )


def test_self_collision_audit_recomputes_registered_edges(tmp_path: Path) -> None:
    output = run_self_collision_audit(tmp_path / "audit", config=_config())

    result = validate_self_collision_audit(output)
    assert result["valid"] is True
    assert result["edges"] == 2
    assert result["false_safe"] == 0
    assert "edge_endpoints_and_decisions_recomputed" in result["checks"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["by_stratum"]["known_intermediate"]["conservative_rejections"] >= 0


def test_self_collision_audit_rejects_manifest_tampering(tmp_path: Path) -> None:
    output = run_self_collision_audit(tmp_path / "audit", config=_config())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["bytes"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest|inventory"):
        validate_self_collision_audit(output)


def test_self_collision_audit_rejects_resigned_decision_tamper(
    tmp_path: Path,
) -> None:
    source = run_self_collision_audit(tmp_path / "source", config=_config())
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    csv_path = copied / "per_edge.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    rows[0]["continuous_valid"] = (
        "False" if rows[0]["continuous_valid"] == "True" else "True"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="value mismatch|edge"):
        validate_self_collision_audit(copied)
