import csv
import json
from pathlib import Path
import shutil

import pytest

from armbench.mujoco_sim.swept_audit import (
    SweptAuditConfig,
    _write_manifest,
    run_swept_collision_audit,
    validate_swept_collision_audit,
)


def test_swept_audit_writes_no_false_safe_artifact(tmp_path: Path) -> None:
    output = run_swept_collision_audit(
        tmp_path / "audit",
        config=SweptAuditConfig(
            scenarios=("free_space", "single_block"),
            samples_per_scenario=2,
            seed=20260808,
        ),
    )

    result = validate_swept_collision_audit(output)
    summary = json.loads((output / "summary.json").read_text("utf-8"))
    assert result["valid"] is True
    assert result["edges"] == 4
    assert result["false_safe"] == 0
    assert "edge_endpoints_and_decisions_recomputed" in result["checks"]
    assert summary["overall"]["conservative_rejections"] >= 0


def test_swept_audit_rejects_manifest_tampering(tmp_path: Path) -> None:
    output = run_swept_collision_audit(
        tmp_path / "audit",
        config=SweptAuditConfig(
            scenarios=("free_space",),
            samples_per_scenario=1,
            seed=20260808,
        ),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][0]["bytes"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest|inventory"):
        validate_swept_collision_audit(output)


def test_swept_audit_rejects_resigned_decision_tamper(
    tmp_path: Path,
) -> None:
    source = run_swept_collision_audit(
        tmp_path / "source",
        config=SweptAuditConfig(
            scenarios=("free_space",),
            samples_per_scenario=1,
            seed=20260808,
        ),
    )
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    csv_path = copied / "per_edge.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    rows[0]["swept_valid"] = (
        "False" if rows[0]["swept_valid"] == "True" else "True"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="edge recomputation failed"):
        validate_swept_collision_audit(copied)
