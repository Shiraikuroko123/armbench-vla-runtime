import json
from pathlib import Path

from armbench.mujoco_sim.swept_audit import (
    SweptAuditConfig,
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

    try:
        validate_swept_collision_audit(output)
    except ValueError as error:
        assert "manifest" in str(error) or "inventory" in str(error)
    else:
        raise AssertionError("tampered swept audit unexpectedly validated")
