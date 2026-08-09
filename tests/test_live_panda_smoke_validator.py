from __future__ import annotations

import json
from pathlib import Path
import shutil

from integrations.openpi.validate_live_panda_smoke import validate_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT_ROOT / "evidence" / "g01_live_panda_smoke_final_001"


def test_saved_live_pi05_panda_bundle_validates() -> None:
    report = validate_bundle(EVIDENCE)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["metrics"]["accepted_responses"] == 35
    assert report["metrics"]["video_frames"] == 94
    assert report["metrics"]["armbench_commit"] == (
        "2f92db28e0bf3b30ad5482bb519377bd4d43b927"
    )


def test_live_pi05_panda_validator_rejects_tampered_summary(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "tampered"
    shutil.copytree(EVIDENCE, artifact)
    summary_path = artifact / "run" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scripted_policy"] = True
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = validate_bundle(artifact)

    assert report["valid"] is False
    assert any("SHA-256 mismatch: summary.json" in error for error in report["errors"])
    assert any("scripted_policy mismatch" in error for error in report["errors"])
