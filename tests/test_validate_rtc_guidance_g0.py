from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pytest

from integrations.openpi.validate_rtc_guidance_g0 import G0ValidationError
from integrations.openpi.validate_rtc_guidance_g0 import validate_artifact


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/pi05_rtc_guidance_g0_001"


def test_committed_rtc_guidance_g0_evidence_validates() -> None:
    result = validate_artifact(EVIDENCE, repository_root=ROOT)

    assert result["valid"] is True
    assert result["weighted_model_residual_ratio"] < 1.0
    assert result["warm_wall_p95_ms"] < 200.0


def test_validator_rejects_report_hash_mismatch(tmp_path: pathlib.Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    report = copied / "report.json"
    report.write_bytes(report.read_bytes() + b"\n")

    with pytest.raises(G0ValidationError, match="byte count mismatch"):
        validate_artifact(copied, repository_root=ROOT)


def test_validator_recomputes_residual_claim(tmp_path: pathlib.Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    report_path = copied / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gates"]["guided_weighted_model_rmse"] = report["gates"][
        "baseline_weighted_model_rmse"
    ]
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"]["bytes"] = report_path.stat().st_size
    manifest["report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(G0ValidationError, match="did not improve"):
        validate_artifact(copied, repository_root=ROOT)
