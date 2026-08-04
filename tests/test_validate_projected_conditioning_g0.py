from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from integrations.openpi.validate_projected_conditioning_g0 import (
    G0ValidationError,
    validate_directory,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT_ROOT / "evidence/pi05_projected_conditioning_g0_001"


def test_committed_projected_conditioning_g0_is_valid() -> None:
    report = validate_directory(EVIDENCE, project_root=PROJECT_ROOT)

    assert report["passed"] is True
    assert report["gates"]["max_model_residual"] == 0.0


def test_report_mutation_fails_closed(tmp_path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    report_path = copied / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gates"]["max_model_residual"] = 0.1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(G0ValidationError, match="byte count|SHA-256"):
        validate_directory(copied, project_root=PROJECT_ROOT)
