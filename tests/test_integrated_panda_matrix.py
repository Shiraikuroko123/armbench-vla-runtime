from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from armbench.vla.integrated_panda_matrix import (
    IntegratedPandaMatrixConfig,
    _write_manifest,
    run_integrated_panda_fault_matrix,
    validate_integrated_panda_fault_matrix,
)


@pytest.fixture(scope="module")
def matrix_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = IntegratedPandaMatrixConfig(
        scenarios=("free_space",),
        payload_masses_kg=(0.0,),
        include_special_cases=False,
    )
    return run_integrated_panda_fault_matrix(
        tmp_path_factory.mktemp("integrated_matrix") / "artifact",
        config,
    )


def test_integrated_matrix_reruns_registered_cases(
    matrix_artifact: Path,
) -> None:
    result = validate_integrated_panda_fault_matrix(matrix_artifact)

    assert result["valid"] is True
    assert result["cases"] == 4
    assert result["expected_matches"] == 4
    assert result["accepted_plans"] == 2
    assert result["verified_brakes"] == 1
    assert result["holds"] == 1
    assert result["unrecoverable_stops"] == 0
    assert "all_supervisor_decisions_rerun" in result["checks"]


def test_integrated_matrix_rejects_manifest_tampering(
    matrix_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "tampered_manifest"
    shutil.copytree(matrix_artifact, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["size_bytes"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest|inventory"):
        validate_integrated_panda_fault_matrix(copied)


def test_integrated_matrix_rejects_resigned_decision_tampering(
    matrix_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "resigned_decision"
    shutil.copytree(matrix_artifact, copied)
    csv_path = copied / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    rows[0]["status"] = "hold"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="recomputation"):
        validate_integrated_panda_fault_matrix(copied)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenarios": ("missing",)},
        {"payload_masses_kg": (-0.1,)},
        {"faults": ("nominal",)},
        {"nominal_horizon": 0},
        {"include_special_cases": 1},
    ],
)
def test_integrated_matrix_rejects_invalid_config(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IntegratedPandaMatrixConfig(**kwargs)
