from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from armbench.vla.cpu_runtime_completion import (
    CPU_RUNTIME_CSV_FIELDS,
    CPURuntimeMatrixConfig,
    _summary_markdown,
    _write_manifest,
    run_cpu_runtime_completion,
    validate_cpu_runtime_completion,
)
from armbench.vla.serialization import write_json


@pytest.fixture(scope="module")
def cpu_runtime_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_cpu_runtime_completion(
        tmp_path_factory.mktemp("cpu_runtime_completion") / "artifact"
    )


def _rewrite_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CPU_RUNTIME_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_cpu_runtime_artifact_reruns_all_registered_cases(
    cpu_runtime_artifact: Path,
) -> None:
    result = validate_cpu_runtime_completion(cpu_runtime_artifact)

    assert result["valid"] is True
    assert result["cases"] == 17
    assert result["expected_matches"] == 17
    assert result["accepted_plans"] == 6
    assert result["holds"] == 10
    assert result["unrecoverable_stops"] == 1
    assert result["partial_prefix_exposed"] == 0
    assert "implementation_source_hashes" in result["checks"]
    assert "registered_cases_rerun" in result["checks"]


def test_cpu_runtime_rejects_unsigned_file_tampering(
    cpu_runtime_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "unsigned_tampering"
    shutil.copytree(cpu_runtime_artifact, copied)
    summary_path = copied / "summary.json"
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest"):
        validate_cpu_runtime_completion(copied)


def test_cpu_runtime_rejects_resigned_semantic_tampering(
    cpu_runtime_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "resigned_semantic_tampering"
    shutil.copytree(cpu_runtime_artifact, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    forged_reason = "qp_continuous_collision_and_braking_forged"
    summary["rows"][0]["reason"] = forged_reason
    write_json(summary_path, summary)

    csv_path = copied / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["reason"] = forged_reason
    _rewrite_csv(csv_path, rows)
    (copied / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    _write_manifest(copied)

    with pytest.raises(ValueError, match="recomputation"):
        validate_cpu_runtime_completion(copied)


def test_cpu_runtime_rejects_resigned_source_hash_tampering(
    cpu_runtime_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "resigned_source_hash_tampering"
    shutil.copytree(cpu_runtime_artifact, copied)
    provenance_path = copied / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    first_source = provenance["implementation_files"][0]
    provenance["implementation_sha256"][first_source] = "0" * 64
    write_json(provenance_path, provenance)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="implementation hash"):
        validate_cpu_runtime_completion(copied)


def test_cpu_runtime_rejects_resigned_case_definition_tampering(
    cpu_runtime_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "resigned_case_tampering"
    shutil.copytree(cpu_runtime_artifact, copied)
    cases_path = copied / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases[0]["latency_ms"] = 5.0
    write_json(cases_path, cases)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="registered cases"):
        validate_cpu_runtime_completion(copied)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_deadline_ms": 0.0},
        {"supervision_budget_ms": float("nan")},
        {"qp_step_budget_ms": -1.0},
        {"control_period_ms": True},
    ],
)
def test_cpu_runtime_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CPURuntimeMatrixConfig(**kwargs)
