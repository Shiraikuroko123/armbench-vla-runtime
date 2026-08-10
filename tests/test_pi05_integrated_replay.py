from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from armbench.vla.pi05_archive_replay import (
    Pi05ArchiveReplayError,
    _write_root_manifest,
)
from armbench.vla.pi05_integrated_replay import (
    MODES,
    Pi05IntegratedReplayConfig,
    execute_pi05_integrated_cpu_replay,
    validate_pi05_integrated_cpu_replay,
)
from test_pi05_archive_replay import _source_artifact


@pytest.fixture(scope="module")
def integrated_replay_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("pi05_integrated_replay")
    source = _source_artifact(root)
    output = root / "artifact"
    execute_pi05_integrated_cpu_replay(
        source,
        output,
        Pi05IntegratedReplayConfig(
            chunk_count=6,
            scenarios=("free_space",),
            modes=MODES,
        ),
    )
    return output, source


def _copy_artifact(source: Path, destination: Path) -> Path:
    import shutil

    shutil.copytree(source, destination)
    return destination


def test_integrated_replay_is_paired_and_all_or_none(
    integrated_replay_artifact: tuple[Path, Path],
) -> None:
    artifact, _ = integrated_replay_artifact
    with (artifact / "per_case.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    with np.load(artifact / "trajectories.npz", allow_pickle=False) as trace:
        assert trace["candidate_actions"].shape == (18, 10, 8)
        assert trace["candidate_positions"].shape == (18, 11, 7)
        assert set(trace["published_lengths"].tolist()) <= {0, 10}
        for index, row in enumerate(rows):
            if row["status"] == "execute":
                np.testing.assert_array_equal(
                    trace["published_actions"][index],
                    trace["candidate_actions"][index],
                )
            else:
                assert not np.any(trace["published_actions"][index])

    assert len(rows) == 18
    assert {row["mode"] for row in rows} == set(MODES)
    assert all(row["partial_prefix_exposed"] == "False" for row in rows)
    assert all(row["all_or_none_publication"] == "True" for row in rows)
    assert all(
        (row["integrated_atomic_gate_used"] == "True")
        == (row["mode"] == "full_assurance")
        for row in rows
    )


def test_integrated_replay_validator_rebinds_source(
    integrated_replay_artifact: tuple[Path, Path],
) -> None:
    artifact, source = integrated_replay_artifact

    result = validate_pi05_integrated_cpu_replay(artifact, source)

    assert result["valid"] is True
    assert result["cases"] == 18
    assert result["selected_responses"] == 6
    assert result["source_reverified"] is True
    assert result["protocol_conformant"] is False
    assert "dynamics_braking_predicates_recomputed" in result["checks"]


def test_validator_rejects_resigned_trajectory_identity_tamper(
    tmp_path: Path,
    integrated_replay_artifact: tuple[Path, Path],
) -> None:
    artifact, source = integrated_replay_artifact
    copied = _copy_artifact(artifact, tmp_path / "trajectory-tamper")
    trajectory_path = copied / "trajectories.npz"
    with np.load(trajectory_path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    arrays["source_row_indices"][0] += 1
    np.savez_compressed(trajectory_path, **arrays)
    _write_root_manifest(copied)

    with pytest.raises(Pi05ArchiveReplayError, match="case identity mismatch"):
        validate_pi05_integrated_cpu_replay(copied, source)


def test_validator_rejects_resigned_csv_publication_tamper(
    tmp_path: Path,
    integrated_replay_artifact: tuple[Path, Path],
) -> None:
    artifact, source = integrated_replay_artifact
    copied = _copy_artifact(artifact, tmp_path / "csv-tamper")
    csv_path = copied / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["published_action_count"] = "0"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_root_manifest(copied)

    with pytest.raises(Pi05ArchiveReplayError, match="trajectory lengths"):
        validate_pi05_integrated_cpu_replay(copied, source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_count": 0},
        {"selection_seed": -1},
        {"scenarios": ("unknown",)},
        {"modes": ("unknown",)},
        {"software_budget_ms": float("nan")},
        {"worker_timeout_s": True},
    ],
)
def test_integrated_replay_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Pi05IntegratedReplayConfig(**kwargs)
