from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
import pytest

from armbench.cli import build_parser
from armbench.vla.optimized_cpu_replay import (
    OptimizedCPUReplayConfig,
    _write_manifest,
    execute_optimized_cpu_replay,
    validate_optimized_cpu_replay,
)


@pytest.fixture(scope="module")
def input_artifact() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "pi05_integrated_panda_cpu_replay_270_001"
    )
    if not path.is_dir():
        pytest.skip("registered optimized replay input artifact is unavailable")
    return path


@pytest.fixture(scope="module")
def optimized_artifact(
    tmp_path_factory: pytest.TempPathFactory, input_artifact: Path
) -> Path:
    return execute_optimized_cpu_replay(
        input_artifact,
        tmp_path_factory.mktemp("optimized-replay") / "artifact",
        OptimizedCPUReplayConfig(chunk_count=1),
    )


def test_optimized_replay_roundtrip_recomputes_candidate_safety(
    optimized_artifact: Path, input_artifact: Path
) -> None:
    result = validate_optimized_cpu_replay(optimized_artifact, input_artifact)

    assert result["valid"] is True
    assert result["cases"] == 6
    assert result["unsafe_plans_published"] == 0
    assert result["partial_prefixes_exposed"] == 0
    assert result["checks"] == [
        "recursive_manifest_and_exact_file_set",
        "frozen_input_manifest_and_response_bindings",
        "implementation_and_protocol_hashes",
        "all_or_none_trajectory_publication",
        "candidate_kinematics_recomputed",
        "continuous_collision_and_braking_audit_recomputed",
        "summary_and_markdown_recomputed",
    ]


def test_resigned_candidate_trajectory_tamper_is_rejected(
    optimized_artifact: Path, input_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "trajectory-tamper"
    shutil.copytree(optimized_artifact, copied)
    trajectory_path = copied / "trajectories.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["candidate_actions"][0, 0, 0] += 0.25
    np.savez_compressed(trajectory_path, **arrays)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="does not integrate"):
        validate_optimized_cpu_replay(copied, input_artifact)


def test_resigned_response_identity_tamper_is_rejected(
    optimized_artifact: Path, input_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "source-tamper"
    shutil.copytree(optimized_artifact, copied)
    csv_path = copied / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["response_action_sha256"] = "0" * 64
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="source binding"):
        validate_optimized_cpu_replay(copied, input_artifact)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_count": 0},
        {"chunk_count": True},
        {"operational_budget_ms": 0.0},
        {"diagnostic_budget_ms": 20.0},
        {"response_deadline_ms": "200"},
        {"qp_step_budget_ms": float("nan")},
        {"worker_timeout_s": False},
        {"poll_interval_s": 0.0},
    ],
)
def test_optimized_replay_configuration_fails_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        OptimizedCPUReplayConfig(**kwargs)


def test_optimized_replay_cli_exposes_run_and_validation_inputs() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "vla-panda-optimized-replay",
            "input-artifact",
            "--output-directory",
            "output-artifact",
        ]
    )
    validate = parser.parse_args(
        [
            "vla-panda-optimized-replay-validate",
            "output-artifact",
            "input-artifact",
        ]
    )

    assert run.input_directory == Path("input-artifact")
    assert run.chunks == 30
    assert run.operational_budget_ms == 20.0
    assert validate.directory == Path("output-artifact")
    assert validate.input_directory == Path("input-artifact")
