from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from armbench.cli import build_parser
from armbench.vla.windowed_cpu_replay import (
    WindowedCPUReplayConfig,
    _write_manifest,
    execute_windowed_cpu_replay,
    validate_windowed_cpu_replay,
)


@pytest.fixture(scope="module")
def input_artifact() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "pi05_integrated_panda_cpu_replay_270_001"
    )
    if not path.is_dir():
        pytest.skip("registered windowed replay input artifact is unavailable")
    return path


@pytest.fixture(scope="module")
def windowed_artifact(
    tmp_path_factory: pytest.TempPathFactory, input_artifact: Path
) -> Path:
    return execute_windowed_cpu_replay(
        input_artifact,
        tmp_path_factory.mktemp("windowed-replay") / "artifact",
        WindowedCPUReplayConfig(chunk_count=1),
    )


def test_windowed_replay_roundtrip_recomputes_window_safety(
    windowed_artifact: Path, input_artifact: Path
) -> None:
    result = validate_windowed_cpu_replay(windowed_artifact, input_artifact)

    assert result["valid"] is True
    assert result["cases"] == 6
    assert result["unsafe_windows_published"] == 0
    assert result["partial_windows_exposed"] == 0
    assert result["checks"] == [
        "recursive_manifest_and_exact_file_set",
        "frozen_input_manifest_and_response_bindings",
        "implementation_and_protocol_hashes",
        "window_level_all_or_none_publication",
        "candidate_kinematics_recomputed",
        "continuous_collision_and_braking_audit_recomputed",
        "source_chunk_prefix_semantics_checked",
        "summary_and_markdown_recomputed",
    ]


def test_resigned_window_trajectory_tamper_is_rejected(
    windowed_artifact: Path, input_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "trajectory-tamper"
    shutil.copytree(windowed_artifact, copied)
    trajectory_path = copied / "trajectories.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["candidate_actions"][0, 0, 0] += 0.25
    np.savez_compressed(trajectory_path, **arrays)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="does not integrate"):
        validate_windowed_cpu_replay(copied, input_artifact)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_count": 0},
        {"chunk_count": True},
        {"supervision_budget_ms": 0.0},
        {"response_deadline_ms": "200"},
        {"qp_step_budget_ms": float("nan")},
        {"worker_timeout_s": False},
        {"poll_interval_s": 0.0},
    ],
)
def test_windowed_replay_configuration_fails_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        WindowedCPUReplayConfig(**kwargs)


def test_windowed_replay_cli_exposes_run_and_validation_inputs() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "vla-panda-windowed-replay",
            "input-artifact",
            "--output-directory",
            "output-artifact",
        ]
    )
    validate = parser.parse_args(
        [
            "vla-panda-windowed-replay-validate",
            "output-artifact",
            "input-artifact",
        ]
    )

    assert run.input_directory == Path("input-artifact")
    assert run.chunks == 30
    assert run.supervision_budget_ms == 20.0
    assert validate.directory == Path("output-artifact")
    assert validate.input_directory == Path("input-artifact")
