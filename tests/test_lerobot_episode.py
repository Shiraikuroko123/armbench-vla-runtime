from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from armbench.vla.lerobot_episode import (
    LeRobotEpisodeError,
    _write_manifest,
    replay_lerobot_episode,
    run_lerobot_episode_smoke,
    validate_lerobot_episode,
)


def test_smoke_episode_replays_watchdog_and_lerobot_mapping(
    tmp_path: Path,
) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")

    result = validate_lerobot_episode(output)
    replay = replay_lerobot_episode(output)
    metadata = json.loads((output / "metadata.json").read_text("utf-8"))

    assert result["valid"]
    assert result["frames"] == 5
    assert result["executed_commands"] == 3
    assert result["held_commands"] == 2
    assert result["reset_events"] == 1
    assert result["reason_counts"] == {
        "command_valid": 3,
        "fault_latched": 1,
        "observation_deadline_exceeded": 1,
    }
    assert replay["all_watchdog_decisions_matched"]
    assert replay["replayed_frames"] == 5
    assert not any(metadata["claims"].values())
    assert metadata["lerobot_style_interface"][
        "official_lerobot_dataset_storage"
    ] is False


def test_manifest_rejects_byte_tamper(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["frames"] = 99
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(LeRobotEpisodeError, match="manifest"):
        validate_lerobot_episode(output)


def test_resigned_watchdog_decision_tamper_is_recomputed(
    tmp_path: Path,
) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    frames_path = output / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text("utf-8").splitlines()]
    rows[0]["watchdog_decision"]["reason"] = "forged_valid_reason"
    frames_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="not reproducible"):
        validate_lerobot_episode(output)


def test_resigned_sequence_tamper_is_rejected_before_replay(
    tmp_path: Path,
) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    archive_path = output / "episode.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["command_sequence_ids"][2] = arrays["command_sequence_ids"][1]
    np.savez_compressed(archive_path, **arrays)
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="sequence IDs"):
        validate_lerobot_episode(output)


def test_resigned_deleted_claim_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    del metadata["claims"]["physical_robot_connected"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="claim boundary"):
        validate_lerobot_episode(output)


def test_resigned_unknown_metadata_field_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["unverified_runtime"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="metadata schema"):
        validate_lerobot_episode(output)


def test_resigned_unknown_frame_field_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    frames_path = output / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text("utf-8").splitlines()]
    rows[0]["policy_checkpoint_executed"] = True
    frames_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="schema/index"):
        validate_lerobot_episode(output)


def test_resigned_string_frame_timestamp_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    frames_path = output / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text("utf-8").splitlines()]
    rows[0]["captured_at_s"] = str(rows[0]["captured_at_s"])
    frames_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="schema/index"):
        validate_lerobot_episode(output)


def test_resigned_string_watchdog_config_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["watchdog_config"]["max_action_age_s"] = "0.1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_manifest(output)

    with pytest.raises(LeRobotEpisodeError, match="watchdog configuration"):
        validate_lerobot_episode(output)


def test_manifest_unknown_field_is_rejected(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["trusted"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LeRobotEpisodeError, match="manifest schema"):
        validate_lerobot_episode(output)


def test_smoke_refuses_to_overwrite_existing_episode(tmp_path: Path) -> None:
    output = run_lerobot_episode_smoke(tmp_path / "episode")

    with pytest.raises(LeRobotEpisodeError, match="already exists"):
        run_lerobot_episode_smoke(output)
