from __future__ import annotations

import json
from pathlib import Path

import pytest

import armbench.vla.official_lerobot as official
from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    PANDA_RUNTIME_ACTION_SPACE_ID,
)
from armbench.vla.official_lerobot import (
    OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION,
    OFFICIAL_LEROBOT_ROBOT_TYPE,
    OFFICIAL_LEROBOT_VERSION,
    OfficialLeRobotError,
    export_official_lerobot_episode,
    official_lerobot_diagnostic,
    official_panda_features,
    run_official_lerobot_smoke,
    validate_official_lerobot_episode,
)


def test_official_version_and_panda_feature_contract_are_pinned() -> None:
    features = official_panda_features()

    assert OFFICIAL_LEROBOT_VERSION == "0.4.4"
    assert OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION == "v3.0"
    assert OFFICIAL_LEROBOT_ROBOT_TYPE == "panda_armbench_runtime"
    assert features["observation.images.exterior"] == {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    }
    assert features["observation.state"]["shape"] == (8,)
    assert features["action"]["shape"] == (8,)
    assert features["observation.state"]["names"][0].endswith("position_rad")
    assert features["action"]["names"][0].endswith("velocity_rad_s")
    assert all("so101" not in name.lower() for name in features["action"]["names"])
    assert len(PANDA_RUNTIME_ACTION_SEMANTICS_SHA256) == 64
    assert PANDA_RUNTIME_ACTION_SPACE_ID.startswith("armbench.panda.")


def test_missing_official_dependency_fails_with_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> str:
        raise official.PackageNotFoundError

    monkeypatch.setattr(official, "_distribution_version", missing)

    diagnostic = official_lerobot_diagnostic()
    with pytest.raises(OfficialLeRobotError, match="setup_official_lerobot"):
        official._load_official_dataset_api()

    assert diagnostic["available"] is False
    assert diagnostic["installed_version"] is None
    assert diagnostic["required_version"] == OFFICIAL_LEROBOT_VERSION


def test_wrong_official_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(official, "_distribution_version", lambda _: "0.4.3")

    with pytest.raises(OfficialLeRobotError, match="version mismatch"):
        official._load_official_dataset_api()


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation.state", ["0.0"] * 7 + ["0.5"]),
        ("action", [False] * 7 + [True]),
        ("observation.state", [0.0] * 7 + [1.01]),
        ("action", [0.0] * 7 + [-0.01]),
    ],
)
def test_official_export_rejects_coerced_or_out_of_range_frames(
    tmp_path: Path, field: str, value: object
) -> None:
    frames = official._smoke_frames()
    frames[0] = {**frames[0], field: value}

    with pytest.raises(ValueError, match="frame values"):
        export_official_lerobot_episode(tmp_path / "official", frames)

    assert not (tmp_path / "official").exists()


def _official_api_available() -> bool:
    return bool(official_lerobot_diagnostic()["available"])


@pytest.mark.skipif(
    not _official_api_available(),
    reason="run scripts/setup_official_lerobot.ps1 for the isolated loader env",
)
def test_pinned_official_loader_roundtrip(tmp_path: Path) -> None:
    output = run_official_lerobot_smoke(tmp_path / "official")

    result = validate_official_lerobot_episode(output)
    summary = json.loads((output / "summary.json").read_text("utf-8"))

    assert result["valid"]
    assert result["frames"] == 3
    assert result["field_checks"] == {
        "images": 6,
        "state": 3,
        "action": 3,
        "task": 3,
        "timestamp": 3,
    }
    assert result["so101_actions_used"] is False
    assert summary["claims"]["official_lerobot_package_used"] is True
    assert summary["claims"]["official_lerobot_dataset_loader_used"] is True
    assert summary["claims"]["physical_robot_connected"] is False
    assert summary["embodiment_boundary"] == {
        "dataset_robot_type": "panda_armbench_runtime",
        "panda_joint_velocity_actions": True,
        "so101_joint_position_actions": False,
        "panda_to_so101_conversion_performed": False,
    }


@pytest.mark.skipif(
    not _official_api_available(),
    reason="run scripts/setup_official_lerobot.ps1 for the isolated loader env",
)
def test_official_artifact_manifest_rejects_summary_tamper(
    tmp_path: Path,
) -> None:
    output = run_official_lerobot_smoke(tmp_path / "official")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["frames"] = 99
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(OfficialLeRobotError, match="manifest inventory"):
        validate_official_lerobot_episode(output)


@pytest.mark.skipif(
    not _official_api_available(),
    reason="run scripts/setup_official_lerobot.ps1 for the isolated loader env",
)
def test_official_resigned_claim_deletion_is_rejected(tmp_path: Path) -> None:
    output = run_official_lerobot_smoke(tmp_path / "official")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    del summary["claims"]["physical_robot_connected"]
    official.write_json(summary_path, summary)
    official._write_manifest(output)

    with pytest.raises(OfficialLeRobotError, match="claim boundary"):
        validate_official_lerobot_episode(output)
