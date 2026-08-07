from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from armbench.vla.lerobot_adapter import (
    LEROBOT_STYLE_FRAME_KEYS,
    PANDA_RUNTIME_ACTION_SEMANTICS_ID,
    LeRobotFrameAdapter,
)
from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
)
from armbench.vla.provider_contract import _fixture_observation


def test_frame_adapter_emits_exact_lerobot_style_keys_and_dtypes() -> None:
    observation = _fixture_observation()
    action = np.linspace(-0.2, 0.5, 8)

    frame = LeRobotFrameAdapter().to_frame(
        observation,
        action,
        action_semantics_id=PANDA_RUNTIME_ACTION_SEMANTICS_ID,
    )

    assert tuple(frame) == LEROBOT_STYLE_FRAME_KEYS
    assert frame["observation.images.exterior"].shape == (224, 224, 3)
    assert frame["observation.images.exterior"].dtype == np.uint8
    assert frame["observation.images.wrist"].dtype == np.uint8
    assert frame["observation.state"].shape == (8,)
    assert frame["observation.state"].dtype == np.float32
    assert frame["action"].shape == (8,)
    assert frame["action"].dtype == np.float32
    assert frame["task"] == observation.prompt


def test_frame_adapter_returns_copies() -> None:
    observation = _fixture_observation()
    action = np.zeros(8)
    frame = LeRobotFrameAdapter().to_frame(
        observation,
        action,
        action_semantics_id=PANDA_RUNTIME_ACTION_SEMANTICS_ID,
    )

    frame["observation.images.exterior"][0, 0, 0] = 255
    frame["observation.state"][0] = 99.0
    frame["action"][0] = 99.0

    assert observation.exterior_image[0, 0, 0] != 255
    assert observation.state[0] != 99.0
    assert action[0] != 99.0


def test_frame_adapter_rejects_wrong_action_semantics() -> None:
    with pytest.raises(ValueError, match="semantics mismatch"):
        LeRobotFrameAdapter().to_frame(
            _fixture_observation(),
            np.zeros(8),
            action_semantics_id="libero.ee_delta_pose_gripper.v1",
        )


def test_frame_adapter_rejects_wrong_semantics_hash() -> None:
    with pytest.raises(ValueError, match="semantics mismatch"):
        LeRobotFrameAdapter().to_frame(
            _fixture_observation(),
            np.zeros(8),
            action_semantics_id=PANDA_RUNTIME_ACTION_SEMANTICS_ID,
            action_semantics_sha256="0" * 64,
        )

    assert PANDA_RUNTIME_ACTION_SEMANTICS_SHA256 != "0" * 64


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_action_semantics_id": "forged.panda.action.v1"},
        {"expected_action_semantics_sha256": "0" * 64},
    ],
)
def test_frame_adapter_cannot_register_forged_semantics(
    changes: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="unregistered"):
        LeRobotFrameAdapter(**changes)


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(7),
        np.zeros((1, 8)),
        np.full(8, np.nan),
        np.array([0.0] * 7 + [1.01]),
        ["0.0"] * 7 + ["0.5"],
        [False] * 7 + [True],
    ],
)
def test_frame_adapter_rejects_invalid_actions(action: object) -> None:
    with pytest.raises(ValueError, match="finite 8-vector"):
        LeRobotFrameAdapter().to_frame(
            replace(_fixture_observation(), sequence_id=2),
            action,  # type: ignore[arg-type]
            action_semantics_id=PANDA_RUNTIME_ACTION_SEMANTICS_ID,
        )


def test_frame_adapter_rejects_out_of_range_observation_gripper() -> None:
    observation = replace(
        _fixture_observation(), gripper_position=np.array([1.01])
    )

    with pytest.raises(ValueError, match="observation gripper"):
        LeRobotFrameAdapter().to_frame(
            observation,
            np.zeros(8),
            action_semantics_id=PANDA_RUNTIME_ACTION_SEMANTICS_ID,
        )
