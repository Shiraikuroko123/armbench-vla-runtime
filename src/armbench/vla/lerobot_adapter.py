"""LeRobot-style frame mapping for ArmBench's Panda/DROID runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import ArrayLike

from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    PANDA_RUNTIME_ACTION_SPACE_ID,
)
from armbench.vla.types import DROID_ACTION_DIM, VLAObservation


PANDA_RUNTIME_ACTION_SEMANTICS_ID = PANDA_RUNTIME_ACTION_SPACE_ID
LEROBOT_STYLE_FRAME_KEYS = (
    "observation.images.exterior",
    "observation.images.wrist",
    "observation.state",
    "action",
    "task",
)


def frame_array_sha256(value: ArrayLike, *, dtype: str) -> str:
    array = np.asarray(value, dtype=dtype, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError("frame array must be finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class LeRobotFrameAdapter:
    """Map one guarded Panda command to LeRobot ``add_frame``-style keys.

    This produces an in-memory API-shaped record. It does not claim byte-level
    compatibility with any particular LeRobotDataset storage version.
    """

    expected_action_semantics_id: str = PANDA_RUNTIME_ACTION_SEMANTICS_ID
    expected_action_semantics_sha256: str = (
        PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
    )

    def __post_init__(self) -> None:
        if (
            self.expected_action_semantics_id
            != PANDA_RUNTIME_ACTION_SEMANTICS_ID
            or self.expected_action_semantics_sha256
            != PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
        ):
            raise ValueError("unregistered Panda action semantics identity")

    def to_frame(
        self,
        observation: VLAObservation,
        action: ArrayLike,
        *,
        action_semantics_id: str,
        action_semantics_sha256: str = PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    ) -> dict[str, object]:
        if (
            action_semantics_id != self.expected_action_semantics_id
            or action_semantics_sha256
            != self.expected_action_semantics_sha256
        ):
            raise ValueError(
                "action semantics mismatch at LeRobot frame boundary: "
                f"expected {self.expected_action_semantics_id!r}/"
                f"{self.expected_action_semantics_sha256}, got "
                f"{action_semantics_id!r}/{action_semantics_sha256}"
            )
        raw = np.asarray(action)
        if raw.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "LeRobot frame action must be a finite 8-vector of numeric values"
            )
        command = np.asarray(raw, dtype=np.float32)
        if (
            command.shape != (DROID_ACTION_DIM,)
            or not np.all(np.isfinite(command))
            or not 0.0 <= command[7] <= 1.0
        ):
            raise ValueError(
                "LeRobot frame action must be a finite 8-vector of numeric values "
                "with gripper in [0, 1]"
            )
        state = observation.state.astype(np.float32, copy=True)
        if not 0.0 <= state[7] <= 1.0:
            raise ValueError("LeRobot observation gripper must be in [0, 1]")
        return {
            "observation.images.exterior": observation.exterior_image.copy(),
            "observation.images.wrist": observation.wrist_image.copy(),
            "observation.state": state,
            "action": command.copy(),
            "task": observation.prompt,
        }
