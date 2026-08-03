"""Validated data contracts at the VLA policy/runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
DROID_IMAGE_SHAPE = (224, 224, 3)
DROID_STATE_DIM = 8
DROID_ACTION_DIM = 8
PI05_DROID_ACTION_HORIZON = 15


def _uint8_image(value: ArrayLike, label: str) -> UInt8Array:
    image = np.asarray(value)
    if image.shape != DROID_IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8 with shape {DROID_IMAGE_SHAPE}")
    result = image.copy()
    result.flags.writeable = False
    return result


def _float_vector(value: ArrayLike, length: int, label: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite vector with length {length}")
    result = vector.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class VLAObservation:
    """One pi0/pi0.5-DROID-compatible policy observation."""

    exterior_image: UInt8Array
    wrist_image: UInt8Array
    joint_position: FloatArray
    gripper_position: FloatArray
    prompt: str
    sequence_id: int = 0
    captured_at_s: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if self.sequence_id < 0 or not np.isfinite(self.captured_at_s):
            raise ValueError("observation sequence/time is invalid")
        object.__setattr__(
            self, "exterior_image", _uint8_image(self.exterior_image, "exterior_image")
        )
        object.__setattr__(
            self, "wrist_image", _uint8_image(self.wrist_image, "wrist_image")
        )
        object.__setattr__(
            self,
            "joint_position",
            _float_vector(self.joint_position, 7, "joint_position"),
        )
        object.__setattr__(
            self,
            "gripper_position",
            _float_vector(self.gripper_position, 1, "gripper_position"),
        )

    @property
    def state(self) -> FloatArray:
        return np.concatenate([self.joint_position, self.gripper_position])

    def to_openpi_droid(self) -> dict[str, object]:
        """Use the exact input keys consumed by OpenPI's DroidInputs transform."""

        return {
            "observation/exterior_image_1_left": self.exterior_image,
            "observation/wrist_image_left": self.wrist_image,
            "observation/joint_position": self.joint_position,
            "observation/gripper_position": self.gripper_position,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class ActionChunk:
    """A DROID joint-velocity/gripper-position action chunk."""

    actions: FloatArray
    source: str
    observation_sequence_id: int
    inference_latency_ms: float
    received_at_s: float = field(default_factory=time.monotonic)
    server_timing: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != DROID_ACTION_DIM:
            raise ValueError(
                f"actions must have shape (horizon, {DROID_ACTION_DIM})"
            )
        if len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be nonempty and finite")
        if not self.source.strip() or self.observation_sequence_id < 0:
            raise ValueError("action source/observation sequence is invalid")
        if self.inference_latency_ms < 0.0 or not np.isfinite(
            self.inference_latency_ms
        ):
            raise ValueError("inference latency must be finite and nonnegative")
        if not np.isfinite(self.received_at_s):
            raise ValueError("action receive time must be finite")
        normalized_timing = {
            str(key): float(value) for key, value in self.server_timing.items()
        }
        if any(not np.isfinite(value) for value in normalized_timing.values()):
            raise ValueError("server timing must contain finite values")
        result = actions.copy()
        result.flags.writeable = False
        object.__setattr__(self, "actions", result)
        object.__setattr__(self, "server_timing", normalized_timing)

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    def age_ms(self, observation: VLAObservation) -> float:
        if observation.sequence_id != self.observation_sequence_id:
            raise ValueError("action chunk does not match the observation sequence")
        return max(0.0, (self.received_at_s - observation.captured_at_s) * 1000.0)
