"""Runtime validation for image/state observations before VLA inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armbench.vla.types import VLAObservation


@dataclass(frozen=True)
class ObservationGuardConfig:
    min_image_std: float = 5.0
    motion_threshold_rad: float = 0.005

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.min_image_std, self.motion_threshold_rad], dtype=float
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "observation guard thresholds must be finite and nonnegative"
            )


@dataclass(frozen=True)
class ObservationCheck:
    sequence_id: int
    healthy: bool
    reasons: tuple[str, ...]
    exterior_image_std: float
    wrist_image_std: float
    state_change_rad: float | None
    exterior_frame_replayed: bool
    wrist_frame_replayed: bool

    def metrics(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "healthy": self.healthy,
            "reasons": list(self.reasons),
            "exterior_image_std": self.exterior_image_std,
            "wrist_image_std": self.wrist_image_std,
            "state_change_rad": self.state_change_rad,
            "exterior_frame_replayed": self.exterior_frame_replayed,
            "wrist_frame_replayed": self.wrist_frame_replayed,
        }


class ObservationRejectedError(ValueError):
    def __init__(self, check: ObservationCheck) -> None:
        self.check = check
        super().__init__("; ".join(check.reasons))


class VLAObservationGuard:
    """Reject blank or replayed camera data before calling a VLA policy."""

    def __init__(
        self, config: ObservationGuardConfig = ObservationGuardConfig()
    ) -> None:
        self.config = config
        self._previous: VLAObservation | None = None

    def reset(self) -> None:
        self._previous = None

    def check(self, observation: VLAObservation) -> ObservationCheck:
        exterior_std = float(observation.exterior_image.std())
        wrist_std = float(observation.wrist_image.std())
        reasons: list[str] = []
        if exterior_std < self.config.min_image_std:
            reasons.append("exterior_image_low_information")
        if wrist_std < self.config.min_image_std:
            reasons.append("wrist_image_low_information")

        state_change: float | None = None
        exterior_replayed = False
        wrist_replayed = False
        previous = self._previous
        if previous is not None:
            if observation.sequence_id <= previous.sequence_id:
                reasons.append("nonmonotonic_observation_sequence")
            if observation.captured_at_s <= previous.captured_at_s:
                reasons.append("nonmonotonic_observation_time")
            state_change = float(
                np.max(
                    np.abs(
                        observation.joint_position - previous.joint_position
                    )
                )
            )
            exterior_replayed = bool(
                np.array_equal(
                    observation.exterior_image, previous.exterior_image
                )
            )
            wrist_replayed = bool(
                np.array_equal(observation.wrist_image, previous.wrist_image)
            )
            if state_change > self.config.motion_threshold_rad:
                if exterior_replayed:
                    reasons.append("exterior_frame_replayed_during_motion")
                if wrist_replayed:
                    reasons.append("wrist_frame_replayed_during_motion")

        result = ObservationCheck(
            sequence_id=observation.sequence_id,
            healthy=not reasons,
            reasons=tuple(reasons),
            exterior_image_std=exterior_std,
            wrist_image_std=wrist_std,
            state_change_rad=state_change,
            exterior_frame_replayed=exterior_replayed,
            wrist_frame_replayed=wrist_replayed,
        )
        if not result.healthy:
            raise ObservationRejectedError(result)
        self._previous = observation
        return result
