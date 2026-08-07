"""Deadline-aware action selection for asynchronous policy outcomes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.async_worker import PolicyOutcome
from armbench.vla.types import DROID_ACTION_DIM

FloatArray = NDArray[np.float64]


def _finite_clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _readonly_action(action: ArrayLike, action_dim: int) -> FloatArray:
    value = np.asarray(action, dtype=float)
    if value.shape != (action_dim,) or not np.all(np.isfinite(value)):
        raise ValueError(f"action must be a finite vector with length {action_dim}")
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class AsyncDispatchConfig:
    action_period_s: float = 1.0 / 15.0
    deadline_s: float = 0.2
    action_dim: int = DROID_ACTION_DIM
    boundary_tolerance_s: float = 1e-9

    def __post_init__(self) -> None:
        timing = np.asarray(
            [self.action_period_s, self.deadline_s, self.boundary_tolerance_s],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(timing))
            or self.action_period_s <= 0.0
            or self.deadline_s < 0.0
            or self.boundary_tolerance_s < 0.0
            or self.action_dim <= 0
        ):
            raise ValueError("async dispatch configuration is invalid")


@dataclass(frozen=True)
class DispatchUpdate:
    status: str
    reason: str
    request_id: int
    observation_sequence_id: int
    observation_age_ms: float
    action_offset: int | None


@dataclass(frozen=True)
class AsyncCommandDecision:
    status: str
    reason: str
    action: FloatArray
    request_id: int | None
    observation_sequence_id: int | None
    observation_age_ms: float | None
    action_index: int | None

    @property
    def holding(self) -> bool:
        return self.status == "hold"

    def metrics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "observation_age_ms": self.observation_age_ms,
            "action_index": self.action_index,
            "action": self.action.tolist(),
        }


class AsyncChunkDispatcher:
    """Select time-aligned actions from completed asynchronous responses."""

    def __init__(
        self,
        config: AsyncDispatchConfig = AsyncDispatchConfig(),
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._active: PolicyOutcome | None = None
        self._latest_response_request_id = -1
        self._hold_reason: str | None = "no_policy_response"
        self._hold_request_id: int | None = None
        self._hold_observation_sequence_id: int | None = None
        self._hold_observation_age_ms: float | None = None
        self._accepted = 0
        self._rejected = 0
        self._executed = 0
        self._holds = 0

    def _stale_steps(self, age_s: float) -> int:
        if age_s <= self.config.boundary_tolerance_s:
            return 0
        return int(
            math.ceil(
                (age_s - self.config.boundary_tolerance_s)
                / self.config.action_period_s
            )
        )

    def _age_s(self, outcome: PolicyOutcome, now_s: float) -> float:
        age_s = now_s - outcome.observation.captured_at_s
        if age_s < -self.config.boundary_tolerance_s:
            raise ValueError("monotonic clock precedes observation capture")
        return max(0.0, age_s)

    def publish(
        self, outcome: PolicyOutcome, *, now_s: float | None = None
    ) -> DispatchUpdate:
        if not isinstance(outcome, PolicyOutcome):
            raise TypeError("outcome must be a PolicyOutcome")
        now = _finite_clock_value(self._clock) if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        try:
            age_s = self._age_s(outcome, now)
        except ValueError:
            age_s = 0.0
            reason = "clock_regression"
        else:
            reason = ""
        if outcome.request_id <= self._latest_response_request_id:
            self._rejected += 1
            return DispatchUpdate(
                "rejected",
                "superseded_response",
                outcome.request_id,
                outcome.observation.sequence_id,
                age_s * 1000.0,
                None,
            )
        self._latest_response_request_id = outcome.request_id
        if reason:
            return self._reject(outcome, reason, age_s, None)
        if not outcome.succeeded:
            return self._reject(outcome, "policy_failure", age_s, None)
        if age_s > self.config.deadline_s + self.config.boundary_tolerance_s:
            return self._reject(outcome, "deadline_exceeded", age_s, None)
        assert outcome.chunk is not None
        action_offset = self._stale_steps(age_s)
        if action_offset >= outcome.chunk.horizon:
            return self._reject(
                outcome, "action_horizon_exhausted", age_s, action_offset
            )
        self._active = outcome
        self._hold_reason = None
        self._hold_request_id = None
        self._hold_observation_sequence_id = None
        self._hold_observation_age_ms = None
        self._accepted += 1
        return DispatchUpdate(
            "accepted",
            "fresh_suffix_available",
            outcome.request_id,
            outcome.observation.sequence_id,
            age_s * 1000.0,
            action_offset,
        )

    def _reject(
        self,
        outcome: PolicyOutcome,
        reason: str,
        age_s: float,
        action_offset: int | None,
    ) -> DispatchUpdate:
        self._active = None
        self._hold_reason = reason
        self._hold_request_id = outcome.request_id
        self._hold_observation_sequence_id = outcome.observation.sequence_id
        self._hold_observation_age_ms = age_s * 1000.0
        self._rejected += 1
        return DispatchUpdate(
            "rejected",
            reason,
            outcome.request_id,
            outcome.observation.sequence_id,
            age_s * 1000.0,
            action_offset,
        )

    def select(
        self,
        hold_action: ArrayLike,
        *,
        now_s: float | None = None,
    ) -> AsyncCommandDecision:
        hold = _readonly_action(hold_action, self.config.action_dim)
        now = _finite_clock_value(self._clock) if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("now_s must be finite")
        outcome = self._active
        if outcome is None:
            self._holds += 1
            return AsyncCommandDecision(
                "hold",
                self._hold_reason or "no_policy_response",
                hold,
                self._hold_request_id,
                self._hold_observation_sequence_id,
                self._hold_observation_age_ms,
                None,
            )
        try:
            age_s = self._age_s(outcome, now)
        except ValueError:
            self._active = None
            self._hold_reason = "clock_regression"
            self._hold_request_id = outcome.request_id
            self._hold_observation_sequence_id = outcome.observation.sequence_id
            self._hold_observation_age_ms = None
            self._holds += 1
            return AsyncCommandDecision(
                "hold",
                "clock_regression",
                hold,
                outcome.request_id,
                outcome.observation.sequence_id,
                None,
                None,
            )
        if age_s > self.config.deadline_s + self.config.boundary_tolerance_s:
            self._active = None
            self._hold_reason = "deadline_exceeded"
            self._hold_request_id = outcome.request_id
            self._hold_observation_sequence_id = outcome.observation.sequence_id
            self._hold_observation_age_ms = age_s * 1000.0
            self._holds += 1
            return AsyncCommandDecision(
                "hold",
                "deadline_exceeded",
                hold,
                outcome.request_id,
                outcome.observation.sequence_id,
                age_s * 1000.0,
                None,
            )
        assert outcome.chunk is not None
        action_index = self._stale_steps(age_s)
        if action_index >= outcome.chunk.horizon:
            self._active = None
            self._hold_reason = "action_horizon_exhausted"
            self._hold_request_id = outcome.request_id
            self._hold_observation_sequence_id = outcome.observation.sequence_id
            self._hold_observation_age_ms = age_s * 1000.0
            self._holds += 1
            return AsyncCommandDecision(
                "hold",
                "action_horizon_exhausted",
                hold,
                outcome.request_id,
                outcome.observation.sequence_id,
                age_s * 1000.0,
                action_index,
            )
        action = _readonly_action(
            outcome.chunk.actions[action_index], self.config.action_dim
        )
        self._executed += 1
        return AsyncCommandDecision(
            "execute",
            "fresh_suffix_available",
            action,
            outcome.request_id,
            outcome.observation.sequence_id,
            age_s * 1000.0,
            action_index,
        )

    def metrics(self) -> dict[str, object]:
        return {
            "accepted_responses": self._accepted,
            "rejected_responses": self._rejected,
            "executed_commands": self._executed,
            "hold_commands": self._holds,
            "active_request_id": (
                self._active.request_id if self._active is not None else None
            ),
            "hold_reason": self._hold_reason,
            "hold_request_id": self._hold_request_id,
            "hold_observation_sequence_id": self._hold_observation_sequence_id,
            "hold_observation_age_ms": self._hold_observation_age_ms,
        }
