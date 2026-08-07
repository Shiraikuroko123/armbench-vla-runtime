"""Fail-closed command watchdog at the actuator send boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.types import DROID_ACTION_DIM


FloatArray = NDArray[np.float64]
WatchdogStatus = Literal["execute", "hold"]
WATCHDOG_DECISION_SCHEMA = "armbench.command_watchdog_decision.v1"
PANDA_RUNTIME_ACTION_SPACE_ID = (
    "armbench.panda.joint_velocity_gripper_position.v1"
)
_RUNTIME_SEMANTICS = {
    "schema_version": "armbench.runtime_action_semantics.v1",
    "action_space_id": PANDA_RUNTIME_ACTION_SPACE_ID,
    "action_dim": DROID_ACTION_DIM,
    "action_order": [
        "panda_joint1_velocity_rad_s",
        "panda_joint2_velocity_rad_s",
        "panda_joint3_velocity_rad_s",
        "panda_joint4_velocity_rad_s",
        "panda_joint5_velocity_rad_s",
        "panda_joint6_velocity_rad_s",
        "panda_joint7_velocity_rad_s",
        "gripper_position_normalized",
    ],
    "coordinate_frame": "panda_joint_space",
    "joint_units": "rad_s",
    "gripper_convention": "zero_closed_one_open",
    "gripper_range": [0.0, 1.0],
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


PANDA_RUNTIME_ACTION_SEMANTICS_SHA256 = hashlib.sha256(
    _canonical_json(_RUNTIME_SEMANTICS)
).hexdigest()


def runtime_action_semantics() -> dict[str, object]:
    """Return a fresh, JSON-serializable runtime action contract."""

    return json.loads(_canonical_json(_RUNTIME_SEMANTICS).decode("ascii"))


def _readonly_action(value: ArrayLike) -> FloatArray | None:
    try:
        action = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if action.shape != (DROID_ACTION_DIM,) or not np.all(np.isfinite(action)):
        return None
    result = action.copy()
    result.flags.writeable = False
    return result


def _action_sha256(action: FloatArray) -> str:
    canonical = np.asarray(action, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _optional_int(value: object) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _optional_finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class CommandWatchdogConfig:
    max_observation_age_s: float = 0.4
    max_action_age_s: float = 0.1
    heartbeat_timeout_s: float = 0.2
    fallback_gripper_position: float = 1.0
    action_semantics_id: str = PANDA_RUNTIME_ACTION_SPACE_ID
    action_semantics_sha256: str = PANDA_RUNTIME_ACTION_SEMANTICS_SHA256

    def __post_init__(self) -> None:
        timing = np.asarray(
            [
                self.max_observation_age_s,
                self.max_action_age_s,
                self.heartbeat_timeout_s,
                self.fallback_gripper_position,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(timing))
            or self.max_observation_age_s < 0.0
            or self.max_action_age_s < 0.0
            or self.heartbeat_timeout_s <= 0.0
            or not 0.0 <= self.fallback_gripper_position <= 1.0
            or self.action_semantics_id != PANDA_RUNTIME_ACTION_SPACE_ID
            or self.action_semantics_sha256
            != PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
        ):
            raise ValueError("command watchdog configuration is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_observation_age_s": self.max_observation_age_s,
            "max_action_age_s": self.max_action_age_s,
            "heartbeat_timeout_s": self.heartbeat_timeout_s,
            "fallback_gripper_position": self.fallback_gripper_position,
            "action_semantics_id": self.action_semantics_id,
            "action_semantics_sha256": self.action_semantics_sha256,
        }


@dataclass(frozen=True)
class WatchdogDecision:
    status: WatchdogStatus
    reason: str
    action: FloatArray
    latched: bool
    fault_reason: str | None
    command_sequence_id: int | None
    observation_sequence_id: int | None
    captured_at_s: float | None
    issued_at_s: float | None
    evaluated_at_s: float | None
    observation_age_ms: float | None
    action_age_ms: float | None
    input_action_sha256: str | None
    action_semantics_id: str
    action_semantics_sha256: str

    def __post_init__(self) -> None:
        action = _readonly_action(self.action)
        if (
            self.status not in {"execute", "hold"}
            or not self.reason.strip()
            or action is None
            or self.latched != (self.fault_reason is not None)
        ):
            raise ValueError("watchdog decision is invalid")
        object.__setattr__(self, "action", action)

    @property
    def holding(self) -> bool:
        return self.status == "hold"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe evidence suitable for deterministic replay."""

        return {
            "schema_version": WATCHDOG_DECISION_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "action": self.action.tolist(),
            "output_action_sha256": _action_sha256(self.action),
            "latched": self.latched,
            "fault_reason": self.fault_reason,
            "command_sequence_id": self.command_sequence_id,
            "observation_sequence_id": self.observation_sequence_id,
            "captured_at_s": self.captured_at_s,
            "issued_at_s": self.issued_at_s,
            "evaluated_at_s": self.evaluated_at_s,
            "observation_age_ms": self.observation_age_ms,
            "action_age_ms": self.action_age_ms,
            "input_action_sha256": self.input_action_sha256,
            "action_semantics_id": self.action_semantics_id,
            "action_semantics_sha256": self.action_semantics_sha256,
        }


class ActuatorCommandWatchdog:
    """Validate the final command envelope and latch every protocol fault.

    The dispatcher decides which action is temporally useful and the guard
    repairs its trajectory. This watchdog is intentionally narrower: it checks
    the ordering, timestamps, liveness, and exact action contract immediately
    before an actuator transport would send the selected command.
    """

    def __init__(
        self, config: CommandWatchdogConfig = CommandWatchdogConfig()
    ) -> None:
        self.config = config
        self._last_command_sequence_id: int | None = None
        self._last_observation_sequence_id: int | None = None
        self._last_captured_at_s: float | None = None
        self._last_issued_at_s: float | None = None
        self._last_evaluated_at_s: float | None = None
        self._last_command_activity_s: float | None = None
        self._accept_not_before_s: float | None = None
        self._fault_reason: str | None = None
        self._accepted_commands = 0
        self._rejected_commands = 0
        self._heartbeat_timeouts = 0
        self._resets = 0

    @property
    def latched(self) -> bool:
        return self._fault_reason is not None

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    def _hold_action(self, gripper_position: object) -> FloatArray:
        gripper = _optional_finite(gripper_position)
        if gripper is None or not 0.0 <= gripper <= 1.0:
            gripper = self.config.fallback_gripper_position
        action = np.zeros(DROID_ACTION_DIM, dtype=float)
        action[7] = gripper
        action.flags.writeable = False
        return action

    def _decision(
        self,
        *,
        status: WatchdogStatus,
        reason: str,
        action: FloatArray,
        command_sequence_id: int | None,
        observation_sequence_id: int | None,
        captured_at_s: float | None,
        issued_at_s: float | None,
        evaluated_at_s: float | None,
        observation_age_ms: float | None,
        action_age_ms: float | None,
        input_action_sha256: str | None,
        action_semantics_id: str,
        action_semantics_sha256: str,
    ) -> WatchdogDecision:
        return WatchdogDecision(
            status=status,
            reason=reason,
            action=action,
            latched=self.latched,
            fault_reason=self._fault_reason,
            command_sequence_id=command_sequence_id,
            observation_sequence_id=observation_sequence_id,
            captured_at_s=captured_at_s,
            issued_at_s=issued_at_s,
            evaluated_at_s=evaluated_at_s,
            observation_age_ms=observation_age_ms,
            action_age_ms=action_age_ms,
            input_action_sha256=input_action_sha256,
            action_semantics_id=action_semantics_id,
            action_semantics_sha256=action_semantics_sha256,
        )

    def _reject(
        self,
        reason: str,
        *,
        hold: FloatArray,
        command_sequence_id: int | None,
        observation_sequence_id: int | None,
        captured_at_s: float | None,
        issued_at_s: float | None,
        evaluated_at_s: float | None,
        observation_age_ms: float | None,
        action_age_ms: float | None,
        input_action_sha256: str | None,
        action_semantics_id: str,
        action_semantics_sha256: str,
    ) -> WatchdogDecision:
        self._fault_reason = reason
        if evaluated_at_s is not None and (
            self._last_evaluated_at_s is None
            or evaluated_at_s > self._last_evaluated_at_s
        ):
            self._last_evaluated_at_s = evaluated_at_s
        self._rejected_commands += 1
        return self._decision(
            status="hold",
            reason=reason,
            action=hold,
            command_sequence_id=command_sequence_id,
            observation_sequence_id=observation_sequence_id,
            captured_at_s=captured_at_s,
            issued_at_s=issued_at_s,
            evaluated_at_s=evaluated_at_s,
            observation_age_ms=observation_age_ms,
            action_age_ms=action_age_ms,
            input_action_sha256=input_action_sha256,
            action_semantics_id=action_semantics_id,
            action_semantics_sha256=action_semantics_sha256,
        )

    def evaluate(
        self,
        action: ArrayLike,
        *,
        command_sequence_id: int,
        observation_sequence_id: int,
        captured_at_s: float,
        issued_at_s: float,
        evaluated_at_s: float,
        gripper_position: float,
        action_semantics_id: str = PANDA_RUNTIME_ACTION_SPACE_ID,
        action_semantics_sha256: str = PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    ) -> WatchdogDecision:
        """Return an executable action or a deterministic, latched hold."""

        hold = self._hold_action(gripper_position)
        command_id = _optional_int(command_sequence_id)
        observation_id = _optional_int(observation_sequence_id)
        captured = _optional_finite(captured_at_s)
        issued = _optional_finite(issued_at_s)
        evaluated = _optional_finite(evaluated_at_s)
        hold_gripper = _optional_finite(gripper_position)
        checked_action = _readonly_action(action)
        input_hash = (
            _action_sha256(checked_action) if checked_action is not None else None
        )
        observation_age_ms = (
            (evaluated - captured) * 1000.0
            if evaluated is not None and captured is not None
            else None
        )
        action_age_ms = (
            (evaluated - issued) * 1000.0
            if evaluated is not None and issued is not None
            else None
        )
        decision_fields = {
            "hold": hold,
            "command_sequence_id": command_id,
            "observation_sequence_id": observation_id,
            "captured_at_s": captured,
            "issued_at_s": issued,
            "evaluated_at_s": evaluated,
            "observation_age_ms": observation_age_ms,
            "action_age_ms": action_age_ms,
            "input_action_sha256": input_hash,
            "action_semantics_id": str(action_semantics_id),
            "action_semantics_sha256": str(action_semantics_sha256),
        }
        if self.latched:
            return self._decision(
                status="hold",
                reason="fault_latched",
                action=hold,
                command_sequence_id=command_id,
                observation_sequence_id=observation_id,
                captured_at_s=captured,
                issued_at_s=issued,
                evaluated_at_s=evaluated,
                observation_age_ms=observation_age_ms,
                action_age_ms=action_age_ms,
                input_action_sha256=input_hash,
                action_semantics_id=str(action_semantics_id),
                action_semantics_sha256=str(action_semantics_sha256),
            )

        reason: str | None = None
        if checked_action is None:
            reason = "invalid_action"
        elif command_id is None:
            reason = "invalid_command_sequence"
        elif observation_id is None:
            reason = "invalid_observation_sequence"
        elif captured is None or issued is None or evaluated is None:
            reason = "invalid_timestamp"
        elif (
            action_semantics_id != self.config.action_semantics_id
            or action_semantics_sha256 != self.config.action_semantics_sha256
        ):
            reason = "action_semantics_mismatch"
        elif hold_gripper is None or not 0.0 <= hold_gripper <= 1.0:
            reason = "invalid_hold_gripper"
        elif not captured <= issued <= evaluated:
            reason = "timestamp_order_violation"
        elif (
            self._last_evaluated_at_s is not None
            and evaluated < self._last_evaluated_at_s
        ):
            reason = "evaluation_time_regression"
        elif (
            self._last_command_sequence_id is not None
            and command_id <= self._last_command_sequence_id
        ):
            reason = "command_sequence_not_increasing"
        elif (
            self._last_observation_sequence_id is not None
            and observation_id < self._last_observation_sequence_id
        ):
            reason = "observation_sequence_regression"
        elif (
            self._last_captured_at_s is not None
            and captured < self._last_captured_at_s
        ):
            reason = "capture_time_regression"
        elif (
            self._last_issued_at_s is not None
            and issued < self._last_issued_at_s
        ):
            reason = "issue_time_regression"
        elif self._accept_not_before_s is not None and issued < self._accept_not_before_s:
            reason = "command_predates_reset"
        elif evaluated - captured > self.config.max_observation_age_s:
            reason = "observation_deadline_exceeded"
        elif evaluated - issued > self.config.max_action_age_s:
            reason = "action_deadline_exceeded"
        if reason is not None:
            return self._reject(reason, **decision_fields)

        assert checked_action is not None
        assert command_id is not None and observation_id is not None
        assert captured is not None and issued is not None and evaluated is not None
        self._last_command_sequence_id = command_id
        self._last_observation_sequence_id = observation_id
        self._last_captured_at_s = captured
        self._last_issued_at_s = issued
        self._last_evaluated_at_s = evaluated
        self._last_command_activity_s = evaluated
        self._accepted_commands += 1
        return self._decision(
            status="execute",
            reason="command_valid",
            action=checked_action,
            command_sequence_id=command_id,
            observation_sequence_id=observation_id,
            captured_at_s=captured,
            issued_at_s=issued,
            evaluated_at_s=evaluated,
            observation_age_ms=observation_age_ms,
            action_age_ms=action_age_ms,
            input_action_sha256=input_hash,
            action_semantics_id=str(action_semantics_id),
            action_semantics_sha256=str(action_semantics_sha256),
        )

    def poll(
        self, *, evaluated_at_s: float, gripper_position: float
    ) -> WatchdogDecision | None:
        """Return a hold on absent/expired heartbeat, otherwise no intervention."""

        hold = self._hold_action(gripper_position)
        evaluated = _optional_finite(evaluated_at_s)
        fields = {
            "command_sequence_id": self._last_command_sequence_id,
            "observation_sequence_id": self._last_observation_sequence_id,
            "captured_at_s": self._last_captured_at_s,
            "issued_at_s": self._last_issued_at_s,
            "evaluated_at_s": evaluated,
            "observation_age_ms": (
                None
                if evaluated is None or self._last_captured_at_s is None
                else (evaluated - self._last_captured_at_s) * 1000.0
            ),
            "action_age_ms": (
                None
                if evaluated is None or self._last_issued_at_s is None
                else (evaluated - self._last_issued_at_s) * 1000.0
            ),
            "input_action_sha256": None,
            "action_semantics_id": self.config.action_semantics_id,
            "action_semantics_sha256": self.config.action_semantics_sha256,
        }
        if self.latched:
            return self._decision(
                status="hold",
                reason="fault_latched",
                action=hold,
                **fields,
            )
        if evaluated is None:
            return self._reject("invalid_timestamp", hold=hold, **fields)
        if (
            self._last_evaluated_at_s is not None
            and evaluated < self._last_evaluated_at_s
        ):
            return self._reject(
                "evaluation_time_regression", hold=hold, **fields
            )
        self._last_evaluated_at_s = evaluated
        if self._last_command_activity_s is None:
            return self._decision(
                status="hold",
                reason="awaiting_first_command",
                action=hold,
                **fields,
            )
        if evaluated - self._last_command_activity_s > self.config.heartbeat_timeout_s:
            self._heartbeat_timeouts += 1
            return self._reject("heartbeat_timeout", hold=hold, **fields)
        return None

    def reset(self, *, evaluated_at_s: float) -> None:
        """Explicitly clear a latch without clearing replay-protection state."""

        evaluated = _optional_finite(evaluated_at_s)
        if evaluated is None or (
            self._last_evaluated_at_s is not None
            and evaluated < self._last_evaluated_at_s
        ):
            raise ValueError("watchdog reset time is invalid or regressed")
        self._fault_reason = None
        self._last_evaluated_at_s = evaluated
        self._last_command_activity_s = evaluated
        self._accept_not_before_s = evaluated
        self._resets += 1

    def metrics(self) -> dict[str, object]:
        return {
            "accepted_commands": self._accepted_commands,
            "rejected_commands": self._rejected_commands,
            "heartbeat_timeouts": self._heartbeat_timeouts,
            "resets": self._resets,
            "latched": self.latched,
            "fault_reason": self._fault_reason,
            "last_command_sequence_id": self._last_command_sequence_id,
            "last_observation_sequence_id": self._last_observation_sequence_id,
            "last_evaluated_at_s": self._last_evaluated_at_s,
            "action_semantics_id": self.config.action_semantics_id,
            "action_semantics_sha256": self.config.action_semantics_sha256,
        }
