from __future__ import annotations

import json

import numpy as np
import pytest

from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    PANDA_RUNTIME_ACTION_SPACE_ID,
    ActuatorCommandWatchdog,
    CommandWatchdogConfig,
    runtime_action_semantics,
)


def _action(gripper: float = 0.6) -> np.ndarray:
    action = np.linspace(-0.3, 0.3, 8)
    action[7] = gripper
    return action


def _evaluate(
    watchdog: ActuatorCommandWatchdog,
    *,
    action: object | None = None,
    command_sequence_id: int = 0,
    observation_sequence_id: int = 0,
    captured_at_s: float = 10.0,
    issued_at_s: float = 10.05,
    evaluated_at_s: float = 10.06,
    gripper_position: float = 0.4,
    action_semantics_id: str = PANDA_RUNTIME_ACTION_SPACE_ID,
    action_semantics_sha256: str = PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
):
    return watchdog.evaluate(
        _action() if action is None else action,
        command_sequence_id=command_sequence_id,
        observation_sequence_id=observation_sequence_id,
        captured_at_s=captured_at_s,
        issued_at_s=issued_at_s,
        evaluated_at_s=evaluated_at_s,
        gripper_position=gripper_position,
        action_semantics_id=action_semantics_id,
        action_semantics_sha256=action_semantics_sha256,
    )


def test_runtime_semantics_are_exact_and_hash_bound() -> None:
    semantics = runtime_action_semantics()

    assert semantics["action_space_id"] == PANDA_RUNTIME_ACTION_SPACE_ID
    assert semantics["action_dim"] == 8
    assert semantics["action_order"][-1] == "gripper_position_normalized"
    assert len(PANDA_RUNTIME_ACTION_SEMANTICS_SHA256) == 64
    semantics["action_order"][0] = "tampered"
    assert runtime_action_semantics()["action_order"][0].startswith("panda_joint1")


def test_valid_decision_is_json_serializable_and_deterministically_recomputed() -> None:
    first = _evaluate(ActuatorCommandWatchdog())
    second = _evaluate(ActuatorCommandWatchdog())

    assert first.status == "execute"
    assert not first.holding
    assert not first.action.flags.writeable
    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict(), allow_nan=False))["status"] == "execute"
    assert first.to_dict()["input_action_sha256"] == first.to_dict()[
        "output_action_sha256"
    ]


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(7),
        np.zeros((1, 8)),
        np.array([0.0] * 7 + [float("nan")]),
        "not-an-action",
    ],
)
def test_invalid_action_latches_deterministic_hold(action: object) -> None:
    watchdog = ActuatorCommandWatchdog()

    rejected = _evaluate(watchdog, action=action, gripper_position=0.35)
    still_latched = _evaluate(watchdog, command_sequence_id=1)

    assert rejected.status == "hold"
    assert rejected.reason == "invalid_action"
    assert rejected.fault_reason == "invalid_action"
    np.testing.assert_array_equal(
        rejected.action, np.array([0.0] * 7 + [0.35])
    )
    assert still_latched.reason == "fault_latched"
    assert still_latched.fault_reason == "invalid_action"


def test_command_sequence_strictly_increases_and_observation_may_repeat() -> None:
    watchdog = ActuatorCommandWatchdog()

    assert _evaluate(watchdog, command_sequence_id=4, observation_sequence_id=2).status == "execute"
    repeated_observation = _evaluate(
        watchdog,
        command_sequence_id=5,
        observation_sequence_id=2,
        captured_at_s=10.0,
        issued_at_s=10.06,
        evaluated_at_s=10.07,
    )
    duplicate_command = _evaluate(
        watchdog,
        command_sequence_id=5,
        observation_sequence_id=2,
        captured_at_s=10.0,
        issued_at_s=10.07,
        evaluated_at_s=10.08,
    )

    assert repeated_observation.status == "execute"
    assert duplicate_command.reason == "command_sequence_not_increasing"
    assert duplicate_command.latched


def test_observation_sequence_cannot_regress() -> None:
    watchdog = ActuatorCommandWatchdog()
    _evaluate(watchdog, command_sequence_id=1, observation_sequence_id=3)

    decision = _evaluate(
        watchdog,
        command_sequence_id=2,
        observation_sequence_id=2,
        captured_at_s=10.01,
        issued_at_s=10.06,
        evaluated_at_s=10.07,
    )

    assert decision.reason == "observation_sequence_regression"
    assert decision.holding


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"captured_at_s": 10.06}, "timestamp_order_violation"),
        ({"issued_at_s": 10.07}, "timestamp_order_violation"),
        ({"captured_at_s": float("nan")}, "invalid_timestamp"),
    ],
)
def test_timestamp_envelope_is_fail_closed(
    changes: dict[str, float], reason: str
) -> None:
    decision = _evaluate(ActuatorCommandWatchdog(), **changes)

    assert decision.reason == reason
    assert decision.latched
    json.dumps(decision.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("captured_at_s", 9.99, "capture_time_regression"),
        ("issued_at_s", 10.049, "issue_time_regression"),
        ("evaluated_at_s", 10.059, "evaluation_time_regression"),
    ],
)
def test_each_timestamp_stream_must_be_monotonic(
    field: str, value: float, reason: str
) -> None:
    watchdog = ActuatorCommandWatchdog()
    _evaluate(watchdog, command_sequence_id=0)
    kwargs = {
        "command_sequence_id": 1,
        "observation_sequence_id": 1,
        "captured_at_s": 10.01,
        "issued_at_s": 10.055,
        "evaluated_at_s": 10.07,
    }
    kwargs[field] = value

    decision = _evaluate(watchdog, **kwargs)

    assert decision.reason == reason


def test_observation_and_action_deadline_boundaries() -> None:
    config = CommandWatchdogConfig(
        max_observation_age_s=0.2,
        max_action_age_s=0.05,
        heartbeat_timeout_s=0.3,
    )
    at_boundary = _evaluate(
        ActuatorCommandWatchdog(config),
        captured_at_s=10.0,
        issued_at_s=10.15,
        evaluated_at_s=10.2,
    )
    stale_observation = _evaluate(
        ActuatorCommandWatchdog(config),
        captured_at_s=10.0,
        issued_at_s=10.151,
        evaluated_at_s=10.201,
    )
    stale_action = _evaluate(
        ActuatorCommandWatchdog(config),
        captured_at_s=10.15,
        issued_at_s=10.15,
        evaluated_at_s=10.201,
    )

    assert at_boundary.status == "execute"
    assert stale_observation.reason == "observation_deadline_exceeded"
    assert stale_action.reason == "action_deadline_exceeded"


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_semantics_id", "wrong.action.space"),
        ("action_semantics_sha256", "0" * 64),
    ],
)
def test_runtime_semantic_mismatch_is_rejected(field: str, value: str) -> None:
    decision = _evaluate(ActuatorCommandWatchdog(), **{field: value})

    assert decision.reason == "action_semantics_mismatch"
    assert decision.latched


def test_heartbeat_poll_is_quiet_until_timeout_then_latches_hold() -> None:
    watchdog = ActuatorCommandWatchdog(
        CommandWatchdogConfig(heartbeat_timeout_s=0.1)
    )
    _evaluate(watchdog, evaluated_at_s=10.06)

    assert watchdog.poll(evaluated_at_s=10.16, gripper_position=0.25) is None
    timeout = watchdog.poll(evaluated_at_s=10.161, gripper_position=0.25)

    assert timeout is not None
    assert timeout.reason == "heartbeat_timeout"
    assert timeout.latched
    np.testing.assert_array_equal(timeout.action, [0.0] * 7 + [0.25])
    assert watchdog.metrics()["heartbeat_timeouts"] == 1


def test_poll_before_first_command_holds_without_latching() -> None:
    watchdog = ActuatorCommandWatchdog()

    decision = watchdog.poll(evaluated_at_s=1.0, gripper_position=0.7)

    assert decision is not None
    assert decision.reason == "awaiting_first_command"
    assert not decision.latched


def test_explicit_reset_recovers_but_does_not_clear_replay_protection() -> None:
    watchdog = ActuatorCommandWatchdog()
    _evaluate(watchdog, command_sequence_id=3, observation_sequence_id=2)
    fault = _evaluate(
        watchdog,
        command_sequence_id=3,
        observation_sequence_id=2,
        captured_at_s=10.0,
        issued_at_s=10.07,
        evaluated_at_s=10.08,
    )
    assert fault.latched

    watchdog.reset(evaluated_at_s=11.0)
    old_queued = _evaluate(
        watchdog,
        command_sequence_id=4,
        observation_sequence_id=2,
        captured_at_s=10.9,
        issued_at_s=10.99,
        evaluated_at_s=11.01,
    )
    assert old_queued.reason == "command_predates_reset"

    watchdog.reset(evaluated_at_s=11.02)
    recovered = _evaluate(
        watchdog,
        command_sequence_id=4,
        observation_sequence_id=2,
        captured_at_s=11.02,
        issued_at_s=11.03,
        evaluated_at_s=11.04,
    )

    assert recovered.status == "execute"
    assert not recovered.latched
    assert watchdog.metrics()["resets"] == 2


def test_invalid_hold_gripper_uses_configured_fallback_and_latches() -> None:
    watchdog = ActuatorCommandWatchdog(
        CommandWatchdogConfig(fallback_gripper_position=0.8)
    )

    decision = _evaluate(watchdog, gripper_position=float("nan"))

    assert decision.reason == "invalid_hold_gripper"
    np.testing.assert_array_equal(decision.action, [0.0] * 7 + [0.8])


def test_reset_rejects_nonfinite_or_regressed_administrative_time() -> None:
    watchdog = ActuatorCommandWatchdog()
    _evaluate(watchdog)

    with pytest.raises(ValueError, match="reset time"):
        watchdog.reset(evaluated_at_s=float("nan"))
    with pytest.raises(ValueError, match="reset time"):
        watchdog.reset(evaluated_at_s=10.0)
