from __future__ import annotations

import pytest

from integrations.openpi.deadline_alignment import (
    CEIL,
    EXECUTE,
    FAIL_CLOSED,
    FLOOR,
    HOLD_REFRESH,
    AlignmentConfig,
    completed_control_steps,
    estimate_stale_steps,
    keyed_discrete_jitter_ms,
    plan_alignment,
)


@pytest.mark.parametrize(
    "age_ms, floor_steps, ceil_steps",
    [
        (0.0, 0, 0),
        (49.999, 0, 1),
        (50.0, 1, 1),
        (50.001, 1, 2),
        (200.0, 4, 4),
    ],
)
def test_measured_age_boundaries_are_explicit(
    age_ms: float, floor_steps: int, ceil_steps: int
) -> None:
    assert completed_control_steps(age_ms, 50.0) == floor_steps
    assert estimate_stale_steps(age_ms, 50.0, FLOOR) == floor_steps
    assert estimate_stale_steps(age_ms, 50.0, CEIL) == ceil_steps


def test_alignment_selects_conservative_fresh_suffix() -> None:
    decision = plan_alignment(
        observation_age_ms=120.0,
        action_chunk_steps=10,
        replan_steps=5,
        refresh_index=0,
        config=AlignmentConfig(control_period_ms=50.0, rounding=CEIL),
    )

    assert decision.disposition == EXECUTE
    assert decision.stale_steps == 3
    assert decision.action_offset_steps == 3
    assert decision.selected_stop_step == 8
    assert decision.available_suffix_steps == 7
    assert not decision.deadline_exceeded
    assert not decision.horizon_overrun


def test_horizon_overrun_holds_then_fails_closed_at_refresh_bound() -> None:
    config = AlignmentConfig(max_refreshes=1)
    first = plan_alignment(
        observation_age_ms=251.0,
        action_chunk_steps=10,
        replan_steps=5,
        refresh_index=0,
        config=config,
    )
    final = plan_alignment(
        observation_age_ms=251.0,
        action_chunk_steps=10,
        replan_steps=5,
        refresh_index=1,
        config=config,
    )

    assert first.horizon_overrun
    assert first.disposition == HOLD_REFRESH
    assert final.disposition == FAIL_CLOSED
    assert final.reason == "horizon_overrun"


def test_deadline_is_independent_of_remaining_action_suffix() -> None:
    decision = plan_alignment(
        observation_age_ms=201.0,
        action_chunk_steps=20,
        replan_steps=1,
        refresh_index=0,
        config=AlignmentConfig(deadline_ms=200.0, max_refreshes=0),
    )

    assert decision.deadline_exceeded
    assert not decision.horizon_overrun
    assert decision.disposition == FAIL_CLOSED
    assert decision.reason == "deadline_exceeded"


def test_pair_keyed_jitter_is_mode_order_independent() -> None:
    key = ("libero_goal", 3, 17, 5)
    forward = [
        keyed_discrete_jitter_ms(
            seed=7,
            pairing_key=key,
            query_index=query,
            values_ms=(0.0, 40.0, 80.0, 160.0),
        )
        for query in range(12)
    ]
    reverse = {
        query: keyed_discrete_jitter_ms(
            seed=7,
            pairing_key=key,
            query_index=query,
            values_ms=(0.0, 40.0, 80.0, 160.0),
        )
        for query in reversed(range(12))
    }

    assert forward == [reverse[index] for index in range(12)]
    assert set(forward).issubset({0.0, 40.0, 80.0, 160.0})
    assert forward != [
        keyed_discrete_jitter_ms(
            seed=8,
            pairing_key=key,
            query_index=query,
            values_ms=(0.0, 40.0, 80.0, 160.0),
        )
        for query in range(12)
    ]


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: estimate_stale_steps(float("nan"), 50.0), "observation_age_ms"),
        (lambda: estimate_stale_steps(-1.0, 50.0), "observation_age_ms"),
        (lambda: estimate_stale_steps(1.0, 0.0), "control_period_ms"),
        (
            lambda: AlignmentConfig(rounding="nearest"),
            "rounding",
        ),
        (
            lambda: keyed_discrete_jitter_ms(
                seed=7, pairing_key=("suite",), query_index=0, values_ms=()
            ),
            "values_ms",
        ),
    ],
)
def test_invalid_timing_inputs_fail_closed(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
