"""Pure helpers for non-oracle, deadline-aware action-chunk alignment.

The official LIBERO runtime also supports a legacy fixed-step protocol.  This
module deliberately has no LIBERO or OpenPI imports so the measured-age policy
can be tested independently of either stack and reused by a future robot
adapter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Optional, Sequence, Tuple


CEIL = "ceil"
FLOOR = "floor"
VALID_ROUNDING = (CEIL, FLOOR)

EXECUTE = "execute"
HOLD_REFRESH = "hold_refresh"
FAIL_CLOSED = "fail_closed"


@dataclasses.dataclass(frozen=True)
class AlignmentConfig:
    """Frozen decision parameters for one temporal-alignment protocol."""

    control_period_ms: float = 50.0
    rounding: str = CEIL
    deadline_ms: Optional[float] = None
    max_refreshes: int = 2
    boundary_tolerance_ms: float = 1e-9

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.control_period_ms)
            or self.control_period_ms <= 0.0
        ):
            raise ValueError("control_period_ms must be finite and positive")
        if self.rounding not in VALID_ROUNDING:
            raise ValueError("rounding must be one of: %s" % ", ".join(VALID_ROUNDING))
        if self.deadline_ms is not None and (
            not math.isfinite(self.deadline_ms) or self.deadline_ms < 0.0
        ):
            raise ValueError("deadline_ms must be finite and nonnegative")
        if (
            isinstance(self.max_refreshes, bool)
            or not isinstance(self.max_refreshes, int)
            or self.max_refreshes < 0
        ):
            raise ValueError("max_refreshes must be a nonnegative integer")
        if (
            not math.isfinite(self.boundary_tolerance_ms)
            or self.boundary_tolerance_ms < 0.0
        ):
            raise ValueError(
                "boundary_tolerance_ms must be finite and nonnegative"
            )


@dataclasses.dataclass(frozen=True)
class AlignmentDecision:
    """Auditable action selection or fail-closed decision for one response."""

    observation_age_ms: float
    stale_steps: int
    action_chunk_steps: int
    replan_steps: int
    action_offset_steps: int
    selected_stop_step: int
    available_suffix_steps: int
    deadline_exceeded: bool
    horizon_overrun: bool
    refresh_index: int
    disposition: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.disposition == EXECUTE

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _finite_age_ms(observation_age_ms: float) -> float:
    try:
        age = float(observation_age_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_age_ms must be numeric") from exc
    if not math.isfinite(age) or age < 0.0:
        raise ValueError("observation_age_ms must be finite and nonnegative")
    return age


def completed_control_steps(
    observation_age_ms: float,
    control_period_ms: float,
    boundary_tolerance_ms: float = 1e-9,
) -> int:
    """Return full controller ticks elapsed while a response was unavailable."""

    age = _finite_age_ms(observation_age_ms)
    if not math.isfinite(control_period_ms) or control_period_ms <= 0.0:
        raise ValueError("control_period_ms must be finite and positive")
    if not math.isfinite(boundary_tolerance_ms) or boundary_tolerance_ms < 0.0:
        raise ValueError("boundary_tolerance_ms must be finite and nonnegative")
    return int(math.floor((age + boundary_tolerance_ms) / control_period_ms))


def estimate_stale_steps(
    observation_age_ms: float,
    control_period_ms: float,
    rounding: str = CEIL,
    boundary_tolerance_ms: float = 1e-9,
) -> int:
    """Convert measured observation age to a pre-registered discrete offset."""

    age = _finite_age_ms(observation_age_ms)
    if not math.isfinite(control_period_ms) or control_period_ms <= 0.0:
        raise ValueError("control_period_ms must be finite and positive")
    if rounding not in VALID_ROUNDING:
        raise ValueError("rounding must be one of: %s" % ", ".join(VALID_ROUNDING))
    if not math.isfinite(boundary_tolerance_ms) or boundary_tolerance_ms < 0.0:
        raise ValueError("boundary_tolerance_ms must be finite and nonnegative")
    if rounding == FLOOR:
        return completed_control_steps(
            age, control_period_ms, boundary_tolerance_ms
        )
    if age <= boundary_tolerance_ms:
        return 0
    return int(
        math.ceil((age - boundary_tolerance_ms) / control_period_ms)
    )


def plan_alignment(
    *,
    observation_age_ms: float,
    action_chunk_steps: int,
    replan_steps: int,
    refresh_index: int,
    config: AlignmentConfig,
) -> AlignmentDecision:
    """Select a fresh suffix or request a bounded fail-closed refresh."""

    for name, value, allow_zero in (
        ("action_chunk_steps", action_chunk_steps, False),
        ("replan_steps", replan_steps, False),
        ("refresh_index", refresh_index, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if allow_zero else 1)
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError("%s must be a %s integer" % (name, qualifier))

    age = _finite_age_ms(observation_age_ms)
    stale_steps = estimate_stale_steps(
        age,
        config.control_period_ms,
        config.rounding,
        config.boundary_tolerance_ms,
    )
    selected_stop = stale_steps + replan_steps
    available_suffix = max(0, action_chunk_steps - stale_steps)
    deadline_exceeded = (
        config.deadline_ms is not None
        and age > config.deadline_ms + config.boundary_tolerance_ms
    )
    horizon_overrun = selected_stop > action_chunk_steps

    if deadline_exceeded or horizon_overrun:
        disposition = (
            HOLD_REFRESH if refresh_index < config.max_refreshes else FAIL_CLOSED
        )
        if deadline_exceeded and horizon_overrun:
            reason = "deadline_and_horizon_overrun"
        elif deadline_exceeded:
            reason = "deadline_exceeded"
        else:
            reason = "horizon_overrun"
    else:
        disposition = EXECUTE
        reason = "fresh_suffix_available"

    return AlignmentDecision(
        observation_age_ms=age,
        stale_steps=stale_steps,
        action_chunk_steps=action_chunk_steps,
        replan_steps=replan_steps,
        action_offset_steps=stale_steps,
        selected_stop_step=selected_stop,
        available_suffix_steps=available_suffix,
        deadline_exceeded=deadline_exceeded,
        horizon_overrun=horizon_overrun,
        refresh_index=refresh_index,
        disposition=disposition,
        reason=reason,
    )


def keyed_discrete_jitter_ms(
    *,
    seed: int,
    pairing_key: Sequence[Any],
    query_index: int,
    values_ms: Sequence[float],
) -> float:
    """Sample paired jitter without sharing mutable RNG state across modes.

    ``pairing_key`` must omit the runtime mode.  The same task/episode/query
    therefore receives the same jitter regardless of execution order.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(query_index, bool)
        or not isinstance(query_index, int)
        or query_index < 0
    ):
        raise ValueError("query_index must be a nonnegative integer")
    values: Tuple[float, ...] = tuple(float(value) for value in values_ms)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("values_ms must be finite, nonnegative, and nonempty")
    try:
        payload = json.dumps(
            {
                "seed": seed,
                "pairing_key": list(pairing_key),
                "query_index": query_index,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("pairing_key must contain JSON-serializable values") from exc
    digest = hashlib.sha256(payload).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]
