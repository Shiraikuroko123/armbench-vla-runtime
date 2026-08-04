"""Reference contracts for policy-internal real-time action chunking.

The overlap scheduler and prefix weights follow the public RTC evaluator at
commit 9296f31. Projected flow inpainting is a separate hard-conditioning
ablation for OpenPI's t=1 (noise) to t=0 (action) convention; it is not RTC's
pseudoinverse guidance method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


RTC_REFERENCE_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"
PrefixSchedule = Literal["linear", "exp", "ones", "zeros"]
VelocityFunction = Callable[[np.ndarray, float], np.ndarray]


class RealtimeChunkingError(ValueError):
    """Raised when an overlap or conditioning contract is unsafe or ambiguous."""


@dataclass(frozen=True)
class CommittedActionCondition:
    reference_actions: np.ndarray
    hard_prefix_mask: np.ndarray
    guidance_weights: np.ndarray
    inference_delay: int
    execute_horizon: int


@dataclass(frozen=True)
class OverlapPlan:
    inference_actions: np.ndarray
    response_actions: np.ndarray
    actions_to_execute: np.ndarray
    next_reference_actions: np.ndarray
    inference_delay: int
    execute_horizon: int


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RealtimeChunkingError("%s must be a positive integer" % name)
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RealtimeChunkingError("%s must be a nonnegative integer" % name)
    return value


def _finite_chunk(value: object, name: str) -> np.ndarray:
    try:
        chunk = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RealtimeChunkingError("%s must be a numeric action chunk" % name) from exc
    if chunk.ndim != 2 or chunk.shape[0] <= 0 or chunk.shape[1] <= 0:
        raise RealtimeChunkingError("%s must have shape (horizon, action_dim)" % name)
    if not np.all(np.isfinite(chunk)):
        raise RealtimeChunkingError("%s must contain only finite values" % name)
    return np.array(chunk, dtype=np.float64, order="C", copy=True)


def prefix_attention_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixSchedule = "exp",
) -> np.ndarray:
    """Reproduce RTC's prefix-attention schedule with strict host validation."""

    start = _nonnegative_integer(start, "start")
    end = _nonnegative_integer(end, "end")
    total = _positive_integer(total, "total")
    if start > total or end > total:
        raise RealtimeChunkingError("start and end must not exceed total")
    if schedule not in ("linear", "exp", "ones", "zeros"):
        raise RealtimeChunkingError("unsupported prefix schedule: %r" % schedule)

    effective_start = min(start, end)
    index = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float64)
    elif schedule == "zeros":
        weights = (index < effective_start).astype(np.float64)
    else:
        denominator = end - effective_start + 1
        weights = np.clip(
            (effective_start - 1 - index) / denominator + 1.0,
            0.0,
            1.0,
        )
        if schedule == "exp":
            weights = weights * np.expm1(weights) / np.expm1(1.0)
    return np.where(index >= end, 0.0, weights)


def build_committed_action_condition(
    reference_actions: object,
    *,
    inference_delay: int,
    execute_horizon: int,
    schedule: PrefixSchedule = "exp",
) -> CommittedActionCondition:
    """Build hard-prefix and RTC-guidance views of the same reference chunk."""

    reference = _finite_chunk(reference_actions, "reference_actions")
    delay = _nonnegative_integer(inference_delay, "inference_delay")
    execute = _positive_integer(execute_horizon, "execute_horizon")
    action_horizon = reference.shape[0]
    if delay > execute:
        raise RealtimeChunkingError("inference_delay must not exceed execute_horizon")
    if execute > action_horizon:
        raise RealtimeChunkingError("execute_horizon must not exceed action horizon")

    hard_mask = np.arange(action_horizon) < delay
    weights = prefix_attention_weights(
        delay,
        action_horizon - execute,
        action_horizon,
        schedule,
    )
    reference.setflags(write=False)
    hard_mask.setflags(write=False)
    weights.setflags(write=False)
    return CommittedActionCondition(
        reference_actions=reference,
        hard_prefix_mask=hard_mask,
        guidance_weights=weights,
        inference_delay=delay,
        execute_horizon=execute,
    )


def build_overlap_plan(
    previous_actions: object,
    response_actions: object,
    *,
    inference_delay: int,
    execute_horizon: int,
) -> OverlapPlan:
    """Build RTC's fixed-width old-prefix plus new-suffix execution window."""

    previous = _finite_chunk(previous_actions, "previous_actions")
    response = _finite_chunk(response_actions, "response_actions")
    if previous.shape != response.shape:
        raise RealtimeChunkingError("previous and response chunks must have equal shape")
    delay = _nonnegative_integer(inference_delay, "inference_delay")
    execute = _positive_integer(execute_horizon, "execute_horizon")
    action_horizon, action_dim = previous.shape
    if delay > execute:
        raise RealtimeChunkingError("inference_delay must not exceed execute_horizon")
    if execute > action_horizon:
        raise RealtimeChunkingError("execute_horizon must not exceed action horizon")

    during_inference = previous[:delay].copy()
    after_response = response[delay:execute].copy()
    actions_to_execute = np.concatenate(
        (during_inference, after_response), axis=0
    )
    if actions_to_execute.shape != (execute, action_dim):
        raise AssertionError("overlap scheduler changed the fixed execution horizon")
    next_reference = np.concatenate(
        (
            response[execute:],
            np.zeros((execute, action_dim), dtype=np.float64),
        ),
        axis=0,
    )

    for array in (
        during_inference,
        after_response,
        actions_to_execute,
        next_reference,
    ):
        array.setflags(write=False)
    return OverlapPlan(
        inference_actions=during_inference,
        response_actions=after_response,
        actions_to_execute=actions_to_execute,
        next_reference_actions=next_reference,
        inference_delay=delay,
        execute_horizon=execute,
    )


def _batched_condition_inputs(
    initial_noise: object,
    condition_actions: object,
    condition_mask: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    try:
        noise = np.asarray(initial_noise, dtype=np.float64)
        condition = np.asarray(condition_actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RealtimeChunkingError("noise and condition must be numeric arrays") from exc
    unbatched = noise.ndim == 2
    if unbatched:
        noise = noise[None, ...]
        condition = condition[None, ...]
    if noise.ndim != 3 or noise.shape != condition.shape:
        raise RealtimeChunkingError(
            "noise and condition must share shape (batch, horizon, action_dim)"
        )
    if min(noise.shape) <= 0:
        raise RealtimeChunkingError("noise and condition dimensions must be nonempty")
    if not np.all(np.isfinite(noise)) or not np.all(np.isfinite(condition)):
        raise RealtimeChunkingError("noise and condition must contain only finite values")

    mask = np.asarray(condition_mask)
    if unbatched and mask.ndim == 1:
        mask = mask[None, ...]
    if mask.shape != noise.shape[:2] or mask.dtype != np.bool_:
        raise RealtimeChunkingError("condition_mask must be boolean with shape (batch, horizon)")
    for row in mask:
        false_seen = np.maximum.accumulate(~row)
        if np.any(row & false_seen):
            raise RealtimeChunkingError("condition_mask must select a contiguous prefix")
    return (
        np.array(noise, dtype=np.float64, order="C", copy=True),
        np.array(condition, dtype=np.float64, order="C", copy=True),
        np.array(mask, dtype=np.bool_, order="C", copy=True),
        unbatched,
    )


def project_conditioned_state(
    state: np.ndarray,
    initial_noise: np.ndarray,
    condition_actions: np.ndarray,
    condition_mask: np.ndarray,
    time: float,
) -> np.ndarray:
    """Project known slots onto OpenPI's t*noise + (1-t)*action path."""

    if not np.isfinite(time) or time < 0.0 or time > 1.0:
        raise RealtimeChunkingError("time must be finite and within [0, 1]")
    known_state = time * initial_noise + (1.0 - time) * condition_actions
    return np.where(condition_mask[..., None], known_state, state)


def projected_flow_inpainting(
    initial_noise: object,
    condition_actions: object,
    condition_mask: object,
    velocity: VelocityFunction,
    *,
    num_steps: int = 10,
) -> np.ndarray:
    """Euler-sample with hard projected prefixes under OpenPI's reverse time."""

    steps = _positive_integer(num_steps, "num_steps")
    if not callable(velocity):
        raise RealtimeChunkingError("velocity must be callable")
    noise, condition, mask, unbatched = _batched_condition_inputs(
        initial_noise, condition_actions, condition_mask
    )
    state = noise.copy()
    time = 1.0
    dt = -1.0 / steps
    for step_index in range(steps):
        state = project_conditioned_state(state, noise, condition, mask, time)
        raw_velocity = velocity(state.copy(), time)
        try:
            velocity_array = np.asarray(raw_velocity, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RealtimeChunkingError("velocity output must be numeric") from exc
        if velocity_array.shape != state.shape:
            raise RealtimeChunkingError("velocity output shape does not match sampler state")
        if not np.all(np.isfinite(velocity_array)):
            raise RealtimeChunkingError("velocity output must contain only finite values")
        next_time = 0.0 if step_index == steps - 1 else max(0.0, time + dt)
        state = state + dt * velocity_array
        state = project_conditioned_state(
            state, noise, condition, mask, next_time
        )
        time = next_time
    output = state[0] if unbatched else state
    return np.array(output, dtype=np.float64, order="C", copy=True)
