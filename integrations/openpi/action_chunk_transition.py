"""Model-independent action-chunk transition capture and serialization."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


TRANSITION_SCHEMA_VERSION = "armbench.action_chunk_transition.v1"
ARCHIVE_SCHEMA_VERSION = "armbench.action_chunk_transition_archive.v1"
SOURCE_PADDING = np.uint8(0)
SOURCE_OLD_PREFIX = np.uint8(1)
SOURCE_NEW_SUFFIX = np.uint8(2)
ARCHIVE_KEYS = frozenset(
    {
        "has_previous_reference",
        "previous_reference",
        "response_actions",
        "executed_window",
        "action_source",
        "next_reference",
        "executed_length",
        "old_prefix_steps",
        "new_suffix_steps",
    }
)


class ActionChunkTransitionError(ValueError):
    pass


def canonical_action_sha256(actions: object) -> str:
    canonical = np.asarray(actions, dtype="<f4", order="C")
    if canonical.ndim != 2 or not np.all(np.isfinite(canonical)):
        raise ActionChunkTransitionError("actions must be a finite two-dimensional array")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_chunk(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="<f4", order="C")
    except (TypeError, ValueError) as exc:
        raise ActionChunkTransitionError(f"{name} must be numeric") from exc
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ActionChunkTransitionError(f"{name} must be a finite two-dimensional array")
    return np.array(array, dtype="<f4", order="C", copy=True)


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ActionChunkTransitionError(f"{name} must be a nonnegative integer")
    return value


@dataclasses.dataclass(frozen=True)
class ActionChunkTransition:
    has_previous_reference: bool
    previous_reference: np.ndarray
    response_actions: np.ndarray
    executed_window: np.ndarray
    action_source: np.ndarray
    next_reference: np.ndarray
    executed_length: int
    old_prefix_steps: int
    new_suffix_steps: int

    def __post_init__(self) -> None:
        for array in (
            self.previous_reference,
            self.response_actions,
            self.executed_window,
            self.action_source,
            self.next_reference,
        ):
            array.setflags(write=False)


def build_action_chunk_transition(
    previous_reference: Optional[object],
    response_actions: object,
    next_reference: object,
    *,
    inference_delay: int,
    execute_horizon: int,
    executed_old: int,
    executed_new: int,
) -> ActionChunkTransition:
    """Capture the exact scheduler transition, including partial execution."""

    delay = _count(inference_delay, "inference_delay")
    execute = _count(execute_horizon, "execute_horizon")
    if execute == 0:
        raise ActionChunkTransitionError("execute_horizon must be positive")
    if delay > execute:
        raise ActionChunkTransitionError("inference_delay must not exceed execute_horizon")
    old_count = _count(executed_old, "executed_old")
    new_count = _count(executed_new, "executed_new")
    response = _canonical_chunk(response_actions, "response_actions")
    horizon, action_dim = response.shape
    if execute > horizon:
        raise ActionChunkTransitionError("execute_horizon exceeds the action horizon")

    has_previous = previous_reference is not None
    if has_previous:
        previous = _canonical_chunk(previous_reference, "previous_reference")
        if previous.shape != response.shape:
            raise ActionChunkTransitionError("previous and response chunks must have equal shape")
        if old_count > delay or new_count > execute - delay:
            raise ActionChunkTransitionError("executed phase count exceeds its scheduled segment")
        if old_count < delay and new_count:
            raise ActionChunkTransitionError("new suffix cannot execute before the old prefix completes")
        scheduled = np.concatenate((previous[:delay], response[delay:execute]), axis=0)
        sources = np.concatenate(
            (
                np.full(old_count, SOURCE_OLD_PREFIX, dtype=np.uint8),
                np.full(new_count, SOURCE_NEW_SUFFIX, dtype=np.uint8),
            )
        )
    else:
        previous = np.zeros_like(response)
        if old_count:
            raise ActionChunkTransitionError("bootstrap transition cannot execute an old prefix")
        if new_count > execute:
            raise ActionChunkTransitionError("bootstrap execution exceeds execute_horizon")
        scheduled = response[:execute]
        sources = np.full(new_count, SOURCE_NEW_SUFFIX, dtype=np.uint8)

    executed_length = old_count + new_count
    executed_window = np.zeros((execute, action_dim), dtype="<f4")
    executed_window[:executed_length] = scheduled[:executed_length]
    action_source = np.full(execute, SOURCE_PADDING, dtype=np.uint8)
    action_source[:executed_length] = sources

    expected_next = np.concatenate(
        (
            response[execute:],
            np.zeros((execute, action_dim), dtype="<f4"),
        ),
        axis=0,
    )
    next_chunk = _canonical_chunk(next_reference, "next_reference")
    if next_chunk.shape != response.shape or not np.array_equal(next_chunk, expected_next):
        raise ActionChunkTransitionError("next_reference does not match the scheduler shift")

    return ActionChunkTransition(
        has_previous_reference=has_previous,
        previous_reference=previous,
        response_actions=response,
        executed_window=executed_window,
        action_source=action_source,
        next_reference=next_chunk,
        executed_length=executed_length,
        old_prefix_steps=old_count,
        new_suffix_steps=new_count,
    )


def write_transition_archive(
    path: pathlib.Path, transitions: Sequence[ActionChunkTransition]
) -> None:
    if not transitions:
        raise ActionChunkTransitionError("transition archive cannot be empty")
    first = transitions[0]
    horizon, action_dim = first.response_actions.shape
    execute = first.executed_window.shape[0]
    for transition in transitions:
        if transition.response_actions.shape != (horizon, action_dim):
            raise ActionChunkTransitionError("archive transitions have different action shapes")
        if transition.executed_window.shape != (execute, action_dim):
            raise ActionChunkTransitionError("archive transitions have different execute horizons")

    arrays: Dict[str, np.ndarray] = {
        "has_previous_reference": np.asarray(
            [row.has_previous_reference for row in transitions], dtype=np.bool_
        ),
        "previous_reference": np.stack([row.previous_reference for row in transitions]),
        "response_actions": np.stack([row.response_actions for row in transitions]),
        "executed_window": np.stack([row.executed_window for row in transitions]),
        "action_source": np.stack([row.action_source for row in transitions]),
        "next_reference": np.stack([row.next_reference for row in transitions]),
        "executed_length": np.asarray(
            [row.executed_length for row in transitions], dtype="<i4"
        ),
        "old_prefix_steps": np.asarray(
            [row.old_prefix_steps for row in transitions], dtype="<i4"
        ),
        "new_suffix_steps": np.asarray(
            [row.new_suffix_steps for row in transitions], dtype="<i4"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_transition_archive(path: pathlib.Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != ARCHIVE_KEYS:
                raise ActionChunkTransitionError("transition archive fields do not match the schema")
            return {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except ActionChunkTransitionError:
        raise
    except Exception as exc:
        raise ActionChunkTransitionError("cannot read transition archive") from exc


def validate_transition_arrays(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    queries: Sequence[Mapping[str, Any]],
    *,
    bootstrap_response_sha256_by_episode: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Recompute scheduler equations and cross-query reference chains."""

    if descriptor.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ActionChunkTransitionError("transition descriptor schema mismatch")
    scheduler = descriptor.get("scheduler")
    rows = descriptor.get("rows")
    if not isinstance(scheduler, Mapping) or not isinstance(rows, list):
        raise ActionChunkTransitionError("transition descriptor sections are missing")
    horizon = int(scheduler.get("action_horizon", -1))
    action_dim = int(scheduler.get("action_dim", -1))
    execute = int(scheduler.get("execute_horizon", -1))
    delay = int(scheduler.get("inference_delay", -1))
    if horizon <= 0 or action_dim <= 0 or execute <= 0 or delay < 0 or delay > execute:
        raise ActionChunkTransitionError("transition scheduler dimensions are invalid")
    count = len(rows)
    if count == 0 or len(queries) != count or set(arrays) != ARCHIVE_KEYS:
        raise ActionChunkTransitionError("transition row count or archive fields mismatch")
    expected_shapes = {
        "has_previous_reference": (count,),
        "previous_reference": (count, horizon, action_dim),
        "response_actions": (count, horizon, action_dim),
        "executed_window": (count, execute, action_dim),
        "action_source": (count, execute),
        "next_reference": (count, horizon, action_dim),
        "executed_length": (count,),
        "old_prefix_steps": (count,),
        "new_suffix_steps": (count,),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise ActionChunkTransitionError(f"transition array shape mismatch: {key}")
    for key in ("previous_reference", "response_actions", "executed_window", "next_reference"):
        value = np.asarray(arrays[key])
        if value.dtype != np.dtype("<f4") or not np.all(np.isfinite(value)):
            raise ActionChunkTransitionError(f"transition action array is not finite float32: {key}")
    if np.asarray(arrays["has_previous_reference"]).dtype != np.bool_:
        raise ActionChunkTransitionError("has_previous_reference must be boolean")
    if np.asarray(arrays["action_source"]).dtype != np.uint8:
        raise ActionChunkTransitionError("action_source must be uint8")
    for key in ("executed_length", "old_prefix_steps", "new_suffix_steps"):
        if np.asarray(arrays[key]).dtype != np.dtype("<i4"):
            raise ActionChunkTransitionError(f"transition count array must be int32: {key}")

    previous_by_episode: Dict[str, np.ndarray] = {}
    for index, (row, query) in enumerate(zip(rows, queries)):
        if not isinstance(row, Mapping) or not isinstance(query, Mapping):
            raise ActionChunkTransitionError("transition rows and queries must be objects")
        for key in ("episode_id", "pair_id", "method", "query_index"):
            if row.get(key) != query.get(key):
                raise ActionChunkTransitionError(f"transition/query identity mismatch: {key}")
        episode_id = str(row["episode_id"])
        bootstrap = bool(query.get("bootstrap"))
        has_previous = bool(arrays["has_previous_reference"][index])
        previous = arrays["previous_reference"][index]
        response = arrays["response_actions"][index]
        executed = arrays["executed_window"][index]
        source = arrays["action_source"][index]
        next_reference = arrays["next_reference"][index]
        length = int(arrays["executed_length"][index])
        old_count = int(arrays["old_prefix_steps"][index])
        new_count = int(arrays["new_suffix_steps"][index])
        if (length, old_count, new_count) != (
            int(query.get("executed_steps", -1)),
            int(query.get("old_prefix_steps", -1)),
            int(query.get("new_suffix_steps", -1)),
        ):
            raise ActionChunkTransitionError("transition execution counts mismatch query")
        if length != old_count + new_count or length < 0 or length > execute:
            raise ActionChunkTransitionError("transition execution counts are inconsistent")

        if bootstrap:
            if has_previous or episode_id in previous_by_episode or np.any(previous):
                raise ActionChunkTransitionError("bootstrap reference contract failed")
            scheduled = response[:execute]
            expected_source = np.full(length, SOURCE_NEW_SUFFIX, dtype=np.uint8)
        else:
            if not has_previous:
                raise ActionChunkTransitionError("nonbootstrap transition lacks its reference chain")
            if episode_id not in previous_by_episode:
                bootstrap_hash = (
                    None
                    if bootstrap_response_sha256_by_episode is None
                    else bootstrap_response_sha256_by_episode.get(episode_id)
                )
                if canonical_action_sha256(previous) != bootstrap_hash:
                    raise ActionChunkTransitionError(
                        "nonbootstrap transition lacks its reference-only bootstrap"
                    )
            elif not np.array_equal(previous, previous_by_episode[episode_id]):
                raise ActionChunkTransitionError("cross-query reference chain mismatch")
            scheduled = np.concatenate((previous[:delay], response[delay:execute]), axis=0)
            expected_source = np.concatenate(
                (
                    np.full(old_count, SOURCE_OLD_PREFIX, dtype=np.uint8),
                    np.full(new_count, SOURCE_NEW_SUFFIX, dtype=np.uint8),
                )
            )
        if not np.array_equal(executed[:length], scheduled[:length]) or np.any(executed[length:]):
            raise ActionChunkTransitionError("executed window does not match overlap scheduling")
        if not np.array_equal(source[:length], expected_source) or np.any(source[length:]):
            raise ActionChunkTransitionError("action source labels do not match execution")
        expected_next = np.concatenate(
            (response[execute:], np.zeros((execute, action_dim), dtype="<f4")), axis=0
        )
        if not np.array_equal(next_reference, expected_next):
            raise ActionChunkTransitionError("next reference does not match response shift")
        if canonical_action_sha256(response) != query.get("response_action_sha256"):
            raise ActionChunkTransitionError("response action hash mismatch")
        if canonical_action_sha256(next_reference) != query.get("next_reference_sha256"):
            raise ActionChunkTransitionError("next reference hash mismatch")
        previous_by_episode[episode_id] = next_reference

    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "valid": True,
        "transition_count": count,
        "episode_count": len(previous_by_episode),
        "action_horizon": horizon,
        "action_dim": action_dim,
        "execute_horizon": execute,
        "inference_delay": delay,
    }
