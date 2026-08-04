"""Runtime primitives for paired OpenPI pi0.5-LIBERO experiments.

This module intentionally does not import ``armbench`` or LIBERO. The official
OpenPI LIBERO image uses Python 3.8, while the main ArmBench package targets
Python 3.10. Keeping the runtime boundary small lets the same state machine run
inside the official container and in lightweight local tests.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import time
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ASYNC_UNGUARDED = "async_unguarded"
STATE_GUARD = "state_guard"
VALID_MODES = (ASYNC_UNGUARDED, STATE_GUARD)

LIBERO_ACTION_DIM = 7
LIBERO_STATE_DIM = 8
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)


class PolicyResponseError(ValueError):
    """Raised when a policy response violates the LIBERO action contract."""


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for one paired LIBERO runtime condition."""

    mode: str
    replan_steps: int
    latency_steps: int
    max_task_steps: int
    num_steps_wait: int = 10
    resize_size: int = 224
    position_threshold_m: float = 0.01
    orientation_threshold_rad: float = 0.10
    gripper_threshold: float = 0.05
    max_requeries: int = 2
    record_video: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError("mode must be one of: %s" % ", ".join(VALID_MODES))
        _require_positive_int("replan_steps", self.replan_steps)
        _require_nonnegative_int("latency_steps", self.latency_steps)
        _require_positive_int("max_task_steps", self.max_task_steps)
        _require_nonnegative_int("num_steps_wait", self.num_steps_wait)
        _require_positive_int("resize_size", self.resize_size)
        _require_nonnegative_int("max_requeries", self.max_requeries)
        for name, value in (
            ("position_threshold_m", self.position_threshold_m),
            ("orientation_threshold_rad", self.orientation_threshold_rad),
            ("gripper_threshold", self.gripper_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and nonnegative" % name)

    @property
    def environment_step_limit(self) -> int:
        return self.num_steps_wait + self.max_task_steps


@dataclasses.dataclass(frozen=True)
class RobotStateSnapshot:
    eef_position: np.ndarray
    eef_quaternion: np.ndarray
    gripper_position: np.ndarray


@dataclasses.dataclass(frozen=True)
class StateMismatch:
    position_m: float
    orientation_rad: float
    gripper_linf: float

    def rejection_reasons(self, config: RuntimeConfig) -> List[str]:
        reasons = []
        if self.position_m > config.position_threshold_m:
            reasons.append("position_mismatch")
        if self.orientation_rad > config.orientation_threshold_rad:
            reasons.append("orientation_mismatch")
        if self.gripper_linf > config.gripper_threshold:
            reasons.append("gripper_mismatch")
        return reasons

    def to_dict(self) -> Dict[str, float]:
        return {
            "position_m": self.position_m,
            "orientation_rad": self.orientation_rad,
            "gripper_linf": self.gripper_linf,
        }


@dataclasses.dataclass(frozen=True)
class QueryRecord:
    query_index: int
    observation_step: int
    response_step: int
    inference_latency_ms: float
    injected_latency_steps_requested: int
    injected_latency_steps_executed: int
    action_chunk_steps: int
    accepted: bool
    decision: str
    rejection_reasons: Tuple[str, ...]
    mismatch: Optional[StateMismatch]
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_index": self.query_index,
            "observation_step": self.observation_step,
            "response_step": self.response_step,
            "inference_latency_ms": self.inference_latency_ms,
            "injected_latency_steps_requested": self.injected_latency_steps_requested,
            "injected_latency_steps_executed": self.injected_latency_steps_executed,
            "action_chunk_steps": self.action_chunk_steps,
            "accepted": self.accepted,
            "decision": self.decision,
            "rejection_reasons": list(self.rejection_reasons),
            "mismatch": None if self.mismatch is None else self.mismatch.to_dict(),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclasses.dataclass
class EpisodeResult:
    success: bool
    termination_reason: str
    initial_state_sha256: str
    environment_steps: int
    task_action_steps: int
    latency_action_steps: int
    policy_queries: int
    accepted_chunks: int
    rejected_chunks: int
    stale_chunks_executed: int
    interventions: int
    query_records: List[QueryRecord]
    replay_frames: List[np.ndarray]
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "termination_reason": self.termination_reason,
            "initial_state_sha256": self.initial_state_sha256,
            "environment_steps": self.environment_steps,
            "task_action_steps": self.task_action_steps,
            "latency_action_steps": self.latency_action_steps,
            "policy_queries": self.policy_queries,
            "accepted_chunks": self.accepted_chunks,
            "rejected_chunks": self.rejected_chunks,
            "stale_chunks_executed": self.stale_chunks_executed,
            "interventions": self.interventions,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


RequestBuilder = Callable[[Mapping[str, Any], str, int], Dict[str, Any]]
FrameExtractor = Callable[[Mapping[str, Any], int], np.ndarray]


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)


def quat_to_axis_angle(quaternion: Sequence[float]) -> np.ndarray:
    """Match the official OpenPI LIBERO evaluator's xyzw conversion."""

    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("LIBERO end-effector quaternion must be finite with shape (4,)")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)


def extract_robot_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Build the exact 8-D state consumed by the official LIBERO policy."""

    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
        )
    )
    if state.shape != (LIBERO_STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError("LIBERO robot state must be finite with shape (8,)")
    return state


def snapshot_robot_state(observation: Mapping[str, Any]) -> RobotStateSnapshot:
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64).copy()
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("end-effector position must be finite with shape (3,)")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("end-effector quaternion must be finite with shape (4,)")
    if gripper.shape != (2,) or not np.all(np.isfinite(gripper)):
        raise ValueError("gripper position must be finite with shape (2,)")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= np.finfo(np.float64).eps:
        raise ValueError("end-effector quaternion must have nonzero norm")
    return RobotStateSnapshot(position, quaternion / quaternion_norm, gripper)


def state_mismatch(
    query_state: RobotStateSnapshot,
    current_state: RobotStateSnapshot,
) -> StateMismatch:
    position_m = float(
        np.linalg.norm(current_state.eef_position - query_state.eef_position)
    )
    quaternion_dot = float(
        np.dot(current_state.eef_quaternion, query_state.eef_quaternion)
    )
    orientation_rad = 2.0 * math.acos(np.clip(abs(quaternion_dot), 0.0, 1.0))
    gripper_linf = float(
        np.max(np.abs(current_state.gripper_position - query_state.gripper_position))
    )
    return StateMismatch(position_m, orientation_rad, gripper_linf)


def preprocess_libero_image(image: Any, resize_size: int) -> np.ndarray:
    """Apply the official 180-degree rotation and padded uint8 resize."""

    from openpi_client import image_tools

    rotated = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(rotated, resize_size, resize_size)
    )


def build_libero_request(
    observation: Mapping[str, Any],
    task_description: str,
    resize_size: int,
) -> Dict[str, Any]:
    """Build the official ``pi05_libero`` websocket request payload."""

    return {
        "observation/image": preprocess_libero_image(
            observation["agentview_image"], resize_size
        ),
        "observation/wrist_image": preprocess_libero_image(
            observation["robot0_eye_in_hand_image"], resize_size
        ),
        "observation/state": extract_robot_state(observation),
        "prompt": str(task_description),
    }


def extract_replay_frame(
    observation: Mapping[str, Any], resize_size: int
) -> np.ndarray:
    return preprocess_libero_image(observation["agentview_image"], resize_size)


def validate_action_chunk(response: Mapping[str, Any], replan_steps: int) -> np.ndarray:
    if not isinstance(response, Mapping) or "actions" not in response:
        raise PolicyResponseError("policy response must contain an 'actions' field")
    try:
        actions = np.asarray(response["actions"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyResponseError("policy actions must be numeric") from exc
    if actions.ndim != 2 or actions.shape[1] != LIBERO_ACTION_DIM:
        raise PolicyResponseError(
            "pi05_libero actions must have shape (horizon, 7), got %s"
            % (actions.shape,)
        )
    if actions.shape[0] < replan_steps:
        raise PolicyResponseError(
            "policy returned %d actions but replan_steps=%d"
            % (actions.shape[0], replan_steps)
        )
    if not np.all(np.isfinite(actions)):
        raise PolicyResponseError("policy actions must all be finite")
    return actions


def hold_action(previous_action: Sequence[float]) -> np.ndarray:
    """Stop Cartesian motion while preserving the last gripper command."""

    previous = np.asarray(previous_action, dtype=np.float64)
    if previous.shape != (LIBERO_ACTION_DIM,) or not np.all(np.isfinite(previous)):
        raise ValueError("previous action must be finite with shape (7,)")
    hold = np.zeros(LIBERO_ACTION_DIM, dtype=np.float64)
    hold[-1] = previous[-1]
    return hold


def initial_state_digest(initial_state: Any) -> str:
    array = np.ascontiguousarray(np.asarray(initial_state))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _environment_step(
    environment: Any, action: np.ndarray
) -> Tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
    result = environment.step(action.tolist())
    if not isinstance(result, tuple):
        raise TypeError("environment.step must return a tuple")
    if len(result) == 4:
        observation, reward, done, info = result
    elif len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        raise ValueError("environment.step must return 4 or 5 values")
    if not isinstance(observation, Mapping):
        raise TypeError("environment observation must be a mapping")
    if not isinstance(info, Mapping):
        info = {"raw_info": repr(info)}
    return observation, float(reward), bool(done), info


def run_episode(
    environment: Any,
    policy: Any,
    initial_state: Any,
    task_description: str,
    config: RuntimeConfig,
    request_builder: RequestBuilder = build_libero_request,
    frame_extractor: FrameExtractor = extract_replay_frame,
    clock: Callable[[], float] = time.perf_counter,
) -> EpisodeResult:
    """Run one LIBERO episode with an explicit asynchronous inference model.

    Inference remains a blocking websocket call, but the environment is advanced
    by ``latency_steps`` using the last commanded action before the response is
    consumed. This deterministic simulation models a controller continuing to
    run while an asynchronous policy query is pending.
    """

    digest = initial_state_digest(initial_state)
    query_records: List[QueryRecord] = []
    replay_frames: List[np.ndarray] = []
    action_plan: Deque[np.ndarray] = collections.deque()
    last_action = LIBERO_DUMMY_ACTION.copy()
    environment_steps = 0
    task_action_steps = 0
    latency_action_steps = 0
    accepted_chunks = 0
    rejected_chunks = 0
    stale_chunks_executed = 0
    consecutive_rejections = 0
    done = False

    def finish(
        success: bool,
        reason: str,
        failure_type: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            success=success,
            termination_reason=reason,
            initial_state_sha256=digest,
            environment_steps=environment_steps,
            task_action_steps=task_action_steps,
            latency_action_steps=latency_action_steps,
            policy_queries=len(query_records),
            accepted_chunks=accepted_chunks,
            rejected_chunks=rejected_chunks,
            stale_chunks_executed=stale_chunks_executed,
            interventions=rejected_chunks,
            query_records=query_records,
            replay_frames=replay_frames,
            failure_type=failure_type,
            failure_message=failure_message,
        )

    try:
        environment.reset()
        observation = environment.set_init_state(np.array(initial_state, copy=True))
        if not isinstance(observation, Mapping):
            raise TypeError("environment.set_init_state must return an observation mapping")

        for _ in range(config.num_steps_wait):
            if environment_steps >= config.environment_step_limit:
                break
            observation, _, done, _ = _environment_step(environment, LIBERO_DUMMY_ACTION)
            environment_steps += 1
            if done:
                return finish(True, "success_during_stabilization")

        while environment_steps < config.environment_step_limit:
            if config.record_video:
                replay_frames.append(frame_extractor(observation, config.resize_size))

            if not action_plan:
                query_index = len(query_records)
                observation_step = environment_steps
                query_state = snapshot_robot_state(observation)
                try:
                    request = request_builder(
                        observation, task_description, config.resize_size
                    )
                    inference_start = clock()
                    response = policy.infer(request)
                    inference_latency_ms = max(0.0, (clock() - inference_start) * 1000.0)
                    actions = validate_action_chunk(response, config.replan_steps)
                except Exception as exc:
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=0.0,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=0,
                            action_chunk_steps=0,
                            accepted=False,
                            decision="policy_or_observation_error",
                            rejection_reasons=(),
                            mismatch=None,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                    reason = (
                        "invalid_policy_response"
                        if isinstance(exc, PolicyResponseError)
                        else "infrastructure_failure"
                    )
                    return finish(False, reason, type(exc).__name__, str(exc))

                delay_steps_executed = 0
                for _ in range(config.latency_steps):
                    if environment_steps >= config.environment_step_limit:
                        break
                    observation, _, done, _ = _environment_step(environment, last_action)
                    environment_steps += 1
                    latency_action_steps += 1
                    delay_steps_executed += 1
                    if config.record_video:
                        replay_frames.append(
                            frame_extractor(observation, config.resize_size)
                        )
                    if done:
                        break

                mismatch = state_mismatch(
                    query_state, snapshot_robot_state(observation)
                )
                if done:
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=inference_latency_ms,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=delay_steps_executed,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="success_during_inference_delay",
                            rejection_reasons=(),
                            mismatch=mismatch,
                        )
                    )
                    return finish(True, "success_during_inference_delay")

                if environment_steps >= config.environment_step_limit:
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=inference_latency_ms,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=delay_steps_executed,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="step_budget_exhausted_during_delay",
                            rejection_reasons=(),
                            mismatch=mismatch,
                        )
                    )
                    return finish(False, "step_limit")

                rejection_reasons = mismatch.rejection_reasons(config)
                if config.mode == STATE_GUARD and rejection_reasons:
                    rejected_chunks += 1
                    consecutive_rejections += 1
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=inference_latency_ms,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=delay_steps_executed,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="rejected_state_mismatch",
                            rejection_reasons=tuple(rejection_reasons),
                            mismatch=mismatch,
                        )
                    )
                    last_action = hold_action(last_action)
                    if consecutive_rejections > config.max_requeries:
                        return finish(False, "max_requeries_exceeded")
                    continue

                consecutive_rejections = 0
                accepted_chunks += 1
                action_plan.extend(
                    np.asarray(action, dtype=np.float64).copy()
                    for action in actions[: config.replan_steps]
                )
                query_records.append(
                    QueryRecord(
                        query_index=query_index,
                        observation_step=observation_step,
                        response_step=environment_steps,
                        inference_latency_ms=inference_latency_ms,
                        injected_latency_steps_requested=config.latency_steps,
                        injected_latency_steps_executed=delay_steps_executed,
                        action_chunk_steps=int(actions.shape[0]),
                        accepted=True,
                        decision=(
                            "accepted_unguarded"
                            if config.mode == ASYNC_UNGUARDED
                            else "accepted_state_guard"
                        ),
                        rejection_reasons=tuple(rejection_reasons),
                        mismatch=mismatch,
                    )
                )
                if delay_steps_executed > 0:
                    stale_chunks_executed += 1

            action = action_plan.popleft()
            observation, _, done, _ = _environment_step(environment, action)
            last_action = action.copy()
            environment_steps += 1
            task_action_steps += 1
            if done:
                return finish(True, "task_success")

        return finish(False, "step_limit")
    except Exception as exc:
        return finish(False, "infrastructure_failure", type(exc).__name__, str(exc))
