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

from integrations.openpi.deadline_alignment import (
    CEIL,
    FAIL_CLOSED,
    AlignmentConfig,
    completed_control_steps,
    plan_alignment,
)


ASYNC_UNGUARDED = "async_unguarded"
LATENCY_ALIGNED = "latency_aligned"
STATE_GUARD = "state_guard"
FIXED_REFRESH = "fixed_refresh"
VALID_MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED, STATE_GUARD, FIXED_REFRESH)

FIXED_STEP_LATENCY = "fixed_steps"
MEASURED_WALL_LATENCY = "measured_wall"
VALID_LATENCY_SOURCES = (FIXED_STEP_LATENCY, MEASURED_WALL_LATENCY)

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
    fixed_refresh_interval: Optional[int] = None
    record_video: bool = False
    latency_source: str = FIXED_STEP_LATENCY
    control_period_ms: float = 50.0
    age_rounding: str = CEIL
    deadline_ms: Optional[float] = None
    max_age_refreshes: int = 2

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError("mode must be one of: %s" % ", ".join(VALID_MODES))
        _require_positive_int("replan_steps", self.replan_steps)
        _require_nonnegative_int("latency_steps", self.latency_steps)
        _require_positive_int("max_task_steps", self.max_task_steps)
        _require_nonnegative_int("num_steps_wait", self.num_steps_wait)
        _require_positive_int("resize_size", self.resize_size)
        _require_nonnegative_int("max_requeries", self.max_requeries)
        if self.fixed_refresh_interval is not None:
            _require_positive_int(
                "fixed_refresh_interval", self.fixed_refresh_interval
            )
        if self.mode == FIXED_REFRESH:
            if self.fixed_refresh_interval is None:
                raise ValueError(
                    "fixed_refresh_interval is required for fixed_refresh mode"
                )
            if self.max_requeries < 1:
                raise ValueError("fixed_refresh mode requires max_requeries >= 1")
        if self.latency_source not in VALID_LATENCY_SOURCES:
            raise ValueError(
                "latency_source must be one of: %s"
                % ", ".join(VALID_LATENCY_SOURCES)
            )
        if self.latency_source == MEASURED_WALL_LATENCY:
            if self.mode not in (ASYNC_UNGUARDED, LATENCY_ALIGNED):
                raise ValueError(
                    "measured_wall latency supports async_unguarded and "
                    "latency_aligned modes only"
                )
            if self.latency_steps != 0:
                raise ValueError(
                    "measured_wall latency requires latency_steps=0; use a "
                    "response jitter callback for controlled delay"
                )
        AlignmentConfig(
            control_period_ms=self.control_period_ms,
            rounding=self.age_rounding,
            deadline_ms=self.deadline_ms,
            max_refreshes=self.max_age_refreshes,
        )
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
    policy_inference_latency_ms: Optional[float] = None
    server_inference_latency_ms: Optional[float] = None
    error_stage: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_source: str = FIXED_STEP_LATENCY
    observation_age_ms: Optional[float] = None
    response_jitter_ms: float = 0.0
    measured_stale_steps: Optional[int] = None
    action_offset_steps: int = 0
    available_suffix_steps: Optional[int] = None
    deadline_exceeded: bool = False
    horizon_overrun: bool = False
    age_refresh_index: int = 0
    fallback_hold_steps: int = 0

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
            "policy_inference_latency_ms": self.policy_inference_latency_ms,
            "server_inference_latency_ms": self.server_inference_latency_ms,
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "latency_source": self.latency_source,
            "observation_age_ms": self.observation_age_ms,
            "response_jitter_ms": self.response_jitter_ms,
            "measured_stale_steps": self.measured_stale_steps,
            "action_offset_steps": self.action_offset_steps,
            "available_suffix_steps": self.available_suffix_steps,
            "deadline_exceeded": self.deadline_exceeded,
            "horizon_overrun": self.horizon_overrun,
            "age_refresh_index": self.age_refresh_index,
            "fallback_hold_steps": self.fallback_hold_steps,
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
    stale_action_steps: int
    interventions: int
    query_records: List[QueryRecord]
    replay_frames: List[np.ndarray]
    failure_category: Optional[str] = None
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None
    deadline_misses: int = 0
    horizon_overruns: int = 0
    age_refreshes: int = 0
    fallback_hold_steps: int = 0

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
            "stale_action_steps": self.stale_action_steps,
            "interventions": self.interventions,
            "deadline_misses": self.deadline_misses,
            "horizon_overruns": self.horizon_overruns,
            "age_refreshes": self.age_refreshes,
            "fallback_hold_steps": self.fallback_hold_steps,
            "failure_category": self.failure_category,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


RequestBuilder = Callable[[Mapping[str, Any], str, int], Dict[str, Any]]
FrameExtractor = Callable[[Mapping[str, Any], int], np.ndarray]
JitterProvider = Callable[[int], float]


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)


def _clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _elapsed_ms(start_s: float, end_s: float, label: str) -> float:
    elapsed = (end_s - start_s) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("monotonic clock moved backwards during %s" % label)
    return elapsed


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


def validate_action_chunk(response: Mapping[str, Any], required_steps: int) -> np.ndarray:
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
    if actions.shape[0] < required_steps:
        raise PolicyResponseError(
            "policy returned %d actions but required_action_steps=%d"
            % (actions.shape[0], required_steps)
        )
    if not np.all(np.isfinite(actions)):
        raise PolicyResponseError("policy actions must all be finite")
    return actions


def response_timing_ms(
    response: Mapping[str, Any], section: str
) -> Optional[float]:
    timing = response.get(section)
    if not isinstance(timing, Mapping):
        return None
    try:
        value = float(timing["infer_ms"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


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
    response_jitter_ms: Optional[JitterProvider] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> EpisodeResult:
    """Run one LIBERO episode with a post-response catch-up simulation.

    The websocket call remains blocking. In ``fixed_steps`` mode the environment
    is advanced by the registered delay after the response returns. In
    ``measured_wall`` mode full controller ticks are derived from end-to-end
    observation age. Both paths simulate a separately ticking controller; they
    do not provide a concurrent real-time watchdog or an OS scheduling guarantee.
    """

    digest = initial_state_digest(initial_state)
    query_records: List[QueryRecord] = []
    replay_frames: List[np.ndarray] = []
    action_plan: Deque[np.ndarray] = collections.deque()
    action_plan_stale: Deque[bool] = collections.deque()
    last_action = LIBERO_DUMMY_ACTION.copy()
    environment_steps = 0
    task_action_steps = 0
    latency_action_steps = 0
    accepted_chunks = 0
    rejected_chunks = 0
    stale_chunks_executed = 0
    stale_action_steps = 0
    consecutive_rejections = 0
    deadline_misses = 0
    horizon_overruns = 0
    age_refreshes = 0
    consecutive_age_refreshes = 0
    fallback_hold_steps = 0
    done = False
    alignment_config = AlignmentConfig(
        control_period_ms=config.control_period_ms,
        rounding=config.age_rounding,
        deadline_ms=config.deadline_ms,
        max_refreshes=config.max_age_refreshes,
    )

    def finish(
        success: bool,
        reason: str,
        failure_category: Optional[str] = None,
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
            stale_action_steps=stale_action_steps,
            interventions=rejected_chunks,
            deadline_misses=deadline_misses,
            horizon_overruns=horizon_overruns,
            age_refreshes=age_refreshes,
            fallback_hold_steps=fallback_hold_steps,
            query_records=query_records,
            replay_frames=replay_frames,
            failure_category=failure_category,
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

        while environment_steps < config.environment_step_limit:
            if config.record_video:
                replay_frames.append(frame_extractor(observation, config.resize_size))

            if not action_plan:
                query_index = len(query_records)
                observation_step = environment_steps
                failure_stage = "latency_measurement"
                inference_start: Optional[float] = None
                inference_latency_ms = 0.0
                observed_age_ms: Optional[float] = None
                jitter_ms = 0.0
                measured_stale_steps: Optional[int] = None
                environment_delay_steps = config.latency_steps
                try:
                    observation_captured_at = _clock_value(clock)
                    failure_stage = "observation_build"
                    query_state = snapshot_robot_state(observation)
                    request = request_builder(
                        observation, task_description, config.resize_size
                    )
                    failure_stage = "policy_inference"
                    inference_start = _clock_value(clock)
                    response = policy.infer(request)
                    failure_stage = "latency_measurement"
                    inference_finished = _clock_value(clock)
                    inference_latency_ms = _elapsed_ms(
                        inference_start,
                        inference_finished,
                        "policy inference",
                    )
                    failure_stage = "response_delivery"
                    if response_jitter_ms is not None:
                        jitter_ms = float(response_jitter_ms(query_index))
                        if not math.isfinite(jitter_ms) or jitter_ms < 0.0:
                            raise ValueError(
                                "response jitter must be finite and nonnegative"
                            )
                        if jitter_ms > 0.0:
                            sleeper(jitter_ms / 1000.0)
                    response_ready_at = _clock_value(clock)
                    observed_age_ms = _elapsed_ms(
                        observation_captured_at,
                        response_ready_at,
                        "observation age measurement",
                    )
                    if config.latency_source == MEASURED_WALL_LATENCY:
                        environment_delay_steps = completed_control_steps(
                            observed_age_ms, config.control_period_ms
                        )
                    failure_stage = "policy_response_validation"
                    required_action_steps = config.replan_steps + (
                        config.latency_steps
                        if (
                            config.mode == LATENCY_ALIGNED
                            and config.latency_source == FIXED_STEP_LATENCY
                        )
                        else 0
                    )
                    actions = validate_action_chunk(
                        response, required_action_steps
                    )
                    policy_inference_latency_ms = response_timing_ms(
                        response, "policy_timing"
                    )
                    server_inference_latency_ms = response_timing_ms(
                        response, "server_timing"
                    )
                    alignment_decision = None
                    if config.latency_source == MEASURED_WALL_LATENCY:
                        alignment_decision = plan_alignment(
                            observation_age_ms=observed_age_ms,
                            action_chunk_steps=int(actions.shape[0]),
                            replan_steps=config.replan_steps,
                            refresh_index=consecutive_age_refreshes,
                            config=alignment_config,
                        )
                        measured_stale_steps = alignment_decision.stale_steps
                        if alignment_decision.deadline_exceeded:
                            deadline_misses += 1
                        if alignment_decision.horizon_overrun:
                            horizon_overruns += 1
                except Exception as exc:
                    failed_latency_ms = inference_latency_ms
                    if failure_stage == "observation_build":
                        failure_category = "observation_contract"
                        termination_reason = "invalid_observation"
                    elif failure_stage == "policy_response_validation":
                        failure_category = "policy_contract"
                        termination_reason = "invalid_policy_response"
                    elif failure_stage in (
                        "latency_measurement",
                        "response_delivery",
                    ):
                        failure_category = "latency_measurement"
                        termination_reason = "invalid_latency_measurement"
                    elif isinstance(exc, TimeoutError):
                        failure_category = "policy_timeout"
                        termination_reason = "policy_timeout"
                    else:
                        failure_category = "policy_transport_or_server"
                        termination_reason = "policy_inference_failure"
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=failed_latency_ms,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=0,
                            action_chunk_steps=0,
                            accepted=False,
                            decision="%s_error" % failure_stage,
                            rejection_reasons=(),
                            mismatch=None,
                            policy_inference_latency_ms=None,
                            server_inference_latency_ms=None,
                            error_stage=failure_stage,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            latency_source=config.latency_source,
                            observation_age_ms=observed_age_ms,
                            response_jitter_ms=jitter_ms,
                        )
                    )
                    return finish(
                        False,
                        termination_reason,
                        failure_category,
                        type(exc).__name__,
                        str(exc),
                    )

                delay_steps_executed = 0
                for _ in range(environment_delay_steps):
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

                recorded_injected_steps = (
                    delay_steps_executed
                    if config.latency_source == FIXED_STEP_LATENCY
                    else 0
                )
                if alignment_decision is None:
                    timing_fields = {
                        "latency_source": config.latency_source,
                        "observation_age_ms": observed_age_ms,
                        "response_jitter_ms": jitter_ms,
                    }
                else:
                    aligned_mode = config.mode == LATENCY_ALIGNED
                    timing_fields = {
                        "latency_source": config.latency_source,
                        "observation_age_ms": observed_age_ms,
                        "response_jitter_ms": jitter_ms,
                        "measured_stale_steps": measured_stale_steps,
                        "action_offset_steps": (
                            alignment_decision.action_offset_steps
                            if aligned_mode
                            else 0
                        ),
                        "available_suffix_steps": (
                            alignment_decision.available_suffix_steps
                        ),
                        "deadline_exceeded": alignment_decision.deadline_exceeded,
                        "horizon_overrun": alignment_decision.horizon_overrun,
                        "age_refresh_index": consecutive_age_refreshes,
                    }

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
                            injected_latency_steps_executed=recorded_injected_steps,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="success_during_inference_delay",
                            rejection_reasons=(),
                            mismatch=mismatch,
                            policy_inference_latency_ms=policy_inference_latency_ms,
                            server_inference_latency_ms=server_inference_latency_ms,
                            **timing_fields,
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
                            injected_latency_steps_executed=recorded_injected_steps,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="step_budget_exhausted_during_delay",
                            rejection_reasons=(),
                            mismatch=mismatch,
                            policy_inference_latency_ms=policy_inference_latency_ms,
                            server_inference_latency_ms=server_inference_latency_ms,
                            **timing_fields,
                        )
                    )
                    return finish(False, "step_limit")

                if (
                    config.latency_source == MEASURED_WALL_LATENCY
                    and config.mode == LATENCY_ALIGNED
                    and alignment_decision is not None
                    and not alignment_decision.accepted
                ):
                    rejected_chunks += 1
                    reason = alignment_decision.reason
                    fail_closed = alignment_decision.disposition == FAIL_CLOSED
                    response_environment_step = environment_steps
                    last_action = hold_action(last_action)
                    query_hold_steps = 0
                    if environment_steps < config.environment_step_limit:
                        observation, _, done, _ = _environment_step(
                            environment, last_action
                        )
                        environment_steps += 1
                        fallback_hold_steps += 1
                        query_hold_steps = 1
                        if config.record_video:
                            replay_frames.append(
                                frame_extractor(observation, config.resize_size)
                            )
                    rejection_timing_fields = dict(timing_fields)
                    rejection_timing_fields["fallback_hold_steps"] = (
                        query_hold_steps
                    )
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=response_environment_step,
                            inference_latency_ms=inference_latency_ms,
                            injected_latency_steps_requested=0,
                            injected_latency_steps_executed=0,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision="rejected_%s_%s"
                            % (
                                reason,
                                "fail_closed" if fail_closed else "hold_refresh",
                            ),
                            rejection_reasons=(reason,),
                            mismatch=mismatch,
                            policy_inference_latency_ms=policy_inference_latency_ms,
                            server_inference_latency_ms=server_inference_latency_ms,
                            **rejection_timing_fields,
                        )
                    )
                    if done:
                        return finish(True, "success_during_fallback_hold")
                    if fail_closed:
                        if reason == "deadline_exceeded":
                            exhausted_reason = "deadline_refresh_exhausted"
                        elif reason == "horizon_overrun":
                            exhausted_reason = "stale_horizon_refresh_exhausted"
                        else:
                            exhausted_reason = (
                                "deadline_and_horizon_refresh_exhausted"
                            )
                        return finish(
                            False,
                            exhausted_reason,
                        )
                    age_refreshes += 1
                    consecutive_age_refreshes += 1
                    if environment_steps >= config.environment_step_limit:
                        return finish(False, "step_limit")
                    continue

                mismatch_reasons = mismatch.rejection_reasons(config)
                scheduled_refresh = (
                    config.mode == FIXED_REFRESH
                    and consecutive_rejections == 0
                    and (accepted_chunks + 1) % int(config.fixed_refresh_interval) == 0
                )
                should_reject = (
                    config.mode == STATE_GUARD and bool(mismatch_reasons)
                ) or scheduled_refresh
                rejection_reasons = (
                    ("scheduled_refresh",)
                    if scheduled_refresh
                    else tuple(mismatch_reasons)
                )
                if should_reject:
                    rejected_chunks += 1
                    consecutive_rejections += 1
                    query_records.append(
                        QueryRecord(
                            query_index=query_index,
                            observation_step=observation_step,
                            response_step=environment_steps,
                            inference_latency_ms=inference_latency_ms,
                            injected_latency_steps_requested=config.latency_steps,
                            injected_latency_steps_executed=recorded_injected_steps,
                            action_chunk_steps=int(actions.shape[0]),
                            accepted=False,
                            decision=(
                                "rejected_fixed_refresh"
                                if scheduled_refresh
                                else "rejected_state_mismatch"
                            ),
                            rejection_reasons=rejection_reasons,
                            mismatch=mismatch,
                            policy_inference_latency_ms=policy_inference_latency_ms,
                            server_inference_latency_ms=server_inference_latency_ms,
                            **timing_fields,
                        )
                    )
                    last_action = hold_action(last_action)
                    if consecutive_rejections > config.max_requeries:
                        return finish(False, "max_requeries_exceeded")
                    continue

                consecutive_rejections = 0
                consecutive_age_refreshes = 0
                accepted_chunks += 1
                action_offset = (
                    (
                        alignment_decision.action_offset_steps
                        if config.latency_source == MEASURED_WALL_LATENCY
                        else delay_steps_executed
                    )
                    if config.mode == LATENCY_ALIGNED
                    else 0
                )
                selected_actions = actions[
                    action_offset : action_offset + config.replan_steps
                ]
                action_plan.extend(
                    np.asarray(action, dtype=np.float64).copy()
                    for action in selected_actions
                )
                chunk_is_stale = (
                    measured_stale_steps > 0
                    if measured_stale_steps is not None
                    else delay_steps_executed > 0
                )
                action_plan_stale.extend(
                    chunk_is_stale for _ in range(config.replan_steps)
                )
                query_records.append(
                    QueryRecord(
                        query_index=query_index,
                        observation_step=observation_step,
                        response_step=environment_steps,
                        inference_latency_ms=inference_latency_ms,
                        injected_latency_steps_requested=config.latency_steps,
                        injected_latency_steps_executed=recorded_injected_steps,
                        action_chunk_steps=int(actions.shape[0]),
                        accepted=True,
                        decision=(
                            "accepted_unguarded"
                            if config.mode == ASYNC_UNGUARDED
                            else (
                                (
                                    "accepted_measured_latency_aligned"
                                    if config.latency_source
                                    == MEASURED_WALL_LATENCY
                                    else "accepted_latency_aligned"
                                )
                                if config.mode == LATENCY_ALIGNED
                                else (
                                    "accepted_state_guard"
                                    if config.mode == STATE_GUARD
                                    else "accepted_fixed_refresh"
                                )
                            )
                        ),
                        rejection_reasons=tuple(mismatch_reasons),
                        mismatch=mismatch,
                        policy_inference_latency_ms=policy_inference_latency_ms,
                        server_inference_latency_ms=server_inference_latency_ms,
                        **timing_fields,
                    )
                )
                if chunk_is_stale:
                    stale_chunks_executed += 1

            action = action_plan.popleft()
            action_is_stale = action_plan_stale.popleft()
            observation, _, done, _ = _environment_step(environment, action)
            last_action = action.copy()
            environment_steps += 1
            task_action_steps += 1
            if action_is_stale:
                stale_action_steps += 1
            if done:
                return finish(True, "task_success")

        return finish(False, "step_limit")
    except Exception as exc:
        return finish(
            False,
            "environment_runtime_failure",
            "environment_runtime",
            type(exc).__name__,
            str(exc),
        )
