"""Closed-loop LIBERO runtime for fair pi0.5 overlap comparisons."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.action_chunk_transition import (
    ActionChunkTransition,
    build_action_chunk_transition,
    canonical_action_sha256,
)
from integrations.openpi.libero_runtime import (
    LIBERO_ACTION_DIM,
    LIBERO_DUMMY_ACTION,
    build_libero_request,
    extract_replay_frame,
    initial_state_digest,
    response_timing_ms,
    validate_action_chunk,
)
from integrations.openpi.realtime_chunking import build_overlap_plan
from integrations.openpi.serve_policy_attested import (
    POLICY_CONDITIONING_METHOD,
    POLICY_CONDITIONING_REQUEST_FIELD,
    POLICY_CONDITIONING_RESPONSE_FIELD,
    POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
    POLICY_SAMPLING_GENERATOR,
    POLICY_SAMPLING_REQUEST_FIELD,
    POLICY_SAMPLING_RESPONSE_FIELD,
    POLICY_SAMPLING_SCHEMA_VERSION,
    build_policy_conditioning_control,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)


OVERLAP_UNCONDITIONED = "overlap_unconditioned"
PROJECTED_OVERLAP = "projected_overlap"
VALID_OVERLAP_METHODS = (OVERLAP_UNCONDITIONED, PROJECTED_OVERLAP)
SamplingControlBuilder = Callable[[int], Mapping[str, Any]]


class OverlapRuntimeError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class OverlapRuntimeConfig:
    method: str
    execute_horizon: int = 5
    inference_delay_steps: int = 4
    action_horizon: int = 10
    max_task_steps: int = 520
    num_steps_wait: int = 10
    resize_size: int = 224
    record_video: bool = False
    max_condition_residual: float = 1e-6

    def __post_init__(self) -> None:
        if self.method not in VALID_OVERLAP_METHODS:
            raise ValueError("unsupported overlap method")
        for name, value in (
            ("execute_horizon", self.execute_horizon),
            ("action_horizon", self.action_horizon),
            ("max_task_steps", self.max_task_steps),
            ("resize_size", self.resize_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if type(self.inference_delay_steps) is not int or self.inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be a nonnegative integer")
        if type(self.num_steps_wait) is not int or self.num_steps_wait < 0:
            raise ValueError("num_steps_wait must be a nonnegative integer")
        if self.inference_delay_steps > self.execute_horizon:
            raise ValueError("inference_delay_steps must not exceed execute_horizon")
        if self.execute_horizon > self.action_horizon:
            raise ValueError("execute_horizon must not exceed action_horizon")
        if self.action_horizon != 10:
            raise ValueError("pi0.5 LIBERO overlap requires action_horizon=10")
        if not math.isfinite(self.max_condition_residual) or self.max_condition_residual <= 0:
            raise ValueError("max_condition_residual must be finite and positive")


@dataclasses.dataclass(frozen=True)
class OverlapQueryRecord:
    query_index: int
    bootstrap: bool
    observation_step: int
    response_step: int
    completed_step: int
    inference_latency_ms: float
    policy_inference_latency_ms: Optional[float]
    server_inference_latency_ms: Optional[float]
    response_action_sha256: str
    next_reference_sha256: str
    sampling_key_sha256: Optional[str]
    sampling_noise_sha256: Optional[str]
    condition_raw_actions_sha256: Optional[str]
    condition_model_actions_sha256: Optional[str]
    condition_mask_sha256: Optional[str]
    max_model_residual: Optional[float]
    old_prefix_steps: int
    new_suffix_steps: int
    executed_steps: int
    seam_motion_l2: Optional[float]
    seam_gripper_abs: Optional[float]
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class OverlapEpisodeResult:
    success: bool
    termination_reason: str
    initial_state_sha256: str
    environment_steps: int
    task_action_steps: int
    policy_queries: int
    bootstrap_queries: int
    conditioned_queries: int
    query_records: List[OverlapQueryRecord]
    transition_records: List[ActionChunkTransition]
    replay_frames: List[np.ndarray]
    failure_stage: Optional[str] = None
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = dataclasses.asdict(self)
        value.pop("replay_frames")
        value.pop("transition_records")
        return value


def _environment_step(
    environment: Any, action: np.ndarray
) -> Tuple[Mapping[str, Any], bool]:
    result = environment.step(np.asarray(action, dtype=np.float64).tolist())
    if not isinstance(result, tuple):
        raise TypeError("environment.step must return a tuple")
    if len(result) == 4:
        observation, _, done, _ = result
    elif len(result) == 5:
        observation, _, terminated, truncated, _ = result
        done = bool(terminated) or bool(truncated)
    else:
        raise ValueError("environment.step must return 4 or 5 values")
    if not isinstance(observation, Mapping):
        raise TypeError("environment observation must be a mapping")
    return observation, bool(done)


def _validate_sampling_trace(
    response: Mapping[str, Any], control: Optional[Mapping[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    if control is None:
        if POLICY_SAMPLING_RESPONSE_FIELD in response:
            raise OverlapRuntimeError("unexpected policy sampling trace")
        return None, None
    trace = response.get(POLICY_SAMPLING_RESPONSE_FIELD)
    if not isinstance(trace, Mapping):
        raise OverlapRuntimeError("policy sampling trace is missing")
    key_sha256 = control.get("key_sha256")
    expected_noise = policy_sampling_noise_sha256(policy_sampling_noise(key_sha256))
    expected = {
        "schema_version": POLICY_SAMPLING_SCHEMA_VERSION,
        "namespace": control.get("namespace"),
        "key_sha256": key_sha256,
        "noise_sha256": expected_noise,
        "generator": POLICY_SAMPLING_GENERATOR,
    }
    if dict(trace) != expected:
        raise OverlapRuntimeError("policy sampling trace does not match its request")
    return str(key_sha256), expected_noise


def _validate_conditioning_trace(
    response: Mapping[str, Any],
    control: Optional[Mapping[str, Any]],
    config: OverlapRuntimeConfig,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    if control is None:
        if POLICY_CONDITIONING_RESPONSE_FIELD in response:
            raise OverlapRuntimeError("unexpected policy conditioning trace")
        return None, None, None, None
    trace = response.get(POLICY_CONDITIONING_RESPONSE_FIELD)
    if not isinstance(trace, Mapping):
        raise OverlapRuntimeError("policy conditioning trace is missing")
    expected_fixed = {
        "schema_version": POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
        "method": POLICY_CONDITIONING_METHOD,
        "inference_delay": config.inference_delay_steps,
        "execute_horizon": config.execute_horizon,
        "raw_actions_sha256": control["raw_actions_sha256"],
        "mask_sha256": control["mask_sha256"],
    }
    for key, value in expected_fixed.items():
        if trace.get(key) != value:
            raise OverlapRuntimeError("policy conditioning trace mismatch: %s" % key)
    model_hash = trace.get("model_actions_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise OverlapRuntimeError("model condition hash is invalid")
    try:
        residual = float(trace["max_model_residual"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OverlapRuntimeError("condition residual is invalid") from exc
    if not math.isfinite(residual) or residual >= config.max_condition_residual:
        raise OverlapRuntimeError("condition residual exceeded the frozen gate")
    return (
        str(control["raw_actions_sha256"]),
        model_hash,
        str(control["mask_sha256"]),
        residual,
    )


def _seam_metrics(
    reference: Optional[np.ndarray],
    response: np.ndarray,
    delay: int,
    execute_horizon: int,
) -> Tuple[Optional[float], Optional[float]]:
    if reference is None or delay <= 0 or delay >= execute_horizon:
        return None, None
    old_action = reference[delay - 1]
    new_action = response[delay]
    return (
        float(np.linalg.norm(new_action[:6] - old_action[:6])),
        float(abs(new_action[-1] - old_action[-1])),
    )


def run_overlap_episode(
    environment: Any,
    policy: Any,
    initial_state: Any,
    task_description: str,
    config: OverlapRuntimeConfig,
    *,
    sampling_control_builder: Optional[SamplingControlBuilder] = None,
    request_builder: Callable[..., Dict[str, Any]] = build_libero_request,
    frame_extractor: Callable[..., np.ndarray] = extract_replay_frame,
    clock: Callable[[], float] = time.perf_counter,
) -> OverlapEpisodeResult:
    query_records: List[OverlapQueryRecord] = []
    transition_records: List[ActionChunkTransition] = []
    replay_frames: List[np.ndarray] = []
    reference: Optional[np.ndarray] = None
    environment_steps = 0
    task_action_steps = 0
    conditioned_queries = 0
    observation: Mapping[str, Any]

    def finish(
        success: bool,
        reason: str,
        *,
        failure_stage: Optional[str] = None,
        failure: Optional[BaseException] = None,
    ) -> OverlapEpisodeResult:
        return OverlapEpisodeResult(
            success=success,
            termination_reason=reason,
            initial_state_sha256=initial_state_digest(initial_state),
            environment_steps=environment_steps,
            task_action_steps=task_action_steps,
            policy_queries=len(query_records),
            bootstrap_queries=sum(record.bootstrap for record in query_records),
            conditioned_queries=conditioned_queries,
            query_records=query_records,
            transition_records=transition_records,
            replay_frames=replay_frames,
            failure_stage=failure_stage,
            failure_type=None if failure is None else type(failure).__name__,
            failure_message=None if failure is None else str(failure),
        )

    try:
        environment.reset()
        observation = environment.set_init_state(np.array(initial_state, copy=True))
        if not isinstance(observation, Mapping):
            raise TypeError("environment.set_init_state must return an observation mapping")
        for _ in range(config.num_steps_wait):
            observation, done = _environment_step(environment, LIBERO_DUMMY_ACTION)
            environment_steps += 1
            if done:
                return finish(True, "success_during_stabilization")
    except Exception as exc:
        return finish(
            False,
            "environment_initialization_failure",
            failure_stage="environment_initialization",
            failure=exc,
        )

    while task_action_steps < config.max_task_steps:
        query_index = len(query_records)
        bootstrap = reference is None
        observation_step = environment_steps
        failure_stage = "observation_build"
        try:
            request = request_builder(observation, task_description, config.resize_size)
            sampling_control = (
                None
                if sampling_control_builder is None
                else dict(sampling_control_builder(query_index))
            )
            if sampling_control is not None:
                request[POLICY_SAMPLING_REQUEST_FIELD] = sampling_control
            condition_control = None
            if not bootstrap and config.method == PROJECTED_OVERLAP:
                condition_control = build_policy_conditioning_control(
                    reference,
                    inference_delay=config.inference_delay_steps,
                    execute_horizon=config.execute_horizon,
                    action_horizon=config.action_horizon,
                    raw_action_dim=LIBERO_ACTION_DIM,
                )
                request[POLICY_CONDITIONING_REQUEST_FIELD] = condition_control

            failure_stage = "policy_inference"
            started = clock()
            response = policy.infer(request)
            finished = clock()
            inference_latency_ms = (finished - started) * 1000.0
            if not math.isfinite(inference_latency_ms) or inference_latency_ms < 0:
                raise OverlapRuntimeError("policy inference latency is invalid")
            if not isinstance(response, Mapping):
                raise OverlapRuntimeError("policy response must be an object")
            failure_stage = "policy_response_validation"
            response_actions = np.asarray(
                validate_action_chunk(response, config.action_horizon),
                dtype="<f4",
                order="C",
            )
            sampling_key, sampling_noise = _validate_sampling_trace(
                response, sampling_control
            )
            (
                condition_raw_hash,
                condition_model_hash,
                condition_mask_hash,
                condition_residual,
            ) = _validate_conditioning_trace(response, condition_control, config)
            policy_ms = response_timing_ms(response, "policy_timing")
            server_ms = response_timing_ms(response, "server_timing")

            if bootstrap:
                old_prefix = np.empty((0, LIBERO_ACTION_DIM), dtype=np.float32)
                new_suffix = response_actions[: config.execute_horizon]
                next_reference = np.concatenate(
                    (
                        response_actions[config.execute_horizon :],
                        np.zeros(
                            (config.execute_horizon, LIBERO_ACTION_DIM),
                            dtype=np.float32,
                        ),
                    ),
                    axis=0,
                )
            else:
                plan = build_overlap_plan(
                    reference,
                    response_actions,
                    inference_delay=config.inference_delay_steps,
                    execute_horizon=config.execute_horizon,
                )
                old_prefix = np.asarray(plan.inference_actions, dtype=np.float32)
                new_suffix = np.asarray(plan.response_actions, dtype=np.float32)
                next_reference = np.asarray(
                    plan.next_reference_actions, dtype=np.float32
                )
            seam_motion, seam_gripper = _seam_metrics(
                reference,
                response_actions,
                config.inference_delay_steps,
                config.execute_horizon,
            )
        except Exception as exc:
            return finish(
                False,
                "overlap_query_failure",
                failure_stage=failure_stage,
                failure=exc,
            )

        response_step = environment_steps
        executed_old = 0
        executed_new = 0
        done = False
        try:
            for phase, actions in (("old", old_prefix), ("new", new_suffix)):
                for action in actions:
                    if task_action_steps >= config.max_task_steps:
                        break
                    if config.record_video:
                        replay_frames.append(
                            frame_extractor(observation, config.resize_size)
                        )
                    observation, done = _environment_step(environment, action)
                    environment_steps += 1
                    task_action_steps += 1
                    if phase == "old":
                        executed_old += 1
                    else:
                        executed_new += 1
                    if done:
                        break
                if done or task_action_steps >= config.max_task_steps:
                    break
        except Exception as exc:
            return finish(
                False,
                "environment_step_failure",
                failure_stage="environment_step",
                failure=exc,
            )

        transition_records.append(
            build_action_chunk_transition(
                reference,
                response_actions,
                next_reference,
                inference_delay=config.inference_delay_steps,
                execute_horizon=config.execute_horizon,
                executed_old=executed_old,
                executed_new=executed_new,
            )
        )
        query_records.append(
            OverlapQueryRecord(
                query_index=query_index,
                bootstrap=bootstrap,
                observation_step=observation_step,
                response_step=response_step,
                completed_step=environment_steps,
                inference_latency_ms=inference_latency_ms,
                policy_inference_latency_ms=policy_ms,
                server_inference_latency_ms=server_ms,
                response_action_sha256=canonical_action_sha256(response_actions),
                next_reference_sha256=canonical_action_sha256(next_reference),
                sampling_key_sha256=sampling_key,
                sampling_noise_sha256=sampling_noise,
                condition_raw_actions_sha256=condition_raw_hash,
                condition_model_actions_sha256=condition_model_hash,
                condition_mask_sha256=condition_mask_hash,
                max_model_residual=condition_residual,
                old_prefix_steps=executed_old,
                new_suffix_steps=executed_new,
                executed_steps=executed_old + executed_new,
                seam_motion_l2=seam_motion,
                seam_gripper_abs=seam_gripper,
                decision=(
                    "bootstrap_unconditioned"
                    if bootstrap
                    else (
                        "accepted_projected_overlap"
                        if config.method == PROJECTED_OVERLAP
                        else "accepted_unconditioned_overlap"
                    )
                ),
            )
        )
        if not bootstrap and config.method == PROJECTED_OVERLAP:
            conditioned_queries += 1
        reference = np.array(next_reference, dtype="<f4", order="C", copy=True)
        if done:
            return finish(True, "task_success")

    return finish(False, "step_limit")
