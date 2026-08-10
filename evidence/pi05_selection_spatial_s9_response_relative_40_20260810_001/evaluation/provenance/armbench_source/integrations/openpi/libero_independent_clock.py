"""True independent-clock runtime adapters for official pi0.5-LIBERO tasks.

The generic scheduler lives in :mod:`armbench.vla.independent_clock`.  This
module binds that scheduler to the official LIBERO observation/action contract
without importing LIBERO itself, which keeps the adapter unit-testable on a
CPU-only checkout.  The parent process owns and advances the environment while
the spawned provider process owns the blocking OpenPI websocket client.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from armbench.vla.independent_clock import (
    AGE_ALIGNED_SUFFIX,
    IndependentClockConfig,
    IndependentClockResult,
    RESPONSE_RELATIVE_CHUNK,
    VALID_ACTION_SELECTION_MODES,
    run_independent_clock,
)
from integrations.openpi.libero_runtime import (
    LIBERO_ACTION_DIM,
    LIBERO_DUMMY_ACTION,
    _environment_step,
    build_libero_request,
    extract_replay_frame,
    hold_action,
    initial_state_digest,
    response_timing_ms,
    validate_action_chunk,
)
from integrations.openpi.serve_policy_attested import (
    POLICY_SAMPLING_GENERATOR,
    POLICY_SAMPLING_REQUEST_FIELD,
    POLICY_SAMPLING_RESPONSE_FIELD,
    POLICY_SAMPLING_SCORED_NAMESPACE,
    build_policy_sampling_control,
    policy_sampling_contract,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)


SCHEMA_VERSION = "armbench.pi05_libero_independent_clock.v1"
PI05_LIBERO_ACTION_HORIZON = 10


def _update_policy_input_digest(
    digest: Any,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    payload: bytes,
) -> None:
    for value in (
        name.encode("utf-8"),
        dtype.encode("ascii"),
        ",".join(str(dimension) for dimension in shape).encode("ascii"),
        payload,
    ):
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)


def canonical_policy_input_sha256(request: Mapping[str, Any]) -> str:
    """Hash the exact images, state, and prompt sent to the LIBERO policy."""

    if not isinstance(request, Mapping):
        raise ValueError("policy request must be a mapping")
    digest = hashlib.sha256()
    digest.update(b"armbench.pi05.libero.policy_input.v1\0")
    for name in ("observation/image", "observation/wrist_image"):
        value = request.get(name)
        if not isinstance(value, np.ndarray):
            raise ValueError("%s must be a numpy array" % name)
        if value.shape != (224, 224, 3) or value.dtype != np.dtype(np.uint8):
            raise ValueError("%s must have shape (224, 224, 3) and dtype uint8" % name)
        canonical = np.ascontiguousarray(value)
        _update_policy_input_digest(
            digest,
            name,
            "uint8",
            tuple(canonical.shape),
            canonical.tobytes(order="C"),
        )

    state = request.get("observation/state")
    if not isinstance(state, np.ndarray):
        raise ValueError("observation/state must be a numpy array")
    if state.shape != (8,) or state.dtype != np.dtype(np.float64):
        raise ValueError("observation/state must have shape (8,) and dtype float64")
    if not np.all(np.isfinite(state)):
        raise ValueError("observation/state must contain only finite values")
    canonical_state = np.asarray(state, dtype="<f8", order="C")
    _update_policy_input_digest(
        digest,
        "observation/state",
        "float64-le",
        tuple(canonical_state.shape),
        canonical_state.tobytes(order="C"),
    )

    prompt = request.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    prompt_bytes = prompt.encode("utf-8")
    _update_policy_input_digest(digest, "prompt", "utf-8", (), prompt_bytes)
    return digest.hexdigest()


def canonical_action_chunk_sha256(actions: Any) -> str:
    """Hash a finite action chunk with explicit shape and little-endian bytes."""

    array = np.asarray(actions, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != LIBERO_ACTION_DIM
        or array.shape[0] <= 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("actions must be finite with shape (horizon, 7)")
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class IndependentLiberoRequestBuilder:
    """Build an official request plus a mode-independent sampling key."""

    task_description: str
    task_suite: str
    task_id: int
    episode_index: int
    seed: int
    replan_steps: int = 1
    resize_size: int = 224

    def __post_init__(self) -> None:
        if not self.task_description.strip():
            raise ValueError("task_description must be nonempty")
        if not self.task_suite.strip():
            raise ValueError("task_suite must be nonempty")
        for name, value in (
            ("task_id", self.task_id),
            ("episode_index", self.episode_index),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
        if (
            isinstance(self.replan_steps, bool)
            or not isinstance(self.replan_steps, int)
            or self.replan_steps <= 0
        ):
            raise ValueError("replan_steps must be a positive integer")
        if self.resize_size != 224:
            raise ValueError("official pi05_libero requests require 224px images")

    @property
    def pairing_key(self) -> tuple[object, ...]:
        return (
            self.task_suite,
            self.task_id,
            self.episode_index,
            self.replan_steps,
        )

    def __call__(
        self,
        observation: Mapping[str, Any],
        sequence_id: int,
        captured_at_s: float,
    ) -> dict[str, Any]:
        del captured_at_s
        request = build_libero_request(
            observation, self.task_description, self.resize_size
        )
        request[POLICY_SAMPLING_REQUEST_FIELD] = build_policy_sampling_control(
            POLICY_SAMPLING_SCORED_NAMESPACE,
            self.seed,
            self.pairing_key,
            sequence_id,
        )
        return request


@dataclass(frozen=True)
class OpenPIIndependentClockProviderFactory:
    """Pickle-safe factory that opens the websocket in the spawned child."""

    host: str
    port: int
    startup_timeout_s: float
    inference_timeout_s: float
    action_horizon: int = PI05_LIBERO_ACTION_HORIZON

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must be nonempty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be in [1, 65535]")
        for name, value in (
            ("startup_timeout_s", self.startup_timeout_s),
            ("inference_timeout_s", self.inference_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and positive" % name)
        if self.action_horizon != PI05_LIBERO_ACTION_HORIZON:
            raise ValueError("official pi05_libero action horizon must be 10")

    def __call__(self) -> "_OpenPIIndependentClockProvider":
        return _OpenPIIndependentClockProvider(self)


class _OpenPIIndependentClockProvider:
    """Validate OpenPI actions and sampling evidence before publication."""

    def __init__(self, config: OpenPIIndependentClockProviderFactory) -> None:
        from integrations.openpi.libero_runtime_eval import BoundedOpenPIClient

        self._action_horizon = config.action_horizon
        self._client = BoundedOpenPIClient(
            config.host,
            config.port,
            startup_timeout_s=config.startup_timeout_s,
            inference_timeout_s=config.inference_timeout_s,
        )

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        control = request.get(POLICY_SAMPLING_REQUEST_FIELD)
        if not isinstance(control, Mapping):
            raise ValueError("independent-clock request lacks policy sampling control")
        policy_input_sha256 = canonical_policy_input_sha256(request)
        response = self._client.infer(request)
        actions = validate_action_chunk(response, self._action_horizon)
        if actions.shape[0] != self._action_horizon:
            raise ValueError("pi05_libero response action horizon must equal 10")

        audit = response.get(POLICY_SAMPLING_RESPONSE_FIELD)
        if not isinstance(audit, Mapping):
            raise ValueError("OpenPI response lacks policy sampling audit")
        key_sha256 = str(control.get("key_sha256", ""))
        expected_audit = {
            "schema_version": policy_sampling_contract()["schema_version"],
            "namespace": POLICY_SAMPLING_SCORED_NAMESPACE,
            "key_sha256": key_sha256,
            "noise_sha256": policy_sampling_noise_sha256(
                policy_sampling_noise(key_sha256)
            ),
            "generator": POLICY_SAMPLING_GENERATOR,
        }
        if dict(audit) != expected_audit:
            raise ValueError("OpenPI policy sampling audit mismatch")

        return {
            "actions": actions,
            "source": "official_openpi_pi05_libero",
            "response_metadata": {
                "action_chunk_sha256": canonical_action_chunk_sha256(actions),
                "policy_input_sha256": policy_input_sha256,
                "policy_sampling": dict(audit),
                "policy_inference_latency_ms": response_timing_ms(
                    response, "policy_timing"
                ),
                "server_inference_latency_ms": response_timing_ms(
                    response, "server_timing"
                ),
            },
        }

    def close(self) -> None:
        self._client.close()


def libero_dynamic_hold(previous_action: np.ndarray) -> np.ndarray:
    """Stop Cartesian motion while retaining the last gripper command."""

    previous = np.asarray(previous_action, dtype=np.float64).copy()
    if previous.shape != (LIBERO_ACTION_DIM,) or not np.all(np.isfinite(previous)):
        raise ValueError("previous LIBERO action must be finite with shape (7,)")
    if np.array_equal(previous, np.zeros(LIBERO_ACTION_DIM, dtype=np.float64)):
        previous = LIBERO_DUMMY_ACTION.copy()
    return hold_action(previous)


class LiberoIndependentClockEnvironment:
    """Parent-owned official LIBERO environment with frozen initialization."""

    def __init__(
        self,
        environment: Any,
        initial_state: Any,
        *,
        num_steps_wait: int = 10,
        resize_size: int = 224,
        record_video: bool = False,
    ) -> None:
        if isinstance(num_steps_wait, bool) or num_steps_wait < 0:
            raise ValueError("num_steps_wait must be a nonnegative integer")
        if resize_size != 224:
            raise ValueError("official pi05_libero evaluation requires 224px images")
        self.environment = environment
        self.initial_state = np.array(initial_state, copy=True)
        self.num_steps_wait = int(num_steps_wait)
        self.resize_size = int(resize_size)
        self.record_video = bool(record_video)
        self.initial_state_sha256 = initial_state_digest(self.initial_state)
        self.observation: Mapping[str, Any] | None = None
        self.stabilization_steps = 0
        self.task_steps = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.final_info: Mapping[str, Any] = {}
        self.replay_frames: list[np.ndarray] = []

    def _require_observation(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("LIBERO environment observation must be a mapping")
        return value

    def reset(self) -> Mapping[str, Any]:
        self.environment.reset()
        observation = self.environment.set_init_state(self.initial_state.copy())
        self.observation = self._require_observation(observation)
        self.stabilization_steps = 0
        self.task_steps = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.final_info = {}
        self.replay_frames = []
        for _ in range(self.num_steps_wait):
            observation, reward, done, info = _environment_step(
                self.environment, LIBERO_DUMMY_ACTION
            )
            self.observation = self._require_observation(observation)
            self.stabilization_steps += 1
            self.cumulative_reward += float(reward)
            self.done = bool(done)
            self.final_info = dict(info)
            if self.done:
                break
        if self.record_video:
            self.replay_frames.append(
                extract_replay_frame(self.observation, self.resize_size)
            )
        return self.observation

    def observe(self) -> Mapping[str, Any]:
        if self.observation is None:
            raise RuntimeError("LIBERO environment must be reset before observe")
        return self.observation

    def step(
        self, action: np.ndarray
    ) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        if self.observation is None:
            raise RuntimeError("LIBERO environment must be reset before step")
        if self.done:
            return self.observation, 0.0, True, self.final_info
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (LIBERO_ACTION_DIM,) or not np.all(
            np.isfinite(action_array)
        ):
            raise ValueError("LIBERO action must be finite with shape (7,)")
        observation, reward, done, info = _environment_step(
            self.environment, action_array
        )
        self.observation = self._require_observation(observation)
        self.task_steps += 1
        self.cumulative_reward += float(reward)
        self.done = bool(done)
        self.final_info = dict(info)
        if self.record_video:
            self.replay_frames.append(
                extract_replay_frame(self.observation, self.resize_size)
            )
        return self.observation, float(reward), self.done, self.final_info


@dataclass(frozen=True)
class LiberoIndependentClockEpisodeResult:
    """Task outcome and raw scheduler evidence for one official episode."""

    schema_version: str
    task_success: bool
    termination_reason: str
    initial_state_sha256: str
    stabilization_steps: int
    task_steps: int
    cumulative_reward: float
    final_info: Mapping[str, Any]
    runtime: IndependentClockResult
    replay_frames: tuple[np.ndarray, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_success": self.task_success,
            "termination_reason": self.termination_reason,
            "initial_state_sha256": self.initial_state_sha256,
            "stabilization_steps": self.stabilization_steps,
            "task_steps": self.task_steps,
            "cumulative_reward": self.cumulative_reward,
            "final_info": dict(self.final_info),
            "runtime": self.runtime.to_dict(),
        }


def run_libero_independent_clock_episode(
    environment: Any,
    provider_factory: Any,
    initial_state: Any,
    request_builder: IndependentLiberoRequestBuilder,
    *,
    control_period_s: float = 0.05,
    deadline_s: float = 0.20,
    max_task_steps: int,
    num_steps_wait: int = 10,
    submit_every_ticks: int = 1,
    action_selection_mode: str = AGE_ALIGNED_SUFFIX,
    startup_timeout_s: float = 1200.0,
    shutdown_timeout_s: float = 5.0,
    record_video: bool = False,
) -> LiberoIndependentClockEpisodeResult:
    """Run one task while simulation ticks during blocking policy inference."""

    adapter = LiberoIndependentClockEnvironment(
        environment,
        initial_state,
        num_steps_wait=num_steps_wait,
        resize_size=request_builder.resize_size,
        record_video=record_video,
    )
    runtime = run_independent_clock(
        adapter,
        provider_factory,
        config=IndependentClockConfig(
            control_period_s=control_period_s,
            action_period_s=control_period_s,
            deadline_s=deadline_s,
            max_ticks=max_task_steps,
            action_dim=LIBERO_ACTION_DIM,
            submit_every_ticks=submit_every_ticks,
            action_selection_mode=action_selection_mode,
            startup_timeout_s=startup_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
        ),
        observation_builder=request_builder,
        hold_action=libero_dynamic_hold,
    )
    success = bool(adapter.done and runtime.environment_done)
    termination_reason = "task_success" if success else runtime.termination_reason
    return LiberoIndependentClockEpisodeResult(
        schema_version=SCHEMA_VERSION,
        task_success=success,
        termination_reason=termination_reason,
        initial_state_sha256=adapter.initial_state_sha256,
        stabilization_steps=adapter.stabilization_steps,
        task_steps=adapter.task_steps,
        cumulative_reward=float(adapter.cumulative_reward),
        final_info=dict(adapter.final_info),
        runtime=runtime,
        replay_frames=tuple(adapter.replay_frames),
    )


__all__ = [
    "AGE_ALIGNED_SUFFIX",
    "IndependentLiberoRequestBuilder",
    "LiberoIndependentClockEnvironment",
    "LiberoIndependentClockEpisodeResult",
    "OpenPIIndependentClockProviderFactory",
    "PI05_LIBERO_ACTION_HORIZON",
    "RESPONSE_RELATIVE_CHUNK",
    "SCHEMA_VERSION",
    "VALID_ACTION_SELECTION_MODES",
    "canonical_action_chunk_sha256",
    "canonical_policy_input_sha256",
    "libero_dynamic_hold",
    "run_libero_independent_clock_episode",
]
