"""Probe an OpenPI server with an exact recorded DROID request."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from armbench.benchmark import environment_metadata
from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.benchmark import (
    OPENPI_COMMIT,
    _guard_config,
    _package_version,
    _write_json,
    load_vla_config,
)
from armbench.vla.policy import OpenPIPolicyClient
from armbench.vla.request_replay import load_recorded_openpi_request
from armbench.vla.types import VLAObservation
from armbench.vla.guard import ActionChunkGuard


RECORDED_OPENPI_PROBE_ARTIFACT_TYPE = "armbench_recorded_openpi_probe_v2"


class RecordedProbeValidationError(ValueError):
    """Raised when a recorded-request probe artifact is inconsistent."""


@dataclass(frozen=True)
class RecordedProbeValidationResult:
    directory: str
    request_payload_sha256: str
    request_payload_size_bytes: int
    action_sha256: str
    action_shape: tuple[int, int]
    action_dtype: str
    guard_safe_after: bool
    policy_provenance: str
    checks: tuple[str, ...]

    def metrics(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "request_payload_sha256": self.request_payload_sha256,
            "request_payload_size_bytes": self.request_payload_size_bytes,
            "action_sha256": self.action_sha256,
            "action_shape": list(self.action_shape),
            "action_dtype": self.action_dtype,
            "guard_safe_after": self.guard_safe_after,
            "policy_provenance": self.policy_provenance,
            "checks": list(self.checks),
            "valid": True,
        }


def _probe_require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordedProbeValidationError(message)


def _probe_json(path: Path) -> dict[str, object]:
    _probe_require(path.is_file() and path.stat().st_size > 0, f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecordedProbeValidationError(f"invalid JSON: {path}") from error
    _probe_require(isinstance(value, dict), f"JSON root must be a mapping: {path}")
    return dict(value)


def _sha256(value: object, label: str) -> str:
    _probe_require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label}",
    )
    return value


def _finite_array(
    value: np.ndarray, shape: tuple[int, ...], label: str
) -> None:
    _probe_require(value.shape == shape, f"unexpected {label} shape")
    _probe_require(
        np.issubdtype(value.dtype, np.number),
        f"{label} must have a numeric dtype",
    )
    _probe_require(bool(np.all(np.isfinite(value))), f"{label} contains nonfinite values")


def _finite_number(value: object, label: str) -> float:
    _probe_require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    number = float(value)
    _probe_require(bool(np.isfinite(number)), f"{label} must be finite")
    return number


def validate_recorded_openpi_probe(
    directory: Path,
) -> RecordedProbeValidationResult:
    """Cross-check a fixed-request OpenPI probe without executing physics."""

    root = directory.resolve()
    _probe_require(root.is_dir(), f"artifact directory does not exist: {root}")
    response = _probe_json(root / "response.json")
    environment = _probe_json(root / "environment.json")
    summary_path = root / "summary.md"
    _probe_require(
        summary_path.is_file() and summary_path.stat().st_size > 0,
        f"missing file: {summary_path}",
    )

    _probe_require(
        response.get("artifact_type") == RECORDED_OPENPI_PROBE_ARTIFACT_TYPE,
        "unexpected recorded probe artifact type",
    )
    _probe_require(
        response.get("remote_policy_response_validated") is True,
        "remote policy response is not validated",
    )
    _probe_require(
        response.get("checkpoint_identity_verified") is False,
        "checkpoint identity must remain unverified",
    )
    _probe_require(
        response.get("physics_executed") is False,
        "recorded probe must not claim physics execution",
    )
    _probe_require(
        response.get("physical_safe") is None,
        "recorded probe must not claim a physical-safety outcome",
    )
    policy_provenance = response.get("policy_provenance")
    _probe_require(
        isinstance(policy_provenance, str) and bool(policy_provenance.strip()),
        "policy provenance must be nonempty",
    )

    source_request = response.get("source_request")
    _probe_require(isinstance(source_request, dict), "source_request must be a mapping")
    request_hash = _sha256(
        source_request.get("packed_payload_sha256"),
        "source request payload SHA-256",
    )
    _probe_require(source_request.get("replayable") is True, "source request is not replayable")
    server_hash_value = source_request.get("server_payload_sha256")
    server_matches = source_request.get("server_payload_matches")
    if server_hash_value is None:
        _probe_require(
            server_matches is None,
            "source server payload match must be null when no server hash exists",
        )
    else:
        server_hash = _sha256(server_hash_value, "source server payload SHA-256")
        _probe_require(
            server_matches is (request_hash == server_hash),
            "source server payload match flag is inconsistent",
        )

    trace_path = root / "response.npz"
    _probe_require(
        trace_path.is_file() and trace_path.stat().st_size > 0,
        f"missing file: {trace_path}",
    )
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            _probe_require(
                set(trace.files)
                == {"raw_actions", "guarded_actions", "predicted_positions"},
                "response.npz has unexpected arrays",
            )
            raw_actions = np.asarray(trace["raw_actions"])
            guarded_actions = np.asarray(trace["guarded_actions"])
            predicted_positions = np.asarray(trace["predicted_positions"])
    except RecordedProbeValidationError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise RecordedProbeValidationError(
            f"cannot load recorded probe arrays: {trace_path}"
        ) from error
    _finite_array(raw_actions, (15, 8), "raw actions")
    _finite_array(guarded_actions, (15, 8), "guarded actions")
    _finite_array(predicted_positions, (16, 7), "predicted positions")

    request_path = root / "request.msgpack"
    _probe_require(
        request_path.is_file() and request_path.stat().st_size > 0,
        f"missing file: {request_path}",
    )
    request_payload = request_path.read_bytes()
    _probe_require(
        response.get("request_payload_embedded") is True,
        "recorded request payload is not marked embedded",
    )
    _probe_require(
        response.get("request_payload_size_bytes") == len(request_payload),
        "recorded request payload size mismatch",
    )
    _probe_require(
        hashlib.sha256(request_payload).hexdigest() == request_hash,
        "recorded request payload SHA-256 mismatch",
    )
    try:
        from openpi_client import msgpack_numpy

        unpacked_request = msgpack_numpy.unpackb(request_payload)
    except Exception as error:
        raise RecordedProbeValidationError(
            "recorded request payload cannot be unpacked"
        ) from error
    _probe_require(
        isinstance(unpacked_request, dict),
        "recorded request payload must unpack to a mapping",
    )
    expected_keys = source_request.get("openpi_keys")
    _probe_require(
        isinstance(expected_keys, list)
        and list(unpacked_request) == expected_keys,
        "recorded request keys mismatch",
    )
    try:
        exterior = np.asarray(
            unpacked_request["observation/exterior_image_1_left"]
        )
        wrist = np.asarray(unpacked_request["observation/wrist_image_left"])
        joints = np.asarray(unpacked_request["observation/joint_position"])
        gripper = np.asarray(unpacked_request["observation/gripper_position"])
        prompt = unpacked_request["prompt"]
    except KeyError as error:
        raise RecordedProbeValidationError(
            "recorded request payload is missing a DROID field"
        ) from error
    _probe_require(
        list(exterior.shape) == source_request.get("exterior_shape")
        and str(exterior.dtype) == source_request.get("exterior_dtype")
        and hashlib.sha256(exterior.tobytes(order="C")).hexdigest()
        == source_request.get("exterior_sha256"),
        "recorded exterior image metadata mismatch",
    )
    _probe_require(
        list(wrist.shape) == source_request.get("wrist_shape")
        and str(wrist.dtype) == source_request.get("wrist_dtype")
        and hashlib.sha256(wrist.tobytes(order="C")).hexdigest()
        == source_request.get("wrist_sha256"),
        "recorded wrist image metadata mismatch",
    )
    _probe_require(
        joints.tolist() == source_request.get("joint_position"),
        "recorded joint position mismatch",
    )
    _probe_require(
        gripper.tolist() == source_request.get("gripper_position"),
        "recorded gripper position mismatch",
    )
    _probe_require(
        prompt == source_request.get("prompt"),
        "recorded prompt mismatch",
    )

    action_hash = _sha256(response.get("action_sha256"), "raw action SHA-256")
    recomputed_action_hash = hashlib.sha256(
        raw_actions.tobytes(order="C")
    ).hexdigest()
    _probe_require(
        action_hash == recomputed_action_hash,
        "raw action SHA-256 mismatch",
    )
    _probe_require(response.get("action_shape") == [15, 8], "JSON action shape mismatch")
    _probe_require(
        response.get("action_dtype") == str(raw_actions.dtype),
        "JSON action dtype mismatch",
    )
    action_min = _finite_number(response.get("action_min"), "JSON action minimum")
    action_max = _finite_number(response.get("action_max"), "JSON action maximum")
    _probe_require(
        action_min == float(np.min(raw_actions)),
        "JSON action minimum mismatch",
    )
    _probe_require(
        action_max == float(np.max(raw_actions)),
        "JSON action maximum mismatch",
    )

    guard = response.get("guard")
    _probe_require(isinstance(guard, dict), "guard metrics must be a mapping")
    guard_safe = response.get("guard_safe_after")
    _probe_require(isinstance(guard_safe, bool), "guard_safe_after must be boolean")
    _probe_require(
        guard.get("safe_after_guard") is guard_safe,
        "guard safety fields disagree",
    )

    probe_environment = environment.get("recorded_openpi_probe")
    _probe_require(
        isinstance(probe_environment, dict),
        "environment recorded probe metadata is missing",
    )
    expected_environment = {
        "remote_policy_response_validated": True,
        "checkpoint_identity_verified": False,
        "policy_provenance": policy_provenance,
        "server": response.get("server"),
        "source_request_payload_sha256": request_hash,
        "request_payload_embedded": True,
        "request_payload_size_bytes": len(request_payload),
        "response_action_sha256": action_hash,
        "physics_executed": False,
    }
    for key, expected in expected_environment.items():
        _probe_require(
            probe_environment.get(key) == expected,
            f"environment metadata mismatch: {key}",
        )

    summary = summary_path.read_text(encoding="utf-8")
    _probe_require(request_hash in summary, "summary request hash mismatch")
    _probe_require(action_hash in summary, "summary action hash mismatch")
    _probe_require(
        "Checkpoint identity verified by protocol: `false`" in summary,
        "summary checkpoint claim mismatch",
    )
    _probe_require(
        "Physics executed: `false`" in summary,
        "summary physics claim mismatch",
    )
    return RecordedProbeValidationResult(
        directory=str(root),
        request_payload_sha256=request_hash,
        request_payload_size_bytes=len(request_payload),
        action_sha256=action_hash,
        action_shape=(15, 8),
        action_dtype=str(raw_actions.dtype),
        guard_safe_after=guard_safe,
        policy_provenance=policy_provenance,
        checks=(
            "artifact_contract",
            "source_request_hash",
            "embedded_request_payload",
            "finite_action_arrays",
            "raw_action_hash",
            "guard_consistency",
            "environment_consistency",
            "summary_claim_boundaries",
        ),
    )


def execute_recorded_openpi_probe(
    config_path: Path,
    artifact_directory: Path,
    output_directory: Path,
    *,
    host: str,
    port: int,
    query_index: int = 0,
    scenario: str | None = None,
    payload_mass: float | None = None,
    execution_horizon: int | None = None,
    api_key: str | None = None,
    connect_timeout_s: float = 3.0,
    inference_timeout_s: float = 1.0,
    policy_provenance: str = "remote_server_unverified",
) -> Path:
    """Run bounded inference on one recorded request without physics execution."""

    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists: {output_directory}"
        )
    if not policy_provenance.strip():
        raise ValueError("policy_provenance must be nonempty")
    config = load_vla_config(config_path)
    recorded = load_recorded_openpi_request(
        artifact_directory,
        query_index=query_index,
        scenario=scenario,
        payload_mass=payload_mass,
        execution_horizon=execution_horizon,
    )
    scenarios = mujoco_scenarios()
    if recorded.scenario not in scenarios:
        raise ValueError(
            f"recorded scenario is unavailable: {recorded.scenario}"
        )
    source = recorded.observation
    observation = VLAObservation(
        exterior_image=source.exterior_image,
        wrist_image=source.wrist_image,
        joint_position=source.joint_position,
        gripper_position=source.gripper_position,
        prompt=source.prompt,
        sequence_id=source.sequence_id,
        captured_at_s=monotonic(),
    )
    expected_horizon = int(dict(config["openpi_contract"])["action_horizon"])
    with OpenPIPolicyClient(
        host=host,
        port=port,
        api_key=api_key,
        expected_horizon=expected_horizon,
        connect_timeout_s=connect_timeout_s,
        inference_timeout_s=inference_timeout_s,
    ) as client:
        server_metadata = client.server_metadata
        chunk = client.infer(observation)

    scenario_definition = scenarios[recorded.scenario]
    raw_guard = dict(config["guard"])
    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(
            scenario_definition.obstacles,
            float(raw_guard["clearance_m"]),
        )
    )
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(
            guard_robot,
            resolution=float(raw_guard["collision_resolution_rad"]),
        ),
        _guard_config(config),
    )
    guard_result = guard.guard(
        observation.joint_position,
        float(observation.gripper_position[0]),
        observation,
        chunk,
    )
    action_sha256 = hashlib.sha256(
        chunk.actions.tobytes(order="C")
    ).hexdigest()
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "request.msgpack").write_bytes(recorded.packed_payload)
    np.savez_compressed(
        output_directory / "response.npz",
        raw_actions=chunk.actions,
        guarded_actions=guard_result.guarded_actions,
        predicted_positions=guard_result.predicted_positions,
    )
    response = {
        "artifact_type": RECORDED_OPENPI_PROBE_ARTIFACT_TYPE,
        "source_artifact": str(artifact_directory.resolve()),
        "source_request": recorded.metrics(),
        "request_payload_embedded": True,
        "request_payload_size_bytes": len(recorded.packed_payload),
        "server": f"{host}:{port}",
        "server_metadata": server_metadata,
        "policy_provenance": policy_provenance,
        "remote_policy_response_validated": True,
        "checkpoint_identity_verified": False,
        "openpi_commit": OPENPI_COMMIT,
        "action_shape": list(chunk.actions.shape),
        "action_dtype": str(chunk.actions.dtype),
        "action_sha256": action_sha256,
        "action_min": float(np.min(chunk.actions)),
        "action_max": float(np.max(chunk.actions)),
        "client_inference_latency_ms": chunk.inference_latency_ms,
        "action_age_ms": chunk.age_ms(observation),
        "server_timing": dict(chunk.server_timing),
        "guard_safe_after": guard_result.safe_after_guard,
        "guard": guard_result.metrics(),
        "physics_executed": False,
        "physical_safe": None,
    }
    _write_json(output_directory / "response.json", response)
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["packages"].update(
        {
            "msgpack": _package_version("msgpack"),
            "mujoco": _package_version("mujoco"),
            "openpi-client": _package_version("openpi-client"),
            "websockets": _package_version("websockets"),
        }
    )
    metadata["recorded_openpi_probe"] = {
        "remote_policy_response_validated": True,
        "checkpoint_identity_verified": False,
        "policy_provenance": policy_provenance,
        "server": f"{host}:{port}",
        "source_request_payload_sha256": recorded.packed_payload_sha256,
        "request_payload_embedded": True,
        "request_payload_size_bytes": len(recorded.packed_payload),
        "response_action_sha256": action_sha256,
        "physics_executed": False,
    }
    _write_json(output_directory / "environment.json", metadata)
    (output_directory / "summary.md").write_text(
        "\n".join(
            [
                "# Recorded OpenPI request probe",
                "",
                f"- Source: `{artifact_directory.resolve()}` query `{query_index}`",
                f"- Request payload SHA-256: `{recorded.packed_payload_sha256}`",
                f"- Embedded request bytes: `{len(recorded.packed_payload)}`",
                f"- Server: `{host}:{port}`",
                f"- Policy provenance: `{policy_provenance}`",
                "- Remote 15x8 response validated: `true`",
                "- Checkpoint identity verified by protocol: `false`",
                f"- Response action SHA-256: `{action_sha256}`",
                f"- Client inference latency ms: `{chunk.inference_latency_ms:.3f}`",
                f"- Guard safe after: `{str(guard_result.safe_after_guard).lower()}`",
                "- Physics executed: `false`",
                "",
                "This artifact compares a server response on a fixed recorded "
                "request. It is not a task rollout or physical-safety result.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return output_directory
