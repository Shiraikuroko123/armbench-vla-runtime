"""Probe an OpenPI server with an exact recorded DROID request."""

from __future__ import annotations

import hashlib
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
    np.savez_compressed(
        output_directory / "response.npz",
        raw_actions=chunk.actions,
        guarded_actions=guard_result.guarded_actions,
        predicted_positions=guard_result.predicted_positions,
    )
    response = {
        "artifact_type": "armbench_recorded_openpi_probe_v1",
        "source_artifact": str(artifact_directory.resolve()),
        "source_request": recorded.metrics(),
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
