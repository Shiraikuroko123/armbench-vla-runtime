"""Load and verify exact OpenPI DROID requests from online artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from armbench.vla.artifact import validate_online_artifact
from armbench.vla.types import VLAObservation


@dataclass(frozen=True)
class RecordedOpenPIRequest:
    directory: str
    scenario: str
    payload_mass: float
    execution_horizon: int
    query_index: int
    observation: VLAObservation
    packed_payload_sha256: str
    server_payload_sha256: str | None

    @property
    def server_payload_matches(self) -> bool | None:
        if self.server_payload_sha256 is None:
            return None
        return self.packed_payload_sha256 == self.server_payload_sha256

    def openpi_request(self) -> dict[str, object]:
        return self.observation.to_openpi_droid()

    def metrics(self) -> dict[str, object]:
        request = self.openpi_request()
        exterior = np.asarray(request["observation/exterior_image_1_left"])
        wrist = np.asarray(request["observation/wrist_image_left"])
        joints = np.asarray(request["observation/joint_position"])
        gripper = np.asarray(request["observation/gripper_position"])
        return {
            "directory": self.directory,
            "scenario": self.scenario,
            "payload_mass": self.payload_mass,
            "execution_horizon": self.execution_horizon,
            "query_index": self.query_index,
            "openpi_keys": list(request),
            "exterior_shape": list(exterior.shape),
            "exterior_dtype": str(exterior.dtype),
            "exterior_sha256": hashlib.sha256(
                exterior.tobytes(order="C")
            ).hexdigest(),
            "wrist_shape": list(wrist.shape),
            "wrist_dtype": str(wrist.dtype),
            "wrist_sha256": hashlib.sha256(
                wrist.tobytes(order="C")
            ).hexdigest(),
            "joint_position": joints.tolist(),
            "gripper_position": gripper.tolist(),
            "prompt": self.observation.prompt,
            "sequence_id": self.observation.sequence_id,
            "packed_payload_sha256": self.packed_payload_sha256,
            "server_payload_sha256": self.server_payload_sha256,
            "server_payload_matches": self.server_payload_matches,
            "replayable": True,
        }


def _aggregate_rows(directory: Path) -> list[dict[str, object]]:
    value = json.loads(
        (directory / "aggregate.json").read_text(encoding="utf-8")
    )
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError("aggregate.json must contain a list of mappings")
    return [dict(row) for row in value]


def _select_episode(
    rows: list[dict[str, object]],
    *,
    scenario: str | None,
    payload_mass: float | None,
    execution_horizon: int | None,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if (scenario is None or str(row["scenario"]) == scenario)
        and (
            payload_mass is None
            or np.isclose(float(row["payload_mass"]), payload_mass)
        )
        and (
            execution_horizon is None
            or int(row["execution_horizon"]) == execution_horizon
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "request inspection requires exactly one matching episode; "
            f"matched {len(selected)}"
        )
    return selected[0]


def _server_payload_hash(directory: Path, query_index: int) -> str | None:
    audit_path = directory / "loopback_server.json"
    if not audit_path.is_file():
        return None
    value = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise ValueError("loopback_server.json has an invalid request audit")
    matches = [
        request
        for request in value["requests"]
        if isinstance(request, dict)
        and int(request.get("request_index", -1)) == query_index
    ]
    if len(matches) != 1:
        raise ValueError("loopback request audit does not match query index")
    payload_hash = matches[0].get("request_payload_sha256")
    if payload_hash is None:
        return None
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise ValueError("loopback request payload hash is invalid")
    return payload_hash


def load_recorded_openpi_request(
    directory: Path,
    *,
    query_index: int = 0,
    scenario: str | None = None,
    payload_mass: float | None = None,
    execution_horizon: int | None = None,
) -> RecordedOpenPIRequest:
    """Reconstruct one exact five-key DROID request from a validated artifact."""

    if query_index < 0:
        raise ValueError("query_index must be nonnegative")
    root = directory.resolve()
    validation = validate_online_artifact(root)
    if validation.replayable_requests == 0:
        raise ValueError(
            "artifact has no replayable requests; record it with "
            "--save-observations using a request-metadata-capable build"
        )
    row = _select_episode(
        _aggregate_rows(root),
        scenario=scenario,
        payload_mass=payload_mass,
        execution_horizon=execution_horizon,
    )
    trace_path = root / str(row["trace"])
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            exterior_images = np.asarray(trace["exterior_images"])
            wrist_images = np.asarray(trace["wrist_images"])
            positions = np.asarray(trace["observation_positions"])
            grippers = np.asarray(trace["observation_gripper_positions"])
            prompts = np.asarray(trace["prompts"])
            sequence_ids = np.asarray(trace["action_offsets"])
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"cannot load replayable trace: {trace_path}") from error
    if query_index >= len(exterior_images):
        raise IndexError(
            f"query_index {query_index} outside [0, {len(exterior_images)})"
        )
    observation = VLAObservation(
        exterior_image=exterior_images[query_index],
        wrist_image=wrist_images[query_index],
        joint_position=positions[query_index],
        gripper_position=grippers[query_index],
        prompt=str(prompts[query_index]),
        sequence_id=int(sequence_ids[query_index]),
        captured_at_s=0.0,
    )
    try:
        from openpi_client import msgpack_numpy
    except ImportError as error:
        raise RuntimeError(
            "OpenPI client is required to pack a recorded request"
        ) from error
    packed = msgpack_numpy.packb(observation.to_openpi_droid())
    packed_hash = hashlib.sha256(packed).hexdigest()
    server_hash = _server_payload_hash(root, query_index)
    if server_hash is not None and packed_hash != server_hash:
        raise ValueError(
            "repacked OpenPI payload does not match the server-received payload"
        )
    return RecordedOpenPIRequest(
        directory=str(root),
        scenario=str(row["scenario"]),
        payload_mass=float(row["payload_mass"]),
        execution_horizon=int(row["execution_horizon"]),
        query_index=query_index,
        observation=observation,
        packed_payload_sha256=packed_hash,
        server_payload_sha256=server_hash,
    )
