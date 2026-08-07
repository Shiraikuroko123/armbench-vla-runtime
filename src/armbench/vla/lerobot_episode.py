"""Hash-manifested LeRobot-style episode export and deterministic replay."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    PANDA_RUNTIME_ACTION_SPACE_ID,
    ActuatorCommandWatchdog,
    CommandWatchdogConfig,
    WatchdogDecision,
    runtime_action_semantics,
)
from armbench.vla.lerobot_adapter import (
    LEROBOT_STYLE_FRAME_KEYS,
    LeRobotFrameAdapter,
    frame_array_sha256,
)
from armbench.vla.types import DROID_IMAGE_SHAPE, VLAObservation


FloatArray = NDArray[np.float64]
LEROBOT_EPISODE_SCHEMA = "armbench.lerobot_style_episode.v1"
LEROBOT_FRAME_SCHEMA = "armbench.lerobot_style_frame.v1"
LEROBOT_SUMMARY_SCHEMA = "armbench.lerobot_style_episode_summary.v1"
LEROBOT_MANIFEST_SCHEMA = "armbench.lerobot_style_episode_manifest.v1"
_EPISODE_FILES = {
    "episode.npz",
    "frames.jsonl",
    "metadata.json",
    "summary.json",
    "manifest.json",
}


class LeRobotEpisodeError(ValueError):
    """Raised when an episode cannot be exported or deterministically replayed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LeRobotEpisodeError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_manifest(root: Path) -> None:
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    document = {
        "schema_version": LEROBOT_MANIFEST_SCHEMA,
        "files": files,
        "inventory_sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
    }
    (root / "manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_manifest(root: Path) -> str:
    _require(root.is_dir(), f"LeRobot-style episode not found: {root}")
    _require(
        {path.name for path in root.iterdir() if path.is_file()} == _EPISODE_FILES,
        "LeRobot-style episode file set is invalid",
    )
    try:
        manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeRobotEpisodeError("episode manifest is invalid") from error
    _require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == LEROBOT_MANIFEST_SCHEMA,
        "episode manifest schema mismatch",
    )
    expected = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    _require(manifest.get("files") == expected, "episode manifest inventory mismatch")
    inventory_sha = hashlib.sha256(_canonical_json(expected)).hexdigest()
    _require(
        manifest.get("inventory_sha256") == inventory_sha,
        "episode manifest inventory hash mismatch",
    )
    return inventory_sha


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeRobotEpisodeError(f"invalid JSON: {path.name}") from error
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                _require(
                    isinstance(value, dict),
                    f"frames.jsonl line {line_number} is not an object",
                )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise LeRobotEpisodeError("frames.jsonl is invalid") from error
    _require(bool(rows), "frames.jsonl contains no frames")
    return rows


@dataclass(frozen=True)
class _RecordedFrame:
    observation: VLAObservation
    requested_action: FloatArray
    command_sequence_id: int
    issued_at_s: float
    evaluated_at_s: float
    reset_before: bool
    reset_at_s: float | None
    input_action_semantics_id: str
    input_action_semantics_sha256: str
    decision: WatchdogDecision


class LeRobotEpisodeRecorder:
    """Record actuator-bound commands with LeRobot-style frame projections."""

    def __init__(
        self,
        *,
        episode_id: str,
        source: str,
        watchdog_config: CommandWatchdogConfig = CommandWatchdogConfig(),
    ) -> None:
        if not episode_id.strip() or not source.strip():
            raise ValueError("episode ID and source must be nonempty")
        self.episode_id = episode_id
        self.source = source
        self.watchdog_config = watchdog_config
        self.watchdog = ActuatorCommandWatchdog(watchdog_config)
        self.frame_adapter = LeRobotFrameAdapter()
        self._records: list[_RecordedFrame] = []

    @property
    def frame_count(self) -> int:
        return len(self._records)

    def append(
        self,
        observation: VLAObservation,
        requested_action: ArrayLike,
        *,
        command_sequence_id: int,
        issued_at_s: float,
        evaluated_at_s: float,
        reset_before: bool = False,
        reset_at_s: float | None = None,
        action_semantics_id: str = PANDA_RUNTIME_ACTION_SPACE_ID,
        action_semantics_sha256: str = PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    ) -> WatchdogDecision:
        action = np.asarray(requested_action, dtype=float)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError("episode requested action must be a finite 8-vector")
        if reset_before:
            if reset_at_s is None:
                raise ValueError("reset_at_s is required when reset_before is true")
            self.watchdog.reset(evaluated_at_s=reset_at_s)
        elif reset_at_s is not None:
            raise ValueError("reset_at_s requires reset_before")
        decision = self.watchdog.evaluate(
            action,
            command_sequence_id=command_sequence_id,
            observation_sequence_id=observation.sequence_id,
            captured_at_s=observation.captured_at_s,
            issued_at_s=issued_at_s,
            evaluated_at_s=evaluated_at_s,
            gripper_position=float(observation.gripper_position[0]),
            action_semantics_id=action_semantics_id,
            action_semantics_sha256=action_semantics_sha256,
        )
        self.frame_adapter.to_frame(
            observation,
            decision.action,
            action_semantics_id=PANDA_RUNTIME_ACTION_SPACE_ID,
            action_semantics_sha256=PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
        )
        copied = action.copy()
        copied.flags.writeable = False
        self._records.append(
            _RecordedFrame(
                observation=observation,
                requested_action=copied,
                command_sequence_id=command_sequence_id,
                issued_at_s=float(issued_at_s),
                evaluated_at_s=float(evaluated_at_s),
                reset_before=bool(reset_before),
                reset_at_s=(None if reset_at_s is None else float(reset_at_s)),
                input_action_semantics_id=action_semantics_id,
                input_action_semantics_sha256=action_semantics_sha256,
                decision=decision,
            )
        )
        return decision

    def export(self, directory: Path) -> Path:
        root = directory.resolve()
        _require(not root.exists(), f"episode directory already exists: {root}")
        _require(bool(self._records), "cannot export an empty episode")
        root.mkdir(parents=True)
        observations = [record.observation for record in self._records]
        np.savez_compressed(
            root / "episode.npz",
            exterior_images=np.stack(
                [observation.exterior_image for observation in observations]
            ).astype(np.uint8),
            wrist_images=np.stack(
                [observation.wrist_image for observation in observations]
            ).astype(np.uint8),
            states=np.stack([observation.state for observation in observations]).astype(
                np.float32
            ),
            requested_actions=np.stack(
                [record.requested_action for record in self._records]
            ).astype(np.float64),
            dispatched_actions=np.stack(
                [record.decision.action for record in self._records]
            ).astype(np.float64),
            watchdog_gripper_positions=np.asarray(
                [
                    float(record.observation.gripper_position[0])
                    for record in self._records
                ],
                dtype=np.float64,
            ),
            command_sequence_ids=np.asarray(
                [record.command_sequence_id for record in self._records],
                dtype=np.int64,
            ),
            observation_sequence_ids=np.asarray(
                [observation.sequence_id for observation in observations],
                dtype=np.int64,
            ),
            captured_at_s=np.asarray(
                [observation.captured_at_s for observation in observations],
                dtype=np.float64,
            ),
            issued_at_s=np.asarray(
                [record.issued_at_s for record in self._records], dtype=np.float64
            ),
            evaluated_at_s=np.asarray(
                [record.evaluated_at_s for record in self._records],
                dtype=np.float64,
            ),
            reset_before=np.asarray(
                [record.reset_before for record in self._records], dtype=np.bool_
            ),
            reset_at_s=np.asarray(
                [
                    -1.0 if record.reset_at_s is None else record.reset_at_s
                    for record in self._records
                ],
                dtype=np.float64,
            ),
        )
        rows = []
        for frame_index, record in enumerate(self._records):
            observation = record.observation
            hashes = {
                "exterior_image_sha256": frame_array_sha256(
                    observation.exterior_image, dtype="uint8"
                ),
                "wrist_image_sha256": frame_array_sha256(
                    observation.wrist_image, dtype="uint8"
                ),
                "state_sha256": frame_array_sha256(
                    observation.state, dtype="<f4"
                ),
                "requested_action_sha256": frame_array_sha256(
                    record.requested_action, dtype="<f8"
                ),
                "dispatched_action_sha256": frame_array_sha256(
                    record.decision.action, dtype="<f8"
                ),
                "task_sha256": _text_sha256(observation.prompt),
            }
            rows.append(
                {
                    "schema_version": LEROBOT_FRAME_SCHEMA,
                    "frame_index": frame_index,
                    "command_sequence_id": record.command_sequence_id,
                    "observation_sequence_id": observation.sequence_id,
                    "captured_at_s": observation.captured_at_s,
                    "issued_at_s": record.issued_at_s,
                    "evaluated_at_s": record.evaluated_at_s,
                    "watchdog_gripper_position": float(
                        observation.gripper_position[0]
                    ),
                    "reset_before": record.reset_before,
                    "reset_at_s": record.reset_at_s,
                    "task": observation.prompt,
                    "input_action_semantics_id": record.input_action_semantics_id,
                    "input_action_semantics_sha256": (
                        record.input_action_semantics_sha256
                    ),
                    "watchdog_decision": record.decision.to_dict(),
                    "hashes": hashes,
                }
            )
        with (root / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        metadata = {
            "schema_version": LEROBOT_EPISODE_SCHEMA,
            "scope": "cpu_only_lerobot_style_frame_and_actuator_boundary",
            "episode_id": self.episode_id,
            "source": self.source,
            "frame_count": len(rows),
            "lerobot_style_interface": {
                "add_frame_keys": list(LEROBOT_STYLE_FRAME_KEYS),
                "official_lerobot_dataset_storage": False,
                "official_lerobot_runtime_validated": False,
            },
            "runtime_action_semantics": runtime_action_semantics(),
            "runtime_action_semantics_sha256": (
                PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
            ),
            "watchdog_config": self.watchdog_config.to_dict(),
            "claims": {
                "official_lerobot_package_used": False,
                "official_lerobot_dataset_validated": False,
                "physical_robot_connected": False,
                "hardware_estop_integrated": False,
                "hard_realtime_guarantee": False,
                "robot_safety_certification": False,
                "learned_policy_checkpoint_executed": False,
            },
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reason_counts = Counter(record.decision.reason for record in self._records)
        summary = {
            "schema_version": LEROBOT_SUMMARY_SCHEMA,
            "episode_id": self.episode_id,
            "frames": len(rows),
            "executed_commands": sum(
                record.decision.status == "execute" for record in self._records
            ),
            "held_commands": sum(
                record.decision.status == "hold" for record in self._records
            ),
            "reset_events": sum(record.reset_before for record in self._records),
            "reason_counts": dict(sorted(reason_counts.items())),
            "watchdog_metrics": self.watchdog.metrics(),
            "deterministic_replay_required": True,
        }
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_manifest(root)
        validate_lerobot_episode(root)
        return root


def _load_arrays(root: Path) -> dict[str, np.ndarray]:
    expected = {
        "exterior_images",
        "wrist_images",
        "states",
        "requested_actions",
        "dispatched_actions",
        "watchdog_gripper_positions",
        "command_sequence_ids",
        "observation_sequence_ids",
        "captured_at_s",
        "issued_at_s",
        "evaluated_at_s",
        "reset_before",
        "reset_at_s",
    }
    try:
        with np.load(root / "episode.npz", allow_pickle=False) as archive:
            _require(set(archive.files) == expected, "episode NPZ keys mismatch")
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except LeRobotEpisodeError:
        raise
    except Exception as error:
        raise LeRobotEpisodeError("episode NPZ cannot be loaded") from error


def _validate_array_contract(arrays: Mapping[str, np.ndarray], count: int) -> None:
    _require(
        arrays["exterior_images"].shape == (count, *DROID_IMAGE_SHAPE)
        and arrays["exterior_images"].dtype == np.uint8
        and arrays["wrist_images"].shape == (count, *DROID_IMAGE_SHAPE)
        and arrays["wrist_images"].dtype == np.uint8,
        "episode image arrays are invalid",
    )
    expected_shapes = {
        "states": (count, 8),
        "requested_actions": (count, 8),
        "dispatched_actions": (count, 8),
        "watchdog_gripper_positions": (count,),
        "command_sequence_ids": (count,),
        "observation_sequence_ids": (count,),
        "captured_at_s": (count,),
        "issued_at_s": (count,),
        "evaluated_at_s": (count,),
        "reset_before": (count,),
        "reset_at_s": (count,),
    }
    _require(
        all(arrays[key].shape == shape for key, shape in expected_shapes.items()),
        "episode array shapes are invalid",
    )
    _require(
        arrays["states"].dtype == np.float32
        and arrays["requested_actions"].dtype == np.float64
        and arrays["dispatched_actions"].dtype == np.float64
        and arrays["watchdog_gripper_positions"].dtype == np.float64
        and arrays["command_sequence_ids"].dtype == np.int64
        and arrays["observation_sequence_ids"].dtype == np.int64
        and arrays["reset_before"].dtype == np.bool_
        and all(
            arrays[key].dtype == np.float64
            for key in ("captured_at_s", "issued_at_s", "evaluated_at_s", "reset_at_s")
        ),
        "episode array dtypes are invalid",
    )
    _require(
        all(np.all(np.isfinite(arrays[key])) for key in (
            "states",
            "requested_actions",
            "dispatched_actions",
            "watchdog_gripper_positions",
            "captured_at_s",
            "issued_at_s",
            "evaluated_at_s",
            "reset_at_s",
        )),
        "episode numeric arrays contain non-finite values",
    )
    command_ids = arrays["command_sequence_ids"]
    observation_ids = arrays["observation_sequence_ids"]
    captured = arrays["captured_at_s"]
    issued = arrays["issued_at_s"]
    evaluated = arrays["evaluated_at_s"]
    _require(
        np.all(command_ids >= 0)
        and np.all(np.diff(command_ids) > 0)
        and np.all(observation_ids >= 0)
        and np.all(np.diff(observation_ids) >= 0),
        "episode sequence IDs are not monotonic",
    )
    _require(
        np.all(np.diff(captured) >= 0.0)
        and np.all(np.diff(issued) >= 0.0)
        and np.all(np.diff(evaluated) >= 0.0)
        and np.all(captured <= issued)
        and np.all(issued <= evaluated),
        "episode timestamps are not monotonic or ordered",
    )


def _config_from_metadata(value: object) -> CommandWatchdogConfig:
    _require(isinstance(value, Mapping), "watchdog configuration is invalid")
    try:
        return CommandWatchdogConfig(
            max_observation_age_s=float(value["max_observation_age_s"]),
            max_action_age_s=float(value["max_action_age_s"]),
            heartbeat_timeout_s=float(value["heartbeat_timeout_s"]),
            fallback_gripper_position=float(value["fallback_gripper_position"]),
            action_semantics_id=str(value["action_semantics_id"]),
            action_semantics_sha256=str(value["action_semantics_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LeRobotEpisodeError("watchdog configuration is invalid") from error


def validate_lerobot_episode(directory: Path) -> dict[str, object]:
    """Verify hashes, schema, frame mapping, and every watchdog decision."""

    root = directory.resolve()
    inventory_sha = _validate_manifest(root)
    metadata = _json(root / "metadata.json")
    summary = _json(root / "summary.json")
    rows = _jsonl(root / "frames.jsonl")
    _require(
        metadata.get("schema_version") == LEROBOT_EPISODE_SCHEMA
        and metadata.get("scope")
        == "cpu_only_lerobot_style_frame_and_actuator_boundary",
        "episode metadata schema/scope mismatch",
    )
    _require(
        summary.get("schema_version") == LEROBOT_SUMMARY_SCHEMA,
        "episode summary schema mismatch",
    )
    count = len(rows)
    _require(
        metadata.get("frame_count") == count
        and summary.get("frames") == count,
        "episode frame count mismatch",
    )
    interface = metadata.get("lerobot_style_interface")
    _require(
        isinstance(interface, Mapping)
        and interface.get("add_frame_keys") == list(LEROBOT_STYLE_FRAME_KEYS)
        and interface.get("official_lerobot_dataset_storage") is False
        and interface.get("official_lerobot_runtime_validated") is False,
        "LeRobot-style interface claim is invalid",
    )
    _require(
        metadata.get("runtime_action_semantics") == runtime_action_semantics()
        and metadata.get("runtime_action_semantics_sha256")
        == PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
        "runtime action semantics mismatch",
    )
    claims = metadata.get("claims")
    _require(
        isinstance(claims, Mapping)
        and bool(claims)
        and not any(bool(value) for value in claims.values()),
        "episode claim boundary is invalid",
    )
    config = _config_from_metadata(metadata.get("watchdog_config"))
    arrays = _load_arrays(root)
    _validate_array_contract(arrays, count)
    watchdog = ActuatorCommandWatchdog(config)
    adapter = LeRobotFrameAdapter()
    replayed_decisions: list[WatchdogDecision] = []
    reason_counts: Counter[str] = Counter()
    reset_events = 0
    for index, row in enumerate(rows):
        _require(
            row.get("schema_version") == LEROBOT_FRAME_SCHEMA
            and row.get("frame_index") == index,
            f"frame {index} schema/index mismatch",
        )
        command_id = int(arrays["command_sequence_ids"][index])
        observation_id = int(arrays["observation_sequence_ids"][index])
        captured = float(arrays["captured_at_s"][index])
        issued = float(arrays["issued_at_s"][index])
        evaluated = float(arrays["evaluated_at_s"][index])
        reset_before = bool(arrays["reset_before"][index])
        reset_at = float(arrays["reset_at_s"][index])
        watchdog_gripper = float(arrays["watchdog_gripper_positions"][index])
        _require(
            row.get("command_sequence_id") == command_id
            and row.get("observation_sequence_id") == observation_id
            and row.get("captured_at_s") == captured
            and row.get("issued_at_s") == issued
            and row.get("evaluated_at_s") == evaluated
            and row.get("watchdog_gripper_position") == watchdog_gripper
            and row.get("reset_before") is reset_before,
            f"frame {index} scalar/array mismatch",
        )
        if reset_before:
            _require(row.get("reset_at_s") == reset_at, f"frame {index} reset mismatch")
            watchdog.reset(evaluated_at_s=reset_at)
            reset_events += 1
        else:
            _require(
                row.get("reset_at_s") is None and reset_at == -1.0,
                f"frame {index} unexpected reset",
            )
        task = row.get("task")
        _require(isinstance(task, str) and task.strip(), f"frame {index} task invalid")
        state = arrays["states"][index]
        observation = VLAObservation(
            exterior_image=arrays["exterior_images"][index],
            wrist_image=arrays["wrist_images"][index],
            joint_position=state[:7],
            gripper_position=state[7:],
            prompt=task,
            sequence_id=observation_id,
            captured_at_s=captured,
        )
        semantics_id = str(row.get("input_action_semantics_id", ""))
        semantics_sha = str(row.get("input_action_semantics_sha256", ""))
        decision = watchdog.evaluate(
            arrays["requested_actions"][index],
            command_sequence_id=command_id,
            observation_sequence_id=observation_id,
            captured_at_s=captured,
            issued_at_s=issued,
            evaluated_at_s=evaluated,
            gripper_position=watchdog_gripper,
            action_semantics_id=semantics_id,
            action_semantics_sha256=semantics_sha,
        )
        _require(
            np.isclose(float(state[7]), watchdog_gripper, atol=1e-7, rtol=0.0),
            f"frame {index} LeRobot/watchdog gripper mismatch",
        )
        _require(
            row.get("watchdog_decision") == decision.to_dict(),
            f"frame {index} watchdog decision is not reproducible",
        )
        _require(
            np.array_equal(decision.action, arrays["dispatched_actions"][index]),
            f"frame {index} dispatched action mismatch",
        )
        frame = adapter.to_frame(
            observation,
            decision.action,
            action_semantics_id=PANDA_RUNTIME_ACTION_SPACE_ID,
            action_semantics_sha256=PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
        )
        _require(tuple(frame) == LEROBOT_STYLE_FRAME_KEYS, "LeRobot frame keys changed")
        _require(
            np.array_equal(
                frame["action"], decision.action.astype(np.float32)
            ),
            f"frame {index} LeRobot action projection mismatch",
        )
        hashes = {
            "exterior_image_sha256": frame_array_sha256(
                frame["observation.images.exterior"], dtype="uint8"
            ),
            "wrist_image_sha256": frame_array_sha256(
                frame["observation.images.wrist"], dtype="uint8"
            ),
            "state_sha256": frame_array_sha256(
                frame["observation.state"], dtype="<f4"
            ),
            "requested_action_sha256": frame_array_sha256(
                arrays["requested_actions"][index], dtype="<f8"
            ),
            "dispatched_action_sha256": frame_array_sha256(
                arrays["dispatched_actions"][index], dtype="<f8"
            ),
            "task_sha256": _text_sha256(task),
        }
        _require(row.get("hashes") == hashes, f"frame {index} content hash mismatch")
        replayed_decisions.append(decision)
        reason_counts[decision.reason] += 1
    expected_summary = {
        "schema_version": LEROBOT_SUMMARY_SCHEMA,
        "episode_id": str(metadata["episode_id"]),
        "frames": count,
        "executed_commands": sum(
            decision.status == "execute" for decision in replayed_decisions
        ),
        "held_commands": sum(
            decision.status == "hold" for decision in replayed_decisions
        ),
        "reset_events": reset_events,
        "reason_counts": dict(sorted(reason_counts.items())),
        "watchdog_metrics": watchdog.metrics(),
        "deterministic_replay_required": True,
    }
    _require(summary == expected_summary, "episode summary is not reproducible")
    return {
        "valid": True,
        "directory": str(root),
        "episode_id": metadata["episode_id"],
        "frames": count,
        "executed_commands": expected_summary["executed_commands"],
        "held_commands": expected_summary["held_commands"],
        "reset_events": reset_events,
        "reason_counts": expected_summary["reason_counts"],
        "manifest_inventory_sha256": inventory_sha,
        "official_lerobot_runtime_validated": False,
        "physical_robot_connected": False,
        "checks": [
            "manifest_sizes_and_hashes",
            "array_schema_shape_dtype",
            "sequence_and_timestamp_monotonicity",
            "lerobot_style_frame_mapping",
            "frame_content_hashes",
            "watchdog_decisions_recomputed",
            "summary_recomputed",
            "explicit_claim_boundary",
        ],
    }


def replay_lerobot_episode(directory: Path) -> dict[str, object]:
    result = validate_lerobot_episode(directory)
    return {
        **result,
        "replayed_frames": result["frames"],
        "all_watchdog_decisions_matched": True,
    }


def _smoke_observation(sequence_id: int, captured_at_s: float) -> VLAObservation:
    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    exterior = np.zeros(DROID_IMAGE_SHAPE, dtype=np.uint8)
    wrist = np.zeros(DROID_IMAGE_SHAPE, dtype=np.uint8)
    exterior[sequence_id % 8 :: 8, :, 0] = 60 + 10 * sequence_id
    wrist[:, sequence_id % 8 :: 8, 1] = 90 + 10 * sequence_id
    q = mujoco_scenarios()["free_space"].start.copy()
    q[0] += 0.01 * sequence_id
    return VLAObservation(
        exterior_image=exterior,
        wrist_image=wrist,
        joint_position=q,
        gripper_position=np.array([0.65]),
        prompt="move to the inspection pose",
        sequence_id=sequence_id,
        captured_at_s=captured_at_s,
    )


def run_lerobot_episode_smoke(directory: Path) -> Path:
    """Exercise accepted, stale, latched, reset, and recovered command paths."""

    recorder = LeRobotEpisodeRecorder(
        episode_id="lerobot_style_watchdog_smoke_001",
        source="scripted_non_learned_cpu_bridge_fixture",
        watchdog_config=CommandWatchdogConfig(
            max_observation_age_s=0.2,
            max_action_age_s=0.1,
            heartbeat_timeout_s=0.25,
        ),
    )
    action = np.zeros(8, dtype=float)
    action[:7] = [0.08, -0.04, 0.02, 0.0, 0.0, 0.01, -0.02]
    action[7] = 0.65
    observation0 = _smoke_observation(0, 100.0)
    recorder.append(
        observation0,
        action,
        command_sequence_id=0,
        issued_at_s=100.03,
        evaluated_at_s=100.04,
    )
    recorder.append(
        observation0,
        action * np.array([0.8] * 7 + [1.0]),
        command_sequence_id=1,
        issued_at_s=100.08,
        evaluated_at_s=100.09,
    )
    recorder.append(
        _smoke_observation(1, 100.10),
        action,
        command_sequence_id=2,
        issued_at_s=100.31,
        evaluated_at_s=100.32,
    )
    recorder.append(
        _smoke_observation(2, 100.33),
        action,
        command_sequence_id=3,
        issued_at_s=100.34,
        evaluated_at_s=100.35,
    )
    recorder.append(
        _smoke_observation(3, 100.36),
        action,
        command_sequence_id=4,
        issued_at_s=100.37,
        evaluated_at_s=100.38,
        reset_before=True,
        reset_at_s=100.36,
    )
    return recorder.export(directory)
