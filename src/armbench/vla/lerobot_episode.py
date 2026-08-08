"""Hash-manifested LeRobot-style episode export and deterministic replay."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
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
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    json_equal,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    write_json,
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
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_FILE_FIELDS = {"path", "size_bytes", "sha256"}
_METADATA_FIELDS = {
    "schema_version",
    "scope",
    "episode_id",
    "source",
    "frame_count",
    "lerobot_style_interface",
    "runtime_action_semantics",
    "runtime_action_semantics_sha256",
    "watchdog_config",
    "claims",
}
_INTERFACE_FIELDS = {
    "add_frame_keys",
    "official_lerobot_dataset_storage",
    "official_lerobot_runtime_validated",
}
_CLAIM_FIELDS = {
    "official_lerobot_package_used",
    "official_lerobot_dataset_validated",
    "physical_robot_connected",
    "hardware_estop_integrated",
    "hard_realtime_guarantee",
    "robot_safety_certification",
    "learned_policy_checkpoint_executed",
}
_WATCHDOG_CONFIG_FIELDS = {
    "max_observation_age_s",
    "max_action_age_s",
    "heartbeat_timeout_s",
    "fallback_gripper_position",
    "action_semantics_id",
    "action_semantics_sha256",
}
_FRAME_FIELDS = {
    "schema_version",
    "frame_index",
    "command_sequence_id",
    "observation_sequence_id",
    "captured_at_s",
    "issued_at_s",
    "evaluated_at_s",
    "watchdog_gripper_position",
    "reset_before",
    "reset_at_s",
    "task",
    "input_action_semantics_id",
    "input_action_semantics_sha256",
    "watchdog_decision",
    "hashes",
}
_FRAME_HASH_FIELDS = {
    "exterior_image_sha256",
    "wrist_image_sha256",
    "state_sha256",
    "requested_action_sha256",
    "dispatched_action_sha256",
    "task_sha256",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "episode_id",
    "frames",
    "executed_commands",
    "held_commands",
    "reset_events",
    "reason_counts",
    "watchdog_metrics",
    "deterministic_replay_required",
}


class LeRobotEpisodeError(ValueError):
    """Raised when an episode cannot be exported or deterministically replayed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LeRobotEpisodeError(message)


def _is_json_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _runtime_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _text_sha256(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _write_manifest(root: Path) -> None:
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    document = {
        "schema_version": LEROBOT_MANIFEST_SCHEMA,
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json(files)),
    }
    write_json(root / "manifest.json", document)


def _validate_manifest(root: Path) -> str:
    _require(root.is_dir(), f"LeRobot-style episode not found: {root}")
    _require(
        {path.name for path in root.iterdir() if path.is_file()} == _EPISODE_FILES,
        "LeRobot-style episode file set is invalid",
    )
    try:
        manifest = strict_json_loads((root / "manifest.json").read_text("utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise LeRobotEpisodeError("episode manifest is invalid") from error
    _require(
        has_exact_fields(manifest, _MANIFEST_FIELDS)
        and manifest["schema_version"] == LEROBOT_MANIFEST_SCHEMA
        and isinstance(manifest["files"], list)
        and is_sha256(manifest["inventory_sha256"]),
        "episode manifest schema mismatch",
    )
    for entry in manifest["files"]:
        _require(
            has_exact_fields(entry, _MANIFEST_FILE_FIELDS)
            and isinstance(entry["path"], str)
            and type(entry["size_bytes"]) is int
            and entry["size_bytes"] >= 0
            and is_sha256(entry["sha256"]),
            "episode manifest entry schema mismatch",
        )
    expected = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    _require(
        json_equal(manifest["files"], expected),
        "episode manifest inventory mismatch",
    )
    inventory_sha = sha256_bytes(canonical_json(expected))
    _require(
        manifest["inventory_sha256"] == inventory_sha,
        "episode manifest inventory hash mismatch",
    )
    return inventory_sha


def _json(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text("utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise LeRobotEpisodeError(f"invalid JSON: {path.name}") from error
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = strict_json_loads(line)
                _require(
                    isinstance(value, dict),
                    f"frames.jsonl line {line_number} is not an object",
                )
                rows.append(value)
    except (OSError, TypeError, ValueError) as error:
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
        if (
            not isinstance(episode_id, str)
            or not episode_id.strip()
            or not isinstance(source, str)
            or not source.strip()
        ):
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
        raw_action = np.asarray(requested_action)
        if raw_action.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "episode requested action must be a finite numeric 8-vector"
            )
        action = np.asarray(raw_action, dtype=float)
        if (
            action.shape != (8,)
            or not np.all(np.isfinite(action))
            or not 0.0 <= action[7] <= 1.0
        ):
            raise ValueError(
                "episode requested action must be a finite numeric 8-vector "
                "with gripper in [0, 1]"
            )
        if type(reset_before) is not bool:
            raise ValueError("reset_before must be a boolean")
        if type(command_sequence_id) is not int or command_sequence_id < 0:
            raise ValueError("command_sequence_id must be a nonnegative integer")
        if type(observation.sequence_id) is not int:
            raise ValueError("observation sequence ID must be an integer")
        if not isinstance(action_semantics_id, str) or not isinstance(
            action_semantics_sha256, str
        ):
            raise ValueError("input action semantics identity must be strings")
        captured_at = _runtime_float(
            observation.captured_at_s, "observation captured_at_s"
        )
        issued_at = _runtime_float(issued_at_s, "issued_at_s")
        evaluated_at = _runtime_float(evaluated_at_s, "evaluated_at_s")
        gripper_position = _runtime_float(
            observation.gripper_position[0], "observation gripper"
        )
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("observation gripper must be in [0, 1]")
        if reset_before:
            if reset_at_s is None:
                raise ValueError("reset_at_s is required when reset_before is true")
            reset_at = _runtime_float(reset_at_s, "reset_at_s")
            self.watchdog.reset(evaluated_at_s=reset_at)
        elif reset_at_s is not None:
            raise ValueError("reset_at_s requires reset_before")
        else:
            reset_at = None
        decision = self.watchdog.evaluate(
            action,
            command_sequence_id=command_sequence_id,
            observation_sequence_id=observation.sequence_id,
            captured_at_s=captured_at,
            issued_at_s=issued_at,
            evaluated_at_s=evaluated_at,
            gripper_position=gripper_position,
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
                issued_at_s=issued_at,
                evaluated_at_s=evaluated_at,
                reset_before=reset_before,
                reset_at_s=reset_at,
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
                    "captured_at_s": float(observation.captured_at_s),
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
        write_json(root / "metadata.json", metadata)
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
        write_json(root / "summary.json", summary)
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
    _require(
        np.all((0.0 <= arrays["states"][:, 7]) & (arrays["states"][:, 7] <= 1.0))
        and np.all(
            (0.0 <= arrays["requested_actions"][:, 7])
            & (arrays["requested_actions"][:, 7] <= 1.0)
        )
        and np.all(
            (0.0 <= arrays["dispatched_actions"][:, 7])
            & (arrays["dispatched_actions"][:, 7] <= 1.0)
        )
        and np.all(
            (0.0 <= arrays["watchdog_gripper_positions"])
            & (arrays["watchdog_gripper_positions"] <= 1.0)
        ),
        "episode normalized gripper values are invalid",
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
    _require(
        has_exact_fields(value, _WATCHDOG_CONFIG_FIELDS),
        "watchdog configuration is invalid",
    )
    try:
        return CommandWatchdogConfig(
            max_observation_age_s=value["max_observation_age_s"],
            max_action_age_s=value["max_action_age_s"],
            heartbeat_timeout_s=value["heartbeat_timeout_s"],
            fallback_gripper_position=value["fallback_gripper_position"],
            action_semantics_id=value["action_semantics_id"],
            action_semantics_sha256=value["action_semantics_sha256"],
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
        has_exact_fields(metadata, _METADATA_FIELDS)
        and metadata["schema_version"] == LEROBOT_EPISODE_SCHEMA
        and metadata["scope"]
        == "cpu_only_lerobot_style_frame_and_actuator_boundary",
        "episode metadata schema/scope mismatch",
    )
    _require(
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == LEROBOT_SUMMARY_SCHEMA,
        "episode summary schema mismatch",
    )
    _require(
        isinstance(metadata["episode_id"], str)
        and bool(metadata["episode_id"].strip())
        and isinstance(metadata["source"], str)
        and bool(metadata["source"].strip())
        and type(metadata["frame_count"]) is int,
        "episode metadata identity/count is invalid",
    )
    count = len(rows)
    _require(
        metadata["frame_count"] == count
        and type(summary["frames"]) is int
        and summary["frames"] == count,
        "episode frame count mismatch",
    )
    interface = metadata["lerobot_style_interface"]
    _require(
        has_exact_fields(interface, _INTERFACE_FIELDS)
        and interface["add_frame_keys"] == list(LEROBOT_STYLE_FRAME_KEYS)
        and interface["official_lerobot_dataset_storage"] is False
        and interface["official_lerobot_runtime_validated"] is False,
        "LeRobot-style interface claim is invalid",
    )
    _require(
        json_equal(
            metadata["runtime_action_semantics"], runtime_action_semantics()
        )
        and metadata["runtime_action_semantics_sha256"]
        == PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
        "runtime action semantics mismatch",
    )
    claims = metadata["claims"]
    _require(
            has_exact_fields(claims, _CLAIM_FIELDS)
        and all(value is False for value in claims.values()),
        "episode claim boundary is invalid",
    )
    config = _config_from_metadata(metadata["watchdog_config"])
    arrays = _load_arrays(root)
    _validate_array_contract(arrays, count)
    watchdog = ActuatorCommandWatchdog(config)
    adapter = LeRobotFrameAdapter()
    replayed_decisions: list[WatchdogDecision] = []
    reason_counts: Counter[str] = Counter()
    reset_events = 0
    for index, row in enumerate(rows):
        _require(
            has_exact_fields(row, _FRAME_FIELDS)
            and row["schema_version"] == LEROBOT_FRAME_SCHEMA
            and type(row["frame_index"]) is int
            and row["frame_index"] == index
            and type(row["command_sequence_id"]) is int
            and type(row["observation_sequence_id"]) is int
            and _is_json_float(row["captured_at_s"])
            and _is_json_float(row["issued_at_s"])
            and _is_json_float(row["evaluated_at_s"])
            and _is_json_float(row["watchdog_gripper_position"])
            and type(row["reset_before"]) is bool
            and (
                row["reset_at_s"] is None
                or _is_json_float(row["reset_at_s"])
            )
            and isinstance(row["task"], str)
            and bool(row["task"].strip())
            and isinstance(row["input_action_semantics_id"], str)
            and isinstance(row["input_action_semantics_sha256"], str)
            and isinstance(row["watchdog_decision"], Mapping)
            and has_exact_fields(row["hashes"], _FRAME_HASH_FIELDS)
            and all(is_sha256(value) for value in row["hashes"].values()),
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
            row["command_sequence_id"] == command_id
            and row["observation_sequence_id"] == observation_id
            and row["captured_at_s"] == captured
            and row["issued_at_s"] == issued
            and row["evaluated_at_s"] == evaluated
            and row["watchdog_gripper_position"] == watchdog_gripper
            and row["reset_before"] is reset_before,
            f"frame {index} scalar/array mismatch",
        )
        if reset_before:
            _require(row["reset_at_s"] == reset_at, f"frame {index} reset mismatch")
            watchdog.reset(evaluated_at_s=reset_at)
            reset_events += 1
        else:
            _require(
                row["reset_at_s"] is None and reset_at == -1.0,
                f"frame {index} unexpected reset",
            )
        task = row["task"]
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
        semantics_id = row["input_action_semantics_id"]
        semantics_sha = row["input_action_semantics_sha256"]
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
            json_equal(row["watchdog_decision"], decision.to_dict()),
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
        _require(
            json_equal(row["hashes"], hashes),
            f"frame {index} content hash mismatch",
        )
        replayed_decisions.append(decision)
        reason_counts[decision.reason] += 1
    expected_summary = {
        "schema_version": LEROBOT_SUMMARY_SCHEMA,
        "episode_id": metadata["episode_id"],
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
    _require(
        json_equal(summary, expected_summary),
        "episode summary is not reproducible",
    )
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
