"""Cross-file validation for schema-v5 online VLA run artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ONLINE_ARTIFACT_SCHEMA_VERSION = 5


class ArtifactValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactValidationResult:
    directory: str
    schema_version: int
    episodes: int
    observation_cycles: int
    policy_queries: int
    action_rows: int
    full_observation_frames: int
    videos_decoded: int
    aggregate_sha256: str
    checks: tuple[str, ...]

    def metrics(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "schema_version": self.schema_version,
            "episodes": self.episodes,
            "observation_cycles": self.observation_cycles,
            "policy_queries": self.policy_queries,
            "action_rows": self.action_rows,
            "full_observation_frames": self.full_observation_frames,
            "videos_decoded": self.videos_decoded,
            "aggregate_sha256": self.aggregate_sha256,
            "checks": list(self.checks),
            "valid": True,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def _json(path: Path) -> object:
    _require(path.is_file() and path.stat().st_size > 0, f"missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid JSON: {path}") from error


def _csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file() and path.stat().st_size > 0, f"missing file: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ArtifactValidationError(f"invalid CSV: {path}") from error
    _require(bool(rows), f"CSV has no data rows: {path}")
    return rows


def _key(row: dict[str, object] | dict[str, str]) -> tuple[str, float, int]:
    try:
        return (
            str(row["scenario"]),
            float(row["payload_mass"]),
            int(row["execution_horizon"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError("artifact row has an invalid episode key") from error


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise ArtifactValidationError(f"invalid boolean value: {value!r}")


def _image_hash(path: Path) -> str:
    _require(path.is_file() and path.stat().st_size > 0, f"missing image: {path}")
    try:
        image = imageio.imread(path)
    except Exception as error:
        raise ArtifactValidationError(f"image cannot be decoded: {path}") from error
    _require(image.shape == (224, 224, 3), f"unexpected image shape: {path}")
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _array_image_hash(image: np.ndarray, label: str) -> str:
    _require(
        image.shape == (224, 224, 3) and image.dtype == np.uint8,
        f"unexpected recorded image shape/dtype: {label}",
    )
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _decode_video(path: Path) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing video: {path}")
    try:
        reader = imageio.get_reader(path)
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()
    except Exception as error:
        raise ArtifactValidationError(f"video cannot be decoded: {path}") from error
    _require(
        frame.ndim == 3 and frame.shape[2] == 3 and float(frame.std()) > 1.0,
        f"video first frame is blank or malformed: {path}",
    )


def validate_online_artifact(
    directory: Path, *, decode_videos: bool = False
) -> ArtifactValidationResult:
    root = directory.resolve()
    _require(root.is_dir(), f"artifact directory does not exist: {root}")
    aggregate_path = root / "aggregate.json"
    aggregate = _json(aggregate_path)
    _require(
        isinstance(aggregate, list) and bool(aggregate),
        "aggregate.json must contain a nonempty list",
    )
    _require(
        all(isinstance(row, dict) for row in aggregate),
        "aggregate rows must be mappings",
    )
    rows: list[dict[str, object]] = [dict(row) for row in aggregate]
    schema_versions = {
        int(row.get("artifact_schema_version", -1)) for row in rows
    }
    _require(
        schema_versions == {ONLINE_ARTIFACT_SCHEMA_VERSION},
        "validator requires online artifact schema version "
        f"{ONLINE_ARTIFACT_SCHEMA_VERSION}, got {sorted(schema_versions)}",
    )

    episode_rows = _csv(root / "per_episode.csv")
    chunk_rows = _csv(root / "per_chunk.csv")
    action_rows = _csv(root / "per_action.csv")
    _require(len(episode_rows) == len(rows), "per_episode row count mismatch")
    row_keys = [_key(row) for row in rows]
    _require(len(row_keys) == len(set(row_keys)), "episode keys are not unique")
    _require(
        set(row_keys) == {_key(row) for row in episode_rows},
        "per_episode keys do not match aggregate.json",
    )
    known_keys = set(row_keys)
    _require(
        all(_key(row) in known_keys for row in chunk_rows),
        "per_chunk contains an unknown episode key",
    )
    _require(
        all(_key(row) in known_keys for row in action_rows),
        "per_action contains an unknown episode key",
    )

    total_cycles = 0
    total_policy_queries = 0
    videos_decoded = 0
    failure_audit_traces = 0
    full_observation_frames = 0
    for row, episode_key in zip(rows, row_keys, strict=True):
        chunks = [item for item in chunk_rows if _key(item) == episode_key]
        actions = [item for item in action_rows if _key(item) == episode_key]
        cycles = int(row["observation_cycles"])
        policy_queries = int(row["policy_queries"])
        action_steps = int(row["action_steps"])
        _require(len(chunks) == cycles, f"chunk count mismatch for {episode_key}")
        _require(
            len(actions) == cycles * 15,
            f"action audit count mismatch for {episode_key}",
        )
        _require(
            sum(_bool(item["policy_inference_attempted"]) for item in chunks)
            == policy_queries,
            f"policy query count mismatch for {episode_key}",
        )
        _require(
            sum(_bool(item["executed"]) for item in actions) == action_steps,
            f"executed action count mismatch for {episode_key}",
        )
        _require(
            int(row["camera_audit_queries"]) == cycles,
            f"camera audit count mismatch for {episode_key}",
        )
        safe_from_counts = all(
            int(row[field]) == 0
            for field in (
                "obstacle_contact_steps",
                "self_contact_steps",
                "joint_limit_violation_steps",
            )
        )
        _require(
            _bool(row["physical_safe"]) == safe_from_counts,
            f"physical safety fields disagree for {episode_key}",
        )
        _require(
            _bool(row["safe_task_success"])
            == (_bool(row["task_success"]) and _bool(row["physical_safe"])),
            f"safe task success fields disagree for {episode_key}",
        )

        trace_path = root / str(row["trace"])
        _require(
            trace_path.is_file() and trace_path.stat().st_size > 0,
            f"missing trace for {episode_key}",
        )
        try:
            with np.load(trace_path, allow_pickle=False) as trace:
                guarded = np.asarray(trace["guarded_action_chunks"])
                raw = np.asarray(trace["raw_action_chunks"])
                predicted = np.asarray(trace["predicted_position_chunks"])
                attempted = np.asarray(trace["policy_inference_attempted"])
                exterior_hashes = np.asarray(trace["exterior_image_sha256"])
                wrist_hashes = np.asarray(trace["wrist_image_sha256"])
                exterior_thumbnails = np.asarray(
                    trace["exterior_image_thumbnails"]
                )
                wrist_thumbnails = np.asarray(trace["wrist_image_thumbnails"])
                failure_fields = (
                    "failure_stages",
                    "failure_types",
                    "failure_messages",
                )
                present_failure_fields = {
                    field for field in failure_fields if field in trace.files
                }
                _require(
                    not present_failure_fields
                    or present_failure_fields == set(failure_fields),
                    f"incomplete failure audit trace for {episode_key}",
                )
                failure_values = (
                    {
                        field: np.asarray(trace[field])
                        for field in failure_fields
                    }
                    if present_failure_fields
                    else None
                )
                full_image_fields = ("exterior_images", "wrist_images")
                present_full_image_fields = {
                    field
                    for field in full_image_fields
                    if field in trace.files
                }
                _require(
                    not present_full_image_fields
                    or present_full_image_fields == set(full_image_fields),
                    f"incomplete full observation trace for {episode_key}",
                )
                full_image_values = (
                    {
                        field: np.asarray(trace[field])
                        for field in full_image_fields
                    }
                    if present_full_image_fields
                    else None
                )
        except (OSError, KeyError, ValueError) as error:
            raise ArtifactValidationError(
                f"invalid NPZ trace for {episode_key}"
            ) from error
        _require(guarded.shape == (cycles, 15, 8), "guarded action shape mismatch")
        _require(raw.shape == (cycles, 15, 8), "raw action shape mismatch")
        _require(predicted.shape == (cycles, 16, 7), "prediction shape mismatch")
        _require(attempted.shape == (cycles,), "attempted-query shape mismatch")
        _require(
            int(np.count_nonzero(attempted)) == policy_queries,
            f"NPZ policy query count mismatch for {episode_key}",
        )
        _require(
            exterior_thumbnails.shape == (cycles, 16, 16, 3)
            and wrist_thumbnails.shape == (cycles, 16, 16, 3),
            f"camera thumbnail shape mismatch for {episode_key}",
        )
        csv_exterior_hashes = [item["exterior_image_sha256"] for item in chunks]
        csv_wrist_hashes = [item["wrist_image_sha256"] for item in chunks]
        _require(
            exterior_hashes.tolist() == csv_exterior_hashes,
            f"exterior hash trace mismatch for {episode_key}",
        )
        _require(
            wrist_hashes.tolist() == csv_wrist_hashes,
            f"wrist hash trace mismatch for {episode_key}",
        )
        full_images_declared = _bool(
            row.get("full_observations_recorded", False)
        )
        _require(
            full_images_declared == (full_image_values is not None),
            f"full observation declaration mismatch for {episode_key}",
        )
        if full_image_values is not None:
            exterior_images = full_image_values["exterior_images"]
            wrist_images = full_image_values["wrist_images"]
            _require(
                exterior_images.shape == (cycles, 224, 224, 3)
                and wrist_images.shape == (cycles, 224, 224, 3)
                and exterior_images.dtype == np.uint8
                and wrist_images.dtype == np.uint8,
                f"full observation array shape/dtype mismatch for {episode_key}",
            )
            _require(
                [
                    _array_image_hash(image, f"{episode_key}:exterior:{index}")
                    for index, image in enumerate(exterior_images)
                ]
                == csv_exterior_hashes,
                f"full exterior observation hashes mismatch for {episode_key}",
            )
            _require(
                [
                    _array_image_hash(image, f"{episode_key}:wrist:{index}")
                    for index, image in enumerate(wrist_images)
                ]
                == csv_wrist_hashes,
                f"full wrist observation hashes mismatch for {episode_key}",
            )
            expected_full_frames = cycles * 2
            _require(
                int(row.get("full_observation_frame_count", -1))
                == expected_full_frames,
                f"full observation frame count mismatch for {episode_key}",
            )
            full_observation_frames += expected_full_frames
        if failure_values is not None:
            for field, csv_field in (
                ("failure_stages", "failure_stage"),
                ("failure_types", "failure_type"),
                ("failure_messages", "failure_message"),
            ):
                _require(
                    failure_values[field].shape == (cycles,),
                    f"{field} shape mismatch for {episode_key}",
                )
                _require(
                    failure_values[field].tolist()
                    == [item[csv_field] for item in chunks],
                    f"{field} trace mismatch for {episode_key}",
                )
            failure_audit_traces += 1
        _require(
            len(set(csv_exterior_hashes))
            == int(row["unique_exterior_observation_hashes"]),
            f"exterior unique hash count mismatch for {episode_key}",
        )
        _require(
            len(set(csv_wrist_hashes))
            == int(row["unique_wrist_observation_hashes"]),
            f"wrist unique hash count mismatch for {episode_key}",
        )
        _require(
            _image_hash(root / str(row["external_image"]))
            == csv_exterior_hashes[0],
            f"first exterior image hash mismatch for {episode_key}",
        )
        _require(
            _image_hash(root / str(row["wrist_image"])) == csv_wrist_hashes[0],
            f"first wrist image hash mismatch for {episode_key}",
        )
        _require(
            _image_hash(root / str(row["last_external_image"]))
            == csv_exterior_hashes[-1],
            f"last exterior image hash mismatch for {episode_key}",
        )
        _require(
            _image_hash(root / str(row["last_wrist_image"]))
            == csv_wrist_hashes[-1],
            f"last wrist image hash mismatch for {episode_key}",
        )
        if "validated_remote_chunks" in row:
            _require(
                _bool(row["remote_policy_response_validated"])
                == (int(row["validated_remote_chunks"]) > 0),
                f"remote response fields disagree for {episode_key}",
            )
        video_path = row.get("video_path")
        if decode_videos and video_path:
            _decode_video(root / str(video_path))
            videos_decoded += 1
        total_cycles += cycles
        total_policy_queries += policy_queries

    config = _json(root / "config.json")
    environment = _json(root / "environment.json")
    _require(isinstance(config, dict), "config.json must contain a mapping")
    _require(
        isinstance(environment, dict)
        and isinstance(environment.get("vla_online"), dict)
        and int(environment["vla_online"].get("artifact_schema_version", -1))
        == ONLINE_ARTIFACT_SCHEMA_VERSION,
        "environment schema does not match aggregate.json",
    )
    for required in ("overview.png", "summary.md"):
        path = root / required
        _require(path.is_file() and path.stat().st_size > 0, f"missing file: {path}")
    aggregate_sha256 = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    checks = (
        "required_files_nonempty",
        "episode_keys_aligned",
        "query_and_action_counts_aligned",
        "safety_fields_consistent",
        "npz_shapes_and_counts_aligned",
        "camera_hashes_and_thumbnails_aligned",
        "saved_observation_hashes_verified",
    )
    if failure_audit_traces == len(rows):
        checks += ("failure_audit_aligned",)
    if full_observation_frames:
        checks += ("full_observation_frames_hashed",)
    checks += (("videos_decoded",) if decode_videos else ())
    return ArtifactValidationResult(
        directory=str(root),
        schema_version=ONLINE_ARTIFACT_SCHEMA_VERSION,
        episodes=len(rows),
        observation_cycles=total_cycles,
        policy_queries=total_policy_queries,
        action_rows=len(action_rows),
        full_observation_frames=full_observation_frames,
        videos_decoded=videos_decoded,
        aggregate_sha256=aggregate_sha256,
        checks=checks,
    )
