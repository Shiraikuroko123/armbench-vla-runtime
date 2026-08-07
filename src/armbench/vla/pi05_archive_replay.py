"""Auditable offline replay of frozen official pi0.5 LIBERO responses."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.model import MENAGERIE_COMMIT, default_panda_scene_path
from armbench.mujoco_sim.scenarios import (
    MUJOCO_SCENARIO_VERSION,
    mujoco_scenarios,
)
from armbench.vla.cartesian_adapter import (
    LIBERO_ACTION_SPACE_ID,
    LIBERO_CONTROLLER_SEMANTICS_ID,
    PANDA_KINEMATIC_CONTROL_POINT_ID,
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
)
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.types import DROID_IMAGE_SHAPE, VLAObservation


FloatArray = NDArray[np.float32]
ROOT_MANIFEST_SCHEMA = "armbench.root_manifest.v1"
TRANSITION_DESCRIPTOR_SCHEMA = "armbench.action_chunk_transition_archive.v1"
REPLAY_PROVENANCE_SCHEMA = "armbench.pi05_panda_archive_replay.v1"
REPLAY_SUMMARY_SCHEMA = "armbench.pi05_panda_archive_replay_summary.v1"
REPLAY_SCOPE = "offline_frozen_policy_response_replay"

PI05_POLICY_FAMILY = "pi0.5"
PI05_POLICY_CONFIG = "pi05_libero"
PI05_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
PI05_CHECKPOINT_SHA256 = (
    "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"
)
PI05_ACTION_ADAPTER = "openpi.pi05_libero.raw_action_v1"
PI05_ACTION_ADAPTER_SOURCE = "integrations/openpi/libero_runtime.py"

ARCHIVE_KEYS = frozenset(
    {
        "has_previous_reference",
        "previous_reference",
        "response_actions",
        "executed_window",
        "action_source",
        "next_reference",
        "executed_length",
        "old_prefix_steps",
        "new_suffix_steps",
    }
)

CSV_FIELDS = (
    "case_id",
    "selection_index",
    "source_row_index",
    "task_suite",
    "task_id",
    "method",
    "pair_id",
    "episode_id",
    "episode_index",
    "query_index",
    "scenario",
    "response_action_sha256",
    "horizon",
    "inference_latency_ms",
    "policy_inference_latency_ms",
    "server_inference_latency_ms",
    "input_max_abs",
    "input_clipped_steps",
    "velocity_limited_steps",
    "adapter_max_residual_norm",
    "adapter_min_singular_value",
    "adapter_latency_ms",
    "raw_path_safe",
    "deadline_exceeded",
    "fallback_latched",
    "fallback_reason",
    "unsafe_raw_steps",
    "intervention_steps",
    "hold_steps",
    "slew_limited_steps",
    "acceleration_override_steps",
    "max_raw_acceleration_rad_s2",
    "max_guarded_acceleration_rad_s2",
    "guard_latency_ms",
    "safe_after_guard",
    "guarded_path_safe",
    "raw_hand_displacement_m",
    "guarded_hand_displacement_m",
)

_CSV_INTEGER_FIELDS = frozenset(
    {
        "selection_index",
        "source_row_index",
        "task_id",
        "episode_index",
        "query_index",
        "horizon",
        "input_clipped_steps",
        "velocity_limited_steps",
        "unsafe_raw_steps",
        "intervention_steps",
        "hold_steps",
        "slew_limited_steps",
        "acceleration_override_steps",
    }
)
_CSV_FLOAT_FIELDS = frozenset(
    {
        "inference_latency_ms",
        "policy_inference_latency_ms",
        "server_inference_latency_ms",
        "input_max_abs",
        "adapter_max_residual_norm",
        "adapter_min_singular_value",
        "adapter_latency_ms",
        "max_raw_acceleration_rad_s2",
        "max_guarded_acceleration_rad_s2",
        "guard_latency_ms",
        "raw_hand_displacement_m",
        "guarded_hand_displacement_m",
    }
)
_CSV_BOOLEAN_FIELDS = frozenset(
    {
        "raw_path_safe",
        "deadline_exceeded",
        "fallback_latched",
        "safe_after_guard",
        "guarded_path_safe",
    }
)


class Pi05ArchiveReplayError(ValueError):
    """Raised when source or derived replay evidence violates its contract."""


@dataclass(frozen=True)
class ValidatedPi05Archive:
    root: Path
    root_manifest: Mapping[str, Any]
    root_manifest_sha256: str
    descriptor: Mapping[str, Any]
    environment: Mapping[str, Any]
    queries: tuple[Mapping[str, Any], ...]
    transition_queries: tuple[Mapping[str, Any], ...]
    arrays: Mapping[str, np.ndarray]
    response_hashes: tuple[str, ...]

    @property
    def transition_count(self) -> int:
        return len(self.transition_queries)


@dataclass(frozen=True)
class ArchiveReplayConfig:
    chunk_count: int = 90
    selection_seed: int = 20260807
    scenarios: tuple[str, ...] = (
        "free_space",
        "single_block",
        "narrow_gate",
    )
    deadline_ms: float = 200.0
    collision_resolution_rad: float = 0.02

    def __post_init__(self) -> None:
        if type(self.chunk_count) is not int or self.chunk_count <= 0:
            raise ValueError("chunk_count must be a positive integer")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("selection_seed must be a nonnegative integer")
        known = mujoco_scenarios()
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or any(name not in known for name in self.scenarios)
        ):
            raise ValueError("scenarios must be unique known MuJoCo scenarios")
        if not np.isfinite(self.deadline_ms) or self.deadline_ms < 0.0:
            raise ValueError("deadline_ms must be finite and nonnegative")
        if (
            not np.isfinite(self.collision_resolution_rad)
            or self.collision_resolution_rad <= 0.0
        ):
            raise ValueError(
                "collision_resolution_rad must be finite and positive"
            )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pi05ArchiveReplayError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                Pi05ArchiveReplayError(f"nonfinite JSON token: {token}")
            ),
        )
    except Pi05ArchiveReplayError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Pi05ArchiveReplayError(f"cannot read JSON: {path}") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_action_sha256(actions: object) -> str:
    canonical = np.asarray(actions, dtype="<f4", order="C")
    _require(
        canonical.ndim == 2 and bool(np.all(np.isfinite(canonical))),
        "actions must be a finite two-dimensional array",
    )
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_manifest_hash(files: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _validate_root_manifest(root: Path) -> Mapping[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    _require(isinstance(manifest, Mapping), "root manifest must be an object")
    _require(
        manifest.get("schema_version") == ROOT_MANIFEST_SCHEMA,
        "root manifest schema mismatch",
    )
    _require(
        set(manifest) == {"schema_version", "files", "files_sha256"},
        "root manifest fields do not match the schema",
    )
    files = manifest.get("files")
    _require(isinstance(files, list), "root manifest file inventory is missing")
    recorded_paths: list[str] = []
    for item in files:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"path", "bytes", "sha256"},
            "root manifest contains an invalid file entry",
        )
        relative = item.get("path")
        _require(isinstance(relative, str) and bool(relative), "invalid file path")
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute()
            and ".." not in pure.parts
            and "\\" not in relative
            and relative != "manifest.json",
            f"unsafe manifest path: {relative}",
        )
        size = item.get("bytes")
        digest = item.get("sha256")
        _require(type(size) is int and size >= 0, f"invalid file size: {relative}")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"invalid file hash: {relative}",
        )
        recorded_paths.append(relative)

    _require(
        recorded_paths == sorted(set(recorded_paths)),
        "root manifest paths must be unique and sorted",
    )
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _require(
        recorded_paths == actual_paths,
        "root manifest inventory does not match artifact files",
    )
    for item in files:
        relative = str(item["path"])
        path = root.joinpath(*PurePosixPath(relative).parts)
        _require(
            path.is_file() and path.stat().st_size == item["bytes"],
            f"artifact file size mismatch: {relative}",
        )
        _require(
            _sha256_file(path) == item["sha256"],
            f"artifact file hash mismatch: {relative}",
        )
    _require(
        _canonical_manifest_hash(files) == manifest.get("files_sha256"),
        "root manifest aggregate hash mismatch",
    )
    return manifest


def _write_root_manifest(root: Path) -> None:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        root / "manifest.json",
        {
            "schema_version": ROOT_MANIFEST_SCHEMA,
            "files": files,
            "files_sha256": _canonical_manifest_hash(files),
        },
    )


def _finite_number(value: object, label: str, *, nonnegative: bool = False) -> float:
    _require(type(value) in {int, float}, f"{label} must be numeric")
    number = float(value)
    _require(np.isfinite(number), f"{label} must be finite")
    if nonnegative:
        _require(number >= 0.0, f"{label} must be nonnegative")
    return number


def _validate_policy_provenance(
    descriptor: Mapping[str, Any], environment: Mapping[str, Any]
) -> None:
    _require(
        descriptor.get("policy")
        == {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
        },
        "transition descriptor policy provenance mismatch",
    )
    adapter = descriptor.get("action_adapter")
    _require(isinstance(adapter, Mapping), "action adapter provenance is missing")
    _require(
        adapter.get("action_space_id") == LIBERO_ACTION_SPACE_ID
        and adapter.get("name") == PI05_ACTION_ADAPTER
        and adapter.get("source") == PI05_ACTION_ADAPTER_SOURCE
        and adapter.get("version") == 1,
        "action adapter provenance mismatch",
    )
    adapter_hash = adapter.get("source_sha256")
    _require(
        isinstance(adapter_hash, str) and len(adapter_hash) == 64,
        "action adapter source hash is missing",
    )

    attestation = environment.get("server_attestation")
    source_hashes = environment.get("source_sha256")
    _require(
        isinstance(attestation, Mapping) and isinstance(source_hashes, Mapping),
        "server or source attestation is missing",
    )
    _require(
        attestation.get("policy_loaded") is True
        and attestation.get("policy_config") == PI05_POLICY_CONFIG
        and attestation.get("checkpoint_uri") == PI05_CHECKPOINT
        and attestation.get("checkpoint_content_sha256")
        == PI05_CHECKPOINT_SHA256,
        "official policy checkpoint attestation mismatch",
    )
    _require(
        attestation.get("openpi_tracked_clean") is True
        and attestation.get("openpi_submodules_clean") is True,
        "OpenPI source attestation is not clean",
    )
    _require(
        source_hashes.get(PI05_ACTION_ADAPTER_SOURCE) == adapter_hash,
        "action adapter source attestations disagree",
    )


def _load_transition_arrays(
    path: Path, count: int, horizon: int, action_dim: int, execute: int
) -> Mapping[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            _require(
                set(loaded.files) == ARCHIVE_KEYS,
                "transition archive fields do not match the schema",
            )
            arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except Pi05ArchiveReplayError:
        raise
    except Exception as error:
        raise Pi05ArchiveReplayError("cannot read transition archive") from error

    expected_shapes = {
        "has_previous_reference": (count,),
        "previous_reference": (count, horizon, action_dim),
        "response_actions": (count, horizon, action_dim),
        "executed_window": (count, execute, action_dim),
        "action_source": (count, execute),
        "next_reference": (count, horizon, action_dim),
        "executed_length": (count,),
        "old_prefix_steps": (count,),
        "new_suffix_steps": (count,),
    }
    for key, shape in expected_shapes.items():
        _require(arrays[key].shape == shape, f"transition shape mismatch: {key}")
    for key in (
        "previous_reference",
        "response_actions",
        "executed_window",
        "next_reference",
    ):
        _require(
            arrays[key].dtype == np.dtype("<f4")
            and bool(np.all(np.isfinite(arrays[key]))),
            f"transition action array must be finite float32: {key}",
        )
    _require(
        arrays["has_previous_reference"].dtype == np.bool_,
        "has_previous_reference must be boolean",
    )
    _require(
        arrays["action_source"].dtype == np.uint8
        and bool(np.all(arrays["action_source"] <= 2)),
        "action_source must contain uint8 scheduler labels",
    )
    for key in ("executed_length", "old_prefix_steps", "new_suffix_steps"):
        _require(
            arrays[key].dtype == np.dtype("<i4"),
            f"transition count array must be int32: {key}",
        )
    return arrays


def _validate_query(query: Mapping[str, Any], *, bootstrap: bool) -> None:
    _require(query.get("bootstrap") is bootstrap, "query bootstrap flag mismatch")
    for field in ("episode_id", "pair_id", "method", "task_suite"):
        _require(
            isinstance(query.get(field), str) and bool(query[field]),
            f"query field is missing: {field}",
        )
    for field in ("task_id", "episode_index", "query_index"):
        _require(
            type(query.get(field)) is int and int(query[field]) >= 0,
            f"query integer field is invalid: {field}",
        )
    for field in (
        "inference_latency_ms",
        "policy_inference_latency_ms",
        "server_inference_latency_ms",
    ):
        _finite_number(query.get(field), field, nonnegative=True)
    for field in ("response_action_sha256", "next_reference_sha256"):
        digest = query.get(field)
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"query hash is invalid: {field}",
        )


def _validate_transition_contract(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    queries: Sequence[Mapping[str, Any]],
    bootstrap_hashes: Mapping[str, str],
) -> tuple[str, ...]:
    scheduler = descriptor["scheduler"]
    execute = int(scheduler["execute_horizon"])
    delay = int(scheduler["inference_delay"])
    action_dim = int(scheduler["action_dim"])
    previous_by_episode: dict[str, np.ndarray] = {}
    response_hashes: list[str] = []

    for index, (row, query) in enumerate(zip(descriptor["rows"], queries)):
        _require(
            isinstance(row, Mapping) and isinstance(query, Mapping),
            "transition rows and queries must be objects",
        )
        for field in ("episode_id", "pair_id", "method", "query_index"):
            _require(
                row.get(field) == query.get(field),
                f"transition/query identity mismatch: {field}",
            )
        episode_id = str(row["episode_id"])
        _require(
            bool(arrays["has_previous_reference"][index]),
            "nonbootstrap transition lacks its reference chain",
        )
        previous = arrays["previous_reference"][index]
        response = arrays["response_actions"][index]
        executed = arrays["executed_window"][index]
        source = arrays["action_source"][index]
        next_reference = arrays["next_reference"][index]
        length = int(arrays["executed_length"][index])
        old_count = int(arrays["old_prefix_steps"][index])
        new_count = int(arrays["new_suffix_steps"][index])
        _require(
            (length, old_count, new_count)
            == (
                int(query.get("executed_steps", -1)),
                int(query.get("old_prefix_steps", -1)),
                int(query.get("new_suffix_steps", -1)),
            ),
            "transition execution counts mismatch query",
        )
        _require(
            length == old_count + new_count and 0 <= length <= execute,
            "transition execution counts are inconsistent",
        )
        if episode_id not in previous_by_episode:
            _require(
                _canonical_action_sha256(previous)
                == bootstrap_hashes.get(episode_id),
                "initial transition does not match its bootstrap response",
            )
        else:
            _require(
                np.array_equal(previous, previous_by_episode[episode_id]),
                "cross-query reference chain mismatch",
            )

        scheduled = np.concatenate(
            (previous[:delay], response[delay:execute]), axis=0
        )
        expected_source = np.concatenate(
            (
                np.full(old_count, 1, dtype=np.uint8),
                np.full(new_count, 2, dtype=np.uint8),
            )
        )
        _require(
            np.array_equal(executed[:length], scheduled[:length])
            and not np.any(executed[length:]),
            "executed window does not match overlap scheduling",
        )
        _require(
            np.array_equal(source[:length], expected_source)
            and not np.any(source[length:]),
            "action source labels do not match execution",
        )
        expected_next = np.concatenate(
            (
                response[execute:],
                np.zeros((execute, action_dim), dtype="<f4"),
            ),
            axis=0,
        )
        _require(
            np.array_equal(next_reference, expected_next),
            "next reference does not match response shift",
        )
        response_hash = _canonical_action_sha256(response)
        _require(
            response_hash == query.get("response_action_sha256"),
            "response action hash mismatch",
        )
        _require(
            _canonical_action_sha256(next_reference)
            == query.get("next_reference_sha256"),
            "next reference hash mismatch",
        )
        response_hashes.append(response_hash)
        previous_by_episode[episode_id] = next_reference
    return tuple(response_hashes)


def validate_pi05_source_archive(directory: Path) -> ValidatedPi05Archive:
    """Validate every source byte and every archived transition equation."""

    root = directory.resolve()
    _require(root.is_dir(), f"source artifact directory not found: {root}")
    root_manifest = _validate_root_manifest(root)
    descriptor = _load_json(root / "transition_descriptor.json")
    environment = _load_json(root / "environment.json")
    queries_value = _load_json(root / "queries.json")
    _require(isinstance(descriptor, Mapping), "transition descriptor is invalid")
    _require(isinstance(environment, Mapping), "environment attestation is invalid")
    _require(isinstance(queries_value, list), "queries.json must contain an array")
    _require(
        descriptor.get("schema_version") == TRANSITION_DESCRIPTOR_SCHEMA,
        "transition descriptor schema mismatch",
    )
    _validate_policy_provenance(descriptor, environment)

    scheduler = descriptor.get("scheduler")
    rows = descriptor.get("rows")
    _require(
        isinstance(scheduler, Mapping) and isinstance(rows, list) and bool(rows),
        "transition descriptor sections are missing",
    )
    _require(
        scheduler
        == {
            "action_dim": 7,
            "action_horizon": 10,
            "execute_horizon": 5,
            "inference_delay": 4,
        },
        "transition scheduler contract mismatch",
    )
    archive = descriptor.get("archive")
    _require(isinstance(archive, Mapping), "transition archive descriptor missing")
    _require(
        set(archive) == {"path", "bytes", "sha256"}
        and archive.get("path") == "transitions.npz",
        "transition archive descriptor fields mismatch",
    )
    archive_path = root / "transitions.npz"
    _require(
        archive_path.stat().st_size == archive.get("bytes")
        and _sha256_file(archive_path) == archive.get("sha256"),
        "transition archive bytes do not match its descriptor",
    )

    queries = tuple(queries_value)
    bootstrap_hashes: dict[str, str] = {}
    transition_queries: list[Mapping[str, Any]] = []
    for value in queries:
        _require(isinstance(value, Mapping), "query rows must be objects")
        bootstrap = value.get("bootstrap") is True
        _validate_query(value, bootstrap=bootstrap)
        if bootstrap:
            episode_id = str(value["episode_id"])
            _require(
                episode_id not in bootstrap_hashes,
                "episode contains duplicate bootstrap queries",
            )
            bootstrap_hashes[episode_id] = str(value["response_action_sha256"])
        else:
            transition_queries.append(value)
    _require(
        len(transition_queries) == len(rows),
        "nonbootstrap query count does not match descriptor rows",
    )
    _require(
        {str(query["episode_id"]) for query in transition_queries}
        == set(bootstrap_hashes),
        "bootstrap coverage does not match transition episodes",
    )

    arrays = _load_transition_arrays(
        archive_path,
        len(rows),
        int(scheduler["action_horizon"]),
        int(scheduler["action_dim"]),
        int(scheduler["execute_horizon"]),
    )
    response_hashes = _validate_transition_contract(
        descriptor,
        arrays,
        transition_queries,
        bootstrap_hashes,
    )
    return ValidatedPi05Archive(
        root=root,
        root_manifest=root_manifest,
        root_manifest_sha256=_sha256_file(root / "manifest.json"),
        descriptor=descriptor,
        environment=environment,
        queries=queries,
        transition_queries=tuple(transition_queries),
        arrays=arrays,
        response_hashes=response_hashes,
    )


def select_stratified_chunks(
    archive: ValidatedPi05Archive,
    chunk_count: int,
    selection_seed: int,
) -> tuple[int, ...]:
    """Select an equal deterministic sample from each task/method stratum."""

    _require(type(chunk_count) is int and chunk_count > 0, "invalid chunk count")
    _require(
        type(selection_seed) is int and selection_seed >= 0,
        "invalid selection seed",
    )
    groups: dict[tuple[int, str], list[int]] = {}
    for index, query in enumerate(archive.transition_queries):
        key = (int(query["task_id"]), str(query["method"]))
        groups.setdefault(key, []).append(index)
    strata = sorted(groups)
    _require(bool(strata), "source archive contains no task/method strata")
    _require(
        chunk_count % len(strata) == 0,
        f"chunk_count must be divisible by the {len(strata)} source strata",
    )
    per_stratum = chunk_count // len(strata)
    _require(per_stratum > 0, "chunk_count must cover every source stratum")

    selected: list[int] = []
    for task_id, method in strata:
        candidates = groups[(task_id, method)]

        def rank(index: int) -> tuple[bytes, int]:
            query = archive.transition_queries[index]
            identity = "\0".join(
                (
                    str(selection_seed),
                    str(task_id),
                    method,
                    str(query["episode_id"]),
                    str(query["query_index"]),
                    str(index),
                )
            ).encode("utf-8")
            return hashlib.sha256(identity).digest(), index

        ranked = sorted(candidates, key=rank)
        _require(
            len(ranked) >= per_stratum,
            f"stratum {(task_id, method)} has fewer than {per_stratum} rows",
        )
        selected.extend(ranked[:per_stratum])
    return tuple(
        sorted(
            selected,
            key=lambda index: (
                int(archive.transition_queries[index]["task_id"]),
                str(archive.transition_queries[index]["method"]),
                index,
            ),
        )
    )


def _case_record(
    archive: ValidatedPi05Archive,
    source_index: int,
    selection_index: int,
    scenario_name: str,
    robot: MuJoCoPanda,
    config: ArchiveReplayConfig,
) -> dict[str, Any]:
    query = archive.transition_queries[source_index]
    source_actions = archive.arrays["response_actions"][source_index]
    scenario = mujoco_scenarios()[scenario_name]
    checker = MuJoCoCollisionChecker(
        robot, resolution=config.collision_resolution_rad
    )
    adapter_config = CartesianAdapterConfig()
    adapter = PandaCartesianActionAdapter(robot, adapter_config)
    captured_at_s = 1000.0
    inference_latency_ms = float(query["inference_latency_ms"])
    adapted = adapter.adapt(
        source_actions,
        scenario.start,
        source=(
            f"official_pi05_frozen_response:{archive.response_hashes[source_index]}"
        ),
        observation_sequence_id=source_index,
        inference_latency_ms=inference_latency_ms,
        received_at_s=captured_at_s + inference_latency_ms / 1000.0,
    )
    raw_path_safe = checker.path_is_valid(adapted.predicted_positions)
    image = np.zeros(DROID_IMAGE_SHAPE, dtype=np.uint8)
    observation = VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=scenario.start,
        gripper_position=np.array([1.0]),
        prompt="offline frozen pi0.5 response replay",
        sequence_id=source_index,
        captured_at_s=captured_at_s,
    )
    guard = ActionChunkGuard(
        checker,
        GuardConfig(
            control_dt_s=adapter_config.control_dt_s,
            deadline_ms=config.deadline_ms,
            joint_velocity_clip_rad_s=float(np.max(robot.velocity_limits)),
        ),
    )
    guarded = guard.guard(
        scenario.start,
        1.0,
        observation,
        adapted.chunk,
    )
    guarded_path_safe = checker.path_is_valid(guarded.predicted_positions)
    start_hand = robot.hand_position(scenario.start)
    raw_end_hand = robot.hand_position(adapted.predicted_positions[-1])
    guarded_end_hand = robot.hand_position(guarded.predicted_positions[-1])
    adapter_metrics = adapted.metrics()
    guard_metrics = guarded.metrics()
    case_id = f"row_{source_index:05d}__{scenario_name}"
    return {
        "case_id": case_id,
        "selection_index": selection_index,
        "source_row_index": source_index,
        "task_suite": str(query["task_suite"]),
        "task_id": int(query["task_id"]),
        "method": str(query["method"]),
        "pair_id": str(query["pair_id"]),
        "episode_id": str(query["episode_id"]),
        "episode_index": int(query["episode_index"]),
        "query_index": int(query["query_index"]),
        "scenario": scenario_name,
        "response_action_sha256": archive.response_hashes[source_index],
        "horizon": int(source_actions.shape[0]),
        "inference_latency_ms": inference_latency_ms,
        "policy_inference_latency_ms": float(query["policy_inference_latency_ms"]),
        "server_inference_latency_ms": float(query["server_inference_latency_ms"]),
        "input_max_abs": float(np.max(np.abs(source_actions))),
        "input_clipped_steps": int(adapter_metrics["clipped_input_steps"]),
        "velocity_limited_steps": int(adapter_metrics["velocity_limited_steps"]),
        "adapter_max_residual_norm": float(adapter_metrics["max_residual_norm"]),
        "adapter_min_singular_value": float(
            adapter_metrics["minimum_singular_value"]
        ),
        "adapter_latency_ms": float(adapter_metrics["adapter_latency_ms"]),
        "raw_path_safe": bool(raw_path_safe),
        "deadline_exceeded": bool(guard_metrics["deadline_exceeded"]),
        "fallback_latched": bool(guard_metrics["fallback_latched"]),
        "fallback_reason": guard_metrics["fallback_reason"],
        "unsafe_raw_steps": int(guard_metrics["unsafe_raw_steps"]),
        "intervention_steps": int(guard_metrics["intervention_steps"]),
        "hold_steps": int(guard_metrics["hold_steps"]),
        "slew_limited_steps": int(guard_metrics["slew_limited_steps"]),
        "acceleration_override_steps": int(
            guard_metrics["acceleration_override_steps"]
        ),
        "max_raw_acceleration_rad_s2": float(
            guard_metrics["max_raw_acceleration_rad_s2"]
        ),
        "max_guarded_acceleration_rad_s2": float(
            guard_metrics["max_guarded_acceleration_rad_s2"]
        ),
        "guard_latency_ms": float(guard_metrics["guard_latency_ms"]),
        "safe_after_guard": bool(guard_metrics["safe_after_guard"]),
        "guarded_path_safe": bool(guarded_path_safe),
        "raw_hand_displacement_m": float(np.linalg.norm(raw_end_hand - start_hand)),
        "guarded_hand_displacement_m": float(
            np.linalg.norm(guarded_end_hand - start_hand)
        ),
    }


def _p95(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    _require(count > 0, "cannot summarize an empty replay")
    return {
        "cases": count,
        "unique_chunks": len({int(row["source_row_index"]) for row in rows}),
        "deadline_exceeded_cases": sum(bool(row["deadline_exceeded"]) for row in rows),
        "deadline_exceeded_rate": fmean(
            bool(row["deadline_exceeded"]) for row in rows
        ),
        "input_clipped_cases": sum(int(row["input_clipped_steps"]) > 0 for row in rows),
        "input_clipped_step_rate": sum(
            int(row["input_clipped_steps"]) for row in rows
        )
        / sum(int(row["horizon"]) for row in rows),
        "raw_path_invalid_cases": sum(not bool(row["raw_path_safe"]) for row in rows),
        "intervention_cases": sum(int(row["intervention_steps"]) > 0 for row in rows),
        "intervention_step_rate": sum(
            int(row["intervention_steps"]) for row in rows
        )
        / sum(int(row["horizon"]) for row in rows),
        "acceleration_override_cases": sum(
            int(row["acceleration_override_steps"]) > 0 for row in rows
        ),
        "hold_step_rate": sum(int(row["hold_steps"]) for row in rows)
        / sum(int(row["horizon"]) for row in rows),
        "guard_safe_cases": sum(bool(row["safe_after_guard"]) for row in rows),
        "guarded_path_valid_cases": sum(
            bool(row["guarded_path_safe"]) for row in rows
        ),
        "mean_adapter_latency_ms": fmean(
            float(row["adapter_latency_ms"]) for row in rows
        ),
        "p95_adapter_latency_ms": _p95(
            [float(row["adapter_latency_ms"]) for row in rows]
        ),
        "mean_guard_latency_ms": fmean(
            float(row["guard_latency_ms"]) for row in rows
        ),
        "p95_guard_latency_ms": _p95(
            [float(row["guard_latency_ms"]) for row in rows]
        ),
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_transition_count: int,
    source_hashes_verified: int,
    selection_seed: int,
) -> dict[str, Any]:
    selected_by_stratum: dict[str, int] = {}
    unique_rows: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        source_index = int(row["source_row_index"])
        unique_rows.setdefault(source_index, row)
    for row in unique_rows.values():
        key = f"task_{int(row['task_id']):02d}/{row['method']}"
        selected_by_stratum[key] = selected_by_stratum.get(key, 0) + 1
    source_latencies = [
        float(row["inference_latency_ms"]) for row in unique_rows.values()
    ]
    methods = sorted({str(row["method"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    return {
        "schema_version": REPLAY_SUMMARY_SCHEMA,
        "scope": REPLAY_SCOPE,
        "source_policy_checkpoint_attested": True,
        "policy_checkpoint_executed_in_replay": False,
        "task_success_evaluated": False,
        "panda_closed_loop_executed": False,
        "source_validation": {
            "transition_count": source_transition_count,
            "response_action_hashes_verified": source_hashes_verified,
        },
        "selection": {
            "algorithm": "sha256_rank_equal_per_task_method_v1",
            "seed": selection_seed,
            "chunk_count": len(unique_rows),
            "stratum_count": len(selected_by_stratum),
            "selected_by_stratum": dict(sorted(selected_by_stratum.items())),
            "source_inference_latency_ms": {
                "mean": fmean(source_latencies),
                "p95": _p95(source_latencies),
                "max": max(source_latencies),
            },
        },
        "overall": _aggregate_rows(rows),
        "by_method": {
            method: _aggregate_rows(
                [row for row in rows if row["method"] == method]
            )
            for method in methods
        },
        "by_scenario": {
            scenario: _aggregate_rows(
                [row for row in rows if row["scenario"] == scenario]
            )
            for scenario in scenarios
        },
        "claim_boundary": [
            "Archived policy responses were replayed; the checkpoint was not executed.",
            "The Panda path was kinematic offline lookahead, not closed-loop control.",
            "No LIBERO or Panda task-success metric was evaluated.",
            "Resolution-bounded collision checks are not a safety certification.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "CSV schema mismatch")
            raw_rows = list(reader)
    except Pi05ArchiveReplayError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise Pi05ArchiveReplayError("cannot read replay CSV") from error
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for field in CSV_FIELDS:
            value = raw[field]
            if field in _CSV_INTEGER_FIELDS:
                try:
                    row[field] = int(value)
                except ValueError as error:
                    raise Pi05ArchiveReplayError(
                        f"invalid CSV integer: {field}"
                    ) from error
            elif field in _CSV_FLOAT_FIELDS:
                try:
                    row[field] = float(value)
                except ValueError as error:
                    raise Pi05ArchiveReplayError(
                        f"invalid CSV float: {field}"
                    ) from error
                _require(np.isfinite(row[field]), f"nonfinite CSV float: {field}")
            elif field in _CSV_BOOLEAN_FIELDS:
                _require(value in {"True", "False"}, f"invalid CSV boolean: {field}")
                row[field] = value == "True"
            elif field == "fallback_reason":
                row[field] = value or None
            else:
                _require(bool(value), f"empty CSV string: {field}")
                row[field] = value
        rows.append(row)
    return rows


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    selection = summary["selection"]
    lines = [
        "# Frozen pi0.5 response replay on the Panda guard path",
        "",
        "This report replays hash-verified official pi0.5 LIBERO action responses",
        "through the local Cartesian adapter and runtime guard. It does not rerun",
        "the policy checkpoint or execute a closed-loop task.",
        "",
        "## Coverage",
        "",
        f"- Selected chunks: {selection['chunk_count']}",
        f"- Task/method strata: {selection['stratum_count']}",
        f"- Independent Panda cases: {overall['cases']}",
        f"- Source response hashes verified: {summary['source_validation']['response_action_hashes_verified']}",
        "",
        "## Runtime observations",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Deadline-exceeded cases | {overall['deadline_exceeded_cases']} / {overall['cases']} |",
        f"| Input-clipped cases | {overall['input_clipped_cases']} / {overall['cases']} |",
        f"| Raw path-invalid cases | {overall['raw_path_invalid_cases']} / {overall['cases']} |",
        f"| Cases with guard intervention | {overall['intervention_cases']} / {overall['cases']} |",
        f"| Acceleration-conflict cases | {overall['acceleration_override_cases']} / {overall['cases']} |",
        f"| Guard-safe cases | {overall['guard_safe_cases']} / {overall['cases']} |",
        f"| Guarded path-valid cases | {overall['guarded_path_valid_cases']} / {overall['cases']} |",
        f"| P95 adapter latency | {overall['p95_adapter_latency_ms']:.3f} ms |",
        f"| P95 guard latency | {overall['p95_guard_latency_ms']:.3f} ms |",
        "",
        "## Claim boundary",
        "",
        "- The source checkpoint and response hashes are attested, but the checkpoint was not executed in this replay.",
        "- No task-success metric or Panda closed-loop execution was evaluated.",
        "- Differential IK is not equivalent to LIBERO's torque-level OSC controller.",
        "- Resolution-bounded collision checks are not continuous-collision or physical-safety certification.",
        "",
    ]
    return "\n".join(lines)


def _provenance(
    archive: ValidatedPi05Archive,
    config: ArchiveReplayConfig,
    selected: Sequence[int],
) -> dict[str, Any]:
    source_id = (
        archive.root.parent.name
        if archive.root.name == "evaluation"
        else archive.root.name
    )
    source_scene = default_panda_scene_path()
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("cartesian_adapter.py"),
        Path(__file__).with_name("guard.py"),
    )
    return {
        "schema_version": REPLAY_PROVENANCE_SCHEMA,
        "scope": REPLAY_SCOPE,
        "source_policy_checkpoint_attested": True,
        "policy_checkpoint_executed_in_replay": False,
        "task_success_evaluated": False,
        "panda_closed_loop_executed": False,
        "source": {
            "artifact_id": source_id,
            "subdirectory": archive.root.name,
            "root_manifest_sha256": archive.root_manifest_sha256,
            "root_manifest_files_sha256": archive.root_manifest["files_sha256"],
            "transition_archive_sha256": archive.descriptor["archive"]["sha256"],
            "transition_archive_bytes": archive.descriptor["archive"]["bytes"],
            "transition_count": archive.transition_count,
            "response_action_hashes_verified": len(archive.response_hashes),
        },
        "policy": dict(archive.descriptor["policy"]),
        "source_action_adapter": dict(archive.descriptor["action_adapter"]),
        "local_adapter": {
            "controller_semantics_id": LIBERO_CONTROLLER_SEMANTICS_ID,
            "kinematic_control_point_id": PANDA_KINEMATIC_CONTROL_POINT_ID,
            "method": "damped_least_squares_differential_ik",
        },
        "runtime_guard": {
            "deadline_ms": config.deadline_ms,
            "collision_resolution_rad": config.collision_resolution_rad,
            "deadline_latch_reset_per_case": True,
            "scenario_start_reset_per_case": True,
        },
        "local_runtime": {
            "python": platform.python_version(),
            "python_implementation": sys.implementation.name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
            "panda_scene_sha256": _sha256_file(source_scene),
            "implementation_sha256": {
                f"armbench/vla/{path.name}": _sha256_file(path)
                for path in implementation_paths
            },
        },
        "selection": {
            "algorithm": "sha256_rank_equal_per_task_method_v1",
            "seed": config.selection_seed,
            "chunk_count": len(selected),
            "source_row_indices": list(selected),
            "scenarios": list(config.scenarios),
            "mujoco_scenario_version": MUJOCO_SCENARIO_VERSION,
        },
        "limitations": [
            "Frozen responses are cross-controller inputs, not Panda policy outputs.",
            "Local offline lookahead does not model closed-loop observation feedback.",
            "The local hand-body control point differs from LIBERO grip_site.",
            "Collision checking is resolution-bounded joint-space interpolation.",
        ],
    }


def execute_pi05_archive_replay(
    source_directory: Path,
    output_directory: Path,
    config: ArchiveReplayConfig = ArchiveReplayConfig(),
) -> Path:
    """Validate, sample, and replay frozen responses without model inference."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    archive = validate_pi05_source_archive(source_directory)
    selected = select_stratified_chunks(
        archive,
        config.chunk_count,
        config.selection_seed,
    )

    scenarios = mujoco_scenarios()
    robots = {
        name: MuJoCoPanda.create(obstacles=scenarios[name].obstacles)
        for name in config.scenarios
    }
    rows: list[dict[str, Any]] = []
    for selection_index, source_index in enumerate(selected):
        for scenario_name in config.scenarios:
            rows.append(
                _case_record(
                    archive,
                    source_index,
                    selection_index,
                    scenario_name,
                    robots[scenario_name],
                    config,
                )
            )

    provenance = _provenance(archive, config, selected)
    summary = _build_summary(
        rows,
        source_transition_count=archive.transition_count,
        source_hashes_verified=len(archive.response_hashes),
        selection_seed=config.selection_seed,
    )
    output.mkdir(parents=True)
    _write_json(output / "provenance.json", provenance)
    _write_csv(output / "per_chunk.csv", rows)
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_root_manifest(output)
    validate_pi05_replay_artifact(output)
    return output


def validate_pi05_replay_artifact(
    directory: Path,
    source_directory: Path | None = None,
) -> dict[str, Any]:
    """Validate a replay report, optionally binding it back to source bytes."""

    root = directory.resolve()
    _require(root.is_dir(), f"replay artifact directory not found: {root}")
    manifest = _validate_root_manifest(root)
    provenance = _load_json(root / "provenance.json")
    summary = _load_json(root / "summary.json")
    rows = _read_csv(root / "per_chunk.csv")
    _require(isinstance(provenance, Mapping), "replay provenance must be an object")
    _require(isinstance(summary, Mapping), "replay summary must be an object")
    _require(
        provenance.get("schema_version") == REPLAY_PROVENANCE_SCHEMA,
        "replay provenance schema mismatch",
    )
    _require(
        summary.get("schema_version") == REPLAY_SUMMARY_SCHEMA,
        "replay summary schema mismatch",
    )
    for document in (provenance, summary):
        _require(document.get("scope") == REPLAY_SCOPE, "replay scope mismatch")
        _require(
            document.get("source_policy_checkpoint_attested") is True,
            "source checkpoint attestation flag is missing",
        )
        for field in (
            "policy_checkpoint_executed_in_replay",
            "task_success_evaluated",
            "panda_closed_loop_executed",
        ):
            _require(document.get(field) is False, f"invalid claim flag: {field}")
    _require(
        provenance.get("policy")
        == {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
        },
        "derived report policy provenance mismatch",
    )
    source_adapter = provenance.get("source_action_adapter")
    _require(
        isinstance(source_adapter, Mapping)
        and source_adapter.get("action_space_id") == LIBERO_ACTION_SPACE_ID
        and source_adapter.get("name") == PI05_ACTION_ADAPTER
        and source_adapter.get("source") == PI05_ACTION_ADAPTER_SOURCE
        and source_adapter.get("version") == 1,
        "derived report action adapter provenance mismatch",
    )
    _require(bool(rows), "replay CSV contains no cases")
    case_ids = [str(row["case_id"]) for row in rows]
    _require(len(case_ids) == len(set(case_ids)), "replay case IDs are duplicated")
    selected = provenance.get("selection")
    source = provenance.get("source")
    _require(
        isinstance(selected, Mapping) and isinstance(source, Mapping),
        "replay provenance sections are missing",
    )
    selected_indices = selected.get("source_row_indices")
    selected_scenarios = selected.get("scenarios")
    _require(
        isinstance(selected_indices, list)
        and all(type(index) is int and index >= 0 for index in selected_indices)
        and len(selected_indices) == len(set(selected_indices))
        and len(selected_indices) == selected.get("chunk_count"),
        "replay provenance contains an invalid source selection",
    )
    _require(
        isinstance(selected_scenarios, list)
        and bool(selected_scenarios)
        and len(selected_scenarios) == len(set(selected_scenarios))
        and all(scenario in mujoco_scenarios() for scenario in selected_scenarios),
        "replay provenance contains invalid scenarios",
    )
    expected_cases = {
        (source_index, scenario)
        for source_index in selected_indices
        for scenario in selected_scenarios
    }
    actual_cases = {
        (int(row["source_row_index"]), str(row["scenario"])) for row in rows
    }
    _require(
        actual_cases == expected_cases and len(rows) == len(expected_cases),
        "CSV does not contain the complete selected chunk/scenario matrix",
    )
    selection_order = {
        source_index: selection_index
        for selection_index, source_index in enumerate(selected_indices)
    }
    for row in rows:
        source_index = int(row["source_row_index"])
        scenario = str(row["scenario"])
        _require(
            int(row["selection_index"]) == selection_order[source_index]
            and row["case_id"] == f"row_{source_index:05d}__{scenario}",
            "CSV case identity does not match the declared selection order",
        )
    expected = _build_summary(
        rows,
        source_transition_count=int(source["transition_count"]),
        source_hashes_verified=int(source["response_action_hashes_verified"]),
        selection_seed=int(selected["seed"]),
    )
    _require(summary == expected, "replay summary is not reproducible from CSV")
    markdown = (root / "summary.md").read_text(encoding="utf-8")
    _require(
        "checkpoint was not executed" in markdown
        and "No task-success metric" in markdown,
        "human-readable claim boundary is missing",
    )
    checks = [
        "manifest_inventory_sizes_and_hashes",
        "claim_boundary_flags",
        "csv_schema_and_case_identity",
        "complete_chunk_scenario_matrix",
        "summary_recomputed_from_csv",
        "selection_matches_provenance",
    ]
    if source_directory is not None:
        source_archive = validate_pi05_source_archive(source_directory)
        _require(
            source_archive.root_manifest_sha256
            == source.get("root_manifest_sha256")
            and source_archive.root_manifest["files_sha256"]
            == source.get("root_manifest_files_sha256")
            and source_archive.descriptor["archive"]["sha256"]
            == source.get("transition_archive_sha256")
            and source_archive.transition_count == source.get("transition_count")
            and len(source_archive.response_hashes)
            == source.get("response_action_hashes_verified"),
            "derived report does not match the supplied source artifact",
        )
        for row in rows:
            source_index = int(row["source_row_index"])
            _require(
                source_index < source_archive.transition_count,
                "derived report source row is outside the source archive",
            )
            query = source_archive.transition_queries[source_index]
            expected_identity = {
                "task_suite": str(query["task_suite"]),
                "task_id": int(query["task_id"]),
                "method": str(query["method"]),
                "pair_id": str(query["pair_id"]),
                "episode_id": str(query["episode_id"]),
                "episode_index": int(query["episode_index"]),
                "query_index": int(query["query_index"]),
                "response_action_sha256": source_archive.response_hashes[
                    source_index
                ],
                "inference_latency_ms": float(query["inference_latency_ms"]),
                "policy_inference_latency_ms": float(
                    query["policy_inference_latency_ms"]
                ),
                "server_inference_latency_ms": float(
                    query["server_inference_latency_ms"]
                ),
            }
            _require(
                all(row[field] == value for field, value in expected_identity.items()),
                "derived CSV row does not match the supplied source archive",
            )
        checks.append("source_archive_reverified")
    return {
        "valid": True,
        "scope": REPLAY_SCOPE,
        "chunks": int(summary["selection"]["chunk_count"]),
        "cases": int(summary["overall"]["cases"]),
        "manifest_files_sha256": manifest["files_sha256"],
        "checks": checks,
    }
