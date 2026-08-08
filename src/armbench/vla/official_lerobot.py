"""Official LeRobotDataset v3.0 round-trip for Panda runtime records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path

import numpy as np

from armbench.vla.command_watchdog import (
    PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
    PANDA_RUNTIME_ACTION_SPACE_ID,
)
from armbench.vla.lerobot_adapter import (
    LEROBOT_STYLE_FRAME_KEYS,
    LeRobotFrameAdapter,
)
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    json_equal,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    write_json,
)
from armbench.vla.types import DROID_IMAGE_SHAPE, VLAObservation


OFFICIAL_LEROBOT_VERSION = "0.4.4"
OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION = "v3.0"
OFFICIAL_LEROBOT_REPO_ID = "armbench/official-panda-roundtrip"
OFFICIAL_LEROBOT_ROBOT_TYPE = "panda_armbench_runtime"
OFFICIAL_LEROBOT_SCHEMA = "armbench.official_lerobot_roundtrip.v1"
OFFICIAL_LEROBOT_MANIFEST_SCHEMA = (
    "armbench.official_lerobot_roundtrip_manifest.v1"
)
OFFICIAL_DATASET_DIRECTORY = "dataset"
_ROOT_FILES = {"dataset", "expected.npz", "summary.json", "manifest.json"}
_SUMMARY_FIELDS = {
    "schema_version",
    "official_lerobot_version",
    "official_dataset_codebase_version",
    "repo_id",
    "robot_type",
    "dataset_directory",
    "fps",
    "frames",
    "features",
    "implementation_sha256",
    "runtime_action_contract",
    "embodiment_boundary",
    "roundtrip_fields",
    "claims",
}
_RUNTIME_CONTRACT_FIELDS = {
    "embodiment",
    "action_semantics_id",
    "action_semantics_sha256",
    "action_kind",
    "action_dim",
}
_EMBODIMENT_BOUNDARY_FIELDS = {
    "dataset_robot_type",
    "panda_joint_velocity_actions",
    "so101_joint_position_actions",
    "panda_to_so101_conversion_performed",
}
_CLAIM_FIELDS = {
    "official_lerobot_package_used",
    "official_lerobot_dataset_loader_used",
    "official_lerobot_roundtrip_validated",
    "so101_robot_connected",
    "panda_robot_connected",
    "physical_robot_connected",
    "hardware_driver_validated",
    "learned_policy_checkpoint_executed",
}
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256"}
_CUSTOM_FEATURE_KEYS = (
    "observation.images.exterior",
    "observation.images.wrist",
    "observation.state",
    "action",
)
_PANDA_STATE_NAMES = tuple(
    [f"panda_joint{index}_position_rad" for index in range(1, 8)]
    + ["gripper_position_normalized"]
)
_PANDA_ACTION_NAMES = tuple(
    [f"panda_joint{index}_velocity_rad_s" for index in range(1, 8)]
    + ["gripper_position_normalized"]
)


def _implementation_paths() -> dict[str, Path]:
    return {
        "armbench/vla/official_lerobot.py": Path(__file__),
        "armbench/vla/lerobot_adapter.py": Path(__file__).with_name(
            "lerobot_adapter.py"
        ),
        "armbench/vla/command_watchdog.py": Path(__file__).with_name(
            "command_watchdog.py"
        ),
    }


class OfficialLeRobotError(RuntimeError):
    """Raised when the pinned official loader contract cannot be established."""


def official_panda_features() -> dict[str, dict[str, object]]:
    """Return the exact official dataset feature declaration for Panda Hx8."""

    return {
        "observation.images.exterior": {
            "dtype": "image",
            "shape": DROID_IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": DROID_IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(_PANDA_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(_PANDA_ACTION_NAMES),
        },
    }


def _load_official_dataset_api() -> tuple[type, str]:
    try:
        installed = _distribution_version("lerobot")
    except PackageNotFoundError as error:
        raise OfficialLeRobotError(
            "official LeRobot is not installed in this interpreter; create "
            "the isolated pinned environment with "
            "`scripts\\setup_official_lerobot.ps1`"
        ) from error
    if installed != OFFICIAL_LEROBOT_VERSION:
        raise OfficialLeRobotError(
            "official LeRobot version mismatch: expected "
            f"{OFFICIAL_LEROBOT_VERSION}, found {installed}"
        )
    try:
        from lerobot.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            LeRobotDataset,
        )
    except Exception as error:
        raise OfficialLeRobotError(
            "official LeRobotDataset could not be imported with its runtime "
            f"dependencies: {type(error).__name__}: {error}"
        ) from error
    if CODEBASE_VERSION != OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION:
        raise OfficialLeRobotError(
            "official LeRobot dataset codebase mismatch: expected "
            f"{OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION}, found "
            f"{CODEBASE_VERSION}"
        )
    return LeRobotDataset, str(CODEBASE_VERSION)


def official_lerobot_diagnostic() -> dict[str, object]:
    """Report whether the exact official dataset API is usable."""

    try:
        dataset_class, codebase_version = _load_official_dataset_api()
    except OfficialLeRobotError as error:
        try:
            installed: str | None = _distribution_version("lerobot")
        except PackageNotFoundError:
            installed = None
        return {
            "available": False,
            "required_version": OFFICIAL_LEROBOT_VERSION,
            "installed_version": installed,
            "dataset_codebase_version": None,
            "dataset_class": None,
            "diagnostic": str(error),
        }
    return {
        "available": True,
        "required_version": OFFICIAL_LEROBOT_VERSION,
        "installed_version": OFFICIAL_LEROBOT_VERSION,
        "dataset_codebase_version": codebase_version,
        "dataset_class": f"{dataset_class.__module__}.{dataset_class.__name__}",
        "diagnostic": "pinned official LeRobotDataset API is available",
    }


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() != "manifest.json"
    ]


def _write_manifest(root: Path) -> None:
    files = _inventory(root)
    write_json(
        root / "manifest.json",
        {
            "schema_version": OFFICIAL_LEROBOT_MANIFEST_SCHEMA,
            "files": files,
            "inventory_sha256": sha256_bytes(canonical_json(files)),
        },
    )


def _validate_manifest(root: Path) -> str:
    try:
        document = strict_json_load(root / "manifest.json")
    except (OSError, TypeError, ValueError) as error:
        raise OfficialLeRobotError("official round-trip manifest is invalid") from error
    if not (
        has_exact_fields(document, _MANIFEST_FIELDS)
        and document["schema_version"] == OFFICIAL_LEROBOT_MANIFEST_SCHEMA
        and isinstance(document["files"], list)
        and is_sha256(document["inventory_sha256"])
    ):
        raise OfficialLeRobotError("official round-trip manifest schema mismatch")
    for entry in document["files"]:
        if not (
            has_exact_fields(entry, _MANIFEST_ENTRY_FIELDS)
            and isinstance(entry["path"], str)
            and type(entry["size_bytes"]) is int
            and entry["size_bytes"] >= 0
            and is_sha256(entry["sha256"])
        ):
            raise OfficialLeRobotError("official manifest entry schema mismatch")
    expected = _inventory(root)
    if not json_equal(document["files"], expected):
        raise OfficialLeRobotError("official round-trip manifest inventory mismatch")
    inventory_hash = sha256_bytes(canonical_json(expected))
    if document["inventory_sha256"] != inventory_hash:
        raise OfficialLeRobotError("official manifest aggregate hash mismatch")
    return inventory_hash


def _to_numpy(value: object) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def _loaded_uint8_image(value: object, label: str) -> np.ndarray:
    image = _to_numpy(value)
    if image.shape != (3, 224, 224) or not np.all(np.isfinite(image)):
        raise OfficialLeRobotError(f"official loader returned invalid {label}")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise OfficialLeRobotError(f"official loader returned out-of-range {label}")
    return np.rint(np.transpose(image, (1, 2, 0)) * 255.0).astype(np.uint8)


def _scalar(value: object, label: str) -> float:
    array = _to_numpy(value)
    if array.size != 1 or not np.all(np.isfinite(array)):
        raise OfficialLeRobotError(f"official loader returned invalid {label}")
    return float(array.reshape(-1)[0])


def _verify_loaded_dataset(
    dataset_class: type,
    dataset_root: Path,
    expected: Mapping[str, np.ndarray],
    *,
    fps: int,
) -> dict[str, object]:
    try:
        dataset = dataset_class(
            repo_id=OFFICIAL_LEROBOT_REPO_ID,
            root=dataset_root,
            download_videos=False,
        )
    except Exception as error:
        raise OfficialLeRobotError(
            "official LeRobotDataset failed to load the local export: "
            f"{type(error).__name__}: {error}"
        ) from error
    frame_count = len(expected["states"])
    if len(dataset) != frame_count:
        raise OfficialLeRobotError("official loader frame count mismatch")
    features = official_panda_features()
    loaded_features = {
        key: dict(dataset.features[key]) for key in _CUSTOM_FEATURE_KEYS
    }
    if not json_equal(loaded_features, features):
        raise OfficialLeRobotError("official loader feature contract mismatch")
    if getattr(dataset.meta, "robot_type", None) != OFFICIAL_LEROBOT_ROBOT_TYPE:
        raise OfficialLeRobotError("official loader robot type mismatch")

    checks = {key: 0 for key in ("images", "state", "action", "task", "timestamp")}
    for index in range(frame_count):
        item = dataset[index]
        exterior = _loaded_uint8_image(
            item["observation.images.exterior"], "exterior image"
        )
        wrist = _loaded_uint8_image(
            item["observation.images.wrist"], "wrist image"
        )
        if not np.array_equal(exterior, expected["exterior_images"][index]):
            raise OfficialLeRobotError("official exterior image round-trip mismatch")
        if not np.array_equal(wrist, expected["wrist_images"][index]):
            raise OfficialLeRobotError("official wrist image round-trip mismatch")
        checks["images"] += 2

        state = _to_numpy(item["observation.state"]).astype(np.float32)
        action = _to_numpy(item["action"]).astype(np.float32)
        if not np.array_equal(state, expected["states"][index]):
            raise OfficialLeRobotError("official state round-trip mismatch")
        if not np.array_equal(action, expected["actions"][index]):
            raise OfficialLeRobotError("official action round-trip mismatch")
        checks["state"] += 1
        checks["action"] += 1

        if item["task"] != str(expected["tasks"][index]):
            raise OfficialLeRobotError("official task round-trip mismatch")
        checks["task"] += 1
        timestamp = _scalar(item["timestamp"], "timestamp")
        expected_timestamp = float(expected["timestamps"][index])
        if timestamp != expected_timestamp or timestamp != float(
            np.float32(index / fps)
        ):
            raise OfficialLeRobotError("official timestamp round-trip mismatch")
        checks["timestamp"] += 1

        if int(_scalar(item["frame_index"], "frame index")) != index:
            raise OfficialLeRobotError("official frame index mismatch")
        if int(_scalar(item["episode_index"], "episode index")) != 0:
            raise OfficialLeRobotError("official episode index mismatch")
    return {
        "frames": frame_count,
        "field_checks": checks,
        "official_loader_class": (
            f"{dataset_class.__module__}.{dataset_class.__name__}"
        ),
    }


def _validated_frames(
    frames: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    if not frames:
        raise ValueError("at least one LeRobot frame is required")
    expected_keys = set(LEROBOT_STYLE_FRAME_KEYS)
    normalized: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, Mapping) or set(frame) != expected_keys:
            raise ValueError("official LeRobot frame keys do not match the adapter")
        exterior = np.asarray(frame["observation.images.exterior"])
        wrist = np.asarray(frame["observation.images.wrist"])
        raw_state = np.asarray(frame["observation.state"])
        raw_action = np.asarray(frame["action"])
        task = frame["task"]
        if (
            exterior.shape != DROID_IMAGE_SHAPE
            or exterior.dtype != np.uint8
            or wrist.shape != DROID_IMAGE_SHAPE
            or wrist.dtype != np.uint8
            or raw_state.dtype.kind not in {"i", "u", "f"}
            or raw_action.dtype.kind not in {"i", "u", "f"}
        ):
            raise ValueError("official LeRobot frame values are invalid")
        state = np.asarray(raw_state, dtype=np.float32)
        action = np.asarray(raw_action, dtype=np.float32)
        if (
            state.shape != (8,)
            or action.shape != (8,)
            or not np.all(np.isfinite(state))
            or not np.all(np.isfinite(action))
            or not 0.0 <= state[7] <= 1.0
            or not 0.0 <= action[7] <= 1.0
            or not isinstance(task, str)
            or not task.strip()
        ):
            raise ValueError("official LeRobot frame values are invalid")
        normalized.append(
            {
                "observation.images.exterior": exterior.copy(),
                "observation.images.wrist": wrist.copy(),
                "observation.state": state.copy(),
                "action": action.copy(),
                "task": task,
            }
        )
    expected = {
        "exterior_images": np.stack(
            [frame["observation.images.exterior"] for frame in normalized]
        ).astype(np.uint8),
        "wrist_images": np.stack(
            [frame["observation.images.wrist"] for frame in normalized]
        ).astype(np.uint8),
        "states": np.stack(
            [frame["observation.state"] for frame in normalized]
        ).astype(np.float32),
        "actions": np.stack([frame["action"] for frame in normalized]).astype(
            np.float32
        ),
        "tasks": np.asarray([frame["task"] for frame in normalized]),
    }
    return normalized, expected


def export_official_lerobot_episode(
    directory: Path,
    frames: Sequence[Mapping[str, object]],
    *,
    fps: int = 10,
) -> Path:
    """Export and reload one episode with the pinned official API."""

    if type(fps) is not int or fps <= 0:
        raise ValueError("official LeRobot fps must be a positive integer")
    root = directory.resolve()
    if root.exists():
        raise OfficialLeRobotError(f"output directory already exists: {root}")
    normalized, expected = _validated_frames(frames)
    dataset_class, codebase_version = _load_official_dataset_api()
    root.mkdir(parents=True)
    dataset_root = root / OFFICIAL_DATASET_DIRECTORY
    try:
        dataset = dataset_class.create(
            repo_id=OFFICIAL_LEROBOT_REPO_ID,
            fps=fps,
            features=official_panda_features(),
            root=dataset_root,
            robot_type=OFFICIAL_LEROBOT_ROBOT_TYPE,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        for frame in normalized:
            dataset.add_frame(dict(frame))
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
    except Exception as error:
        raise OfficialLeRobotError(
            "official LeRobotDataset export failed: "
            f"{type(error).__name__}: {error}"
        ) from error

    expected["timestamps"] = np.asarray(
        [np.float32(index / fps) for index in range(len(normalized))],
        dtype=np.float32,
    )
    np.savez_compressed(root / "expected.npz", **expected)
    verification = _verify_loaded_dataset(
        dataset_class,
        dataset_root,
        expected,
        fps=fps,
    )
    summary = {
        "schema_version": OFFICIAL_LEROBOT_SCHEMA,
        "official_lerobot_version": OFFICIAL_LEROBOT_VERSION,
        "official_dataset_codebase_version": codebase_version,
        "repo_id": OFFICIAL_LEROBOT_REPO_ID,
        "robot_type": OFFICIAL_LEROBOT_ROBOT_TYPE,
        "dataset_directory": OFFICIAL_DATASET_DIRECTORY,
        "fps": fps,
        "frames": len(normalized),
        "features": official_panda_features(),
        "implementation_sha256": {
            label: sha256_file(path)
            for label, path in _implementation_paths().items()
        },
        "runtime_action_contract": {
            "embodiment": "franka_panda_7dof",
            "action_semantics_id": PANDA_RUNTIME_ACTION_SPACE_ID,
            "action_semantics_sha256": PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
            "action_kind": "seven_joint_velocity_plus_normalized_gripper_position",
            "action_dim": 8,
        },
        "embodiment_boundary": {
            "dataset_robot_type": OFFICIAL_LEROBOT_ROBOT_TYPE,
            "panda_joint_velocity_actions": True,
            "so101_joint_position_actions": False,
            "panda_to_so101_conversion_performed": False,
        },
        "roundtrip_fields": verification["field_checks"],
        "claims": {
            "official_lerobot_package_used": True,
            "official_lerobot_dataset_loader_used": True,
            "official_lerobot_roundtrip_validated": True,
            "so101_robot_connected": False,
            "panda_robot_connected": False,
            "physical_robot_connected": False,
            "hardware_driver_validated": False,
            "learned_policy_checkpoint_executed": False,
        },
    }
    write_json(root / "summary.json", summary)
    _write_manifest(root)
    validate_official_lerobot_episode(root)
    return root


def _load_expected(root: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(root / "expected.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "exterior_images",
                "wrist_images",
                "states",
                "actions",
                "tasks",
                "timestamps",
            }:
                raise OfficialLeRobotError("official expected array keys mismatch")
            expected = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        raise OfficialLeRobotError("official expected arrays are invalid") from error
    count = len(expected["states"])
    if not (
        count > 0
        and expected["exterior_images"].shape == (count, *DROID_IMAGE_SHAPE)
        and expected["exterior_images"].dtype == np.uint8
        and expected["wrist_images"].shape == (count, *DROID_IMAGE_SHAPE)
        and expected["wrist_images"].dtype == np.uint8
        and expected["states"].shape == (count, 8)
        and expected["states"].dtype == np.float32
        and expected["actions"].shape == (count, 8)
        and expected["actions"].dtype == np.float32
        and expected["tasks"].shape == (count,)
        and expected["tasks"].dtype.kind == "U"
        and expected["timestamps"].shape == (count,)
        and expected["timestamps"].dtype == np.float32
        and np.all(np.isfinite(expected["states"]))
        and np.all(np.isfinite(expected["actions"]))
        and np.all(np.isfinite(expected["timestamps"]))
        and np.all(
            (0.0 <= expected["states"][:, 7])
            & (expected["states"][:, 7] <= 1.0)
        )
        and np.all(
            (0.0 <= expected["actions"][:, 7])
            & (expected["actions"][:, 7] <= 1.0)
        )
    ):
        raise OfficialLeRobotError("official expected array contract mismatch")
    return expected


def validate_official_lerobot_episode(directory: Path) -> dict[str, object]:
    """Reload and compare a preserved artifact with official LeRobotDataset."""

    root = directory.resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != _ROOT_FILES:
        raise OfficialLeRobotError("official round-trip root file set is invalid")
    inventory_hash = _validate_manifest(root)
    try:
        summary = strict_json_load(root / "summary.json")
    except (OSError, TypeError, ValueError) as error:
        raise OfficialLeRobotError("official round-trip summary is invalid") from error
    if not (
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == OFFICIAL_LEROBOT_SCHEMA
        and summary["official_lerobot_version"] == OFFICIAL_LEROBOT_VERSION
        and summary["official_dataset_codebase_version"]
        == OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION
        and summary["repo_id"] == OFFICIAL_LEROBOT_REPO_ID
        and summary["robot_type"] == OFFICIAL_LEROBOT_ROBOT_TYPE
        and summary["dataset_directory"] == OFFICIAL_DATASET_DIRECTORY
        and type(summary["fps"]) is int
        and summary["fps"] > 0
        and type(summary["frames"]) is int
        and summary["frames"] > 0
        and json_equal(summary["features"], official_panda_features())
    ):
        raise OfficialLeRobotError("official round-trip summary schema mismatch")
    implementation_hashes = summary["implementation_sha256"]
    implementation_paths = _implementation_paths()
    if not (
        isinstance(implementation_hashes, Mapping)
        and set(implementation_hashes) == set(implementation_paths)
        and all(is_sha256(value) for value in implementation_hashes.values())
        and all(
            implementation_hashes[label] == sha256_file(path)
            for label, path in implementation_paths.items()
        )
    ):
        raise OfficialLeRobotError("official round-trip implementation hash mismatch")
    contract = summary["runtime_action_contract"]
    boundary = summary["embodiment_boundary"]
    claims = summary["claims"]
    if not (
        has_exact_fields(contract, _RUNTIME_CONTRACT_FIELDS)
        and contract["embodiment"] == "franka_panda_7dof"
        and contract["action_semantics_id"] == PANDA_RUNTIME_ACTION_SPACE_ID
        and contract["action_semantics_sha256"]
        == PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
        and contract["action_kind"]
        == "seven_joint_velocity_plus_normalized_gripper_position"
        and contract["action_dim"] == 8
        and is_sha256(contract["action_semantics_sha256"])
    ):
        raise OfficialLeRobotError("official Panda action contract mismatch")
    if not (
        has_exact_fields(boundary, _EMBODIMENT_BOUNDARY_FIELDS)
        and boundary["dataset_robot_type"] == OFFICIAL_LEROBOT_ROBOT_TYPE
        and boundary["panda_joint_velocity_actions"] is True
        and boundary["so101_joint_position_actions"] is False
        and boundary["panda_to_so101_conversion_performed"] is False
    ):
        raise OfficialLeRobotError("Panda/SO-101 embodiment boundary mismatch")
    if not (
        has_exact_fields(claims, _CLAIM_FIELDS)
        and claims["official_lerobot_package_used"] is True
        and claims["official_lerobot_dataset_loader_used"] is True
        and claims["official_lerobot_roundtrip_validated"] is True
        and all(
            claims[field] is False
            for field in _CLAIM_FIELDS
            if field
            not in {
                "official_lerobot_package_used",
                "official_lerobot_dataset_loader_used",
                "official_lerobot_roundtrip_validated",
            }
        )
    ):
        raise OfficialLeRobotError("official round-trip claim boundary mismatch")

    dataset_class, codebase_version = _load_official_dataset_api()
    expected = _load_expected(root)
    if summary["frames"] != len(expected["states"]):
        raise OfficialLeRobotError("official summary frame count mismatch")
    verification = _verify_loaded_dataset(
        dataset_class,
        root / OFFICIAL_DATASET_DIRECTORY,
        expected,
        fps=summary["fps"],
    )
    if not json_equal(summary["roundtrip_fields"], verification["field_checks"]):
        raise OfficialLeRobotError("official round-trip field counts mismatch")
    return {
        "valid": True,
        "directory": str(root),
        "official_lerobot_version": OFFICIAL_LEROBOT_VERSION,
        "official_dataset_codebase_version": codebase_version,
        "official_loader_class": verification["official_loader_class"],
        "frames": verification["frames"],
        "field_checks": verification["field_checks"],
        "runtime_action_semantics_id": PANDA_RUNTIME_ACTION_SPACE_ID,
        "runtime_action_semantics_sha256": (
            PANDA_RUNTIME_ACTION_SEMANTICS_SHA256
        ),
        "so101_actions_used": False,
        "manifest_inventory_sha256": inventory_hash,
        "checks": [
            "pinned_official_distribution_version",
            "official_dataset_v3_loader",
            "armbench_implementation_hashes",
            "lossless_png_image_roundtrip",
            "state_action_task_timestamp_roundtrip",
            "panda_hx8_semantics_bound",
            "so101_semantics_excluded",
            "recursive_file_manifest",
        ],
    }


def _smoke_frames() -> list[dict[str, object]]:
    adapter = LeRobotFrameAdapter()
    y, x = np.indices((224, 224))
    frames: list[dict[str, object]] = []
    prompt = "move the Panda gripper through the registered joint-space path"
    for index in range(3):
        exterior = np.stack(
            [
                (x + 17 * index) % 256,
                (y + 29 * index) % 256,
                (x + y + 11 * index) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        wrist = np.roll(exterior, shift=index + 1, axis=1)
        observation = VLAObservation(
            exterior_image=exterior,
            wrist_image=wrist,
            joint_position=np.linspace(-0.3, 0.3, 7) + 0.01 * index,
            gripper_position=np.asarray([0.8 - 0.1 * index]),
            prompt=prompt,
            sequence_id=index,
            captured_at_s=index / 10.0,
        )
        action = np.zeros(8, dtype=float)
        action[:7] = 0.02 * (index + 1) * np.asarray(
            [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1]
        )
        action[7] = float(observation.gripper_position[0])
        frames.append(
            adapter.to_frame(
                observation,
                action,
                action_semantics_id=PANDA_RUNTIME_ACTION_SPACE_ID,
                action_semantics_sha256=PANDA_RUNTIME_ACTION_SEMANTICS_SHA256,
            )
        )
    return frames


def run_official_lerobot_smoke(directory: Path) -> Path:
    """Create the deterministic three-frame official loader artifact."""

    return export_official_lerobot_episode(directory, _smoke_frames(), fps=10)


__all__ = [
    "OFFICIAL_LEROBOT_DATASET_CODEBASE_VERSION",
    "OFFICIAL_LEROBOT_ROBOT_TYPE",
    "OFFICIAL_LEROBOT_VERSION",
    "OfficialLeRobotError",
    "export_official_lerobot_episode",
    "official_lerobot_diagnostic",
    "official_panda_features",
    "run_official_lerobot_smoke",
    "validate_official_lerobot_episode",
]
