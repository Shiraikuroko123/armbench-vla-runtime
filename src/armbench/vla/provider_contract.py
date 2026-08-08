"""Provider-neutral action semantics and auditable frozen-response policies.

This module keeps provider-native action chunks outside the runtime contract
until their semantics have been checked and an explicit adapter has converted
them to ArmBench's Panda/DROID ``Hx8`` action representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.cartesian_adapter import (
    LIBERO_ACTION_DIM,
    LIBERO_ACTION_SPACE_ID,
    LIBERO_CONTROLLER_SEMANTICS_ID,
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
)
from armbench.vla.serialization import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    write_json,
)
from armbench.vla.types import ActionChunk, VLAObservation


FloatArray = NDArray[np.float64]
PROVIDER_BUNDLE_SCHEMA = "armbench.frozen_action_provider.v1"
PROVIDER_MANIFEST_SCHEMA = "armbench.frozen_action_provider_manifest.v1"
PROVIDER_AUDIT_SCHEMA = "armbench.provider_contract_audit.v1"
PROVIDER_AUDIT_MANIFEST_SCHEMA = "armbench.provider_contract_audit_manifest.v1"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _json_string(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a JSON string")
    return value


def _json_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _json_string(value)


def _json_bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("expected a JSON boolean")
    return value


def _json_int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected a JSON integer")
    return value


def _json_number(value: object) -> float:
    if type(value) not in {int, float}:
        raise TypeError("expected a JSON number")
    return float(value)


def _json_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON string array")
    return tuple(_json_string(item) for item in value)


def _json_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON integer array")
    return tuple(_json_int(item) for item in value)


class ProviderContractError(ValueError):
    """Raised when provider identity, semantics, or frozen bytes are invalid."""


class SemanticCompatibilityError(ProviderContractError):
    """Raised before a provider with incompatible actions reaches the runtime."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderContractError(message)


def canonical_action_sha256(actions: ArrayLike) -> str:
    """Hash action values in one dtype, byte order, and memory order."""

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ProviderContractError("actions must be a nonempty matrix")
    try:
        canonical = np.asarray(values, dtype="<f4", order="C")
    except (TypeError, ValueError) as error:
        raise ProviderContractError("actions cannot be converted to float32") from error
    if not np.all(np.isfinite(canonical)):
        raise ProviderContractError("actions must be finite")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def canonical_observation_sha256(observation: VLAObservation) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(observation.exterior_image, dtype=np.uint8).tobytes())
    digest.update(np.asarray(observation.wrist_image, dtype=np.uint8).tobytes())
    digest.update(
        np.asarray(observation.state, dtype="<f8", order="C").tobytes(order="C")
    )
    digest.update(observation.prompt.encode("utf-8"))
    digest.update(int(observation.sequence_id).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionSemantics:
    """Fields that must agree before numerically similar actions are adapted."""

    action_space_id: str
    action_dim: int
    action_order: tuple[str, ...]
    control_period_s: float
    coordinate_frame: str
    normalized_min: float
    normalized_max: float
    translation_delta_scale_m: float
    rotation_representation: str
    rotation_delta_scale_rad: float
    gripper_convention: str
    controller_semantics_id: str

    def __post_init__(self) -> None:
        numeric = np.asarray(
            [
                self.control_period_s,
                self.normalized_min,
                self.normalized_max,
                self.translation_delta_scale_m,
                self.rotation_delta_scale_rad,
            ],
            dtype=float,
        )
        strings = (
            self.action_space_id,
            self.coordinate_frame,
            self.rotation_representation,
            self.gripper_convention,
            self.controller_semantics_id,
        )
        if (
            type(self.action_dim) is not int
            or self.action_dim <= 0
            or len(self.action_order) != self.action_dim
            or len(set(self.action_order)) != self.action_dim
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.action_order
            )
            or any(not isinstance(item, str) or not item.strip() for item in strings)
            or not np.all(np.isfinite(numeric))
            or self.control_period_s <= 0.0
            or self.normalized_min >= self.normalized_max
            or self.translation_delta_scale_m <= 0.0
            or self.rotation_delta_scale_rad <= 0.0
        ):
            raise ProviderContractError("action semantics are invalid")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["action_order"] = list(self.action_order)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ActionSemantics":
        try:
            _require(
                set(value)
                == {
                    "action_space_id",
                    "action_dim",
                    "action_order",
                    "control_period_s",
                    "coordinate_frame",
                    "normalized_min",
                    "normalized_max",
                    "translation_delta_scale_m",
                    "rotation_representation",
                    "rotation_delta_scale_rad",
                    "gripper_convention",
                    "controller_semantics_id",
                },
                "action semantics document fields are invalid",
            )
            return cls(
                action_space_id=_json_string(value["action_space_id"]),
                action_dim=_json_int(value["action_dim"]),
                action_order=_json_string_tuple(value["action_order"]),
                control_period_s=_json_number(value["control_period_s"]),
                coordinate_frame=_json_string(value["coordinate_frame"]),
                normalized_min=_json_number(value["normalized_min"]),
                normalized_max=_json_number(value["normalized_max"]),
                translation_delta_scale_m=_json_number(
                    value["translation_delta_scale_m"]
                ),
                rotation_representation=_json_string(value["rotation_representation"]),
                rotation_delta_scale_rad=_json_number(
                    value["rotation_delta_scale_rad"]
                ),
                gripper_convention=_json_string(value["gripper_convention"]),
                controller_semantics_id=_json_string(value["controller_semantics_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderContractError(
                "action semantics document is invalid"
            ) from error

    @property
    def semantic_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()))


def libero_cartesian_semantics(
    config: CartesianAdapterConfig = CartesianAdapterConfig(),
) -> ActionSemantics:
    return ActionSemantics(
        action_space_id=LIBERO_ACTION_SPACE_ID,
        action_dim=LIBERO_ACTION_DIM,
        action_order=("dx", "dy", "dz", "dax", "day", "daz", "gripper"),
        control_period_s=config.control_dt_s,
        coordinate_frame=config.action_frame,
        normalized_min=-config.normalized_action_limit,
        normalized_max=config.normalized_action_limit,
        translation_delta_scale_m=config.translation_delta_scale_m,
        rotation_representation="axis_angle_delta",
        rotation_delta_scale_rad=config.rotation_delta_scale_rad,
        gripper_convention="minus_one_open_plus_one_closed",
        controller_semantics_id=LIBERO_CONTROLLER_SEMANTICS_ID,
    )


def require_semantic_compatibility(
    actual: ActionSemantics,
    expected: ActionSemantics,
) -> None:
    mismatches = [
        key
        for key in expected.to_dict()
        if actual.to_dict()[key] != expected.to_dict()[key]
    ]
    if mismatches:
        raise SemanticCompatibilityError(
            "provider action semantics are incompatible: " + ", ".join(mismatches)
        )


@dataclass(frozen=True)
class ProviderIdentity:
    provider_id: str
    model_family: str
    implementation_repository: str
    implementation_revision: str
    checkpoint_reference: str | None
    checkpoint_sha256: str | None
    checkpoint_identity_status: str
    response_origin: str
    checkpoint_executed_during_capture: bool
    checkpoint_executed_this_run: bool = False

    def __post_init__(self) -> None:
        if (
            not self.provider_id.strip()
            or not self.model_family.strip()
            or not self.implementation_repository.strip()
            or not _HEX_40.fullmatch(self.implementation_revision)
            or self.checkpoint_identity_status
            not in {"content_attested", "declared_unverified", "not_applicable"}
            or self.response_origin
            not in {
                "checkpoint_capture",
                "attested_archive",
                "synthetic_contract_fixture",
                "live_checkpoint_inference",
            }
        ):
            raise ProviderContractError("provider identity is invalid")
        if self.checkpoint_sha256 is not None and not _HEX_64.fullmatch(
            self.checkpoint_sha256
        ):
            raise ProviderContractError("checkpoint SHA-256 is invalid")
        if self.checkpoint_identity_status == "content_attested":
            if self.checkpoint_sha256 is None or self.checkpoint_reference is None:
                raise ProviderContractError(
                    "content-attested checkpoints require reference and SHA-256"
                )
        elif self.checkpoint_sha256 is not None:
            raise ProviderContractError(
                "unattested checkpoint identities cannot carry a content hash"
            )
        if self.response_origin == "synthetic_contract_fixture" and (
            self.checkpoint_executed_during_capture
            or self.checkpoint_identity_status == "content_attested"
        ):
            raise ProviderContractError(
                "synthetic fixtures cannot claim checkpoint execution or attestation"
            )
        if self.response_origin == "live_checkpoint_inference":
            if (
                not self.checkpoint_executed_this_run
                or self.checkpoint_executed_during_capture
                or self.checkpoint_identity_status != "content_attested"
            ):
                raise ProviderContractError(
                    "live providers require current-run execution and checkpoint "
                    "content attestation"
                )
        elif self.checkpoint_executed_this_run:
            raise ProviderContractError(
                "non-live providers cannot claim current-run checkpoint execution"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderIdentity":
        try:
            _require(
                set(value)
                == {
                    "provider_id",
                    "model_family",
                    "implementation_repository",
                    "implementation_revision",
                    "checkpoint_reference",
                    "checkpoint_sha256",
                    "checkpoint_identity_status",
                    "response_origin",
                    "checkpoint_executed_during_capture",
                    "checkpoint_executed_this_run",
                },
                "provider identity document fields are invalid",
            )
            return cls(
                provider_id=_json_string(value["provider_id"]),
                model_family=_json_string(value["model_family"]),
                implementation_repository=_json_string(
                    value["implementation_repository"]
                ),
                implementation_revision=_json_string(value["implementation_revision"]),
                checkpoint_reference=_json_optional_string(
                    value["checkpoint_reference"]
                ),
                checkpoint_sha256=_json_optional_string(value["checkpoint_sha256"]),
                checkpoint_identity_status=_json_string(
                    value["checkpoint_identity_status"]
                ),
                response_origin=_json_string(value["response_origin"]),
                checkpoint_executed_during_capture=_json_bool(
                    value["checkpoint_executed_during_capture"]
                ),
                checkpoint_executed_this_run=_json_bool(
                    value["checkpoint_executed_this_run"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderContractError(
                "provider identity document is invalid"
            ) from error


@dataclass(frozen=True)
class RawActionChunk:
    actions: FloatArray
    semantics: ActionSemantics
    source: str
    observation_sequence_id: int
    inference_latency_ms: float
    received_at_s: float
    response_sha256: str

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=float)
        if (
            actions.ndim != 2
            or actions.shape[0] == 0
            or actions.shape[1] != self.semantics.action_dim
            or not np.all(np.isfinite(actions))
            or not self.source.strip()
            or self.observation_sequence_id < 0
            or not np.isfinite(self.inference_latency_ms)
            or self.inference_latency_ms < 0.0
            or not np.isfinite(self.received_at_s)
            or not _HEX_64.fullmatch(self.response_sha256)
            or canonical_action_sha256(actions) != self.response_sha256
        ):
            raise ProviderContractError("raw action chunk is invalid")
        copied = actions.copy()
        copied.flags.writeable = False
        object.__setattr__(self, "actions", copied)


class RawActionChunkProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    @property
    def semantics(self) -> ActionSemantics: ...

    def infer_raw(self, observation: VLAObservation) -> RawActionChunk: ...


@dataclass(frozen=True)
class FrozenResponseRecord:
    observation_sequence_id: int
    actions: FloatArray
    inference_latency_ms: float
    observation_sha256: str | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=float)
        if (
            type(self.observation_sequence_id) is not int
            or self.observation_sequence_id < 0
            or actions.ndim != 2
            or actions.shape[0] == 0
            or not np.all(np.isfinite(actions))
            or not np.isfinite(self.inference_latency_ms)
            or self.inference_latency_ms < 0.0
            or (
                self.observation_sha256 is not None
                and not _HEX_64.fullmatch(self.observation_sha256)
            )
        ):
            raise ProviderContractError("frozen response record is invalid")
        copied = actions.copy()
        copied.flags.writeable = False
        object.__setattr__(self, "actions", copied)


def _inventory(
    root: Path, *, exclude_root_manifest: bool = False
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if exclude_root_manifest and relative == "manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _write_manifest(root: Path, *, schema: str, recursive: bool) -> None:
    files = (
        _inventory(root, exclude_root_manifest=True)
        if recursive
        else [
            {
                "path": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in ("provider.json", "responses.npz")
        ]
    )
    manifest = {
        "schema_version": schema,
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json(files)),
    }
    write_json(root / "manifest.json", manifest)


def _validate_manifest(root: Path, *, schema: str, recursive: bool) -> None:
    try:
        manifest = strict_json_load(root / "manifest.json")
    except (OSError, TypeError, ValueError) as error:
        raise ProviderContractError(
            "provider manifest is missing or invalid"
        ) from error
    _require(
        isinstance(manifest, Mapping) and manifest.get("schema_version") == schema,
        "provider manifest schema mismatch",
    )
    _require(
        set(manifest) == {"schema_version", "files", "inventory_sha256"},
        "provider manifest fields are invalid",
    )
    files = manifest.get("files")
    _require(isinstance(files, list), "provider manifest file list is invalid")
    expected = (
        _inventory(root, exclude_root_manifest=True)
        if recursive
        else [
            {
                "path": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in ("provider.json", "responses.npz")
            if (root / name).is_file()
        ]
    )
    _require(files == expected, "provider manifest file inventory mismatch")
    _require(
        manifest.get("inventory_sha256")
        == sha256_bytes(canonical_json(expected)),
        "provider manifest inventory hash mismatch",
    )


def write_frozen_provider_bundle(
    directory: Path,
    *,
    identity: ProviderIdentity,
    semantics: ActionSemantics,
    responses: Sequence[FrozenResponseRecord],
) -> Path:
    root = directory.resolve()
    _require(not root.exists(), f"provider directory already exists: {root}")
    _require(bool(responses), "at least one frozen response is required")
    sequence_ids = [record.observation_sequence_id for record in responses]
    _require(
        len(sequence_ids) == len(set(sequence_ids)),
        "frozen response sequence IDs must be unique",
    )
    _require(
        all(record.actions.shape[1] == semantics.action_dim for record in responses),
        "frozen response width does not match action semantics",
    )
    root.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    response_rows: list[dict[str, object]] = []
    for index, record in enumerate(responses):
        key = f"response_{index:04d}"
        arrays[key] = np.asarray(record.actions, dtype="<f4", order="C")
        response_rows.append(
            {
                "array_key": key,
                "observation_sequence_id": record.observation_sequence_id,
                "observation_sha256": record.observation_sha256,
                "inference_latency_ms": record.inference_latency_ms,
                "shape": list(record.actions.shape),
                "response_sha256": canonical_action_sha256(record.actions),
            }
        )
    np.savez_compressed(root / "responses.npz", **arrays)
    descriptor = {
        "schema_version": PROVIDER_BUNDLE_SCHEMA,
        "scope": "frozen_provider_contract_without_checkpoint_inference",
        "identity": identity.to_dict(),
        "semantics": semantics.to_dict(),
        "semantics_sha256": semantics.semantic_sha256,
        "responses": response_rows,
    }
    write_json(root / "provider.json", descriptor)
    _write_manifest(root, schema=PROVIDER_MANIFEST_SCHEMA, recursive=False)
    validate_frozen_provider_bundle(root)
    return root


def validate_frozen_provider_bundle(directory: Path) -> dict[str, object]:
    root = directory.resolve()
    _require(root.is_dir(), f"provider directory not found: {root}")
    _validate_manifest(root, schema=PROVIDER_MANIFEST_SCHEMA, recursive=False)
    _require(
        {path.name for path in root.iterdir()}
        == {"provider.json", "responses.npz", "manifest.json"},
        "provider directory contains undeclared files",
    )
    try:
        descriptor = strict_json_load(root / "provider.json")
    except (OSError, TypeError, ValueError) as error:
        raise ProviderContractError("provider descriptor is invalid") from error
    _require(
        isinstance(descriptor, Mapping)
        and descriptor.get("schema_version") == PROVIDER_BUNDLE_SCHEMA
        and descriptor.get("scope")
        == "frozen_provider_contract_without_checkpoint_inference",
        "provider descriptor schema/scope mismatch",
    )
    _require(
        set(descriptor)
        == {
            "schema_version",
            "scope",
            "identity",
            "semantics",
            "semantics_sha256",
            "responses",
        },
        "provider descriptor fields are invalid",
    )
    identity_raw = descriptor.get("identity")
    semantics_raw = descriptor.get("semantics")
    _require(
        isinstance(identity_raw, Mapping) and isinstance(semantics_raw, Mapping),
        "provider descriptor identity/semantics are invalid",
    )
    identity = ProviderIdentity.from_dict(identity_raw)
    semantics = ActionSemantics.from_dict(semantics_raw)
    _require(
        descriptor.get("semantics_sha256") == semantics.semantic_sha256,
        "provider semantics hash mismatch",
    )
    response_rows = descriptor.get("responses")
    _require(
        isinstance(response_rows, list) and bool(response_rows),
        "provider response table is invalid",
    )
    try:
        archive = np.load(root / "responses.npz", allow_pickle=False)
    except Exception as error:
        raise ProviderContractError(
            "provider response archive cannot be loaded"
        ) from error
    records: list[FrozenResponseRecord] = []
    try:
        declared_keys: set[str] = set()
        sequence_ids: set[int] = set()
        for row in response_rows:
            _require(isinstance(row, Mapping), "provider response row is invalid")
            _require(
                set(row)
                == {
                    "array_key",
                    "observation_sequence_id",
                    "observation_sha256",
                    "inference_latency_ms",
                    "shape",
                    "response_sha256",
                },
                "provider response row fields are invalid",
            )
            key = _json_string(row["array_key"])
            _require(
                re.fullmatch(r"response_[0-9]{4}", key) is not None,
                "provider response array key is invalid",
            )
            _require(key not in declared_keys, "provider response key is duplicated")
            declared_keys.add(key)
            try:
                sequence_id = _json_int(row["observation_sequence_id"])
                latency_ms = _json_number(row["inference_latency_ms"])
                shape = _json_int_tuple(row["shape"])
                actions = np.asarray(archive[key], dtype=float)
            except (KeyError, TypeError, ValueError) as error:
                raise ProviderContractError(
                    "provider response row is malformed"
                ) from error
            _require(
                sequence_id not in sequence_ids,
                "provider response sequence is duplicated",
            )
            sequence_ids.add(sequence_id)
            _require(
                actions.shape == shape
                and actions.ndim == 2
                and actions.shape[1] == semantics.action_dim,
                "provider response shape is invalid",
            )
            _require(
                row.get("response_sha256") == canonical_action_sha256(actions),
                "provider response action hash mismatch",
            )
            observation_sha = _json_optional_string(row["observation_sha256"])
            record = FrozenResponseRecord(
                observation_sequence_id=sequence_id,
                actions=actions,
                inference_latency_ms=latency_ms,
                observation_sha256=observation_sha,
            )
            records.append(record)
        _require(set(archive.files) == declared_keys, "provider archive keys mismatch")
    finally:
        archive.close()
    return {
        "valid": True,
        "directory": str(root),
        "provider_id": identity.provider_id,
        "model_family": identity.model_family,
        "responses": len(records),
        "semantics_sha256": semantics.semantic_sha256,
        "checkpoint_executed_this_run": False,
        "checks": [
            "manifest_and_file_hashes",
            "provider_identity_claim_boundary",
            "action_semantics_hash",
            "response_shape_and_canonical_hash",
        ],
    }


class FrozenResponseProvider:
    """Deterministic provider that replays validated response bytes once."""

    def __init__(
        self,
        identity: ProviderIdentity,
        semantics: ActionSemantics,
        records: Sequence[FrozenResponseRecord],
    ) -> None:
        if not records:
            raise ProviderContractError("frozen provider requires responses")
        self._identity = identity
        self._semantics = semantics
        self._records = tuple(records)
        self._index = 0

    @classmethod
    def from_directory(cls, directory: Path) -> "FrozenResponseProvider":
        validate_frozen_provider_bundle(directory)
        root = directory.resolve()
        descriptor = strict_json_load(root / "provider.json")
        identity = ProviderIdentity.from_dict(descriptor["identity"])
        semantics = ActionSemantics.from_dict(descriptor["semantics"])
        rows = descriptor["responses"]
        with np.load(root / "responses.npz", allow_pickle=False) as archive:
            records = [
                FrozenResponseRecord(
                    observation_sequence_id=int(row["observation_sequence_id"]),
                    actions=np.asarray(archive[str(row["array_key"])], dtype=float),
                    inference_latency_ms=float(row["inference_latency_ms"]),
                    observation_sha256=(
                        None
                        if row.get("observation_sha256") is None
                        else str(row["observation_sha256"])
                    ),
                )
                for row in rows
            ]
        return cls(identity, semantics, records)

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    @property
    def semantics(self) -> ActionSemantics:
        return self._semantics

    def infer_raw(self, observation: VLAObservation) -> RawActionChunk:
        if self._index >= len(self._records):
            raise RuntimeError("frozen provider responses exhausted")
        record = self._records[self._index]
        if record.observation_sequence_id != observation.sequence_id:
            raise ProviderContractError(
                "frozen response sequence does not match observation"
            )
        if (
            record.observation_sha256 is not None
            and record.observation_sha256 != canonical_observation_sha256(observation)
        ):
            raise ProviderContractError(
                "frozen response observation hash does not match"
            )
        self._index += 1
        response_sha = canonical_action_sha256(record.actions)
        return RawActionChunk(
            actions=record.actions,
            semantics=self.semantics,
            source=(f"frozen:{self.identity.provider_id}:{response_sha[:12]}"),
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=record.inference_latency_ms,
            received_at_s=(
                observation.captured_at_s + record.inference_latency_ms / 1000.0
            ),
            response_sha256=response_sha,
        )

    def reset(self) -> None:
        self._index = 0


class AdaptedActionChunkPolicy:
    """Gate provider semantics, then adapt raw Cartesian actions to Panda Hx8."""

    def __init__(
        self,
        provider: RawActionChunkProvider,
        adapter: PandaCartesianActionAdapter,
    ) -> None:
        self.provider = provider
        self.adapter = adapter
        self.expected_semantics = libero_cartesian_semantics(adapter.config)
        require_semantic_compatibility(provider.semantics, self.expected_semantics)

    @property
    def identity(self) -> ProviderIdentity:
        return self.provider.identity

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "source_semantics": self.provider.semantics.to_dict(),
            "source_semantics_sha256": self.provider.semantics.semantic_sha256,
            "adapter": "PandaCartesianActionAdapter",
            "checkpoint_executed_this_run": (
                self.identity.checkpoint_executed_this_run
            ),
        }

    def infer(self, observation: VLAObservation) -> ActionChunk:
        raw = self.provider.infer_raw(observation)
        require_semantic_compatibility(raw.semantics, self.expected_semantics)
        if raw.observation_sequence_id != observation.sequence_id:
            raise ProviderContractError(
                "provider response sequence does not match observation"
            )
        return self.adapter.adapt(
            raw.actions,
            observation.joint_position,
            source=raw.source,
            observation_sequence_id=raw.observation_sequence_id,
            inference_latency_ms=raw.inference_latency_ms,
            received_at_s=raw.received_at_s,
        ).chunk


def _fixture_identity() -> ProviderIdentity:
    return ProviderIdentity(
        provider_id="openvla_oft_libero_contract_fixture",
        model_family="OpenVLA-OFT",
        implementation_repository="https://github.com/moojink/openvla-oft",
        implementation_revision="e4287e94541f459edc4feabc4e181f537cd569a8",
        checkpoint_reference="not_loaded_cpu_contract_fixture",
        checkpoint_sha256=None,
        checkpoint_identity_status="declared_unverified",
        response_origin="synthetic_contract_fixture",
        checkpoint_executed_during_capture=False,
        checkpoint_executed_this_run=False,
    )


def _fixture_observation() -> VLAObservation:
    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    q_start = mujoco_scenarios()["free_space"].start
    exterior = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.zeros_like(exterior)
    exterior[::8, :, 0] = 80
    wrist[:, ::8, 1] = 120
    return VLAObservation(
        exterior_image=exterior,
        wrist_image=wrist,
        joint_position=q_start,
        gripper_position=np.array([1.0]),
        prompt="move the end effector forward",
        sequence_id=0,
        captured_at_s=100.0,
    )


def _fixture_actions() -> FloatArray:
    actions = np.zeros((6, LIBERO_ACTION_DIM), dtype=float)
    actions[:, 0] = 0.08
    actions[:, 2] = -0.02
    actions[:, 6] = -1.0
    return actions


def _mismatch_cases(semantics: ActionSemantics) -> dict[str, ActionSemantics]:
    return {
        "coordinate_frame": replace(semantics, coordinate_frame="tool"),
        "control_period": replace(semantics, control_period_s=0.1),
        "rotation_representation": replace(
            semantics, rotation_representation="euler_xyz_delta"
        ),
        "gripper_convention": replace(
            semantics, gripper_convention="minus_one_closed_plus_one_open"
        ),
        "controller_semantics": replace(
            semantics, controller_semantics_id="unverified-controller"
        ),
    }


def _audit_claims() -> dict[str, bool]:
    return {
        "openvla_oft_checkpoint_executed": False,
        "checkpoint_response_captured": False,
        "cross_model_task_success_measured": False,
        "gpu_latency_measured": False,
        "official_lerobot_runtime_used": False,
    }


def _audit_limitations() -> list[str]:
    return [
        "The OpenVLA-OFT-named response is a synthetic ABI fixture, not model output.",
        "The run validates provider metadata, exact action semantics, and adapter integration only.",
        "No cross-model task-success, generalization, or GPU-latency claim is supported.",
    ]


def run_provider_contract_audit(directory: Path) -> Path:
    """Create a self-validating CPU-only second-provider contract artifact."""

    from armbench.mujoco_sim import MuJoCoPanda

    root = directory.resolve()
    _require(not root.exists(), f"audit directory already exists: {root}")
    root.mkdir(parents=True)
    observation = _fixture_observation()
    semantics = libero_cartesian_semantics()
    raw_actions = _fixture_actions()
    provider_root = write_frozen_provider_bundle(
        root / "provider",
        identity=_fixture_identity(),
        semantics=semantics,
        responses=(
            FrozenResponseRecord(
                observation_sequence_id=0,
                actions=raw_actions,
                inference_latency_ms=37.0,
                observation_sha256=canonical_observation_sha256(observation),
            ),
        ),
    )
    provider = FrozenResponseProvider.from_directory(provider_root)
    robot = MuJoCoPanda.create(obstacles=())
    policy = AdaptedActionChunkPolicy(
        provider,
        PandaCartesianActionAdapter(robot),
    )
    chunk = policy.infer(observation)
    mismatch_rejections: dict[str, str] = {}
    for label, candidate in _mismatch_cases(semantics).items():
        try:
            require_semantic_compatibility(candidate, semantics)
        except SemanticCompatibilityError as error:
            mismatch_rejections[label] = str(error)
    np.savez_compressed(
        root / "audit_arrays.npz",
        exterior_image=observation.exterior_image,
        wrist_image=observation.wrist_image,
        joint_position=observation.joint_position,
        gripper_position=observation.gripper_position,
        raw_actions=np.asarray(raw_actions, dtype="<f4"),
        adapted_actions=np.asarray(chunk.actions, dtype="<f8"),
    )
    summary = {
        "schema_version": PROVIDER_AUDIT_SCHEMA,
        "scope": "cpu_only_provider_abi_semantic_gate_and_adapter",
        "provider": policy.provenance,
        "observation_sha256": canonical_observation_sha256(observation),
        "raw_response_sha256": canonical_action_sha256(raw_actions),
        "raw_shape": list(raw_actions.shape),
        "adapted_shape": list(chunk.actions.shape),
        "adapted_action_sha256": canonical_action_sha256(chunk.actions),
        "semantic_mismatch_cases": mismatch_rejections,
        "semantic_mismatch_rejections": len(mismatch_rejections),
        "claims": _audit_claims(),
        "limitations": _audit_limitations(),
    }
    write_json(root / "summary.json", summary)
    _write_manifest(
        root,
        schema=PROVIDER_AUDIT_MANIFEST_SCHEMA,
        recursive=True,
    )
    validate_provider_contract_audit(root)
    return root


def validate_provider_contract_audit(directory: Path) -> dict[str, object]:
    root = directory.resolve()
    _require(root.is_dir(), f"provider audit directory not found: {root}")
    _validate_manifest(
        root,
        schema=PROVIDER_AUDIT_MANIFEST_SCHEMA,
        recursive=True,
    )
    try:
        summary = strict_json_load(root / "summary.json")
    except (OSError, TypeError, ValueError) as error:
        raise ProviderContractError("provider audit summary is invalid") from error
    _require(
        isinstance(summary, Mapping)
        and summary.get("schema_version") == PROVIDER_AUDIT_SCHEMA
        and summary.get("scope") == "cpu_only_provider_abi_semantic_gate_and_adapter",
        "provider audit summary schema/scope mismatch",
    )
    _require(
        set(summary)
        == {
            "schema_version",
            "scope",
            "provider",
            "observation_sha256",
            "raw_response_sha256",
            "raw_shape",
            "adapted_shape",
            "adapted_action_sha256",
            "semantic_mismatch_cases",
            "semantic_mismatch_rejections",
            "claims",
            "limitations",
        },
        "provider audit summary fields are invalid",
    )
    validate_frozen_provider_bundle(root / "provider")
    try:
        archive = np.load(root / "audit_arrays.npz", allow_pickle=False)
    except Exception as error:
        raise ProviderContractError("provider audit arrays cannot be loaded") from error
    try:
        expected_keys = {
            "exterior_image",
            "wrist_image",
            "joint_position",
            "gripper_position",
            "raw_actions",
            "adapted_actions",
        }
        _require(
            set(archive.files) == expected_keys, "provider audit array keys mismatch"
        )
        observation = VLAObservation(
            exterior_image=archive["exterior_image"],
            wrist_image=archive["wrist_image"],
            joint_position=archive["joint_position"],
            gripper_position=archive["gripper_position"],
            prompt="move the end effector forward",
            sequence_id=0,
            captured_at_s=100.0,
        )
        raw_actions = np.asarray(archive["raw_actions"], dtype=float)
        stored_adapted = np.asarray(archive["adapted_actions"], dtype=float)
    finally:
        archive.close()
    _require(
        summary.get("observation_sha256") == canonical_observation_sha256(observation),
        "provider audit observation hash mismatch",
    )
    _require(
        summary.get("raw_response_sha256") == canonical_action_sha256(raw_actions),
        "provider audit raw response hash mismatch",
    )
    provider = FrozenResponseProvider.from_directory(root / "provider")
    from armbench.mujoco_sim import MuJoCoPanda

    policy = AdaptedActionChunkPolicy(
        provider,
        PandaCartesianActionAdapter(MuJoCoPanda.create(obstacles=())),
    )
    replayed = policy.infer(observation)
    _require(
        summary.get("provider") == policy.provenance,
        "provider audit provider provenance mismatch",
    )
    _require(
        summary.get("raw_shape") == list(raw_actions.shape)
        and summary.get("adapted_shape") == list(replayed.actions.shape),
        "provider audit action shapes mismatch",
    )
    _require(
        np.array_equal(replayed.actions, stored_adapted),
        "provider audit adapted actions are not reproducible",
    )
    _require(
        summary.get("adapted_action_sha256")
        == canonical_action_sha256(replayed.actions),
        "provider audit adapted action hash mismatch",
    )
    mismatches = _mismatch_cases(provider.semantics)
    mismatch_rejections: dict[str, str] = {}
    for label, candidate in mismatches.items():
        try:
            require_semantic_compatibility(candidate, provider.semantics)
        except SemanticCompatibilityError as error:
            mismatch_rejections[label] = str(error)
    _require(
        mismatch_rejections == summary.get("semantic_mismatch_cases")
        and len(mismatch_rejections)
        == len(mismatches)
        == summary.get("semantic_mismatch_rejections"),
        "provider audit semantic rejection count mismatch",
    )
    _require(
        summary.get("claims") == _audit_claims()
        and summary.get("limitations") == _audit_limitations(),
        "provider audit claim boundary is invalid",
    )
    return {
        "valid": True,
        "directory": str(root),
        "provider_id": provider.identity.provider_id,
        "model_family": provider.identity.model_family,
        "semantic_mismatch_rejections": len(mismatch_rejections),
        "raw_shape": list(raw_actions.shape),
        "adapted_shape": list(replayed.actions.shape),
        "checkpoint_executed_this_run": False,
        "checks": [
            "recursive_manifest_and_hashes",
            "frozen_provider_bundle",
            "observation_response_binding",
            "semantic_fail_closed_matrix",
            "deterministic_adapter_replay",
            "summary_fields_recomputed",
            "explicit_claim_boundary",
        ],
    }
