"""Reproducible CPU audit for dynamics-feasible Panda stopping trajectories."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import mujoco
import numpy as np

from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.dynamics_braking import (
    DynamicsBrakingConfig,
    generate_dynamics_validated_brake,
)
from armbench.mujoco_sim.model import (
    MENAGERIE_COMMIT,
    MuJoCoPanda,
    default_panda_scene_path,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    strict_json_loads,
    write_json,
)


AUDIT_SCHEMA = "armbench.dynamics_braking_audit.v1"
AUDIT_SCOPE = "sampled_inverse_dynamics_stop_with_continuous_collision_edges"
DEFAULT_VELOCITY_PROFILES = (
    "stationary",
    "low_forward",
    "low_reverse",
    "high_forward",
    "high_reverse",
)
CSV_FIELDS = (
    "schema_version",
    "case_id",
    "payload_mass_kg",
    "joint_damping_scale",
    "velocity_profile",
    "initial_velocity_rad_s",
    "validated",
    "failure_reason",
    "stop_time_s",
    "stopping_distance_l2_rad",
    "max_joint_stopping_distance_rad",
    "max_torque_ratio",
    "evaluated_samples",
    "latency_ms",
)
_SUMMARY_FIELDS = {
    "schema_version",
    "scope",
    "configuration",
    "environment",
    "panda_scene_sha256",
    "implementation_sha256",
    "overall",
    "by_payload_kg",
    "claim_boundary",
}
_ENVIRONMENT_FIELDS = {
    "mujoco_version",
    "numpy_version",
    "menagerie_commit",
}
_OVERALL_FIELDS = {
    "cases",
    "validated_stops",
    "fail_closed_rejections",
    "rejection_rate",
    "maximum_stop_time_s",
    "maximum_stopping_distance_l2_rad",
    "maximum_torque_ratio",
    "p95_validation_latency_ms",
    "maximum_validation_latency_ms",
}
_CONFIG_FIELDS = {
    "payload_masses_kg",
    "joint_damping_scales",
    "velocity_profiles",
    "sample_dt_s",
    "acceleration_limit_rad_s2",
    "max_stop_time_s",
    "actuator_force_limit_scale",
}
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256"}
_CLAIM_BOUNDARY = [
    "The candidate stop is sampled at the registered period.",
    "Inverse dynamics uses the compiled MuJoCo Panda and actuator limits.",
    "Collision edges use conservative MuJoCo pair-distance bounds.",
    "No closed-loop tracking, hard-real-time, or hardware claim is made.",
]


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class DynamicsBrakingAuditConfig:
    """Registered matrix for the local Panda dynamics audit."""

    payload_masses_kg: tuple[float, ...] = (0.0, 0.5, 1.0)
    joint_damping_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    velocity_profiles: tuple[str, ...] = DEFAULT_VELOCITY_PROFILES
    sample_dt_s: float = 0.01
    acceleration_limit_rad_s2: float = 5.0
    max_stop_time_s: float = 1.0
    actuator_force_limit_scale: float = 0.8

    def __post_init__(self) -> None:
        raw_payloads = np.asarray(self.payload_masses_kg)
        raw_damping = np.asarray(self.joint_damping_scales)
        if (
            raw_payloads.dtype.kind not in {"i", "u", "f"}
            or raw_damping.dtype.kind not in {"i", "u", "f"}
        ):
            raise ValueError("audit payload and damping values must be numeric")
        payloads = np.asarray(raw_payloads, dtype=float)
        damping = np.asarray(raw_damping, dtype=float)
        sample_dt = _finite_float(self.sample_dt_s, "sample_dt_s")
        acceleration_limit = _finite_float(
            self.acceleration_limit_rad_s2, "acceleration_limit_rad_s2"
        )
        max_stop_time = _finite_float(self.max_stop_time_s, "max_stop_time_s")
        force_scale = _finite_float(
            self.actuator_force_limit_scale, "actuator_force_limit_scale"
        )
        if (
            len(payloads) == 0
            or not np.all(np.isfinite(payloads))
            or np.any(payloads < 0.0)
            or len(set(float(value) for value in payloads)) != len(payloads)
        ):
            raise ValueError("audit payload masses must be unique and nonnegative")
        if (
            len(damping) == 0
            or not np.all(np.isfinite(damping))
            or np.any(damping <= 0.0)
            or len(set(float(value) for value in damping)) != len(damping)
        ):
            raise ValueError("audit damping scales must be unique and positive")
        if (
            not isinstance(self.velocity_profiles, tuple)
            or not self.velocity_profiles
            or any(type(name) is not str for name in self.velocity_profiles)
            or len(set(self.velocity_profiles)) != len(self.velocity_profiles)
            or any(name not in DEFAULT_VELOCITY_PROFILES for name in self.velocity_profiles)
        ):
            raise ValueError("audit velocity profiles are invalid")
        if any(
            value <= 0.0
            for value in (
                sample_dt,
                acceleration_limit,
                max_stop_time,
                force_scale,
            )
        ):
            raise ValueError("audit timing and physical limits must be positive")
        if force_scale > 1.0:
            raise ValueError("audit actuator force scale cannot exceed one")
        object.__setattr__(
            self, "payload_masses_kg", tuple(float(value) for value in payloads)
        )
        object.__setattr__(
            self, "joint_damping_scales", tuple(float(value) for value in damping)
        )
        object.__setattr__(self, "velocity_profiles", tuple(self.velocity_profiles))
        object.__setattr__(self, "sample_dt_s", sample_dt)
        object.__setattr__(self, "acceleration_limit_rad_s2", acceleration_limit)
        object.__setattr__(self, "max_stop_time_s", max_stop_time)
        object.__setattr__(self, "actuator_force_limit_scale", force_scale)


def _velocity_profile(name: str) -> np.ndarray:
    direction = np.asarray([1.0, -0.8, 0.6, -0.5, 0.4, -0.3, 0.2])
    profiles = {
        "stationary": np.zeros(7),
        "low_forward": 0.25 * direction,
        "low_reverse": -0.25 * direction,
        "high_forward": direction,
        "high_reverse": -direction,
    }
    try:
        return profiles[name].copy()
    except KeyError as error:
        raise ValueError(f"unknown braking velocity profile: {name}") from error


def _braking_config(settings: DynamicsBrakingAuditConfig) -> DynamicsBrakingConfig:
    return DynamicsBrakingConfig(
        sample_dt_s=settings.sample_dt_s,
        joint_acceleration_limits_rad_s2=(
            settings.acceleration_limit_rad_s2,
        )
        * 7,
        max_stop_time_s=settings.max_stop_time_s,
        actuator_force_limit_scale=settings.actuator_force_limit_scale,
        check_inter_sample_edges=True,
    )


def _case_id(payload: float, damping: float, profile: str) -> str:
    return f"payload_{payload:04.1f}kg__damping_{damping:04.1f}x__{profile}"


def _make_runtime(
    payload: float, damping: float
) -> tuple[MuJoCoPanda, ContinuousMuJoCoCollisionChecker]:
    robot = MuJoCoPanda.create(obstacles=(), payload_mass=payload)
    robot.model.dof_damping[robot.arm_dof_addresses] *= damping
    return robot, ContinuousMuJoCoCollisionChecker(robot)


def _evaluate_case(
    robot: MuJoCoPanda,
    checker: ContinuousMuJoCoCollisionChecker,
    settings: DynamicsBrakingAuditConfig,
    *,
    payload: float,
    damping: float,
    profile: str,
) -> dict[str, object]:
    velocity = _velocity_profile(profile)
    started = perf_counter()
    result = generate_dynamics_validated_brake(
        robot,
        checker,
        mujoco_scenarios()["free_space"].start,
        velocity,
        _braking_config(settings),
    )
    latency_ms = (perf_counter() - started) * 1000.0
    return {
        "schema_version": AUDIT_SCHEMA,
        "case_id": _case_id(payload, damping, profile),
        "payload_mass_kg": payload,
        "joint_damping_scale": damping,
        "velocity_profile": profile,
        "initial_velocity_rad_s": canonical_json(velocity.tolist()).decode("ascii"),
        "validated": result.validated,
        "failure_reason": result.failure_reason or "",
        "stop_time_s": result.stop_time_s,
        "stopping_distance_l2_rad": result.stopping_distance_l2_rad,
        "max_joint_stopping_distance_rad": result.max_joint_stopping_distance_rad,
        "max_torque_ratio": result.max_torque_ratio,
        "evaluated_samples": result.evaluated_samples,
        "latency_ms": latency_ms,
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty dynamics audit")
    latencies = np.asarray([float(row["latency_ms"]) for row in rows])
    torque_ratios = [
        float(row["max_torque_ratio"])
        for row in rows
        if row["max_torque_ratio"] is not None
    ]
    validated = sum(bool(row["validated"]) for row in rows)
    rejected = len(rows) - validated
    return {
        "cases": len(rows),
        "validated_stops": validated,
        "fail_closed_rejections": rejected,
        "rejection_rate": rejected / len(rows),
        "maximum_stop_time_s": max(float(row["stop_time_s"]) for row in rows),
        "maximum_stopping_distance_l2_rad": max(
            float(row["stopping_distance_l2_rad"]) for row in rows
        ),
        "maximum_torque_ratio": max(torque_ratios) if torque_ratios else None,
        "p95_validation_latency_ms": float(np.percentile(latencies, 95)),
        "maximum_validation_latency_ms": float(np.max(latencies)),
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]
    if not isinstance(overall, Mapping):
        raise ValueError("dynamics audit overall summary is invalid")
    return (
        "# Panda dynamics braking audit\n\n"
        f"Cases: {overall['cases']}\n\n"
        f"Validated stops: {overall['validated_stops']}\n\n"
        f"Fail-closed rejections: {overall['fail_closed_rejections']}\n\n"
        f"Maximum stop time: {float(overall['maximum_stop_time_s']):.6f} s\n\n"
        "Maximum joint-space stopping distance: "
        f"{float(overall['maximum_stopping_distance_l2_rad']):.6f} rad\n\n"
        "This is sampled MuJoCo inverse-dynamics evidence with continuous "
        "collision checks between samples, not a hardware safety certificate.\n"
    )


def _write_manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": f"{AUDIT_SCHEMA}.manifest",
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json(files)),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def run_dynamics_braking_audit(
    output_directory: Path,
    config: DynamicsBrakingAuditConfig = DynamicsBrakingAuditConfig(),
) -> Path:
    """Run the registered CPU matrix and write a self-validating artifact."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"dynamics audit output already exists: {output}")
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for payload in config.payload_masses_kg:
        for damping in config.joint_damping_scales:
            robot, checker = _make_runtime(payload, damping)
            for profile in config.velocity_profiles:
                rows.append(
                    _evaluate_case(
                        robot,
                        checker,
                        config,
                        payload=payload,
                        damping=damping,
                        profile=profile,
                    )
                )

    with (output / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": AUDIT_SCHEMA,
        "scope": AUDIT_SCOPE,
        "configuration": asdict(config),
        "environment": {
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "panda_scene_sha256": sha256_file(default_panda_scene_path()),
        "implementation_sha256": {
            "armbench/mujoco_sim/dynamics_braking.py": sha256_file(
                Path(__file__).with_name("dynamics_braking.py")
            ),
            "armbench/mujoco_sim/dynamics_braking_audit.py": sha256_file(
                Path(__file__)
            ),
            "armbench/mujoco_sim/continuous_collision.py": sha256_file(
                Path(__file__).with_name("continuous_collision.py")
            ),
        },
        "overall": _aggregate(rows),
        "by_payload_kg": {
            f"{payload:g}": _aggregate(
                [row for row in rows if float(row["payload_mass_kg"]) == payload]
            )
            for payload in config.payload_masses_kg
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    _write_manifest(output)
    validate_dynamics_braking_audit(output)
    return output


def _load_config(value: object) -> DynamicsBrakingAuditConfig:
    if not has_exact_fields(value, _CONFIG_FIELDS):
        raise ValueError("dynamics audit configuration is invalid")
    try:
        config = DynamicsBrakingAuditConfig(
            payload_masses_kg=tuple(value["payload_masses_kg"]),
            joint_damping_scales=tuple(value["joint_damping_scales"]),
            velocity_profiles=tuple(value["velocity_profiles"]),
            sample_dt_s=float(value["sample_dt_s"]),
            acceleration_limit_rad_s2=float(value["acceleration_limit_rad_s2"]),
            max_stop_time_s=float(value["max_stop_time_s"]),
            actuator_force_limit_scale=float(value["actuator_force_limit_scale"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dynamics audit configuration is invalid") from error
    if canonical_json(value) != canonical_json(asdict(config)):
        raise ValueError("dynamics audit configuration is not canonical")
    return config


def _validate_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    expected_files = {"per_case.csv", "summary.json", "summary.md"}
    if not (
        has_exact_fields(manifest, _MANIFEST_FIELDS)
        and manifest["schema_version"] == f"{AUDIT_SCHEMA}.manifest"
        and isinstance(manifest["files"], list)
        and is_sha256(manifest["inventory_sha256"])
        and {
            item["path"]
            for item in manifest["files"]
            if has_exact_fields(item, _MANIFEST_ENTRY_FIELDS)
        }
        == expected_files
        and manifest["inventory_sha256"]
        == sha256_bytes(canonical_json(manifest["files"]))
    ):
        raise ValueError("dynamics audit manifest is invalid")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError("dynamics audit directory contains undeclared files")
    for item in manifest["files"]:
        if not (
            has_exact_fields(item, _MANIFEST_ENTRY_FIELDS)
            and isinstance(item["path"], str)
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0
            and is_sha256(item["sha256"])
        ):
            raise ValueError("dynamics audit manifest entry is invalid")
        relative = Path(item["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"dynamics audit manifest mismatch: {relative}")
    return str(manifest["inventory_sha256"])


def _parse_row(raw: Mapping[str, str]) -> dict[str, object]:
    if set(raw) != set(CSV_FIELDS):
        raise ValueError("dynamics audit row fields are invalid")
    if raw["schema_version"] != AUDIT_SCHEMA:
        raise ValueError("dynamics audit row schema is invalid")
    if raw["validated"] not in {"True", "False"}:
        raise ValueError("dynamics audit row boolean is invalid")
    try:
        vector_value = strict_json_loads(raw["initial_velocity_rad_s"])
    except (TypeError, ValueError) as error:
        raise ValueError("dynamics audit initial velocity vector is invalid") from error
    if not (
        isinstance(vector_value, list)
        and len(vector_value) == 7
        and all(
            type(value) in {int, float} and np.isfinite(value)
            for value in vector_value
        )
    ):
        raise ValueError("dynamics audit initial velocity vector is invalid")
    try:
        vector = np.asarray(vector_value, dtype=float)
        row = {
            **raw,
            "payload_mass_kg": float(raw["payload_mass_kg"]),
            "joint_damping_scale": float(raw["joint_damping_scale"]),
            "initial_velocity_rad_s": canonical_json(vector.tolist()).decode("ascii"),
            "validated": raw["validated"] == "True",
            "stop_time_s": float(raw["stop_time_s"]),
            "stopping_distance_l2_rad": float(raw["stopping_distance_l2_rad"]),
            "max_joint_stopping_distance_rad": float(
                raw["max_joint_stopping_distance_rad"]
            ),
            "max_torque_ratio": (
                None if raw["max_torque_ratio"] == "" else float(raw["max_torque_ratio"])
            ),
            "evaluated_samples": int(raw["evaluated_samples"]),
            "latency_ms": float(raw["latency_ms"]),
        }
    except (TypeError, ValueError) as error:
        raise ValueError("dynamics audit row value is invalid") from error
    numeric = [
        row["payload_mass_kg"],
        row["joint_damping_scale"],
        row["stop_time_s"],
        row["stopping_distance_l2_rad"],
        row["max_joint_stopping_distance_rad"],
        row["latency_ms"],
    ]
    if (
        vector.shape != (7,)
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(np.asarray(numeric, dtype=float)))
        or any(float(value) < 0.0 for value in numeric)
        or row["evaluated_samples"] < 0
        or (
            row["max_torque_ratio"] is not None
            and (
                not np.isfinite(row["max_torque_ratio"])
                or row["max_torque_ratio"] < 0.0
            )
        )
    ):
        raise ValueError("dynamics audit row numeric contract is invalid")
    return row


def _same_deterministic_result(
    stored: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    exact_fields = {
        "schema_version",
        "case_id",
        "velocity_profile",
        "initial_velocity_rad_s",
        "validated",
        "failure_reason",
        "evaluated_samples",
    }
    float_fields = {
        "payload_mass_kg",
        "joint_damping_scale",
        "stop_time_s",
        "stopping_distance_l2_rad",
        "max_joint_stopping_distance_rad",
    }
    if any(stored[field] != expected[field] for field in exact_fields):
        return False
    if any(
        not np.isclose(
            float(stored[field]), float(expected[field]), rtol=1e-12, atol=1e-12
        )
        for field in float_fields
    ):
        return False
    stored_ratio = stored["max_torque_ratio"]
    expected_ratio = expected["max_torque_ratio"]
    return (stored_ratio is None and expected_ratio is None) or (
        stored_ratio is not None
        and expected_ratio is not None
        and np.isclose(
            float(stored_ratio), float(expected_ratio), rtol=1e-12, atol=1e-12
        )
    )


def _compare_aggregate(actual: object, expected: Mapping[str, object]) -> None:
    if not has_exact_fields(actual, _OVERALL_FIELDS):
        raise ValueError("dynamics audit aggregate fields are invalid")
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("dynamics audit aggregate mismatch")


def validate_dynamics_braking_audit(directory: Path) -> dict[str, object]:
    """Verify hashes, rerun every physical case, and recompute aggregates."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dynamics audit directory not found: {root}")
    inventory_hash = _validate_manifest(root)
    summary = strict_json_load(root / "summary.json")
    if not (
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == AUDIT_SCHEMA
        and summary["scope"] == AUDIT_SCOPE
        and summary["panda_scene_sha256"] == sha256_file(default_panda_scene_path())
        and canonical_json(summary["claim_boundary"])
        == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("dynamics audit summary is invalid")
    config = _load_config(summary["configuration"])
    environment = summary["environment"]
    if not (
        has_exact_fields(environment, _ENVIRONMENT_FIELDS)
        and environment["mujoco_version"] == mujoco.__version__
        and environment["numpy_version"] == np.__version__
        and environment["menagerie_commit"] == MENAGERIE_COMMIT
    ):
        raise ValueError("dynamics audit environment mismatch")
    hashes = summary["implementation_sha256"]
    expected_hash_paths = {
        "armbench/mujoco_sim/dynamics_braking.py": Path(__file__).with_name(
            "dynamics_braking.py"
        ),
        "armbench/mujoco_sim/dynamics_braking_audit.py": Path(__file__),
        "armbench/mujoco_sim/continuous_collision.py": Path(__file__).with_name(
            "continuous_collision.py"
        ),
    }
    if not isinstance(hashes, Mapping) or set(hashes) != set(expected_hash_paths):
        raise ValueError("dynamics audit implementation hashes are invalid")
    if any(
        hashes[label] != sha256_file(path)
        for label, path in expected_hash_paths.items()
    ):
        raise ValueError("dynamics audit implementation hash mismatch")

    with (root / "per_case.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("dynamics audit CSV fields are invalid")
        stored_rows = [_parse_row(row) for row in reader]
    expected_count = (
        len(config.payload_masses_kg)
        * len(config.joint_damping_scales)
        * len(config.velocity_profiles)
    )
    if len(stored_rows) != expected_count:
        raise ValueError("dynamics audit case count is invalid")

    expected_rows: list[dict[str, object]] = []
    for payload in config.payload_masses_kg:
        for damping in config.joint_damping_scales:
            robot, checker = _make_runtime(payload, damping)
            for profile in config.velocity_profiles:
                expected_rows.append(
                    _evaluate_case(
                        robot,
                        checker,
                        config,
                        payload=payload,
                        damping=damping,
                        profile=profile,
                    )
                )
    if len({row["case_id"] for row in stored_rows}) != expected_count:
        raise ValueError("dynamics audit case identities are not unique")
    for stored, expected in zip(stored_rows, expected_rows):
        if not _same_deterministic_result(stored, expected):
            raise ValueError(f"dynamics audit recomputation failed: {stored['case_id']}")

    overall = _aggregate(stored_rows)
    _compare_aggregate(summary["overall"], overall)
    by_payload = summary["by_payload_kg"]
    expected_payload_keys = {f"{payload:g}" for payload in config.payload_masses_kg}
    if not isinstance(by_payload, Mapping) or set(by_payload) != expected_payload_keys:
        raise ValueError("dynamics audit payload aggregates are invalid")
    for payload in config.payload_masses_kg:
        rows = [
            row for row in stored_rows if float(row["payload_mass_kg"]) == payload
        ]
        _compare_aggregate(by_payload[f"{payload:g}"], _aggregate(rows))
    if (root / "summary.md").read_text("utf-8") != _summary_markdown(summary):
        raise ValueError("dynamics audit Markdown summary is not reproducible")
    return {
        "valid": True,
        "scope": AUDIT_SCOPE,
        "cases": expected_count,
        "validated_stops": overall["validated_stops"],
        "fail_closed_rejections": overall["fail_closed_rejections"],
        "manifest_inventory_sha256": inventory_hash,
        "checks": [
            "recursive_manifest",
            "all_cases_rerun_with_mujoco_inverse_dynamics",
            "continuous_collision_edges_rerun",
            "payload_and_damping_aggregates_recomputed",
        ],
    }


__all__ = [
    "DynamicsBrakingAuditConfig",
    "run_dynamics_braking_audit",
    "validate_dynamics_braking_audit",
]
