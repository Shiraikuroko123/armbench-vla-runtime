"""Reproducible continuous self-collision audit for the Menagerie Panda."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionConfig,
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MuJoCoPanda, default_panda_scene_path


AUDIT_SCHEMA = "armbench.mujoco_self_collision_audit.v1"
AUDIT_SCOPE = "continuous_self_collision_certificate_vs_dense_sampled_oracle"
STRATA = ("known_intermediate", "local", "global")
CSV_FIELDS = (
    "schema_version",
    "stratum",
    "edge_index",
    "q_start",
    "q_end",
    "endpoint_start_valid",
    "endpoint_end_valid",
    "continuous_status",
    "continuous_reason",
    "continuous_collision_pair",
    "continuous_valid",
    "continuous_pair_evaluations",
    "continuous_subintervals",
    "continuous_max_depth",
    "continuous_minimum_distance_m",
    "continuous_motion_bound_m",
    "dense_samples",
    "dense_valid",
    "false_safe",
    "conservative_rejection",
    "continuous_latency_ms",
    "dense_latency_ms",
)

# A fixed edge whose endpoints are collision-free but whose linear interpolation
# crosses a Panda self-collision. It is kept as a mechanism control in every
# regenerated artifact rather than relying on a lucky random draw.
KNOWN_SELF_START = np.asarray(
    [
        2.013896901192687,
        0.8514937392438364,
        2.128626643260985,
        -0.7308358555657621,
        1.5703065944221684,
        1.5832426543430034,
        0.5959999237674181,
    ],
    dtype=float,
)
KNOWN_SELF_END = np.asarray(
    [
        2.044760389840342,
        -1.7175851740339965,
        2.558238709554596,
        -2.9380304930478514,
        -2.797519848117953,
        1.1846938393238287,
        -2.8554195536676996,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SelfCollisionAuditConfig:
    """Frozen protocol for the self-collision certificate audit."""

    strata: tuple[str, ...] = STRATA
    samples_per_stratum: int = 24
    seed: int = 20260808
    dense_resolution_rad: float = 0.002
    continuous_max_depth: int = 16
    continuous_minimum_interval_rad: float = 1e-6
    continuous_max_pair_evaluations: int = 250_000

    def __post_init__(self) -> None:
        if (
            not self.strata
            or len(set(self.strata)) != len(self.strata)
            or any(name not in STRATA for name in self.strata)
        ):
            raise ValueError("self-collision strata are invalid")
        if type(self.samples_per_stratum) is not int or self.samples_per_stratum <= 0:
            raise ValueError("samples_per_stratum must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not np.isfinite(self.dense_resolution_rad)
            or self.dense_resolution_rad <= 0.0
        ):
            raise ValueError("dense_resolution_rad must be finite and positive")
        if type(self.continuous_max_depth) is not int or self.continuous_max_depth < 0:
            raise ValueError("continuous_max_depth must be a nonnegative integer")
        if (
            not np.isfinite(self.continuous_minimum_interval_rad)
            or self.continuous_minimum_interval_rad <= 0.0
        ):
            raise ValueError(
                "continuous_minimum_interval_rad must be finite and positive"
            )
        if (
            type(self.continuous_max_pair_evaluations) is not int
            or self.continuous_max_pair_evaluations <= 0
        ):
            raise ValueError(
                "continuous_max_pair_evaluations must be a positive integer"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _write_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": f"{AUDIT_SCHEMA}.manifest",
        "files": files,
        "inventory_sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _clip_interior(robot: MuJoCoPanda, q: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    return np.clip(q, robot.lower_limits + epsilon, robot.upper_limits - epsilon)


def _build_edges(
    robot: MuJoCoPanda, settings: SelfCollisionAuditConfig
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Generate the exact deterministic edge list used by run and validate."""

    rng = np.random.default_rng(settings.seed)
    edges: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    base = np.asarray(
        [0.0, -0.75, 0.0, -2.25, 0.0, 1.55, 0.75], dtype=float
    )
    for stratum in settings.strata:
        for edge_index in range(settings.samples_per_stratum):
            if stratum == "known_intermediate" and edge_index == 0:
                start = KNOWN_SELF_START.copy()
                end = KNOWN_SELF_END.copy()
            elif stratum == "local":
                start = _clip_interior(robot, base + rng.normal(0.0, 0.14, 7))
                end = _clip_interior(robot, start + rng.normal(0.0, 0.28, 7))
            else:
                start = _clip_interior(
                    robot, rng.uniform(robot.lower_limits, robot.upper_limits)
                )
                end = _clip_interior(
                    robot, rng.uniform(robot.lower_limits, robot.upper_limits)
                )
            edges.append((stratum, edge_index, start, end))
    return edges


def _make_checkers(
    settings: SelfCollisionAuditConfig,
) -> tuple[MuJoCoPanda, ContinuousMuJoCoCollisionChecker, MuJoCoCollisionChecker]:
    robot = MuJoCoPanda.create(obstacles=())
    continuous = ContinuousMuJoCoCollisionChecker(
        robot,
        ContinuousCollisionConfig(
            max_depth=settings.continuous_max_depth,
            minimum_interval_rad=settings.continuous_minimum_interval_rad,
            max_pair_evaluations=settings.continuous_max_pair_evaluations,
            include_static_obstacles=False,
            include_self_collision=True,
        ),
    )
    dense = MuJoCoCollisionChecker(robot, resolution=settings.dense_resolution_rad)
    return robot, continuous, dense


def _bool(value: bool) -> str:
    return "True" if bool(value) else "False"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"self-collision boolean value is invalid: {value!r}")


def _float_or_blank(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _evaluate_edge(
    continuous: ContinuousMuJoCoCollisionChecker,
    dense: MuJoCoCollisionChecker,
    stratum: str,
    edge_index: int,
    start: np.ndarray,
    end: np.ndarray,
) -> dict[str, Any]:
    start_valid = dense.configuration_is_valid(start)
    end_valid = dense.configuration_is_valid(end)
    started = perf_counter()
    certificate = continuous.edge_certificate(start, end)
    continuous_latency_ms = (perf_counter() - started) * 1000.0
    started = perf_counter()
    dense_valid = dense.edge_is_valid(start, end)
    dense_latency_ms = (perf_counter() - started) * 1000.0
    return {
        "schema_version": AUDIT_SCHEMA,
        "stratum": stratum,
        "edge_index": edge_index,
        "q_start": json.dumps(start.tolist(), separators=(",", ":")),
        "q_end": json.dumps(end.tolist(), separators=(",", ":")),
        "endpoint_start_valid": _bool(start_valid),
        "endpoint_end_valid": _bool(end_valid),
        "continuous_status": certificate.status,
        "continuous_reason": certificate.reason,
        "continuous_collision_pair": certificate.collision_pair or "",
        "continuous_valid": _bool(certificate.certified_safe),
        "continuous_pair_evaluations": certificate.pair_evaluations,
        "continuous_subintervals": certificate.subintervals_evaluated,
        "continuous_max_depth": certificate.maximum_depth_reached,
        "continuous_minimum_distance_m": _float_or_blank(
            certificate.minimum_sampled_distance_m
        ),
        "continuous_motion_bound_m": certificate.maximum_interval_motion_bound_m,
        "dense_samples": dense.edge_sample_count(start, end),
        "dense_valid": _bool(dense_valid),
        "false_safe": _bool(certificate.certified_safe and not dense_valid),
        "conservative_rejection": _bool(
            not certificate.certified_safe and dense_valid
        ),
        "continuous_latency_ms": continuous_latency_ms,
        "dense_latency_ms": dense_latency_ms,
    }


def _typed_row(raw: Mapping[str, str]) -> dict[str, Any]:
    def parse_bool(name: str) -> bool:
        value = raw.get(name)
        if value not in {"True", "False"}:
            raise ValueError(f"self-collision boolean is invalid: {name}")
        return value == "True"

    def parse_float(name: str, *, allow_blank: bool = False) -> float | None:
        value = raw.get(name, "")
        if allow_blank and value == "":
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"self-collision numeric field is invalid: {name}") from error
        if not np.isfinite(parsed):
            raise ValueError(f"self-collision numeric field is nonfinite: {name}")
        return parsed

    result: dict[str, Any] = dict(raw)
    try:
        result["edge_index"] = int(raw["edge_index"])
        result["continuous_pair_evaluations"] = int(
            raw["continuous_pair_evaluations"]
        )
        result["continuous_subintervals"] = int(raw["continuous_subintervals"])
        result["continuous_max_depth"] = int(raw["continuous_max_depth"])
        result["dense_samples"] = int(raw["dense_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("self-collision integer field is invalid") from error
    for name in (
        "continuous_minimum_distance_m",
        "continuous_motion_bound_m",
        "continuous_latency_ms",
        "dense_latency_ms",
    ):
        result[name] = parse_float(name, allow_blank=name == "continuous_minimum_distance_m")
    for name in (
        "endpoint_start_valid",
        "endpoint_end_valid",
        "continuous_valid",
        "dense_valid",
        "false_safe",
        "conservative_rejection",
    ):
        result[name] = parse_bool(name)
    if result["edge_index"] < 0 or result["dense_samples"] <= 0:
        raise ValueError("self-collision integer field is out of range")
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty self-collision audit")
    continuous_latency = [float(row["continuous_latency_ms"]) for row in rows]
    dense_latency = [float(row["dense_latency_ms"]) for row in rows]
    return {
        "edges": len(rows),
        "endpoint_safe_edges": sum(
            _as_bool(row["endpoint_start_valid"])
            and _as_bool(row["endpoint_end_valid"])
            for row in rows
        ),
        "continuous_safe": sum(_as_bool(row["continuous_valid"]) for row in rows),
        "dense_safe": sum(_as_bool(row["dense_valid"]) for row in rows),
        "false_safe": sum(_as_bool(row["false_safe"]) for row in rows),
        "conservative_rejections": sum(
            _as_bool(row["conservative_rejection"]) for row in rows
        ),
        "p95_continuous_latency_ms": _percentile(continuous_latency, 95),
        "maximum_continuous_latency_ms": max(continuous_latency),
        "p95_dense_latency_ms": _percentile(dense_latency, 95),
        "maximum_dense_latency_ms": max(dense_latency),
        "maximum_pair_evaluations": max(
            int(row["continuous_pair_evaluations"]) for row in rows
        ),
    }


def _stratum_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "edges": len(rows),
        "endpoint_safe_edges": sum(
            _as_bool(row["endpoint_start_valid"])
            and _as_bool(row["endpoint_end_valid"])
            for row in rows
        ),
        "continuous_safe": sum(_as_bool(row["continuous_valid"]) for row in rows),
        "dense_safe": sum(_as_bool(row["dense_valid"]) for row in rows),
        "false_safe": sum(_as_bool(row["false_safe"]) for row in rows),
        "conservative_rejections": sum(
            _as_bool(row["conservative_rejection"]) for row in rows
        ),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    return (
        "# Panda self-collision audit\n\n"
        f"Edges: {overall['edges']}\n\n"
        f"Endpoint-safe edges: {overall['endpoint_safe_edges']}\n\n"
        f"False-safe decisions against dense oracle: {overall['false_safe']}\n\n"
        f"Conservative rejections: {overall['conservative_rejections']}\n\n"
        "The certificate covers linear joint interpolation over the compiled "
        "MuJoCo geometry. The dense oracle is sampled evidence, not an analytic "
        "proof or a hardware safety certificate.\n"
    )


def run_self_collision_audit(
    output_directory: Path,
    *,
    config: SelfCollisionAuditConfig | None = None,
) -> Path:
    """Run and persist the frozen self-collision matrix."""

    settings = config or SelfCollisionAuditConfig()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"output directory must not already exist: {output}")
    output.mkdir(parents=True)
    robot, continuous, dense = _make_checkers(settings)
    rows = [
        _evaluate_edge(continuous, dense, stratum, edge_index, start, end)
        for stratum, edge_index, start, end in _build_edges(robot, settings)
    ]
    with (output / "per_edge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    by_stratum = {
        stratum: _stratum_summary(
            [row for row in rows if row["stratum"] == stratum]
        )
        for stratum in settings.strata
    }
    summary = {
        "schema_version": AUDIT_SCHEMA,
        "scope": AUDIT_SCOPE,
        "configuration": asdict(settings),
        "panda_scene_sha256": _sha256(default_panda_scene_path()),
        "implementation_sha256": {
            "armbench/mujoco_sim/self_collision_audit.py": _sha256(Path(__file__)),
            "armbench/mujoco_sim/continuous_collision.py": _sha256(
                Path(__file__).with_name("continuous_collision.py")
            ),
            "armbench/mujoco_sim/collision.py": _sha256(
                Path(__file__).with_name("collision.py")
            ),
        },
        "overall": _aggregate(rows),
        "by_stratum": by_stratum,
        "claim_boundary": [
            "The continuous checker covers linear joint interpolation and compiled MuJoCo geometry.",
            "The dense comparison is a sampled oracle, not an analytic proof.",
            "The result does not establish physical robot safety, hard real-time behavior, or emergency stopping.",
        ],
    }
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    _write_manifest(output)
    return output


def _compare_value(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f"self-collision value mismatch: {label}")
    elif actual != expected:
        raise ValueError(f"self-collision value mismatch: {label}")


def validate_self_collision_audit(directory: Path) -> dict[str, Any]:
    """Recompute every self-collision decision and verify the manifest."""

    root = directory.resolve()
    if not root.is_dir():
        raise ValueError(f"self-collision audit directory not found: {root}")
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("self-collision audit JSON is unreadable") from error
    if summary.get("schema_version") != AUDIT_SCHEMA or summary.get("scope") != AUDIT_SCOPE:
        raise ValueError("self-collision audit summary schema is invalid")
    if manifest.get("schema_version") != f"{AUDIT_SCHEMA}.manifest":
        raise ValueError("self-collision audit manifest schema is invalid")
    files = manifest.get("files")
    expected_files = {"per_edge.csv", "summary.json", "summary.md"}
    if (
        not isinstance(files, list)
        or any(not isinstance(item, Mapping) for item in files)
        or {str(item.get("path")) for item in files} != expected_files
    ):
        raise ValueError("self-collision audit manifest file set is invalid")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError("self-collision audit directory contains undeclared files")
    if hashlib.sha256(_canonical_json(files)).hexdigest() != manifest.get(
        "inventory_sha256"
    ):
        raise ValueError("self-collision audit inventory hash is invalid")
    for item in files:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("self-collision audit manifest path is invalid")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError(f"self-collision audit manifest mismatch: {item['path']}")

    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("self-collision audit configuration is invalid")
    try:
        settings = SelfCollisionAuditConfig(
            strata=tuple(str(value) for value in configuration["strata"]),
            samples_per_stratum=int(configuration["samples_per_stratum"]),
            seed=int(configuration["seed"]),
            dense_resolution_rad=float(configuration["dense_resolution_rad"]),
            continuous_max_depth=int(configuration["continuous_max_depth"]),
            continuous_minimum_interval_rad=float(
                configuration["continuous_minimum_interval_rad"]
            ),
            continuous_max_pair_evaluations=int(
                configuration["continuous_max_pair_evaluations"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("self-collision audit configuration is invalid") from error
    if json.loads(json.dumps(asdict(settings))) != dict(configuration):
        raise ValueError("self-collision audit configuration is not canonical")
    if summary.get("panda_scene_sha256") != _sha256(default_panda_scene_path()):
        raise ValueError("self-collision audit Panda scene hash is invalid")
    expected_hashes = {
        "armbench/mujoco_sim/self_collision_audit.py": _sha256(Path(__file__)),
        "armbench/mujoco_sim/continuous_collision.py": _sha256(
            Path(__file__).with_name("continuous_collision.py")
        ),
        "armbench/mujoco_sim/collision.py": _sha256(
            Path(__file__).with_name("collision.py")
        ),
    }
    if summary.get("implementation_sha256") != expected_hashes:
        raise ValueError("self-collision audit implementation hashes are invalid")

    with (root / "per_edge.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("self-collision audit CSV fields are invalid")
        raw_rows = list(reader)
    expected_edges = _build_edges(MuJoCoPanda.create(obstacles=()), settings)
    if len(raw_rows) != len(expected_edges):
        raise ValueError("self-collision audit edge count is invalid")
    robot, continuous, dense = _make_checkers(settings)
    typed_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for raw, (stratum, edge_index, start, end) in zip(raw_rows, expected_edges):
        row = _typed_row(raw)
        identity = (stratum, edge_index)
        if identity in identities or (row["stratum"], row["edge_index"]) != identity:
            raise ValueError("self-collision edge identity is invalid")
        identities.add(identity)
        for name, expected in (("q_start", start), ("q_end", end)):
            try:
                vector = np.asarray(json.loads(row[name]), dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"self-collision vector is invalid: {name}") from error
            if vector.shape != (7,) or not np.allclose(vector, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"self-collision edge endpoint mismatch: {identity}")
        expected_row = _evaluate_edge(continuous, dense, stratum, edge_index, start, end)
        for name in (
            "endpoint_start_valid",
            "endpoint_end_valid",
            "continuous_status",
            "continuous_reason",
            "continuous_collision_pair",
            "continuous_valid",
            "continuous_pair_evaluations",
            "continuous_subintervals",
            "continuous_max_depth",
            "dense_samples",
            "dense_valid",
            "false_safe",
            "conservative_rejection",
        ):
            _compare_value(row[name], _typed_row(expected_row)[name], f"{identity}.{name}")
        for name in ("continuous_minimum_distance_m", "continuous_motion_bound_m"):
            actual = row[name]
            expected = (
                None
                if expected_row[name] == ""
                else float(expected_row[name])
            )
            if actual is None or expected is None:
                if actual != expected:
                    raise ValueError(f"self-collision value mismatch: {identity}.{name}")
            else:
                _compare_value(actual, expected, f"{identity}.{name}")
        for name in ("continuous_latency_ms", "dense_latency_ms"):
            if row[name] < 0.0:
                raise ValueError(f"self-collision latency is invalid: {identity}.{name}")
        typed_rows.append(row)

    expected_overall = _aggregate(typed_rows)
    if summary.get("overall") != expected_overall:
        # Latency is intentionally nondeterministic; compare its shape and all
        # deterministic counts exactly, then accept the recorded timing summary.
        actual_overall = summary.get("overall")
        if not isinstance(actual_overall, Mapping):
            raise ValueError("self-collision overall summary is invalid")
        for name in (
            "edges",
            "endpoint_safe_edges",
            "continuous_safe",
            "dense_safe",
            "false_safe",
            "conservative_rejections",
            "maximum_pair_evaluations",
        ):
            if int(actual_overall[name]) != int(expected_overall[name]):
                raise ValueError(f"self-collision aggregate mismatch: {name}")
        for name in (
            "p95_continuous_latency_ms",
            "maximum_continuous_latency_ms",
            "p95_dense_latency_ms",
            "maximum_dense_latency_ms",
        ):
            if not np.isfinite(float(actual_overall[name])) or float(actual_overall[name]) < 0.0:
                raise ValueError(f"self-collision aggregate latency is invalid: {name}")
    expected_by_stratum = {
        stratum: _stratum_summary(
            [row for row in typed_rows if row["stratum"] == stratum]
        )
        for stratum in settings.strata
    }
    if summary.get("by_stratum") != expected_by_stratum:
        raise ValueError("self-collision stratum summary mismatch")
    return {
        "valid": True,
        "edges": len(typed_rows),
        "endpoint_safe_edges": expected_overall["endpoint_safe_edges"],
        "false_safe": expected_overall["false_safe"],
        "conservative_rejections": expected_overall["conservative_rejections"],
        "checks": [
            "manifest_hashes",
            "edge_endpoints_and_decisions_recomputed",
            "aggregate_counts_recomputed",
            "implementation_and_scene_hashes",
        ],
    }
