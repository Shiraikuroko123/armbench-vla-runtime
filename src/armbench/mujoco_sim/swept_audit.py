"""CPU-only audit for the clearance-backed MuJoCo swept checker."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.model import MuJoCoPanda, default_panda_scene_path
from armbench.mujoco_sim.scenarios import mujoco_scenarios


AUDIT_SCHEMA = "armbench.mujoco_swept_audit.v1"
AUDIT_SCOPE = "clearance_backed_swept_checker_vs_dense_sampled_oracle"
CSV_FIELDS = (
    "schema_version",
    "scenario",
    "edge_index",
    "q_start",
    "q_end",
    "workspace_bound_m",
    "swept_samples",
    "swept_valid",
    "dense_valid",
    "false_safe",
    "conservative_rejection",
    "swept_latency_ms",
    "dense_latency_ms",
)


@dataclass(frozen=True)
class SweptAuditConfig:
    scenarios: tuple[str, ...] = ("free_space", "single_block", "narrow_gate")
    samples_per_scenario: int = 24
    seed: int = 20260808
    clearance_m: float = 0.02
    sampled_resolution_rad: float = 0.05
    dense_resolution_rad: float = 0.002

    def __post_init__(self) -> None:
        available = mujoco_scenarios()
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or any(name not in available for name in self.scenarios)
        ):
            raise ValueError("scenarios must be unique known MuJoCo scenarios")
        if self.samples_per_scenario <= 0:
            raise ValueError("samples_per_scenario must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not np.isfinite(self.clearance_m) or self.clearance_m <= 0.0:
            raise ValueError("clearance_m must be finite and positive")
        if (
            not np.isfinite(self.sampled_resolution_rad)
            or self.sampled_resolution_rad <= 0.0
            or not np.isfinite(self.dense_resolution_rad)
            or self.dense_resolution_rad <= 0.0
            or self.dense_resolution_rad >= self.sampled_resolution_rad
        ):
            raise ValueError("audit resolutions are invalid")


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


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty swept audit")
    return {
        "edges": len(rows),
        "swept_valid": sum(bool(row["swept_valid"]) for row in rows),
        "dense_valid": sum(bool(row["dense_valid"]) for row in rows),
        "false_safe": sum(bool(row["false_safe"]) for row in rows),
        "conservative_rejections": sum(
            bool(row["conservative_rejection"]) for row in rows
        ),
        "mean_swept_samples": float(
            np.mean([int(row["swept_samples"]) for row in rows])
        ),
        "p95_swept_latency_ms": _percentile(
            [float(row["swept_latency_ms"]) for row in rows], 95
        ),
        "p95_dense_latency_ms": _percentile(
            [float(row["dense_latency_ms"]) for row in rows], 95
        ),
        "max_workspace_bound_m": max(
            float(row["workspace_bound_m"]) for row in rows
        ),
    }


def _scenario_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "edges": len(rows),
        "false_safe": sum(bool(row["false_safe"]) for row in rows),
        "conservative_rejections": sum(
            bool(row["conservative_rejection"]) for row in rows
        ),
        "swept_valid": sum(bool(row["swept_valid"]) for row in rows),
        "dense_valid": sum(bool(row["dense_valid"]) for row in rows),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    return (
        "# MuJoCo swept collision audit\n\n"
        f"Edges: {summary['overall']['edges']}\n\n"
        "False-safe edges against dense oracle: "
        f"{summary['overall']['false_safe']}\n\n"
        "Conservative rejections: "
        f"{summary['overall']['conservative_rejections']}\n\n"
        "The dense oracle is sampled evidence, not a continuous proof.\n"
    )


def _build_edges(
    scenario_name: str,
    robot: MuJoCoPanda,
    samples: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    scenario = mujoco_scenarios()[scenario_name]
    edges = [(scenario.start.copy(), scenario.goal.copy())]
    for _ in range(max(0, samples - 1)):
        start = rng.uniform(robot.lower_limits, robot.upper_limits)
        end = rng.uniform(robot.lower_limits, robot.upper_limits)
        edges.append((start, end))
    return edges


def _write_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    inventory = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": f"{AUDIT_SCHEMA}.manifest",
        "files": files,
        "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def run_swept_collision_audit(
    output_directory: Path,
    *,
    config: SweptAuditConfig | None = None,
) -> Path:
    """Compare the conservative checker with a denser sampled oracle."""

    settings = config or SweptAuditConfig()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    output.mkdir(parents=True)
    rng = np.random.default_rng(settings.seed)
    rows: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {}

    for scenario_name in settings.scenarios:
        scenario = mujoco_scenarios()[scenario_name]
        inflated = inflate_obstacles(scenario.obstacles, settings.clearance_m)
        swept_robot = MuJoCoPanda.create(obstacles=inflated)
        dense_robot = MuJoCoPanda.create(obstacles=inflated)
        swept = MuJoCoCollisionChecker(
            swept_robot,
            resolution=settings.sampled_resolution_rad,
            swept_obstacle_margin_m=settings.clearance_m,
        )
        dense = MuJoCoCollisionChecker(
            dense_robot,
            resolution=settings.dense_resolution_rad,
        )
        scenario_rows: list[dict[str, Any]] = []
        for edge_index, (start, end) in enumerate(
            _build_edges(scenario_name, swept_robot, settings.samples_per_scenario, rng)
        ):
            bound_m = swept.edge_workspace_motion_bound(start, end)
            sample_count = swept.edge_sample_count(start, end)
            started = perf_counter()
            swept_valid = swept.edge_is_valid(start, end)
            swept_ms = (perf_counter() - started) * 1000.0
            started = perf_counter()
            dense_valid = dense.edge_is_valid(start, end)
            dense_ms = (perf_counter() - started) * 1000.0
            row = {
                "schema_version": AUDIT_SCHEMA,
                "scenario": scenario_name,
                "edge_index": edge_index,
                "q_start": json.dumps(start.tolist(), separators=(",", ":")),
                "q_end": json.dumps(end.tolist(), separators=(",", ":")),
                "workspace_bound_m": bound_m,
                "swept_samples": sample_count,
                "swept_valid": swept_valid,
                "dense_valid": dense_valid,
                "false_safe": swept_valid and not dense_valid,
                "conservative_rejection": not swept_valid and dense_valid,
                "swept_latency_ms": swept_ms,
                "dense_latency_ms": dense_ms,
            }
            rows.append(row)
            scenario_rows.append(row)
        by_scenario[scenario_name] = scenario_rows

    with (output / "per_edge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "scope": AUDIT_SCOPE,
        "configuration": asdict(settings),
        "panda_scene_sha256": _sha256(default_panda_scene_path()),
        "implementation_sha256": {
            "armbench/mujoco_sim/swept_audit.py": _sha256(Path(__file__)),
            "armbench/mujoco_sim/collision.py": _sha256(
                Path(__file__).with_name("collision.py")
            ),
        },
        "overall": _aggregate(rows),
        "by_scenario": {},
        "claim_boundary": [
            "The dense comparison is a stronger sampled oracle, not an analytic proof.",
            "The certificate covers static obstacles represented by the configured clearance.",
            "Self-collision remains sampled and is not a continuous certificate.",
            "No physical robot or hard-real-time claim is made.",
        ],
    }
    for name, scenario_rows in by_scenario.items():
        summary["by_scenario"][name] = _scenario_counts(scenario_rows)
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_manifest(output)
    return output


def _parse_bool(value: str, label: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"swept audit boolean is invalid: {label}")
    return value == "True"


def _parse_vector(value: str, label: str) -> np.ndarray:
    try:
        vector = np.asarray(json.loads(value), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"swept audit vector is invalid: {label}") from error
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"swept audit vector is invalid: {label}")
    return vector


def _compare_aggregate(
    actual: Any, expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"swept audit aggregate fields are invalid: {label}")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if not np.isclose(
                float(actual_value), expected_value, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    f"swept audit aggregate mismatch: {label}.{key}"
                )
        elif int(actual_value) != expected_value:
            raise ValueError(
                f"swept audit aggregate mismatch: {label}.{key}"
            )


def validate_swept_collision_audit(directory: Path) -> dict[str, Any]:
    """Recompute every edge decision, aggregate, and protected file."""

    root = directory.resolve()
    if not root.is_dir():
        raise ValueError(f"swept audit directory not found: {root}")
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != AUDIT_SCHEMA
        or summary.get("scope") != AUDIT_SCOPE
    ):
        raise ValueError("swept audit summary schema is invalid")

    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != f"{AUDIT_SCHEMA}.manifest":
        raise ValueError("swept audit manifest schema is invalid")
    files = manifest.get("files")
    expected_files = {"per_edge.csv", "summary.json", "summary.md"}
    if (
        not isinstance(files, list)
        or any(not isinstance(item, Mapping) for item in files)
        or {str(item.get("path")) for item in files} != expected_files
    ):
        raise ValueError("swept audit manifest file set is invalid")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError("swept audit directory contains undeclared files")
    inventory = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(inventory).hexdigest() != manifest.get(
        "inventory_sha256"
    ):
        raise ValueError("swept audit inventory hash is invalid")
    for item in files:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("swept audit manifest path is invalid")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError(f"swept audit manifest mismatch: {item['path']}")

    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("swept audit configuration is invalid")
    try:
        settings = SweptAuditConfig(
            scenarios=tuple(str(value) for value in configuration["scenarios"]),
            samples_per_scenario=int(configuration["samples_per_scenario"]),
            seed=int(configuration["seed"]),
            clearance_m=float(configuration["clearance_m"]),
            sampled_resolution_rad=float(
                configuration["sampled_resolution_rad"]
            ),
            dense_resolution_rad=float(configuration["dense_resolution_rad"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("swept audit configuration is invalid") from error
    normalized_settings = json.loads(json.dumps(asdict(settings)))
    if dict(configuration) != normalized_settings:
        raise ValueError("swept audit configuration is not canonical")

    implementation_hashes = summary.get("implementation_sha256")
    expected_hash_keys = {
        "armbench/mujoco_sim/swept_audit.py",
        "armbench/mujoco_sim/collision.py",
    }
    if (
        not isinstance(implementation_hashes, Mapping)
        or set(implementation_hashes) != expected_hash_keys
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in implementation_hashes.values()
        )
    ):
        raise ValueError("swept audit implementation hashes are invalid")
    scene_hash = summary.get("panda_scene_sha256")
    if scene_hash != _sha256(default_panda_scene_path()):
        raise ValueError("swept audit Panda scene hash is invalid")

    with (root / "per_edge.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("swept audit CSV field set is invalid")
        raw_rows = list(reader)
    expected_edges = len(settings.scenarios) * settings.samples_per_scenario
    if len(raw_rows) != expected_edges:
        raise ValueError("swept audit edge count is invalid")

    checkers: dict[
        str, tuple[MuJoCoCollisionChecker, MuJoCoCollisionChecker]
    ] = {}
    for scenario_name in settings.scenarios:
        scenario = mujoco_scenarios()[scenario_name]
        inflated = inflate_obstacles(scenario.obstacles, settings.clearance_m)
        checkers[scenario_name] = (
            MuJoCoCollisionChecker(
                MuJoCoPanda.create(obstacles=inflated),
                resolution=settings.sampled_resolution_rad,
                swept_obstacle_margin_m=settings.clearance_m,
            ),
            MuJoCoCollisionChecker(
                MuJoCoPanda.create(obstacles=inflated),
                resolution=settings.dense_resolution_rad,
            ),
        )

    typed_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for raw in raw_rows:
        if raw["schema_version"] != AUDIT_SCHEMA:
            raise ValueError("swept audit CSV schema is invalid")
        scenario_name = raw["scenario"]
        if scenario_name not in checkers:
            raise ValueError("swept audit CSV scenario is invalid")
        try:
            edge_index = int(raw["edge_index"])
            workspace_bound_m = float(raw["workspace_bound_m"])
            swept_samples = int(raw["swept_samples"])
            swept_latency_ms = float(raw["swept_latency_ms"])
            dense_latency_ms = float(raw["dense_latency_ms"])
        except ValueError as error:
            raise ValueError("swept audit numeric field is invalid") from error
        identity = (scenario_name, edge_index)
        if identity in identities or edge_index < 0:
            raise ValueError("swept audit edge identity is invalid")
        identities.add(identity)
        start = _parse_vector(raw["q_start"], f"{identity}.q_start")
        end = _parse_vector(raw["q_end"], f"{identity}.q_end")
        swept, dense = checkers[scenario_name]
        if not swept.robot.within_limits(start) or not swept.robot.within_limits(end):
            raise ValueError("swept audit edge endpoint violates joint limits")
        expected_bound = swept.edge_workspace_motion_bound(start, end)
        expected_samples = swept.edge_sample_count(start, end)
        expected_swept_valid = swept.edge_is_valid(start, end)
        expected_dense_valid = dense.edge_is_valid(start, end)
        stored_swept_valid = _parse_bool(
            raw["swept_valid"], f"{identity}.swept_valid"
        )
        stored_dense_valid = _parse_bool(
            raw["dense_valid"], f"{identity}.dense_valid"
        )
        expected_false_safe = expected_swept_valid and not expected_dense_valid
        expected_conservative = not expected_swept_valid and expected_dense_valid
        if (
            not np.isfinite(workspace_bound_m)
            or workspace_bound_m < 0.0
            or not np.isclose(
                workspace_bound_m, expected_bound, rtol=1e-12, atol=1e-12
            )
            or swept_samples != expected_samples
            or stored_swept_valid != expected_swept_valid
            or stored_dense_valid != expected_dense_valid
            or _parse_bool(raw["false_safe"], f"{identity}.false_safe")
            != expected_false_safe
            or _parse_bool(
                raw["conservative_rejection"],
                f"{identity}.conservative_rejection",
            )
            != expected_conservative
            or not np.isfinite(swept_latency_ms)
            or swept_latency_ms < 0.0
            or not np.isfinite(dense_latency_ms)
            or dense_latency_ms < 0.0
        ):
            raise ValueError(f"swept audit edge recomputation failed: {identity}")
        typed_rows.append(
            {
                **raw,
                "edge_index": edge_index,
                "workspace_bound_m": workspace_bound_m,
                "swept_samples": swept_samples,
                "swept_valid": stored_swept_valid,
                "dense_valid": stored_dense_valid,
                "false_safe": expected_false_safe,
                "conservative_rejection": expected_conservative,
                "swept_latency_ms": swept_latency_ms,
                "dense_latency_ms": dense_latency_ms,
            }
        )

    for scenario_name in settings.scenarios:
        expected_indices = set(range(settings.samples_per_scenario))
        actual_indices = {
            int(row["edge_index"])
            for row in typed_rows
            if row["scenario"] == scenario_name
        }
        if actual_indices != expected_indices:
            raise ValueError("swept audit scenario edge indices are incomplete")
    expected_overall = _aggregate(typed_rows)
    _compare_aggregate(summary.get("overall"), expected_overall, "overall")
    by_scenario = summary.get("by_scenario")
    if not isinstance(by_scenario, Mapping) or set(by_scenario) != set(
        settings.scenarios
    ):
        raise ValueError("swept audit scenario summary is invalid")
    for scenario_name in settings.scenarios:
        scenario_rows = [
            row for row in typed_rows if row["scenario"] == scenario_name
        ]
        _compare_aggregate(
            by_scenario[scenario_name],
            _scenario_counts(scenario_rows),
            scenario_name,
        )
    if expected_overall["false_safe"]:
        raise ValueError("swept checker accepted an edge rejected by dense oracle")
    if (root / "summary.md").read_text("utf-8") != _summary_markdown(summary):
        raise ValueError("swept audit Markdown summary is not reproducible")
    return {
        "valid": True,
        "scope": summary["scope"],
        "edges": len(typed_rows),
        "false_safe": expected_overall["false_safe"],
        "manifest_inventory_sha256": manifest["inventory_sha256"],
        "checks": [
            "manifest_inventory_sizes_and_hashes",
            "edge_endpoints_and_decisions_recomputed",
            "summary_recomputed_from_csv",
            "markdown_summary_recomputed",
        ],
    }
