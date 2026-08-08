"""Recomputable CPU fault matrix for the integrated Panda supervisor."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
import osqp

from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import (
    MENAGERIE_COMMIT,
    MuJoCoPanda,
    default_panda_scene_path,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaDecision,
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
)
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    write_json,
)
from armbench.vla.types import ActionChunk


MATRIX_SCHEMA = "armbench.integrated_panda_fault_matrix.v1"
MATRIX_SCOPE = "registered_scripted_faults_through_integrated_cpu_supervisor"
REGISTERED_FAULTS = (
    "nominal",
    "velocity_spike",
    "stale_response",
    "state_mismatch",
)
CSV_FIELDS = (
    "schema_version",
    "case_id",
    "scenario",
    "payload_mass_kg",
    "fault",
    "expected_status",
    "expected_failure_stage",
    "status",
    "reason",
    "failure_stage",
    "failure_index",
    "expected_match",
    "policy_actions_executable",
    "policy_action_count",
    "qp_feasible",
    "qp_intervention_steps",
    "qp_fallback_steps",
    "continuous_edges_checked",
    "continuous_rejection_status",
    "continuous_pair_evaluations",
    "conservative_rejection",
    "braking_boundaries_checked",
    "braking_invariant_complete",
    "fallback_validated",
    "fallback_failure_reason",
    "goal_distance_before_rad",
    "goal_distance_after_rad",
    "task_progress_rad",
    "max_inverse_dynamics_torque_ratio",
    "total_latency_ms",
    "qp_latency_ms",
    "continuous_collision_latency_ms",
    "dynamics_braking_latency_ms",
    "fallback_brake_latency_ms",
)
_SUMMARY_FIELDS = {
    "schema_version",
    "scope",
    "configuration",
    "environment",
    "panda_scene_sha256",
    "implementation_sha256",
    "overall",
    "by_fault",
    "claim_boundary",
}
_ENVIRONMENT_FIELDS = {
    "mujoco_version",
    "numpy_version",
    "osqp_version",
    "menagerie_commit",
}
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256"}
_CASE_DOCUMENT_FIELDS = {"schema_version", "cases"}
_CASE_FIELDS = {
    "case_id",
    "scenario",
    "payload_mass_kg",
    "fault",
    "q_rad",
    "qvel_rad_s",
    "observed_q_rad",
    "response_age_ms",
    "actions",
    "target_q_rad",
    "expected_status",
    "expected_failure_stage",
}
_AGGREGATE_FIELDS = {
    "cases",
    "expected_matches",
    "accepted_plans",
    "verified_brakes",
    "holds",
    "unrecoverable_stops",
    "fallback_covered_cases",
    "qp_intervention_steps",
    "continuous_edges_checked",
    "conservative_rejections",
    "mean_accepted_task_progress_rad",
    "p95_supervision_latency_ms",
    "maximum_supervision_latency_ms",
    "maximum_inverse_dynamics_torque_ratio",
}
_CLAIM_BOUNDARY = [
    "Inputs are deterministic scripted joint-velocity chunks, not learned-policy outputs.",
    "The supervisor is synchronous CPU code and does not claim hard-real-time scheduling.",
    "Continuous certificates cover the registered MuJoCo geometry and linear joint edges.",
    "Inverse dynamics checks model feasibility, not physical-robot emergency-stop safety.",
    "This matrix does not execute a task in closed-loop MuJoCo physics.",
]
_SELF_START = np.array(
    [
        2.013896901192687,
        0.8514937392438364,
        2.128626643260985,
        -0.7308358555657621,
        1.5703065944221684,
        1.5832426543430034,
        0.5959999237674181,
    ]
)
_SELF_END = np.array(
    [
        2.044760389840342,
        -1.7175851740339965,
        2.558238709554596,
        -2.9380304930478514,
        -2.797519848117953,
        1.1846938393238287,
        -2.8554195536676996,
    ]
)


@dataclass(frozen=True)
class IntegratedPandaMatrixConfig:
    """Registered dimensions and limits for the local CPU matrix."""

    scenarios: tuple[str, ...] = ("free_space", "single_block", "narrow_gate")
    payload_masses_kg: tuple[float, ...] = (0.0, 0.5)
    faults: tuple[str, ...] = REGISTERED_FAULTS
    control_dt_s: float = 0.05
    response_deadline_ms: float = 5000.0
    supervision_budget_ms: float = 5000.0
    qp_step_budget_ms: float = 500.0
    max_state_mismatch_rad: float = 0.05
    nominal_horizon: int = 3
    intermediate_self_collision_horizon: int = 120
    include_special_cases: bool = True

    def __post_init__(self) -> None:
        available = set(mujoco_scenarios())
        if (
            not isinstance(self.scenarios, tuple)
            or not self.scenarios
            or any(type(name) is not str or name not in available for name in self.scenarios)
            or len(set(self.scenarios)) != len(self.scenarios)
        ):
            raise ValueError("matrix scenarios are invalid")
        payloads = np.asarray(self.payload_masses_kg)
        if payloads.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("matrix payloads must be numeric")
        payload_values = np.asarray(payloads, dtype=float)
        if (
            payload_values.ndim != 1
            or len(payload_values) == 0
            or not np.all(np.isfinite(payload_values))
            or np.any(payload_values < 0.0)
            or len(set(float(value) for value in payload_values))
            != len(payload_values)
        ):
            raise ValueError("matrix payloads must be unique and nonnegative")
        if (
            not isinstance(self.faults, tuple)
            or set(self.faults) != set(REGISTERED_FAULTS)
            or len(self.faults) != len(REGISTERED_FAULTS)
        ):
            raise ValueError("matrix faults must contain the registered set")
        numeric = np.asarray(
            [
                self.control_dt_s,
                self.response_deadline_ms,
                self.supervision_budget_ms,
                self.qp_step_budget_ms,
                self.max_state_mismatch_rad,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(numeric))
            or np.any(numeric[:4] <= 0.0)
            or numeric[4] < 0.0
            or type(self.nominal_horizon) is not int
            or self.nominal_horizon <= 0
            or type(self.intermediate_self_collision_horizon) is not int
            or self.intermediate_self_collision_horizon <= 1
            or type(self.include_special_cases) is not bool
        ):
            raise ValueError("matrix timing and horizons are invalid")
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(
            self,
            "payload_masses_kg",
            tuple(float(value) for value in payload_values),
        )
        object.__setattr__(self, "faults", tuple(self.faults))


def _guard_config(config: IntegratedPandaMatrixConfig) -> IntegratedPandaGuardConfig:
    return IntegratedPandaGuardConfig(
        control_dt_s=config.control_dt_s,
        response_deadline_ms=config.response_deadline_ms,
        supervision_budget_ms=config.supervision_budget_ms,
        max_state_mismatch_rad=config.max_state_mismatch_rad,
        qp_step_budget_ms=config.qp_step_budget_ms,
    )


def _case_specs(config: IntegratedPandaMatrixConfig) -> list[dict[str, object]]:
    scenarios = mujoco_scenarios()
    cases: list[dict[str, object]] = []
    for scenario_name in config.scenarios:
        scenario = scenarios[scenario_name]
        direction = scenario.goal - scenario.start
        direction /= np.max(np.abs(direction))
        nominal = np.zeros((config.nominal_horizon, 8), dtype=float)
        nominal[:, :7] = 0.10 * direction
        nominal[:, 7] = 0.5
        spike = nominal.copy()
        spike[:, :7] = 5.0 * direction
        for payload in config.payload_masses_kg:
            for fault in config.faults:
                qvel = np.zeros(7)
                observed = scenario.start.copy()
                response_age_ms = 0.0
                actions = nominal.copy()
                expected_status = "accepted"
                expected_stage: str | None = None
                if fault == "velocity_spike":
                    actions = spike.copy()
                elif fault == "stale_response":
                    qvel = 0.10 * direction
                    response_age_ms = config.response_deadline_ms + 1.0
                    expected_status = "verified_brake"
                    expected_stage = "deadline"
                elif fault == "state_mismatch":
                    observed[0] += config.max_state_mismatch_rad + 0.01
                    expected_status = "hold"
                    expected_stage = "state_alignment"
                cases.append(
                    {
                        "case_id": (
                            f"{scenario_name}__payload_{payload:.1f}kg__{fault}"
                        ),
                        "scenario": scenario_name,
                        "payload_mass_kg": payload,
                        "fault": fault,
                        "q_rad": scenario.start.tolist(),
                        "qvel_rad_s": qvel.tolist(),
                        "observed_q_rad": observed.tolist(),
                        "response_age_ms": response_age_ms,
                        "actions": actions.tolist(),
                        "target_q_rad": scenario.goal.tolist(),
                        "expected_status": expected_status,
                        "expected_failure_stage": expected_stage,
                    }
                )

    if not config.include_special_cases:
        return cases

    horizon = config.intermediate_self_collision_horizon
    self_actions = np.zeros((horizon, 8), dtype=float)
    self_actions[:, :7] = (_SELF_END - _SELF_START) / (
        horizon * config.control_dt_s
    )
    self_actions[:, 7] = 0.5
    cases.append(
        {
            "case_id": "free_space__payload_0.0kg__intermediate_self_collision",
            "scenario": "free_space",
            "payload_mass_kg": 0.0,
            "fault": "intermediate_self_collision",
            "q_rad": _SELF_START.tolist(),
            "qvel_rad_s": np.zeros(7).tolist(),
            "observed_q_rad": _SELF_START.tolist(),
            "response_age_ms": 0.0,
            "actions": self_actions.tolist(),
            "target_q_rad": _SELF_END.tolist(),
            "expected_status": "hold",
            "expected_failure_stage": "continuous_collision",
        }
    )

    upper = MuJoCoPanda.create(obstacles=()).upper_limits - 0.001
    for payload in config.payload_masses_kg:
        velocity = np.zeros(7)
        velocity[0] = 0.2
        cases.append(
            {
                "case_id": f"free_space__payload_{payload:.1f}kg__near_limit_stop",
                "scenario": "free_space",
                "payload_mass_kg": payload,
                "fault": "near_limit_stop",
                "q_rad": upper.tolist(),
                "qvel_rad_s": velocity.tolist(),
                "observed_q_rad": upper.tolist(),
                "response_age_ms": config.response_deadline_ms + 1.0,
                "actions": np.zeros((config.nominal_horizon, 8)).tolist(),
                "target_q_rad": scenarios["free_space"].goal.tolist(),
                "expected_status": "unrecoverable_stop",
                "expected_failure_stage": "deadline",
            }
        )
    return cases


def _decision_row(
    spec: Mapping[str, object], decision: IntegratedPandaDecision
) -> dict[str, object]:
    q = np.asarray(spec["q_rad"], dtype=float)
    target = np.asarray(spec["target_q_rad"], dtype=float)
    final = decision.predicted_positions[-1]
    before = float(np.linalg.norm(target - q))
    after = float(np.linalg.norm(target - final))
    metrics = decision.metrics()
    rejection_status = ""
    if decision.edge_certificates and not decision.edge_certificates[-1].certified_safe:
        rejection_status = decision.edge_certificates[-1].status
    return {
        "schema_version": MATRIX_SCHEMA,
        "case_id": spec["case_id"],
        "scenario": spec["scenario"],
        "payload_mass_kg": spec["payload_mass_kg"],
        "fault": spec["fault"],
        "expected_status": spec["expected_status"],
        "expected_failure_stage": spec["expected_failure_stage"] or "",
        "status": decision.status,
        "reason": decision.reason,
        "failure_stage": decision.failure_stage or "",
        "failure_index": (
            "" if decision.failure_index is None else decision.failure_index
        ),
        "expected_match": (
            decision.status == spec["expected_status"]
            and (decision.failure_stage or "")
            == (spec["expected_failure_stage"] or "")
        ),
        "policy_actions_executable": decision.policy_actions_executable,
        "policy_action_count": len(decision.executable_actions),
        "qp_feasible": (
            "" if decision.qp_result is None else decision.qp_result.feasible
        ),
        "qp_intervention_steps": decision.intervention_steps,
        "qp_fallback_steps": (
            decision.qp_result.fallback_steps if decision.qp_result else 0
        ),
        "continuous_edges_checked": len(decision.edge_certificates),
        "continuous_rejection_status": rejection_status,
        "continuous_pair_evaluations": sum(
            item.pair_evaluations for item in decision.edge_certificates
        ),
        "conservative_rejection": decision.conservative_rejection,
        "braking_boundaries_checked": len(decision.braking_certificates),
        "braking_invariant_complete": decision.fallback_covered,
        "fallback_validated": (
            ""
            if decision.fallback_brake is None
            else decision.fallback_brake.validated
        ),
        "fallback_failure_reason": (
            ""
            if decision.fallback_brake is None
            else decision.fallback_brake.failure_reason or ""
        ),
        "goal_distance_before_rad": before,
        "goal_distance_after_rad": after,
        "task_progress_rad": before - after,
        "max_inverse_dynamics_torque_ratio": metrics[
            "max_inverse_dynamics_torque_ratio"
        ],
        "total_latency_ms": decision.total_latency_ms,
        "qp_latency_ms": decision.stage_latencies_ms.get("qp_projection", 0.0),
        "continuous_collision_latency_ms": decision.stage_latencies_ms.get(
            "continuous_collision", 0.0
        ),
        "dynamics_braking_latency_ms": decision.stage_latencies_ms.get(
            "dynamics_braking", 0.0
        ),
        "fallback_brake_latency_ms": decision.stage_latencies_ms.get(
            "fallback_brake", 0.0
        ),
    }


def _evaluate_case(
    spec: Mapping[str, object], config: IntegratedPandaMatrixConfig
) -> dict[str, object]:
    scenario = mujoco_scenarios()[str(spec["scenario"])]
    robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        payload_mass=float(spec["payload_mass_kg"]),
    )
    checker = ContinuousMuJoCoCollisionChecker(robot)
    supervisor = IntegratedPandaSupervisor(robot, checker, _guard_config(config))
    chunk = ActionChunk(
        actions=np.asarray(spec["actions"], dtype=float),
        source="registered_scripted_nonlearned_fault_matrix",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
        received_at_s=1.0,
    )
    decision = supervisor.supervise(
        spec["q_rad"],
        spec["qvel_rad_s"],
        chunk,
        observed_q=spec["observed_q_rad"],
        response_age_ms=float(spec["response_age_ms"]),
    )
    return _decision_row(spec, decision)


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty integrated matrix")
    latencies = np.asarray([float(row["total_latency_ms"]) for row in rows])
    accepted_progress = [
        float(row["task_progress_rad"])
        for row in rows
        if row["status"] == "accepted"
    ]
    torque_ratios = [
        float(row["max_inverse_dynamics_torque_ratio"])
        for row in rows
        if row["max_inverse_dynamics_torque_ratio"] is not None
    ]
    return {
        "cases": len(rows),
        "expected_matches": sum(bool(row["expected_match"]) for row in rows),
        "accepted_plans": sum(row["status"] == "accepted" for row in rows),
        "verified_brakes": sum(
            row["status"] == "verified_brake" for row in rows
        ),
        "holds": sum(row["status"] == "hold" for row in rows),
        "unrecoverable_stops": sum(
            row["status"] == "unrecoverable_stop" for row in rows
        ),
        "fallback_covered_cases": sum(
            bool(row["braking_invariant_complete"]) for row in rows
        ),
        "qp_intervention_steps": sum(
            int(row["qp_intervention_steps"]) for row in rows
        ),
        "continuous_edges_checked": sum(
            int(row["continuous_edges_checked"]) for row in rows
        ),
        "conservative_rejections": sum(
            bool(row["conservative_rejection"]) for row in rows
        ),
        "mean_accepted_task_progress_rad": (
            float(np.mean(accepted_progress)) if accepted_progress else 0.0
        ),
        "p95_supervision_latency_ms": float(np.percentile(latencies, 95)),
        "maximum_supervision_latency_ms": float(np.max(latencies)),
        "maximum_inverse_dynamics_torque_ratio": (
            max(torque_ratios) if torque_ratios else None
        ),
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]
    by_fault = summary["by_fault"]
    if not isinstance(overall, Mapping) or not isinstance(by_fault, Mapping):
        raise ValueError("integrated matrix summary is invalid")
    lines = [
        "# Integrated Panda supervisor fault matrix",
        "",
        f"Registered cases: {overall['cases']}",
        "",
        f"Expected outcomes matched: {overall['expected_matches']}/{overall['cases']}",
        "",
        f"Accepted plans: {overall['accepted_plans']}",
        "",
        f"Verified braking fallbacks: {overall['verified_brakes']}",
        "",
        f"Fail-closed holds: {overall['holds']}",
        "",
        f"Unrecoverable stop states: {overall['unrecoverable_stops']}",
        "",
        "| Fault | Cases | Accepted | Verified brake | Hold | Unrecoverable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fault in sorted(by_fault):
        aggregate = by_fault[fault]
        lines.append(
            f"| {fault} | {aggregate['cases']} | {aggregate['accepted_plans']} | "
            f"{aggregate['verified_brakes']} | {aggregate['holds']} | "
            f"{aggregate['unrecoverable_stops']} |"
        )
    lines.extend(
        [
            "",
            "The matrix uses scripted actions and synchronous CPU supervision. ",
            "It is not learned-policy, hard-real-time, closed-loop physics, or ",
            "physical-robot safety evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_manifest(root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": f"{MATRIX_SCHEMA}.manifest",
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json(files)),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _implementation_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "armbench/vla/integrated_panda_matrix.py": Path(__file__),
        "armbench/vla/integrated_panda_guard.py": Path(__file__).with_name(
            "integrated_panda_guard.py"
        ),
        "armbench/vla/qp_projection.py": Path(__file__).with_name(
            "qp_projection.py"
        ),
        "armbench/mujoco_sim/continuous_collision.py": root
        / "mujoco_sim"
        / "continuous_collision.py",
        "armbench/mujoco_sim/dynamics_braking.py": root
        / "mujoco_sim"
        / "dynamics_braking.py",
    }


def run_integrated_panda_fault_matrix(
    output_directory: Path,
    config: IntegratedPandaMatrixConfig = IntegratedPandaMatrixConfig(),
) -> Path:
    """Execute the registered matrix and write a self-validating artifact."""

    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"integrated matrix output already exists: {output}")
    output.mkdir(parents=True)
    specs = _case_specs(config)
    rows = [_evaluate_case(spec, config) for spec in specs]
    with (output / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "cases.json",
        {"schema_version": MATRIX_SCHEMA, "cases": specs},
    )
    faults = tuple(dict.fromkeys(str(row["fault"]) for row in rows))
    summary = {
        "schema_version": MATRIX_SCHEMA,
        "scope": MATRIX_SCOPE,
        "configuration": asdict(config),
        "environment": {
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "osqp_version": osqp.__version__,
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "panda_scene_sha256": sha256_file(default_panda_scene_path()),
        "implementation_sha256": {
            label: sha256_file(path) for label, path in _implementation_paths().items()
        },
        "overall": _aggregate(rows),
        "by_fault": {
            fault: _aggregate([row for row in rows if row["fault"] == fault])
            for fault in faults
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    _write_manifest(output)
    validate_integrated_panda_fault_matrix(output)
    return output


def _validate_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    expected_files = {"cases.json", "per_case.csv", "summary.json", "summary.md"}
    if not (
        has_exact_fields(manifest, _MANIFEST_FIELDS)
        and manifest["schema_version"] == f"{MATRIX_SCHEMA}.manifest"
        and isinstance(manifest["files"], list)
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"]
        == sha256_bytes(canonical_json(manifest["files"]))
    ):
        raise ValueError("integrated matrix manifest is invalid")
    declared = set()
    for item in manifest["files"]:
        if not (
            has_exact_fields(item, _MANIFEST_ENTRY_FIELDS)
            and isinstance(item["path"], str)
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0
            and is_sha256(item["sha256"])
        ):
            raise ValueError("integrated matrix manifest entry is invalid")
        relative = Path(item["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"integrated matrix manifest mismatch: {relative}")
        declared.add(item["path"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if declared != expected_files or actual != expected_files:
        raise ValueError("integrated matrix artifact file set is invalid")
    return str(manifest["inventory_sha256"])


def _load_config(value: object) -> IntegratedPandaMatrixConfig:
    if not isinstance(value, Mapping):
        raise ValueError("integrated matrix configuration is invalid")
    try:
        config = IntegratedPandaMatrixConfig(
            scenarios=tuple(value["scenarios"]),
            payload_masses_kg=tuple(value["payload_masses_kg"]),
            faults=tuple(value["faults"]),
            control_dt_s=value["control_dt_s"],
            response_deadline_ms=value["response_deadline_ms"],
            supervision_budget_ms=value["supervision_budget_ms"],
            qp_step_budget_ms=value["qp_step_budget_ms"],
            max_state_mismatch_rad=value["max_state_mismatch_rad"],
            nominal_horizon=value["nominal_horizon"],
            intermediate_self_collision_horizon=value[
                "intermediate_self_collision_horizon"
            ],
            include_special_cases=value["include_special_cases"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("integrated matrix configuration is invalid") from error
    if canonical_json(value) != canonical_json(asdict(config)):
        raise ValueError("integrated matrix configuration is not canonical")
    return config


def _parse_optional_bool(value: str) -> bool | None:
    if value == "":
        return None
    if value not in {"True", "False"}:
        raise ValueError("integrated matrix optional boolean is invalid")
    return value == "True"


def _parse_row(raw: Mapping[str, str]) -> dict[str, object]:
    if set(raw) != set(CSV_FIELDS) or raw["schema_version"] != MATRIX_SCHEMA:
        raise ValueError("integrated matrix row fields are invalid")
    boolean_fields = (
        "expected_match",
        "policy_actions_executable",
        "conservative_rejection",
        "braking_invariant_complete",
    )
    if any(raw[field] not in {"True", "False"} for field in boolean_fields):
        raise ValueError("integrated matrix row boolean is invalid")
    optional_float_fields = ("max_inverse_dynamics_torque_ratio",)
    float_fields = (
        "payload_mass_kg",
        "goal_distance_before_rad",
        "goal_distance_after_rad",
        "task_progress_rad",
        "total_latency_ms",
        "qp_latency_ms",
        "continuous_collision_latency_ms",
        "dynamics_braking_latency_ms",
        "fallback_brake_latency_ms",
    )
    integer_fields = (
        "policy_action_count",
        "qp_intervention_steps",
        "qp_fallback_steps",
        "continuous_edges_checked",
        "continuous_pair_evaluations",
        "braking_boundaries_checked",
    )
    try:
        row: dict[str, object] = dict(raw)
        for field in boolean_fields:
            row[field] = raw[field] == "True"
        row["qp_feasible"] = _parse_optional_bool(raw["qp_feasible"])
        row["fallback_validated"] = _parse_optional_bool(
            raw["fallback_validated"]
        )
        row["failure_index"] = (
            None if raw["failure_index"] == "" else int(raw["failure_index"])
        )
        for field in float_fields:
            row[field] = float(raw[field])
        for field in optional_float_fields:
            row[field] = None if raw[field] == "" else float(raw[field])
        for field in integer_fields:
            row[field] = int(raw[field])
    except (TypeError, ValueError) as error:
        raise ValueError("integrated matrix row value is invalid") from error
    numeric = [float(row[field]) for field in float_fields]
    numeric.extend(float(row[field]) for field in integer_fields)
    if (
        not np.all(np.isfinite(numeric))
        or any(float(row[field]) < 0.0 for field in float_fields if field != "task_progress_rad")
        or any(int(row[field]) < 0 for field in integer_fields)
        or row["failure_index"] is not None
        and int(row["failure_index"]) < 0
    ):
        raise ValueError("integrated matrix row numeric contract is invalid")
    ratio = row["max_inverse_dynamics_torque_ratio"]
    if ratio is not None and (not np.isfinite(ratio) or float(ratio) < 0.0):
        raise ValueError("integrated matrix torque ratio is invalid")
    return row


def _same_deterministic_row(
    stored: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    ignored = {
        "total_latency_ms",
        "qp_latency_ms",
        "continuous_collision_latency_ms",
        "dynamics_braking_latency_ms",
        "fallback_brake_latency_ms",
        "max_inverse_dynamics_torque_ratio",
        "goal_distance_before_rad",
        "goal_distance_after_rad",
        "task_progress_rad",
    }
    if any(
        stored[field] != expected[field]
        for field in CSV_FIELDS
        if field not in ignored
    ):
        return False
    for field in (
        "goal_distance_before_rad",
        "goal_distance_after_rad",
        "task_progress_rad",
    ):
        if not np.isclose(
            float(stored[field]), float(expected[field]), rtol=1e-12, atol=1e-12
        ):
            return False
    left = stored["max_inverse_dynamics_torque_ratio"]
    right = expected["max_inverse_dynamics_torque_ratio"]
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12)
    )


def _normalize_generated_row(row: Mapping[str, object]) -> dict[str, object]:
    """Apply the same scalar encoding used by ``csv.DictWriter`` and parser."""

    encoded = {
        field: "" if row[field] is None else str(row[field]) for field in CSV_FIELDS
    }
    return _parse_row(encoded)


def _validate_aggregate(actual: object, expected: Mapping[str, object]) -> None:
    if not has_exact_fields(actual, _AGGREGATE_FIELDS):
        raise ValueError("integrated matrix aggregate fields are invalid")
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("integrated matrix aggregate mismatch")


def validate_integrated_panda_fault_matrix(directory: Path) -> dict[str, object]:
    """Verify hashes, rebuild registered inputs, rerun cases, and aggregate."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"integrated matrix directory not found: {root}")
    inventory_hash = _validate_manifest(root)
    summary = strict_json_load(root / "summary.json")
    if not (
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == MATRIX_SCHEMA
        and summary["scope"] == MATRIX_SCOPE
        and summary["panda_scene_sha256"] == sha256_file(default_panda_scene_path())
        and canonical_json(summary["claim_boundary"]) == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("integrated matrix summary is invalid")
    config = _load_config(summary["configuration"])
    environment = summary["environment"]
    if not (
        has_exact_fields(environment, _ENVIRONMENT_FIELDS)
        and environment["mujoco_version"] == mujoco.__version__
        and environment["numpy_version"] == np.__version__
        and environment["osqp_version"] == osqp.__version__
        and environment["menagerie_commit"] == MENAGERIE_COMMIT
    ):
        raise ValueError("integrated matrix environment mismatch")
    expected_paths = _implementation_paths()
    hashes = summary["implementation_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(expected_paths):
        raise ValueError("integrated matrix implementation hashes are invalid")
    if any(hashes[label] != sha256_file(path) for label, path in expected_paths.items()):
        raise ValueError("integrated matrix implementation hash mismatch")

    case_document = strict_json_load(root / "cases.json")
    expected_specs = _case_specs(config)
    if not (
        has_exact_fields(case_document, _CASE_DOCUMENT_FIELDS)
        and case_document["schema_version"] == MATRIX_SCHEMA
        and isinstance(case_document["cases"], list)
        and all(has_exact_fields(item, _CASE_FIELDS) for item in case_document["cases"])
        and canonical_json(case_document["cases"]) == canonical_json(expected_specs)
    ):
        raise ValueError("integrated matrix registered cases are invalid")

    with (root / "per_case.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("integrated matrix CSV fields are invalid")
        stored_rows = [_parse_row(row) for row in reader]
    if len(stored_rows) != len(expected_specs):
        raise ValueError("integrated matrix case count is invalid")
    if len({str(row["case_id"]) for row in stored_rows}) != len(stored_rows):
        raise ValueError("integrated matrix case identities are not unique")

    replayed = [
        _normalize_generated_row(_evaluate_case(spec, config))
        for spec in expected_specs
    ]
    for stored, expected in zip(stored_rows, replayed):
        if not _same_deterministic_row(stored, expected):
            raise ValueError(f"integrated matrix recomputation failed: {stored['case_id']}")
    if not all(bool(row["expected_match"]) for row in stored_rows):
        raise ValueError("integrated matrix registered outcome mismatch")

    overall = _aggregate(stored_rows)
    _validate_aggregate(summary["overall"], overall)
    fault_order = tuple(dict.fromkeys(str(row["fault"]) for row in stored_rows))
    by_fault = summary["by_fault"]
    if not isinstance(by_fault, Mapping) or tuple(by_fault) != tuple(sorted(fault_order)):
        raise ValueError("integrated matrix fault aggregate keys are invalid")
    for fault in fault_order:
        subset = [row for row in stored_rows if row["fault"] == fault]
        _validate_aggregate(by_fault[fault], _aggregate(subset))
    if (root / "summary.md").read_text("utf-8") != _summary_markdown(summary):
        raise ValueError("integrated matrix Markdown summary is not reproducible")
    return {
        "valid": True,
        "scope": MATRIX_SCOPE,
        "cases": len(stored_rows),
        "expected_matches": overall["expected_matches"],
        "accepted_plans": overall["accepted_plans"],
        "verified_brakes": overall["verified_brakes"],
        "holds": overall["holds"],
        "unrecoverable_stops": overall["unrecoverable_stops"],
        "manifest_inventory_sha256": inventory_hash,
        "checks": [
            "recursive_manifest",
            "registered_inputs_rebuilt",
            "all_supervisor_decisions_rerun",
            "fault_aggregates_recomputed",
            "no_partial_policy_action_on_rejection",
        ],
    }


__all__ = [
    "IntegratedPandaMatrixConfig",
    "run_integrated_panda_fault_matrix",
    "validate_integrated_panda_fault_matrix",
]
