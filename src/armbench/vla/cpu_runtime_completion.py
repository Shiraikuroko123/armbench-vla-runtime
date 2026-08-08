"""Recomputable CPU completion matrix for the provider-to-Panda boundary.

This artifact is intentionally narrower than a VLA benchmark.  It exercises
the same asynchronous policy mailbox, the integrated QP/collision/braking
supervisor, and the reset-generation gate with mock, frozen, and
provider-compatible contract fixtures.  It records malformed, non-finite,
disconnect, stale, state-mismatch, budget, and replay failures in one table.

The contract fixture is not a downloaded checkpoint.  A real OpenPI provider is
still a GPU/server milestone; this matrix proves that its provider interface
can enter the same runtime without making a false end-to-end claim.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.async_worker import LatestPolicyWorker, PolicyOutcome
from armbench.vla.cartesian_adapter import PandaCartesianActionAdapter
from armbench.vla.integrated_panda_async import (
    AtomicPandaPlanGate,
    LatestIntegratedPandaWorker,
)
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
)
from armbench.vla.integrated_panda_matrix import (
    _SELF_END,
    _SELF_START,
)
from armbench.vla.integrated_panda_task import make_integrated_task_checker
from armbench.vla.provider_contract import (
    ActionSemantics,
    AdaptedActionChunkPolicy,
    FrozenResponseProvider,
    FrozenResponseRecord,
    ProviderIdentity,
    libero_cartesian_semantics,
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
from armbench.vla.types import ActionChunk, VLAObservation


FloatArray = np.ndarray
CPU_RUNTIME_SCHEMA = "armbench.cpu_runtime_completion.v1"
CPU_RUNTIME_MANIFEST_SCHEMA = "armbench.cpu_runtime_completion_manifest.v1"
CPU_RUNTIME_SCOPE = "provider_neutral_async_integrated_panda_cpu_reference"
CPU_RUNTIME_CSV_FIELDS = (
    "schema_version",
    "case_id",
    "provider",
    "fault",
    "expected_status",
    "expected_reason_prefix",
    "policy_succeeded",
    "policy_failure_type",
    "policy_worker_latency_ms",
    "policy_control_ticks",
    "policy_worker_thread_separate",
    "assurance_control_ticks",
    "assurance_worker_latency_ms",
    "assurance_worker_thread_separate",
    "status",
    "reason",
    "supervisor_status",
    "response_age_ms",
    "policy_action_count",
    "policy_actions_executable",
    "partial_prefix_exposed",
    "fallback_validated",
    "expected_match",
    "case_elapsed_ms",
)
_ARTIFACT_FILES = {
    "cases.json",
    "per_case.csv",
    "provenance.json",
    "summary.json",
    "summary.md",
}
_MANIFEST_FIELDS = {"schema_version", "files", "inventory_sha256"}
_MANIFEST_ENTRY_FIELDS = {"path", "size_bytes", "sha256"}
_CASE_FIELDS = {
    "case_id",
    "provider",
    "fault",
    "latency_ms",
    "expected_status",
    "expected_reason_prefix",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "scope",
    "overall",
    "by_provider",
    "claim_boundary",
    "configuration",
    "rows",
}
_OVERALL_FIELDS = {
    "cases",
    "expected_matches",
    "accepted_plans",
    "holds",
    "verified_brakes",
    "unrecoverable_stops",
    "partial_prefix_exposed",
    "separate_policy_workers",
    "separate_assurance_workers",
    "p95_assurance_worker_latency_ms",
    "max_assurance_worker_latency_ms",
}
_PROVIDER_AGGREGATE_FIELDS = {"cases", "expected_matches", "accepted_plans"}
_CONFIGURATION_FIELDS = {
    "response_deadline_ms",
    "supervision_budget_ms",
    "qp_step_budget_ms",
    "control_period_ms",
}
_PROVENANCE_FIELDS = {
    "schema_version",
    "provider_modes",
    "implementation_files",
    "implementation_sha256",
    "claim_boundary",
}
_PROVIDER_MODES = {
    "mock_hx8": "direct scripted Hx8 test policy",
    "frozen_hx7_adapter": (
        "validated frozen Hx7 provider plus Panda Jacobian adapter"
    ),
    "openpi_interface_fixture": (
        "OpenPI-compatible contract fixture; checkpoint not loaded"
    ),
}
_CLAIM_BOUNDARY = [
    (
        "Policy sources are scripted, frozen, or provider-compatible fixtures; "
        "no learned checkpoint is executed here."
    ),
    (
        "Supervisor and assurance workers run on CPU and use best-effort "
        "Python scheduling."
    ),
    (
        "A rejected result exposes zero policy actions; a hardware actuator "
        "must consume the separate brake certificate."
    ),
    (
        "The interface fixture does not establish pi0.5 task success or "
        "real-robot safety."
    ),
]
_IMPLEMENTATION_RELATIVE_PATHS = (
    "armbench/vla/cpu_runtime_completion.py",
    "armbench/vla/integrated_panda_async.py",
    "armbench/vla/integrated_panda_guard.py",
    "armbench/vla/integrated_panda_task.py",
    "armbench/vla/async_worker.py",
    "armbench/vla/provider_contract.py",
    "armbench/vla/cartesian_adapter.py",
    "armbench/vla/qp_projection.py",
    "armbench/mujoco_sim/continuous_collision.py",
    "armbench/mujoco_sim/dynamics_braking.py",
)


def _image(value: int = 0) -> np.ndarray:
    return np.full((224, 224, 3), value, dtype=np.uint8)


def _observation(
    q: np.ndarray,
    *,
    sequence_id: int = 0,
    captured_at_s: float | None = None,
) -> VLAObservation:
    return VLAObservation(
        exterior_image=_image(32),
        wrist_image=_image(64),
        joint_position=q,
        gripper_position=np.array([1.0]),
        prompt="move the Panda gripper to the registered goal",
        sequence_id=sequence_id,
        captured_at_s=(time.monotonic() if captured_at_s is None else captured_at_s),
    )


def _joint_chunk(
    observation: VLAObservation,
    *,
    horizon: int = 2,
    velocity: float = 0.02,
) -> ActionChunk:
    actions = np.zeros((horizon, 8), dtype=float)
    actions[:, 0] = velocity
    actions[:, 7] = float(observation.gripper_position[0])
    return ActionChunk(
        actions=actions,
        source="cpu_completion_mock_policy",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=0.0,
    )


class _MockPolicy:
    """Small Hx8 policy used to exercise the runtime without a checkpoint."""

    def __init__(self, *, latency_ms: float = 0.0) -> None:
        self.latency_ms = float(latency_ms)

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started = time.monotonic()
        if self.latency_ms > 0.0:
            time.sleep(self.latency_ms / 1000.0)
        received = time.monotonic()
        chunk = _joint_chunk(observation)
        return ActionChunk(
            actions=chunk.actions,
            source="cpu_completion_mock_policy",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received - started) * 1000.0,
            received_at_s=received,
        )


class _FaultPolicy:
    """Deliberately broken policy calls for the fail-closed rows."""

    def __init__(self, fault: str) -> None:
        self.fault = fault

    def infer(self, observation: VLAObservation) -> object:
        if self.fault == "malformed_shape":
            return {"actions": [[0.0] * 7]}
        if self.fault == "nonfinite":
            return ActionChunk(
                actions=np.full((2, 8), np.nan),
                source="invalid_nonfinite_policy",
                observation_sequence_id=observation.sequence_id,
                inference_latency_ms=0.0,
            )
        if self.fault == "disconnect":
            raise ConnectionError("provider socket disconnected")
        if self.fault == "timeout":
            raise TimeoutError("provider response deadline elapsed")
        if self.fault == "sequence_mismatch":
            chunk = _joint_chunk(observation)
            return ActionChunk(
                actions=chunk.actions,
                source="invalid_sequence_policy",
                observation_sequence_id=observation.sequence_id + 1,
                inference_latency_ms=0.0,
            )
        raise ValueError(f"unknown fault policy: {self.fault}")


class _IntermediateCollisionPolicy:
    """Drive the registered self-collision edge through the real supervisor."""

    def infer(self, observation: VLAObservation) -> ActionChunk:
        horizon = 120
        actions = np.zeros((horizon, 8), dtype=float)
        actions[:, :7] = (_SELF_END - _SELF_START) / (horizon * 0.05)
        actions[:, 7] = 1.0
        return ActionChunk(
            actions=actions,
            source="cpu_completion_intermediate_self_collision",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=0.0,
        )


class _ContractFixtureProvider:
    """Provider-compatible Hx7 source with explicit non-checkpoint identity."""

    def __init__(self, semantics: ActionSemantics) -> None:
        self._semantics = semantics
        self._identity = ProviderIdentity(
            provider_id="openpi_interface_contract_fixture",
            model_family="OpenPI-compatible provider",
            implementation_repository="https://github.com/Physical-Intelligence/openpi",
            implementation_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
            checkpoint_reference=None,
            checkpoint_sha256=None,
            checkpoint_identity_status="not_applicable",
            response_origin="synthetic_contract_fixture",
            checkpoint_executed_during_capture=False,
            checkpoint_executed_this_run=False,
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    @property
    def semantics(self) -> ActionSemantics:
        return self._semantics

    def infer_raw(self, observation: VLAObservation) -> Any:
        actions = np.zeros((2, 7), dtype=float)
        actions[:, 6] = -1.0
        from armbench.vla.provider_contract import (
            RawActionChunk,
            canonical_action_sha256,
        )

        return RawActionChunk(
            actions=actions,
            semantics=self.semantics,
            source="openpi_interface_contract_fixture",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=0.0,
            received_at_s=time.monotonic(),
            response_sha256=canonical_action_sha256(actions),
        )


def _frozen_policy(robot: MuJoCoPanda) -> AdaptedActionChunkPolicy:
    semantics = libero_cartesian_semantics()
    identity = ProviderIdentity(
        provider_id="frozen_cpu_contract_fixture",
        model_family="Frozen provider fixture",
        implementation_repository="https://github.com/Physical-Intelligence/openpi",
        implementation_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
        checkpoint_reference=None,
        checkpoint_sha256=None,
        checkpoint_identity_status="not_applicable",
        response_origin="synthetic_contract_fixture",
        checkpoint_executed_during_capture=False,
        checkpoint_executed_this_run=False,
    )
    provider = FrozenResponseProvider(
        identity,
        semantics,
        (
            FrozenResponseRecord(
                observation_sequence_id=0,
                actions=np.column_stack(
                    (np.zeros((2, 6), dtype=float), -np.ones(2))
                ),
                inference_latency_ms=0.0,
            ),
        ),
    )
    return AdaptedActionChunkPolicy(provider, PandaCartesianActionAdapter(robot))


def _contract_fixture_policy(robot: MuJoCoPanda) -> AdaptedActionChunkPolicy:
    return AdaptedActionChunkPolicy(
        _ContractFixtureProvider(libero_cartesian_semantics()),
        PandaCartesianActionAdapter(robot),
    )


@dataclass(frozen=True)
class CPURuntimeMatrixConfig:
    """Stable controls for the local completion matrix."""

    response_deadline_ms: float = 20_000.0
    supervision_budget_ms: float = 20_000.0
    qp_step_budget_ms: float = 500.0
    control_period_ms: float = 1.0

    def __post_init__(self) -> None:
        raw_values = (
            self.response_deadline_ms,
            self.supervision_budget_ms,
            self.qp_step_budget_ms,
            self.control_period_ms,
        )
        if any(type(value) not in {int, float} for value in raw_values):
            raise ValueError("CPU runtime matrix timing must be numeric")
        values = np.asarray(
            raw_values,
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("CPU runtime matrix timing is invalid")


def _case_specs() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for latency in (0.0, 40.0, 80.0, 160.0):
        cases.append(
            {
                "case_id": f"mock_latency_{int(latency):03d}ms",
                "provider": "mock_hx8",
                "fault": "none",
                "latency_ms": latency,
                "expected_status": "execute",
                "expected_reason_prefix": "qp_continuous_collision_and_braking",
            }
        )
    cases.extend(
        [
            {
                "case_id": "frozen_provider_nominal",
                "provider": "frozen_hx7_adapter",
                "fault": "none",
                "latency_ms": 0.0,
                "expected_status": "execute",
                "expected_reason_prefix": "qp_continuous_collision_and_braking",
            },
            {
                "case_id": "provider_interface_fixture_nominal",
                "provider": "openpi_interface_fixture",
                "fault": "none",
                "latency_ms": 0.0,
                "expected_status": "execute",
                "expected_reason_prefix": "qp_continuous_collision_and_braking",
            },
        ]
    )
    for fault in ("malformed_shape", "nonfinite", "disconnect", "timeout", "sequence_mismatch"):
        cases.append(
            {
                "case_id": f"provider_fault_{fault}",
                "provider": "fault_injector",
                "fault": fault,
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "provider_failure",
            }
        )
    cases.extend(
        [
            {
                "case_id": "stale_observation",
                "provider": "mock_hx8",
                "fault": "stale",
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "response_deadline_exceeded_before_supervision",
            },
            {
                "case_id": "state_mismatch",
                "provider": "mock_hx8",
                "fault": "state_mismatch",
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "observation_state_mismatch",
            },
            {
                "case_id": "reset_replay",
                "provider": "mock_hx8",
                "fault": "reset_before_commit",
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "reset_generation_mismatch",
            },
            {
                "case_id": "supervision_budget_fail_closed",
                "provider": "mock_hx8",
                "fault": "tiny_supervision_budget",
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "supervision_budget_exceeded",
            },
            {
                "case_id": "near_limit_unrecoverable_stop",
                "provider": "mock_hx8",
                "fault": "near_limit_stop",
                "latency_ms": 0.0,
                "expected_status": "unrecoverable_stop",
                "expected_reason_prefix": "response_deadline_exceeded_before_supervision",
            },
            {
                "case_id": "intermediate_collision_fail_closed",
                "provider": "mock_hx8",
                "fault": "intermediate_self_collision",
                "latency_ms": 0.0,
                "expected_status": "hold",
                "expected_reason_prefix": "continuous_edge_rejected",
            },
        ]
    )
    return cases


def _supervisor_for_case(
    config: CPURuntimeMatrixConfig,
    *,
    tiny_budget: bool = False,
) -> tuple[IntegratedPandaSupervisor, np.ndarray, np.ndarray, np.ndarray]:
    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=())
    checker, _ = make_integrated_task_checker(robot)
    supervisor_config = IntegratedPandaGuardConfig(
        response_deadline_ms=config.response_deadline_ms,
        supervision_budget_ms=(0.0001 if tiny_budget else config.supervision_budget_ms),
        qp_step_budget_ms=config.qp_step_budget_ms,
    )
    return (
        IntegratedPandaSupervisor(robot, checker, supervisor_config),
        scenario.start.copy(),
        np.zeros(7, dtype=float),
        scenario.goal.copy(),
    )


def _policy_for_case(
    case: Mapping[str, object],
    robot: MuJoCoPanda,
    observation: VLAObservation,
) -> object:
    provider = str(case["provider"])
    fault = str(case["fault"])
    if provider == "frozen_hx7_adapter":
        return _frozen_policy(robot)
    if provider == "openpi_interface_fixture":
        return _contract_fixture_policy(robot)
    if provider == "fault_injector":
        return _FaultPolicy(fault)
    if fault == "intermediate_self_collision":
        return _IntermediateCollisionPolicy()
    return _MockPolicy(latency_ms=float(case["latency_ms"]))


def _wait_policy(
    worker: LatestPolicyWorker,
    *,
    control_period_s: float,
    timeout_s: float = 30.0,
) -> tuple[PolicyOutcome, int]:
    deadline = time.monotonic() + timeout_s
    ticks = 0
    while time.monotonic() < deadline:
        outcomes = worker.drain()
        if outcomes:
            return outcomes[0], ticks
        ticks += 1
        time.sleep(control_period_s)
    raise TimeoutError("CPU runtime policy worker timed out")


def _wait_assurance(
    worker: LatestIntegratedPandaWorker,
    *,
    control_period_s: float,
    timeout_s: float = 45.0,
) -> tuple[object, int]:
    deadline = time.monotonic() + timeout_s
    ticks = 0
    while time.monotonic() < deadline:
        outcomes = worker.drain()
        if outcomes:
            return outcomes[0], ticks
        ticks += 1
        time.sleep(control_period_s)
    raise TimeoutError("CPU runtime assurance worker timed out")


def _run_case(
    case: Mapping[str, object], config: CPURuntimeMatrixConfig
) -> dict[str, object]:
    fault = str(case["fault"])
    supervisor, q_start, qvel, _ = _supervisor_for_case(
        config, tiny_budget=fault == "tiny_supervision_budget"
    )
    if fault == "intermediate_self_collision":
        q_start = _SELF_START.copy()
    if fault == "near_limit_stop":
        q_start = supervisor.robot.upper_limits - 0.001
        qvel[0] = 0.2
    observed_q = q_start.copy()
    if fault == "state_mismatch":
        observed_q[0] += supervisor.config.max_state_mismatch_rad + 0.01
    captured_at_s = time.monotonic()
    if fault in {"stale", "near_limit_stop"}:
        captured_at_s -= config.response_deadline_ms / 1000.0 + 1.0
    observation = _observation(observed_q, captured_at_s=captured_at_s)

    # The policy adapter owns its own kinematic model so provider conversion is
    # independent of the supervisor's MuJoCo data object.
    policy_robot = MuJoCoPanda.create(obstacles=())
    policy = _policy_for_case(case, policy_robot, observation)

    # Stage 1 is deliberately asynchronous: a slow or failing provider must
    # never execute on the thread that advances the controller clock.
    policy_worker = LatestPolicyWorker(policy)
    policy_started = time.monotonic()
    try:
        policy_worker.submit(observation)
        policy_outcome, policy_ticks = _wait_policy(
            policy_worker,
            control_period_s=config.control_period_ms / 1000.0,
        )
    finally:
        policy_worker.close(timeout_s=5.0)

    row: dict[str, object] = {
        "schema_version": CPU_RUNTIME_SCHEMA,
        "case_id": case["case_id"],
        "provider": case["provider"],
        "fault": fault,
        "expected_status": case["expected_status"],
        "expected_reason_prefix": case["expected_reason_prefix"],
        "policy_succeeded": policy_outcome.succeeded,
        "policy_failure_type": policy_outcome.failure_type or "",
        "policy_worker_latency_ms": policy_outcome.worker_latency_ms,
        "policy_control_ticks": policy_ticks,
        "policy_worker_thread_separate": (
            policy_outcome.worker_thread_id != threading.get_ident()
        ),
        "assurance_control_ticks": 0,
        "assurance_worker_latency_ms": 0.0,
        "assurance_worker_thread_separate": False,
        "status": "hold",
        "reason": "provider_failure:unknown",
        "supervisor_status": "",
        "response_age_ms": max(
            0.0,
            (policy_outcome.finished_at_s - observation.captured_at_s) * 1000.0,
        ),
        "policy_action_count": 0,
        "policy_actions_executable": False,
        "partial_prefix_exposed": False,
        "fallback_validated": False,
    }

    if not policy_outcome.succeeded or policy_outcome.chunk is None:
        row["reason"] = f"provider_failure:{policy_outcome.failure_type}"
        row["policy_failure_type"] = policy_outcome.failure_type or "unknown"
    else:
        # Stage 2 owns the expensive QP, swept-collision, and braking checks on
        # a second worker. The gate is the only control-side publication point.
        assurance_worker = LatestIntegratedPandaWorker(supervisor)
        gate = AtomicPandaPlanGate(supervisor)
        try:
            assurance_worker.submit(
                generation=gate.generation,
                observation_sequence_id=observation.sequence_id,
                q=q_start,
                qvel=qvel,
                observed_q=observed_q,
                response_age_ms=row["response_age_ms"],
                chunk=policy_outcome.chunk,
            )
            assurance_outcome, assurance_ticks = _wait_assurance(
                assurance_worker,
                control_period_s=config.control_period_ms / 1000.0,
            )
            if fault == "reset_before_commit":
                gate.reset()
            # Commit rechecks deadline, robot state, ordering, and reset epoch.
            # Rejected chunks publish an empty Hx8 array, never a safe-looking
            # prefix from a plan whose later edge failed.
            atomic = gate.commit(assurance_outcome, q_now=q_start)
        finally:
            assurance_worker.close(timeout_s=45.0)
        row.update(
            {
                "assurance_control_ticks": assurance_ticks,
                "assurance_worker_latency_ms": assurance_outcome.worker_latency_ms,
                "assurance_worker_thread_separate": (
                    assurance_outcome.worker_thread_id
                    != threading.get_ident()
                ),
                "status": atomic.status,
                "reason": atomic.reason,
                "supervisor_status": atomic.supervisor_status or "",
                "response_age_ms": atomic.response_age_ms,
                "policy_action_count": len(atomic.policy_actions),
                "policy_actions_executable": atomic.policy_actions_executable,
                "partial_prefix_exposed": (
                    not atomic.policy_actions_executable
                    and len(atomic.policy_actions) != 0
                ),
                "fallback_validated": atomic.fallback_validated,
            }
        )

    row["expected_match"] = bool(
        row["status"] == case["expected_status"]
        and str(row["reason"]).startswith(str(case["expected_reason_prefix"]))
    )
    row["case_elapsed_ms"] = (time.monotonic() - policy_started) * 1000.0
    return row


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _implementation_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(source_root / Path(relative))
        for relative in _IMPLEMENTATION_RELATIVE_PATHS
    }


def _write_manifest(root: Path) -> None:
    files = _inventory(root)
    write_json(
        root / "manifest.json",
        {
            "schema_version": CPU_RUNTIME_MANIFEST_SCHEMA,
            "files": files,
            "inventory_sha256": sha256_bytes(canonical_json(files)),
        },
    )


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": CPU_RUNTIME_SCHEMA,
        "scope": CPU_RUNTIME_SCOPE,
        "overall": {
            "cases": len(rows),
            "expected_matches": sum(bool(row["expected_match"]) for row in rows),
            "accepted_plans": sum(row["status"] == "execute" for row in rows),
            "holds": sum(row["status"] == "hold" for row in rows),
            "verified_brakes": sum(
                row["status"] == "verified_brake" for row in rows
            ),
            "unrecoverable_stops": sum(
                row["status"] == "unrecoverable_stop" for row in rows
            ),
            "partial_prefix_exposed": sum(
                bool(row["partial_prefix_exposed"]) for row in rows
            ),
            "separate_policy_workers": sum(
                bool(row["policy_worker_thread_separate"]) for row in rows
            ),
            "separate_assurance_workers": sum(
                bool(row["assurance_worker_thread_separate"]) for row in rows
            ),
            "p95_assurance_worker_latency_ms": float(
                np.percentile(
                    [float(row["assurance_worker_latency_ms"]) for row in rows],
                    95,
                )
            ),
            "max_assurance_worker_latency_ms": max(
                float(row["assurance_worker_latency_ms"]) for row in rows
            ),
        },
        "by_provider": {
            provider: {
                "cases": sum(row["provider"] == provider for row in rows),
                "expected_matches": sum(
                    row["provider"] == provider and bool(row["expected_match"])
                    for row in rows
                ),
                "accepted_plans": sum(
                    row["provider"] == provider and row["status"] == "execute"
                    for row in rows
                ),
            }
            for provider in sorted({str(row["provider"]) for row in rows})
        },
        "claim_boundary": list(_CLAIM_BOUNDARY),
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]
    lines = [
        "# CPU runtime completion matrix",
        "",
        "Provider-neutral asynchronous assurance boundary; scripted/frozen/contract fixtures only.",
        "",
        f"- Cases: {overall['cases']}",
        f"- Expected outcomes: {overall['expected_matches']}/{overall['cases']}",
        f"- Complete plans published: {overall['accepted_plans']}",
        f"- Holds: {overall['holds']}",
        f"- Unrecoverable stops: {overall['unrecoverable_stops']}",
        f"- Partial policy prefixes exposed: {overall['partial_prefix_exposed']}",
        f"- Assurance worker P95: {overall['p95_assurance_worker_latency_ms']:.3f} ms",
        "",
        "| Case | Provider | Fault | Status | Reason | Match |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['case_id']} | {row['provider']} | {row['fault']} | "
            f"{row['status']} | {row['reason']} | {row['expected_match']} |"
        )
    return "\n".join(lines) + "\n"


def run_cpu_runtime_completion(
    output_directory: Path,
    config: CPURuntimeMatrixConfig = CPURuntimeMatrixConfig(),
) -> Path:
    """Run and save the registered CPU-only runtime boundary matrix."""

    output = output_directory.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"CPU runtime output already exists: {output}")
    else:
        output.mkdir(parents=True)
    specs = _case_specs()
    rows = [_run_case(spec, config) for spec in specs]
    summary = _summary(rows)
    summary["configuration"] = {
        "response_deadline_ms": config.response_deadline_ms,
        "supervision_budget_ms": config.supervision_budget_ms,
        "qp_step_budget_ms": config.qp_step_budget_ms,
        "control_period_ms": config.control_period_ms,
    }
    summary["rows"] = rows
    write_json(output / "cases.json", specs)
    write_json(output / "summary.json", summary)
    write_json(
        output / "provenance.json",
        {
            "schema_version": CPU_RUNTIME_SCHEMA,
            "provider_modes": dict(_PROVIDER_MODES),
            "implementation_files": list(_IMPLEMENTATION_RELATIVE_PATHS),
            "implementation_sha256": _implementation_hashes(),
            "claim_boundary": summary["claim_boundary"],
        },
    )
    with (output / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CPU_RUNTIME_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_for_markdown = dict(summary)
    summary_for_markdown["rows"] = rows
    (output / "summary.md").write_text(
        _summary_markdown(summary_for_markdown), encoding="utf-8"
    )
    _write_manifest(output)
    validate_cpu_runtime_completion(output)
    return output


def _validate_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != _ARTIFACT_FILES:
        raise ValueError("CPU runtime artifact file set is invalid")
    if any((root / relative).is_symlink() for relative in _ARTIFACT_FILES):
        raise ValueError("CPU runtime artifact cannot contain symbolic links")

    expected = _inventory(root)
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not (
        has_exact_fields(manifest, _MANIFEST_FIELDS)
        and manifest["schema_version"] == CPU_RUNTIME_MANIFEST_SCHEMA
        and isinstance(files, list)
        and all(has_exact_fields(item, _MANIFEST_ENTRY_FIELDS) for item in files)
        and all(isinstance(item["path"], str) for item in files)
        and all(type(item["size_bytes"]) is int for item in files)
        and all(item["size_bytes"] >= 0 for item in files)
        and all(is_sha256(item["sha256"]) for item in files)
        and files == expected
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"] == sha256_bytes(canonical_json(expected))
    ):
        raise ValueError("CPU runtime manifest mismatch")
    return str(manifest["inventory_sha256"])


def _load_config(value: object) -> CPURuntimeMatrixConfig:
    if not has_exact_fields(value, _CONFIGURATION_FIELDS):
        raise ValueError("CPU runtime configuration is invalid")
    try:
        config = CPURuntimeMatrixConfig(
            response_deadline_ms=value["response_deadline_ms"],
            supervision_budget_ms=value["supervision_budget_ms"],
            qp_step_budget_ms=value["qp_step_budget_ms"],
            control_period_ms=value["control_period_ms"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("CPU runtime configuration is invalid") from error
    if canonical_json(value) != canonical_json(asdict(config)):
        raise ValueError("CPU runtime configuration is not canonical")
    return config


_ROW_BOOLEAN_FIELDS = (
    "policy_succeeded",
    "policy_worker_thread_separate",
    "assurance_worker_thread_separate",
    "policy_actions_executable",
    "partial_prefix_exposed",
    "fallback_validated",
    "expected_match",
)
_ROW_INTEGER_FIELDS = (
    "policy_control_ticks",
    "assurance_control_ticks",
    "policy_action_count",
)
_ROW_FLOAT_FIELDS = (
    "policy_worker_latency_ms",
    "assurance_worker_latency_ms",
    "response_age_ms",
    "case_elapsed_ms",
)


def _validate_row_values(value: object) -> dict[str, object]:
    if not has_exact_fields(value, set(CPU_RUNTIME_CSV_FIELDS)):
        raise ValueError("CPU runtime row fields are invalid")
    row = dict(value)
    if any(type(row[field]) is not bool for field in _ROW_BOOLEAN_FIELDS):
        raise ValueError("CPU runtime row boolean is invalid")
    if any(type(row[field]) is not int for field in _ROW_INTEGER_FIELDS):
        raise ValueError("CPU runtime row integer is invalid")
    if any(
        type(row[field]) not in {int, float} for field in _ROW_FLOAT_FIELDS
    ):
        raise ValueError("CPU runtime row float is invalid")
    numeric = [float(row[field]) for field in _ROW_FLOAT_FIELDS]
    numeric.extend(float(row[field]) for field in _ROW_INTEGER_FIELDS)
    if not np.all(np.isfinite(numeric)) or any(item < 0.0 for item in numeric):
        raise ValueError("CPU runtime row numeric contract is invalid")

    string_fields = set(CPU_RUNTIME_CSV_FIELDS).difference(
        _ROW_BOOLEAN_FIELDS + _ROW_INTEGER_FIELDS + _ROW_FLOAT_FIELDS
    )
    if any(not isinstance(row[field], str) for field in string_fields):
        raise ValueError("CPU runtime row string is invalid")
    if (
        row["schema_version"] != CPU_RUNTIME_SCHEMA
        or not row["case_id"]
        or not row["reason"]
        or row["status"]
        not in {"execute", "verified_brake", "hold", "unrecoverable_stop"}
        or row["expected_status"]
        not in {"execute", "verified_brake", "hold", "unrecoverable_stop"}
        or row["supervisor_status"]
        not in {"", "accepted", "verified_brake", "hold", "unrecoverable_stop"}
    ):
        raise ValueError("CPU runtime row status contract is invalid")
    expected_match = bool(
        row["status"] == row["expected_status"]
        and str(row["reason"]).startswith(str(row["expected_reason_prefix"]))
    )
    if bool(row["expected_match"]) != expected_match:
        raise ValueError("CPU runtime row expected outcome flag is inconsistent")
    executable = bool(row["policy_actions_executable"])
    action_count = int(row["policy_action_count"])
    if (row["status"] == "execute") != executable:
        raise ValueError("CPU runtime row execution flag is inconsistent")
    if (executable and action_count <= 0) or (not executable and action_count != 0):
        raise ValueError("CPU runtime row action publication is inconsistent")
    if bool(row["partial_prefix_exposed"]):
        raise ValueError("CPU runtime exposed a partial policy prefix")
    if not bool(row["policy_worker_thread_separate"]):
        raise ValueError("CPU runtime policy work ran on the control thread")
    if bool(row["policy_succeeded"]) != bool(
        row["assurance_worker_thread_separate"]
    ):
        raise ValueError("CPU runtime assurance worker routing is inconsistent")
    return row


def _parse_csv_row(raw: Mapping[str, str]) -> dict[str, object]:
    if set(raw) != set(CPU_RUNTIME_CSV_FIELDS):
        raise ValueError("CPU runtime CSV row fields are invalid")
    row: dict[str, object] = dict(raw)
    try:
        for field in _ROW_BOOLEAN_FIELDS:
            if raw[field] not in {"True", "False"}:
                raise ValueError("CPU runtime CSV boolean is invalid")
            row[field] = raw[field] == "True"
        for field in _ROW_INTEGER_FIELDS:
            row[field] = int(raw[field])
        for field in _ROW_FLOAT_FIELDS:
            row[field] = float(raw[field])
    except (TypeError, ValueError) as error:
        raise ValueError("CPU runtime CSV value is invalid") from error
    return _validate_row_values(row)


def _validate_provenance(root: Path, claim_boundary: object) -> None:
    provenance = strict_json_load(root / "provenance.json")
    current_hashes = _implementation_hashes()
    if not (
        has_exact_fields(provenance, _PROVENANCE_FIELDS)
        and provenance["schema_version"] == CPU_RUNTIME_SCHEMA
        and canonical_json(provenance["provider_modes"])
        == canonical_json(_PROVIDER_MODES)
        and provenance["implementation_files"]
        == list(_IMPLEMENTATION_RELATIVE_PATHS)
        and isinstance(provenance["implementation_sha256"], Mapping)
        and set(provenance["implementation_sha256"])
        == set(_IMPLEMENTATION_RELATIVE_PATHS)
        and all(
            is_sha256(provenance["implementation_sha256"][relative])
            for relative in _IMPLEMENTATION_RELATIVE_PATHS
        )
        and provenance["implementation_sha256"] == current_hashes
        and canonical_json(provenance["claim_boundary"])
        == canonical_json(claim_boundary)
    ):
        raise ValueError("CPU runtime provenance or implementation hash mismatch")


def _same_recomputed_row(
    stored: Mapping[str, object], fresh: Mapping[str, object]
) -> bool:
    # Wall-clock latency and polling counts are observations, not deterministic
    # semantics. Everything that can change the actuator decision is replayed.
    stable_fields = (
        "schema_version",
        "case_id",
        "provider",
        "fault",
        "expected_status",
        "expected_reason_prefix",
        "policy_succeeded",
        "policy_failure_type",
        "policy_worker_thread_separate",
        "assurance_worker_thread_separate",
        "status",
        "reason",
        "supervisor_status",
        "policy_action_count",
        "policy_actions_executable",
        "partial_prefix_exposed",
        "fallback_validated",
        "expected_match",
    )
    return all(stored[field] == fresh[field] for field in stable_fields)


def validate_cpu_runtime_completion(directory: Path) -> dict[str, object]:
    """Verify hashes, cross-file semantics, and every registered case."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CPU runtime directory not found: {root}")
    inventory_hash = _validate_manifest(root)
    summary = strict_json_load(root / "summary.json")
    specs = strict_json_load(root / "cases.json")
    expected_specs = _case_specs()
    # Registered inputs are source-defined. Re-signing a modified cases.json
    # therefore cannot silently redefine what the artifact claims to test.
    if not (
        isinstance(specs, list)
        and all(has_exact_fields(spec, _CASE_FIELDS) for spec in specs)
        and canonical_json(specs) == canonical_json(expected_specs)
        and len({spec["case_id"] for spec in specs}) == len(specs)
    ):
        raise ValueError("CPU runtime registered cases are invalid")
    if not (
        has_exact_fields(summary, _SUMMARY_FIELDS)
        and summary["schema_version"] == CPU_RUNTIME_SCHEMA
        and summary["scope"] == CPU_RUNTIME_SCOPE
        and canonical_json(summary["claim_boundary"])
        == canonical_json(_CLAIM_BOUNDARY)
        and has_exact_fields(summary["overall"], _OVERALL_FIELDS)
        and isinstance(summary["rows"], list)
        and isinstance(summary["by_provider"], Mapping)
    ):
        raise ValueError("CPU runtime summary is invalid")
    config = _load_config(summary["configuration"])
    _validate_provenance(root, summary["claim_boundary"])

    with (root / "per_case.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CPU_RUNTIME_CSV_FIELDS:
            raise ValueError("CPU runtime CSV fields are invalid")
        stored_rows = [_parse_csv_row(row) for row in reader]
    summary_rows = [_validate_row_values(row) for row in summary["rows"]]
    if (
        len(stored_rows) != len(expected_specs)
        or len(summary_rows) != len(expected_specs)
        or len({row["case_id"] for row in stored_rows}) != len(stored_rows)
        or canonical_json(stored_rows) != canonical_json(summary_rows)
    ):
        raise ValueError("CPU runtime CSV and summary rows disagree")
    if [row["case_id"] for row in stored_rows] != [
        spec["case_id"] for spec in expected_specs
    ]:
        raise ValueError("CPU runtime case order is invalid")
    for row, spec in zip(stored_rows, expected_specs):
        if any(
            row[field] != spec[field]
            for field in (
                "case_id",
                "provider",
                "fault",
                "expected_status",
                "expected_reason_prefix",
            )
        ):
            raise ValueError("CPU runtime row does not match its registered case")

    stored_aggregate = _summary(stored_rows)
    # CSV is the machine-readable table; summary JSON/Markdown are derived
    # views. They must agree before the slower semantic replay begins.
    if canonical_json(summary["overall"]) != canonical_json(
        stored_aggregate["overall"]
    ):
        raise ValueError("CPU runtime overall aggregate mismatch")
    by_provider = summary["by_provider"]
    if set(by_provider) != set(stored_aggregate["by_provider"]) or any(
        not has_exact_fields(value, _PROVIDER_AGGREGATE_FIELDS)
        for value in by_provider.values()
    ):
        raise ValueError("CPU runtime provider aggregate fields are invalid")
    if canonical_json(by_provider) != canonical_json(
        stored_aggregate["by_provider"]
    ):
        raise ValueError("CPU runtime provider aggregate mismatch")
    if not all(bool(row["expected_match"]) for row in stored_rows):
        raise ValueError("CPU runtime registered outcome mismatch")
    if (root / "summary.md").read_text(encoding="utf-8") != _summary_markdown(
        summary
    ):
        raise ValueError("CPU runtime Markdown summary is not reproducible")

    recomputed = [_validate_row_values(_run_case(spec, config)) for spec in specs]
    for stored, fresh in zip(stored_rows, recomputed):
        if not _same_recomputed_row(stored, fresh):
            raise ValueError(
                f"CPU runtime recomputation mismatch: {stored['case_id']}"
            )
    recomputed_summary = _summary(recomputed)
    return {
        "valid": True,
        "scope": CPU_RUNTIME_SCOPE,
        "cases": len(recomputed),
        "expected_matches": int(recomputed_summary["overall"]["expected_matches"]),
        "accepted_plans": int(recomputed_summary["overall"]["accepted_plans"]),
        "holds": int(recomputed_summary["overall"]["holds"]),
        "unrecoverable_stops": int(recomputed_summary["overall"]["unrecoverable_stops"]),
        "partial_prefix_exposed": 0,
        "manifest_inventory_sha256": inventory_hash,
        "checks": [
            "recursive_manifest_and_exact_file_set",
            "implementation_source_hashes",
            "cross_file_rows_and_aggregates",
            "registered_cases_rerun",
            "provider_modes_replayed",
            "atomic_generation_and_prefix_invariant",
            "markdown_summary_reproduced",
        ],
    }


__all__ = [
    "CPU_RUNTIME_CSV_FIELDS",
    "CPURuntimeMatrixConfig",
    "run_cpu_runtime_completion",
    "validate_cpu_runtime_completion",
]
