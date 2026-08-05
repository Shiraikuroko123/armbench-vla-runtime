from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

import integrations.openpi.rtc_overlap_pilot as pilot
from integrations.openpi.rtc_overlap_pilot import OVERLAP_UNCONDITIONED
from integrations.openpi.rtc_overlap_pilot import PilotValidationError
from integrations.openpi.rtc_overlap_pilot import PROJECTED_OVERLAP
from integrations.openpi.rtc_overlap_pilot import RTC_GUIDED_OVERLAP
from integrations.openpi.rtc_overlap_pilot import _validate_query_pairing
from integrations.openpi.rtc_overlap_pilot import build_cells
from integrations.openpi.rtc_overlap_pilot import summarize


def test_three_arm_cells_use_latin_rotation() -> None:
    cells = build_cells(
        "libero_10",
        [0, 1, 2],
        [0],
        execute_horizon=5,
        inference_delay_steps=4,
    )

    assert len(cells) == 9
    assert [cell.method for cell in cells] == [
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
        OVERLAP_UNCONDITIONED,
        RTC_GUIDED_OVERLAP,
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
    ]
    for start in range(0, len(cells), 3):
        assert len({cell.pair_id for cell in cells[start : start + 3]}) == 1


def test_three_arm_summary_keeps_hard_and_soft_residuals_separate() -> None:
    episodes = []
    for pair_id, success in (("a", False), ("b", True)):
        episodes.extend(
            [
                {
                    "pair_id": pair_id,
                    "method": OVERLAP_UNCONDITIONED,
                    "success": success,
                    "policy_queries": 3,
                },
                {
                    "pair_id": pair_id,
                    "method": PROJECTED_OVERLAP,
                    "success": True,
                    "policy_queries": 3,
                },
                {
                    "pair_id": pair_id,
                    "method": RTC_GUIDED_OVERLAP,
                    "success": pair_id == "a",
                    "policy_queries": 3,
                },
            ]
        )
    queries = []
    for method in (
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
    ):
        queries.extend(
            [
                {
                    "method": method,
                    "bootstrap": True,
                    "seam_motion_l2": None,
                    "seam_gripper_abs": None,
                    "max_model_residual": None,
                    "weighted_model_rmse": None,
                },
                {
                    "method": method,
                    "bootstrap": False,
                    "seam_motion_l2": 0.2,
                    "seam_gripper_abs": 0.1,
                    "max_model_residual": 0.0 if method == PROJECTED_OVERLAP else None,
                    "weighted_model_rmse": 0.03
                    if method == RTC_GUIDED_OVERLAP
                    else None,
                },
            ]
        )

    summary = summarize(episodes, queries)

    assert summary["complete_triplets"] == 2
    assert summary["methods"][PROJECTED_OVERLAP]["max_model_residual"] == 0.0
    assert summary["methods"][RTC_GUIDED_OVERLAP]["mean_weighted_model_rmse"] == 0.03
    assert summary["contrasts_vs_unconditioned"][PROJECTED_OVERLAP]["wins"] == 1
    assert summary["contrasts_vs_unconditioned"][RTC_GUIDED_OVERLAP]["wins"] == 1


def _paired_bootstrap_queries():
    return [
        {
            "pair_id": "pair",
            "method": method,
            "query_index": 0,
            "policy_input_sha256": "a" * 64,
            "response_action_sha256": "b" * 64,
            "sampling_key_sha256": "c" * 64,
            "sampling_noise_sha256": "d" * 64,
        }
        for method in (
            OVERLAP_UNCONDITIONED,
            PROJECTED_OVERLAP,
            RTC_GUIDED_OVERLAP,
        )
    ]


def test_query_zero_pairing_requires_identical_inputs_responses_and_sampling() -> None:
    _validate_query_pairing(_paired_bootstrap_queries())


@pytest.mark.parametrize(
    "field",
    (
        "policy_input_sha256",
        "response_action_sha256",
        "sampling_key_sha256",
        "sampling_noise_sha256",
    ),
)
def test_query_zero_pairing_fails_closed_on_any_mismatch(field) -> None:
    queries = _paired_bootstrap_queries()
    queries[-1][field] = "e" * 64

    with pytest.raises(PilotValidationError, match="query-0"):
        _validate_query_pairing(queries)


@pytest.mark.parametrize("run_failure", (False, True))
def test_execute_uses_and_closes_a_fresh_environment_for_every_cell(
    tmp_path, monkeypatch, run_failure
) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def get_server_metadata(self):
            return {}

        def close(self):
            self.closed = True

    class FakeEnvironment:
        def __init__(self):
            self.seeds = []
            self.closed = False

        def seed(self, value):
            self.seeds.append(value)

        def close(self):
            self.closed = True

    class FakeRecord:
        query_index = 0

        def to_dict(self):
            return {
                "query_index": 0,
                "bootstrap": True,
                "policy_input_sha256": "a" * 64,
                "response_action_sha256": "b" * 64,
                "sampling_key_sha256": "c" * 64,
                "sampling_noise_sha256": "d" * 64,
                "executed_steps": 0,
                "decision": "bootstrap_reference_only",
            }

    class FakeResult:
        success = True
        replay_frames = []
        query_records = [FakeRecord()]
        transition_records = []
        failure_type = None
        failure_message = None

        def to_dict(self):
            return {
                "success": True,
                "termination_reason": "test_complete",
                "initial_state_sha256": "e" * 64,
                "environment_steps": 0,
                "task_action_steps": 0,
                "policy_queries": 1,
                "bootstrap_queries": 1,
                "conditioned_queries": 0,
                "failure_stage": None,
                "failure_type": None,
                "failure_message": None,
            }

    class FakeTaskSuite:
        def get_task(self, _task_id):
            return types.SimpleNamespace(language="test task")

        def get_task_init_states(self, _task_id):
            return [np.asarray([0.0], dtype=np.float64)]

    benchmark_module = types.SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_10": FakeTaskSuite}
    )
    libero_package = types.ModuleType("libero")
    libero_package.__path__ = []
    libero_submodule = types.ModuleType("libero.libero")
    libero_submodule.benchmark = benchmark_module
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_submodule)

    environments = []

    def make_environment(_task, _seed):
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    def run_episode(*_args, **_kwargs):
        if run_failure:
            raise RuntimeError("test run failure")
        return FakeResult()

    project_root = pathlib.Path(__file__).resolve().parents[1]
    openpi_root = tmp_path / "openpi"
    openpi_root.mkdir()
    args = types.SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        openpi_root=str(openpi_root),
        armbench_root=str(project_root),
        host="127.0.0.1",
        port=8000,
        server_startup_timeout_s=1.0,
        inference_timeout_s=1.0,
        task_suite="libero_10",
        execute_horizon=5,
        inference_delay_steps=4,
        sampling_seed=1,
        environment_seed=7,
        max_task_steps=1,
        video_mode="none",
    )
    cells = build_cells(
        "libero_10",
        [0],
        [0],
        execute_horizon=5,
        inference_delay_steps=4,
    )
    monkeypatch.setattr(pilot, "BoundedOpenPIClient", FakeClient)
    monkeypatch.setattr(pilot, "_validate_server_metadata", lambda *_args: {})
    monkeypatch.setattr(pilot, "_make_libero_environment", make_environment)
    monkeypatch.setattr(pilot, "run_overlap_episode", run_episode)
    monkeypatch.setattr(
        pilot,
        "_command_output",
        lambda command, cwd=None: (
            pilot.OPENPI_RTC_GUIDANCE_COMMIT
            if tuple(command[:3]) == ("git", "rev-parse", "HEAD")
            and pathlib.Path(cwd) == openpi_root
            else "armbench-test-commit"
            if tuple(command[:3]) == ("git", "rev-parse", "HEAD")
            else ""
        ),
    )
    monkeypatch.setattr(
        pilot,
        "write_transition_archive",
        lambda path, _rows: pathlib.Path(path).write_bytes(b"test archive"),
    )
    monkeypatch.setattr(pilot, "validate_artifact", lambda _path: {})

    if run_failure:
        with pytest.raises(RuntimeError, match="test run failure"):
            pilot.execute(args, cells)
        assert len(environments) == 1
    else:
        assert pilot.execute(args, cells) == 0
        assert len(environments) == len(cells) == 3
        assert len({id(environment) for environment in environments}) == 3
    assert all(environment.seeds == [7] for environment in environments)
    assert all(environment.closed for environment in environments)
