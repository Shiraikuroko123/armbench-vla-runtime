from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

import integrations.openpi.rtc_overlap_primary_analysis as analysis_module
from integrations.openpi import rtc_overlap_pilot
from integrations.openpi.rtc_overlap_primary_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FROZEN_ARMBENCH_COMMIT,
    FROZEN_CHECKPOINT,
    FROZEN_CHECKPOINT_CONTENT_SHA256,
    FROZEN_EXTERNAL_PROTOCOL_COMMIT,
    FROZEN_OPENPI_EXTENSION_COMMIT,
    FROZEN_OPENPI_UPSTREAM_COMMIT,
    SAMPLING_SEEDS,
    AnalysisError,
    analyze_artifacts,
    generate_report,
    validate_analysis_manifest,
)


METHODS = tuple(rtc_overlap_pilot.V2_OVERLAP_METHODS)
BASELINE = rtc_overlap_pilot.OVERLAP_UNCONDITIONED
PROJECTED = rtc_overlap_pilot.PROJECTED_OVERLAP
RTC = rtc_overlap_pilot.RTC_GUIDED_OVERLAP


def _write_json(path: pathlib.Path, value, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _artifact(root: pathlib.Path, sampling_seed: int) -> pathlib.Path:
    root.mkdir(parents=True)
    cells = rtc_overlap_pilot.build_cells(
        "libero_10",
        list(range(10)),
        list(range(2, 7)),
        execute_horizon=5,
        inference_delay_steps=4,
    )
    protocol = {
        "schema_version": rtc_overlap_pilot.SCHEMA_VERSION,
        "pilot_only": True,
        "policy_config": "pi05_libero",
        "checkpoint": FROZEN_CHECKPOINT,
        "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
        "openpi_upstream_commit": FROZEN_OPENPI_UPSTREAM_COMMIT,
        "openpi_extension_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
        "task_suite": "libero_10",
        "task_ids": list(range(10)),
        "episode_indices": list(range(2, 7)),
        "methods": list(METHODS),
        "execute_horizon": 5,
        "inference_delay_steps": 4,
        "action_horizon": 10,
        "control_period_ms": 50.0,
        "fixed_delay_ms": 200.0,
        "sampling_seed": sampling_seed,
        "pairing_key_fields": [
            "task_suite",
            "task_id",
            "episode_index",
            "execute_horizon",
            "query_index",
        ],
        "bootstrap_rule": (
            "query 0 generates an unexecuted reference; query 1 samples from "
            "the same observation"
        ),
        "scheduler": "old[:d] + new[d:E], then new[E:H] + zeros(E)",
        "matrix": [cell.to_dict() for cell in cells],
        "planned_rollouts": 150,
        "complete_triplets": 50,
        "video_mode": "failures",
    }
    source_hashes = {
        source: _digest("source:" + source) for source in rtc_overlap_pilot.SOURCE_FILES
    }
    environment = {
        "schema_version": rtc_overlap_pilot.SCHEMA_VERSION,
        "armbench_commit": FROZEN_ARMBENCH_COMMIT,
        "armbench_status": "",
        "command": [
            "/armbench/integrations/openpi/rtc_overlap_pilot.py",
            "run",
            "--output-dir",
            "/armbench_results/pi05_rtc_overlap_primary_v3_seed_%d_001/evaluation"
            % sampling_seed,
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--openpi-root",
            "/app",
            "--armbench-root",
            "/armbench",
            "--task-suite",
            "libero_10",
            "--task-ids",
            "all",
            "--episode-indices",
            "2,3,4,5,6",
            "--sampling-seed",
            str(sampling_seed),
            "--environment-seed",
            "7",
            "--video-mode",
            "failures",
            "--server-startup-timeout-s",
            "120",
            "--inference-timeout-s",
            "600",
        ],
        "openpi_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
        "openpi_status": "",
        "source_sha256": source_hashes,
        "server_attestation": {
            "schema_version": "armbench.openpi_server_attestation.v1",
            "policy_config": "pi05_libero",
            "checkpoint_uri": FROZEN_CHECKPOINT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
            "openpi_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
            "openpi_upstream_base_commit": FROZEN_OPENPI_UPSTREAM_COMMIT,
            "openpi_tracked_clean": True,
            "openpi_submodules_clean": True,
            "action_horizon": 10,
            "openpi_extension_files": {
                "src/openpi/models/pi0.py": _digest("pi0"),
                "src/openpi/policies/policy.py": _digest("policy"),
            },
            "server_source_sha256": source_hashes[
                "integrations/openpi/serve_policy_attested.py"
            ],
        },
    }
    episodes = []
    queries = []
    for cell in cells:
        identity = {
            "schema_version": rtc_overlap_pilot.SCHEMA_VERSION,
            **cell.to_dict(),
        }
        state_hash = _digest("state:%d:%d" % (cell.task_id, cell.episode_index))
        baseline_success = cell.episode_index != 2
        success = baseline_success
        if cell.method == PROJECTED and cell.task_id < 5:
            success = True
        if cell.method == RTC:
            success = True
        baseline_motion = (
            0.20
            + 0.01 * cell.task_id
            + 0.001 * cell.episode_index
            + 0.000001 * (sampling_seed - SAMPLING_SEEDS[0])
        )
        baseline_gripper = 0.10 + 0.002 * cell.task_id
        motion_offset = {BASELINE: 0.0, PROJECTED: -0.05, RTC: -0.025}[cell.method]
        gripper_offset = {BASELINE: 0.0, PROJECTED: -0.04, RTC: -0.03}[cell.method]
        episodes.append(
            {
                **identity,
                "success": success,
                "policy_queries": 2,
                "initial_state_sha256": state_hash,
                "task_description": "task %d" % cell.task_id,
                "wall_time_s": 1.0,
            }
        )
        for query_index, bootstrap in enumerate((True, False)):
            sampling_key = _digest(
                "key:%d:%s:%d" % (sampling_seed, cell.pair_id, query_index)
            )
            response_identity = "response:%d:%s:%d" % (
                sampling_seed,
                cell.pair_id,
                query_index,
            )
            policy_input_identity = "input:%d:%s:%d" % (
                sampling_seed,
                cell.pair_id,
                query_index,
            )
            if not bootstrap:
                response_identity += ":" + cell.method
                policy_input_identity += ":" + cell.method
            queries.append(
                {
                    **identity,
                    "query_index": query_index,
                    "bootstrap": bootstrap,
                    "sampling_key_sha256": sampling_key,
                    "sampling_noise_sha256": _digest("noise:" + sampling_key),
                    "policy_input_sha256": _digest(policy_input_identity),
                    "response_action_sha256": _digest(response_identity),
                    "condition_raw_actions_sha256": None,
                    "condition_model_actions_sha256": None,
                    "condition_mask_sha256": None,
                    "max_model_residual": None,
                    "guidance_raw_actions_sha256": None,
                    "guidance_model_actions_sha256": None,
                    "guidance_weights_sha256": None,
                    "max_weighted_model_residual": None,
                    "weighted_model_rmse": None,
                    "seam_motion_l2": (
                        None if bootstrap else baseline_motion + motion_offset
                    ),
                    "seam_gripper_abs": (
                        None if bootstrap else baseline_gripper + gripper_offset
                    ),
                }
            )
    values = {
        "resolved_protocol.json": protocol,
        "environment.json": environment,
        "episodes.json": episodes,
        "queries.json": queries,
        "progress.json": {"planned": 150, "completed": 150, "complete": True},
        "summary.json": {"schema_version": rtc_overlap_pilot.SCHEMA_VERSION},
        "transition_descriptor.json": {"schema_version": "test.transition.v1"},
        "manifest.json": {"schema_version": "armbench.root_manifest.v1"},
    }
    for relative, value in values.items():
        _write_json(root / relative, value)
    return root


@pytest.fixture
def artifacts(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def validate(path: pathlib.Path):
        root = pathlib.Path(path).resolve()
        calls.append(root)
        return {"schema_version": "test.validator.v1", "valid": True}

    monkeypatch.setattr(rtc_overlap_pilot, "validate_artifact", validate)
    roots = [_artifact(tmp_path / ("seed-%d" % seed), seed) for seed in SAMPLING_SEEDS]
    return roots, calls


def test_combines_exact_held_out_matrix_with_frozen_statistics(artifacts) -> None:
    roots, calls = artifacts
    analysis, rows = analyze_artifacts(roots)

    assert calls[:2] == [root.resolve() for root in roots]
    assert len(calls) == 4
    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert ANALYSIS_SCHEMA_VERSION.endswith(".v2")
    assert FROZEN_ARMBENCH_COMMIT == "44c358731c5493284b74bb29eefa7d538d0f38dd"
    assert FROZEN_EXTERNAL_PROTOCOL_COMMIT == "509f6f4cbcc9e8b02804edf640e565673d4a3855"
    assert analysis["cohort"]["matched_triplets"] == 100
    assert analysis["cohort"]["rollouts"] == 300
    assert analysis["cohort"]["sampling_seeds"] == list(SAMPLING_SEEDS)
    assert len(rows) == 100
    assert len({row["triplet_id"] for row in rows}) == 100
    assert {
        (row["task_id"], row["episode_index"], row["sampling_seed"]) for row in rows
    } == {
        (task_id, episode_index, seed)
        for task_id in range(10)
        for episode_index in range(2, 7)
        for seed in SAMPLING_SEEDS
    }
    assert analysis["success"]["methods"][BASELINE]["successes"] == 80
    assert analysis["success"]["methods"][PROJECTED]["successes"] == 90
    assert analysis["success"]["methods"][RTC]["successes"] == 100
    projected = analysis["success"]["contrasts_vs_unconditioned"][PROJECTED]
    assert (
        projected["candidate_wins"],
        projected["candidate_losses"],
        projected["ties"],
    ) == (
        10,
        0,
        90,
    )
    assert projected["mcnemar_exact_two_sided_p"] == pytest.approx(2 / 1024)
    assert projected["success_improvement_supported"] is True
    sign_flip = projected["exact_task_sign_flip"]
    assert sign_flip["enumerated_assignments"] == 1024
    assert sign_flip["extreme_assignments"] == 64
    assert sign_flip["exact_p"] == 0.0625
    rtc_sign_flip = analysis["success"]["contrasts_vs_unconditioned"][RTC][
        "exact_task_sign_flip"
    ]
    assert rtc_sign_flip["extreme_assignments"] == 2
    assert rtc_sign_flip["exact_p"] == 2 / 1024
    motion = analysis["seam"]["seam_motion_l2"]
    assert motion["methods"][BASELINE]["valid_rollouts"] == 100
    assert motion["methods"][BASELINE]["scored_transitions"] == 100
    assert motion["contrasts_vs_unconditioned"][PROJECTED][
        "paired_episode_mean_difference"
    ] == pytest.approx(-0.05)
    assert len(analysis["per_task"]) == 10
    assert len(analysis["leave_one_task_out"]["omissions"]) == 10
    assert analysis["statistics"]["bootstrap_seed"] == BOOTSTRAP_SEED
    assert analysis["statistics"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert (
        analysis["frozen_identity"]["external_held_out_protocol_commit"]
        == FROZEN_EXTERNAL_PROTOCOL_COMMIT
    )
    assert all(source["raw_protocol_pilot_only"] for source in analysis["sources"])
    assert all(
        source["source_schema_version"].endswith(".v3")
        for source in analysis["sources"]
    )
    assert all(
        source["bootstrap_triplets_bitwise_verified"] == 50
        for source in analysis["sources"]
    )
    assert all(
        source["bootstrap_pairing_fields"]
        == [
            "policy_input_sha256",
            "response_action_sha256",
            "sampling_key_sha256",
            "sampling_noise_sha256",
        ]
        for source in analysis["sources"]
    )
    assert all(
        source["frozen_environment_command_verified"] is True
        for source in analysis["sources"]
    )
    assert "pilot_only=true" in analysis["claim_boundary"]
    assert "preserved v2 attempts are excluded" in analysis["claim_boundary"]
    assert (
        analysis["protocol_provenance"]["rejected_v2_attempts"]["included_in_estimates"]
        is False
    )


def test_transactional_outputs_are_deterministic_and_manifest_bound(
    artifacts, tmp_path: pathlib.Path
) -> None:
    roots, _ = artifacts
    out_a = tmp_path / "analysis-a"
    out_b = tmp_path / "analysis-b"
    generate_report(roots, out_a)
    generate_report(list(reversed(roots)), out_b)

    expected = {"analysis.json", "per_triplet.csv", "summary.md", "manifest.json"}
    assert {path.name for path in out_a.iterdir()} == expected
    assert {path.name for path in out_b.iterdir()} == expected
    for name in expected:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    validation = validate_analysis_manifest(out_a)
    assert validation["valid"], validation["errors"]
    with (out_a / "per_triplet.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    assert len({row["triplet_id"] for row in rows}) == 100
    assert {int(row["sampling_seed"]) for row in rows} == set(SAMPLING_SEEDS)


def test_seams_are_averaged_within_rollout_before_combining(artifacts) -> None:
    roots, _ = artifacts
    episodes_path = roots[0] / "episodes.json"
    queries_path = roots[0] / "queries.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    episode_id = "libero_10__task_00__episode_02__%s" % BASELINE
    episode = next(row for row in episodes if row["episode_id"] == episode_id)
    episode["policy_queries"] = 3
    extra = dict(
        next(
            row
            for row in queries
            if row["episode_id"] == episode_id and row["query_index"] == 1
        )
    )
    key = _digest("key:%d:%s:%d" % (SAMPLING_SEEDS[0], episode["pair_id"], 2))
    extra.update(
        {
            "query_index": 2,
            "sampling_key_sha256": key,
            "sampling_noise_sha256": _digest("noise:" + key),
            "seam_motion_l2": 1.0,
            "seam_gripper_abs": 0.5,
        }
    )
    queries.append(extra)
    _write_json(episodes_path, episodes)
    _write_json(queries_path, queries)

    analysis, rows = analyze_artifacts(roots)
    selected = next(
        row
        for row in rows
        if row["task_id"] == 0
        and row["episode_index"] == 2
        and row["sampling_seed"] == SAMPLING_SEEDS[0]
    )
    assert selected["%s_seam_motion_l2" % BASELINE] == pytest.approx(
        (0.202 + 1.0) / 2.0
    )
    assert selected["%s_scored_transition_queries" % BASELINE] == 2
    assert analysis["cohort"]["scored_transitions_by_method"][BASELINE] == 101
    assert (
        analysis["seam"]["seam_motion_l2"]["methods"][BASELINE]["valid_rollouts"] == 100
    )


def test_exact_task_sign_flip_zero_effect_boundary_is_one() -> None:
    rows = [
        {
            "task_id": task_id,
            "episode_index": episode_index,
            "sampling_seed": sampling_seed,
            "difference": 0.0,
        }
        for task_id in range(10)
        for episode_index in range(2, 7)
        for sampling_seed in SAMPLING_SEEDS
    ]

    result = analysis_module._exact_task_sign_flip(rows, "difference")

    assert result["enumerated_assignments"] == 1024
    assert result["extreme_assignments"] == 1024
    assert result["exact_p"] == 1.0


def test_rejects_real_bootstrap_response_mismatch_with_same_key_and_noise(
    artifacts,
) -> None:
    roots, _ = artifacts
    path = roots[0] / "queries.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    affected = 0
    for row in rows:
        if (
            row["query_index"] == 0
            and row["method"] == RTC
            and row["task_id"] in (3, 8, 9)
        ):
            row["response_action_sha256"] = _digest(
                "mismatched:%d:%d" % (row["task_id"], row["episode_index"])
            )
            affected += 1
    assert affected == 15
    _write_json(path, rows)

    with pytest.raises(
        AnalysisError,
        match="query0 response action mismatch across methods for task 3 episode 2",
    ):
        analyze_artifacts(roots)


def test_rejects_bootstrap_policy_input_mismatch_with_other_hashes_fixed(
    artifacts,
) -> None:
    roots, _ = artifacts
    path = roots[0] / "queries.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    target = next(
        row
        for row in rows
        if row["query_index"] == 0
        and row["task_id"] == 3
        and row["episode_index"] == 2
        and row["method"] == RTC
    )
    target["policy_input_sha256"] = _digest("stale-policy-image")
    _write_json(path, rows)

    with pytest.raises(
        AnalysisError,
        match="query0 policy input mismatch across methods for task 3 episode 2",
    ):
        analyze_artifacts(roots)


def test_rejects_conditioning_or_guidance_on_reference_bootstrap(artifacts) -> None:
    roots, _ = artifacts
    path = roots[0] / "queries.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    target = next(
        row
        for row in rows
        if row["query_index"] == 0
        and row["task_id"] == 0
        and row["episode_index"] == 2
        and row["method"] == RTC
    )
    target["guidance_raw_actions_sha256"] = _digest("invalid-bootstrap-guidance")
    _write_json(path, rows)

    with pytest.raises(
        AnalysisError,
        match="query0 conditioning/guidance audit must be null",
    ):
        analyze_artifacts(roots)


def _set_command_option(command, option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            "environment_seed",
            "frozen value mismatch for --environment-seed",
        ),
        ("max_task_steps", "must not contain --max-task-steps"),
        ("video_mode", "frozen value mismatch for --video-mode"),
        ("sampling_seed", "frozen value mismatch for --sampling-seed"),
        ("duplicate", "contains duplicate option --host"),
        ("unknown", "contains unknown option --unknown-primary-option"),
        ("not_string_list", "must be a nonempty string list"),
    ],
)
def test_environment_command_deviations_fail_closed(
    artifacts, mutation: str, message: str
) -> None:
    roots, _ = artifacts
    path = roots[0] / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    command = environment["command"]
    if mutation == "environment_seed":
        _set_command_option(command, "--environment-seed", "8")
    elif mutation == "max_task_steps":
        command.extend(("--max-task-steps", "520"))
    elif mutation == "video_mode":
        _set_command_option(command, "--video-mode", "all")
    elif mutation == "sampling_seed":
        _set_command_option(command, "--sampling-seed", str(SAMPLING_SEEDS[1]))
    elif mutation == "duplicate":
        command.extend(("--host", "127.0.0.1"))
    elif mutation == "unknown":
        command.extend(("--unknown-primary-option", "value"))
    else:
        environment["command"] = "not-a-string-list"
    _write_json(path, environment)

    with pytest.raises(AnalysisError, match=message):
        analyze_artifacts(roots)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_seed", "each frozen sampling seed exactly once"),
        ("missing", "exactly 150 rollouts"),
        ("initial_state", "initial-state mismatch across sampling artifacts"),
        ("nonfinite", "non-finite"),
        ("source_identity", "source implementation or recording identity mismatch"),
        ("sampling_duplicate", "cross-artifact duplicate sampling key"),
    ],
)
def test_rejects_duplicate_missing_mismatched_and_nonfinite_sources(
    artifacts, mutation: str, message: str
) -> None:
    roots, _ = artifacts
    if mutation == "duplicate_seed":
        protocol_path = roots[1] / "resolved_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["sampling_seed"] = SAMPLING_SEEDS[0]
        _write_json(protocol_path, protocol)
        environment_path = roots[1] / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        command = environment["command"]
        _set_command_option(command, "--sampling-seed", str(SAMPLING_SEEDS[0]))
        _set_command_option(
            command,
            "--output-dir",
            "/armbench_results/pi05_rtc_overlap_primary_v3_seed_%d_001/evaluation"
            % SAMPLING_SEEDS[0],
        )
        _write_json(environment_path, environment)
    elif mutation == "missing":
        path = roots[1] / "episodes.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows.pop()
        _write_json(path, rows)
    elif mutation == "initial_state":
        path = roots[1] / "episodes.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            if row["task_id"] == 0 and row["episode_index"] == 2:
                row["initial_state_sha256"] = "f" * 64
        _write_json(path, rows)
    elif mutation == "nonfinite":
        path = roots[1] / "episodes.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[0]["wall_time_s"] = float("nan")
        _write_json(path, rows, allow_nan=True)
    elif mutation == "source_identity":
        path = roots[1] / "environment.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_sha256"]["integrations/openpi/realtime_chunking.py"] = "f" * 64
        _write_json(path, value)
    else:
        first_queries = json.loads(
            (roots[0] / "queries.json").read_text(encoding="utf-8")
        )
        path = roots[1] / "queries.json"
        second_queries = json.loads(path.read_text(encoding="utf-8"))
        first_keys = {
            (row["task_id"], row["episode_index"], row["query_index"]): row[
                "sampling_key_sha256"
            ]
            for row in first_queries
        }
        for row in second_queries:
            key = (row["task_id"], row["episode_index"], row["query_index"])
            row["sampling_key_sha256"] = first_keys[key]
        _write_json(path, second_queries)

    with pytest.raises(AnalysisError, match=message):
        analyze_artifacts(roots)


def test_existing_output_is_not_touched(artifacts, tmp_path: pathlib.Path) -> None:
    roots, calls = artifacts
    output = tmp_path / "analysis"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AnalysisError, match="already exists"):
        generate_report(roots, output)
    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_exactly_two_sources_are_required_before_validation(
    artifacts,
) -> None:
    roots, calls = artifacts

    with pytest.raises(AnalysisError, match="exactly two"):
        analyze_artifacts(roots[:1])
    assert calls == []
