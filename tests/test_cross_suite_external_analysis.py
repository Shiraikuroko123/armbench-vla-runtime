from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

import integrations.openpi.cross_suite_external_analysis as module
from integrations.openpi.cross_suite_external_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    ASYNC_UNGUARDED,
    EMPTY_GIT_DIFF_SHA256,
    FROZEN_ARMBENCH_RUN_COMMIT,
    FROZEN_CHECKPOINT,
    FROZEN_CHECKPOINT_CONTENT_SHA256,
    FROZEN_OPENPI_COMMIT,
    FROZEN_POLICY_CONFIG,
    FROZEN_RUN_IDS,
    FROZEN_SUITES,
    LATENCY_ALIGNED,
    execute_analysis,
    validate_analysis_manifest,
)
from integrations.openpi.libero_compose_run import validate_directory_manifest


FIELDS = (
    "schema_version", "episode_id", "pair_id", "condition_order", "task_suite",
    "task_id", "episode_index", "task_description", "mode", "replan_steps",
    "latency_steps", "injected_latency_ms", "fixed_refresh_interval", "seed",
    "success", "initial_state_sha256", "policy_queries", "failure_category",
    "failure_type", "failure_message", "video_required", "video_path",
    "video_error_type", "video_error_message",
)


@pytest.fixture
def stub_strict_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "validate_run_manifest",
        lambda _root: {
            "valid": True, "complete": True, "errors": [], "files_checked": 137,
        },
    )


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(data: bytes):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _rows(suite: str, aligned_gain: int, runtime_failure: bool = False):
    rows = []
    order = 0
    for task_id in range(10):
        for episode_index in range(5):
            pair_index = task_id * 5 + episode_index
            pair_id = "%s/task_%03d/episode_%03d/h_05/l_004" % (
                suite, task_id, episode_index,
            )
            mode_order = (
                (ASYNC_UNGUARDED, LATENCY_ALIGNED)
                if pair_index % 2 == 0 else (LATENCY_ALIGNED, ASYNC_UNGUARDED)
            )
            for mode in mode_order:
                baseline_success = pair_index < 15
                success = baseline_success or (
                    mode == LATENCY_ALIGNED and pair_index < 15 + aligned_gain
                )
                failed = runtime_failure and pair_index == 49 and mode == ASYNC_UNGUARDED
                rows.append({
                    "schema_version": "armbench.pi05_libero_async.v1",
                    "episode_id": pair_id + "__" + mode, "pair_id": pair_id,
                    "condition_order": order, "task_suite": suite, "task_id": task_id,
                    "episode_index": episode_index, "task_description": "task %d" % task_id,
                    "mode": mode, "replan_steps": 5, "latency_steps": 4,
                    "injected_latency_ms": 200.0, "fixed_refresh_interval": "",
                    "seed": 7, "success": False if failed else success,
                    "initial_state_sha256": hashlib.sha256(
                        (suite + "/%d/%d" % (task_id, episode_index)).encode("ascii")
                    ).hexdigest(),
                    "policy_queries": 20 if mode == ASYNC_UNGUARDED else 12,
                    "failure_category": "policy_timeout" if failed else "",
                    "failure_type": "TimeoutError" if failed else "",
                    "failure_message": "timed out" if failed else "",
                    "video_required": True,
                    "video_path": "videos/" + (pair_id + "__" + mode).replace("/", "__") + ".mp4",
                    "video_error_type": "",
                    "video_error_message": "",
                })
                order += 1
    return rows


def _make_run(
    parent: pathlib.Path,
    suite: str,
    run_id: str,
    *,
    rows=None,
    protocol_mutation=None,
    environment_mutation=None,
) -> pathlib.Path:
    root = parent / run_id
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    selected = list(_rows(suite, {"libero_object": 5, "libero_goal": 10, "libero_10": 15}[suite]) if rows is None else rows)
    source = evaluation / "per_episode.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    protocol = {
        "schema_version": "armbench.pi05_libero_async.v1",
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "policy_config": FROZEN_POLICY_CONFIG,
        "declared_checkpoint": FROZEN_CHECKPOINT,
        "checkpoint_provenance": "server_attestation_with_checkpoint_content_sha256",
        "runtime_failure_policy": "abort_formal_run",
        "seed": 7, "bootstrap_resamples": 10000,
        "episode_budget": {
            "official_suite_task_steps": {
                "libero_spatial": 220, "libero_object": 280,
                "libero_goal": 300, "libero_10": 520, "libero_90": 400,
            },
            "stabilization_steps": 10,
            "task_steps_override": None,
        },
        "experimental_mechanism": {
            "fixed_refresh_interval": None,
            "step_budget": (
                "Injected delay steps consume the same environment-step budget "
                "as task actions."
            ),
        },
        "thresholds": {
            "position_m": 0.01, "orientation_rad": 0.1,
            "gripper_linf": 0.05, "max_requeries": 2,
        },
        "timeouts": {"server_startup_s": 1200.0, "policy_inference_s": 600.0},
        "official_protocol": {
            "action_dimension": 7, "camera_rotation_degrees": 180,
            "control_frequency_hz": 20, "control_period_ms": 50.0,
            "environment_render_resolution": [256, 256], "resize": [224, 224],
            "state_dimension": 8, "task_success_source": "LIBERO environment done",
            "video_playback_fps": 10,
        },
        "matrix": {
            "task_suites": [suite], "task_ids": list(range(10)),
            "episode_indices": list(range(5)),
            "modes": [ASYNC_UNGUARDED, LATENCY_ALIGNED], "replan_steps": [5],
            "latency_steps": [4], "matched_condition_groups": 50, "rollouts": 100,
        },
    }
    if protocol_mutation:
        protocol_mutation(protocol)
    environment = {
        "armbench_git_commit": FROZEN_ARMBENCH_RUN_COMMIT,
        "armbench_git_status": "", "armbench_git_diff_sha256": EMPTY_GIT_DIFF_SHA256,
        "openpi_git_commit": FROZEN_OPENPI_COMMIT,
        "arguments": {
            "video_mode": "all", "command": "run", "allow_commit_mismatch": False,
            "allow_unattested_server": False, "checkpoint": FROZEN_CHECKPOINT,
            "expected_openpi_commit": FROZEN_OPENPI_COMMIT,
            "max_task_steps": None, "continue_after_runtime_failure": False,
            "resize_size": 224, "num_steps_wait": 10, "max_requeries": 2,
            "inference_timeout_s": 600.0, "position_threshold_m": 0.01,
            "orientation_threshold_rad": 0.1, "gripper_threshold": 0.05,
            "fixed_refresh_interval": None, "task_suite": suite,
            "modes": "async_unguarded,latency_aligned", "task_ids": "all",
            "episode_indices": "0:5", "replan_steps": "5", "latency_steps": "4",
            "seed": 7, "bootstrap_resamples": 10000,
            "max_consecutive_infrastructure_failures": 3,
            "server_startup_timeout_s": 1200.0,
        },
        "server_metadata": {"armbench_server_attestation": {
            "policy_config": FROZEN_POLICY_CONFIG, "checkpoint_uri": FROZEN_CHECKPOINT,
            "openpi_commit": FROZEN_OPENPI_COMMIT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
            "action_horizon": 10,
        }},
    }
    if environment_mutation:
        environment_mutation(environment)
    progress = {"planned_rollouts": 100, "completed_rollouts": 100, "complete": True}
    values = {
        "resolved_protocol.json": protocol,
        "environment.json": environment,
        "progress.json": progress,
    }
    for name, value in values.items():
        _write_json(evaluation / name, value)
    for row in selected:
        video = evaluation / pathlib.PurePosixPath(row["video_path"])
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((row["episode_id"] + "\n").encode("ascii"))
    eval_files = {}
    for name in ("per_episode.csv", "resolved_protocol.json", "environment.json", "progress.json"):
        eval_files[name] = _record((evaluation / name).read_bytes())
    for video in sorted((evaluation / "videos").iterdir()):
        eval_files[video.relative_to(evaluation).as_posix()] = _record(video.read_bytes())
    _write_json(evaluation / "manifest.json", {
        "schema_version": "armbench.pi05_libero_async.v1", "files": eval_files,
    })
    root_files = {"evaluation/" + name: record for name, record in eval_files.items()}
    root_files["evaluation/manifest.json"] = _record((evaluation / "manifest.json").read_bytes())
    _write_json(root / "manifest.json", {
        "schema_version": "armbench.pi05_libero_container_run.v1",
        "complete": True, "files": root_files,
    })
    return root


def _resign_evaluation_file(run: pathlib.Path, relative: str) -> None:
    evaluation_manifest_path = run / "evaluation" / "manifest.json"
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    data = (run / "evaluation" / pathlib.PurePosixPath(relative)).read_bytes()
    evaluation_manifest["files"][relative] = _record(data)
    _write_json(evaluation_manifest_path, evaluation_manifest)
    root_manifest_path = run / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["files"]["evaluation/" + relative] = _record(data)
    root_manifest["files"]["evaluation/manifest.json"] = _record(
        evaluation_manifest_path.read_bytes()
    )
    _write_json(root_manifest_path, root_manifest)


def _make_family(tmp_path: pathlib.Path, runtime_failure: bool = False):
    return [
        _make_run(
            tmp_path, suite, run_id,
            rows=_rows(suite, gain, runtime_failure and suite == "libero_goal"),
        )
        for suite, run_id, gain in zip(FROZEN_SUITES, FROZEN_RUN_IDS, (5, 10, 15))
    ]


def test_writes_deterministic_cross_suite_itt_outputs(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    runs = _make_family(tmp_path / "inputs", runtime_failure=True)
    output = execute_analysis(runs, tmp_path / "analysis")
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))

    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis["itt"] == {"rollouts": 300, "pairs": 150, "runtime_failures_retained": 1}
    assert [row["task_suite"] for row in analysis["suite_results"]] == list(FROZEN_SUITES)
    assert [row["aligned_wins"] for row in analysis["suite_results"]] == [5, 10, 15]
    assert all(row["analysis_role"] == "confirmatory_suite" for row in analysis["suite_results"])
    assert all(row["mcnemar_holm_p"] >= row["mcnemar_exact_p"] for row in analysis["suite_results"])
    pooled = analysis["pooled_descriptive"]
    assert pooled["paired_n"] == 150
    assert pooled["analysis_scope"] == "pooled_descriptive_only_no_p_value"
    assert not any("mcnemar" in key or "reject" in key for key in pooled)
    assert len(analysis["runtime_failures"]) == 1
    assert analysis["runtime_failures"][0]["success"] is False
    assert analysis["acceptance"]["passed"] is False
    assert analysis["acceptance"]["zero_runtime_failures"] is False
    assert analysis["acceptance"]["videos_verified"] == 300
    assert analysis["acceptance"]["suites"][1]["passed"] is False
    assert [source["run_id"] for source in analysis["sources"]] == list(FROZEN_RUN_IDS)

    expected_counts = {"suite_results.csv": 3, "task_descriptives.csv": 30, "per_pair.csv": 150, "pooled_descriptive.csv": 1}
    for name, count in expected_counts.items():
        with (output / name).open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == count
    report = validate_analysis_manifest(output)
    assert report["valid"], report["errors"]
    generic_report = validate_directory_manifest(
        output, expected_schema=ANALYSIS_SCHEMA_VERSION
    )
    assert generic_report["valid"], generic_report["errors"]
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "No pooled p value" in summary
    assert "Formal acceptance: **FAILED**" in summary


def test_outputs_are_byte_identical_across_source_roots(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    runs_a = _make_family(tmp_path / "a")
    runs_b = _make_family(tmp_path / "b")
    out_a = execute_analysis(runs_a, tmp_path / "out-a")
    out_b = execute_analysis(runs_b, tmp_path / "out-b")
    analysis = json.loads((out_a / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["acceptance"]["passed"] is True
    assert analysis["acceptance"]["complete_suites"] == 3
    assert analysis["acceptance"]["videos_verified"] == 300
    assert {path.name for path in out_a.iterdir()} == {path.name for path in out_b.iterdir()}
    for path in out_a.iterdir():
        assert path.read_bytes() == (out_b / path.name).read_bytes()


def test_rejects_protocol_deviation_and_manifest_tampering(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    runs = _make_family(tmp_path / "protocol")
    protocol = runs[1] / "evaluation" / "resolved_protocol.json"
    value = json.loads(protocol.read_text(encoding="utf-8"))
    value["matrix"]["latency_steps"] = [3]
    _write_json(protocol, value)
    # Resigning only the nested/root records proves frozen protocol checks are independent.
    data = protocol.read_bytes()
    for manifest_path, key in (
        (runs[1] / "evaluation" / "manifest.json", "resolved_protocol.json"),
        (runs[1] / "manifest.json", "evaluation/resolved_protocol.json"),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][key] = _record(data)
        _write_json(manifest_path, manifest)
    root_manifest_path = runs[1] / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["files"]["evaluation/manifest.json"] = _record(
        (runs[1] / "evaluation" / "manifest.json").read_bytes()
    )
    _write_json(root_manifest_path, root_manifest)
    with pytest.raises(ValueError, match="frozen matrix mismatch: latency_steps"):
        execute_analysis(runs, tmp_path / "bad-protocol-output")

    runs = _make_family(tmp_path / "tamper")
    with (runs[0] / "evaluation" / "per_episode.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="mismatch for per_episode.csv"):
        execute_analysis(runs, tmp_path / "tampered-output")


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("arguments", "max_task_steps", 40, "frozen arguments mismatch: max_task_steps"),
        (
            "arguments",
            "continue_after_runtime_failure",
            True,
            "frozen arguments mismatch: continue_after_runtime_failure",
        ),
        ("arguments", "num_steps_wait", 5, "frozen arguments mismatch: num_steps_wait"),
        (
            "episode_budget",
            "task_steps_override",
            40,
            "frozen episode budget mismatch: task_steps_override",
        ),
        (
            "episode_budget",
            "stabilization_steps",
            5,
            "frozen episode budget mismatch: stabilization_steps",
        ),
    ],
)
def test_rejects_episode_budget_and_behavior_parameter_deviations(
    tmp_path: pathlib.Path,
    stub_strict_validation: None,
    target: str,
    field: str,
    value,
    message: str,
) -> None:
    if target == "arguments":
        run = _make_run(
            tmp_path,
            FROZEN_SUITES[0],
            FROZEN_RUN_IDS[0],
            environment_mutation=lambda environment: environment["arguments"].__setitem__(
                field, value
            ),
        )
    else:
        run = _make_run(
            tmp_path,
            FROZEN_SUITES[0],
            FROZEN_RUN_IDS[0],
            protocol_mutation=lambda protocol: protocol["episode_budget"].__setitem__(
                field, value
            ),
        )
    with pytest.raises(ValueError, match=message):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])


def test_rejects_missing_unsafe_duplicate_and_tampered_videos(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    run = _make_run(tmp_path / "missing", FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    first_video = next((run / "evaluation" / "videos").iterdir())
    first_video.unlink()
    with pytest.raises(ValueError, match="manifest-bound file must be regular"):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])

    run = _make_run(tmp_path / "tampered", FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    first_video = next((run / "evaluation" / "videos").iterdir())
    first_video.write_bytes(b"tampered-video\n")
    with pytest.raises(ValueError, match="mismatch for videos/"):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])

    for case, replacement, message in (
        ("unsafe", "../escape.mp4", "unsafe or empty video_path"),
        ("duplicate", None, "video_path values must be unique"),
    ):
        run = _make_run(tmp_path / case, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
        source = run / "evaluation" / "per_episode.csv"
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["video_path"] = replacement or rows[1]["video_path"]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        _resign_evaluation_file(run, "per_episode.csv")
        with pytest.raises(ValueError, match=message):
            module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])

    run = _make_run(tmp_path / "encoding-error", FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    source = run / "evaluation" / "per_episode.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["video_error_type"] = "RuntimeError"
    rows[0]["video_error_message"] = "encoder failed"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _resign_evaluation_file(run, "per_episode.csv")
    with pytest.raises(ValueError, match="records a video encoding error"):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])


def test_detects_manifest_toctou_and_validator_count_change(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _make_run(tmp_path / "manifest", FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    calls = 0

    def mutate_on_second(_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            with (run / "manifest.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
        return {"valid": True, "complete": True, "errors": [], "files_checked": 137}

    monkeypatch.setattr(module, "validate_run_manifest", mutate_on_second)
    with pytest.raises(ValueError, match="root manifest changed during source snapshot"):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    assert calls == 2

    run = _make_run(tmp_path / "count", FROZEN_SUITES[0], FROZEN_RUN_IDS[0])
    calls = 0

    def change_count(_root):
        nonlocal calls
        calls += 1
        return {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 137 if calls == 1 else 138,
        }

    monkeypatch.setattr(module, "validate_run_manifest", change_count)
    with pytest.raises(ValueError, match="file count changed"):
        module.validate_frozen_source(run, FROZEN_SUITES[0], FROZEN_RUN_IDS[0])


def test_rejects_wrong_suite_order_and_broken_pairing(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    runs = _make_family(tmp_path / "order")
    with pytest.raises(ValueError, match="order/name mismatch"):
        execute_analysis([runs[1], runs[0], runs[2]], tmp_path / "wrong-order")

    rows = _rows("libero_object", 5)
    rows[1]["initial_state_sha256"] = "f" * 64
    broken = _make_run(tmp_path / "pair", FROZEN_SUITES[0], FROZEN_RUN_IDS[0], rows=rows)
    other = [
        _make_run(tmp_path / "pair", suite, run_id)
        for suite, run_id in zip(FROZEN_SUITES[1:], FROZEN_RUN_IDS[1:])
    ]
    with pytest.raises(ValueError, match="initial_state_sha256 mismatch"):
        execute_analysis([broken, *other], tmp_path / "broken-pair")


def test_requires_fresh_strict_validation(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _make_family(tmp_path / "inputs")
    monkeypatch.setattr(module, "validate_run_manifest", lambda _root: {
        "valid": False, "complete": False, "errors": ["unprotected rogue file"], "files_checked": 0,
    })
    with pytest.raises(ValueError, match="strict full run validation failed.*rogue"):
        execute_analysis(runs, tmp_path / "output")


def test_analysis_manifest_fails_closed_and_binds_source_hashes(
    tmp_path: pathlib.Path, stub_strict_validation: None,
) -> None:
    output = execute_analysis(_make_family(tmp_path / "inputs"), tmp_path / "analysis")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["root_manifest_sha256"] = "f" * 64
    _write_json(manifest_path, manifest)
    report = validate_analysis_manifest(output)
    assert report["valid"] is False
    assert any("source hashes do not match" in error for error in report["errors"])

    output = execute_analysis(_make_family(tmp_path / "missing-inputs"), tmp_path / "missing")
    (output / "summary.md").unlink()
    report = validate_analysis_manifest(output)
    assert report["valid"] is False
    assert any("fixed six outputs" in error or "output set mismatch" in error for error in report["errors"])

    output = execute_analysis(_make_family(tmp_path / "malformed-inputs"), tmp_path / "malformed")
    manifest_path = output / "manifest.json"
    manifest_path.write_text("{malformed", encoding="utf-8")
    report = validate_analysis_manifest(output)
    assert report["valid"] is False
    assert any("malformed" in error for error in report["errors"])

    manifest_path.unlink()
    report = validate_analysis_manifest(output)
    assert report["valid"] is False
    assert any("missing" in error for error in report["errors"])


def test_atomic_failure_refuses_overwrite_and_unsafe_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    stub_strict_validation: None,
) -> None:
    runs = _make_family(tmp_path / "inputs")
    output = tmp_path / "analysis"
    original = module._write_csv

    def fail(path, rows):
        if path.name == "per_pair.csv":
            raise OSError("injected write failure")
        original(path, rows)

    monkeypatch.setattr(module, "_write_csv", fail)
    with pytest.raises(OSError, match="injected write failure"):
        execute_analysis(runs, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".analysis.tmp-*"))

    monkeypatch.setattr(module, "_write_csv", original)
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        execute_analysis(runs, output)
    with pytest.raises(ValueError, match="outside every frozen input"):
        execute_analysis(runs, runs[0] / "derived")


REAL_OBJECT_RUN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "evidence"
    / "cloud"
    / "external"
    / FROZEN_RUN_IDS[0]
)


@pytest.mark.skipif(
    not REAL_OBJECT_RUN.is_dir(),
    reason="downloaded formal Object artifact is not available",
)
def test_real_object_artifact_passes_unstubbed_strict_source_validation() -> None:
    source = module.validate_frozen_source(
        REAL_OBJECT_RUN, FROZEN_SUITES[0], FROZEN_RUN_IDS[0]
    )
    assert source.complete is True
    assert source.videos_verified == 100
    assert source.independently_validated_files > 100
