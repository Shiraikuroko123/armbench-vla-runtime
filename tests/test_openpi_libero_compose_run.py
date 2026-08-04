from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import integrations.openpi.libero_compose_run as compose_run
from integrations.openpi.libero_compose_run import (
    ATTESTATION_SCHEMA_VERSION,
    DEFAULT_CHECKPOINT,
    OPENPI_COMMIT,
    PREFLIGHT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    _validated_run_directory,
    _normalize_libero_args,
    _resolved_project_name,
    build_manifest_files,
    finalize_run,
    validate_run_manifest,
    write_json,
)


@pytest.fixture(autouse=True)
def _stub_evaluation_validator(monkeypatch):
    def validate(path):
        return SimpleNamespace(
            to_dict=lambda: {
                "schema_version": "armbench.libero_artifact_validation.v1",
                "artifact": str(path),
                "valid": True,
                "errors": [],
                "warnings": [],
                "checks": ["test fixture"],
            }
        )

    monkeypatch.setattr(compose_run, "validate_artifact", validate)


def _process_record(returncode=0, attempted=True):
    return {
        "schema_version": "armbench.compose_process.v1",
        "attempted": attempted,
        "argv": ["docker", "compose"],
        "started_at_utc": "2026-08-04T00:00:00+00:00",
        "duration_s": 1.0,
        "returncode": returncode,
        "stdout": "",
        "stderr": "",
    }


def _make_complete_run(run_directory):
    evaluation = run_directory / "evaluation"
    source = (
        evaluation
        / "provenance"
        / "armbench_source"
        / "integrations"
        / "openpi"
        / "serve_policy_attested.py"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"attested server source\n")
    write_json(
        run_directory / "preflight.json",
        {"schema_version": PREFLIGHT_SCHEMA_VERSION, "ready": True},
    )
    write_json(
        run_directory / "resolved_compose_config.json",
        {
            "services": {
                "runtime": {"command": "libero_runtime_eval /evaluation"},
                "openpi_server": {"command": "serve_policy_attested.py"},
            }
        },
    )
    inventory = [
        {
            "path": "params/model.bin",
            "bytes": 5,
            "sha256": hashlib.sha256(b"model").hexdigest(),
        }
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "policy_loaded": True,
        "policy_config": "pi05_libero",
        "checkpoint_uri": DEFAULT_CHECKPOINT,
        "checkpoint_file_count": 1,
        "checkpoint_total_bytes": 5,
        "checkpoint_content_sha256": inventory_sha256,
        "checkpoint_files": inventory,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_tracked_clean": True,
        "openpi_tracked_status": "",
        "openpi_submodules_clean": True,
        "action_horizon": 10,
        "model_action_dim": 32,
        "server_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    write_json(
        run_directory / "checkpoint_attestation.json",
        attestation,
    )
    (run_directory / "openpi_server.log").write_bytes(b"Serving attested policy\n")
    write_json(run_directory / "compose_up.json", _process_record())
    write_json(run_directory / "compose_stop.json", _process_record())
    write_json(evaluation / "integrity.json", {"valid": True})
    write_json(
        evaluation / "environment.json",
        {
            "server_metadata": {
                "armbench_server_attestation": {
                    key: value
                    for key, value in attestation.items()
                    if key not in {"checkpoint_files", "checkpoint_local_path"}
                }
            }
        },
    )
    (evaluation / "summary.md").write_text("complete\n", encoding="utf-8")
    write_json(
        evaluation / "manifest.json",
        {
            "schema_version": "armbench.pi05_libero_async.v1",
            "files": build_manifest_files(evaluation),
        },
    )


def test_finalize_hashes_server_log_attestation_and_nested_manifest(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _make_complete_run(run_directory)

    result = finalize_run(run_directory)

    assert result["complete"] is True
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == RUN_SCHEMA_VERSION
    assert manifest["complete"] is True
    assert manifest["files"]["openpi_server.log"]["sha256"] == hashlib.sha256(
        b"Serving attested policy\n"
    ).hexdigest()
    assert "checkpoint_attestation.json" in manifest["files"]
    assert "resolved_compose_config.json" in manifest["files"]
    assert "preflight.json" in manifest["files"]
    assert "artifact_validation.json" in manifest["files"]
    assert "evaluation/manifest.json" in manifest["files"]
    assert validate_run_manifest(run_directory)["valid"] is True


def test_manifest_detects_server_log_change_after_finalize(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _make_complete_run(run_directory)
    finalize_run(run_directory)

    with (run_directory / "openpi_server.log").open("a", encoding="utf-8") as handle:
        handle.write("late write\n")

    validation = validate_run_manifest(run_directory)

    assert validation["valid"] is False
    assert "SHA-256 mismatch: openpi_server.log" in validation["errors"]


def test_finalize_is_incomplete_when_compose_stop_was_not_confirmed(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _make_complete_run(run_directory)
    write_json(run_directory / "compose_stop.json", _process_record(returncode=1))

    result = finalize_run(run_directory)
    finalization = json.loads(
        (run_directory / "finalization.json").read_text(encoding="utf-8")
    )

    assert result["complete"] is False
    assert "docker compose stop was not confirmed successful" in result["errors"]
    assert finalization["server_log_hashed_only_after_compose_stop"] is False
    validation = validate_run_manifest(run_directory)
    assert validation["valid"] is False
    assert "root manifest is explicitly incomplete" in validation["errors"]


def test_finalize_rejects_failed_independent_artifact_validation(
    tmp_path, monkeypatch
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    _make_complete_run(run_directory)
    monkeypatch.setattr(
        compose_run,
        "validate_artifact",
        lambda path: SimpleNamespace(
            to_dict=lambda: {
                "schema_version": "armbench.libero_artifact_validation.v1",
                "artifact": str(path),
                "valid": False,
                "errors": ["[recomputation] forged aggregate"],
                "warnings": [],
                "checks": [],
            }
        ),
    )

    result = finalize_run(run_directory)

    assert result["complete"] is False
    assert any("forged aggregate" in error for error in result["errors"])


def test_run_directory_rejects_traversal_and_existing_evidence(tmp_path):
    with pytest.raises(ValueError, match="run-id"):
        _validated_run_directory(tmp_path, "../escape")

    run_directory = _validated_run_directory(tmp_path, "valid_run-01")
    (run_directory / "existing.txt").write_text("evidence", encoding="ascii")
    with pytest.raises(FileExistsError, match="absent or empty"):
        _validated_run_directory(tmp_path, "valid_run-01")


def test_compose_project_name_is_safe_for_run_ids_with_dots():
    assert _resolved_project_name("smoke.20260804", None) == "armbench-smoke-20260804"
    assert _resolved_project_name("ignored", "Study_Run.01") == "study_run-01"


def test_libero_args_cannot_override_formal_provenance_options():
    assert _normalize_libero_args("--modes async_unguarded,state_guard --task-ids 0:2") == (
        "--modes async_unguarded,state_guard --task-ids 0:2"
    )
    with pytest.raises(ValueError, match="protected option"):
        _normalize_libero_args("--allow-unattested-server")
    with pytest.raises(ValueError, match="protected option"):
        _normalize_libero_args("--allow-u")
    with pytest.raises(ValueError, match="metacharacters"):
        _normalize_libero_args("--task-ids 0; touch /tmp/forged")


def test_execute_run_stops_compose_before_finalizing(tmp_path, monkeypatch):
    events = []
    args = SimpleNamespace(
        openpi_root=tmp_path / "openpi",
        armbench_root=tmp_path / "armbench",
        results_root=tmp_path / "results",
        run_id="ordered-run",
        policy_port=8000,
        server_wait_attempts=3,
        libero_args="",
        fixed_refresh_interval=4,
        skip_container_gpu_probe=False,
        project_name=None,
        no_build=True,
        stop_timeout_s=5,
    )
    args.openpi_root.mkdir()
    args.armbench_root.mkdir()
    monkeypatch.setattr(compose_run, "collect_facts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        compose_run, "evaluate_preflight", lambda facts: {"ready": True}
    )

    def fake_capture(argv, cwd, environment, timeout_s):
        if argv[-3:] == ["config", "--format", "json"]:
            events.append("config")
            assert environment["ARMBENCH_LIBERO_ARGS"] == "--fixed-refresh-interval 4"
            record = _process_record()
            record["stdout"] = json.dumps({"services": {}})
            return record
        events.append("stop")
        return _process_record()

    def fake_up(argv, cwd, environment, log_path):
        events.append("up")
        log_path.write_text("compose output\n", encoding="utf-8")
        return _process_record()

    def fake_finalize(run_directory):
        events.append("finalize")
        assert (run_directory / "compose_stop.json").is_file()
        return {"complete": True}

    monkeypatch.setattr(compose_run, "_run_capture", fake_capture)
    monkeypatch.setattr(compose_run, "_stream_compose_up", fake_up)
    monkeypatch.setattr(compose_run, "finalize_run", fake_finalize)

    assert compose_run.execute_run(args) == 0
    assert events == ["config", "up", "stop", "finalize"]


def test_execute_run_finalizes_when_compose_stream_raises(tmp_path, monkeypatch):
    events = []
    args = SimpleNamespace(
        openpi_root=tmp_path / "openpi",
        armbench_root=tmp_path / "armbench",
        results_root=tmp_path / "results",
        run_id="stream-failure",
        policy_port=8000,
        server_wait_attempts=3,
        libero_args="",
        fixed_refresh_interval=None,
        skip_container_gpu_probe=False,
        project_name=None,
        no_build=True,
        stop_timeout_s=5,
    )
    args.openpi_root.mkdir()
    args.armbench_root.mkdir()
    monkeypatch.setattr(compose_run, "collect_facts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        compose_run, "evaluate_preflight", lambda facts: {"ready": True}
    )

    def fake_capture(argv, cwd, environment, timeout_s):
        if argv[-3:] == ["config", "--format", "json"]:
            record = _process_record()
            record["stdout"] = json.dumps({"services": {}})
            return record
        events.append("stop")
        return _process_record()

    monkeypatch.setattr(compose_run, "_run_capture", fake_capture)
    monkeypatch.setattr(
        compose_run,
        "_stream_compose_up",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stream failed")),
    )
    monkeypatch.setattr(
        compose_run,
        "finalize_run",
        lambda run_directory: events.append("finalize") or {"complete": False},
    )

    assert compose_run.execute_run(args) == 2
    assert events == ["stop", "finalize"]
    record = json.loads(
        (args.results_root / args.run_id / "compose_up.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["returncode"] is None
    assert "stream failed" in record["stderr"]
