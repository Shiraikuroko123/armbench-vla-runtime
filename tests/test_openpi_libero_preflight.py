import json
from pathlib import Path

from integrations.openpi.preflight import (
    CUDA_PROBE_IMAGE,
    OPENPI_COMMIT,
    REQUIRED_OPENPI_PATHS,
    evaluate_preflight,
    parse_nvidia_smi_csv,
    parse_submodule_status,
    probe_results_directory,
)


def _command(ok=True, stdout="", stderr=""):
    return {
        "argv": [],
        "ok": ok,
        "returncode": 0 if ok else 1,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": 1.0,
    }


def _ready_facts():
    device = {
        "name": "NVIDIA RTX 4090",
        "uuid": "GPU-test",
        "driver_version": "555.42",
        "memory_total_mb": 24564.0,
    }
    return {
        "collected_at_utc": "2026-08-04T00:00:00+00:00",
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "machine": "x86_64",
            "python": "3.11.9",
        },
        "openpi": {
            "root": "/opt/openpi",
            "required_paths": {path: True for path in REQUIRED_OPENPI_PATHS},
            "commit": OPENPI_COMMIT,
            "commit_command": _command(stdout=OPENPI_COMMIT + "\n"),
            "tracked_status": "",
            "tracked_status_command": _command(),
            "submodules": [
                {
                    "status": " ",
                    "commit": "a" * 40,
                    "path": "third_party/libero",
                }
            ],
            "submodule_command": _command(),
        },
        "results": {"root": "/results", "writable": True, "error": ""},
        "gpu": {
            "devices": [device],
            "nvidia_smi_command": _command(),
            "container_probe": {
                "attempted": True,
                "image": CUDA_PROBE_IMAGE,
                "devices": [device],
                "command": _command(),
                "skipped_reason": "",
            },
        },
        "docker": {
            "version_command": _command(),
            "info_command": _command(),
            "compose_command": _command(stdout="2.29.1\n"),
            "client_version": "27.2.0",
            "server_version": "27.2.0",
            "compose_version": "2.29.1",
            "os_type": "linux",
            "operating_system": "Ubuntu 22.04",
            "architecture": "x86_64",
            "storage_driver": "overlay2",
            "default_runtime": "runc",
            "runtime_names": ["io.containerd.runc.v2", "nvidia", "runc"],
            "cpu_count": 16,
            "memory_bytes": 68719476736,
        },
    }


def test_evaluate_preflight_accepts_complete_pinned_linux_gpu_host():
    report = evaluate_preflight(_ready_facts())

    assert report["ready"] is True
    assert all(check["passed"] for check in report["checks"])
    assert report["expected_openpi_commit"] == OPENPI_COMMIT
    assert report["generated_at_utc"] == "2026-08-04T00:00:00+00:00"
    assert json.loads(json.dumps(report))["schema_version"] == report["schema_version"]


def test_evaluate_preflight_fails_closed_without_linux_or_gpu():
    facts = _ready_facts()
    facts["platform"]["system"] = "Windows"
    facts["gpu"]["devices"] = []
    facts["gpu"]["nvidia_smi_command"] = _command(False, stderr="not found")
    facts["gpu"]["container_probe"] = {
        "attempted": False,
        "image": CUDA_PROBE_IMAGE,
        "devices": [],
        "command": _command(False),
        "skipped_reason": "host prerequisites failed; container probe was not attempted",
    }

    report = evaluate_preflight(facts)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["ready"] is False
    assert {"host_linux", "host_nvidia_gpu", "container_nvidia_gpu"} <= failed


def test_evaluate_preflight_rejects_commit_dirty_tree_and_unpinned_submodule():
    facts = _ready_facts()
    facts["openpi"]["commit"] = "f" * 40
    facts["openpi"]["tracked_status"] = " M src/openpi/models/model.py\n"
    facts["openpi"]["submodules"][0]["status"] = "+"

    report = evaluate_preflight(facts)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["ready"] is False
    assert {
        "openpi_pinned_commit",
        "openpi_worktree_clean",
        "openpi_submodules_clean",
        "libero_submodule_pinned",
    } <= failed


def test_nvidia_and_submodule_parsers_preserve_reproducibility_fields():
    devices = parse_nvidia_smi_csv(
        "NVIDIA RTX 4090, GPU-abc, 555.42, 24564\n"
        "malformed,row\n"
    )
    submodules = parse_submodule_status(
        " aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa third_party/libero (heads/main)\n"
        "-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb third_party/aloha\n"
    )

    assert devices == [
        {
            "name": "NVIDIA RTX 4090",
            "uuid": "GPU-abc",
            "driver_version": "555.42",
            "memory_total_mb": 24564.0,
        }
    ]
    assert submodules[0]["status"] == " "
    assert submodules[0]["path"] == "third_party/libero"
    assert submodules[1]["status"] == "-"


def test_results_directory_probe_creates_directory_and_cleans_sentinel(tmp_path):
    results = tmp_path / "nested" / "results"

    record = probe_results_directory(results)

    assert record["writable"] is True
    assert results.is_dir()
    assert list(results.iterdir()) == []


def test_compose_overlay_mounts_artifacts_and_runs_the_standalone_evaluator():
    compose_path = (
        Path(__file__).parents[1]
        / "integrations"
        / "openpi"
        / "compose.libero-runtime.yml"
    )
    compose = compose_path.read_text(encoding="utf-8")

    assert "${ARMBENCH_ROOT:?" in compose
    assert "${ARMBENCH_RESULTS_ROOT:?" in compose
    assert "target: /armbench" in compose
    assert "target: /armbench_results" in compose
    assert "python -m integrations.openpi.libero_runtime_eval run" in compose
    assert 'run_directory="/armbench_results/$${ARMBENCH_RUN_ID}"' in compose
    assert "serve_policy_attested.py" in compose
    assert "uv run --frozen python" in compose
    assert "checkpoint_attestation.json" in compose
    assert "openpi_server.log" in compose
    assert '--output-dir "/armbench_results/$${ARMBENCH_RUN_ID}/evaluation"' in compose
    assert "--openpi-root /app" in compose
    assert "--resize-size 224" in compose
    assert "--modes async_unguarded,state_guard" in compose
    assert "ARMBENCH_SERVER_ARGS: >-" in compose
    assert "--policy-config pi05_libero" in compose
    assert "--checkpoint gs://openpi-assets/checkpoints/pi05_libero" in compose
    assert "GPU reservations are inherited from the pinned official Compose file" in compose
