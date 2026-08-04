"""Fail-closed host preflight for the official OpenPI pi0.5-LIBERO run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "armbench.openpi_libero_preflight.v1"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CUDA_PROBE_IMAGE = (
    "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@"
    "sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0"
)
REQUIRED_OPENPI_PATHS = (
    "examples/libero/compose.yml",
    "examples/libero/Dockerfile",
    "scripts/serve_policy.py",
    "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    "third_party/libero/requirements.txt",
)

CommandRunner = Callable[[Sequence[str], Optional[pathlib.Path], float], Mapping[str, Any]]


def run_command(
    argv: Sequence[str], cwd: Optional[pathlib.Path] = None, timeout_s: float = 30.0
) -> Dict[str, Any]:
    """Run a command without a shell and return a JSON-serializable record."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "argv": list(argv),
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except FileNotFoundError as exc:
        return {
            "argv": list(argv),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        return {
            "argv": list(argv),
            "ok": False,
            "returncode": None,
            "stdout": stdout or "",
            "stderr": (stderr or "") + "\ncommand timed out after %.1fs" % timeout_s,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def parse_nvidia_smi_csv(output: str) -> List[Dict[str, Any]]:
    """Parse the stable no-header nvidia-smi query used by both GPU probes."""

    devices: List[Dict[str, Any]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        try:
            memory_total_mb = float(row[3].strip())
        except ValueError:
            continue
        devices.append(
            {
                "name": row[0].strip(),
                "uuid": row[1].strip(),
                "driver_version": row[2].strip(),
                "memory_total_mb": memory_total_mb,
            }
        )
    return devices


def parse_submodule_status(output: str) -> List[Dict[str, str]]:
    """Preserve git's leading status marker for reproducibility checks."""

    entries: List[Dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        marker = line[0] if line[0] in (" ", "-", "+", "U") else "?"
        body = line[1:].strip() if marker != "?" else line.strip()
        fields = body.split()
        if len(fields) < 2:
            continue
        entries.append({"status": marker, "commit": fields[0], "path": fields[1]})
    return entries


def _parse_json_stdout(command: Mapping[str, Any]) -> Dict[str, Any]:
    if not command.get("ok"):
        return {}
    try:
        parsed = json.loads(str(command.get("stdout", "")))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _command_diagnostic(command: Mapping[str, Any]) -> str:
    detail = str(command.get("stderr") or command.get("stdout") or "command failed")
    return detail.strip()[-2000:]


def probe_results_directory(path: pathlib.Path) -> Dict[str, Any]:
    """Create and remove a private sentinel to verify the bind target is writable."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        sentinel = path / (".armbench-preflight-%s-%s.tmp" % (os.getpid(), uuid.uuid4().hex))
        with sentinel.open("xb") as handle:
            handle.write(b"armbench-preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
        sentinel.unlink()
        return {"root": str(path.resolve()), "writable": True, "error": ""}
    except OSError as exc:
        return {"root": str(path.resolve()), "writable": False, "error": str(exc)}


def collect_facts(
    openpi_root: pathlib.Path,
    results_root: pathlib.Path,
    command_runner: CommandRunner = run_command,
    platform_system: Optional[str] = None,
    probe_container_gpu: bool = True,
) -> Dict[str, Any]:
    """Collect host facts. Evaluation is deliberately separate and pure."""

    system = platform_system or platform.system()
    root = openpi_root.resolve()
    required_paths = {
        relative: (root / pathlib.PurePosixPath(relative)).is_file()
        for relative in REQUIRED_OPENPI_PATHS
    }

    git_commit_command = command_runner(("git", "rev-parse", "HEAD"), root, 15.0)
    git_status_command = command_runner(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        root,
        15.0,
    )
    submodule_command = command_runner(
        ("git", "submodule", "status", "--recursive"), root, 30.0
    )
    nvidia_command = command_runner(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        None,
        20.0,
    )
    docker_version_command = command_runner(
        ("docker", "version", "--format", "{{json .}}"), None, 30.0
    )
    docker_info_command = command_runner(
        ("docker", "info", "--format", "{{json .}}"), None, 30.0
    )
    compose_command = command_runner(("docker", "compose", "version", "--short"), None, 30.0)

    nvidia_devices = parse_nvidia_smi_csv(str(nvidia_command.get("stdout", "")))
    docker_version_raw = _parse_json_stdout(docker_version_command)
    docker_info_raw = _parse_json_stdout(docker_info_command)
    runtimes = docker_info_raw.get("Runtimes", {})
    runtime_names = sorted(runtimes) if isinstance(runtimes, dict) else []

    should_probe = bool(
        probe_container_gpu
        and system.lower() == "linux"
        and nvidia_command.get("ok")
        and nvidia_devices
        and docker_version_command.get("ok")
    )
    if should_probe:
        container_command = command_runner(
            (
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "nvidia-smi",
                CUDA_PROBE_IMAGE,
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            None,
            900.0,
        )
        container_devices = parse_nvidia_smi_csv(str(container_command.get("stdout", "")))
        skipped_reason = ""
    else:
        container_command = {
            "argv": [],
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "duration_ms": 0.0,
        }
        container_devices = []
        if not probe_container_gpu:
            skipped_reason = "disabled by --skip-container-gpu-probe"
        else:
            skipped_reason = "host prerequisites failed; container probe was not attempted"

    client = docker_version_raw.get("Client", {})
    server = docker_version_raw.get("Server", {})
    return {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "openpi": {
            "root": str(root),
            "required_paths": required_paths,
            "commit": str(git_commit_command.get("stdout", "")).strip(),
            "commit_command": dict(git_commit_command),
            "tracked_status": str(git_status_command.get("stdout", "")),
            "tracked_status_command": dict(git_status_command),
            "submodules": parse_submodule_status(str(submodule_command.get("stdout", ""))),
            "submodule_command": dict(submodule_command),
        },
        "results": probe_results_directory(results_root),
        "gpu": {
            "devices": nvidia_devices,
            "nvidia_smi_command": dict(nvidia_command),
            "container_probe": {
                "attempted": should_probe,
                "image": CUDA_PROBE_IMAGE,
                "devices": container_devices,
                "command": dict(container_command),
                "skipped_reason": skipped_reason,
            },
        },
        "docker": {
            "version_command": dict(docker_version_command),
            "info_command": dict(docker_info_command),
            "compose_command": dict(compose_command),
            "client_version": client.get("Version") if isinstance(client, dict) else None,
            "server_version": server.get("Version") if isinstance(server, dict) else None,
            "compose_version": str(compose_command.get("stdout", "")).strip(),
            "os_type": docker_info_raw.get("OSType"),
            "operating_system": docker_info_raw.get("OperatingSystem"),
            "architecture": docker_info_raw.get("Architecture"),
            "storage_driver": docker_info_raw.get("Driver"),
            "default_runtime": docker_info_raw.get("DefaultRuntime"),
            "runtime_names": runtime_names,
            "cpu_count": docker_info_raw.get("NCPU"),
            "memory_bytes": docker_info_raw.get("MemTotal"),
        },
    }


def evaluate_preflight(
    facts: Mapping[str, Any], expected_commit: str = OPENPI_COMMIT
) -> Dict[str, Any]:
    """Turn collected facts into deterministic, fail-closed readiness checks."""

    checks: List[Dict[str, Any]] = []

    def add_check(
        name: str,
        passed: bool,
        summary: str,
        remediation: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        checks.append(
            {
                "name": name,
                "required": True,
                "passed": bool(passed),
                "summary": summary,
                "remediation": "" if passed else remediation,
                "evidence": dict(evidence or {}),
            }
        )

    platform_facts = facts.get("platform", {})
    system = str(platform_facts.get("system", ""))
    add_check(
        "host_linux",
        system.lower() == "linux",
        "Host operating system is %s" % (system or "unknown"),
        "Run the official LIBERO containers on a native Linux NVIDIA host.",
        {"system": system, "release": platform_facts.get("release")},
    )

    openpi = facts.get("openpi", {})
    required_paths = openpi.get("required_paths", {})
    missing_paths = sorted(path for path, exists in required_paths.items() if not exists)
    add_check(
        "openpi_layout",
        bool(required_paths) and not missing_paths,
        "Official OpenPI/LIBERO files are present" if not missing_paths else "Missing OpenPI files",
        "Clone the pinned OpenPI repository and initialize its submodules.",
        {"root": openpi.get("root"), "missing_paths": missing_paths},
    )

    commit_command = openpi.get("commit_command", {})
    actual_commit = str(openpi.get("commit", ""))
    commit_ok = bool(commit_command.get("ok")) and actual_commit == expected_commit
    add_check(
        "openpi_pinned_commit",
        commit_ok,
        "OpenPI HEAD is %s" % (actual_commit or "unavailable"),
        "Checkout OpenPI commit %s before producing evidence." % expected_commit,
        {
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "diagnostic": "" if commit_command.get("ok") else _command_diagnostic(commit_command),
        },
    )

    status_command = openpi.get("tracked_status_command", {})
    tracked_status = str(openpi.get("tracked_status", ""))
    worktree_ok = bool(status_command.get("ok")) and not tracked_status.strip()
    add_check(
        "openpi_worktree_clean",
        worktree_ok,
        "OpenPI worktree has no tracked or untracked changes"
        if worktree_ok
        else "OpenPI has tracked or untracked changes",
        "Use a completely clean pinned OpenPI worktree before producing evidence.",
        {"worktree_status": tracked_status.strip()},
    )

    submodule_command = openpi.get("submodule_command", {})
    submodule_entries = openpi.get("submodules", [])
    all_submodules_ok = bool(
        submodule_command.get("ok")
        and all(entry.get("status") == " " for entry in submodule_entries)
    )
    add_check(
        "openpi_submodules_clean",
        all_submodules_ok,
        "All recursive submodules are initialized at recorded commits"
        if all_submodules_ok
        else "One or more recursive submodules are uninitialized or modified",
        "Run `git submodule update --init --recursive` and restore modified submodules.",
        {"entries": submodule_entries},
    )
    libero_entries = [
        entry for entry in submodule_entries if entry.get("path") == "third_party/libero"
    ]
    libero_ok = bool(
        submodule_command.get("ok")
        and len(libero_entries) == 1
        and libero_entries[0].get("status") == " "
    )
    add_check(
        "libero_submodule_pinned",
        libero_ok,
        "LIBERO submodule is initialized at the pinned revision"
        if libero_ok
        else "LIBERO submodule is absent, uninitialized, or at a different revision",
        "Run: git submodule update --init --recursive",
        {"entries": libero_entries},
    )

    results = facts.get("results", {})
    add_check(
        "results_directory_writable",
        bool(results.get("writable")),
        "Results directory is writable"
        if results.get("writable")
        else "Results directory is not writable",
        "Create a writable host directory for ARMBENCH_RESULTS_ROOT.",
        {"root": results.get("root"), "error": results.get("error", "")},
    )

    gpu = facts.get("gpu", {})
    nvidia_command = gpu.get("nvidia_smi_command", {})
    devices = gpu.get("devices", [])
    gpu_ok = bool(nvidia_command.get("ok") and devices)
    add_check(
        "host_nvidia_gpu",
        gpu_ok,
        "Detected %d NVIDIA GPU(s)" % len(devices),
        "Install a compatible NVIDIA driver or use a Linux NVIDIA GPU host.",
        {
            "devices": devices,
            "diagnostic": "" if nvidia_command.get("ok") else _command_diagnostic(nvidia_command),
        },
    )

    docker = facts.get("docker", {})
    version_command = docker.get("version_command", {})
    docker_ok = bool(
        version_command.get("ok")
        and docker.get("client_version")
        and docker.get("server_version")
    )
    add_check(
        "docker_engine",
        docker_ok,
        "Docker client and daemon are reachable" if docker_ok else "Docker daemon is unavailable",
        "Install Docker Engine and make its daemon accessible to the current user.",
        {
            "client_version": docker.get("client_version"),
            "server_version": docker.get("server_version"),
            "diagnostic": "" if version_command.get("ok") else _command_diagnostic(version_command),
        },
    )

    info_command = docker.get("info_command", {})
    linux_engine_ok = bool(info_command.get("ok") and str(docker.get("os_type", "")).lower() == "linux")
    add_check(
        "docker_linux_engine",
        linux_engine_ok,
        "Docker daemon uses Linux containers"
        if linux_engine_ok
        else "Docker daemon is not reporting a Linux engine",
        "Use a native Linux Docker Engine for the official host-networked benchmark.",
        {
            "os_type": docker.get("os_type"),
            "operating_system": docker.get("operating_system"),
            "architecture": docker.get("architecture"),
            "storage_driver": docker.get("storage_driver"),
            "default_runtime": docker.get("default_runtime"),
            "runtime_names": docker.get("runtime_names", []),
        },
    )

    compose_command = docker.get("compose_command", {})
    compose_ok = bool(compose_command.get("ok") and docker.get("compose_version"))
    add_check(
        "docker_compose_v2",
        compose_ok,
        "Docker Compose version is %s" % (docker.get("compose_version") or "unavailable"),
        "Install the Docker Compose v2 plugin.",
        {
            "version": docker.get("compose_version"),
            "diagnostic": "" if compose_command.get("ok") else _command_diagnostic(compose_command),
        },
    )

    container_probe = gpu.get("container_probe", {})
    container_command = container_probe.get("command", {})
    container_devices = container_probe.get("devices", [])
    container_ok = bool(
        container_probe.get("attempted") and container_command.get("ok") and container_devices
    )
    add_check(
        "container_nvidia_gpu",
        container_ok,
        "CUDA container can access %d NVIDIA GPU(s)" % len(container_devices),
        "Install NVIDIA Container Toolkit and verify `docker run --gpus all` succeeds.",
        {
            "attempted": bool(container_probe.get("attempted")),
            "image": container_probe.get("image"),
            "devices": container_devices,
            "skipped_reason": container_probe.get("skipped_reason", ""),
            "diagnostic": "" if container_command.get("ok") else _command_diagnostic(container_command),
        },
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": facts.get("collected_at_utc"),
        "ready": all(check["passed"] for check in checks if check["required"]),
        "expected_openpi_commit": expected_commit,
        "checks": checks,
        "environment": {
            "platform": dict(platform_facts),
            "openpi_root": openpi.get("root"),
            "results_root": results.get("root"),
            "gpu_devices": devices,
            "docker": {
                key: docker.get(key)
                for key in (
                    "client_version",
                    "server_version",
                    "compose_version",
                    "os_type",
                    "operating_system",
                    "architecture",
                    "storage_driver",
                    "default_runtime",
                    "runtime_names",
                    "cpu_count",
                    "memory_bytes",
                )
            },
            "container_probe_image": container_probe.get("image"),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--openpi-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--expected-openpi-commit", default=OPENPI_COMMIT)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument(
        "--skip-container-gpu-probe",
        action="store_true",
        help="Collect diagnostics without the definitive container GPU check; readiness will fail.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    facts = collect_facts(
        args.openpi_root,
        args.results_root,
        probe_container_gpu=not args.skip_container_gpu_probe,
    )
    report = evaluate_preflight(facts, expected_commit=args.expected_openpi_commit)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
