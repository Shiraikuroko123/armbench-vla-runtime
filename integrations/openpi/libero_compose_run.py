"""Orchestrate and finalize an evidence-bearing OpenPI pi0.5-LIBERO run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from integrations.openpi.preflight import collect_facts, evaluate_preflight
from integrations.openpi.validate_libero_artifact import validate_artifact


RUN_SCHEMA_VERSION = "armbench.pi05_libero_container_run.v1"
PROCESS_SCHEMA_VERSION = "armbench.compose_process.v1"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
ATTESTATION_SCHEMA_VERSION = "armbench.openpi_server_attestation.v1"
PREFLIGHT_SCHEMA_VERSION = "armbench.openpi_libero_preflight.v1"
EVALUATION_SCHEMA_VERSION = "armbench.pi05_libero_async.v1"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
ROOT_REQUIRED_FILES = (
    "preflight.json",
    "resolved_compose_config.json",
    "checkpoint_attestation.json",
    "openpi_server.log",
    "compose_up.json",
    "compose_stop.json",
    "artifact_validation.json",
    "evaluation/integrity.json",
    "evaluation/environment.json",
    "evaluation/manifest.json",
)

FORBIDDEN_LIBERO_ARGUMENTS = frozenset(
    (
        "--output-dir",
        "--host",
        "--port",
        "--openpi-root",
        "--armbench-root",
        "--expected-openpi-commit",
        "--checkpoint",
        "--server-launch-args",
        "--resize-size",
        "--num-steps-wait",
        "--allow-commit-mismatch",
        "--allow-unattested-server",
        "--continue-after-runtime-failure",
        "--fixed-refresh-interval",
    )
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: pathlib.Path, label: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        errors.append("missing %s: %s" % (label, path.name))
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append("invalid %s JSON: %s" % (label, exc))
        return None
    if not isinstance(parsed, dict):
        errors.append("%s must contain a JSON object" % label)
        return None
    return parsed


def _artifact_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact tree must not contain symlinks: %s" % path)
        if (
            not path.is_file()
            or path == root / "manifest.json"
            or path.suffix == ".tmp"
        ):
            continue
        yield path


def build_manifest_files(root: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for path in _artifact_files(root):
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


def validate_directory_manifest(
    root: pathlib.Path, expected_schema: Optional[str] = None
) -> Dict[str, Any]:
    errors: List[str] = []
    manifest = _read_json(root / "manifest.json", "manifest", errors)
    if manifest is None:
        return {"valid": False, "errors": errors, "files_checked": 0}
    if expected_schema is not None and manifest.get("schema_version") != expected_schema:
        errors.append(
            "manifest schema_version=%r, expected %r"
            % (manifest.get("schema_version"), expected_schema)
        )
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        errors.append("manifest files must be an object")
        return {"valid": False, "errors": errors, "files_checked": 0}
    try:
        actual_paths = {
            path.relative_to(root).as_posix(): path for path in _artifact_files(root)
        }
    except ValueError as exc:
        errors.append(str(exc))
        return {"valid": False, "errors": errors, "files_checked": 0}
    expected_names = set(expected_files)
    actual_names = set(actual_paths)
    for relative in sorted(expected_names - actual_names):
        errors.append("manifest-listed file is missing: %s" % relative)
    for relative in sorted(actual_names - expected_names):
        errors.append("file is not protected by manifest: %s" % relative)
    checked = 0
    for relative in sorted(expected_names & actual_names):
        record = expected_files.get(relative)
        if not isinstance(record, dict):
            errors.append("invalid manifest record: %s" % relative)
            continue
        path = actual_paths[relative]
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if record.get("bytes") != actual_size:
            errors.append("byte count mismatch: %s" % relative)
        if record.get("sha256") != actual_sha256:
            errors.append("SHA-256 mismatch: %s" % relative)
        checked += 1
    return {"valid": not errors, "errors": errors, "files_checked": checked}


def _validate_checkpoint_attestation(
    attestation: Mapping[str, Any], errors: List[str]
) -> None:
    expected = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "policy_loaded": True,
        "policy_config": "pi05_libero",
        "checkpoint_uri": DEFAULT_CHECKPOINT,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_tracked_clean": True,
        "openpi_tracked_status": "",
        "openpi_submodules_clean": True,
        "action_horizon": 10,
    }
    for key, value in expected.items():
        observed = attestation.get(key)
        if observed != value or (
            isinstance(value, bool) and type(observed) is not bool
        ):
            errors.append(
                "checkpoint attestation %s=%r, expected %r"
                % (key, observed, value)
            )
    files = attestation.get("checkpoint_files")
    if not isinstance(files, list) or not files:
        errors.append("checkpoint attestation has no file inventory")
        return
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    content_sha256 = hashlib.sha256(canonical).hexdigest()
    if attestation.get("checkpoint_content_sha256") != content_sha256:
        errors.append("checkpoint attestation inventory hash mismatch")
    sizes = [item.get("bytes") for item in files if isinstance(item, dict)]
    if len(sizes) != len(files) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sizes
    ):
        errors.append("checkpoint attestation contains invalid file sizes")
    else:
        file_count = attestation.get("checkpoint_file_count")
        total_bytes = attestation.get("checkpoint_total_bytes")
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count != len(files)
        ):
            errors.append("checkpoint attestation file count mismatch")
        if (
            isinstance(total_bytes, bool)
            or not isinstance(total_bytes, int)
            or total_bytes != sum(sizes)
            or total_bytes <= 0
        ):
            errors.append("checkpoint attestation total byte count mismatch")
    for item in files:
        if not isinstance(item, dict):
            errors.append("checkpoint attestation contains a non-object file record")
            break
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or pathlib.PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in pathlib.PurePosixPath(relative).parts)
        ):
            errors.append("checkpoint attestation contains an unsafe file path")
            break
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            errors.append("checkpoint attestation contains an invalid file digest")
            break
    paths = [item.get("path") for item in files if isinstance(item, dict)]
    if len(paths) != len(set(paths)):
        errors.append("checkpoint attestation contains duplicate file paths")
    for field in ("checkpoint_content_sha256", "server_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(attestation.get(field, ""))):
            errors.append("checkpoint attestation %s is not a SHA-256" % field)
    model_action_dim = attestation.get("model_action_dim")
    if (
        isinstance(model_action_dim, bool)
        or not isinstance(model_action_dim, int)
        or model_action_dim <= 0
    ):
        errors.append("checkpoint attestation model_action_dim must be positive")


def _portable_artifact_report(report: Any) -> Dict[str, Any]:
    value = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    value["artifact"] = "evaluation"
    return value


def evaluate_run_contents(
    run_directory: pathlib.Path,
    artifact_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    for relative in ROOT_REQUIRED_FILES:
        path = run_directory / pathlib.PurePosixPath(relative)
        if not path.is_file():
            errors.append("required run artifact is missing: %s" % relative)

    preflight = _read_json(run_directory / "preflight.json", "preflight", errors)
    if preflight is not None:
        if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
            errors.append("preflight schema is not recognized")
        if preflight.get("ready") is not True:
            errors.append("preflight did not pass")

    independently_computed = (
        dict(artifact_report)
        if artifact_report is not None
        else _portable_artifact_report(
            validate_artifact(run_directory / "evaluation")
        )
    )
    recorded_artifact_report = _read_json(
        run_directory / "artifact_validation.json",
        "artifact validation",
        errors,
    )
    if recorded_artifact_report != independently_computed:
        errors.append(
            "recorded artifact validation does not match independent recomputation"
        )
    if independently_computed.get("valid") is not True:
        validation_errors = independently_computed.get("errors")
        if isinstance(validation_errors, list) and validation_errors:
            errors.extend(
                "evaluation artifact %s" % error for error in validation_errors
            )
        else:
            errors.append("independent evaluation artifact validation failed")
    warnings = independently_computed.get("warnings")
    if isinstance(warnings, list) and warnings:
        errors.extend(
            "formal container run rejects validator warning: %s" % warning
            for warning in warnings
        )

    compose_config = _read_json(
        run_directory / "resolved_compose_config.json", "resolved compose config", errors
    )
    if compose_config is not None:
        services = compose_config.get("services", {})
        runtime_text = json.dumps(services.get("runtime", {}), sort_keys=True)
        server_text = json.dumps(services.get("openpi_server", {}), sort_keys=True)
        if "libero_runtime_eval" not in runtime_text or "/evaluation" not in runtime_text:
            errors.append("resolved runtime service does not target the evaluator subdirectory")
        if "serve_policy_attested.py" not in server_text:
            errors.append("resolved server service does not use the attested entrypoint")

    attestation = _read_json(
        run_directory / "checkpoint_attestation.json", "checkpoint attestation", errors
    )
    if attestation is not None:
        _validate_checkpoint_attestation(attestation, errors)
        snapshot = (
            run_directory
            / "evaluation"
            / "provenance"
            / "armbench_source"
            / "integrations"
            / "openpi"
            / "serve_policy_attested.py"
        )
        if snapshot.is_file():
            if attestation.get("server_source_sha256") != sha256_file(snapshot):
                errors.append("attested server source does not match the evaluation snapshot")
        else:
            errors.append("evaluation snapshot is missing the attested server source")

        environment_record = _read_json(
            run_directory / "evaluation" / "environment.json",
            "evaluation environment",
            errors,
        )
        if environment_record is not None:
            server_metadata = environment_record.get("server_metadata", {})
            observed_attestation = (
                server_metadata.get("armbench_server_attestation")
                if isinstance(server_metadata, dict)
                else None
            )
            expected_public_attestation = {
                key: value
                for key, value in attestation.items()
                if key not in {"checkpoint_files", "checkpoint_local_path"}
            }
            if observed_attestation != expected_public_attestation:
                errors.append(
                    "checkpoint attestation file does not match policy server metadata"
                )

    server_log = run_directory / "openpi_server.log"
    if server_log.is_file() and server_log.stat().st_size <= 0:
        errors.append("openpi_server.log is empty")

    compose_up = _read_json(run_directory / "compose_up.json", "compose up record", errors)
    if compose_up is not None and compose_up.get("returncode") != 0:
        errors.append("docker compose up did not exit successfully")
    compose_stop = _read_json(
        run_directory / "compose_stop.json", "compose stop record", errors
    )
    compose_stop_confirmed = False
    if compose_stop is not None:
        compose_stop_confirmed = bool(
            compose_stop.get("attempted") is True
            and compose_stop.get("returncode") == 0
        )
        if not compose_stop_confirmed:
            errors.append("docker compose stop was not confirmed successful")

    integrity = _read_json(
        run_directory / "evaluation" / "integrity.json", "evaluation integrity", errors
    )
    if integrity is not None and integrity.get("valid") is not True:
        errors.append("evaluation integrity check did not pass")
    nested = validate_directory_manifest(
        run_directory / "evaluation", expected_schema=EVALUATION_SCHEMA_VERSION
    )
    if not nested["valid"]:
        errors.extend("evaluation %s" % error for error in nested["errors"])
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "complete": not errors,
        "errors": errors,
        "compose_stop_confirmed": compose_stop_confirmed,
        "artifact_validation_valid": independently_computed.get("valid") is True,
        "evaluation_manifest_files_checked": nested["files_checked"],
    }


def finalize_run(run_directory: pathlib.Path) -> Dict[str, Any]:
    root = run_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError("run directory does not exist: %s" % root)
    artifact_report = _portable_artifact_report(
        validate_artifact(root / "evaluation")
    )
    write_json(root / "artifact_validation.json", artifact_report)
    assessment = evaluate_run_contents(root, artifact_report)
    finalization = dict(assessment)
    finalization["finalized_at_utc"] = _utc_now()
    finalization["server_log_hashed_only_after_compose_stop"] = bool(
        assessment["compose_stop_confirmed"]
    )
    write_json(root / "finalization.json", finalization)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "complete": assessment["complete"],
        "validation_errors": assessment["errors"],
        "files": build_manifest_files(root),
    }
    write_json(root / "manifest.json", manifest)
    validation = validate_directory_manifest(root, expected_schema=RUN_SCHEMA_VERSION)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "complete": bool(assessment["complete"] and validation["valid"]),
        "errors": list(assessment["errors"]) + list(validation["errors"]),
        "manifest_files": len(manifest["files"]),
        "manifest_validation": validation,
    }


def validate_run_manifest(run_directory: pathlib.Path) -> Dict[str, Any]:
    root = run_directory.resolve()
    manifest_validation = validate_directory_manifest(
        root, expected_schema=RUN_SCHEMA_VERSION
    )
    errors = list(manifest_validation["errors"])
    manifest_errors: List[str] = []
    manifest = _read_json(root / "manifest.json", "root manifest", manifest_errors)
    errors.extend(manifest_errors)
    if manifest is not None and manifest.get("complete") is not True:
        errors.append("root manifest is explicitly incomplete")
    artifact_report = _portable_artifact_report(
        validate_artifact(root / "evaluation")
    )
    contents = evaluate_run_contents(root, artifact_report)
    errors.extend(contents["errors"])
    unique_errors = list(dict.fromkeys(errors))
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "valid": not unique_errors,
        "complete": bool(
            not unique_errors
            and manifest is not None
            and manifest.get("complete") is True
        ),
        "errors": unique_errors,
        "files_checked": manifest_validation["files_checked"],
        "artifact_validation": artifact_report,
        "content_assessment": contents,
    }


def _process_record(
    argv: Sequence[str], returncode: Optional[int], started_at: str, duration_s: float,
    *, attempted: bool = True, stdout: str = "", stderr: str = ""
) -> Dict[str, Any]:
    return {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "attempted": attempted,
        "argv": list(argv),
        "started_at_utc": started_at,
        "duration_s": round(duration_s, 3),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _run_capture(
    argv: Sequence[str], cwd: pathlib.Path, environment: Mapping[str, str], timeout_s: float
) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return _process_record(
            argv,
            completed.returncode,
            started_at,
            time.perf_counter() - started,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _process_record(
            argv,
            None,
            started_at,
            time.perf_counter() - started,
            stderr=str(exc),
        )


def _stream_compose_up(
    argv: Sequence[str], cwd: pathlib.Path, environment: Mapping[str, str], log_path: pathlib.Path
) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    returncode: Optional[int] = None
    error = ""
    process: Optional[subprocess.Popen[str]] = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                returncode = 130
        except OSError as exc:
            error = str(exc)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process is not None:
                returncode = process.returncode
    return _process_record(
        argv,
        returncode,
        started_at,
        time.perf_counter() - started,
        stderr=error,
    )


def _validated_run_directory(results_root: pathlib.Path, run_id: str) -> pathlib.Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run-id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    root = results_root.resolve()
    run_directory = (root / run_id).resolve()
    if run_directory.parent != root:
        raise ValueError("run directory escaped results root")
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError("run directory must be absent or empty: %s" % run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def _compose_prefix(
    openpi_root: pathlib.Path, armbench_root: pathlib.Path, project_name: str
) -> List[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--project-directory",
        str(openpi_root),
        "-f",
        str(openpi_root / "examples" / "libero" / "compose.yml"),
        "-f",
        str(armbench_root / "integrations" / "openpi" / "compose.libero-runtime.yml"),
    ]


def _resolved_project_name(run_id: str, requested: Optional[str]) -> str:
    candidate = requested or ("armbench-%s" % run_id.lower())
    normalized = re.sub(r"[^a-z0-9_-]+", "-", candidate.lower()).strip("-_")
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError("could not derive a valid Docker Compose project name")
    return normalized[:128]


def _normalize_libero_args(value: str) -> str:
    if any(character in value for character in "\r\n;&|`$<>(){}[]*?!\\\"'"):
        raise ValueError("libero-args contains unsupported shell metacharacters")
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError as exc:
        raise ValueError("libero-args could not be parsed: %s" % exc)
    if any(any(character.isspace() for character in token) for token in tokens):
        raise ValueError("libero-args values cannot contain whitespace")
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in FORBIDDEN_LIBERO_ARGUMENTS or any(
            forbidden.startswith(option) for forbidden in FORBIDDEN_LIBERO_ARGUMENTS
        ):
            raise ValueError("libero-args cannot override protected option %s" % option)
    return " ".join(tokens)


def execute_run(args: argparse.Namespace) -> int:
    openpi_root = args.openpi_root.resolve()
    armbench_root = args.armbench_root.resolve()
    results_root = args.results_root.resolve()
    libero_args = _normalize_libero_args(args.libero_args.strip())
    if args.fixed_refresh_interval is not None:
        fixed_refresh_argument = "--fixed-refresh-interval %d" % args.fixed_refresh_interval
        libero_args = " ".join(part for part in (libero_args, fixed_refresh_argument) if part)
    run_directory = _validated_run_directory(results_root, args.run_id)
    environment = dict(os.environ)
    environment.update(
        {
            "PWD": str(openpi_root),
            "ARMBENCH_ROOT": str(armbench_root),
            "ARMBENCH_RESULTS_ROOT": str(results_root),
            "ARMBENCH_RUN_ID": args.run_id,
            "ARMBENCH_POLICY_PORT": str(args.policy_port),
            "ARMBENCH_SERVER_WAIT_ATTEMPTS": str(args.server_wait_attempts),
            "ARMBENCH_LIBERO_ARGS": libero_args,
        }
    )

    preflight = evaluate_preflight(
        collect_facts(
            openpi_root,
            results_root,
            probe_container_gpu=not args.skip_container_gpu_probe,
        )
    )
    write_json(run_directory / "preflight.json", preflight)
    project_name = _resolved_project_name(args.run_id, args.project_name)
    prefix = _compose_prefix(openpi_root, armbench_root, project_name)
    if not preflight["ready"]:
        write_json(
            run_directory / "compose_up.json",
            _process_record(
                [], None, _utc_now(), 0.0, attempted=False, stderr="preflight failed"
            ),
        )
        write_json(
            run_directory / "compose_stop.json",
            _process_record(
                [],
                None,
                _utc_now(),
                0.0,
                attempted=False,
                stderr="compose was not started",
            ),
        )
        result = finalize_run(run_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    config_command = prefix + ["config", "--format", "json"]
    config_record = _run_capture(config_command, openpi_root, environment, 120.0)
    if config_record["returncode"] != 0:
        write_json(run_directory / "compose_config_error.json", config_record)
        write_json(
            run_directory / "compose_up.json",
            _process_record(
                [],
                None,
                _utc_now(),
                0.0,
                attempted=False,
                stderr="compose config failed",
            ),
        )
        write_json(
            run_directory / "compose_stop.json",
            _process_record(
                [],
                None,
                _utc_now(),
                0.0,
                attempted=False,
                stderr="compose was not started",
            ),
        )
        result = finalize_run(run_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    try:
        resolved_config = json.loads(config_record["stdout"])
    except ValueError as exc:
        write_json(
            run_directory / "compose_config_error.json",
            dict(config_record, parse_error=str(exc)),
        )
        result = finalize_run(run_directory)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    write_json(run_directory / "resolved_compose_config.json", resolved_config)

    up_command = prefix + ["up"]
    if not args.no_build:
        up_command.append("--build")
    up_command.extend(["--abort-on-container-exit", "--exit-code-from", "runtime"])
    stop_command = prefix + ["stop", "--timeout", str(args.stop_timeout_s)]
    try:
        try:
            up_record = _stream_compose_up(
                up_command,
                openpi_root,
                environment,
                run_directory / "compose_up.log",
            )
        except Exception as exc:
            up_record = _process_record(
                up_command,
                None,
                _utc_now(),
                0.0,
                stderr="%s: %s" % (type(exc).__name__, exc),
            )
        write_json(run_directory / "compose_up.json", up_record)
    finally:
        stop_record = _run_capture(
            stop_command, openpi_root, environment, args.stop_timeout_s + 60
        )
        write_json(run_directory / "compose_stop.json", stop_record)
    result = finalize_run(run_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if up_record["returncode"] == 0 and result["complete"] else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Preflight, run, stop, and finalize Compose",
        allow_abbrev=False,
    )
    run_parser.add_argument("--openpi-root", type=pathlib.Path, required=True)
    run_parser.add_argument(
        "--armbench-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    run_parser.add_argument("--results-root", type=pathlib.Path, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--project-name")
    run_parser.add_argument("--policy-port", type=int, default=8000)
    run_parser.add_argument("--server-wait-attempts", type=int, default=360)
    run_parser.add_argument("--stop-timeout-s", type=int, default=60)
    run_parser.add_argument("--libero-args", default=os.environ.get("ARMBENCH_LIBERO_ARGS", ""))
    run_parser.add_argument("--fixed-refresh-interval", type=int)
    run_parser.add_argument("--no-build", action="store_true")
    run_parser.add_argument("--skip-container-gpu-probe", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Regenerate the root manifest after Compose has stopped",
        allow_abbrev=False,
    )
    finalize_parser.add_argument("run_directory", type=pathlib.Path)
    validate_parser = subparsers.add_parser(
        "validate", help="Validate the root run manifest", allow_abbrev=False
    )
    validate_parser.add_argument("run_directory", type=pathlib.Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        if args.policy_port <= 0 or args.policy_port > 65535:
            raise ValueError("policy-port must be between 1 and 65535")
        if args.server_wait_attempts <= 0 or args.stop_timeout_s <= 0:
            raise ValueError("wait attempts and stop timeout must be positive")
        if args.fixed_refresh_interval is not None and args.fixed_refresh_interval <= 0:
            raise ValueError("fixed-refresh-interval must be positive when provided")
        return execute_run(args)
    if args.command == "finalize":
        result = finalize_run(args.run_directory)
    else:
        result = validate_run_manifest(args.run_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("complete", result.get("valid", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
