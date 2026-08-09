"""Independently validate an attested live pi0.5-to-Panda smoke bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import imageio.v2 as imageio
import numpy as np


RUN_SCHEMA = "armbench.live_pi05_panda_smoke.v1"
VALIDATION_SCHEMA = "armbench.live_pi05_panda_smoke_validation.v1"
ATTESTATION_SCHEMA = "armbench.openpi_server_attestation.v1"
CHECKPOINT_CRC_SCHEMA = "armbench.gcs_checkpoint_crc32c.v1"
CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi05_libero"
POLICY_CONFIG = "pi05_libero"
PROVIDER_ID = "openpi_pi05_libero_live"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CHECKPOINT_FILE_COUNT = 16
CHECKPOINT_TOTAL_BYTES = 12_439_085_481
RUN_FILES = {
    "events.json",
    "panda_trace.mp4",
    "summary.json",
    "trace.npz",
}
TRACE_ARRAYS = {
    "scheduled_wall_times_s": (None,),
    "actual_wall_times_s": (None,),
    "simulated_times_s": (None,),
    "desired_positions": (None, 7),
    "actual_positions": (None, 7),
    "command_velocities": (None, 7),
    "command_statuses": (None,),
    "observation_ages_ms": (None,),
    "request_ids": (None,),
    "action_indices": (None,),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==\Z")


class _Collector:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks: list[str] = []
        self.metrics: dict[str, object] = {}

    def error(self, section: str, message: str) -> None:
        self.errors.append(f"[{section}] {message}")

    def checked(self, message: str) -> None:
        self.checks.append(message)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path, section: str, collector: _Collector) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        collector.error(section, f"cannot read strict JSON {path.name}: {error}")
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, section: str, collector: _Collector) -> dict[str, Any]:
    if not isinstance(value, dict):
        collector.error(section, "expected a JSON object")
        return {}
    return value


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _safe_direct_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.as_posix() != value:
        return None
    return value


def _validate_run_manifest(run: Path, collector: _Collector) -> None:
    manifest = _mapping(
        _read_json(run / "manifest.json", "manifest", collector),
        "manifest",
        collector,
    )
    if manifest.get("schema_version") != RUN_SCHEMA:
        collector.error("manifest", "schema_version mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        collector.error("manifest", "files must be an array")
        return
    paths: list[str] = []
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(entries):
        section = f"manifest.files[{index}]"
        entry = _mapping(raw, section, collector)
        relative = _safe_direct_path(entry.get("path"))
        if relative is None:
            collector.error(section, "path must be one safe direct child")
            continue
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            collector.error(section, "size_bytes must be a nonnegative integer")
            continue
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            collector.error(section, "sha256 must be lowercase hexadecimal")
            continue
        path = run / relative
        if not path.is_file():
            collector.error(section, f"missing file: {relative}")
            continue
        if path.stat().st_size != size:
            collector.error(section, f"size mismatch: {relative}")
        if _sha256_file(path) != digest:
            collector.error(section, f"SHA-256 mismatch: {relative}")
        paths.append(relative)
        normalized.append(
            {"path": relative, "size_bytes": size, "sha256": digest}
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        collector.error("manifest", "file inventory must be sorted and unique")
    if set(paths) != RUN_FILES:
        collector.error("manifest", f"file set mismatch: {sorted(set(paths))}")
    actual = {path.name for path in run.iterdir() if path.is_file()}
    if actual != RUN_FILES | {"manifest.json"}:
        collector.error("manifest", f"unexpected run files: {sorted(actual)}")
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode()
    expected_inventory = hashlib.sha256(canonical).hexdigest()
    if manifest.get("inventory_sha256") != expected_inventory:
        collector.error("manifest", "inventory_sha256 mismatch")
    collector.checked("run manifest hashes and exact file inventory")


def _validate_summary(
    run: Path, collector: _Collector
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _mapping(
        _read_json(run / "summary.json", "summary", collector),
        "summary",
        collector,
    )
    expected = {
        "schema_version": RUN_SCHEMA,
        "scope": "attested_live_pi05_libero_to_panda_mujoco_smoke",
        "provider_id": PROVIDER_ID,
        "response_origin": "live_checkpoint_inference",
        "scripted_policy": False,
        "policy_checkpoint_executed": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            collector.error("summary", f"{key} mismatch")
    response_sha = summary.get("response_sha256")
    if not isinstance(response_sha, str) or _SHA256.fullmatch(response_sha) is None:
        collector.error("summary", "response_sha256 is not a SHA-256 digest")
        response_sha = ""

    identity = _mapping(
        _mapping(
            summary.get("policy_provenance"),
            "summary.policy_provenance",
            collector,
        ).get("identity"),
        "summary.policy_provenance.identity",
        collector,
    )
    identity_expected = {
        "provider_id": PROVIDER_ID,
        "implementation_revision": OPENPI_COMMIT,
        "checkpoint_reference": CHECKPOINT_URI,
        "checkpoint_identity_status": "content_attested",
        "response_origin": "live_checkpoint_inference",
        "checkpoint_executed_this_run": True,
    }
    for key, value in identity_expected.items():
        if identity.get(key) != value:
            collector.error("identity", f"{key} mismatch")
    checkpoint_sha = identity.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or _SHA256.fullmatch(checkpoint_sha) is None:
        collector.error("identity", "checkpoint_sha256 is invalid")

    response = _mapping(
        summary.get("last_policy_response"),
        "summary.last_policy_response",
        collector,
    )
    if response.get("response_sha256") != response_sha:
        collector.error("response", "last response digest differs from summary")
    if response.get("provider_id") != PROVIDER_ID:
        collector.error("response", "provider_id mismatch")
    for key in (
        "provider_inference_latency_ms",
        "adapter_latency_ms",
        "total_latency_ms",
    ):
        if not _finite_nonnegative(response.get(key)):
            collector.error("response", f"{key} is invalid")

    episode = _mapping(summary.get("episode"), "summary.episode", collector)
    episode_expected = {
        "scenario": "free_space",
        "mode": "braking_invariant",
        "policy_checkpoint_executed": True,
        "scripted_policy": False,
        "physics_executed": True,
        "hard_realtime_claim": False,
        "physical_safety_claim": False,
        "separate_policy_thread": True,
        "physical_safe": True,
    }
    for key, value in episode_expected.items():
        if episode.get(key) != value:
            collector.error("episode", f"{key} mismatch")
    policy_source = episode.get("policy_source")
    if not isinstance(policy_source, str) or response_sha not in policy_source:
        collector.error("episode", "policy_source lacks the final response digest")
    for key in ("control_ticks", "control_ticks_during_inference", "accepted_responses"):
        if type(episode.get(key)) is not int or int(episode[key]) <= 0:
            collector.error("episode", f"{key} must be a positive integer")
    for key in (
        "obstacle_contact_steps",
        "self_contact_steps",
        "joint_limit_violation_steps",
        "abrupt_stop_violations",
        "unsafe_prepared_plans",
        "repair_selection_deadline_exceedances",
        "policy_failures",
    ):
        if episode.get(key) != 0:
            collector.error("episode", f"{key} must equal zero")
    for key in (
        "mean_policy_latency_ms",
        "p95_policy_latency_ms",
        "max_policy_latency_ms",
        "p95_control_tick_lateness_ms",
        "max_control_tick_lateness_ms",
    ):
        if not _finite_nonnegative(episode.get(key)):
            collector.error("episode", f"{key} is invalid")

    protocol = _mapping(summary.get("protocol"), "summary.protocol", collector)
    if protocol.get("scenario") != episode.get("scenario"):
        collector.error("protocol", "scenario differs from episode")
    if protocol.get("mode") != episode.get("mode"):
        collector.error("protocol", "mode differs from episode")
    if protocol.get("action_horizon") != 10:
        collector.error("protocol", "action_horizon must equal 10")
    reference_steps = protocol.get("reference_steps")
    extra_steps = protocol.get("extra_action_steps")
    max_steps = protocol.get("max_action_steps")
    if not all(type(value) is int and value >= 0 for value in (reference_steps, extra_steps)):
        collector.error("protocol", "step counts are invalid")
    elif max_steps != reference_steps + extra_steps:
        collector.error("protocol", "max_action_steps does not match step counts")
    if not _finite_nonnegative(protocol.get("deadline_ms")):
        collector.error("protocol", "deadline_ms is invalid")

    implementation = _mapping(
        summary.get("implementation"), "summary.implementation", collector
    )
    commit = implementation.get("armbench_commit")
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        collector.error("implementation", "armbench_commit is invalid")
    if implementation.get("armbench_tracked_clean") is not True:
        collector.error("implementation", "tracked worktree was not clean")
    files = implementation.get("implementation_files")
    if not isinstance(files, list) or len(files) < 6:
        collector.error("implementation", "implementation file inventory is incomplete")
    else:
        paths = []
        for entry in files:
            if not isinstance(entry, dict):
                collector.error("implementation", "file record is not an object")
                continue
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str) or not path:
                collector.error("implementation", "file path is invalid")
            else:
                paths.append(path)
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                collector.error("implementation", "file SHA-256 is invalid")
        if len(paths) != len(set(paths)):
            collector.error("implementation", "file paths are duplicated")

    collector.metrics.update(
        {
            "armbench_commit": commit,
            "checkpoint_sha256": checkpoint_sha,
            "response_sha256": response_sha,
            "accepted_responses": episode.get("accepted_responses"),
            "mean_policy_latency_ms": episode.get("mean_policy_latency_ms"),
            "p95_policy_latency_ms": episode.get("p95_policy_latency_ms"),
            "control_ticks": episode.get("control_ticks"),
            "control_ticks_during_inference": episode.get(
                "control_ticks_during_inference"
            ),
        }
    )
    collector.checked("live provider identity, response digest, protocol, and claim flags")
    return summary, identity


def _validate_attestation(
    bundle: Path,
    summary: dict[str, Any],
    identity: dict[str, Any],
    collector: _Collector,
) -> dict[str, Any]:
    attestation = _mapping(
        _read_json(
            bundle / "server" / "checkpoint_attestation.json",
            "attestation",
            collector,
        ),
        "attestation",
        collector,
    )
    expected = {
        "schema_version": ATTESTATION_SCHEMA,
        "policy_loaded": True,
        "policy_config": POLICY_CONFIG,
        "checkpoint_uri": CHECKPOINT_URI,
        "checkpoint_file_count": CHECKPOINT_FILE_COUNT,
        "checkpoint_total_bytes": CHECKPOINT_TOTAL_BYTES,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_tracked_clean": True,
        "openpi_submodules_clean": True,
        "action_horizon": 10,
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            collector.error("attestation", f"{key} mismatch")
    checkpoint_sha = attestation.get("checkpoint_content_sha256")
    if checkpoint_sha != identity.get("checkpoint_sha256"):
        collector.error("attestation", "checkpoint digest differs from identity")
    files = attestation.get("checkpoint_files")
    if not isinstance(files, list) or len(files) != CHECKPOINT_FILE_COUNT:
        collector.error("attestation", "checkpoint file inventory is incomplete")
    else:
        canonical = json.dumps(
            files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        if hashlib.sha256(canonical).hexdigest() != checkpoint_sha:
            collector.error("attestation", "checkpoint inventory digest mismatch")
        sizes = [entry.get("bytes") for entry in files if isinstance(entry, dict)]
        if (
            len(sizes) != CHECKPOINT_FILE_COUNT
            or any(type(size) is not int or size < 0 for size in sizes)
            or sum(sizes) != CHECKPOINT_TOTAL_BYTES
        ):
            collector.error("attestation", "checkpoint inventory sizes mismatch")

    metadata = _mapping(
        summary.get("server_metadata"), "summary.server_metadata", collector
    )
    public = _mapping(
        metadata.get("armbench_server_attestation"),
        "summary.server_metadata.armbench_server_attestation",
        collector,
    )
    for key in expected:
        if public.get(key) != attestation.get(key):
            collector.error("attestation", f"public metadata differs for {key}")
    if public.get("checkpoint_content_sha256") != checkpoint_sha:
        collector.error("attestation", "public checkpoint digest mismatch")
    log_path = bundle / "server" / "openpi_server.log"
    try:
        log = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        collector.error("server_log", f"cannot read server log: {error}")
    else:
        if f"Serving attested policy checkpoint_sha256={checkpoint_sha}" not in log:
            collector.error("server_log", "serving checkpoint line is absent")
        if "server listening on 0.0.0.0:8000" not in log:
            collector.error("server_log", "listen line is absent")
        if "Traceback (most recent call last)" in log:
            collector.error("server_log", "server log contains a traceback")
    collector.checked("external checkpoint attestation and public server metadata")
    return attestation


def _validate_crc32c(
    bundle: Path, attestation: dict[str, Any], collector: _Collector
) -> None:
    crc = _mapping(
        _read_json(
            bundle / "checkpoint_crc32c_verification.json",
            "crc32c",
            collector,
        ),
        "crc32c",
        collector,
    )
    expected = {
        "schema_version": CHECKPOINT_CRC_SCHEMA,
        "checkpoint_uri": CHECKPOINT_URI,
        "file_count": CHECKPOINT_FILE_COUNT,
        "total_bytes": CHECKPOINT_TOTAL_BYTES,
        "all_crc32c_matched": True,
        "errors": [],
    }
    for key, value in expected.items():
        if crc.get(key) != value:
            collector.error("crc32c", f"{key} mismatch")
    records = crc.get("records")
    files = attestation.get("checkpoint_files")
    attested_sizes = {
        entry["path"]: entry["bytes"]
        for entry in files
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and type(entry.get("bytes")) is int
    } if isinstance(files, list) else {}
    if not isinstance(records, list) or len(records) != CHECKPOINT_FILE_COUNT:
        collector.error("crc32c", "record inventory is incomplete")
        return
    observed: dict[str, int] = {}
    for index, record in enumerate(records):
        section = f"crc32c.records[{index}]"
        if not isinstance(record, dict):
            collector.error(section, "record is not an object")
            continue
        path = record.get("path")
        size = record.get("size_bytes")
        checksum = record.get("crc32c")
        if not isinstance(path, str) or not path:
            collector.error(section, "path is invalid")
            continue
        if path in observed:
            collector.error(section, "path is duplicated")
        if type(size) is not int or size < 0:
            collector.error(section, "size_bytes is invalid")
            continue
        observed[path] = size
        if record.get("expected_size_bytes") != size:
            collector.error(section, "expected size differs from measured size")
        if record.get("matched") is not True:
            collector.error(section, "record is not marked matched")
        if checksum != record.get("expected_crc32c"):
            collector.error(section, "CRC32C differs from expected value")
        if not isinstance(checksum, str) or _CRC32C.fullmatch(checksum) is None:
            collector.error(section, "CRC32C encoding is invalid")
        else:
            try:
                if len(base64.b64decode(checksum, validate=True)) != 4:
                    collector.error(section, "CRC32C must decode to four bytes")
            except ValueError:
                collector.error(section, "CRC32C is not valid base64")
    if observed != attested_sizes:
        collector.error("crc32c", "GCS and attested path/size inventories differ")
    collector.checked("16-object GCS CRC32C inventory matches checkpoint attestation")


def _validate_trace(run: Path, summary: dict[str, Any], collector: _Collector) -> None:
    try:
        with np.load(run / "trace.npz", allow_pickle=False) as trace:
            if set(trace.files) != set(TRACE_ARRAYS):
                collector.error("trace", f"array set mismatch: {sorted(trace.files)}")
                return
            control_ticks = summary.get("episode", {}).get("control_ticks")
            for name, shape in TRACE_ARRAYS.items():
                array = trace[name]
                expected_shape = tuple(
                    control_ticks if value is None else value for value in shape
                )
                if array.shape != expected_shape:
                    collector.error("trace", f"{name} shape mismatch")
            for name in (
                "scheduled_wall_times_s",
                "actual_wall_times_s",
                "simulated_times_s",
            ):
                values = trace[name]
                if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
                    collector.error("trace", f"{name} is not finite/strictly increasing")
            for name in (
                "desired_positions",
                "actual_positions",
                "command_velocities",
            ):
                if not np.all(np.isfinite(trace[name])):
                    collector.error("trace", f"{name} contains non-finite values")
            ages = trace["observation_ages_ms"]
            finite_ages = ages[np.isfinite(ages)]
            if len(finite_ages) == 0 or np.any(finite_ages < 0.0):
                collector.error("trace", "observation ages are absent or negative")
            statuses = trace["command_statuses"]
            if not np.any(np.char.startswith(statuses, "execute:")):
                collector.error("trace", "no execute command was recorded")
    except (OSError, ValueError, KeyError) as error:
        collector.error("trace", f"cannot validate trace.npz: {error}")
    collector.checked("trace shapes, monotonic clocks, finite state, and execution status")


def _validate_events(run: Path, summary: dict[str, Any], collector: _Collector) -> None:
    events = _read_json(run / "events.json", "events", collector)
    if not isinstance(events, list) or not events:
        collector.error("events", "events must be a non-empty array")
        return
    if any(not isinstance(event, dict) for event in events):
        collector.error("events", "every event must be an object")
        return
    policy_outcomes = [event for event in events if event.get("event") == "policy_outcome"]
    accepted = [
        event
        for event in policy_outcomes
        if event.get("dispatch_status") == "accepted"
    ]
    episode = _mapping(summary.get("episode"), "summary.episode", collector)
    if len(accepted) != episode.get("accepted_responses"):
        collector.error("events", "accepted response count differs from summary")
    if any(event.get("succeeded") is not True for event in accepted):
        collector.error("events", "an accepted policy response did not succeed")
    if not any(
        isinstance(event.get("policy_source"), str)
        and re.search(r"response_sha256:[0-9a-f]{64}\Z", event["policy_source"])
        for event in accepted
    ):
        collector.error("events", "accepted live response digest is absent")
    submissions = [event for event in events if event.get("event") == "policy_submission"]
    if len(submissions) < len(policy_outcomes):
        collector.error("events", "more policy outcomes than submissions")
    collector.metrics["event_count"] = len(events)
    collector.checked("event lifecycle and accepted-response aggregate")


def _validate_video(run: Path, collector: _Collector) -> None:
    path = run / "panda_trace.mp4"
    frame_count = 0
    first: np.ndarray | None = None
    last: np.ndarray | None = None
    minimum_std = math.inf
    try:
        reader = imageio.get_reader(path)
        try:
            for frame in reader:
                pixels = np.asarray(frame)
                if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
                    collector.error("video", "frame shape or dtype mismatch")
                    break
                if first is None:
                    first = pixels.copy()
                last = pixels.copy()
                minimum_std = min(minimum_std, float(np.std(pixels)))
                frame_count += 1
        finally:
            reader.close()
    except (OSError, RuntimeError, ValueError) as error:
        collector.error("video", f"cannot decode MP4: {error}")
        return
    if frame_count < 2 or first is None or last is None:
        collector.error("video", "video has fewer than two decodable frames")
    else:
        if minimum_std <= 10.0:
            collector.error("video", "at least one frame is visually blank")
        motion = float(np.mean(np.abs(first.astype(float) - last.astype(float))))
        if motion <= 1.0:
            collector.error("video", "first/last frames do not show visible motion")
        collector.metrics["video_frames"] = frame_count
        collector.metrics["video_first_last_mean_abs_difference"] = motion
    collector.checked("MP4 decodes, is nonblank, and contains visible motion")


def validate_bundle(bundle: Path) -> dict[str, object]:
    root = bundle.resolve()
    collector = _Collector()
    if not root.is_dir():
        collector.error("bundle", "artifact directory does not exist")
    run = root / "run"
    if not run.is_dir():
        collector.error("bundle", "run directory is missing")
    if collector.errors:
        return {
            "schema_version": VALIDATION_SCHEMA,
            "artifact": bundle.as_posix(),
            "valid": False,
            "errors": collector.errors,
            "checks": collector.checks,
            "metrics": collector.metrics,
        }
    _validate_run_manifest(run, collector)
    summary, identity = _validate_summary(run, collector)
    attestation = _validate_attestation(
        root, summary, identity, collector
    )
    _validate_crc32c(root, attestation, collector)
    _validate_trace(run, summary, collector)
    _validate_events(run, summary, collector)
    _validate_video(run, collector)
    return {
        "schema_version": VALIDATION_SCHEMA,
        "artifact": bundle.as_posix(),
        "valid": not collector.errors,
        "errors": collector.errors,
        "checks": collector.checks,
        "metrics": collector.metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_bundle(args.bundle)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    if args.json:
        print(encoded, end="")
    else:
        print("valid" if report["valid"] else "invalid")
        for error in report["errors"]:
            print(error)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
