"""Independently validate a pi0.5 RTC-guidance G0 evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any, Mapping, Optional, Sequence


EVIDENCE_SCHEMA_VERSION = "armbench.pi05_rtc_guidance_g0_evidence.v1"
REPORT_SCHEMA_VERSION = "armbench.pi05_rtc_guidance_g0.v1"
METHOD = "rtc_pseudoinverse_guidance"


class G0ValidationError(ValueError):
    pass


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                G0ValidationError("nonfinite JSON token: %s" % token)
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise G0ValidationError("could not load JSON: %s" % path) from exc


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise G0ValidationError("could not hash file: %s" % path) from exc
    return digest.hexdigest()


def _canonical_source_sha256(path: pathlib.Path) -> str:
    try:
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
    except OSError as exc:
        raise G0ValidationError("could not read source: %s" % path) from exc
    return hashlib.sha256(canonical).hexdigest()


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G0ValidationError("%s must be numeric" % name)
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise G0ValidationError("%s is outside its finite range" % name)
    return number


def validate_artifact(
    evidence_directory: pathlib.Path,
    *,
    repository_root: Optional[pathlib.Path] = None,
) -> Mapping[str, Any]:
    evidence = evidence_directory.resolve()
    root = (
        pathlib.Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve()
    )
    manifest = _load_json(evidence / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise G0ValidationError("evidence manifest must be an object")
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise G0ValidationError("evidence schema mismatch")

    report_entry = manifest.get("report")
    if not isinstance(report_entry, Mapping):
        raise G0ValidationError("report manifest entry is missing")
    if set(report_entry) != {"path", "bytes", "sha256"}:
        raise G0ValidationError("report manifest fields do not match the contract")
    report_path = (evidence / str(report_entry["path"])).resolve()
    if report_path.parent != evidence:
        raise G0ValidationError("report path escapes the evidence directory")
    if report_path.stat().st_size != report_entry.get("bytes"):
        raise G0ValidationError("report byte count mismatch")
    if _sha256_file(report_path) != report_entry.get("sha256"):
        raise G0ValidationError("report SHA-256 mismatch")
    report = _load_json(report_path)
    if not isinstance(report, Mapping):
        raise G0ValidationError("G0 report must be an object")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise G0ValidationError("G0 report schema mismatch")
    if report.get("method") != METHOD or report.get("passed") is not True:
        raise G0ValidationError("G0 report did not record a passing RTC method")

    identity = report.get("identity")
    gates = report.get("gates")
    memory = report.get("device_memory")
    if not all(isinstance(value, Mapping) for value in (identity, gates, memory)):
        raise G0ValidationError("G0 identity, gates, or memory is missing")
    if identity.get("openpi_commit") != manifest.get("openpi_extension_commit"):
        raise G0ValidationError("OpenPI extension identity mismatch")
    if identity.get("rtc_reference_commit") != manifest.get("rtc_reference_commit"):
        raise G0ValidationError("RTC reference identity mismatch")
    for gate in (
        "baseline_exact_repeatability",
        "guided_exact_repeatability",
        "zero_weight_exact_parity",
        "finite_guided_actions",
        "weighted_model_residual_reduction",
        "latency_gate",
        "memory_gate",
    ):
        if gates.get(gate) is not True:
            raise G0ValidationError("required gate is not true: %s" % gate)

    baseline_rmse = _finite_number(
        gates.get("baseline_weighted_model_rmse"),
        "baseline weighted model RMSE",
        positive=True,
    )
    guided_rmse = _finite_number(
        gates.get("guided_weighted_model_rmse"),
        "guided weighted model RMSE",
        positive=True,
    )
    residual_ratio = _finite_number(
        gates.get("weighted_model_residual_ratio"),
        "weighted model residual ratio",
        positive=True,
    )
    if not guided_rmse < baseline_rmse:
        raise G0ValidationError("guided residual did not improve on baseline")
    if not math.isclose(
        guided_rmse / baseline_rmse,
        residual_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise G0ValidationError("weighted residual ratio was not reproducible")

    p95 = _finite_number(gates.get("warm_wall_p95_ms"), "warm wall P95", positive=True)
    max_p95 = _finite_number(
        gates.get("max_warm_wall_p95_ms"), "maximum warm wall P95", positive=True
    )
    ratio = _finite_number(
        gates.get("warm_wall_p95_ratio"), "warm wall P95 ratio", positive=True
    )
    max_ratio = _finite_number(
        gates.get("max_warm_wall_p95_ratio"),
        "maximum warm wall P95 ratio",
        positive=True,
    )
    if p95 > max_p95 or ratio > max_ratio:
        raise G0ValidationError("latency values exceed the recorded gates")
    peak = _finite_number(memory.get("peak_bytes_in_use"), "peak device bytes", positive=True)
    max_peak = _finite_number(
        gates.get("max_peak_device_bytes"), "maximum peak device bytes", positive=True
    )
    if peak > max_peak:
        raise G0ValidationError("device memory exceeds the recorded gate")

    extension_entry = manifest.get("extension_manifest")
    if not isinstance(extension_entry, Mapping):
        raise G0ValidationError("extension manifest entry is missing")
    extension_path = (root / str(extension_entry.get("path"))).resolve()
    if _sha256_file(extension_path) != extension_entry.get("sha256"):
        raise G0ValidationError("extension manifest SHA-256 mismatch")
    extension = _load_json(extension_path)
    if extension.get("extension_commit") != identity.get("openpi_commit"):
        raise G0ValidationError("extension manifest commit mismatch")
    if extension.get("rtc_reference_commit") != identity.get("rtc_reference_commit"):
        raise G0ValidationError("extension manifest RTC reference mismatch")
    if extension.get("production_source_sha256") != identity.get("source_sha256"):
        raise G0ValidationError("production source hashes mismatch")
    gate_source = root / "integrations/openpi/rtc_guidance_g0.py"
    if _canonical_source_sha256(gate_source) != identity.get(
        "gate_source_canonical_sha256"
    ):
        raise G0ValidationError("G0 source hash mismatch")

    return {
        "valid": True,
        "report_sha256": report_entry["sha256"],
        "openpi_commit": identity["openpi_commit"],
        "checkpoint_content_sha256": identity["checkpoint_content_sha256"],
        "weighted_model_residual_ratio": residual_ratio,
        "warm_wall_p95_ms": p95,
        "peak_bytes_in_use": int(peak),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence_directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_artifact(pathlib.Path(args.evidence_directory))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
