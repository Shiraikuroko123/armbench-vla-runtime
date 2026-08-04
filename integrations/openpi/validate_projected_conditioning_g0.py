"""Validate a projected-conditioning G0 artifact and its code identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any, Mapping, Optional, Sequence

from integrations.openpi.projected_conditioning_g0 import (
    MAX_PEAK_DEVICE_BYTES,
    MAX_WARM_MODEL_P95_RATIO,
    SCHEMA_VERSION,
    _canonical_source_sha256,
)


class G0ValidationError(ValueError):
    pass


def _load_json(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                G0ValidationError(f"nonfinite JSON token: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise G0ValidationError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise G0ValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise G0ValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G0ValidationError(f"{name} must be finite")
    return result


def validate_directory(
    evidence_directory: pathlib.Path,
    *,
    project_root: pathlib.Path | None = None,
) -> Mapping[str, Any]:
    evidence_directory = evidence_directory.resolve()
    project_root = (
        pathlib.Path(__file__).resolve().parents[2]
        if project_root is None
        else project_root.resolve()
    )
    manifest = _load_json(evidence_directory / "manifest.json")
    if set(manifest) != {
        "schema_version",
        "run_id",
        "report",
        "checkpoint_content_sha256",
        "openpi_extension_commit",
        "openpi_extension_manifest",
    }:
        raise G0ValidationError("evidence manifest fields do not match the contract")
    if manifest.get("schema_version") != "armbench.evidence_manifest.v1":
        raise G0ValidationError("evidence manifest schema mismatch")
    report_meta = manifest.get("report")
    if not isinstance(report_meta, Mapping) or set(report_meta) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise G0ValidationError("report manifest is invalid")
    report_path = evidence_directory / str(report_meta["path"])
    if not report_path.is_file():
        raise G0ValidationError("G0 report is missing")
    if report_path.stat().st_size != report_meta.get("bytes"):
        raise G0ValidationError("G0 report byte count mismatch")
    if _sha256_file(report_path) != report_meta.get("sha256"):
        raise G0ValidationError("G0 report SHA-256 mismatch")

    extension_manifest_path = project_root / str(
        manifest["openpi_extension_manifest"]
    )
    extension = _load_json(extension_manifest_path)
    patch_meta = extension.get("patch")
    if not isinstance(patch_meta, Mapping):
        raise G0ValidationError("OpenPI patch manifest is invalid")
    patch_path = extension_manifest_path.parent / str(patch_meta.get("path"))
    if patch_path.stat().st_size != patch_meta.get("bytes"):
        raise G0ValidationError("OpenPI patch byte count mismatch")
    if _sha256_file(patch_path) != patch_meta.get("sha256"):
        raise G0ValidationError("OpenPI patch SHA-256 mismatch")
    if extension.get("extension_commit") != manifest.get("openpi_extension_commit"):
        raise G0ValidationError("OpenPI extension commit mismatch")

    report = _load_json(report_path)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("passed") is not True:
        raise G0ValidationError("G0 report did not pass the expected schema")
    identity = report.get("identity")
    gates = report.get("gates")
    contract = report.get("contract")
    latency = report.get("warm_latency")
    memory = report.get("device_memory")
    if not all(isinstance(value, Mapping) for value in (identity, gates, contract, latency, memory)):
        raise G0ValidationError("G0 report sections are missing")
    if identity.get("checkpoint_content_sha256") != manifest.get("checkpoint_content_sha256"):
        raise G0ValidationError("checkpoint content hash mismatch")
    if identity.get("openpi_commit") != extension.get("extension_commit"):
        raise G0ValidationError("report OpenPI commit mismatch")
    if identity.get("openpi_status") != "":
        raise G0ValidationError("report OpenPI worktree was not clean")
    if identity.get("source_sha256") != extension.get("production_source_sha256"):
        raise G0ValidationError("OpenPI production source hashes mismatch")
    gate_source = project_root / "integrations/openpi/projected_conditioning_g0.py"
    if identity.get("gate_source_canonical_sha256") != _canonical_source_sha256(gate_source):
        raise G0ValidationError("G0 source hash mismatch")
    if contract.get("raw_action_shape") != [10, 7] or contract.get("model_action_shape") != [10, 32]:
        raise G0ValidationError("G0 action shape contract mismatch")

    for field in (
        "baseline_exact_repeatability",
        "empty_mask_exact_parity",
        "latency_gate",
        "memory_gate",
    ):
        if gates.get(field) is not True:
            raise G0ValidationError(f"G0 gate failed: {field}")
    if _require_finite(gates.get("max_model_residual"), "max_model_residual") >= 1e-6:
        raise G0ValidationError("model residual gate failed")
    if _require_finite(gates.get("max_raw_prefix_residual"), "max_raw_prefix_residual") >= 1e-6:
        raise G0ValidationError("raw residual gate failed")
    ratio = _require_finite(gates.get("warm_model_p95_ratio"), "warm_model_p95_ratio")
    if gates.get("max_warm_model_p95_ratio") != MAX_WARM_MODEL_P95_RATIO:
        raise G0ValidationError("latency threshold was not prespecified")
    if ratio > MAX_WARM_MODEL_P95_RATIO:
        raise G0ValidationError("latency ratio gate failed")
    if gates.get("max_peak_device_bytes") != MAX_PEAK_DEVICE_BYTES:
        raise G0ValidationError("memory threshold was not prespecified")
    peak = _require_finite(memory.get("peak_bytes_in_use"), "peak_bytes_in_use")
    if peak > MAX_PEAK_DEVICE_BYTES:
        raise G0ValidationError("device memory gate failed")

    baseline_p95 = _require_finite(
        latency["baseline_model"].get("p95_ms"), "baseline_model.p95_ms"
    )
    conditioned_p95 = _require_finite(
        latency["conditioned_model"].get("p95_ms"), "conditioned_model.p95_ms"
    )
    if latency["baseline_model"].get("count") != 20 or latency["conditioned_model"].get("count") != 20:
        raise G0ValidationError("G0 warm inference count mismatch")
    if not math.isclose(conditioned_p95 / baseline_p95, ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise G0ValidationError("reported latency ratio is inconsistent")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evidence_directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_directory(pathlib.Path(args.evidence_directory))
    print(
        "projected-conditioning G0 valid: "
        f"commit={report['identity']['openpi_commit']} "
        f"ratio={report['gates']['warm_model_p95_ratio']:.4f} "
        f"peak_bytes={report['device_memory']['peak_bytes_in_use']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
