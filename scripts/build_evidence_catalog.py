"""Build and verify the repository-wide read-only evidence catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from collections import Counter
from typing import Any, Mapping, Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "configs" / "evidence_catalog.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "docs" / "evidence_catalog.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "docs" / "EVIDENCE_CATALOG.md"
SOURCE_SCHEMA = "armbench.evidence_catalog_source.v1"
OUTPUT_SCHEMA = "armbench.evidence_catalog.v1"
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]*$")
VALIDATOR_STATUSES = frozenset(
    {"available", "available_with_limitation", "not_available"}
)


def _strict_json(path: pathlib.Path) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict catalog source {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("catalog source must be a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _repository_path(value: Any, label: str) -> tuple[str, pathlib.Path]:
    raw = _require_string(value, label)
    pure = pathlib.PurePosixPath(raw)
    if (
        raw != pure.as_posix()
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or "\\" in raw
    ):
        raise ValueError(f"{label} must be a normalized repository-relative path")
    path = (PROJECT_ROOT / pathlib.Path(*pure.parts)).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} does not resolve to a regular file: {raw}")
    return raw, path


def _tracked_evidence_files() -> dict[str, list[tuple[pathlib.Path, str, int]]]:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "--stage",
                "-z",
                "--",
                "evidence",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "catalog inventory requires a Git checkout with git available"
        ) from exc
    records: list[tuple[str, str, pathlib.PurePosixPath]] = []
    object_ids: list[str] = []
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            pure = pathlib.PurePosixPath(raw_path.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("Git returned malformed evidence index metadata") from exc
        if mode not in {"100644", "100755"} or stage != "0":
            raise ValueError(f"unsupported evidence index entry: {pure}")
        if len(pure.parts) < 3 or pure.parts[0] != "evidence":
            raise ValueError(f"unexpected tracked evidence path: {pure}")
        records.append((mode, object_id, pure))
        object_ids.append(object_id)

    try:
        sizes_result = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "cat-file",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            input=("\n".join(object_ids) + "\n").encode("ascii"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect Git evidence blobs") from exc
    sizes: dict[str, int] = {}
    try:
        for line in sizes_result.stdout.decode("ascii").splitlines():
            object_id, object_type, raw_size = line.split()
            if object_type != "blob":
                raise ValueError(f"evidence object is not a blob: {object_id}")
            sizes[object_id] = int(raw_size)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Git returned malformed evidence blob metadata") from exc

    grouped: dict[str, list[tuple[pathlib.Path, str, int]]] = {}
    for _mode, object_id, pure in records:
        path = (PROJECT_ROOT / pathlib.Path(*pure.parts)).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"tracked evidence path is not a regular file: {pure}")
        if object_id not in sizes:
            raise ValueError(f"missing Git blob metadata for evidence path: {pure}")
        grouped.setdefault(pure.parts[1], []).append(
            (path, object_id, sizes[object_id])
        )
    return grouped


def _artifact_inventory(
    root: pathlib.Path, tracked_files: Sequence[tuple[pathlib.Path, str, int]]
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"artifact root is not a regular directory: {root}")
    files = list(tracked_files)
    if not files:
        raise ValueError(f"artifact has no Git-tracked files: {root}")
    files.sort(key=lambda item: item[0].relative_to(root).as_posix())
    digest = hashlib.sha256()
    total_bytes = 0
    for path, object_id, size in files:
        relative = path.relative_to(root).as_posix()
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(object_id.encode("ascii"))
        digest.update(b"\n")
    return {
        "fingerprint_basis": "sha256_over_git_blob_ids_paths_and_sizes",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _validate_validator(value: Any, artifact_id: str) -> dict[str, Any]:
    label = f"{artifact_id}.validator"
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    status = _require_string(value.get("status"), f"{label}.status")
    if status not in VALIDATOR_STATUSES:
        raise ValueError(f"{label}.status is unsupported: {status}")
    if status == "not_available":
        reason = _require_string(value.get("reason"), f"{label}.reason")
        if set(value) != {"status", "reason"}:
            raise ValueError(f"{label} has unexpected fields")
        return {"status": status, "reason": reason}
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(token, str) or not token for token in argv)
    ):
        raise ValueError(f"{label}.argv must be a non-empty string array")
    scope = _require_string(value.get("scope"), f"{label}.scope")
    if set(value) != {"status", "argv", "scope"}:
        raise ValueError(f"{label} has unexpected fields")
    return {"status": status, "argv": argv, "scope": scope}


def _validate_source(source: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"catalog source schema must be {SOURCE_SCHEMA}")
    if set(source) != {"schema_version", "classes", "artifacts"}:
        raise ValueError("catalog source has unexpected top-level fields")
    raw_classes = source.get("classes")
    raw_artifacts = source.get("artifacts")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("catalog classes must be a non-empty array")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("catalog artifacts must be a non-empty array")

    classes: list[dict[str, str]] = []
    class_ids: set[str] = set()
    for index, value in enumerate(raw_classes):
        if not isinstance(value, Mapping) or set(value) != {
            "id",
            "label",
            "description",
        }:
            raise ValueError(f"classes[{index}] has an invalid shape")
        class_id = _require_string(value.get("id"), f"classes[{index}].id")
        if not IDENTIFIER.fullmatch(class_id) or class_id in class_ids:
            raise ValueError(f"class id is invalid or duplicated: {class_id}")
        class_ids.add(class_id)
        classes.append(
            {
                "id": class_id,
                "label": _require_string(value.get("label"), f"{class_id}.label"),
                "description": _require_string(
                    value.get("description"), f"{class_id}.description"
                ),
            }
        )

    required = {
        "id",
        "title",
        "class",
        "policy_provenance",
        "result",
        "protocol",
        "manifest",
        "raw",
        "validator",
        "claim_boundary",
    }
    artifacts: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    for index, value in enumerate(raw_artifacts):
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"artifacts[{index}] has an invalid shape")
        artifact_id = _require_string(value.get("id"), f"artifacts[{index}].id")
        if not IDENTIFIER.fullmatch(artifact_id) or artifact_id in artifact_ids:
            raise ValueError(f"artifact id is invalid or duplicated: {artifact_id}")
        artifact_ids.add(artifact_id)
        class_id = _require_string(value.get("class"), f"{artifact_id}.class")
        if class_id not in class_ids:
            raise ValueError(f"{artifact_id}.class is not declared: {class_id}")
        policy = _require_string(
            value.get("policy_provenance"), f"{artifact_id}.policy_provenance"
        )
        if not IDENTIFIER.fullmatch(policy):
            raise ValueError(f"{artifact_id}.policy_provenance is not canonical")

        result, result_path = _repository_path(
            value.get("result"), f"{artifact_id}.result"
        )
        protocol, _ = _repository_path(
            value.get("protocol"), f"{artifact_id}.protocol"
        )
        artifact_root = (PROJECT_ROOT / "evidence" / artifact_id).resolve()
        try:
            result_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(f"{artifact_id}.result must be inside its artifact") from exc

        manifest_value = value.get("manifest")
        manifest: str | None = None
        if manifest_value is not None:
            manifest, manifest_path = _repository_path(
                manifest_value, f"{artifact_id}.manifest"
            )
            try:
                manifest_path.relative_to(artifact_root)
            except ValueError as exc:
                raise ValueError(
                    f"{artifact_id}.manifest must be inside its artifact"
                ) from exc

        raw_value = value.get("raw")
        if not isinstance(raw_value, list) or not raw_value:
            raise ValueError(f"{artifact_id}.raw must be a non-empty array")
        raw: list[str] = []
        for raw_index, item in enumerate(raw_value):
            raw_path, resolved = _repository_path(
                item, f"{artifact_id}.raw[{raw_index}]"
            )
            try:
                resolved.relative_to(artifact_root)
            except ValueError as exc:
                raise ValueError(
                    f"{artifact_id}.raw[{raw_index}] must be inside its artifact"
                ) from exc
            raw.append(raw_path)
        if len(raw) != len(set(raw)):
            raise ValueError(f"{artifact_id}.raw contains duplicate paths")

        artifacts.append(
            {
                "id": artifact_id,
                "title": _require_string(value.get("title"), f"{artifact_id}.title"),
                "class": class_id,
                "policy_provenance": policy,
                "result": result,
                "protocol": protocol,
                "manifest": manifest,
                "raw": raw,
                "validator": _validate_validator(value.get("validator"), artifact_id),
                "claim_boundary": _require_string(
                    value.get("claim_boundary"), f"{artifact_id}.claim_boundary"
                ),
            }
        )

    actual_ids = {
        path.name
        for path in (PROJECT_ROOT / "evidence").iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if artifact_ids != actual_ids:
        missing = sorted(actual_ids - artifact_ids)
        unknown = sorted(artifact_ids - actual_ids)
        raise ValueError(
            "catalog coverage mismatch: "
            f"missing={missing or 'none'}, unknown={unknown or 'none'}"
        )
    return classes, artifacts


def build_catalog(source_path: pathlib.Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source = _strict_json(source_path)
    classes, artifacts = _validate_source(source)
    tracked_files = _tracked_evidence_files()
    artifact_ids = {entry["id"] for entry in artifacts}
    if set(tracked_files) != artifact_ids:
        raise ValueError("tracked evidence files do not match the registered directories")
    class_counts = Counter(entry["class"] for entry in artifacts)
    output_artifacts: list[dict[str, Any]] = []
    global_digest = hashlib.sha256()
    total_files = 0
    total_bytes = 0
    for entry in sorted(artifacts, key=lambda item: item["id"]):
        inventory = _artifact_inventory(
            PROJECT_ROOT / "evidence" / entry["id"], tracked_files[entry["id"]]
        )
        total_files += inventory["file_count"]
        total_bytes += inventory["total_bytes"]
        global_digest.update(entry["id"].encode("ascii"))
        global_digest.update(b"\0")
        global_digest.update(inventory["tree_sha256"].encode("ascii"))
        global_digest.update(b"\n")
        output_artifacts.append({**entry, "inventory": inventory})
    return {
        "schema_version": OUTPUT_SCHEMA,
        "source": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_count": len(output_artifacts),
        "inventory": {
            "fingerprint_basis": "sha256_over_git_blob_ids_paths_and_sizes",
            "file_count": total_files,
            "total_bytes": total_bytes,
            "tree_sha256": global_digest.hexdigest(),
        },
        "class_counts": {
            item["id"]: class_counts[item["id"]] for item in classes
        },
        "classes": classes,
        "artifacts": output_artifacts,
    }


def _json_bytes(catalog: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(catalog, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _link(path: str) -> str:
    relative = pathlib.Path(
        os.path.relpath(PROJECT_ROOT / pathlib.PurePosixPath(path), DEFAULT_MARKDOWN_OUTPUT.parent)
    )
    return relative.as_posix()


def _command(argv: Sequence[str]) -> str:
    return shlex.join(argv)


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _markdown(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# Evidence catalog",
        "",
        "This file is generated by `python scripts/build_evidence_catalog.py`.",
        "Do not edit it manually and do not modify preserved `evidence/` artifacts.",
        "The catalog distinguishes scientific results from pilots, component gates,",
        "scripted runtime checks, and rejected runs.",
        "",
        "## Repository inventory",
        "",
        f"- Artifact directories: {catalog['artifact_count']}",
        f"- Files: {catalog['inventory']['file_count']}",
        f"- Stored size: {_format_bytes(catalog['inventory']['total_bytes'])}",
        f"- Catalog tree SHA-256: `{catalog['inventory']['tree_sha256']}`",
        "- Fingerprint basis: Git blob IDs + canonical paths + blob sizes",
        "- Machine-readable form: [evidence_catalog.json](evidence_catalog.json)",
        "",
        "Run `python scripts/build_evidence_catalog.py --check` from the repository",
        "root to verify complete directory coverage, every linked file, and all",
        "Git-tracked content fingerprints. This check validates catalog integrity; study-specific",
        "commands below define the stronger scientific validation available per artifact.",
        "",
        "## Evidence classes",
        "",
        "| Class | Count | Meaning |",
        "| --- | ---: | --- |",
    ]
    for item in catalog["classes"]:
        lines.append(
            f"| {item['label']} | {catalog['class_counts'][item['id']]} | "
            f"{item['description']} |"
        )

    lines.extend(
        [
            "",
            "## Primary offline acceptance",
            "",
            "These commands validate saved evidence and rebuild derived outputs. They do",
            "not rerun model inference and do not create new task-success observations.",
            "",
            "```text",
            "python -m integrations.openpi.alignment_acceptance --no-open",
            "python -m integrations.openpi.measured_age_confirmatory_acceptance --no-open",
            "python -m integrations.openpi.rtc_overlap_primary_dashboard",
            "```",
        ]
    )

    artifacts = catalog["artifacts"]
    for class_entry in catalog["classes"]:
        members = [item for item in artifacts if item["class"] == class_entry["id"]]
        if not members:
            continue
        lines.extend(
            [
                "",
                f"## {class_entry['label']}",
                "",
                class_entry["description"],
            ]
        )
        for item in members:
            result = _link(item["result"])
            protocol = _link(item["protocol"])
            manifest = (
                f"[{pathlib.PurePosixPath(item['manifest']).name}]({_link(item['manifest'])})"
                if item["manifest"]
                else "Not retained"
            )
            raw_links = ", ".join(
                f"[{pathlib.PurePosixPath(path).name}]({_link(path)})"
                for path in item["raw"]
            )
            validator = item["validator"]
            if validator["status"] == "not_available":
                validation = f"Not available. {validator['reason']}"
            else:
                limitation = (
                    " Limited validator." if validator["status"] == "available_with_limitation" else ""
                )
                validation = (
                    f"`{_command(validator['argv'])}`.{limitation} {validator['scope']}"
                )
            inventory = item["inventory"]
            lines.extend(
                [
                    "",
                    f"### {item['title']}",
                    "",
                    f"- Artifact: `{item['id']}`",
                    f"- Policy provenance: `{item['policy_provenance']}`",
                    f"- Result: [{pathlib.PurePosixPath(item['result']).name}]({result})",
                    f"- Protocol: [{pathlib.PurePosixPath(item['protocol']).name}]({protocol})",
                    f"- Manifest: {manifest}",
                    f"- Raw review files: {raw_links}",
                    f"- Validator: {validation}",
                    f"- Claim boundary: {item['claim_boundary']}",
                    f"- Inventory: {inventory['file_count']} files, "
                    f"{_format_bytes(inventory['total_bytes'])}, "
                    f"tree SHA-256 `{inventory['tree_sha256']}`",
                ]
            )
    return "\n".join(lines) + "\n"


def _check_output(path: pathlib.Path, expected: bytes) -> str | None:
    if not path.is_file():
        return f"missing generated catalog output: {path.relative_to(PROJECT_ROOT)}"
    if path.read_bytes() != expected:
        return f"stale generated catalog output: {path.relative_to(PROJECT_ROOT)}"
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json-output", type=pathlib.Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output", type=pathlib.Path, default=DEFAULT_MARKDOWN_OUTPUT
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in generated outputs differ from the evidence tree",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog = build_catalog(args.source.resolve())
        json_bytes = _json_bytes(catalog)
        markdown_bytes = _markdown(catalog).encode("utf-8")
        if args.check:
            errors = [
                error
                for error in (
                    _check_output(args.json_output.resolve(), json_bytes),
                    _check_output(args.markdown_output.resolve(), markdown_bytes),
                )
                if error is not None
            ]
            if errors:
                raise ValueError("; ".join(errors))
        else:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_bytes(json_bytes)
            args.markdown_output.write_bytes(markdown_bytes)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "checked": bool(args.check),
                "artifact_count": catalog["artifact_count"],
                "file_count": catalog["inventory"]["file_count"],
                "tree_sha256": catalog["inventory"]["tree_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
