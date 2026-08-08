"""Small, deterministic serialization helpers for VLA artifacts.

The runtime produces evidence that is consumed by both humans and validators.
Keeping the encoding rules in one module prevents provider bundles and
LeRobot-style episodes from silently developing different integrity semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: object) -> bytes:
    """Return the repository's stable JSON byte representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def strict_json_loads(value: str) -> Any:
    """Parse JSON while rejecting duplicate fields and non-standard constants."""

    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_fields,
        parse_constant=_reject_json_constant,
    )


def strict_json_load(path: Path) -> Any:
    """Read and strictly parse a UTF-8 JSON document."""

    return strict_json_loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest for an in-memory payload."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the complete artifact into memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def has_exact_fields(value: object, fields: set[str]) -> bool:
    """Check that a decoded JSON object has exactly the expected keys."""

    return isinstance(value, dict) and set(value) == fields


def is_sha256(value: object) -> bool:
    """Return whether *value* is a lowercase hexadecimal SHA-256 digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def json_equal(left: object, right: object) -> bool:
    """Compare JSON-compatible values using their canonical bytes."""

    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def write_json(path: Path, value: object) -> None:
    """Write a human-readable, deterministic JSON document."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "canonical_json",
    "has_exact_fields",
    "is_sha256",
    "json_equal",
    "sha256_bytes",
    "sha256_file",
    "strict_json_load",
    "strict_json_loads",
    "write_json",
]
