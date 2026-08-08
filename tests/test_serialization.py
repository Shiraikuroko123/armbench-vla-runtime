from __future__ import annotations

import hashlib

import pytest

from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    json_equal,
    sha256_file,
    strict_json_loads,
    write_json,
)


def test_canonical_json_is_stable_and_rejects_non_finite_values() -> None:
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"value": float("nan")})


def test_strict_json_rejects_duplicate_fields_and_constants() -> None:
    with pytest.raises(ValueError, match="duplicate JSON field"):
        strict_json_loads('{"value": 1, "value": 2}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        strict_json_loads('{"value": NaN}')


def test_manifest_helpers_and_json_writer(tmp_path) -> None:
    path = tmp_path / "document.json"
    write_json(path, {"message": "ok", "values": [1, 2]})

    assert strict_json_loads(path.read_text(encoding="utf-8")) == {
        "message": "ok",
        "values": [1, 2],
    }
    assert has_exact_fields({"message": "ok"}, {"message"})
    assert not has_exact_fields({"message": "ok", "extra": True}, {"message"})
    assert json_equal({"x": 1, "y": [2]}, {"y": [2], "x": 1})
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert is_sha256("a" * 64)
    assert not is_sha256("A" * 64)
