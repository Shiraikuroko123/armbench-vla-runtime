from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.projected_conditioning_g0 import (
    _canonical_source_sha256,
    _checkpoint_manifest,
    _fixed_observation,
    _percentile,
    _timing_summary,
)


def test_fixed_observation_is_deterministic_and_libero_shaped() -> None:
    first = _fixed_observation()
    second = _fixed_observation()

    assert first["observation/state"].shape == (8,)
    assert first["observation/image"].shape == (224, 224, 3)
    assert first["observation/wrist_image"].shape == (224, 224, 3)
    for key in ("observation/state", "observation/image", "observation/wrist_image"):
        np.testing.assert_array_equal(first[key], second[key])


def test_timing_summary_reports_prespecified_percentiles() -> None:
    summary = _timing_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == 2.5
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["max_ms"] == 4.0


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _percentile([], 95.0)


def test_checkpoint_manifest_hashes_all_files_deterministically(tmp_path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "weights.bin").write_bytes(b"weights")
    (tmp_path / "metadata.json").write_bytes(b"{}\n")

    first = _checkpoint_manifest(tmp_path)
    second = _checkpoint_manifest(tmp_path)

    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 10
    assert len(first["content_sha256"]) == 64


def test_source_hash_is_independent_of_crlf(tmp_path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"print('x')\n")
    crlf.write_bytes(b"print('x')\r\n")

    assert _canonical_source_sha256(lf) == _canonical_source_sha256(crlf)
