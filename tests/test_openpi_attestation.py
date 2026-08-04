from __future__ import annotations

import hashlib
import json

from integrations.openpi.serve_policy_attested import (
    _submodules_are_clean,
    checkpoint_content_manifest,
    public_attestation,
)


def test_checkpoint_manifest_hashes_every_file_deterministically(tmp_path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "a.bin").write_bytes(b"alpha")
    (tmp_path / "metadata.json").write_bytes(b"{}\n")

    first = checkpoint_content_manifest(tmp_path)
    second = checkpoint_content_manifest(tmp_path)

    assert first == second
    assert first["checkpoint_file_count"] == 2
    assert first["checkpoint_total_bytes"] == 8
    by_path = {item["path"]: item for item in first["checkpoint_files"]}
    assert by_path["params/a.bin"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    canonical = json.dumps(
        first["checkpoint_files"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert first["checkpoint_content_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_public_attestation_omits_local_paths_and_file_inventory() -> None:
    public = public_attestation(
        {
            "schema_version": "test",
            "checkpoint_local_path": "/private/cache",
            "checkpoint_files": [{"path": "secret"}],
            "checkpoint_content_sha256": "a" * 64,
        }
    )

    assert public == {
        "schema_version": "test",
        "checkpoint_content_sha256": "a" * 64,
    }


def test_submodule_status_rejects_uninitialized_or_modified_entries() -> None:
    assert _submodules_are_clean("")
    assert _submodules_are_clean(" abc123 submodule (heads/main)")
    assert not _submodules_are_clean("-abc123 submodule")
    assert not _submodules_are_clean("+abc123 submodule (heads/main-1-gabc123)")
