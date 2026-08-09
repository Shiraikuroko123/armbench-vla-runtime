from __future__ import annotations

import json

from scripts import build_evidence_catalog as catalog


def _saved_catalog() -> dict:
    return json.loads(catalog.DEFAULT_JSON_OUTPUT.read_text(encoding="utf-8"))


def test_generated_catalog_matches_all_tracked_evidence() -> None:
    generated = catalog.build_catalog()

    assert generated == _saved_catalog()
    assert generated["artifact_count"] == 30
    assert generated["inventory"]["file_count"] == 1519
    assert len(generated["inventory"]["tree_sha256"]) == 64
    assert {item["id"] for item in generated["artifacts"]} == {
        path.name for path in (catalog.PROJECT_ROOT / "evidence").iterdir() if path.is_dir()
    }


def test_catalog_keeps_claim_classes_and_policy_provenance_distinct() -> None:
    saved = _saved_catalog()
    artifacts = {item["id"]: item for item in saved["artifacts"]}

    assert saved["class_counts"]["primary"] == 3
    assert saved["class_counts"]["exploratory"] == 7
    assert saved["class_counts"]["rejected"] == 2
    assert all(
        item["policy_provenance"] == "official_openpi_pi05_libero"
        for item in saved["artifacts"]
        if item["class"] in {"primary", "primary_source", "rejected"}
    )
    assert all(
        item["policy_provenance"].startswith("scripted_non_learned")
        for item in saved["artifacts"]
        if item["class"] == "scripted_runtime"
    )
    assert artifacts["pi05_rtc_overlap_primary_v3_300_001"]["class"] == "primary"
    assert "null success-improvement result" in artifacts[
        "pi05_rtc_overlap_primary_v3_300_001"
    ]["claim_boundary"]
    assert artifacts["pi05_rtc_overlap_primary_seed_20260806_001"][
        "class"
    ] == "rejected"
    g02 = artifacts["pi05_libero_independent_clock_core_40_001"]
    assert g02["class"] == "exploratory"
    assert g02["policy_provenance"] == "official_openpi_pi05_libero"
    assert "not an official LIBERO leaderboard score" in g02["claim_boundary"]
