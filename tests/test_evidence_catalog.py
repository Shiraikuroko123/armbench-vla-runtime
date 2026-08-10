from __future__ import annotations

import json

from scripts import build_evidence_catalog as catalog


def _saved_catalog() -> dict:
    return json.loads(catalog.DEFAULT_JSON_OUTPUT.read_text(encoding="utf-8"))


def test_generated_catalog_matches_all_tracked_evidence() -> None:
    generated = catalog.build_catalog()

    assert generated == _saved_catalog()
    assert generated["artifact_count"] == 56
    assert generated["inventory"]["file_count"] == 4270
    assert len(generated["inventory"]["tree_sha256"]) == 64
    assert {item["id"] for item in generated["artifacts"]} == {
        path.name for path in (catalog.PROJECT_ROOT / "evidence").iterdir() if path.is_dir()
    }


def test_catalog_keeps_claim_classes_and_policy_provenance_distinct() -> None:
    saved = _saved_catalog()
    artifacts = {item["id"]: item for item in saved["artifacts"]}

    assert saved["class_counts"]["primary"] == 3
    assert saved["class_counts"]["exploratory"] == 31
    assert saved["class_counts"]["mechanism_gate"] == 6
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
    assert saved["class_counts"]["exploratory"] == 31
    assert artifacts["g03_independent_clock_object_40_20260810_001"]["class"] == "exploratory"
    assert artifacts["g04_spatial_deadline50_40_20260810_001"]["class"] == "exploratory"
    assert artifacts["g05_spatial_deadline150_40_20260810_001"]["class"] == "exploratory"
    assert artifacts["g06_spatial_deadline175_40_20260810_001"]["class"] == "exploratory"
    assert artifacts[
        "pi05_libero_spatial_deadline150_seed8_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_libero_spatial_deadline175_seed8_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts["pi05_selection_smoke_age_aligned_seed7_1_20260810_001"][
        "class"
    ] == "mechanism_gate"
    assert artifacts[
        "pi05_selection_smoke_response_relative_seed7_1_20260810_001"
    ]["class"] == "mechanism_gate"
    assert artifacts[
        "pi05_selection_spatial_s7_age_aligned_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_selection_spatial_s7_response_relative_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_selection_spatial_s8_age_aligned_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_selection_spatial_s8_response_relative_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_selection_spatial_s9_age_aligned_40_20260810_001"
    ]["class"] == "exploratory"
    assert artifacts[
        "pi05_selection_spatial_s9_response_relative_40_20260810_001"
    ]["class"] == "exploratory"
    assert "not method-effect evidence" in artifacts[
        "pi05_selection_smoke_age_aligned_seed7_1_20260810_001"
    ]["claim_boundary"]
