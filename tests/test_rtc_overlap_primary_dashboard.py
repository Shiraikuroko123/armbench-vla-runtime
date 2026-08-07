from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import pytest

from integrations.openpi import rtc_overlap_primary_dashboard as dashboard


def _payload(html: str) -> dict:
    marker = '<script id="armbench-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_rebuild_equivalence_allows_python_patch_version_difference() -> None:
    saved = {
        "implementation": {
            "python_version": "3.10.8",
            "numpy_version": "1.26.4",
            "analyzer_sha256": "a" * 64,
        },
        "result": {"successes": 97},
    }
    recomputed = {
        **saved,
        "implementation": {
            **saved["implementation"],
            "python_version": "3.10.20",
        },
    }

    dashboard._validate_rebuild_equivalence(saved, recomputed)


def test_rebuild_equivalence_rejects_result_or_dependency_change() -> None:
    saved = {
        "implementation": {
            "python_version": "3.10.8",
            "numpy_version": "1.26.4",
            "analyzer_sha256": "a" * 64,
        },
        "result": {"successes": 97},
    }
    changed_result = {**saved, "result": {"successes": 96}}
    changed_numpy = {
        **saved,
        "implementation": {
            **saved["implementation"],
            "numpy_version": "2.0.0",
        },
    }

    with pytest.raises(ValueError, match="disagrees"):
        dashboard._validate_rebuild_equivalence(saved, changed_result)
    with pytest.raises(ValueError, match="NumPy"):
        dashboard._validate_rebuild_equivalence(saved, changed_numpy)


def test_builds_fail_closed_dashboard_from_corrected_primary_evidence(
) -> None:
    output_parent = pathlib.Path(
        tempfile.mkdtemp(prefix="rtc-primary-dashboard-", dir=dashboard.PROJECT_ROOT)
    )
    output = output_parent / "index.html"
    try:
        result = dashboard.build_dashboard(output=output)

        assert result == {
            "output": str(output.resolve()),
            "rollouts": 300,
            "triplets": 100,
            "failure_videos_verified": 10,
            "tasks": 10,
        }
        html = output.read_text(encoding="utf-8")
        payload = _payload(html)
        assert payload["schemaVersion"] == dashboard.DASHBOARD_SCHEMA_VERSION
        assert payload["cohort"]["rollouts"] == 300
        assert payload["cohort"]["triplets"] == 100
        assert len(payload["triplets"]) == 100
        assert sum(
            method["video"] is not None
            for triplet in payload["triplets"]
            for method in triplet["methods"]
        ) == 10
        assert all(
            contrast["successImprovementSupported"] is False
            for contrast in payload["contrasts"]
        )
        assert "frozen_environment_command_verified" not in html
        assert "RTC overlap primary evidence" in html
    finally:
        shutil.rmtree(output_parent)
