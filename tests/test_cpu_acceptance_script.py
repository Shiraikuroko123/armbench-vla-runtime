from __future__ import annotations

import json
import pathlib
import sys

import pytest

import scripts.accept_cpu as acceptance


def _passed(spec_id: str) -> dict[str, object]:
    return {
        "id": spec_id,
        "status": "passed",
        "returncode": 0,
        "duration_s": 0.01,
        "command": [sys.executable, "-c", "pass"],
        "stdout_tail": "",
        "stderr_tail": "",
    }


def test_registered_checks_have_unique_ids_and_existing_inputs() -> None:
    specs = acceptance._specs()
    ids = [spec["id"] for spec in specs]

    assert len(ids) == len(set(ids))
    for spec in specs:
        for token in spec["argv"]:
            if token.startswith(("reports/", "evidence/", "scripts/")):
                assert (acceptance.PROJECT_ROOT / token).exists(), token


@pytest.mark.parametrize(
    ("require_official", "expected_overall", "expected_status"),
    (
        (False, "passed", "skipped"),
        (True, "failed", "failed"),
    ),
)
def test_acceptance_records_missing_optional_environment(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    require_official: bool,
    expected_overall: str,
    expected_status: str,
) -> None:
    monkeypatch.setattr(acceptance, "_specs", lambda: [{"id": "probe"}])
    monkeypatch.setattr(
        acceptance,
        "_run",
        lambda _python, spec, _timeout: _passed(spec["id"]),
    )

    summary = acceptance.run_acceptance(
        python=pathlib.Path(sys.executable),
        output=tmp_path,
        timeout_s=1.0,
        official_python=None,
        require_official=require_official,
        full_tests=False,
    )

    assert summary["overall"] == expected_overall
    assert summary["checks"][-1]["status"] == expected_status
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
        "schema_version"
    ] == "armbench.cpu_acceptance.v1"
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "learned VLA checkpoint" in markdown
