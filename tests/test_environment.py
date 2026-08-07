from pathlib import Path

from armbench.environment import (
    EnvironmentCheck,
    EnvironmentReport,
    format_environment_report,
)
from armbench.mujoco_sim.model import (
    MENAGERIE_ROOT_ENV,
    PANDA_SCENE_ENV,
    default_panda_scene_path,
    panda_scene_candidates,
)


def test_direct_scene_override_has_priority(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "custom" / "scene.xml"
    scene.parent.mkdir()
    scene.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv(PANDA_SCENE_ENV, str(scene))
    monkeypatch.delenv(MENAGERIE_ROOT_ENV, raising=False)

    assert panda_scene_candidates()[0] == scene.resolve()
    assert default_panda_scene_path() == scene.resolve()


def test_menagerie_root_accepts_repository_or_model_directory(
    monkeypatch, tmp_path: Path
) -> None:
    repository_scene = tmp_path / "franka_emika_panda" / "scene.xml"
    repository_scene.parent.mkdir()
    repository_scene.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.delenv(PANDA_SCENE_ENV, raising=False)
    monkeypatch.setenv(MENAGERIE_ROOT_ENV, str(tmp_path))
    assert default_panda_scene_path() == repository_scene.resolve()

    monkeypatch.setenv(MENAGERIE_ROOT_ENV, str(repository_scene.parent))
    assert default_panda_scene_path() == repository_scene.resolve()


def test_environment_report_distinguishes_optional_checks() -> None:
    report = EnvironmentReport(
        armbench_version="test",
        platform="test-platform",
        executable="python",
        checks=(
            EnvironmentCheck("required", True, True, "ready"),
            EnvironmentCheck("optional", False, False, "not installed"),
        ),
    )

    rendered = format_environment_report(report)

    assert report.ready
    assert "[PASS] required: ready" in rendered
    assert "[INFO] optional: not installed" in rendered
    assert rendered.endswith("Result: READY")
