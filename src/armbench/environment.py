"""Local environment diagnostics for CPU-only ArmBench workflows."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg

from armbench import __version__
from armbench.mujoco_sim.model import MuJoCoPanda, default_panda_scene_path


@dataclass(frozen=True)
class EnvironmentCheck:
    name: str
    ok: bool
    required: bool
    detail: str


@dataclass(frozen=True)
class EnvironmentReport:
    armbench_version: str
    platform: str
    executable: str
    checks: tuple[EnvironmentCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "armbench_version": self.armbench_version,
            "platform": self.platform,
            "executable": self.executable,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }


def _package_check(distribution: str) -> EnvironmentCheck:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return EnvironmentCheck(distribution, False, True, "not installed")
    return EnvironmentCheck(distribution, True, True, version)


def collect_environment_report(*, require_vla: bool = False) -> EnvironmentReport:
    checks: list[EnvironmentCheck] = []
    python_ok = sys.version_info[:2] == (3, 10)
    checks.append(
        EnvironmentCheck(
            "python",
            python_ok,
            True,
            f"{platform.python_version()} (requires 3.10.x)",
        )
    )
    checks.extend(
        _package_check(distribution)
        for distribution in ("numpy", "mujoco", "matplotlib", "imageio")
    )
    try:
        scene_path = default_panda_scene_path()
        robot = MuJoCoPanda.create(scene_path=scene_path)
        model_detail = (
            f"{scene_path} (nq={robot.model.nq}, nv={robot.model.nv}, "
            f"nu={robot.model.nu})"
        )
        checks.append(EnvironmentCheck("panda_model", True, True, model_detail))
    except Exception as error:
        checks.append(
            EnvironmentCheck("panda_model", False, True, str(error).strip())
        )
    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        checks.append(
            EnvironmentCheck("ffmpeg", ffmpeg_path.is_file(), False, str(ffmpeg_path))
        )
    except Exception as error:
        checks.append(EnvironmentCheck("ffmpeg", False, False, str(error).strip()))
    openpi_available = importlib.util.find_spec("openpi_client") is not None
    checks.append(
        EnvironmentCheck(
            "openpi_client",
            openpi_available,
            require_vla,
            "installed" if openpi_available else "optional VLA client not installed",
        )
    )
    return EnvironmentReport(
        armbench_version=__version__,
        platform=platform.platform(),
        executable=str(Path(sys.executable).resolve()),
        checks=tuple(checks),
    )


def format_environment_report(report: EnvironmentReport) -> str:
    lines = [
        f"ArmBench {report.armbench_version} environment",
        f"Platform: {report.platform}",
        f"Python: {report.executable}",
        "",
    ]
    for check in report.checks:
        if check.ok:
            status = "PASS"
        elif check.required:
            status = "FAIL"
        else:
            status = "INFO"
        lines.append(f"[{status}] {check.name}: {check.detail}")
    lines.extend(("", f"Result: {'READY' if report.ready else 'BLOCKED'}"))
    return "\n".join(lines)
