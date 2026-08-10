"""Run the saved ArmBench CPU evidence through its read-only validators.

The command is intentionally orchestration-only: it never regenerates or
rewrites a checked-in artifact. Results are written below ``output/`` so the
acceptance log can be inspected locally without polluting the evidence tree.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "cpu_acceptance"


def _python_candidates() -> list[pathlib.Path]:
    names = (
        ".venv-lerobot-0.4.4",
        ".venv-lerobot",
        ".venv-lerobot-cpython",
    )
    candidates: list[pathlib.Path] = []
    for name in names:
        candidates.append(PROJECT_ROOT.parent / name / "Scripts" / "python.exe")
        candidates.append(PROJECT_ROOT.parent / name / "bin" / "python")
        candidates.append(PROJECT_ROOT / name / "Scripts" / "python.exe")
        candidates.append(PROJECT_ROOT / name / "bin" / "python")
    return candidates


def _find_official_python(explicit: str | None) -> pathlib.Path | None:
    if explicit:
        path = pathlib.Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    for path in _python_candidates():
        if path.is_file() and _supports_official_lerobot(path):
            return path
    return None


def _supports_official_lerobot(python: pathlib.Path) -> bool:
    probe = (
        "import importlib.metadata as m; import armbench; "
        "raise SystemExit(0 if m.version('lerobot') == '0.4.4' else 1)"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _specs() -> list[dict[str, Any]]:
    specs = [
        {"id": "doctor", "argv": ["-m", "armbench", "doctor"]},
        {
            "id": "mujoco_scenarios",
            "argv": ["-m", "armbench", "mujoco-validate"],
        },
        {"id": "qp_projection", "argv": ["-m", "armbench", "vla-qp-smoke"]},
        {
            "id": "integrated_guard",
            "argv": ["-m", "armbench", "vla-integrated-guard-smoke"],
        },
        {
            "id": "swept_collision",
            "argv": [
                "-m",
                "armbench",
                "mujoco-swept-validate",
                "reports/mujoco_swept_audit_001",
            ],
        },
        {
            "id": "self_collision",
            "argv": [
                "-m",
                "armbench",
                "mujoco-self-collision-validate",
                "reports/mujoco_self_collision_audit_001",
            ],
        },
        {
            "id": "dynamics_braking",
            "argv": [
                "-m",
                "armbench",
                "mujoco-dynamics-braking-validate",
                "reports/dynamics_braking_audit_001",
            ],
        },
        {
            "id": "provider_contract",
            "argv": [
                "-m",
                "armbench",
                "vla-provider-audit-validate",
                "reports/provider_contract_audit_001",
            ],
        },
        {
            "id": "lerobot_watchdog",
            "argv": [
                "-m",
                "armbench",
                "vla-lerobot-validate",
                "reports/lerobot_style_watchdog_001",
            ],
        },
        {
            "id": "frozen_pi05_replay",
            "argv": [
                "-m",
                "armbench",
                "vla-panda-archive-replay-validate",
                "reports/pi05_panda_archive_replay_90_001",
            ],
        },
        {
            "id": "braking_repair",
            "argv": [
                "-m",
                "armbench",
                "vla-panda-braking-repair-validate",
                "reports/pi05_panda_braking_repair_90_001",
                "--source-directory",
                "evidence/pi05_rtc_overlap_primary_v3_seed_20260807_001/evaluation",
            ],
        },
        {
            "id": "integrated_fault_matrix",
            "argv": [
                "-m",
                "armbench",
                "vla-integrated-fault-validate",
                "reports/integrated_panda_fault_matrix_001",
            ],
        },
        {
            "id": "integrated_tasks",
            "argv": [
                "-m",
                "armbench",
                "vla-integrated-task-validate",
                "reports/integrated_panda_task_001",
            ],
        },
        {
            "id": "async_panda_closed_loop",
            "argv": [
                "-m",
                "armbench",
                "vla-panda-async-validate",
                "reports/async_panda_closed_loop_400ms_20mm_v3_001",
            ],
        },
        {
            "id": "cpu_runtime_boundary",
            "argv": [
                "-m",
                "armbench",
                "vla-cpu-runtime-validate",
                "reports/cpu_runtime_completion_001",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_core",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/pi05_libero_independent_clock_core_40_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_visual",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/pi05_libero_independent_clock_visual_success_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_object",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/g03_independent_clock_object_40_20260810_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_deadline50",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/g04_spatial_deadline50_40_20260810_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_deadline150",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/g05_spatial_deadline150_40_20260810_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_deadline175",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/g06_spatial_deadline175_40_20260810_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_deadline150_seed8",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/pi05_libero_spatial_deadline150_seed8_40_20260810_001/evaluation",
                "--json",
            ],
        },
        {
            "id": "pi05_libero_independent_clock_deadline175_seed8",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                "evidence/pi05_libero_spatial_deadline175_seed8_40_20260810_001/evaluation",
                "--json",
            ],
        },
    ]

    additional_independent_clock_ids = (
        "pi05_object_deadline150_seed7_40_20260810_001",
        "pi05_object_deadline150_seed8_40_20260810_001",
        "pi05_object_deadline175_seed7_40_20260810_001",
        "pi05_object_deadline175_seed8_40_20260810_001",
        "pi05_object_deadline200_seed8_40_20260810_001",
        "pi05_spatial_deadline150_seed9_40_20260810_001",
        "pi05_spatial_deadline155_seed7_40_20260810_001",
        "pi05_spatial_deadline155_seed8_40_20260810_001",
        "pi05_spatial_deadline155_seed9_40_20260810_001",
        "pi05_spatial_deadline175_seed9_40_20260810_001",
        "pi05_spatial_deadline200_seed8_40_20260810_001",
        "pi05_spatial_deadline200_seed9_40_20260810_001",
        "pi05_selection_smoke_age_aligned_seed7_1_20260810_001",
        "pi05_selection_smoke_response_relative_seed7_1_20260810_001",
        "pi05_selection_spatial_s7_age_aligned_40_20260810_001",
        "pi05_selection_spatial_s7_response_relative_40_20260810_001",
        "pi05_selection_spatial_s8_age_aligned_40_20260810_001",
        "pi05_selection_spatial_s8_response_relative_40_20260810_001",
    )
    specs.extend(
        {
            "id": f"validate_{artifact_id}",
            "argv": [
                "-m",
                "integrations.openpi.validate_libero_independent_clock",
                f"evidence/{artifact_id}/evaluation",
                "--json",
            ],
        }
        for artifact_id in additional_independent_clock_ids
    )

    deadline_artifact_ids = (
        "pi05_object_deadline150_seed7_40_20260810_001",
        "pi05_object_deadline175_seed7_40_20260810_001",
        "g03_independent_clock_object_40_20260810_001",
        "pi05_object_deadline150_seed8_40_20260810_001",
        "pi05_object_deadline175_seed8_40_20260810_001",
        "pi05_object_deadline200_seed8_40_20260810_001",
        "g05_spatial_deadline150_40_20260810_001",
        "pi05_spatial_deadline155_seed7_40_20260810_001",
        "g06_spatial_deadline175_40_20260810_001",
        "pi05_libero_independent_clock_core_40_001",
        "pi05_libero_spatial_deadline150_seed8_40_20260810_001",
        "pi05_spatial_deadline155_seed8_40_20260810_001",
        "pi05_libero_spatial_deadline175_seed8_40_20260810_001",
        "pi05_spatial_deadline200_seed8_40_20260810_001",
        "pi05_spatial_deadline150_seed9_40_20260810_001",
        "pi05_spatial_deadline155_seed9_40_20260810_001",
        "pi05_spatial_deadline175_seed9_40_20260810_001",
        "pi05_spatial_deadline200_seed9_40_20260810_001",
    )
    smoke_artifact_ids = (
        "pi05_selection_smoke_age_aligned_seed7_1_20260810_001",
        "pi05_selection_smoke_response_relative_seed7_1_20260810_001",
    )
    specs.extend(
        [
            {
                "id": "pi05_deadline_multisuite_report_720",
                "argv": [
                    "scripts/build_pi05_deadline_report.py",
                    *[
                        f"evidence/{artifact_id}/evaluation"
                        for artifact_id in deadline_artifact_ids
                    ],
                    "--output-dir",
                    "reports/pi05_deadline_multisuite_report_720_20260810_001",
                    "--check",
                ],
            },
            {
                "id": "pi05_selection_smoke_report",
                "argv": [
                    "scripts/build_pi05_selection_report.py",
                    *[
                        f"evidence/{artifact_id}/evaluation"
                        for artifact_id in smoke_artifact_ids
                    ],
                    "--profile",
                    "smoke",
                    "--output-dir",
                    "reports/pi05_selection_smoke_report_20260810_001",
                    "--check",
                ],
            },
            {
                "id": "evidence_catalog",
                "argv": ["scripts/build_evidence_catalog.py", "--check"],
            },
        ]
    )
    return specs


def _tail(value: str, limit: int = 2400) -> str:
    value = value.strip()
    return value if len(value) <= limit else "..." + value[-limit:]


def _run(
    python: pathlib.Path,
    spec: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    command = [str(python), *list(spec["argv"])]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        return {
            "id": spec["id"],
            "status": status,
            "returncode": completed.returncode,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "id": spec["id"],
            "status": "failed",
            "returncode": None,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout_tail": _tail(str(exc.stdout or "")),
            "stderr_tail": f"timeout after {timeout_s:.1f}s",
        }
    except OSError as exc:
        return {
            "id": spec["id"],
            "status": "failed",
            "returncode": None,
            "duration_s": round(time.perf_counter() - started, 3),
            "command": command,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ArmBench CPU acceptance",
        "",
        f"- Overall: **{summary['overall'].upper()}**",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Python: `{summary['python']}`",
        f"- Passed: `{summary['counts']['passed']}`; skipped: `{summary['counts']['skipped']}`; failed: `{summary['counts']['failed']}`",
        "",
        "This log reruns saved validators. It does not claim a learned VLA checkpoint, a real robot, hard real time, or a safety certification.",
        "",
        "| Check | Status | Duration (s) |",
        "| --- | --- | ---: |",
    ]
    for result in summary["checks"]:
        lines.append(
            f"| `{result['id']}` | `{result['status']}` | {result['duration_s']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_acceptance(
    *,
    python: pathlib.Path,
    output: pathlib.Path,
    timeout_s: float,
    official_python: pathlib.Path | None,
    require_official: bool,
    full_tests: bool,
) -> dict[str, Any]:
    checks = [_run(python, spec, timeout_s) for spec in _specs()]

    if official_python is None:
        status = "failed" if require_official else "skipped"
        checks.append(
            {
                "id": "official_lerobot_roundtrip",
                "status": status,
                "returncode": None,
                "duration_s": 0.0,
                "command": [],
                "stdout_tail": "",
                "stderr_tail": "official LeRobot environment not found",
            }
        )
    else:
        checks.append(
            _run(
                official_python,
                {
                    "id": "official_lerobot_roundtrip",
                    "argv": [
                        "-m",
                        "armbench",
                        "vla-lerobot-official-validate",
                        "reports/official_lerobot_roundtrip_001",
                    ],
                },
                timeout_s,
            )
        )

    if full_tests:
        checks.append(
            _run(
                python,
                {"id": "pytest_full", "argv": ["-m", "pytest", "-q"]},
                max(timeout_s, 900.0),
            )
        )

    counts = {
        state: sum(result["status"] == state for result in checks)
        for state in ("passed", "skipped", "failed")
    }
    summary = {
        "schema_version": "armbench.cpu_acceptance.v1",
        "overall": "passed" if counts["failed"] == 0 else "failed",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": str(python),
        "official_python": str(official_python) if official_python else None,
        "counts": counts,
        "checks": checks,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rerun the saved ArmBench CPU evidence validators"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="ArmBench Python interpreter (default: current interpreter)",
    )
    parser.add_argument(
        "--official-python",
        help="isolated Python interpreter containing lerobot==0.4.4",
    )
    parser.add_argument(
        "--require-official",
        action="store_true",
        help="fail when the official LeRobot environment is unavailable",
    )
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="also run the complete pytest suite after artifact checks",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="per-check timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="directory for the ignored local acceptance log",
    )
    parser.add_argument("--json", action="store_true", help="print the full JSON summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    python = pathlib.Path(args.python).expanduser().resolve()
    if not python.is_file():
        print(f"ArmBench Python not found: {python}", file=sys.stderr)
        return 2
    official_python = _find_official_python(args.official_python)
    summary = run_acceptance(
        python=python,
        output=pathlib.Path(args.output).expanduser().resolve(),
        timeout_s=max(args.timeout_s, 1.0),
        official_python=official_python,
        require_official=args.require_official,
        full_tests=args.full_tests,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        for result in summary["checks"]:
            print(
                f"[{result['status'].upper():7}] {result['id']} "
                f"({result['duration_s']:.2f}s)"
            )
        print(
            f"CPU acceptance: {summary['overall'].upper()} "
            f"({summary['counts']['passed']} passed, "
            f"{summary['counts']['skipped']} skipped, "
            f"{summary['counts']['failed']} failed)"
        )
        print(f"Saved: {pathlib.Path(args.output).resolve()}")
    return 0 if summary["overall"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
