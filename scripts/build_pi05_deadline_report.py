"""Build a deterministic, validator-backed pi0.5 deadline report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import pathlib
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openpi.validate_libero_independent_clock import (  # noqa: E402
    validate_artifact,
)


SCHEMA_VERSION = "armbench.pi05_deadline_report.v1"
MANIFEST_SCHEMA_VERSION = "armbench.pi05_deadline_report_manifest.v1"
OUTPUT_NAMES = (
    "deadline_cells.csv",
    "summary.json",
    "summary.md",
    "manifest.json",
)


def _strict_json(path: pathlib.Path) -> Any:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {path}: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON {path}: {exc}") from exc


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError(
            "Wilson interval requires 0 <= successes <= total and total > 0"
        )
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _evaluation_root(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    if resolved.name == "evaluation":
        return resolved
    candidate = resolved / "evaluation"
    if candidate.is_dir():
        return candidate
    raise ValueError(f"artifact has no evaluation directory: {path}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_deadline_cell(path: pathlib.Path) -> dict[str, Any]:
    evaluation = _evaluation_root(path)
    report = validate_artifact(evaluation)
    if not report.valid:
        raise ValueError(
            "artifact validator failed for %s: %s"
            % (evaluation, "; ".join(report.errors))
        )

    protocol = _require_mapping(
        _strict_json(evaluation / "resolved_protocol.json"), "resolved_protocol"
    )
    aggregate = _require_mapping(
        _strict_json(evaluation / "aggregate.json"), "aggregate"
    )
    environment = _require_mapping(
        _strict_json(evaluation / "environment.json"), "environment"
    )
    rows = _strict_json(evaluation / "per_episode.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"per_episode.json must contain rows: {evaluation}")

    runtime_protocol = _require_mapping(protocol.get("runtime"), "protocol.runtime")
    sampling = _require_mapping(
        protocol.get("policy_sampling"), "protocol.policy_sampling"
    )
    matrix = _require_mapping(protocol.get("matrix"), "protocol.matrix")
    matrix_cells = matrix.get("cells")
    if not isinstance(matrix_cells, list) or not matrix_cells:
        raise ValueError("protocol.matrix.cells must be a non-empty array")
    suites = {
        str(_require_mapping(cell, "matrix cell").get("task_suite"))
        for cell in matrix_cells
    }
    if len(suites) != 1:
        raise ValueError(f"artifact mixes task suites: {sorted(suites)}")
    task_suite = next(iter(suites))

    reason_counts: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    raw_tick_count = 0
    for runtime_path in sorted((evaluation / "episodes").glob("*/runtime.json")):
        episode = _require_mapping(_strict_json(runtime_path), str(runtime_path))
        runtime = _require_mapping(episode.get("runtime"), f"{runtime_path}.runtime")
        ticks = runtime.get("ticks")
        if not isinstance(ticks, list):
            raise ValueError(f"{runtime_path}.runtime.ticks must be an array")
        for raw_tick in ticks:
            tick = _require_mapping(raw_tick, f"{runtime_path}.tick")
            status = str(tick.get("status"))
            reason = str(tick.get("reason"))
            status_counts[status] += 1
            reason_counts[f"{status}:{reason}"] += 1
            raw_tick_count += 1

    rollouts = int(aggregate["completed_rollouts"])
    successes = sum(
        bool(_require_mapping(row, "episode row")["task_success"]) for row in rows
    )
    if rollouts != len(rows) or successes != int(aggregate["task_successes"]):
        raise ValueError(
            "aggregate rollout or success counts disagree with per-episode rows"
        )
    if raw_tick_count != int(aggregate["total_control_ticks"]):
        raise ValueError(
            "aggregate control-tick count disagrees with raw runtime ticks"
        )
    if status_counts["execute"] != int(aggregate["total_executes"]):
        raise ValueError("aggregate execute count disagrees with raw runtime ticks")
    if status_counts["hold"] != int(aggregate["total_holds"]):
        raise ValueError("aggregate hold count disagrees with raw runtime ticks")

    task_groups: dict[int, list[bool]] = defaultdict(list)
    for raw_row in rows:
        row = _require_mapping(raw_row, "episode row")
        task_groups[int(row["task_id"])].append(bool(row["task_success"]))
    task_clusters = [
        {
            "task_id": task_id,
            "rollouts": len(outcomes),
            "successes": sum(outcomes),
            "success_rate": sum(outcomes) / len(outcomes),
        }
        for task_id, outcomes in sorted(task_groups.items())
    ]

    lower, upper = wilson_interval(successes, rollouts)
    total_executes = int(aggregate["total_executes"])
    total_holds = int(aggregate["total_holds"])
    total_decisions = total_executes + total_holds
    if total_decisions <= 0:
        raise ValueError("artifact contains no control decisions")

    artifact_id = evaluation.parent.name
    return {
        "artifact_id": artifact_id,
        "source_manifest_sha256": _sha256(evaluation / "manifest.json"),
        "task_suite": task_suite,
        "seed": int(sampling["seed"]),
        "deadline_ms": float(runtime_protocol["deadline_ms"]),
        "rollouts": rollouts,
        "task_successes": successes,
        "task_success_rate": successes / rollouts,
        "wilson95_low": lower,
        "wilson95_high": upper,
        "task_clusters": task_clusters,
        "tasks_all_success": sum(
            cluster["successes"] == cluster["rollouts"] for cluster in task_clusters
        ),
        "tasks_any_success": sum(cluster["successes"] > 0 for cluster in task_clusters),
        "control_ticks": raw_tick_count,
        "ticks_during_inference": int(aggregate["total_ticks_during_inference"]),
        "episodes_with_inference_overlap": int(
            aggregate["episodes_with_inference_overlap"]
        ),
        "execute_ticks": total_executes,
        "hold_ticks": total_holds,
        "execute_duty_cycle": total_executes / total_decisions,
        "deadline_hold_ticks": reason_counts["hold:deadline_exceeded"],
        "no_response_hold_ticks": reason_counts["hold:no_policy_response"],
        "response_level_deadline_rejections": int(
            aggregate["total_deadline_exceeded_responses"]
        ),
        "provider_failures": int(aggregate["total_failed_responses"]),
        "checkpoint": str(protocol["checkpoint"]),
        "openpi_commit": str(protocol["openpi_commit"]),
        "armbench_commit": environment.get("armbench_git_commit"),
        "control_period_ms": float(runtime_protocol["control_period_ms"]),
        "action_horizon": int(protocol["official_protocol"]["action_horizon"]),
    }


def build_summary(paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one artifact is required")
    cells = [_load_deadline_cell(path) for path in paths]
    cells.sort(key=lambda cell: (cell["task_suite"], cell["seed"], cell["deadline_ms"]))

    identities: set[tuple[str, int, float]] = set()
    for cell in cells:
        identity = (cell["task_suite"], cell["seed"], cell["deadline_ms"])
        if identity in identities:
            raise ValueError(f"duplicate suite/seed/deadline cell: {identity}")
        identities.add(identity)

    comparisons = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["task_suite"], cell["seed"])].append(cell)
    for (task_suite, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda cell: cell["deadline_ms"])
        for lower, upper in zip(ordered, ordered[1:]):
            comparisons.append(
                {
                    "task_suite": task_suite,
                    "seed": seed,
                    "from_deadline_ms": lower["deadline_ms"],
                    "to_deadline_ms": upper["deadline_ms"],
                    "task_success_rate_difference": (
                        upper["task_success_rate"] - lower["task_success_rate"]
                    ),
                    "execute_duty_cycle_difference": (
                        upper["execute_duty_cycle"] - lower["execute_duty_cycle"]
                    ),
                    "deadline_hold_tick_difference": (
                        upper["deadline_hold_ticks"] - lower["deadline_hold_ticks"]
                    ),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(cells),
        "total_rollouts": sum(cell["rollouts"] for cell in cells),
        "cells": cells,
        "adjacent_deadline_comparisons": comparisons,
        "analysis_boundary": (
            "Registered benchmark episodes are reported by suite, seed, and task "
            "cluster. They are not pooled as iid deployment draws or a universal "
            "VLA deadline estimate."
        ),
        "claim_boundaries": [
            "not an official LIBERO leaderboard score",
            "not a hard-real-time guarantee",
            "not hardware safety or real-robot deployment evidence",
            "not cross-model superiority",
        ],
    }


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _render_csv(summary: Mapping[str, Any]) -> str:
    fields = (
        "artifact_id",
        "task_suite",
        "seed",
        "deadline_ms",
        "rollouts",
        "task_successes",
        "task_success_rate",
        "wilson95_low",
        "wilson95_high",
        "tasks_all_success",
        "tasks_any_success",
        "control_ticks",
        "ticks_during_inference",
        "execute_ticks",
        "hold_ticks",
        "execute_duty_cycle",
        "deadline_hold_ticks",
        "no_response_hold_ticks",
        "response_level_deadline_rejections",
        "provider_failures",
        "source_manifest_sha256",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cell in summary["cells"]:
        row = {field: cell[field] for field in fields}
        for field in (
            "deadline_ms",
            "task_success_rate",
            "wilson95_low",
            "wilson95_high",
            "execute_duty_cycle",
        ):
            row[field] = _format_number(float(row[field]))
        writer.writerow(row)
    return output.getvalue()


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# pi0.5-LIBERO independent-clock deadline report",
        "",
        "Every source artifact passed the independent validator before this report was built.",
        "Tick-level deadline holds and response-level deadline rejections remain separate metrics.",
        "",
        "| Suite | Seed | Deadline | Task success (Wilson 95% CI) | Execute duty | Deadline hold ticks | Response rejections | Provider failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in summary["cells"]:
        lines.append(
            "| {task_suite} | {seed} | {deadline:g} ms | {successes}/{rollouts} "
            "({rate:.1%}, {low:.1%}-{high:.1%}) | {duty:.1%} | {holds:,} | "
            "{rejections:,} | {failures:,} |".format(
                task_suite=cell["task_suite"],
                seed=cell["seed"],
                deadline=cell["deadline_ms"],
                successes=cell["task_successes"],
                rollouts=cell["rollouts"],
                rate=cell["task_success_rate"],
                low=cell["wilson95_low"],
                high=cell["wilson95_high"],
                duty=cell["execute_duty_cycle"],
                holds=cell["deadline_hold_ticks"],
                rejections=cell["response_level_deadline_rejections"],
                failures=cell["provider_failures"],
            )
        )

    lines.extend(["", "## Adjacent registered deadlines", ""])
    if summary["adjacent_deadline_comparisons"]:
        lines.extend(
            [
                "| Suite | Seed | Comparison | Success-rate difference | Execute-duty difference | Deadline-hold difference |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in summary["adjacent_deadline_comparisons"]:
            lines.append(
                "| {task_suite} | {seed} | {start:g} -> {end:g} ms | {success:+.1%} | "
                "{duty:+.1%} | {holds:+,} |".format(
                    task_suite=comparison["task_suite"],
                    seed=comparison["seed"],
                    start=comparison["from_deadline_ms"],
                    end=comparison["to_deadline_ms"],
                    success=comparison["task_success_rate_difference"],
                    duty=comparison["execute_duty_cycle_difference"],
                    holds=comparison["deadline_hold_tick_difference"],
                )
            )
    else:
        lines.append("No suite/seed group contains more than one registered deadline.")

    lines.extend(
        [
            "",
            "## Statistical boundary",
            "",
            str(summary["analysis_boundary"]),
            "Wilson intervals describe the registered episode cells and do not correct for task clustering.",
            "",
            "## Source artifacts",
            "",
        ]
    )
    for cell in summary["cells"]:
        lines.append(
            f"- `{cell['artifact_id']}`: manifest `{cell['source_manifest_sha256']}`"
        )
    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in summary["claim_boundaries"])
    return "\n".join(lines) + "\n"


def render_outputs(summary: Mapping[str, Any]) -> dict[str, bytes]:
    outputs = {
        "deadline_cells.csv": _render_csv(summary).encode("utf-8"),
        "summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        "summary.md": _render_markdown(summary).encode("utf-8"),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(outputs.items())
        },
    }
    outputs["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return outputs


def write_outputs(output: pathlib.Path, outputs: Mapping[str, bytes]) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        temporary = output / f"{name}.tmp"
        temporary.write_bytes(outputs[name])
        temporary.replace(output / name)


def check_outputs(output: pathlib.Path, outputs: Mapping[str, bytes]) -> None:
    output = output.resolve()
    for name in OUTPUT_NAMES:
        path = output / name
        if not path.is_file():
            raise ValueError(f"missing generated report file: {path}")
        if path.read_bytes() != outputs[name]:
            raise ValueError(f"generated report is stale: {path}")
    extras = sorted(
        path.name for path in output.iterdir() if path.name not in OUTPUT_NAMES
    )
    if extras:
        raise ValueError(f"unexpected generated report files: {extras}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifacts", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_summary(args.artifacts)
    outputs = render_outputs(summary)
    if args.check:
        check_outputs(args.output_dir, outputs)
    else:
        write_outputs(args.output_dir, outputs)
    print(
        json.dumps(
            {
                "artifact_count": summary["artifact_count"],
                "checked": bool(args.check),
                "output_dir": str(args.output_dir.resolve()),
                "total_rollouts": summary["total_rollouts"],
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
