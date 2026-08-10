"""Build a validator-backed report for independent-clock action selection modes."""

from __future__ import annotations

import argparse
from collections import Counter
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

from integrations.openpi.libero_independent_clock import (  # noqa: E402
    AGE_ALIGNED_SUFFIX,
    RESPONSE_RELATIVE_CHUNK,
)
from integrations.openpi.validate_libero_independent_clock import (  # noqa: E402
    validate_artifact,
)


SCHEMA_VERSION = "armbench.pi05_selection_report.v1"
MANIFEST_SCHEMA_VERSION = "armbench.pi05_selection_report_manifest.v1"
OUTPUT_NAMES = ("pairs.csv", "summary.json", "summary.md", "manifest.json")
MODES = (AGE_ALIGNED_SUFFIX, RESPONSE_RELATIVE_CHUNK)
PAIR_FIELDS = (
    "task_suite",
    "seed",
    "task_id",
    "episode_index",
    "age_aligned_success",
    "response_relative_success",
    "success_difference",
    "age_aligned_execute_duty",
    "response_relative_execute_duty",
    "execute_duty_difference",
    "initial_state_sha256",
    "policy_input_sha256",
    "sampling_key_sha256",
    "sampling_noise_sha256",
    "query0_action_chunk_sha256",
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _evaluation_root(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    if resolved.name == "evaluation":
        return resolved
    candidate = resolved / "evaluation"
    if candidate.is_dir():
        return candidate
    raise ValueError(f"artifact has no evaluation directory: {path}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mcnemar_exact(aligned_only: int, relative_only: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant pairs."""

    if aligned_only < 0 or relative_only < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = aligned_only + relative_only
    if discordant == 0:
        return 1.0
    lower = min(aligned_only, relative_only)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _query_zero(runtime: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    requests = runtime.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{label} requests must be an array")
    candidates = [
        request
        for request in requests
        if isinstance(request, Mapping)
        and request.get("observation_sequence_id") == 0
        and request.get("actions") is not None
    ]
    if len(candidates) != 1:
        raise ValueError(f"{label} must contain exactly one completed query-0 request")
    return candidates[0]


def _load_artifact(path: pathlib.Path) -> dict[str, Any]:
    evaluation = _evaluation_root(path)
    validation = validate_artifact(evaluation)
    if not validation.valid:
        raise ValueError(
            "artifact validator failed for %s: %s"
            % (evaluation, "; ".join(validation.errors))
        )

    protocol = _require_mapping(
        _strict_json(evaluation / "resolved_protocol.json"), "resolved_protocol"
    )
    runtime_config = _require_mapping(protocol.get("runtime"), "protocol.runtime")
    sampling = _require_mapping(
        protocol.get("policy_sampling"), "protocol.policy_sampling"
    )
    matrix = _require_mapping(protocol.get("matrix"), "protocol.matrix")
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{evaluation} has no matrix cells")
    mode = str(runtime_config.get("action_selection_mode", ""))
    if mode not in MODES:
        raise ValueError(f"{evaluation} does not declare a scored selection mode")
    if (
        runtime_config.get("policy_input_audit")
        != "canonical_pi05_libero_request_sha256_v1"
    ):
        raise ValueError(f"{evaluation} does not require policy-input hashing")

    aggregate = _require_mapping(
        _strict_json(evaluation / "aggregate.json"), "aggregate"
    )
    if aggregate.get("complete") is not True:
        raise ValueError(f"{evaluation} is incomplete")
    if int(aggregate.get("total_failed_responses", -1)) != 0:
        raise ValueError(f"{evaluation} contains provider failures")
    if int(aggregate.get("episodes_with_inference_overlap", -1)) != len(cells):
        raise ValueError(f"{evaluation} lacks inference/simulation overlap")

    environment = _require_mapping(
        _strict_json(evaluation / "environment.json"), "environment"
    )
    server_metadata = _require_mapping(
        environment.get("server_metadata"), "environment.server_metadata"
    )
    attestation = _require_mapping(
        server_metadata.get("armbench_server_attestation"), "server attestation"
    )

    records: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for raw_cell in cells:
        cell = _require_mapping(raw_cell, "matrix cell")
        suite = str(cell["task_suite"])
        task_id = int(cell["task_id"])
        episode_index = int(cell["episode_index"])
        seed = int(sampling["seed"])
        episode_id = str(cell["episode_id"])
        payload = _require_mapping(
            _strict_json(evaluation / "episodes" / episode_id / "runtime.json"),
            f"{episode_id}.runtime",
        )
        runtime = _require_mapping(payload.get("runtime"), f"{episode_id}.runtime")
        query = _query_zero(runtime, episode_id)
        metadata = _require_mapping(
            query.get("response_metadata"), f"{episode_id}.query0.metadata"
        )
        sampling_audit = _require_mapping(
            metadata.get("policy_sampling"), f"{episode_id}.query0.sampling"
        )
        ticks = runtime.get("ticks")
        if not isinstance(ticks, list) or not ticks:
            raise ValueError(f"{episode_id} has no control ticks")
        execute_ticks = sum(
            isinstance(tick, Mapping) and tick.get("status") == "execute"
            for tick in ticks
        )
        hold_reasons = Counter(
            str(tick.get("reason"))
            for tick in ticks
            if isinstance(tick, Mapping) and tick.get("status") == "hold"
        )
        action_indices = Counter(
            int(tick["action_index"])
            for tick in ticks
            if isinstance(tick, Mapping)
            and tick.get("status") == "execute"
            and isinstance(tick.get("action_index"), int)
        )
        identity = (suite, seed, task_id, episode_index)
        if identity in records:
            raise ValueError(f"duplicate episode identity in {evaluation}: {identity}")
        records[identity] = {
            "task_success": bool(payload["task_success"]),
            "initial_state_sha256": str(payload["initial_state_sha256"]),
            "policy_input_sha256": str(metadata.get("policy_input_sha256", "")),
            "sampling_key_sha256": str(sampling_audit.get("key_sha256", "")),
            "sampling_noise_sha256": str(sampling_audit.get("noise_sha256", "")),
            "action_chunk_sha256": str(metadata.get("action_chunk_sha256", "")),
            "execute_ticks": execute_ticks,
            "control_ticks": len(ticks),
            "hold_reasons": dict(sorted(hold_reasons.items())),
            "action_indices": {
                str(index): count for index, count in sorted(action_indices.items())
            },
        }

    return {
        "artifact_id": evaluation.parent.name,
        "evaluation": evaluation,
        "manifest_sha256": _sha256(evaluation / "manifest.json"),
        "mode": mode,
        "seed": int(sampling["seed"]),
        "checkpoint": str(protocol["checkpoint"]),
        "checkpoint_content_sha256": str(attestation["checkpoint_content_sha256"]),
        "openpi_commit": str(protocol["openpi_commit"]),
        "armbench_commit": str(environment.get("armbench_git_commit")),
        "runtime_source_sha256": environment.get("runtime_source_sha256"),
        "control_period_ms": float(runtime_config["control_period_ms"]),
        "deadline_ms": float(runtime_config["deadline_ms"]),
        "submit_every_ticks": int(runtime_config["submit_every_ticks"]),
        "records": records,
    }


def _require_equal(values: Sequence[Any], label: str) -> Any:
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"paired artifacts disagree on {label}")
    return values[0]


def build_summary(paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least two artifacts are required")
    artifacts = [_load_artifact(path) for path in paths]
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for artifact in artifacts:
        key = (artifact["seed"], artifact["mode"])
        if key in grouped:
            raise ValueError(f"duplicate seed/mode artifact: {key}")
        grouped[key] = artifact
    seeds = sorted({artifact["seed"] for artifact in artifacts})
    for seed in seeds:
        missing = [mode for mode in MODES if (seed, mode) not in grouped]
        if missing:
            raise ValueError(f"seed {seed} is missing modes: {', '.join(missing)}")

    _require_equal([artifact["checkpoint"] for artifact in artifacts], "checkpoint")
    _require_equal(
        [artifact["checkpoint_content_sha256"] for artifact in artifacts],
        "checkpoint content",
    )
    _require_equal(
        [artifact["openpi_commit"] for artifact in artifacts], "OpenPI commit"
    )
    _require_equal(
        [artifact["armbench_commit"] for artifact in artifacts], "ArmBench commit"
    )
    _require_equal(
        [artifact["runtime_source_sha256"] for artifact in artifacts],
        "runtime source hashes",
    )
    _require_equal(
        [artifact["control_period_ms"] for artifact in artifacts], "control period"
    )
    _require_equal([artifact["deadline_ms"] for artifact in artifacts], "deadline")
    _require_equal(
        [artifact["submit_every_ticks"] for artifact in artifacts],
        "submission cadence",
    )

    pairs: list[dict[str, Any]] = []
    mode_totals = {
        mode: {
            "rollouts": 0,
            "successes": 0,
            "execute_ticks": 0,
            "control_ticks": 0,
            "hold_reasons": Counter(),
            "action_indices": Counter(),
        }
        for mode in MODES
    }
    seed_summaries = []
    for seed in seeds:
        aligned = grouped[(seed, AGE_ALIGNED_SUFFIX)]
        relative = grouped[(seed, RESPONSE_RELATIVE_CHUNK)]
        if set(aligned["records"]) != set(relative["records"]):
            raise ValueError(
                f"seed {seed} mode artifacts have different episode matrices"
            )
        seed_aligned_successes = 0
        seed_relative_successes = 0
        for identity in sorted(aligned["records"]):
            left = aligned["records"][identity]
            right = relative["records"][identity]
            pairing_fields = (
                "initial_state_sha256",
                "policy_input_sha256",
                "sampling_key_sha256",
                "sampling_noise_sha256",
                "action_chunk_sha256",
            )
            mismatches = [
                field for field in pairing_fields if left[field] != right[field]
            ]
            if mismatches:
                raise ValueError(
                    "query-0 pairing failed for %s: %s"
                    % (identity, ", ".join(mismatches))
                )
            if any(
                not isinstance(left[field], str) or len(left[field]) != 64
                for field in pairing_fields
            ):
                raise ValueError(f"query-0 pairing hash is invalid for {identity}")
            aligned_success = bool(left["task_success"])
            relative_success = bool(right["task_success"])
            seed_aligned_successes += aligned_success
            seed_relative_successes += relative_success
            pair = {
                "task_suite": identity[0],
                "seed": identity[1],
                "task_id": identity[2],
                "episode_index": identity[3],
                "age_aligned_success": aligned_success,
                "response_relative_success": relative_success,
                "success_difference": int(aligned_success) - int(relative_success),
                "age_aligned_execute_duty": left["execute_ticks"]
                / left["control_ticks"],
                "response_relative_execute_duty": right["execute_ticks"]
                / right["control_ticks"],
                "execute_duty_difference": left["execute_ticks"] / left["control_ticks"]
                - right["execute_ticks"] / right["control_ticks"],
                "initial_state_sha256": left["initial_state_sha256"],
                "policy_input_sha256": left["policy_input_sha256"],
                "sampling_key_sha256": left["sampling_key_sha256"],
                "sampling_noise_sha256": left["sampling_noise_sha256"],
                "query0_action_chunk_sha256": left["action_chunk_sha256"],
            }
            pairs.append(pair)
            for mode, record in (
                (AGE_ALIGNED_SUFFIX, left),
                (RESPONSE_RELATIVE_CHUNK, right),
            ):
                total = mode_totals[mode]
                total["rollouts"] += 1
                total["successes"] += bool(record["task_success"])
                total["execute_ticks"] += record["execute_ticks"]
                total["control_ticks"] += record["control_ticks"]
                total["hold_reasons"].update(record["hold_reasons"])
                total["action_indices"].update(record["action_indices"])
        seed_summaries.append(
            {
                "seed": seed,
                "pairs": len(aligned["records"]),
                "age_aligned_successes": seed_aligned_successes,
                "response_relative_successes": seed_relative_successes,
            }
        )

    aligned_only = sum(
        pair["age_aligned_success"] and not pair["response_relative_success"]
        for pair in pairs
    )
    relative_only = sum(
        pair["response_relative_success"] and not pair["age_aligned_success"]
        for pair in pairs
    )
    both_success = sum(
        pair["age_aligned_success"] and pair["response_relative_success"]
        for pair in pairs
    )
    both_failure = len(pairs) - aligned_only - relative_only - both_success
    rendered_modes = {}
    for mode, totals in mode_totals.items():
        rendered_modes[mode] = {
            "rollouts": totals["rollouts"],
            "successes": totals["successes"],
            "success_rate": totals["successes"] / totals["rollouts"],
            "execute_ticks": totals["execute_ticks"],
            "control_ticks": totals["control_ticks"],
            "execute_duty_cycle": totals["execute_ticks"] / totals["control_ticks"],
            "hold_reasons": dict(sorted(totals["hold_reasons"].items())),
            "action_indices": dict(sorted(totals["action_indices"].items())),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(artifacts),
        "pair_count": len(pairs),
        "total_rollouts": 2 * len(pairs),
        "modes": rendered_modes,
        "paired_success": {
            "both_success": both_success,
            "age_aligned_only": aligned_only,
            "response_relative_only": relative_only,
            "both_failure": both_failure,
            "success_rate_difference": (
                rendered_modes[AGE_ALIGNED_SUFFIX]["success_rate"]
                - rendered_modes[RESPONSE_RELATIVE_CHUNK]["success_rate"]
            ),
            "mcnemar_exact_two_sided_p": mcnemar_exact(aligned_only, relative_only),
        },
        "seed_summaries": seed_summaries,
        "pairs": pairs,
        "pairing_gate": {
            "valid": True,
            "checked_fields": [
                "initial_state_sha256",
                "policy_input_sha256",
                "sampling_key_sha256",
                "sampling_noise_sha256",
                "query0_action_chunk_sha256",
            ],
        },
        "sources": [
            {
                "artifact_id": artifact["artifact_id"],
                "mode": artifact["mode"],
                "seed": artifact["seed"],
                "manifest_sha256": artifact["manifest_sha256"],
            }
            for artifact in sorted(
                artifacts, key=lambda item: (item["seed"], item["mode"])
            )
        ],
        "analysis_boundary": (
            "Registered episodes are paired within task, initial state, and joint seed. "
            "Task and seed blocks are not iid deployment samples, and query-0 equality "
            "does not imply later observations remain equal after modes diverge."
        ),
        "claim_boundaries": [
            "not an official LIBERO leaderboard score",
            "not a hard-real-time guarantee",
            "not hardware safety or real-robot deployment evidence",
            "not cross-model superiority",
        ],
    }


def _format_float(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _render_csv(summary: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PAIR_FIELDS, lineterminator="\n")
    writer.writeheader()
    for pair in summary["pairs"]:
        row = dict(pair)
        for field in (
            "age_aligned_execute_duty",
            "response_relative_execute_duty",
            "execute_duty_difference",
        ):
            row[field] = _format_float(float(row[field]))
        writer.writerow(row)
    return output.getvalue()


def _render_markdown(summary: Mapping[str, Any]) -> str:
    aligned = summary["modes"][AGE_ALIGNED_SUFFIX]
    relative = summary["modes"][RESPONSE_RELATIVE_CHUNK]
    paired = summary["paired_success"]
    lines = [
        "# pi0.5 independent-clock action-selection report",
        "",
        "Every source artifact passed the independent validator and every query-0 pairing gate passed.",
        "",
        "| Mode | Success | Execute duty | Control ticks |",
        "| --- | ---: | ---: | ---: |",
        "| `{}` | {}/{} ({:.1%}) | {:.1%} | {:,} |".format(
            AGE_ALIGNED_SUFFIX,
            aligned["successes"],
            aligned["rollouts"],
            aligned["success_rate"],
            aligned["execute_duty_cycle"],
            aligned["control_ticks"],
        ),
        "| `{}` | {}/{} ({:.1%}) | {:.1%} | {:,} |".format(
            RESPONSE_RELATIVE_CHUNK,
            relative["successes"],
            relative["rollouts"],
            relative["success_rate"],
            relative["execute_duty_cycle"],
            relative["control_ticks"],
        ),
        "",
        "## Paired outcome",
        "",
        "- Pairs: `{}`".format(summary["pair_count"]),
        "- Both success / aligned only / response-relative only / both failure: "
        "`{}` / `{}` / `{}` / `{}`".format(
            paired["both_success"],
            paired["age_aligned_only"],
            paired["response_relative_only"],
            paired["both_failure"],
        ),
        "- Success-rate difference (`age_aligned_suffix - response_relative_chunk`): `{:+.2%}`".format(
            paired["success_rate_difference"]
        ),
        "- Exact two-sided McNemar p: `{:.8g}`".format(
            paired["mcnemar_exact_two_sided_p"]
        ),
        "",
        "## Seed blocks",
        "",
        "| Joint seed | Pairs | Age-aligned success | Response-relative success |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary["seed_summaries"]:
        lines.append(
            "| {seed} | {pairs} | {age}/{pairs} | {relative}/{pairs} |".format(
                seed=row["seed"],
                pairs=row["pairs"],
                age=row["age_aligned_successes"],
                relative=row["response_relative_successes"],
            )
        )
    lines.extend(
        [
            "",
            "## Pairing gate",
            "",
            "All paired episodes matched initial state plus query-0 policy input, sampling key, sampling noise, and action chunk hashes.",
            "",
            "## Analysis boundary",
            "",
            str(summary["analysis_boundary"]),
            "",
            "## Source artifacts",
            "",
        ]
    )
    for source in summary["sources"]:
        lines.append(
            "- `{artifact_id}`: mode `{mode}`, seed `{seed}`, manifest `{manifest_sha256}`".format(
                **source
            )
        )
    lines.extend(["", "## Claim boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in summary["claim_boundaries"])
    return "\n".join(lines) + "\n"


def render_outputs(summary: Mapping[str, Any]) -> dict[str, bytes]:
    outputs = {
        "pairs.csv": _render_csv(summary).encode("utf-8"),
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
                "pair_count": summary["pair_count"],
                "pairing_gate_valid": summary["pairing_gate"]["valid"],
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
