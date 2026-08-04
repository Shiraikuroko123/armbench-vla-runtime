"""Read-only paired analysis for a frozen pi0.5-LIBERO alignment run."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.libero_compose_run import validate_run_manifest


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_libero_latency_aligned_analysis.v1"
SOURCE_SCHEMA_VERSION = "armbench.pi05_libero_async.v1"
ROOT_SCHEMA_VERSION = "armbench.pi05_libero_container_run.v1"
ASYNC_UNGUARDED = "async_unguarded"
LATENCY_ALIGNED = "latency_aligned"
TARGET_MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED)
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_RESAMPLES = 10_000
FROZEN_SUITE = "libero_spatial"
FROZEN_TASK_IDS = tuple(range(10))
FROZEN_EPISODE_INDICES = tuple(range(5))
FROZEN_REPLAN_STEPS = (5,)
FROZEN_LATENCY_STEPS = (0, 2, 4)
FROZEN_PRIMARY_LATENCY_STEPS = 4
FROZEN_SEED = 7
SOURCE_ARTIFACT_PATH = "evaluation/per_episode.csv"
FROZEN_ARMBENCH_RUN_COMMIT = "30676d2d3ff43e3df0750e2ad01f94748293cff5"
FROZEN_ALIGNMENT_IMPLEMENTATION_COMMIT = (
    "cccbe351a1a4523c65d01eff2997580f7ca83649"
)
FROZEN_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
FROZEN_POLICY_CONFIG = "pi05_libero"
FROZEN_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
FROZEN_CHECKPOINT_CONTENT_SHA256 = (
    "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"
)
EMPTY_GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
PAIR_KEY_FIELDS = (
    "task_suite",
    "task_id",
    "episode_index",
    "replan_steps",
    "latency_steps",
    "seed",
    "initial_state_sha256",
)
REQUIRED_COLUMNS = frozenset(
    (
        "schema_version",
        "episode_id",
        "pair_id",
        "condition_order",
        "task_suite",
        "task_id",
        "episode_index",
        "task_description",
        "mode",
        "replan_steps",
        "latency_steps",
        "injected_latency_ms",
        "fixed_refresh_interval",
        "seed",
        "success",
        "initial_state_sha256",
        "policy_queries",
        "failure_category",
        "failure_type",
        "failure_message",
        "video_required",
    )
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*\Z")


@dataclasses.dataclass(frozen=True, order=True)
class BasePairKey:
    task_suite: str
    task_id: int
    episode_index: int
    replan_steps: int
    latency_steps: int
    seed: int


@dataclasses.dataclass(frozen=True, order=True)
class PairKey:
    task_suite: str
    task_id: int
    episode_index: int
    replan_steps: int
    latency_steps: int
    seed: int
    initial_state_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Episode:
    episode_id: str
    pair_id: str
    condition_order: int
    task_description: str
    mode: str
    key: PairKey
    success: bool
    policy_queries: int
    failure_category: str
    failure_type: str
    failure_message: str
    video_required: bool

    @property
    def base_key(self) -> BasePairKey:
        return BasePairKey(
            self.key.task_suite,
            self.key.task_id,
            self.key.episode_index,
            self.key.replan_steps,
            self.key.latency_steps,
            self.key.seed,
        )

    @property
    def runtime_failure(self) -> bool:
        return bool(self.failure_category or self.failure_type)


@dataclasses.dataclass(frozen=True)
class FrozenProtocol:
    task_suite: str
    task_ids: Tuple[int, ...]
    episode_indices: Tuple[int, ...]
    replan_steps: Tuple[int, ...]
    latency_steps: Tuple[int, ...]
    seed: int
    planned_rollouts: int
    matched_groups: int
    bootstrap_resamples: int


@dataclasses.dataclass(frozen=True)
class SourceEvidence:
    csv_path: pathlib.Path
    csv_data: bytes
    csv_sha256: str
    csv_bytes: int
    evaluation_manifest_sha256: str
    root_manifest_sha256: str
    protocol_sha256: str
    environment_sha256: str
    independently_validated_files: int
    protocol: FrozenProtocol


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_mapping(path: pathlib.Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("%s is not readable valid JSON: %s" % (label, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("%s must contain a JSON object" % label)
    return value


def _strict_int(value: Any, label: str) -> int:
    text = str(value)
    if not _NONNEGATIVE_INTEGER.fullmatch(text):
        raise ValueError("%s must be a canonical nonnegative integer" % label)
    return int(text)


def _strict_int_list(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a nonempty list" % label)
    parsed = tuple(_strict_int(item, label) for item in value)
    if len(parsed) != len(set(parsed)) or tuple(sorted(parsed)) != parsed:
        raise ValueError("%s must be sorted and unique" % label)
    return parsed


def _strict_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % label)


def _manifest_file_record(
    manifest: Mapping[str, Any], relative: str, path: pathlib.Path, label: str
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(relative), dict):
        raise ValueError("%s does not protect %s" % (label, relative))
    record = files[relative]
    if record.get("bytes") != path.stat().st_size:
        raise ValueError("%s byte count mismatch for %s" % (label, relative))
    if record.get("sha256") != _sha256(path):
        raise ValueError("%s SHA-256 mismatch for %s" % (label, relative))


def _manifest_buffer_record(
    manifest: Mapping[str, Any], relative: str, data: bytes, label: str
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(relative), dict):
        raise ValueError("%s does not protect %s" % (label, relative))
    record = files[relative]
    if record.get("bytes") != len(data):
        raise ValueError("%s byte count mismatch for %s" % (label, relative))
    if record.get("sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("%s SHA-256 mismatch for %s" % (label, relative))


def _validate_frozen_identity(
    protocol: Mapping[str, Any], environment: Mapping[str, Any]
) -> None:
    expected_protocol = {
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "policy_config": FROZEN_POLICY_CONFIG,
        "declared_checkpoint": FROZEN_CHECKPOINT,
        "checkpoint_provenance": (
            "server_attestation_with_checkpoint_content_sha256"
        ),
        "runtime_failure_policy": "abort_formal_run",
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError("frozen protocol identity mismatch: %s" % field)
    expected_environment = {
        "armbench_git_commit": FROZEN_ARMBENCH_RUN_COMMIT,
        "armbench_git_status": "",
        "armbench_git_diff_sha256": EMPTY_GIT_DIFF_SHA256,
        "openpi_git_commit": FROZEN_OPENPI_COMMIT,
    }
    for field, expected in expected_environment.items():
        if environment.get(field) != expected:
            raise ValueError("frozen environment identity mismatch: %s" % field)
    arguments = environment.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("frozen environment has no arguments object")
    if arguments.get("video_mode") != "all":
        raise ValueError("frozen confirmatory run requires video_mode=all")
    if arguments.get("allow_unattested_server") is not False:
        raise ValueError("frozen confirmatory run requires formal server attestation")
    metadata = environment.get("server_metadata")
    attestation = (
        metadata.get("armbench_server_attestation")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(attestation, Mapping):
        raise ValueError("frozen confirmatory run has no server attestation")
    expected_attestation = {
        "policy_config": FROZEN_POLICY_CONFIG,
        "checkpoint_uri": FROZEN_CHECKPOINT,
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            raise ValueError("frozen checkpoint attestation mismatch: %s" % field)


def _validate_protocol(
    value: Mapping[str, Any], progress: Mapping[str, Any]
) -> FrozenProtocol:
    if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("resolved protocol schema is not recognized")
    matrix = value.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("resolved protocol has no matrix")
    modes = matrix.get("modes")
    if (
        not isinstance(modes, list)
        or any(not isinstance(mode, str) for mode in modes)
        or tuple(sorted(modes)) != tuple(sorted(TARGET_MODES))
    ):
        raise ValueError("protocol must contain exactly async_unguarded and latency_aligned")
    suites = matrix.get("task_suites")
    if suites != [FROZEN_SUITE]:
        raise ValueError("protocol task suite does not match the frozen alignment study")
    task_ids = _strict_int_list(matrix.get("task_ids"), "matrix.task_ids")
    episodes = _strict_int_list(
        matrix.get("episode_indices"), "matrix.episode_indices"
    )
    replans = _strict_int_list(matrix.get("replan_steps"), "matrix.replan_steps")
    latencies = _strict_int_list(
        matrix.get("latency_steps"), "matrix.latency_steps"
    )
    seed = _strict_int(value.get("seed"), "protocol.seed")
    bootstrap = _strict_int(
        value.get("bootstrap_resamples"), "protocol.bootstrap_resamples"
    )
    frozen = (
        task_ids == FROZEN_TASK_IDS
        and episodes == FROZEN_EPISODE_INDICES
        and replans == FROZEN_REPLAN_STEPS
        and latencies == FROZEN_LATENCY_STEPS
        and seed == FROZEN_SEED
        and bootstrap == BOOTSTRAP_RESAMPLES
    )
    if not frozen:
        raise ValueError("resolved protocol does not match the frozen confirmatory matrix")
    matched_groups = len(task_ids) * len(episodes) * len(replans) * len(latencies)
    planned = matched_groups * len(TARGET_MODES)
    if _strict_int(
        matrix.get("matched_condition_groups"), "matrix.matched_condition_groups"
    ) != matched_groups:
        raise ValueError("matrix matched_condition_groups mismatch")
    if _strict_int(matrix.get("rollouts"), "matrix.rollouts") != planned:
        raise ValueError("matrix rollout count mismatch")
    if _strict_int(
        progress.get("planned_rollouts"), "progress.planned_rollouts"
    ) != planned:
        raise ValueError("progress planned_rollouts mismatch")
    completed = _strict_int(
        progress.get("completed_rollouts"), "progress.completed_rollouts"
    )
    if completed != planned or progress.get("complete") is not True:
        raise ValueError("analysis requires a complete frozen run")
    return FrozenProtocol(
        task_suite=FROZEN_SUITE,
        task_ids=task_ids,
        episode_indices=episodes,
        replan_steps=replans,
        latency_steps=latencies,
        seed=seed,
        planned_rollouts=planned,
        matched_groups=matched_groups,
        bootstrap_resamples=bootstrap,
    )


def validate_frozen_source(per_episode_csv: pathlib.Path) -> SourceEvidence:
    source = per_episode_csv.resolve()
    if source.name != "per_episode.csv" or source.parent.name != "evaluation":
        raise ValueError("input must be an artifact evaluation/per_episode.csv")
    if source.is_symlink() or not source.is_file():
        raise ValueError("per_episode.csv must be a regular file")
    evaluation = source.parent
    root = evaluation.parent
    independent_validation = validate_run_manifest(root)
    if (
        not isinstance(independent_validation, Mapping)
        or independent_validation.get("valid") is not True
        or independent_validation.get("complete") is not True
    ):
        errors = (
            independent_validation.get("errors", [])
            if isinstance(independent_validation, Mapping)
            else []
        )
        detail = "; ".join(str(error) for error in list(errors)[:3])
        raise ValueError(
            "independent full run validation failed%s"
            % ((": " + detail) if detail else "")
        )
    files_checked = independent_validation.get("files_checked")
    if isinstance(files_checked, bool) or not isinstance(files_checked, int):
        raise ValueError("independent full run validation returned no file count")
    evaluation_manifest_path = evaluation / "manifest.json"
    root_manifest_path = root / "manifest.json"
    evaluation_manifest = _json_mapping(
        evaluation_manifest_path, "evaluation manifest"
    )
    root_manifest = _json_mapping(root_manifest_path, "root manifest")
    if evaluation_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("evaluation manifest schema is not recognized")
    if root_manifest.get("schema_version") != ROOT_SCHEMA_VERSION:
        raise ValueError("root manifest schema is not recognized")
    if root_manifest.get("complete") is not True:
        raise ValueError("root manifest is not complete")
    _manifest_file_record(
        evaluation_manifest, "per_episode.csv", source, "evaluation manifest"
    )
    _manifest_file_record(
        root_manifest,
        "evaluation/per_episode.csv",
        source,
        "root manifest",
    )
    progress = _json_mapping(evaluation / "progress.json", "progress")
    integrity = _json_mapping(evaluation / "integrity.json", "integrity")
    finalization = _json_mapping(root / "finalization.json", "finalization")
    artifact_validation = _json_mapping(
        root / "artifact_validation.json", "artifact validation"
    )
    protected_evaluation_files = (
        "progress.json",
        "integrity.json",
        "resolved_protocol.json",
        "environment.json",
    )
    for relative in protected_evaluation_files:
        path = evaluation / relative
        _manifest_file_record(
            evaluation_manifest, relative, path, "evaluation manifest"
        )
        _manifest_file_record(
            root_manifest,
            "evaluation/" + relative,
            path,
            "root manifest",
        )
    _manifest_file_record(
        root_manifest,
        "evaluation/manifest.json",
        evaluation_manifest_path,
        "root manifest",
    )
    for relative in ("finalization.json", "artifact_validation.json"):
        _manifest_file_record(
            root_manifest, relative, root / relative, "root manifest"
        )
    if integrity.get("valid") is not True:
        raise ValueError("evaluation integrity is not valid")
    if finalization.get("complete") is not True:
        raise ValueError("root finalization is not complete")
    if artifact_validation.get("valid") is not True:
        raise ValueError("recorded independent artifact validation is not valid")
    protocol_path = evaluation / "resolved_protocol.json"
    protocol_value = _json_mapping(protocol_path, "resolved protocol")
    environment_path = evaluation / "environment.json"
    environment_value = _json_mapping(environment_path, "environment")
    _validate_frozen_identity(protocol_value, environment_value)
    protocol = _validate_protocol(protocol_value, progress)
    try:
        csv_data = source.read_bytes()
    except OSError as exc:
        raise ValueError("per_episode.csv cannot be snapshotted: %s" % exc) from exc
    _manifest_buffer_record(
        evaluation_manifest, "per_episode.csv", csv_data, "evaluation manifest"
    )
    _manifest_buffer_record(
        root_manifest,
        SOURCE_ARTIFACT_PATH,
        csv_data,
        "root manifest",
    )
    return SourceEvidence(
        csv_path=source,
        csv_data=csv_data,
        csv_sha256=hashlib.sha256(csv_data).hexdigest(),
        csv_bytes=len(csv_data),
        evaluation_manifest_sha256=_sha256(evaluation_manifest_path),
        root_manifest_sha256=_sha256(root_manifest_path),
        protocol_sha256=_sha256(protocol_path),
        environment_sha256=_sha256(environment_path),
        independently_validated_files=files_checked,
        protocol=protocol,
    )


def _parse_episode(raw: Mapping[str, str], row_number: int) -> Episode:
    label = "per_episode.csv row %d" % row_number
    if raw.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("%s schema_version mismatch" % label)
    mode = str(raw.get("mode", ""))
    if mode not in TARGET_MODES:
        raise ValueError("%s contains unexpected mode %r" % (label, mode))
    if raw.get("fixed_refresh_interval", "") != "":
        raise ValueError("%s mixes a fixed-refresh protocol" % label)
    task_suite = str(raw.get("task_suite", ""))
    task_id = _strict_int(raw.get("task_id"), label + ".task_id")
    episode_index = _strict_int(
        raw.get("episode_index"), label + ".episode_index"
    )
    replan_steps = _strict_int(
        raw.get("replan_steps"), label + ".replan_steps"
    )
    latency_steps = _strict_int(
        raw.get("latency_steps"), label + ".latency_steps"
    )
    seed = _strict_int(raw.get("seed"), label + ".seed")
    initial_hash = str(raw.get("initial_state_sha256", ""))
    if not _SHA256.fullmatch(initial_hash):
        raise ValueError("%s initial_state_sha256 is invalid" % label)
    try:
        injected_latency_ms = float(str(raw.get("injected_latency_ms", "")))
    except ValueError as exc:
        raise ValueError("%s injected_latency_ms is invalid" % label) from exc
    if not math.isfinite(injected_latency_ms) or not math.isclose(
        injected_latency_ms, latency_steps * 50.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("%s injected latency does not match the 20 Hz protocol" % label)
    success = _strict_bool(str(raw.get("success", "")), label + ".success")
    failure_category = str(raw.get("failure_category", ""))
    failure_type = str(raw.get("failure_type", ""))
    if success and (failure_category or failure_type):
        raise ValueError("%s cannot be successful and record a runtime failure" % label)
    video_required = _strict_bool(
        str(raw.get("video_required", "")), label + ".video_required"
    )
    if not video_required:
        raise ValueError("%s violates frozen video_mode=all coverage" % label)
    return Episode(
        episode_id=str(raw.get("episode_id", "")),
        pair_id=str(raw.get("pair_id", "")),
        condition_order=_strict_int(
            raw.get("condition_order"), label + ".condition_order"
        ),
        task_description=str(raw.get("task_description", "")),
        mode=mode,
        key=PairKey(
            task_suite,
            task_id,
            episode_index,
            replan_steps,
            latency_steps,
            seed,
            initial_hash,
        ),
        success=success,
        policy_queries=_strict_int(
            raw.get("policy_queries"), label + ".policy_queries"
        ),
        failure_category=failure_category,
        failure_type=failure_type,
        failure_message=str(raw.get("failure_message", "")),
        video_required=video_required,
    )


def load_episodes(source: SourceEvidence) -> List[Episode]:
    try:
        csv_text = source.csv_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("per_episode.csv is not valid UTF-8: %s" % exc) from exc
    with io.StringIO(csv_text, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError("per_episode.csv contains duplicate header fields")
        missing_columns = sorted(REQUIRED_COLUMNS - set(fields))
        if missing_columns:
            raise ValueError(
                "per_episode.csv is missing required columns: %s"
                % ", ".join(missing_columns)
            )
        episodes = [_parse_episode(row, index) for index, row in enumerate(reader, 2)]
    if len(episodes) != source.protocol.planned_rollouts:
        raise ValueError(
            "per_episode.csv contains %d rows, expected %d"
            % (len(episodes), source.protocol.planned_rollouts)
        )
    return episodes


def _expected_base_keys(protocol: FrozenProtocol) -> set[BasePairKey]:
    return {
        BasePairKey(protocol.task_suite, task, episode, replan, latency, protocol.seed)
        for task in protocol.task_ids
        for episode in protocol.episode_indices
        for replan in protocol.replan_steps
        for latency in protocol.latency_steps
    }


def pair_episodes(
    episodes: Sequence[Episode], protocol: FrozenProtocol
) -> List[Tuple[PairKey, Episode, Episode]]:
    expected = _expected_base_keys(protocol)
    grouped: Dict[BasePairKey, Dict[str, Episode]] = {}
    episode_ids: set[str] = set()
    condition_orders: set[int] = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ValueError("duplicate episode_id: %s" % episode.episode_id)
        episode_ids.add(episode.episode_id)
        if episode.condition_order in condition_orders:
            raise ValueError("duplicate condition_order: %d" % episode.condition_order)
        condition_orders.add(episode.condition_order)
        if episode.base_key not in expected:
            raise ValueError("episode is outside the frozen matrix: %r" % (episode.base_key,))
        modes = grouped.setdefault(episode.base_key, {})
        if episode.mode in modes:
            raise ValueError(
                "duplicate mode %s for pair %r" % (episode.mode, episode.base_key)
            )
        modes[episode.mode] = episode
    if sorted(condition_orders) != list(range(len(episodes))):
        raise ValueError("condition_order must be contiguous from zero")
    missing_groups = sorted(expected - set(grouped))
    extra_groups = sorted(set(grouped) - expected)
    if missing_groups or extra_groups:
        raise ValueError(
            "frozen matrix group mismatch: missing=%r extra=%r"
            % (missing_groups, extra_groups)
        )
    pairs: List[Tuple[PairKey, Episode, Episode]] = []
    for base_key in sorted(expected):
        modes = grouped[base_key]
        missing_modes = sorted(set(TARGET_MODES) - set(modes))
        extra_modes = sorted(set(modes) - set(TARGET_MODES))
        if missing_modes or extra_modes:
            raise ValueError(
                "pair %r mode mismatch: missing=%r extra=%r"
                % (base_key, missing_modes, extra_modes)
            )
        baseline = modes[ASYNC_UNGUARDED]
        aligned = modes[LATENCY_ALIGNED]
        if baseline.key.initial_state_sha256 != aligned.key.initial_state_sha256:
            raise ValueError("initial_state_sha256 mismatch for pair %r" % (base_key,))
        if baseline.pair_id != aligned.pair_id:
            raise ValueError("pair_id mismatch for pair %r" % (base_key,))
        if baseline.task_description != aligned.task_description:
            raise ValueError("task description mismatch for pair %r" % (base_key,))
        if abs(baseline.condition_order - aligned.condition_order) != 1:
            raise ValueError("pair modes are not adjacent for pair %r" % (base_key,))
        pairs.append((baseline.key, baseline, aligned))
    ordered_pairs = sorted(
        pairs,
        key=lambda pair: min(pair[1].condition_order, pair[2].condition_order),
    )
    first_mode_order: Optional[Tuple[str, str]] = None
    for pair_index, (_, baseline, aligned) in enumerate(ordered_pairs):
        episodes_by_order = sorted(
            (baseline, aligned), key=lambda episode: episode.condition_order
        )
        actual_orders = [episode.condition_order for episode in episodes_by_order]
        expected_orders = [pair_index * 2, pair_index * 2 + 1]
        if actual_orders != expected_orders:
            raise ValueError(
                "pair %s is not condition-order block %r"
                % (baseline.pair_id, expected_orders)
            )
        actual_mode_order = tuple(episode.mode for episode in episodes_by_order)
        if first_mode_order is None:
            first_mode_order = actual_mode_order
        desired = (
            first_mode_order
            if pair_index % 2 == 0
            else tuple(reversed(first_mode_order))
        )
        if actual_mode_order != desired:
            raise ValueError(
                "pair %s does not follow alternating mode order" % baseline.pair_id
            )
    return pairs


def _wilson_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires a positive trial count")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_bootstrap_interval(
    differences: Sequence[float], resamples: int
) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one pair")
    if values.size == 1:
        return float(values[0]), float(values[0])
    random = np.random.default_rng(BOOTSTRAP_SEED)
    indices = random.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _mcnemar_exact_p(aligned_wins: int, baseline_wins: int) -> float:
    discordant = aligned_wins + baseline_wins
    if discordant == 0:
        return 1.0
    smaller = min(aligned_wins, baseline_wins)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2.0 ** discordant))


def _holm_adjust(rows: List[Dict[str, Any]]) -> None:
    ordered = sorted(
        range(len(rows)), key=lambda index: float(rows[index]["mcnemar_exact_p"])
    )
    running_maximum = 0.0
    hypotheses = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(
            1.0,
            float(rows[index]["mcnemar_exact_p"]) * (hypotheses - rank),
        )
        running_maximum = max(running_maximum, adjusted)
        rows[index]["mcnemar_holm_p"] = running_maximum


def _failure_breakdown(episodes: Iterable[Episode], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for episode in episodes:
        if not episode.runtime_failure:
            continue
        value = str(getattr(episode, field)) or "unspecified"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _pair_row(key: PairKey, baseline: Episode, aligned: Episode) -> Dict[str, Any]:
    return {
        **key.to_dict(),
        "pair_id": baseline.pair_id,
        "async_unguarded_success": baseline.success,
        "latency_aligned_success": aligned.success,
        "aligned_minus_async_success": int(aligned.success) - int(baseline.success),
        "async_unguarded_policy_queries": baseline.policy_queries,
        "latency_aligned_policy_queries": aligned.policy_queries,
        "aligned_minus_async_policy_queries": (
            aligned.policy_queries - baseline.policy_queries
        ),
        "async_unguarded_runtime_failure": baseline.runtime_failure,
        "latency_aligned_runtime_failure": aligned.runtime_failure,
        "async_unguarded_failure_category": baseline.failure_category,
        "latency_aligned_failure_category": aligned.failure_category,
        "async_unguarded_failure_type": baseline.failure_type,
        "latency_aligned_failure_type": aligned.failure_type,
    }


def analyze_pairs(
    pairs: Sequence[Tuple[PairKey, Episode, Episode]], protocol: FrozenProtocol
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    pair_rows = [_pair_row(*pair) for pair in pairs]
    strata: List[Dict[str, Any]] = []
    runtime_failures: List[Dict[str, Any]] = []
    for _, baseline, aligned in pairs:
        for episode in (baseline, aligned):
            if episode.runtime_failure:
                runtime_failures.append(
                    {
                        **episode.key.to_dict(),
                        "episode_id": episode.episode_id,
                        "mode": episode.mode,
                        "success": episode.success,
                        "failure_category": episode.failure_category,
                        "failure_type": episode.failure_type,
                        "failure_message": episode.failure_message,
                    }
                )
    for latency in protocol.latency_steps:
        selected = [pair for pair in pairs if pair[0].latency_steps == latency]
        baseline_rows = [pair[1] for pair in selected]
        aligned_rows = [pair[2] for pair in selected]
        baseline_successes = sum(row.success for row in baseline_rows)
        aligned_successes = sum(row.success for row in aligned_rows)
        baseline_ci = _wilson_interval(baseline_successes, len(baseline_rows))
        aligned_ci = _wilson_interval(aligned_successes, len(aligned_rows))
        differences = [
            float(aligned.success) - float(baseline.success)
            for _, baseline, aligned in selected
        ]
        bootstrap = _paired_bootstrap_interval(
            differences, protocol.bootstrap_resamples
        )
        aligned_wins = sum(value > 0.0 for value in differences)
        baseline_wins = sum(value < 0.0 for value in differences)
        baseline_queries = [row.policy_queries for row in baseline_rows]
        aligned_queries = [row.policy_queries for row in aligned_rows]
        query_differences = [
            aligned.policy_queries - baseline.policy_queries
            for _, baseline, aligned in selected
        ]
        strata.append(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "task_suite": protocol.task_suite,
                "replan_steps": protocol.replan_steps[0],
                "latency_steps": latency,
                "analysis_role": (
                    "primary"
                    if latency == FROZEN_PRIMARY_LATENCY_STEPS
                    else "prespecified_secondary"
                ),
                "injected_latency_ms": latency * 50.0,
                "paired_n": len(selected),
                "async_unguarded_n": len(baseline_rows),
                "async_unguarded_successes": baseline_successes,
                "async_unguarded_success_rate": baseline_successes
                / len(baseline_rows),
                "async_unguarded_wilson95_low": baseline_ci[0],
                "async_unguarded_wilson95_high": baseline_ci[1],
                "latency_aligned_n": len(aligned_rows),
                "latency_aligned_successes": aligned_successes,
                "latency_aligned_success_rate": aligned_successes
                / len(aligned_rows),
                "latency_aligned_wilson95_low": aligned_ci[0],
                "latency_aligned_wilson95_high": aligned_ci[1],
                "aligned_minus_async_success_rate_difference": float(
                    np.mean(differences)
                ),
                "paired_bootstrap95_low": bootstrap[0],
                "paired_bootstrap95_high": bootstrap[1],
                "aligned_wins": aligned_wins,
                "async_unguarded_wins": baseline_wins,
                "ties": sum(value == 0.0 for value in differences),
                "mcnemar_exact_p": _mcnemar_exact_p(
                    aligned_wins, baseline_wins
                ),
                "mcnemar_holm_p": None,
                "async_unguarded_mean_policy_queries": float(
                    np.mean(baseline_queries)
                ),
                "latency_aligned_mean_policy_queries": float(
                    np.mean(aligned_queries)
                ),
                "aligned_minus_async_mean_policy_queries": float(
                    np.mean(query_differences)
                ),
                "async_unguarded_runtime_failures": sum(
                    row.runtime_failure for row in baseline_rows
                ),
                "latency_aligned_runtime_failures": sum(
                    row.runtime_failure for row in aligned_rows
                ),
                "async_unguarded_failure_categories": _failure_breakdown(
                    baseline_rows, "failure_category"
                ),
                "latency_aligned_failure_categories": _failure_breakdown(
                    aligned_rows, "failure_category"
                ),
                "async_unguarded_failure_types": _failure_breakdown(
                    baseline_rows, "failure_type"
                ),
                "latency_aligned_failure_types": _failure_breakdown(
                    aligned_rows, "failure_type"
                ),
            }
        )
    if len(strata) != 3:
        raise ValueError("Holm family must contain exactly three latency strata")
    _holm_adjust(strata)
    task_descriptives: List[Dict[str, Any]] = []
    for task_id in protocol.task_ids:
        for latency in protocol.latency_steps:
            selected = [
                pair
                for pair in pairs
                if pair[0].task_id == task_id and pair[0].latency_steps == latency
            ]
            baseline_rows = [pair[1] for pair in selected]
            aligned_rows = [pair[2] for pair in selected]
            expected_n = len(protocol.episode_indices) * len(protocol.replan_steps)
            if len(selected) != expected_n:
                raise ValueError(
                    "task-latency descriptive cell has %d pairs, expected %d"
                    % (len(selected), expected_n)
                )
            baseline_successes = sum(row.success for row in baseline_rows)
            aligned_successes = sum(row.success for row in aligned_rows)
            task_descriptives.append(
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "analysis_scope": "descriptive_only_no_task_level_inference",
                    "task_suite": protocol.task_suite,
                    "task_id": task_id,
                    "replan_steps": protocol.replan_steps[0],
                    "latency_steps": latency,
                    "injected_latency_ms": latency * 50.0,
                    "paired_n": len(selected),
                    "async_unguarded_n": len(baseline_rows),
                    "async_unguarded_successes": baseline_successes,
                    "async_unguarded_success_rate": baseline_successes
                    / len(baseline_rows),
                    "latency_aligned_n": len(aligned_rows),
                    "latency_aligned_successes": aligned_successes,
                    "latency_aligned_success_rate": aligned_successes
                    / len(aligned_rows),
                    "aligned_minus_async_success_rate_difference": (
                        aligned_successes / len(aligned_rows)
                        - baseline_successes / len(baseline_rows)
                    ),
                    "async_unguarded_mean_policy_queries": float(
                        np.mean([row.policy_queries for row in baseline_rows])
                    ),
                    "latency_aligned_mean_policy_queries": float(
                        np.mean([row.policy_queries for row in aligned_rows])
                    ),
                    "aligned_minus_async_mean_policy_queries": float(
                        np.mean(
                            [
                                aligned.policy_queries - baseline.policy_queries
                                for _, baseline, aligned in selected
                            ]
                        )
                    ),
                    "async_unguarded_runtime_failures": sum(
                        row.runtime_failure for row in baseline_rows
                    ),
                    "latency_aligned_runtime_failures": sum(
                        row.runtime_failure for row in aligned_rows
                    ),
                }
            )
    expected_descriptive_rows = len(protocol.task_ids) * len(protocol.latency_steps)
    if len(task_descriptives) != expected_descriptive_rows:
        raise ValueError(
            "task-latency descriptive table has %d rows, expected %d"
            % (len(task_descriptives), expected_descriptive_rows)
        )
    return strata, task_descriptives, pair_rows, runtime_failures


def build_analysis(
    source: SourceEvidence,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes = load_episodes(source)
    pairs = pair_episodes(episodes, source.protocol)
    strata, task_descriptives, pair_rows, failures = analyze_pairs(
        pairs, source.protocol
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "frozen_read_only_latency_aligned_vs_async_unguarded_itt",
        "source": {
            "per_episode_csv": SOURCE_ARTIFACT_PATH,
            "per_episode_csv_sha256": source.csv_sha256,
            "per_episode_csv_bytes": source.csv_bytes,
            "evaluation_manifest_sha256": source.evaluation_manifest_sha256,
            "root_manifest_sha256": source.root_manifest_sha256,
            "resolved_protocol_sha256": source.protocol_sha256,
            "environment_sha256": source.environment_sha256,
            "independently_validated_files": source.independently_validated_files,
        },
        "frozen_identity": {
            "armbench_run_commit": FROZEN_ARMBENCH_RUN_COMMIT,
            "temporal_alignment_implementation_commit": (
                FROZEN_ALIGNMENT_IMPLEMENTATION_COMMIT
            ),
            "openpi_commit": FROZEN_OPENPI_COMMIT,
            "policy_config": FROZEN_POLICY_CONFIG,
            "checkpoint": FROZEN_CHECKPOINT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
            "video_mode": "all",
        },
        "contrast": {
            "comparison_mode": LATENCY_ALIGNED,
            "reference_mode": ASYNC_UNGUARDED,
            "difference_direction": "latency_aligned - async_unguarded",
        },
        "pair_key_fields": list(PAIR_KEY_FIELDS),
        "protocol": {
            "task_suite": source.protocol.task_suite,
            "task_ids": list(source.protocol.task_ids),
            "episode_indices": list(source.protocol.episode_indices),
            "modes": list(TARGET_MODES),
            "replan_steps": list(source.protocol.replan_steps),
            "latency_steps": list(source.protocol.latency_steps),
            "primary_latency_steps": FROZEN_PRIMARY_LATENCY_STEPS,
            "secondary_latency_steps": [
                latency
                for latency in source.protocol.latency_steps
                if latency != FROZEN_PRIMARY_LATENCY_STEPS
            ],
            "seed": source.protocol.seed,
            "planned_rollouts": source.protocol.planned_rollouts,
            "matched_condition_groups": source.protocol.matched_groups,
        },
        "bootstrap": {
            "method": (
                "paired percentile bootstrap of mean "
                "(latency_aligned - async_unguarded) success"
            ),
            "seed": BOOTSTRAP_SEED,
            "resamples": source.protocol.bootstrap_resamples,
            "confidence_level": 0.95,
            "inferential_role": (
                "descriptive marginal uncertainty; confirmatory decisions use "
                "the Holm-adjusted exact McNemar tests"
            ),
        },
        "multiplicity": {
            "method": "Holm step-down correction",
            "family": "three suite-level latency-stratified exact McNemar tests",
            "hypotheses": len(strata),
            "confirmatory_decision_statistic": "mcnemar_holm_p",
        },
        "itt": {
            "rollouts": len(episodes),
            "pairs": len(pairs),
            "runtime_failures_retained": len(failures),
        },
        "task_latency_descriptives": {
            "artifact": "task_latency_descriptives.csv",
            "inference": "none",
            "rows": len(task_descriptives),
            "scope": "task x latency; five matched conditions per row",
        },
        "runtime_failures": failures,
        "latency_strata": strata,
        "claim_boundary": (
            "Training-free temporal alignment under deterministic injected LIBERO "
            "delay; not a real-time, safety, dynamics, or real-robot guarantee."
        ),
    }
    return analysis, task_descriptives, pair_rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _summary(analysis: Mapping[str, Any]) -> str:
    bootstrap = analysis["bootstrap"]
    lines = [
        "# Frozen pi0.5-LIBERO latency-alignment analysis",
        "",
        "This is a read-only intention-to-treat analysis of the manifest-bound "
        "`evaluation/per_episode.csv`.",
        "",
        "- Source SHA-256: `%s`" % analysis["source"]["per_episode_csv_sha256"],
        "- Rollouts/pairs: `%d/%d`"
        % (analysis["itt"]["rollouts"], analysis["itt"]["pairs"]),
        "- Runtime failures retained in ITT: `%d`"
        % analysis["itt"]["runtime_failures_retained"],
        "- Difference direction: `latency_aligned - async_unguarded`",
        "- Primary comparison: delay `4`; delays `0` and `2` are prespecified secondary",
        "- Paired bootstrap seed/resamples: `%d/%d`"
        % (bootstrap["seed"], bootstrap["resamples"]),
        "- Confirmatory decisions: Holm-adjusted exact McNemar; bootstrap intervals "
        "are descriptive marginal uncertainty and need not agree with test decisions",
        "- Task x latency rows: descriptive only; no task-level significance tests",
        "",
        "| Delay | Role | Async success (95% Wilson) | Aligned success (95% Wilson) | "
        "Aligned - async (bootstrap 95%) | Wins/losses/ties | McNemar raw/Holm | "
        "Mean queries async/aligned/(aligned - async) | "
        "Runtime failures async/aligned |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in analysis["latency_strata"]:
        lines.append(
            "| %d | %s | %d/%d (%.3f, [%.3f, %.3f]) | "
            "%d/%d (%.3f, [%.3f, %.3f]) | %.3f ([%.3f, %.3f]) | "
            "%d/%d/%d | %.4g/%.4g | %.2f/%.2f/%.2f | %d/%d |"
            % (
                row["latency_steps"],
                row["analysis_role"],
                row["async_unguarded_successes"],
                row["async_unguarded_n"],
                row["async_unguarded_success_rate"],
                row["async_unguarded_wilson95_low"],
                row["async_unguarded_wilson95_high"],
                row["latency_aligned_successes"],
                row["latency_aligned_n"],
                row["latency_aligned_success_rate"],
                row["latency_aligned_wilson95_low"],
                row["latency_aligned_wilson95_high"],
                row["aligned_minus_async_success_rate_difference"],
                row["paired_bootstrap95_low"],
                row["paired_bootstrap95_high"],
                row["aligned_wins"],
                row["async_unguarded_wins"],
                row["ties"],
                row["mcnemar_exact_p"],
                row["mcnemar_holm_p"],
                row["async_unguarded_mean_policy_queries"],
                row["latency_aligned_mean_policy_queries"],
                row["aligned_minus_async_mean_policy_queries"],
                row["async_unguarded_runtime_failures"],
                row["latency_aligned_runtime_failures"],
            )
        )
    lines.extend(("", str(analysis["claim_boundary"]), ""))
    return "\n".join(lines)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(row[field]) for field in fields} for row in rows
        )


def write_analysis(
    source: SourceEvidence,
    analysis: Mapping[str, Any],
    task_descriptives: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    output_directory: pathlib.Path,
) -> pathlib.Path:
    output = output_directory.resolve()
    artifact_root = source.csv_path.parent.parent
    if output == artifact_root or artifact_root in output.parents:
        raise ValueError("output directory must be outside the frozen input artifact")
    if output.exists():
        raise FileExistsError("output directory already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=".%s.tmp-" % output.name, dir=str(output.parent))
    )
    try:
        _write_json(staging / "analysis.json", analysis)
        _write_csv(staging / "latency_strata.csv", analysis["latency_strata"])
        _write_csv(staging / "task_latency_descriptives.csv", task_descriptives)
        _write_csv(staging / "per_pair.csv", pair_rows)
        (staging / "summary.md").write_text(
            _summary(analysis), encoding="utf-8", newline="\n"
        )
        files = {}
        for path in sorted(staging.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "source_per_episode_csv_sha256": source.csv_sha256,
                "files": files,
            },
        )
        if output.exists():
            raise FileExistsError("output directory already exists: %s" % output)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def execute_analysis(
    per_episode_csv: pathlib.Path, output_directory: pathlib.Path
) -> pathlib.Path:
    source = validate_frozen_source(per_episode_csv)
    analysis, task_descriptives, pair_rows = build_analysis(source)
    return write_analysis(
        source, analysis, task_descriptives, pair_rows, output_directory
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "per_episode_csv",
        type=pathlib.Path,
        help="Frozen run's evaluation/per_episode.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        required=True,
        help="New analysis directory outside the frozen artifact",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = execute_analysis(args.per_episode_csv, args.output_directory)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "valid": True,
                "output_directory": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
