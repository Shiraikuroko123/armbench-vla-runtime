#!/usr/bin/env bash
set -Eeuo pipefail

readonly ARMBENCH_ROOT="${ARMBENCH_ROOT:?set ARMBENCH_ROOT to the ArmBench checkout}"
readonly OPENPI_ROOT="${OPENPI_ROOT:?set OPENPI_ROOT to the pinned OpenPI checkout}"
readonly RESULTS_ROOT="${RESULTS_ROOT:?set RESULTS_ROOT to a persistent results directory}"
readonly OPENPI_DATA_HOME="${OPENPI_DATA_HOME:?set OPENPI_DATA_HOME to the populated persistent OpenPI asset cache}"
readonly ARMBENCH_EXPECTED_COMMIT="${ARMBENCH_EXPECTED_COMMIT:?set ARMBENCH_EXPECTED_COMMIT to the frozen clean commit}"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"
readonly RUN_ID="${RUN_ID:-pi05_libero_measured_age_confirmatory_001}"
readonly RUN_DIRECTORY="${RESULTS_ROOT}/${RUN_ID}"
readonly PLAN_PATH="${RESULTS_ROOT}/${RUN_ID}.plan.json"
readonly PLAN_SHA256_PATH="${PLAN_PATH}.sha256"
readonly PROTOCOL_SOURCES_PATH="${RESULTS_ROOT}/${RUN_ID}.protocol_sources.tar.gz"
readonly PROTOCOL_SOURCES_SHA256_PATH="${PROTOCOL_SOURCES_PATH}.sha256"
readonly PROTOCOL_COMMIT_PATH="${RESULTS_ROOT}/${RUN_ID}.protocol_commit.txt"
readonly VALIDATION_PATH="${RESULTS_ROOT}/${RUN_ID}.root_validation.json"
readonly ARCHIVE_PATH="${RESULTS_ROOT}/${RUN_ID}.tar.gz"
readonly PARTIAL_ARCHIVE_PATH="${ARCHIVE_PATH}.partial"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Invalid RUN_ID: $RUN_ID" >&2
  exit 64
fi
if [[ ! -d "$ARMBENCH_ROOT/.git" ]]; then
  echo "ARMBENCH_ROOT is not a Git checkout: $ARMBENCH_ROOT" >&2
  exit 66
fi
if [[ ! -d "$OPENPI_ROOT/.git" ]]; then
  echo "OPENPI_ROOT is not a Git checkout: $OPENPI_ROOT" >&2
  exit 66
fi
if [[ ! -f "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params/_METADATA" ]] || \
   [[ ! -f "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params/manifest.ocdbt" ]] || \
   [[ ! -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets" ]]; then
  echo "OPENPI_DATA_HOME does not contain a complete pi05_libero cache: $OPENPI_DATA_HOME" >&2
  exit 66
fi
export OPENPI_DATA_HOME

readonly ACTUAL_COMMIT="$(git -C "$ARMBENCH_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$ARMBENCH_EXPECTED_COMMIT" ]]; then
  echo "ArmBench commit mismatch: expected $ARMBENCH_EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 65
fi
if [[ -n "$(git -C "$ARMBENCH_ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "ArmBench worktree must be clean before the confirmatory run" >&2
  git -C "$ARMBENCH_ROOT" status --short >&2
  exit 65
fi

mkdir -p "$RESULTS_ROOT"
for target in \
  "$RUN_DIRECTORY" \
  "$PLAN_PATH" \
  "$PLAN_SHA256_PATH" \
  "$PROTOCOL_SOURCES_PATH" \
  "$PROTOCOL_SOURCES_SHA256_PATH" \
  "$PROTOCOL_COMMIT_PATH" \
  "$VALIDATION_PATH" \
  "$ARCHIVE_PATH" \
  "$PARTIAL_ARCHIVE_PATH" \
  "${ARCHIVE_PATH}.sha256"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing confirmatory output: $target" >&2
    exit 73
  fi
done

cd "$ARMBENCH_ROOT"

git archive \
  --format=tar.gz \
  --output="$PROTOCOL_SOURCES_PATH" \
  "$ACTUAL_COMMIT" -- \
  docs/PI05_MEASURED_AGE_CONFIRMATORY_FREEZE.md \
  docs/research/pi05_measured_age_confirmatory_exact_power.json \
  integrations/openpi/measured_age_confirmatory_analysis.py \
  scripts/power_measured_age_confirmatory.py \
  scripts/run_pi05_measured_age_confirmatory.sh
printf '%s\n' "$ACTUAL_COMMIT" >"$PROTOCOL_COMMIT_PATH"
(
  cd "$RESULTS_ROOT"
  sha256sum "$(basename "$PROTOCOL_SOURCES_PATH")" \
    >"$(basename "$PROTOCOL_SOURCES_SHA256_PATH")"
)

readonly -a PROTOCOL_ARGS=(
  --task-suite libero_spatial
  --task-ids all
  --episode-indices 5:17
  --modes async_unguarded,latency_aligned
  --replan-steps 5
  --control-period-ms 50
  --age-rounding ceil
  --deadline-ms 250
  --max-age-refreshes 2
  --jitter-values-ms 0,40,80,160
  --warmup-queries 3
  --warmup-task-id 0
  --warmup-episode-index 49
  --seed 7
)
readonly -a RUN_ONLY_ARGS=(--video-mode all)

"$PYTHON_BIN" -m integrations.openpi.measured_age_libero_eval plan \
  "${PROTOCOL_ARGS[@]}" >"$PLAN_PATH"

"$PYTHON_BIN" - "$PLAN_PATH" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if plan.get("schema_version") != "armbench.pi05_libero_measured_age.v2":
    raise SystemExit("measured-age plan schema mismatch")
matrix = plan.get("matrix", {})
if matrix.get("rollouts") != 240:
    raise SystemExit("expected exactly 240 scored rollouts, got %r" % matrix.get("rollouts"))
pair_count = matrix.get("matched_condition_groups", matrix.get("paired_conditions"))
if pair_count != 120:
    raise SystemExit("expected exactly 120 matched pairs, got %r" % pair_count)
if matrix.get("episode_indices") != list(range(5, 17)):
    raise SystemExit("confirmatory episode split mismatch")
cells = plan.get("registered_cells")
if not isinstance(cells, list) or len(cells) != 240:
    raise SystemExit("expected exactly 240 registered cells")
if [cell.get("condition_order") for cell in cells] != list(range(240)):
    raise SystemExit("registered condition order is not consecutive")
for task_id in range(10):
    task_cells = [cell for cell in cells if cell.get("task_id") == task_id]
    if len(task_cells) != 24:
        raise SystemExit("task %d does not contain 24 cells" % task_id)
    task_pairs = [task_cells[index:index + 2] for index in range(0, 24, 2)]
    if any(
        pair[0].get("pair_id") != pair[1].get("pair_id")
        or {cell.get("mode") for cell in pair}
        != {"async_unguarded", "latency_aligned"}
        for pair in task_pairs
    ):
        raise SystemExit("task %d does not contain adjacent complete pairs" % task_id)
    first_modes = [pair[0].get("mode") for pair in task_pairs]
    if first_modes.count("async_unguarded") != 6:
        raise SystemExit("task %d async-first count mismatch" % task_id)
    if first_modes.count("latency_aligned") != 6:
        raise SystemExit("task %d aligned-first count mismatch" % task_id)
PY

(
  cd "$RESULTS_ROOT"
  sha256sum "$(basename "$PLAN_PATH")" >"$(basename "$PLAN_SHA256_PATH")"
)

for token in "${PROTOCOL_ARGS[@]}" "${RUN_ONLY_ARGS[@]}"; do
  if [[ ! "$token" =~ ^[A-Za-z0-9_./,:+-]+$ ]]; then
    echo "Frozen evaluator argument contains unsupported characters: $token" >&2
    exit 64
  fi
done
readonly EVALUATOR_ARGS="${PROTOCOL_ARGS[*]} ${RUN_ONLY_ARGS[*]}"

set +e
"$PYTHON_BIN" -m integrations.openpi.measured_age_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --openpi-data-home "$OPENPI_DATA_HOME" \
  --run-id "$RUN_ID" \
  --no-build \
  --evaluator-args "$EVALUATOR_ARGS"
run_status=$?

"$PYTHON_BIN" -m integrations.openpi.measured_age_compose_run validate \
  "$RUN_DIRECTORY" >"$VALIDATION_PATH"
validation_status=$?
set -e

tar -C "$RESULTS_ROOT" -czf "$PARTIAL_ARCHIVE_PATH" \
  "$RUN_ID" \
  "$(basename "$PLAN_PATH")" \
  "$(basename "$PLAN_SHA256_PATH")" \
  "$(basename "$VALIDATION_PATH")" \
  "$(basename "$PROTOCOL_SOURCES_PATH")" \
  "$(basename "$PROTOCOL_SOURCES_SHA256_PATH")" \
  "$(basename "$PROTOCOL_COMMIT_PATH")"
mv "$PARTIAL_ARCHIVE_PATH" "$ARCHIVE_PATH"
(
  cd "$RESULTS_ROOT"
  sha256sum "$(basename "$ARCHIVE_PATH")" >"$(basename "${ARCHIVE_PATH}.sha256")"
)

printf 'PLAN=%s\n' "$PLAN_PATH"
printf 'PLAN_SHA256=%s\n' "$PLAN_SHA256_PATH"
printf 'PROTOCOL_SOURCES=%s\n' "$PROTOCOL_SOURCES_PATH"
printf 'PROTOCOL_SOURCES_SHA256=%s\n' "$PROTOCOL_SOURCES_SHA256_PATH"
printf 'PROTOCOL_COMMIT=%s\n' "$PROTOCOL_COMMIT_PATH"
printf 'RUN=%s\n' "$RUN_DIRECTORY"
printf 'VALIDATION=%s\n' "$VALIDATION_PATH"
printf 'ARCHIVE=%s\n' "$ARCHIVE_PATH"
printf 'SHA256=%s\n' "${ARCHIVE_PATH}.sha256"

if ((run_status != 0)); then
  exit "$run_status"
fi
exit "$validation_status"
