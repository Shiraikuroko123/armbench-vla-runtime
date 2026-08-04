#!/usr/bin/env bash
set -Eeuo pipefail

readonly ARMBENCH_ROOT="${ARMBENCH_ROOT:?set ARMBENCH_ROOT to the ArmBench checkout}"
readonly OPENPI_ROOT="${OPENPI_ROOT:?set OPENPI_ROOT to the pinned OpenPI checkout}"
readonly RESULTS_ROOT="${RESULTS_ROOT:?set RESULTS_ROOT to a persistent results directory}"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"
readonly RUN_ID="${RUN_ID:-pi05_libero_measured_age_pilot_001}"
readonly RUN_DIRECTORY="${RESULTS_ROOT}/${RUN_ID}"
readonly PLAN_PATH="${RESULTS_ROOT}/${RUN_ID}.plan.json"
readonly VALIDATION_PATH="${RESULTS_ROOT}/${RUN_ID}.root_validation.json"
readonly ARCHIVE_PATH="${RESULTS_ROOT}/${RUN_ID}.tar.gz"

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
mkdir -p "$RESULTS_ROOT"
if [[ -e "$RUN_DIRECTORY" ]] && [[ -n "$(find "$RUN_DIRECTORY" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Run directory must be absent or empty: $RUN_DIRECTORY" >&2
  exit 73
fi
if [[ -e "$ARCHIVE_PATH" || -e "${ARCHIVE_PATH}.sha256" ]]; then
  echo "Refusing to overwrite an existing archive or digest" >&2
  exit 73
fi

cd "$ARMBENCH_ROOT"

readonly -a PROTOCOL_ARGS=(
  --task-suite libero_spatial
  --task-ids all
  --episode-indices 0:2
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
readonly -a RUN_ONLY_ARGS=(
  --video-mode all
)

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
if matrix.get("rollouts") != 40:
    raise SystemExit("expected exactly 40 scored rollouts, got %r" % matrix.get("rollouts"))
pair_count = matrix.get("matched_condition_groups", matrix.get("paired_conditions"))
if pair_count != 20:
    raise SystemExit("expected exactly 20 matched pairs, got %r" % pair_count)
PY

printf -v EVALUATOR_ARGS '%q ' "${PROTOCOL_ARGS[@]}" "${RUN_ONLY_ARGS[@]}"
# The root runner rejects shell metacharacters, so the frozen values above must
# remain simple option/value tokens. Convert Bash's escaped representation back
# to the exact space-delimited form accepted by the runner.
EVALUATOR_ARGS="${EVALUATOR_ARGS% }"
if [[ "$EVALUATOR_ARGS" == *\\* ]]; then
  echo "Frozen evaluator arguments unexpectedly require shell escaping" >&2
  exit 64
fi

set +e
"$PYTHON_BIN" -m integrations.openpi.measured_age_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --no-build \
  --evaluator-args "$EVALUATOR_ARGS"
run_status=$?

"$PYTHON_BIN" -m integrations.openpi.measured_age_compose_run validate \
  "$RUN_DIRECTORY" >"$VALIDATION_PATH"
validation_status=$?
set -e

tar -C "$RESULTS_ROOT" -czf "$ARCHIVE_PATH" "$RUN_ID"
sha256sum "$ARCHIVE_PATH" >"${ARCHIVE_PATH}.sha256"

printf 'PLAN=%s\n' "$PLAN_PATH"
printf 'RUN=%s\n' "$RUN_DIRECTORY"
printf 'VALIDATION=%s\n' "$VALIDATION_PATH"
printf 'ARCHIVE=%s\n' "$ARCHIVE_PATH"
printf 'SHA256=%s\n' "${ARCHIVE_PATH}.sha256"

if ((run_status != 0)); then
  exit "$run_status"
fi
exit "$validation_status"
