#!/usr/bin/env bash
set -Eeuo pipefail

readonly SUITES=(libero_object libero_goal libero_10)
readonly RUN_IDS=(
  pi05_libero_object_alignment_external_001
  pi05_libero_goal_alignment_external_001
  pi05_libero_10_alignment_external_001
)

require_directory() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    printf '%s is not a directory: %s\n' "$label" "$path" >&2
    exit 64
  fi
}

require_empty_target() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    printf '%s already exists; refusing to overwrite: %s\n' "$label" "$path" >&2
    exit 65
  fi
}

for command_name in python3 git tar sha256sum tee; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'required command is unavailable: %s\n' "$command_name" >&2
    exit 69
  fi
done

: "${OPENPI_ROOT:?set OPENPI_ROOT to the pinned OpenPI checkout}"
: "${ARMBENCH_ROOT:?set ARMBENCH_ROOT to the ArmBench checkout}"
: "${ARMBENCH_RESULTS_ROOT:?set ARMBENCH_RESULTS_ROOT to the results directory}"
: "${ARMBENCH_ARCHIVE_ROOT:?set ARMBENCH_ARCHIVE_ROOT to the archive directory}"
: "${ARMBENCH_EXPECTED_COMMIT:?set ARMBENCH_EXPECTED_COMMIT to the frozen clean commit}"

require_directory OPENPI_ROOT "$OPENPI_ROOT"
require_directory ARMBENCH_ROOT "$ARMBENCH_ROOT"
mkdir -p "$ARMBENCH_RESULTS_ROOT" "$ARMBENCH_ARCHIVE_ROOT"

readonly ACTUAL_COMMIT="$(git -C "$ARMBENCH_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$ARMBENCH_EXPECTED_COMMIT" ]]; then
  printf 'ArmBench commit mismatch: expected %s, got %s\n' \
    "$ARMBENCH_EXPECTED_COMMIT" "$ACTUAL_COMMIT" >&2
  exit 65
fi
if [[ -n "$(git -C "$ARMBENCH_ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'ArmBench worktree must be clean before a formal run\n' >&2
  git -C "$ARMBENCH_ROOT" status --short >&2
  exit 65
fi

cd "$ARMBENCH_ROOT"

for index in "${!SUITES[@]}"; do
  suite="${SUITES[$index]}"
  run_id="${RUN_IDS[$index]}"
  run_directory="$ARMBENCH_RESULTS_ROOT/$run_id"
  archive="$ARMBENCH_ARCHIVE_ROOT/$run_id.full.tar.gz"
  checksum="$archive.sha256"
  launcher_log="$ARMBENCH_ARCHIVE_ROOT/$run_id.launcher.log"
  validation="$ARMBENCH_ARCHIVE_ROOT/$run_id.validation.json"
  plan="$ARMBENCH_ARCHIVE_ROOT/$run_id.plan.json"

  require_empty_target 'run directory' "$run_directory"
  require_empty_target archive "$archive"
  require_empty_target checksum "$checksum"
  require_empty_target 'launcher log' "$launcher_log"
  require_empty_target validation "$validation"
  require_empty_target plan "$plan"

  python3 -m integrations.openpi.libero_runtime_eval plan \
    --task-suite "$suite" \
    --task-ids all \
    --episode-indices 0:5 \
    --modes async_unguarded,latency_aligned \
    --replan-steps 5 \
    --latency-steps 4 \
    >"$plan"

  python3 - "$plan" "$suite" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
suite = sys.argv[2]
plan = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "rollouts": 100,
    "matched_condition_groups": 50,
    "task_suites": [suite],
    "task_ids": list(range(10)),
    "episode_indices": list(range(5)),
    "modes": ["async_unguarded", "latency_aligned"],
    "replan_steps": [5],
    "latency_steps": [4],
}
errors = [key for key, value in expected.items() if plan.get(key) != value]
if errors:
    raise SystemExit("frozen plan mismatch: " + ", ".join(errors))
PY

  printf 'Starting %s (%s) at %s\n' "$run_id" "$suite" "$(date -Is)"
  set +e
  python3 -m integrations.openpi.libero_compose_run run \
    --openpi-root "$OPENPI_ROOT" \
    --armbench-root "$ARMBENCH_ROOT" \
    --results-root "$ARMBENCH_RESULTS_ROOT" \
    --run-id "$run_id" \
    --libero-args "--task-suite $suite --task-ids all --episode-indices 0:5 --modes async_unguarded,latency_aligned --replan-steps 5 --latency-steps 4 --seed 7 --bootstrap-resamples 10000 --video-mode all" \
    2>&1 | tee "$launcher_log"
  run_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$run_status" -ne 0 ]]; then
    printf 'Formal run failed with status %s; preserved %s\n' \
      "$run_status" "$run_directory" >&2
    exit "$run_status"
  fi

  python3 -m integrations.openpi.libero_compose_run validate "$run_directory" \
    | tee "$validation"
  python3 - "$validation" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("valid") is not True or report.get("complete") is not True:
    raise SystemExit("final validation did not report valid=true and complete=true")
PY

  temporary_archive="$archive.partial"
  require_empty_target 'partial archive' "$temporary_archive"
  tar -C "$ARMBENCH_RESULTS_ROOT" -czf "$temporary_archive" "$run_id"
  mv "$temporary_archive" "$archive"
  (
    cd "$ARMBENCH_ARCHIVE_ROOT"
    sha256sum "$(basename "$archive")" >"$(basename "$checksum")"
  )
  sync
  printf 'Archived %s at %s\n' "$run_id" "$(date -Is)"
done

printf 'All frozen cross-suite runs completed and archived at %s\n' "$(date -Is)"
