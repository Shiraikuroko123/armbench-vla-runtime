#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=/workspace/armbench/project
readonly RESULTS=/workspace/armbench-results
readonly IMAGE=armbench-libero-runtime:py311
readonly RUNTIME_COMMIT=1551900d2c66b0e8a1d46af51ee5df53e8c63bcc
readonly REPORT_COMMIT=60e027c48e5d9e4cdd1bf843cec94acff99ed1c9
readonly SECOND_CELL_CUTOFF_UTC=2026-08-10T07:00:00Z
readonly ALIGNED_ID=pi05_selection_object_s7_age_aligned_40_20260810_001
readonly RELATIVE_ID=pi05_selection_object_s7_response_relative_40_20260810_001
readonly REPORT_ID=pi05_selection_object_report_80_20260810_001

docker_args=(
  --network host
  --shm-size 8g
  -v "$RESULTS:/armbench_results"
  -v /workspace/openpi:/app
  -v "$REPO:/armbench:ro"
  -e PYTHONPATH=/armbench:/armbench/src:/app:/app/packages/openpi-client/src:/app/third_party/libero
  -e MUJOCO_GL=egl
  -e PYOPENGL_PLATFORM=egl
  -e MUJOCO_EGL_DEVICE_ID=0
  -e LIBERO_CONFIG_PATH=/tmp/libero
  -e GIT_CONFIG_COUNT=2
  -e GIT_CONFIG_KEY_0=safe.directory
  -e GIT_CONFIG_VALUE_0=/app
  -e GIT_CONFIG_KEY_1=safe.directory
  -e GIT_CONFIG_VALUE_1=/armbench
  -e "ARMBENCH_SERVER_ARGS=--policy-config pi05_libero --checkpoint gs://openpi-assets/checkpoints/pi05_libero --openpi-root /app --expected-openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac --attestation-output /armbench_results/server/checkpoint_attestation.json --host 0.0.0.0 --port 8000"
)

run_cell() {
  local container_name=$1
  local run_id=$2
  local mode=$3

  printf 'CELL_START %s %s %s\n' "$(date -Is)" "$run_id" "$mode"
  sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
  sudo docker run --name "$container_name" "${docker_args[@]}" "$IMAGE" \
    /bin/bash -lc "source /.venv/bin/activate; exec python -m integrations.openpi.libero_independent_clock_eval run \
      --output-dir /armbench_results/$run_id/evaluation \
      --host 127.0.0.1 --port 8000 \
      --openpi-root /app --armbench-root /armbench \
      --task-suite libero_object \
      --task-ids 0:10 --episode-indices 4:8 \
      --deadline-ms 175 --seed 7 \
      --action-selection-mode $mode \
      --video-mode failures"

  sudo docker run --rm "${docker_args[@]}" "$IMAGE" \
    /bin/bash -lc "source /.venv/bin/activate; python -m integrations.openpi.validate_libero_independent_clock /armbench_results/$run_id/evaluation --json"

  jq -e '
    .complete == true and
    .planned_rollouts == 40 and
    .completed_rollouts == 40 and
    .total_failed_responses == 0 and
    .episodes_with_inference_overlap == 40
  ' "$RESULTS/$run_id/evaluation/aggregate.json" >/dev/null

  (
    cd "$RESULTS"
    sudo tar -czf "$run_id.tar.gz" "$run_id"
    sha256sum "$run_id.tar.gz" | sudo tee "$run_id.tar.gz.sha256"
  )
  printf 'CELL_COMPLETE %s %s\n' "$(date -Is)" "$run_id"
}

git -C "$REPO" checkout --detach "$RUNTIME_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$RUNTIME_COMMIT"

run_cell armbench-object-selection-s7-aligned "$ALIGNED_ID" age_aligned_suffix

cutoff_epoch=$(date -u -d "$SECOND_CELL_CUTOFF_UTC" +%s)
if (( $(date -u +%s) >= cutoff_epoch )); then
  printf 'SECOND_CELL_NOT_STARTED %s cutoff=%s\n' "$(date -Is)" "$SECOND_CELL_CUTOFF_UTC"
  exit 3
fi

run_cell armbench-object-selection-s7-relative "$RELATIVE_ID" response_relative_chunk

git -C "$REPO" checkout --detach "$REPORT_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$REPORT_COMMIT"
sudo docker run --rm "${docker_args[@]}" "$IMAGE" \
  /bin/bash -lc "source /.venv/bin/activate; python scripts/build_pi05_selection_report.py \
    /armbench_results/$ALIGNED_ID \
    /armbench_results/$RELATIVE_ID \
    --profile smoke \
    --output-dir /armbench_results/$REPORT_ID"

(
  cd "$RESULTS"
  sudo tar -czf "$REPORT_ID.tar.gz" "$REPORT_ID"
  sha256sum "$REPORT_ID.tar.gz" | sudo tee "$REPORT_ID.tar.gz.sha256"
  sudo touch OBJECT_SELECTION_SEED7_COMPLETE
)
printf 'OBJECT_SELECTION_SEED7_COMPLETE %s\n' "$(date -Is)"
