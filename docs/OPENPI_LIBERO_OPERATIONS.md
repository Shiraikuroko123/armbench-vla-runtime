# OpenPI/LIBERO operations guide

## Scope

This guide covers new diagnostic and registered pi0.5-LIBERO runs on a
Linux/NVIDIA host. It describes the transactional runner, validation lifecycle,
and artifact handling. Completed study IDs and preserved evidence directories
must not be reused.

Study-specific matrices remain governed by their frozen protocols. This guide
does not redefine an existing primary endpoint or authorize pooling across
studies.

## Requirements

| Component | Requirement |
| --- | --- |
| Host | Ubuntu or compatible Linux distribution |
| GPU | NVIDIA GPU visible inside containers |
| Container runtime | Docker Engine, Compose v2, NVIDIA Container Toolkit |
| OpenPI | Clean checkout at the commit required by the target protocol |
| ArmBench | Clean checkout at the registered run commit |
| Checkpoint cache | Persistent volume with sufficient space for pi05_libero |
| Results storage | Persistent volume outside the container filesystem |

The deterministic alignment studies use OpenPI commit:

~~~text
15a9616a00943ada6c20a0f158e3adb39df2ccac
~~~

Other protocols may bind a different ArmBench or OpenPI extension commit.
Always use the identity recorded in the corresponding frozen protocol.

## Workspace layout

~~~text
/workspace/openpi/            pinned OpenPI checkout
/workspace/armbench/project/  ArmBench checkout
/workspace/openpi-cache/      checkpoint and model cache
/workspace/armbench-results/  persistent run artifacts
~~~

Set explicit absolute paths:

~~~bash
export OPENPI_ROOT=/workspace/openpi
export ARMBENCH_ROOT=/workspace/armbench/project
export ARMBENCH_RESULTS_ROOT=/workspace/armbench-results
export OPENPI_DATA_HOME=/workspace/openpi-cache

mkdir -p "$ARMBENCH_RESULTS_ROOT" "$OPENPI_DATA_HOME"
cd "$ARMBENCH_ROOT"
~~~

## Transactional runner

integrations.openpi.libero_compose_run performs the complete run lifecycle:

1. host, repository, dependency, storage, and container-GPU preflight;
2. resolved Compose generation;
3. attested policy server startup;
4. evaluator execution;
5. container shutdown;
6. nested artifact validation;
7. root finalization and manifest generation.

Do not replace this sequence with manually assembled docker compose commands.
The runner records failure state and attempts shutdown/finalization even when
evaluation does not complete.

The following diagnostic overrides weaken provenance and are not valid for a
registered study:

- --skip-container-gpu-probe
- --allow-unattested-server
- --allow-commit-mismatch
- --continue-after-runtime-failure

## Two-rollout smoke test

Use a new run ID:

~~~bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_smoke_new_001 \
  --libero-args "--task-suite libero_spatial --task-ids 0 --episode-indices 0:2 --modes async_unguarded --replan-steps 5 --latency-steps 0 --seed 7 --video-mode all"
~~~

Expected behavior:

- preflight reports ready;
- checkpoint attestation is available before the first rollout;
- two rollouts reach a terminal state;
- Compose is stopped;
- the root artifact is finalized;
- validation reports valid=true.

The smoke test validates infrastructure and checkpoint connectivity. Two
rollouts do not support a performance estimate.

## Validation and finalization

The run command finalizes automatically. Explicit commands are available for
inspection or for refinalization after adding permitted administrative records:

~~~bash
export RUN_DIRECTORY="$ARMBENCH_RESULTS_ROOT/pi05_libero_smoke_new_001"

python3 -m integrations.openpi.libero_compose_run finalize "$RUN_DIRECTORY"
python3 -m integrations.openpi.libero_compose_run validate "$RUN_DIRECTORY"
~~~

Acceptance requires:

- finalization complete=true;
- finalization errors=[];
- manifest validation valid=true;
- independent artifact validation valid=true;
- no unexpected files outside the manifest;
- matrix, episode, query, statistic, and required-video checks complete.

Finalization is content-sensitive. Any permitted change after finalization
requires another finalization pass. Raw experimental records must not be edited
to make validation succeed.

## Registered study drivers

Use the repository driver associated with the target protocol:

| Study family | Driver |
| --- | --- |
| Measured-age confirmation | scripts/run_pi05_measured_age_confirmatory.sh |
| Cross-suite external validation | scripts/run_pi05_cross_suite_external.sh |
| Measured-age exploratory pilot | scripts/run_pi05_measured_age_pilot.sh |
| Deterministic LIBERO matrix | integrations.openpi.libero_compose_run |

Before execution, inspect the driver constants, expected repository commit,
run ID, task matrix, sampling contract, and output root. A modified matrix is a
new study and requires a new protocol and run ID.

RTC corrected-v3 evidence is already complete. Its preserved result should be
inspected with scripts/rtc_primary_acceptance.cmd rather than rerun under the
same identity.

## Artifact retention

Retain the complete run directory, not only summary files or videos. Required
material includes:

- preflight and resolved configuration;
- checkpoint and source attestation;
- server and evaluator logs;
- root and nested manifests;
- per-episode and per-query records;
- analysis inputs and derived outputs;
- required videos;
- finalization and validation reports.

Before terminating a cloud instance:

1. stop all containers;
2. validate the source run directory;
3. archive and hash the complete run directory;
4. download the archive and hash record;
5. verify the downloaded hash;
6. validate the downloaded copy where supported;
7. retain billing and GPU environment metadata;
8. delete the cloud instance only after the local copy is verified.

## Cost control

Run the smallest smoke test before a registered matrix. Estimate compute from
measured pilot wall time rather than nominal rollout count:

~~~text
conservative GPU hours =
  planned rollouts * pilot P95 seconds per rollout / 3600
~~~

Include checkpoint download time, failed startup attempts, validation, archive
creation, storage, and transfer in the budget. Keep a separate reserve for one
full failed condition group or interrupted run.

## Failure policy

- Preserve incomplete artifacts and logs.
- Do not delete registered failures from an intention-to-test denominator.
- Do not resume into an existing nonempty run directory.
- Use a new run ID for every diagnostic rerun.
- Distinguish infrastructure failure from task failure.
- Freeze any protocol correction before observing replacement outcomes.

For preserved outcomes and their interpretation, see [Results](RESULTS.md).
