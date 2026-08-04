# pi0.5 Cross-Suite Temporal-Alignment External Validation Freeze

Frozen on 2026-08-04 before any `latency_aligned` rollout on the registered
LIBERO Object, Goal, or LIBERO-10 task/initial-state cells.

## Motivation and claim

The completed confirmatory study used all ten LIBERO Spatial tasks. This
external validation asks whether its 200 ms temporal-alignment effect persists
without tuning on three different LIBERO task distributions. It does not test
measured network jitter, a real-time guarantee, collision safety, dynamics
shift, a different VLA checkpoint, or a real robot.

The method remains training-free: after four injected 20 Hz delay steps, the
dispatcher drops the corresponding four-action prefix from the returned
ten-action chunk and executes the following five actions. No threshold,
checkpoint, prompt, action horizon, or task-specific parameter may be changed
after observing an external-validation outcome.

## Frozen identity

- Policy config: `pi05_libero`.
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`.
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Temporal-alignment implementation commit:
  `cccbe351a1a4523c65d01eff2997580f7ca83649`.
- Formal run source: the clean ArmBench commit containing this freeze; the
  exact commit and source hashes must be attested inside every run artifact.

## Frozen matrix

- Suites, in execution order: `libero_object`, `libero_goal`, `libero_10`.
- Tasks: all ten official tasks in each suite.
- Initial states: official indices `0,1,2,3,4` for every task.
- Modes: `async_unguarded,latency_aligned`.
- Replan horizon: five actions.
- Injected delay: four 20 Hz control steps, reported as 200 ms.
- Seed: `7`.
- Video mode: `all`.
- Mode order: the evaluator's existing adjacent-pair alternating order.
- Planned size: 50 matched pairs and 100 rollouts per suite; 150 matched pairs
  and 300 rollouts in the external-validation family.

Run IDs are fixed as:

- `pi05_libero_object_alignment_external_001`
- `pi05_libero_goal_alignment_external_001`
- `pi05_libero_10_alignment_external_001`

## Confirmatory analysis

The unit is a matched task/initial-state pair. Infrastructure and protocol
failures remain in the intention-to-treat denominator as unsuccessful
rollouts. Each suite reports:

- baseline and aligned success counts/rates with Wilson 95% intervals;
- `latency_aligned - async_unguarded` paired success-rate difference;
- deterministic 10,000-resample paired bootstrap 95% interval, descriptive;
- aligned wins, baseline wins, ties, and two-sided exact McNemar p-value;
- mean policy queries and runtime/infrastructure failure breakdowns.

The three suite-level exact McNemar tests form one confirmatory family and use
Holm step-down correction. The corrected suite-level tests are the
confirmatory decisions. Task-level rows and a pooled 150-pair summary are
descriptive only; the pooled summary must not be presented as a new population
level significance test. Negative and null suite results remain in every
artifact and report.

## Execution commands

Each suite uses the same command template, changing only `SUITE` and `RUN_ID`:

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --libero-args "--task-suite $SUITE --task-ids all --episode-indices 0:5 --modes async_unguarded,latency_aligned --replan-steps 5 --latency-steps 4 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

## Acceptance and stopping rules

Every run must independently finish 100/100 planned rollouts, stop its Compose
services, finalize both manifests, pass the strict artifact validator, retain
all 100 videos, and report zero non-performance runtime failures. A failure
does not authorize deletion, row repair, reuse of a run ID, or substitution of
another initial state. A replacement run, if operationally necessary, uses a
new ID and both artifacts are retained.

The three suites run serially on one policy server/GPU allocation. After each
suite, its finalized directory is archived and copied off the expiring cloud
instance before the next optional extension begins. If time is insufficient,
completed suites remain reported in the frozen order and the missing suite is
explicitly marked not run; no completed suite is selected based on its result.
