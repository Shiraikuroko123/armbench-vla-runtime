# pi0.5 LIBERO temporal-alignment confirmatory freeze

Frozen on 2026-08-04 before starting `pi05_libero_alignment_core_001`.

## Prior information already observed

This protocol is not the original state-guard preregistration. It is a new
confirmatory protocol frozen after two explicitly exploratory studies:

- `pi05_libero_pilot_002`, initial states 45 and 46: `async_unguarded` 7/20,
  fixed `state_guard` 6/20. The guard produced 282 position-triggered
  rejections and 11 `max_requeries_exceeded` terminations.
- `pi05_libero_alignment_pilot_001`, initial states 47 and 48:
  `async_unguarded` 5/20 and `latency_aligned` 20/20 at delay four.

The infrastructure smoke used task 0, initial states 0 and 1, at zero injected
delay with `async_unguarded`. Those two nominal outcomes were observed before
this freeze. No `latency_aligned` rollout has used any confirmatory initial
state. The method and the matrix below must not change in response to
confirmatory outcomes.

## Frozen implementation

- ArmBench run commit: `30676d2d3ff43e3df0750e2ad01f94748293cff5`.
- Temporal-alignment implementation commit:
  `cccbe351a1a4523c65d01eff2997580f7ca83649`.
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Policy config/checkpoint: `pi05_libero`,
  `gs://openpi-assets/checkpoints/pi05_libero`.
- Attested checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.

`latency_aligned` advances the environment for the registered injected delay
using the last dispatched action, then discards exactly the same number of
leading actions from the newly returned chunk. It dispatches the following
`replan_steps` actions. It does not retrain pi0.5, inspect task success, use a
dynamics model, or alter model inputs. The run fails before rollout unless
`latency_steps + replan_steps <= 10`.

## Frozen matrix

- Suite: all ten `libero_spatial` tasks.
- Initial states: `0:5` for every task.
- Modes: `async_unguarded,latency_aligned`.
- Replan horizon: 5.
- Injected delay steps: `0,2,4`, corresponding to 0, 100, and 200 ms at 20 Hz.
- Environment seed: 7.
- Videos: all successful and failed episodes.
- Total: 300 rollouts in 150 matched task/initial-state/horizon/delay groups.
- Run ID: `pi05_libero_alignment_core_001`.

Modes within each group remain adjacent and their order alternates. Matching
means task, initial state, seed, horizon, delay, and initial-state hash match.
It does not mean the policy-server random stream is reset between modes.

## Outcomes and analysis

The primary outcome is official LIBERO task completion. The primary estimand is
the intention-to-treat success-rate difference
`latency_aligned - async_unguarded` at horizon five and delay four over all 50
matched conditions.

The primary report must include:

- both success counts and Wilson 95% intervals;
- the paired success-rate difference and paired bootstrap 95% interval;
- aligned wins, unguarded wins, ties, and the two-sided exact McNemar p-value;
- planned/completed rollouts and every non-efficacy runtime failure;
- mean policy queries and their paired difference.

Delay zero and delay two are prespecified secondary conditions. Their same
statistics must be reported, including negative results. A Holm correction is
applied across the three suite-level McNemar tests. Task-level rows are
descriptive because each task contributes only five conditions per delay.

No episode, task, or initial state may be removed because of task failure.
Infrastructure or protocol failures remain in the ITT denominator and are
reported separately from policy efficacy. Per-protocol sensitivity excludes
only the failure categories already frozen in the evaluator.

## Stop and claim rules

The transactional runner aborts the formal run on a policy transport, timeout,
observation contract, or environment infrastructure failure. The incomplete
run and all logs are retained under the original ID; any restart uses a new ID
and is reported. Action chunks shorter than `delay + replan` fail closed.

No performance claim is allowed unless root and nested validation both report
`valid=true`, the run is complete, all 300 videos required by the protocol are
present, and checkpoint/source attestation matches this freeze.

Even a positive result supports only training-free temporal alignment under
deterministic injected LIBERO delay. It is not a real-time guarantee, collision
certificate, dynamics-aware repair result, real-robot result, or proof for
measured network jitter.

## Frozen cloud command

```bash
env PYTHONPATH=/workspace/armbench/project \
    OPENPI_DATA_HOME=/workspace/openpi-cache \
  /workspace/armbench/.venv/bin/python \
  -m integrations.openpi.libero_compose_run run \
  --openpi-root /workspace/openpi \
  --armbench-root /workspace/armbench/project \
  --results-root /workspace/armbench-results \
  --run-id pi05_libero_alignment_core_001 \
  --no-build \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 0:5 --modes async_unguarded,latency_aligned --replan-steps 5 --latency-steps 0,2,4 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

After completion, finalize and validate the run, archive the complete run and
deployment logs with SHA-256, download them off the temporary server, and
publish the validated tables plus the complete archive through the GitHub
repository/release.
