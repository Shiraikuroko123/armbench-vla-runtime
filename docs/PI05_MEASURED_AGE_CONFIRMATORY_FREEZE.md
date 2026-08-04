# pi0.5 LIBERO measured-age confirmatory freeze

Frozen on 2026-08-05 before launching any scored confirmatory rollout.

## Status and prior information

This is a prospective confirmatory protocol for the measured-age temporal
alignment runtime. The separately registered
`pi05_libero_measured_age_pilot_001` used episode indices `0:2` and remains
exploratory. Its outcomes informed the paired alternatives below; they are
planning assumptions, not known population probabilities or observed power.

Historical Spatial evidence scored episode indices `0:5`; additional scored
windows used `45:47` and `47:49`, while episode 49 is reserved for warm-up.
The confirmatory split therefore uses the fully unobserved episode indices
`5:17`. No scored outcome from those indices may be inspected before this
freeze. The matrix, runtime parameters, primary outcome, test, analysis
population, and stopping rules below cannot change in response to results.

## Frozen implementation and model identity

- Policy config/checkpoint: official `pi05_libero`,
  `gs://openpi-assets/checkpoints/pi05_libero`.
- OpenPI and ArmBench revisions, clean-worktree state, checkpoint content hash,
  policy-load attestation, and source manifests are recorded by the existing
  transactional runner.
- Latency source: measured wall-clock observation age. Hidden `latency_steps`
  is zero and is not used by the dispatcher.
- Candidate: `latency_aligned`; reference: `async_unguarded`.
- Neither mode retrains or fine-tunes pi0.5.
- Policy sampling noise is supplied explicitly for every request. Its versioned
  SHA-256 key is derived from seed, namespace, suite, task, episode, horizon,
  and query index; runtime mode is excluded. The request binding and realized
  noise hash are persisted per query and independently validated. Warm-up uses
  its own namespace.

The exact revisions and checkpoint digest are recorded before launch and
enforced by clean-commit and attestation checks. A revision change requires a
new protocol version and run ID.

## Frozen matrix and order

- Suite: all 10 tasks in `libero_spatial`.
- Held-out episode indices: `5:17` for every task (12 initial states).
- Modes: `async_unguarded,latency_aligned`.
- Total: 240 scored rollouts in 120 matched pairs.
- Matching fields: suite, task, episode, replan horizon, seed, and initial-state
  hash.
- Environment seed: 7.
- Videos: every scored episode, whether successful or failed.
- Run ID: `pi05_libero_measured_age_confirmatory_001`.

Both modes in a pair are adjacent. The evaluator's deterministic alternating
order is frozen: every task must contain six async-first and six aligned-first
pairs, and condition orders must be consecutive integers 0 through 239. The
complete registered plan and SHA-256 are written before the first scored
request. A Git-archive of this freeze, the launcher, exact-power artifact, power
implementation, and confirmatory analyzer is generated from the enforced commit
and included with its SHA-256 in the final evidence archive. Warm-up is not
scored and does not enter efficacy or query outcomes.

## Frozen runtime parameters

These match the measured-age pilot:

- replan horizon: 5 actions;
- controller period: 50 ms;
- action-offset rounding: `ceil`;
- deadline: 250 ms;
- maximum consecutive age refreshes: 2;
- paired response-jitter candidates: `0,40,80,160` ms;
- warm-up queries: 3;
- warm-up task ID / episode index: `0 / 49`;
- evaluator seed: 7;
- video mode: `all`.

The keyed jitter and policy-noise schedules exclude runtime mode. Matching
pair/query indices therefore receive the same requested jitter and pi0.5
diffusion noise even when modes make different total numbers of queries. The
OpenPI wrapper must reject a malformed or missing noise binding; it cannot fall
back to mutable server RNG. Realized age, jitter, noise binding, and decision
remain recorded per query.

The evaluator is still blocking. Catch-up models an independently ticking
controller after response arrival; this is not operating-system hard real time.
No deadline, rounding rule, jitter candidate, noise derivation, refresh limit,
horizon, task, episode, warm-up, seed, mode, or outcome can be tuned after
launch.

## Primary estimand and test

There is one primary hypothesis. The outcome is official LIBERO task
completion. The ITT estimand is the paired success-rate difference
`latency_aligned - async_unguarded` across all 120 pairs. It targets the fixed,
episode-weighted 10-task by 12-state Spatial grid; these are not 120 independent
tasks or a random sample of all robot tasks.

The single primary test is a two-sided exact McNemar test at `alpha=.05`.
Candidate-only means aligned success/reference failure; reference-only means
the reverse. With `d` discordant pairs, the p-value is twice the smaller
`Binomial(d,.5)` tail, capped at one. A positive primary result requires both
`p<=.05` and more candidate-only than reference-only pairs. There is no
multiplicity adjustment because there is one primary test.

All 240 planned rollouts remain in the ITT denominator. Task, deadline,
horizon, and fail-closed outcomes are never deleted. Infrastructure failures
are retained and handled by the whole-run rule below. The report includes mode
success counts and Wilson intervals, discordant/tied counts, paired risk
difference and paired bootstrap interval, exact p-value, completion accounting,
failure taxonomy, age and latency summaries, query burden, accepted/rejected
chunks, deadline/horizon events, refreshes, fallback holds, and catch-up steps.

## Frozen cluster and order sensitivities

Task dependence is handled by pre-specified secondary analyses:

- per-task mode success rates, candidate/reference wins, and risk difference;
- 10,000 task-cluster bootstrap resamples, sampling the ten tasks with
  replacement while keeping all 12 pairs in a sampled task, seed `20260805`;
- exhaustive `2^10` task-level sign-flip test of mean task risk difference;
- ten leave-one-task-out risk differences;
- descriptive effect and discordance in async-first and aligned-first strata.

These analyses cannot replace the primary pooled test. If the pooled result is
positive but cluster sensitivities disagree, the claim remains limited to the
registered grid and cannot be generalized across tasks.

## A priori exact power

The smallest effect of interest is net paired improvement `.15`. The primary
pilot-informed alternative additionally assumes candidate-only `.20` and
reference-only `.05`, hence total discordance `.25`; total discordance is a
separate assumption, not part of the SESOI. Sensitivity alternatives are
`.18/.05` and `.22/.03`. A symmetric `.10/.10` null exposes exact-test
discreteness.

Power is computed without Monte Carlo error by enumerating
`D~Binomial(n,p10+p01)` and `C|D~Binomial(D,p10/(p10+p01))`, applying the exact
planned rejection rule. At 120 pairs:

| Paired assumption | Exact rejection probability |
| --- | ---: |
| `.20/.05` primary alternative | 0.903514 |
| `.18/.05` lower sensitivity | 0.823937 |
| `.22/.03` upper sensitivity | 0.993202 |
| `.10/.10` symmetric null | 0.029613 |

Run the reproducible calculation from the repository root:

```powershell
& '..\.venv\Scripts\python.exe' scripts\power_measured_age_confirmatory.py `
  --output results\pi05_measured_age_confirmatory_power.json
```

The byte-reproducible default output is frozen at
`docs/research/pi05_measured_age_confirmatory_exact_power.json` and checked by
the test suite.

This is prospective sensitivity analysis, not post-hoc power. It assumes
independent pairs and does not absorb within-task dependence, which is why the
cluster analyses are mandatory. The 120-pair matrix is a complete held-out
`10 tasks x 12 states` grid and exceeds 80% exact power under the lower
sensitivity alternative.

## Stop, restart, and validation rules

The runner fails closed on source/checkpoint mismatch, dirty worktree,
incomplete cache, malformed noise binding, transport/response error, invalid
plan, or an existing output path. No primary outcome, cumulative success count,
McNemar statistic, or effect estimate is inspected before all 240 registered
rollouts finish. There is no efficacy early stop or p-value-driven extension.

A confirmed environment, transport, or server infrastructure failure
invalidates the whole run rather than a selected pair. The incomplete artifact
and logs are retained; a disclosed new run ID restarts warm-up and all 240
rollouts. Any server restart has the same full-restart rule. Code or parameter
changes require a new protocol version and cannot be merged into this run.

No confirmatory claim is allowed unless root and nested validators report a
complete valid 240-rollout artifact, all videos and hashes are present, exactly
120 complete pairs and the frozen order are present, policy noise/jitter
bindings validate, and source/checkpoint attestations match this freeze.

Even a positive result supports only training-free measured-age temporal
alignment for this official pi0.5-LIBERO simulation protocol. It is not a real
network latency, hard-real-time, collision-safety, cross-model, cross-suite, or
real-robot claim.

## Frozen launch command

```bash
ARMBENCH_ROOT=/workspace/armbench/project \
OPENPI_ROOT=/workspace/openpi \
RESULTS_ROOT=/workspace/armbench-results \
OPENPI_DATA_HOME=/workspace/openpi-cache \
ARMBENCH_EXPECTED_COMMIT=<frozen-clean-armbench-commit> \
PYTHON_BIN=/workspace/armbench/.venv/bin/python \
bash scripts/run_pi05_measured_age_confirmatory.sh
```

Before execution, the generated plan must report schema
`armbench.pi05_libero_measured_age.v2`, 240 rollouts, 120 pairs, and episode
indices `[5,6,7,8,9,10,11,12,13,14,15,16]`; every task must pass the registered
order checks.
