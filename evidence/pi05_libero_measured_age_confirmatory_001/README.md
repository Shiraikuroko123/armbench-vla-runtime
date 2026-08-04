# pi0.5 LIBERO measured-age confirmatory evidence

This directory preserves the complete held-out run, two independently
recomputed analyses, and every scored video for
`pi05_libero_measured_age_confirmatory_001`.

## Research question

Can a frozen VLA remain effective under measured, variable response age when
the runtime receives no hidden injected-delay label? Both modes use the
attested official `pi05_libero` checkpoint, identical initial states, and the
same mode-independent response jitter and explicit pi0.5 flow-sampling noise
for every shared pair/query identity.

`async_unguarded` always dispatches from action index zero.
`latency_aligned` converts measured observation age to a conservative ceil
offset, executes only a complete five-action suffix, and holds then refreshes
when the 250 ms deadline or ten-action horizon is exceeded. Neither mode
changes or fine-tunes the VLA.

The protocol was frozen at commit
`12070625cd6f46186282317262065d015c8fbe27` before any scored held-out episode
was inspected. It contains all ten LIBERO Spatial tasks, episode indices 5-16,
two counterbalanced modes, 120 matched pairs, and 240 scored rollouts.

## Confirmatory result

The difference below is `latency_aligned - async_unguarded`.

| Measure | Async unguarded | Measured-age aligned | Paired result |
| --- | ---: | ---: | ---: |
| Success | 88/120 (73.33%) | 116/120 (96.67%) | +23.33 points, pair bootstrap 95% [+15.00, +31.67] |
| Paired outcomes | - | - | 32 aligned wins / 4 async wins / 84 ties |
| Exact McNemar | - | - | two-sided `p=1.941574737e-6` |
| Whole-task bootstrap | - | - | 95% [+10.83, +38.33] |
| Exact task sign flip | - | - | `p=0.015625`, all `2^10` assignments |
| Leave-one-task-out | - | - | +17.59 to +25.93 points |
| Mean policy queries | 22.75 | 15.14 | -7.61, bootstrap 95% [-8.78, -6.44] |
| Deadline / horizon events | 48 / 48 | 25 / 25 | descriptive |
| Hold-refresh interventions | 0 | 25 | registered bounded fallback |

The single prespecified primary test is positive. The effect also remains
positive under whole-task resampling, exact task sign flips, and every
leave-one-task-out analysis. Condition order was balanced, but effect magnitude
differs by stratum: +13.33 points when `async_unguarded` ran first and +33.33
points when `latency_aligned` ran first. This heterogeneity is disclosed rather
than pooled away.

All 240 assigned attempts, 4,547 scored policy queries, and 240 videos remain
in the artifact. There were no fail-closed policy queries or infrastructure
failures. The aligned runtime executed 25 registered hold-refresh
interventions.

## Why this supersedes the pilot for efficacy

The historical 40-rollout pilot paired response jitter but did not pair
OpenPI's mutable policy RNG between modes. It remains valid exploratory
evidence, but its success difference has a latent sampling-noise confound.
This held-out study uses an explicit mode-independent `10 x 32` float32 noise
tensor keyed by pair/query identity. The server echoes its key and realized
noise hash, and the independent validator recomputes every tensor and binding.

The confirmatory result does not retroactively change or pool with the pilot.

## Provenance and validation

- ArmBench run/protocol commit:
  `12070625cd6f46186282317262065d015c8fbe27`.
- OpenPI commit:
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Official config/checkpoint: `pi05_libero` /
  `gs://openpi-assets/checkpoints/pi05_libero`.
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.
- Evaluation-manifest SHA-256:
  `ce439762242328ceeee5177c470ee498becfe422beaeec868a9b2637eddc8d39`.
- `per_episode.csv` SHA-256:
  `4574c8946f02c07cc63e923e0a815366b5502154b8d847e8d799d6fb99c000e8`.
- `per_query.csv` SHA-256:
  `2fdd961205f0d28d8b2a30bfab764ae7d2b6caf9466bc932c2e0965cb29602af`.
- Full cloud archive SHA-256:
  `57236b693ed284f57e3e845656f9e195d81bcd8211551b62a895d5ee71e7d0a4`.
- Root validation: `valid=true`, `complete=true`, 271 files checked.
- Nested evaluation validation: `valid=true`, 261 files checked, no warning or
  error.
- GitHub Release:
  [`evidence-pi05-libero-measured-age-confirmatory-001`](https://github.com/Shiraikuroko123/armbench-vla-runtime/releases/tag/evidence-pi05-libero-measured-age-confirmatory-001).
- Offline paired-video dashboard:
  [`reports/pi05_libero_measured_age_confirmatory_001/index.html`](../../reports/pi05_libero_measured_age_confirmatory_001/index.html).

From the repository root, the shortest complete acceptance is:

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
```

It validates the root container, independently recomputes the confirmatory
analysis from protected CSV files, checks analysis/source hash binding,
rebuilds the dashboard, verifies all 240 manifest-bound video files, and opens
the page. For a noninteractive check, append `-NoOpen`.

The individual reproduction commands are:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.measured_age_compose_run validate `
  'evidence\pi05_libero_measured_age_confirmatory_001\run'

& '..\.venv\Scripts\python.exe' -m integrations.openpi.measured_age_confirmatory_analysis validate `
  'evidence\pi05_libero_measured_age_confirmatory_001\confirmatory_analysis' `
  --json

& '..\.venv\Scripts\python.exe' -m integrations.openpi.measured_age_confirmatory_acceptance `
  --no-open
```

## Claim boundary

This is simulation-only evidence for training-free measured-age suffix
selection with a frozen pi0.5-LIBERO checkpoint. Inference remains blocking and
controller catch-up is simulated after response arrival. It does not establish
true concurrent inference/controller execution, hard real-time guarantees,
physical collision or dynamics safety, another VLA family, real-robot validity,
or RTC-style policy-internal continuation.
