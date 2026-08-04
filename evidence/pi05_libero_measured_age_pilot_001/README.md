# pi0.5 LIBERO measured-age pilot evidence

This directory preserves the complete validated run and strict paired analysis
for `pi05_libero_measured_age_pilot_001`. It is an exploratory pilot, not a
confirmatory efficacy study.

## Research question

Can a frozen VLA handle measured, variable response age without receiving the
experimenter's hidden delay label? Both modes use the official `pi05_libero`
checkpoint and experience controller catch-up plus the same SHA-256-keyed
response-jitter schedule. `async_unguarded` dispatches from action index zero.
`latency_aligned` conservatively converts measured observation age to a ceil
offset, executes only a complete five-action suffix, and holds then refreshes
when the 250 ms deadline or ten-action horizon is exceeded.

The registered matrix contains all 10 LIBERO Spatial tasks, initial states 0
and 1, and both modes: 40 scored rollouts in 20 matched pairs. Three attested
queries on task 0, state 49 were completed before scoring. Their ages were
18,424.229 ms, 84.145 ms, and 79.465 ms; the cold-start query is therefore not
assigned to either experimental mode.

## Pilot result

All 40 assigned attempts and all 810 scored policy queries remain in the
artifact. The difference below is `latency_aligned - async_unguarded`.

| Measure | Async unguarded | Measured-age aligned | Paired result |
| --- | ---: | ---: | ---: |
| Success | 14/20 (70%) | 19/20 (95%) | +25 points, bootstrap 95% [+10, +45] |
| Paired outcomes | - | - | 5 aligned wins / 0 async wins / 15 ties |
| Exact McNemar | - | - | two-sided `p=0.0625` |
| Mean policy queries | 24.6 | 15.9 | -8.7, bootstrap 95% [-10.95, -6.50] |
| Observation-age P95 / max | 244.740 / 250.166 ms | 244.587 / 251.428 ms | descriptive |
| Deadline / horizon events | 1 / 1 | 1 / 1 | descriptive |
| Hold-refresh events | 0 | 1 | bounded fallback executed as registered |

The five net wins and query reduction justify a larger, separately frozen
confirmatory matrix. They do not establish a statistically significant success
claim at the conventional 0.05 level: with only five discordant pairs, the
exact two-sided result is `p=0.0625`. No parameter or condition was changed
after the run began.

The Compose evaluation completed in 413.917 seconds on an RTX 4090. All 40
videos are present. There were no infrastructure, transport, policy-contract,
or video failures. One aligned query crossed the registered deadline/horizon,
was rejected, executed one fallback hold, and refreshed successfully.

## Provenance and validation

- ArmBench run commit:
  `b1835dabf2b76714bda01eeae43516f99ddc0505`.
- Analysis commit:
  `5467e7a7a631d2ef7aef7c5ff264e0ea1b56af88`.
- OpenPI commit:
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Official config/checkpoint: `pi05_libero` /
  `gs://openpi-assets/checkpoints/pi05_libero`.
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.
- Source evaluation-manifest SHA-256:
  `fb233967926e78ba26231556f70e1835909273e58813b0761e63a2f7f70bf02a`.
- Source `per_episode.csv` SHA-256:
  `a40b434e5e6870bb4d57f0a71579a95be62a2270c91705c887316426b02d9478`.
- Source `per_query.csv` SHA-256:
  `4080763fdda776c6ed78ad164610821facf0dc541ced25bfd693e2ff45f8b31f`.
- Analysis-source SHA-256:
  `2a966c28f2fe1fcf0e6a78c76b516fce146a9c97468228a9297a37a54c38aa74`.
- Full cloud archive SHA-256:
  `77636e8c46282bd9fa75b2636b8a5ce649c0f3a4cf235c86b4f240e080b9aa33`.
- GitHub Release:
  [`evidence-pi05-libero-measured-age-pilot-001`](https://github.com/Shiraikuroko123/armbench-vla-runtime/releases/tag/evidence-pi05-libero-measured-age-pilot-001).
- Offline paired-video acceptance dashboard:
  [`reports/pi05_libero_measured_age_pilot_001/index.html`](../../reports/pi05_libero_measured_age_pilot_001/index.html).
- Root validation: `valid=true`, `complete=true`, 70 files checked.
- Nested evaluation validation: `valid=true`, no warnings or errors.

The complete run is in [`run`](run), and the derived output is in
[`analysis`](analysis). The analyzer validates the source twice, snapshots the
manifest-bound bytes, reconstructs every pair, and records the exact analyzer
and validator source hashes before writing a separate manifest.

From the repository root:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.measured_age_compose_run validate `
  'evidence\pi05_libero_measured_age_pilot_001\run'

& '..\.venv\Scripts\python.exe' -m integrations.openpi.validate_measured_age_artifact `
  'evidence\pi05_libero_measured_age_pilot_001\run\evaluation'

& '..\.venv\Scripts\python.exe' -m integrations.openpi.measured_age_analysis `
  'evidence\pi05_libero_measured_age_pilot_001\run\evaluation' `
  --output-directory 'results\pi05_libero_measured_age_pilot_001-analysis-reproduction' `
  --bootstrap-resamples 10000 --bootstrap-seed 20260805 --json
```

## Claim boundary

This is simulation-only evidence for training-free action-suffix selection
under measured client-visible age and registered response-delivery jitter.
Inference is still blocking, controller catch-up is simulated after response
arrival, and the study covers one checkpoint family and 20 pairs. It is not a
hard real-time guarantee, physical safety certificate, real-robot experiment,
trained-policy contribution, or top-venue-scale confirmatory result.
