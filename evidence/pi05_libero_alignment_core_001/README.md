# pi0.5 LIBERO temporal-alignment confirmatory evidence

This directory preserves the complete validated run and a read-only paired
analysis for `pi05_libero_alignment_core_001`.

## Research question

An asynchronous VLA returns an action chunk conditioned on an observation that
is already stale when execution begins. The baseline starts at action index
zero after the registered delay. `latency_aligned` is training-free: after `d`
delay steps, it discards actions `[0:d]` and executes the following five-action
suffix. A short policy chunk fails closed.

The confirmatory protocol was frozen before these initial states were used:
all 10 LIBERO Spatial tasks, five initial states per task, two dispatch modes,
and delays of 0, 2, and 4 control steps at 20 Hz. This produced 300 rollouts in
150 matched task/state/delay groups.

## Confirmatory result

All failures remain in the intention-to-treat denominator. The difference is
`latency_aligned - async_unguarded`.

| Injected delay | Role | Async success | Aligned success | Paired difference (bootstrap 95%) | Aligned wins / async wins / ties | Exact McNemar raw / Holm | Mean policy queries async / aligned |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | secondary | 49/50 | 50/50 | +2 points [0, +6] | 1 / 0 / 49 | 1.000 / 1.000 | 22.14 / 21.50 |
| 100 ms | secondary | 41/50 | 48/50 | +14 points [+2, +26] | 9 / 2 / 39 | 0.0654 / 0.1309 | 21.80 / 16.18 |
| 200 ms | **primary** | **18/50** | **50/50** | **+64 points [+50, +76]** | **32 / 0 / 18** | **4.66e-10 / 1.40e-9** | **22.96 / 12.82** |

The frozen primary comparison supports a large benefit under deterministic
200 ms injected delay. The 100 ms secondary comparison is directionally
positive but is not significant after Holm correction. Zero delay is a ceiling
condition. Bootstrap intervals describe marginal uncertainty; the
Holm-adjusted exact McNemar tests are the prespecified confirmatory decisions.

No transport, timeout, policy-contract, environment, or video failure occurred.
The Compose evaluation took 2,087.395 seconds. All 300 requested videos are
present.

## Provenance and validation

- ArmBench run commit: `30676d2d3ff43e3df0750e2ad01f94748293cff5`.
- Alignment implementation: `cccbe351a1a4523c65d01eff2997580f7ca83649`.
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Official config/checkpoint: `pi05_libero` /
  `gs://openpi-assets/checkpoints/pi05_libero`.
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.
- Source `per_episode.csv` SHA-256:
  `d9c14a651dcfdeb90eb50b5997793808d67aa23f7d8672a5a4fe6155559ead23`.
- Full archive SHA-256:
  `045da367f718aeabfb829166f0e98c005a8cbaebff9a6a2f270c1dd50aff4a29`.
- Root validation: `valid=true`, `complete=true`, 331 protected files checked.
- Nested evaluation validation: `valid=true`, with no warnings or errors.

The full run is in [`run`](run), and the derived report is in
[`analysis`](analysis). The analysis reruns the complete root validator, checks
the exact frozen provenance, snapshots the manifest-bound CSV bytes, enforces
all 150 pairs, and writes its output atomically.

From the repository root on this workspace:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.libero_compose_run validate `
  'evidence\pi05_libero_alignment_core_001\run'

& '..\.venv\Scripts\python.exe' -m integrations.openpi.latency_aligned_analysis `
  'evidence\pi05_libero_alignment_core_001\run\evaluation\per_episode.csv' `
  --output-directory 'results\pi05_libero_alignment_core_001-analysis-reproduction'
```

## Deterministic video audit

The representative pair is the smallest task ID and initial-state index among
the primary-delay aligned wins: task 0, initial state 1.

- [Async baseline failure](run/evaluation/videos/libero_spatial__task_000__episode_001__h_05__l_004__async_unguarded__failure.mp4)
- [Latency-aligned success](run/evaluation/videos/libero_spatial__task_000__episode_001__h_05__l_004__latency_aligned__success.mp4)

## Claim boundary

This is evidence for training-free temporal action-chunk alignment under
deterministic injected LIBERO delay. It is not a hard real-time guarantee,
collision certificate, dynamics-aware repair result, measured network-jitter
study, real-robot experiment, or claim that the method retrained pi0.5.
