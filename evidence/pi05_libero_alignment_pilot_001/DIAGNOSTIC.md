# pi0.5 LIBERO temporal-alignment pilot

This directory preserves the complete, validated exploratory run
`pi05_libero_alignment_pilot_001`. It was used to decide whether temporal
alignment was worth a separately frozen confirmatory experiment; it is not
confirmatory evidence itself.

## Protocol

- Official OpenPI `pi05_libero` checkpoint at commit
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- All 10 `libero_spatial` tasks, initial states 47 and 48.
- Horizon 5 and four injected delay steps (200 ms at 20 Hz).
- `async_unguarded` versus training-free `latency_aligned` dispatch.
- 40 planned and completed rollouts, with every video retained.

## Exploratory observation

| Mode | Official successes | Mean policy queries |
| --- | ---: | ---: |
| `async_unguarded` | 5/20 | 24.25 |
| `latency_aligned` | 20/20 | 13.80 |

There were no recorded transport, policy-contract, timeout, environment, or
video failures. These two initial states were selected after an earlier guard
pilot, so the large difference must not be presented as an unbiased estimate.
The confirmatory protocol was frozen separately in
[`docs/PI05_ALIGNMENT_CONFIRMATORY_FREEZE.md`](../../docs/PI05_ALIGNMENT_CONFIRMATORY_FREEZE.md)
before any confirmatory rollout was started.

The embedded evaluator version predates the external aligned-mode comparator,
so its state-guard-only `paired_comparisons` tables are empty by design. The raw
`per_episode.csv` remains immutable and is the source for any later aligned
comparison.

## Validation

The copied artifact independently passes root and nested validation:

- `complete=true` and `valid=true`;
- 71 root-manifest files checked;
- 61 evaluation-manifest files checked;
- checkpoint content SHA-256
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.

From the repository root:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.libero_compose_run validate `
  'evidence\pi05_libero_alignment_pilot_001\run'
```

The claim is limited to deterministic injected observation/action delay in
LIBERO simulation. It is not a hard real-time guarantee, a real-robot result,
or a continuous-collision safety certificate.
