# Windowed Panda CPU Repeatability Audit

Status: completed follow-up to the atomic certification-window audit. The
formal artifact is
[`reports/pi05_windowed_cpu_repeatability_180_002`](../reports/pi05_windowed_cpu_repeatability_180_002/summary.md).

## Question

The paired window audit showed that a one-action (`H=1`) publication window can
finish within the registered software budget on the reference host, while the
original ten-action (`H=10`) chunk cannot. This study asks whether that
publication behavior survives a fresh Python process and bounded host
contention. It does not rerun the `pi0.5` checkpoint or create a task-success
measurement.

## Fixed matrix

The protocol is frozen in
[`pi05_windowed_cpu_repeatability_protocol_20260811.json`](research/pi05_windowed_cpu_repeatability_protocol_20260811.json).
It fixes:

- three idle and three four-worker CPU-load fresh-process trials;
- the same 30 frozen `pi0.5` response rows crossed with three Panda scenes;
- paired `H=10` full-chunk and `H=1` atomic certification-window profiles;
- a 20 ms supervision budget, 200 ms response deadline, and 5 ms QP step
  budget; and
- complete child logs, nested manifests, SHA-256 bindings, and an independent
  root validator.

All six trials in `180_002` validated. The checkpoint was not rerun.

## Result

| Condition | Profile | Execute counts | P95 supervisor, mean +/- stdev | Safe candidates | Unsafe windows | Partial windows |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Idle | `full_chunk_h10` | `[0, 0, 0]` | 27.175 +/- 1.347 ms | 66/90 | 0 | 0 |
| Idle | `certified_window_h1` | `[90, 90, 90]` | 6.585 +/- 0.980 ms | 90/90 | 0 | 0 |
| Four CPU workers | `full_chunk_h10` | `[0, 0, 0]` | 31.355 +/- 2.096 ms | 66/90 | 0 | 0 |
| Four CPU workers | `certified_window_h1` | `[89, 84, 86]` | 19.980 +/- 5.300 ms | 90/90 | 0 | 0 |

The result supports a narrow engineering conclusion: reducing the certified
publication horizon to one action preserves the all-or-none window boundary
and is substantially more repeatable than full-chunk certification on this
host. It does **not** establish hard real time, physical safety, or deployment
performance. `H=1` changes the source-chunk contract: the ten-action response
is exposed progressively as a sequence of independently certified windows.

## Reproduce and validate

From the repository root on Windows PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-windowed-repeatability-validate `
  reports\pi05_windowed_cpu_repeatability_180_002 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

The validator recomputes the root inventory, protocol and source bindings,
all six child replay manifests, candidate kinematics, continuous collision and
braking predicates, publication lengths, and condition-level statistics. The
validated root inventory SHA-256 is
`7cc99c4420b706a30fcb3beee774b937d0a3eb619fa17f704f59419c89323306`.

To create a new matrix rather than overwriting the checked-in artifact:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-windowed-repeatability `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_windowed_cpu_repeatability
```

## Claim boundary

The three repeats per condition are descriptive engineering evidence, not an
inferential timing study. The CPU-load condition is a bounded local stress,
not hardware qualification. Frozen responses are cross-controller Panda
inputs; no Panda observation is returned to the policy. The report does not
measure task success, hard-real-time scheduling, physical-robot safety, or
cross-machine generalization.
