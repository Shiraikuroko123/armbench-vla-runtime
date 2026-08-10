# Optimized CPU Repeatability Audit

Status: completed local CPU follow-up to the v0.2.0 optimized assurance
audit. The preserved artifact is
[`reports/pi05_optimized_cpu_repeatability_20260811_001`](../reports/pi05_optimized_cpu_repeatability_20260811_001/summary.md).

## Question

The frozen v0.2.0 audit recorded one complete publication in 90 cases under
the 20 ms operational profile. Its P50 and P95 were already above the budget,
so the next question was whether that boundary result repeats across cold
processes and under bounded host contention.

This study does not change the v0.2.0 implementation or retroactively rewrite
its frozen decision. It tests how stable that decision is.

## Fixed matrix

The protocol is stored at
[`pi05_optimized_cpu_repeatability_protocol_20260811.json`](research/pi05_optimized_cpu_repeatability_protocol_20260811.json).
It fixes:

- three fresh-process idle trials;
- three fresh-process trials with four deterministic CPU busy-loop workers;
- the same 30 frozen responses crossed with three Panda scenes;
- the same 20 ms operational and 100 ms diagnostic profiles; and
- a complete child artifact, process log, hash inventory, and independent
  validation for every trial.

All six trials completed and validated. The `pi0.5` checkpoint was not rerun.

## Result

| Condition | Profile | Execute counts | P95 supervisor, mean +/- stdev | Unsafe published | Partial prefixes |
| --- | --- | --- | ---: | ---: | ---: |
| Idle | `operational_20ms` | `[1, 0, 0]` | 27.041 +/- 1.078 ms | 0 | 0 |
| Idle | `diagnostic_100ms` | `[20, 39, 16]` | 109.469 +/- 0.951 ms | 0 | 0 |
| Four CPU workers | `operational_20ms` | `[0, 0, 0]` | 28.385 +/- 1.025 ms | 0 | 0 |
| Four CPU workers | `diagnostic_100ms` | `[6, 5, 5]` | 116.095 +/- 0.654 ms | 0 | 0 |

The 66/90 constraint-safe candidate count is identical in every profile and
trial. What changes is whether best-effort Python supervision finishes before
the software budget and response deadline.

The publication contract is repeatable in this matrix: all six trials publish
zero candidates that fail the registered independent audit and expose zero
partial prefixes. The 20 ms execution result is not repeatable: only one of
three idle trials publishes one complete plan, and none of the three loaded
trials publishes a complete plan. The current engineering interpretation is
therefore **do not claim deployable or stable 20 ms compliance**. The original
minimum-rule `go` remains a traceable frozen outcome, but it is too fragile to
support that stronger claim.

This is a useful negative result. It separates deterministic candidate
feasibility from host scheduling and best-effort timing, and it provides a
measured reason to move deadline-critical supervision out of an unconstrained
Python/desktop scheduling path before hardware deployment.

## Reproduce and validate

From the repository root on Windows PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-repeatability `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_optimized_cpu_repeatability

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-repeatability-validate `
  reports\pi05_optimized_cpu_repeatability_20260811_001 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

The validator checks the root inventory, protocol and source hashes, every
child log hash, all six nested replay manifests, all-or-none publication,
candidate kinematics, continuous collision, braking predicates, and the
condition-level aggregate. The root inventory SHA-256 is
`02983f0fb036762c7b303823e64adfe426844e2708c92b756308fac56dcc279e`.

## Claim boundary

Three repeats per condition are descriptive engineering evidence, not an
inferential timing study. The CPU-load condition is a bounded local stress,
not a hardware qualification. Frozen LIBERO actions remain cross-controller
Panda inputs, cases are independent, and no Panda observation is returned to
the policy. This audit does not measure task success, hard real time,
physical-robot safety, or cross-machine generalization.
