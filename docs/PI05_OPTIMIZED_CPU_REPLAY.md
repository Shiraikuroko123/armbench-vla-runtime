# Optimized pi0.5-to-Panda CPU Assurance Replay

Status: R02 CPU optimization audit. The preserved artifact is
[`reports/pi05_optimized_cpu_replay_180_001`](../reports/pi05_optimized_cpu_replay_180_001/summary.md).
It succeeds the retained
[`20 ms no-go baseline`](PI05_INTEGRATED_PANDA_CPU_REPLAY.md); it does not
rewrite that earlier result.

## Question

The baseline showed that a complete Panda assurance worker could fail closed
but could not publish any fully checked plan inside a 20 ms software budget.
This audit asks whether CPU-only implementation changes can make at least one
complete plan atomically executable without weakening the registered
kinematic, continuous-collision, braking, deadline, or publication checks.

The optimized path adds:

- a persistent OSQP workspace for the joint position, velocity, and
  acceleration projection;
- conservative broad-phase pruning with vectorized interval bounds before
  exact MuJoCo distance queries;
- reusable inverse-dynamics and braking workspaces;
- safe-only configuration certificate reuse, reset between formal cases; and
- one optimized supervisor that publishes a complete plan or a zero-motion
  hold, never a partially accepted prefix.

## Fixed engineering audit

The protocol is stored at
[`docs/research/pi05_optimized_cpu_replay_protocol_20260810.json`](research/pi05_optimized_cpu_replay_protocol_20260810.json).
It was fixed after optimization profiling, so this is an engineering audit,
not a preregistered inferential study.

The runner reuses the 30 attested frozen responses and three Panda scenes from
the baseline. Each of the resulting 90 cases is evaluated under two profiles:

| Profile | Purpose | Supervisor budget | Response deadline |
| --- | --- | ---: | ---: |
| `operational_20ms` | Primary engineering go/no-go | 20 ms | 200 ms |
| `diagnostic_100ms` | Separate timing diagnosis | 100 ms | 200 ms |

Workers remain alive within a scene/profile, but startup is excluded from
request latency and every formal case clears its safe-configuration cache. The
100 ms profile cannot override the operational decision.

## Result

| Profile | Execute / cases | Constraint-safe candidates | Unsafe published | Budget misses | Deadline misses | P50 / P95 supervisor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `operational_20ms` | 1/90 | 66/90 | 0 | 69 | 3 | 21.493 / 23.888 ms |
| `diagnostic_100ms` | 58/90 | 66/90 | 0 | 3 | 8 | 49.560 / 86.559 ms |

Both profiles expose zero partial prefixes. The broad phase prunes 95.55% of
registered pair tests in the 20 ms profile and 94.99% in the diagnostic
profile before exact pair evaluation.

The frozen operational decision is **`go`** because one complete plan is
executed, no published plan fails the independent registered-constraint
audit, the artifact validates, and no partial prefix is exposed. This is a
minimum-rule pass, not stable 20 ms performance: 89/90 operational cases hold,
the operational P50 and P95 both exceed 20 ms, and the maximum supervisor
latency is 24.550 ms. The audit therefore shows a large feasibility
improvement while leaving timing determinism as the next CPU question.

For engineering context, the retained baseline full-assurance path executed
0/90 cases and found 5/90 constraint-safe candidates at 42.974 ms P95. The
optimized path executes 1/90 and finds 66/90 at 23.888 ms P95. This is a
versioned implementation progression over the same frozen input matrix, not a
statistical method comparison.

## Reproduce and validate

From the repository root on Windows PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-replay `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_optimized_cpu_replay

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-replay-validate `
  reports\pi05_optimized_cpu_replay_180_001 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

The validator checks the recursive manifest, protocol and input binding,
implementation and scene hashes, all 180 row identities, candidate and
published trajectory integration, independent kinematic/collision/braking
predicates, all-or-none publication, aggregates, and Markdown. Tests also
confirm that a re-signed semantic modification is rejected.

## Claim boundary

The official `pi0.5` checkpoint is not rerun. The responses are LIBERO actions
adapted across controllers to Panda, not native Panda policy outputs. Cases are
independent and do not return Panda observations to the policy, so this audit
does not measure task success. Python wall-clock measurements are not an
operating-system hard-real-time guarantee, and MuJoCo checks are not a
physical-robot safety certificate.
