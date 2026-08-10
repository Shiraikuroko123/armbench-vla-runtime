# Frozen pi0.5 Responses Through Integrated Panda Assurance

Status: current R02 CPU preflight. The scored artifact is
[`reports/pi05_integrated_panda_cpu_replay_270_001`](../reports/pi05_integrated_panda_cpu_replay_270_001/summary.md).

## Question

This stage asks whether the existing Panda runtime can accept an attested
frozen `pi0.5` LIBERO response under one fixed 200 ms response deadline and a
20 ms CPU assurance budget. It compares the same adapted action chunk and the
same initial Panda state under three modes:

| Mode | Runtime decision boundary |
| --- | --- |
| `direct_dispatch` | Publish the complete adapted chunk without QP, collision rejection, or braking repair. |
| `qp_projection` | Apply OSQP joint position, velocity, and acceleration projection, then publish or hold on timing/QP failure. |
| `full_assurance` | Run QP, continuous static/self-collision certification, and inverse-dynamics stopping checks on a separate worker; activate only through `AtomicPandaPlanGate`. |

The checkpoint is not executed. The source archive was produced earlier by the
official Physical Intelligence `pi05_libero` checkpoint and is revalidated down
to all 7,934 response hashes before replay.

```text
attested frozen pi0.5 Hx7 response
                 |
                 v
LIBERO semantic gate + Panda differential IK (Hx8)
                 |
       +---------+----------+
       |         |          |
       v         v          v
    direct      OSQP     OSQP + continuous collision
   dispatch   projection  + inverse-dynamics braking
       |         |          |
       +---------+----------+
                 v
    all-or-none publication or fail-closed hold
                 |
                 v
 CSV + candidate/published NPZ + hashes + validator
```

## Frozen protocol

The protocol was committed before implementation and scoring at
[`docs/research/pi05_integrated_panda_cpu_protocol_20260810.json`](research/pi05_integrated_panda_cpu_protocol_20260810.json).
It fixes 30 response chunks selected equally across the 30 task/method strata,
three Panda scenes, all three modes, a 20 ms software budget, and a 200 ms
response deadline. The resulting matrix contains 270 rows.

## Result

| Mode | Execute / cases | Constraint-safe candidates | Unsafe plans published | 20 ms misses | P95 mode latency | Mean motion retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct dispatch | 90/90 | 0/90 | 90 | 0 | 0.002 ms | 1.000 |
| QP projection | 1/90 | 5/90 | 1 | 89 | 28.848 ms | 0.009 |
| Full assurance | 0/90 | 5/90 | 0 | 90 | 42.974 ms | 0.000 |

The registered decision is `no_go`: the current best-effort Python
implementation cannot publish a fully assured response under 20 ms. It fails
closed and exposes zero partial action prefixes. This result also separates two
problems that must not be conflated:

1. **Latency:** QP alone misses the budget in 89/90 cases; full supervision
   misses it in 90/90.
2. **Candidate feasibility:** after QP, only 5/90 candidates pass every
   registered kinematic, continuous-collision, and braking predicate.

The earlier braking-repair result is not contradictory. That experiment used a
bounded trajectory-scale search with a lighter offline edge/terminal-stop
contract and reported 12.777 ms P95. This stage adds OSQP, registered continuous
static and self-collision certificates, an inverse-dynamics stop at every action
boundary, and activation-time publication checks.

## Reproduce and validate

From the repository root on Windows PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-integrated-replay `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_integrated_panda_cpu_replay

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-integrated-replay-validate `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

The validator checks the recursive manifest, frozen protocol, implementation
and scene hashes, source response metadata, complete paired matrix, all-or-none
publication, trajectory integration, continuous collision, inverse-dynamics
braking, aggregates, and Markdown. Tests also re-sign modified CSV/NPZ files and
confirm that semantic tampering is rejected.

## Next CPU iteration

The next implementation should retain the frozen result and create a new
artifact. The local priorities are persistent/prebuilt OSQP solves, reusable
MuJoCo braking workspaces and actuator limits, conservative broad-phase
collision rejection, certificate reuse, and a whole-chunk repair that improves
feasibility before the expensive exact checks. GPU inference is not needed for
that work.

## Claim boundary

Hand displacement is a command-retention proxy, not task progress. Every case
resets from a Panda scenario start and does not feed observations back to the
policy. The result is not a task-success experiment, hard-real-time guarantee,
physical-robot test, safety certificate, or cross-model result.
