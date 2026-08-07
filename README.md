[Project website](https://shiraikuroko123.github.io/armbench-vla-runtime/) | [简体中文](README_ZH.md) | [Documentation](docs/README.md)

[![CPU CI](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml)

# ArmBench

ArmBench is a runtime and evaluation platform for action-chunk
vision-language-action (VLA) policies. It studies what happens after a policy
has produced a short future action sequence but before that sequence reaches a
robot controller.

The repository contains two separately validated paths:

- A local 7-DoF MuJoCo Panda execution base for motion planning, constrained
  tracking, protocol validation, and fault handling.
- An official-checkpoint evaluation path for Physical Intelligence's
  `pi0.5` (pi-zero-point-five) VLA model through OpenPI and LIBERO.

They share runtime contracts, telemetry, validation, and artifact tooling. They
are not yet a verified end-to-end `pi0.5`-to-Panda deployment.

## The problem

An action-chunk policy generates several future actions from one observation.
Inference takes time. When the response arrives, its early actions can describe
time slots that have already passed. Blindly executing from action index zero
therefore creates a stale-action error.

ArmBench measures observation age, selects an unexpired action suffix, and
uses a bounded hold/refresh path when the response misses its deadline. The
runtime also rejects malformed, non-finite, disconnected, stale, or
state-inconsistent policy responses.

## Architecture

```text
images + language + robot state
              |
              v
Physical Intelligence pi0.5 VLA via OpenPI
              |
              v
          action chunk
              |
              v
temporal supervisor + response validation
              |
              +--> LIBERO closed-loop evaluation
              |
              +--> local Panda runtime validation
                     (planning, tracking, guard, fault injection)
```

The Panda and LIBERO paths use different action contracts and report separate
results. See [Architecture and claim boundaries](docs/PROJECT_ARCHITECTURE.md)
for the full design and the current integration gap.

## Validated evidence

| Study | Evidence | Interpretation |
| --- | --- | --- |
| Measured-age temporal alignment | Official `pi0.5`-LIBERO Spatial, 120 matched pairs: 88/120 to 116/120, +23.33 points, exact McNemar `p=1.94e-6` | Training-free, observation-age-based suffix selection improves this frozen simulation matrix |
| Cross-suite validation | Object, Goal, and LIBERO-10: 300 rollouts / 150 pairs, 83/150 to 141/150 | Extends deterministic-delay evidence within the same model family and simulator suite |
| RTC-style continuation | 300 rollouts / 100 matched triplets: 96/100 baseline, 97/100 hard projection, 97/100 RTC guidance | No task-success superiority; motion-seam measurements remain exploratory |
| Braking-invariant Panda repair | 270 paired offline cases: 264/270 to 270/270 registered constraints, all 6 legacy conflicts resolved, 0 regressions | Frozen `pi0.5` responses replayed through the Panda adapter; no task-success or hard-real-time claim |
| Asynchronous Panda closed loop | 27 CPU wall-clock cases with clearance-backed swept obstacle checks: braking invariant was physically safe in 9/9 with 0 abrupt stops and 0 repair-budget misses; legacy recorded 311 abrupt stops and unguarded 289 | Live dual-camera, policy-worker, dispatcher, repair, and torque-control integration using a scripted non-learned policy; not learned-policy efficacy or physical certification |
| Clearance-backed swept audit | 72 seeded MuJoCo edges across three scenes: 0 false-safe decisions against a denser sampled oracle | Conservative static-obstacle audit; self-collision and continuous physical safety remain out of scope |

Full protocols, validators, statistics, and limitations are in
[Results](docs/RESULTS.md).

## Scope

- ArmBench does not train or fine-tune `pi0.5`.
- Official-checkpoint results are simulation-only LIBERO evidence.
- The temporal studies use blocking inference plus simulator catch-up, not an
  operating-system-level hard-real-time control loop.
- Panda guard evidence does not certify collision safety and is not evidence
  that `pi0.5` controls a Panda robot.
- No Isaac Lab, ROS2, real Franka Panda, or safety PLC integration is claimed.

## Local CPU quickstart

The local Panda path requires Git and CPython 3.10. It does not require a GPU,
an OpenPI server, or a robot. From a fresh clone on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
& '.\.venv\Scripts\python.exe' -m armbench doctor
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view --scenario narrow_gate
```

On Ubuntu or another supported Linux distribution:

```bash
./scripts/setup_local.sh
./.venv/bin/python -m armbench doctor
./.venv/bin/python -m armbench mujoco-view --scenario narrow_gate
```

The setup script installs the CPU dependencies and checks out the pinned Panda
model under `.cache/`. Add `-WithVla` on PowerShell or `--with-vla` on Linux
only when the lightweight OpenPI client is needed; this does not download a
`pi0.5` checkpoint. See [local setup and support](docs/LOCAL_SETUP.md) for
manual installation, model overrides, and headless environments.

For a bounded local acceptance check on Windows:

```powershell
.\scripts\vla_demo.cmd -CheckOnly
```

The model-free asynchronous harness verifies that a blocking policy call runs
on a separate worker while the control side continues to tick:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke
```

This is component-level scripted evidence, not a new `pi0.5` task-success
result. See [non-blocking runtime harness](docs/ASYNC_RUNTIME.md).

The CPU-only Cartesian adapter smoke maps a scripted LIBERO-style `H x 7`
end-effector chunk through the MuJoCo Panda hand Jacobian and the existing
joint-space guard:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-adapter-smoke
```

This closes a component-level action-semantics boundary. It does not run
`pi0.5` or establish an end-to-end deployment. See
[LIBERO-to-Panda Cartesian adapter](docs/PANDA_CARTESIAN_ADAPTER.md).

The next CPU-only acceptance path replays hash-verified action chunks that were
previously produced by the attested official `pi0.5` LIBERO checkpoint:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-archive-replay `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_archive_replay --chunks 90
```

The preserved 90-chunk report covers 270 independent Panda lookahead cases.
It found 36 invalid raw paths and six cases where collision avoidance and the
configured acceleration bound could not both be satisfied. This is offline
cross-controller diagnostic evidence: the checkpoint was not rerun, no Panda
closed loop was executed, and no task-success claim is made. See
[frozen pi0.5 response replay](docs/PI05_PANDA_ARCHIVE_REPLAY.md).

The next CPU-only stage compares the legacy per-step guard with a
trajectory-level braking-invariant repair on the same frozen responses:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-braking-repair `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_braking_repair_90_001 `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-braking-repair-validate `
  results\pi05_panda_braking_repair_90_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

The checked-in [braking-repair report](reports/pi05_panda_braking_repair_90_001/summary.md)
shows 270/270 cases satisfying the registered constraints, resolving all six
legacy collision/acceleration conflicts with zero regressions. The report is a
paired offline diagnostic: it does not rerun `pi0.5`, close the Panda feedback
loop, or provide a physical-safety or hard-real-time guarantee. For trajectory
inspection, use the `raw_positions`, `legacy_positions`, and `repair_positions`
arrays with `mujoco-view`; the method and claim boundary are documented in
[deadline-bounded braking-invariant repair](docs/PI05_PANDA_BRAKING_REPAIR.md).

The local asynchronous closed-loop stage connects live dual-camera capture, a
blocking policy worker, observation-age suffix selection, deadline fallback,
braking repair, and torque-controlled Panda physics. Its built-in policy is
scripted and non-learned so the entire runtime can be accepted on a CPU:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-async-run `
  --output-directory results\async_panda_quick `
  --scenario single_block --quick --deadline-ms 400

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-async-validate results\async_panda_quick
```

Each case preserves wall-clock events and measured MuJoCo traces. Add
`--videos` for post-run MP4 rendering, or replay `actual_positions` with
`mujoco-view`. This closes the local runtime/control feedback loop; it does not
execute a learned VLA checkpoint or establish hard-real-time or robot-safety
claims. See [asynchronous Panda closed-loop runtime](docs/ASYNC_PANDA_CLOSED_LOOP.md).

The preserved [27-case v3 report](reports/async_panda_closed_loop_400ms_20mm_v3_001/summary.md)
uses five fixed latencies plus jitter, response loss, payload, and persistent
action-fault conditions. Braking-invariant execution recorded 0 abrupt-stop
violations and 9/9 physically safe traces, with 7.81 ms P95 and 19.01 ms
maximum repair latency. Its static-obstacle edges use the recorded 20 mm
clearance-backed swept subdivision; self-collision remains sampled. It reached
the target in only 1/9 conditions, making the safety/progress and local CPU-
throughput limits explicit rather than hiding them behind an aggregate success
claim.

## Review preserved evidence

These commands validate stored artifacts and rebuild offline dashboards. They
do not rerun model inference and do not require a GPU.

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd
```

Browse the [evidence catalog](docs/EVIDENCE_CATALOG.md) to distinguish primary
results, pilots, integration gates, rejected runs, and scripted runtime checks.
Its [machine-readable form](docs/evidence_catalog.json) links every preserved
artifact to its result, protocol, manifest, raw review files, validator, and
explicit claim boundary.

For local setup, remote OpenPI execution, debugging, and environment support,
use [the documentation index](docs/README.md).

## Repository layout

```text
src/armbench/        local Panda planning, control, MuJoCo, and runtime code
integrations/openpi/ official-checkpoint evaluators, analyses, and patches
tests/               unit and integration tests
scripts/             launch and acceptance commands
docs/                design, operations, protocols, and audits
evidence/            preserved experiment artifacts
reports/             offline dashboards generated from evidence
```

ArmBench is released under the MIT License. Upstream software, models, and
assets retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
