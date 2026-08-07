[Project website](https://shiraikuroko123.github.io/armbench-vla-runtime/) | [简体中文](README_ZH.md) | [Documentation](docs/README.md)

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

## Review preserved evidence

These commands validate stored artifacts and rebuild offline dashboards. They
do not rerun model inference and do not require a GPU.

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd
```

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
