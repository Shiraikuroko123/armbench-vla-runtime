# ArmBench Architecture and Claim Boundaries

Status: Current. Updated: 2026-08-05.

## Purpose

ArmBench is an engineering platform for evaluating how action-chunk VLA
policies behave between policy response and robot execution. It is not a new
foundation model, a `pi0.5` training project, a real-robot deployment, or a
collision-safety certification system.

The platform has two independently validated execution paths and one shared
runtime and evidence layer. This separation is intentional: it prevents a
local Panda guard result from being presented as a `pi0.5`-LIBERO result, or a
LIBERO task-success result from being presented as a Panda safety result.

## Terminology

- **`pi0.5`**: Physical Intelligence's pi-zero-point-five
  vision-language-action model. Use `pi0.5 VLA` after the first definition.
- **OpenPI**: the upstream model and inference implementation used to access
  the `pi0.5` checkpoint. It is not a separate model.
- **Action chunk**: a short sequence of future control actions predicted from
  one observation.
- **Measured age**: elapsed time from observation capture until a policy
  response becomes available to the controller.
- **Fail closed**: reject an invalid or unusable response and command the
  registered hold/fallback behavior rather than guessing an action.

## Components

### 1. Seven-DoF Panda constrained-execution base

The local MuJoCo Panda path provides the classical robotics base:

- RRT-Connect/RRT* planning, path smoothing, and time parameterization;
- PD/LQR trajectory tracking and disturbance-oriented execution checks;
- sampled configuration and interpolated-edge collision checks;
- joint, velocity, gripper, and state-consistency checks;
- dual-camera observations, transport tests, and deterministic fault injection.

This path validates runtime contracts and controlled failure behavior on a
seven-DoF arm model. Its collision checks are sampled checks, not a continuous
collision certificate.

### 2. `pi0.5` VLA temporal-evaluation path

The official-checkpoint path runs an attested `pi05_libero` checkpoint through
OpenPI on LIBERO. The policy receives canonicalized images, robot state, and a
language prompt, then produces an action chunk.

The central runtime question is whether the chunk remains temporally usable
when inference is delayed. The training-free measured-age dispatcher computes a
conservative suffix offset from response age, uses that suffix only when it
fits the remaining horizon and deadline, and otherwise enters bounded
hold/refresh behavior.

### 3. Shared runtime and evidence layer

Both paths reuse the engineering layer below:

```text
observation/state/prompt contract
             |
             v
policy transport and response validation
             |
             v
age/deadline/state checks and fail-closed supervision
             |
             v
per-query and per-action traces, video, manifests, validators, dashboards
```

Shared tooling does not mean shared experimental semantics. The LIBERO and
Panda action spaces, policies, execution environments, and result claims are
kept separate.

## Current evidence

| Component | Validated result | Claim boundary |
| --- | --- | --- |
| Measured-age dispatcher | `pi0.5`-LIBERO Spatial, 120 matched pairs: 88/120 to 116/120, +23.33 points, exact McNemar `p=1.94e-6` | One frozen official checkpoint and simulation matrix |
| Cross-suite validation | Object, Goal, and LIBERO-10, 300 rollouts / 150 pairs: 83/150 to 141/150 | Same model family and simulator suite; deterministic-delay evidence |
| RTC-style sampler extension | 300 matched triplets: 96/100 baseline, 97/100 hard projection, 97/100 RTC | No task-success superiority; seam metrics are exploratory |
| Panda runtime | Protocol/guard/fault traces in local MuJoCo | Not official `pi0.5` policy efficacy or physical safety proof |

Detailed results are in [Results](RESULTS.md). Frozen protocols and audits are
listed in the [documentation index](README.md).

## What is integrated today

The project currently integrates a common runtime interface and evidence model,
not a verified direct control chain from `pi0.5` LIBERO action outputs to Panda
joint commands. A direct chain would need an explicit action adapter, declared
coordinate frames and gripper semantics, inverse kinematics or constrained
projection, time synchronization, and new end-to-end experiments.

Do not write or say that `pi0.5` has been deployed on a Panda robot, that the
Panda guard certifies VLA safety, or that the simulation results establish
hard-real-time behavior.

## Next integration milestone

The technically meaningful next step is not a cosmetic simulator change. It is
to connect the two paths through a declared adapter and evaluate:

1. independently scheduled inference and control with stale-response discard;
2. deadline-bounded constrained projection, including continuous collision and
   dynamics limits;
3. a second open action-chunk policy under the same frozen protocol; and
4. a simulator-to-hardware or LeRobot-compatible adapter with separate safety
   and timing evidence.

Until then, the accurate public description is: a seven-DoF constrained
execution base plus a `pi0.5` VLA runtime-evaluation path, sharing auditable
runtime infrastructure.
