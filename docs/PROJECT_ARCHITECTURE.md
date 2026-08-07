# ArmBench Architecture and Claim Boundaries

Status: Current. Updated: 2026-08-07.

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

The maintained runtime now also contains a component-level non-blocking
harness: a blocking policy runs on a dedicated worker, a latest-only pending
mailbox bounds backlog, and the control side rejects superseded, failed,
deadline-missed, or horizon-exhausted responses. This harness uses a scripted
policy for CPU validation. The same scheduling contract is now connected to a
local torque-controlled MuJoCo Panda loop with live dual-camera capture,
measured-state terminal braking, and trace-derived validation. It has not
replaced the blocking-inference plus simulator-catch-up evaluator used by the
completed `pi0.5` studies.

## Current evidence

| Component | Validated result | Claim boundary |
| --- | --- | --- |
| Measured-age dispatcher | `pi0.5`-LIBERO Spatial, 120 matched pairs: 88/120 to 116/120, +23.33 points, exact McNemar `p=1.94e-6` | One frozen official checkpoint and simulation matrix |
| Cross-suite validation | Object, Goal, and LIBERO-10, 300 rollouts / 150 pairs: 83/150 to 141/150 | Same model family and simulator suite; deterministic-delay evidence |
| RTC-style sampler extension | 300 matched triplets: 96/100 baseline, 97/100 hard projection, 97/100 RTC | No task-success superiority; seam metrics are exploratory |
| Panda runtime | Protocol/guard/fault traces in local MuJoCo | Not official `pi0.5` policy efficacy or physical safety proof |
| Threaded runtime harness | Separate worker/control thread IDs, continued control ticks, latest-only replacement, and deadline tests | Scripted component evidence; no LIBERO or Panda task-success claim |
| Asynchronous Panda closed loop | 27 cases: braking invariant 9/9 physically safe and 0 abrupt stops; legacy 9/9 and 266 abrupt stops; unguarded 8/9 and 211 | Scripted single-run engineering matrix; not learned-policy efficacy, a statistical superiority test, hard real time, or physical safety certification |
| Cartesian action adapter | Scripted `H x 7` LIBERO-style chunk mapped through the Panda Jacobian into the existing `H x 8` guard contract | Component smoke only; no official checkpoint, task-success, or controller-equivalence claim |
| Frozen-response Panda replay | 7,934 official response hashes verified; 90 chunks replayed across three Panda scenes | Offline cross-controller diagnostic; no checkpoint execution, feedback loop, or task-success claim |
| Braking-invariant repair | 270 paired frozen-response cases: 264/270 to 270/270 registered constraints, all 6 legacy conflicts resolved, zero regressions | Training-free trajectory repair diagnostic; measured software budget, not hard real time or physical safety |

Detailed results are in [Results](RESULTS.md). Frozen protocols and audits are
listed in the [documentation index](README.md).

## What is integrated today

The project now includes a component-level Cartesian adapter that maps a
declared `H x 7` LIBERO-style end-effector chunk to the Panda runtime's `H x 8`
joint-velocity/gripper contract. It uses the MuJoCo Panda hand Jacobian, damped
least-squares differential inverse kinematics, joint-limit-aware scaling, and
the existing guard. Its deterministic CPU smoke is documented in
[LIBERO-to-Panda Cartesian adapter](PANDA_CARTESIAN_ADAPTER.md).

The adapter is now also exercised by a strictly validated offline replay of
frozen official-checkpoint responses. The replay checks every preserved
response hash, samples equally by LIBERO task and runtime method, resets each
Panda case independently, and emits a self-validating CSV/JSON artifact. See
[frozen pi0.5 response replay](PI05_PANDA_ARCHIVE_REPLAY.md).

The same replay archive now feeds a second, paired diagnostic: a
trajectory-level braking-invariant repair. It searches a bounded set of whole-
chunk velocity scales, validates configuration and interpolated-edge collision
constraints, and appends a terminal deceleration path before selecting a
candidate. On the preserved 270-case matrix it resolves all six legacy
collision/acceleration conflicts with zero repair regressions. See
[deadline-bounded braking-invariant repair](PI05_PANDA_BRAKING_REPAIR.md).

The asynchronous Panda runtime now integrates the local half of the chain. A
latest-only camera worker timestamps live Panda state and two simulated images;
a separate blocking scripted policy returns action chunks; every control tick
performs observation-age suffix dispatch and deadline checks; and PD plus bias
compensation applies torque-limited commands to MuJoCo. Expired or failed
responses trigger a stop rebuilt and checked from measured state. The resulting
events, NPZ traces, provenance, hashes, and recomputed metrics form a
self-validating artifact. See
[asynchronous Panda closed-loop runtime](ASYNC_PANDA_CLOSED_LOOP.md).

This is still not a verified live control chain from official `pi0.5`
inference to Panda execution. Scale, coordinate-frame, clipping, and gripper
conventions are now attested against LIBERO commit `f78abd68` and robosuite
`1.4.1`, but the differential-IK adapter is not dynamically equivalent to
robosuite's torque-level OSC. The official-checkpoint worker is also not
connected to this Panda loop or an independently ticking LIBERO actuator loop.
End-to-end claims require learned-policy integration, time synchronization,
and a new frozen experiment.

Do not write or say that `pi0.5` has been deployed on a Panda robot, that the
Panda guard certifies VLA safety, or that the simulation results establish
hard-real-time behavior.

Independent inference/control scheduling is implemented and tested in the
local scripted Panda actuator loop, but it is not yet wired to an official
checkpoint or the LIBERO evaluator. Python threads also provide no operating
system scheduling or worst-case latency guarantee.

## Next integration milestone

The technically meaningful next step is not a cosmetic simulator change. It is
to connect the implemented component adapter to a complete evaluator and test:

1. wire the independently scheduled runtime and the repair layer into an
   end-to-end evaluator and preserve stale-response discard in task-level
   evidence;
2. replace resolution-bounded edge sampling with a validated continuous or
   conservative swept-volume collision check and report dynamics limits;
3. reproduce the repair protocol on a second open action-chunk policy; and
4. add a simulator-to-hardware or LeRobot-compatible adapter with separate
   watchdog, safety, and timing evidence.

Until then, the accurate public description is: a seven-DoF constrained
execution base plus a `pi0.5` VLA runtime-evaluation path, with an explicit
component-level Cartesian adapter and shared auditable runtime infrastructure.
