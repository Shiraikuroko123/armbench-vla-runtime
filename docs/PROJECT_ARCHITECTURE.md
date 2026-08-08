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
- sampled configuration checks plus a fail-closed, clearance-backed continuous
  edge certificate for the declared static-obstacle and self-collision pairs;
- joint, velocity, gripper, and state-consistency checks;
- constrained QP action projection and sampled inverse-dynamics braking checks;
- dual-camera observations, transport tests, and deterministic fault injection.

This path validates runtime contracts and controlled failure behavior on a
seven-DoF arm model. The continuous checker is conservative for the compiled
MuJoCo geometry and declared joint-linear interpolation. The preserved audit
matrix covers static obstacles; broad self-collision coverage and physical
model accuracy remain open questions.

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
| Asynchronous Panda closed loop | 27 cases with clearance-backed swept obstacle checks: braking invariant 9/9 physically safe and 0 abrupt stops; legacy 9/9 and 311 abrupt stops; unguarded 8/9 and 289 | Scripted single-run engineering matrix; not learned-policy efficacy, a statistical superiority test, hard real time, or physical safety certification |
| Cartesian action adapter | Scripted `H x 7` LIBERO-style chunk mapped through the Panda Jacobian into the existing `H x 8` guard contract | Component smoke only; no official checkpoint, task-success, or controller-equivalence claim |
| Frozen-response Panda replay | 7,934 official response hashes verified; 90 chunks replayed across three Panda scenes | Offline cross-controller diagnostic; no checkpoint execution, feedback loop, or task-success claim |
| Braking-invariant repair | 270 paired frozen-response cases: 264/270 to 270/270 registered constraints, all 6 legacy conflicts resolved, zero regressions | Training-free trajectory repair diagnostic; measured software budget, not hard real time or physical safety |
| Continuous collision edges | 72 seeded static-obstacle edges: 0 false-safe decisions against a denser sampled oracle | Conservative compiled-geometry audit; self-collision is not broadly audited |
| Dynamics-feasible braking | 45/45 registered payload, damping, and velocity cases pass inverse-dynamics effort and edge checks | Sampled MuJoCo model feasibility; no closed-loop or hardware claim |
| Provider-neutral ABI | Synthetic OpenVLA-OFT-named `6x7` fixture bound to one observation, adapted to `6x8`, with 5/5 registered semantic mismatches rejected | Interface portability only; no OpenVLA-OFT checkpoint execution or cross-model task result |
| LeRobot-style actuator boundary | Five replayable frames: three executes, one stale-observation hold, one latched hold, one explicit reset | In-memory frame compatibility and software watchdog only; no official LeRobot runtime, driver, or robot |
| Official LeRobotDataset round-trip | Isolated `lerobot==0.4.4`/v3.0 loader reloads three Panda frames with images, state, action, task, and timestamps | Dataset serialization only; no policy checkpoint, SO-101 conversion, or driver |

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

The local MuJoCo checker now also exposes a clearance-backed swept static-
obstacle audit. It derives conservative per-joint workspace displacement
radii, subdivides edges against the configured static clearance, and compares
the result with a denser sampled oracle. The preserved 72-edge audit has zero
false-safe decisions. Self-collision remains sampled, and the audit is not a
continuous or physical-robot safety certificate. See
[MuJoCo swept collision audit](MUJOCO_SWEPT_AUDIT.md).

The asynchronous Panda runtime now integrates the local half of the chain. A
latest-only camera worker timestamps live Panda state and two simulated images;
a separate blocking scripted policy returns action chunks; every control tick
performs observation-age suffix dispatch and deadline checks; and PD plus bias
compensation applies torque-limited commands to MuJoCo. Expired or failed
responses trigger a stop rebuilt and checked from measured state. The resulting
events, NPZ traces, provenance, hashes, and recomputed metrics form a
self-validating artifact. The current checker also derives a conservative
per-joint workspace-motion bound and subdivides each clearance-backed edge;
the preserved v3 matrix records these bounds and the 20 mm margin in its
provenance. Self-collision remains sampled and dynamics are not certified. See
[asynchronous Panda closed-loop runtime](ASYNC_PANDA_CLOSED_LOOP.md).

The policy boundary is now provider-neutral at the ABI level. Frozen response
bundles carry model-family identity, checkpoint-attestation status, observation
binding, and a canonical action-semantics hash. Provider-native actions do not
enter the runtime until an exact gate accepts coordinate frame, control period,
normalization, scale, rotation, gripper, and controller fields and an explicit
adapter emits the Panda `Hx8` contract. The preserved second-family fixture is
synthetic, so this is portability evidence rather than cross-model efficacy.
See [provider-neutral action contract](PROVIDER_CONTRACT.md).

The actuator boundary now also exposes LeRobot-style in-memory frame keys and
a fail-closed command watchdog. Hash-manifested episode records preserve input
and dispatched commands, time/sequence metadata, latch/reset events, and enough
bytes for deterministic decision replay. A separate isolated exporter now writes
the same Panda Hx8 semantics through the official `LeRobotDataset` v3.0 loader
and checks the round-trip fields. This is dataset compatibility evidence, not a
policy, SO-101, or robot-driver integration. See [LeRobot-style runtime bridge](LEROBOT_RUNTIME_BRIDGE.md)
and [official LeRobotDataset round-trip](OFFICIAL_LEROBOT_ROUNDTRIP.md).

The Panda safety boundary also includes a dynamics-feasible braking audit. It
constructs sampled constant-deceleration stops, checks joint limits and
continuous collision edges, and recomputes MuJoCo inverse-dynamics effort under
registered payload and damping changes. The preserved 45-case matrix is model
feasibility evidence; it is not a hard-real-time, emergency-stop, or physical
robot certificate. See [Panda dynamics braking audit](DYNAMICS_BRAKING_AUDIT.md).

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

1. replace the second-family synthetic fixture with attested, checkpoint-backed
   OpenVLA-OFT captures and run a preregistered cross-model closed-loop matrix;
2. wire the official LeRobot dataset boundary and Panda watchdog to a concrete
   driver only after action semantics and reset behavior are specified;
3. expand the continuous self-collision audit and connect the dynamics-aware
   braking result to task-level online execution; and
4. add calibrated hardware timing, emergency-stop integration, and repeated
   physical fault-injection evidence.

Until then, the accurate public description is: a seven-DoF constrained
execution base plus a `pi0.5` VLA runtime-evaluation path, with provider-neutral
action semantics, an official LeRobot dataset round-trip, dynamics-aware
MuJoCo braking audits, and shared auditable runtime infrastructure. Learned
second-model and physical-robot evidence remain future work.
