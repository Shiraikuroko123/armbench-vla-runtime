# VLA Interview Guide

## Graduate-student assessment

This project is now credible as a primary resume project for **VLA deployment,
embodied systems, robot learning infrastructure, simulation/evaluation, or
robotics software** roles. It demonstrates that you can connect a policy API to
robot observations/actions, reason about asynchronous execution, preserve
provenance, inject failures, and verify behavior in physics.

It is only a supporting project for **VLA model architecture, pretraining, or
fine-tuning research** roles. There is no learned model training, dataset
pipeline, representation ablation, or real-checkpoint task benchmark yet. A
single remote `vla-probe` would prove integration, not close that gap.

## Ninety-second explanation

ArmBench is an OpenPI-compatible runtime assurance layer for VLA action chunks.
The simulated Franka Panda produces an exterior image, a wrist image, seven
joint positions, a gripper position, and a language prompt using the exact
`pi05_droid` input keys. A common policy interface can call the official OpenPI
WebSocket client or a deterministic fault-injection policy. It validates the
returned 15-by-8 DROID chunk before torque execution: stale chunks trigger a
latched hold, joint velocities and gripper commands are bounded, and each joint
edge is checked against MuJoCo Panda meshes with 20 mm clearance. Unsafe actions
are backtracked or replaced by hold. In two fixed scenes, the guard preserved
both safe streams, reduced 2,314 injected contact steps to zero, and had a worst
per-case P95 of 7.96 ms on an Intel laptop. The benchmark uses scripted actions,
not a pi0.5 checkpoint, so the claim is runtime integration and evaluation, not
learned-policy performance or certified safety.

## What you actually built

- The Panda camera/proprioception adapter and visible task target.
- Immutable Python data contracts for DROID observations and action chunks.
- A strict wrapper around official `openpi-client`, including a real local
  protocol round-trip test and a real-server probe command.
- A stateful runtime guard with sequence/age checks, deadline latch/reset,
  action bounds, joint limits, sampled mesh-edge lookahead, backtracking, and
  hold fallback.
- Deterministic safe/fault streams, inference jitter schedules, and matched
  guarded/unguarded physics cases.
- Per-case, per-chunk, and per-action audit trails with raw and executed actions,
  reasons, scales, predicted/actual states, images, plots, and videos.
- A Windows self-locating launcher, VS Code debug configurations, tests, pinned
  third-party versions, and explicit claim boundaries.

The official Panda assets, MuJoCo engine, and OpenPI client are dependencies,
not original work. Say exactly that.

## Questions to expect

### What is the difference between this and pi0/pi0.5?

pi0/pi0.5 is the learned policy that maps images, language, and state to an
action chunk. ArmBench is the runtime around that policy: it creates the request,
transports it, validates the reply, handles deadlines, checks execution
constraints, applies or rejects actions, and measures the physical result.

The project has an OpenPI client path but the tracked experiment uses a
non-learned source. Therefore “OpenPI-compatible” is correct; “deployed pi0.5”
is not yet correct.

### Why use the `pi05_droid` contract?

DROID uses a Franka setup and exposes a concrete public interface: two 224x224
RGB images, seven joint positions, one gripper position, a prompt, and joint
velocity plus gripper action chunks. That makes the policy/runtime boundary
testable without inventing a private VLA API. The contract and client are pinned
to one upstream commit because horizons and example code can change.

### Why is the action horizon 15?

The pinned `pi05_droid` model config sets an action horizon of 15 and action
dimension 8. DROID runs at about 15 Hz. The official example has an older comment
mentioning 10 actions, so the implementation trusts the pinned model config and
strictly rejects any other response shape.

### Are actions normalized?

The first seven values are DROID joint-velocity commands. The official DROID
example clips the returned action to `[-1, 1]` before stepping a joint-velocity
environment. ArmBench therefore clips to +/-1 rad/s and then applies the Panda
joint-specific hardware velocity limits. It does not multiply a normalized
fraction by each hardware limit.

### Why a latched deadline fallback?

A 240 ms chunk exceeds the configured 200 ms age threshold. Initially the
runtime held only that chunk and resumed the remaining precomputed stream; the
state mismatch produced 43 contact steps in a regression run. The corrected
state machine enters hold and requires explicit reset/resynchronization. The
formal rerun had zero contacts but did not complete the task. This is a concrete
example of safety taking priority over availability.

### Why not execute only the first action and re-query immediately?

That is another valid closed-loop design, but it increases inference traffic
and may not fit model latency. This benchmark validates complete chunks because
chunking is the public OpenPI/DROID interface. A follow-up experiment should
vary the executed open-loop horizon and compare responsiveness, deadline rate,
smoothness, and task success.

### Is the guard guaranteed safe?

No. Configuration contacts use MuJoCo meshes, but edge validation samples joint
interpolation at 0.02 rad resolution. Physics can deviate from the predicted
path. The runtime bounds command-space acceleration, but it has no jerk bound,
dynamics-level reachable set, or formal proof. The evidence is zero contact in
fixed cases, not a theorem or safety certification.

### Why does the collision guard fail the task?

The injected stream goes through an obstacle. The local repair scales each
unsafe velocity and eventually holds; it cannot invent a new path around the
obstacle. Zero contact with large goal error is a correct refusal, not task
success. A recovery planner or a newly conditioned VLA chunk is needed to finish.

### Why have RRT-Connect in a VLA project?

It produces a known-safe positive-control stream and supplies the geometry,
clearance, and physics substrate used to test the VLA runtime. It is not
presented as the VLA itself. Comparing a safe planner stream, an unsafe direct
stream, guarded execution, and unguarded execution gives the safety layer both
positive and negative controls.

### Why MuJoCo instead of Isaac Gym or Isaac Lab?

The available host has no NVIDIA GPU. MuJoCo provides official Panda meshes,
contacts, torque dynamics, rendering, and reproducible low-volume evaluation on
the Intel laptop. Isaac Lab is a simulator/training framework, not a VLA model;
it is valuable when large parallel GPU rollouts or NVIDIA sensor pipelines are
needed. Porting the evaluator later would be additional cross-simulator
validation, not a prerequisite for this runtime result.

### Can it run on a real Panda?

No. A real deployment needs a `libfranka` or ROS2 adapter, calibrated camera and
joint conventions, deterministic command timing, watchdogs, emergency-stop
integration, collision thresholds, network failure handling, and hardware
validation. The present guard is research simulation code, not safety-rated.

### What would you do next with a GPU budget?

First run `pi05_droid` on a remote RTX 4090 and store honest probe artifacts.
Then build a closed-loop rollout with controlled open-loop horizons and compare
pi0, pi0.5, and a non-learned baseline under paired observations, seeds, jitter,
and fault injection. Because these synthetic scenes are out of DROID's training
distribution, task claims would require adaptation or an appropriate benchmark
such as LIBERO, plus confidence intervals and failure taxonomy.

## How to live-debug it in an interview

1. Run `scripts\vla_demo.cmd -CheckOnly` from an arbitrary directory.
2. Open the formal exterior/wrist images and explain every request tensor.
3. Filter `per_action.csv` for `backtracked:` and show raw versus executed action.
4. Filter for `deadline_latched` and explain why later fast chunks still hold.
5. Play guarded and unguarded fault videos side by side.
6. Set a breakpoint in `ActionChunkGuard.guard`, change one backtracking scale,
   rerun a new quick ID, and predict what metric should move.

Do not edit the formal evidence directory. New runs are immutable and use new
IDs.

## Honest claim boundaries

Do not claim:

- pi0/pi0.5 checkpoint results without a real `probe.json` and rollout artifact;
- model training, fine-tuning, or DROID/LIBERO dataset evaluation;
- analytic continuous collision detection;
- OS-level hard real-time scheduling;
- jerk constraints or dynamics-verified acceleration guarantees;
- Isaac Lab, ROS2, `libfranka`, or real-robot deployment;
- formal or certified safety.

The strongest current claim is an end-to-end, OpenPI-compatible VLA
runtime/evaluation system with deterministic fault injection, physics evidence,
deadline state handling, and action-level auditability.

## Resume wording

> Built an OpenPI-compatible VLA action runtime for a MuJoCo Franka Panda,
> converting dual 224x224 RGB views, language, and proprioception into the
> pi0.5-DROID remote inference contract and validating 15x8 action chunks with
> deadline-latched fallback, joint/velocity constraints, mesh-collision
> lookahead, and action backtracking. Across two fixed fault-injection scenes,
> preserved 2/2 safe trajectories and reduced 2,314 injected contact steps to
> zero, with guard P95 at most 7.96 ms on an Intel laptop; packaged per-action
> audit logs, tests, camera evidence, and MP4 replays.

Use “OpenPI-compatible,” not “pi0.5 deployment,” until real checkpoint evidence
exists.

## Before placing it on a resume

You should be able to do all of the following without reading a prepared answer:

1. draw the five runtime boundaries and state every tensor shape;
2. explain why pi0/pi0.5 is not a simulator and why Isaac Lab is not a VLA;
3. reproduce the quick run and locate one rejected action in `per_action.csv`;
4. explain the deadline-latch regression and the safety/task-success tradeoff;
5. distinguish sampled kinematic validity from contact-free physics execution;
6. change a prompt, latency schedule, or guard threshold and predict the result;
7. state which code/assets are yours and which are pinned dependencies.

AI assistance produced substantial implementation and documentation. Your
defensible ownership comes from being able to reproduce, inspect, modify,
debug, and explain the system. Do not imply that you independently typed code
you cannot maintain.
