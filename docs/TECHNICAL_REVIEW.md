# ArmBench technical review guide

## Purpose

This document provides a structured review of the implementation, experimental
evidence, and operating boundaries. It is intended for maintainers and
technical reviewers who need to trace a result from its system assumption to
code, runtime records, and validation output.

For commands and installation issues, use [Troubleshooting](DEBUGGING.md). For
complete numerical results and provenance, use [Results](RESULTS.md). Review
[Architecture and claim boundaries](PROJECT_ARCHITECTURE.md) before treating
the local Panda path and the Physical Intelligence `pi0.5` VLA path as one
system claim.

## System summary

ArmBench is a runtime and evaluation layer for action-chunk VLA policies. Its
primary method addresses observation/action misalignment caused by inference
latency: when a response is d control periods old, the dispatcher selects a
suffix that begins at action d instead of executing the chunk from action zero.

A separate policy-internal path evaluates hard projected overlap and RTC-style
denoised-action VJP guidance in the pinned pi0.5 flow sampler. It conditions a
new sample on controls already committed by the scheduler. This is not the same
operation as suffix selection.

The project also includes a local MuJoCo/Panda runtime for observation and
action contracts, fault injection, action-level guards, and torque-controlled
closed-loop execution.

## Execution boundaries

| Boundary | Official checkpoint path | Local validation path |
| --- | --- | --- |
| Policy | Attested pi05_libero checkpoint | OpenPI remote server or deterministic test fixture |
| Observation | LIBERO images, language, and robot state through official transforms | Two 224x224 RGB cameras, seven joints, gripper, prompt |
| Action | pi0.5 LIBERO action horizon 10 through official transforms | DROID-compatible 15x8 joint-velocity/gripper chunk |
| Execution | LIBERO closed-loop simulation | MuJoCo Menagerie Franka Panda |
| Primary use | Method evaluation | Contract, guard, and fault-response validation |

Evidence from one path is not used to establish claims about the other.

## Evidence summary

| Study | Result used in the current project |
| --- | --- |
| Deterministic alignment core | At 200 ms, 18/50 asynchronous vs 50/50 aligned; +64 points, Holm-adjusted McNemar p=1.40e-9 |
| Measured-age confirmation | 88/120 baseline vs 116/120 aligned; +23.33 points, McNemar p=1.94e-6 |
| Cross-suite validation | Prespecified Object, Goal, and LIBERO-10 tests each remain significant after Holm correction |
| Corrected-v3 RTC overlap | 96/100 unconditioned and 97/100 for each conditioned method; no success advantage, Holm-adjusted p=1.0 |
| Local runtime fault matrix | Deterministic protocol and safety-fault handling with explicitly non-learned fixtures |

These studies answer different questions and must not be pooled.

## Code map

| Concern | Primary implementation |
| --- | --- |
| Fixed-delay action selection | integrations/openpi/libero_runtime.py |
| Measured-age timing and suffix decisions | integrations/openpi/deadline_alignment.py |
| Explicit pi0.5 sampling noise and server attestation | integrations/openpi/serve_policy_attested.py |
| RTC/projected-overlap scheduling | integrations/openpi/projected_overlap_runtime.py |
| RTC combined analysis and validation | integrations/openpi/rtc_overlap_primary_analysis.py |
| Local OpenPI observation contract | src/armbench/vla/observation.py |
| Bounded policy transport | src/armbench/vla/policy.py |
| Runtime supervision and deadline latch | src/armbench/vla/runtime.py |
| Action-chunk guard | src/armbench/vla/guard.py |
| MuJoCo closed-loop execution | src/armbench/vla/online.py |
| Artifact integrity | src/armbench/vla/artifact.py and study-specific validators |

## Review topics

### Temporal alignment assumption

For a control period Δt and measured observation age a, the conservative
alignment index is derived from a / Δt and bounded by the action horizon. The
dispatcher fails closed when the selected start plus the requested execution
horizon exceeds the available chunk.

The method assumes action k is the policy command appropriate approximately k
control periods after the source observation. That assumption is testable but
is not a model-predictive guarantee.

### Fixed delay versus measured age

The deterministic study supplies a known injected delay to isolate the causal
effect of suffix selection. The measured-age runtime instead derives the index
from timestamps visible to the client. It records capture, request, response,
jitter, catch-up, and dispatch times so the validator can recompute the
decision.

The measured-age confirmation pairs both response jitter and explicit pi0.5
flow-sampling noise. The earlier pilot paired jitter but not policy noise and
therefore remains exploratory.

### Suffix selection versus RTC overlap

Suffix selection advances the simulator for the inferred delay and then
executes a later slice from the returned chunk. RTC overlap commits part of an
existing chunk while the next chunk is being sampled, then conditions the new
sample on those committed controls.

For OpenPI time t=1 at noise and t=0 at action, the implemented denoised-action
estimate is:

~~~text
D_t(x) = x - t * v_openpi(x, t)
error  = weights * (reference - D_t(x))
v_guided = v_openpi - gain(t) * J_D(x)^T * error
~~~

The sign accounts for OpenPI Euler integration with negative dt. The
transpose-Jacobian product is computed with jax.vjp.

### RTC v2 rejection and corrected-v3

RTC v2 reused one LIBERO environment across method conditions. Reset did not
restore identical policy images for tasks 3, 8, and 9, so query-zero actions
diverged even though declared state and sampling noise matched. This invalidated
the matched comparison.

Corrected-v3 creates a new environment per rollout and gates finalization on
four query-zero hashes: policy input, response action, sampling key, and
sampling noise. The corrected matrix is a complete 300-rollout rerun on
held-out seeds. The v2 results are retained only for audit.

### Why reinforcement learning is not part of the current method

The primary intervention is a deterministic runtime synchronization rule.
Introducing RL would add policy optimization without resolving the timing
contract that the experiment isolates. RL becomes relevant if the runtime must
learn a sequential decision, such as whether to hold, refresh, shorten a
chunk, or request takeover under uncertain latency and state.

The absence of RL is therefore a scope decision, not evidence that RL is
unnecessary for broader embodied-control problems.

### Action semantics

The official LIBERO path uses the checkpoint's own transforms and ten-action
horizon. The local DROID path accepts exactly a 15x8 response: seven
joint-velocity controls and one normalized gripper command per step. The two
contracts are not interchangeable.

The local runtime clips commands to the declared policy range and Panda limits,
applies acceleration slew limits, checks joint and sampled collision
constraints, and backtracks unsafe steps. Every modification is written to
per_action.csv with the raw action, executed action, scale, and reason.

### Deadline and transport failures

Connection refusal, timeout, malformed shape, nonfinite values, stale state,
and observation replay are converted into an explicit hold before the action
guard. A deadline miss latches the supervisor; later responses do not
automatically resume an older open-loop stream. Recovery requires an explicit
reset or a registered refresh path.

### Observation refresh

Each online query records image hashes, adjacent-frame pixel deltas,
proprioception, sequence identifiers, and timestamps. Optional full-frame
recording permits exact request replay. Camera-freeze tests replay an earlier
frame while proprioception changes and verify that the second policy call is
rejected before inference.

These checks detect the registered blank/replay faults. They do not establish
general semantic sensor validity.

### Safety interpretation

The local guard demonstrates bounded response to registered faults in MuJoCo.
It is not a formal safety controller. Collision checks use MuJoCo geometry at
sampled configurations and interpolated joint-space edges; command slew limits
do not certify physical acceleration or jerk.

A held episode may remain collision-free while failing its task. Safety and
task completion are reported as separate outcomes.

### Simulator choice

MuJoCo supports the available CPU/Windows environment, official Panda assets,
contact dynamics, and deterministic local tests. Isaac Lab would be appropriate
for GPU-parallel simulation, RL training, or Omniverse sensor workflows. Adding
a simulator does not change whether the action source is learned.

### Hardware deployment

The current repository does not include ROS2, libfranka, calibration,
command-rate enforcement, hardware watchdogs, emergency-stop integration, or
a safety PLC adapter. A real Panda deployment requires each of those layers
and a new hardware validation protocol.

## Review workflow

### 1. Verify the installation

~~~powershell
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -CheckOnly
~~~

### 2. Validate preserved checkpoint evidence

~~~powershell
.\scripts\alignment_acceptance.cmd -NoOpen
.\scripts\measured_age_confirmatory_acceptance.cmd -NoOpen
.\scripts\rtc_primary_acceptance.cmd -NoOpen
~~~

Each command must return a zero exit status and a report with valid=true before
its result is cited.

### 3. Inspect one matched pair

Open the generated dashboard, select the registered primary condition, and
play both videos together. Cross-check the task, initial state, condition,
success label, query count, and video path against the bound CSV.

### 4. Trace one action

For a local MuJoCo artifact:

1. locate a modified row in per_action.csv;
2. inspect the raw and executed action;
3. follow the recorded reason into ActionChunkGuard.guard;
4. verify the corresponding state and contact trace in the NPZ and video.

### 5. Reproduce a diagnostic

Run a new smoke ID rather than modifying preserved evidence:

~~~powershell
& '..\.venv\Scripts\python.exe' -m armbench vla-online-run --quick --videos --save-observations --run-id review_smoke
~~~

## Evidence interpretation

Artifact validation establishes that files, matrix membership, derived
statistics, and required videos agree with the bound manifests. Checkpoint
attestation binds the run to recorded local checkpoint content and source
state. Neither mechanism authenticates the upstream publisher or certifies
physical safety.

Negative and invalid results are handled differently:

- a valid negative result remains part of the evidence;
- an infrastructure failure remains in the registered denominator when the
  protocol requires intention-to-test;
- a comparison that violates its pairing invariant is excluded from
  method-effect estimation and retained as an audit record.

## Current limitations

- one official VLA checkpoint family;
- simulation-only execution;
- blocking inference with simulator catch-up rather than independent control
  and inference clocks;
- no hardware adapter or safety certification;
- no learned fallback, takeover, or recovery policy;
- the braking-invariant repair is validated only on frozen responses and does
  not yet run in the task-level online evaluator;
- RTC success comparison is underpowered for small effects and currently
  supports no superiority claim.

## Recommended next milestones

1. Wire the independently scheduled runtime and braking repair into an
   end-to-end evaluator, recording observation, action, and commitment age at
   each control tick.
2. Replace resolution-bounded edge sampling with a validated continuous or
   conservative swept-volume collision check and report dynamics limits.
3. Reproduce the runtime method on a second open VLA family without forcing
   incompatible action semantics into the existing adapter.
4. Evaluate calibrated abstention or takeover under registered faults.
5. Add a hardware adapter only after watchdog, calibration, and emergency-stop
   requirements are specified and tested.

## Maintainer readiness

A maintainer should be able to:

- explain both runtime schedulers without conflating them;
- reconstruct the primary paired statistics from source CSV files;
- distinguish policy noise pairing from environment-state pairing;
- identify the first failed boundary in an observation-to-physics trace;
- add a new run ID without mutating preserved evidence;
- state the checkpoint, simulator, and hardware limits of every reported
  result.
