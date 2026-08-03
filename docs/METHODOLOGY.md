# Methodology and claim boundaries

## VLA runtime question

The primary VLA experiment asks a systems question: can a training-free runtime
layer reject stale or kinematically unsafe DROID action chunks while preserving
a known-safe chunk stream? It does not ask whether pi0.5 solves the synthetic
task. The tracked experiment uses deterministic non-learned action sources so
that faults and expected outcomes are controlled.

The policy/runtime boundary is compatible with OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and model config `pi05_droid`:

```text
observation/exterior_image_1_left : uint8[224, 224, 3]
observation/wrist_image_left      : uint8[224, 224, 3]
observation/joint_position        : float[7]
observation/gripper_position      : float[1]
prompt                            : nonempty string
actions                           : float[15, 8]
```

The pinned official client supplies MessagePack NumPy serialization and the
wire contract. ArmBench wraps those bytes in a bounded WebSocket transport
because the upstream client retries refused connections forever and performs an
unbounded inference receive. ArmBench validates the exact input keys and refuses
a response whose horizon, dimension, or finite-value contract differs. Invalid
policy data is converted by the supervisor into a provenance-safe hold before it
can reach the motion guard. Each valid observation and action chunk also carries
a local sequence ID, capture/receive timestamps, policy-source label, client
latency, and optional server timing.

## VLA observation construction

The Panda model is composed with two cameras before MuJoCo compilation. The
exterior camera views the complete workcell. The fixed hand-mounted camera uses
a wide field of view and offset selected from the official `hand` body frame so
that obstacles and target remain visible. Both views are rendered as 224x224
RGB `uint8`. Automated tests require red obstacle and green target pixels in
both views.

The green target is placed at the official MJCF hand position for the scenario
goal and has contact disabled. It provides a visual task referent but cannot
affect collision or physics metrics. The language prompts explicitly identify
the red obstacles and goal. Joint and finger positions come from the same
MuJoCo state used for rendering.

## DROID action semantics

The first seven output dimensions are joint-velocity commands; the eighth is a
gripper-position command. The pinned OpenPI DROID example clips every returned
dimension to `[-1, 1]` before stepping a joint-velocity environment. ArmBench
uses +/-1 rad/s for the seven velocity commands and also caps them by the Panda
joint-specific velocity limits. The gripper value is clipped to `[0, 1]`.

This is not implemented as a normalized fraction multiplied by each hardware
limit. That alternative would incorrectly amplify a DROID command by roughly
the Panda velocity limit.

## Runtime assurance algorithm

For a chunk captured at time `t_obs` and received at `t_recv`, end-to-end age is
`1000 * (t_recv - t_obs)` milliseconds. A chunk older than 200 ms is replaced by
hold. The failure is latched for the remainder of the episode until an explicit
`reset`, because automatically resuming an old open-loop stream after a missed
chunk can be inconsistent with the actual state.

For a fresh finite action, the runtime:

1. rejects a policy/execution joint-state mismatch above 0.05 rad;
2. clips joint velocity and gripper values to the configured bounds;
3. limits velocity change relative to the last executed command;
4. integrates one 15 Hz step from the current predicted joint state;
5. checks Panda joint limits and enabled MuJoCo mesh contacts;
6. checks the complete joint edge at no more than 0.02 rad interpolation spacing;
7. tries velocity scales `1.0`, `0.75`, `0.5`, `0.25`, then `0.0`;
8. commits the first valid candidate and repeats from that predicted state.

The planning/check model inflates each physical obstacle by 20 mm. Execution
uses the physical radius. The output records every scale, reason, raw/executed
action, and before/after joint state. This is sampled runtime validation, not a
control-barrier-function proof or continuous swept-volume guarantee.

## Receding-horizon physics loop

The online benchmark does not precompute a complete guarded trajectory and run
physics afterward. It captures exterior RGB, wrist RGB, joint position, and
gripper position from one live torque-controlled MuJoCo state; requests and
guards a 15x8 chunk; executes only a configured prefix of 1, 5, or 15 actions;
then repeats from the resulting actual `qpos` and `qvel`. The guard's previous
velocity is synchronized to the last action that was actually executed, not the
unexecuted tail of the chunk. Every query preserves the raw and guarded action
chunks, client/end-to-end timing, optional server timing, and termination
reason. A positive `max_policy_queries` budget can bound remote inference cost;
reaching it causes a pose hold and is reported as `query_budget`.
When enabled, online MP4 frames are rendered during these same physics steps
from the exterior camera. They are not reconstructed later from joint traces.
Online artifact schema version 3 also normalizes every checked action into
`per_action.csv`. The `executed` field separates the action prefix applied to
physics from the unexecuted remainder that was only validated; raw/guarded
values, intervention reason/scale, and predicted before/after state remain
query-addressable.

`vla-openpi-run` replaces the reference policy with the bounded transport built
on official OpenPI serialization while retaining the same live physics loop. It
records server metadata and separates attempted remote calls from validated
replies. The artifact marks `remote_policy_response_validated=true` only if at
least one remote reply passes the `15x8` and finite-value contract and is bound
to the current observation record. `checkpoint_identity_verified` remains false
because the OpenPI wire protocol does not attest the loaded checkpoint. A
malformed reply or timeout advances MuJoCo under hold for the measured wait,
then produces a latched runtime fallback rather than a fabricated successful
inference.

The bundled online policy follows a collision-free reference and is labeled
`scripted_non_learned_reference`. It exists to isolate the effect and query cost
of feedback horizon. Its configured latency is synthetic: MuJoCo advances under
a pose-hold controller for that duration, then the response is checked against
the new dispatch state. This supports controlled delay/state-mismatch tests but
is not measured pi0/pi0.5 server latency or OS scheduling evidence.

A latency schedule is indexed by policy query and repeats deterministically.
This matters when comparing horizons: one query always consumes one schedule
entry even though that reply may execute 1, 5, or 15 actions. The complete
resolved profile and per-query realized waits are stored with the run.

The optional online state-jump fault changes one MuJoCo joint after observation
capture and immediately before guard dispatch. It is indexed by policy query
and records the exact injected seven-joint offset. The default 0.08 rad example
crosses the 0.05 rad consistency threshold, so the expected outcome is a
latched hold, physical safety, and task incompletion. Directly changing `qpos`
makes the fault deterministic and attributable; it is not presented as a
physically modeled external impulse.

## VLA fault protocol

The positive-control stream is produced by fixed-seed RRT-Connect, mesh-checked
shortcutting, and velocity timing, then converted into 15 Hz DROID-shaped
chunks. The negative-control stream directly interpolates start to goal through
the obstacle. Both are exposed through the same `ActionChunkPolicy.infer`
interface and labeled `scripted_non_learned`.

Each of two scenarios runs guarded and unguarded under three conditions:

- safe stream with repeating `0/40/80/160 ms` inference jitter;
- direct collision fault at 50 ms;
- safe stream with repeating `0/40/80/240 ms` jitter, where 240 ms crosses the
  deadline and activates the latch.

Task success and physical safety are separate. Physical safety requires zero
obstacle contacts, self-contacts, and joint-limit samples. Task success requires
final maximum joint error no greater than 0.05 rad. A guard that stops a
collision but does not reach the goal is safe and task-incomplete.

## Primary robot model

The primary experiment loads `franka_emika_panda/scene.xml` from MuJoCo
Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. ArmBench maps the
seven arm joints, two finger joints, link bodies, position limits, velocity
limits, and Panda effort limits by name. Obstacles and the optional payload are
composed into an `MjSpec` before compilation; upstream model files are never
modified.

The model is compiled twice for distinct purposes. Planning models retain the
Menagerie position actuators but are used kinematically through `mj_forward`.
Execution models zero the seven arm actuator gain/bias terms and apply joint
torques through `qfrc_applied`. The finger actuator remains available to hold
the gripper closed.

## Scenario protocol

`single_block` contains one 55 mm sphere. `narrow_gate` contains three 55 mm
spheres. Both use the same versioned start and goal configurations. For every
configured clearance, automated validation requires:

- the start to be valid;
- the goal to be valid;
- direct joint interpolation to be blocked.

The robust profile inflates obstacle radii by 20 mm during planning. Execution
always restores the physical 55 mm obstacle radii, so clearance is a planning
constraint rather than a larger execution obstacle.

## Mesh collision checking

At a configuration, MuJoCo computes enabled contacts between the compiled
Panda collision meshes and obstacle geoms. Enabled robot self-contacts are also
treated as invalid. An edge is sampled by joint interpolation so that the
maximum change of any joint between checks is no greater than 0.05 rad.

This is a resolution-bounded discrete check. Although mesh contact is more
faithful than the legacy capsule skeleton, it is not analytic continuous
collision detection and must not be described as a swept-volume guarantee.

## Planner comparison

RRT-Connect and RRT* share the robot model, collision checker, start/goal,
clearance, 0.35 rad step size, 2 s wall-clock deadline, and paired seeds. The
RRT* implementation stops after its first feasible path. The comparison is
therefore time to feasibility under a fixed deadline, not asymptotic path
optimality.

Latency percentiles include failures because a deadline miss is operationally
meaningful. Path statistics include successes only. Success uncertainty uses a
Wilson 95% interval. The formal physics run uses 10 seeds per
scene/clearance/planner condition; that is suitable for a reproducible project
comparison but too small for a broad algorithmic superiority claim.

## Post-processing and timing

Random shortcutting accepts a replacement only after checking the complete new
edge and revalidates the final path. First-order time parameterization limits
each joint velocity using the slowest joint in a segment. It does not constrain
acceleration or jerk. The VLA runtime adds a separate command-space acceleration
limit while validating action chunks; this does not change the planner's
first-order trajectory parameterization.

The `nominal_fast` profile runs at 35% of Panda velocity limits. The matched
`nominal_slow` and `clearance_slow` profiles both use 10%; their only planning
difference is 0 versus 20 mm obstacle inflation.

## Rigid-body execution

MuJoCo advances at 2 ms and the controller updates at 10 ms. At a control tick,
the torque request is

```text
tau = Kp * (q_desired - q_observed)
    + Kd * (dq_desired - dq_observed)
    + qfrc_bias
```

`qfrc_bias` supplies MuJoCo gravity, Coriolis, and passive-force compensation.
The result is clipped independently to the Panda effort limit for every joint.
Observation delay is implemented with a control-rate history buffer; 0/40/80
ms correspond to 0/4/8 control ticks. The command trajectory itself is not
delayed.

The payload condition adds a 0.5 kg box rigidly below the hand. Its geometry is
included during planning and execution, while its mass and inertia affect only
execution dynamics.

## Safety outcome

An execution is a `safe_success` only when all conditions hold:

- maximum final joint error is no greater than 0.05 rad;
- obstacle contact steps equal zero;
- self-contact steps equal zero;
- joint-limit violation steps equal zero.

RMSE alone is intentionally insufficient. The benchmark also records torque
saturation, contact events and duration, maximum normal contact force, maximum
penetration, self-contact, and joint-limit samples. MuJoCo contacts can be
stiff and the reported peak force is simulator/model dependent; it is useful
for comparing these cases, not as a certified physical load.

## Capsule consistency audit

The secondary approximation connects consecutive official MuJoCo body origins
with 50 mm-radius capsules. It uses the official kinematic state, not the
legacy DH model. For each fixed random configuration, its obstacle label is
compared with the exact MuJoCo mesh-contact label. The audit reports both
dangerous false-safe and conservative false-collision outcomes.

## Reproducibility

Every stochastic component receives an explicit NumPy generator. Each run
stores its resolved config, Git state, Python/dependency/platform metadata, raw
planning rows, raw execution rows, all collision samples, aggregate JSON,
successful paths, control traces, videos, posters, and a timestamped log. A run
directory is created with `exist_ok=False` and cannot be silently overwritten.

The VLA artifact additionally stores policy/OpenPI provenance, camera inputs,
per-case metrics, every chunk deadline/latch record, every raw and repaired
action, predicted/desired/actual joint arrays, an overview figure, and selected
MP4 executions. Local artifacts explicitly set
`remote_policy_response_validated=false` and
`checkpoint_identity_verified=false`.
A separate online artifact stores every re-observed joint state and action
offset, the 1/5/15 horizon comparison, first/last camera frames, and live physics
traces.
A successful `vla-probe` writes a separate artifact with that field set to true
only after a remote response passes validation. `vla-openpi-run` applies the
same rule to a query-bounded, receding-horizon rollout and also stores every raw
and guarded remote chunk.

Wall-clock planning latency varies with host load. Fixed seeds preserve sampled
sequences, but a search near the 2 s boundary can change status on a different
machine. Hardware, software, commit, and raw timeout rows must accompany any
latency claim.

## Legacy baseline

Version 0.1 used a standard-DH NumPy model and decoupled joint dynamics. Its
kinematic frame does not match the official Menagerie Panda: measured hand
position differences were approximately 0.588 m at the benchmark start and
0.440 m at the goal. Those results remain useful for testing the independent
planner/control code but are historical algorithm results only. They must not
be cited as Panda mesh geometry, rigid-body simulation, or real-robot evidence.

## Unimplemented scope

There is no real robot adapter, ROS2 node, `libfranka` integration, emergency
stop, safety PLC, or hardware-in-the-loop result. Environments are spherical,
and there is no tracked real pi0/pi0.5 checkpoint rollout. The current jitter is
a deterministic injected schedule; dropped/corrupted frames, OS-level hard
real-time scheduling, arbitrary workcell meshes, jerk constraints,
dynamics-level acceleration guarantees, uncertainty calibration, cross-model
comparison, and real-Panda experiments are future work.
