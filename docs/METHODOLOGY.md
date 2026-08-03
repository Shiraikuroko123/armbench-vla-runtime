# Methodology and claim boundaries

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
acceleration or jerk.

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
delay is deterministic, and no camera/VLA inference is present. Random jitter,
dropped observations, hard compute deadlines, arbitrary workcell meshes,
acceleration/jerk timing, and real-Panda experiments are future work.
