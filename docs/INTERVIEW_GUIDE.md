# Interview guide

## Ninety-second explanation

ArmBench studies the gap between a collision-free joint-space path and a path
that remains executable under rigid-body dynamics. It uses the pinned official
MuJoCo Menagerie Franka Panda instead of treating a custom DH skeleton as the
robot. RRT-Connect and a first-solution RRT* baseline plan with mesh contacts;
the selected path is shortcut-smoothed, velocity-timed, and tracked by a
torque-limited PD controller with gravity/bias compensation. The execution
matrix injects 0/40/80 ms feedback delay and a 0.5 kg payload, then reports goal
error, torque saturation, contacts, force, penetration, self-collision, and
joint-limit violations. The controlled result is that the same slow controller
has brief contacts in 12/12 cases without clearance and zero contacts in 12/12
cases with 20 mm planning clearance. It is a simulation result, not a real
robot safety guarantee.

## What was built

- A runtime adapter around the official MJCF with explicit name-based Panda
  joint, body, actuator, limit, and effort mapping.
- Contact-backed configuration/edge validation compatible with the independent
  RRT implementations.
- Planning clearance through obstacle inflation, including attached-payload
  geometry during planning.
- Torque execution with delayed observation history, MuJoCo bias compensation,
  effort clipping, and contact-force collection.
- A fixed-seed benchmark that retains failures and writes portable raw data,
  metadata, paths, traces, summaries, and videos.
- A mesh-versus-capsule disagreement audit, automated physics/render tests, and
  interactive replay for recorded failures.

## Questions to be ready for

### Why MuJoCo instead of Isaac Lab?

The experiment needs accurate CPU rigid-body dynamics, official Panda meshes,
contacts, and reproducible low-volume execution. MuJoCo provides that on the
available Intel-only machine. Isaac Lab would add an NVIDIA/CUDA dependency and
is more useful when large parallel GPU environments become part of the study.

### Why does 20 mm clearance help?

Planning checks the reference path, but execution follows it with finite error.
Inflating obstacles makes the planned configuration-space path farther from
contact. In the matched comparison the controller gains and speed are fixed, so
the changed path clearance is the relevant intervention.

### Is collision checking continuous?

No. Configurations use exact MuJoCo mesh contacts, while edges are interpolated
at at most 0.05 rad per joint. It is resolution-bounded and can still miss a
collision between samples. Analytic or conservative continuous collision
checking is future work.

### Why is RRT* slower and less successful?

This implementation uses rewiring and a fixed 2 s deadline, then reports its
first feasible path. These particular constrained scenes favor bidirectional
RRT-Connect. The experiment does not run RRT* to asymptotic convergence and
does not establish universal planner superiority.

### What exactly is delayed?

Only measured joint position/velocity used by feedback. A history buffer
selects 0/4/8 controller ticks behind at 10 ms per tick. The reference command
continues on schedule. This models observation/transport staleness, not random
jitter or variable VLA inference time.

### Why add `qfrc_bias`?

The feedback term should correct tracking error rather than continually fight
gravity and velocity-dependent bias. MuJoCo computes these generalized forces;
adding them is inverse-dynamics bias compensation. The combined request is
still clipped to Panda effort limits.

### Can it control a real Panda?

No. A real deployment needs a `libfranka` or ROS2 control interface, real-time
command timing, watchdogs, emergency-stop integration, collision thresholds,
calibration, and hardware validation. The existing trace is a simulated joint
trajectory, not a certified robot command stream.

### Why audit capsules if mesh checking already exists?

It quantifies the risk of the cheaper model instead of only asserting that it
is approximate. Nine of 1,202 fixed samples were dangerous false-safe labels,
so the project has direct evidence for retaining mesh contacts in the primary
pipeline.

## Honest claim boundaries

Do not say that the project implements continuous collision guarantees, VLA
inference, Isaac Lab, ROS2, real-time scheduling, or real-robot safety. Do not
combine the legacy DH benchmark statistics with the MuJoCo results. Do not call
10 seeds a publication-scale statistical study.

The strongest defensible claim is an end-to-end, reproducible simulation
benchmark with a controlled clearance comparison, realistic Panda dynamics,
failure-preserving data, tests, and visual evidence.

## Before placing it on a resume

Run the full tests, reproduce a quick artifact, open both safe and failed traces
in the viewer, change one scenario safely, and explain the controller equation
without reading the source. If AI-assisted development is discussed, describe
it honestly: tool assistance produced substantial implementation and
documentation, while your defensible ownership comes from understanding,
reproducing, debugging, modifying, and presenting the system rather than
claiming every line was typed unaided.
