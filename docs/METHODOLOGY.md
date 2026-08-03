# Methodology and claim boundaries

## Geometry

At each sampled configuration, every kinematic link is treated as a capsule:
the distance from each spherical obstacle center to the complete link segment
is compared with obstacle radius + link radius + safety margin. This catches
mid-link collisions that a joint-point approximation misses.

An edge in joint space is checked by linear interpolation. The maximum change
of any joint between adjacent checks is 0.05 rad in the default configuration.
This is a resolution-bounded check, not an analytic continuous swept-volume
proof. Self-collision, attached objects, and environment geometry other than
spheres are not modeled in version 0.1.

## Planner comparison

RRT-Connect and RRT* share the robot model, joint limits, collision checker,
step size, timeout, scenarios, and seeds. The RRT* baseline stops at its first
feasible solution; the experiment therefore compares time to feasibility, not
RRT* asymptotic path quality. Latency percentiles include failed trials because
a timeout is operationally meaningful. Path metrics include successful trials
only. Success uncertainty is reported with a Wilson 95% interval.

## Post-processing

Shortcut smoothing accepts a replacement only after checking the complete new
edge and validates the full path again at the end. Time parameterization uses
the slowest joint in each segment and a 35% speed scale. It enforces velocity
limits but does not explicitly constrain acceleration or jerk.

## Tracking model

The plant is seven decoupled double integrators with viscous damping. Commands
are acceleration-like inputs and are clipped per joint. The benchmark injects
0/40/80 ms observation delay, measurement/process noise, and nominal/1.25x
inertia. This is a controlled algorithmic test, not Panda rigid-body dynamics,
Isaac Lab, MuJoCo, ROS2, or real-robot validation.

Safety counters distinguish sampled joint-limit violations, sampled obstacle
collisions, and invalid transitions between adjacent control steps. The last
counter reuses the same resolution-bounded edge checker as the planner.

## Reproducibility

All stochastic components receive an explicit `numpy.random.Generator`.
Planner seeds and controller noise seeds are recorded separately. Every run
stores the resolved config, Python/dependency/platform metadata, Git state,
raw trial tables, aggregate metrics, successful paths, failures, and figures.
Wall-clock latency is expected to vary across machines even when paths match.
