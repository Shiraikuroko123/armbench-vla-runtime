# LIBERO-to-Panda Cartesian Adapter

Status: Current component-level implementation

## Purpose

This adapter closes one software boundary between ArmBench's two testbeds. It
maps a finite `H x 7` action chunk identified as
`libero.ee_delta_pose_gripper.v1` to the local Panda guard's `H x 8` contract:
seven joint velocities in radians per second plus a normalized gripper command.

It does not establish that the official `pi0.5` checkpoint controls the local
Panda. The current acceptance command uses a scripted Cartesian chunk and no
model inference.

## Mapping

For each raw Cartesian action, the adapter:

1. clips the six motion coordinates to the declared input interval;
2. applies the source-attested `0.05 m` translation and `0.5 rad` rotation
   scales at the LIBERO control period of `0.05 s`;
3. interprets the default command in the base/world frame and converts an
   explicitly configured tool-frame command when needed;
4. evaluates the Menagerie Panda hand-body geometric Jacobian;
5. solves damped least-squares differential inverse kinematics;
6. uniformly scales joint velocity to robot limits and a joint-limit margin;
7. maps LIBERO `-1=open, +1=closed` to the local Panda
   `0=closed, 1=open` convention; and
8. preserves source, observation sequence, inference latency, and receive time
   in the resulting action chunk.

Official archived responses can exceed the nominal input interval. Matching
robosuite, the adapter records those steps and clips them before scaling.

The existing `ActionChunkGuard` then applies cross-step acceleration limiting,
configuration checks, resolution-bounded collision edge checks, deadline and
state-mismatch latches, backtracking, and hold fallback.

## CPU acceptance

After the normal local setup, run:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-adapter-smoke
```

The command creates a deterministic ten-step Cartesian chunk, maps it through
the MuJoCo Panda Jacobian, applies the existing guard, and checks finite output,
positive commanded hand displacement, and a valid guarded path. Its report is
explicitly labeled `scripted_cartesian_adapter_component_only`.

## Claim boundary

- The default action semantics are attested to OpenPI's LIBERO submodule at
  commit `f78abd68`, which creates a 20 Hz robosuite `OSC_POSE` environment,
  and robosuite `1.4.1`. Its controller config clips at `[-1, 1]`, scales
  translation to `+/-0.05 m` and rotation to `+/-0.5 rad`, and applies deltas
  in the base/world frame. Panda gripper source code defines `-1=open` and
  `+1=closed`.
- Damped least squares plus deterministic scaling is not a quadratic-program
  safety filter, and it is not dynamically equivalent to robosuite's
  torque-level operational-space controller.
- The local kinematic control point is the Menagerie `hand` body origin, not
  robosuite's `grip_site`; the report exposes this distinction explicitly.
- Collision checking interpolates joint-space edges at a configured resolution;
  it is not continuous-collision certification.
- The smoke command does not use OpenPI, `pi0.5`, LIBERO task execution, a real
  robot, ROS2, or a safety PLC.

## Completed next boundary

Frozen official LIBERO policy responses now pass through this adapter in a
strictly validated offline replay. See
[frozen pi0.5 response replay](PI05_PANDA_ARCHIVE_REPLAY.md). The remaining
end-to-end milestone is to wrap live responses in an independently scheduled
Panda evaluator with feedback observations and a frozen paired protocol. Until
then, the local result remains a cross-controller diagnostic and must not be
reported as a reproduced LIBERO task score.
