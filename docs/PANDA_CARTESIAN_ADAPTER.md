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

For each normalized Cartesian action, the adapter:

1. clips the six motion coordinates to a declared input interval;
2. applies explicit translation and rotation scales;
3. converts tool-frame commands to the MuJoCo world frame when configured;
4. evaluates the Menagerie Panda hand-body geometric Jacobian;
5. solves damped least-squares differential inverse kinematics;
6. uniformly scales joint velocity to robot limits and a joint-limit margin;
7. maps the signed gripper command from `[-1, 1]` to `[0, 1]`; and
8. preserves source, observation sequence, inference latency, and receive time
   in the resulting action chunk.

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

- Default scales are conservative component-test parameters. A formal LIBERO
  comparison must freeze its upstream OSC controller units, frame, clipping,
  and gripper semantics first.
- Damped least squares plus deterministic scaling is not a quadratic-program
  safety filter.
- Collision checking interpolates joint-space edges at a configured resolution;
  it is not continuous-collision certification.
- The smoke command does not use OpenPI, `pi0.5`, LIBERO task execution, a real
  robot, ROS2, or a safety PLC.

## Next end-to-end milestone

Wrap the official LIBERO policy response with this declared adapter contract,
run policy inference on the maintained worker while the simulator advances on
an independent control clock, and compare unguarded, measured-age, and
measured-age-plus-feasibility modes under one frozen paired protocol. Replace
the component scales with values attested from the actual controller before
interpreting task outcomes.
