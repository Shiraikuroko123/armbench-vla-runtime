# Third-party notices

## PythonRobotics

- Project: https://github.com/AtsushiSakai/PythonRobotics
- Pinned commit: `b38c510e083d69a5755d98d0680bd50f3d9a91fa`
- License: MIT
- Use: reference implementation and cross-check for the seven-joint Panda
  kinematic parameter table and the upstream RRT* teaching baseline.

The planner, collision checker, benchmark harness, post-processing, and
controllers in this repository are independent implementations. The Panda
joint limits and velocity limits are documented by Franka Robotics:
https://frankarobotics.github.io/docs/control_parameters.html

## MuJoCo Menagerie: Franka Emika Panda

- Project: https://github.com/google-deepmind/mujoco_menagerie
- Pinned commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- Model directory: `franka_emika_panda`
- License: Apache-2.0
- Use: authoritative MJCF kinematics, inertial parameters, collision meshes,
  position limits, actuators, and visual assets for physics validation.

The model remains in the separate workspace-level `upstream` checkout and is
loaded at runtime. ArmBench does not modify or redistribute its mesh assets.

## Physical Intelligence OpenPI client

- Project: https://github.com/Physical-Intelligence/openpi
- Pinned commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Package: `packages/openpi-client`
- License: Apache-2.0
- Use: official MessagePack/WebSocket protocol for remote pi0/pi0.5 policy
  inference and the DROID observation/action contract.

The full OpenPI model stack is not embedded in ArmBench. It remains an optional
Ubuntu/NVIDIA policy-server environment; only the lightweight client is used by
the Windows MuJoCo runtime.
