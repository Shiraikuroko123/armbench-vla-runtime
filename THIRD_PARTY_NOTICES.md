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

