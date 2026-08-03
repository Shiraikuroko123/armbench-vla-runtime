"""MuJoCo camera and proprioception adapter for pi0/pi0.5-DROID inputs."""

from __future__ import annotations

import time

import mujoco
import numpy as np

from armbench.mujoco_sim.model import (
    MuJoCoPanda,
    VLA_EXTERNAL_CAMERA,
    VLA_WRIST_CAMERA,
)
from armbench.vla.types import VLAObservation


class MuJoCoDroidObservationBuilder:
    def __init__(self, robot: MuJoCoPanda, *, image_size: int = 224) -> None:
        if image_size != 224:
            raise ValueError("OpenPI DROID observations currently require 224px images")
        for name in (VLA_EXTERNAL_CAMERA, VLA_WRIST_CAMERA):
            if mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(
                    "Panda model does not contain VLA cameras; create it with "
                    "vla_cameras=True"
                )
        self.robot = robot
        self.renderer = mujoco.Renderer(
            robot.model, height=image_size, width=image_size
        )

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "MuJoCoDroidObservationBuilder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _render(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render().copy()

    def capture(
        self,
        data: mujoco.MjData,
        *,
        prompt: str,
        sequence_id: int,
        captured_at_s: float | None = None,
    ) -> VLAObservation:
        capture_started = time.monotonic() if captured_at_s is None else captured_at_s
        joint_position = data.qpos[self.robot.arm_qpos_addresses].copy()
        finger_position = data.qpos[self.robot.finger_qpos_addresses]
        gripper_normalized = np.array(
            [np.clip(float(np.mean(finger_position)) / 0.04, 0.0, 1.0)]
        )
        return VLAObservation(
            exterior_image=self._render(data, VLA_EXTERNAL_CAMERA),
            wrist_image=self._render(data, VLA_WRIST_CAMERA),
            joint_position=joint_position,
            gripper_position=gripper_normalized,
            prompt=prompt,
            sequence_id=sequence_id,
            captured_at_s=capture_started,
        )
