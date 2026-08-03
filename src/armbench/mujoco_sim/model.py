"""Runtime composition and state mapping for the Menagerie Panda model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.geometry import Sphere
from armbench.model import PANDA_VELOCITY

FloatArray = NDArray[np.float64]
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ARM_BODY_NAMES = (
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
)
ARM_FORCE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def default_panda_scene_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    path = (
        project_root.parent
        / "upstream"
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "scene.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Menagerie Panda scene not found. Expected sparse checkout at "
            f"{path}"
        )
    return path


def _safe_name(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_")
    return cleaned or "obstacle"


class MuJoCoPanda:
    """Compiled Panda scene with planner-compatible kinematic methods."""

    name = "franka_panda_mujoco_menagerie"
    dof = 7

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        scene_path: Path,
        obstacles: Sequence[Sphere],
        obstacle_geom_labels: dict[int, str],
        payload_mass: float,
    ) -> None:
        self.model = model
        self.scene_path = scene_path
        self.obstacles = tuple(obstacles)
        self.obstacle_geom_labels = dict(obstacle_geom_labels)
        self.payload_mass = float(payload_mass)
        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self.finger_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINT_NAMES
            ],
            dtype=int,
        )
        if np.any(self.arm_joint_ids < 0) or np.any(self.finger_joint_ids < 0):
            raise ValueError("Menagerie Panda joint mapping is incomplete")
        self.arm_qpos_addresses = model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_addresses = model.jnt_dofadr[self.arm_joint_ids].copy()
        self.finger_qpos_addresses = model.jnt_qposadr[self.finger_joint_ids].copy()
        self.lower_limits = model.jnt_range[self.arm_joint_ids, 0].copy()
        self.upper_limits = model.jnt_range[self.arm_joint_ids, 1].copy()
        self.velocity_limits = PANDA_VELOCITY.copy()
        self.force_limits = ARM_FORCE_LIMITS.copy()
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ARM_BODY_NAMES
            ],
            dtype=int,
        )
        if np.any(self.body_ids < 0):
            raise ValueError("Menagerie Panda body mapping is incomplete")
        self.robot_geom_ids = frozenset(
            int(geom_id)
            for geom_id, body_id in enumerate(model.geom_bodyid)
            if int(body_id) != 0
        )
        self._fk_data = mujoco.MjData(model)

    @classmethod
    def create(
        cls,
        *,
        scene_path: Path | None = None,
        obstacles: Sequence[Sphere] = (),
        payload_mass: float = 0.0,
        torque_control: bool = False,
    ) -> "MuJoCoPanda":
        if payload_mass < 0.0:
            raise ValueError("payload mass cannot be negative")
        resolved = (scene_path or default_panda_scene_path()).resolve()
        spec = mujoco.MjSpec.from_file(str(resolved))
        if spec is None:
            raise ValueError(f"failed to parse MuJoCo scene: {resolved}")
        obstacle_names: list[tuple[str, str]] = []
        for index, obstacle in enumerate(obstacles):
            name = f"armbench_obstacle_{index}_{_safe_name(obstacle.label)}"
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=obstacle.center,
                size=[obstacle.radius, 0.0, 0.0],
                rgba=[0.82, 0.16, 0.12, 1.0],
                contype=1,
                conaffinity=1,
                friction=[0.8, 0.01, 0.001],
            )
            obstacle_names.append((name, obstacle.label))
        if payload_mass > 0.0:
            hand = spec.body("hand")
            if hand is None:
                raise ValueError("Menagerie Panda hand body not found")
            payload = hand.add_body(name="armbench_payload", pos=[0.0, 0.0, 0.13])
            payload.add_geom(
                name="armbench_payload_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.025, 0.025, 0.04],
                mass=payload_mass,
                rgba=[0.12, 0.35, 0.75, 1.0],
                contype=1,
                conaffinity=1,
            )
        model = spec.compile()
        if torque_control:
            model.actuator_gainprm[:7] = 0.0
            model.actuator_biasprm[:7] = 0.0
        labels: dict[int, str] = {}
        for name, label in obstacle_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"compiled obstacle geom missing: {name}")
            labels[geom_id] = label
        return cls(
            model,
            scene_path=resolved,
            obstacles=obstacles,
            obstacle_geom_labels=labels,
            payload_mass=payload_mass,
        )

    def validate_configuration(self, q: ArrayLike) -> FloatArray:
        configuration = np.asarray(q, dtype=float)
        if configuration.shape != (self.dof,):
            raise ValueError(f"configuration must have shape ({self.dof},)")
        if not np.all(np.isfinite(configuration)):
            raise ValueError("configuration must contain only finite values")
        return configuration

    def within_limits(self, q: ArrayLike, *, atol: float = 1e-12) -> bool:
        configuration = self.validate_configuration(q)
        return bool(
            np.all(configuration >= self.lower_limits - atol)
            and np.all(configuration <= self.upper_limits + atol)
        )

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return rng.uniform(self.lower_limits, self.upper_limits)

    def set_configuration(
        self,
        data: mujoco.MjData,
        q: ArrayLike,
        *,
        finger_position: float = 0.04,
        forward: bool = True,
    ) -> None:
        configuration = self.validate_configuration(q)
        data.qpos[self.arm_qpos_addresses] = configuration
        data.qpos[self.finger_qpos_addresses] = finger_position
        data.qvel[:] = 0.0
        if forward:
            mujoco.mj_forward(self.model, data)

    def forward_points(self, q: ArrayLike) -> FloatArray:
        self.set_configuration(self._fk_data, q)
        return self._fk_data.xpos[self.body_ids].copy()

    def body_name_for_geom(self, geom_id: int) -> str:
        body_id = int(self.model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name or f"body_{body_id}"

    def obstacle_contacts(self, data: mujoco.MjData) -> list[tuple[int, int, str, str]]:
        contacts: list[tuple[int, int, str, str]] = []
        obstacle_ids = self.obstacle_geom_labels
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in obstacle_ids and second in self.robot_geom_ids:
                contacts.append(
                    (index, second, obstacle_ids[first], self.body_name_for_geom(second))
                )
            elif second in obstacle_ids and first in self.robot_geom_ids:
                contacts.append(
                    (index, first, obstacle_ids[second], self.body_name_for_geom(first))
                )
        return contacts

    def self_contacts(self, data: mujoco.MjData) -> list[tuple[int, str, str]]:
        contacts: list[tuple[int, str, str]] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.robot_geom_ids and second in self.robot_geom_ids:
                first_body = self.body_name_for_geom(first)
                second_body = self.body_name_for_geom(second)
                if first_body != second_body:
                    contacts.append((index, first_body, second_body))
        return contacts

