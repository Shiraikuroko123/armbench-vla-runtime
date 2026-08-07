"""Runtime composition and state mapping for the Menagerie Panda model."""

from __future__ import annotations

import os
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
VLA_EXTERNAL_CAMERA = "armbench_vla_external"
VLA_WRIST_CAMERA = "armbench_vla_wrist"
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
PANDA_SCENE_ENV = "ARMBENCH_PANDA_SCENE"
MENAGERIE_ROOT_ENV = "ARMBENCH_MENAGERIE_ROOT"


def panda_scene_candidates() -> tuple[Path, ...]:
    """Return portable, ordered locations for the pinned Panda scene."""

    project_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    scene_override = os.environ.get(PANDA_SCENE_ENV)
    if scene_override:
        candidates.append(Path(scene_override))
    root_override = os.environ.get(MENAGERIE_ROOT_ENV)
    if root_override:
        root = Path(root_override)
        candidates.append(root / "franka_emika_panda" / "scene.xml")
        candidates.append(root / "scene.xml")
    candidates.extend(
        (
            project_root
            / ".cache"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml",
            project_root.parent
            / "upstream"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml",
            Path.cwd()
            / ".cache"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml",
            Path.cwd()
            / "upstream"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml",
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return tuple(unique)


def default_panda_scene_path() -> Path:
    candidates = panda_scene_candidates()
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "MuJoCo Menagerie Panda scene not found. Run scripts/setup_local.ps1 "
        f"or set {PANDA_SCENE_ENV}. Searched:\n  - {searched}"
    )


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
        self.hand_body_id = int(self.body_ids[-1])
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
        vla_cameras: bool = False,
        goal_marker: Sequence[float] | None = None,
    ) -> "MuJoCoPanda":
        if payload_mass < 0.0:
            raise ValueError("payload mass cannot be negative")
        if goal_marker is not None:
            goal_position = np.asarray(goal_marker, dtype=float)
            if goal_position.shape != (3,) or not np.all(np.isfinite(goal_position)):
                raise ValueError("goal_marker must be a finite 3-D position")
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
        if vla_cameras:
            target = spec.worldbody.add_body(
                name="armbench_vla_camera_target", pos=[0.35, 0.05, 0.55]
            )
            del target
            external_camera = spec.worldbody.add_camera(
                name=VLA_EXTERNAL_CAMERA,
                pos=[1.05, -1.0, 1.15],
                fovy=50.0,
            )
            external_camera.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
            external_camera.targetbody = "armbench_vla_camera_target"
            hand = spec.body("hand")
            if hand is None:
                raise ValueError("Menagerie Panda hand body not found")
            hand.add_camera(
                name=VLA_WRIST_CAMERA,
                pos=[-0.06, -0.04, -0.04],
                # Look between the obstacle corridor and goal in hand coordinates.
                quat=[0.66961113, -0.63053439, -0.26906847, 0.28574372],
                fovy=100.0,
            )
        if goal_marker is not None:
            spec.worldbody.add_geom(
                name="armbench_visual_goal",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=goal_position,
                size=[0.035, 0.0, 0.0],
                rgba=[0.12, 0.78, 0.22, 1.0],
                contype=0,
                conaffinity=0,
            )
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

    def hand_position(self, q: ArrayLike) -> FloatArray:
        position, _ = self.hand_pose(q)
        return position

    def hand_pose(self, q: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return the Panda hand-body position and rotation in the world frame."""

        self.set_configuration(self._fk_data, q)
        position = self._fk_data.xpos[self.hand_body_id].copy()
        rotation = self._fk_data.xmat[self.hand_body_id].reshape(3, 3).copy()
        return position, rotation

    def hand_jacobian(self, q: ArrayLike) -> FloatArray:
        """Return a world-frame 6x7 geometric Jacobian for the hand body."""

        self.set_configuration(self._fk_data, q)
        jacobian_position = np.zeros((3, self.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacBody(
            self.model,
            self._fk_data,
            jacobian_position,
            jacobian_rotation,
            self.hand_body_id,
        )
        return np.vstack(
            (
                jacobian_position[:, self.arm_dof_addresses],
                jacobian_rotation[:, self.arm_dof_addresses],
            )
        )

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
