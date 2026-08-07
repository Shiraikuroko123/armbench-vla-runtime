"""MuJoCo contact-based configuration and conservative edge validation."""

from __future__ import annotations

from math import ceil
from typing import Iterable

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from armbench.collision import CollisionStats
from armbench.mujoco_sim.model import MuJoCoPanda


class MuJoCoCollisionChecker:
    """Planner-compatible collision checker using compiled mesh contacts.

    ``swept_obstacle_margin_m`` enables a clearance-backed conservative edge
    certificate. Callers must construct ``robot`` with static obstacles
    inflated by at least this margin. The joint-space subdivision then bounds
    the maximum motion of every downstream collision geometry between samples;
    a collision-free sample sequence is sufficient for the uninflated static
    obstacle. Self-collision remains checked at the same samples and is not
    described as a continuous certificate.
    """

    def __init__(
        self,
        robot: MuJoCoPanda,
        *,
        resolution: float = 0.05,
        swept_obstacle_margin_m: float = 0.0,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("collision resolution must be positive")
        if (
            not np.isfinite(swept_obstacle_margin_m)
            or swept_obstacle_margin_m < 0.0
        ):
            raise ValueError(
                "swept obstacle margin must be finite and nonnegative"
            )
        self.robot = robot
        self.resolution = float(resolution)
        self.swept_obstacle_margin_m = float(swept_obstacle_margin_m)
        self.joint_motion_radii_m = self._compute_joint_motion_radii()
        self.stats = CollisionStats()
        self.data = mujoco.MjData(robot.model)

    def _compute_joint_motion_radii(self) -> np.ndarray:
        """Bound point displacement per radian of each Panda joint.

        The bound sums fixed body offsets and geom bounding radii along each
        descendant chain. It intentionally over-approximates the true radius
        so that the resulting swept subdivision is conservative.
        """

        model = self.robot.model
        active_geoms = tuple(
            int(geom_id)
            for geom_id in self.robot.robot_geom_ids
            if int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )

        def is_descendant(body_id: int, ancestor_id: int) -> bool:
            current = int(body_id)
            while current != 0:
                if current == ancestor_id:
                    return True
                current = int(model.body_parentid[current])
            return False

        radii: list[float] = []
        for joint_id in self.robot.arm_joint_ids:
            joint_body = int(model.jnt_bodyid[joint_id])
            joint_offset = float(np.linalg.norm(model.jnt_pos[joint_id]))
            maximum = 0.0
            for geom_id in active_geoms:
                geom_body = int(model.geom_bodyid[geom_id])
                if not is_descendant(geom_body, joint_body):
                    continue
                bound = (
                    joint_offset
                    + float(np.linalg.norm(model.geom_pos[geom_id]))
                    + float(model.geom_rbound[geom_id])
                )
                current = geom_body
                while current != joint_body:
                    bound += float(np.linalg.norm(model.body_pos[current]))
                    current = int(model.body_parentid[current])
                maximum = max(maximum, bound)
            radii.append(maximum)
        result = np.asarray(radii, dtype=float)
        if result.shape != (self.robot.dof,) or not np.all(np.isfinite(result)):
            raise RuntimeError("failed to derive finite Panda motion bounds")
        return result

    def edge_workspace_motion_bound(
        self, q_start: ArrayLike, q_end: ArrayLike
    ) -> float:
        """Return a conservative workspace displacement bound for an edge."""

        start = self.robot.validate_configuration(q_start)
        end = self.robot.validate_configuration(q_end)
        return float(np.dot(self.joint_motion_radii_m, np.abs(end - start)))

    def edge_sample_count(self, q_start: ArrayLike, q_end: ArrayLike) -> int:
        """Return the resolution and clearance-backed sample count."""

        start = self.robot.validate_configuration(q_start)
        end = self.robot.validate_configuration(q_end)
        max_delta = float(np.max(np.abs(end - start)))
        resolution_samples = max(1, int(ceil(max_delta / self.resolution)))
        if self.swept_obstacle_margin_m <= 0.0:
            return resolution_samples
        workspace_bound = self.edge_workspace_motion_bound(start, end)
        # A half-margin leaves numerical room at either endpoint of a subedge.
        swept_samples = max(
            1,
            int(ceil(workspace_bound / (0.5 * self.swept_obstacle_margin_m))),
        )
        return max(resolution_samples, swept_samples)

    def reset_stats(self) -> None:
        self.stats = CollisionStats()

    def configuration_failure(self, q: ArrayLike) -> str | None:
        self.stats.configuration_queries += 1
        try:
            configuration = self.robot.validate_configuration(q)
        except ValueError:
            return "invalid_configuration"
        if not self.robot.within_limits(configuration):
            return "joint_limit"
        self.robot.set_configuration(self.data, configuration)
        contacts = self.robot.obstacle_contacts(self.data)
        self.stats.link_obstacle_tests += len(self.robot.obstacles) * self.robot.dof
        if contacts:
            _, _, obstacle, body = contacts[0]
            return f"collision:{obstacle}:{body}"
        self_contacts = self.robot.self_contacts(self.data)
        if self_contacts:
            _, first_body, second_body = self_contacts[0]
            return f"self_collision:{first_body}:{second_body}"
        return None

    def configuration_is_valid(self, q: ArrayLike) -> bool:
        return self.configuration_failure(q) is None

    def edge_is_valid(self, q_start: ArrayLike, q_end: ArrayLike) -> bool:
        self.stats.edge_queries += 1
        start = self.robot.validate_configuration(q_start)
        end = self.robot.validate_configuration(q_end)
        sample_count = self.edge_sample_count(start, end)
        if self.swept_obstacle_margin_m > 0.0:
            self.stats.swept_edge_queries += 1
            self.stats.swept_subdivision_samples += sample_count + 1
        for fraction in np.linspace(0.0, 1.0, sample_count + 1):
            if not self.configuration_is_valid(start + fraction * (end - start)):
                return False
        return True

    def path_is_valid(self, path: Iterable[ArrayLike]) -> bool:
        configurations = [self.robot.validate_configuration(q) for q in path]
        if not configurations:
            return False
        if len(configurations) == 1:
            return self.configuration_is_valid(configurations[0])
        return all(
            self.edge_is_valid(start, end)
            for start, end in zip(configurations[:-1], configurations[1:])
        )
