"""MuJoCo contact-based configuration and edge validation."""

from __future__ import annotations

from math import ceil
from typing import Iterable

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from armbench.collision import CollisionStats
from armbench.mujoco_sim.model import MuJoCoPanda


class MuJoCoCollisionChecker:
    """Planner-compatible collision checker using compiled mesh contacts."""

    def __init__(self, robot: MuJoCoPanda, *, resolution: float = 0.05) -> None:
        if resolution <= 0.0:
            raise ValueError("collision resolution must be positive")
        self.robot = robot
        self.resolution = float(resolution)
        self.stats = CollisionStats()
        self.data = mujoco.MjData(robot.model)

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
        max_delta = float(np.max(np.abs(end - start)))
        sample_count = max(1, int(ceil(max_delta / self.resolution)))
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
