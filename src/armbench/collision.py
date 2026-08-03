"""Configuration and resolution-bounded edge collision validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from armbench.geometry import Sphere, segment_intersects_sphere
from armbench.model import RobotModel


@dataclass
class CollisionStats:
    configuration_queries: int = 0
    edge_queries: int = 0
    link_obstacle_tests: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class CollisionChecker:
    """Validate arm states against joint limits and spherical obstacles.

    A configuration test checks each finite link segment exactly against each
    sphere. An edge test linearly interpolates joint space at no more than
    ``resolution`` radians in any joint. It is resolution-bounded validation,
    not an analytic swept-volume proof between samples.
    """

    def __init__(
        self,
        robot: RobotModel,
        obstacles: Sequence[Sphere] = (),
        *,
        link_radius: float = 0.035,
        safety_margin: float = 0.01,
        resolution: float = 0.05,
    ) -> None:
        if link_radius < 0.0 or safety_margin < 0.0:
            raise ValueError("link radius and safety margin cannot be negative")
        if resolution <= 0.0:
            raise ValueError("collision resolution must be positive")
        self.robot = robot
        self.obstacles = tuple(obstacles)
        self.link_radius = float(link_radius)
        self.safety_margin = float(safety_margin)
        self.resolution = float(resolution)
        self.stats = CollisionStats()

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
        points = self.robot.forward_points(configuration)
        for link_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            for obstacle in self.obstacles:
                self.stats.link_obstacle_tests += 1
                if segment_intersects_sphere(
                    start,
                    end,
                    obstacle,
                    link_radius=self.link_radius,
                    safety_margin=self.safety_margin,
                ):
                    return f"collision:{obstacle.label}:link_{link_index}"
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

