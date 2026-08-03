"""Panda scenarios calibrated against Menagerie collision meshes."""

from __future__ import annotations

import numpy as np

from armbench.geometry import Sphere
from armbench.scenario import Scenario

MUJOCO_SCENARIO_VERSION = "2.0-mujoco"
_START = np.array([0.0, -0.75, 0.0, -2.25, 0.0, 1.55, 0.75])
_GOAL = np.array([1.2, 0.35, -0.8, -1.35, 0.7, 2.25, -0.65])


def mujoco_scenarios() -> dict[str, Scenario]:
    scenarios = (
        Scenario(
            name="free_space",
            start=_START,
            goal=_GOAL,
            obstacles=(),
            description="Menagerie mesh sanity check without added obstacles.",
            version=MUJOCO_SCENARIO_VERSION,
        ),
        Scenario(
            name="single_block",
            start=_START,
            goal=_GOAL,
            obstacles=(
                Sphere(np.array([0.42, 0.12, 0.70]), 0.055, "center_block"),
            ),
            description="A sphere intersects direct interpolation but not either endpoint.",
            version=MUJOCO_SCENARIO_VERSION,
        ),
        Scenario(
            name="narrow_gate",
            start=_START,
            goal=_GOAL,
            obstacles=(
                Sphere(np.array([0.46, -0.08, 0.78]), 0.055, "gate_lower"),
                Sphere(np.array([0.46, 0.08, 0.70]), 0.055, "gate_center"),
                Sphere(np.array([0.38, 0.24, 0.78]), 0.055, "gate_upper"),
            ),
            description=(
                "Three mesh-contact spheres block direct joint interpolation while "
                "keeping both endpoints valid with 20 mm planning clearance."
            ),
            version=MUJOCO_SCENARIO_VERSION,
        ),
    )
    return {scenario.name: scenario for scenario in scenarios}
