import numpy as np
import pytest

from armbench.geometry import (
    Sphere,
    closest_point_on_segment,
    point_to_segment_distance,
    segment_intersects_sphere,
)


def test_sphere_near_segment_midpoint_is_detected() -> None:
    sphere = Sphere(np.array([0.5, 0.09, 0.0]), 0.04)

    assert segment_intersects_sphere(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        sphere,
        link_radius=0.03,
        safety_margin=0.02,
    )


def test_projection_is_clamped_to_segment_not_infinite_line() -> None:
    point = np.array([2.0, 0.1, 0.0])
    closest = closest_point_on_segment(
        point, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    )

    np.testing.assert_allclose(closest, [1.0, 0.0, 0.0])
    assert point_to_segment_distance(
        point, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    ) == pytest.approx(np.hypot(1.0, 0.1))


def test_zero_length_segment_reduces_to_point_distance() -> None:
    endpoint = np.array([1.0, 2.0, 3.0])
    assert point_to_segment_distance([1.0, 2.0, 5.0], endpoint, endpoint) == 2.0


def test_exact_safety_boundary_counts_as_contact() -> None:
    sphere = Sphere(np.array([0.5, 0.2, 0.0]), 0.10)
    assert segment_intersects_sphere(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        sphere,
        link_radius=0.05,
        safety_margin=0.05,
    )


def test_sphere_near_joint_endpoint_is_detected() -> None:
    sphere = Sphere(np.array([1.02, 0.0, 0.0]), 0.03)
    assert segment_intersects_sphere([0, 0, 0], [1, 0, 0], sphere)

