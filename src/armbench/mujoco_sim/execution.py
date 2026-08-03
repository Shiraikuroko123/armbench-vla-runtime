"""Torque-controlled execution of planned joint trajectories in MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.postprocess.time_parameterization import Trajectory

FloatArray = NDArray[np.float64]
DEFAULT_KP = np.array([120.0, 120.0, 100.0, 100.0, 60.0, 40.0, 30.0])
DEFAULT_KD = np.array([22.0, 22.0, 18.0, 18.0, 12.0, 8.0, 6.0])


@dataclass(frozen=True)
class PhysicsExecutionResult:
    feedback_mode: str
    delay_ms: int
    payload_mass: float
    times: FloatArray
    desired_positions: FloatArray
    actual_positions: FloatArray
    applied_torques: FloatArray
    rmse: float
    max_tracking_error: float
    final_goal_error: float
    torque_saturation_count: int
    joint_limit_violation_steps: int
    obstacle_contact_steps: int
    obstacle_contact_events: int
    obstacle_contact_duration_s: float
    max_contact_force_n: float
    max_penetration_m: float
    self_contact_steps: int
    safe_success: bool
    video_path: str | None

    def metrics(self) -> dict[str, object]:
        return {
            "feedback_mode": self.feedback_mode,
            "delay_ms": self.delay_ms,
            "payload_mass": self.payload_mass,
            "rmse": self.rmse,
            "max_tracking_error": self.max_tracking_error,
            "final_goal_error": self.final_goal_error,
            "torque_saturation_count": self.torque_saturation_count,
            "joint_limit_violation_steps": self.joint_limit_violation_steps,
            "obstacle_contact_steps": self.obstacle_contact_steps,
            "obstacle_contact_events": self.obstacle_contact_events,
            "obstacle_contact_duration_s": self.obstacle_contact_duration_s,
            "max_contact_force_n": self.max_contact_force_n,
            "max_penetration_m": self.max_penetration_m,
            "self_contact_steps": self.self_contact_steps,
            "safe_success": self.safe_success,
            "video_path": self.video_path,
        }


def _gain_array(value: float | ArrayLike, dof: int, label: str) -> FloatArray:
    gain = np.broadcast_to(np.asarray(value, dtype=float), (dof,)).copy()
    if np.any(gain < 0.0):
        raise ValueError(f"{label} gains cannot be negative")
    return gain


def execute_trajectory(
    robot: MuJoCoPanda,
    trajectory: Trajectory,
    *,
    delay_ms: int = 0,
    control_dt: float = 0.01,
    kp: float | ArrayLike = DEFAULT_KP,
    kd: float | ArrayLike = DEFAULT_KD,
    warmup_s: float = 0.3,
    hold_s: float = 1.0,
    goal_tolerance: float = 0.05,
    feedback_mode: str = "delayed",
    video_path: Path | None = None,
    video_fps: int = 30,
    render_size: tuple[int, int] = (640, 480),
) -> PhysicsExecutionResult:
    """Execute a trajectory with delayed joint feedback and bias compensation.

    The arm actuators in ``robot`` must have zero gain/bias (create the scene
    with ``torque_control=True``). Commands are applied through
    ``qfrc_applied`` and clipped to the Panda effort limits in the MJCF.
    """

    if trajectory.dof != robot.dof:
        raise ValueError("trajectory width must match the Panda arm")
    if delay_ms < 0 or control_dt <= 0.0 or warmup_s < 0.0 or hold_s < 0.0:
        raise ValueError("invalid delay or timing parameter")
    if feedback_mode not in {"delayed", "velocity_prediction"}:
        raise ValueError("feedback_mode must be 'delayed' or 'velocity_prediction'")
    physics_dt = float(robot.model.opt.timestep)
    control_stride_float = control_dt / physics_dt
    control_stride = int(round(control_stride_float))
    if control_stride <= 0 or not np.isclose(control_stride_float, control_stride):
        raise ValueError("control_dt must be an integer multiple of MuJoCo timestep")
    delay_steps_float = (delay_ms / 1000.0) / control_dt
    delay_steps = int(round(delay_steps_float))
    if not np.isclose(delay_steps_float, delay_steps):
        raise ValueError("delay_ms must be an integer multiple of control_dt")
    proportional = _gain_array(kp, robot.dof, "kp")
    derivative = _gain_array(kd, robot.dof, "kd")

    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, trajectory.positions[0])
    data.ctrl[7] = 255.0
    arm_dofs = robot.arm_dof_addresses
    arm_qpos = robot.arm_qpos_addresses
    total_time = warmup_s + trajectory.duration + hold_s
    total_steps = int(np.ceil(total_time / physics_dt))
    history: list[tuple[FloatArray, FloatArray]] = [
        (data.qpos[arm_qpos].copy(), data.qvel[arm_dofs].copy())
    ]
    times: list[float] = []
    desired_positions: list[FloatArray] = []
    actual_positions: list[FloatArray] = []
    applied_torques: list[FloatArray] = []
    torque_saturation_count = 0
    joint_limit_violation_steps = 0
    obstacle_contact_steps = 0
    obstacle_contact_events = 0
    max_contact_force = 0.0
    max_penetration = 0.0
    self_contact_steps = 0
    previous_obstacle_contact = False

    writer = None
    renderer = None
    camera = None
    next_frame_time = 0.0
    resolved_video: str | None = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = render_size
        renderer = mujoco.Renderer(robot.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.22, 0.08, 0.58]
        camera.distance = 1.55
        camera.azimuth = 135.0
        camera.elevation = -22.0
        writer = imageio.get_writer(
            str(video_path),
            fps=video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        resolved_video = str(video_path.resolve())

    try:
        for physics_step in range(total_steps + 1):
            if physics_step % control_stride == 0:
                trajectory_time = np.clip(data.time - warmup_s, 0.0, trajectory.duration)
                desired_q, desired_dq = trajectory.sample(np.array([trajectory_time]))
                current_q = data.qpos[arm_qpos].copy()
                current_dq = data.qvel[arm_dofs].copy()
                history.append((current_q, current_dq))
                observed_index = max(0, len(history) - 1 - delay_steps)
                observed_q, observed_dq = history[observed_index]
                feedback_q = observed_q
                if feedback_mode == "velocity_prediction" and delay_steps > 0:
                    feedback_q = observed_q + (delay_ms / 1000.0) * observed_dq
                feedback = proportional * (desired_q[0] - feedback_q) + derivative * (
                    desired_dq[0] - observed_dq
                )
                requested = feedback + data.qfrc_bias[arm_dofs]
                applied = np.clip(requested, -robot.force_limits, robot.force_limits)
                torque_saturation_count += int(
                    np.count_nonzero(np.abs(requested - applied) > 1e-10)
                )
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[arm_dofs] = applied
                times.append(float(data.time))
                desired_positions.append(desired_q[0].copy())
                actual_positions.append(current_q)
                applied_torques.append(applied.copy())

            if physics_step == total_steps:
                break
            mujoco.mj_step(robot.model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise RuntimeError("MuJoCo state became non-finite")
            q = data.qpos[arm_qpos]
            if np.any(q < robot.lower_limits - 1e-6) or np.any(
                q > robot.upper_limits + 1e-6
            ):
                joint_limit_violation_steps += 1
            obstacle_contacts = robot.obstacle_contacts(data)
            has_obstacle_contact = bool(obstacle_contacts)
            if has_obstacle_contact:
                obstacle_contact_steps += 1
                if not previous_obstacle_contact:
                    obstacle_contact_events += 1
                for contact_index, _, _, _ in obstacle_contacts:
                    contact = data.contact[contact_index]
                    force = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(robot.model, data, contact_index, force)
                    max_contact_force = max(max_contact_force, abs(float(force[0])))
                    max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
            previous_obstacle_contact = has_obstacle_contact
            if robot.self_contacts(data):
                self_contact_steps += 1

            if writer is not None and renderer is not None and camera is not None:
                while data.time + 1e-12 >= next_frame_time:
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
                    next_frame_time += 1.0 / video_fps
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    times_array = np.asarray(times)
    desired_array = np.asarray(desired_positions)
    actual_array = np.asarray(actual_positions)
    torque_array = np.asarray(applied_torques)
    errors = actual_array - desired_array
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_tracking_error = float(np.max(np.abs(errors)))
    final_goal_error = float(np.max(np.abs(actual_array[-1] - trajectory.positions[-1])))
    safe_success = bool(
        final_goal_error <= goal_tolerance
        and obstacle_contact_steps == 0
        and joint_limit_violation_steps == 0
        and self_contact_steps == 0
    )
    return PhysicsExecutionResult(
        feedback_mode=feedback_mode,
        delay_ms=delay_ms,
        payload_mass=robot.payload_mass,
        times=times_array,
        desired_positions=desired_array,
        actual_positions=actual_array,
        applied_torques=torque_array,
        rmse=rmse,
        max_tracking_error=max_tracking_error,
        final_goal_error=final_goal_error,
        torque_saturation_count=torque_saturation_count,
        joint_limit_violation_steps=joint_limit_violation_steps,
        obstacle_contact_steps=obstacle_contact_steps,
        obstacle_contact_events=obstacle_contact_events,
        obstacle_contact_duration_s=obstacle_contact_steps * physics_dt,
        max_contact_force_n=max_contact_force,
        max_penetration_m=max_penetration,
        self_contact_steps=self_contact_steps,
        safe_success=safe_success,
        video_path=resolved_video,
    )
