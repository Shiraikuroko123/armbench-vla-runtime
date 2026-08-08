# Dynamics-feasible Panda braking

The earlier braking invariant checked joint position, velocity, acceleration,
and collision constraints. Those checks did not establish that the compiled
Panda model could produce the required stopping torques after payload or joint
damping changed. This stage adds a fail-closed inverse-dynamics boundary.

For a registered initial joint velocity, ArmBench constructs a synchronized
constant-deceleration stop. Every sampled state is checked against Panda joint
position, velocity, and acceleration limits. Every adjacent position pair is
then checked by the continuous MuJoCo pair-distance certificate. Finally,
`mujoco.mj_inverse` recomputes the generalized effort and compares it with 80%
of the compiled actuator force range. A missing or ambiguous actuator mapping,
nonfinite dynamics result, collision, limit violation, or excessive effort
rejects the complete stop before execution.

The checked-in audit covers 45 deterministic cases:

- payload: 0, 0.5, and 1 kg;
- arm-joint viscous damping scale: 0.5, 1, and 2;
- initial velocity: stationary, low/high forward, and low/high reverse.

All 45 registered stops validated. The largest stop time was 0.20 s, the
largest joint-space stopping distance was 0.159374 rad, and the largest
actuator-limit ratio was 0.424543. The artifact records each case in CSV,
binds every file with a recursive SHA-256 manifest, and reruns every MuJoCo
inverse-dynamics and continuous-collision decision during validation.

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  mujoco-dynamics-braking-validate `
  reports\dynamics_braking_audit_001
```

This is sampled model-based feasibility evidence. It does not demonstrate
closed-loop tracking, bounded jerk, operating-system hard real time, model
accuracy on a physical Panda, or certified emergency stopping.
