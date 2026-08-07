# Clearance-backed swept collision audit

Status: current CPU-only collision-validation method and audit.

The Panda runtime constructs its guard scene with a configured static-obstacle
clearance. The swept checker derives an over-approximate workspace displacement
radius for every arm joint from the compiled Menagerie geometry. An edge is
subdivided until the maximum bound for one subedge is at most half the static
clearance, while retaining the configured joint-resolution lower bound.

This is a conservative certificate for static obstacles represented by that
clearance. Self-collision is still checked at the sampled configurations and
is not a continuous certificate. The dense comparison below is a stronger
sampled oracle, not an analytic proof.

## Run

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-swept-audit `
  --output-directory reports\mujoco_swept_audit_001 `
  --samples-per-scenario 24 --clearance-mm 20 `
  --sampled-resolution-rad 0.05 --dense-resolution-rad 0.002

& '.\.venv\Scripts\python.exe' -m armbench mujoco-swept-validate `
  reports\mujoco_swept_audit_001
```

The audit samples the direct start-to-goal edge plus seeded random joint-space
edges in `free_space`, `single_block`, and `narrow_gate`. It records every
edge decision, workspace bound, sample count, latency, and whether the
conservative checker accepted an edge rejected by the dense oracle.

## Preserved result

The 72-edge artifact in
[`reports/mujoco_swept_audit_001`](../reports/mujoco_swept_audit_001/summary.json)
contains zero false-safe edges and zero conservative rejections. The result is
an implementation audit, not evidence of continuous collision safety, hard
real-time behavior, or a physical robot.
