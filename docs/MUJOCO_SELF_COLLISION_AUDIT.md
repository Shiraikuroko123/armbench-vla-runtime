# Panda continuous self-collision audit

Status: current CPU-only geometry audit.

The local Panda checker now evaluates self-collision pairs with a conservative
distance-bound certificate over a linear joint-space edge. This audit compares
that decision with a denser sampled MuJoCo oracle on the same compiled
Menagerie geometry. The matrix contains a fixed intermediate-collision control
and seeded local/global edges; endpoint validity is recorded separately from
edge validity.

## Run

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-audit `
  --output-directory reports\mujoco_self_collision_audit_001 `
  --samples-per-stratum 24 --dense-resolution-rad 0.002

& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
```

The output is self-validating. `per_edge.csv` preserves the endpoints,
continuous certificate status and witness pair, dense-oracle decision, and
both measured latencies. `manifest.json` protects every generated file and the
summary records the Panda scene and implementation hashes.

## Preserved result

The preserved 72-edge artifact in
[`reports/mujoco_self_collision_audit_001`](../reports/mujoco_self_collision_audit_001/summary.json)
contains 70 edges with collision-free endpoints, zero false-safe decisions,
and 21 conservative rejections. The continuous checker is therefore fail
closed relative to this sampled oracle, at the cost of rejecting some edges
that the oracle accepts.

This is an audit of linear interpolation and the pinned compiled MuJoCo
geometry. The dense oracle is sampled evidence, not an analytic proof. The
result does not establish physical-robot safety, hard-real-time behavior,
emergency stopping, or dynamics under a real payload.
