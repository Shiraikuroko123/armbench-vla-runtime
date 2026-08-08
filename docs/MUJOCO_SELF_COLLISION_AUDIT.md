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

## Visual acceptance

After the numeric validator passes, replay the registered mechanism-control edge
with contact markers enabled:

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-view `
  'D:\arm-planning-control-project\project\reports\mujoco_self_collision_audit_001' `
  --stratum known_intermediate --edge-index 0 --speed 0.75
```

The MuJoCo window linearly interpolates `q_start -> q_end`. Contact points show
the observed self-contact interval; the console prints the registered
certificate decision and observed contact pairs when the window closes. Use
`--skip-validation` only after running the validator above. This is a kinematic
visual replay and does not replace the manifest-backed result.
