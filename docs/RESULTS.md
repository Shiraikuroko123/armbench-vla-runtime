# Verified result snapshots

## Primary OpenPI-contract VLA runtime result

### Provenance

- Run ID: `vla_guard_formal_20260803`
- Source commit: `b6996edf1ea1c04cd16100857321e95b576b8c06`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- OpenPI config contract: `pi05_droid`, 15x8 actions at about 15 Hz
- Policy source: `scripted_non_learned`
- Actual OpenPI checkpoint inference: `false`
- Python / MuJoCo: 3.10.8 / 3.11.0
- CPU / graphics: Intel i9-12900H / Intel Iris Xe
- Protocol: 2 scenes x 3 fault/timing conditions x guarded/unguarded
- Artifact size: 12 cases, 128 chunks, 1,920 action records, 3 MP4 files

The run environment marked Git dirty only because the pre-existing untracked
`MJMODEL.TXT` was visible to `git status`; no source file differed from commit
`b6996ed`. The complete snapshot is in
[`evidence/vla_guard_formal_20260803`](../evidence/vla_guard_formal_20260803/summary.md).

### Outcomes

| Scene | Condition | Mode | Task | Physical safe | Contacts | Interventions |
|---|---|---|---:|---:|---:|---:|
| single block | safe 0/40/80/160 ms jitter | unguarded | yes | yes | 0 | 0 |
| single block | safe 0/40/80/160 ms jitter | guarded | yes | yes | 0 | 0 |
| single block | direct collision fault | unguarded | no | no | 673 | 0 |
| single block | direct collision fault | guarded | no | yes | 0 | 40 |
| single block | mixed 0/40/80/240 ms jitter | unguarded | yes | yes | 0 | 0 |
| single block | mixed 0/40/80/240 ms jitter | guarded | no | yes | 0 | 195 |
| narrow gate | safe 0/40/80/160 ms jitter | unguarded | yes | yes | 0 | 0 |
| narrow gate | safe 0/40/80/160 ms jitter | guarded | yes | yes | 0 | 0 |
| narrow gate | direct collision fault | unguarded | no | no | 1,641 | 0 |
| narrow gate | direct collision fault | guarded | no | yes | 0 | 40 |
| narrow gate | mixed 0/40/80/240 ms jitter | unguarded | yes | yes | 0 | 0 |
| narrow gate | mixed 0/40/80/240 ms jitter | guarded | no | yes | 0 | 150 |

The guard preserved both positive-control streams without modifying any action.
It reduced 2,314 contact simulation steps from the direct fault streams to zero,
but those guarded episodes intentionally stopped before the goal. The mixed
deadline schedules had seven total 240 ms chunks. The first miss in each guarded
episode latched hold, giving zero contact and task failure rather than an unsafe
resume.

All 6/6 guarded cases were physically safe. Guard P95 ranged from 5.36 to 7.96
ms per case; the maximum was 7.959 ms in the narrow-gate collision fault. These
timings are from the fixed Intel host and include sampled MuJoCo mesh-edge
checks, not remote policy inference.

The experiment validates the runtime contract, fault response, and physics
outcome. It does not establish pi0/pi0.5 task performance, learned-policy
generalization, real-time scheduling, or formal safety.

## Classical MuJoCo planning/control foundation

### Provenance

- Run ID: `mujoco_formal_20260803`
- Code commit: `aa9f185044317a9c56406a95207b4b64ddc0e338`
- Git state at run start: clean
- Menagerie commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- Python / MuJoCo: 3.10.8 / 3.11.0
- CPU: Intel Core i9-12900H
- Renderer verified on Intel Iris Xe without CUDA
- Recorded runtime: approximately 133 seconds
- Planning protocol: 10 paired seeds per scene/clearance/planner
- Execution protocol: 2 scenes x 3 profiles x 3 delays x 2 payloads

The tracked snapshot is in
[`evidence/mujoco_formal_20260803`](../evidence/mujoco_formal_20260803/summary.md).
It contains the resolved config, clean environment record, 80 raw planning
trials, 36 raw execution rows, 1,202 collision-audit samples, aggregate JSON,
and four MP4 recordings. The local untracked result additionally contains 53
successful-path files and 36 controller traces.

### Planning

Latency includes both successes and deadline failures. Path statistics use
successful trials only.

| Scene | Clearance | Planner | Success | P50 ms | P95 ms |
|---|---:|---|---:|---:|---:|
| single block | 0 mm | RRT-Connect | 10/10 | 23.8 | 99.1 |
| single block | 0 mm | RRT* | 5/10 | 1178.9 | 2003.8 |
| single block | 20 mm | RRT-Connect | 10/10 | 30.6 | 123.3 |
| single block | 20 mm | RRT* | 4/10 | 2000.9 | 2004.7 |
| narrow gate | 0 mm | RRT-Connect | 10/10 | 27.5 | 58.0 |
| narrow gate | 0 mm | RRT* | 2/10 | 2000.7 | 2005.3 |
| narrow gate | 20 mm | RRT-Connect | 10/10 | 29.4 | 89.3 |
| narrow gate | 20 mm | RRT* | 2/10 | 2001.7 | 2004.5 |

All 53 successful smoothed paths were revalidated against the physical-radius
MuJoCo mesh model. This scene favors rapid bidirectional connection. RRT* is a
first-solution, deadline-bounded baseline here; this table is not a general
planner ranking.

### Physics matrix

Each profile contains 12 deterministic scene/delay/payload combinations.

| Profile | Clearance | Safe | RMSE range rad | Contact-step range | Total limit steps |
|---|---:|---:|---:|---:|---:|
| nominal fast | 0 mm | 2/12 | 0.0181-0.8425 | 0-363 | 1096 |
| nominal slow | 0 mm | 0/12 | 0.0208-0.0317 | 5-12 | 0 |
| clearance slow | 20 mm | 12/12 | 0.0220-0.0306 | 0 | 0 |

`nominal_slow` and `clearance_slow` use identical speed and gains. The matched
comparison therefore shows that controller tuning alone did not remove the
brief contacts; adding planning clearance did in these fixed scenes.

Selected rows illustrate why multiple safety counters are required:

| Scene/profile | Delay | Payload | RMSE | Contacts | Final error | Outcome |
|---|---:|---:|---:|---:|---:|---|
| single, nominal fast | 0 ms | 0.0 kg | 0.0183 | 3 | 0.0015 | contact |
| single, nominal fast | 80 ms | 0.0 kg | 0.7992 | 0 | 1.9109 | limit/goal failure |
| single, clearance slow | 80 ms | 0.0 kg | 0.0226 | 0 | 0.0113 | safe success |
| gate, nominal fast | 80 ms | 0.5 kg | 0.8425 | 112 | 2.5869 | contact/limit failure |
| gate, clearance slow | 80 ms | 0.5 kg | 0.0230 | 0 | 0.0202 | safe success |

### Collision approximation audit

| Scene | Samples | True collision | True safe | False safe | False collision |
|---|---:|---:|---:|---:|---:|
| single block | 601 | 11 | 582 | 6 | 2 |
| narrow gate | 601 | 20 | 576 | 3 | 2 |

The 9 false-safe samples show that the body-origin capsule skeleton cannot
replace mesh checks even when it uses official MuJoCo kinematics.

## Historical NumPy/DH result

Run `formal_v1_30seed_20260803` at commit `a19883c` contains 180 planning and 60
decoupled-control trials. It remains locally available as an algorithm baseline
and is reproducible with `armbench run`. It is not the primary result because
its DH hand positions differ from the official Panda MJCF by approximately
0.588 m at the start and 0.440 m at the goal, and its plant is not rigid-body
physics. Do not combine its statistics with the MuJoCo tables above.
