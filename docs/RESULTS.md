# Verified result snapshots

## Primary online VLA runtime result

### Provenance

- Run ID: `vla_online_formal_20260804`
- Source commit: `3e58994d4fd29ed4020c40b5d5ece1e9abed4afd`
- OpenPI config contract: `pi05_droid`, 15x8 actions at about 15 Hz
- Policy source: `scripted_non_learned_reference`
- Actual OpenPI checkpoint inference: `false`
- Python / MuJoCo: 3.10.8 / 3.11.0
- Protocol: 2 scenes x 2 payloads x execution horizons 1/5/15
- Artifact size: 12 episodes, 1,082 chunk records, 48 observation PNGs, 12 NPZ traces

The run environment marked Git dirty only because the pre-existing untracked
`MJMODEL.TXT` was visible to `git status`; no source file differed from commit
`3e58994`. The complete snapshot is in
[`evidence/vla_online_formal_20260804`](../evidence/vla_online_formal_20260804/summary.md).

### Outcomes

| Scene | Payload kg | Horizon | Queries | Task | Safe | Goal error rad | RMSE rad |
|---|---:|---:|---:|---:|---:|---:|---:|
| single block | 0.0 | 1 | 233 | yes | yes | 0.00081 | 0.00078 |
| single block | 0.0 | 5 | 47 | yes | yes | 0.00029 | 0.00167 |
| single block | 0.0 | 15 | 16 | yes | yes | 0.00015 | 0.00213 |
| single block | 0.5 | 1 | 233 | yes | yes | 0.00091 | 0.00088 |
| single block | 0.5 | 5 | 47 | yes | yes | 0.00037 | 0.00179 |
| single block | 0.5 | 15 | 16 | yes | yes | 0.00017 | 0.00221 |
| narrow gate | 0.0 | 1 | 193 | yes | yes | 0.00370 | 0.00089 |
| narrow gate | 0.0 | 5 | 39 | yes | yes | 0.00258 | 0.00196 |
| narrow gate | 0.0 | 15 | 13 | yes | yes | 0.00230 | 0.00277 |
| narrow gate | 0.5 | 1 | 193 | yes | yes | 0.00448 | 0.00097 |
| narrow gate | 0.5 | 5 | 39 | yes | yes | 0.00330 | 0.00202 |
| narrow gate | 0.5 | 15 | 13 | yes | yes | 0.00288 | 0.00283 |

All 12/12 episodes were physically safe and task-successful, with no runtime
fallback, deadline, or state-mismatch event in the zero-latency matrix. Horizon
1 recaptured state and both cameras after every action and had the lowest RMSE;
horizon 15 reduced query count by roughly 14-15x. This is a responsiveness-cost
comparison, not evidence that a learned VLA solved the task.

### Live-video contrast

The optional online recorder was verified from source commit
`f51196e6b6e2c6c3382025df04c17911d41d1803`. Both cases use the same
`single_block`, zero-payload, horizon-15 setup and the explicitly non-learned
reference policy.

| Case | Queries | Termination | Task | Safe | Video |
|---|---:|---|---:|---:|---|
| Nominal | 16 | `goal_reached` | yes | yes | [490 frames, 2.47 MB](../evidence/vla_online_visual_nominal_20260804/videos/single_block__payload_0kg__horizon_15.mp4) |
| 0.08 rad joint-1 jump | 1 | `guard_fallback:state_mismatch` | no | yes | [40 frames, 0.12 MB](../evidence/vla_online_visual_state_jump_20260804/videos/single_block__payload_0kg__horizon_15.mp4) |

The nominal MP4 SHA-256 is
`A315DDEA5A65088EDD796B28C2EAB881005796C1430556597CE845A03EF3310F`;
the state-jump MP4 SHA-256 is
`2B7F033673C1E1F96260253FBF1A156C254226398E0D63E4D4FEC7BEEFC9E662`.
Both decode as nonblank `640x480` H.264 at 30 fps. Frames were captured during
the physics loop, not reconstructed from the NPZ trace.

The nominal environment snapshot lists only the pre-existing untracked
`MJMODEL.TXT`. The state-jump snapshot also lists the nominal evidence directory
because the two immutable runs were generated sequentially before this evidence
commit; neither snapshot reports a modified source file.

### Online latency-jitter contrast

Source commit: `fc9bd4150afea861530f36aca7ed91613d8e67ba`. Both runs use
the same `single_block`, zero-payload, horizon-15 reference-policy setup. The
only changed variable is the fourth entry of the repeating synthetic latency
profile.

| Profile ms | Queries | Simulated wait s | Deadline chunks | Termination | Goal error rad | Task | Safe |
|---|---:|---:|---:|---|---:|---:|---:|
| 0/40/80/160 | 16 | 1.12 | 0 | `goal_reached` | 0.00021 | yes | yes |
| 0/40/80/240 | 4 | 0.36 | 1 | `guard_fallback:deadline` | 1.32538 | no | yes |

The [below-deadline artifact](../evidence/vla_online_jitter_safe_20260804/summary.md)
shows four complete schedule cycles. The
[deadline artifact](../evidence/vla_online_jitter_deadline_20260804/summary.md)
executes three valid chunks, advances physics for the 240 ms fourth wait, then
latches hold. Aggregate SHA-256 values are respectively
`87757576A21C970D6835B0DBF2FD2FE8A5B4FAE644D97B304232B3A143CCE525`
and
`BAFE7C6CFB6F9B085D13C585B0D435E5AF99CADE77DF782D6AB0F736184AB704`.
These are controlled synthetic delays, not measured pi0/pi0.5 inference or an
OS hard-real-time experiment.

The safe-jitter environment snapshot lists only `MJMODEL.TXT`; the deadline
snapshot additionally lists the previously generated safe-jitter directory.
Neither records a modified source file.

### Pre-inference camera-replay contrast

Source commit: `5a14c613221d4842a6f2a50a56c4de4d81a22e55`. Both runs use
the same `single_block`, zero-payload, horizon-15 reference-policy setup and
schema-v5 observation audit. The only changed variable is whether observation
cycle 1 replays both frames from cycle 0.

| Input | Observation cycles | Policy queries | Unique exterior/wrist hashes | Rejects | Termination | Goal error rad | Task | Safe |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Live cameras | 16 | 16 | 16/16 | 0 | `goal_reached` | 0.00015 | yes | yes |
| Both frames replayed at cycle 1 | 2 | 1 | 1/1 | 1 | `runtime_fallback:observation_validation` | 1.24075 | no | yes |

In the [replay artifact](../evidence/vla_online_camera_freeze_20260804/summary.md),
the second `per_chunk.csv` row has the same two SHA-256 values as the first row,
zero mean absolute pixel delta, both `*_frame_replayed` flags, and
`policy_inference_attempted=false`. The
[nominal artifact](../evidence/vla_online_camera_nominal_20260804/summary.md)
has 16 unique hashes per camera and no observation rejection. Aggregate
SHA-256 values are respectively
`3AAFBCF669FC74A0A88AF0E2627DA9428EB953C024AF233984FA217D6766C632`
for replay and
`13F9CEC7CAA1169640C2D8821881F9243C51084131983EE9DF6925ECBC26AF55`
for nominal.

Both environment snapshots list only the pre-existing untracked `MJMODEL.TXT`
and no modified source file. Both MP4 files decode at `640x480`; they were
captured from the same physics loops. This establishes exact replay detection
before policy inference, not general image freshness or semantic correctness.

## OpenPI-contract fault-response result

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
