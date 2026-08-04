# Verified result snapshots

## Primary pi0.5-LIBERO confirmatory result

### Provenance and protocol

- Run ID: `pi05_libero_alignment_core_001`
- ArmBench run commit: `30676d2d3ff43e3df0750e2ad01f94748293cff5`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy/checkpoint: official `pi05_libero`
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Matrix: 10 LIBERO Spatial tasks x 5 initial states x 3 delays x 2 modes
- Evidence: 300/300 rollouts, 300 videos, zero infrastructure failures
- Validation: root `valid=true`, 331 protected files; nested evaluation
  `valid=true`

`latency_aligned` does not retrain the policy. After `d` injected delay steps,
it discards the first `d` returned actions and dispatches the following
five-action suffix. The frozen primary comparison is 200 ms; 0 and 100 ms are
prespecified secondary conditions.

### Outcomes

| Delay | Role | Async success | Aligned success | Aligned - async (bootstrap 95%) | Wins / losses / ties | McNemar raw / Holm | Mean queries async / aligned |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 ms | secondary | 49/50 | 50/50 | +2 points [0, +6] | 1 / 0 / 49 | 1.000 / 1.000 | 22.14 / 21.50 |
| 100 ms | secondary | 41/50 | 48/50 | +14 points [+2, +26] | 9 / 2 / 39 | 0.0654 / 0.1309 | 21.80 / 16.18 |
| 200 ms | primary | 18/50 | 50/50 | +64 points [+50, +76] | 32 / 0 / 18 | 4.66e-10 / 1.40e-9 | 22.96 / 12.82 |

The 200 ms primary result supports the training-free alignment mechanism under
deterministic injected delay. The 100 ms result does not survive Holm
correction. Bootstrap intervals are descriptive; the Holm-adjusted exact
McNemar tests are the confirmatory decisions. This is not measured network
jitter, a hard real-time guarantee, a real-robot result, or a safety
certificate. The complete artifact and analysis are in
[`evidence/pi05_libero_alignment_core_001`](../evidence/pi05_libero_alignment_core_001/README.md).

## Local scripted online VLA runtime result

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

## Exact OpenPI request replay

### Provenance

- Run ID: `vla_openpi_request_replay_20260804`
- Source commit: `f507cf249548cfc52e69626eceb81d8260afa75e`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy source: `scripted_non_learned_loopback`
- Actual OpenPI checkpoint inference: `false`
- Artifact: 2 queries, 4 original 224x224 frames, 2 complete request metadata
  records, 30 action-audit rows, 1 NPZ trace, and 1 MP4

The environment snapshot lists only the pre-existing untracked `MJMODEL.TXT`.
The artifact validator reports `replayable_requests=2`, decodes the MP4, and
checks JSON/CSV/NPZ counts, image hashes, request metadata, and safety fields.
Its aggregate SHA-256 is
`5995F6771A5B4158788A91AE02C326A2A02074FD7F9611A2A2766A2572F025C3`.

### Byte-level round trip

| Query | Sequence | Exterior SHA prefix | Wrist SHA prefix | Payload SHA-256 | Server match |
|---:|---:|---|---|---|---:|
| 0 | 0 | `74be2029` | `4ac8fa3e` | `b67b5dbcb6105695b69e84626d30655d113344e42ece1c1da89f7f3c270f7c52` | yes |
| 1 | 1 | `c047aae8` | `12c7686e` | `f08c32767625fa205c0fdec700b50ed5211da37dd17dc4df16e7ae721edf7a60` | yes |

For each query, `vla-request-inspect` reconstructed the exact five DROID keys
from saved images, seven-joint state, gripper state, and prompt, then used the
official OpenPI serializer. The reconstructed payload SHA equals the SHA of the
raw bytes received by the loopback server before unpacking. The two-query budget
terminates as `query_budget`, so this is request/protocol evidence rather than a
task-completion or learned-policy result.

## OpenPI wire-fault matrix

### Provenance

- Run ID: `vla_openpi_fault_matrix_20260804`
- Source commit: `9a90c2247cfc9faf37d1faf5904b109012912660`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy source: `scripted_non_learned_loopback`
- Actual OpenPI checkpoint inference: `false`
- Protocol: nominal positive control plus wrong shape, nonfinite, disconnect,
  and timeout; one query per matched case
- Artifact size: 5 child episodes, 75 action-audit rows, 10 complete 224x224
  query frames, 5 NPZ traces, and 5 MP4 files

The matrix was generated from committed source; all five environment snapshots
list only the pre-existing untracked `MJMODEL.TXT`. The complete report is
[`evidence/vla_openpi_fault_matrix_20260804`](../evidence/vla_openpi_fault_matrix_20260804/summary.md),
and the structured matrix SHA-256 is
`399D938A864342EC2296F68B0F7B4CCE8E537DFF2C876A8753082C17A318F70E`.

### Outcomes

| Mode | Requests | Valid chunks | Fallbacks | Failure type | Safe | Violation steps | Hash pair |
|---|---:|---:|---:|---|---:|---:|---:|
| none | 1 | 1 | 0 | - | yes | 0 | match |
| malformed shape | 1 | 0 | 1 | `ValueError` | yes | 0 | match |
| nonfinite | 1 | 0 | 1 | `ValueError` | yes | 0 | match |
| disconnect | 1 | 0 | 1 | `ConnectionClosedError` | yes | 0 | match |
| timeout | 1 | 0 | 1 | `TimeoutError` | yes | 0 | match |

All 4/4 deterministic faults produced zero validated remote chunks and one
latched policy-inference fallback. The nominal control produced one validated
chunk and no fallback. Every child schema-v5 artifact passed JSON/CSV/NPZ,
camera-hash, complete-frame, safety-field, and MP4 decoding checks. Aggregate
SHA-256 values are recorded in `matrix.json` for every child.

The experiment fixes execution horizon 1, payload 0 kg, one query, a 100 ms
client timeout, and a 250 ms injected timeout response. The one-query budget
means all cases intentionally stop before task completion. This validates the
tested runtime failure paths and evidence plumbing, not pi0/pi0.5 competence,
real-network failure rates, or certified safety.

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
