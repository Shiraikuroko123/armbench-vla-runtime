# Verified results

This document is the current numerical record for ArmBench. Each section binds
an outcome to its protocol, source identity, and evidence artifact. Official
checkpoint studies, exploratory sampler studies, local deterministic runtime
tests, and historical planner results are reported separately and are not
pooled.

Model terminology and the separation between the Physical Intelligence
`pi0.5` VLA path and the local seven-DoF Panda path are defined in
[Architecture and claim boundaries](PROJECT_ARCHITECTURE.md).

Fields named `safe` or `physical_safe` refer to registered simulation
predicates defined by the corresponding protocol. They are not physical-safety
certification.

## pi0.5 RTC guidance G0

### Provenance and protocol

- Evidence ID: `pi05_rtc_guidance_g0_001`
- ArmBench evidence commit: `7ed75a9`
- OpenPI upstream / extension:
  `15a9616a00943ada6c20a0f158e3adb39df2ccac` /
  `54592c7148ba69bf52757385502782f80f2285e0`
- RTC reference commit: `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`
- Policy/checkpoint: official `pi05_libero`
- Matrix: one fixed observation and explicit sampling-noise tensor, 20 warm
  queries per measured path

### Outcomes

| Measure | Baseline | RTC guided | Result |
|---|---:|---:|---:|
| Weighted model RMSE | 0.0814669 | 0.0270451 | residual ratio 0.3320 |
| Warm wall P95 | 79.5584 ms | 108.0617 ms | 1.3583x |
| Exact repeatability | yes | yes | explicit noise |
| Zero-weight parity | - | - | bitwise exact |
| Peak JAX bytes in use | - | 6.43 GiB | below 23 GiB gate |

This establishes model-side integration feasibility and the correction
direction. A fixed observation cannot establish closed-loop task efficacy. The
complete report and validator are in
[`evidence/pi05_rtc_guidance_g0_001`](../evidence/pi05_rtc_guidance_g0_001/README.md).

## RTC overlap v2: invalidated comparison

The original 60-rollout three-method artifact remains immutable, but it is not
valid matched-effect evidence. A post-run invariant audit found query-0 action
mismatches in 6/20 triplets. Reusing one LIBERO environment across methods
carried visual state across resets for tasks 3, 8, and 9, despite matching
declared state, prompt, sampling key, and sampling noise. Its success and seam
numbers are excluded from every estimate below. Structural manifest validity
does not repair a broken causal comparison.

The preserved raw artifact and root-cause reproduction are documented in
[`RTC_OVERLAP_PAIRING_AUDIT_20260805.md`](research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md).

## Corrected-v3 pi0.5 RTC overlap held-out primary

### Provenance and protocol

- Evidence IDs: `pi05_rtc_overlap_primary_v3_seed_20260806_001` and
  `pi05_rtc_overlap_primary_v3_seed_20260807_001`
- Corrected evaluator commit:
  `44c358731c5493284b74bb29eefa7d538d0f38dd`
- Frozen corrected protocol commit:
  `509f6f4cbcc9e8b02804edf640e565673d4a3855`
- Policy/checkpoint: official `pi05_libero`, content SHA-256
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Matrix: 10 LIBERO-10 tasks x 5 initial states x 2 sampling seeds x 3
  methods, 300 rollouts and 100 matched triplets
- Pairing gate: fresh environment per rollout; identical query-0 policy input,
  response action, sampling key, and sampling noise within every triplet
- Validation: both v3 raw artifacts valid; deterministic combined analysis and
  analysis-manifest rebuild valid; ten failure videos decoded and bound

### Outcomes

| Method | Success (Wilson 95%) | Motion seam mean / median | Gripper seam mean / median | Scored transitions |
|---|---:|---:|---:|---:|
| Unconditioned overlap | 96/100 (0.960 [0.902,0.984]) | 0.106729 / 0.102259 | 0.053754 / 0.044599 | 5,270 |
| Hard projected overlap | 97/100 (0.970 [0.915,0.990]) | 0.083089 / 0.079802 | 0.055490 / 0.045002 | 5,264 |
| RTC-guided overlap | 97/100 (0.970 [0.915,0.990]) | 0.087204 / 0.084136 | 0.043190 / 0.041612 | 5,323 |

| Contrast vs unconditioned | Success difference (task-block 95%) | Wins/losses/ties | McNemar raw/Holm | Motion seam difference (task-block 95%) |
|---|---:|---:|---:|---:|
| Hard projection | +0.010 [-0.040,+0.060] | 4/3/93 | 1/1 | -0.023640 [-0.028187,-0.019207] |
| RTC guidance | +0.010 [-0.040,+0.080] | 3/2/95 | 1/1 | -0.019524 [-0.023128,-0.016142] |

Neither conditioned method establishes task-success superiority. Motion seam
is an exploratory process metric; its interval is not promoted to a safety,
task-success, or deployment claim. The result covers one pi0.5 checkpoint on
fixed LIBERO-10 simulation tasks and does not establish true concurrent
inference/control, collision safety, cross-policy generality, or real hardware.

The [combined analysis](../evidence/pi05_rtc_overlap_primary_v3_300_001/analysis/summary.md),
[corrected protocol](research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md), and
[offline dashboard](../reports/pi05_rtc_overlap_primary_v3_300_001/index.html)
are the current evidence. Validate them with
`scripts\rtc_primary_acceptance.cmd -NoOpen`.

## pi0.5-LIBERO held-out measured-age confirmation

### Provenance and protocol

- Run ID: `pi05_libero_measured_age_confirmatory_001`
- Frozen protocol/run commit:
  `12070625cd6f46186282317262065d015c8fbe27`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy/checkpoint: official `pi05_libero`
- Matrix: 10 Spatial tasks x 12 held-out initial states x 2 modes
- Evidence: 240/240 rollouts, 240 videos, 4,547 scored policy queries
- Pairing: identical initial state, keyed response jitter, and explicit
  mode-independent `10 x 32` float32 pi0.5 sampling noise
- Validation: root `valid=true`, 271 protected files; evaluation `valid=true`,
  261 protected files; independently recomputed noise and statistics

### Outcomes

| Measure | Async unguarded | Measured-age aligned | Paired result |
|---|---:|---:|---:|
| Success | 88/120 | 116/120 | +23.33 points, pair bootstrap 95% [+15.00,+31.67] |
| Discordance | - | - | 32 aligned wins / 4 async wins / 84 ties |
| Primary test | - | - | exact two-sided McNemar `p=1.941574737e-6` |
| Whole-task bootstrap | - | - | 95% [+10.83,+38.33] |
| Task sign flip | - | - | exhaustive `2^10`, `p=0.015625` |
| Mean policy queries | 22.75 | 15.14 | -7.61, bootstrap 95% [-8.78,-6.44] |

Every leave-one-task-out effect remains positive (+17.59 to +25.93 points).
Condition-order strata are also both positive, but differ in magnitude:
+13.33 points when the baseline ran first and +33.33 points when alignment ran
first. The result supports training-free suffix selection under measured
client-visible age in blocking LIBERO simulation. It is not evidence of true
concurrent control, hard real-time behavior, physical safety, cross-model
generality, RTC-style policy-internal continuation, or a real robot. The full
artifact and paired dashboard are in
[`evidence/pi05_libero_measured_age_confirmatory_001`](../evidence/pi05_libero_measured_age_confirmatory_001/README.md).

## pi0.5-LIBERO cross-suite external validation

### Provenance and protocol

- Run IDs: `pi05_libero_{object,goal,10}_alignment_external_001`
- ArmBench run commit: `92ff977fb830505118f7a522ed4a8d91b3a02965`
- Temporal-alignment implementation commit:
  `cccbe351a1a4523c65d01eff2997580f7ca83649`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy/checkpoint: official `pi05_libero`
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Matrix: 3 suites x 10 tasks x 5 initial states x 2 modes at 200 ms
- Evidence: 300/300 rollouts, 300 videos, zero runtime/infrastructure failures
- Validation: each source `valid=true`, 131 protected files; derived analysis
  `valid=true`, six protected outputs

The external protocol was frozen after the Spatial confirmatory result and
before any registered Object, Goal, or LIBERO-10 rollout. It keeps the
checkpoint, prompt path, five-action horizon, four-step delay, seed, thresholds,
task budget, and analysis family fixed.

### Outcomes

| Suite | Async success | Aligned success | Difference (bootstrap 95%) | Wins / losses / ties | McNemar raw / Holm | Mean queries async / aligned |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO Object | 25/50 | 49/50 | +48 points [+34, +62] | 25 / 1 / 24 | 8.05e-7 / 2.41e-6 | 28.96 / 17.40 |
| LIBERO Goal | 35/50 | 47/50 | +24 points [+10, +38] | 14 / 2 / 34 | 0.00418 / 0.00418 | 23.96 / 14.46 |
| LIBERO-10 | 23/50 | 45/50 | +44 points [+26, +60] | 25 / 3 / 22 | 2.74e-5 / 5.49e-5 | 49.40 / 34.44 |

All three suite-level exact tests reject after Holm correction. The pooled
150-pair row is descriptive only: success is 83/150 versus 141/150, difference
+38.7 points with bootstrap 95% interval [+29.3,+47.3], and mean queries are
34.11 versus 22.10. No pooled p value is computed. Task-level rows are also
descriptive only.

This supports generalization of the training-free dispatcher across three
additional LIBERO task distributions under deterministic 200 ms injected
delay. It is not evidence for measured network jitter, other checkpoints,
real-time guarantees, collision safety, dynamics shift, or a real robot. The
offline dashboard is
[`reports/pi05_cross_suite_external_001/index.html`](../reports/pi05_cross_suite_external_001/index.html).

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

## Frozen pi0.5 response replay on the Panda guard path

### Provenance and protocol

- Derived report ID: `pi05_panda_archive_replay_90_001`
- Source artifact:
  `pi05_rtc_overlap_primary_v3_seed_20260807_001/evaluation`
- Policy/checkpoint provenance: official `pi05_libero`, content SHA-256
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Source validation: root inventory and hashes, scheduler chains, NPZ contract,
  and 7,934/7,934 response-action hashes verified
- Selection: 90 chunks, three from each of 30 LIBERO-10 task/method strata
- Local matrix: three Panda scenes, 270 independent kinematic lookahead cases,
  fresh scenario state and guard per case
- Explicit scope: checkpoint not executed in replay; no Panda closed loop and no
  task-success evaluation

### Outcomes

| Measure | Result |
| --- | ---: |
| Selected chunks containing at least one clipped input | 89/90 |
| Raw Panda lookahead paths invalid | 36/270 |
| Cases with guard intervention | 226/270 |
| Source latency above the 200 ms deadline | 3/270 |
| Guarded paths valid under 0.02 rad edge sampling | 270/270 |
| Cases satisfying every registered guard constraint | 264/270 |
| Collision/acceleration conflict cases | 6/270 |

The six conflict cases required a collision-valid stop or hold whose immediate
velocity change exceeded the configured acceleration bound. They remain failed
guard cases even though the resulting position path passes discrete collision
sampling. This result identifies a concrete empty-feasible-set condition for
future trajectory repair; it does not establish physical safety or policy task
efficacy. Method rows are diagnostics, not a registered comparison.

The [derived report](../reports/pi05_panda_archive_replay_90_001/summary.md) stores
per-case CSV data, aggregates, source and implementation hashes, local runtime
versions, explicit claim flags, and a self-validating file manifest. The full
method and reproduction boundary is documented in
[frozen pi0.5 response replay](PI05_PANDA_ARCHIVE_REPLAY.md).

## Braking-invariant repair on frozen pi0.5 responses

### Provenance and protocol

- Derived report ID: `pi05_panda_braking_repair_90_001`
- Source artifact: `pi05_rtc_overlap_primary_v3_seed_20260807_001/evaluation`
- Source validation: 7,934/7,934 response-action hashes reverified before the
  paired matrix was executed
- Selection: the same 90 frozen chunks and three independent Panda scenarios
  used by the archive replay (270 cases total)
- Baseline: existing greedy per-step guard
- Repair: whole-chunk scale search over `1.0, 0.75, 0.5, 0.25, 0.0`, with
  configuration, velocity, acceleration, edge-collision, and terminal-braking
  checks
- Selection budget: 20 ms measured software budget; not a hard-real-time
  scheduler
- Explicit scope: no policy execution, Panda feedback loop, or task-success
  evaluation

### Outcomes

| Measure | Greedy guard | Braking-invariant repair |
| --- | ---: | ---: |
| All registered constraints satisfied | 264/270 | 270/270 |
| Position path valid under 0.02 rad edge sampling | 270/270 | 270/270 |
| Collision/acceleration conflict cases | 6 | 0 |
| Repair regressions | - | 0 |
| P95 software selection latency | 13.550 ms | 12.777 ms |
| Maximum software selection latency | - | 19.979 ms |
| Selection-budget exceedances | - | 0/270 |
| Terminal braking path valid | - | 270/270 |

The repair resolves the exact six cases in which the legacy guard's collision
response and acceleration bound conflicted. Four cases select a zero-scale
hold, while the remaining cases select a reduced nonzero scale; the complete
distribution is preserved in `summary.json`. No repair regression was observed
in the paired matrix.

This is a registered engineering diagnostic, not a policy-effect comparison:
the source responses are frozen and replayed offline, and the repair does not
optimize task progress. The report and its validator are in
[`reports/pi05_panda_braking_repair_90_001`](../reports/pi05_panda_braking_repair_90_001/summary.md)
and [the method document](PI05_PANDA_BRAKING_REPAIR.md). The result does not
establish task-success improvement, continuous collision safety, physical
safety, or worst-case hard-real-time behavior.

## Asynchronous Panda closed-loop runtime

### Provenance and protocol

- Run ID: `async_panda_closed_loop_400ms_20mm_v3_001`
- Source commit: `fe7d3171ba54642ca335111110105392916394a8`;
  exact runtime implementation hashes are recorded in provenance
- Policy: `scripted_non_learned_async_reference`; no `pi0`/`pi0.5`
  checkpoint was executed
- Execution: 100 Hz best-effort wall-clock control, 15 Hz action period,
  torque-controlled MuJoCo Panda, live 224x224 exterior/wrist images, separate
  latest-only observation and blocking-policy workers
- Dispatcher: observation age measured from state capture, expired-prefix
  removal, 400 ms response deadline, and per-control-tick activation checks
- Matrix: `single_block`, 232 reference actions plus 45 terminal steps; three
  modes crossed with 0/40/80/160/240 ms fixed delay, 80 +/- 25 ms jitter, 10%
  response loss, 0.5 kg payload, and a persistent 2.5 rad/s joint-0 fault
- Geometry: 20 mm planning clearance inherited by runtime checks; static-
  obstacle edges use per-joint workspace-motion subdivision at half the margin,
  while self-collision remains sampled
- Artifact: 27 NPZ traces, 18.43 MB JSONL event log, CSV, summaries,
  provenance, and manifest; 23,827,218 manifest-protected bytes
- Manifest inventory SHA-256:
  `185ae41fd98055f9e4d223750cba1d8e07ffb6689f65a4a7388b47853af0b8ef`

### Outcomes

| Mode | Cases satisfying physical predicate | Target reached | Abrupt-stop violations | Repair-budget misses | P95 / max repair latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unguarded | 8/9 | 6/9 | 289 | 0 | 0 / 0 ms |
| Legacy greedy | 9/9 | 1/9 | 311 | 0 | 9.954 / 22.841 ms |
| Braking invariant | 9/9 | 1/9 | 0 | 0 | 7.805 / 19.014 ms |

The registered physical predicate is zero MuJoCo obstacle/self-contact steps
and zero joint-limit-violation steps. No mode contacted the obstacle in this
matrix. The unguarded persistent-fault case failed because it accumulated
1,722 joint-limit-violation steps; both guarded modes recorded zero. The
braking-invariant mode also kept every evaluated command transition within the
15 rad/s^2 acceleration bound. Its zero repair-budget misses mean no
`BrakingTrajectoryGuard` selection exceeded the 20 ms measured software
budget in this run; this remains a best-effort measurement, not a hard bound.

All three modes exercised measured-age dispatch: 7,616 executed command
switches used a positive action index and zero used index 0. The 160 ms
condition had a cross-mode mean hold rate of 0.925 and 199 deadline rejections;
the 240 ms condition held at every action boundary and recorded 207 deadline
rejections. This exposes the local dual-camera CPU pipeline's throughput limit
under a 400 ms deadline.

The guarded modes reached the target in only 1/9 conditions, compared with 6/9
for unguarded execution. This is a visible safety/progress tradeoff, not a
learned-policy task-success comparison. There is one best-effort wall-clock run
per mode/condition, so scheduling noise is not paired and no statistical
superiority claim is made. The self-validating report is in
[`reports/async_panda_closed_loop_400ms_20mm_v3_001`](../reports/async_panda_closed_loop_400ms_20mm_v3_001/summary.md),
and the implementation/claim boundary is documented in
[asynchronous Panda closed-loop runtime](ASYNC_PANDA_CLOSED_LOOP.md).

## Clearance-backed MuJoCo swept audit

### Provenance and protocol

- Run ID: `mujoco_swept_audit_001`
- Source: implementation hashes recorded in the artifact summary and protected
  by the root manifest
- Matrix: three Panda scenes x 24 edges = 72 seeded edges
- Swept checker: 0.05 rad joint-resolution lower bound plus per-joint workspace
  displacement subdivision at half of the 20 mm static-obstacle clearance
- Comparison oracle: the same inflated MuJoCo geometry sampled at 0.002 rad
- Scope: static-obstacle certificate only; self-collision remains sampled

### Outcomes

| Metric | Result |
| --- | ---: |
| Edges | 72 |
| Swept / dense accepted | 49 / 49 |
| False-safe edges | 0 |
| Conservative rejections | 0 |
| Mean swept samples per edge | 744.7 |
| P95 swept / dense latency | 129.07 / 321.50 ms |

The broad random edges make the cost of a conservative bound visible; they are
not representative of short control actions. The dense oracle remains sampled
and therefore does not convert this audit into an analytic continuous-collision
proof. The self-validating artifact is
[`reports/mujoco_swept_audit_001`](../reports/mujoco_swept_audit_001/summary.json),
and the method boundary is documented in
[clearance-backed swept collision audit](MUJOCO_SWEPT_AUDIT.md).

## Provider-neutral second-family contract audit

- Run ID: `provider_contract_audit_001`
- Provider identity: `openvla_oft_libero_contract_fixture`
- Upstream family/revision declaration: OpenVLA-OFT / `e4287e9`
- Response origin: `synthetic_contract_fixture`
- Raw/adapted shapes: `6x7` LIBERO-style / `6x8` Panda runtime
- Observation binding: exact dual-image, state, prompt, and sequence SHA-256
- Semantic mismatch matrix: 5/5 rejected (frame, period, rotation encoding,
  gripper convention, and controller semantics)
- Root inventory SHA-256:
  `41bd52c99a645ff26c47c1d992e0d5fc9c889fa6e69ca6967388caf9e8673812`

The validator checks both root and nested manifests, provider identity and
claim flags, canonical action/semantic hashes, observation-response binding,
all registered semantic rejections, and deterministic Panda adapter replay.
This demonstrates a second model-family ABI and fail-closed semantic boundary.
It does not execute or capture output from an OpenVLA-OFT checkpoint, measure
GPU latency, or establish cross-model task success. The artifact is
[`reports/provider_contract_audit_001`](../reports/provider_contract_audit_001/summary.json)
and the method is documented in
[provider-neutral action contract](PROVIDER_CONTRACT.md).

## LeRobot-style actuator-boundary replay

- Run ID: `lerobot_style_watchdog_001`
- Frames: 5
- Decisions: 3 execute / 2 hold
- Fault path: stale observation -> latched hold -> explicit reset -> recovery
- Registered Panda action semantics SHA-256:
  `f2133fdd533e6a50ebe18400b70a316b1c674dd0ddd5e788d13f5c39c5873ddd`
- Root inventory SHA-256:
  `7bf9a72ff642cc0386148f3ba5d8f2cdd4d2458fec19a739f5b13b19bf1643c5`

The artifact exposes LeRobot `add_frame`-style in-memory keys for two images,
state, action, and task. Its software watchdog checks action semantics,
sequence and timestamp monotonicity, observation/action deadlines, heartbeat,
fault latching, and reset replay protection. Validation reconstructs every
observation, replays the watchdog state machine, regenerates all frame hashes,
and recomputes the summary even after a manifest has been re-signed.

Neither the official `lerobot` package nor a physical robot was used. This is
not official LeRobotDataset storage validation, a driver integration, a
hardware emergency stop, hard-real-time behavior, or safety certification. The
artifact is
[`reports/lerobot_style_watchdog_001`](../reports/lerobot_style_watchdog_001/summary.json)
and the implementation boundary is documented in
[LeRobot-style runtime bridge](LEROBOT_RUNTIME_BRIDGE.md).

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

All 12/12 episodes satisfied the registered simulation safety and task-success
predicates, with no runtime fallback, deadline, or state-mismatch event in the
zero-latency matrix. Horizon
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

All 6/6 guarded cases satisfied the registered simulation safety predicate.
Guard P95 ranged from 5.36 to 7.96 ms per case; the maximum was 7.959 ms in the
narrow-gate collision fault. These
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
