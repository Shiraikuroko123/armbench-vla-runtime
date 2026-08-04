# Debugging ArmBench VLA Runtime

Debug this project as five separate boundaries:

```text
MuJoCo observation -> OpenPI request -> action chunk -> supervisor/guard -> physics
```

Do not tune the planner when the camera is wrong, and do not tune the controller
when the server returned a malformed action. Find the first boundary whose
contract is false.

## 1. Recover from the PowerShell path error

The command `..\.venv\Scripts\python.exe` only works when PowerShell is already
inside `D:\arm-planning-control-project\project`. Your earlier terminal was in
`C:\WINDOWS\system32`, so `..` pointed to `C:\WINDOWS`, where no environment
exists.

Run the self-locating check from any directory:

```powershell
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -CheckOnly
```

Or set explicit paths once per terminal:

```powershell
$ArmbenchProject = 'D:\arm-planning-control-project\project'
$ArmbenchPython = 'D:\arm-planning-control-project\.venv\Scripts\python.exe'
Set-Location $ArmbenchProject
& $ArmbenchPython -c `
  "import mujoco, openpi_client, armbench; print(mujoco.__version__)"
```

Expected: MuJoCo `3.11.0` and no import error.

## 2. Run the smallest contract tests

```powershell
& $ArmbenchPython -m pytest tests\test_vla.py -q
```

The tests cover:

1. exact `pi05_droid` request keys and `(15, 8)` response shape;
2. a real local MessagePack/WebSocket round trip through official
   `openpi-client`;
3. exterior/wrist camera shape, dtype, and visible red/green task objects;
4. collision-fault intervention and safe predicted positions;
5. stale-chunk hold and deadline latch/reset behavior;
6. policy/contract failure conversion to a latched runtime hold;
7. a complete artifact with honest `scripted_non_learned` provenance.

If these fail, do not start a formal run.

## 3. Create a cheap end-to-end artifact

```powershell
& $ArmbenchPython -m armbench vla-guard-run --quick `
  --run-id debug_vla_01
```

Use a new run ID on every attempt. Inspect in this order:

1. `run.log`: last completed physical case;
2. `summary.md` and `overview.png`: high-level outcome and camera inputs;
3. `per_case.csv`: task success, physical safety, contacts, guard P95;
4. `per_chunk.csv`: inference age, deadline miss, latch state;
5. `per_action.csv`: exact raw/executed actions, scale, and reason;
6. `<case>.npz`: raw actions and predicted/desired/actual joint arrays.

Expected quick behavior:

| Condition | Unguarded | Guarded |
|---|---|---|
| Safe jitter | completes without contact | unchanged, completes without contact |
| Collision fault | contacts obstacle | zero contact, stops before goal |
| Mixed deadline jitter | ignores deadline | latches hold, zero contact, task incomplete |

Task failure in the last two guarded rows is intentional. A rejected unsafe
command must not be reported as task success.

For the live-state horizon comparison, run:

```powershell
& $ArmbenchPython -m armbench vla-online-run --quick --videos `
  --save-observations --run-id debug_online_01
```

The quick run executes horizons 1 and 15. Verify that `policy_queries` is much
higher for horizon 1, while both rows have `online_physics_feedback=True` and
`camera_recapture_per_query=True`. Each online `per_chunk.csv` row stores the
actual observation joint state and the post-execution state.
The matching `videos/*.mp4` is recorded from the same live MuJoCo state; it is
not a kinematic reconstruction from the saved NPZ.
With `--save-observations`, the NPZ additionally contains
`exterior_images` and `wrist_images` with shape `(queries, 224, 224, 3)` and
`uint8` dtype. Rehash any indexed frame and compare it with the same
`per_chunk.csv` row before debugging policy behavior.
It also stores `observation_gripper_positions` and `prompts`; joint positions and
sequence/action offsets are already query-addressable. To inspect the exact
five-key request without printing image arrays:

```powershell
& $ArmbenchPython -m armbench vla-request-inspect `
  'results\debug_online_01' `
  --scenario single_block --payload 0 --horizon 15 --query 0
```

For a loopback artifact, `server_payload_matches=true` proves the reconstructed
MessagePack bytes hash to the same value observed by the local server. A remote
artifact has no server-side hash unless you preserve one separately.
Use online `per_action.csv` to distinguish the checked 15-action tail from the
prefix that was actually sent to physics. Filter `executed=True`, then inspect
`raw_action`, `guarded_action`, `reason`, `scale`, `q_before`, and `q_after`.

To force a deadline in the online physics loop, add
`--policy-latency-ms 240`. The run should execute one held prefix, report one
deadline chunk, remain physically safe, and fail the task rather than silently
continuing the reference stream.

To exercise jitter after several successful chunks, use
`--policy-latency-schedule-ms 0 40 80 160` for a below-deadline profile or
`--policy-latency-schedule-ms 0 40 80 240` for a fourth-query deadline. The
profile repeats by policy query, not by executed action index. Verify the exact
sequence in `per_chunk.csv` and the resolved profile in `config.json`.

To reproduce a dispatch-state consistency failure, run:

```powershell
& $ArmbenchPython -m armbench vla-online-run --quick `
  --run-id debug_state_jump_01 `
  --state-jump-joint 1 --state-jump-rad 0.08
```

Both rows should report `fault_injections=1`, `state_mismatch_chunks=1`,
`guard_fallback_chunks=1`, `physical_safe=True`, and `task_success=False`.
The matching `per_chunk.csv` row preserves the observed state, injected offset,
dispatch state, measured mismatch, and fallback reason. This deterministic
joint-state jump is a boundary test, not a simulated contact impulse.

To reject a frozen observation before it reaches the policy, run:

```powershell
& $ArmbenchPython -m armbench vla-online-run `
  --scenarios single_block --horizons 15 --payloads 0 `
  --run-id debug_camera_freeze_01 `
  --freeze-camera both --freeze-camera-query 1
```

Expect `observation_cycles=2`, `policy_queries=1`,
`observation_rejection_chunks=1`, and
`termination_reason=runtime_fallback:observation_validation`. The rejected row
must have repeated camera hashes, zero frame deltas, both replay reasons, and
`policy_inference_attempted=False`. This distinction verifies the network policy
was never called with the rejected observation.

Validate the completed schema-v5 artifact before reading individual metrics:

```powershell
& $ArmbenchPython -m armbench vla-artifact-validate `
  evidence\vla_online_camera_nominal_20260804 --decode-videos
```

The command is read-only. It aligns episode keys and row counts across
`aggregate.json`, `per_episode.csv`, `per_chunk.csv`, `per_action.csv`, and each
NPZ trace; recomputes the first/last camera PNG hashes; and optionally decodes
each MP4. When full observations are present, it rehashes every 224x224 frame
and rejects missing cameras, partial query coverage, wrong dtype/shape, or a
frame-count mismatch. Put a breakpoint in
`vla/artifact.py: validate_online_artifact` when a run is complete but one
evidence file disagrees. `valid=true` establishes internal artifact consistency,
not checkpoint identity or certified safety.
`replayable_requests=0` is not a validation failure: it means optional complete
images or request metadata were not recorded, as in older schema-v5 artifacts.

## 4. Debug the observation boundary

Open `observations/<scenario>_external.png` and
`observations/<scenario>_wrist.png`. Both must be 224x224 RGB images showing red
obstacles and the green target. The wrist view may include the gripper edge, but
the task objects must remain visible.

Set breakpoints in:

- `mujoco_sim/model.py: MuJoCoPanda.create` for camera mounts/goal marker;
- `vla/observation.py: MuJoCoDroidObservationBuilder.capture` for render and
  proprioception;
- `vla/types.py: VLAObservation.to_openpi_droid` for final official keys.

At `capture`, inspect image `shape`, `dtype`, `std`, the seven-element
`joint_position`, normalized one-element `gripper_position`, `sequence_id`, and
`captured_at_s`.

## 5. Debug the OpenPI boundary

First run the complete wire path without a GPU:

```powershell
& $ArmbenchPython -m armbench vla-loopback-run `
  --scenario single_block --horizon 5 --max-policy-queries 3 `
  --video --save-observations `
  --output-directory 'results\openpi_loopback_debug_01'
```

Set breakpoints in `vla/loopback.py`, `vla/policy.py`, and `vla/runtime.py`.
Compare `loopback_server.json` request hashes with `per_chunk.csv`. This path
uses the real serializer, socket transport, response validator, guard, and live
physics loop, but its server is deterministic and non-learned.

To debug fail-closed behavior rather than the nominal response, add one of
`--fault-mode malformed_shape`, `nonfinite`, `disconnect`, or `timeout`. For a
bounded timeout reproduction:

```powershell
& $ArmbenchPython -m armbench vla-loopback-run `
  --scenario single_block --horizon 1 --max-policy-queries 1 `
  --fault-mode timeout --fault-delay-ms 250 `
  --inference-timeout-s 0.1 `
  --output-directory 'results\openpi_loopback_timeout_debug_01'
```

Break in `vla/loopback.py: OpenPIProtocolLoopbackServer._handler`, then in
`vla/policy.py: BoundedOpenPIBackend.infer`, and finally in
`vla/runtime.py: VLARuntimeSupervisor.infer_and_guard`. The server audit must
show one injected fault; `per_chunk.csv` must show
`policy_inference_attempted=True`, `validated_policy_response=False`, no raw
action, and `failure_stage=policy_inference`. Inspect `failure_type` and
`failure_message` for the client-visible root cause: the injected modes produce
`ValueError`, `ConnectionClosedError`, or `TimeoutError`. The same values must
appear in the NPZ failure arrays and aggregate type list. The aggregate must
report zero validated chunks, one runtime fallback, task failure, and physical
safety. A timeout delay must be greater than the client inference timeout.

Run every wire fault under matched conditions:

```powershell
& $ArmbenchPython -m armbench vla-loopback-matrix `
  --output-directory 'results\openpi_fault_matrix_debug_01'
```

Open `overview.png`, then compare `matrix.csv` with each child
`loopback_server.json` and `per_chunk.csv`. The command exits nonzero when any
case fails its expected authority, hash, or physical-safety condition. For a
faster breakpoint cycle, add `--fault-modes none nonfinite`; do not use the
reduced subset as the four-fault evidence claim.

Only after the nominal loopback passes, replace it with a real GPU server:

```powershell
& $ArmbenchPython -m armbench vla-probe `
  --host '<GPU_SERVER_IP>' --port 8000 `
  --scenario single_block `
  --output-directory 'results\openpi_probe_debug_01'
```

Set breakpoints in `vla/policy.py: OpenPIPolicyClient.infer`. Verify:

- the request has exactly two images, joint state, gripper state, and prompt;
- `server_metadata` identifies the expected server/config;
- `actions.shape == (15, 8)`;
- `source == "openpi_remote"`;
- `inference_latency_ms` measures the client call;
- action age from observation capture is not greater than the configured
  deadline unless you expect fallback.

An unreachable server raises `ConnectionError`; a stalled WebSocket handshake
or inference raises `TimeoutError`. Both leave no output directory. Set
`--connect-timeout-s` and `--inference-timeout-s` explicitly when diagnosing a
remote server. A successful `probe.json` has
`remote_policy_response_validated: true`; it still has
`checkpoint_identity_verified: false` because the wire protocol cannot prove
which checkpoint the server loaded. Preserve the server launch command and log.

Only after the probe passes, test the bounded closed loop:

```powershell
& $ArmbenchPython -m armbench vla-openpi-run `
  --host '<GPU_SERVER_IP>' --port 8000 `
  --scenario single_block --horizon 1 `
  --max-policy-queries 3 `
  --video `
  --output-directory 'results\openpi_online_debug_01'
```

Inspect `summary.md`, then `per_chunk.csv`. Check
`validated_policy_response`, `client_inference_latency_ms`,
`policy_latency_ms`, `server_timing`, both image SHA-256 values, adjacent-frame
pixel deltas, raw/guarded actions, and
`termination_reason`. `remote_policy_response_validated=true` requires at least
one validated `15x8` reply. If the connection succeeds but inference times out or
returns a malformed chunk, the supervisor advances simulated time for the wait,
executes a latched hold, and writes a failure artifact with the field set to
false. A connection/handshake failure occurs before an output directory exists.
The NPZ trace contains a 16x16 RGB thumbnail from both cameras for every query;
use it to debug observation order without loading full-size frame sequences.

For a deterministic input-side comparison before a closed loop, replay a saved
request:

```powershell
& $ArmbenchPython -m armbench vla-recorded-probe `
  evidence\vla_openpi_request_replay_20260804 `
  --query 1 --host '<GPU_SERVER_IP>' --port 8000 `
  --connect-timeout-s 3 --inference-timeout-s 1 `
  --output-directory 'results\recorded_probe_debug_01'

& $ArmbenchPython -m armbench vla-recorded-probe-validate `
  'results\recorded_probe_debug_01'

& $ArmbenchPython -m armbench vla-recorded-probe-compare `
  'results\recorded_probe_server_a' `
  'results\recorded_probe_server_b' `
  --left-label 'server A' --right-label 'server B' `
  --output-directory 'results\recorded_probe_comparison_01'

& $ArmbenchPython -m armbench vla-recorded-probe-compare-validate `
  'results\recorded_probe_comparison_01'

& $ArmbenchPython -m armbench vla-recorded-probe-batch-compare `
  'results\server_a_probe_cohort' `
  'results\server_b_probe_cohort' `
  --left-label 'server A' --right-label 'server B' `
  --output-directory 'results\probe_batch_comparison_01'
```

Set breakpoints in `vla/request_replay.py`, `vla/policy.py`, and
`vla/replay_probe.py`. Verify `response.json` keeps the source payload SHA,
strict response shape/hash, latency, metadata, and guard result. The NPZ contains
raw and guarded actions plus predicted positions. There is no MuJoCo stepping in
this command, so `guard_safe_after` is a sampled kinematic check and
`physical_safe` must remain null.

The validator is intentionally independent of the remote server. It reloads the
saved arrays, requires finite `(15, 8)` raw and guarded actions and finite
`(16, 7)` predicted positions, recomputes the raw-action SHA-256, and rejects
inconsistent JSON, environment, summary, or claim-boundary fields. A validation
failure means the directory must not be used for a checkpoint comparison.

For the paired comparator, first inspect
`comparison.json.request_payload_sha256`; both input artifacts must contain that
same hash. Then inspect `per_step.csv` for the raw velocity, gripper, and guarded
action differences. `comparison.png` summarizes differences by chunk step and
action dimension. Latency values remain host/network observations and labels do
not attest server checkpoint identity.

The comparison validator reloads `paired_responses.npz`, recomputes both action
hashes and every reported error metric, checks all 15 CSV rows and eight action
dimensions, verifies report-file hashes, and decodes the plot. The response
comparison is therefore locally auditable without reconnecting to either
server. The original replay artifact is still required to reconstruct and view
the request represented by the payload SHA.

For a batch comparison, each cohort root may contain nested probe directories.
The command accepts exactly one validated artifact per request SHA on each side;
duplicates are ambiguous repeated samples and missing hashes break pairing, so
both are explicit errors. Inspect `per_pair.csv`, then open each path in
`child_comparison` and run the single-comparison validator when diagnosing an
outlier. Bootstrap seed and resample count are stored in `batch.json`.

## 6. Debug the runtime guard

Set the main breakpoint at `vla/guard.py: ActionChunkGuard.guard`. For one bad
action inspect:

- `raw_velocity` and `raw_gripper`;
- `raw_failure`;
- `deadline_exceeded`, `state_mismatch_rad`, and `fallback_reason`;
- `previous_velocity`, `max_raw_acceleration`, and `candidate_velocity`;
- each `scale` in `(1.0, 0.75, 0.5, 0.25, 0.0)`;
- `q_before`, `candidate_q`, and `selected_q`;
- `configuration_failure` and `edge_is_valid` in the MuJoCo checker.

Action-level reasons have these meanings:

| Reason | Meaning |
|---|---|
| `accepted` | raw command passed all configured checks |
| `action_bounds_repaired` | value was clipped into the command bounds |
| `slew_rate_repaired` | velocity was limited relative to the previous command |
| `backtracked:<failure>` | velocity was scaled to make the edge valid |
| `deadline` | this chunk crossed the 200 ms threshold |
| `deadline_latched` | an earlier miss keeps the episode in hold |
| `state_mismatch` | execution state differs from the policy observation |
| `state_mismatch_latched` | an earlier state mismatch keeps the episode in hold |
| `guard_disabled` | unguarded comparison; no safety conclusion was computed |

Call `guard.reset(previous_joint_velocity=measured_velocity)` only after explicit
resynchronization, or `guard.reset()` for a stationary new episode. Do not clear
the latch merely because a later server reply is fast.

## 7. Separate predicted safety from physical safety

`executed_kinematic_valid=True` means the sampled reference path passed the
checker. `physical_safe=True` additionally requires zero MuJoCo obstacle/self
contacts and zero joint-limit violation steps during torque execution.

If the predicted path is valid but physics contacts an obstacle:

1. compare `predicted_positions`, `desired_positions`, and `actual_positions`;
2. inspect RMSE and maximum contact force in `per_case.csv`;
3. replay the `.npz` through `mujoco-view`;
4. check command discontinuities around a repaired/held action;
5. increase model fidelity or clearance only after identifying the mechanism.

Do not call the current sampled edge check continuous collision detection.

## 8. Visual replay

Replay a VLA case from its root-level NPZ:

```powershell
& $ArmbenchPython -m armbench mujoco-view `
  --scenario single_block --play `
  --trace 'evidence\vla_guard_formal_20260803\single_block__fresh_collision_fault__guarded.npz' `
  --array actual_positions
```

Compare the protected command:

```powershell
& $ArmbenchPython -m armbench mujoco-view `
  --scenario single_block --play `
  --trace 'evidence\vla_guard_formal_20260803\single_block__fresh_collision_fault__guarded.npz' `
  --array predicted_positions
```

The viewer is a kinematic replay. The recorded MP4 is the physics execution.

## 9. VS Code debugging

Open `D:\arm-planning-control-project\project` as the VS Code workspace. The
tracked `.vscode/launch.json` provides:

- `ArmBench: VLA tests`;
- `ArmBench: VLA quick benchmark`;
- `ArmBench: OpenPI remote probe`;
- `ArmBench: OpenPI local loopback`;
- `ArmBench: OpenPI online closed loop`;
- `ArmBench: MuJoCo trajectory viewer`.

Choose a configuration in **Run and Debug**, set a breakpoint in the files
listed above, and press F5. Enter a new run directory name when prompted.

## Code map

| Question | File / function |
|---|---|
| What is the OpenPI data contract? | `vla/types.py` |
| How is the OpenPI protocol called? | `vla/policy.py: BoundedOpenPIBackend` |
| How can I debug the wire path locally? | `vla/loopback.py` |
| How are MuJoCo observations built? | `vla/observation.py` |
| Why was an observation rejected? | `vla/observation_guard.py` |
| Why was an action changed? | `vla/guard.py: ActionChunkGuard.guard` |
| How are fault matrices executed? | `vla/benchmark.py: execute_vla_guard_benchmark` |
| Where is a real server probed? | `vla/benchmark.py: execute_openpi_probe` |
| Where is a remote closed loop run? | `vla/online_benchmark.py: execute_openpi_online_run` |
| Which geom caused contact? | `mujoco_sim/model.py: obstacle_contacts` |
| How are torques applied? | `mujoco_sim/execution.py: execute_trajectory` |
| Where are CLI commands wired? | `cli.py` |

## Common failures

### `..\.venv\Scripts\python.exe` is not recognized

Your current directory is wrong. Use the absolute `$ArmbenchPython` shown in
section 1 or run `scripts\vla_demo.cmd`.

### OpenPI client import fails

Install the VLA extra with the pinned official client:

```powershell
& $ArmbenchPython -m pip install --editable `
  'D:\arm-planning-control-project\project[test,vla]'
```

### Menagerie model is missing

Verify
`D:\arm-planning-control-project\upstream\mujoco_menagerie\franka_emika_panda\scene.xml`.
Do not copy only `scene.xml`; it references included XML and mesh assets.

### OpenGL or camera test fails

Update the graphics driver, close programs consuming graphics contexts, and run
only `test_mujoco_builder_captures_nonblank_droid_observation`. A portfolio
video should not be claimed until both cameras pass the color/visibility test.

### Result directory already exists

Runs are immutable by design. Choose a new `--run-id` or probe output path; do
not overwrite evidence while debugging.

### Planning latency differs

Seeds preserve sampled sequences, not wall-clock load. Compare status, nodes,
paths, and the environment record before treating latency drift as a regression.
