[Project website](https://shiraikuroko123.github.io/armbench-vla-runtime/) | [简体中文](README_ZH.md) | [Documentation](docs/README.md)

[![CPU CI](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml)

# ArmBench

ArmBench is a runtime and evaluation platform for action-chunk
vision-language-action (VLA) policies. It studies what happens after a policy
has produced a short future action sequence but before that sequence reaches a
robot controller.

The repository contains two separately validated paths:

- A local 7-DoF MuJoCo Panda execution base for motion planning, constrained
  tracking, protocol validation, and fault handling.
- An official-checkpoint evaluation path for Physical Intelligence's
  `pi0.5` (pi-zero-point-five) VLA model through OpenPI and LIBERO.

They share runtime contracts, telemetry, validation, and artifact tooling. A
content-attested integration gate now connects a live `pi0.5`-LIBERO response
to the asynchronous Panda runtime through an explicit Cartesian action
adapter. This is not yet an official task-aligned Panda evaluation, hardware
deployment, or safety certification.

The G02 pilot additionally evaluates the attested checkpoint in LIBERO with
simulation and blocking inference on distinct processes and clocks. All 40
rollouts completed, 38 succeeded, and every raw request, action chunk, hold,
initial state, failure, and provenance record is retained for CPU-only
validation. The result is exploratory and is not an official leaderboard score.

The registered deadline follow-up keeps the checkpoint and runtime contract
fixed across 18 independently validated cells and 720 rollouts. Spatial seeds
7/8/9 each record 0/40 at 150 ms and 36-40/40 at 155 ms; Object seeds 7/8 each
record 0/40 at 150 ms and 37-39/40 at 175 ms. Raising the deadline beyond those
transition cells does not produce a consistent success improvement. The
pattern is consistent with service latency crossing discrete 20 Hz controller
ticks, not a universal VLA threshold. The balanced report is in
[`reports/pi05_deadline_multisuite_report_720_20260810_001`](reports/pi05_deadline_multisuite_report_720_20260810_001/summary.md).

The naming change reflects an engineering progression, not a replacement of
the original project. The seven-DoF planning and tracking benchmark became the
Panda execution substrate; ArmBench adds the action-chunk timing, validation,
atomic publication, and fallback boundary in front of that substrate.

## The problem

An action-chunk policy generates several future actions from one observation.
Inference takes time. When the response arrives, its early actions can describe
time slots that have already passed. Blindly executing from action index zero
therefore creates a stale-action error.

ArmBench measures observation age, selects an unexpired action suffix, and
uses a bounded hold/refresh path when the response misses its deadline. The
runtime also rejects malformed, non-finite, disconnected, stale, or
state-inconsistent policy responses.

## Architecture

```text
images + language + robot state
              |
              v
Physical Intelligence pi0.5 VLA via OpenPI
              |
              v
          action chunk
              |
              v
temporal supervisor + response validation
              |
              +--> LIBERO closed-loop evaluation
              |
              +--> local Panda runtime validation
                     (planning, OSQP projection, continuous collision,
                      braking invariant, torque execution, fault injection)
```

The Panda and LIBERO paths use different action contracts and report separate
results. See [Architecture and claim boundaries](docs/PROJECT_ARCHITECTURE.md)
for the full design and the current integration gap.

## Validated evidence

| Study | Evidence | Interpretation |
| --- | --- | --- |
| Live `pi0.5`-to-Panda integration gate | Official OpenPI `pi05_libero` checkpoint, 35 accepted responses, mean/P95 policy latency 82.75/89.56 ms, 290/311 control ticks concurrent with inference, 0 registered simulation violations | A content-attested Hx7 response crossed the Panda Hx8 adapter and asynchronous runtime; the probe target was not reached, so this is integration evidence rather than task competence |
| Independent-clock `pi0.5`-LIBERO pilot | LIBERO Spatial tasks 0-9, episodes 0-3: 40/40 completed, 38/40 task successes (95.0%), 4,521 control ticks during inference, 0 deadline-exceeded or failed responses | The official attested checkpoint was evaluated while simulation and blocking inference advanced in separate processes/clocks; this is a simulation pilot, not a leaderboard score or hard-real-time guarantee |
| G03 cross-suite independent-clock extension | LIBERO Object tasks 0-9, episodes 0-3: 40/40 completed, 39/40 task successes (97.5%), 6,161/6,263 ticks during inference, 5,530 execute and 733 hold ticks | Same attested checkpoint and protocol on a second suite; exploratory transfer evidence, not a complete-suite score |
| G04 50 ms deadline stress | LIBERO Spatial tasks 0-9, episodes 0-3: 40/40 completed, 0/40 successes, 0 execute and 8,800 hold ticks, 5,474 deadline exceedances | Fail-closed behavior under an intentionally infeasible deadline; not a model-quality ranking |
| G05 150 ms deadline stress | Same 40-rollout Spatial matrix: 0/40 successes, 2,309 execute and 6,491 hold ticks, 16 deadline exceedances | Intermediate exploratory stress point: actions execute, but the available duty cycle is insufficient for task completion |
| G06 175 ms deadline stress | Same 40-rollout Spatial matrix: 38/40 successes, 3,942 execute and 613 hold ticks, 0 deadline/provider failures | Exploratory transition point; not a universal deadline threshold or hard-real-time claim |
| Registered deadline study | 18 cells / 720 rollouts: Spatial seeds 7/8/9 at 150/155/175/200 ms and Object seeds 7/8 at 150/175/200 ms; all source validators and the combined report pass | Service- and clock-specific simulation evidence; cells remain seed-stratified and do not establish a universal threshold, iid deployment estimate, or hard-real-time guarantee |
| Measured-age temporal alignment | Official `pi0.5`-LIBERO Spatial, 120 matched pairs: 88/120 to 116/120, +23.33 points, exact McNemar `p=1.94e-6` | Training-free, observation-age-based suffix selection improves this frozen simulation matrix |
| Cross-suite validation | Object, Goal, and LIBERO-10: 300 rollouts / 150 pairs, 83/150 to 141/150 | Extends deterministic-delay evidence within the same model family and simulator suite |
| RTC-style continuation | 300 rollouts / 100 matched triplets: 96/100 baseline, 97/100 hard projection, 97/100 RTC guidance | No task-success superiority; motion-seam measurements remain exploratory |
| Braking-invariant Panda repair | 270 paired offline cases: 264/270 to 270/270 registered constraints, all 6 legacy conflicts resolved, 0 regressions | Frozen `pi0.5` responses replayed through the Panda adapter; no task-success or hard-real-time claim |
| Integrated Panda supervisor | 27/27 registered fault outcomes reproduced: 12 accepted plans, 6 verified brakes, 7 holds, 2 unrecoverable stops; no rejected chunk exposed a partial prefix | Atomic CPU reference chain combining OSQP kinematic projection, continuous collision certificates, and a stop invariant; scripted actions, not hard real time |
| Asynchronous assurance publication | 17/17 provider, timing, state, reset, budget, and collision outcomes reproduced: 6 complete plans, 10 holds, 1 unrecoverable stop, 0 partial prefixes | Separate policy/assurance workers and activation-time atomic gate; mock, frozen, and interface fixtures only, not a learned checkpoint |
| Assured Panda task execution | 2/2 MuJoCo targets reached under nominal and 0.5 kg/80 ms conditions; 351/351 motion edges and stop boundaries certified, zero registered contact/limit/saturation events | Offline assurance followed by torque-controlled joint-waypoint execution; not a learned VLA, manipulation, or hardware result |
| Asynchronous Panda closed loop | 27 CPU wall-clock cases with clearance-backed swept obstacle checks: braking invariant was physically safe in 9/9 with 0 abrupt stops and 0 repair-budget misses; legacy recorded 311 abrupt stops and unguarded 289 | Live dual-camera, policy-worker, dispatcher, repair, and torque-control integration using a scripted non-learned policy; not learned-policy efficacy or physical certification |
| Clearance-backed swept audit | 72 seeded MuJoCo edges across three scenes: 0 false-safe decisions against a denser sampled oracle | Conservative static-obstacle audit; self-collision and continuous physical safety remain out of scope |
| Continuous self-collision audit | 72 seeded Panda joint-space edges: 70 with valid endpoints, 0 false-safe decisions, 21 conservative rejections against a 0.002 rad sampled oracle | Fail-closed geometry audit for linear interpolation; not a physical or hard-real-time safety certificate |
| Provider-neutral action contract | OpenVLA-OFT-named CPU fixture: `6x7` to `6x8`, exact observation binding, 5/5 semantic mismatches rejected | Second-model-family ABI evidence only; no OpenVLA-OFT checkpoint was executed |
| LeRobot-style actuator boundary | 5-frame replay: 3 commands executed, stale observation held, latch preserved, explicit reset recovered | API-shaped frame and software-watchdog evidence; separate official-loader result is reported below |
| Official LeRobotDataset round-trip | Pinned `lerobot==0.4.4`, v3.0 `LeRobotDataset`, 3 frames, images/state/action/task/timestamp all round-tripped | Isolated dataset serialization evidence for Panda Hx8 semantics; no policy checkpoint or robot driver |
| MuJoCo dynamics braking audit | 45/45 registered payload, damping, and velocity cases passed inverse-dynamics and continuous-edge checks; max torque ratio 0.424543 | Sampled model feasibility evidence; no hard-real-time or emergency-stop certification |

Full protocols, validators, statistics, and limitations are in
[Results](docs/RESULTS.md).

The checked-in live bundle can be independently validated without a GPU. This
checks preserved checkpoint identity, response provenance, events, clocks,
state traces, and the complete MP4; it does not rerun model inference:

```powershell
& '.\.venv\Scripts\python.exe' -m `
  integrations.openpi.validate_live_panda_smoke `
  '.\evidence\g01_live_panda_smoke_final_001' --json
```

The G02 independent-clock pilot is also fully preserved and can be validated
without a GPU. The core result and the separate visual-success run have their
own manifests and must not be pooled:

```powershell
& '.\.venv\Scripts\python.exe' -m `
  integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_libero_independent_clock_core_40_001\evaluation' --json

& '.\.venv\Scripts\python.exe' -m `
  integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_libero_independent_clock_visual_success_001\evaluation' --json
```

The core artifact records raw requests, action chunks, hold/execute decisions,
process IDs, initial states, failures, videos, and source hashes. See the
[G02 pilot report](docs/G02_INDEPENDENT_CLOCK_PILOT.md) and the
[evidence catalog](docs/EVIDENCE_CATALOG.md) for the exact protocol and claim
boundary.

## Scope

- ArmBench does not train or fine-tune `pi0.5`.
- Official-checkpoint results are simulation-only. The LIBERO studies report
  official task protocols; the Panda G01 artifact is a separate integration
  probe and reports `target_reached=false`.
- The G02 independent-clock result is a 40-rollout LIBERO Spatial pilot. Its
  38/40 success rate is not an official leaderboard score, a method comparison,
  a hard-real-time guarantee, or hardware-safety evidence.
- G03 and G04 are separately registered transfer and deadline-stress artifacts;
  they are not pooled with G02 as one confirmatory estimate.
- G05 is an exploratory 150 ms stress point between the 50 ms and 200 ms
  conditions; it is reported separately and is not a universal deadline claim.
- G06 is a second exploratory transition point at 175 ms; its 38/40 result is
  not pooled with G02 or presented as a threshold certification.
- The temporal studies use blocking inference plus simulator catch-up, not an
  operating-system-level hard-real-time control loop.
- Integrated Panda assurance has synchronous task evidence and an asynchronous
  worker/publication boundary. Its 2/2 reached task actions remain scripted
  RRT-Connect; those results must not be relabeled as `pi0.5` task outcomes.
- No Isaac Lab, ROS2, real Franka Panda, or safety PLC integration is claimed.
- The LeRobot bridge is an in-memory, CPU-only compatibility contract. A
  separate isolated environment validates one three-frame export through the
  pinned official `LeRobotDataset` v3.0 API; this is dataset serialization
  evidence, not a policy, driver, or hardware claim.

## Local CPU quickstart

The local Panda path requires Git and CPython 3.10. It does not require a GPU,
an OpenPI server, or a robot. From a fresh clone on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
& '.\.venv\Scripts\python.exe' -m armbench doctor
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view --scenario narrow_gate
```

On Ubuntu or another supported Linux distribution:

```bash
./scripts/setup_local.sh
./.venv/bin/python -m armbench doctor
./.venv/bin/python -m armbench mujoco-view --scenario narrow_gate
```

The setup script installs the CPU dependencies and checks out the pinned Panda
model under `.cache/`. Add `-WithVla` on PowerShell or `--with-vla` on Linux
only when the lightweight OpenPI client is needed; this does not download a
`pi0.5` checkpoint. See [local setup and support](docs/LOCAL_SETUP.md) for
manual installation, model overrides, and headless environments.

The strongest local CPU acceptance path now recomputes the 27-case integrated
fault matrix, replays the 17-case asynchronous publication matrix, and reruns
planning, assurance, and MuJoCo physics for two saved Panda tasks. The script
resolves its own repository and Python paths. Run it from the repository root,
or pass its absolute path when launching elsewhere:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

The first command is numeric acceptance; the second also replays the recorded
MuJoCo trajectory. See [integrated Panda action assurance](docs/INTEGRATED_PANDA_ASSURANCE.md)
for the method, results, fixed-gripper collision rule, and latency boundary.
The provider-to-assurance threading and atomic activation path is documented in
[the CPU completion guide](docs/CPU_RUNTIME_COMPLETION_ZH.md).

To rerun every saved local CPU artifact with one cross-platform command, use
`scripts/accept_cpu.py`. It validates collision, dynamics, provider, LeRobot,
frozen-response, Panda task, asynchronous, both preserved G02 independent-clock
artifacts, and the evidence catalog. It writes an ignored summary under
`output/cpu_acceptance/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_cpu.ps1'
```

The official LeRobotDataset check is automatically included when an isolated
`lerobot==0.4.4` interpreter is available. Add `--full-tests` before a release
to run the complete CPU test suite as well. See [one-click CPU acceptance](docs/CPU_ACCEPTANCE_ZH.md)
for the environment and claim boundary.

For a bounded local acceptance check on Windows:

```powershell
.\scripts\vla_demo.cmd -CheckOnly
```

The model-free asynchronous harness verifies that a blocking policy call runs
on a separate worker while the control side continues to tick:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke
```

This is component-level scripted evidence, not a new `pi0.5` task-success
result. See [non-blocking runtime harness](docs/ASYNC_RUNTIME.md).

The CPU-only Cartesian adapter smoke maps a scripted LIBERO-style `H x 7`
end-effector chunk through the MuJoCo Panda hand Jacobian and the existing
joint-space guard:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-adapter-smoke
```

This CPU command closes a component-level action-semantics boundary. It does
not run `pi0.5` or establish an end-to-end deployment. See
[LIBERO-to-Panda Cartesian adapter](docs/PANDA_CARTESIAN_ADAPTER.md).

The next CPU-only acceptance path replays hash-verified action chunks that were
previously produced by the attested official `pi0.5` LIBERO checkpoint:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-archive-replay `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_archive_replay --chunks 90
```

The preserved 90-chunk report covers 270 independent Panda lookahead cases.
It found 36 invalid raw paths and six cases where collision avoidance and the
configured acceleration bound could not both be satisfied. This is offline
cross-controller diagnostic evidence: the checkpoint was not rerun, no Panda
closed loop was executed, and no task-success claim is made. See
[frozen pi0.5 response replay](docs/PI05_PANDA_ARCHIVE_REPLAY.md).

The next CPU-only stage compares the legacy per-step guard with a
trajectory-level braking-invariant repair on the same frozen responses:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-braking-repair `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_braking_repair_90_001 `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-braking-repair-validate `
  results\pi05_panda_braking_repair_90_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

The checked-in [braking-repair report](reports/pi05_panda_braking_repair_90_001/summary.md)
shows 270/270 cases satisfying the registered constraints, resolving all six
legacy collision/acceleration conflicts with zero regressions. The report is a
paired offline diagnostic: it does not rerun `pi0.5`, close the Panda feedback
loop, or provide a physical-safety or hard-real-time guarantee. For trajectory
inspection, use the `raw_positions`, `legacy_positions`, and `repair_positions`
arrays with `mujoco-view`; the method and claim boundary are documented in
[deadline-bounded braking-invariant repair](docs/PI05_PANDA_BRAKING_REPAIR.md).

The local asynchronous closed-loop stage connects live dual-camera capture, a
blocking policy worker, observation-age suffix selection, deadline fallback,
braking repair, and torque-controlled Panda physics. Its built-in policy is
scripted and non-learned so the entire runtime can be accepted on a CPU:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-async-run `
  --output-directory results\async_panda_quick `
  --scenario single_block --quick --deadline-ms 400

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-async-validate results\async_panda_quick
```

Each case preserves wall-clock events and measured MuJoCo traces. Add
`--videos` for post-run MP4 rendering, or replay `actual_positions` with
`mujoco-view`. This closes the local runtime/control feedback loop; it does not
execute a learned VLA checkpoint or establish hard-real-time or robot-safety
claims. See [asynchronous Panda closed-loop runtime](docs/ASYNC_PANDA_CLOSED_LOOP.md).

The preserved [27-case v3 report](reports/async_panda_closed_loop_400ms_20mm_v3_001/summary.md)
uses five fixed latencies plus jitter, response loss, payload, and persistent
action-fault conditions. Braking-invariant execution recorded 0 abrupt-stop
violations and 9/9 physically safe traces, with 7.81 ms P95 and 19.01 ms
maximum repair latency. Its static-obstacle edges use the recorded 20 mm
clearance-backed swept subdivision; the separate self-collision audit is
reported below. It reached
the target in only 1/9 conditions, making the safety/progress and local CPU-
throughput limits explicit rather than hiding them behind an aggregate success
claim.

The separate [continuous self-collision audit](reports/mujoco_self_collision_audit_001/summary.md)
rechecks 72 seeded Panda joint-space edges against a 0.002 rad sampled oracle:
70 edges have valid endpoints, false-safe is 0, and 21 edges are conservatively
rejected. Recompute it without a GPU:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
```

For a visual check of the registered intermediate-collision control, open the
same edge in the MuJoCo viewer after validation:

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-view `
  'D:\arm-planning-control-project\project\reports\mujoco_self_collision_audit_001' `
  --stratum known_intermediate --edge-index 0 --speed 0.75 --skip-validation
```

The contact overlay is kinematic visual evidence; the manifest-backed validator
remains the acceptance authority.

The certificate is limited to linear interpolation and compiled MuJoCo
geometry; it is not a physical safety or hard-real-time result.

The provider-neutral CPU audit demonstrates how a second action-chunk model
family reaches the existing runtime without treating every `Hx7` tensor as the
same action space:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-provider-audit-validate `
  reports\provider_contract_audit_001
```

The checked-in fixture is labeled `OpenVLA-OFT` to exercise that provider ABI,
but it is synthetic and does not contain checkpoint output. The exact semantic
gate and claim boundary are documented in
[provider-neutral action contract](docs/PROVIDER_CONTRACT.md).

The final CPU-only boundary maps guarded Panda commands to LeRobot-style
`add_frame` keys, applies an actuator-side command watchdog, and exports a
hash-manifested episode whose decisions are recomputed during offline replay:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-lerobot-replay `
  reports\lerobot_style_watchdog_001
```

This validates the software interface, stale-command hold, latch, and reset
path. It does not execute a learned policy or connect a robot. The separate
official loader round-trip can be checked with:

```powershell
& '.\.venv-lerobot-0.4.4\Scripts\python.exe' -m armbench `
  vla-lerobot-official-validate reports\official_lerobot_roundtrip_001
```

See [LeRobot-style runtime bridge](docs/LEROBOT_RUNTIME_BRIDGE.md) and
[official LeRobotDataset round-trip](docs/OFFICIAL_LEROBOT_ROUNDTRIP.md).

The dynamics feasibility artifact is checked independently with:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  mujoco-dynamics-braking-validate reports\dynamics_braking_audit_001
```

It is sampled MuJoCo evidence, not a physical emergency-stop certificate. See
[Panda dynamics braking audit](docs/DYNAMICS_BRAKING_AUDIT.md).

## Review preserved evidence

These commands validate stored artifacts and rebuild offline dashboards. They
do not rerun model inference and do not require a GPU.

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd
```

Browse the [evidence catalog](docs/EVIDENCE_CATALOG.md) to distinguish primary
results, pilots, integration gates, rejected runs, and scripted runtime checks.
Its [machine-readable form](docs/evidence_catalog.json) links every preserved
artifact to its result, protocol, manifest, raw review files, validator, and
explicit claim boundary.

For local setup, remote OpenPI execution, debugging, and environment support,
use [the documentation index](docs/README.md).

## Repository layout

```text
src/armbench/        local Panda planning, control, MuJoCo, and runtime code
integrations/openpi/ official-checkpoint evaluators, analyses, and patches
tests/               unit and integration tests
scripts/             launch and acceptance commands
docs/                design, operations, protocols, and audits
evidence/            preserved experiment artifacts
reports/             offline dashboards generated from evidence
```

ArmBench is released under the MIT License. Upstream software, models, and
assets retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
