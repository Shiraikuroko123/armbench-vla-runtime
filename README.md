[English](README.md) | [简体中文](README_ZH.md)

# ArmBench

Reproducible runtime evaluation for action-chunk vision-language-action
policies.

ArmBench studies two deployment problems: action chunks that become stale
while inference is in flight, and discontinuities between committed and newly
sampled actions. It combines official pi0.5/LIBERO experiments with a local
MuJoCo/Panda runtime for protocol validation, fault injection, and
action-level inspection.

> Project status: research prototype. The repository contains validated
> simulation evidence for an official pi0.5 checkpoint. It does not train
> pi0.5, provide a hard-real-time guarantee, certify collision safety, or
> report real-robot results.

## Problem

An action-chunk policy predicts several future controls from one observation.
If inference consumes d control periods, the first d actions describe time
steps that have already passed by the time the response arrives. Executing the
chunk from index zero introduces a temporal mismatch between observation and
control.

ArmBench evaluates two complementary runtime strategies:

1. Temporal alignment selects the suffix whose first action matches the
   measured observation age. It operates outside the policy and does not
   modify checkpoint weights.
2. Projected overlap and RTC-style guidance operate inside the pi0.5 flow
   sampler. They condition a new chunk on actions that the controller has
   already committed to execute.

These strategies use different scheduler semantics and are evaluated as
separate methods.

## System architecture

~~~text
images + language + robot state
               |
               v
      pi0.5 / OpenPI policy
               |
               v
          action chunk
               |
               v
  temporal alignment or overlap guidance
               |
               v
 deadline, state, kinematic, and collision checks
               |
               v
        LIBERO or MuJoCo execution
               |
               v
 traces + videos + statistics + manifests
~~~

The repository has two execution paths:

| Path | Purpose | Policy source | Execution environment |
| --- | --- | --- | --- |
| Official checkpoint evaluation | Closed-loop method evaluation | Attested pi05_libero checkpoint | LIBERO on Linux/NVIDIA |
| Local runtime validation | Protocol, guard, and fault handling | OpenPI server or explicitly labeled deterministic fixture | MuJoCo Panda on Windows/CPU |

Results and action semantics are not combined across these paths.

## Implemented components

- Training-free fixed-delay and measured-age action-chunk alignment.
- Hard projected overlap and RTC-style denoised-action VJP guidance in the
  pinned pi0.5 flow sampler.
- Attested OpenPI evaluation with explicit sampling noise and matched
  closed-loop conditions.
- Strict observation/action contracts, bounded WebSocket inference, deadline
  latching, state-consistency checks, and fail-closed supervision.
- Panda joint, velocity, acceleration, gripper, and sampled mesh-edge checks
  with action backtracking.
- Dual-camera MuJoCo feedback, torque-controlled execution, and deterministic
  sensor, transport, state, and latency fault injection.
- Read-only validators, content-addressed manifests, paired statistical
  analyses, and offline video dashboards.

## Validated studies

Each row below is a separate registered or exploratory study. Results are not
pooled across protocols.

| Study | Matrix | Primary result | Interpretation |
| --- | ---: | --- | --- |
| Deterministic temporal alignment | 300 rollouts / 150 pairs | At 200 ms: 18/50 asynchronous vs 50/50 aligned; +64 points, bootstrap 95% CI [+50,+76], Holm-adjusted McNemar p=1.40e-9 | Supports suffix alignment under deterministic injected LIBERO delay |
| Measured-age confirmation | 240 rollouts / 120 pairs | 88/120 baseline vs 116/120 aligned; +23.33 points, pair bootstrap 95% CI [+15.00,+31.67], McNemar p=1.94e-6 | Supports client-visible age alignment with paired jitter and policy noise |
| Cross-suite validation | 300 rollouts / 150 pairs | 83/150 asynchronous vs 141/150 aligned descriptively; Object, Goal, and LIBERO-10 tests each pass Holm correction | Extends deterministic-delay evidence to three additional task suites |
| Corrected-v3 RTC overlap | 300 rollouts / 100 triplets | 96/100 unconditioned, 97/100 hard projection, 97/100 RTC; both success contrasts have Holm-adjusted p=1.0 | No task-success advantage; motion-seam reductions remain exploratory |
| Hard-projection pilot | 40 rollouts / 20 pairs | 19/20 unconditioned vs 18/20 projected; McNemar p=1.0 | Integration evidence only |

Full provenance, confidence intervals, runtime measurements, and study-specific
limitations are recorded in [Results](docs/RESULTS.md).

### Corrected-v3 pairing

The original RTC v2 comparison reused a LIBERO environment across method
conditions. A later invariant audit found that tasks 3, 8, and 9 produced
different query-zero policy images and actions despite matching declared state,
prompt, sampling key, and sampling noise. The v2 artifacts remain available as
audit records but are excluded from method-effect estimates.

Corrected-v3 creates a fresh environment for every rollout and requires four
matching query-zero hashes within each method triplet: policy input, response
action, sampling key, and sampling noise. The 300-rollout v3 matrix is a
complete rerun on held-out seeds; no v2 outcomes were pooled or repaired.

See the [pairing audit](docs/research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md),
[corrected protocol](docs/research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md), and
[acceptance guide](docs/RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE.md).

## Inspect preserved results

The acceptance workflows validate preserved artifacts and rebuild local
dashboards. They do not rerun policy inference and do not require a GPU.

From the repository root on Windows:

~~~powershell
.\scripts\rtc_primary_acceptance.cmd
.\scripts\measured_age_confirmatory_acceptance.cmd
& '..\.venv\Scripts\python.exe' -m integrations.openpi.acceptance_dashboard --open
& '..\.venv\Scripts\python.exe' -m integrations.openpi.cross_suite_dashboard --open
~~~

Append -NoOpen to either command script for noninteractive validation. A valid
RTC acceptance run reports:

~~~text
valid=true
rollouts=300
triplets=100
failure_videos_verified=10
tasks=10
~~~

The generated dashboards are stored under reports/ and remain fully offline.

## Local installation

The tested Windows environment uses Python 3.10.8. The virtual environment and
MuJoCo Menagerie checkout are workspace-level siblings of the repository.

~~~powershell
$Workspace = 'D:\arm-planning-control-project'
Set-Location $Workspace

git clone https://github.com/Shiraikuroko123/armbench-vla-runtime.git project
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git upstream\mujoco_menagerie
git -C upstream\mujoco_menagerie sparse-checkout init --cone
git -C upstream\mujoco_menagerie sparse-checkout set franka_emika_panda
git -C upstream\mujoco_menagerie checkout 71f066ad0be9cd271f7ed58c030243ef157af9f4

py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --editable '.\project[test,vla]'
~~~

If the workspace already exists, the self-locating launcher can run from any
PowerShell directory:

~~~powershell
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -CheckOnly
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd'
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -Formal
~~~

| Command | Behavior |
| --- | --- |
| -CheckOnly | Validate dependencies, scenarios, camera contracts, protocol, guard, and saved artifacts |
| default | Run one local smoke scenario without video |
| -Formal | Run the two-scene local matrix and render three MP4 files |

The local reference policy is a deterministic test fixture. Connect a remote
OpenPI server before making a learned-policy claim for the MuJoCo path.

## Official checkpoint execution

New pi0.5 rollouts require Ubuntu, an NVIDIA GPU, the pinned OpenPI checkout,
and the official checkpoint cache. The container workflow, preflight gates,
budget controls, and validation commands are documented in the
[OpenPI/LIBERO operations guide](docs/OPENPI_LIBERO_OPERATIONS.md).

The existing result dashboards can be inspected without that environment.

## Repository layout

~~~text
src/armbench/                 local planning, control, MuJoCo, VLA runtime
integrations/openpi/          official-checkpoint evaluators and analysis
integrations/openpi/patches/  pinned OpenPI sampler extensions
tests/                        unit and integration tests
scripts/                      repeatable launch and acceptance commands
docs/                         architecture, methods, protocols, and operations
evidence/                     preserved experiment artifacts
reports/                      generated offline dashboards
configs/                      benchmark and scenario configuration
~~~

Start with the [documentation index](docs/README.md) for the distinction
between current guidance, frozen protocols, audits, and historical runbooks.

## Reproducibility model

Formal artifacts bind the following information:

- repository and OpenPI commits;
- checkpoint URI and content SHA-256;
- resolved protocol and matrix;
- explicit policy sampling noise where required by the study;
- per-episode, per-query, and per-action records;
- analysis inputs and derived outputs;
- required video paths and hashes.

Validators fail closed on missing, modified, duplicated, noncanonical, or
out-of-matrix data. This provides artifact integrity and internal consistency;
it is not publisher authentication or a physical-safety certificate.

## Environment support

| Capability | Verified environment |
| --- | --- |
| Artifact validation and dashboards | Windows/CPU |
| Local MuJoCo runtime | Windows, CPU + OpenGL; no CUDA required |
| OpenPI client | Windows with bounded WebSocket/MessagePack transport |
| Official pi0.5 inference | Separate Ubuntu/NVIDIA host; more than 8 GB VRAM per upstream guidance |
| Isaac Gym / Isaac Lab | Not integrated |
| Real Franka Panda | Not integrated; no ROS2, libfranka, calibration, watchdog, or safety PLC adapter |

Other embodiments require new observation transforms, action semantics,
kinematic limits, and simulator or hardware adapters.

## Verification

Run the complete test suite from the repository root:

~~~powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
~~~

For runtime diagnosis, use the boundary-oriented
[troubleshooting guide](docs/DEBUGGING.md).

## Scope

- Official-checkpoint results cover one pi0.5-LIBERO checkpoint in simulation.
- Deterministic-delay and measured-age studies use blocking policy calls with
  post-response simulator catch-up, not independently scheduled inference and
  control threads.
- Motion seam is a process metric, not a task-success or safety endpoint.
- MuJoCo collision checks sample configurations and interpolated edges; they
  are not analytic continuous collision detection.
- The runtime constrains command slew but does not certify acceleration, jerk,
  or physical tracking.

## License and upstream software

ArmBench is released under the MIT License. Upstream models, datasets, assets,
and code retain their original licenses. See
[Third-party notices](THIRD_PARTY_NOTICES.md).
