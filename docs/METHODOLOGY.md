# Evaluation methodology and system boundaries

## Document scope

ArmBench contains two independent evidence paths. Official pi0.5-LIBERO studies
evaluate temporal alignment and sampler-internal overlap methods with an
attested checkpoint. The local MuJoCo/DROID path evaluates protocol contracts,
runtime guards, deterministic faults, and Panda physics with either a remote
OpenPI server or explicitly labeled non-learned fixtures.

This document specifies the local MuJoCo/DROID methodology and the shared
artifact model. Official-checkpoint protocols and outcomes are documented in
[Results](RESULTS.md), [Measured-age temporal alignment](MEASURED_LATENCY_RUNTIME.md),
and [RTC-guided pi0.5 integration](RTC_PI05_INTEGRATION.md). Conclusions are not
transferred between paths without a registered cross-path experiment.

## Local runtime validation question

The local experiment asks whether a runtime layer can reject stale or
kinematically infeasible DROID action chunks while leaving a registered
positive-control stream unchanged. Deterministic non-learned action sources
make fault conditions and expected responses reproducible; they do not measure
pi0.5 task competence.

The policy/runtime boundary is compatible with OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and model config `pi05_droid`:

```text
observation/exterior_image_1_left : uint8[224, 224, 3]
observation/wrist_image_left      : uint8[224, 224, 3]
observation/joint_position        : float[7]
observation/gripper_position      : float[1]
prompt                            : nonempty string
actions                           : float[15, 8]
```

The pinned official client supplies MessagePack NumPy serialization and the
wire contract. ArmBench wraps those bytes in a bounded WebSocket transport
because the upstream client retries refused connections forever and performs an
unbounded inference receive. ArmBench validates the exact input keys and refuses
a response whose horizon, dimension, or finite-value contract differs. Invalid
policy data is converted by the supervisor into a provenance-safe hold before it
can reach the motion guard. Each valid observation and action chunk also carries
a local sequence ID, capture/receive timestamps, policy-source label, client
latency, and optional server timing.

## VLA observation construction

The Panda model is composed with two cameras before MuJoCo compilation. The
exterior camera views the complete workcell. The fixed hand-mounted camera uses
a wide field of view and offset selected from the official `hand` body frame so
that obstacles and target remain visible. Both views are rendered as 224x224
RGB `uint8`. Automated tests require red obstacle and green target pixels in
both views.

The green target is placed at the official MJCF hand position for the scenario
goal and has contact disabled. It provides a visual task referent but cannot
affect collision or physics metrics. The language prompts explicitly identify
the red obstacles and goal. Joint and finger positions come from the same
MuJoCo state used for rendering.

Before policy inference, an observation guard checks both image standard
deviations, sequence/time monotonicity, and exact equality with the preceding
accepted frames. A repeated frame is rejected only when maximum joint-state
change exceeds the configured 0.005 rad threshold; identical images while the
robot is stationary are allowed. Rejection enters the same latched runtime hold
without calling the policy. This is a deterministic replay heuristic, not a
general semantic image-quality model.

## DROID action semantics

The first seven output dimensions are joint-velocity commands; the eighth is a
gripper-position command. The pinned OpenPI DROID example clips every returned
dimension to `[-1, 1]` before stepping a joint-velocity environment. ArmBench
uses +/-1 rad/s for the seven velocity commands and also caps them by the Panda
joint-specific velocity limits. The gripper value is clipped to `[0, 1]`.

This is not implemented as a normalized fraction multiplied by each hardware
limit. That alternative would incorrectly amplify a DROID command by roughly
the Panda velocity limit.

## Runtime validation pipeline

For a chunk captured at time `t_obs` and received at `t_recv`, end-to-end age is
`1000 * (t_recv - t_obs)` milliseconds. A chunk older than 200 ms is replaced by
hold. The failure is latched for the remainder of the episode until an explicit
`reset`, because automatically resuming an old open-loop stream after a missed
chunk can be inconsistent with the actual state.

For a fresh finite action, the runtime:

1. rejects a policy/execution joint-state mismatch above 0.05 rad;
2. clips joint velocity and gripper values to the configured bounds;
3. limits velocity change relative to the last executed command;
4. integrates one 15 Hz step from the current predicted joint state;
5. checks Panda joint limits and enabled MuJoCo mesh contacts;
6. checks the complete joint edge at no more than 0.02 rad interpolation spacing;
7. tries velocity scales `1.0`, `0.75`, `0.5`, `0.25`, then `0.0`;
8. commits the first valid candidate and repeats from that predicted state.

The planning/check model inflates each physical obstacle by 20 mm. Execution
uses the physical radius. The output records every scale, reason, raw/executed
action, and before/after joint state. This is sampled runtime validation, not a
control-barrier-function proof or continuous swept-volume guarantee.

## Receding-horizon physics loop

The online benchmark does not precompute a complete guarded trajectory and run
physics afterward. It captures exterior RGB, wrist RGB, joint position, and
gripper position from one live torque-controlled MuJoCo state; requests and
guards a 15x8 chunk; executes only a configured prefix of 1, 5, or 15 actions;
then repeats from the resulting actual `qpos` and `qvel`. The guard's previous
velocity is synchronized to the last action that was actually executed, not the
unexecuted tail of the chunk. Every query preserves the raw and guarded action
chunks, client/end-to-end timing, optional server timing, and termination
reason. A positive `max_policy_queries` budget can bound remote inference cost;
reaching it causes a pose hold and is reported as `query_budget`.
When enabled, online MP4 frames are rendered during these same physics steps
from the exterior camera. They are not reconstructed later from joint traces.
Online artifact schema version 5 also normalizes every checked action into
`per_action.csv`. The `executed` field separates the action prefix applied to
physics from the unexecuted remainder that was only validated; raw/guarded
values, intervention reason/scale, and predicted before/after state remain
query-addressable. Every query also records SHA-256 for both full RGB frames and
the mean absolute pixel delta from the preceding query. The NPZ stores matching
16x16 RGB thumbnail sequences, preserving low-cost visual audit evidence without
retaining every full 224x224 frame.

Full query images are opt-in because a two-camera 224x224 RGB pair is about
301 KiB before compression. With `--save-observations`, every query stores both
original `uint8` frames in the NPZ alongside their CSV/NPZ hashes. The validator
requires complete dual-camera coverage, exact `(queries, 224, 224, 3)` shapes,
and a SHA-256 match for every frame. The lightweight default retains hashes,
frame deltas, 16x16 thumbnails, and first/last full PNGs but cannot exactly
replay every policy input.

Every new online trace also stores the observed seven-joint vector, normalized
gripper scalar, prompt, and sequence/action offset per query. When full images
are present, `vla-request-inspect` reconstructs the exact five-key DROID mapping
and packs it with official OpenPI MessagePack serialization. The loopback server
hashes the raw received payload before unpacking; equality with the reconstructed
payload hash provides a byte-level round trip. Older artifacts lacking gripper
or prompt arrays remain schema-valid but report zero replayable requests.

`vla-recorded-probe` sends one reconstructed request through the same bounded
OpenPI client used by the online runtime. It validates a finite 15x8 reply,
stores raw/guarded action arrays and predicted positions, and hashes the raw
response for paired server comparison. The observation gets a fresh local
capture timestamp before inference because timestamps are runtime metadata, not
DROID payload fields; this lets the guard measure the replay call's actual age
without changing request bytes. No physics is stepped, so the artifact reports
no physical-safety outcome.

Probe artifact v2 embeds the exact official MessagePack request sent to the
server. Validation rehashes and unpacks it, then reconciles the two RGB arrays,
proprioception, prompt, key order, and source request metadata. This removes the
original replay directory as a dependency for inspecting one fixed probe input.
It does not prove which checkpoint processed those bytes.

`vla-recorded-probe-sweep` collects an explicitly bounded query-index cohort
from one replayable source artifact. All requests are reconstructed and hash
checked before creating the output directory. Each query then uses an isolated
client connection so one malformed response, timeout, or disconnect is retained
as a per-query failure without contaminating later samples. Successful children
must pass the independent probe validator. A complete sweep requires every
planned query to succeed; neither successful nor failed rows execute physics.

Its independent validator requires one ordered CSV row per selected request,
revalidates every successful child, and requires failed rows to retain bounded
exception type/message fields without response claims. It reconciles all counts
and file hashes. Internal validity and completion are separate: a correctly
recorded partial failure is valid evidence of an incomplete sweep, and the CLI
still returns nonzero.

`vla-recorded-probe-validate` independently reloads that probe artifact and
cross-checks its four files. It recomputes the raw-response hash from the exact
array bytes, verifies shape/dtype/range metadata, finite guarded trajectories,
source-request and environment hashes, and the explicit absence of physics and
checkpoint-attestation claims. This detects post-run corruption and coordinated
metadata inconsistencies; it does not establish that the remote server loaded a
particular learned checkpoint.

`vla-recorded-probe-compare` performs a paired descriptive comparison only after
both inputs pass that validator and their serialized request hashes match. It
reports raw and guarded action RMSE/max differences, predicted joint-position
differences, per-step velocity/gripper norms, guard interventions, and observed
client latency. Reusing one exact request removes input variation for that pair,
but one request is not a benchmark distribution and latency is not controlled
across independently provisioned servers. User labels are retained as labels,
not promoted to checkpoint attestation.

Paired report v2 requires byte-identical source requests and embeds one exact
MessagePack payload alongside both validated responses, guarded chunks, and
predicted joint paths. Its independent validator reconstructs the DROID input,
recomputes action hashes and all aggregate/per-step/per-dimension differences,
then checks CSV, plot, and environment hashes. The fixed input/output comparison
is reproducible without source probes or a live policy server.

`vla-recorded-probe-batch-compare` extends exact pairing to two artifact
cohorts. It requires identical request-hash sets and exactly one response per
hash on each side, validates every source and child comparison, then aggregates
raw/guarded RMSE and paired latency/guard effects. Mean uncertainty uses a
deterministic percentile bootstrap over request pairs (seed 20260804, 10,000
resamples). This quantifies variation within the selected request cohort only:
temporally adjacent observations may be correlated, and no success labels or
physics outcomes are present.

The independent batch validator walks the full evidence hierarchy rather than
trusting the top-level table: child response arrays regenerate child metrics,
child metrics regenerate per-pair rows, and those rows regenerate aggregate and
bootstrap statistics. Stored hashes bind each child and top-level JSON/CSV/plot
to the reported environment. This is consistency and corruption detection, not
a cryptographic signature against an adversary who can rewrite every file.

`vla-openpi-run` replaces the reference policy with the bounded transport built
on official OpenPI serialization while retaining the same live physics loop. It
records server metadata and separates attempted remote calls from validated
replies. The artifact marks `remote_policy_response_validated=true` only if at
least one remote reply passes the `15x8` and finite-value contract and is bound
to the current observation record. `checkpoint_identity_verified` remains false
because the OpenPI wire protocol does not attest the loaded checkpoint. A
malformed reply or timeout advances MuJoCo under hold for the measured wait,
then produces a latched runtime fallback rather than a fabricated successful
inference.

`vla-loopback-run` exercises that identical transport and execution path against
an ephemeral local server. The server validates the exact DROID keys and image,
state, gripper, and prompt contracts, then returns deterministic reference
actions. It stores a separate request audit with both image hashes. Its policy
provenance is always `scripted_non_learned_loopback`; a valid network response
does not become evidence of learned-policy or checkpoint performance.

The loopback can inject one response fault at a selected request: a wrong action
shape, a nonfinite action, a server-side close before response, or a response
delayed beyond the bounded client timeout. Injection occurs only after the
server validates and hashes the DROID observation. The malformed responses pass
through the real MessagePack/WebSocket client; the supervisor must reject them,
advance physics under pose hold for measured client wait time, execute no remote
action, and retain a safe failure artifact. These controlled cases establish
failure-path behavior for the tested faults, not network reliability statistics.
Each chunk retains the failure stage, client exception type, and a 500-character
message in both CSV and NPZ form; the aggregate retains the unique stage/type
set. These are runtime observations, not inferred server-side causes.

`vla-loopback-matrix` holds scene, payload, execution horizon, query budget,
camera recording, and client timeout constant while changing only the injected
fault. Its nominal positive control must produce one validated chunk without a
runtime fallback. Every fault case must produce zero validated chunks, one
policy-inference fallback, matching server/client camera hashes, and zero
contact/self-contact/joint-limit steps. The combined manifest reports a Boolean
matrix result and hashes the structured matrix JSON; each child artifact remains
independently validatable.

The bundled online policy follows a collision-free reference and is labeled
`scripted_non_learned_reference`. It exists to isolate the effect and query cost
of feedback horizon. Its configured latency is synthetic: MuJoCo advances under
a pose-hold controller for that duration, then the response is checked against
the new dispatch state. This supports controlled delay/state-mismatch tests but
is not measured pi0/pi0.5 server latency or OS scheduling evidence.

A latency schedule is indexed by policy query and repeats deterministically.
This matters when comparing horizons: one query always consumes one schedule
entry even though that reply may execute 1, 5, or 15 actions. The complete
resolved profile and per-query realized waits are stored with the run.

The optional online state-jump fault changes one MuJoCo joint after observation
capture and immediately before guard dispatch. It is indexed by policy query
and records the exact injected seven-joint offset. The default 0.08 rad example
crosses the 0.05 rad consistency threshold, so the expected outcome is a
latched hold, physical safety, and task incompletion. Directly changing `qpos`
makes the fault deterministic and attributable; it is not presented as a
physically modeled external impulse.

## VLA fault protocol

The positive-control stream is produced by fixed-seed RRT-Connect, mesh-checked
shortcutting, and velocity timing, then converted into 15 Hz DROID-shaped
chunks. The negative-control stream directly interpolates start to goal through
the obstacle. Both are exposed through the same `ActionChunkPolicy.infer`
interface and labeled `scripted_non_learned`.

Each of two scenarios runs guarded and unguarded under three conditions:

- safe stream with repeating `0/40/80/160 ms` inference jitter;
- direct collision fault at 50 ms;
- safe stream with repeating `0/40/80/240 ms` jitter, where 240 ms crosses the
  deadline and activates the latch.

Task success and physical safety are separate. Physical safety requires zero
obstacle contacts, self-contacts, and joint-limit samples. Task success requires
final maximum joint error no greater than 0.05 rad. A guard that stops a
collision but does not reach the goal is safe and task-incomplete.

## Primary robot model

The primary experiment loads `franka_emika_panda/scene.xml` from MuJoCo
Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. ArmBench maps the
seven arm joints, two finger joints, link bodies, position limits, velocity
limits, and Panda effort limits by name. Obstacles and the optional payload are
composed into an `MjSpec` before compilation; upstream model files are never
modified.

The model is compiled twice for distinct purposes. Planning models retain the
Menagerie position actuators but are used kinematically through `mj_forward`.
Execution models zero the seven arm actuator gain/bias terms and apply joint
torques through `qfrc_applied`. The finger actuator remains available to hold
the gripper closed.

## Scenario protocol

`single_block` contains one 55 mm sphere. `narrow_gate` contains three 55 mm
spheres. Both use the same versioned start and goal configurations. For every
configured clearance, automated validation requires:

- the start to be valid;
- the goal to be valid;
- direct joint interpolation to be blocked.

The robust profile inflates obstacle radii by 20 mm during planning. Execution
always restores the physical 55 mm obstacle radii, so clearance is a planning
constraint rather than a larger execution obstacle.

## Mesh collision checking

At a configuration, MuJoCo computes enabled contacts between the compiled
Panda collision meshes and obstacle geoms. Enabled robot self-contacts are also
treated as invalid. An edge is sampled by joint interpolation so that the
maximum change of any joint between checks is no greater than 0.05 rad.

This is a resolution-bounded discrete check. Although mesh contact is more
faithful than the legacy capsule skeleton, it is not analytic continuous
collision detection and does not provide a swept-volume guarantee.

## Planner comparison

RRT-Connect and RRT* share the robot model, collision checker, start/goal,
clearance, 0.35 rad step size, 2 s wall-clock deadline, and paired seeds. The
RRT* implementation stops after its first feasible path. The comparison is
therefore time to feasibility under a fixed deadline, not asymptotic path
optimality.

Latency percentiles include failures because a deadline miss is operationally
meaningful. Path statistics include successes only. Success uncertainty uses a
Wilson 95% interval. The formal physics run uses 10 seeds per
scene/clearance/planner condition; that is suitable for a reproducible project
comparison but too small for a broad algorithmic superiority claim.

## Post-processing and timing

Random shortcutting accepts a replacement only after checking the complete new
edge and revalidates the final path. First-order time parameterization limits
each joint velocity using the slowest joint in a segment. It does not constrain
acceleration or jerk. The VLA runtime adds a separate command-space acceleration
limit while validating action chunks; this does not change the planner's
first-order trajectory parameterization.

The `nominal_fast` profile runs at 35% of Panda velocity limits. The matched
`nominal_slow` and `clearance_slow` profiles both use 10%; their only planning
difference is 0 versus 20 mm obstacle inflation.

## Rigid-body execution

MuJoCo advances at 2 ms and the controller updates at 10 ms. At a control tick,
the torque request is

```text
tau = Kp * (q_desired - q_observed)
    + Kd * (dq_desired - dq_observed)
    + qfrc_bias
```

`qfrc_bias` supplies MuJoCo gravity, Coriolis, and passive-force compensation.
The result is clipped independently to the Panda effort limit for every joint.
Observation delay is implemented with a control-rate history buffer; 0/40/80
ms correspond to 0/4/8 control ticks. The command trajectory itself is not
delayed.

The payload condition adds a 0.5 kg box rigidly below the hand. Its geometry is
included during planning and execution, while its mass and inertia affect only
execution dynamics.

## Safety outcome

An execution is a `safe_success` only when all conditions hold:

- maximum final joint error is no greater than 0.05 rad;
- obstacle contact steps equal zero;
- self-contact steps equal zero;
- joint-limit violation steps equal zero.

RMSE alone is intentionally insufficient. The benchmark also records torque
saturation, contact events and duration, maximum normal contact force, maximum
penetration, self-contact, and joint-limit samples. MuJoCo contacts can be
stiff and the reported peak force is simulator/model dependent; it is useful
for comparing these cases, not as a certified physical load.

## Capsule consistency audit

The secondary approximation connects consecutive official MuJoCo body origins
with 50 mm-radius capsules. It uses the official kinematic state, not the
legacy DH model. For each fixed random configuration, its obstacle label is
compared with the exact MuJoCo mesh-contact label. The audit reports both
dangerous false-safe and conservative false-collision outcomes.

## Reproducibility

Every stochastic component receives an explicit NumPy generator. Each run
stores its resolved config, Git state, Python/dependency/platform metadata, raw
planning rows, raw execution rows, all collision samples, aggregate JSON,
successful paths, control traces, videos, posters, and a timestamped log. A run
directory is created with `exist_ok=False` and cannot be silently overwritten.

The VLA artifact additionally stores policy/OpenPI provenance, camera inputs,
per-case metrics, every chunk deadline/latch record, every raw and repaired
action, predicted/desired/actual joint arrays, an overview figure, and selected
MP4 executions. Local artifacts explicitly set
`remote_policy_response_validated=false` and
`checkpoint_identity_verified=false`.
A separate online artifact stores every re-observed joint state and action
offset, the 1/5/15 horizon comparison, first/last camera frames, and live physics
traces.
A successful `vla-probe` writes a separate artifact with that field set to true
only after a remote response passes validation. `vla-openpi-run` applies the
same rule to a query-bounded, receding-horizon rollout and also stores every raw
and guarded remote chunk.

The schema-v5 online artifact has a separate read-only validator. It requires
the JSON, CSV, and NPZ episode keys and counts to agree; checks query attempts,
executed actions, safety booleans, trace array shapes, per-cycle camera hashes,
thumbnail counts, and first/last saved-image hashes; and can decode every
referenced MP4 first frame. It reports the byte-level SHA-256 of
`aggregate.json`. When the additive failure-audit arrays are present, all three
must exist and exactly match the per-chunk CSV. Older schema-v5 evidence without
these optional arrays remains valid. Optional full-observation arrays follow the
same additive rule and are rehashed frame by frame. This is a reproducibility
and corruption check, not a signed chain of custody, model-identity attestation,
or proof that the safety metrics generalize beyond the recorded episodes.

Wall-clock planning latency varies with host load. Fixed seeds preserve sampled
sequences, but a search near the 2 s boundary can change status on a different
machine. Hardware, software, commit, and raw timeout rows must accompany any
latency claim.

## Legacy baseline

Version 0.1 used a standard-DH NumPy model and decoupled joint dynamics. Its
kinematic frame does not match the official Menagerie Panda: measured hand
position differences were approximately 0.588 m at the benchmark start and
0.440 m at the goal. Those results remain useful for testing the independent
planner/control code but are historical algorithm results. They are not
evidence for Panda mesh geometry, rigid-body simulation, or real-robot
behavior.

## Current limitations

The repository has no real-robot adapter, ROS2 node, `libfranka` integration,
emergency-stop interface, safety PLC, or hardware-in-the-loop result.
Official-checkpoint evidence covers one pi0.5-LIBERO model family in simulation.
The local fault schedule is deterministic; partially stale or semantically
corrupted frames remain outside the exact-replay detector.

OS-level hard-real-time scheduling, arbitrary workcell meshes, analytic
continuous collision detection, jerk constraints, dynamics-level acceleration
guarantees, uncertainty calibration, cross-model evaluation, and real-Panda
experiments remain outside the implemented scope.
