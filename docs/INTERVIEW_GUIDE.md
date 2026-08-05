# VLA Interview Guide

## Graduate-student assessment

This project is now credible as a primary resume project for **VLA deployment,
embodied systems, robot learning infrastructure, simulation/evaluation, or
robotics software** roles. It demonstrates that you can connect a policy API to
robot observations/actions, reason about asynchronous execution, preserve
provenance, inject failures, and verify behavior in physics.

The policy-internal RTC extension also makes it relevant to **VLA inference and
runtime-method** roles: it requires understanding the pi0.5 flow sampler,
reverse-time integration, normalization/padding, VJP guidance, and matched
closed-loop evaluation. It remains a runtime-method project rather than a new
foundation-model architecture.

It is only a supporting project for **VLA model architecture, pretraining, or
fine-tuning research** roles. There is no learned model training, dataset
pipeline, or representation ablation. It now includes an attested official
pi0.5 checkpoint benchmark, but that demonstrates runtime/evaluation research,
not model-learning research.

## Ninety-second explanation

ArmBench studies stale action chunks in asynchronous VLA deployment. I added a
training-free dispatcher that uses measured observation age to skip the stale
prefix and execute a still-valid suffix. In the held-out 240-rollout study, I
paired initial states, response jitter, and explicit pi0.5 sampling noise;
success improved from 88/120 to 116/120, a +23.3-point paired effect with exact
McNemar `p=1.94e-6` and whole-task bootstrap 95% interval [+10.8,+38.3]. Mean
policy queries fell from 22.75 to 15.14. Separate deterministic-delay studies
cover another 600 rollouts across 40 LIBERO tasks, including three external
suites whose prespecified exact tests all survive Holm correction. All 840
confirmatory/external rollouts, failures, videos, source/checkpoint identity,
and manifests are preserved and independently validated. The result is
simulation-only and does not claim pi0.5 training, concurrent hard real-time
execution, or certified safety.

Separately, I ported RTC-style denoised-action VJP guidance into the pinned
pi0.5 sampler. A 20-query G0 gate established bitwise zero-guidance parity,
residual direction, 108.06 ms guided wall P95, and 6.43 GiB peak JAX memory.
After rejecting a v2 comparison whose environment reuse broke query-0 pairing,
I reran a corrected 300-rollout held-out matrix with fresh environments and a
four-hash pairing gate. Baseline, hard projection, and RTC guidance succeeded
on 96/100, 97/100, and 97/100; Holm-adjusted `p=1.0`, so I make no RTC
task-success claim. Motion seam decreased only as an exploratory process
metric.

## What you actually built

- The `latency_aligned` pi0.5-LIBERO dispatch mode, horizon preflight, strict
  short-chunk failure, and delay-matched action-prefix removal.
- A measured-age runtime with conservative suffix selection, deadline-bounded
  hold-refresh, counterbalanced execution, and mode-independent response-jitter
  and `10 x 32` pi0.5 sampling-noise pairing.
- A held-out confirmatory analyzer with exact paired McNemar, pair bootstrap,
  whole-task bootstrap, exhaustive task sign flips, leave-one-task-out effects,
  condition-order strata, and a 240-video acceptance dashboard.
- A transactional official-checkpoint evaluator with fixed matrices,
  checkpoint/source attestation, full video retention, and nested validation.
- A read-only paired ITT analyzer with Wilson intervals, deterministic paired
  bootstrap, exact McNemar/Holm tests, and descriptive task-level tables.
- A fail-closed cross-suite analyzer and offline dashboard binding three source
  manifests, 150 matched pairs, 30 task rows, and 300 playable videos.
- A reverse-time OpenPI RTC extension using `jax.vjp`, explicit sampling noise,
  attested sampler/source identity, and a fixed-observation latency/memory gate.
- A reference-only-bootstrap three-method overlap evaluator, a fresh-environment
  remediation for query-0 visual carryover, a four-hash pairing gate, and a
  deterministic 300-rollout held-out analyzer/dashboard.
- The Panda camera/proprioception adapter and visible task target.
- A pre-inference observation guard for blank images, nonmonotonic capture
  sequence/time, and exact camera replay during measured robot motion.
- Immutable Python data contracts for DROID observations and action chunks.
- A bounded WebSocket transport using official OpenPI MessagePack serialization,
  including round-trip, refusal, stalled-inference, and real-server probe paths.
- A one-command local OpenPI loopback that exercises the real wire/client/runtime
  path and independently audits request hashes without claiming learned inference.
- Deterministic wrong-shape, nonfinite, disconnect, and timeout injections on
  that socket path, with fail-closed MuJoCo artifacts and server-side audits.
- A one-command matched fault matrix with a nominal positive control, child
  artifact validation, request-hash pairing, CI status, and comparison plot.
- A stateful runtime guard with sequence/age checks, deadline latch/reset,
  action bounds, joint limits, sampled mesh-edge lookahead, backtracking, and
  hold fallback.
- Deterministic safe/fault streams, inference jitter schedules, and matched
  guarded/unguarded physics cases.
- A receding-horizon live physics loop with actual state/camera feedback and
  1/5/15-action query-cost comparison under 0/0.5 kg payloads.
- A query-bounded `vla-openpi-run` path that puts the real WebSocket client in
  that loop and distinguishes attempts, valid replies, and fail-closed holds.
- Optional live-physics MP4 recording for both reference-policy and remote
  OpenPI online episodes.
- Per-case, per-chunk, and per-action audit trails with raw and executed actions,
  reasons, scales, predicted/actual states, images, plots, and videos.
- Opt-in exact dual-camera query recording for offline input replay, with every
  224x224 frame rehashed against the per-chunk wire audit.
- A complete DROID request loader that aligns images, joint/gripper state,
  prompt, and sequence, then verifies repacked MessagePack bytes against the
  server-received payload SHA when loopback evidence is available.
- A recorded-request OpenPI probe for paired checkpoint/server comparison,
  retaining fixed input SHA, validated action SHA, timing, metadata, and guard
  output plus the exact official MessagePack input bytes without mislabeling
  offline inference as a physics rollout.
- An independent recorded-probe validator that recomputes response bytes and
  cross-checks arrays, metadata, provenance, and non-physics claim boundaries.
- A bounded recorded-request sweep that preflights exact inputs, isolates each
  server call, continues after a failed response, and distinguishes a partial
  cohort from complete evidence through structured rows and process exit code.
- A sweep validator that preserves valid failure evidence while preventing an
  incomplete collection from being reported as a complete model cohort.
- A same-request paired server comparator with per-step/per-dimension action
  deltas, guard effects, latency, provenance, and an explicit hash mismatch
  refusal path for future pi0 versus pi0.5 experiments; response snapshots and
  an embedded exact request plus an independent report validator make every
  displayed input/output delta recomputable.
- A request-hash-indexed cohort comparator that rejects missing/duplicate pairs,
  validates every child, and reports deterministic descriptive/bootstrap
  statistics without presenting action difference as task performance.
- A hierarchical cohort validator that rebuilds child, row, aggregate, and plot
  evidence and catches a modified metric before it can be cited.
- Client-visible failure stage, exception type, and bounded message mirrored
  between per-chunk CSV and NPZ traces for remote-server diagnosis.
- A schema-v5 artifact validator that cross-checks JSON/CSV/NPZ counts, camera
  hashes, safety fields, array shapes, and optionally decodes recorded videos.
- A Windows self-locating launcher, VS Code debug configurations, tests, pinned
  third-party versions, and explicit claim boundaries.

The official Panda assets, MuJoCo engine, and OpenPI serializer are dependencies,
not original work. Say exactly that; the bounded transport and runtime contracts
are ArmBench code.

## Questions to expect

### What is the difference between this and pi0/pi0.5?

pi0/pi0.5 is the learned policy that maps images, language, and state to an
action chunk. ArmBench is the runtime around that policy: it creates the request,
transports it, validates the reply, handles deadlines, checks execution
constraints, applies or rejects actions, and measures the physical result.

The formal LIBERO path loads the attested official `pi05_libero` checkpoint and
runs it in a closed-loop simulator, so "deployed and evaluated the frozen pi0.5
checkpoint in LIBERO" is correct. The separate local Panda/DROID path still
uses scripted non-learned action sources in its tracked physics experiments.
Neither path trained or fine-tuned pi0.5, and neither is a real-robot
deployment.

### Why did you not use reinforcement learning?

The research question is a runtime synchronization failure, not policy
learning: after inference consumes four control periods, the first four actions
in the returned chunk refer to states the environment has already passed. A
deterministic suffix alignment is therefore the direct baseline and does not
need to modify pi0.5. Adding PPO by itself would not make the causal claim
stronger; it would introduce training variance and a second source of behavior.

This means the project is strongest for VLA deployment, runtime, evaluation,
and embodied-systems roles. It is not a substitute for a model-training project
when applying specifically to VLA pretraining, fine-tuning, or RL research.

A defensible learned extension is a frozen-VLA query-level scheduler. At each
pi0.5 response it would choose to execute an aligned prefix of 1, 3, or 5
actions, or hold and refresh, with a success-versus-query-cost objective. That
is a sequential decision problem suitable for PPO because the choice changes
later observations and query count. It requires new online rollouts, held-out
tasks/suites, multiple training seeds, and fixed non-learned scheduler
baselines. The existing query logs do not contain full action chunks,
restorable simulator states, or counterfactual outcomes, so they must not be
described as an offline-RL dataset.

### Why use the `pi05_droid` contract?

DROID uses a Franka setup and exposes a concrete public interface: two 224x224
RGB images, seven joint positions, one gripper position, a prompt, and joint
velocity plus gripper action chunks. That makes the policy/runtime boundary
testable without inventing a private VLA API. The contract and client are pinned
to one upstream commit because horizons and example code can change.

### Why is the action horizon 15?

The pinned `pi05_droid` model config sets an action horizon of 15 and action
dimension 8. DROID runs at about 15 Hz. The official example has an older comment
mentioning 10 actions, so the implementation trusts the pinned model config and
strictly rejects any other response shape.

### Are actions normalized?

The first seven values are DROID joint-velocity commands. The official DROID
example clips the returned action to `[-1, 1]` before stepping a joint-velocity
environment. ArmBench therefore clips to +/-1 rad/s and then applies the Panda
joint-specific hardware velocity limits. It does not multiply a normalized
fraction by each hardware limit.

### Why a latched deadline fallback?

A 240 ms chunk exceeds the configured 200 ms age threshold. Initially the
runtime held only that chunk and resumed the remaining precomputed stream; the
state mismatch produced 43 contact steps in a regression run. The corrected
state machine enters hold and requires explicit reset/resynchronization. The
formal rerun had zero contacts but did not complete the task. This is a concrete
example of safety taking priority over availability.

The live online loop has a separate matched check: repeating
`0/40/80/160 ms` completed safely, while changing only the final entry to
`240 ms` triggered a fourth-query hold. That evidence isolates the configured
200 ms deadline from scene, payload, horizon, and action-source changes.

### What happens when the policy server fails?

The bounded client closes its connection on an invalid response, disconnect, or
receive timeout. The runtime catches the exception at the policy boundary,
advances MuJoCo for the measured wait under a pose-hold controller, produces no
raw remote action, and executes a latched 15x8 hold chunk. The loopback fault
matrix sends all four cases through the real MessagePack/WebSocket path. Each
case records one attempted request, zero validated chunks, task failure, and zero
contacts. This tests deterministic injected failures; it does not estimate real
server availability. The client-visible exception type is retained separately
from the injected server cause, so a real deployment does not depend on access
to server logs.

The matrix is stronger than four screenshots because it fixes the scene,
horizon, payload, and one-query budget, includes a nominal positive control, and
computes pass/fail from structured child artifacts. It still establishes only
the tested deterministic failure paths, not arbitrary network fault coverage.

### Why not execute only the first action and re-query immediately?

The fault-injection benchmark validates complete chunks because chunking is the
public OpenPI/DROID interface. The separate `vla-online-run` experiment now
executes prefixes of 1, 5, or 15 actions and recaptures actual MuJoCo state plus
both cameras each time. Horizon 1 provides the highest feedback frequency but
requires many more policy queries. The current comparison uses a non-learned
reference policy, so it measures runtime behavior rather than VLA competence.

### How do you know the cameras were really refreshed?

Every observation cycle records SHA-256 and adjacent-frame pixel delta for both
full frames, plus 16x16 RGB thumbnails in the NPZ trace. The runtime separately
compares the images with the preceding accepted observation. If proprioception
changed by more than 0.005 rad while either image is byte-identical, it rejects
the observation before policy inference and latches hold. A deterministic frozen
camera test therefore has two observation cycles but only one policy query. This
detects exact replay, not every possible stale or corrupted image.

For selected debug/evidence runs, `--save-observations` also stores both original
224x224 frames for every query. The validator recomputes all hashes, so these
runs support exact input inspection rather than only freshness evidence. It is
opt-in because full sweeps can otherwise spend hundreds of megabytes on images.

Do not call images alone a replayable request. A request is reported replayable
only when both images, joint state, gripper state, prompt, and sequence metadata
align. In a loopback run, the strongest check is equality between the SHA-256 of
the reconstructed official MessagePack payload and the raw payload received by
the server.

The tracked two-query replay artifact demonstrates that check twice with
different live camera/state observations. Both payload hashes match exactly;
the artifact still uses a scripted non-learned reply and must not be described
as a pi0/pi0.5 rollout.

The tracked matched evidence is concrete: nominal execution produced 16 unique
hashes per camera and completed, while replaying both cycle-0 frames at cycle 1
produced one observation rejection and only one policy call. The rejection run
remained contact-free but intentionally did not complete the task.

### Is the guard guaranteed safe?

No. Configuration contacts use MuJoCo meshes, but edge validation samples joint
interpolation at 0.02 rad resolution. Physics can deviate from the predicted
path. The runtime bounds command-space acceleration, but it has no jerk bound,
dynamics-level reachable set, or formal proof. The evidence is zero contact in
fixed cases, not a theorem or safety certification.

### Why does the collision guard fail the task?

The injected stream goes through an obstacle. The local repair scales each
unsafe velocity and eventually holds; it cannot invent a new path around the
obstacle. Zero contact with large goal error is a correct refusal, not task
success. A recovery planner or a newly conditioned VLA chunk is needed to finish.

### Why have RRT-Connect in a VLA project?

It produces a known-safe positive-control stream and supplies the geometry,
clearance, and physics substrate used to test the VLA runtime. It is not
presented as the VLA itself. Comparing a safe planner stream, an unsafe direct
stream, guarded execution, and unguarded execution gives the safety layer both
positive and negative controls.

### Why MuJoCo instead of Isaac Gym or Isaac Lab?

The available host has no NVIDIA GPU. MuJoCo provides official Panda meshes,
contacts, torque dynamics, rendering, and reproducible low-volume evaluation on
the Intel laptop. Isaac Lab is a simulator/training framework, not a VLA model;
it is valuable when large parallel GPU rollouts or NVIDIA sensor pipelines are
needed. Porting the evaluator later would be additional cross-simulator
validation, not a prerequisite for this runtime result.

### Can it run on a real Panda?

No. A real deployment needs a `libfranka` or ROS2 adapter, calibrated camera and
joint conventions, deterministic command timing, watchdogs, emergency-stop
integration, collision thresholds, network failure handling, and hardware
validation. The present guard is research simulation code, not safety-rated.

### What would you do next with a GPU budget?

The corrected 300-rollout RTC overlap matrix is complete and does not establish
task-success superiority. The next systems experiment should separate inference
and control into independently ticking loops and measure observation/action age
rather than blocking and injecting a fixed delay. After that, repeat a reduced
frozen matrix with a second open VLA checkpoint. A simulator port or RL run is
useful only when it tests one of those scientific gaps.

## How to live-debug it in an interview

1. Run `scripts\vla_demo.cmd -CheckOnly` from an arbitrary directory.
2. Open the formal exterior/wrist images and explain every request tensor.
3. Filter `per_action.csv` for `backtracked:` and show raw versus executed action.
4. Filter for `deadline_latched` and explain why later fast chunks still hold.
5. Play guarded and unguarded fault videos side by side.
6. Set a breakpoint in `ActionChunkGuard.guard`, change one backtracking scale,
   rerun a new quick ID, and predict what metric should move.

Do not edit the formal evidence directory. New runs are immutable and use new
IDs.

## Honest claim boundaries

Do not claim:

- pi0/pi0.5 checkpoint results without real probe and closed-loop artifacts;
- RTC task-success improvement from the fixed-observation G0, rejected v2
  artifacts, or corrected-v3 held-out matrix;
- a specific checkpoint identity without the matching GPU server launch log;
- model training, fine-tuning, or DROID/LIBERO dataset evaluation;
- analytic continuous collision detection;
- OS-level hard real-time scheduling;
- general camera-fault coverage beyond tested blank/exact-replay cases;
- jerk constraints or dynamics-verified acceleration guarantees;
- Isaac Lab, ROS2, `libfranka`, or real-robot deployment;
- formal or certified safety.

The strongest current claim is an end-to-end, OpenPI-compatible VLA
runtime/evaluation system with live receding-horizon physics feedback,
deterministic fault injection, bounded failure handling, and action-level
auditability.

## Resume wording

> Built an OpenPI-compatible VLA runtime and attested pi0.5-LIBERO evaluator.
> Implemented training-free measured-age action-chunk alignment with paired
> response jitter and policy-sampling noise; in a frozen 240-rollout held-out
> study, improved success from 88/120 to 116/120 (+23.3 points, exact McNemar
> p=1.94e-6; whole-task bootstrap 95% CI [+10.8,+38.3]) while reducing mean
> policy queries from 22.75 to 15.14. Preserved 840 confirmatory/external
> rollouts and videos with checkpoint/source attestation, immutable manifests,
> independent validators, and one-command paired visual acceptance.

For a VLA inference/runtime role, add a separate sentence rather than changing
the confirmed headline result:

> Extended the pinned OpenPI pi0.5 flow sampler with training-free RTC-style
> denoised-action VJP guidance; verified bitwise legacy parity, residual
> direction, latency/memory gates, and a corrected 300-rollout LIBERO-10
> three-method evaluation with fresh-environment and four-hash pairing gates.
> The study found no task-success superiority; action-seam changes are reported
> only as exploratory process evidence.

Use "evaluated the attested official pi0.5-LIBERO checkpoint," not "trained
pi0.5" or "deployed on a real robot."

## Before placing it on a resume

You should be able to do all of the following without reading a prepared answer:

1. explain why delay makes an action prefix stale and why skipping it is only a
   discrete temporal-alignment assumption;
2. reproduce root validation and the paired analysis from `per_episode.csv`;
3. explain paired McNemar, pair versus whole-task bootstrap, task sign flips,
   the +23.3-point measured-age effect, and the condition-order strata;
4. draw the LIBERO and MuJoCo runtime boundaries without mixing their tensor
   shapes or action semantics;
5. explain why pi0/pi0.5 is not a simulator and why Isaac Lab is not a VLA;
6. locate a matched baseline-failure/aligned-success pair and both videos;
7. explain the deadline latch and the safety/task-success tradeoff in the local
   scripted MuJoCo evidence;
8. distinguish artifact consistency, checkpoint attestation, and publisher
   authenticity;
9. state which code/assets are yours and which are pinned dependencies;
10. explain why the measured-age pilot's unpaired policy RNG is a confound and
    how explicit keyed sampling noise fixes it in the measured-age confirmation;
11. derive `D_t(x)=x-t v(x,t)` and explain why the OpenPI correction uses a
   minus sign when Euler integration has `dt < 0`;
12. explain the difference between G0 model-space residual reduction, pilot
    task efficacy, hard projection, and soft VJP guidance;
13. explain reference-only bootstrap, keyed-noise triplets, Holm correction,
    and why seam is averaged within episode before comparing methods;
14. explain why matching sampling noise did not rescue v2 after its policy
    images diverged, and how corrected-v3's fresh environments and query-0
    four-hash gate restore the matched comparison.

AI assistance produced substantial implementation and documentation. Your
defensible ownership comes from being able to reproduce, inspect, modify,
debug, and explain the system. Do not imply that you independently typed code
you cannot maintain.
