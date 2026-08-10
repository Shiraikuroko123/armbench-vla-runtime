# ArmBench Website Product Brief

## Primary job

Show a reviewer or robotics hiring engineer both how an attested pi0.5
checkpoint response crosses the policy-to-Panda boundary and how the same
checkpoint runs in a true independent-clock LIBERO pilot, with failures and
GPU-free validation preserved.

## One-sentence claim

ArmBench connects content-attested pi0.5-LIBERO responses to an asynchronous
seven-DoF MuJoCo Panda runtime through explicit action semantics, measured-age
dispatch, braking-invariant repair, and reviewable execution evidence.

## Information architecture

1. Full-bleed live-checkpoint Panda simulation hero with explicit `SIMULATION`,
   `LIVE PI0.5 CHECKPOINT`, and target-not-reached disclosure.
2. Artifact bar linking source, G01 live evidence, the full catalog, and docs.
3. Current live-evidence strip and four artifact ledgers.
4. Assurance flow and atomic decision table.
5. Registered 27-case fault matrix with outcome distribution.
6. Two assured MuJoCo tasks, physical predicates, gripper collision boundary,
   and offline latency disclosure.
7. Live bridge scope followed by the separately scoped π0.5-LIBERO outcome
   study with synchronized authentic comparison clips, plus an official
   LeRobotDataset v3.0 Panda H×8 roundtrip.
8. G02 independent-clock pilot with a 40-rollout metric strip, one separately
   labeled visual-success run, and both retained core-pilot failure videos.
9. Registered 18-cell / 720-rollout Spatial and Object deadline study,
   presented as a service- and clock-specific runtime-budget curve.
10. Frozen 120-pair / 240-rollout held-out comparison of age-aligned suffix
    and response-relative chunk selection, with seed blocks and pairing gates.
11. One-command full local CPU acceptance, focused Panda replay commands, a
   checked-in VS Code workspace, and a four-step acceptance path.
12. Verified 17-case asynchronous CPU boundary disclosure.
13. Proven / not-claimed scope block and architecture link.

## Audience

- Embodied-AI and robotics hiring teams assessing implementation depth.
- Researchers checking whether a result is recomputable and correctly scoped.
- Engineers who want to run the saved reports without a GPU or cloud host.

## Results contract

| Artifact | Exact result | Boundary |
| --- | --- | --- |
| `g01_live_panda_smoke_final_001` | 35 accepted live responses; mean/P95 82.75/89.56 ms; 290/311 control ticks during inference; 0 registered simulation violations; 94 video frames | official checkpoint integration probe; target not reached; no official task, hardware, hard-real-time, or safety claim |
| `pi05_libero_independent_clock_core_40_001` | 40/40 completed; 38/40 task success; 4,521/4,623 ticks during inference; 4,031 execute and 592 hold ticks; 0 deadline/provider failures | official-checkpoint LIBERO Spatial pilot; not a leaderboard score, method comparison, hard-real-time guarantee, or hardware result |
| `g03_independent_clock_object_40_20260810_001` | 40/40 completed; 39/40 task success; 6,161/6,263 ticks during inference; 5,530 execute and 733 hold ticks | exploratory cross-suite Object transfer; not pooled with G02 |
| `g04_spatial_deadline50_40_20260810_001` | 40/40 completed; 0/40 task success; 0 execute and 8,800 hold ticks; 5,474 deadline exceedances | fail-closed tight-deadline stress; not a model-quality failure or universal threshold |
| `g05_spatial_deadline150_40_20260810_001` | 40/40 completed; 0/40 task success; 2,309 execute and 6,491 hold ticks; 16 deadline exceedances | exploratory intermediate deadline point; action execution resumes but remains insufficient for this task budget |
| `g06_spatial_deadline175_40_20260810_001` | 40/40 completed; 38/40 task success; 3,942 execute and 613 hold ticks; 0 deadline/provider failures | exploratory transition point; not a universal threshold or hard-real-time result |
| `pi05_deadline_multisuite_report_720_20260810_001` | 18 cells / 720 rollouts; Spatial seeds 7/8/9 at 150/155/175/200 ms and Object seeds 7/8 at 150/175/200 ms; all source validators pass | seed- and suite-stratified simulation evidence; not a universal threshold, iid deployment estimate, or no-deadline baseline |
| `pi05_selection_heldout_report_240_20260810_001` | 120 pairs / 240 rollouts; age-aligned 114/120, response-relative 100/120; +11.67 pp; exact McNemar p=0.00936; 30-block bootstrap 95% [+1.67,+21.67] | protocol frozen before execution; exploratory result for one checkpoint/suite; both modes retain the same deadline/hold rule; no cross-model, hardware, or leaderboard claim |
| `pi05_libero_independent_clock_visual_success_001` | 1/1 curated success clip with 105/106 ticks during inference | media-only run; never pooled with the 40-rollout result |
| `cpu_runtime_completion_001` | 17/17 expected outcomes; 6 complete plans, 10 holds, 1 unrecoverable stop; zero partial prefixes; assurance-worker P95 281.2 ms | scripted/frozen/contract fixtures; best-effort Python threads; no learned checkpoint |
| `integrated_panda_fault_matrix_001` | 27/27 expected outcomes; 12 accepted, 6 verified brakes, 7 holds, 2 unrecoverable stops; 124 edges | scripted inputs; synchronous offline CPU |
| `integrated_panda_task_001` | 2/2 target reaches; 351/351 motion edges and braking boundaries; zero registered contacts, self contacts, limit violations, or torque saturation | MuJoCo torque execution; joint-waypoint task |
| π0.5 measured-age run | 88/120 baseline versus 116/120 aligned; +23.33 pp; exact McNemar p = 1.94e-6 | independent LIBERO experiment; no Panda adapter |
| `official_lerobot_roundtrip_001` | official LeRobot 0.4.4 / LeRobotDataset v3.0 reload; 3 frames, 6 image fields, Panda H×8 semantics | CPU dataset-interface roundtrip; no SO-101 conversion, policy training, or robot connection |

The task artifact reports 5.27 s and 10.20 s full-horizon offline supervision. The
fixed-open `left_finger` / `right_finger` body pair is the only explicit allowed
collision boundary and expands to 36 geom pairs. Neither fact is presented as a
hard-real-time or physical safety guarantee.

## Limitations shown on-page

- Panda task actions with 2/2 target reaches are scripted RRT-Connect
  references. G01 must not be used to relabel those task outcomes as pi0.5.
- There is no real robot, grasping success, ROS2 deployment, or hard-real-time
  scheduling claim.
- G01 connects the runtime tracks but remains separate from official LIBERO
  task-outcome evidence and reports `target_reached=false`.
- G02 is a 40-rollout simulation pilot. Its 38/40 result is not an official
  leaderboard entry or evidence of method superiority, hardware safety, or
  hard-real-time scheduling.
- The complete deadline report contains 18 registered cells and remains
  service-, clock-, checkpoint-, and simulator-specific. G04 at 50 ms stays a
  separate stress control and 200 ms is a nominal reference, not no-deadline.
- The held-out selection comparison covers one checkpoint, one suite, three
  joint seeds, and episodes 4-7. Query-0 equality does not imply later
  observations remain equal after the modes diverge.
- `response_relative_chunk` is not unguarded; it retains the same 175 ms
  observation-age deadline and fail-closed hold rule.
- The asynchronous CPU artifact validates a provider-to-supervisor software
  boundary; it is not a closed-loop task-success or deadline-guarantee result.

## Deployment

The site is static and dependency-free for GitHub Pages. All paths are relative
to `site/`, metadata points to the public repository, and the page works with a
local static server as well as the repository Pages path.
