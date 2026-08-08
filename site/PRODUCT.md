# ArmBench Website Product Brief

## Primary Job

ArmBench is a small robotics engineering project page. It summarizes saved VLA
experiments and Panda simulation checks without presenting the repository as a
product, safety system, or complete research platform. The page helps a visitor
answer three questions quickly:

1. What timing problem does ArmBench study?
2. What experiments and software modules are included?
3. What has not been implemented or tested?

## Content Contract

- **Identity:** ArmBench, a project for VLA action-chunk timing and Panda
  simulation checks.
- **Method:** skip the stale prefix of an action chunk; request a new action or
  hold when the remaining horizon is invalid.
- **Test paths:** the official Physical Intelligence π0.5 checkpoint in LIBERO
  simulation, plus a separate seven-DoF Panda path in MuJoCo.
- **Shown materials:** Panda rollout footage, a matched π0.5-LIBERO example,
  paired statistics, and saved result files.
- **Repository files:** source, result-file index, statistical scripts,
  debugging guide, check commands, license, and third-party notices.
- **Status:** simulation-only, no training or fine-tuning, updated 2026-08-08.

## Audience

- Embodied-AI and robotics hiring teams reviewing a student project.
- VLA and robot-learning engineers checking experiment settings and statistics.
- Developers deciding whether they can run or extend the repository.
- Technical visitors who need the central timing issue without reading a paper.

The public page contains no resume, phone number, private email address, cloud
credential, raw home path, or personally identifying data beyond the public
GitHub account that owns the repository.

## Results Order

The first viewport shows the project name, a plain description, a MuJoCo Panda
rollout, its simulation conditions, and links to the results and repository.
The file links remain visible as the next-scroll cue.

The page then presents three result groups:

| Study | Scope | Reported result |
| --- | --- | --- |
| Measured-age primary | Spatial, 120 matched pairs | 88/120 to 116/120; +23.33 pp |
| Cross-suite extension | Object, Goal, LIBERO-10; 150 pairs | 83/150 to 141/150 |
| Corrected-v3 RTC extension | 100 matched triplets | 96/97/97; Holm p = 1.0 |

The RTC row is intentionally a null result. It is not presented as another
layer of support for the measured-age result.

## Page Structure

1. Full-bleed Panda simulation hero with explicit test conditions.
2. Code, result files, statistics, and local-run file dock.
3. Tabbed measured-age, cross-suite, and RTC results.
4. Synchronized matched-video player with selection disclosure.
5. Interactive observation-age timeline using the frozen 50 ms rule.
6. Two independent testbeds and their unclosed integration gap.
7. Implemented modules: provider adaptation, LeRobot-style watchdog replay,
   trajectory repair, asynchronous Panda control, result-file checks, and
   clearance-based collision sampling.
8. Scripted non-learned fault tests with links to saved files.
9. Checked scope and unfinished work.
10. GPU-free commands for checking saved measured-age and RTC results.

## Reported Results

- Measured age: 120 matched pairs, 88/120 baseline and 116/120 aligned,
  +23.33 percentage points, exact two-sided McNemar p = 1.94e-6, and pair
  bootstrap 95% interval [+15.00, +31.67] pp.
- Cross-suite: 300 rollouts / 150 matched pairs, 83/150 baseline and 141/150
  aligned. The pooled row is descriptive; inference remains suite-level.
- Corrected-v3 RTC: 300 rollouts / 100 matched triplets with 96/100, 97/100,
  and 97/100 successes. Holm-adjusted p = 1.0; superiority is unsupported.
- Panda: local planning, time parameterization, torque-limited joint PD,
  action validation, collision sampling, and bounded scripted fault responses.
- Provider-neutral action contract: a synthetic OpenVLA-OFT-named fixture maps
  `6x7` to `6x8` and rejects 5/5 semantic mismatches; no OpenVLA-OFT checkpoint
  was executed.
- LeRobot-style boundary: a five-frame in-memory replay executes three valid
  commands, holds a stale observation, preserves a fault latch, and recovers
  only after explicit reset; no official LeRobot runtime is used.
- Braking repair and swept audit: 270 paired offline cases satisfy all
  registered constraints after repair, while 72 seeded static-obstacle edges
  produce zero false-safe decisions against a denser sampled oracle.
- Asynchronous Panda loop: 27 wall-clock CPU cases connect dual-camera capture,
  a blocking scripted policy worker, temporal dispatch, trajectory repair, and
  torque control. Braking-invariant mode records 9/9 physical predicates and
  zero abrupt-stop violations, but reaches the target in only 1/9 cases.

## Limits

- No π0.5 training or fine-tuning.
- No real robot, hard-real-time, or collision-safety certification.
- Blocking inference with post-response catch-up simulation, not concurrent
  inference and control.
- No verified π0.5-to-Panda action adapter or end-to-end deployment.
- Scripted fault cards do not run π0.5 and do not establish VLA safety.
- No cross-checkpoint generalization, Isaac Lab, or ROS2 claim.
- LIBERO and Panda outcomes remain separate evidence domains.
- The provider, watchdog, and result-file modules are software checks only;
  they are not hardware drivers or safety-controller tests.

## Delivery

The site is dependency-free HTML, CSS, and JavaScript deployed with GitHub
Pages. It is bilingual, keyboard accessible, responsive from 320 px upward,
and usable with reduced motion or data saving enabled. No analytics, remote
fonts, generated robot imagery, or third-party runtime assets are loaded.
The deployment workflow checks JavaScript syntax and validates the HTML before
publishing.

The structure was informed by OpenVLA, Open X-Embodiment / RT-X, Diffusion
Policy, Mobile ALOHA, Octo, 3D Diffusion Policy, and Physical Intelligence's
π0.5 page. No template code or page media was copied.
