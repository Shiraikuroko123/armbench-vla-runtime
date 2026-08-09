# ArmBench Website Product Brief

## Primary job

Show a reviewer or robotics hiring engineer exactly what the project currently
does at the policy-to-controller boundary, how to reproduce it on CPU, and what
the simulation evidence does not cover.

## One-sentence claim

ArmBench is a CPU-recomputable reference supervisor that accepts a complete
seven-DoF Panda action chunk only after timing, OSQP kinematic projection,
continuous collision, and dynamics-feasible braking checks, then sends the
accepted trajectory to MuJoCo torque execution.

## Information architecture

1. Full-bleed Panda simulation hero with explicit `SIMULATION` and
   `SCRIPTED RRT-CONNECT` boundary labels.
2. Artifact bar linking source, asynchronous CPU report, fault report, and task
   report.
3. Current evidence strip and three artifact ledgers.
4. Assurance flow and atomic decision table.
5. Registered 27-case fault matrix with outcome distribution.
6. Two assured MuJoCo tasks, physical predicates, gripper collision boundary,
   and offline latency disclosure.
7. Separate π0.5-LIBERO evidence with synchronized authentic comparison clips,
   plus an official LeRobotDataset v3.0 Panda H×8 roundtrip.
8. One-command full local CPU acceptance, focused Panda replay commands, a
   checked-in VS Code workspace, and a four-step acceptance path.
9. Verified 17-case asynchronous CPU boundary disclosure.
10. Proven / not-claimed scope block and architecture link.

## Audience

- Embodied-AI and robotics hiring teams assessing implementation depth.
- Researchers checking whether a result is recomputable and correctly scoped.
- Engineers who want to run the saved reports without a GPU or cloud host.

## Results contract

| Artifact | Exact result | Boundary |
| --- | --- | --- |
| `cpu_runtime_completion_001` | 17/17 expected outcomes; 6 complete plans, 10 holds, 1 unrecoverable stop; zero partial prefixes; assurance-worker P95 438 ms | scripted/frozen/contract fixtures; best-effort Python threads; no learned checkpoint |
| `integrated_panda_fault_matrix_001` | 27/27 expected outcomes; 12 accepted, 6 verified brakes, 7 holds, 2 unrecoverable stops; 124 edges | scripted inputs; synchronous offline CPU |
| `integrated_panda_task_001` | 2/2 target reaches; 351/351 motion edges and braking boundaries; zero registered contacts, self contacts, limit violations, or torque saturation | MuJoCo torque execution; joint-waypoint task |
| π0.5 measured-age run | 88/120 baseline versus 116/120 aligned; +23.33 pp; exact McNemar p = 1.94e-6 | independent LIBERO experiment; no Panda adapter |
| `official_lerobot_roundtrip_001` | official LeRobot 0.4.4 / LeRobotDataset v3.0 reload; 3 frames, 6 image fields, Panda H×8 semantics | CPU dataset-interface roundtrip; no SO-101 conversion, policy training, or robot connection |

The task artifact reports 5.27 s and 10.20 s full-horizon offline supervision. The
fixed-open `left_finger` / `right_finger` body pair is the only explicit allowed
collision boundary and expands to 36 geom pairs. Neither fact is presented as a
hard-real-time or physical safety guarantee.

## Limitations shown on-page

- Panda task actions are scripted RRT-Connect references. The asynchronous CPU
  matrix uses scripted, frozen, and provider-compatible fixtures, not learned
  VLA outputs.
- There is no real robot, grasping success, ROS2 deployment, or hard-real-time
  scheduling claim.
- π0.5-LIBERO and Panda-MuJoCo evidence remain separate tracks.
- The asynchronous CPU artifact validates a provider-to-supervisor software
  boundary; it is not a closed-loop task-success or deadline-guarantee result.

## Deployment

The site is static and dependency-free for GitHub Pages. All paths are relative
to `site/`, metadata points to the public repository, and the page works with a
local static server as well as the repository Pages path.
