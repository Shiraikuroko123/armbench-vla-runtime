# ArmBench Website Product Brief

## Product

ArmBench is a public technical project page for a runtime and evaluation
platform for action-chunk vision-language-action policies. The page must make
one idea understandable without prior robotics knowledge: inference consumes
time, so the first actions in a returned chunk may already be stale when they
reach the controller.

ArmBench measures the age of the source observation, selects an unexpired
action suffix, and fails closed when the remaining horizon or deadline is
invalid. It also preserves protocols, matched trials, videos, manifests, and
validators so a reviewer can audit the result rather than trust a demo reel.

## Audience

- Robotics and embodied-AI hiring teams evaluating engineering depth.
- VLA and robot-learning researchers checking the method and statistics.
- Engineers who want to inspect, validate, or extend the repository.
- General technical visitors who need a concise explanation before the details.

The public page contains no resume, phone number, private email address, cloud
credentials, or personally identifying information beyond the public GitHub
account that owns the repository.

## Core Story

1. Show the system moving in the first viewport.
2. Explain stale action prefixes with an interactive 50 ms control timeline.
3. Show the frozen Physical Intelligence pi0.5 VLA results, including a matched
   baseline-failure/aligned-success video pair.
4. Report both positive and null results. The corrected RTC comparison must say
   that it establishes no task-success superiority.
5. Separate the official-checkpoint LIBERO path from the local seven-DoF
   MuJoCo Panda validation path.
6. End with commands and links a reviewer can use to verify the artifacts.

## Verified Claims

- Held-out measured-age study: 120 matched pairs, 88/120 baseline successes and
  116/120 aligned successes, +23.33 percentage points, exact two-sided McNemar
  p=1.941574737e-6.
- Cross-suite deterministic-delay validation: 300 rollouts / 150 matched pairs,
  83/150 baseline successes and 141/150 aligned successes.
- Corrected-v3 RTC comparison: 300 rollouts / 100 matched triplets, with
  96/100, 97/100, and 97/100 successes. Holm-adjusted p=1.0; no task-success
  superiority is supported.
- The local MuJoCo path exercises planning, tracking, action validation, fault
  injection, and bounded response on a seven-DoF Panda model.

## Claim Boundaries

- No pi0.5 training or fine-tuning.
- Simulation-only official-checkpoint evidence.
- Blocking inference with simulated controller catch-up, not hard real time.
- No verified pi0.5-to-Panda end-to-end deployment.
- No real robot, Isaac Lab, ROS2, safety PLC, or collision-safety certification.
- Panda and LIBERO outcomes remain separate evidence domains.

## Delivery

The site is dependency-free HTML, CSS, and JavaScript deployed through GitHub
Pages. It is bilingual, keyboard accessible, responsive from 320 px mobile to
wide desktop, usable with reduced motion, and built entirely from repository
evidence. No analytics or third-party runtime assets are loaded.

## Reference Synthesis

The visual study covered OpenVLA, Open X-Embodiment / RT-X, Diffusion Policy,
Mobile ALOHA, Octo, 3D Diffusion Policy, and Physical Intelligence's pi0.5 page.
The useful shared pattern is evidence-first communication: working robot media,
one method diagram, quantitative comparison, code/paper links, and limitations.
ArmBench adopts that evidence order but replaces the conventional academic
paper template with a runtime-observatory interface suited to its timing and
verification contribution.
