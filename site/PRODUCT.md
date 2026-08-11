# VLA-Sync website product brief

## Primary job

Give both a non-specialist and a robotics hiring engineer a layered,
evidence-backed answer to four questions:

1. In plain language, what problem does the project solve?
2. What timing failure occurs when an action-chunk VLA responds late?
3. What does VLA-Sync change at runtime, without retraining the VLA?
4. Which conclusions are supported by saved artifacts, and where do they stop?

## Positioning

VLA-Sync is an auditable asynchronous runtime and evaluation system for
action-chunk VLA policies under inference latency. It is not a new VLA model,
a training framework, a real-robot safety controller, or a leaderboard entry.

The public explanation starts with a simpler formulation: the VLA proposes
what the robot should do, while VLA-Sync validates whether that proposal is
still temporally valid and executable now. This framing must always preserve the separation between
LIBERO task-timing evidence and Panda low-level constraint evidence.

The system runs inference and control on independent clocks, measures the age
of each accepted response, selects an unexpired action suffix, and enters a
fail-closed hold/refresh path when the registered deadline is exceeded.

## Evidence contract

VLA-Sync deliberately keeps two evidence tracks separate:

| Track | What it establishes | What it does not establish |
| --- | --- | --- |
| Official Physical Intelligence π0.5 VLA checkpoint for LIBERO | independent-clock closed-loop execution, deadline behavior, and frozen action-selection comparisons | cross-model generality, real-robot deployment, or a universal deadline |
| Seven-DoF MuJoCo Panda substrate | RRT-Connect references, OSQP projection, continuous collision checks, braking, and recomputable execution traces | π0.5 VLA Panda task success or hardware safety |

The hero montage is a synchronized, equal-size smoke comparison from the
LIBERO track. Both panes use seed 7, task 0, episode 4, and the same initial
state; both succeed. The pair illustrates the two action-selection semantics
but is not included in the frozen-240 formal statistics. Panda evidence stays
in its own rollout tab and is never pooled into a π0.5 task-success claim.

## Decisive results

| Study | Result | Boundary |
| --- | --- | --- |
| Frozen held-out selection | 120 pairs / 240 rollouts; age-aligned suffix 114/120, response-relative chunk 100/120; +11.67 pp; exact McNemar p=0.0093553; 30-block bootstrap 95% CI [+1.67,+21.67] | one checkpoint family, one LIBERO suite, three joint seeds |
| Registered deadline matrix | 18 independently validated cells / 720 rollouts; Spatial and Object suites; all provider-failure counts zero | service-, clock-, checkpoint-, simulator-, and protocol-specific |
| Independent-clock pilot | 40/40 complete, 38/40 successful; 4,521/4,623 control ticks overlap inference | simulation pilot, not a leaderboard result |
| Panda scripted execution | 2/2 target reaches; 351/351 motion and braking boundaries; zero registered contacts, limit violations, or torque saturation | scripted RRT-Connect references, not π0.5 VLA competence |
| Optimized Panda CPU assurance | Original 180-case audit: 20 ms profile 1/90 execute, 66/90 constraint-safe, zero unsafe publications, zero partial prefixes, 23.888 ms P95 | frozen minimum-rule `go`, retained as the v0.2.0 historical result |
| Optimized CPU repeatability | Six fresh-process 180-case reruns: idle execute `[1,0,0]`, four-worker load `[0,0,0]`; every trial retains 66/90 safe candidates and zero unsafe/prefix exposure | fail-closed publication repeats; stable or deployable 20 ms compliance is not supported |
| Atomic certification-window repeatability | Six paired fresh-process audits: `H=1` execute `[90,90,90]` idle and `[89,84,86]` under four-worker load; paired `H=10` execute `[0,0,0]`; zero unsafe or partial windows | one-action certification is the repeatable CPU publication candidate on this host; source-chunk atomicity changes and hard real time is not established |
| Reproducibility surface | 836 pytest passes, 50 CPU-acceptance passes, 56 registered artifacts / 4,270 catalogued files | validates saved evidence and software contracts; it does not rerun the checkpoint on CPU |

## Information architecture

1. **Hero:** literal project identity, a plain-language problem statement,
   artifact rail, authentic two-track simulation montage, and four decisive
   metrics with non-specialist labels.
2. **60-second explanation:** see, wait, align, check, and execute/hold flow;
   a four-term glossary; and an explicit LIBERO/Panda evidence split.
3. **Abstract:** concise system definition and explicit non-claims.
4. **Motivation:** observation age, in-flight inference, stale action chunks,
   and deadline behavior on one timing diagram.
5. **Method:** perception/policy/runtime flow followed by visibly separate
   LIBERO and Panda evidence branches.
6. **Evaluation:** held-out paired result, exact statistics, seed table,
   registered deadline curve, optimized CPU repeatability, and the paired
   atomic-window audit.
7. **Rollouts:** validator-retained failures, the independent-clock pilot,
   and the Panda execution substrate in task tabs.
8. **Reproduce:** GPU-free validation commands and the current verification
   ledger.
9. **Engineering roadmap:** established conclusions and prioritized
   cross-model, online-repair, and deployment-realism improvements.

Every technical section includes a visible “this section answers” sentence so
readers can decide whether to stop at the explanation or inspect the evidence.

## Audience and actions

- Hiring engineers inspect the system boundary, method, failures, and code.
- Researchers inspect conditions, paired statistics, manifests, and reports.
- Reproducers run the CPU acceptance path without renting another GPU.

Primary actions are Code, Results, Architecture, and Evidence. The site does
not expose Paper, Model, or Dataset actions because VLA-Sync has not released
those artifacts as project-owned research outputs.

## Deployment

The site is dependency-free static HTML/CSS/JavaScript for GitHub Pages. Media
uses local posters, one optional above-fold autoplay, and below-fold lazy
attachment. The public repository is the canonical source; the local site can
be served from `site/` with any static HTTP server.
