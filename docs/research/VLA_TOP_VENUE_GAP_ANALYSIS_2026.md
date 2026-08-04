# VLA top-venue engineering gap analysis

Status checked: 2026-08-05. This is a targeted engineering review, not a claim
of exhaustive coverage. Formal proceedings and journal records are separated
from arXiv-only work. The reproducible metadata catalog is
`vla_runtime_literature_metadata.json`; the refresh and validation command is
`scripts/verify_vla_runtime_literature.py`.

![ArmBench method position and research gap](figures/armbench_vla_method_gap.png)

## Bottom line

ArmBench is now a credible systems and evaluation project, but it is not yet a
top-conference method paper. Its strongest evidence is unusually disciplined
for a portfolio project: an attested official pi0.5 checkpoint, paired LIBERO
conditions, frozen protocols, exact paired tests, multiplicity correction,
bootstrap intervals, complete videos, and manifest-bound artifacts. The
training-free temporal dispatcher also produced a large effect that persisted
across a separately frozen three-suite validation under deterministic 200 ms
delay.

The scientific gap is not simply "no RL." The closest formal comparator is RTC
(NeurIPS 2025), which changes flow-policy inference itself by freezing already
committed actions and inpainting a continuation. ArmBench currently sees only a
completed action chunk. It now has an exploratory measured-age suffix-selection
pilot and a frozen held-out successor protocol. The pilot did not pair OpenPI's
mutable policy-sampling RNG, so its efficacy result remains a mechanism signal,
not confirmatory evidence. ArmBench cannot yet claim policy-internal
replanning, multi-model generality, or real-robot validity.

The next defensible project thesis is:

> Can a frozen VLA remain useful under measured, variable response age when a
> runtime must choose a fresh suffix or fail closed before its deadline?

That question is closer to the project's demonstrated strength than adding a
small PPO run that does not address stale actions.

## Retrieval and evidence classes

The bounded catalog contains eleven primary works selected because they expose
a concrete engineering route relevant to VLA runtime reliability:

- formal VLA foundations and adaptation: RT-2, OpenVLA, OpenVLA-OFT, FAST, and
  ConRFT;
- formal reinforcement-learning routes: DPPO and HIL-SERL;
- formal asynchronous runtime work: RTC;
- recent arXiv-only asynchronous routes: VLASH, FutureRTC, and Action
  ControlNet.

Publication identity was checked against official RSS, PMLR, ICLR, NeurIPS, or
publisher pages where available. arXiv was used for versioned preprint metadata,
OpenAlex for identifier reconciliation, and Crossref for registered DOI metadata.
An arXiv DOI never promotes a work to "formal." API failures are retained in
the metadata artifact rather than interpreted as absence.

### Status and code snapshot

The table below freezes both publication status and public-code state to the
2026-08-05 access date. A repository HEAD is provenance for this audit, not a
claim that the authors designated that commit as a paper release.

| Work | Status on access date | Official code snapshot | What is actually runnable |
| --- | --- | --- | --- |
| RTC | NeurIPS 2025 formal paper ([venue](https://neurips.cc/virtual/2025/poster/117747)) | [Kinetix repository at `9296f31`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/commit/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b) | Public code reproduces the Kinetix route and supplies roughly 60 GiB of expert assets; it does not expose the paper's real-robot pi0.5 stack as a drop-in ArmBench server |
| OpenVLA-OFT | RSS 2025 formal paper ([proceedings](https://www.roboticsproceedings.org/rss21/p017.html)) | [official repository at `e4287e9`](https://github.com/moojink/openvla-oft/commit/e4287e94541f459edc4feabc4e181f537cd569a8) | Pretrained checkpoints and native LIBERO evaluation are public; inference needs about 16 GB VRAM, while the documented training recipes need about 25-62 GB per GPU and the paper recipe used up to 8 A100s |
| DPPO | ICLR 2025 formal paper ([venue](https://iclr.cc/virtual/2025/poster/28475)) | [official repository at `cc7234a`](https://github.com/irom-princeton/dppo/commit/cc7234ad7ff39a8f32de3af903606723a16f0648) | Pretrained policies and RL fine-tuning configs are public for Gym, Robomimic, D3IL, and Furniture-Bench; it expects Linux/NVIDIA and uses 40-50 parallel CPU environments or Isaac Gym for Furniture-Bench |
| HIL-SERL | Science Robotics 2025 formal paper ([DOI](https://doi.org/10.1126/scirobotics.ads5033)) | [official repository at `c32939b`](https://github.com/rail-berkeley/hil-serl/commit/c32939bccb65f3b8c43a9f9add3d322d4ab0264a) | The actor/learner, reward-classifier, demonstration, and human-correction stack is public, but the intended evidence path requires a Franka setup, operator time, and real online training |
| VLASH | arXiv-only preprint, `2512.01031`, on the access date ([preprint](https://arxiv.org/abs/2512.01031)) | [official repository at `22cbabf`](https://github.com/mit-han-lab/vlash/commit/22cbabfee0f57874987c75a35a7dac129e695db0) | pi0/pi0.5 asynchronous training and inference examples are public and LoRA is advertised below 12 GB; the repository snapshot has no released LIBERO config or tagged checkpoint, so a matched ArmBench reproduction still needs integration and training work |

RTC is not a same-command baseline for this repository. Its public evaluator
loads RTC's own Kinetix flow checkpoints; ArmBench calls a completed-chunk
OpenPI websocket boundary. A faithful same-checkpoint comparison therefore
requires exposing committed actions and timing inside OpenPI's flow sampler.
OpenVLA-OFT is the more practical second model family because its four LIBERO
checkpoints and evaluator are public, but it tests cross-model portability, not
policy-internal RTC inpainting.

## What the strongest systems actually add

| Route | Main intervention | Training / equipment burden | What ArmBench should learn from it |
| --- | --- | --- | --- |
| RT-2, CoRL 2023 | Co-fine-tunes vision-language and robot-action data into a generalist VLA | Large, closed training stack and robot data | Establishes the model class, but is not a practical personal-project baseline |
| OpenVLA, CoRL 2024 | Open 7B VLA and reproducible adaptation path | Substantial GPU fine-tuning, public checkpoints | Best route to a second model family and cross-model runtime evidence |
| OpenVLA-OFT, RSS 2025 | Parallel continuous action generation and action chunking with optimized supervised fine-tuning | Offline training plus simulation and real ALOHA evaluation | Demonstrates that decoding and training changes must be separated from runtime-only gains |
| FAST, RSS 2025 | Compresses continuous robot actions into efficient tokens | Tokenizer and VLA training | Relevant to serving cost and horizon design, not by itself a stale-response solution |
| ConRFT, RSS 2025 | Reinforced VLA fine-tuning through a consistency-policy route | Offline/online adaptation and intervention data | A serious RL comparison, far beyond a decorative PPO baseline |
| DPPO, ICLR 2025 | Policy-gradient fine-tuning for diffusion policies | RL fine-tuning on simulated continuous-control and robot-learning tasks; the paper also reports zero-shot hardware deployment | Useful only if the research question becomes reward-driven policy improvement |
| HIL-SERL, Science Robotics 2025 | Real-world online RL supported by demonstrations and human corrections | Real robot, human supervision, and online RL | Shows why real-world RL evidence is expensive and why a toy simulation run is not equivalent |
| RTC, NeurIPS 2025 | Freezes committed flow-policy actions and inpaints a consistent continuation at inference time | Requires access inside the flow-policy sampling process; includes real-robot evidence | Closest direct comparator and the present method-quality target |
| VLASH, arXiv 2025 | Rolls the robot state forward with the previous action chunk, then fine-tunes with state/action offsets for asynchronous execution | Public pi0/pi0.5 training code advertises LoRA below 12 GB, but matched LIBERO artifacts are not released | Directly targets stale state, but is an arXiv-only learned comparison rather than a training-free drop-in |
| FutureRTC, arXiv 2026 | Anticipatory conditioning and learned execution-time context | Learned prediction/adaptation modules | Raises the bar beyond time-only prefix selection; preprint evidence must be treated cautiously |
| Action ControlNet, arXiv 2026 | Lightweight delay-aware adapter for smooth asynchronous handoff | Parameter-efficient adapter training | A useful learned-adapter control, but not training-free; preprint only as of the access date |

## Where ArmBench is already strong

ArmBench should be presented as a runtime reliability and evidence-engineering
project, not as a newly trained policy:

- It evaluates the pinned official `pi05_libero` checkpoint rather than a mock
  policy for its formal LIBERO claims.
- It preserves the training-free intervention boundary: the VLA checkpoint is
  frozen, and only action dispatch changes.
- The confirmatory Spatial study and separately frozen Object, Goal, and
  LIBERO-10 external study cover 40 tasks and 600 rollouts in total. These are
  two study families, not one post-hoc pooled experiment.
- Every matched condition keeps the same initial state, task, horizon, and seed;
  success is analyzed with paired methods rather than independent proportions.
- Source, checkpoint, protocol, CSV, manifest, and video consistency are
  cross-checked by separate repository validators. Runtime failures remain in
  intention-to-test counts.

This is enough for a strong graduate-level portfolio project because an
interviewer can run the validator, inspect a matched video pair, recompute the
statistics, and trace the action-selection code. It is not enough for a top
paper because the current causal intervention is still narrow.

The portfolio evidence itself is traceable to repository history:

- frozen confirmatory evidence: `632c043f6d8b44450f15d50571f4e686ad20d08a`,
  followed by validated release documentation at `b0c6b39b21cc59dec843117a06db86ea0260d365`;
- cross-suite external validation: `a5232d54e46d96a27911892d9360f5a93693e612`,
  covering the separately frozen Object, Goal, and LIBERO-10 study;
- measured-age runtime core: `39faa089329c8e8466437b6c98c763d86cd52df7`,
  and auditable pilot driver: `d098dd285e3dc4b434d42888f91049f6da7cd385`,
  followed by the scored pilot at run commit
  `b1835dabf2b76714bda01eeae43516f99ddc0505`.

Those commits are present on the configured `origin/main`, but the repository
returned HTTP 404 to an unauthenticated link check on the access date. An
interviewer cannot audit them until the repository or a release is made public,
or access is granted. Local hashes alone are not public provenance.

ArmBench has no formal paper or arXiv preprint. These are released engineering
and experimental artifacts, not a publication record.

## Exact gap to a top-venue method paper

### 1. Oracle timing versus measured timing

The frozen ArmBench study injects four known delay steps and gives the same
number directly to the dispatcher. That isolates a causal mechanism cleanly,
but it is an oracle protocol. A deployed runtime observes timestamps and
response arrival; it does not receive the experimenter's hidden delay label.

The measured-age core added after the frozen study closes the software part of
this gap. It measures end-to-end observation age, applies a pre-registered
floor or conservative-ceil conversion, and checks whether the requested suffix
remains inside the action horizon. Its separately registered 40-rollout pi0.5
pilot completed with valid artifacts. That legacy artifact proves the runtime
and timing path, but its two modes consumed different draws from the server's
mutable policy RNG. The held-out successor supplies mode-independent explicit
pi0.5 sampling noise and independently validates every request binding. Neither
study may be used to relabel the old 200 ms result.

### 2. Fixed suffix skipping versus policy-consistent continuation

ArmBench assumes that action `k` is the appropriate command after roughly `k`
control periods. RTC instead operates inside flow inference and constructs a
continuation consistent with already committed actions. VLASH and FutureRTC add
future-state information. The present websocket API returns only a completed
chunk, so an honest RTC baseline requires an OpenPI server/model extension; it
cannot be recreated by renaming an external array slice.

### 3. One VLA checkpoint versus method generality

The formal result uses one pi0.5-LIBERO checkpoint. Four suites establish task
breadth, not model breadth. A method claim should include at least one genuinely
different open checkpoint family, such as OpenVLA-OFT through its native
evaluation stack, and should preserve each model's action semantics rather than
forcing incompatible outputs into one wrapper.

### 4. Simulation versus deployment evidence

LIBERO is appropriate for controlled paired statistics, but RTC and HIL-SERL
include hardware evidence. A real deployment needs independently ticking
control and inference loops, timestamped sensor capture, watchdogs, reconnect
behavior, command limits, and an emergency-stop boundary. Post-hoc advancement
of a blocked simulator is a useful model, not a hard-real-time controller.

### 5. Temporal feasibility versus physical feasibility

The formal LIBERO path checks temporal suffix availability but does not project
pi0.5 end-effector actions through joint, velocity, acceleration, and continuous
collision constraints. ArmBench has these ingredients in its separate MuJoCo
Panda path, but the two evidence paths are not yet one causal experiment.

## Why adding RL now would not automatically improve the project

DPPO, ConRFT, and HIL-SERL make explicit policy-learning claims supported by
substantive training and task evaluation; HIL-SERL additionally provides
real-hardware evidence. Running PPO for a few hours on a toy reward would add a
framework name while weakening the central story: it would not explain stale
action chunks, would not be comparable to the frozen pi0.5 result, and would
introduce reward-design, environment-step, initialization, and training-seed
confounds. It would also turn the clean question "does dispatch alignment fix
an old action?" into the different question "did policy optimization learn a
better policy?"

RL becomes justified only after choosing a different research question, such
as learning recovery after a fail-closed intervention or adapting a compact
latency predictor. At that point the minimum credible comparison includes
behavior cloning or supervised adaptation, multiple training seeds, learning
curves, matched environment steps, and a no-adaptation runtime control.

## Staged route toward the literature

### Stage A: non-oracle measured-age pilot (completed)

Freeze a new schema and run one minimal, explicitly exploratory pi0.5-LIBERO
pilot. The smallest useful design is 10 Spatial tasks x 2 fixed initial states
x 2 modes (`async_unguarded` and measured-age `latency_aligned`), for 20 paired
groups and 40 scored rollouts. Before randomizing a scored condition, run three
attested, unscored warm-up queries with the same checkpoint and tensor shapes.
Use a mode-independent SHA-256 keyed jitter schedule over 0/40/80/160 ms,
conservative-ceil rounding at 20 Hz, a fixed deadline, and bounded hold-refresh.
Counterbalance mode order within each pair.

The registered run completed all 40 attempts in 413.917 seconds of Compose
execution on the working RTX 4090/OpenPI setup; it did not train the policy.
The artifact preserves the frozen protocol, warm-up log, server
launch/commit/checkpoint attestation, 40 episode rows, 810 scored query rows,
40 videos, paired summary, failure taxonomy, manifests, and separate read-only
validation and analysis reports.

The exploratory result was 14/20 versus 19/20 successes, a +25-point paired
difference with bootstrap 95% interval [+10,+45], 5/0/15 aligned wins/losses/
ties, and exact two-sided McNemar `p=0.0625`. Mean policy queries fell from
24.6 to 15.9. Both modes recorded one approximately 251 ms deadline/horizon
event; the aligned mode executed its registered hold-refresh path once. This is
a credible mechanism signal and systems artifact, not a confirmatory efficacy
claim. In particular, keyed jitter was paired but the legacy server's mutable
policy-sampling RNG was not. The artifact remains valid for what it recorded;
the unpaired latent policy noise is a design limitation, not a corrupt-file
error.

The operational acceptance bar must be fixed before launch:

- all 40 assigned attempts remain in intention-to-test accounting, including
  runtime failures;
- all query ages, ceil offsets, deadline decisions, and selected suffixes are
  recomputable from raw timestamps;
- baseline and candidate use the same keyed jitter value for every common
  pair/query index, independent of execution order;
- future efficacy runs also use the same explicit pi0.5 sampling noise for
  every common pair/query index and persist the request and realized-noise
  hashes;
- every artifact and video referenced by the manifest exists, hashes correctly,
  and the independent validator returns `valid=true`;
- no efficacy claim is made from this pilot alone. Its success-rate difference,
  intervention rate, deadline misses, horizon overruns, refreshes, policy-query
  count, and age P50/P95/max determine whether a larger confirmatory matrix is
  scientifically justified.

Historical external artifacts contain 8,431 recorded policy queries. Their
suite P95 client times are about 82-83 ms, but each suite also has an
approximately 18.5 s first-query outlier consistent with compilation or
warm-up.
This is calibration evidence, not a registered outcome. A measured-latency
study must run an attested, unscored warm-up before condition randomization or
the first assigned mode will absorb a severe order effect.

### Stage A2: held-out measured-age confirmation (frozen)

The successor protocol is frozen at commit `12070625cd6f46186282317262065d015c8fbe27`
before inspecting any scored episode in its held-out split. It uses all ten
Spatial tasks, episode indices `5:17`, two adjacent counterbalanced modes, 120
matched pairs, and 240 rollouts. Every shared pair/query index receives the same
mode-independent jitter and explicit `10 x 32` pi0.5 flow-sampling noise.

The sole primary test is a two-sided exact McNemar test at `.05`; a positive
claim additionally requires more aligned-only than baseline-only successes.
The protocol pre-specifies 10,000 whole-task bootstrap resamples, exhaustive
`2^10` task sign flips, leave-one-task-out effects, condition-first strata, and
a no-interim-inspection rule. Exact power is 0.824 under the lower registered
`.18/.05` discordance alternative and 0.904 under the primary `.20/.05`
alternative. Until all 240 rollouts and both validators finish, its efficacy
status is deliberately unresolved.

### Stage B: direct asynchronous-method baseline

Extend the pinned OpenPI server so the evaluator can pass committed actions and
execution timestamps into the flow sampler. Reproduce RTC or a clearly named
approximation on the same checkpoint and tasks. Compare completed-chunk prefix
selection against policy-internal continuation under the identical latency
trace. This is the shortest route from a strong project to a research-method
contribution.

### Stage C: cross-model and cross-simulator validity

Add OpenVLA-OFT or another open action-chunk model, then repeat a reduced frozen
matrix. Port only the evaluator contract to Isaac Lab if parallel GPU physics
or sensor randomization earns its cost; changing simulator without a new
scientific control is not itself a contribution.

### Stage D: physical constraints and hardware

Join the LIBERO temporal supervisor with the Panda feasibility layer through a
declared action adapter. Add IK/QP projection, continuous collision checking,
and deadline-bounded hold/recovery. A LeRobot-compatible hardware adapter or a
small real-Panda collaboration would then test whether the measured-age logic
survives real scheduling and sensor timestamps.

## Acceptance bar for future claims

The next public result should not be called complete until all of the following
are true:

- old deterministic evidence still validates byte-for-byte under its original
  schema;
- the new protocol is frozen before rollouts and has a distinct run ID;
- warm-up, jitter generation, timestamp origin, rounding, deadline, horizon,
  retry limit, and failure inclusion are recorded;
- paired modes receive the same initial state and keyed jitter sequence;
- every measured offset and decision can be recomputed from raw query records;
- a tampered age, offset, or deadline decision is rejected by a separate
  read-only validator;
- the dashboard labels pilot, confirmatory, and external evidence separately;
- any top-paper comparison is run from its real implementation or explicitly
  labeled an approximation.

## Primary identifiers

- RT-2: PMLR 229, <https://proceedings.mlr.press/v229/zitkovich23a.html>
- OpenVLA: PMLR 270, <https://proceedings.mlr.press/v270/kim25c.html>
- OpenVLA-OFT: <https://doi.org/10.15607/RSS.2025.XXI.017>
- FAST: <https://doi.org/10.15607/RSS.2025.XXI.012>
- ConRFT: <https://doi.org/10.15607/RSS.2025.XXI.019>
- DPPO: <https://iclr.cc/virtual/2025/poster/28475>
- HIL-SERL: <https://doi.org/10.1126/scirobotics.ads5033>
- RTC: <https://neurips.cc/virtual/2025/poster/117747>
- VLASH: <https://arxiv.org/abs/2512.01031>
- FutureRTC: <https://arxiv.org/abs/2607.24008>
- Action ControlNet: <https://arxiv.org/abs/2606.25985>
