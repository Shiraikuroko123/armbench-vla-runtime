# Engineering gap analysis for a VLA systems paper

Status checked: 2026-08-05. This is a targeted engineering review, not a claim
of exhaustive coverage. Formal proceedings and journal records are separated
from arXiv-only work. The reproducible metadata catalog is
`vla_runtime_literature_metadata.json`; the refresh and validation command is
`scripts/verify_vla_runtime_literature.py`.

![ArmBench method position and research gap](figures/armbench_vla_method_gap.png)

## Assessment

ArmBench is an independently auditable systems and evaluation study; it is not
yet a top-conference method paper. The evidence package includes an attested
official pi0.5 checkpoint, paired LIBERO conditions, frozen protocols, exact
paired tests, multiplicity correction, bootstrap intervals, complete videos,
and manifest-bound artifacts. The training-free temporal dispatcher produced a
large effect that persisted across a separately frozen three-suite validation
under deterministic 200 ms delay.

The remaining scientific gap is broader than the absence of reinforcement
learning. The closest formal comparator is RTC (NeurIPS 2025), which modifies
flow-policy inference by freezing committed actions and inpainting a
continuation. ArmBench includes a held-out 120-pair measured-age confirmation
and pi0.5 sampler extensions for hard committed-prefix projection and soft
denoised-action VJP guidance. The measured-age study pairs explicit pi0.5
flow-sampling noise, improves success by 23.33 points, and remains positive
under task-cluster sensitivity analyses.

The corrected-v3 300-rollout RTC comparison found 96/100 baseline success and
97/100 for both hard projection and RTC guidance, with Holm-adjusted `p=1.0`.
It provides exploratory motion-seam evidence but no task-success superiority.
Current evidence therefore does not establish RTC-guidance efficacy,
cross-policy generality, independently scheduled control, or real-robot
validity.

The next research question is:

> Under independently ticking inference and control, when does sampler-internal
> committed-action guidance improve temporal continuity without sacrificing
> task progress?

That question follows directly from the demonstrated runtime mechanism; a
small PPO experiment would address a different problem.

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

| Work | Status on access date | Official code snapshot | Available implementation surface |
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

## Capabilities established by comparison systems

| Route | Main intervention | Training / equipment burden | What ArmBench should learn from it |
| --- | --- | --- | --- |
| RT-2, CoRL 2023 | Co-fine-tunes vision-language and robot-action data into a generalist VLA | Large, closed training stack and robot data | Establishes the model class, but is not a practical personal-project baseline |
| OpenVLA, CoRL 2024 | Open 7B VLA and reproducible adaptation path | Substantial GPU fine-tuning, public checkpoints | Candidate second model family for cross-model runtime evidence |
| OpenVLA-OFT, RSS 2025 | Parallel continuous action generation and action chunking with optimized supervised fine-tuning | Offline training plus simulation and real ALOHA evaluation | Demonstrates that decoding and training changes must be separated from runtime-only gains |
| FAST, RSS 2025 | Compresses continuous robot actions into efficient tokens | Tokenizer and VLA training | Relevant to serving cost and horizon design, not by itself a stale-response solution |
| ConRFT, RSS 2025 | Reinforced VLA fine-tuning through a consistency-policy route | Offline/online adaptation and intervention data | A serious RL comparison, far beyond a decorative PPO baseline |
| DPPO, ICLR 2025 | Policy-gradient fine-tuning for diffusion policies | RL fine-tuning on simulated continuous-control and robot-learning tasks; the paper also reports zero-shot hardware deployment | Relevant when the research question includes reward-driven policy improvement |
| HIL-SERL, Science Robotics 2025 | Real-world online RL supported by demonstrations and human corrections | Real robot, human supervision, and online RL | Establishes the training and hardware evidence expected for real-world RL claims |
| RTC, NeurIPS 2025 | Freezes committed flow-policy actions and inpaints a consistent continuation at inference time | Requires access inside the flow-policy sampling process; includes real-robot evidence | Closest direct comparator and the present method-quality target |
| VLASH, arXiv 2025 | Rolls the robot state forward with the previous action chunk, then fine-tunes with state/action offsets for asynchronous execution | Public pi0/pi0.5 training code advertises LoRA below 12 GB, but matched LIBERO artifacts are not released | Directly targets stale state, but is an arXiv-only learned comparison rather than a training-free drop-in |
| FutureRTC, arXiv 2026 | Anticipatory conditioning and learned execution-time context | Learned prediction/adaptation modules | Raises the bar beyond time-only prefix selection; preprint evidence must be treated cautiously |
| Action ControlNet, arXiv 2026 | Lightweight delay-aware adapter for smooth asynchronous handoff | Parameter-efficient adapter training | A useful learned-adapter control, but not training-free; preprint only as of the access date |

## Current engineering strengths

ArmBench should be presented as a runtime reliability and evidence-engineering
project, not as a newly trained policy:

- It evaluates the pinned official `pi05_libero` checkpoint rather than a test
  fixture for its formal LIBERO claims.
- It preserves the training-free intervention boundary: the VLA checkpoint is
  frozen, and only action dispatch changes.
- The deterministic Spatial study and separately frozen Object, Goal, and
  LIBERO-10 external study cover 40 tasks and 600 rollouts. A distinct
  measured-age confirmation adds 240 held-out rollouts with paired policy
  noise. These are three study families, not one post-hoc pooled experiment.
- Every matched condition keeps the same initial state, task, horizon, and seed;
  success is analyzed with paired methods rather than independent proportions.
- Source, checkpoint, protocol, CSV, manifest, and video consistency are
  cross-checked by separate repository validators. Runtime failures remain in
  intention-to-test counts.

These properties make the study independently reviewable: a reviewer can run
the validators, inspect a matched video pair, recompute the statistics, and
trace the action-selection code. They do not close the publication gap because
the current causal intervention remains narrow.

The evidence lineage is traceable to repository history:

- frozen confirmatory evidence: `632c043f6d8b44450f15d50571f4e686ad20d08a`,
  followed by validated release documentation at `b0c6b39b21cc59dec843117a06db86ea0260d365`;
- cross-suite external validation: `a5232d54e46d96a27911892d9360f5a93693e612`,
  covering the separately frozen Object, Goal, and LIBERO-10 study;
- measured-age runtime core: `39faa089329c8e8466437b6c98c763d86cd52df7`,
  and auditable pilot driver: `d098dd285e3dc4b434d42888f91049f6da7cd385`,
  followed by the scored pilot at run commit
  `b1835dabf2b76714bda01eeae43516f99ddc0505`;
- held-out measured-age protocol and scored run:
  `12070625cd6f46186282317262065d015c8fbe27`.

These commits are present on the configured `origin/main`. Repository access
and validated release archives provide the review surface; local commit hashes
alone are not independently retrievable provenance.

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
pi0.5 sampling noise and independently validates every request binding. Its
120-pair primary result is positive, so measured-age suffix selection is now a
confirmed simulation result rather than only a software path. Neither study
may be used to relabel the old deterministic 200 ms result, and post-response
catch-up still falls short of independently ticking inference and control.

### 2. Fixed suffix skipping versus policy-consistent continuation

ArmBench assumes that action `k` is the appropriate command after roughly `k`
control periods. RTC instead operates inside flow inference and constructs a
continuation consistent with already committed actions. VLASH and FutureRTC add
future-state information. The present websocket API returns only a completed
chunk, so a methodologically valid RTC baseline requires an OpenPI server/model
extension. Renaming an external array slice would not implement the same
method.

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

## Role of reinforcement learning

DPPO, ConRFT, and HIL-SERL make policy-learning claims supported by substantive
training and task evaluation; HIL-SERL additionally provides real-hardware
evidence. A small PPO experiment would not be a valid control for the current
runtime method: it would address policy optimization, introduce reward,
environment-step, initialization, and training-seed variables, and would not be
comparable to the frozen pi0.5 intervention.

RL becomes justified only after choosing a different research question, such
as learning recovery after a fail-closed intervention or adapting a compact
latency predictor. At that point the minimum informative comparison includes
behavior cloning or supervised adaptation, multiple training seeds, learning
curves, matched environment steps, and a no-adaptation runtime control.

## Staged route toward the literature

### Stage A: non-oracle measured-age pilot (completed)

Freeze a new schema and run one minimal, explicitly exploratory pi0.5-LIBERO
pilot. The minimal registered design is 10 Spatial tasks x 2 fixed initial states
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
a mechanism signal and systems artifact, not a confirmatory efficacy
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

### Stage A2: held-out measured-age confirmation (completed)

The successor protocol is frozen at commit `12070625cd6f46186282317262065d015c8fbe27`
before inspecting any scored episode in its held-out split. It uses all ten
Spatial tasks, episode indices `5:17`, two adjacent counterbalanced modes, 120
matched pairs, and 240 rollouts. Every shared pair/query index receives the same
mode-independent jitter and explicit `10 x 32` pi0.5 flow-sampling noise.

The study completed all 240 rollouts with no infrastructure failure. Success
was 88/120 for `async_unguarded` and 116/120 for `latency_aligned`: +23.33
points, 32/4/84 wins/losses/ties, and exact two-sided McNemar
`p=1.941574737e-6`. The pair bootstrap 95% interval is [+15.00,+31.67].
Prespecified robustness checks remain positive: whole-task bootstrap 95%
[+10.83,+38.33], exhaustive `2^10` task sign-flip `p=0.015625`, and
leave-one-task-out effects [+17.59,+25.93]. Both condition-first strata are
positive (+13.33 and +33.33 points), although their magnitude difference is a
remaining order-sensitivity warning. This closes the measured-age evidence
stage, not the policy-internal RTC, cross-model, concurrency, or hardware gaps.

### Stage B: direct asynchronous-method baseline

The pinned OpenPI server now accepts committed actions inside the flow sampler
and exposes a clearly named RTC-style approximation on the same checkpoint and
tasks. The next direct comparison must place completed-chunk prefix selection
and policy-internal continuation under the same independently ticking latency
trace; the present overlap evaluator still blocks on each response.

Stage B0 now has an executable reference contract in
`integrations/openpi/realtime_chunking.py`. It reproduces RTC commit `9296f31`'s
fixed-width overlap window `old[:d] + new[d:E]`, the subsequent `E`-step shift
and zero padding, and all four public prefix-weight schedules. It also contains
a hard projected-flow ablation for OpenPI's opposite `t=1 -> 0` convention.

The model-side hard-projection ablation is now complete. OpenPI extension
commit `2c8e61d5fbfde4b670ae428ef4b8440d35c1c7fd` normalizes and pads LIBERO
reference actions to `10 x 32` and projects committed prefixes inside every
pi0.5 Euler step. Its G0 artifact established legacy parity, zero prefix
residual, bounded latency, and bounded memory. ArmBench commit
`baceec016c89bf1def6ea156f63d13c2c7f65d6a` then ran a 20-pair LIBERO-10
pilot with the exact fixed-width overlap scheduler and paired explicit policy
noise.

That pilot is a useful negative result. Unconditioned overlap succeeded on
19/20 pairs and hard projected overlap on 18/20, with 0/1/19 projected
wins/losses/ties and exact McNemar `p=1.0`. Projection reduced mean motion seam
from `0.10937` to `0.08689` but did not improve task success and increased the
mean gripper seam. All 40 videos and 2,165 transitions validate against an
independent float32 transcript. The project must therefore not promote hard
projection as RTC or scale this exact ablation as though efficacy were already
established.

The soft-guidance implementation is now complete. OpenPI extension commit
`54592c7148ba69bf52757385502782f80f2285e0` applies the denoised-action VJP
correction inside pi0.5's reverse-time Euler loop. The attested G0 artifact
established bitwise zero-guidance parity, a weighted model-residual ratio of
`0.3320`, guided warm wall P95 of `108.06 ms`, 6.43 GiB peak JAX bytes in use,
and exact repeatability under explicit noise.

ArmBench commit `2aef062256fc3f6257f9f58d68c3f18c07d1b0b8` then added a
reference-only-bootstrap, three-method evaluator. Its first 60-rollout v2
execution was subsequently rejected: environment reuse changed the query-0
policy images and actions in 6/20 triplets. The same failure affected 15/50
triplets in each v2 held-out seed. Those outcomes remain preserved but are
excluded from method-effect estimates.

The corrected evaluator at commit
`44c358731c5493284b74bb29eefa7d538d0f38dd` creates a fresh environment per
rollout and enforces matching query-0 policy-input, response-action,
sampling-key, and sampling-noise hashes. Its separately frozen v3 matrix
completed 300 rollouts and 100 matched triplets: unconditioned overlap succeeded
on 96/100, hard projection on 97/100, and RTC guidance on 97/100. Both
prespecified contrasts had a `+1` point risk difference and Holm-adjusted exact
McNemar `p=1.0`. Hard projection and RTC guidance reduced exploratory motion
seam by `0.023640` and `0.019524`, respectively, but this is not task-success,
safety, or superiority evidence.

The remaining direct-method gap is therefore not another blocking rollout
matrix. It is to replace blocking simulation with independently ticking
inference and control loops and then repeat a frozen comparison under genuine
response age and jitter. The existing suffix-selection rollouts remain a
different baseline because they advance `d + E` rather than exactly `E` steps
per query.

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

The next public result should satisfy all of the following acceptance criteria:

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
