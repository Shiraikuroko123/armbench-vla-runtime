# pi0.5 RTC overlap: held-out 300-rollout protocol

> **Historical v2 protocol — superseded.** The v2 execution was rejected after
> a query-0 pairing audit found environment carryover. Do not use its outcomes
> for method-effect claims. The active replacement is
> [`RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md`](RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md),
> whose corrected 300-rollout matrix is complete.

Status: analysis plan frozen on 2026-08-05 before inspecting any held-out
outcomes. The first seed job had started when this file was committed, so this
is an auditable prospective project protocol, not a third-party preregistration.

## Question and claim boundary

The experiment asks whether policy-internal RTC-style VJP guidance changes
closed-loop LIBERO-10 task success and action-chunk seam size relative to the
same frozen pi0.5 sampler without overlap conditioning. Hard projected
inpainting is retained as a mechanistic ablation.

The experiment does not test independently ticking inference and control,
deadline guarantees, collision safety, a second policy, or real hardware. A
positive result would support a same-checkpoint LIBERO simulation result only;
it would not establish full equivalence to the RTC paper.

## Frozen implementation

- ArmBench evaluator commit: `2aef062256fc3f6257f9f58d68c3f18c07d1b0b8`
- OpenPI upstream base: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- OpenPI RTC extension: `54592c7148ba69bf52757385502782f80f2285e0`
- Checkpoint config: `pi05_libero`
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Scheduler: `H=10`, `E=5`, injected delay `d=4`
- Methods: `overlap_unconditioned`, `projected_overlap`, and
  `rtc_guided_overlap`

Every triplet uses the same initial state and explicit pi0.5 sampling-noise
key. Method order follows the evaluator's three-way Latin rotation. Query zero
only establishes an unexecuted reference; query one reuses the initial
observation. Each later execution uses `old[:d] + new[d:E]`, then shifts
`new[E:H]` into the next zero-padded reference.

## Cohorts

The development pilot uses initial-state indices `0,1` and sampling seed
`20260805`; its 60 rollouts are excluded from the held-out analysis.

The held-out matrix is:

```text
10 LIBERO-10 tasks
x 5 initial-state indices (2,3,4,5,6)
x 2 policy sampling seeds (20260806, 20260807)
x 3 methods
= 300 closed-loop rollouts = 100 matched triplets
```

Both 150-rollout seed artifacts must finish and validate. A duplicate, missing,
or mismatched triplet fails the combined analysis. Infrastructure failures are
not selectively replaced; a restarted seed receives a new artifact identifier
and the failed attempt remains recorded.

## Frozen analysis

The experimental unit for paired outcome records is one
task/state/sampling-seed triplet. Task structure is retained explicitly because
five states and two sampling seeds are nested within each of only ten fixed
benchmark tasks.

For each method, report successes, rollouts, success rate, and a 95% Wilson
interval. The only planned success contrasts are each conditioned method minus
`overlap_unconditioned`. Report paired risk difference, wins/losses/ties, and
two-sided exact McNemar p-values with Holm step-down correction across the two
contrasts. A success-improvement statement requires a positive risk difference
and Holm-adjusted `p < 0.05`.

Also report a whole-task block-bootstrap 95% interval with 10,000 resamples and
seed `20260805`, an exact task-level sign-flip diagnostic, per-task effects,
condition-order strata, and leave-one-task-out ranges. These quantify dependence
and task sensitivity; they do not turn the ten LIBERO tasks into a random sample
of real manipulation tasks.

For motion and gripper seams, exclude bootstrap queries and first average within
each rollout so every rollout has equal weight. Report method-level mean,
standard deviation, median, IQR, valid rollout count, and transition count.
For both conditioned methods, report paired episode-level differences relative
to unconditioned overlap and whole-task block-bootstrap 95% intervals. Seam
results are exploratory process metrics and receive no superiority p-value.
Negative `candidate - unconditioned` seam differences indicate smaller seams.

The hard-projection residual and RTC weighted model RMSE are implementation
diagnostics, not task-efficacy outcomes. Query-level means from the evaluator's
runtime summary are not used as inferential seam estimates.

## Release gate

No held-out result is reported until both raw artifacts have been downloaded,
their archive hashes match the cloud copies, each root manifest and transition
archive validates locally, and the combined analysis is reproducible from the
preserved JSON and NPZ records. Failure videos are retained; absence of a video
for a successful rollout is expected under `video_mode=failures`.
