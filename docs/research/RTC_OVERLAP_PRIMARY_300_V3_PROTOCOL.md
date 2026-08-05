# pi0.5 RTC overlap: corrected held-out 300-rollout protocol

Status: frozen on 2026-08-05 before any v3 rollout. This protocol replaces the
v2 execution attempt, not its original analysis question. It is an auditable
project protocol rather than a third-party preregistration.

## Why a corrected execution is required

The v2 evaluator reused one LIBERO environment across method conditions. A
post-run pairing audit found that tasks 3, 8, and 9 retained visual state across
resets: the two policy images changed while the declared initial-state hash,
robot state, prompt, sampling key, and sampling noise remained fixed. Query-0
actions consequently differed in 15/50 triplets in each held-out seed attempt.

Both v2 attempts are preserved but excluded from every method-effect estimate.
The evidence and root-cause reproduction are recorded in
[`RTC_OVERLAP_PAIRING_AUDIT_20260805.md`](RTC_OVERLAP_PAIRING_AUDIT_20260805.md).
No v2 outcome is used to tune the v3 method, matrix, or analysis.

## Frozen implementation and pairing gate

- Corrected ArmBench evaluator commit:
  `44c358731c5493284b74bb29eefa7d538d0f38dd`
- OpenPI upstream base: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- OpenPI RTC extension: `54592c7148ba69bf52757385502782f80f2285e0`
- Checkpoint config: `pi05_libero`
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- Scheduler: `H=10`, `E=5`, injected delay `d=4`
- Methods: `overlap_unconditioned`, `projected_overlap`, and
  `rtc_guided_overlap`
- Runtime schema: `armbench.pi05_rtc_overlap_pilot.v3`

Every rollout constructs and closes a fresh LIBERO environment. Every query
records a canonical SHA-256 over the exact two `224 x 224 x 3 uint8` policy
images, eight-dimensional little-endian `float64` robot state, and UTF-8 prompt.
Before finalization, every query-0 triplet must have exactly one row per method
and one shared policy-input hash, response-action hash, sampling-key hash, and
sampling-noise hash. Any mismatch invalidates the entire artifact before a root
manifest is written.

The reference-only bootstrap and overlap execution remain unchanged: query 0
establishes an unexecuted reference, query 1 reuses the initial observation,
execution uses `old[:d] + new[d:E]`, and the next reference is
`new[E:H] + zeros(E)`.

## Corrected cohorts

The required preflight smoke is:

```text
tasks 3, 8, 9 x state 2 x seed 20260808 x 3 methods = 9 rollouts
```

It is an implementation gate only and is excluded from all outcome analysis.
The full run may start only if its query-0 input, response, key, and noise hashes
all match within every triplet and the v3 validator succeeds.

The corrected held-out matrix retains the prospectively selected cells:

```text
10 LIBERO-10 tasks
x 5 initial-state indices (2,3,4,5,6)
x 2 policy sampling seeds (20260806, 20260807)
x 3 methods
= 300 rollouts = 100 matched triplets
```

New run identifiers are required:

- `pi05_rtc_overlap_primary_v3_seed_20260806_001`
- `pi05_rtc_overlap_primary_v3_seed_20260807_001`

Both complete v3 artifacts are required. The rejected v2 attempts remain in the
repository and are not selectively repaired or pooled.

## Frozen analysis

The analysis remains the one frozen before v2 outcomes were inspected. For each
method, report success count/rate and a 95% Wilson interval. Compare hard
projection and RTC guidance separately with unconditioned overlap using paired
risk difference, wins/losses/ties, two-sided exact McNemar tests, and Holm
correction across the two comparisons. A task-success improvement statement
requires a positive risk difference and Holm-adjusted `p < 0.05`.

Also report a 10,000-resample whole-task block bootstrap with seed `20260805`,
an exact task-level sign-flip diagnostic, per-task effects, method-order strata,
and leave-one-task-out ranges.

For motion and gripper seams, exclude bootstrap queries and first average within
each rollout. Report method summaries and paired conditioned-minus-unconditioned
episode differences with whole-task bootstrap intervals. Seam metrics remain
exploratory process outcomes and receive no superiority p-value. Hard-projection
residual and RTC weighted model RMSE remain implementation diagnostics.

## Claim and release boundary

The study covers one frozen pi0.5 checkpoint on ten fixed LIBERO-10 simulation
tasks. It does not test independently ticking inference/control, hard deadlines,
collision safety, a second policy, or real hardware, and it is not evidence of
training or fine-tuning pi0.5.

No v3 outcome may be reported until both raw artifacts have completed, passed
their v3 validator and query-0 pairing gate, been archived and downloaded with
matching cloud/local SHA-256 values, and passed a deterministic combined-analysis
rebuild from preserved JSON and NPZ records.
