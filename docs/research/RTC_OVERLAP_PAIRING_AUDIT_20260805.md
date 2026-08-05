# RTC overlap query-0 pairing audit

Status: the v2 pilot and both v2 held-out seed attempts are rejected from
matched-effect analysis. Their raw files remain immutable evidence of the
attempt and of the audit failure.

## Detection

The v2 evaluator paired the declared LIBERO initial-state index and explicit
pi0.5 sampling key, but it did not hash the four model inputs or require the
unconditioned query-0 response to match across the three methods. An independent
review after the first held-out seed had been preserved found the missing
invariant. Commit `2a558ad` makes both the primary analyzer and offline dashboard
fail closed on this condition.

The audit found query-0 response-action mismatches despite identical sampling
key and noise hashes:

| Artifact | Mismatched triplets | Affected cells |
| --- | ---: | --- |
| `pi05_rtc_overlap_pilot_001` | 6/20 | tasks 3, 8, 9; states 0-1 |
| `pi05_rtc_overlap_primary_seed_20260806_001` | 15/50 | tasks 3, 8, 9; states 2-6 |
| `pi05_rtc_overlap_primary_seed_20260807_001` | 15/50 | tasks 3, 8, 9; states 2-6 |

In every affected triplet, all three methods had one shared sampling-key hash
and one shared sampling-noise hash but three distinct query-0 action hashes.
Reconstruction of the first held-out seed from `transitions.npz` showed that
this was not hash sensitivity to insignificant serialization noise: the largest
single action-element difference was `0.236883`, and the largest action-chunk
L2 difference was `0.701880`.

## Root cause reproduction

A read-only cloud diagnostic used task 3, initial-state index 2, environment
seed 7, ten dummy stabilization steps, and the production request builder. It
hashed the two `224 x 224 x 3` policy images, eight-dimensional robot state, and
prompt after each initialization.

| Initialization strategy | Agent image hashes | Wrist image hashes | State hashes | Prompt hashes |
| --- | ---: | ---: | ---: | ---: |
| Reuse one LIBERO environment for three resets | 3 | 3 | 1 | 1 |
| Construct and close a fresh environment each time | 1 | 1 | 1 | 1 |

The repeated fresh-environment image hashes were
`a11d2caefe51d89ee1f4c449755a40feeb8fc73e5687655d7c79889d0494c980`
and
`fd7bdb9373956801d938dea9b67321ca9c7670c27af0f93914811a2df449e6e1`.
The diagnosis therefore isolates visual state carried across environment reuse;
it does not implicate the keyed pi0.5 noise contract.

## Consequence and remediation

The v2 root manifests, transition archives, videos, checkpoint attestation, and
legacy validator reports remain internally valid. That integrity is necessary
but not sufficient for a matched comparison. Success and seam summaries from
these three artifacts must not be used as evidence that one method is better or
worse.

The corrected v3 evaluator must:

1. construct and close a fresh LIBERO environment for every rollout;
2. record a canonical digest of both policy images, robot state, and prompt;
3. require query-0 input, response, sampling-key, and sampling-noise hashes to
   match across all three methods before finalization; and
4. use new run identifiers while retaining both rejected held-out attempts.

The full corrected matrix may start only after a task 3/8/9 smoke test passes
these invariants.

## Transfer records

- Seed `20260806` archive SHA-256:
  `70e65a48bbfe7e4256450b4b13116b9c288b02f966b6c5c66744ac8997880dc6`
- Seed `20260807` archive SHA-256:
  `0449734c0a51ee482a5fa040a883f014c3559b541a6b05b3aa2f4b9bf620f45f`

Each value matched between the cloud archive and the local download before
extraction.
