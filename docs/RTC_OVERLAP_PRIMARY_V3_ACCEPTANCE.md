[English](RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE.md) | [简体中文](RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE_ZH.md)

# RTC overlap corrected-v3: evidence acceptance

## Evidence lineage

The RTC implementation progressed through staged integration and evaluation.
Each stage retains its own validity status; later execution does not
retroactively validate an earlier comparison:

| Stage | Purpose | Current status |
| --- | --- | --- |
| Fixed-observation G0 | Sampler parity, guidance direction, latency, memory | Valid implementation evidence; not task efficacy |
| 40-rollout hard-projection pilot | Separate two-method mechanism exploration | Preserved exploratory evidence; not pooled with RTC v3 |
| 60-rollout RTC v2 attempt | First three-method closed-loop integration | Rejected for query-0 environment carryover |
| Two 50-triplet v2 held-out attempts | Intended primary comparison | Rejected for the same pairing failure |
| Corrected-v3 smoke | Fresh-environment and four-hash pairing gate | Passed; excluded from outcomes |
| Corrected-v3 held-out primary | 100 matched triplets, 300 rollouts | Complete and current |

The v2 attempts were excluded because their causal comparison was invalid, not
because their numerical outcomes differed from v3. Reusing one LIBERO
environment changed the two policy images for tasks 3, 8, and 9.
Sampling keys and noise still matched, but query-0 inputs and actions did not.
The [pairing audit](research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md) preserves
the detection, reproduction, and remediation.

Corrected-v3 constructs and closes a fresh environment for every rollout. It
requires identical query-0 policy-input, response-action, sampling-key, and
sampling-noise hashes across the three methods before finalization. Any mismatch
invalidates the artifact.

## Validated result

| Method | Success | Motion seam mean | Gripper seam mean |
| --- | ---: | ---: | ---: |
| Unconditioned overlap | 96/100 | 0.106729 | 0.053754 |
| Hard projection | 97/100 | 0.083089 | 0.055490 |
| RTC guidance | 97/100 | 0.087204 | 0.043190 |

Both conditioned methods have a `+1` percentage-point task-success difference
and raw/Holm exact McNemar `p=1.0`. No task-success superiority is supported.
The motion-seam differences are exploratory process evidence only:

- hard projection: `-0.023640`, task-block 95% CI
  `[-0.028187,-0.019207]`;
- RTC guidance: `-0.019524`, task-block 95% CI
  `[-0.023128,-0.016142]`.

## Acceptance command

From any Windows directory:

```powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd
```

Use `-NoOpen` to validate without opening the browser:

```powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd -NoOpen
```

The command fails closed unless both v3 raw artifacts validate, the combined
analysis rebuilds from those sources, the analysis manifest is valid, all 100
triplets bind to the rebuilt analysis, and all ten referenced failure videos
exist and hash correctly. A valid run reports:

```text
valid=true
rollouts=300
triplets=100
failure_videos_verified=10
tasks=10
```

The generated dashboard is
[`reports/pi05_rtc_overlap_primary_v3_300_001/index.html`](../reports/pi05_rtc_overlap_primary_v3_300_001/index.html).

## Scope

This is same-checkpoint pi0.5 evidence on ten fixed LIBERO-10 simulation tasks.
It does not establish independent control/inference timing, hard deadlines,
collision safety, cross-policy generality, real-hardware efficacy, or training
and fine-tuning of pi0.5.
