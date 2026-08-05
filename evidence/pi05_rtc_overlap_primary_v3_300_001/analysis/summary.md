# pi0.5 RTC overlap corrected-v3 held-out primary analysis

- Matrix: 10 tasks x 5 initial states x 2 sampling seeds x 3 methods (300 rollouts)
- Matched task/state/seed triplets: `100`
- Whole-task bootstrap seed/resamples: `20260805/10000`
- Confirmatory family: two exact McNemar tests with Holm correction

| Method | Success (Wilson 95%) | Motion seam mean / median | Gripper seam mean / median | Scored transitions |
| --- | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 96/100 (0.960 [0.902, 0.984]) | 0.106729 / 0.102259 | 0.053754 / 0.044599 | 5270 |
| projected_overlap | 97/100 (0.970 [0.915, 0.990]) | 0.083089 / 0.079802 | 0.055490 / 0.045002 | 5264 |
| rtc_guided_overlap | 97/100 (0.970 [0.915, 0.990]) | 0.087204 / 0.084136 | 0.043190 / 0.041612 | 5323 |

| Contrast vs unconditioned | Success difference (task-block 95%) | Wins/losses/ties | McNemar raw/Holm | Task sign-flip p | Motion seam difference (task-block 95%) | Gripper seam difference (task-block 95%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| projected_overlap | +0.010 [-0.040, +0.060] | 4/3/93 | 1/1 | 1 | -0.023640 [-0.028187, -0.019207] | +0.001736 [-0.012925, +0.014252] |
| rtc_guided_overlap | +0.010 [-0.040, +0.080] | 3/2/95 | 1/1 | 1 | -0.019524 [-0.023128, -0.016142] | -0.010564 [-0.024923, +0.001622] |

Same-checkpoint pi0.5 evidence on ten fixed LIBERO-10 simulation tasks. Each immutable corrected-v3 evaluator protocol retains pilot_only=true; the held-out primary role comes from external frozen protocol commit 509f6f4cbcc9e8b02804edf640e565673d4a3855 and disjoint state/seed cohorts. The preserved v2 attempts are excluded from every estimate. It does not establish independent control/inference timing, deadline or collision safety, cross-policy generalization, or real-hardware efficacy.
