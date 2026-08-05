# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 49/50 | 0.980 | 53.74 | 0.106325 | 0.069468 | n/a | n/a |
| projected_overlap | 49/50 | 0.980 | 53.84 | 0.084958 | 0.065637 | 0 | n/a |
| rtc_guided_overlap | 46/50 | 0.920 | 55.82 | 0.088694 | 0.040661 | n/a | 0.026591 |

- Complete triplets: 50
- projected_overlap vs unconditioned: delta 0.000, W/L/T 1/1/48, exact p 1
- rtc_guided_overlap vs unconditioned: delta -0.060, W/L/T 0/3/47, exact p 0.25
