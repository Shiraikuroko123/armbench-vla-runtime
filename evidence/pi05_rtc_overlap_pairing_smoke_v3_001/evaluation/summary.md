# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 2/3 | 0.667 | 70.00 | 0.121432 | 0.041137 | n/a | n/a |
| projected_overlap | 3/3 | 1.000 | 65.33 | 0.082114 | 0.023001 | 0 | n/a |
| rtc_guided_overlap | 2/3 | 0.667 | 71.00 | 0.093141 | 0.021163 | n/a | 0.027602 |

- Complete triplets: 3
- projected_overlap vs unconditioned: delta 0.333, W/L/T 1/0/2, exact p 1
- rtc_guided_overlap vs unconditioned: delta 0.000, W/L/T 0/0/3, exact p 1
