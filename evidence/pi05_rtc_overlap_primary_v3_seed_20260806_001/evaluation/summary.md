# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 48/50 | 0.960 | 53.98 | 0.108941 | 0.056634 | n/a | n/a |
| projected_overlap | 48/50 | 0.960 | 53.16 | 0.082604 | 0.056885 | 0 | n/a |
| rtc_guided_overlap | 48/50 | 0.960 | 54.32 | 0.088376 | 0.047357 | n/a | 0.027122 |

- Complete triplets: 50
- projected_overlap vs unconditioned: delta 0.000, W/L/T 2/2/46, exact p 1
- rtc_guided_overlap vs unconditioned: delta 0.000, W/L/T 1/1/48, exact p 1
