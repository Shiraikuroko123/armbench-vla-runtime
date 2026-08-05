# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 20/20 | 1.000 | 50.70 | 0.105079 | 0.064747 | n/a | n/a |
| projected_overlap | 19/20 | 0.950 | 52.95 | 0.083083 | 0.046394 | 0 | n/a |
| rtc_guided_overlap | 19/20 | 0.950 | 53.65 | 0.083156 | 0.042177 | n/a | 0.024735 |

- Complete triplets: 20
- projected_overlap vs unconditioned: delta -0.050, W/L/T 0/1/19, exact p 1
- rtc_guided_overlap vs unconditioned: delta -0.050, W/L/T 0/1/19, exact p 1
