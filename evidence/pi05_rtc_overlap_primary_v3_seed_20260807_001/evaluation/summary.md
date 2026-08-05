# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 48/50 | 0.960 | 53.42 | 0.107149 | 0.062200 | n/a | n/a |
| projected_overlap | 49/50 | 0.980 | 54.12 | 0.084027 | 0.068982 | 0 | n/a |
| rtc_guided_overlap | 49/50 | 0.980 | 54.14 | 0.087586 | 0.043562 | n/a | 0.026524 |

- Complete triplets: 50
- projected_overlap vs unconditioned: delta 0.020, W/L/T 2/1/47, exact p 1
- rtc_guided_overlap vs unconditioned: delta 0.020, W/L/T 2/1/47, exact p 1
