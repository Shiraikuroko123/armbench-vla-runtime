# pi0.5 three-arm RTC overlap pilot

This is a paired three-arm pilot, not a confirmatory efficacy claim. All methods use reference-only bootstrap, the same overlap scheduler, and keyed noise.

| Method | Success | Rate | Mean queries | Motion seam | Gripper seam | Hard residual | RTC RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 47/50 | 0.940 | 55.70 | 0.110930 | 0.065924 | n/a | n/a |
| projected_overlap | 47/50 | 0.940 | 55.72 | 0.082421 | 0.059418 | 0 | n/a |
| rtc_guided_overlap | 47/50 | 0.940 | 54.16 | 0.085038 | 0.051234 | n/a | 0.026640 |

- Complete triplets: 50
- projected_overlap vs unconditioned: delta 0.000, W/L/T 2/2/46, exact p 1
- rtc_guided_overlap vs unconditioned: delta 0.000, W/L/T 1/1/48, exact p 1
