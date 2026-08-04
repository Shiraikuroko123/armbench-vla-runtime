# pi0.5 projected-overlap pilot

This is a paired pilot, not a confirmatory efficacy claim. Both methods use the same fixed-width overlap scheduler and keyed policy noise.

| Method | Success | Rate | Mean queries | Mean motion seam | Mean gripper seam | Max residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_unconditioned | 19/20 | 0.950 | 53.20 | 0.109373 | 0.048326 | n/a |
| projected_overlap | 18/20 | 0.900 | 55.05 | 0.086893 | 0.059185 | 0 |

- Paired rollouts: 20
- Success difference: -0.050
- Wins/losses/ties: 0/1/19
- Exact McNemar p: 1
