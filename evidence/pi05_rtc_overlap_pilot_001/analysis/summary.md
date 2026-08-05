# RTC overlap pilot analysis

- Source manifest SHA-256: `b9505af7c5185e1d5bf3af22a273708ec51b0a40a3c387a7865cc6777d6b07bb`
- Matrix: 10 tasks x 2 initial states x 3 methods (60 rollouts)
- Task-block bootstrap seed/resamples: `20260805/10000`

| Method | Success (Wilson 95%) | Motion seam mean / median | Gripper seam mean / median |
| --- | ---: | ---: | ---: |
| overlap_unconditioned | 20/20 [0.839, 1.000] | 0.104452 / 0.105460 | 0.058052 / 0.046187 |
| projected_overlap | 19/20 [0.764, 0.991] | 0.083129 / 0.082088 | 0.045345 / 0.042563 |
| rtc_guided_overlap | 19/20 [0.764, 0.991] | 0.084002 / 0.080663 | 0.039617 / 0.035865 |

| Contrast vs unconditioned | Success diff | McNemar p | Holm p | Motion seam mean diff (task-block 95%) | Gripper seam mean diff (task-block 95%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| projected_overlap | -0.050 | 1 | 1 | -0.021322 [-0.032036, -0.012622] | -0.012707 [-0.029111, +0.002804] |
| rtc_guided_overlap | -0.050 | 1 | 1 | -0.020450 [-0.030301, -0.011399] | -0.018435 [-0.037893, +0.000875] |

Simulation-only 10-task pilot with two initial states per task. Task-block bootstrap intervals are descriptive; the matrix is not a submission-scale estimate of general VLA task performance.
