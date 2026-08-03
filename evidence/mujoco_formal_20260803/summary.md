# MuJoCo Panda benchmark summary

This artifact uses the pinned MuJoCo Menagerie Panda model. Planning contacts use compiled collision meshes; execution uses 2 ms rigid-body physics and torque-limited joint PD control.

## Planning

| Scenario | Clearance | Planner | Success | P50 ms | P95 ms |
|---|---:|---|---:|---:|---:|
| narrow_gate | 0 mm | rrt_connect | 10/10 | 27.5 | 58.0 |
| narrow_gate | 0 mm | rrt_star | 2/10 | 2000.7 | 2005.3 |
| narrow_gate | 20 mm | rrt_connect | 10/10 | 29.4 | 89.3 |
| narrow_gate | 20 mm | rrt_star | 2/10 | 2001.7 | 2004.5 |
| single_block | 0 mm | rrt_connect | 10/10 | 23.8 | 99.1 |
| single_block | 0 mm | rrt_star | 5/10 | 1178.9 | 2003.8 |
| single_block | 20 mm | rrt_connect | 10/10 | 30.6 | 123.3 |
| single_block | 20 mm | rrt_star | 4/10 | 2000.9 | 2004.7 |

## Physics execution

| Scenario | Profile | Delay | Payload | RMSE rad | Contact steps | Max force N | Safe |
|---|---|---:|---:|---:|---:|---:|---:|
| single_block | nominal_fast | 0 ms | 0.0 kg | 0.0183 | 3 | 112.77 | False |
| single_block | nominal_fast | 40 ms | 0.0 kg | 0.3261 | 179 | 4360.09 | False |
| single_block | nominal_fast | 80 ms | 0.0 kg | 0.7992 | 0 | 0.00 | False |
| single_block | nominal_fast | 0 ms | 0.5 kg | 0.0185 | 3 | 141.35 | False |
| single_block | nominal_fast | 40 ms | 0.5 kg | 0.4060 | 58 | 4933.67 | False |
| single_block | nominal_fast | 80 ms | 0.5 kg | 0.8409 | 90 | 4103.07 | False |
| single_block | nominal_slow | 0 ms | 0.0 kg | 0.0303 | 5 | 32.04 | False |
| single_block | nominal_slow | 40 ms | 0.0 kg | 0.0253 | 6 | 32.78 | False |
| single_block | nominal_slow | 80 ms | 0.0 kg | 0.0208 | 7 | 33.69 | False |
| single_block | nominal_slow | 0 ms | 0.5 kg | 0.0314 | 10 | 37.75 | False |
| single_block | nominal_slow | 40 ms | 0.5 kg | 0.0266 | 11 | 37.99 | False |
| single_block | nominal_slow | 80 ms | 0.5 kg | 0.0226 | 12 | 47.80 | False |
| single_block | clearance_slow | 0 ms | 0.0 kg | 0.0284 | 0 | 0.00 | True |
| single_block | clearance_slow | 40 ms | 0.0 kg | 0.0252 | 0 | 0.00 | True |
| single_block | clearance_slow | 80 ms | 0.0 kg | 0.0226 | 0 | 0.00 | True |
| single_block | clearance_slow | 0 ms | 0.5 kg | 0.0301 | 0 | 0.00 | True |
| single_block | clearance_slow | 40 ms | 0.5 kg | 0.0272 | 0 | 0.00 | True |
| single_block | clearance_slow | 80 ms | 0.5 kg | 0.0248 | 0 | 0.00 | True |
| narrow_gate | nominal_fast | 0 ms | 0.0 kg | 0.0181 | 0 | 0.00 | True |
| narrow_gate | nominal_fast | 40 ms | 0.0 kg | 0.4679 | 12 | 701.35 | False |
| narrow_gate | nominal_fast | 80 ms | 0.0 kg | 0.8072 | 46 | 1945.39 | False |
| narrow_gate | nominal_fast | 0 ms | 0.5 kg | 0.0184 | 0 | 0.00 | True |
| narrow_gate | nominal_fast | 40 ms | 0.5 kg | 0.3515 | 363 | 6341.24 | False |
| narrow_gate | nominal_fast | 80 ms | 0.5 kg | 0.8425 | 112 | 4141.67 | False |
| narrow_gate | nominal_slow | 0 ms | 0.0 kg | 0.0308 | 8 | 31.23 | False |
| narrow_gate | nominal_slow | 40 ms | 0.0 kg | 0.0264 | 10 | 35.53 | False |
| narrow_gate | nominal_slow | 80 ms | 0.0 kg | 0.0225 | 12 | 40.50 | False |
| narrow_gate | nominal_slow | 0 ms | 0.5 kg | 0.0317 | 9 | 35.41 | False |
| narrow_gate | nominal_slow | 40 ms | 0.5 kg | 0.0275 | 10 | 39.39 | False |
| narrow_gate | nominal_slow | 80 ms | 0.5 kg | 0.0238 | 11 | 43.75 | False |
| narrow_gate | clearance_slow | 0 ms | 0.0 kg | 0.0300 | 0 | 0.00 | True |
| narrow_gate | clearance_slow | 40 ms | 0.0 kg | 0.0257 | 0 | 0.00 | True |
| narrow_gate | clearance_slow | 80 ms | 0.0 kg | 0.0220 | 0 | 0.00 | True |
| narrow_gate | clearance_slow | 0 ms | 0.5 kg | 0.0306 | 0 | 0.00 | True |
| narrow_gate | clearance_slow | 40 ms | 0.5 kg | 0.0265 | 0 | 0.00 | True |
| narrow_gate | clearance_slow | 80 ms | 0.5 kg | 0.0230 | 0 | 0.00 | True |

## Capsule approximation versus mesh contacts

| Scenario | Samples | False safe | False collision | Disagreement |
|---|---:|---:|---:|---:|
| single_block | 601 | 6 | 2 | 1.3% |
| narrow_gate | 601 | 3 | 2 | 0.8% |

`safe_success` requires final joint error within tolerance, zero environment contact steps, zero self-contact steps, and zero joint-limit violation steps. This is simulation evidence, not real-robot validation.
