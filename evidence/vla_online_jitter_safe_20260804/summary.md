# Receding-horizon MuJoCo VLA runtime benchmark

This benchmark executes only a prefix of each 15x8 action chunk, then recaptures both 224x224 cameras and actual MuJoCo joint state before the next policy query.

The policy is `scripted_non_learned_reference`. No pi0/pi0.5 checkpoint or learned-policy inference was used.

Repeating synthetic latency profile (ms): `[0.0, 40.0, 80.0, 160.0]`.

| Scenario | Payload kg | Horizon | Queries | Termination | Task | Safe | Faults | Deadlines | State mismatches | Goal error rad | RMSE rad |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single_block | 0 | 15 | 16 | goal_reached | True | True | 0 | 0 | 0 | 0.00021 | 0.00237 |

The latency profile is synthetic in the reference-policy benchmark. MuJoCo advances under a pose-hold controller for each delay before the response is guarded; it is not measured model latency.

The comparison isolates runtime feedback frequency and physics tracking. It is not evidence of VLA task competence.
