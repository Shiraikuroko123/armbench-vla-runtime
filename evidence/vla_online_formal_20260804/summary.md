# Receding-horizon MuJoCo VLA runtime benchmark

This benchmark executes only a prefix of each 15x8 action chunk, then recaptures both 224x224 cameras and actual MuJoCo joint state before the next policy query.

The policy is `scripted_non_learned_reference`. No pi0/pi0.5 checkpoint or learned-policy inference was used.

| Scenario | Payload kg | Horizon | Queries | Task | Safe | Goal error rad | RMSE rad |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_block | 0 | 1 | 233 | True | True | 0.00081 | 0.00078 |
| single_block | 0 | 5 | 47 | True | True | 0.00029 | 0.00167 |
| single_block | 0 | 15 | 16 | True | True | 0.00015 | 0.00213 |
| single_block | 0.5 | 1 | 233 | True | True | 0.00091 | 0.00088 |
| single_block | 0.5 | 5 | 47 | True | True | 0.00037 | 0.00179 |
| single_block | 0.5 | 15 | 16 | True | True | 0.00017 | 0.00221 |
| narrow_gate | 0 | 1 | 193 | True | True | 0.00370 | 0.00089 |
| narrow_gate | 0 | 5 | 39 | True | True | 0.00258 | 0.00196 |
| narrow_gate | 0 | 15 | 13 | True | True | 0.00230 | 0.00277 |
| narrow_gate | 0.5 | 1 | 193 | True | True | 0.00448 | 0.00097 |
| narrow_gate | 0.5 | 5 | 39 | True | True | 0.00330 | 0.00202 |
| narrow_gate | 0.5 | 15 | 13 | True | True | 0.00288 | 0.00283 |

`policy_latency_ms` is synthetic in the reference-policy benchmark. MuJoCo advances under a pose-hold controller for that duration before the response is guarded; it is not measured model latency.

The comparison isolates runtime feedback frequency and physics tracking. It is not evidence of VLA task competence.
