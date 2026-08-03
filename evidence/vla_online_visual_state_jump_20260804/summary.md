# Receding-horizon MuJoCo VLA runtime benchmark

This benchmark executes only a prefix of each 15x8 action chunk, then recaptures both 224x224 cameras and actual MuJoCo joint state before the next policy query.

The policy is `scripted_non_learned_reference`. No pi0/pi0.5 checkpoint or learned-policy inference was used.

| Scenario | Payload kg | Horizon | Queries | Termination | Task | Safe | Faults | Deadlines | State mismatches | Goal error rad | RMSE rad |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single_block | 0 | 15 | 1 | guard_fallback:state_mismatch | False | True | 1 | 0 | 1 | 1.40000 | 0.00000 |

`policy_latency_ms` is synthetic in the reference-policy benchmark. MuJoCo advances under a pose-hold controller for that duration before the response is guarded; it is not measured model latency.

The comparison isolates runtime feedback frequency and physics tracking. It is not evidence of VLA task competence.

The optional state jump is a deterministic fault injected directly into MuJoCo joint state after observation capture. It tests dispatch-state consistency handling; it is not a modeled contact impulse.
