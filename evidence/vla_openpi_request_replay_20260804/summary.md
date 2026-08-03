# OpenPI-compatible remote closed-loop MuJoCo run

- Server: `127.0.0.1:61190`
- Server metadata: `{"armbench_loopback": true, "checkpoint_identity_verified": false, "fault_delay_ms": 250.0, "fault_mode": "none", "fault_request_index": 0, "model_config": "armbench_scripted_droid_loopback", "policy_source": "scripted_non_learned_loopback"}`
- Validated remote 15x8 chunks: `2`
- Policy provenance: `scripted_non_learned_loopback`
- Remote policy response validated: `true`
- Checkpoint identity verified by protocol: `false`
- Termination: `query_budget`

| Scenario | Horizon | Queries | Valid replies | Task | Safe | P95 end-to-end ms | Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_block | 1 | 2 | 2 | False | True | 123.45 | 0 |

This command uses the real bounded OpenPI WebSocket transport and recaptures live MuJoCo state and both cameras between action prefixes. A connected server without a validated reply is not counted as a validated remote policy response.

This server is a scripted non-learned protocol diagnostic. No learned checkpoint produced these actions; task outcomes are integration results only.
