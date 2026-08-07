# Frozen pi0.5 response replay on the Panda guard path

This report replays hash-verified official pi0.5 LIBERO action responses
through the local Cartesian adapter and runtime guard. It does not rerun
the policy checkpoint or execute a closed-loop task.

## Coverage

- Selected chunks: 90
- Task/method strata: 30
- Independent Panda cases: 270
- Source response hashes verified: 7934

## Runtime observations

| Metric | Value |
| --- | ---: |
| Deadline-exceeded cases | 3 / 270 |
| Input-clipped cases | 267 / 270 |
| Raw path-invalid cases | 36 / 270 |
| Cases with guard intervention | 226 / 270 |
| Acceleration-conflict cases | 6 / 270 |
| Guard-safe cases | 264 / 270 |
| Guarded path-valid cases | 270 / 270 |
| P95 adapter latency | 5.759 ms |
| P95 guard latency | 16.115 ms |

## Claim boundary

- The source checkpoint and response hashes are attested, but the checkpoint was not executed in this replay.
- No task-success metric or Panda closed-loop execution was evaluated.
- Differential IK is not equivalent to LIBERO's torque-level OSC controller.
- Resolution-bounded collision checks are not continuous-collision or physical-safety certification.
