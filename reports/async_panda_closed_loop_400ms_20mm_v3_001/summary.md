# Asynchronous Panda closed-loop runtime

A blocking scripted policy runs on a worker thread while a best-effort
periodic control loop continues torque-controlled MuJoCo execution.
Camera/state acquisition uses a latest-only worker; response age includes
sensor acquisition and policy inference.

| Mode | Cases | Target reached | Physically safe | Hold rate | Abrupt stops | Repair budget misses | Contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| braking_invariant | 9 | 1 | 9 | 0.304 | 0 | 0 | 0 |
| legacy_greedy | 9 | 1 | 9 | 0.343 | 311 | 0 | 0 |
| unguarded | 9 | 6 | 8 | 0.335 | 289 | 0 | 0 |

## Claim boundary

The policy is scripted and non-learned; no pi0/pi0.5 checkpoint was
executed. Deadlines are measured software budgets, not an OS hard-real-
time guarantee. MuJoCo contact outcomes are not physical-robot safety
certification.
