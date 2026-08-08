# Integrated Panda supervisor fault matrix

Registered cases: 27

Expected outcomes matched: 27/27

Accepted plans: 12

Verified braking fallbacks: 6

Fail-closed holds: 7

Unrecoverable stop states: 2

| Fault | Cases | Accepted | Verified brake | Hold | Unrecoverable |
|---|---:|---:|---:|---:|---:|
| intermediate_self_collision | 1 | 0 | 0 | 1 | 0 |
| near_limit_stop | 2 | 0 | 0 | 0 | 2 |
| nominal | 6 | 6 | 0 | 0 | 0 |
| stale_response | 6 | 0 | 6 | 0 | 0 |
| state_mismatch | 6 | 0 | 0 | 6 | 0 |
| velocity_spike | 6 | 6 | 0 | 0 | 0 |

The matrix uses scripted actions and synchronous CPU supervision. 
It is not learned-policy, hard-real-time, closed-loop physics, or 
physical-robot safety evidence.
