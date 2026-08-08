# Integrated Panda task execution

Safe task successes: 2/2

| Case | Scenario | Payload | Delay | Guard | Target | Physical safety |
|---|---|---:|---:|---|---|---|
| single_block_goal | single_block | 0.0 kg | 0 ms | accepted | True | True |
| narrow_gate_payload_delay_goal | narrow_gate | 0.5 kg | 80 ms | accepted | True | True |

The action source is a scripted RRT-Connect reference. Assurance is 
computed offline before MuJoCo torque execution; this is not a learned 
VLA, hard-real-time, or physical-robot safety result.
