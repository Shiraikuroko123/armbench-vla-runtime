# OpenPI-contract VLA action guard benchmark

![VLA runtime benchmark overview](overview.png)

**Policy provenance:** scripted non-learned action streams. No pi0 or pi0.5 checkpoint was used in this local run.

The interface matches `pi05_droid` at OpenPI commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`: two 224x224 RGB images, 8-D state, language prompt, and 15x8 action chunks.

| Scenario | Condition | Mode | Interventions | Deadline chunks | Guard P95 ms | Contacts | Task | Safe |
|---|---|---|---:|---:|---:|---:|---:|---:|
| single_block | fresh_safe_jitter | unguarded | 0 | 0 | 0.000 | 0 | True | True |
| single_block | fresh_safe_jitter | guarded | 0 | 0 | 5.395 | 0 | True | True |
| single_block | fresh_collision_fault | unguarded | 0 | 0 | 0.000 | 673 | False | False |
| single_block | fresh_collision_fault | guarded | 40 | 0 | 6.364 | 0 | False | True |
| single_block | mixed_deadline_jitter | unguarded | 0 | 4 | 0.000 | 0 | True | True |
| single_block | mixed_deadline_jitter | guarded | 195 | 4 | 5.365 | 0 | False | True |
| narrow_gate | fresh_safe_jitter | unguarded | 0 | 0 | 0.000 | 0 | True | True |
| narrow_gate | fresh_safe_jitter | guarded | 0 | 0 | 5.362 | 0 | True | True |
| narrow_gate | fresh_collision_fault | unguarded | 0 | 0 | 0.000 | 1641 | False | False |
| narrow_gate | fresh_collision_fault | guarded | 40 | 0 | 7.959 | 0 | False | True |
| narrow_gate | mixed_deadline_jitter | unguarded | 0 | 3 | 0.000 | 0 | True | True |
| narrow_gate | mixed_deadline_jitter | guarded | 150 | 3 | 5.690 | 0 | False | True |

This artifact validates the policy/runtime contract and guard logic. It is not evidence of learned-policy task performance. Use `armbench vla-probe` with an official remote OpenPI server before making a pi0/pi0.5 inference claim.

A deadline miss latches hold until an explicit runtime reset; later fresh chunks do not silently resume an open-loop stream.
