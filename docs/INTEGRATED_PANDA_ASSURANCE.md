# Integrated Panda action assurance

Status: Current. Updated: 2026-08-08.

## Engineering question

An action-chunk policy can return a numerically valid tensor that is still
unsafe or unusable at execution time. The response may be stale, refer to an
old robot state, violate joint kinematics, cross an obstacle between sampled
waypoints, or leave the arm unable to stop within the registered model limits.

ArmBench's integrated Panda supervisor is a CPU reference implementation for
that policy-to-controller boundary. It accepts a complete chunk only after all
registered checks pass. A rejected chunk exposes no successful partial prefix.

## Assurance chain

```text
measured q/qdot + observed q + response age + Hx8 action chunk
                              |
                              v
                 deadline and state-alignment gate
                              |
                              v
       OSQP projection: joint position, velocity, acceleration,
                       and declared linear constraints
                              |
                              v
       continuous static/self-collision certificate for every edge
                              |
                              v
       dynamics-feasible stop certificate at every action boundary
                              |
                              v
       accepted | verified_brake | hold | unrecoverable_stop
```

OSQP enforces the kinematic box and declared linear constraints. Collision is
checked after projection by a separate fail-closed continuous-edge checker; it
is not encoded as a convex QP collision constraint. Stop feasibility is checked
with sampled MuJoCo inverse dynamics and registered actuator margins.

The decision is atomic:

- `accepted`: the complete projected chunk and one stop certificate per action
  boundary are available;
- `verified_brake`: the policy chunk is rejected while the moving measured
  state has a validated braking trajectory;
- `hold`: the policy chunk is rejected while the stationary measured state has
  a validated zero-motion fallback;
- `unrecoverable_stop`: even the model-based fallback cannot be certified, so
  no policy action is exposed and higher-level intervention is required.

## Preserved evidence

### Registered fault matrix

[`integrated_panda_fault_matrix_001`](../reports/integrated_panda_fault_matrix_001/summary.md)
contains 27 deterministic cases across three scenes, two payloads, nominal
chunks, velocity spikes, stale responses, state mismatch, an intermediate
self-collision control, and near-limit stopping states.

- 27/27 registered outcomes were reproduced;
- 12 complete plans were accepted;
- 6 stale responses entered `verified_brake`;
- 7 cases entered fail-closed `hold`;
- 2 near-limit states were reported as `unrecoverable_stop`;
- 124 continuous edges were checked;
- no rejected case exposed a partial policy action.

P95 supervision latency was 588.49 ms and the maximum was 2.972 s on the
recorded CPU host. These are offline reference timings, not a real-time result.

### Closed-loop MuJoCo task execution

[`integrated_panda_task_001`](../reports/integrated_panda_task_001/summary.md)
rebuilds RRT-Connect references, supervises the complete chunks, and then
executes the accepted trajectories through torque-controlled MuJoCo physics.

| Case | Conditions | Target error | Tracking RMSE | Registered outcome |
| --- | --- | ---: | ---: | --- |
| `single_block_goal` | 0 kg, 0 ms delay | 0.00343 rad | 0.00607 rad | target reached, physically safe |
| `narrow_gate_payload_delay_goal` | 0.5 kg, 80 ms feedback delay | 0.01962 rad | 0.03057 rad | target reached, physically safe |

Across both cases, 351/351 motion edges and 351/351 action-boundary braking
states were certified. Execution recorded zero obstacle contacts, self
contacts, joint-limit violations, and torque-saturation events.

The task checker has one explicit allowed-collision rule for the fixed-open
gripper: the `left_finger`/`right_finger` body pair is excluded. That single
body rule expands to 36 geometry-pair combinations in the compiled model. All
other registered Panda self-collision pairs remain enabled.

Full-chunk supervision took 5.27 s and 10.20 s for the two trajectories. It was
performed before physics execution. This evidence closes the local
planning-assurance-execution integration path, but it does not establish online
deadline feasibility.

## Reproduce and inspect

From the repository root on Windows:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-fault-validate `
  reports\integrated_panda_fault_matrix_001

& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-task-validate `
  reports\integrated_panda_task_001
```

The validators do more than verify stored hashes. The first rebuilds all
registered inputs and reruns every supervisor decision. The second reruns
planning, assurance, MuJoCo physics, trajectory arrays, and aggregate metrics.

The repository also provides a self-locating acceptance script. Run it from the
repository root, or pass its absolute path when launching elsewhere:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

The visual replay shows the recorded `actual_positions` trace. It is useful for
inspection, while the manifest-backed rerun remains the acceptance authority.

## Claim boundary and next step

The action source in these two artifacts is a scripted RRT-Connect reference,
not a learned VLA. The result covers joint-waypoint reaching in MuJoCo, not
grasping, object manipulation, a physical robot, hard real time, or certified
safety. It requires no GPU or server to validate.

The next research step that genuinely requires GPU access is to replace the
scripted source with frozen and then live outputs from at least two learned VLA
families, preserve exact action semantics, and evaluate the same supervisor in
an asynchronous task loop. Online deployment also requires reducing or
parallelizing the continuous and braking checks to fit a declared deadline.
