# Debugging ArmBench

This guide follows the shortest path from environment failure to a specific
planner, collision, controller, or rendering defect.

## 1. Verify the environment boundary

Run from the project directory:

```powershell
& '..\.venv\Scripts\python.exe' -c `
  "import mujoco, armbench; print(mujoco.__version__)"
& '..\.venv\Scripts\python.exe' -m armbench mujoco-validate
```

Expected: MuJoCo `3.11.0` and four passing records covering both scenes at 0
and 20 mm clearance. A missing Menagerie checkout raises a path containing the
exact expected model location.

If `mujoco-validate` reports an endpoint collision, do not increase planner
iterations. Fix the scene first. Each scene must keep both endpoints valid at
the largest configured clearance while blocking direct interpolation.

## 2. Run the contract tests

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
```

The MuJoCo tests cover joint-name mapping, official limits, endpoint/direct-edge
geometry, named obstacle contact, torque execution, nonblank rendering, the
minimal artifact contract, and trace loading. A model or renderer regression
should fail here before a long experiment is started.

## 3. Create a cheap smoke artifact

```powershell
& '..\.venv\Scripts\python.exe' -m armbench mujoco-run --quick `
  --run-id debug_smoke
```

This uses two planning seeds, 40 collision samples, and skips rigid-body
execution. Inspect these files in order:

1. `scenario_validation.json`: protocol errors.
2. `planning_per_trial.csv`: status, seed, latency, and failure detail.
3. `collision_samples.csv`: exact versus capsule labels and the seven-joint
   configuration for every mismatch.
4. `run.log`: the last completed case if a run stops.

## 4. Inspect geometry interactively

Open the actual obstacle geometry:

```powershell
& '..\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block --pose start
```

Open the 20 mm planning volumes and attached payload:

```powershell
& '..\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario narrow_gate --clearance-mm 20 --payload 0.5
```

Closing the viewer prints the inspected joint vector, hand position, obstacle
contacts, and self-contacts as JSON. Use `--pose goal` for the other endpoint.

## 5. Replay a failed execution

Every physics case stores the command, measured position, applied torque, and
control timestamps in `traces/<case>.npz`. Replay measured motion at real time:

```powershell
& '..\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block --play `
  --trace 'results\mujoco_formal_20260803\traces\single_block__nominal_fast__delay_080ms__payload_0.0kg.npz'
```

Useful variants:

```powershell
# Compare the desired motion.
& '..\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block --play --array desired_positions `
  --trace '<trace.npz>'

# Freeze the measured state at one control sample.
& '..\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block --frame 120 --trace '<trace.npz>'
```

The viewer is kinematic replay of recorded states. It does not rerun the
controller, so it is safe to use for isolating the first visibly bad frame.

## 6. Read the metrics correctly

- Low RMSE with nonzero contact means the path has insufficient clearance; it
  is not a successful safety result.
- Zero contact with a large final error or joint-limit count means the
  controller failed without hitting the configured obstacle.
- High torque saturation under delay usually indicates stale-feedback
  oscillation. Compare the trace with the 0 ms case before changing the planner.
- Small contact penetration can still produce a high peak force in a stiff
  contact model. Use steps/events/duration together with force.
- A planner timeout is retained as data. It must not be removed from latency
  percentiles or rerun with a hand-picked seed.

## 7. Code entry points and breakpoints

| Question | File / function |
|---|---|
| Did the official MJCF map correctly? | `mujoco_sim/model.py: MuJoCoPanda.create` |
| Why is this pose invalid? | `mujoco_sim/collision.py: configuration_failure` |
| Which interpolation sample fails? | `mujoco_sim/collision.py: edge_is_valid` |
| Why did search stop? | `planners/rrt_connect.py: plan` or `rrt_star.py: plan` |
| Was smoothing responsible? | `postprocess/shortcut.py: shortcut_path` |
| What torque was requested/clipped? | `mujoco_sim/execution.py: execute_trajectory` |
| Which geom contacted which link? | `mujoco_sim/model.py: obstacle_contacts` |
| How was a case configured? | `mujoco_sim/benchmark.py: _run_execution` |
| Why will a trace not load? | `mujoco_sim/viewer.py: load_pose_sequence` |

At the controller breakpoint, inspect `desired_q`, `observed_q`, `observed_dq`,
`requested`, `applied`, and `data.qfrc_bias`. Delay is represented by
`observed_index`; for 80 ms at a 10 ms control period it should select eight
control ticks behind the current observation.

## Common failures

### Menagerie model not found

Repeat the sparse-checkout commands in the README and verify that
`..\upstream\mujoco_menagerie\franka_emika_panda\scene.xml` exists. Do not copy
only `scene.xml`; it references meshes and included files in the model folder.

### OpenGL or blank frames

Update the graphics driver, close software that has exhausted graphics
contexts, and rerun only `test_offscreen_render_is_nonblank`. Physics and
planning can still run with `mujoco-run --no-videos`, but a portfolio video
should not be claimed until the render test passes.

### Start or goal suddenly collides

Check both physical and inflated obstacles. A center that is valid at 55 mm can
be invalid after 20 mm inflation. Keep the automated four-record validation as
the acceptance criterion.

### Different planning latency

Paths are seed-controlled; wall time is not. Check CPU load and compare status,
iterations, nodes, and path arrays before treating latency drift as an
algorithm regression. A seed near the 2 s deadline can change status across
hosts.
