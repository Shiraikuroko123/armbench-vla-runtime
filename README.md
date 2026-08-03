# ArmBench: CPU-Only 7-DoF Planning and Tracking Benchmark

ArmBench is a reproducible seven-joint manipulator benchmark built for a CPU
development environment. It independently implements bounded RRT-Connect and
an RRT* first-solution baseline, whole-link collision checks, collision-aware
shortcut smoothing, velocity-limited timing, and delayed PD/LQR tracking.

The project starts from a narrow question: how much correctness and evidence
is missing between a successful sampling-planner animation and an executable,
repeatable planning-and-control result?

## What is implemented

- Pure NumPy standard-DH forward kinematics for a Franka Panda parameter table.
- Published per-joint position and velocity limits instead of a shared range.
- Capsule-to-sphere collision at every complete link, including degenerate links.
- Resolution-bounded joint-space edge validation at 0.05 rad.
- RRT-Connect with `TRAPPED/ADVANCED/REACHED`, timeout, and iteration bounds.
- RRT* rewiring baseline with an explicit first-feasible-solution stopping rule.
- Three fixed scenarios, 30 paired planner seeds, failure taxonomy, and Wilson CIs.
- Collision-revalidated shortcut smoothing and first-order velocity timing.
- Saturated PD and discrete LQR tracking with 0/40/80 ms observation delay,
  stochastic noise, and 1.0x/1.25x load models.
- Separate tracking counters for joint-limit samples, obstacle-collision samples,
  and invalid transitions between adjacent control ticks.
- A run artifact containing config, environment, raw CSV, aggregates, paths,
  failures, compressed traces, figures, and a Markdown summary.

## Setup on this workspace

From `D:\arm-planning-control-project`:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --editable '.\project'
Set-Location '.\project'
```

Validate the fixed scenarios and run tests:

```powershell
& '..\.venv\Scripts\python.exe' -m armbench validate
& '..\.venv\Scripts\python.exe' -m pytest -q
```

Run a small smoke experiment (3 planning seeds, 2 control seeds):

```powershell
& '..\.venv\Scripts\python.exe' -m armbench run --quick --run-id smoke
```

Run the formal 30-seed planning experiment and configured control trials:

```powershell
& '..\.venv\Scripts\python.exe' -m armbench run
```

Each run is written to `results/<run_id>/`. An existing run directory is never
overwritten.

The verified local result snapshot is documented in
[`docs/RESULTS.md`](docs/RESULTS.md). Its formal run used commit `a19883c`, 30
paired planning seeds per scenario/planner, and five controller noise seeds.

## Output contract

```text
results/<run_id>/
  config.json
  environment.json
  per_trial.csv
  control_per_trial.csv
  aggregate.json
  summary.md
  paths/
  failures/
  control_traces/
  figures/
  run.log
```

Planning P50/P95 includes successes and failures; path statistics use successful
trials only. Raw failures remain in `per_trial.csv` and receive a JSON record.
Three synthetic diagnostics verify the start-collision, goal-collision, and
deadline failure paths but are excluded from performance aggregates.

## Upstream boundary

The pinned PythonRobotics checkout lives outside this repository at
`..\upstream\PythonRobotics` and remains unmodified. It was used as a reference
for the Panda parameter table and upstream teaching baseline. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.

## Limitations

This version uses spherical workspace obstacles, does not check self-collision,
and uses interpolation rather than an analytic swept-volume guarantee. Timing
enforces velocity but not acceleration/jerk. Tracking uses simplified decoupled
joint dynamics, not a rigid-body simulator or real robot. These constraints are
detailed in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and must remain visible
in any resume or interview description.

## Evidence-based resume wording

> Built a CPU-only 7-DoF manipulator planning and tracking benchmark with
> whole-link collision checks, bounded RRT-Connect/RRT*, shortcut smoothing,
> and delayed PD/LQR control. Across three fixed scenarios and 30 paired seeds,
> RRT-Connect achieved 30/30 success per scenario; in the constrained passage
> its P95 first-solution latency was 622 ms versus the RRT* baseline's 2013 ms
> at 5/30 success under a 2 s deadline. Reported Wilson intervals, retained all
> failures, and separated tracking collisions from joint-limit violations.

This wording is specific to the checked-in configuration and the machine in the
result report. It must not be reframed as real-robot or rigid-body validation.
