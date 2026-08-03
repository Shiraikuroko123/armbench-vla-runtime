# Formal result snapshot

## Provenance

- Run ID: `formal_v1_30seed_20260803`
- Code commit: `a19883ce551a2ec035002cc51ccf122ded3c6092`
- Git state at run start: clean
- Python: 3.10.8
- NumPy / Matplotlib / pytest: 1.26.4 / 3.7.2 / 8.3.5
- CPU: Intel Core i9-12900H, Windows 10.0.26200
- Planning protocol: 30 paired seeds (`0..29`) for each scenario and planner
- Control protocol: five noise seeds for every controller/delay/load condition

The complete local artifact is in
`results/formal_v1_30seed_20260803/`. It contains 180 planning trials, 60
control trials, 155 verified successful paths, 25 benchmark timeout records,
three synthetic failure diagnostics, compressed traces, and figures.

## Planning

Latency includes successful and failed trials. Path length is computed on
successful trials only. Parentheses show Wilson 95% success intervals.

| Scenario | Planner | Success | P50 ms | P95 ms | Mean raw length | Mean smoothed length |
|---|---|---:|---:|---:|---:|---:|
| free_space | RRT-Connect | 30/30, 100% (88.6-100%) | 3.8 | 6.6 | 2.653 | 2.653 |
| free_space | RRT* | 30/30, 100% (88.6-100%) | 3.3 | 4.2 | 2.653 | 2.653 |
| single_block | RRT-Connect | 30/30, 100% (88.6-100%) | 47.1 | 94.3 | 4.498 | 3.348 |
| single_block | RRT* | 30/30, 100% (88.6-100%) | 232.1 | 1170.5 | 3.507 | 3.351 |
| narrow_passage | RRT-Connect | 30/30, 100% (88.6-100%) | 424.5 | 621.7 | 8.706 | 5.644 |
| narrow_passage | RRT* | 5/30, 16.7% (7.3-33.6%) | 2003.2 | 2012.8 | 4.745 | 4.735 |

The constrained scene favors rapid bidirectional connection. This result does
not establish general planner superiority: the RRT* baseline stops at its first
feasible solution, both planners use one fixed parameter set, obstacles are
spheres, and there is no self-collision model. RRT-Connect's raw paths are also
longer; shortcut smoothing closes much of that gap.

## Tracking

The controller follows the same smoothed `single_block` reference. Reported
values are means over five noise seeds. Load is a scalar multiplier on the
simplified joint inertia.

| Controller | Delay ms | Load | RMSE rad | Collision samples | Limit samples | Invalid edge intervals |
|---|---:|---:|---:|---:|---:|---:|
| PD | 0 | 1.00 | 0.0141 | 0.0 | 0.0 | 0.0 |
| PD | 40 | 1.00 | 0.0166 | 0.0 | 0.0 | 0.0 |
| PD | 80 | 1.00 | 0.0358 | 16.0 | 0.0 | 18.0 |
| PD | 80 | 1.25 | 0.0355 | 12.8 | 0.0 | 14.8 |
| LQR | 0 | 1.00 | 0.0298 | 0.0 | 0.0 | 0.0 |
| LQR | 40 | 1.00 | 0.0270 | 0.0 | 0.0 | 0.0 |
| LQR | 80 | 1.00 | 0.0340 | 0.0 | 0.0 | 0.0 |
| LQR | 80 | 1.25 | 0.0394 | 0.0 | 0.0 | 0.0 |

PD has lower nominal-delay RMSE, but it intersects the obstacle under 80 ms
delay. LQR has somewhat higher RMSE yet remains collision-free in these trials.
This is the main reason safety counters are reported separately from tracking
error. It is an observation in a decoupled double-integrator simulation, not a
claim about physical Panda control.

## Reproduction

From the project directory:

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
& '..\.venv\Scripts\python.exe' -m armbench validate
& '..\.venv\Scripts\python.exe' -m armbench run --run-id <new_run_id>
```

The output directory is never overwritten. Compare raw `per_trial.csv` files,
not only formatted summaries, when checking another run.

