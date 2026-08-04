# pi0.5 LIBERO pilot diagnostic

This is an exploratory diagnostic of `pi05_libero_pilot_002`, not a
confirmatory performance claim. The run used all ten `libero_spatial` tasks,
initial states 45 and 46, `replan_steps=5`, and an injected four-step delay
(200 ms at 20 Hz). The two modes were evaluated on matched task and initial
state conditions.

## Integrity and efficacy

- Planned/completed rollouts: 40/40.
- Non-efficacy runtime or contract failures: 0.
- Root validation: 71 files checked, `valid=true`, no warnings or errors.
- `async_unguarded`: 7/20 successes (35%).
- `state_guard`: 6/20 successes (30%).
- Paired difference, guard minus unguarded: -5 percentage points.
- Paired bootstrap 95% interval: [-30, 20] percentage points.
- Exact McNemar p-value after the registered correction: 1.0.

The pilot is too small to establish superiority or equivalence. The interval
includes both material benefit and material harm.

## Guard failure mechanism

The state guard produced 282 `rejected_state_mismatch` decisions. Every one
included `position_mismatch`; only one also included `orientation_mismatch`.
No rejection was driven by the gripper threshold.

Position displacement between query capture and response consumption:

| Decision | Queries | P50 | P90 | P95 |
| --- | ---: | ---: | ---: | ---: |
| Accepted by state guard | 195 | 0.00598 m | 0.00893 m | 0.00943 m |
| Rejected by state guard | 282 | 0.02648 m | 0.04855 m | 0.05115 m |

The configured position threshold was 0.01 m. During the injected delay the
environment intentionally continued executing the previous action. The guard
therefore treated expected commanded motion as if it were unmodeled state
drift. Eleven of twenty guard episodes terminated at
`max_requeries_exceeded`; the guard averaged 14.1 rejections per episode.

This is evidence that the fixed displacement guard is misspecified for this
asynchronous execution model. It is not evidence that state validation is
generally harmful.

## Next registered engineering step

Keep the original guard and all negative results as baselines. Add a separate
training-free temporal-alignment mode that discards the returned action
chunk's first `latency_steps` actions before dispatch. With a ten-action pi0.5
chunk, horizon five, and delay four, this dispatches actions 4 through 8 rather
than stale actions 0 through 4. The mode must fail closed when the returned
chunk cannot supply `latency_steps + replan_steps` actions.

The temporal-alignment mode is a new exploratory mechanism. It must first run
on initial states disjoint from both this pilot and the planned confirmatory
states. Its result must not be merged into this artifact or described as a
preregistered result.

## Reproduction

From the repository root:

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' `
  -m integrations.openpi.libero_compose_run validate `
  evidence\pi05_libero_pilot_002\run
```

The immutable raw tables are in `run/evaluation/per_episode.csv` and
`run/evaluation/per_query.csv`; the registered summary is in
`run/evaluation/summary.md`. The independent validation output is stored next
to, rather than inside, the immutable run directory.
