# pi0.5-LIBERO independent-clock core pilot

This artifact records the frozen 40-rollout ArmBench G02 pilot. The official,
content-attested `pi05_libero` checkpoint ran in an OpenPI server process while
LIBERO simulation advanced in a separate process with its own control clock.

## Result

- Suite: LIBERO Spatial
- Matrix: tasks 0-9, episodes 0-3, 40 rollouts, seed 7
- Control period: 50 ms (20 Hz); action horizon: 10; deadline: 200 ms
- Completed: 40/40
- Task success: 38/40 (95.0%)
- Episodes with measured inference/simulation overlap: 40/40
- Control ticks during inference: 4,521/4,623
- Execute / hold ticks: 4,031 / 592
- Deadline-exceeded / failed responses: 0 / 0

This is a simulation pilot, not an official LIBERO leaderboard score. It does
not claim hard-real-time scheduling, hardware safety, real-robot deployment,
training or fine-tuning, or method superiority. The two failed episodes are
preserved under `evaluation/videos/` and remain part of the result.

## Recompute

From the repository root, with the project environment installed:

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  'evidence\pi05_libero_independent_clock_core_40_001\evaluation' --json
```

The validator checks manifest hashes, initial states, request lifecycles,
action chunks, independent process IDs, control ticks, derived aggregates,
checkpoint attestation, and the runtime source snapshot. `validator_recomputed.json`
is the last local validation output; the complete raw run is also available as a
release archive when published.
