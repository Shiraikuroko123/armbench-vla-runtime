# pi0.5-LIBERO independent-clock visual success

This artifact is a single authentic success clip captured with the same
attested checkpoint and independent-clock runtime as the core pilot. It exists
for visual inspection and debugging, not for estimating task success.

## Result

- Suite/task/episode: LIBERO Spatial / task 0 / episode 0
- Seed: 7; control period: 50 ms; deadline: 200 ms
- Completed: 1/1; task success: 1/1
- Episodes with measured inference/simulation overlap: 1/1
- Control ticks during inference: 105/106
- Execute / hold ticks: 90 / 16
- Deadline-exceeded / failed responses: 0 / 0
- Video: `evaluation/videos/libero_spatial__task_000__episode_00.mp4`

This curated media run must not be pooled with the 40-rollout core result. It
is simulation-only and does not establish a leaderboard score, hard-real-time
guarantee, hardware safety, or real-robot deployment.

## Recompute

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  'evidence\pi05_libero_independent_clock_visual_success_001\evaluation' --json
```

The full request trace, initial state, provenance snapshot, manifest, and video
are retained under `evaluation/`.
