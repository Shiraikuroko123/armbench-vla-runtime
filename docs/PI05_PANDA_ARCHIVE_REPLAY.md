# Frozen pi0.5 Response Replay on the Panda Guard Path

Status: Current offline integration evidence

## Purpose

This workflow closes the next verifiable boundary after the scripted Cartesian
adapter smoke. It takes action chunks previously produced by the official
Physical Intelligence `pi0.5` LIBERO checkpoint, verifies their preserved
provenance and hashes, then replays a deterministic stratified sample through
the local MuJoCo Panda adapter and runtime guard.

It is a cross-controller offline diagnostic. The policy checkpoint is not
executed again, the chunks are not connected by Panda observations, and no task
success is measured.

```text
frozen official pi0.5 response (10 x 7)
              |
              v
LIBERO semantics + clipping + Panda differential IK
              |
              v
joint velocity / acceleration / deadline / collision guard
              |
              v
auditable per-case CSV + aggregate JSON + file manifest
```

## Source validation

Before any replay case runs, the command verifies:

1. the source manifest inventory, file sizes, individual hashes, and aggregate
   hash;
2. the official `pi05_libero` policy configuration, checkpoint URI, checkpoint
   content hash, and clean OpenPI server attestation;
3. the declared LIBERO action space and action-adapter source hash;
4. every NPZ key, dtype, and shape;
5. all bootstrap-to-transition reference chains and overlap-scheduler
   equations; and
6. all 7,934 canonical response-action hashes, not only the selected sample.

The source `evidence/` directory is read-only. Replay outputs are new derived
reports and never rewrite the frozen experiment.

## Reproduce on CPU

From the repository root on Windows:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-archive-replay `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_archive_replay `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-archive-replay-validate `
  results\pi05_panda_archive_replay `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

Use `./.venv/bin/python` and forward-slash paths on Linux. The command needs the
pinned MuJoCo Menagerie Panda model but no GPU, OpenPI server, or checkpoint
download. Output directories must not already exist.

The selection is equal across the 30 `(task_id, method)` strata: 10 LIBERO-10
tasks, three runtime methods, and three chunks per stratum. Every chunk starts
independently from each scenario's declared Panda start state, with a fresh
guard, so deadline latches and commanded velocity do not leak between cases.

## Preserved 90-chunk report

The checked-in derived report is
[`reports/pi05_panda_archive_replay_90_001`](../reports/pi05_panda_archive_replay_90_001/summary.md).
It contains 90 selected chunks and 270 independent cases across `free_space`,
`single_block`, and `narrow_gate`.

| Observation | Result |
| --- | ---: |
| Source response hashes recomputed | 7,934 / 7,934 |
| Selected chunks containing clipped input | 89 / 90 |
| Raw Panda lookahead paths invalid | 36 / 270 |
| Cases with guard intervention | 226 / 270 |
| Source deadline exceeded at 200 ms | 3 / 270 (one chunk in three scenes) |
| Guarded paths valid under 0.02 rad edge sampling | 270 / 270 |
| Cases satisfying all guard constraints | 264 / 270 |
| Collision/acceleration conflict cases | 6 / 270 |

The six conflict cases are an important negative result. The guard selected a
collision-valid stop/hold path, but the required immediate velocity change
exceeded the configured acceleration bound. The report therefore does not call
those cases safe. This exposes an empty-feasible-set problem for a future
deadline-bounded trajectory repair method rather than hiding it behind a task
success score.

Local adapter and guard timings are recorded in the report, together with OS,
Python, NumPy, MuJoCo, Panda scene, and implementation-source fingerprints.
They are host measurements, not hard-real-time worst-case guarantees.

## Files

- `provenance.json`: source/checkpoint attestations, explicit claim flags,
  selection, runtime environment, and implementation hashes.
- `per_chunk.csv`: one row per selected chunk and Panda scenario.
- `summary.json`: aggregates reproducible from the CSV.
- `summary.md`: compact human review surface.
- `manifest.json`: exact file inventory, byte sizes, and SHA-256 hashes.

## Claim boundary

- `source_policy_checkpoint_attested=true` means the frozen responses trace to
  the attested official checkpoint.
- `policy_checkpoint_executed_in_replay=false` means this CPU workflow did not
  run `pi0.5` inference.
- `task_success_evaluated=false` means these numbers are not LIBERO or Panda
  task-success results.
- `panda_closed_loop_executed=false` means the Panda did not generate new
  observations for subsequent policy calls.
- Differential IK is not dynamically equivalent to LIBERO's torque-level OSC,
  and resolution-bounded edge sampling is not continuous-collision or physical
  safety certification.
