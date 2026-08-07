# Contributing to ArmBench

ArmBench accepts changes that strengthen runtime correctness, reproducibility,
Panda simulation, artifact validation, or documentation without broadening the
published claims beyond available evidence.

## Development setup

Use the [local CPU setup](docs/LOCAL_SETUP.md), then run:

```powershell
& '.\.venv\Scripts\python.exe' -m ruff check .
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Keep changes focused and include tests for behavioral modifications. CPU-only
tests must not download checkpoints or contact an inference service.

## Evidence rules

- Treat existing directories under `evidence/` as immutable records.
- Write diagnostics and reruns under a new, unique run ID.
- Preserve source snapshots, resolved configuration, seeds, environment data,
  manifests, and SHA-256 records for evidence-bearing runs.
- Put large new evidence archives in a GitHub Release. Keep only summaries,
  manifests, checksums, and representative media in the Git tree.
- Report null and invalidated experiments alongside positive results.

## Pull requests

Describe the runtime or evidence contract affected by the change. Include the
commands used for validation and distinguish live-policy evidence from
scripted, replayed, or synthetic tests. Do not describe simulation results as
real-robot safety evidence.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.
