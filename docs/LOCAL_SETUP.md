# Local CPU setup

Status: Operational

This path supports the MuJoCo Panda benchmarks, scripted VLA runtime checks,
artifact validation, and offline dashboards. It does not run a `pi0.5`
checkpoint and does not require a GPU.

## Requirements

- 64-bit CPython 3.10
- Git
- Windows 10/11 with PowerShell 5.1+, or a maintained Linux distribution
- A desktop session for the interactive MuJoCo viewer

The repository, Python environment, model assets, and generated files require
roughly 1-2 GB beyond the existing evidence checkout.

## Automated setup

Run from the repository root.

Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
```

Linux:

```bash
./scripts/setup_local.sh
```

Both scripts create `.venv`, install `.[test]`, and sparse-checkout the Panda
assets from MuJoCo Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4` into
`.cache/mujoco_menagerie`. Pass `-WithVla` or `--with-vla` to install the
OpenPI client adapter as well. A model checkpoint and an inference server are
still separate requirements for live policy execution.

## Manual setup

Create and install the environment:

```powershell
py -3.10 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

Provide the Panda model through one of these environment variables:

```powershell
$env:ARMBENCH_PANDA_SCENE = 'C:\models\mujoco_menagerie\franka_emika_panda\scene.xml'
# Or point to the Menagerie repository root:
$env:ARMBENCH_MENAGERIE_ROOT = 'C:\models\mujoco_menagerie'
```

Without an override, ArmBench searches the repository-local `.cache` first
and then the legacy workspace-level `upstream/mujoco_menagerie` directory.
`armbench doctor --json` reports the exact resolved paths for automation.

## Acceptance commands

```powershell
& '.\.venv\Scripts\python.exe' -m armbench doctor
& '.\.venv\Scripts\python.exe' -m armbench mujoco-validate
.\scripts\vla_demo.cmd -CheckOnly
```

Open the interactive viewer only from a desktop session:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario narrow_gate --clearance-mm 20 --payload 0.5
```

Stored evidence can be validated without model inference:

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd -NoOpen
```

## What works without a server

| Capability | GPU | Network after setup | OpenPI server |
| --- | --- | --- | --- |
| Panda planning, control, and viewer | No | No | No |
| Scripted runtime and fault checks | No | No | No |
| Artifact validation and dashboards | No | No | No |
| Live `pi0.5` inference | Yes | Usually | Yes |

If `doctor` reports `BLOCKED`, resolve its required `FAIL` entries first.
Missing `openpi_client` or FFmpeg is informational unless a command explicitly
requires live VLA transport or video encoding.
