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
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-fault-validate `
  reports\integrated_panda_fault_matrix_001
& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-task-validate `
  reports\integrated_panda_task_001
.\scripts\vla_demo.cmd -CheckOnly
```

The two integrated validators rebuild the registered supervisor inputs. The
task validator also replans and re-executes both torque-controlled MuJoCo cases,
so it normally takes tens of seconds on a laptop CPU. On the current nested
Windows workspace, run both through a script that resolves the repository and
virtual-environment paths. Use an absolute `-File` path when launching outside
the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'
```

After the numeric check, replay the registered self-collision edge with the
contact-point overlay:

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-view `
  'D:\arm-planning-control-project\project\reports\mujoco_self_collision_audit_001' `
  --stratum known_intermediate --edge-index 0 --speed 0.75 --skip-validation
```

Treat the viewer as visual evidence only; the manifest-backed validator remains
the acceptance authority.

Replay the registered payload-and-delay task after numeric acceptance:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

This viewer uses the saved `actual_positions` physics trace. The validator,
not the animation, determines whether planning, assurance, physics, and stored
metrics agree.

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
| Integrated supervisor and two-task physics rerun | No | No | No |
| Artifact validation and dashboards | No | No | No |
| Live `pi0.5` inference | Yes | Usually | Yes |

If `doctor` reports `BLOCKED`, resolve its required `FAIL` entries first.
Missing `openpi_client` or FFmpeg is informational unless a command explicitly
requires live VLA transport or video encoding.
