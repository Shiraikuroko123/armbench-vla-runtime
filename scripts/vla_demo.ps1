[CmdletBinding()]
param(
    [switch]$Formal,
    [switch]$NoVideos,
    [switch]$CheckOnly,
    [string]$RunId = ("vla_demo_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$ArmbenchPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
$MenagerieScene = Join-Path $WorkspaceRoot `
    "upstream\mujoco_menagerie\franka_emika_panda\scene.xml"

if (-not (Test-Path -LiteralPath $ArmbenchPython -PathType Leaf)) {
    throw "Python environment not found: $ArmbenchPython. Run the README setup first."
}
if (-not (Test-Path -LiteralPath $MenagerieScene -PathType Leaf)) {
    throw "Pinned MuJoCo Menagerie Panda model not found: $MenagerieScene"
}

Push-Location $ProjectRoot
try {
    & $ArmbenchPython -c `
        "import mujoco, openpi_client, armbench; print(f'Python VLA client OK; MuJoCo {mujoco.__version__}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency import check failed. Install the project[test,vla] extras."
    }

    & $ArmbenchPython -m armbench mujoco-validate
    if ($LASTEXITCODE -ne 0) {
        throw "MuJoCo scenario validation failed."
    }

    & $ArmbenchPython -m pytest tests\test_vla.py -q
    if ($LASTEXITCODE -ne 0) {
        throw "VLA contract/runtime tests failed."
    }

    if ($CheckOnly) {
        Write-Host "Checks passed. No benchmark was run."
        return
    }

    $BenchmarkArgs = @(
        "-m", "armbench", "vla-guard-run",
        "--run-id", $RunId
    )
    if (-not $Formal) {
        $BenchmarkArgs += "--quick"
    }
    if ($NoVideos) {
        $BenchmarkArgs += "--no-videos"
    }

    & $ArmbenchPython @BenchmarkArgs
    if ($LASTEXITCODE -ne 0) {
        throw "VLA benchmark failed."
    }
}
finally {
    Pop-Location
}
