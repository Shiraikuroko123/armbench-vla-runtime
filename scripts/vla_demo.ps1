[CmdletBinding()]
param(
    [switch]$Formal,
    [switch]$NoVideos,
    [switch]$CheckOnly,
    [string]$Python,
    [string]$RunId = ("vla_demo_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $WorkspaceRoot ".venv\Scripts\python.exe")
)
if ($Python) {
    $ArmbenchPython = $Python
}
else {
    $ArmbenchPython = $PythonCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $ArmbenchPython) {
    throw "Python environment not found. Run .\scripts\setup_local.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $ArmbenchPython -c `
        "import mujoco, armbench; print(f'ArmBench local runtime OK; MuJoCo {mujoco.__version__}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency import check failed. Install the project[test] extras."
    }

    & $ArmbenchPython -m armbench doctor
    if ($LASTEXITCODE -ne 0) {
        throw "ArmBench environment validation failed."
    }

    & $ArmbenchPython -m armbench mujoco-validate
    if ($LASTEXITCODE -ne 0) {
        throw "MuJoCo scenario validation failed."
    }

    & $ArmbenchPython -m pytest `
        tests\test_async_runtime.py tests\test_environment.py -q
    if ($LASTEXITCODE -ne 0) {
        throw "Local runtime/environment tests failed."
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
