[CmdletBinding()]
param(
    [switch]$Visualize,
    [ValidateSet("single_block_goal", "narrow_gate_payload_delay_goal")]
    [string]$Case = "narrow_gate_payload_delay_goal"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $WorkspaceRoot ".venv\Scripts\python.exe")
)
$ArmbenchPython = $PythonCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1

if (-not $ArmbenchPython) {
    $Expected = $PythonCandidates -join " or "
    throw "ArmBench Python was not found. Run scripts\setup_local.ps1 first. Expected $Expected"
}

function Invoke-Armbench([string[]]$CommandArguments) {
    & $ArmbenchPython -m armbench @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "armbench $($CommandArguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$FaultArtifact = Join-Path $ProjectRoot "reports\integrated_panda_fault_matrix_001"
$CpuRuntimeArtifact = Join-Path $ProjectRoot "reports\cpu_runtime_completion_001"
$TaskArtifact = Join-Path $ProjectRoot "reports\integrated_panda_task_001"

Push-Location $ProjectRoot
try {
    Write-Host "[1/4] Checking the local runtime"
    Invoke-Armbench @("doctor")

    Write-Host "[2/4] Recomputing the 27-case integrated fault matrix"
    Invoke-Armbench @("vla-integrated-fault-validate", $FaultArtifact)

    Write-Host "[3/4] Replaying the 17-case asynchronous publication matrix"
    Invoke-Armbench @("vla-cpu-runtime-validate", $CpuRuntimeArtifact)

    Write-Host "[4/4] Replanning and re-executing the two MuJoCo task cases"
    Invoke-Armbench @("vla-integrated-task-validate", $TaskArtifact)

    Write-Host "Integrated Panda acceptance passed."

    if ($Visualize) {
        if ($Case -eq "single_block_goal") {
            $Scenario = "single_block"
            $Payload = "0.0"
        }
        else {
            $Scenario = "narrow_gate"
            $Payload = "0.5"
        }
        $Trace = Join-Path $TaskArtifact "traces\$Case.npz"
        Write-Host "Opening recorded MuJoCo trace: $Case"
        Invoke-Armbench @(
            "mujoco-view",
            "--scenario", $Scenario,
            "--clearance-mm", "20",
            "--payload", $Payload,
            "--trace", $Trace,
            "--array", "actual_positions",
            "--play"
        )
    }
}
finally {
    Pop-Location
}
