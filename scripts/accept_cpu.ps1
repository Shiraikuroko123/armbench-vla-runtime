[CmdletBinding()]
param(
    [switch]$RequireOfficial,
    [switch]$FullTests,
    [switch]$Json,
    [string]$Python,
    [string]$OfficialPython,
    [double]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $WorkspaceRoot ".venv\Scripts\python.exe")
)

if ($Python) {
    $ArmbenchPython = [System.IO.Path]::GetFullPath($Python)
}
else {
    $ArmbenchPython = $PythonCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $ArmbenchPython -or -not (Test-Path -LiteralPath $ArmbenchPython -PathType Leaf)) {
    $Expected = $PythonCandidates -join " or "
    throw "ArmBench Python was not found. Run scripts\setup_local.ps1 first. Expected $Expected"
}

$Arguments = @(
    (Join-Path $ProjectRoot "scripts\accept_cpu.py"),
    "--python", $ArmbenchPython,
    "--timeout-s", $TimeoutSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($OfficialPython) {
    $Arguments += @("--official-python", [System.IO.Path]::GetFullPath($OfficialPython))
}
if ($RequireOfficial) {
    $Arguments += "--require-official"
}
if ($FullTests) {
    $Arguments += "--full-tests"
}
if ($Json) {
    $Arguments += "--json"
}

Push-Location $ProjectRoot
try {
    & $ArmbenchPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CPU acceptance failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
