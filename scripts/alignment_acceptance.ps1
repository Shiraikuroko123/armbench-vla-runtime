[CmdletBinding()]
param(
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$ArmbenchPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ArmbenchPython -PathType Leaf)) {
    throw "Python environment not found: $ArmbenchPython. Complete the README setup first."
}

$AcceptanceArgs = @(
    "-m", "integrations.openpi.alignment_acceptance"
)
if ($NoOpen) {
    $AcceptanceArgs += "--no-open"
}

Push-Location $ProjectRoot
try {
    & $ArmbenchPython @AcceptanceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Temporal-alignment evidence acceptance failed."
    }
}
finally {
    Pop-Location
}
