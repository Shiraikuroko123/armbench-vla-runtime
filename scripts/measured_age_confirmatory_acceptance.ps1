[CmdletBinding()]
param(
    [Alias("no-open")]
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$ArmbenchPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ArmbenchPython -PathType Leaf)) {
    throw "Python environment not found: $ArmbenchPython. Run the README setup first."
}

$AcceptanceArgs = @(
    "-m", "integrations.openpi.measured_age_confirmatory_acceptance"
)
if ($NoOpen) {
    $AcceptanceArgs += "--no-open"
}

Push-Location $ProjectRoot
try {
    & $ArmbenchPython @AcceptanceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Measured-age confirmatory acceptance failed. Do not present the result as validated."
    }
}
finally {
    Pop-Location
}
