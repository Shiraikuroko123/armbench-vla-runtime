[CmdletBinding()]
param(
    [switch]$NoOpen
)

$CompatibilityTarget = Join-Path $PSScriptRoot "alignment_acceptance.ps1"
if ($NoOpen) {
    & $CompatibilityTarget -NoOpen
}
else {
    & $CompatibilityTarget
}
