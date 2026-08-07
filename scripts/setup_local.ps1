[CmdletBinding()]
param(
    [switch]$WithVla,
    [switch]$SkipModels,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$MenagerieRoot = Join-Path $ProjectRoot ".cache\mujoco_menagerie"
$PandaScene = Join-Path $MenagerieRoot "franka_emika_panda\scene.xml"
$MenagerieCommit = "71f066ad0be9cd271f7ed58c030243ef157af9f4"

function Assert-Success([string]$Action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $Python -m venv $VenvRoot
    Assert-Success "Creating the Python environment"
}

& $VenvPython -m pip install --upgrade pip
Assert-Success "Updating pip"

$InstallTarget = ".[test]"
if ($WithVla) {
    $InstallTarget = ".[test,vla]"
}
Push-Location $ProjectRoot
try {
    & $VenvPython -m pip install -e $InstallTarget
    Assert-Success "Installing ArmBench"
}
finally {
    Pop-Location
}

if (-not $SkipModels -and -not (Test-Path -LiteralPath $PandaScene -PathType Leaf)) {
    if (Test-Path -LiteralPath $MenagerieRoot) {
        throw "Incomplete model cache at $MenagerieRoot. Move it aside and rerun setup."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $MenagerieRoot) | Out-Null
    & git clone --filter=blob:none --no-checkout `
        https://github.com/google-deepmind/mujoco_menagerie.git $MenagerieRoot
    Assert-Success "Cloning MuJoCo Menagerie"
    & git -C $MenagerieRoot sparse-checkout init --cone
    Assert-Success "Initializing the sparse model checkout"
    & git -C $MenagerieRoot sparse-checkout set franka_emika_panda
    Assert-Success "Selecting the Panda model"
    & git -C $MenagerieRoot checkout --detach $MenagerieCommit
    Assert-Success "Checking out the pinned Menagerie revision"
}

$DoctorArgs = @("-m", "armbench", "doctor")
if ($WithVla) {
    $DoctorArgs += "--require-vla"
}
& $VenvPython @DoctorArgs
Assert-Success "Validating the ArmBench environment"

Write-Host ""
Write-Host "ArmBench is ready."
Write-Host "Python: $VenvPython"
Write-Host "Viewer: & '$VenvPython' -m armbench mujoco-view --scenario narrow_gate"
