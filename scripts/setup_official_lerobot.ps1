param(
    [string]$VenvPath = "",
    [string]$Python = "C:\Python310\python.exe"
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $VenvPath) {
    $VenvPath = Join-Path $repo ".venv-lerobot-0.4.4"
}
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
if (Test-Path -LiteralPath $VenvPath) {
    throw "Refusing to reuse an existing LeRobot environment: $VenvPath"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python 3.10 interpreter not found: $Python"
}

& $Python -m venv $VenvPath
Assert-NativeSuccess "Create LeRobot virtual environment"
$venvPython = Join-Path $VenvPath "Scripts\python.exe"
$constraints = Join-Path $PSScriptRoot "official_lerobot_windows_py310_constraints.txt"

# LeRobot 0.4.4 resolves to a NumPy 2 environment through rerun-sdk. Keep it
# separate from ArmBench/OpenPI, which deliberately pins NumPy 1.26.4.
& $venvPython -m pip install --disable-pip-version-check `
    --constraint $constraints `
    "lerobot==0.4.4"
Assert-NativeSuccess "Install pinned LeRobot environment"

# These ArmBench imports are needed by the shared CLI. Versions here are
# compatible with NumPy 2 and are intentionally outside ArmBench's main env.
& $venvPython -m pip install --disable-pip-version-check `
    --constraint $constraints `
    "matplotlib==3.10.9" `
    "mujoco==3.11.0" `
    "osqp==1.1.3" `
    "scipy==1.15.3" `
    "websockets==16.1.1" `
    "pytest==8.3.5" `
    "ruff==0.12.12"
Assert-NativeSuccess "Install ArmBench loader dependencies"

# Expose the checkout without registering ArmBench's NumPy 1.26 package
# metadata in this NumPy 2 loader environment.
$sitePackages = (& $venvPython -c "import site; print(site.getsitepackages()[0])").Trim()
Assert-NativeSuccess "Resolve LeRobot site-packages"
$sourcePath = Join-Path $repo "src"
$sourcePth = Join-Path $sitePackages "armbench_checkout.pth"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $sourcePth,
    $sourcePath + [Environment]::NewLine,
    $utf8NoBom
)

& $venvPython -c "import armbench, importlib.metadata as m; assert m.version('lerobot') == '0.4.4'; print('lerobot=' + m.version('lerobot')); print('numpy=' + m.version('numpy')); print('armbench=' + armbench.__file__)"
Assert-NativeSuccess "Validate isolated imports"
& $venvPython -m pip check
Assert-NativeSuccess "Validate isolated dependency graph"
Write-Output "LeRobot environment: $VenvPath"
Write-Output "Smoke command:"
Write-Output "& '$venvPython' -m armbench vla-lerobot-official-smoke --output-directory reports\official_lerobot_roundtrip_001"
