$ErrorActionPreference = "Stop"

$ProjectRoot = Join-Path $PSScriptRoot "..\rust\01-wgpu-vector-add"
$TargetDir = Join-Path $env:LOCALAPPDATA "Lentica\cargo-target\wgpu-vector-add"
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

if (Test-Path $CargoBin) {
    $env:PATH = "$CargoBin;$env:PATH"
}

$env:CARGO_INCREMENTAL = "0"
$env:CARGO_TARGET_DIR = $TargetDir

Push-Location $ProjectRoot
try {
    cargo run
}
finally {
    Pop-Location
}
