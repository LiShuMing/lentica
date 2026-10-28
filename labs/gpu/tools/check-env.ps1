$ErrorActionPreference = "Stop"

Write-Host "Windows GPU:"
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM

Write-Host ""
Write-Host "WSL distributions:"
wsl -l -v

Write-Host ""
Write-Host "WSL user and kernel:"
wsl bash -lc "whoami; uname -a"

Write-Host ""
Write-Host "Optional ROCm tools:"
wsl bash -lc "command -v rocminfo || true; command -v rocm-smi || true; command -v hipcc || true"
