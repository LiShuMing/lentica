# AMD Stack Notes

Fill this in as you inspect the machine.

## Windows

GPU:

```text
TODO
```

Driver:

```text
TODO
```

Check commands:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```

```powershell
dxdiag
```

## WSL

Distribution:

```bash
wsl -l -v
```

GPU visibility:

```bash
lspci | grep -Ei 'vga|display|amd|radeon'
```

ROCm tools, if installed:

```bash
rocminfo
rocm-smi
hipcc --version
```

## Decision Log

| Date | Decision | Reason |
| --- | --- | --- |
| TODO | Start with Rust + wgpu | Stable AMD path on Windows through DX12/Vulkan. |
