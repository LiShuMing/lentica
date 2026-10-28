# wgpu Vector Add

This is the first Rust compute-shader exercise.

Run from this directory:

```bash
cargo run
```

On Windows, prefer the repository helper script:

```powershell
..\..\tools\run-wgpu-vector-add.ps1
```

The script stores Cargo build output in `C:\tmp\gpu-lab-target\wgpu-vector-add`. This avoids lock-file and executable-copy errors that can happen when Windows Cargo writes `target/` into a `\\wsl$` path.

In WSL, GPU compute through `wgpu` may depend on the local graphics stack, so Windows native execution is the preferred first path.
