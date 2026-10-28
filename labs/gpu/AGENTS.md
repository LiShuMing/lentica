# GPU Labs

Follow root AGENTS.md and spec 0001. Keep Rust/wgpu/WGSL, HIP, DirectML, and CUDA
examples separate. Preserve hardware-specific notes as notes, not universal setup
claims. Use active/explicit environments and portable script paths. Do not run
installers, allocate large GPU workloads, or install drivers during ordinary CI.
Never commit target directories, native executables, notebook outputs, or .venv.
