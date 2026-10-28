# WGSL Notes

WGSL is the shader language used by WebGPU and `wgpu`.

Minimal compute shader shape:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
}
```

Key ideas:

- `global_invocation_id` is the global thread-like ID.
- `workgroup_size` controls how many invocations are grouped together.
- Storage buffers are the usual way to pass large arrays into compute shaders.
- Always guard array access when dispatch size is rounded up.
