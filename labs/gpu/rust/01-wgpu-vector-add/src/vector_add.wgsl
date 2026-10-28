@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&output)) {
        output[index] = input_a[index] + input_b[index];
    }
}
