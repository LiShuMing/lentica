use std::sync::mpsc;

use wgpu::util::DeviceExt;

const ELEMENTS: usize = 1024;
const WORKGROUP_SIZE: u32 = 64;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable GPU adapter found");

    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("failed to create device");

    let input_a: Vec<f32> = (0..ELEMENTS).map(|value| value as f32).collect();
    let input_b: Vec<f32> = (0..ELEMENTS).map(|value| (value as f32) * 2.0).collect();
    let zeroes = vec![0.0f32; ELEMENTS];
    let buffer_size = (ELEMENTS * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("output"),
        contents: bytemuck::cast_slice(&zeroes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vector_add.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("vector_add.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vector add pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &bind_group_layout,
        entries: &[
            bind_entry(0, &buffer_a),
            bind_entry(1, &buffer_b),
            bind_entry(2, &output),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((ELEMENTS as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    let result = read_f32_buffer(&device, &readback, ELEMENTS);
    for index in 0..ELEMENTS {
        let expected = input_a[index] + input_b[index];
        assert_eq!(result[index], expected);
    }

    println!("ok: {} GPU additions verified", ELEMENTS);
    println!("sample: {} + {} = {}", input_a[7], input_b[7], result[7]);
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, elements: usize) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().expect("failed to map readback buffer");

    let view = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&view)[..elements].to_vec();
    drop(view);
    buffer.unmap();
    result
}
