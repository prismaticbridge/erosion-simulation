use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use bytemuck::NoUninit;
use honeycomb_3d::GpuContext;
use image::ImageBuffer;
use noise::{NoiseFn, Perlin};
use serde::Deserialize;
use wgpu::{ShaderModuleDescriptor, SubmissionIndex};

const WORKGROUP_SIZE: u32 = 16;

#[derive(Deserialize)]
pub struct Config {
    pub chunk_size: u32,
    pub iterations: u32,
    pub fbm_resolution: u32,
    pub fbm_octaves: u32,
    pub fbm_persistence: f64,
}

const SHADER_DIRECTORY: LazyLock<PathBuf> = LazyLock::new(|| {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        PathBuf::from(manifest_dir).join("shaders")
    } else {
        // Fallback for standalone release distributions outside of cargo
        let mut exe_path = std::env::current_exe().expect("Error: failed to get exe path");
        exe_path.push("shaders");
        exe_path
    }
});

pub struct TerrainGenContext {
    // These are GPU objects required for chunk generation
    accumulate_pipeline: wgpu::ComputePipeline,
    erosion_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,
    // thermal_pipeline: wgpu::ComputePipeline,
    rain_bind_group: wgpu::BindGroup,
    sobel_read_bind_group: wgpu::BindGroup,
    sobel_write_bind_group: wgpu::BindGroup,
    heightmap_bind_groups: [wgpu::BindGroup; 2],
    water_sediment_bind_groups: [wgpu::BindGroup; 2],
    //ping pong textures that are swapped every iteration
    //these aren't needed, but if they get dropped wgpu frees them and then the shaders all crash
    _rain_texture: wgpu::Texture,
    _gradient_texture: wgpu::Texture,
    heightmap_textures: [wgpu::Texture; 2], // gets read at the end
    water_sediment_textures: [wgpu::Texture; 2],

    temp_buffer: Option<wgpu::Buffer>,
    gpu: Arc<GpuContext>,

    // water_sediment_index: u8,
    texture_size: wgpu::Extent3d,

    pub config: Config,
}

fn create_2d_texture(
    device: &wgpu::Device,
    label: &'static str,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub struct LayoutDimensions {
    bytes_per_pixel: u32,
    padded_bytes_per_row: u32,
    buffer_size: u32,
    width: u32,
    height: u32,
    channels: u32,
}

pub fn get_layout_dimensions(texture: &wgpu::Texture) -> LayoutDimensions {
    let bytes_per_pixel = texture
        .format()
        .target_pixel_byte_cost()
        .expect("Cannot get pixels per byte for a texture");
    let unpadded_bytes_per_row = texture.width() * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let buffer_size = texture.height() * padded_bytes_per_row;

    let channels = texture.format().components() as u32;

    LayoutDimensions {
        bytes_per_pixel: bytes_per_pixel,
        padded_bytes_per_row,
        buffer_size,
        width: texture.width(),
        height: texture.height(),
        channels,
    }
}

fn read_to_buffer(
    texture: &wgpu::Texture,
    output_buffer: &mut wgpu::Buffer,
    dimensions: &LayoutDimensions,
    encoder: &mut wgpu::CommandEncoder,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfoBase {
            texture: texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfoBase {
            buffer: output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions.padded_bytes_per_row),
                rows_per_image: Some(dimensions.height),
            },
        },
        wgpu::Extent3d {
            width: dimensions.width,
            height: dimensions.height,
            depth_or_array_layers: 1,
        },
    );
}

fn read_to_cpu(
    buffer: &wgpu::Buffer,
    cpu_dest: &mut Vec<u8>,
    device: &wgpu::Device,
    submission_index: SubmissionIndex,
    dimensions: &LayoutDimensions,
) {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    let _ = device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
        .inspect_err(|e| eprintln!("Failed to poll device: {e}"));

    if rx.recv().is_ok() {
        if let Ok(read_data_slice) = slice.get_mapped_range() {
            let unpadded_bytes_per_row = dimensions.width * dimensions.bytes_per_pixel;
            for chunk in read_data_slice.chunks(dimensions.padded_bytes_per_row as usize) {
                cpu_dest.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
            }
        }
        buffer.unmap();
    } else {
        eprintln!("Error: failed to read data from gpu buffer");
    }
}

pub fn save_heightmap(
    texture: &wgpu::Texture,
    file: &str,
    mut encoder: wgpu::CommandEncoder,
    gpu: &GpuContext,
) {
    let layout_dimensions = get_layout_dimensions(&texture);
    // allocate a buffer of the correct size
    let mut temp_buffer = gpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
        label: None,
        size: layout_dimensions.buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // finish the compute pass
    read_to_buffer(&texture, &mut temp_buffer, &layout_dimensions, &mut encoder);
    let mut output_data = Vec::with_capacity(
        (layout_dimensions.width * layout_dimensions.height * layout_dimensions.bytes_per_pixel)
            as usize,
    );
    let submission_index = gpu.queue.submit(Some(encoder.finish()));
    read_to_cpu(
        &mut temp_buffer,
        &mut output_data,
        &gpu.device,
        submission_index,
        &layout_dimensions,
    );

    let output_floats: &[f32] = bytemuck::cast_slice(&output_data);

    // AI-generated snippet to print metrics, TODO delete it
    let mut max = 0.0;
    if !output_floats.is_empty() {
        let mut min = output_floats[0];
        let mut sum = 0.0;

        for &val in output_floats {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
            sum += val;
        }

        let avg = sum / output_floats.len() as f32;
        println!(
            "Texture Stats -> Min: {:.4}, Max: {:.4}, Avg: {:.4}",
            min, max, avg
        );
    } else {
        println!("Texture Stats -> Buffer is empty");
    }

    let channels = layout_dimensions.channels;

    match channels {
        1 => {
            let mut img: ImageBuffer<image::Luma<u16>, Vec<u16>> =
                ImageBuffer::new(layout_dimensions.width, layout_dimensions.height);

            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let idx = (y * layout_dimensions.width + x) as usize;
                let val_f32 = output_floats[idx];
                let val_u16 = (val_f32.clamp(0.0, 1.0) * 65535.0) as u16;

                *pixel = image::Luma([val_u16]);
            }

            let _ = img
                .save(file)
                .inspect_err(|e| eprintln!("Failed to save image to file: {e}"));
        }
        2 | 3 => {
            let mut img: ImageBuffer<image::Rgb<u16>, Vec<u16>> =
                ImageBuffer::new(layout_dimensions.width, layout_dimensions.height);

            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let base_idx = ((y * layout_dimensions.width + x) * channels as u32) as usize;

                let r_f32 = output_floats[base_idx];
                let g_f32 = output_floats[base_idx + 1];
                let b_f32 = if channels == 3 {
                    output_floats[base_idx + 2]
                } else {
                    0.0 // Pad blue channel to 0 for 2-channel textures
                };

                let r = (r_f32.clamp(0.0, 1.0) * 65535.0) as u16;
                let g = (g_f32.clamp(0.0, 1.0) * 65535.0) as u16;
                let b = (b_f32.clamp(0.0, 1.0) * 65535.0) as u16;

                *pixel = image::Rgb([r, g, b]);
            }

            let _ = img
                .save(file)
                .inspect_err(|e| eprintln!("Failed to save image to file: {e}"));
        }
        other => {
            eprintln!("Unsupported channel count ({other}) in save_heightmap");
        }
    }
}

/// Uploads the raw bytes of a slice to a gpu texture
fn upload_to_texture<T: NoUninit>(
    texture: &mut wgpu::Texture,
    data: &[T],
    gpu: &GpuContext,
    layout: LayoutDimensions,
) {
    let bytes = bytemuck::cast_slice(data);
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            // use the texture's pixel size, not size_of::<T>(), so multi-channel
            // formats (Rg32Float) get the right row pitch
            bytes_per_row: Some(layout.width * layout.bytes_per_pixel),
            rows_per_image: Some(layout.height),
        },
        wgpu::Extent3d {
            width: layout.width,
            height: layout.height,
            depth_or_array_layers: 1,
        },
    );
}

/// Generates a random 0..=1 noise field, box-blurs it by `blur_radius` texels
/// wraps at the edges so the result stays tileable
/// uploads it to a new R32Float texture
pub fn create_noise_map(size: wgpu::Extent3d, blur_radius: u32) -> Vec<f32> {
    let width = size.width as usize;
    let height = size.height as usize;

    // White noise from a cheap LCG, so we don't need a rand dependency.
    let mut state: u32 = 0x1234_5678;
    let mut noise = vec![0.0f32; width * height];
    for value in &mut noise {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *value = (state >> 8) as f32 / (1u32 << 24) as f32;
    }

    // Separable box blur: horizontal pass into `tmp`, vertical pass into `blurred`.
    let radius = blur_radius as isize;
    let kernel = (2 * radius + 1) as f32;
    let mut tmp = vec![0.0f32; noise.len()];
    let mut blurred = vec![0.0f32; noise.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for dx in -radius..=radius {
                let sx = (x as isize + dx).rem_euclid(width as isize) as usize;
                sum += noise[y * width + sx];
            }
            tmp[y * width + x] = sum / kernel;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for dy in -radius..=radius {
                let sy = (y as isize + dy).rem_euclid(height as isize) as usize;
                sum += tmp[sy * width + x];
            }
            blurred[y * width + x] = sum / kernel;
        }
    }

    return blurred;
}

/// Generates an initial heightmap using Fractional Brownian Motion
pub fn generate_chunkable_fbm(
    shape: (u32, u32),
    base_res: (u32, u32),
    octaves: u32,
    persistence: f64,
) -> Vec<f32> {
    let mut total = vec![0.0 as f32; (shape.0 * shape.1) as usize];
    let mut amplitude = 1.0;
    let mut total_amplitude = 0.0;
    let mut freq_mult = 1;

    let seed: u32 = rand::random();
    let perlin = Perlin::new(seed);

    for _ in 0..octaves {
        let res = (base_res.0 * freq_mult, base_res.1 * freq_mult);

        // check that axes are divisible by resolution
        if shape.0 % res.0 == 0 && shape.1 % res.1 == 0 {
            for y in 0..shape.0 {
                for x in 0..shape.1 {
                    // Map coordinates to the frequency resolution scale
                    let nx = (x as f64 / shape.1 as f64) * res.1 as f64;
                    let ny = (y as f64 / shape.0 as f64) * res.0 as f64;
                    let val = perlin.get([nx, ny]);

                    let idx = (y * shape.1 + x) as usize;
                    //graphics card needs f32
                    total[idx] += (val * amplitude) as f32;
                }
            }
            total_amplitude += amplitude;
        }

        amplitude *= persistence;
        freq_mult *= 2;
    }

    // Normalize by total amplitude
    if total_amplitude > 0.0 {
        for val in total.iter_mut() {
            *val /= total_amplitude as f32;
        }
    }

    total
}

impl TerrainGenContext {
    pub fn new(gpu: Arc<GpuContext>) -> Self {
        let config_str =
            std::fs::read_to_string("params.yaml").expect("Failed to read params file");
        let config: Config = noyalib::from_str(&config_str).expect("Failed to parse params file");

        let texture_size = wgpu::Extent3d {
            width: config.chunk_size,
            height: config.chunk_size,
            depth_or_array_layers: 1,
        };

        //Bind groups
        //index 0 rain, read-only
        let rain_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("accumulate bind group laytou"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });

        //index 1 sobel gradients / velocity
        //erosion shader writes, advection shader reads, so more efficient to split
        let sobel_read_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sobel read bind group layout"),
                    entries: &[
                        //normalized sobel gradients
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::ReadOnly,
                                format: wgpu::TextureFormat::Rg32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });
        let sobel_write_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sobel write bind group layout"),
                    entries: &[
                        //normalized sobel gradients
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rg32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        //index 2 heightmap
        //ping pong texture since its sampled and written in the erosion shader
        let heightmap_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("heightmap bind group layout"),
                    entries: &[
                        //heightmap texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        //sampler for heightmap texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        //texture to write height map
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        //index 3 water and sediment
        //ping pong texture since its sampled and written in the advection shader
        let water_sediment_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("water and sediment bind group"),
                    entries: &[
                        //water and sediment
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rg32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        //First compute pass: accumulate
        //handles rain and evaporation
        let accumulate_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("accumulate pipeline layout"),
                    bind_group_layouts: &[
                        Some(&rain_bind_group_layout),
                        None,
                        None,
                        Some(&water_sediment_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let accumulate_shader_path = SHADER_DIRECTORY.join("accumulate.wgsl");
        let accumulate_shader_source = std::fs::read_to_string(&accumulate_shader_path).expect(
            &format!("Failed to read {}", accumulate_shader_path.display()),
        );
        let accumulate_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("accumulate shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(accumulate_shader_source)),
        });
        let accumulate_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("accumulate pipeline"),
                    layout: Some(&accumulate_pipeline_layout),
                    module: &accumulate_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        // Second compute pass: erosion
        // Computes sobel gradients, normalizes them
        // Offsets coordinates by gradient and calculate delta height from that location
        // Calculates velocity and sediment capacity
        // Deposits sediment
        // NOTE: the downhill lookup is bilinear (map_coordinates(order=1) in
        // erosion.py), so the sampler must filter. R32Float/Rg32Float filtering
        // requires the FLOAT32_FILTERABLE feature.
        let erosion_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("sobel pipeline layout"),
                    bind_group_layouts: &[
                        None,
                        Some(&sobel_write_bind_group_layout),
                        Some(&heightmap_bind_group_layout),
                        Some(&water_sediment_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let erosion_shader_path = SHADER_DIRECTORY.join("erosion.wgsl");
        let erosion_shader_source = std::fs::read_to_string(&erosion_shader_path)
            .expect(&format!("Failed to read {}", erosion_shader_path.display()));
        let erosion_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sobel shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(erosion_shader_source)),
        });
        let erosion_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("sobel pipeline"),
                    layout: Some(&erosion_pipeline_layout),
                    module: &erosion_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        // Third shader pass: advection
        let advect_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("advect pipeline layout"),
                    bind_group_layouts: &[
                        None,
                        Some(&sobel_read_bind_group_layout),
                        None,
                        Some(&water_sediment_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let advection_shader_path = SHADER_DIRECTORY.join("advect.wgsl");
        let advect_shader_source = std::fs::read_to_string(&advection_shader_path).expect(
            &format!("Failed to read {}", advection_shader_path.display()),
        );
        let advect_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("advection shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(advect_shader_source)),
        });
        let advect_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("advect pipeline"),
                    layout: Some(&advect_pipeline_layout),
                    module: &advect_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        //         let thermal_pipeline_layout =
        //             gpu.device
        //                 .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //                     label: Some("thermal erosion pipeline"),
        //                     bind_group_layouts: &[None, None, Some(&heightmap_bind_group_layout)],
        //                     immediate_size: 0,
        //                 });
        //
        //         let thermal_erosion_shader_path = SHADER_DIRECTORY.join("thermal.wgsl");
        //         let thermal_erosion_shader_source = std::fs::read_to_string(&thermal_erosion_shader_path)
        //             .expect(&format!(
        //                 "Failed to read {}",
        //                 thermal_erosion_shader_path.display()
        //             ));
        //         let thermal_erosion_shader =
        //             gpu.device
        //                 .create_shader_module(wgpu::ShaderModuleDescriptor {
        //                     label: Some("thermal erosion shader"),
        //                     source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(
        //                         thermal_erosion_shader_source,
        //                     )),
        //                 });
        //
        //         let thermal_pipeline =
        //             gpu.device
        //                 .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //                     label: Some("thermal erosion pipeline"),
        //                     layout: Some(&thermal_pipeline_layout),
        //                     module: &thermal_erosion_shader,
        //                     entry_point: None,
        //                     compilation_options: Default::default(),
        //                     cache: Default::default(),
        //                 });

        let mut rain_texture = create_2d_texture(
            &gpu.device,
            "rain noise",
            texture_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        {
            let noise = create_noise_map(texture_size, 1);
            let layout = get_layout_dimensions(&rain_texture);
            upload_to_texture(&mut rain_texture, &noise, &gpu, layout);
        }

        let gradient_texture = create_2d_texture(
            &gpu.device,
            "sobel gradients",
            texture_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let mut heightmap_textures = [
            create_2d_texture(
                &gpu.device,
                "terrain heightmap",
                texture_size,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
            ),
            create_2d_texture(
                &gpu.device,
                "terrain heightmap",
                texture_size,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
            ),
        ];
        {
            let mut initial_heightmap = generate_chunkable_fbm(
                (config.chunk_size, config.chunk_size),
                (config.fbm_resolution, config.fbm_resolution),
                config.fbm_octaves,
                config.fbm_persistence,
            );
            // erosion.py normalizes the terrain to 0..1 before simulating, and
            // the erosion constants assume that range.
            let min = initial_heightmap
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let max = initial_heightmap
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if max > min {
                for height in &mut initial_heightmap {
                    *height = (*height - min) / (max - min);
                }
            }
            let layout = get_layout_dimensions(&heightmap_textures[0]);
            // The first texture is read from, so heightmap should start in that one
            // Second texture starts blank but the first erosion pass will update it
            upload_to_texture(&mut heightmap_textures[0], &initial_heightmap, &gpu, layout);
            //save initial heightmap
            let heightmap_u8: Vec<u8> = initial_heightmap
                .iter()
                .map(|h| (h * 255.0) as u8)
                .collect();
            image::save_buffer(
                "initial_heightmap.png",
                &heightmap_u8,
                config.chunk_size,
                config.chunk_size,
                image::ExtendedColorType::L8,
            )
            .expect("Failed to save initial heightmap");
        }

        // TODO delete COPY_SRC from both of these
        let mut water_sediment_textures = [
            create_2d_texture(
                &gpu.device,
                "water",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            create_2d_texture(
                &gpu.device,
                "sediment",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
        ];

        {
            // storage textures are not guaranteed to start zeroed, and the first
            // accumulate/advect passes read before writing, so clear them
            let layout = get_layout_dimensions(&water_sediment_textures[0]);
            let zeros = vec![0.0f32; (layout.width * layout.height * 2) as usize];
            for texture in &mut water_sediment_textures {
                upload_to_texture(texture, &zeros, &gpu, get_layout_dimensions(texture));
            }
        }

        let rain_view = rain_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let heightmap_views = [
            heightmap_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            heightmap_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let water_sediment_views = [
            water_sediment_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            water_sediment_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let gradient_view = gradient_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let heightmap_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain heightmap sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let rain_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rain bind group"),
            layout: &rain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&rain_view),
            }],
        });

        let sobel_read_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel bind group"),
            layout: &sobel_read_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gradient_view),
            }],
        });

        let sobel_write_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel bind group"),
            layout: &sobel_write_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gradient_view),
            }],
        });

        let heightmap_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("heightmap bind group"),
                layout: &heightmap_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&heightmap_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&heightmap_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&heightmap_views[1]),
                    },
                ],
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("heightmap bind group"),
                layout: &heightmap_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&heightmap_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&heightmap_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&heightmap_views[0]),
                    },
                ],
            }),
        ];

        let water_sediment_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ping water sediment group"),
                layout: &water_sediment_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&water_sediment_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&water_sediment_views[1]),
                    },
                ],
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ping water sediment group"),
                layout: &water_sediment_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&water_sediment_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&water_sediment_views[0]),
                    },
                ],
            }),
        ];

        Self {
            accumulate_pipeline,
            erosion_pipeline,
            advect_pipeline,
            // thermal_pipeline,
            rain_bind_group,
            sobel_read_bind_group,
            sobel_write_bind_group,
            heightmap_bind_groups,
            water_sediment_bind_groups,
            //for some textures the rust-side handle is never used
            //so prefix with underscore to prevent rustc warning
            _rain_texture: rain_texture,
            _gradient_texture: gradient_texture,
            heightmap_textures,
            water_sediment_textures,
            temp_buffer: None,
            gpu,
            texture_size,
            config,
        }
    }

    fn dispatch_workgroups(&self) -> (u32, u32) {
        (
            self.texture_size.width.div_ceil(WORKGROUP_SIZE),
            self.texture_size.height.div_ceil(WORKGROUP_SIZE),
        )
    }

    // If an encoder is not provided, this will create one
    // This consumes the command encoder for reading data back to the cpu
    pub fn run_chunk_generation(&mut self, encoder: Option<wgpu::CommandEncoder>) -> Vec<f32> {
        let mut encoder =
            encoder.unwrap_or(self.gpu.device.create_command_encoder(&Default::default()));
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let workgroup_dispatch_counts = self.dispatch_workgroups();
        //bind group 0 rain never changes
        pass.set_bind_group(0, Some(&self.rain_bind_group), &[]);
        //bind group 1 sobel gradients is split between read/write but not ping pong
        //So bind it right before the shader runs
        //bind group 2 changes on erosion shader, 3 on advection shader
        //So rebind them when needed
        let mut heightmap_index = 0;
        pass.set_bind_group(2, Some(&self.heightmap_bind_groups[heightmap_index]), &[]);
        let mut water_sediment_index = 0;
        pass.set_bind_group(
            3,
            Some(&self.water_sediment_bind_groups[water_sediment_index]),
            &[],
        );
        for i in 0..self.config.iterations {
            //accumulate flips water/sediment textures
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            water_sediment_index = 1 - water_sediment_index;
            pass.set_bind_group(
                3,
                Some(&self.water_sediment_bind_groups[water_sediment_index]),
                &[],
            );

            //erosion samples from heightmap and then writes, so flip
            //writes to sobels but can be overwritten, so no flip
            pass.set_pipeline(&self.erosion_pipeline);
            pass.set_bind_group(1, Some(&self.sobel_write_bind_group), &[]);
            pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            heightmap_index = 1 - heightmap_index;
            pass.set_bind_group(2, Some(&self.heightmap_bind_groups[heightmap_index]), &[]);
            water_sediment_index = 1 - water_sediment_index;
            pass.set_bind_group(
                3,
                Some(&self.water_sediment_bind_groups[water_sediment_index]),
                &[],
            );

            //advection reads velocity (no flip) and samples and writes to water/sediment (flip)
            pass.set_pipeline(&self.advect_pipeline);
            pass.set_bind_group(1, Some(&self.sobel_read_bind_group), &[]);
            pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            water_sediment_index = 1 - water_sediment_index;
            pass.set_bind_group(
                3,
                Some(&self.water_sediment_bind_groups[water_sediment_index]),
                &[],
            );

            //thermal erosion
            // pass.set_pipeline(&self.thermal_pipeline);
            // pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            // heightmap_index = 1 - heightmap_index;
            // pass.set_bind_group(2, Some(&self.heightmap_bind_groups[heightmap_index]), &[]);

            if i % 16 == 0 || i >= 1000 {
                drop(pass);
                let filename = format!("/tmp/heightmaps/heightmap_{:04}.png", i);
                save_heightmap(
                    &self.heightmap_textures[heightmap_index],
                    &filename,
                    encoder,
                    &self.gpu,
                );
                let filename = format!("/tmp/heightmaps/water_{:04}.png", i);
                save_heightmap(
                    &self.water_sediment_textures[water_sediment_index],
                    &filename,
                    self.gpu.device.create_command_encoder(&Default::default()),
                    &self.gpu,
                );
                encoder = self.gpu.device.create_command_encoder(&Default::default());
                pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, Some(&self.rain_bind_group), &[]);
                pass.set_bind_group(2, Some(&self.heightmap_bind_groups[heightmap_index]), &[]);
                pass.set_bind_group(
                    3,
                    Some(&self.water_sediment_bind_groups[water_sediment_index]),
                    &[],
                );
            }
        }

        //get last written heightmap
        //after the erosion flip, heightmap_index is the texture the last
        //erosion pass wrote to; do not flip again or we read the previous frame
        let layout_dimensions = get_layout_dimensions(&self.heightmap_textures[heightmap_index]);
        //allocate a buffer of the correct size
        if self.temp_buffer.is_none()
            || self.temp_buffer.as_ref().unwrap().size() < layout_dimensions.buffer_size as u64
        {
            self.temp_buffer = Some(self.gpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
                label: None,
                size: layout_dimensions.buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        //finish the compute pass
        drop(pass);
        //allocate a buffer of the correct size
        let mut temp_buffer = self.gpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: None,
            size: layout_dimensions.buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        //copy the texture to a buffer so its readable to cpu
        read_to_buffer(
            &self.heightmap_textures[heightmap_index],
            &mut temp_buffer,
            &layout_dimensions,
            &mut encoder,
        );
        let mut output_data = Vec::with_capacity(
            (layout_dimensions.width * layout_dimensions.height * layout_dimensions.bytes_per_pixel)
                as usize,
        );
        let submission_index = self.gpu.queue.submit(Some(encoder.finish()));
        read_to_cpu(
            &mut temp_buffer,
            &mut output_data,
            &self.gpu.device,
            submission_index,
            &layout_dimensions,
        );

        let output_floats: Vec<f32> = bytemuck::cast_slice(&output_data).into();

        //         let mut img: ImageBuffer<image::Luma<u16>, Vec<u16>> =
        //             ImageBuffer::new(layout_dimensions.width, layout_dimensions.height);
        //
        //         for (x, y, pixel) in img.enumerate_pixels_mut() {
        //             let val_f32 = output_floats[(y * layout_dimensions.width + x) as usize];
        //             let val_u16 = (val_f32.clamp(0.0, 1.0) * 65535.0) as u16;
        //
        //             *pixel = image::Luma([val_u16]);
        //         }
        //
        //         img.save("final_heightmap.png")
        //             .expect("Failed to save heightmap");

        return output_floats;
    }
}
