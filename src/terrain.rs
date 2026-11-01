use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use bytemuck::NoUninit;
use honeycomb_3d::GpuContext;
use image::{ImageBuffer, Rgb};
use noise::{NoiseFn, Perlin};
use wgpu::{ShaderModuleDescriptor, SubmissionIndex};

pub const CHUNK_SIZE: u32 = 512; // texture width/height in texels for one terrain chunk
pub const PAD_SIZE: u32 = 128;
const WORKGROUP_SIZE: u32 = 16;
const TERRAIN_GEN_ITERATIONS: u32 = 512;

// Fractional brownian motion parameters for initial heightmap generation
const FBM_RESOLUTION: u32 = 8;
const FBM_OCTAVES: u32 = 6;
const FBM_PERSISTENCE: f64 = 0.5;

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
    accumulate_pipeline: wgpu::ComputePipeline,
    erosion_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,
    rain_bind_group: wgpu::BindGroup,
    sobel_read_bind_group: wgpu::BindGroup,
    sobel_write_bind_group: wgpu::BindGroup,
    heightmap_bind_groups: [wgpu::BindGroup; 2],
    water_sediment_bind_groups: [wgpu::BindGroup; 2],
    //ping pong textures that are swapped every iteration
    //these aren't needed, but if they get dropped wgpu frees them and then the shaders all crash
    rain_texture: wgpu::Texture,
    gradient_texture: wgpu::Texture,
    heightmap_textures: [wgpu::Texture; 2],
    water_sediment_textures: [wgpu::Texture; 2],

    temp_buffer: Option<wgpu::Buffer>,
    gpu: Arc<GpuContext>,

    // water_sediment_index: u8,
    texture_size: wgpu::Extent3d,
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

struct LayoutDimensions {
    bytes_per_pixel: u32,
    padded_bytes_per_row: u32,
    buffer_size: u32,
    width: u32,
    height: u32,
}

fn get_layout_dimensions(texture: &wgpu::Texture) -> LayoutDimensions {
    let bytes_per_pixel = texture
        .format()
        .target_pixel_byte_cost()
        .expect("Cannot get pixels per byte for a texture");
    let unpadded_bytes_per_row = texture.width() * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let buffer_size = texture.height() * padded_bytes_per_row;

    LayoutDimensions {
        bytes_per_pixel,
        padded_bytes_per_row,
        buffer_size,
        width: texture.width(),
        height: texture.height(),
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

pub fn save_texture_to_file(
    texture: &wgpu::Texture,
    file: &str,
    mut encoder: wgpu::CommandEncoder,
    gpu: &GpuContext,
    channels: u32,
) {
    let output_buffer_dimensions = get_layout_dimensions(&texture);
    //allocate a buffer of the correct size
    let mut temp_buffer = gpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
        label: None,
        size: output_buffer_dimensions.buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    //finish the compute pass
    read_to_buffer(
        &texture,
        &mut temp_buffer,
        &output_buffer_dimensions,
        &mut encoder,
    );
    let mut output_data = Vec::with_capacity(
        (output_buffer_dimensions.width
            * output_buffer_dimensions.height
            * output_buffer_dimensions.bytes_per_pixel) as usize,
    );
    let submission_index = gpu.queue.submit(Some(encoder.finish()));
    read_to_cpu(
        &mut temp_buffer,
        &mut output_data,
        &gpu.device,
        submission_index,
        &output_buffer_dimensions,
    );

    let output_format = match channels {
        1 => image::ExtendedColorType::L8,
        _ => image::ExtendedColorType::Rgb8,
    };

    let output_floats: &[f32] = bytemuck::cast_slice(&output_data);

    let mut img = ImageBuffer::new(
        output_buffer_dimensions.width,
        output_buffer_dimensions.height,
    );
    match channels {
        1 => {
            for (i, pixel) in img.pixels_mut().enumerate() {
                let r = (output_floats[i] * 255.0).clamp(0.0, 255.0) as u8;
                *pixel = Rgb([r, 0, 0]); // G and B padded with 0
            }
        }
        2 => {
            for (i, pixel) in img.pixels_mut().enumerate() {
                let r = (output_floats[i * 2] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (output_floats[i * 2 + 1] * 255.0).clamp(0.0, 255.0) as u8;
                *pixel = Rgb([r, g, 0]); // B padded with 0
            }
        }
        _ => {
            for (i, pixel) in img.pixels_mut().enumerate() {
                let r = (output_floats[i * 3] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (output_floats[i * 3 + 1] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (output_floats[i * 3 + 2] * 255.0).clamp(0.0, 255.0) as u8;
                *pixel = Rgb([r, g, b]);
            }
        }
    }

    img.save(file);
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
            bytes_per_row: Some(layout.width * size_of::<T>() as u32),
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
    /// First render pass does the following:
    ///
    /// `new` allocates every GPU object the pass needs: the 5 textures, the
    /// sampler, the bind group, and the compute pipeline. Textures are
    /// zero-initialized by wgpu; the caller uploads real heights with
    /// `queue.write_texture(&heightmap_texture, ...)` before dispatching.
    /// See [`Self::run_first_pass`].
    pub fn new(gpu: Arc<GpuContext>) -> Self {
        let texture_size = wgpu::Extent3d {
            width: CHUNK_SIZE,
            height: CHUNK_SIZE,
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
        // NOTE: R32Float/Rg32Float are *unfilterable*, so the sampled textures
        // use `Float { filterable: false }` with a `NonFiltering` sampler.
        // Switching the heightmap to a filterable format (e.g. R16Float) would
        // allow linear filtering in the downhill lookup.
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

        let mut rain_texture = create_2d_texture(
            &gpu.device,
            "rain noise",
            texture_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC, // TODO: remove this
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
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC, // TODO remove COPY_SRC
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
            let initial_heightmap = generate_chunkable_fbm(
                (CHUNK_SIZE, CHUNK_SIZE),
                (FBM_RESOLUTION, FBM_RESOLUTION),
                FBM_OCTAVES,
                FBM_PERSISTENCE,
            );
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
                CHUNK_SIZE,
                CHUNK_SIZE,
                image::ExtendedColorType::L8,
            )
            .expect("Failed to save initial heightmap");
        }

        let water_sediment_textures = [
            create_2d_texture(
                &gpu.device,
                "water",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
            ),
            create_2d_texture(
                &gpu.device,
                "sediment",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::STORAGE_BINDING,
            ),
        ];

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
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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
            rain_bind_group,
            sobel_read_bind_group,
            sobel_write_bind_group,
            heightmap_bind_groups,
            water_sediment_bind_groups,
            rain_texture,
            gradient_texture,
            heightmap_textures,
            water_sediment_textures,
            temp_buffer: None,
            gpu,
            texture_size,
        }
    }

    fn dispatch_workgroups(&self) -> (u32, u32) {
        (
            self.texture_size.width.div_ceil(WORKGROUP_SIZE),
            self.texture_size.height.div_ceil(WORKGROUP_SIZE),
        )
    }

    pub fn save_everything(&self) {
        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(&self.rain_texture, "rain.png", encoder, &self.gpu, 1);

        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(&self.gradient_texture, "rain.png", encoder, &self.gpu, 2);

        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(
            &self.heightmap_textures[0],
            "heightmap_0.png",
            encoder,
            &self.gpu,
            1,
        );

        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(
            &self.heightmap_textures[1],
            "heightmap_1.png",
            encoder,
            &self.gpu,
            1,
        );

        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(
            &self.water_sediment_textures[0],
            "water_0.png",
            encoder,
            &self.gpu,
            1,
        );
        let encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        save_texture_to_file(
            &self.water_sediment_textures[1],
            "water_1.png",
            encoder,
            &self.gpu,
            1,
        );
    }

    pub fn run_chunk_generation(&mut self, mut encoder: wgpu::CommandEncoder) -> Vec<f32> {
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
        for _i in 0..TERRAIN_GEN_ITERATIONS {
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
        }

        //get last written heightmap
        //heightmap index corresponds to the read handle, so flip to get write handle
        heightmap_index = 1 - heightmap_index;
        let output_buffer_dimensions =
            get_layout_dimensions(&self.heightmap_textures[heightmap_index]);
        //allocate a buffer of the correct size
        if self.temp_buffer.is_none()
            || self.temp_buffer.as_ref().unwrap().size()
                < output_buffer_dimensions.buffer_size as u64
        {
            self.temp_buffer = Some(self.gpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
                label: None,
                size: output_buffer_dimensions.buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        //finish the compute pass
        drop(pass);
        read_to_buffer(
            &self.heightmap_textures[heightmap_index],
            self.temp_buffer.as_mut().unwrap(),
            &output_buffer_dimensions,
            &mut encoder,
        );
        let mut output_data = Vec::with_capacity(
            (output_buffer_dimensions.width
                * output_buffer_dimensions.height
                * output_buffer_dimensions.bytes_per_pixel) as usize,
        );
        let submission_index = self.gpu.queue.submit(Some(encoder.finish()));
        read_to_cpu(
            self.temp_buffer.as_mut().unwrap(),
            &mut output_data,
            &self.gpu.device,
            submission_index,
            &output_buffer_dimensions,
        );

        let float_slice: &[f32] = bytemuck::cast_slice(&output_data);
        return float_slice.to_vec();
    }
}
