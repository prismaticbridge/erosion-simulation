use std::{fmt::format, path::PathBuf, sync::LazyLock};

use honeycomb_3d::{GpuContext, vertex::Vertex};
use wgpu::{BindGroupLayoutEntry, ShaderModuleDescriptor};

#[rustfmt::skip]
pub const CUBE_VERTICES: [Vertex; 8] = [
    Vertex {
        position: [-0.5, -0.5, -0.5],
        color: [0.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [0.0, 1.0, 1.0],
    }
];

#[rustfmt::skip]
pub const CUBE_INDICES: [u16; 36] = [
    0, 1, 2,
    0, 2, 3,
    1, 5, 6,
    1, 6, 2,
    3, 2, 6,
    3, 6, 7,
    3, 7, 4,
    3, 4, 0,
    0, 4, 5,
    0, 5, 1,
    4, 6, 5,
    4, 7, 6
];

const CHUNK_SIZE: u32 = 512; // texture width/height in texels for one terrain chunk
const PAD_SIZE: u32 = 128;
const WORKGROUP_SIZE: u32 = 16;
const TERRAIN_GEN_ITERATIONS: u32 = 512;

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

    // heightmap_index: u8,
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

/// Generates a random 0..=1 noise field, box-blurs it by `blur_radius` texels
/// (wrapping at the edges so the result stays tileable) and uploads it to a new
/// `R32Float` texture, ready to be used as the rain source.
pub fn create_noise_texture(
    gpu: &GpuContext,
    size: wgpu::Extent3d,
    blur_radius: u32,
) -> wgpu::Texture {
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

    let texture = create_2d_texture(
        &gpu.device,
        "rain noise",
        size,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
    );

    let bytes: Vec<u8> = blurred.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * size_of::<f32>() as u32),
            rows_per_image: Some(size.height),
        },
        size,
    );

    texture
}

impl TerrainGenContext {
    /// First render pass does the following:
    ///
    /// `new` allocates every GPU object the pass needs: the 5 textures, the
    /// sampler, the bind group, and the compute pipeline. Textures are
    /// zero-initialized by wgpu; the caller uploads real heights with
    /// `queue.write_texture(&heightmap_texture, ...)` before dispatching.
    /// See [`Self::run_first_pass`].
    pub fn new(gpu: &GpuContext) -> Self {
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

        let rain_texture = create_noise_texture(gpu, texture_size, 1);

        let gradient_texture = create_2d_texture(
            &gpu.device,
            "sobel gradients",
            texture_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let heightmap_textures = [
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
            // heightmap_index: 0,
            // water_sediment_index: 0,
            texture_size,
        }
    }

    fn dispatch_workgroups(&self) -> (u32, u32) {
        (
            self.texture_size.width.div_ceil(WORKGROUP_SIZE),
            self.texture_size.height.div_ceil(WORKGROUP_SIZE),
        )
    }

    pub fn run_chunk_generation(&self, encoder: &mut wgpu::CommandEncoder) {
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
        for i in 0..TERRAIN_GEN_ITERATIONS {
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
            pass.set_bind_group(1, Some(&self.sobel_read_bind_group), &[]);
            pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            heightmap_index = 1 - heightmap_index;
            pass.set_bind_group(2, Some(&self.heightmap_bind_groups[heightmap_index]), &[]);

            //advection reads velocity (no flip) and samples and writes to water/sediment (flip)
            pass.set_pipeline(&self.advect_pipeline);
            pass.set_bind_group(1, Some(&self.sobel_write_bind_group), &[]);
            pass.dispatch_workgroups(workgroup_dispatch_counts.0, workgroup_dispatch_counts.1, 1);
            water_sediment_index = 1 - water_sediment_index;
            pass.set_bind_group(
                3,
                Some(&self.water_sediment_bind_groups[water_sediment_index]),
                &[],
            );
        }
    }
}
