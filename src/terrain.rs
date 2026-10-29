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
    sobel_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,
    rain_bind_group: wgpu::BindGroup,
    sobel_bind_group: wgpu::BindGroup,
    water_sediment_bind_groups: [wgpu::BindGroup; 2],
    heightmap_bind_groups: [wgpu::BindGroup; 2],
    //ping pong textures that are swapped every iteration
    rain_texture: wgpu::Texture,
    heightmap_textures: [wgpu::Texture; 2],
    water_sediment_textures: [wgpu::Texture; 2],
    gradient_texture: wgpu::Texture,

    heightmap_index: u8,
    water_sediment_index: u8,
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

impl TerrainGenContext {
    /// First render pass does the following:
    /// Computes sobel gradients, normalizes them
    /// Offsets coordinates by gradient and calculate delta height from that location
    /// Calculates velocity and sediment capacity
    /// Deposits sediment
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

        // FIRST SHADER PASS
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

        let accumulate_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("accumulate pipeline layout"),
                    bind_group_layouts: &[
                        Some(&rain_bind_group_layout),
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

        let rain_texture = create_2d_texture(
            &gpu.device,
            "rain texture",
            texture_size,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );

        let water_sediment_textures = [
            create_2d_texture(
                &gpu.device,
                "water",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            create_2d_texture(
                &gpu.device,
                "sediment",
                texture_size,
                wgpu::TextureFormat::Rg32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            ),
        ];

        let water_sediment_views = [
            water_sediment_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            water_sediment_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let rain_view = rain_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // SECOND SHADER PASS
        // NOTE: R32Float/Rg32Float are *unfilterable*, so the sampled textures
        // use `Float { filterable: false }` with a `NonFiltering` sampler.
        // Switching the heightmap to a filterable format (e.g. R16Float) would
        // allow linear filtering in the downhill lookup.
        // TODO: enable float filtering flag if possible
        let sobel_gradient_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sobel bind group layout"),
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
        //ping pong layout for sediment, heightmap
        //this is used in both ping pong render passes
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

        let sobel_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("sobel pipeline layout"),
                    bind_group_layouts: &[
                        Some(&sobel_gradient_bind_group_layout),
                        Some(&heightmap_bind_group_layout),
                        Some(&water_sediment_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let sobel_shader_path = SHADER_DIRECTORY.join("sobel_gradient.wgsl");
        let sobel_shader_source = std::fs::read_to_string(&sobel_shader_path)
            .expect(&format!("Failed to read {}", sobel_shader_path.display()));
        let sobel_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sobel shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(sobel_shader_source)),
        });
        let sobel_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("sobel pipeline"),
                layout: Some(&sobel_pipeline_layout),
                module: &sobel_shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let heightmap_textures = [
            create_2d_texture(
                &gpu.device,
                "terrain heightmap",
                texture_size,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            create_2d_texture(
                &gpu.device,
                "terrain heightmap",
                texture_size,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            ),
        ];
        let gradient_texture = create_2d_texture(
            &gpu.device,
            "sobel gradients",
            texture_size,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

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

        let heightmap_views = [
            heightmap_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            heightmap_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let gradient_view = gradient_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let rain_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rain bind group"),
            layout: &rain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&rain_view),
            }],
        });

        let sobel_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sobel bind group"),
            layout: &sobel_gradient_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gradient_view),
            }],
        });

        // Third shader pass: advection
        let advect_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("advect pipeline layout"),
                    bind_group_layouts: &[
                        Some(&sobel_gradient_bind_group_layout),
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

        let heightmap_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ping bind group"),
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
                label: Some("ping bind group"),
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
            sobel_pipeline,
            accumulate_pipeline,
            rain_bind_group,
            sobel_bind_group,
            water_sediment_bind_groups,
            heightmap_bind_groups,
            rain_texture,
            heightmap_textures,
            water_sediment_textures,
            gradient_texture,
            heightmap_index: 0,
            water_sediment_index: 0,
            texture_size,
        }
    }

    fn dispatch_workgroups(&self) -> (u32, u32) {
        (
            self.texture_size.width.div_ceil(WORKGROUP_SIZE),
            self.texture_size.height.div_ceil(WORKGROUP_SIZE),
        )
    }

    /// Records the first (sobel/erosion) pass into `encoder`.
    /// Upload height/water/sediment data with `queue.write_texture` first.
    pub fn run_first_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sobel first pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sobel_pipeline);
        pass.set_bind_group(0, &self.sobel_bind_group, &[]);
        let (dx, dy) = self.dispatch_workgroups();
        pass.dispatch_workgroups(dx, dy, 1);
    }
}
