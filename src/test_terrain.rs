use std::{
    sync::{Arc, Mutex, OnceLock},
    thread::{JoinHandle, sleep},
    time::{Duration, Instant},
};

use glam::Vec2;
use honeycomb_3d::Renderer;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::Window,
};

use crate::terrain::TerrainGenContext;

struct State {
    renderer: Renderer,
}
pub const REQUESTED_FEATURES: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE;

// Not great for state to have functions but this was the most convenient place
impl State {}

pub struct App {
    state: Option<State>,
    last_time: Instant,
    in_last_sec: f64,
}

impl App {
    pub fn new() -> Self {
        let app = Self {
            state: None,
            last_time: Instant::now(),
            in_last_sec: 0.0,
        };

        return app;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let result = pollster::block_on(Renderer::new(event_loop, &REQUESTED_FEATURES));
            let mut renderer = match result {
                Ok(renderer) => renderer,
                Err(features) => pollster::block_on(Renderer::new(event_loop, &features)).expect(
                    "Failed to create renderer with features that are supposed to be supported",
                ),
            };

            let gpu = renderer.get_gpu_context();
            let terrain_gen = TerrainGenContext::new(&gpu);
            let mut command_encoder =
                gpu.device
                    .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                        label: Some("compute pass"),
                    });
            terrain_gen.run_chunk_generation(&mut command_encoder);

            self.state = Some(State { renderer })
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
    }
}
