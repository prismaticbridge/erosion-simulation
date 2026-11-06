use std::{sync::Arc, time::Instant};

use honeycomb_3d::Renderer;
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop};

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
            let start = Instant::now();
            let result = pollster::block_on(Renderer::new(event_loop, &REQUESTED_FEATURES));
            let end = Instant::now();
            println!("delta time for new: {} ms", (end - start).as_millis());
            let renderer = match result {
                Ok(renderer) => renderer,
                Err(features) => pollster::block_on(Renderer::new(event_loop, &features)).expect(
                    "Failed to create renderer with features that are supposed to be supported",
                ),
            };

            let gpu = renderer.get_gpu_context();
            println!("enabled features: {}", gpu.device.features());
            let mut terrain_gen = TerrainGenContext::new(Arc::clone(&gpu));
            let command_encoder = gpu.device.create_command_encoder(&Default::default());

            let start = Instant::now();
            terrain_gen.run_chunk_generation(command_encoder);
            let end = Instant::now();
            println!("delta time for chunks: {} ms", (end - start).as_millis());

            // let start = Instant::now();
            // terrain_gen.save_everything();
            // let end = Instant::now();
            // println!(
            //     "delta time for read to cpu: {} ms",
            //     (end - start).as_millis()
            // );

            self.state = Some(State { renderer })
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        event_loop.exit();
    }
}
