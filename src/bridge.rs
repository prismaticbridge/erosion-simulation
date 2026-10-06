use std::{
    sync::{Arc, Mutex},
    thread::{JoinHandle, sleep},
    time::{Duration, Instant},
};

use glam::Vec2;
use honeycomb_3d::Renderer;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::Window,
};

use crate::{
    application::Game,
    bridge_data::{Event, NewObjectEvent, RenderState, SharedInputOutput},
};

struct State {
    update_thread_join_handle: Option<JoinHandle<Game>>,
    shared_io: Arc<SharedInputOutput>,
    window: Arc<Window>,
    renderer: Renderer,
}

pub static CAMERA_DIMENSIONS: Mutex<Vec2> = Mutex::new(Vec2 { x: 1.0, y: 1.0 });
pub const UPDATE_HZ: u32 = 60;
pub const UPDATE_LOOP_DELAY: Duration = Duration::from_nanos(1_000_000_000 / UPDATE_HZ as u64);
pub const ANIMATION_TIME: u64 = UPDATE_LOOP_DELAY.as_micros() as u64;
pub static SCREEN_WIDTH: Mutex<u32> = Mutex::new(0);
pub static SCREEN_HEIGHT: Mutex<u32> = Mutex::new(0);

// Not great for state to have functions but this was the most convenient place
impl State {
    pub fn update_window_size_camera_transform(&mut self, width: u32, height: u32) {
        if width < height {
            let scale = width as f32 / height as f32;
            if let Ok(mut dims) = CAMERA_DIMENSIONS.lock() {
                dims.x = 1.0;
                dims.y = 1.0 / scale;
            }
        } else {
            let scale = height as f32 / width as f32;
            if let Ok(mut dims) = CAMERA_DIMENSIONS.lock() {
                dims.x = 1.0 / scale;
                dims.y = 1.0;
            }
        }
    }

    pub fn update_gpu_data(&mut self) {
        let Ok(mut render_state) = self.shared_io.render_state.lock() else {
            return;
        };
        //to avoid borrow checker errors, copy out now
        let active_scene = render_state.active_scene;
        if render_state.dirty {
            self.renderer.set_active_scene(active_scene);
            render_state.dirty = false;
        }
        if render_state.player_state.dirty {
            self.renderer.set_interpolated_target_position(
                render_state.player_state.player_material,
                render_state.player_state.player_mesh,
                render_state.player_state.player_object,
                render_state.player_state.player_position,
                ANIMATION_TIME,
            );
            render_state.player_state.dirty = false;
        }

        //drain events first to avoid holding a borrow
        let requests: Vec<_> = render_state.scenes[active_scene]
            .new_object_requests
            .drain(..)
            .collect();
        for request in requests {
            let object_index = self.renderer.add_object_instance(
                active_scene,
                request.material_idx,
                request.mesh_idx,
                request.transform,
            );
            render_state.scenes[active_scene]
                .new_object_events
                .push(NewObjectEvent {
                    object_index,
                    game_index: request.game_index,
                });
        }
    }
}

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
            let mut renderer = pollster::block_on(Renderer::new(event_loop));

            let window = Arc::clone(&renderer.window);

            let shared_io = Arc::new(SharedInputOutput {
                render_state: std::sync::Mutex::new(RenderState::new()),
                input_events: std::sync::Mutex::new(Vec::new()),
            });
            let mut game = Game::new(&shared_io, &mut renderer);

            let mut next_update_time = Instant::now();
            let join_handle = std::thread::spawn(move || -> Game {
                // let mut last_update = Instant::now();
                // let mut num_updates: f64 = 0.0;
                loop {
                    let now = Instant::now();
                    // num_updates += 1.0;
                    // num_updates /= (now - last_update).as_micros() as f64 / 1000000.0 + 1.0;
                    // last_update = now;
                    // println!("{}", num_updates);
                    if now < next_update_time {
                        // println!(
                        //     "sleeping {} before next update",
                        //     (next_update_time - now).as_millis()
                        // );
                        sleep(next_update_time - now);
                        next_update_time = Instant::now() + UPDATE_LOOP_DELAY; // new now, since we slept in-between
                    } else {
                        // println!("triggering next update immediately");
                        next_update_time = now + UPDATE_LOOP_DELAY;
                    }

                    let should_stop = game.update();
                    if should_stop {
                        break game;
                    }
                }
            });

            self.state = Some(State {
                update_thread_join_handle: Some(join_handle),
                shared_io,
                window,
                renderer,
            })
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                if let Ok(mut events) = state.shared_io.input_events.lock() {
                    events.push(Event::Stop);
                    drop(events);
                    if let Some(join_handle) = state.update_thread_join_handle.take() {
                        let _ = join_handle.join().inspect_err(|e| {
                            println!(
                                "Error joining update thread which should have stopped: {:?}",
                                e
                            )
                        });
                    }
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Ok(mut width) = SCREEN_WIDTH.lock() {
                    *width = size.width
                }
                if let Ok(mut height) = SCREEN_HEIGHT.lock() {
                    *height = size.height
                }
                state.renderer.resize(size.width, size.height);
                state.update_window_size_camera_transform(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                // Drain the latest game state from the update thread.
                state.renderer.update_frame_timestamp(); //new targets should use the same timestamp as this frame
                state.update_gpu_data();
                state.renderer.update_interpolations();
                match state.renderer.render() {
                    Ok(_) => {}
                    Err(honeycomb_3d::utils::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.renderer.resize(size.width, size.height);
                    }
                    Err(_) => eprintln!("Surface error"),
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if let Ok(mut events) = state.shared_io.input_events.lock() {
                    events.push(Event::Key((key_code, key_state)));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
            let now = Instant::now();
            self.in_last_sec /= (now - self.last_time).as_micros() as f64 / 1000000.0 + 1.0;
            self.in_last_sec += 1.0;
            // println!("{}", self.in_last_sec);
            self.last_time = now;
        }
    }
}
