use std::sync::Arc;

use glam::{Quat, Vec3};
use honeycomb_3d::GpuContext;
use winit::{event::ElementState, keyboard::KeyCode};

use crate::{
    bridge::UPDATE_HZ,
    bridge_data::{Event, SceneState, SharedInputOutput},
    chunk::{Chunk, WORLD_HEIGHT},
    terraingen::TerrainGenContext,
};

//this is an upper bound
const NUM_KEYCODES: usize = 256;
const MOUSE_MOVE_FACTOR: f64 = 0.001;
const MOUSE_MOVE_THRESHOLD: f64 = 1e-8;

const MOVEMENT_SPEED: f32 = 10.0;
const MOVEMENT_PER_TICK: f32 = MOVEMENT_SPEED / UPDATE_HZ as f32;

pub struct Game {
    shared_io: Arc<SharedInputOutput>,
    keyboard_input: [bool; NUM_KEYCODES],
    camera_position: Vec3,
    camera_orientation: Quat,
    terrain_generator: TerrainGenContext,
    gpu_context: Arc<GpuContext>,
    gpu_features: wgpu::Features,

    should_stop: bool,
}
impl Game {
    pub fn new(shared_io: &Arc<SharedInputOutput>, renderer: &mut honeycomb_3d::Renderer) -> Self {
        let main_scene = renderer.create_scene();
        // renderer.add_static_object(main_scene, &CUBE_VERTICES, &CUBE_INDICES);
        // if let Ok(mut rs) = shared_io.render_state.lock() {
        //     rs.scenes.push(SceneState::new());
        // }
        let gpu_context = renderer.get_gpu_context();
        let mut terrain_generator = TerrainGenContext::new(Arc::clone(&gpu_context));
        let chunk_heightmap = terrain_generator.run_chunk_generation(None);
        let terrain_chunk = Chunk::new(chunk_heightmap, 0, 0);
        let (vertices, indices) = terrain_chunk.to_3d_geometry();
        renderer.add_static_object(main_scene, &vertices, &indices);
        if let Ok(mut rs) = shared_io.render_state.lock() {
            rs.scenes.push(SceneState::new());
        }
        let gpu_features = renderer.enabled_features;
        Self {
            shared_io: Arc::clone(shared_io),
            keyboard_input: [false; NUM_KEYCODES],
            camera_position: Vec3 {
                x: 0.0,
                y: WORLD_HEIGHT,
                z: -2.0,
            },
            camera_orientation: Quat::IDENTITY,
            terrain_generator,
            gpu_context,
            gpu_features,
            should_stop: false,
        }
    }

    fn update_camera_pose(&mut self, x_offset: f64, y_offset: f64) {
        let mouse_move_squared = x_offset * x_offset + y_offset * y_offset;
        if mouse_move_squared < MOUSE_MOVE_THRESHOLD {
            return;
        }
        let mouse_move_magnitude = mouse_move_squared.sqrt();
        let local_axis = Vec3 {
            x: (-y_offset / mouse_move_magnitude) as f32, // y offset is inverted
            y: (-x_offset / mouse_move_magnitude) as f32,
            z: 0.0,
        };
        let transformation_quat = Quat::from_axis_angle(
            local_axis,
            (mouse_move_magnitude * MOUSE_MOVE_FACTOR) as f32,
        );
        self.camera_orientation = self.camera_orientation * transformation_quat;
    }

    fn update_events(&mut self) {
        let events = match self.shared_io.input_events.lock() {
            Ok(mut events) => std::mem::take(&mut *events),
            Err(e) => {
                eprintln!("failed to aquire lock on events: {e}");
                return;
            }
        };

        for event in events {
            match event {
                Event::Stop => self.should_stop = true,
                Event::MouseMotion((x, y)) => {
                    self.update_camera_pose(x, y);
                }
                Event::Key((keycode, state)) => {
                    self.keyboard_input[keycode as usize] = state == ElementState::Pressed;
                }
            }
        }
    }

    fn update_input(&mut self) {
        const FORWARD_LOCAL: Vec3 = Vec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        };
        if self.keyboard_input[KeyCode::KeyW as usize] {
            let forward_global = self.camera_orientation * FORWARD_LOCAL;
            self.camera_position += forward_global * MOVEMENT_PER_TICK;
        }
        if self.keyboard_input[KeyCode::KeyS as usize] {
            let forward_global = self.camera_orientation * FORWARD_LOCAL;
            self.camera_position -= forward_global * MOVEMENT_PER_TICK;
        }
    }

    pub fn update(&mut self) -> bool {
        self.update_events();
        self.update_input();
        if let Ok(mut rs) = self.shared_io.render_state.lock() {
            rs.camera_state.position = self.camera_position;
            rs.camera_state.orientation = self.camera_orientation;
            rs.camera_state.dirty = true;
        }
        self.should_stop
    }
}
