use std::sync::Mutex;

use glam::{Mat4, Quat, Vec3};
use winit::{event::ElementState, keyboard::KeyCode};

pub enum Event {
    Key((KeyCode, ElementState)),
    MouseMotion((f64, f64)),
    Stop,
}

/// Shared between application and game.
/// bundle these together so there's only one Arc, for better caching.
pub struct SharedInputOutput {
    pub render_state: Mutex<RenderState>,
    pub input_events: Mutex<Vec<Event>>,
}

/// CPU-side render data computed by the update thread each tick.
/// The main thread drains this at the start of each frame.
pub struct RenderState {
    pub camera_state: CameraState,
    pub active_scene: usize,
    pub scenes: Vec<SceneState>,
    pub dirty: bool,
}

pub struct CameraState {
    pub position: Vec3,
    pub orientation: Quat,
    pub dirty: bool,
}

///No scene specification, uses active scene by default
pub struct NewObjectRequest {
    pub material_idx: usize,
    pub mesh_idx: usize,
    pub transform: Mat4,
    pub game_index: usize, //to be returned in the callback
}

pub struct NewObjectEvent {
    pub object_index: usize,
    pub game_index: usize,
}

///Scene index is its index in the array of scenes
pub struct SceneState {
    ///List of new objects to be added on next render() tick
    pub new_object_requests: Vec<NewObjectRequest>,
    ///List of objects that have been added
    pub new_object_events: Vec<NewObjectEvent>,
}

impl RenderState {
    pub fn new() -> Self {
        Self {
            camera_state: CameraState {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                dirty: true,
            },
            scenes: Vec::new(),
            active_scene: 0,
            dirty: false,
        }
    }
}

impl SceneState {
    pub fn new() -> Self {
        Self {
            new_object_requests: Vec::new(),
            new_object_events: Vec::new(),
        }
    }
}
