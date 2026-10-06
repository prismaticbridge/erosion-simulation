use std::sync::Mutex;

use glam::{Mat4, Vec3};
use winit::{event::ElementState, keyboard::KeyCode};

pub enum Event {
    Key((KeyCode, ElementState)),
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
    pub player_state: PlayerState,
    pub active_scene: usize,
    pub scenes: Vec<SceneState>,
    pub dirty: bool,
}

pub struct PlayerState {
    pub player_position: Vec3,
    // renderer handles for the player object — constant after init
    pub player_material: usize,
    pub player_mesh: usize,
    pub player_object: usize,
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
            player_state: PlayerState {
                player_position: Vec3::ZERO,
                player_material: 0,
                player_mesh: 0,
                player_object: 0,
                dirty: false,
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
