use std::sync::Arc;

use glam::{Quat, Vec3};

use crate::bridge_data::SharedInputOutput;

//this is an upper bound
const NUM_KEYCODES: usize = 256;
pub struct Game {
    shared_io: Arc<SharedInputOutput>,
    keyboard_input: [bool; NUM_KEYCODES],
    camera_position: Vec3,
    camera_orientation: Quat,
}
impl Game {
    pub fn new(shared_io: &Arc<SharedInputOutput>, renderer: &mut honeycomb_3d::Renderer) -> Self {
        Self {
            shared_io: Arc::clone(shared_io),
            keyboard_input: [false; NUM_KEYCODES],
            camera_position: Vec3::ZERO,
            camera_orientation: Quat::IDENTITY,
        }
    }
    pub fn update(&mut self) -> bool {
        false
    }
}
