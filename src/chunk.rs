use std::hash::{Hash, Hasher};

use glam::Vec3;
use honeycomb_3d::vertex::Vertex;

//size of each "block" or heightmap pixel
pub const MAP_SCALE: f32 = 0.25;
pub const WORLD_HEIGHT: f32 = 32.0;

pub struct Chunk {
    pub x: i32,
    pub y: i32,

    heightmap: Vec<f32>,
}

impl PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl Eq for Chunk {}
impl Hash for Chunk {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ux = self.x as u32 as u64;
        let uz = self.y as u32 as u64;
        let mut key = (ux << 32) | uz;

        key ^= key >> 33;
        key = key.wrapping_mul(0xff51afd7ed558ccd);
        key ^= key >> 33;
        key = key.wrapping_mul(0xc4ceb9fe1a85ec53);
        key ^= key >> 33;

        state.write_u64(key);
    }
}

const NUM_HILLSHADE_INTERVALS: usize = 4;
//search this to find index to lookup table
const HILLSHADING_THRESHOLDS: [f32; NUM_HILLSHADE_INTERVALS + 1] = [0.0, 0.25, 0.5, 0.8, 1.0];
//Lookup table to mapping heightmap values to terrain colors
const HILLSHADING_COLORS: [[f32; 3]; NUM_HILLSHADE_INTERVALS + 1] = [
    [0.15, 0.3, 0.15],
    [0.3, 0.45, 0.3],
    [0.5, 0.5, 0.35],
    [0.4, 0.36, 0.33],
    [1.0, 1.0, 1.0],
];

fn map_to_color(height: f32) -> [f32; 3] {
    if height <= HILLSHADING_THRESHOLDS[0] {
        return HILLSHADING_COLORS[0];
    }
    let last_idx = NUM_HILLSHADE_INTERVALS;
    if height >= HILLSHADING_THRESHOLDS[last_idx] {
        return HILLSHADING_COLORS[last_idx];
    }

    let result = HILLSHADING_THRESHOLDS.binary_search_by(|f| f.total_cmp(&height));
    let index = match result {
        Ok(i) => i,
        Err(i) => i - 1,
    };

    let first = HILLSHADING_COLORS[index];
    let second = HILLSHADING_COLORS[index + 1];

    let lerp_value = (height - HILLSHADING_THRESHOLDS[index])
        / (HILLSHADING_THRESHOLDS[index + 1] - HILLSHADING_THRESHOLDS[index]);

    let color = [
        second[0] * lerp_value + first[0] * (1.0 - lerp_value),
        second[1] * lerp_value + first[1] * (1.0 - lerp_value),
        second[2] * lerp_value + first[2] * (1.0 - lerp_value),
    ];

    color
}

impl Chunk {
    pub fn new(heightmap: Vec<f32>, x: i32, y: i32) -> Self {
        Self { x, y, heightmap }
    }
    //returns (vertices, indices, normals)
    pub fn to_3d_geometry(&self, chunk_size: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut geometry = Vec::<Vertex>::with_capacity((chunk_size * chunk_size) as usize);
        let mut indices =
            Vec::<u32>::with_capacity(((chunk_size - 1) * (chunk_size - 1) * 6) as usize);

        //build geometry
        //this is confusing! x and y are texture indices
        //but the camera projection uses x and z as horizontal axes
        let chk_sz_i32: i32 = chunk_size as i32;
        for y in 0..chk_sz_i32 as i32 {
            for x in 0..chk_sz_i32 as i32 {
                geometry.push(Vertex {
                    position: [
                        ((x - chk_sz_i32 / 2) as f32) * MAP_SCALE,
                        self.heightmap[(y * chk_sz_i32 + x) as usize] * WORLD_HEIGHT,
                        ((y - chk_sz_i32 / 2) as f32) * MAP_SCALE,
                    ],
                    color: map_to_color(self.heightmap[(y * chk_sz_i32 + x) as usize]),
                    normal: [0.0, 0.0, 0.0], // set these later
                });
            }
        }
        //build triangles
        for y in 0..chunk_size - 1 {
            for x in 0..chunk_size - 1 {
                let topleft = y * chunk_size + x;
                let topright = y * chunk_size + x + 1;
                let bottomleft = (y + 1) * chunk_size + x;
                let bottomright = (y + 1) * chunk_size + x + 1;
                indices.extend([
                    topleft,
                    bottomleft,
                    topright,
                    bottomright,
                    topright,
                    bottomleft,
                ]);
                let topleft_v = Vec3::from_array(geometry[topleft as usize].position);
                let topright_v = Vec3::from_array(geometry[topright as usize].position);
                let bottomleft_v = Vec3::from_array(geometry[bottomleft as usize].position);
                let bottomright_v = Vec3::from_array(geometry[bottomright as usize].position);

                let normal = (bottomleft_v - topleft_v).cross(topright_v - topleft_v);
                for i in 0..3 {
                    geometry[topleft as usize].normal[i] += normal[i];
                    geometry[bottomleft as usize].normal[i] += normal[i];
                    geometry[topright as usize].normal[i] += normal[i];
                }
                let normal = (topright_v - bottomright_v).cross(bottomleft_v - bottomright_v);
                for i in 0..3 {
                    geometry[bottomright as usize].normal[i] += normal[i];
                    geometry[topright as usize].normal[i] += normal[i];
                    geometry[bottomleft as usize].normal[i] += normal[i];
                }
            }
        }

        for vertex in &mut geometry {
            let magnitude = Vec3::from_array(vertex.normal).length();
            vertex.normal[0] /= magnitude;
            vertex.normal[1] /= magnitude;
            vertex.normal[2] /= magnitude;
        }

        // print as output for a .obj file
        //for vertex in &geometry {
        //    println!(
        //        "v {} {} {}",
        //        vertex.position[0], vertex.position[1], vertex.position[2]
        //    );
        //}
        //
        //let mut i = 0;
        //while i < indices.len() {
        //    println!(
        //        "f {} {} {}",
        //        indices[i] + 1,
        //        indices[i + 1] + 1,
        //        indices[i + 2] + 1
        //    );
        //    i += 3;
        //}

        return (geometry, indices);
    }
}
