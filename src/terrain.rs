use honeycomb_3d::vertex::Vertex;

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
