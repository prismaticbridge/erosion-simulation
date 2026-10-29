@group(1) @binding(0) var velocity: texture_storage_2d<rg32float, read>;
@group(3) @binding(0) var water_sediment: texture_storage_2d<rg32float, read>;
@group(3) @binding(1) var write_water_sediment: texture_storage_2d<rg32float, write>;

// var<workgroup> shared_tile: array<array<vec2f, 18>, 18>;
// TODO: workgroup cache optimization

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // let dims = textureDimensions(water_sediment);
    let coords = vec2i(id.xy);

    var accumulated = vec2f(0.0, 0.0);
    for (var x: i32 = -1; x <= 1; x = x + 1) {
        for (var y: i32 = -1; y <= 1; y = y + 1) {
            let load_offset = vec2i(x, y);
            let sample_coords = load_offset + coords;
            let velocity = textureLoad(velocity, sample_coords).xy;
            let offset = vec2f(load_offset);
            let target = offset + velocity;
            let overlap = (1 - abs(target.x)) * (1 - abs(target.y));

            let sample = textureLoad(water_sediment, sample_coords).xy;
            accumulated += sample * overlap;
        }
    }

    textureStore(write_water_sediment, coords, vec4f(accumulated, 0.0, 1.0));
}
