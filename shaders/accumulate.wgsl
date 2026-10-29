// handles evaporation and rain
@group(0) @binding(0) var rain: texture_storage_2d<r32float, read>;
@group(3) @binding(0) var water_sediment: texture_2d<f32>;
@group(3) @binding(1) var write_water_sediment: texture_storage_2d<rg32float, write>;

// TODO: this is basically nothing, tune it
const EVAPORATION_RATE = 1.0 - 0.00004;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // let dims = textureDimensions(water_sediment);
    let coords = vec2i(id.xy);
    var water_sediment_sample = textureLoad(water_sediment, coords, 0);
    let rain_amt = textureLoad(rain, coords).r;

    water_sediment_sample.r = (water_sediment_sample.r + rain_amt) * EVAPORATION_RATE;
    textureStore(write_water_sediment, coords, water_sediment_sample);
}
