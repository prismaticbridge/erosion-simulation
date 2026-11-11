//sobel is calculated once and stored, so no pingpong
//this also represents the velocity
@group(1) @binding(0) var sobel_gradients: texture_storage_2d<rg32float, write>; // normalized
//sample and write height in this shader, so need pingpong
@group(2) @binding(0) var heightmap: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;
@group(2) @binding(2) var write_heightmap: texture_storage_2d<r32float, write>;
//water and sediment get flipped in a different shader, so separate group
//In this shader they only get read from
@group(3) @binding(0) var water_sediment: texture_2d<f32>;
@group(3) @binding(1) var write_water_sediment: texture_storage_2d<rg32float, write>;
// @group(4) @binding(0) var<uniform> time_seed: vec2;

const MIN_HEIGHT_DELTA: f32 = 0.06;
const SEDIMENT_CAPACITY: f32 = 750.0;
const DISSOLVE_RATE: f32 = 0.2; // 0.25
const DEPOSITION_RATE: f32 = 0.05; // 0.001
const WATER_HEIGHT_FACTOR: f32 = 0.0;//0.0015;
const SPIKE_LIMIT: f32 = 0.001;

//convolution kernels
const SOBEL_KERNEL = vec3f(1.0, 2.0, 1.0);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(heightmap);
    // if (id.x >= dims.x || id.y >= dims.y) {
    //     return;
    // }
    let coords = vec2i(id.xy);

    let water_sediment_here = textureLoad(water_sediment, coords, 0).rg;

    // Use this version to add water height
    // let heights_bottom = vec3f(
    //     textureLoad(heightmap, coords + vec2i(-1, -1), 0).r + textureLoad(water_sediment, coords + vec2i(-1, -1), 0).r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(0, -1), 0).r + textureLoad(water_sediment, coords + vec2i(0, -1), 0).r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(1, -1), 0).r + textureLoad(water_sediment, coords + vec2i(1, -1), 0).r * WATER_HEIGHT_FACTOR,
    // );
    // let heights_center = vec3f(
    //     textureLoad(heightmap, coords + vec2i(-1, 0), 0).r + textureLoad(water_sediment, coords + vec2i(-1, 0), 0).r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(0, 0), 0).r + water_sediment_here.r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(1, 0), 0).r + textureLoad(water_sediment, coords + vec2i(1, 0), 0).r * WATER_HEIGHT_FACTOR,
    // );
    // let heights_top = vec3f(
    //     textureLoad(heightmap, coords + vec2i(-1, 1), 0).r + textureLoad(water_sediment, coords + vec2i(-1, 1), 0).r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(0, 1), 0).r + textureLoad(water_sediment, coords + vec2i(0, 1), 0).r * WATER_HEIGHT_FACTOR,
    //     textureLoad(heightmap, coords + vec2i(1, 1), 0).r + textureLoad(water_sediment, coords + vec2i(1, 1), 0).r * WATER_HEIGHT_FACTOR,
    // );

    // Use this version for no adding water height, its more optimized
    let heights_bottom = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, -1), 0).r,
        textureLoad(heightmap, coords + vec2i(0, -1), 0).r,
        textureLoad(heightmap, coords + vec2i(1, -1), 0).r,
    );
    let heights_center = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, 0), 0).r,
        textureLoad(heightmap, coords + vec2i(0, 0), 0).r,
        textureLoad(heightmap, coords + vec2i(1, 0), 0).r,
    );
    let heights_top = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, 1), 0).r,
        textureLoad(heightmap, coords + vec2i(0, 1), 0).r,
        textureLoad(heightmap, coords + vec2i(1, 1), 0).r,
    );

    let sobel_left = dot(vec3f(heights_bottom.x, heights_center.x, heights_top.x), SOBEL_KERNEL);
    let sobel_right = dot(vec3f(heights_bottom.z, heights_center.z, heights_top.z), SOBEL_KERNEL);
    let sobel_x = (sobel_right - sobel_left) / 8.0;

    let sobel_top = dot(heights_top, SOBEL_KERNEL);
    let sobel_bottom = dot(heights_bottom, SOBEL_KERNEL);
    let sobel_y = (sobel_top - sobel_bottom) / 8.0;

    let sobel_gradient = vec2f(sobel_x, sobel_y);
    let grad_len_sq = dot(sobel_gradient, sobel_gradient) + 1e-8; // guard against division by zero
    let sobel_normalized = sobel_gradient * inverseSqrt(grad_len_sq);

    textureStore(sobel_gradients, coords, vec4f(sobel_normalized, 0.0, 1.0));

    // Offset coordinates by gradient and get delta h.
    // textureSample needs normalized UVs; use textureSampleLevel
    let dims_f = vec2f(dims);
    let downhill_px = vec2f(coords) - sobel_normalized;
    let half_texel = vec2f(0.5) / dims_f;
    let downhill_uv = clamp(
        (downhill_px + 0.5) / dims_f,
        half_texel,
        vec2f(1.0) - half_texel
    );
    // let neighbor_height = textureSampleLevel(heightmap, heightmap_sampler, downhill_uv, 0.0).r + textureSampleLevel(water_sediment, heightmap_sampler, downhill_uv, 0.0).r * WATER_HEIGHT_FACTOR;
    let neighbor_height = textureSampleLevel(heightmap, heightmap_sampler, downhill_uv, 0.0).r;
    let height_here = heights_center.y;
    let delta_h = height_here - neighbor_height;

    //calculate velocity downhill based on gradient direction
    let speed = max(delta_h, 0.0);
    let water_here = water_sediment_here.r;
    let sediment_here = water_sediment_here.g;
    let capacity = max(MIN_HEIGHT_DELTA, delta_h) * water_here * speed * SEDIMENT_CAPACITY;

    // Write the height change: positive = deposit, negative = erode.
    var delta: f32 = 0.0;
    // if its true, its a weird sink where the uphill coordinate is lower
    // so try to fill in the sink
    if (delta_h < 0.0) {
        delta = sediment_here; // this gets clamped to delta h later anyways, so its never too large
    }
    else if (sediment_here > capacity) {
        delta = DEPOSITION_RATE * (sediment_here - capacity);
    }
    else {
        delta = DISSOLVE_RATE * (sediment_here - capacity);
    }

    // delta = max(-max(delta_h, 0.0), delta);
    delta = clamp(delta, min(-SPIKE_LIMIT, -delta_h), max(SPIKE_LIMIT, -delta_h));

    // let new_height = delta + height_here - water_sediment_here.r * WATER_HEIGHT_FACTOR;
    let new_height = delta + height_here;
    textureStore(write_heightmap, coords, vec4f(new_height, 0.0, 0.0, 1.0));

    let new_sediment = sediment_here - delta;
    let new_water_sediment = vec4f(water_here, new_sediment, 0.0, 1.0);
    textureStore(write_water_sediment, coords, new_water_sediment);
}
