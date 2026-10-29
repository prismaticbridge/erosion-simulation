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

const MIN_HEIGHT_DELTA: f32 = 0.06;
const SEDIMENT_CAPACITY: f32 = 50.0;
const DISSOLVE_RATE: f32 = 0.25;
const DEPOSITION_RATE: f32 = 0.001;
const SOBEL_KERNEL = vec3f(1.0, 2.0, 1.0);
const REPOSE_SLOPE: f32 = 0.08;

const GAUSSIAN_TOP = vec3f(0.0625, 0.125, 0.0625);
const GAUSSIAN_CENTER = vec3f(0.125,  0.25,  0.125);
//gaussian bottom is the same as top

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(heightmap);
    // if (id.x >= dims.x || id.y >= dims.y) {
    //     return;
    // }
    let coords = vec2i(id.xy);

    // Clamp the +-1 sobel taps so edge texels don't read out of bounds.
    // let c = clamp(coords, vec2i(1), vec2i(dims) - vec2i(2));
    var heights_bottom = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, -1), 0).r,
        textureLoad(heightmap, coords + vec2i(0, -1), 0).r,
        textureLoad(heightmap, coords + vec2i(1, -1), 0).r,
    );
    var heights_center = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, 0), 0).r,
        textureLoad(heightmap, coords + vec2i(0, 0), 0).r,
        textureLoad(heightmap, coords + vec2i(1, 0), 0).r,
    );
    var heights_top = vec3f(
        textureLoad(heightmap, coords + vec2i(-1, 1), 0).r,
        textureLoad(heightmap, coords + vec2i(0, 1), 0).r,
        textureLoad(heightmap, coords + vec2i(1, 1), 0).r,
    );
    //old, slow method:
    // for (var x: i32 = -1; x <= 1; x = x+1) {
    //     for (var y: i32 = -1; y <= 1; y = y+1) {
    //         let index = (y+1) * 3 + (x+1);
    //         let sample_coords = coords + vec2i(x, y));
    //         heights[index] = textureLoad(heightmap, sample_coords, 0).r;
    //     }
    // }

    let sobel_left = dot(vec3f(heights_bottom.x, heights_center.x, heights_top.x), SOBEL_KERNEL);
    let sobel_right = dot(vec3f(heights_bottom.z, heights_center.z, heights_top.z), SOBEL_KERNEL);
    let sobel_x = (sobel_right - sobel_left) / 8.0;

    let sobel_top = dot(heights_top, SOBEL_KERNEL);
    let sobel_bottom = dot(heights_bottom, SOBEL_KERNEL);
    let sobel_y = (sobel_bottom - sobel_top) / 8.0;

    // Guard against normalize(vec2(0)) producing NaN on flat terrain.
    let sobel_gradient = vec2f(sobel_x, sobel_y);
    let grad_len_sq = dot(sobel_gradient, sobel_gradient);
    let norm_gradient = normalize(sobel_gradient);
    let sobel_normalized = select(vec2f(0.0), norm_gradient, grad_len_sq > 1e-6);
    textureStore(sobel_gradients, coords, vec4f(sobel_normalized, 0.0, 1.0));

    // Offset coordinates by gradient and get delta h.
    // textureSample needs normalized UVs; use textureSampleLevel
    let dims_f = vec2f(dims);
    let downhill_px = vec2f(coords) - sobel_normalized;
    let downhill_uv = clamp(downhill_px / dims_f, vec2f(0.0), vec2f(1.0) - 1.0 / dims_f);
    let neighbor_height = textureSampleLevel(heightmap, heightmap_sampler, downhill_uv, 0.0).r;
    let height_here = heights_center.y;
    let delta_h = height_here - neighbor_height;

    // compute thermal erosion
    let gaussian_height = dot(heights_bottom, GAUSSIAN_TOP) +
                          dot(heights_center, GAUSSIAN_CENTER) +
                          dot(heights_top, GAUSSIAN_TOP);
    let height_delta_smooth = select(0.0, gaussian_height - height_here, delta_h > REPOSE_SLOPE);

    //calculate velocity downhill based on gradient direction
    let speed = delta_h * 15.0;
    let water_sediment_here = textureLoad(water_sediment, coords, 0).rg;
    let water_here = water_sediment_here.r;
    let sediment_here = water_sediment_here.g;
    let capacity = max(MIN_HEIGHT_DELTA, delta_h) * water_here * speed * SEDIMENT_CAPACITY;

    // Write the height change: positive = deposit, negative = erode.
    var delta: f32 = 0.0;
    if (delta_h <= 0.0) {
        delta = 0.0;
    }
    else if (sediment_here > capacity) {
        delta = DEPOSITION_RATE * (sediment_here - capacity);
    }
    else {
        delta = -DISSOLVE_RATE * (capacity - sediment_here);
    }

    let new_height = delta + height_here + height_delta_smooth;
    textureStore(write_heightmap, coords, vec4f(new_height, 0.0, 0.0, 1.0));

    let new_sediment = sediment_here - delta;
    let new_water_sediment = vec4f(water_here, new_sediment, 0.0, 1.0);
    textureStore(write_water_sediment, coords, new_water_sediment);
}
