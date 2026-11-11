@group(2) @binding(0) var heightmap: texture_2d<f32>;
@group(2) @binding(1) var heightmap_sampler: sampler;
@group(2) @binding(2) var write_heightmap: texture_storage_2d<r32float, write>;


const GAUSSIAN_TOP = vec3f(0.0625, 0.125, 0.0625);
const GAUSSIAN_CENTER = vec3f(0.125, 0.25, 0.125);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coords = vec2i(id.xy);

    let dims = vec2i(textureDimensions(write_heightmap));
    if (any(coords >= dims) || any(coords < vec2i(0))) {
        return;
    }

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

    let height_here = heights_center.y;

    // this laplacian approximation is scaled by 6x
    let local_laplacian = dot(heights_bottom, LAPLACIAN_TOP) + dot(heights_center, LAPLACIAN_CENTER) + dot(heights_top, LAPLACIAN_TOP);
    let local_gaussian = dot(heights_bottom, GAUSSIAN_TOP) + dot(heights_center, GAUSSIAN_CENTER) + dot(heights_top, GAUSSIAN_TOP);
    let gaussian_height_difference = local_gaussian - height_here;

    let thermal_erosion_amount = gaussian_height_difference * local_laplacian / 42.0;
    let new_height = height_here + thermal_erosion_amount;

    textureStore(write_heightmap, coords, vec4(new_height, 0.0, 0.0, 1.0));
}
