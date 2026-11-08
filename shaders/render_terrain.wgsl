struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(1) @binding(0) var<uniform> light_direction: vec3<f32>;

const DIFFUSE: f32 = 0.2;

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera * vec4<f32>(vertex.position, 1.0);
    let ndotl = max(0.0, dot(vertex.normal, -light_direction));
    let lighting = min(1.0, ndotl + DIFFUSE);
    out.color = vec3<f32>(vertex.color * lighting);
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
