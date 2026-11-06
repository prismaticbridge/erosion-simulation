## Erosion Simulation and Renderer

This project has 2 main parts, erosion simulation and 3d rendering. It uses wgpu and winit for input, rendering, and gpu operations.

### Erosion simulation

The basic method is to generate an initial heightmap using fractional brownian motion, and then simulate hydraulic erosion for many steps. Most of the erosion code is in `terrain.rs` and the shaders directory, which contain compute shaders. The compute pipeline uses ping-pong textures for heightmap and water/sediment, since those need to be read and written to in the same shader pass.

**Technicalities**
At initialization, do the following:

- Allocate all GPU objects needed to do the terrain generation
- Initialize a rain texture with random noise, and a heightmap texture with fractional brownian motion
  At each simulation step, do the following:
- First shader pass: accumulate.wgsl
    - Increase the amount of water on each tile by the rain texture multiplied by a small RAIN_RATE
- Second shader pass: erosion.wgsl
    - Compute sobel gradients
    - Normalize the gradients and store them, these are negative water velocity (since water should flow down)
    - Compute the height delta at each pixel, by doing bilinear sampling on the heightmap at the point backwards from normalized sobels
    - Compute the sediment capacity at each pixel based on water amount, speed, and height delta
    - Compute sediment delta based on sediment capacity
    - Apply erosion or deposition based on sediment and capacity
- Third shader pass: advect.wgsl
    - At each surrounding pixel, look at the velocity and compute how much water and sediment from that pixel would flow into the current one
    - At the current pixel, compute how much water and sediment would stay in the current pixel, based on local velocity
    - Advect water and sediment based on these computations
    - Apply slight evaporation

### Terrain rendering

The heightmap is processed to generate a mesh. This is done in chunk.rs. The algorithm is to create a point at each heightmap pixel, then turn each square into 2 triangles. There are no optimizations so far, it renders every triangle.

This is uploaded as one vertex buffer to the gpu.

Here is the general architecture of honeycomb, the rendering engine:

- Renderer: contains the device and queue, camera, surface, window, render pipelines and bind groups, the depth texture, and a list of scenes
- Only one scene is active at once
- Each scene contains a static geometry buffer and a list of materials
- Every frame, the active scene's static geometry is rendered, and then each material's meshes are rendered
- Each material contains a bind group and texture, a vertex and index buffer, and a list of meshes. A mesh is unique geometry, but all meshes points are stored in the materials vertex buffer
- Each "mesh" contains an instance buffer, which is essentially a list of model transformations. These are rendered using an instanced draw call.
