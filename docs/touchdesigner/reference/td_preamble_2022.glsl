// TouchDesigner GLSL Preamble Reference - Version 2022.20000+
//
// This file documents the GLSL preamble that TouchDesigner injects into shaders
// at runtime. User shaders should NOT include #version directives as TD handles this.
//
// STATUS: SYNTHETIC (needs real extraction from TouchDesigner)
// See TD-001 ticket for extraction procedure.
//
// Extraction Procedure:
// 1. Open TouchDesigner 2022.20000+
// 2. Create a new GLSL TOP
// 3. Enter minimal shader: void main() { fragColor = vec4(1.0); }
// 4. Create an Info DAT and connect it to the GLSL TOP
// 5. In Info DAT parameters, set "Operator" to point to your GLSL TOP
// 6. The Info DAT will show all injected uniforms and preamble code
// 7. Copy the uniform declarations section
// 8. Also check: GLSL TOP -> Right-click -> View -> GLSL Info for additional details
//
// =============================================================================
// FRAGMENT SHADER PREAMBLE (GLSL 330 core)
// =============================================================================

#version 330 core

// Input from vertex shader
in vec2 vUV;                    // Texture coordinates (0-1 range)

// Standard output
layout(location = 0) out vec4 fragColor;

// ----- TouchDesigner Built-in Uniforms -----

// Input textures (up to 8 inputs)
uniform sampler2D sTD2DInputs[8];

// Output info: vec4(width, height, 1/width, 1/height)
uniform vec4 uTDOutputInfo;

// Current render pass index (for multi-pass rendering)
uniform int uTDPass;

// ----- TouchDesigner Helper Functions -----

// Output swizzle for color space conversion (identity in most cases)
vec4 TDOutputSwizzle(vec4 c) { return c; }

// ----- Additional uniforms discovered from TD -----
// TODO: Add any additional uniforms discovered during extraction:
// - uTDTime (time-related uniforms?)
// - uTDFrame (frame number?)
// - uTDGeneral (general info?)
// - Check Info DAT for complete list

// =============================================================================
// COMPUTE SHADER PREAMBLE (GLSL 430 core)
// =============================================================================

/*
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// Input textures
uniform sampler2D sTD2DInputs[8];

// Output images (read-write)
layout(rgba32f) uniform image2D sTD2DOutputs[8];

// TODO: Document compute-specific uniforms:
// - gl_WorkGroupID
// - gl_LocalInvocationID
// - gl_GlobalInvocationID
// - Any TD-specific compute uniforms
*/

// =============================================================================
// NOTES
// =============================================================================
//
// 1. User shaders should NOT include #version - TD injects it
// 2. Fragment shaders use GLSL 330 core for maximum Apple compatibility
// 3. Compute shaders require GLSL 430+ (use only when necessary)
// 4. All effects should use sTD2DInputs[0] for the primary input
// 5. TDOutputSwizzle() should wrap final output for consistency
//
// Example minimal passthrough shader:
//
//   void main() {
//       vec4 color = texture(sTD2DInputs[0], vUV.st);
//       fragColor = TDOutputSwizzle(color);
//   }
//
