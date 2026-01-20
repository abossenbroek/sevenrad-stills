// hello_brightness.frag
// Adjustable brightness effect - Hello World for TD GLSL development
//
// Uniforms:
//   uBrightness - Brightness multiplier (range: 0.0-2.0, default: 1.0)
//
// NOTE: Omit version directive - TouchDesigner auto-injects it
//
// Usage in TouchDesigner:
//   1. Create Text DAT with File parameter pointing to this file
//   2. Create GLSL TOP, set GLSL DAT to the Text DAT
//   3. Add Custom Parameter "Brightness" (Float, 0.0-2.0, default 1.0)
//   4. In GLSL TOP Vectors page: vec0name = "uBrightness", vec0valuex = par.Brightness

uniform float uBrightness;

void main() {
    // Read input texture at current UV coordinate
    vec4 color = texture(sTD2DInputs[0], vUV.st);

    // Apply brightness adjustment (preserving alpha)
    color.rgb *= uBrightness;

    // Output with proper color space handling
    fragColor = TDOutputSwizzle(color);
}
