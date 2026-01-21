// Minimal passthrough shader
// This shader simply reads from input 0 and writes to output
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    fragColor = TDOutputSwizzle(color);
}
