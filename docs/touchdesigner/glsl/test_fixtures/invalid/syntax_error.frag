// Missing semicolon - should fail validation
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st)
    fragColor = TDOutputSwizzle(color);
}
