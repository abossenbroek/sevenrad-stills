// Shader using TD uniforms and custom parameters
uniform float uBrightness;

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    color.rgb *= uBrightness;
    fragColor = TDOutputSwizzle(color);
}
