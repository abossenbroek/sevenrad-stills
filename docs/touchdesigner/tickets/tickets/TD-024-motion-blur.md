# TD-024: motion_blur Effect

---
id: TD-024
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: medium
shader_type: fragment
passes: 1
---

## Description

Implement directional motion blur along specified angle.

## Taichi Reference

`src/sevenrad_stills/operations/motion_blur_taichi.py`

## Acceptance Criteria

- [ ] `motion_blur.frag` shader created
- [ ] Linear kernel along angle
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Angle | Float | 0.0 | Blur direction (degrees) |
| Amount | Float | 10.0 | Blur length |
| Samples | Int | 16 | Number of samples |

## Implementation

```glsl
uniform float uAngle;    // Blur direction in degrees
uniform float uAmount;   // Blur length in pixels
uniform int uSamples;

void main() {
    vec2 uv = vUV.st;
    vec2 texelSize = 1.0 / vec2(textureSize(sTD2DInputs[0], 0));

    // Convert angle to radians and calculate direction
    float rad = radians(uAngle);
    vec2 direction = vec2(cos(rad), sin(rad)) * uAmount * texelSize;

    vec4 color = vec4(0.0);

    for (int i = 0; i < uSamples; i++) {
        // Sample along the blur direction, centered on current pixel
        float t = (float(i) / float(uSamples - 1)) - 0.5;
        vec2 offset = direction * t;
        color += texture(sTD2DInputs[0], uv + offset);
    }

    color /= float(uSamples);
    fragColor = TDOutputSwizzle(color);
}
```

## Files

- `touchdesigner/glsl/effects/motion_blur.frag` (create)
- `touchdesigner/tox/operators/sr_motion_blur.tox` (create)
