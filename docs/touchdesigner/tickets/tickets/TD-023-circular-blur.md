# TD-023: circular_blur Effect

---
id: TD-023
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

Implement disk kernel blur (circular/bokeh blur).

## Taichi Reference

`src/sevenrad_stills/operations/blur_circular_taichi.py`

## Acceptance Criteria

- [ ] `circular_blur.frag` shader created
- [ ] 2D disk kernel sampling
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Radius | Float | 5.0 | Blur radius |
| Samples | Int | 16 | Number of samples |

## Implementation

```glsl
uniform float uRadius;
uniform int uSamples;

// Golden angle for uniform disk distribution
const float GOLDEN_ANGLE = 2.39996323;

void main() {
    vec2 uv = vUV.st;
    vec2 texelSize = 1.0 / vec2(textureSize(sTD2DInputs[0], 0));

    vec4 color = vec4(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < uSamples; i++) {
        // Golden angle spiral for uniform disk sampling
        float r = sqrt(float(i) / float(uSamples)) * uRadius;
        float theta = float(i) * GOLDEN_ANGLE;

        vec2 offset = vec2(cos(theta), sin(theta)) * r * texelSize;
        color += texture(sTD2DInputs[0], uv + offset);
        totalWeight += 1.0;
    }

    color /= totalWeight;
    fragColor = TDOutputSwizzle(color);
}
```

## Files

- `touchdesigner/glsl/effects/circular_blur.frag` (create)
- `touchdesigner/tox/operators/sr_circular_blur.tox` (create)
