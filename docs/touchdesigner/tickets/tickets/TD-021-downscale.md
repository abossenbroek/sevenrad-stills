# TD-021: downscale Effect

---
id: TD-021
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

Implement resolution reduction with box filter.

## Taichi Reference

`src/sevenrad_stills/operations/downscale_taichi.py`

## Acceptance Criteria

- [ ] `downscale.frag` shader created
- [ ] Box filter averaging
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Factor | Int | 2 | Downscale factor |

## Implementation

```glsl
uniform int uFactor;  // Downscale factor (2, 4, 8, etc.)

void main() {
    vec2 uv = vUV.st;
    ivec2 inputSize = textureSize(sTD2DInputs[0], 0);

    // Calculate the block of pixels to average
    vec2 blockStart = floor(uv * vec2(inputSize) / float(uFactor)) * float(uFactor);

    vec4 color = vec4(0.0);
    float count = 0.0;

    // Box filter: average all pixels in the block
    for (int y = 0; y < uFactor; y++) {
        for (int x = 0; x < uFactor; x++) {
            vec2 samplePos = (blockStart + vec2(x, y) + 0.5) / vec2(inputSize);
            color += texture(sTD2DInputs[0], samplePos);
            count += 1.0;
        }
    }

    color /= count;
    fragColor = TDOutputSwizzle(color);
}
```

## Files

- `touchdesigner/glsl/effects/downscale.frag` (create)
- `touchdesigner/tox/operators/sr_downscale.tox` (create)

## Notes (RF-010 fix)

Single-pass box filter implementation matching Taichi reference. For extreme downscale factors (8x+), consider using TD's built-in Resolution TOP followed by this effect for quality, but the shader itself is single-pass.
