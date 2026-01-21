# TD-025: slc_off Effect

---
id: TD-025
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: medium
shader_type: compute
passes: 1
---

## Description

Implement Landsat scan line corrector failure simulation (wedge-shaped mask).

## Taichi Reference

`src/sevenrad_stills/operations/slc_off_taichi.py`

## Acceptance Criteria

- [ ] `slc_off.comp` compute shader created
- [ ] Wedge-shaped mask application
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing
- [ ] Shader uses #version 430 core

## Temporal Behavior

Static - wedge pattern stable per seed.

## Parameters (RF-005 fix - matches Taichi)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Gapwidth | Float | 0.1 | Gap width as fraction of image |
| Scanperiod | Int | 14 | Stripe frequency (Landsat-7 default) |
| Fillmode | Menu | black | black, white, mean |
| Seed | Int | 42 | Random seed |

## Implementation

```glsl
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D inputTex;
layout(rgba32f, binding = 1) uniform image2D outputTex;

uniform float uGapwidth;
uniform int uScanperiod;
uniform int uFillmode;  // 0=black, 1=white, 2=mean
uniform int uSeed;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inputTex);

    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 color = imageLoad(inputTex, pos);

    // Calculate wedge-shaped gap pattern (Landsat SLC-off simulation)
    // Gap widens from center toward edges
    float centerDist = abs(float(pos.x) / float(size.x) - 0.5) * 2.0;
    float gapWidth = uGapwidth * (1.0 + centerDist);

    // Check if pixel falls in gap based on scanline period
    int scanline = pos.y % uScanperiod;
    bool inGap = float(scanline) < gapWidth * float(uScanperiod);

    if (inGap) {
        if (uFillmode == 0) color = vec4(0.0, 0.0, 0.0, 1.0);      // black
        else if (uFillmode == 1) color = vec4(1.0, 1.0, 1.0, 1.0); // white
        // mode 2 (mean) would require neighbor sampling
    }

    imageStore(outputTex, pos, color);
}
```

## Files

- `touchdesigner/glsl/effects/slc_off.comp` (create)
- `touchdesigner/tox/operators/sr_slc_off.tox` (create)

## Notes

Requires GLSL 430+ for compute shader support.
