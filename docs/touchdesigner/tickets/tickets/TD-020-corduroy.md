# TD-020: corduroy Effect

---
id: TD-020
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

Implement scanline-based brightness variation (corduroy pattern).

## Taichi Reference

`src/sevenrad_stills/operations/corduroy_taichi.py`

**Note**: Also consult Max/MSP implementation for additional clarity on algorithm behavior.

## Acceptance Criteria

- [ ] `corduroy.frag` shader created
- [ ] Per-scanline brightness multiplier
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Temporal Behavior

Static - pattern should be stable per seed.

## Parameters (matches Taichi)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Orientation | Menu | horizontal | horizontal, vertical |
| Strength | Float | 0.2 | Brightness variation intensity |
| Density | Float | 1.0 | Line density multiplier |
| Seed | Int | 42 | Random seed |

## Implementation

```glsl
uniform int uOrientation;  // 0=horizontal, 1=vertical
uniform float uStrength;
uniform float uDensity;
uniform int uSeed;

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    ivec2 pos = ivec2(gl_FragCoord.xy);

    // Select scanline based on orientation
    int scanline = (uOrientation == 0) ? pos.y : pos.x;

    // Generate per-line brightness multiplier
    float lineNoise = rand_float(ivec2(scanline, 0), uSeed);
    float multiplier = 1.0 + (lineNoise * 2.0 - 1.0) * uStrength;

    color.rgb *= multiplier;
    fragColor = TDOutputSwizzle(saturate(color));
}
```

## Files

- `touchdesigner/glsl/effects/corduroy.frag` (create)
- `touchdesigner/tox/operators/sr_corduroy.tox` (create)
