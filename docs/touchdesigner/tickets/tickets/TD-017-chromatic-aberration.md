# TD-017: chromatic_aberration Effect

---
id: TD-017
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: simple
shader_type: fragment
passes: 1
---

## Description

Implement chromatic aberration effect with per-channel UV offset.

## Acceptance Criteria

- [ ] `chromatic_aberration.frag` shader created
- [ ] Separate R, G, B channel offsets
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Taichi Reference

`src/sevenrad_stills/operations/chromatic_aberration_taichi.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Shiftx | Float | 0.01 | Horizontal offset |
| Shifty | Float | 0.01 | Vertical offset |

## Files

- `touchdesigner/glsl/effects/chromatic_aberration.frag` (create)
- `touchdesigner/tox/operators/sr_chromatic_aberration.tox` (create)

## Implementation

```glsl
uniform vec2 uShift;

void main() {
    vec2 uv = vUV.st;

    // Sample each channel with offset
    float r = texture(sTD2DInputs[0], uv + uShift).r;
    float g = texture(sTD2DInputs[0], uv).g;
    float b = texture(sTD2DInputs[0], uv - uShift).b;
    float a = texture(sTD2DInputs[0], uv).a;

    fragColor = TDOutputSwizzle(vec4(r, g, b, a));
}
```

## References

- [02-EFFECTS-AND-DEMOS.md](../../02-EFFECTS-AND-DEMOS.md)
