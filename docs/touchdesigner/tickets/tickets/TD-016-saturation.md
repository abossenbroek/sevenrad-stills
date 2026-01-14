# TD-016: saturation Effect

---
id: TD-016
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

Implement saturation adjustment effect using HSV color space.

## Acceptance Criteria

- [ ] `saturation.frag` shader created
- [ ] Supports modes: multiply, add, set
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Taichi Reference

`src/sevenrad_stills/operations/saturation_taichi.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Mode | Menu | multiply | multiply, add, set |
| Value | Float | 1.0 | 0=gray, 1=original, >1=boosted |

## Files

- `touchdesigner/glsl/effects/saturation.frag` (create)
- `touchdesigner/tox/operators/sr_saturation.tox` (create)

## Implementation

```glsl
uniform float uSaturation;
uniform int uMode;  // 0=multiply, 1=add, 2=set

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    vec3 hsv = rgb_to_hsv(color.rgb);

    if (uMode == 0) hsv.y *= uSaturation;
    else if (uMode == 1) hsv.y += uSaturation - 1.0;
    else hsv.y = uSaturation;

    hsv.y = clamp(hsv.y, 0.0, 1.0);
    color.rgb = hsv_to_rgb(hsv);
    fragColor = TDOutputSwizzle(color);
}
```

## References

- [02-EFFECTS-AND-DEMOS.md](../../02-EFFECTS-AND-DEMOS.md)
