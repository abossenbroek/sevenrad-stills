# TD-016: saturation Effect

---
id: TD-016
status: will_not_do
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: simple
shader_type: fragment
passes: 1
resolution: native_td_operator
---

## Status: WILL NOT DO

**Reason**: TouchDesigner provides native [HSV Adjust TOP](https://docs.derivative.ca/HSV_Adjust_TOP) which offers superior functionality:
- Selective saturation control via Saturation Range and Falloff
- Saturation Multiplier (0 = desaturate, 1 = unchanged, 2 = double)
- Hue-based targeting for precise color adjustments
- GPU-accelerated native implementation

**Recommendation**: Use TD's built-in HSV Adjust TOP instead. It provides more control than our simple multiply/add/set modes.

---

## Original Description (Archived)

Implement saturation adjustment effect using HSV color space.

## Original Acceptance Criteria (Archived)

- [ ] ~~`saturation.frag` shader created~~
- [ ] ~~Supports modes: multiply, add, set~~
- [ ] ~~.tox operator packaged with help~~
- [ ] ~~Video-first demo included~~
- [ ] ~~Unit render tests passing~~

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
