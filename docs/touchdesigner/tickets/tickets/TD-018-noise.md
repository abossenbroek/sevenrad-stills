# TD-018: noise Effect

---
id: TD-018
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

Implement noise effect with uniform, gaussian, and salt-pepper modes.

## Acceptance Criteria

- [ ] `noise.frag` shader created
- [ ] Supports modes: uniform, gaussian (RF-011 fix: salt_pepper removed, use TD-019)
- [ ] Animate toggle for temporal variation
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Taichi Reference

`src/sevenrad_stills/operations/noise_taichi.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Mode | Menu | uniform | uniform, gaussian |
| Amount | Float | 0.1 | Noise intensity |
| Seed | Int | 42 | Random seed |
| Animatenoise | Toggle | Off | Per-frame variation |

## Temporal Behavior

- `Animatenoise = 0`: Static noise pattern
- `Animatenoise = 1`: Different noise each frame

## Files

- `touchdesigner/glsl/effects/noise.frag` (create)
- `touchdesigner/tox/operators/sr_noise.tox` (create)

## Implementation

```glsl
uniform int uMode;
uniform float uAmount;
uniform int uSeed;
uniform int uAnimateNoise;
uniform float uTime;

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int seed = getEffectiveSeed(uSeed, uAnimateNoise, uTime);

    if (uMode == 0) {
        // Uniform noise
        float n = rand_float(pos, seed) * 2.0 - 1.0;
        color.rgb += n * uAmount;
    } else if (uMode == 1) {
        // Gaussian noise
        float n = rand_gaussian(pos, seed, uAmount);
        color.rgb += n;
    }
    // RF-011 fix: salt_pepper mode removed - use dedicated TD-019 effect

    fragColor = TDOutputSwizzle(saturate(color));
}
```

## References

- [02-EFFECTS-AND-DEMOS.md](../../02-EFFECTS-AND-DEMOS.md)
