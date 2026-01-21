# TD-027: buffer_corruption Effect

---
id: TD-027
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: complex
shader_type: compute
passes: 1
---

## Description

Implement multi-mode buffer corruption (shift, swap, zero, XOR) for glitch effects.

## Acceptance Criteria

- [ ] `buffer_corruption.comp` compute shader created
- [ ] Shader uses #version 430 core
- [ ] All modes: xor, invert, channel_shuffle (matches Taichi - RF-002 fix)
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Fallback

XOR mode requires bitwise float manipulation which GLSL cannot do natively. If GLSL compute fails for XOR, activate [TD-027a](TD-027a-buffer-corruption-cpp.md).

**Fallback Trigger Criteria (RF-009 fix)**: Activate TD-027a if ANY of:
1. SPIRV-Cross Metal compilation fails for buffer_corruption.comp
2. Perceptual diff SSIM < 0.95 vs Taichi reference
3. Frame time > 8ms at 1080p (2x the 4ms target)

## Temporal Behavior

Configurable via `Animateglitch` parameter.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Mode | Menu | xor | xor, invert, channel_shuffle |
| Amount | Float | 0.1 | Corruption intensity |
| Seed | Int | 42 | Random seed |
| Animateglitch | Toggle | Off | Per-frame variation |

## Files

- `touchdesigner/glsl/effects/buffer_corruption.comp` (create)
- `touchdesigner/tox/operators/sr_buffer_corruption.tox` (create)

## Notes

- Requires GLSL 430+ for compute shader support
- Most complex effect
- XOR mode requires bitwise float manipulation
- C++ fallback pre-approved if needed

## XOR Mode Implementation (TR-013 fix)

GLSL can perform bitwise operations on floats using reinterpretation:

```glsl
// XOR mode implementation using floatBitsToUint/uintBitsToFloat
vec4 applyXorCorruption(vec4 color, uint mask) {
    return vec4(
        uintBitsToFloat(floatBitsToUint(color.r) ^ mask),
        uintBitsToFloat(floatBitsToUint(color.g) ^ mask),
        uintBitsToFloat(floatBitsToUint(color.b) ^ mask),
        color.a  // preserve alpha
    );
}
```

**Note**: Results may produce NaN/Inf values. Clamp output or use `isinf()`/`isnan()` checks. If GLSL XOR produces unacceptable artifacts, activate TD-027a C++ fallback.
