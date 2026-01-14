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
- [ ] All modes: shift, swap, zero (XOR may need C++)
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Fallback

XOR mode may require C++ implementation. If GLSL compute fails for XOR, activate [TD-027a](TD-027a-buffer-corruption-cpp.md).

## Temporal Behavior

Configurable via `Animateglitch` parameter.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Mode | Menu | shift | shift, swap, zero, xor |
| Amount | Float | 0.1 | Corruption intensity |
| Seed | Int | 42 | Random seed |
| Animateglitch | Toggle | Off | Per-frame variation |

## Files

- `touchdesigner/glsl/effects/buffer_corruption.comp` (create)
- `touchdesigner/tox/operators/sr_buffer_corruption.tox` (create)

## Notes

- Most complex effect
- XOR mode requires bitwise float manipulation
- C++ fallback pre-approved if needed
