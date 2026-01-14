# TD-026: band_swap Effect

---
id: TD-026
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

Implement tile-based band permutation effect.

## Acceptance Criteria

- [ ] `band_swap.comp` compute shader created
- [ ] Random band permutation
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Fallback

If GLSL compute fails, activate [TD-026a](TD-026a-band-swap-cpp.md) for C++ implementation.

## Temporal Behavior

Static - band order stable per seed.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Bandheight | Int | 16 | Band height in pixels |
| Swapcount | Int | 5 | Number of swaps |
| Seed | Int | 42 | Random seed |

## Files

- `touchdesigner/glsl/effects/band_swap.comp` (create)
- `touchdesigner/tox/operators/sr_band_swap.tox` (create)

## Notes

Requires GLSL 430+ for compute shader support.
