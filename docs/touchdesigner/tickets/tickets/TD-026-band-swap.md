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

**Fallback Trigger Criteria (RF-009 fix)**: Activate TD-026a if ANY of:
1. SPIRV-Cross Metal compilation fails for band_swap.comp
2. Perceptual diff SSIM < 0.95 vs Taichi reference
3. Frame time > 8ms at 1080p (2x the 4ms target)

## Temporal Behavior

Static - band order stable per seed.

## Parameters (RF-004 fix - matches Taichi)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Tilecount | Int | 8 | Number of tiles to create |
| Permutation | Menu | RGB | RGB, BGR, GRB, GBR, BRG, RBG |
| Tilesizemin | Float | 0.05 | Min tile size as fraction of height |
| Tilesizemax | Float | 0.15 | Max tile size as fraction of height |
| Seed | Int | 42 | Random seed |

## Files

- `touchdesigner/glsl/effects/band_swap.comp` (create)
- `touchdesigner/tox/operators/sr_band_swap.tox` (create)

## Notes

Requires GLSL 430+ for compute shader support.
