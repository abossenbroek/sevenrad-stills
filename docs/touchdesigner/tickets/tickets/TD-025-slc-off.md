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

## Acceptance Criteria

- [ ] `slc_off.comp` compute shader created
- [ ] Wedge-shaped mask application
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Temporal Behavior

Static - wedge pattern stable per seed.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Gap | Float | 0.1 | Gap width |
| Seed | Int | 42 | Random seed |

## Files

- `touchdesigner/glsl/effects/slc_off.comp` (create)
- `touchdesigner/tox/operators/sr_slc_off.tox` (create)

## Notes

Requires GLSL 430+ for compute shader support.
