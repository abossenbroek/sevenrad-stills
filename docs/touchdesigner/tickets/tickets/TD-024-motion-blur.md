# TD-024: motion_blur Effect

---
id: TD-024
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

Implement directional motion blur along specified angle.

## Acceptance Criteria

- [ ] `motion_blur.frag` shader created
- [ ] Linear kernel along angle
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Angle | Float | 0.0 | Blur direction (degrees) |
| Amount | Float | 10.0 | Blur length |
| Samples | Int | 16 | Number of samples |

## Files

- `touchdesigner/glsl/effects/motion_blur.frag` (create)
- `touchdesigner/tox/operators/sr_motion_blur.tox` (create)
