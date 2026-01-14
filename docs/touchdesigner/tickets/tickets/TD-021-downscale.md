# TD-021: downscale Effect

---
id: TD-021
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: medium
shader_type: fragment
passes: 1-2
---

## Description

Implement resolution reduction with box filter.

## Acceptance Criteria

- [ ] `downscale.frag` shader created
- [ ] Box filter averaging
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Factor | Int | 2 | Downscale factor |

## Files

- `touchdesigner/glsl/effects/downscale.frag` (create)
- `touchdesigner/tox/operators/sr_downscale.tox` (create)
