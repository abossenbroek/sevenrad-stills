# TD-023: circular_blur Effect

---
id: TD-023
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

Implement disk kernel blur (circular/bokeh blur).

## Acceptance Criteria

- [ ] `circular_blur.frag` shader created
- [ ] 2D disk kernel sampling
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Radius | Float | 5.0 | Blur radius |
| Samples | Int | 16 | Number of samples |

## Files

- `touchdesigner/glsl/effects/circular_blur.frag` (create)
- `touchdesigner/tox/operators/sr_circular_blur.tox` (create)
