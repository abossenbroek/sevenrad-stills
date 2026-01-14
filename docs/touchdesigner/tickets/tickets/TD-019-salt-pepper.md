# TD-019: salt_pepper Effect

---
id: TD-019
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

Implement dedicated salt and pepper noise effect with dual threshold.

## Acceptance Criteria

- [ ] `salt_pepper.frag` shader created
- [ ] Separate salt/pepper probability controls
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Amount | Float | 0.05 | Overall noise probability |
| Seed | Int | 42 | Random seed |
| Animatenoise | Toggle | Off | Per-frame variation |

## Files

- `touchdesigner/glsl/effects/salt_pepper.frag` (create)
- `touchdesigner/tox/operators/sr_salt_pepper.tox` (create)
