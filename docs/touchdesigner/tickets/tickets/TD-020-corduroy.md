# TD-020: corduroy Effect

---
id: TD-020
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

Implement scanline-based brightness variation (corduroy pattern).

## Acceptance Criteria

- [ ] `corduroy.frag` shader created
- [ ] Per-scanline brightness multiplier
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Temporal Behavior

Static - pattern should be stable per seed.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Amount | Float | 0.1 | Brightness variation |
| Seed | Int | 42 | Random seed |

## Files

- `touchdesigner/glsl/effects/corduroy.frag` (create)
- `touchdesigner/tox/operators/sr_corduroy.tox` (create)
