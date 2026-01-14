# TD-022: gaussian_blur Effect

---
id: TD-022
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: medium
shader_type: fragment
passes: 2
---

## Description

Implement separable Gaussian blur with two passes (horizontal + vertical).

## Acceptance Criteria

- [ ] `gaussian_blur_h.frag` shader created (horizontal pass)
- [ ] `gaussian_blur_v.frag` shader created (vertical pass)
- [ ] Two-pass chaining in .tox
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Taichi Reference

`src/sevenrad_stills/operations/blur_gaussian_taichi.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Kernelsize | Int | 5 | Blur radius |
| Sigma | Float | 1.0 | Gaussian sigma |

## Files

- `touchdesigner/glsl/effects/gaussian_blur_h.frag` (create)
- `touchdesigner/glsl/effects/gaussian_blur_v.frag` (create)
- `touchdesigner/tox/operators/sr_gaussian_blur.tox` (create)

## Notes

- Multi-pass effect requires chained GLSL TOPs in .tox
- Separable convolution for performance
