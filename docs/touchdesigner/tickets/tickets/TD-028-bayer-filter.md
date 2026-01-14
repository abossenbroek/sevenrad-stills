# TD-028: bayer_filter Effect

---
id: TD-028
status: pending
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: complex
shader_type: fragment
passes: 2
---

## Description

Implement CFA (Color Filter Array) mosaic and demosaic simulation.

## Acceptance Criteria

- [ ] `bayer_mosaic.frag` shader created (pass 1)
- [ ] `bayer_demosaic.frag` shader created (pass 2)
- [ ] Two-pass chaining in .tox
- [ ] .tox operator packaged with help
- [ ] Video-first demo included
- [ ] Unit render tests passing

## Taichi Reference

`src/sevenrad_stills/operations/bayer_filter_taichi.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Pattern | Menu | RGGB | RGGB, BGGR, GRBG, GBRG |

## Files

- `touchdesigner/glsl/effects/bayer_mosaic.frag` (create)
- `touchdesigner/glsl/effects/bayer_demosaic.frag` (create)
- `touchdesigner/tox/operators/sr_bayer_filter.tox` (create)

## Notes

- Two-pass effect: mosaic then demosaic
- Tests interaction of 2x2 Bayer pattern with video compression artifacts

## Multi-Pass Specification (RF-003 fix)

| Pass | Shader | Input | Output Format |
|------|--------|-------|---------------|
| 1 (Mosaic) | bayer_mosaic.frag | Source texture | RGBA32F |
| 2 (Demosaic) | bayer_demosaic.frag | Pass 1 output | RGBA32F |

- **Intermediate texture**: Same resolution as input, RGBA32F format
- **Uniform sharing**: Both passes share `Pattern` parameter (RGGB/BGGR/GRBG/GBRG)
- **.tox structure**: glsl_mosaic → Render TOP (cache) → glsl_demosaic
