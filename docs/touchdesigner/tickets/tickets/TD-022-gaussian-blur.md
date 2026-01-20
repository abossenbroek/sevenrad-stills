# TD-022: gaussian_blur Effect

---
id: TD-022
status: will_not_do
priority: medium
phase: 6
depends_on: [TD-015]
blocks: []
complexity: medium
shader_type: fragment
passes: 2
resolution: native_td_operator
---

## Status: WILL NOT DO

**Reason**: TouchDesigner provides native [Blur TOP](https://docs.derivative.ca/Blur_TOP) with Gaussian filter:
- Built-in Gaussian kernel filter type
- Configurable Filter Size (radius in pixels)
- Independent X/Y/Z axis scaling via Filter Scale
- Pre-Shrink option for performance optimization
- Multiple extend modes (Hold, Repeat, Mirror)
- GPU-accelerated native implementation

**Recommendation**: Use TD's built-in Blur TOP with Filter Type set to "Gaussian". It provides all the functionality needed with better performance than a custom GLSL implementation.

---

## Original Description (Archived)

Implement separable Gaussian blur with two passes (horizontal + vertical).

## Original Acceptance Criteria (Archived)

- [ ] ~~`gaussian_blur_h.frag` shader created (horizontal pass)~~
- [ ] ~~`gaussian_blur_v.frag` shader created (vertical pass)~~
- [ ] ~~Two-pass chaining in .tox~~
- [ ] ~~.tox operator packaged with help~~
- [ ] ~~Video-first demo included~~
- [ ] ~~Unit render tests passing~~

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

## Multi-Pass Specification (RF-003 fix)

| Pass | Shader | Input | Output Format |
|------|--------|-------|---------------|
| 1 (H) | gaussian_blur_h.frag | Source texture | RGBA32F |
| 2 (V) | gaussian_blur_v.frag | Pass 1 output | RGBA32F |

- **Intermediate texture**: Same resolution as input, RGBA32F format
- **Uniform sharing**: Both passes share `Kernelsize` and `Sigma` parameters
- **.tox structure**: glsl_blur_h → Render TOP (cache) → glsl_blur_v
