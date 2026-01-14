# TD-015: Create tdCommon.glsl

---
id: TD-015
status: pending
priority: high
phase: 6
depends_on: [TD-004, TD-005, TD-006]
blocks: [TD-016, TD-017, TD-018, TD-019, TD-020, TD-021, TD-022, TD-023, TD-024, TD-025, TD-026, TD-027, TD-028]
---

## Description

Create shared GLSL utilities file with validated PCG hash, bilinear sampling, and HSV conversion functions.

## Acceptance Criteria

- [ ] `touchdesigner/glsl/common/tdCommon.glsl` created
- [ ] Contains validated PCG hash (from TD-004)
- [ ] Contains validated bilinear_sample (from TD-005)
- [ ] Contains validated rgb_to_hsv/hsv_to_rgb (from TD-006)
- [ ] Version header included
- [ ] All utility functions tested

## Files

- `touchdesigner/glsl/common/tdCommon.glsl` (create)

## Contents

```glsl
// tdCommon.glsl - Shared utilities for SevenRad TouchDesigner effects
// Version: 0.1

#define TD_SHADERS_VERSION 0.1

// PCG Random Number Generator
uint pcg_hash(uint input_state) { ... }
float rand_float(ivec2 pos, int seed) { ... }
float rand_gaussian(ivec2 pos, int seed, float sigma) { ... }

// Bilinear Sampling
vec4 bilinear_sample(sampler2D tex, vec2 uv, ivec2 size) { ... }

// Color Space Conversion
vec3 rgb_to_hsv(vec3 rgb) { ... }
vec3 hsv_to_rgb(vec3 hsv) { ... }

// Utility Functions
float luminance(vec3 rgb) { ... }
vec3 saturate(vec3 x) { ... }
vec4 saturate(vec4 x) { ... }

// Temporal
int getEffectiveSeed(int uSeed, int uAnimateNoise, float uTime) { ... }
```

## Notes

- This is the foundation for all effects
- Functions must match Phase 2 validated implementations
- Update version when utilities change

## References

- [02-EFFECTS-AND-DEMOS.md](../../02-EFFECTS-AND-DEMOS.md) - Common GLSL Utilities section
