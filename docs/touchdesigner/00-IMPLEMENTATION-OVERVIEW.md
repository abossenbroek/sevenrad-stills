# TouchDesigner Migration: Implementation Overview

Migrate 14 Taichi GPU effects to TouchDesigner operators using **GLSL 330 core** (fragment shaders) with compute shaders (GLSL 430+) for tile-based effects.

## Related Documents

- [01-LINTING-INFRASTRUCTURE.md](01-LINTING-INFRASTRUCTURE.md) - Linting tools, CI/CD, editor integration
- [02-EFFECTS-AND-DEMOS.md](02-EFFECTS-AND-DEMOS.md) - GLSL effects, .tox structure, demo system
- [03-REMEDIATION-PLAN.md](03-REMEDIATION-PLAN.md) - Pre-implementation validation and infrastructure hardening

---

## Project Goal

Robust linting infrastructure provides fast feedback before TouchDesigner testing, catching LLM hallucinations and human coding errors early.

| Attribute | Value |
|-----------|-------|
| **Platform** | macOS Apple Silicon (M1/M2/M3) via MoltenVK/Metal |
| **TouchDesigner** | 2022.20000+ (Vulkan backend) |
| **GLSL Versions** | 330 core (fragments), 430+ (compute only) |
| **Priority** | Pure GLSL → C++ TOP → Python (fallback) |

---

## Key Architectural Decisions

1. **GLSL Version**: 330 core for fragment shaders (maximum Apple compatibility); 430+ only for compute shaders
2. **Shared Utilities**: Single `tdCommon.glsl` include file for all effects (random, sampling, color)
3. **Linting First**: Complete linting infrastructure before migrating any effects
4. **Prototype Before C++**: Test tile effects (band_swap, buffer_corruption) in pure GLSL before committing to C++ TOPs
5. **Unit Renders**: Store expected 64×64 PNG outputs for deterministic regression testing
6. **GPL-3.0 Isolation**: Run glsl_analyzer only in CI containers to avoid license contamination

---

## Implementation Phases

```
Phase 1: Linting Infrastructure     [See 01-LINTING-INFRASTRUCTURE.md]
   ├── glsl_analyzer, glslangValidator, clang-tidy
   ├── Pre-commit hooks, CI/CD workflow
   └── Test fixtures (valid/invalid shaders)

Phase 2: Common GLSL Utilities      [See 02-EFFECTS-AND-DEMOS.md]
   ├── tdCommon.glsl (random, sampling, color)
   └── Versioned shader headers

Phase 3: Effect Migration           [See 02-EFFECTS-AND-DEMOS.md]
   ├── Simple effects (saturation, noise, chromatic_aberration)
   ├── Multi-pass effects (gaussian_blur, bayer_filter)
   └── Compute shaders (band_swap, buffer_corruption, slc_off)

Phase 4: Demo System                [See 02-EFFECTS-AND-DEMOS.md]
   ├── Per-operator .tox with help
   ├── Master demo project (sr_demo.toe)
   └── Effect chain presets
```

---

## Effect Summary

| Effect | Shader | Passes | Complexity |
|--------|--------|--------|------------|
| saturation | Fragment | 1 | Simple |
| chromatic_aberration | Fragment | 1 | Simple |
| noise (3 modes) | Fragment | 1 | Simple |
| salt_pepper | Fragment | 1 | Simple |
| corduroy | Fragment | 1 | Medium |
| downscale | Fragment | 1-2 | Medium |
| gaussian_blur | Fragment | 2 (H+V) | Medium |
| circular_blur | Fragment | 1 | Medium |
| motion_blur | Fragment | 1 | Medium |
| slc_off | Compute | 1 | Medium |
| band_swap | Compute | 1 | Medium |
| buffer_corruption | Compute | 1 | Complex |
| bayer_filter | Fragment | 2 | Complex |

---

## Success Criteria

1. All 14 effects working in TouchDesigner on macOS Apple Silicon
2. Linting catches syntax/semantic errors before TD testing
3. Each operator has comprehensive demo with help (video-first)
4. CI validates all shaders on every PR (including macOS Metal validation)
5. Master demo project showcases all effects with chain builder
6. Pure GLSL implementation for all effects (C++ only if absolutely necessary)
7. Unit render tests pass with perceptual diff comparison (SSIM >= 0.99)

---

## Gate Criteria (Before Phase 2)

> **IMPORTANT**: Do not proceed to effect implementation until all gates pass.
> See [03-REMEDIATION-PLAN.md](03-REMEDIATION-PLAN.md) for full details.

| Gate | Requirement | Validation |
|------|-------------|------------|
| G1 | macOS CI runner operational | Renders test shader, uploads artifact |
| G2 | TD preamble extracted | validate_glsl.py uses real TD uniforms |
| G3 | PCG hash validated | Bit-identical to Taichi OR divergence documented |
| G4 | Bilinear sampling aligned | Within 1/255 tolerance at test positions |
| G5 | Perceptual diff active | MD5 replaced with SSIM in all tests |
| G6 | MoltenVK validation in CI | SPIRV-Cross Metal compilation passes |
| G7 | Temporal behavior defined | All 14 effects have seed/animation spec |

---

## GLSL Feasibility Checklist

Before attempting pure GLSL for any effect, verify:

- [ ] No random-access writes required (fragment shaders read-only)
- [ ] No inter-pixel communication (each pixel independent)
- [ ] No recursive algorithms
- [ ] Loops bounded to <1024 iterations
- [ ] No bitwise float manipulation (reinterpret_cast equivalent)
- [ ] No geometry shader requirements (Metal limitation)

**Pre-Approved C++ Exceptions:**
- `buffer_corruption` - XOR mode needs bitwise float ops
- `band_swap` - Try GLSL compute first, C++ fallback approved

---

## Platform Notes (macOS Apple Silicon)

- **Graphics API**: Vulkan via MoltenVK → Metal
- **Compute Shaders**: Fully supported on Apple Silicon
- **Geometry Shaders**: NOT supported (Metal limitation)
- **Avoid**: `GL_TEXTURE_RECTANGLE`, implicit LOD functions, non-constant loops >1024 iterations

---

## Directory Structure

```
touchdesigner/
├── glsl/
│   ├── common/
│   │   └── tdCommon.glsl          # Shared utilities
│   ├── effects/
│   │   └── ... (14 effects)
│   └── test_fixtures/
│       ├── valid/
│       └── invalid/
├── cpp/                           # C++ TOPs (only if needed)
├── tox/
│   ├── operators/                 # Individual .tox files
│   └── demo/
│       └── sr_demo.toe            # Master demo project
└── scripts/
    └── validate_glsl.py           # GLSL validation wrapper
```

---

## Critical Files to Reference

### Taichi Source (algorithms to port)
- `src/sevenrad_stills/operations/taichi_kernels/random.py` - PCG RNG
- `src/sevenrad_stills/operations/taichi_kernels/sampling.py` - Bilinear sampling
- `src/sevenrad_stills/operations/saturation_taichi.py` - HSV conversion
- `src/sevenrad_stills/operations/blur_gaussian_taichi.py` - Separable convolution
- `src/sevenrad_stills/operations/bayer_filter_taichi.py` - Mosaic/demosaic

### Existing Infrastructure
- `.pre-commit-config.yaml` - Add GLSL/C++ hooks
- `.github/workflows/` - Add touchdesigner-lint.yml
- `pyproject.toml` - Dev dependencies

---

## Sources

- [TouchDesigner 2022.20000 Release Notes](https://docs.derivative.ca/Release_Notes/2022.20000)
- [TouchDesigner GLSL TOP](https://docs.derivative.ca/GLSL_TOP)
- [TouchDesigner Vulkan Guide](https://derivative.ca/UserGuide/Vulkan)
- [glsl_analyzer GitHub](https://github.com/nolanderc/glsl_analyzer)
