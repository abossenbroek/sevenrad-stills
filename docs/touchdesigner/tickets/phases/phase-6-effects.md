# Phase 6: Effect Implementation

---
complete: false
---

## Overview

Implement all 14 Taichi effects as GLSL shaders and package as .tox operators. This phase is BLOCKED until all gates G1-G7 pass.

## Gate Criteria

**Entry Gate**: All of G1-G7 must pass before starting this phase.

| Gate | Status |
|------|--------|
| G1 | macOS CI operational | [ ] |
| G2 | TD preamble extracted | [ ] |
| G3 | PCG hash validated | [ ] |
| G4 | Bilinear aligned | [ ] |
| G5 | Perceptual diff active | [ ] |
| G6 | MoltenVK validation in CI | [ ] |
| G7 | Temporal behavior defined | [ ] |

## Tickets

| Ticket | Title | Complexity | Status |
|--------|-------|------------|--------|
| [TD-015](../tickets/TD-015-tdcommon-glsl.md) | Create tdCommon.glsl | Foundation | pending |
| [TD-016](../tickets/TD-016-saturation.md) | saturation effect | Simple | pending |
| [TD-017](../tickets/TD-017-chromatic-aberration.md) | chromatic_aberration effect | Simple | pending |
| [TD-018](../tickets/TD-018-noise.md) | noise effect | Simple | pending |
| [TD-019](../tickets/TD-019-salt-pepper.md) | salt_pepper effect | Simple | pending |
| [TD-020](../tickets/TD-020-corduroy.md) | corduroy effect | Medium | pending |
| [TD-021](../tickets/TD-021-downscale.md) | downscale effect | Medium | pending |
| [TD-022](../tickets/TD-022-gaussian-blur.md) | gaussian_blur effect | Medium | pending |
| [TD-023](../tickets/TD-023-circular-blur.md) | circular_blur effect | Medium | pending |
| [TD-024](../tickets/TD-024-motion-blur.md) | motion_blur effect | Medium | pending |
| [TD-025](../tickets/TD-025-slc-off.md) | slc_off effect | Medium | pending |
| [TD-026](../tickets/TD-026-band-swap.md) | band_swap effect | Medium | pending |
| [TD-027](../tickets/TD-027-buffer-corruption.md) | buffer_corruption effect | Complex | pending |
| [TD-028](../tickets/TD-028-bayer-filter.md) | bayer_filter effect | Complex | pending |

## Implementation Order

```
TD-015 (tdCommon.glsl)
    │
    ├──> Simple: TD-016, TD-017, TD-018, TD-019
    │
    ├──> Medium Fragment: TD-020, TD-021, TD-022, TD-023, TD-024
    │
    ├──> Medium Compute: TD-025, TD-026
    │
    └──> Complex: TD-027, TD-028
```

## Notes

- Each effect requires .tox packaging with help and demo
- Video-first demos for all effects
- Run local TD tests before PR (CI is SPIRV-only)

## Completion Checklist

- [ ] All gates G1-G7 verified
- [ ] TD-015 complete (tdCommon.glsl)
- [ ] All 14 effects complete
- [ ] All .tox operators packaged
- [ ] Master demo project (sr_demo.toe) complete
- [ ] Phase marked complete: true
