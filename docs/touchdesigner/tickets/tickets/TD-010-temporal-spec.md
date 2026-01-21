# TD-010: Define Effect Temporal Specifications

---
id: TD-010
status: complete
priority: high
phase: 4
depends_on: []
blocks: [TD-011]
---

## Description

Document temporal behavior (static vs animated seed) for all 14 effects.

## Status: COMPLETE

This ticket is already complete. The temporal behavior table is documented in:
- `docs/touchdesigner/02-EFFECTS-AND-DEMOS.md` (Temporal Behavior Per Effect section)

## Deliverables (Completed)

- [x] Temporal behavior table for all 14 effects
- [x] Seed handling convention documented
- [x] `getEffectiveSeed()` GLSL pattern documented
- [x] TD Python time access corrected (`me.time.frame` not `absTime.frame`)

## Temporal Behavior Summary

| Effect | Seed Behavior | Animate Parameter |
|--------|---------------|-------------------|
| saturation | N/A | - |
| chromatic_aberration | N/A | - |
| noise | Configurable | `Animatenoise` |
| salt_pepper | Configurable | `Animatenoise` |
| corduroy | Static | - |
| gaussian_blur | N/A | - |
| buffer_corruption | Configurable | `Animateglitch` |
| ... | ... | ... |

## References

- [02-EFFECTS-AND-DEMOS.md](../../02-EFFECTS-AND-DEMOS.md) - Video/Temporal Considerations
