# TD-027a: buffer_corruption C++ TOP Fallback

---
id: TD-027a
status: dormant
priority: medium
phase: 5
depends_on: [TD-027]
blocks: []
---

## Description

C++ TOP implementation for buffer_corruption effect if GLSL compute shader fails.

**Status: DORMANT** - Activate only if TD-027 (GLSL compute) tests fail.

## Activation Trigger

Activate this ticket if:
- TD-027 GLSL compute shader fails on Metal
- XOR mode cannot be implemented in GLSL
- MoltenVK translation produces incorrect results

## Acceptance Criteria (if activated)

- [ ] C++ TOP project created using TD SDK
- [ ] Implements all corruption modes (shift, swap, zero, XOR)
- [ ] Performance matches or exceeds GLSL target
- [ ] Packaged as .tox operator

## Why C++ Is Pre-Approved

- XOR mode requires bitwise operations on float memory
- GLSL lacks `reinterpret_cast` equivalent
- C++ provides direct memory manipulation

## XOR Mode Implementation

```cpp
void applyXorCorruption(float* pixels, int width, int height, uint32_t mask) {
    for (int i = 0; i < width * height * 4; i++) {
        // Reinterpret float as uint32_t, XOR, reinterpret back
        uint32_t* asInt = reinterpret_cast<uint32_t*>(&pixels[i]);
        *asInt ^= mask;
    }
}
```

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 5.2
- RF-007 red team finding
- [TouchDesigner C++ TOP docs](https://docs.derivative.ca/Write_a_CPlusPlus_TOP)
