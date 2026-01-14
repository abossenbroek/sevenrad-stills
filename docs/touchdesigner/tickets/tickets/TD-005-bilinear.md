# TD-005: Bilinear Sampling Alignment

---
id: TD-005
status: pending
priority: high
phase: 2
depends_on: [TD-001, TD-002, TD-002a, TD-003]
blocks: [TD-015, TD-017, TD-021, TD-023, TD-024]
---

## Description

Verify GLSL bilinear sampling matches Taichi coordinate system. Test at pixel centers and fractional positions.

**Critical dependency (RF-001)**: Explicitly depends on TD-001 (preamble), not just Phase 1 generally.

## Acceptance Criteria

- [ ] `bilinear_test.frag` shader created
- [ ] `tests/test_bilinear_alignment.py` implemented
- [ ] Pixel center samples match within 0 tolerance
- [ ] Fractional samples match within 1/255 tolerance
- [ ] Coordinate offset corrected if needed

## Test Pattern

64x64 checkerboard:
- Even positions (x+y % 2 == 0): RGB(255, 0, 0) red
- Odd positions: RGB(0, 255, 0) green

## Test Positions

```python
# Pixel centers - must match exactly
PIXEL_CENTERS = [(0,0), (15,15), (31,31), (47,47), (63,63)]

# Fractional - within tolerance
FRACTIONAL = [(0.5, 0.5), (10.25, 20.75), (32.5, 32.5)]
```

## Files

- `touchdesigner/glsl/test_fixtures/bilinear_test.frag` (create)
- `tests/test_bilinear_alignment.py` (create)

## Coordinate Fix Options

If misaligned, try:

```glsl
// Current (possibly wrong):
vec2 pos = uv * vec2(size) - 0.5;

// Alternative A (OpenGL convention):
vec2 pos = uv * vec2(size);

// Alternative B (explicit half-pixel offset):
vec2 pos = (uv - 0.5/vec2(size)) * vec2(size);
```

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 2.2
