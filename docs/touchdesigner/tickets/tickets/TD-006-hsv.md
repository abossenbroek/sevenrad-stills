# TD-006: HSV Conversion Validation

---
id: TD-006
status: pending
priority: high
phase: 2
depends_on: [TD-001, TD-002a, TD-003]
blocks: [TD-015, TD-016]
---

## Description

Verify RGB↔HSV conversion produces correct results and roundtrips without error accumulation.

## Acceptance Criteria

- [ ] `hsv_test.frag` shader created
- [ ] `tests/test_hsv_conversion.py` implemented
- [ ] All test vectors pass (pure colors, grays, edge cases)
- [ ] Roundtrip RGB→HSV→RGB error < 1/255 per channel

## Test Vectors

```json
{
  "conversions": [
    {"rgb": [1.0, 0.0, 0.0], "hsv": [0.0, 1.0, 1.0], "name": "pure red"},
    {"rgb": [0.0, 1.0, 0.0], "hsv": [0.333, 1.0, 1.0], "name": "pure green"},
    {"rgb": [0.0, 0.0, 1.0], "hsv": [0.667, 1.0, 1.0], "name": "pure blue"},
    {"rgb": [1.0, 1.0, 1.0], "hsv": [0.0, 0.0, 1.0], "name": "white"},
    {"rgb": [0.0, 0.0, 0.0], "hsv": [0.0, 0.0, 0.0], "name": "black"},
    {"rgb": [0.5, 0.5, 0.5], "hsv": [0.0, 0.0, 0.5], "name": "gray 50%"},
    {"rgb": [1.0, 0.5, 0.0], "hsv": [0.083, 1.0, 1.0], "name": "orange"},
    {"rgb": [0.5, 0.0, 0.5], "hsv": [0.833, 1.0, 0.5], "name": "purple"}
  ],
  "tolerance": 0.004
}
```

## Files

- `touchdesigner/glsl/test_fixtures/hsv_test.frag` (create)
- `tests/test_hsv_conversion.py` (create)

## Edge Cases

- Hue near 0/360 boundary (red)
- Very low saturation (near-gray)
- Very high/low value (near-white/black)

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 2.3
