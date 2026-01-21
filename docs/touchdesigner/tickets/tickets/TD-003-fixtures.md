# TD-003: Create Validation Fixture Infrastructure

---
id: TD-003
status: pending
priority: high
phase: 1
depends_on: []
blocks: [TD-004, TD-005, TD-006, TD-011]
---

## Description

Create directory structure and test assets for algorithm validation including images, test vectors, TD projects, and video clips.

## Acceptance Criteria

- [ ] `touchdesigner/fixtures/` directory structure created
- [ ] Checkerboard 64x64 PNG for bilinear tests
- [ ] Gradient 64x64 PNG for color tests
- [ ] Solid color test images (red, green, blue, white, black, gray50)
- [ ] `pcg_hash_values.json` with Taichi-generated reference hashes
- [ ] `bilinear_samples.json` with expected sample values
- [ ] `hsv_conversions.json` with RGB↔HSV test vectors
- [ ] Video clips sourced (RF-005: stock footage)

## Files

```
touchdesigner/fixtures/
├── inputs/
│   ├── checkerboard_64x64.png
│   ├── gradient_64x64.png
│   └── solid_colors/
│       ├── red.png
│       ├── green.png
│       ├── blue.png
│       ├── white.png
│       ├── black.png
│       └── gray50.png
├── expected/
│   ├── pcg_hash_values.json
│   ├── bilinear_samples.json
│   └── hsv_conversions.json
├── videos/
│   ├── LICENSE.md              # CC0 attribution
│   ├── static_gradient.mov
│   ├── moving_gradient.mov
│   └── color_bars_60fps.mov
└── projects/
    ├── pcg_hash_test.toe
    ├── bilinear_test.toe
    └── smoke_test.toe
```

## Sub-tasks

- [ ] TD-003a: Create directory structure
- [ ] TD-003b: Generate test images (Python script)
- [ ] TD-003c: Generate PCG hash reference values from Taichi
- [ ] TD-003d: Generate bilinear sample reference values
- [ ] TD-003e: Create HSV conversion test vectors
- [ ] TD-003f: Source CC0 video clips (RF-005 fix)
- [ ] TD-003g: Create minimal TD project templates

## Video Sourcing (RF-005)

Source CC0/public domain video clips from:
- Pexels (pexels.com)
- Pixabay (pixabay.com)
- Coverr (coverr.co)

Requirements:
- 1080p or higher
- ~10 seconds looping
- Varied motion patterns (static, slow pan, fast motion)

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 1.3
