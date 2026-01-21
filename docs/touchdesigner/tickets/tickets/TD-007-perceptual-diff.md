# TD-007: Replace MD5 with Perceptual Diff

---
id: TD-007
status: pending
priority: high
phase: 3
depends_on: [TD-003]
blocks: [TD-011]
---

## Description

Implement SSIM-based perceptual comparison to replace fragile MD5 hash comparison for render tests.

**Two-tier fixtures (RF-004)**: Create preliminary fixtures now, regenerate after Phase 2 confirms algorithm behavior.

## Acceptance Criteria

- [ ] `touchdesigner/scripts/perceptual_diff.py` created
- [ ] `ComparisonResult` dataclass with metrics
- [ ] SSIM threshold configurable (default 0.99)
- [ ] Max pixel diff configurable (default 2)
- [ ] pytest fixture `assert_renders_match` created
- [ ] scikit-image added to dev dependencies
- [ ] Provisional fixtures marked for regeneration after Phase 2

## Files

- `touchdesigner/scripts/perceptual_diff.py` (create)
- `tests/conftest.py` (modify)
- `pyproject.toml` (add scikit-image)

## Implementation

```python
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    passed: bool
    ssim_score: float
    max_pixel_diff: int
    pixels_exceeding_threshold: int
    provisional: bool = False  # RF-004: mark if needs regeneration

def compare_images(actual_path, expected_path,
                   ssim_threshold=0.99, max_pixel_diff=2):
    # ... implementation
```

## Fixture Regeneration (RF-004)

After Phase 2 completes:
1. Check if PCG hash diverged from Taichi
2. If yes, regenerate all noise-based fixtures
3. Update `provisional: false` in fixture metadata

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 3.1
- RF-004 red team finding
