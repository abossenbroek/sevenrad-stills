# TD-011: Video Test Suite

---
id: TD-011
status: pending
priority: high
phase: 4
depends_on: [TD-003, TD-007, TD-010]
blocks: []
---

## Description

Implement temporal stability tests using video input to detect flicker and animation issues.

**Extended scope (RF-006)**: Includes chain integration tests for 2-3 effect chains.

## Acceptance Criteria

- [ ] `tests/test_temporal_stability.py` created
- [ ] Static seed flicker test (SSIM > 0.9999 between frames)
- [ ] Animated seed variation test (SSIM < 0.95 between frames)
- [ ] Blur temporal coherence test
- [ ] Chain integration tests for preset chains (RF-006)
- [ ] Sample videos in fixtures (from TD-003)

## Tests

### Temporal Stability
- `test_static_noise_no_flicker` - Static seed produces identical frames
- `test_animated_noise_varies` - Animated seed produces different frames
- `test_blur_no_temporal_artifacts` - Blur on moving video is smooth

### Chain Integration (RF-006)
- `test_chain_vhs_aesthetic` - 3 effects chained correctly
- `test_chain_satellite_glitch` - 4 effects chained correctly
- `test_chain_no_alpha_corruption` - Alpha preserved through chain
- `test_chain_no_precision_loss` - No visible banding after chain

## Files

- `tests/test_temporal_stability.py` (create)

## Implementation

```python
class TestTemporalStability:
    def test_static_noise_no_flicker(self, render_video_effect):
        frames = render_video_effect("noise",
            params={"Seed": 42, "Animatenoise": 0}, num_frames=10)
        for i in range(1, len(frames)):
            ssim = compare_frames(frames[0], frames[i])
            assert ssim > 0.9999, f"Flicker detected at frame {i}"

class TestChainIntegration:
    def test_chain_vhs_aesthetic(self, render_chain):
        # Test: noise -> chromatic_aberration -> saturation
        result = render_chain(["noise", "chromatic_aberration", "saturation"])
        assert result.alpha_intact
        assert result.no_banding
```

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 4.2
- RF-006 red team finding
