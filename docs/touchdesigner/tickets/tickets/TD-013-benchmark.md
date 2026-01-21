# TD-013: Performance Benchmarking Framework

---
id: TD-013
status: pending
priority: medium
phase: 5
depends_on: [TD-002]
blocks: []
---

## Description

Create benchmarking script to measure effect performance and establish baselines.

## Acceptance Criteria

- [ ] `touchdesigner/scripts/benchmark.py` created
- [ ] Measures avg/min/max frame time
- [ ] Reports FPS and 60fps compliance
- [ ] Baseline JSON output for CI comparison
- [ ] CI step warns if >20% slower than baseline

## Performance Targets (1080p)

| Effect | Target | Notes |
|--------|--------|-------|
| saturation | <2ms | Simple |
| chromatic_aberration | <2ms | Simple |
| noise | <2ms | PCG is fast |
| salt_pepper | <2ms | Simple |
| corduroy | <2ms | Simple |
| downscale | <2ms | Simple |
| gaussian_blur | <8ms | Two passes |
| circular_blur | <8ms | Many samples |
| motion_blur | <4ms | Line kernel |
| bayer_filter | <8ms | Two passes |
| slc_off | <4ms | Compute (RF-006 fix) |
| band_swap | <4ms | Compute (RF-006 fix) |
| buffer_corruption | <4ms | Compute |

**60fps Budget**: 16.67ms per frame

## Files

- `touchdesigner/scripts/benchmark.py` (create)

## Implementation

```python
def benchmark_effect(effect: str, params: dict,
                     resolution: tuple = (1920, 1080),
                     num_frames: int = 100) -> dict:
    import time

    # Warm-up
    for _ in range(10):
        render_frame(effect, params, resolution)

    # Benchmark
    times = []
    for _ in range(num_frames):
        start = time.perf_counter()
        render_frame(effect, params, resolution)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg = sum(times) / len(times)
    return {
        "effect": effect,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "avg_ms": round(avg, 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "fps": round(1000 / avg, 1),
        "meets_60fps": avg < 16.67
    }
```

## Baseline JSON Schema (RF-012 fix)

Example `perf_baseline.json` output:

```json
{
  "version": "1.0",
  "hardware": "Apple M1 Max",
  "td_version": "2022.20000",
  "captured_at": "2024-01-15T10:30:00Z",
  "baselines": {
    "saturation": {
      "resolution": "1920x1080",
      "avg_ms": 1.2,
      "min_ms": 1.0,
      "max_ms": 1.8,
      "fps": 833.3,
      "meets_60fps": true
    },
    "gaussian_blur": {
      "resolution": "1920x1080",
      "avg_ms": 6.5,
      "min_ms": 5.8,
      "max_ms": 7.2,
      "fps": 153.8,
      "meets_60fps": true
    }
  }
}
```

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 5.3
