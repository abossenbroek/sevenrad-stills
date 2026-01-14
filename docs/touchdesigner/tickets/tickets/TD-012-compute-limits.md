# TD-012: Compute Shader Resolution Testing

---
id: TD-012
status: pending
priority: medium
phase: 5
depends_on: [TD-002]
blocks: []
---

## Description

Test compute shaders at various resolutions to document Apple Silicon limits.

**Extended scope (RF-008)**: Includes baseline capture subtask for performance regression detection.

## Acceptance Criteria

- [ ] `tests/test_compute_limits.py` created
- [ ] Tests pass at 1080p and 4K
- [ ] Tests documented at 8K (may fail)
- [ ] Limits documented
- [ ] **Performance baselines captured (RF-008)**

## Resolution Test Matrix

| Resolution | Workgroups (16x16) | Invocations | Target |
|------------|-------------------|-------------|--------|
| 1920x1080 | 120 x 68 | 2M | MUST pass |
| 3840x2160 | 240 x 135 | 8M | MUST pass |
| 7680x4320 | 480 x 270 | 33M | SHOULD pass |
| 8192x8192 | 512 x 512 | 67M | MAY pass |

## Sub-tasks

- [ ] TD-012a: Capture initial performance baselines (RF-008)
- [ ] TD-012b: Test resolution matrix
- [ ] TD-012c: Document limits

## Baseline Capture (RF-008)

First subtask captures baselines:

```python
# touchdesigner/scripts/capture_baseline.py
baselines = {}
for effect in COMPUTE_EFFECTS:
    result = benchmark_effect(effect, resolution=(1920, 1080))
    baselines[effect] = {
        "resolution": "1920x1080",
        "avg_ms": result["avg_ms"],
        "hardware": "M1 Max",
        "td_version": "2022.20000"
    }

with open("perf_baseline.json", "w") as f:
    json.dump(baselines, f, indent=2)
```

## Files

- `tests/test_compute_limits.py` (create)
- `touchdesigner/fixtures/perf_baseline.json` (create)

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 5.1
- RF-008 red team finding
