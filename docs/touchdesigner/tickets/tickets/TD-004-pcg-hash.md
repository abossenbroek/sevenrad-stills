# TD-004: PCG Hash Equivalence Testing

---
id: TD-004
status: pending
priority: critical
phase: 2
depends_on: [TD-001, TD-002, TD-002a, TD-003]
blocks: [TD-015, TD-018, TD-019, TD-027]
---

## Description

Verify GLSL PCG hash produces identical bit patterns to Taichi implementation. If divergence found, implement fallback strategy.

**Critical dependency (RF-001)**: Explicitly depends on TD-001 (preamble) and TD-002a (smoke test), not just Phase 1 generally.

## Acceptance Criteria

- [ ] `pcg_hash_test.frag` shader created
- [ ] `tests/test_pcg_equivalence.py` implemented
- [ ] Tests pass for all seed values OR divergence documented
- [ ] Fallback strategy implemented if divergence found

## Test Seeds

```python
TEST_SEEDS = [0, 1, 42, 65536, 2147483647, 2147483648, 4294967295]
```

## Files

- `touchdesigner/glsl/test_fixtures/pcg_hash_test.frag` (create)
- `touchdesigner/fixtures/projects/pcg_hash_test.toe` (create)
- `tests/test_pcg_equivalence.py` (create)

## Implementation

### Test Shader
```glsl
// Outputs 32-bit hash as RGBA (8 bits per channel)
uniform int uTestSeed;

void main() {
    uint hash = pcg_hash(uint(uTestSeed));
    fragColor = vec4(
        float((hash >> 24u) & 0xFFu) / 255.0,
        float((hash >> 16u) & 0xFFu) / 255.0,
        float((hash >> 8u) & 0xFFu) / 255.0,
        float(hash & 0xFFu) / 255.0
    );
}
```

## Fallback Strategy

If GLSL PCG differs from Taichi:

| Option | Action |
|--------|--------|
| A | Adjust GLSL shifts to match Taichi (explicit uint casts) |
| B | Accept divergence, generate GLSL-specific fixtures |
| C | Switch to xorshift128+ RNG (known portable) |

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 2.1
