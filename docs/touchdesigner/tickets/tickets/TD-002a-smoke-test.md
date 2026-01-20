# TD-002a: Minimal Passthrough Smoke Test

---
id: TD-002a
status: pending
priority: high
phase: 1
depends_on: [TD-001, TD-002]
blocks: [TD-004, TD-005, TD-006]
---

## Description

Create a minimal smoke test that verifies basic GLSL shader compilation and execution in TouchDesigner before running complex algorithm validation.

**Added via RF-003**: Phase 2 was jumping directly to complex PCG hash testing without verifying basic TD shader compilation works at all.

## Acceptance Criteria

- [ ] Minimal passthrough shader created
- [ ] TD project renders solid color output
- [ ] Shader compiles in both glslangValidator AND actual TD
- [ ] Test documented as prerequisite for Phase 2 tickets
- [ ] Local test script created for developers

## Files

- `touchdesigner/glsl/test_fixtures/passthrough_smoke.frag` (create)
- `touchdesigner/fixtures/projects/smoke_test.toe` (create)
- `touchdesigner/scripts/run_smoke_test.py` (create)

## Implementation

### Smoke Test Shader
```glsl
// passthrough_smoke.frag
// Minimal shader to verify TD GLSL pipeline works

void main() {
    // Simply output solid red - proves shader runs
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
```

### Validation Steps
1. Compile with glslangValidator (CI)
2. Compile with spirv-cross to Metal (CI)
3. Run in TouchDesigner locally (manual)
4. Verify output is solid red

## Success Criteria

If this test fails:
- TD installation is broken
- Preamble extraction (TD-001) is wrong
- CI pipeline is misconfigured

Fix these before proceeding to Phase 2.

## References

- RF-003 red team finding
- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md)
