# TD-008: Add MoltenVK/Metal Validation

---
id: TD-008
status: pending
priority: high
phase: 3
depends_on: [TD-002]
blocks: []
---

## Description

Add SPIRV-Cross Metal compilation to CI pipeline to catch MoltenVK translation issues before runtime.

## Acceptance Criteria

- [ ] CI step validates GLSL→SPIRV→Metal path
- [ ] All shaders compile to Metal without errors
- [ ] MoltenVK edge case test fixtures created
- [ ] Known edge cases documented

## Files

- `.github/workflows/touchdesigner-macos.yml` (modify)
- `touchdesigner/glsl/test_fixtures/moltenvk_edge_cases.frag` (create)

## CI Implementation

```yaml
- name: Validate GLSL -> SPIRV -> Metal
  run: |
    for shader in touchdesigner/glsl/**/*.frag; do
      echo "Validating: $shader"
      glslangValidator -V -S frag "$shader" -o /tmp/shader.spv
      spirv-cross --msl /tmp/shader.spv --output /tmp/shader.metal
      xcrun -sdk macosx metal -c /tmp/shader.metal -o /dev/null
      echo "  PASS"
    done
```

## Edge Cases to Test

```glsl
// moltenvk_edge_cases.frag

// TEST 1: Dynamic array indexing
uniform int uDynamicIndex;
vec4 test_dynamic_array() {
    return texture(sTD2DInputs[uDynamicIndex], vUV);
}

// TEST 2: Integer overflow wrapping
uint test_overflow() {
    return 0xFFFFFFFFu + 1u;  // Should wrap to 0
}

// TEST 3: Bitshift on sign bit
uint test_sign_shift() {
    return 0x80000000u >> 31u;  // Should be 1
}
```

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 3.2
