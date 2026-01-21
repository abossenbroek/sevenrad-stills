# TD-001: Extract TouchDesigner GLSL Preamble

---
id: TD-001
status: pending
priority: critical
phase: 1
depends_on: []
blocks: [TD-002a, TD-004, TD-005, TD-006]
---

## Description

Extract the actual GLSL preamble that TouchDesigner injects into shaders, replacing the synthetic/guessed version in `validate_glsl.py`.

The current preamble is hand-crafted and may not match actual TD runtime, causing validation to pass for shaders that fail in TD (or vice versa).

## Acceptance Criteria

- [ ] Minimal passthrough shader created in TD 2022.20000+
- [ ] All injected uniforms documented (names, types, purposes)
- [x] `touchdesigner/reference/td_preamble_2022.glsl` created (synthetic - needs real values)
- [x] `touchdesigner/scripts/validate_glsl.py` created (with synthetic TD_PREAMBLE)
- [ ] TD_PREAMBLE updated with real extracted values
- [ ] Shader that passes validation compiles in actual TD
- [ ] Version differences documented if TD 2023.x/2024.x differ

## Files

- `docs/touchdesigner/reference/td_preamble_2022.glsl` (created)
- `docs/touchdesigner/scripts/validate_glsl.py` (created)
- `docs/touchdesigner/glsl/test_fixtures/valid/` (created - 2 fixtures)
- `docs/touchdesigner/glsl/test_fixtures/invalid/` (created - 2 fixtures)

## Implementation Notes

1. Create minimal passthrough shader in TD:
   ```glsl
   void main() {
       fragColor = texture(sTD2DInputs[0], vUV.st);
   }
   ```

2. Use TD's shader inspection tools or SPIR-V analysis to extract injected code

3. Document all uniforms:
   - `sTD2DInputs[]` - Input textures
   - `uTDOutputInfo` - Output resolution info
   - `vUV` - Texture coordinates
   - Others as discovered

## Extraction Procedure

1. Open TouchDesigner 2022.20000+
2. Create a new GLSL TOP
3. Enter minimal shader: `void main() { fragColor = vec4(1.0); }`
4. Create an Info DAT and connect it to the GLSL TOP
5. In Info DAT parameters, set "Operator" to point to your GLSL TOP
6. The Info DAT will show all injected uniforms and preamble code
7. Copy the uniform declarations section
8. Also check: GLSL TOP → Right-click → View → GLSL Info for additional details

**Note:** Also consult Max/MSP jit.gl.pix implementation for cross-reference if available.

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 1.1
- [TouchDesigner GLSL TOP docs](https://docs.derivative.ca/GLSL_TOP)
