# TD-002: Set Up macOS CI Runner

---
id: TD-002
status: pending
priority: critical
phase: 1
depends_on: []
blocks: [TD-002a, TD-004, TD-005, TD-008, TD-011, TD-012, TD-013]
---

## Description

Configure GitHub Actions macOS runner (M1 Apple Silicon) to validate shaders via SPIRV-Cross Metal compilation.

**Decision (RF-002)**: Use SPIRV-only CI validation. TD render tests run locally before PR submission due to licensing constraints.

## Acceptance Criteria

- [ ] `.github/workflows/touchdesigner-macos.yml` created
- [ ] Workflow runs on `macos-14` (M1 hardware)
- [ ] glslang and spirv-cross installed via brew
- [ ] GLSL→SPIRV→Metal validation pipeline working
- [ ] Local testing requirement documented in CONTRIBUTING.md
- [ ] Workflow triggers on `touchdesigner/**` path changes

## Files

- `.github/workflows/touchdesigner-macos.yml` (create)
- `CONTRIBUTING.md` (modify - add local TD testing requirement)

## Implementation

```yaml
name: TouchDesigner GLSL Validation

on:
  push:
    branches: [main, develop, feature/*]
    paths: ['touchdesigner/**']
  pull_request:
    paths: ['touchdesigner/**']

jobs:
  macos-validation:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: Install shader tools
        run: brew install glslang spirv-cross

      - name: Validate GLSL->SPIRV->Metal
        run: |
          for shader in touchdesigner/glsl/**/*.frag; do
            echo "Validating: $shader"
            glslangValidator -V -S frag "$shader" -o /tmp/shader.spv
            spirv-cross --msl /tmp/shader.spv --output /tmp/shader.metal
            xcrun -sdk macosx metal -c /tmp/shader.metal -o /dev/null
            echo "  PASS"
          done
```

## Notes

- No TD headless rendering in CI (license constraint)
- Developers must run local TD tests before submitting PRs
- CI catches Metal translation issues without TD runtime

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 1.2
