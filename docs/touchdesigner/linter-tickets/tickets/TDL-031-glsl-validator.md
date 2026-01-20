# TDL-031: GLSL Validator Integration

---
id: TDL-031
status: pending
priority: critical
phase: 3
depends_on: [TDL-030]
blocks: []
---

## Problem Statement

GLSL shaders embedded in .text files should be syntax-checked before toecollapse. TouchDesigner injects a preamble with uniforms and functions, so we can't validate the shader as-is. We need a preamble-agnostic approach that catches syntax errors without false positives for undefined TD-provided symbols.

## Acceptance Criteria

- [ ] Integration with glslangValidator (external tool)
- [ ] Minimal preamble for syntax validation (not full TD preamble)
- [ ] Filters "undefined uniform" warnings (TD injects these)
- [ ] Reports syntax errors with adjusted line numbers
- [ ] Detects forbidden `#version` directive (TD injects this)
- [ ] Handles both fragment and compute shaders
- [ ] Test cases with valid and invalid GLSL

## Files to Create

```
td_linter/
├── embedded/
│   └── glsl_validator.py
├── reference/
│   ├── minimal_fragment_preamble.glsl
│   └── minimal_compute_preamble.glsl
└── tests/
    └── test_glsl_validator.py
```

## Research Pointers

### glslangValidator Tool

- Part of Vulkan SDK or standalone: https://github.com/KhronosGroup/glslang
- Install: `brew install glslang` (macOS), `apt install glslang-tools` (Linux)
- Usage: `glslangValidator -S frag shader.frag`

### The Preamble Problem (AG-002)

TouchDesigner injects:
- `#version 330` (or similar)
- Uniform declarations (`sTD2DInputs`, `uTDOutputInfo`)
- Function implementations (`TDOutputSwizzle`)

We don't have the exact preamble. It may vary by TD version and operator settings.

**Solution**: Use minimal preamble, filter undefined warnings.

### Minimal Preamble Design

Fragment shader:
```glsl
#version 330 core
uniform sampler2D sTD2DInputs[8];
in vec2 vUV;
layout(location = 0) out vec4 fragColor;
vec4 TDOutputSwizzle(vec4 c) { return c; }
```

Compute shader:
```glsl
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
// ... minimal declarations
```

### Warning Filtering

Filter glslang output for:
- "undefined uniform" - TD provides these
- "undeclared identifier" - May be TD function
- "use of undeclared" - Same reason

Keep:
- Syntax errors
- Type errors
- Missing main function

### Line Number Adjustment

Preamble adds lines. If error is on line 15 and preamble is 10 lines, actual error is on line 5 of user code.

```python
adjusted_line = reported_line - preamble_lines
```

### Forbidden Patterns

Report error for:
- `#version` in user shader (TD injects this)
- `#extension` without care (may conflict)

### Output Parsing

glslangValidator output format:
```
ERROR: 0:15: 'foo' : undeclared identifier
ERROR: 0:20: syntax error
```

Parse with regex: `r'ERROR:\s*\d+:(\d+):\s*(.+)'`

### Integration with validate_glsl.py

The project already has `validate_glsl.py`. Consider:
- Wrap existing script
- Or incorporate its logic into linter

Check `docs/touchdesigner/scripts/validate_glsl.py` for existing implementation.

## Test Strategy

| Test | Expected |
|------|----------|
| Valid TD shader | No errors |
| Syntax error | ERROR reported |
| Has #version | ERROR (forbidden) |
| Uses undefined TD uniform | Filtered (no error) |

## Limitations to Document

Per AG-002, this validator cannot:
- Verify TD-specific uniforms are correct
- Check TD function signatures
- Catch runtime errors dependent on TD version

Recommend runtime testing in TD for full validation.

## Definition of Done

All acceptance criteria checked. GLSL syntax errors caught, TD-specific warnings filtered.
