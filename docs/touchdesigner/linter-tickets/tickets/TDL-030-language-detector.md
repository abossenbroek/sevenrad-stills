# TDL-030: Language Detector

---
id: TDL-030
status: pending
priority: high
phase: 3
depends_on: [TDL-013]
blocks: [TDL-031, TDL-032]
---

## Problem Statement

.text files can contain GLSL shaders, Python scripts, or plain text. Before validating, we must detect which language a file contains. Misclassification leads to incorrect validation (running GLSL rules on Python code).

## Acceptance Criteria

- [ ] Detects GLSL with high accuracy (vec2/3/4, uniform, sampler2D, etc.)
- [ ] Detects Python with high accuracy (def, import, op(), me., etc.)
- [ ] Falls back to UNKNOWN for ambiguous content
- [ ] Uses context from parent .n file if available (DAT type)
- [ ] Handles .text file header (version number line)
- [ ] Test cases for clear GLSL, clear Python, and ambiguous

## Files to Create

```
td_linter/
├── embedded/
│   ├── __init__.py
│   └── language_detector.py
└── tests/
    └── test_language_detector.py
```

## Research Pointers

### .text File Format

.text files have a header:
```
2
*                  [optional metadata]
[actual content]
```

First line is version number. Skip it before analysis.

### Context-Based Detection

The parent DAT type provides strong hints:
- `execute` DAT -> Python
- `script` DAT -> Python
- `callbacks` DAT -> Python
- Text DAT referenced by GLSL TOP -> GLSL

How to get context? Parse the .n file that owns this .text, check the operator type.

### Content Heuristics

**GLSL indicators** (high confidence):
- `vec2`, `vec3`, `vec4`, `mat2`, `mat3`, `mat4`
- `uniform`, `varying`, `attribute`
- `sampler2D`, `sampler3D`, `samplerCube`
- `gl_FragColor`, `fragColor`
- `void main()`
- TD-specific: `TDOutputSwizzle`, `sTD2DInputs`

**Python indicators** (high confidence):
- `def functionname()`
- `import something`
- `from x import y`
- TD-specific: `op(`, `me.`, `project.`, `absTime`

### Scoring Algorithm

```python
def detect_language(content: str, context: dict) -> Language:
    # Context wins if available
    if context.get('dat_type') in ('execute', 'script', 'callbacks'):
        return Language.PYTHON

    # Content scoring
    glsl_score = count_matches(content, GLSL_PATTERNS)
    python_score = count_matches(content, PYTHON_PATTERNS)

    if glsl_score > python_score + THRESHOLD:
        return Language.GLSL
    if python_score > glsl_score + THRESHOLD:
        return Language.PYTHON
    return Language.UNKNOWN
```

### Pattern Design

Use regex patterns that avoid false positives:
- `r'\bvec[234]\b'` - word boundary prevents matching "vector"
- `r'\bdef\s+\w+\s*\('` - function definition pattern

### Threshold Tuning

How much score difference to require? Consider:
- Too low: Misclassifies mixed/ambiguous files
- Too high: Too many UNKNOWN results

Tune against sample corpus.

### Edge Cases

- Empty files
- Files with only comments
- Files with both Python-like and GLSL-like syntax
- Very short files (not enough signal)

## Test Strategy

| Test | Expected |
|------|----------|
| Pure GLSL shader | GLSL |
| Pure Python script | PYTHON |
| Mixed/ambiguous | UNKNOWN |
| Execute DAT context | PYTHON (via context) |
| Empty file | UNKNOWN |

## Definition of Done

All acceptance criteria checked. Detector correctly classifies sample corpus files.
