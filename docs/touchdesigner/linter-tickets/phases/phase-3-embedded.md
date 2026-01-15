# Phase 3: Embedded Code Validation

---
complete: true
---

## Overview

.text files contain embedded GLSL shaders and Python scripts. This phase integrates external validators (glslangValidator, Python AST) and handles the unique challenges of validating code that runs inside TouchDesigner's runtime.

**Philosophy**: We can't perfectly validate code that depends on a runtime we don't have. Focus on catching what we can (syntax errors), warn about what we can't (undefined TD builtins).

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G3.1 | GLSL syntax errors caught | Invalid shader flagged before toecollapse |
| G3.2 | Python syntax errors caught | SyntaxError detected pre-runtime |
| G3.3 | TD builtins not false-positived | `op()`, `me`, `absTime` not flagged |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-030](../tickets/TDL-030-language-detector.md) | Language Detector | High | pending |
| [TDL-031](../tickets/TDL-031-glsl-validator.md) | GLSL Validator Integration | Critical | pending |
| [TDL-032](../tickets/TDL-032-python-validator.md) | Python AST Validator | High | pending |
| [TDL-033](../tickets/TDL-033-td-stubs.md) | TouchDesigner Python Stubs | Medium | pending |
| [TDL-034](../tickets/TDL-034-expression-validator.md) | Parameter Expression Validator | Medium | pending |

## Dependencies

```
Phase 1 (parsers) ──> TDL-030 ──> TDL-031
                              ├──> TDL-032

TDL-032 ──> TDL-033 (stubs enhance validation)

TDL-011 (parm parser) ──> TDL-034 (expressions in .parm)
```

## Key Challenges

### GLSL Preamble Problem
TouchDesigner injects a preamble with uniforms (`sTD2DInputs`, `uTDOutputInfo`). We don't have the exact preamble, so:
- Use minimal preamble for syntax validation
- Filter "undefined uniform" warnings
- Document limitations clearly

### Python Runtime Problem
TD Python has special globals (`op`, `me`, `parent`, `absTime`). They don't exist at lint time:
- Build comprehensive builtins whitelist
- Allow config for project-specific globals
- Warning severity, not error

### Expression vs Script
.parm mode 49/17 contains Python expressions, not full scripts:
- Parse with `ast.parse(mode='eval')` not `mode='exec'`
- Different validation rules

## Completion Checklist

- [x] TDL-030 complete: GLSL vs Python detected accurately
- [x] TDL-031 complete: GLSL syntax validated
- [x] TDL-032 complete: Python syntax validated
- [x] TDL-033 complete: TD stubs reduce false positives
- [x] TDL-034 complete: Expressions extracted and checked
- [x] G3.1-G3.3 verified
- [x] Phase marked complete: true
