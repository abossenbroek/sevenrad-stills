# td-linter Rules Catalog

This document describes all available lint rules in td-linter.

## Rule Categories

| Code | Category | Description |
|------|----------|-------------|
| S | Syntax | File parsing and syntax validation |
| C | Connection | Operator connections and graph structure |
| T | Type | Type compatibility between operators |
| R | Reference | Operator references and dependencies |
| G | GLSL | Embedded GLSL shader validation |
| P | Python | Embedded Python expression validation |
| F | Performance | Performance optimization rules |

## Syntax Rules (S)

### S001: valid-n-syntax

**Description**: Validates `.n` file syntax.

**Severity**: error

**Fixable**: No

**Details**: Checks that `.n` (node definition) files can be parsed according to the TouchDesigner format grammar.

**Example violation**:
```
Error S001: Syntax error in node definition
  File: /project/ops/broken.n
  Line: 5
  Message: Unexpected token 'invalid'
```

---

### S002: valid-parm-syntax

**Description**: Validates `.parm` file syntax.

**Severity**: error

**Fixable**: No

**Details**: Checks that `.parm` (parameter) files follow the expected format with mode flags and values.

---

### S003: valid-toc

**Description**: Validates `.toc` manifest file.

**Severity**: error

**Fixable**: No

**Details**: Ensures the table of contents file lists valid operator paths.

## Connection Rules (C)

### C001: no-invalid-cycles

**Description**: Detects invalid operator cycles (feedback loops).

**Severity**: error

**Fixable**: No

**Details**: Cycles in the operator graph are invalid unless they contain a feedback operator (`feedback`, `feedbackchop`, `timemachine`, `delay`, `lag`). All-CHOP cycles are also valid.

**Example violation**:
```
Error C001: Invalid cycle detected
  Path: op1 → op2 → op3 → op1
  Message: Cycle does not contain feedback operator
```

---

### C002: no-dangling-inputs

**Description**: Detects inputs referencing non-existent operators.

**Severity**: warning

**Fixable**: No

**Details**: Checks that all input references in `.n` files point to operators that exist in the project.

**Example violation**:
```
Warning C002: Dangling input reference
  Path: /project/ops/myop
  Input: missing_operator
  Message: Referenced operator does not exist
```

---

### C003: no-orphan-operators

**Description**: Detects operators with no connections.

**Severity**: info

**Fixable**: No

**Details**: Identifies operators that have neither inputs nor outputs, which may indicate unused or forgotten operators.

---

### C004: consistent-connections

**Description**: Validates connection consistency.

**Severity**: warning

**Fixable**: No

**Details**: Ensures bidirectional consistency between input declarations and actual connections.

## Type Rules (T)

### T001: type-compatibility

**Description**: Validates operator type compatibility.

**Severity**: error

**Fixable**: No

**Details**: Checks that connected operators have compatible types. For example, a TOP can connect to another TOP or a MAT, but not directly to a CHOP.

**Valid connections**:
| Source | Valid Targets |
|--------|---------------|
| TOP | TOP, MAT |
| CHOP | CHOP |
| SOP | SOP, MAT |
| DAT | DAT |
| COMP | COMP |
| MAT | - (output only) |

**Example violation**:
```
Error T001: Type incompatibility
  Source: /project/ops/noise (CHOP)
  Target: /project/ops/compositor (TOP)
  Message: CHOP cannot connect directly to TOP
```

---

### T002: valid-converter-usage

**Description**: Validates converter operator usage.

**Severity**: warning

**Fixable**: No

**Details**: Checks that converter operators (chopto, sopto, tochop, etc.) are used correctly for cross-family connections.

## Reference Rules (R)

### R001: valid-operator-references

**Description**: Validates operator path references.

**Severity**: error

**Fixable**: No

**Details**: Ensures that operator references in parameters and expressions point to valid operators.

---

### R002: valid-parameter-references

**Description**: Validates parameter references.

**Severity**: warning

**Fixable**: No

**Details**: Checks that parameter references (e.g., `op('name').par.value`) refer to existing parameters.

## GLSL Rules (G)

### G001: valid-glsl-syntax

**Description**: Validates GLSL shader syntax.

**Severity**: error

**Fixable**: No

**Details**: Uses glslangValidator to check embedded GLSL code in `.text` files.

**Example violation**:
```
Error G001: GLSL syntax error
  File: /project/ops/shader.text
  Line: 15
  Message: undeclared identifier 'foo'
```

---

### G002: no-glsl-version

**Description**: Detects `#version` directives in GLSL.

**Severity**: warning

**Fixable**: Yes

**Details**: TouchDesigner manages GLSL versions internally. Explicit `#version` directives can cause compatibility issues.

**Fix**: Removes the `#version` line.

**Example violation**:
```
Warning G002: GLSL version directive found
  File: /project/ops/shader.text
  Line: 1
  Message: Remove #version directive (managed by TouchDesigner)
```

---

### G003: no-precision-qualifiers

**Description**: Detects precision qualifiers in GLSL.

**Severity**: warning

**Fixable**: No

**Details**: Precision qualifiers (`precision mediump float;`) are mobile GLSL and not needed in TouchDesigner.

## Python Rules (P)

### P001: valid-python-expressions

**Description**: Validates Python expressions in parameters.

**Severity**: error

**Fixable**: No

**Details**: Checks that Python expressions in `.parm` files (mode 49) are syntactically valid.

**Example violation**:
```
Error P001: Invalid Python expression
  File: /project/ops/controller.parm
  Parameter: tx
  Expression: absTime.frame * + 5
  Message: Syntax error in expression
```

---

### P002: safe-python-constructs

**Description**: Checks for potentially unsafe Python.

**Severity**: warning

**Fixable**: No

**Details**: Warns about Python constructs that could cause performance issues or unexpected behavior (e.g., imports, exec, eval).

## Performance Rules (F)

### F001: deep-nesting

**Description**: Detects deeply nested operator hierarchies.

**Severity**: warning

**Fixable**: No

**Options**:
- `max_depth`: Maximum nesting depth (default: 10)

**Details**: Excessive nesting can make projects hard to maintain and may impact performance.

**Example violation**:
```
Warning F001: Deep operator nesting
  Path: /project/a/b/c/d/e/f/g/h/i/j/k
  Depth: 11
  Message: Nesting exceeds max_depth of 10
```

---

### F002: excessive-inputs

**Description**: Detects operators with too many inputs.

**Severity**: warning

**Fixable**: No

**Options**:
- `max_inputs`: Maximum input count (default: 16)

**Details**: Operators with many inputs can be difficult to manage and may indicate design issues.

---

### F003: heavy-texture-chains

**Description**: Detects long TOP chains without caching.

**Severity**: warning

**Fixable**: No

**Options**:
- `max_chain_length`: Maximum chain length (default: 8)

**Details**: Long chains of texture operators without cache nodes can cause GPU performance issues.

**Cache breakers**: `cache`, `rendertop`, `feedback`, `feedbacktop`

---

### F004: unoptimized-feedback

**Description**: Detects feedback loops without optimization.

**Severity**: warning

**Fixable**: No

**Details**: Feedback loops should include synchronization operators to prevent excessive recomputation.

**Valid operators in loops**: `feedback`, `feedbackchop`, `cache`, `delay`, `lag`, `timemachine`

---

### F005: cook-every-frame

**Description**: Detects operators cooking every frame unnecessarily.

**Severity**: info

**Fixable**: No

**Details**: Some operators are marked to cook every frame even when their inputs don't change, wasting CPU/GPU resources.

**Exceptions**: Time-based operators like `timer`, `noise`, `lfo`, `beat`, `moviefilein`, etc.

## Configuration Examples

### Enable All Rules

```yaml
extends: strict
```

### Minimal (Syntax Only)

```yaml
extends: minimal
```

### Custom Selection

```yaml
select: [S, C, T, R]
ignore: [F, G003]

rules:
  F001:
    enabled: true
    options:
      max_depth: 15
  F003:
    enabled: true
    severity: error
```

## Rule Discovery

List all available rules:

```bash
td-linter rules
```

Show disabled rules too:

```bash
td-linter rules --show-disabled
```

Filter by category:

```bash
td-linter rules --category F
```
