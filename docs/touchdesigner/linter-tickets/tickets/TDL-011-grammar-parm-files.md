# TDL-011: Tree-sitter Grammar for .parm Files

---
id: TDL-011
status: pending
priority: critical
phase: 1
depends_on: [TDL-001, TDL-002]
blocks: [TDL-013, TDL-034]
---

## Problem Statement

.parm files store operator parameters (values, expressions, modes). Each line is a parameter definition with a mode flag that determines interpretation. The grammar must handle the line-oriented format and correctly identify expression parameters for downstream validation.

## Acceptance Criteria

- [ ] Tree-sitter grammar parses all .parm files in sample corpus
- [ ] Grammar correctly identifies mode flags (from TDL-002 discovery)
- [ ] Grammar extracts expression content for mode 49/17 parameters
- [ ] Grammar handles all value types (numbers, strings, paths, identifiers)
- [ ] Grammar handles ? delimiters correctly
- [ ] Unit tests cover all parameter modes
- [ ] Error recovery produces partial AST for malformed files

## Files to Create

```
tree-sitter-toedir/
├── grammar-parm.js         # Separate grammar for .parm
├── corpus/
│   ├── valid-parm/         # Valid .parm files
│   └── invalid-parm/       # Malformed .parm files
└── ...
```

## Research Pointers

### .parm File Structure (from spec)

```
?
param_name mode value [expression]
param_name mode value
...
?
```

Key observations:
- `?` delimiters mark start and end
- Each line: name, mode (integer), value, optional expression
- Mode determines how to interpret value and expression
- Whitespace-separated fields

### Mode Flags (from TDL-002)

Integrate findings from TDL-002. The grammar should:
- Use semantic `mode_flag` type
- Enable downstream tools to filter by mode

### Value Types

| Type | Pattern | Example |
|------|---------|---------|
| number | `-?[0-9]+(\.[0-9]+)?` | `0.25`, `-10` |
| string | `"[^"]*"` | `"hello"` |
| path | `\./[a-zA-Z0-9_/]+` | `./out1` |
| identifier | `[a-zA-Z_][a-zA-Z0-9_]*` | `hermite` |

### Expression Handling

Mode 49/17 parameters have expressions:
```
tx 49 6531 absTime.frame*.6
```

The grammar should capture `absTime.frame*.6` as the expression field for TDL-034 to validate.

Challenge: Expressions can contain spaces (in strings, function calls). Need to capture everything after the value.

### Line-Oriented Parsing

Unlike .n files, .parm files are strictly line-oriented:
- Each parameter is one line
- Newline is significant

Consider: Should newline be `extras` or part of rules?

### Grammar Architecture Decision

Option A: Single grammar with multiple entry points
Option B: Separate grammars for .n and .parm

Recommendation: Separate grammars (cleaner, easier to maintain)

## Test Strategy

| Test Case | Description |
|-----------|-------------|
| simple.parm | Basic constant parameters |
| expressions.parm | Mode 49 with various expressions |
| strings.parm | String values with special characters |
| paths.parm | Relative and absolute path references |
| complex.parm | Mix of all types |

## Edge Cases from Samples

Watch for these in the sample corpus:
- Empty .parm files (just `?\n?`)
- Very long expressions
- Unicode in strings
- Path references with unusual characters

## Definition of Done

All acceptance criteria checked. Grammar parses entire sample corpus without errors. Mode flags correctly identified.
