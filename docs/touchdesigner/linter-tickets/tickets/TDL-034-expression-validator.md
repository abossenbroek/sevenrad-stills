# TDL-034: Parameter Expression Validator

---
id: TDL-034
status: pending
priority: medium
phase: 3
depends_on: [TDL-011, TDL-002]
blocks: []
---

## Problem Statement

.parm files contain parameter expressions (mode 49/17) that are Python expressions evaluated at runtime. These need separate validation from full Python scripts because they're expressions, not statements, and have a different set of available globals.

## Acceptance Criteria

- [ ] Extracts expressions from .parm files (mode 49 and 17)
- [ ] Validates syntax using `ast.parse(mode='eval')`
- [ ] TD expression globals whitelist (me, op, parent, absTime, etc.)
- [ ] Reports syntax errors with parameter name and line
- [ ] CLI flag `--validate-expressions` enables this check
- [ ] Test cases for valid and invalid expressions

## Files to Create

```
td_linter/
├── embedded/
│   └── expression_validator.py
└── tests/
    └── test_expression_validator.py
```

## Research Pointers

### Expression vs Statement

Python has two parse modes:
- `ast.parse(code, mode='exec')` - Statements (scripts)
- `ast.parse(code, mode='eval')` - Single expression

Expressions:
- `absTime.frame * 0.6` - VALID
- `x = 5` - INVALID (assignment is statement)

### Extracting Expressions

From TDL-002 mode flag discovery:
- Mode 49: Expression mode
- Mode 17: String with expression

.parm line format:
```
tx 49 6531 absTime.frame*.6
```

The expression is the 4th field (after value).

### Extraction Algorithm

```python
def extract_expressions(parm_file):
    for line in parm_file:
        parts = line.split(None, 3)  # Split max 4 parts
        if len(parts) >= 4:
            mode = int(parts[1])
            if mode in (49, 17):
                yield Expression(
                    param=parts[0],
                    expression=parts[3],
                    line=line_number
                )
```

### Expression Globals

Expressions have a limited context:
- `me` - Current operator
- `op()` - Operator lookup
- `parent` - Parent operator
- `absTime` - Time info
- `project` - Project info
- `tdu` - TD utilities
- `math` - Math module (often imported)

Smaller than full script context.

### Validation Logic

```python
try:
    ast.parse(expression, mode='eval')
except SyntaxError as e:
    yield Violation(rule='expression-syntax', ...)

# Also check for undefined names
tree = ast.parse(expression, mode='eval')
names = find_names(tree)
for name in names:
    if name not in EXPRESSION_GLOBALS:
        yield Violation(rule='expression-undefined', severity=WARNING)
```

### CLI Integration

From the spec:
```bash
td-linter lint myproject.toe.dir --validate-expressions
```

This flag triggers expression extraction and validation.

### Edge Cases

- Empty expression field
- Expressions with string quotes
- Multi-part expressions (rare)
- Expressions that span (should they?)

## Performance Note

Expression validation is opt-in (`--validate-expressions`) because:
- Many projects have hundreds of expressions
- Validation adds overhead
- False positives for custom globals

## Definition of Done

All acceptance criteria checked. Expressions extracted and validated with mode 'eval'.
