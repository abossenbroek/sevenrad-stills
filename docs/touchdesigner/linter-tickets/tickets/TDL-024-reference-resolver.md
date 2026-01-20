# TDL-024: Operator Reference Resolver

---
id: TDL-024
status: pending
priority: high
phase: 2
depends_on: [TDL-020]
blocks: []
---

## Problem Statement

.parm files contain operator references (pixeldat, computedat, top, chop paths). These must resolve to existing operators. Unlike .n file inputs, these are parameter values that reference operators by path.

## Acceptance Criteria

- [ ] Relative references (`./out1`) resolved correctly
- [ ] Absolute references (`/project1/geo1`) resolved correctly
- [ ] All reference parameter types checked (pixeldat, computedat, top, chop, dat, sop)
- [ ] Error messages identify parameter name and bad reference
- [ ] Works with container hierarchies
- [ ] Test cases for valid and invalid references

## Files to Create

```
td_linter/
├── rules/
│   └── valid_operator_reference.py
└── tests/
    └── test_operator_reference.py
```

## Research Pointers

### Reference Parameter Types

From the spec, these parameters contain operator references:
- `pixeldat` - DAT reference for GLSL shader code
- `computedat` - DAT reference for compute shader code
- `top` - TOP reference
- `chop` - CHOP reference
- `dat` - DAT reference
- `sop` - SOP reference
- `mat` - MAT reference

### Path Resolution

**Relative paths** (start with `./`):
```
# In /project1/container1/glsl1.parm:
pixeldat 0 ./shader_code
# Resolves to: /project1/container1/shader_code
```

**Absolute paths** (start with `/`):
```
top 0 /project1/moviefilein1
# Resolves to: /project1/moviefilein1
```

**Bare names** (no prefix):
```
top 0 moviefilein1
# Resolves to: sibling in same container
```

### Resolution Algorithm

```python
def resolve_reference(ref_value: str, parm_file_path: Path) -> str:
    if ref_value.startswith('./'):
        # Relative to parent directory
        parent = parm_file_path.parent
        return f"{parent}/{ref_value[2:]}"
    elif ref_value.startswith('/'):
        # Absolute from root
        return ref_value[1:]  # Strip leading /
    else:
        # Sibling in same container
        parent = parm_file_path.parent
        return f"{parent}/{ref_value}"
```

### Validation Against Graph

```python
resolved_path = resolve_reference(ref_value, parm_file_path)
if resolved_path not in graph.nodes:
    yield Violation(...)
```

### Edge Cases

- Empty reference (allowed?)
- Reference to self
- Reference with expression (mode 49)
- References in nested containers
- Unicode in paths

### Which Parameters to Check

Not every parameter with a string value is a reference. Maintain a list of known reference parameters:
```python
REFERENCE_PARAMS = {
    'pixeldat', 'computedat', 'top', 'chop', 'dat', 'sop', 'mat',
    'target', 'source', 'clone', 'externaltox',
    # Add more as discovered
}
```

### Error Message Design

```
valid-operator-reference: Parameter 'pixeldat' in 'glsl1' references non-existent: './shadercode'
Resolved path: project1/container1/shadercode
```

## Future Enhancement

- Validate type compatibility (e.g., `pixeldat` should point to a DAT)
- Suggest similar existing operators (fuzzy match)

## Definition of Done

All acceptance criteria checked. Path references validate against graph.
