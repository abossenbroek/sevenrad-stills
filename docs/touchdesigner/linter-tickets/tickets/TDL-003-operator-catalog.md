# TDL-003: Operator Type Catalog

---
id: TDL-003
status: pending
priority: high
phase: 0
depends_on: []
blocks: [TDL-022]
---

## Problem Statement

The linter needs to know which operators exist, which families they belong to, and what connections are valid between them. The spec's operator lists are acknowledged as "incomplete." Without a comprehensive catalog, type compatibility checking will have false positives and miss real errors.

## Acceptance Criteria

- [ ] All operator types enumerated for each family (TOP, CHOP, SOP, DAT, COMP, MAT)
- [ ] Input/output compatibility documented per operator
- [ ] Conversion operators identified (chopto, toptochop, etc.)
- [ ] Special input operators documented (glsl accepts DAT for shader code)
- [ ] `td_operators.yaml` database created
- [ ] Automated extraction script for future TD versions

## Files to Create

```
reference/
├── td_operators.yaml           # Comprehensive operator database
├── td_operators_schema.yaml    # Schema for the database
└── extract_operators.py        # Script to run in TD
```

## Research Pointers

### TouchDesigner Operator Documentation

- https://docs.derivative.ca/Operator
- https://docs.derivative.ca/TOP (and /CHOP, /SOP, /DAT, /COMP, /MAT)

### Programmatic Enumeration

In TouchDesigner Python:
```python
# Hint: Explore these
td.families  # List of operator families
op('someTOP').OPType  # Operator type
op.create()  # Create operators to inspect
```

Study how to:
1. List all operator types in a family
2. Get input connector information
3. Determine output type
4. Check if an operator is a converter

### Database Schema Design

Design a YAML schema that captures:

```yaml
operators:
  - name: string         # e.g., "displace"
    family: enum         # TOP, CHOP, SOP, DAT, COMP, MAT
    inputs:
      - index: int       # 0, 1, 2...
        accepts: [enum]  # Which families can connect
        required: bool
    outputs:
      family: enum       # What family this outputs
    is_converter: bool   # Does it convert between families?
    description: string
```

### Conversion Operators

Special attention to operators that bridge families:
- `chopto` (CHOP -> TOP)
- `toptochop` (TOP -> CHOP)
- `soptochop`, `choptosop`
- `dattochop`, `choptodat`
- etc.

These must be correctly identified so type checking allows them.

### Special Cases

Some operators accept unusual inputs:
- `glsl` TOP accepts DAT for shader code
- `render` TOP accepts SOP, MAT, COMP
- `geometry` COMP accepts SOP and MAT

Document all exceptions to the "same family only" rule.

## Validation Strategy

1. **Completeness**: Count operators in TD vs catalog
2. **Accuracy**: Verify a sample of operators manually
3. **Edge cases**: Test conversion operators specifically

## Version Considerations

New operators are added in each TD release:
- Include TD version in database
- Plan for database updates

## Definition of Done

All acceptance criteria checked. Database covers all operators in target TD version.
