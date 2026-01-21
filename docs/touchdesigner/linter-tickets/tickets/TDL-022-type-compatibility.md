# TDL-022: Type Compatibility Checker

---
id: TDL-022
status: pending
priority: high
phase: 2
depends_on: [TDL-020, TDL-003]
blocks: []
---

## Problem Statement

Operators can only connect to operators of compatible families (TOP->TOP, CHOP->CHOP) unless using conversion operators. Incompatible connections cause runtime errors. The validator should catch these at lint time.

## Acceptance Criteria

- [ ] Compatibility matrix defines valid connections
- [ ] Conversion operators (chopto, toptochop, etc.) detected and allowed
- [ ] Special input operators handled (glsl accepts DAT)
- [ ] Clear error messages identify incompatible types
- [ ] Uses td_operators.yaml from TDL-003
- [ ] Test cases for valid, invalid, and converter connections

## Files to Create

```
td_linter/
├── rules/
│   └── type_compatibility.py
└── tests/
    └── test_type_compatibility.py
```

## Research Pointers

### Compatibility Matrix

Default rule: Same family connects to same family.

```
TOP -> TOP: YES
TOP -> CHOP: NO (unless converter)
CHOP -> CHOP: YES
...
```

Exception: COMP can connect to TOP (component outputs).

### Conversion Operators

From the spec and TDL-003:
- `chopto` - CHOP input, TOP output
- `toptochop` - TOP input, CHOP output
- `soptochop`, `choptosop`
- `dattochop`, `choptodat`
- `soptodat`, `dattosop`
- etc.

When target is a converter, skip compatibility check.

### Special Input Operators

Some operators accept unusual inputs:
- `glsl` TOP accepts DAT for shader code
- `render` TOP accepts SOP, MAT, COMP
- `geometry` COMP accepts SOP and MAT

These should be in td_operators.yaml with their `accepts` lists.

### Algorithm Sketch

```
For each edge (source -> target):
  1. If target is converter: skip
  2. If target has special_inputs and source.family in special_inputs: skip
  3. If source.family in COMPATIBLE[target.family]: skip
  4. Report type incompatibility
```

### Data Structure Design

Consider how to represent compatibility:

Option A: Matrix (dict of sets)
```python
COMPATIBLE = {
    'TOP': {'TOP'},
    'CHOP': {'CHOP'},
    ...
}
```

Option B: Load from td_operators.yaml
```python
operators = load_yaml('td_operators.yaml')
# Build compatibility from accepts fields
```

### Error Message Design

Good:
```
Type incompatibility: CHOP 'noise1' cannot connect to TOP 'displace1'
Hint: Use 'chopto' to convert CHOP to TOP
```

### Edge Cases

- NULL operators (exist but may have special rules)
- Custom operators (unknown to catalog)
- Operators with multiple inputs of different types

## Future Enhancement

Consider severity levels:
- ERROR for clearly invalid (CHOP -> SOP)
- WARNING for unusual but possibly valid

## Definition of Done

All acceptance criteria checked. Type mismatches detected, converters allowed.
