# TDL-002: Mode Flag Discovery

---
id: TDL-002
status: pending
priority: critical
phase: 0
depends_on: []
blocks: [TDL-011, TDL-034]
---

## Problem Statement

.parm files contain mode flags (0, 17, 32, 49, etc.) whose meanings are undocumented. The spec notes these are "preliminary observations requiring systematic verification." Without knowing what each mode means, the linter can't properly validate parameter expressions or detect invalid configurations.

## Acceptance Criteria

- [ ] TouchDesigner project `mode_flag_discovery.toe` created
- [ ] All parameter mode flags enumerated programmatically via TD Python API
- [ ] Each mode's behavior documented (constant, expression, default, etc.)
- [ ] Test parameters created for each mode type
- [ ] Mode differences between TD versions documented (if any)
- [ ] `mode_flags.yaml` reference file created with all findings

## Files to Create

```
reference/
├── mode_flag_discovery.toe     # TD project for enumeration
├── mode_flags.yaml             # Documented mode values
└── mode_extraction.py          # Script to run inside TD
```

## Research Pointers

### TouchDesigner Parameter System

Study the TD parameter documentation:
- https://docs.derivative.ca/Par_Class
- https://docs.derivative.ca/ParMode_Class
- https://docs.derivative.ca/Parameter

Key classes to investigate:
- `par.mode` - Returns the parameter mode
- `par.eval()` vs `par.val` - Different depending on mode
- `par.expr` - The expression string if mode is expression

### Discovery Approach

1. **Create test parameters**: In TD, create operators with parameters in different modes (constant, expression, bind, etc.)

2. **Export and inspect**: Expand to .toe.dir and examine the .parm files

3. **Correlate values**: Map the mode integer in .parm to the `par.mode` enum in Python

4. **Document patterns**: Note which operators use which modes by default

### Python Enumeration Script

Write a script to run inside TouchDesigner that:
- Iterates over all operators in the project
- For each parameter, logs: name, mode value, mode name, value, expression
- Exports to CSV/YAML for analysis

Hint: `op.pars()` returns all parameters, `par.mode` returns the mode enum.

## Expected Modes (Verify These)

| Value | Suspected Meaning | Verify |
|-------|------------------|--------|
| 0 | Constant | ? |
| 17 | String with expression | ? |
| 32 | Default/unchanged | ? |
| 49 | Expression | ? |
| ? | Export | ? |
| ? | Bind | ? |

## Version Differences

Check if mode values changed between:
- TouchDesigner 2022.20000+
- TouchDesigner 2023.x
- TouchDesigner 2024.x

Document any differences in the reference file.

## Why This Matters

1. **Expression validation**: Mode 49/17 parameters contain Python expressions that need separate validation (TDL-034)
2. **Default detection**: Mode 32 may indicate unchanged defaults (skip validation)
3. **Grammar accuracy**: The .parm grammar needs to know valid mode ranges

## Definition of Done

All acceptance criteria checked. Mode flags fully documented with evidence.
