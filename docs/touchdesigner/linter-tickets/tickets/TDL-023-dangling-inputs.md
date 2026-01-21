# TDL-023: Dangling Input Detector

---
id: TDL-023
status: pending
priority: high
phase: 2
depends_on: [TDL-020]
blocks: []
---

## Problem Statement

Inputs that reference non-existent operators cause errors at runtime. The graph builder (TDL-020) creates placeholder nodes for missing references. This validator reports those placeholders as errors.

## Acceptance Criteria

- [ ] All `MISSING:*` placeholder nodes reported as errors
- [ ] Error message includes: target operator, missing reference name, input index
- [ ] Line number in .n file included if available
- [ ] Test cases for missing references at various depths
- [ ] Handles typos and path errors distinctly if possible

## Files to Create

```
td_linter/
├── rules/
│   └── no_dangling_inputs.py
└── tests/
    └── test_dangling_inputs.py
```

## Research Pointers

### Placeholder Node Convention

From TDL-020, missing references create nodes like:
```
G.add_edge("MISSING:nonexistent", target_path, input_index=0, missing=True)
```

The validator finds these by:
1. Iterating edges
2. Checking if source starts with `MISSING:`
3. Or checking edge attribute `missing=True`

### Error Information

For each dangling input, report:
- **Target**: Which operator has the bad input
- **Reference**: What name was referenced
- **Index**: Which input slot
- **Location**: .n file and line if possible

### Common Causes

1. **Typo**: `moviefilein1` vs `moveifilein1`
2. **Deleted operator**: Reference to operator that was removed
3. **Wrong path**: Relative vs absolute path confusion
4. **Renamed operator**: Old name still in inputs block

### Suggestion Generation

If possible, suggest corrections:
- Fuzzy match against existing operators
- "Did you mean 'moviefilein1'?"

This is a nice-to-have enhancement.

### Algorithm

```python
for source, target, data in G.edges(data=True):
    if source.startswith("MISSING:") or data.get('missing'):
        ref_name = source.replace("MISSING:", "")
        yield Violation(
            rule='no-dangling-inputs',
            message=f"Input references non-existent: '{ref_name}'",
            path=target,
            context={'missing_ref': ref_name, 'input_index': data.get('input_index')}
        )
```

### Error Message Design

```
no-dangling-inputs: Input 0 of 'displace1' references non-existent operator 'moviefliein1'
Location: project1/displace1.n:6
```

## Edge Cases

- Multiple inputs to same missing operator
- Missing operator referenced from multiple places
- Nested path references that partially resolve

## Definition of Done

All acceptance criteria checked. All missing references reported with clear messages.
