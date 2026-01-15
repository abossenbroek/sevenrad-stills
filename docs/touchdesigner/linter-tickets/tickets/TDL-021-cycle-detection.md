# TDL-021: Cycle Detection Algorithm

---
id: TDL-021
status: pending
priority: critical
phase: 2
depends_on: [TDL-020]
blocks: []
---

## Problem Statement

Cycles in operator networks can cause TouchDesigner to hang. However, some cycles are intentional (feedback operators in CHOPs). The validator must distinguish invalid cycles from legitimate feedback loops.

## Acceptance Criteria

- [ ] Cycles detected using NetworkX algorithms
- [ ] CHOP-only cycles allowed (traditional feedback)
- [ ] Feedback operator types whitelist implemented
- [ ] .parm feedback parameter checking implemented
- [ ] Clear error messages include cycle path
- [ ] Suggestion to add feedback operator in error message
- [ ] Test cases for valid and invalid cycles

## Files to Create

```
td_linter/
├── rules/
│   └── no_invalid_cycles.py
└── tests/
    └── test_cycle_detection.py
```

## Research Pointers

### NetworkX Cycle Algorithms

- https://networkx.org/documentation/stable/reference/algorithms/cycles.html
- `nx.simple_cycles(G)` - Find all simple cycles
- `nx.find_cycle(G)` - Find one cycle (fast fail)

Study the difference and choose appropriately.

### Valid vs Invalid Cycles

**Valid (don't report)**:
1. CHOP-only chains (feedback is traditional in CHOPs)
2. Cycles containing feedback operator types
3. Cycles where an operator has `feedback` parameter enabled

**Invalid (report)**:
1. TOP chains looping back without feedback operator
2. SOP chains looping back
3. Any cycle without explicit feedback mechanism

### Feedback Operator Types (AG-001)

From the spec:
```
feedback       # TOP feedback operator
timemachine    # TOP time-based feedback
feedbackchop   # CHOP feedback operator
delay          # CHOP delay (can create intentional feedback)
lag            # CHOP lag (can be part of feedback)
```

These operators legitimize cycles.

### .parm Feedback Property Check

Some operators have a `feedback` parameter that, when enabled, legitimizes feedback:
1. Parse the .parm file for the operator
2. Look for `feedback`, `feedbackmode`, `usefeedback` parameters
3. Check if value is truthy (not 0, 'off', etc.)

### Algorithm Sketch

```
1. Find all cycles: cycles = nx.simple_cycles(G)
2. For each cycle:
   a. If all CHOP: skip (valid)
   b. If contains feedback operator: skip (valid)
   c. If any node has feedback param enabled: skip (valid)
   d. Otherwise: report violation
```

### Error Message Design

Good error message:
```
Invalid cycle detected: displace1 -> blur1 -> displace1
Hint: Add a feedback/timemachine operator to create intentional feedback loop
```

### Edge Cases

- Self-loops (operator connects to itself)
- Very long cycles (10+ operators)
- Multiple cycles sharing edges
- Cycles spanning containers

## Performance Considerations

- `simple_cycles` can be slow on large graphs
- Consider: Should we limit search depth?
- Consider: Early termination after N cycles found?

## Definition of Done

All acceptance criteria checked. Invalid cycles detected, valid feedback allowed.
