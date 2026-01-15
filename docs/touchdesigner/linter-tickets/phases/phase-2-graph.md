# Phase 2: Graph Validation

---
complete: false
---

## Overview

TouchDesigner networks are directed graphs. This phase builds the graph model and implements structural validation: cycle detection, type compatibility, dangling references.

**Philosophy**: Graph theory gives us algorithms. NetworkX gives us implementation. Our job is to map TD semantics onto graph properties.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G2.1 | Graph construction handles all samples | No exceptions on corpus |
| G2.2 | Cycle detection identifies known-bad networks | Test fixture with cycle detected |
| G2.3 | Type checker catches cross-family connections | TOP->CHOP without converter flagged |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-020](../tickets/TDL-020-graph-model.md) | NetworkX Graph Model | Critical | pending |
| [TDL-021](../tickets/TDL-021-cycle-detection.md) | Cycle Detection Algorithm | Critical | pending |
| [TDL-022](../tickets/TDL-022-type-compatibility.md) | Type Compatibility Checker | High | pending |
| [TDL-023](../tickets/TDL-023-dangling-inputs.md) | Dangling Input Detector | High | pending |
| [TDL-024](../tickets/TDL-024-reference-resolver.md) | Operator Reference Resolver | High | pending |

## Dependencies

```
Phase 1 (parsers) ──> TDL-020 ──> TDL-021
                              ├──> TDL-022
                              ├──> TDL-023
                              └──> TDL-024

TDL-003 (operator catalog) ──> TDL-022
```

## Graph Theory Concepts to Study

| Concept | Application |
|---------|-------------|
| Directed Graph (DiGraph) | TD networks have directional flow |
| Cycle Detection | `nx.simple_cycles()` finds loops |
| Strongly Connected Components | Feedback loops form SCCs |
| Topological Sort | Valid DAGs can be topo-sorted |
| Edge Attributes | Store input_index on edges |
| Node Attributes | Store OperatorNode on nodes |

## Completion Checklist

- [ ] TDL-020 complete: Graph builds from parsed ASTs
- [ ] TDL-021 complete: Cycles detected, feedback allowed
- [ ] TDL-022 complete: Type mismatches flagged
- [ ] TDL-023 complete: Missing operators reported
- [ ] TDL-024 complete: ./path and /path resolved
- [ ] G2.1-G2.3 verified
- [ ] Phase marked complete: true
