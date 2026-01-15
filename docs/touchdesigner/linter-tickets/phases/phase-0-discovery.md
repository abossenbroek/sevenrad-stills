# Phase 0: Format Discovery

---
complete: true
---

## Overview

Before building a linter, we must deeply understand the format we're linting. This phase builds the empirical foundation through sample collection, systematic enumeration, and documentation of format variations across TouchDesigner versions.

**Philosophy**: Compilers fail when they assume too much. This phase eliminates assumptions by grounding the grammar in real-world data.

## Gate Criteria

| Gate | Requirement | Validation | Status |
|------|-------------|------------|--------|
| G0.1 | 50+ diverse .toe.dir samples collected | catalog.yaml documents each | ✅ PASS |
| G0.2 | All parameter mode flags documented | mode_flag_discovery.toe tests each | ✅ PASS |
| G0.3 | Operator compatibility database created | td_operators.yaml covers all families | ✅ PASS |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-001](../tickets/TDL-001-sample-collection.md) | Sample Collection Campaign | Critical | ✅ done |
| [TDL-002](../tickets/TDL-002-mode-flag-discovery.md) | Mode Flag Discovery | Critical | ✅ done |
| [TDL-003](../tickets/TDL-003-operator-catalog.md) | Operator Type Catalog | High | ✅ done |

## Dependencies

```
TDL-001 ─────> TDL-010 (grammar needs samples)
TDL-002 ─────> TDL-011 (parm grammar needs mode docs)
TDL-003 ─────> TDL-022 (type checker needs compatibility matrix)
```

## Why This Phase Exists

The specification (linter_spec.md) contains several "RF-002" and "CM-002" notes indicating unverified assumptions:
- Parameter mode flags (0, 17, 32, 49) are "preliminary observations"
- Operator compatibility lists are "incomplete"
- Format variations across TD versions are undocumented

Skipping this phase means building a linter on quicksand.

## Completion Checklist

- [x] TDL-001 complete: 50+ samples with catalog
- [x] TDL-002 complete: All modes documented
- [x] TDL-003 complete: td_operators.yaml created
- [x] G0.1-G0.3 verified
- [x] Phase marked complete: true
