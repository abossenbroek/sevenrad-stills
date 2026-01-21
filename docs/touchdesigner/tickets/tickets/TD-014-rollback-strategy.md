# TD-014: GLSL Feasibility & Rollback Strategy

---
id: TD-014
status: complete
priority: medium
phase: 5
depends_on: []
blocks: []
---

## Description

Document GLSL feasibility checklist and escalation path to C++ TOPs.

## Status: COMPLETE

This ticket is already complete. The feasibility checklist and rollback strategy are documented in:
- `docs/touchdesigner/00-IMPLEMENTATION-OVERVIEW.md` (GLSL Feasibility Checklist section)

## Deliverables (Completed)

- [x] Feasibility checklist documented
- [x] Escalation path defined (GLSL Fragment → Compute → C++ → Python)
- [x] Pre-approved C++ exceptions listed

## GLSL Feasibility Checklist

Before attempting pure GLSL for any effect, verify:
- [ ] No random-access writes required
- [ ] No inter-pixel communication
- [ ] No recursive algorithms
- [ ] Loops bounded to <1024 iterations
- [ ] No bitwise float manipulation
- [ ] No geometry shader requirements

## Escalation Path

```
GLSL Fragment ─[FAIL]─> GLSL Compute ─[FAIL]─> C++ TOP ─[FAIL]─> Python Script TOP
```

## Pre-Approved C++ Exceptions

| Effect | Reason | Decision |
|--------|--------|----------|
| buffer_corruption | XOR mode needs bitwise float ops | C++ approved |
| band_swap | Random tile access | Compute first, C++ fallback |

## References

- [00-IMPLEMENTATION-OVERVIEW.md](../../00-IMPLEMENTATION-OVERVIEW.md)
