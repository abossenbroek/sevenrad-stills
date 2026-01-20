# Phase 5: Performance, Limits & Process

---
complete: false
---

## Overview

Test compute shader resolution limits, establish performance benchmarking framework, and document GLSL feasibility criteria with rollback strategy.

## Gate Criteria

No gates - this phase produces operational guardrails for Phase 6.

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TD-012](../tickets/TD-012-compute-limits.md) | Compute Shader Resolution Testing | Medium | pending |
| [TD-013](../tickets/TD-013-benchmark.md) | Performance Benchmarking Framework | Medium | pending |
| [TD-014](../tickets/TD-014-rollback-strategy.md) | GLSL Feasibility & Rollback Strategy | Medium | complete |

## Dependencies

```
Phase 1 ──> TD-012 (needs macOS CI)
            TD-013 (needs macOS CI)

TD-014 (already complete - documented in 00-IMPLEMENTATION-OVERVIEW.md)
```

## Contingency Tickets (Dormant)

These tickets are created but dormant. Activate only if GLSL compute fails:

| Ticket | Title | Trigger |
|--------|-------|---------|
| [TD-026a](../tickets/TD-026a-band-swap-cpp.md) | band_swap C++ TOP Fallback | GLSL compute fails |
| [TD-027a](../tickets/TD-027a-buffer-corruption-cpp.md) | buffer_corruption C++ TOP Fallback | GLSL compute fails |

(RF-007 fix)

## Notes

- TD-012 includes baseline capture subtask (RF-008 fix)
- TD-014 is already complete (feasibility checklist in overview doc)
- Contingency tickets ready if C++ fallback needed

## Completion Checklist

- [ ] TD-012 complete (with baselines captured)
- [ ] TD-013 complete
- [x] TD-014 complete (documented)
- [ ] Performance baselines stored in perf_baseline.json
- [ ] Compute limits documented
- [ ] Phase marked complete: true
