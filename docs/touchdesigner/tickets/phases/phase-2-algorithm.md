# Phase 2: Algorithm Validation

---
complete: false
---

## Overview

Validate core GLSL utilities (PCG hash, bilinear sampling, HSV conversion) match Taichi implementations before migrating any effects.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G3 | PCG hash validated | Bit-identical to Taichi OR divergence documented |
| G4 | Bilinear sampling aligned | Within 1/255 tolerance at test positions |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TD-004](../tickets/TD-004-pcg-hash.md) | PCG Hash Equivalence Testing | Critical | pending |
| [TD-005](../tickets/TD-005-bilinear.md) | Bilinear Sampling Alignment | High | pending |
| [TD-006](../tickets/TD-006-hsv.md) | HSV Conversion Validation | High | pending |

## Dependencies

```
Phase 1 (G1, G2) ──> TD-002a ──┬──> TD-004 ──> G3
                               ├──> TD-005 ──> G4
                               └──> TD-006
```

**Critical**: TD-004 and TD-005 explicitly depend on TD-002 (preamble) completion, not just Phase 1 generally. (RF-001 fix)

## Notes

- All algorithm tests require TD-002a smoke test to pass first
- If PCG hash diverges, document and generate GLSL-specific fixtures
- Bilinear coordinate fix may be needed (test determines)

## Completion Checklist

- [ ] TD-004 complete (or divergence documented)
- [ ] TD-005 complete (or offset corrected)
- [ ] TD-006 complete
- [ ] G3 verified: PCG hash behavior documented
- [ ] G4 verified: Bilinear within tolerance
- [ ] Phase marked complete: true
