# Phase 3: Testing Infrastructure Hardening

---
complete: false
---

## Overview

Replace fragile MD5 testing with perceptual diff, add MoltenVK/Metal validation to CI, and document known linter gaps.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G5 | Perceptual diff active | MD5 replaced with SSIM in all tests |
| G6 | MoltenVK validation in CI | SPIRV-Cross Metal compilation passes |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TD-007](../tickets/TD-007-perceptual-diff.md) | Replace MD5 with Perceptual Diff | High | pending |
| [TD-008](../tickets/TD-008-moltenvk.md) | Add MoltenVK/Metal Validation | High | pending |
| [TD-009](../tickets/TD-009-linter-gaps.md) | Document Known Linter Gaps | Medium | pending |

## Dependencies

```
Phase 1 ──────────────┬──> TD-007 ──> G5
                      ├──> TD-008 ──> G6
                      └──> TD-009

Phase 2 ··soft··> TD-007 (fixture regeneration)
```

**Note**: TD-007 uses two-tier fixture approach (RF-004 fix):
1. Create preliminary fixtures now (marked provisional)
2. Regenerate after Phase 2 confirms algorithm behavior

## Notes

- TD-007 expected outputs may need regeneration after Phase 2
- TD-008 validates GLSL->SPIRV->Metal in CI (no TD runtime)
- TD-009 documents what glslangValidator cannot catch

## Completion Checklist

- [ ] TD-007 complete (provisional fixtures created)
- [ ] TD-008 complete
- [ ] TD-009 complete
- [ ] G5 verified: SSIM comparison working
- [ ] G6 verified: Metal compilation in CI
- [ ] Fixtures regenerated after Phase 2 (if needed)
- [ ] Phase marked complete: true
