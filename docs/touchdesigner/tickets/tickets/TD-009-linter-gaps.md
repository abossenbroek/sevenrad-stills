# TD-009: Document Known Linter Gaps

---
id: TD-009
status: pending
priority: medium
phase: 3
depends_on: []
blocks: []
---

## Description

Document what glslangValidator does NOT catch and provide workarounds for developers.

## Acceptance Criteria

- [ ] Linter gaps table added to documentation
- [ ] Each gap has documented workaround
- [ ] Developer checklist created

## Gaps to Document

| Gap | Description | Workaround |
|-----|-------------|------------|
| TD uniform conflicts | Using a name TD reserves internally | Check TD docs for reserved names |
| MoltenVK translation | Metal has different semantics | SPIRV-Cross validation in CI |
| Texture binding limits | TD may have different limits | Test at runtime |
| Precision mismatches | highp/mediump/lowp behavior | Always use highp on desktop |
| Non-constant array index | Metal restrictions | Avoid or test specifically |

## Files

- `docs/touchdesigner/03-REMEDIATION-PLAN.md` (modify - add section)

## Developer Checklist

Before submitting PR:
- [ ] Shader passes glslangValidator
- [ ] Shader passes SPIRV-Cross Metal compilation
- [ ] No dynamic array indexing (unless tested)
- [ ] No TD reserved uniform names
- [ ] Local TD render test passes

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 3.3
