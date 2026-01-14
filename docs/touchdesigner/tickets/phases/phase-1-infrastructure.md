# Phase 1: Infrastructure Foundation

---
complete: false
---

## Overview

Establish the testing foundation required for all subsequent validation work. This phase sets up the macOS CI environment, extracts the real TouchDesigner GLSL preamble, and creates all test fixtures.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G1 | macOS CI runner operational | SPIRV-Cross Metal compilation passes |
| G2 | TD preamble extracted | validate_glsl.py uses real TD uniforms |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TD-001](../tickets/TD-001-td-preamble.md) | Extract TD GLSL Preamble | Critical | pending |
| [TD-002](../tickets/TD-002-macos-ci.md) | Set Up macOS CI Runner | Critical | pending |
| [TD-002a](../tickets/TD-002a-smoke-test.md) | Minimal Passthrough Smoke Test | High | pending |
| [TD-003](../tickets/TD-003-fixtures.md) | Create Validation Fixtures | High | pending |

## Dependencies

```
TD-001 ──────────────┐
                     ├──> TD-002a ──> Phase 2
TD-002 ──────────────┤
                     │
TD-003 ──────────────┘
```

## Notes

- TD-002 uses SPIRV-only CI validation (RF-002 fix)
- TD-002a added as smoke test gate (RF-003 fix)
- TD-003 includes stock footage sourcing (RF-005 fix)

## Definition of Done

**Convention (TR-011 fix)**: The Acceptance Criteria listed in each ticket serve as the Definition of Done. A ticket is complete when ALL acceptance criteria checkboxes are checked.

Standard DoD applies to all tickets:
- All acceptance criteria met
- Code reviewed (if applicable)
- Tests passing (if applicable)
- Documentation updated (if applicable)

## Completion Checklist

- [ ] TD-001 complete
- [ ] TD-002 complete
- [ ] TD-002a complete
- [ ] TD-003 complete
- [ ] G1 verified: CI renders test shader
- [ ] G2 verified: Preamble matches actual TD
- [ ] Phase marked complete: true
