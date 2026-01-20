# Phase 4: Rule System

---
complete: true
---

## Overview

Transform hardcoded checks into a configurable rule system. Users should be able to enable/disable rules, adjust severities, and extend with custom rules via YAML configuration.

**Philosophy**: A linter is only as good as its configurability. ESLint succeeded because of `.eslintrc`. We need `.td-linter.yaml`.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G4.1 | Rules load from YAML | Custom config overrides defaults |
| G4.2 | Rules can be disabled | `enabled: false` silences rule |
| G4.3 | Presets work | `extends: [recommended]` applies bundle |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-040](../tickets/TDL-040-rule-schema.md) | YAML Rule Schema | High | ✅ Done |
| [TDL-041](../tickets/TDL-041-rule-loader.md) | Rule Loader Implementation | High | ✅ Done |
| [TDL-042](../tickets/TDL-042-builtin-rules.md) | Built-in Rule Set | High | ✅ Done |
| [TDL-043](../tickets/TDL-043-rule-api.md) | Rule Configuration API | Medium | ✅ Done |

## Dependencies

```
Phases 1-3 (validators) ──> TDL-042 (rules wrap validators)

TDL-040 ──> TDL-041 ──> TDL-042
                    └──> TDL-043
```

## Design Patterns to Study

| Pattern | Application |
|---------|-------------|
| Visitor Pattern | Rules visit AST nodes |
| Strategy Pattern | Different rule implementations |
| Chain of Responsibility | Rule pipeline |
| Factory Pattern | Rule instantiation from config |

## Rule Categories (from spec)

| Category | Examples |
|----------|----------|
| syntax | valid-n-file-syntax, valid-parm-file-syntax |
| connection | no-invalid-cycles, no-dangling-inputs |
| type | type-compatibility |
| reference | valid-operator-reference |
| glsl | glsl-no-version, glsl-syntax |
| python | python-syntax, python-undefined-name |
| performance | deep-nesting, excessive-inputs |
| style | naming-convention, tile-overlap |

## Completion Checklist

- [x] TDL-040 complete: JSON Schema validates configs
- [x] TDL-041 complete: Loader parses YAML, handles extends
- [x] TDL-042 complete: 16 rules in recommended preset (S001-S003, C001-C002, T001, R001-R002, G001-G003, P001-P003, F001-F002)
- [x] TDL-043 complete: API allows runtime rule queries
- [x] G4.1-G4.3 verified
- [x] Phase marked complete: true
