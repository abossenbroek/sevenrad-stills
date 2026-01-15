# Phase 1: Core Infrastructure

---
complete: true
---

## Overview

Build the syntactic foundation: Lark grammars that parse .n and .parm files into ASTs. This phase produces the parser that all subsequent validation builds upon.

**Philosophy**: A grammar is a contract. Get it wrong, and everything downstream fails. Get it right, and everything becomes trivial.

**Implementation Note**: We chose Lark (Python EBNF parser) over tree-sitter for simpler Python integration and faster iteration.

## Gate Criteria

| Gate | Requirement | Validation | Status |
|------|-------------|------------|--------|
| G1.1 | .n grammar parses all samples | Lark parse succeeds on corpus (35/35) | ✅ PASS |
| G1.2 | .parm grammar parses all samples | Lark parse succeeds on corpus (29/29) | ✅ PASS |
| G1.3 | CLI skeleton operational | td-linter --help returns 0 | ✅ PASS |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-010](../tickets/TDL-010-grammar-n-files.md) | Lark Grammar for .n Files | Critical | ✅ done |
| [TDL-011](../tickets/TDL-011-grammar-parm-files.md) | Lark Grammar for .parm Files | Critical | ✅ done |
| [TDL-012](../tickets/TDL-012-toc-parser.md) | TOC Manifest Parser | High | ✅ done |
| [TDL-013](../tickets/TDL-013-python-bindings.md) | Python Parser Integration | High | ✅ done |
| [TDL-014](../tickets/TDL-014-cli-skeleton.md) | CLI Skeleton with Typer | High | ✅ done |

## Dependencies

```
Phase 0 ──────┐
              │
TDL-010 ──────┤
              ├──> TDL-013 ──> Phase 2
TDL-011 ──────┤
              │
TDL-012 ──────┘

TDL-014 (parallel, no deps)
```

## Key Decisions

1. **Why Lark?** Pure Python, excellent error messages, EBNF syntax familiar to most developers. Simpler than tree-sitter for this use case.

2. **Why two grammars?** .n and .parm have different structures. Separate grammars are cleaner than one complex grammar with multiple entry points.

3. **Why not regex?** .n files have nested structures (inputs block). Regex can't handle nesting without becoming unmaintainable.

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_n_parser_gate.py | 82 | Positive + negative cases for G1.1 |
| test_parm_parser_gate.py | 57 | Positive + negative cases for G1.2 |
| test_cli_gate.py | 16 | Positive + negative cases for G1.3 |
| test_toc_parser.py | 5 | TOC parser validation |
| test_lint_flow.py | 6 | Integration tests |

## Completion Checklist

- [x] TDL-010 complete: .n grammar passes corpus (35/35 files)
- [x] TDL-011 complete: .parm grammar passes corpus (29/29 files)
- [x] TDL-012 complete: TOC validation works
- [x] TDL-013 complete: Python can invoke parsers
- [x] TDL-014 complete: CLI structure in place
- [x] G1.1-G1.3 verified (185 tests passing)
- [x] Phase marked complete: true
