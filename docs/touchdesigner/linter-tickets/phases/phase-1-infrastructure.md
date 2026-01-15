# Phase 1: Core Infrastructure

---
complete: false
---

## Overview

Build the syntactic foundation: tree-sitter grammars that parse .n and .parm files into ASTs. This phase produces the parser that all subsequent validation builds upon.

**Philosophy**: A grammar is a contract. Get it wrong, and everything downstream fails. Get it right, and everything becomes trivial.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G1.1 | .n grammar parses all samples | tree-sitter parse succeeds on corpus |
| G1.2 | .parm grammar parses all samples | tree-sitter parse succeeds on corpus |
| G1.3 | CLI skeleton operational | td-linter --help returns 0 |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-010](../tickets/TDL-010-grammar-n-files.md) | Tree-sitter Grammar for .n Files | Critical | pending |
| [TDL-011](../tickets/TDL-011-grammar-parm-files.md) | Tree-sitter Grammar for .parm Files | Critical | pending |
| [TDL-012](../tickets/TDL-012-toc-parser.md) | TOC Manifest Parser | High | pending |
| [TDL-013](../tickets/TDL-013-python-bindings.md) | Python Bindings Integration | High | pending |
| [TDL-014](../tickets/TDL-014-cli-skeleton.md) | CLI Skeleton with Typer | High | pending |

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

1. **Why tree-sitter?** Industry standard for incremental parsing. Great error recovery. Python bindings exist.

2. **Why two grammars?** .n and .parm have different structures. Separate grammars are cleaner than one complex grammar with multiple entry points.

3. **Why not regex?** .n files have nested structures (inputs block). Regex can't handle nesting without becoming unmaintainable.

## Completion Checklist

- [ ] TDL-010 complete: .n grammar passes corpus
- [ ] TDL-011 complete: .parm grammar passes corpus
- [ ] TDL-012 complete: TOC validation works
- [ ] TDL-013 complete: Python can invoke parsers
- [ ] TDL-014 complete: CLI structure in place
- [ ] G1.1-G1.3 verified
- [ ] Phase marked complete: true
