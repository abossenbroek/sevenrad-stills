# TouchDesigner Linter Engineering Tickets

Engineering tickets for building `td-linter`, a validation tool for TouchDesigner `.toe.dir` expanded projects.

**Philosophy**: These tickets guide experienced compiler developers without giving them fish. Each ticket describes what to build and why, provides research pointers, but leaves implementation decisions to the engineer.

## Quick Links

- **Source Specification**: [linter_spec.md](../reference/linter_spec.md)
- **Total Tickets**: 24
- **Phases**: 6 (Phase 0 through Phase 5)

## Phase Overview

| Phase | Focus | Tickets | Description | Status |
|-------|-------|---------|-------------|--------|
| [Phase 0](phases/phase-0-discovery.md) | Discovery | TDL-001 to TDL-003 | Sample collection, format documentation | ✅ Complete |
| [Phase 1](phases/phase-1-infrastructure.md) | Infrastructure | TDL-010 to TDL-014 | Lark grammars, CLI skeleton | ✅ Complete |
| [Phase 2](phases/phase-2-graph.md) | Graph | TDL-020 to TDL-024 | NetworkX model, structural validation | Pending |
| [Phase 3](phases/phase-3-embedded.md) | Embedded Code | TDL-030 to TDL-034 | GLSL/Python validation | Pending |
| [Phase 4](phases/phase-4-rules.md) | Rules | TDL-040 to TDL-043 | YAML configuration, rule system | Pending |
| [Phase 5](phases/phase-5-integration.md) | Integration | TDL-050 to TDL-054 | CI/CD, packaging, distribution | Pending |

## Ticket Index

### Phase 0: Format Discovery ✅
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| [TDL-001](tickets/TDL-001-sample-collection.md) | Sample Collection Campaign | Critical | ✅ Done |
| [TDL-002](tickets/TDL-002-mode-flag-discovery.md) | Mode Flag Discovery | Critical | ✅ Done |
| [TDL-003](tickets/TDL-003-operator-catalog.md) | Operator Type Catalog | High | ✅ Done |

### Phase 1: Core Infrastructure ✅
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| [TDL-010](tickets/TDL-010-grammar-n-files.md) | Lark Grammar for .n Files | Critical | ✅ Done |
| [TDL-011](tickets/TDL-011-grammar-parm-files.md) | Lark Grammar for .parm Files | Critical | ✅ Done |
| [TDL-012](tickets/TDL-012-toc-parser.md) | TOC Manifest Parser | High | ✅ Done |
| [TDL-013](tickets/TDL-013-python-bindings.md) | Python Parser Integration | High | ✅ Done |
| [TDL-014](tickets/TDL-014-cli-skeleton.md) | CLI Skeleton with Typer | High | ✅ Done |

### Phase 2: Graph Validation
| ID | Title | Priority |
|----|-------|----------|
| [TDL-020](tickets/TDL-020-graph-model.md) | NetworkX Graph Model | Critical |
| [TDL-021](tickets/TDL-021-cycle-detection.md) | Cycle Detection Algorithm | Critical |
| [TDL-022](tickets/TDL-022-type-compatibility.md) | Type Compatibility Checker | High |
| [TDL-023](tickets/TDL-023-dangling-inputs.md) | Dangling Input Detector | High |
| [TDL-024](tickets/TDL-024-reference-resolver.md) | Operator Reference Resolver | High |

### Phase 3: Embedded Code Validation
| ID | Title | Priority |
|----|-------|----------|
| [TDL-030](tickets/TDL-030-language-detector.md) | Language Detector | High |
| [TDL-031](tickets/TDL-031-glsl-validator.md) | GLSL Validator Integration | Critical |
| [TDL-032](tickets/TDL-032-python-validator.md) | Python AST Validator | High |
| [TDL-033](tickets/TDL-033-td-stubs.md) | TouchDesigner Python Stubs | Medium |
| [TDL-034](tickets/TDL-034-expression-validator.md) | Parameter Expression Validator | Medium |

### Phase 4: Rule System
| ID | Title | Priority |
|----|-------|----------|
| [TDL-040](tickets/TDL-040-rule-schema.md) | YAML Rule Schema | High |
| [TDL-041](tickets/TDL-041-rule-loader.md) | Rule Loader Implementation | High |
| [TDL-042](tickets/TDL-042-builtin-rules.md) | Built-in Rule Set | High |
| [TDL-043](tickets/TDL-043-rule-api.md) | Rule Configuration API | Medium |

### Phase 5: Integration & Polish
| ID | Title | Priority |
|----|-------|----------|
| [TDL-050](tickets/TDL-050-output-formats.md) | JSON/SARIF Output Formatters | High |
| [TDL-051](tickets/TDL-051-precommit-hook.md) | Pre-commit Hook | High |
| [TDL-052](tickets/TDL-052-github-actions.md) | GitHub Actions Workflow | High |
| [TDL-053](tickets/TDL-053-harness-integration.md) | build_test_harness.py Integration | Medium |
| [TDL-054](tickets/TDL-054-package-distribution.md) | Package Distribution | High |

## Dependency Graph

```
Phase 0: Discovery (parallel work possible)
TDL-001 ────────┐
TDL-002 ────────┼──> Phase 1
TDL-003 ────────┘

Phase 1: Infrastructure
TDL-010 (grammar .n) ────┐
TDL-011 (grammar .parm) ─┼──> TDL-013 (bindings) ──> Phase 2
TDL-012 (TOC) ───────────┘
TDL-014 (CLI) ─────────────────────────────────────> Phase 5

Phase 2: Graph Validation
TDL-020 (model) ──> TDL-021 (cycles)
                ├──> TDL-022 (types) <── TDL-003
                ├──> TDL-023 (dangling)
                └──> TDL-024 (references)

Phase 3: Embedded Code (after Phase 1)
TDL-030 (detector) ──> TDL-031 (GLSL)
                   └──> TDL-032 (Python) ──> TDL-033 (stubs)
TDL-011 ──> TDL-034 (expressions)

Phase 4: Rules (after Phases 2-3)
TDL-040 (schema) ──> TDL-041 (loader) ──> TDL-042 (builtin)
                                     └──> TDL-043 (API)

Phase 5: Integration (after Phase 4)
TDL-050 (formats) ──> TDL-051 (precommit)
                  ├──> TDL-052 (GitHub)
                  └──> TDL-053 (harness)
TDL-054 (package) <── TDL-014
```

## How to Use These Tickets

1. **Start with Phase 0**: Even if you're eager to code, format discovery prevents rework later

2. **Follow dependencies**: Tickets are sequenced. Attempting TDL-021 before TDL-020 will waste time

3. **Research first**: Each ticket has "Research Pointers" - read them before implementing

4. **Acceptance criteria are DoD**: A ticket is done when ALL checkboxes are checked

5. **Don't write code in tickets**: These tickets guide; they don't prescribe

## Key Decisions Left to Engineers

- Graph node ID conventions
- Error message formatting details
- Config file discovery order
- Severity thresholds for each rule
- Performance optimization strategies

## External Resources

All linked in individual tickets, but key ones:
- [Lark Parser Docs](https://lark-parser.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/)
- [TouchDesigner Python API](https://docs.derivative.ca/Python)
- [SARIF Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/)
