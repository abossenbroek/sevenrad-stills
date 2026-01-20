# Phase 5: Integration & Polish

---
complete: true
---

## Overview

Make the linter production-ready: CI/CD integration, output formats for tooling, package distribution. This phase takes the linter from "works on my machine" to "works in every pipeline."

**Philosophy**: Tools that don't integrate don't get used. SARIF for GitHub, JSON for scripts, pre-commit for developers.

## Gate Criteria

| Gate | Requirement | Validation |
|------|-------------|------------|
| G5.1 | SARIF uploads to GitHub | Code scanning shows violations |
| G5.2 | Pre-commit hook works | Commit blocked on error |
| G5.3 | Package installable | `pip install td-linter` succeeds |

## Tickets

| Ticket | Title | Priority | Status |
|--------|-------|----------|--------|
| [TDL-050](../tickets/TDL-050-output-formats.md) | JSON/SARIF Output Formatters | High | **done** |
| [TDL-051](../tickets/TDL-051-precommit-hook.md) | Pre-commit Hook | High | **done** |
| [TDL-052](../tickets/TDL-052-github-actions.md) | GitHub Actions Workflow | High | **done** |
| [TDL-053](../tickets/TDL-053-harness-integration.md) | build_test_harness.py Integration | Medium | **done** |
| [TDL-054](../tickets/TDL-054-package-distribution.md) | Package Distribution | High | **done** |

## Dependencies

```
Phase 4 (rules) ──> TDL-050 ──> TDL-051
                           ├──> TDL-052
                           └──> TDL-053

TDL-014 (CLI) ──> TDL-054 (packaging requires CLI)
```

## Integration Points

| System | Integration Method |
|--------|-------------------|
| GitHub Code Scanning | SARIF format, upload-sarif action |
| Pre-commit Framework | .pre-commit-config.yaml hook |
| Existing build_test_harness.py | Python import, lint before collapse |
| PyPI | pyproject.toml, twine upload |
| Local development | pip install -e . |

## Output Format Requirements

| Format | Consumer | Key Features |
|--------|----------|--------------|
| text | Human terminal | Colors, grouping by file |
| json | Scripts, APIs | Machine-readable, stable schema |
| sarif | GitHub, IDEs | Standard for code scanning |

## Completion Checklist

- [x] TDL-050 complete: All three output formats work
- [x] TDL-051 complete: Hook blocks bad commits
- [x] TDL-052 complete: GH workflow runs on .toe.dir changes
- [x] TDL-053 complete: Harness calls linter pre-collapse
- [x] TDL-054 complete: Package installable via parent package
- [x] G5.1-G5.3 verified
- [x] Phase marked complete: true
