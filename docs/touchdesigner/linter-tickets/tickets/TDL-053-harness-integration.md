# TDL-053: build_test_harness.py Integration

---
id: TDL-053
status: done
priority: medium
phase: 5
depends_on: [TDL-042]
blocks: []
---

## Problem Statement

The existing `build_test_harness.py` script collapses .toe.dir to .toe binary format. Linting should run before collapse to catch errors early. Integration should be seamless - if linting fails, collapse should abort.

## Acceptance Criteria

- [ ] Linter callable from Python (not just CLI)
- [ ] Integration function for build_test_harness.py
- [ ] Lint errors abort collapse
- [ ] Lint warnings logged but don't abort
- [ ] Skip option for bypassing lint
- [ ] Documentation for integration pattern

## Files to Create

```
td_linter/
├── api.py                # Programmatic API
└── tests/
    └── test_api.py

docs/
└── programmatic_usage.md
```

## Research Pointers

### Programmatic API

The CLI wraps functionality that should also be directly callable:

```python
from td_linter import lint

violations = lint(path='myproject.toe.dir')
errors = [v for v in violations if v.severity == Severity.ERROR]

if errors:
    print("Linting failed")
    sys.exit(1)
```

### Integration Pattern

In build_test_harness.py:

```python
def collapse(verbose: bool = False, skip_lint: bool = False) -> int:
    if not skip_lint:
        if not lint_before_collapse(HARNESS_DIR):
            return 1

    # ... existing collapse logic
```

```python
def lint_before_collapse(harness_dir: Path) -> bool:
    """Returns True if lint passes (no errors)."""
    from td_linter import lint, Severity

    violations = lint(harness_dir)
    errors = [v for v in violations if v.severity == Severity.ERROR]

    if errors:
        print(f"ERROR: {len(errors)} linting errors")
        for v in errors:
            print(f"  {v.path}: {v.message}")
        return False

    warnings = [v for v in violations if v.severity == Severity.WARNING]
    if warnings:
        print(f"WARNING: {len(warnings)} warnings (continuing)")

    return True
```

### API Design

```python
# td_linter/api.py

def lint(
    path: str | Path,
    config_path: str | Path | None = None,
    pre_collapse: bool = False,
    validate_expressions: bool = False,
) -> list[Violation]:
    """
    Lint a .toe.dir directory.

    Args:
        path: Path to .toe.dir directory
        config_path: Optional config file
        pre_collapse: Enable relaxed pre-collapse mode
        validate_expressions: Enable expression validation

    Returns:
        List of violations found
    """
```

### Skip Option

Allow bypassing lint when needed:
```bash
python build_test_harness.py collapse --skip-lint
```

Document when this is appropriate (e.g., known issues being fixed separately).

### Existing Script Location

Find and study existing build_test_harness.py:
```
docs/touchdesigner/scripts/build_test_harness.py
```

Understand its structure before integrating.

### Error Reporting

When lint fails:
- List all errors with file paths
- Suggest `--skip-lint` if user wants to proceed anyway
- Return non-zero exit code

### Testing Integration

- Test with clean project (should pass)
- Test with errors (should abort)
- Test with warnings only (should continue)
- Test --skip-lint flag

## Definition of Done

All acceptance criteria checked. build_test_harness.py runs lint before collapse.
