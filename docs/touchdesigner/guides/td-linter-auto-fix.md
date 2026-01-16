# td-linter Auto-Fix Guide

td-linter can automatically fix certain violations. This guide explains how to use the auto-fix feature.

## Basic Usage

```bash
td-linter fix path/to/project.toe.dir
```

This scans the project, identifies fixable violations, and applies fixes.

## Dry-Run Mode

Preview what would be fixed without modifying files:

```bash
td-linter fix project.toe.dir --dry-run
```

### Output

```
Dry-run mode: No files will be modified

Would fix 3 violations:
  [G002] /project/ops/shader.text:1 - Remove GLSL version directive
  [G002] /project/ops/effect.text:1 - Remove GLSL version directive
  [G002] /project/ops/post.text:1 - Remove GLSL version directive

Summary: 3 would be fixed, 5 skipped (not fixable)
```

## Verbose Mode

See detailed information about each fix:

```bash
td-linter fix project.toe.dir --verbose
```

### Output

```
Applying fixes...

[1/3] G002: /project/ops/shader.text
  Line 1: Removing "#version 450"
  ✓ Applied

[2/3] G002: /project/ops/effect.text
  Line 1: Removing "#version 330"
  ✓ Applied

[3/3] G002: /project/ops/post.text
  Line 1: Removing "#version 450 core"
  ✓ Applied

Summary: 3 applied, 0 failed, 5 skipped
```

## Fixable Rules

Not all rules support auto-fix. Currently fixable rules:

| Rule | Name | Description |
|------|------|-------------|
| G002 | no-glsl-version | Removes `#version` directives from GLSL shaders |

Rules that are NOT fixable (require manual intervention):
- Cycle detection (C001) - structural changes needed
- Missing references (C002) - need to create operators
- Type mismatches (T001, T002) - design decisions required

## Rule Selection

Fix only specific rules:

```bash
# Fix only GLSL rules
td-linter fix project.toe.dir --select G

# Fix everything except performance rules
td-linter fix project.toe.dir --ignore F
```

## Configuration

Use a config file to control fix behavior:

```bash
td-linter fix project.toe.dir --config my-config.yaml
```

## Watch Mode Integration

Auto-fix can be combined with watch mode:

```bash
td-linter watch project.toe.dir --fix
```

This automatically applies fixes after each file change.

## Programmatic Usage

For build scripts or CI/CD:

```python
from td_linter.linter import run_lint
from td_linter.fix import FixApplier

# Get violations
violations = list(run_lint(Path("project.toe.dir")))

# Apply fixes
applier = FixApplier(dry_run=False)
result = applier.apply(violations)

print(f"Applied: {result.success_count}")
print(f"Failed: {result.failure_count}")
print(f"Skipped: {result.skipped_count}")
```

## How Fixes Work

### Fix Structure

Each fix contains:
1. **Description**: What the fix does
2. **Replacements**: List of text changes

### Replacement Details

Each replacement specifies:
- **file_path**: File to modify
- **start_line**: First line to replace
- **end_line**: Last line to replace
- **start_col**: Start column (optional)
- **end_col**: End column (optional)
- **new_text**: Replacement text

### Application Order

Fixes are applied bottom-up (highest line numbers first) to preserve line numbers for subsequent fixes.

## Safety Features

1. **Non-destructive**: Only fixable violations are modified
2. **Atomic per-file**: All fixes for a file succeed or fail together
3. **Detailed reporting**: Every action is logged
4. **Dry-run available**: Preview before applying

## Troubleshooting

### "No fixable violations found"

The violations found don't have auto-fix support. Check which rules are fixable.

### "Fix failed"

Check verbose output for the specific error:

```bash
td-linter fix project.toe.dir --verbose
```

Common issues:
- File permissions
- File was modified during fix
- Invalid line numbers (corrupted violation data)

### Fix Applied But Issue Remains

Some fixes may need follow-up:
1. Run lint again to check
2. Some issues require multiple passes
3. Some violations are similar but not identical
