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

## Fix Safety

### Content Hash Verification

td-linter verifies files haven't changed before applying fixes:

1. When a violation is created, the file's SHA-256 hash is recorded
2. Before applying a fix, the current hash is compared
3. If hashes differ, the fix is skipped to prevent data corruption

This prevents issues when:
- You edit a file after running lint
- Another process modifies the file
- Multiple lint runs overlap

### Path Traversal Protection

All file paths in fixes are validated:
- Resolved to absolute paths
- Checked against project boundary
- Symlinks fully resolved

This prevents malicious fixes from escaping the project directory.

### Disable Safety Checks

For automated pipelines where you're confident in the state:

```python
from td_linter.fix import FixApplier

applier = FixApplier(
    verify_hashes=False,  # Skip hash verification
)
```

**Warning:** Only disable safety checks in controlled environments.

## Creating Fixable Rules

Plugin authors can create rules that provide automatic fixes.

### Fix Structure

```python
from td_linter.rules.base import LintRule, Violation, Fix, Replacement

class MyFixableRule(LintRule):
    @property
    def fixable(self) -> bool:
        return True  # Mark as fixable

    def check(self, graph):
        for node in graph.nodes:
            if issue_found:
                yield Violation(
                    rule=self.rule_id,
                    message="Issue description",
                    path=node,
                    fix=Fix(
                        description="What the fix does",
                        replacements=[
                            Replacement(
                                file_path=source_file,
                                start_line=5,  # 1-indexed
                                end_line=6,    # Exclusive
                                new_text="replacement content",
                            )
                        ]
                    )
                )
```

### Replacement Types

**Full line replacement:**
```python
Replacement(
    file_path=path,
    start_line=5,
    end_line=6,  # Replaces line 5
    new_text="new line content\n",
)
```

**Column-specific (single line):**
```python
Replacement(
    file_path=path,
    start_line=5,
    end_line=5,
    start_col=10,  # 0-indexed
    end_col=20,    # Exclusive
    new_text="replaced",
)
```

**Delete lines:**
```python
Replacement(
    file_path=path,
    start_line=5,
    end_line=8,  # Deletes lines 5-7
    new_text="",
)
```

**Insert lines:**
```python
Replacement(
    file_path=path,
    start_line=5,
    end_line=5,  # Insert before line 5
    new_text="new line 1\nnew line 2\n",
)
```

### Adding Content Hash

For safety, include the file's hash when creating the fix:

```python
from td_linter.fix import FixApplier

content_hash = FixApplier.compute_file_hash(source_file)

Replacement(
    file_path=source_file,
    start_line=5,
    end_line=6,
    new_text="fixed content",
    content_hash=content_hash,  # Enables verification
)
```

## Batch Fixing

### Multiple Projects

Fix all projects in a directory:

```bash
#!/bin/bash
for project in *.toe.dir; do
    echo "Fixing: $project"
    td-linter fix "$project" --verbose
done
```

### Python Script

```python
#!/usr/bin/env python3
"""Batch fix multiple projects."""

from pathlib import Path
from td_linter import lint
from td_linter.fix import FixApplier

def fix_all_projects(root_dir: Path):
    results = {}

    for project in root_dir.glob("*.toe.dir"):
        print(f"Processing: {project}")

        violations = lint(project)
        fixable = [v for v in violations if v.fix is not None]

        if not fixable:
            results[project.name] = {"fixed": 0, "skipped": 0}
            continue

        applier = FixApplier(project_root=project)
        result = applier.apply(fixable)

        results[project.name] = {
            "fixed": result.success_count,
            "skipped": result.skipped_count,
            "failed": result.failure_count,
        }

    return results

if __name__ == "__main__":
    results = fix_all_projects(Path("."))
    for project, counts in results.items():
        print(f"{project}: {counts}")
```

### CI/CD Integration

```yaml
# .github/workflows/fix.yml
name: Auto-Fix

on:
  push:
    paths:
      - '**.toe.dir/**'

jobs:
  fix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install td-linter
        run: pip install sevenrad-stills

      - name: Apply fixes
        run: |
          for project in *.toe.dir; do
            td-linter fix "$project"
          done

      - name: Commit fixes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add .
          git diff --staged --quiet || git commit -m "Auto-fix td-linter violations"
          git push
```
