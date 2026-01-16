# TDL-051: Pre-commit Hook

---
id: TDL-051
status: done
priority: high
phase: 5
depends_on: [TDL-050]
blocks: []
---

## Problem Statement

Developers should be able to catch linting errors before committing. Pre-commit is the standard framework for Git hooks. The linter needs a pre-commit hook configuration that runs on .toe.dir changes.

## Acceptance Criteria

- [ ] Hook configuration for .pre-commit-config.yaml
- [ ] Hook runs only on .toe.dir file changes
- [ ] Hook fails commit on errors (exit code 1)
- [ ] Hook allows warnings (configurable)
- [ ] Documentation for hook setup
- [ ] Test hook behavior locally

## Files to Create

```
# Project documentation
docs/
└── precommit_setup.md

# Example .pre-commit-config.yaml
.pre-commit-config.yaml.example
```

## Research Pointers

### Pre-commit Framework

- https://pre-commit.com/
- Hooks defined in `.pre-commit-config.yaml`
- Can use local repos or remote repos

### Hook Configuration Options

**Option 1: Local hook** (for development)
```yaml
repos:
  - repo: local
    hooks:
      - id: td-linter
        name: Lint TouchDesigner projects
        entry: td-linter lint
        language: python
        files: \.toe\.dir/
        pass_filenames: false
```

**Option 2: Remote hook** (for published package)
```yaml
repos:
  - repo: https://github.com/yourorg/td-linter
    rev: v1.0.0
    hooks:
      - id: td-linter
```

### File Matching

The `files` regex determines when hook runs:
- `\.toe\.dir/` - Matches any file inside .toe.dir directories
- Could be more specific: `\.toe\.dir/.*\.(n|parm|text)$`

### Pass Filenames

`pass_filenames: false` means run once on whole project, not per-file. This is appropriate because:
- .toe.dir is validated as a whole
- Graph validation needs all files

### Exit Codes

Pre-commit interprets:
- 0: Success, allow commit
- 1: Failure, block commit

Our linter should:
- Exit 0 if no errors (warnings OK)
- Exit 1 if any errors
- Optional: `--fail-on-warning` to also block on warnings

### Performance Consideration

Pre-commit runs on every commit. For large projects:
- Consider `--pre-collapse` mode for faster validation
- Document expected runtime

### Additional Configuration

Consider hook arguments:
```yaml
hooks:
  - id: td-linter
    args: ['--quiet', '--pre-collapse']
```

### Testing the Hook

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test manually
pre-commit run td-linter --all-files
```

### Documentation Content

Document:
1. Installing pre-commit
2. Adding hook to config
3. Running manually
4. Skipping hook when needed (`--no-verify`)
5. Common issues

## Definition of Done

All acceptance criteria checked. Hook blocks commits with errors, documentation complete.
