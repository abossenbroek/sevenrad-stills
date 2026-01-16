# TDL-052: GitHub Actions Workflow

---
id: TDL-052
status: done
priority: high
phase: 5
depends_on: [TDL-050]
blocks: []
---

## Problem Statement

CI/CD pipelines should automatically lint .toe.dir changes on push and pull request. GitHub Actions is the standard CI for GitHub repos. The workflow should produce SARIF output for GitHub Code Scanning integration.

## Acceptance Criteria

- [ ] Workflow triggers on .toe.dir file changes
- [ ] Workflow installs td-linter and glslang
- [ ] Workflow runs lint with SARIF output
- [ ] SARIF uploads to GitHub Code Scanning
- [ ] Violations appear as PR annotations
- [ ] Workflow example documented

## Files to Create

```
.github/workflows/
└── td-lint.yml           # Workflow definition

docs/
└── github_actions_setup.md
```

## Research Pointers

### GitHub Actions Basics

- https://docs.github.com/en/actions
- Workflows in `.github/workflows/*.yml`
- Triggers: push, pull_request, workflow_dispatch

### Trigger Configuration

```yaml
on:
  push:
    paths:
      - '**/*.toe.dir/**'
  pull_request:
    paths:
      - '**/*.toe.dir/**'
```

Only runs when .toe.dir files change.

### Runner Selection

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
```

Options: ubuntu-latest, macos-latest, windows-latest

Consider: Does linter need macOS for specific features?

### Installing Dependencies

```yaml
steps:
  - uses: actions/checkout@v4

  - name: Set up Python
    uses: actions/setup-python@v5
    with:
      python-version: '3.11'

  - name: Install td-linter
    run: pip install td-linter

  - name: Install glslang
    run: sudo apt-get install -y glslang-tools
```

### Running Lint

```yaml
  - name: Lint .toe.dir
    run: |
      for dir in $(find . -name '*.toe.dir' -type d); do
        td-linter lint "$dir" --format sarif >> results.sarif
      done
```

Or with dedicated script for robustness.

### SARIF Upload

```yaml
  - name: Upload SARIF
    uses: github/codeql-action/upload-sarif@v3
    with:
      sarif_file: results.sarif
```

This integrates with GitHub Code Scanning.

### PR Annotations

When SARIF is uploaded:
- GitHub shows violations as annotations on PR diff
- Security tab shows all violations
- Can require clean scans for merge

### Workflow Example

```yaml
name: TouchDesigner Lint

on:
  push:
    paths: ['**/*.toe.dir/**']
  pull_request:
    paths: ['**/*.toe.dir/**']

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install td-linter

      - run: sudo apt-get install -y glslang-tools

      - name: Lint
        run: |
          find . -name '*.toe.dir' -type d -exec td-linter lint {} --format sarif \; > results.sarif

      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### Testing Workflow

- Test on a feature branch
- Verify SARIF uploads correctly
- Check PR annotations appear

## Definition of Done

All acceptance criteria checked. Workflow runs, uploads SARIF, shows annotations.
