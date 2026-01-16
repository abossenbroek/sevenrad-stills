# td-linter Quickstart Guide

This guide will help you get started with td-linter, a validation tool for TouchDesigner `.toe.dir` projects.

## Installation

td-linter is installed as part of the sevenrad-stills package:

```bash
pip install sevenrad-stills
```

For optional features:

```bash
# Watch mode (file monitoring)
pip install sevenrad-stills[td-linter-watch]

# LSP server (editor integration)
pip install sevenrad-stills[td-linter-lsp]

# All optional features
pip install sevenrad-stills[td-linter-all]
```

## Prerequisites

Before using td-linter, you need an expanded TouchDesigner project:

1. Open your `.toe` file in TouchDesigner
2. Use `toeexpand` to export it as a `.toe.dir` directory
3. The `.toe.dir` contains human-readable text files that td-linter validates

## First Lint Run

Run td-linter on your project:

```bash
td-linter lint path/to/your/project.toe.dir
```

### Example Output

```
╭─────────────────────────────────────────────────────────────────────────────╮
│ td-linter results for project.toe.dir                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Violations: 2 errors, 3 warnings, 0 info                                    │
╰─────────────────────────────────────────────────────────────────────────────╯

Error C001: no-invalid-cycles
  Operator cycle detected: op1 → op2 → op1
  Path: /project/op1

Warning C002: no-dangling-inputs
  Input references non-existent operator: missing_op
  Path: /project/op3
```

## Understanding the Output

Each violation includes:
- **Severity**: error, warning, or info
- **Rule ID**: Category code + number (e.g., C001)
- **Rule Name**: Human-readable name
- **Message**: Description of the issue
- **Path**: Operator path where the issue was found

### Rule Categories

| Code | Category | Description |
|------|----------|-------------|
| S | Syntax | File parsing errors |
| C | Connection | Operator connections and cycles |
| T | Type | Type compatibility between operators |
| R | Reference | Operator references and dependencies |
| G | GLSL | Embedded GLSL shader validation |
| P | Python | Embedded Python expression validation |
| F | Performance | Performance optimization rules |

## Configuration

Create a configuration file to customize td-linter:

```bash
td-linter init
```

This creates `td-linter.yaml`:

```yaml
version: "1.0.0"
extends: recommended

# Enable specific rules
select: [S, C, T, R]

# Disable specific rules
ignore: [F]

# Per-rule configuration
rules:
  F001:
    enabled: true
    severity: warning
    options:
      max_depth: 15
```

### Presets

| Preset | Description |
|--------|-------------|
| `recommended` | Sensible defaults for most projects |
| `strict` | All rules enabled with stricter thresholds |
| `minimal` | Only crash-prevention rules |
| `pedantic` | Maximum strictness |

## Output Formats

td-linter supports multiple output formats:

```bash
# Default human-readable output
td-linter lint project.toe.dir

# JSON for scripts
td-linter lint project.toe.dir --format json

# SARIF for CI/CD tools (GitHub, GitLab)
td-linter lint project.toe.dir --format sarif
```

## Common Options

```bash
# Verbose output
td-linter lint project.toe.dir --verbose

# Quiet (errors only)
td-linter lint project.toe.dir --quiet

# Exit code 1 on warnings (for CI)
td-linter lint project.toe.dir --fail-on-warning

# Use specific config file
td-linter lint project.toe.dir --config my-config.yaml

# Select only syntax rules
td-linter lint project.toe.dir --select S

# Ignore performance rules
td-linter lint project.toe.dir --ignore F
```

## List Available Rules

```bash
td-linter rules
```

Shows all available rules with:
- Rule ID
- Name
- Description
- Category
- Severity
- Enabled status

## Next Steps

- [Watch Mode Guide](td-linter-watch-mode.md) - Live re-linting on file changes
- [Auto-Fix Guide](td-linter-auto-fix.md) - Automatically fix violations
- [Plugin Guide](td-linter-plugins.md) - Write custom rules
- [LSP Integration Guide](td-linter-lsp-integration.md) - Editor integration
- [Rules Catalog](../reference/td-linter-rules-catalog.md) - All rules explained
