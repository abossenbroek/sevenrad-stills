# td-linter Quickstart Guide

This guide will help you get started with td-linter, a validation tool for TouchDesigner `.toe.dir` projects.

## System Requirements

- **Python**: 3.10 or higher
- **TouchDesigner**: Any version that supports `toeexpand`/`toecollapse`
- **GLSL Validation** (optional): `glslangValidator` from the Vulkan SDK

### Installing glslangValidator

GLSL rules (G001, G002, G003) require `glslangValidator` to validate shader syntax. Without it, GLSL validation is skipped.

**macOS (Homebrew):**
```bash
brew install glslang
```

**Ubuntu/Debian:**
```bash
apt install glslang-tools
```

**Windows:**
Download from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or [Khronos releases](https://github.com/KhronosGroup/glslang/releases).

**Verify installation:**
```bash
glslangValidator --version
```

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

**Verify installation:**
```bash
td-linter version
```

## Prerequisites

Before using td-linter, you need an expanded TouchDesigner project:

1. Open your `.toe` file in TouchDesigner
2. Use `toeexpand` to export it as a `.toe.dir` directory
3. The `.toe.dir` contains human-readable text files that td-linter validates

## Project Structure

A `.toe.dir` project has this structure:

```
myproject.toe.dir/
├── .toc              # Table of contents (manifest)
├── local/            # Local scope operators
│   ├── base1.n       # Operator definition
│   ├── base1.parm    # Parameter values
│   └── glsl1/        # GLSL TOP container
│       ├── glsl1.n
│       ├── glsl1.parm
│       └── glsl1.text  # Shader code
└── operators/
    ├── noise1.n
    ├── noise1.parm
    ├── textin1.n
    ├── textin1.parm
    └── textin1.text  # Python script
```

**File types validated:**
| Extension | Content |
|-----------|---------|
| `.n` | Operator definition (family, inputs, position) |
| `.parm` | Parameter values and expressions |
| `.text` | Embedded code (GLSL shaders or Python scripts) |
| `.toc` | Project manifest |

## First Lint Run

Run td-linter on your project:

```bash
td-linter lint path/to/your/project.toe.dir
```

### Example Output

```
Error C001: no-invalid-cycles
  Operator cycle detected: op1 -> op2 -> op1
  Path: /project/op1

Warning C002: no-dangling-inputs
  Input references non-existent operator: missing_op
  Path: /project/op3

Found 1 error, 1 warning
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
| `minimal` | Only crash-prevention rules (S, C001, T001) |
| `pedantic` | Maximum strictness, all rules as errors |

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

Filter by category:
```bash
td-linter rules --category C
```

## Troubleshooting

### "Path not found"

```
Error: Path not found: project.toe.dir
```

**Solution:** Check the path exists and is spelled correctly.

### "Not a directory"

```
Error: Not a directory: project.toe
```

**Solution:** td-linter validates expanded `.toe.dir` directories, not binary `.toe` files. Use `toeexpand` first.

### "glslangValidator not found"

GLSL rules are skipped if `glslangValidator` is not installed. Install it (see System Requirements) or ignore GLSL rules:

```bash
td-linter lint project.toe.dir --ignore G
```

### "watchdog not installed"

Watch mode requires the optional dependency:

```bash
pip install sevenrad-stills[td-linter-watch]
```

### "pygls not installed"

LSP mode requires the optional dependency:

```bash
pip install sevenrad-stills[td-linter-lsp]
```

### Slow on Large Projects

For large projects, consider:
1. Using `--select` to run only specific rule categories
2. Ignoring performance rules: `--ignore F`
3. Using watch mode for incremental feedback

### Permission Errors on macOS

If watching files, grant terminal access to the project directory when prompted by macOS.

## Next Steps

### Immediate Productivity
1. **[Watch Mode Guide](td-linter-watch-mode.md)** - Set up continuous linting for instant feedback while editing
2. **[Auto-Fix Guide](td-linter-auto-fix.md)** - Automatically fix common issues

### Editor Integration
3. **[LSP Integration Guide](td-linter-lsp-integration.md)** - Get real-time feedback in VS Code, Neovim, or other editors

### Customization
4. **[Plugin Guide](td-linter-plugins.md)** - Write custom rules for your team's conventions
5. **[Rules Catalog](../reference/td-linter-rules-catalog.md)** - Understand what each rule checks and why

### Reference
- **[API Reference](../reference/td-linter-api.md)** - Integrate td-linter into build scripts
- **[Configuration Reference](../reference/td-linter-config.md)** - All configuration options
