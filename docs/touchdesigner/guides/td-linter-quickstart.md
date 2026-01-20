# td-linter Quickstart Guide

This guide will help you get started with td-linter, a validation tool for TouchDesigner projects.

## System Requirements

- **Python**: 3.10 or higher
- **TouchDesigner**: Any version that supports `toeexpand`/`toecollapse` (required for direct `.toe` file linting)
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

## Basic Usage

td-linter can lint both binary `.toe` files and expanded `.toe.dir` directories.

### Linting .toe Files Directly (Recommended)

The easiest way to use td-linter is to lint `.toe` files directly:

```bash
td-linter lint project.toe
```

This automatically:
1. Finds your TouchDesigner installation
2. Expands the `.toe` file to a temporary `.toe.dir` directory
3. Validates the expanded project
4. Cleans up the temporary files

### TouchDesigner Path Discovery

td-linter automatically finds TouchDesigner in this order:
1. `--td-path` CLI option
2. `TOUCHDESIGNER_PATH` environment variable
3. Common installation locations:
   - **macOS**: `/Applications/TouchDesigner*.app/Contents/MacOS/`
   - **Windows**: `C:\Program Files\Derivative\TouchDesigner*\bin\`
   - **Linux**: `/opt/TouchDesigner*/bin/`
4. `toeexpand`/`toecollapse` in PATH

### Specifying TouchDesigner Path

If auto-discovery fails, specify the path explicitly:

```bash
# Via CLI option
td-linter lint project.toe --td-path /Applications/TouchDesigner.app/Contents/MacOS

# Via environment variable
export TOUCHDESIGNER_PATH=/Applications/TouchDesigner.app/Contents/MacOS
td-linter lint project.toe
```

### Debugging with --keep-files-after-expand

To inspect the expanded files after linting (useful for debugging):

```bash
td-linter lint project.toe --keep-files-after-expand
```

This keeps the `.toe.dir` directory after linting completes. The path is printed to stdout.

### Linting .toe.dir Directories

You can also lint pre-expanded directories directly:

```bash
td-linter lint project.toe.dir
```

This is useful when:
- You're working with version-controlled `.toe.dir` projects
- TouchDesigner isn't installed on the current machine
- You want to avoid repeated expand/collapse cycles

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
# Lint a .toe file directly (recommended)
td-linter lint path/to/your/project.toe

# Or lint an expanded .toe.dir directory
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
# Lint a .toe file directly
td-linter lint project.toe

# Keep expanded files for debugging
td-linter lint project.toe --keep-files-after-expand

# Specify TouchDesigner path
td-linter lint project.toe --td-path /path/to/TouchDesigner/bin

# Verbose output
td-linter lint project.toe --verbose

# Quiet (errors only)
td-linter lint project.toe --quiet

# Exit code 1 on warnings (for CI)
td-linter lint project.toe --fail-on-warning

# Use specific config file
td-linter lint project.toe --config my-config.yaml

# Select only syntax rules
td-linter lint project.toe --select S

# Ignore performance rules
td-linter lint project.toe --ignore F
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

### "TouchDesigner not found"

```
Error: TouchDesigner not found. Set TOUCHDESIGNER_PATH environment variable or use --td-path option.
```

**Solution:** td-linter needs TouchDesigner to expand `.toe` files. Either:
1. Install TouchDesigner
2. Set `TOUCHDESIGNER_PATH` environment variable to your TD installation
3. Use `--td-path /path/to/TouchDesigner/bin`
4. Add TouchDesigner's bin directory to your PATH

See [TouchDesigner Path Discovery](#touchdesigner-path-discovery) for details.

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
