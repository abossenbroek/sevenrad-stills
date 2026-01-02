# Max Linter

Validates Max/MSP patchers, help files, and GenExpr shaders for common issues before they cause problems in Max.

## Quick Start

```bash
cd max-externals

# Validate all help patchers
make test-help-lint

# Validate specific files
cd linter
uv run max-lint ../help/sr.noise.maxhelp
uv run max-lint ../examples/*.maxpat
```

## What It Checks

### Help Patchers (.maxhelp) and Patchers (.maxpat)

| Check | Description |
|-------|-------------|
| **Context naming** | OpenGL context names must use underscores (`sr_effect_ctx`), not dots |
| **Initialization order** | `jit.world` must be banged before `jit.movie` uses the context |
| **GPU signal flow** | Complete pipeline: `qmetro` → `jit.movie` → `jit.gl.pix` → `jit.pwindow` |
| **Type mismatches** | Matrix outputs can't connect to texture inputs without conversion |
| **Parameter UI** | All shader params should have controls connected via messages |
| **Dial ranges** | Dial min/max/mult must match shader parameter bounds |
| **UI overlaps** | Detects overlapping UI elements that may hide controls |
| **Dead code** | Finds orphaned objects not connected to signal flow |
| **Feedback loops** | Warns about potential infinite loops in signal routing |

### GenExpr Shaders (.genjit)

| Check | Description |
|-------|-------------|
| **Syntax validation** | Parser validates GenExpr grammar |
| **Undefined variables** | References to undeclared variables |
| **Parameter ranges** | Default values must be within min/max bounds |
| **Function calls** | Validates arguments to built-in functions |

## Basic Usage

### From Makefile (Recommended)

```bash
# Lint help patchers (strict mode - warnings are errors)
make test-help-lint

# Lint with verbose output
make test-help-lint-verbose

# Lenient mode (warnings don't fail)
make test-help-lint-lenient

# Lint GenExpr shaders
make test-lint
```

### Direct CLI Usage

```bash
cd linter

# Lint maxhelp files
uv run max-lint ../help/*.maxhelp

# Lint maxpat files
uv run max-lint ../examples/*.maxpat

# Lint both in a directory
uv run max-lint ../help/ ../examples/

# Verbose output (shows passing files)
uv run max-lint -v ../help/*.maxhelp

# Strict mode (warnings become errors)
uv run max-lint --strict ../help/*.maxhelp
```

## Output Formats

The linter supports multiple output formats for different use cases:

### Text (Default)

Human-readable output for terminal use:

```bash
uv run max-lint ../help/sr.noise.maxhelp
```

```
help/sr.noise.maxhelp:
  [ERROR] context-naming (obj-1): Context name 'sr.noise.ctx' contains dots
  [WARNING] dial-range (obj-5): Dial max (100) exceeds param max (1.0)
```

### JSON

Machine-parseable output for tooling:

```bash
uv run max-lint --output-format json ../help/*.maxhelp
```

```json
{
  "results": [
    {
      "file": "help/sr.noise.maxhelp",
      "errors": [{"severity": "ERROR", "rule": "context-naming", ...}],
      "warnings": [],
      "status": "error"
    }
  ],
  "summary": {"total_files": 5, "files_with_errors": 1, "success": false}
}
```

### YAML

Structured output for configuration and reports:

```bash
uv run max-lint --output-format yaml ../help/*.maxhelp
```

### GitHub Actions

Workflow commands that create annotations in pull requests:

```bash
uv run max-lint --output-format github ../help/*.maxhelp
```

```
::error file=help/sr.noise.maxhelp,title=context-naming::Context name contains dots (obj-1)
::warning file=help/sr.noise.maxhelp,title=dial-range::Dial max exceeds param max (obj-5)
```

## CI Integration

### GitHub Actions Workflow

```yaml
name: Lint Max Patchers

on: [push, pull_request]

jobs:
  lint:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: cd max-externals/linter && uv sync

      - name: Lint patchers
        run: |
          cd max-externals/linter
          uv run max-lint --output-format github --strict ../help/*.maxhelp ../examples/*.maxpat
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: max-lint
        name: Lint Max patchers
        entry: bash -c 'cd max-externals/linter && uv run max-lint --strict ../help/*.maxhelp'
        language: system
        files: \.(maxhelp|maxpat)$
        pass_filenames: false
```

## Common Errors and Fixes

### Context Naming

```
[ERROR] context-naming: Context name 'sr.effect.ctx' contains dots
```

**Fix:** Rename context to use underscores: `sr_effect_ctx`

```
jit.world sr_effect_ctx @visible 0
```

### Missing output_texture

```
[ERROR] type-004: jit.movie connected to GPU pipeline but missing @output_texture 1
```

**Fix:** Add `@output_texture 1` to jit.movie for GPU texture output:

```
jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_effect_ctx
```

### Type Mismatch

```
[ERROR] type-002: Matrix outlet cannot connect to texture inlet
```

**Fix:** Ensure jit.movie outputs GPU textures when feeding jit.gl.pix:

```
jit.movie @output_texture 1 → jit.gl.pix
```

### Parameter Not Initialized

```
[ERROR] init-004: Parameter 'amount' control not initialized from loadbang
```

**Fix:** Connect a loadbang through a message to set initial value:

```
loadbang → [0.5] → [amount $1] → jit.gl.pix
```

### Dial Range Mismatch

```
[WARNING] dial-001: Dial range [0-127] doesn't cover param range [0.0-1.0]
```

**Fix:** Adjust dial attributes to match shader parameter:

```
dial @min 0. @size 100 @mult 0.01  // Outputs 0.0-1.0
```

## CLI Reference

```
max-lint [OPTIONS] FILES...

Arguments:
  FILES                  Files or directories to lint (.genjit, .maxhelp, .maxpat)

Options:
  --maxhelp              Lint only .maxhelp files
  --maxpat               Lint only .maxpat files
  --strict               Treat warnings as errors
  -v, --verbose          Show info messages and passing files
  -f, --output-format    Output format: text, json, yaml, github
  --fallback             Skip LSP validation if servers unavailable
  --check-lsp            Check if LSP servers are available
  --c-external           Lint C external source files
  --version              Show version
  --help                 Show help
```

## LSP Server Setup

For GenExpr and GLSL validation, install language servers:

```bash
# GLSL analyzer
brew install glsl_analyzer

# clangd (usually included with Xcode)
xcode-select --install

# Check availability
cd linter && uv run max-lint --check-lsp
```

## Development

```bash
cd linter

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Formatting
uv run ruff format src/
```

## Architecture

```
linter/src/max_linter/
├── cli.py                    # Command-line interface
├── formatters.py             # Output formatters (text/json/yaml/github)
├── lint_error.py             # Error/warning dataclass
├── lint_graph.py             # NetworkX graph analysis
├── validators/
│   └── maxhelp/              # Modular validator mixins
│       ├── context.py        # Context naming validation
│       ├── signal_flow.py    # GPU pipeline validation
│       ├── connection.py     # Type checking
│       ├── dial.py           # Parameter range validation
│       └── ...
├── genexpr/                  # GenExpr parser and validator
└── extractors/               # File format parsers
```
