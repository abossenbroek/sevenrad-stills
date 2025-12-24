# Max Linter

LSP-based linter for Max/MSP GenExpr and GLSL shaders using real language server validation.

## Overview

This package uses actual LSP (Language Server Protocol) communication to validate shader code:

- **glsl_analyzer** - GLSL shader validation
- **clangd** - GenExpr (C-like) syntax validation

## Installation

```bash
cd linter
uv sync
```

### Required LSP Servers

Install the language servers before use:

```bash
# GLSL analyzer
brew install glsl_analyzer

# clangd (usually included with Xcode)
xcode-select --install
# Or install via Homebrew
brew install llvm
```

## Usage

```bash
# Validate genjit files
uv run max-lint ../code/*.genjit

# Check if LSP servers are available
uv run max-lint --check-lsp

# Verbose output
uv run max-lint -v ../code/*.genjit

# Skip validation if LSP unavailable
uv run max-lint --fallback ../code/*.genjit
```

## How It Works

1. **Extract shader code** from .genjit files (JSON patcher format)
2. **Detect language** (GenExpr or GLSL) based on content markers
3. **Validate with LSP**:
   - GLSL shaders → glsl_analyzer
   - GenExpr code → clangd (wrapped in C context)
4. **Report diagnostics** with line numbers and severity

## GenExpr Validation

GenExpr is validated using clangd by:

1. Wrapping code in a C function context
2. Adding type stubs for GenExpr built-ins (vec4, sample, norm, etc.)
3. Filtering diagnostics to remove false positives about GenExpr constructs

## Package Structure

```
linter/
├── pyproject.toml
├── README.md
└── src/
    └── max_linter/
        ├── __init__.py
        ├── cli.py              # Command-line interface
        ├── lsp_client.py       # LSP communication
        ├── results.py          # Diagnostic types
        ├── validators/
        │   ├── __init__.py
        │   ├── glsl.py         # glsl_analyzer wrapper
        │   └── clangd.py       # clangd wrapper
        └── extractors/
            ├── __init__.py
            └── genjit.py       # Extract from .genjit files
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Formatting
uv run ruff format src/
```

## License

Part of the SevenRad Max Externals package.
