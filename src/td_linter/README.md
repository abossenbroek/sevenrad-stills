# td-linter

A validation tool for TouchDesigner `.toe.dir` expanded projects. Catches structural errors, broken references, type mismatches, and embedded code issues before projects are collapsed back to binary format.

## Key Features

- **Grammar-based parsing** - Validates `.n` and `.parm` files using formal Lark grammars
- **Graph analysis** - Uses NetworkX to detect cycles, dangling inputs, and type mismatches
- **Embedded code validation** - Checks GLSL shaders and Python expressions
- **Multiple output formats** - Text (human-readable), JSON, and SARIF (CI/CD integration)
- **LSP server** - Real-time editor integration for VS Code, Neovim, and more
- **Watch mode** - Continuous linting during development
- **Auto-fix** - Automatically fix certain violations
- **Plugin system** - Extend with custom rules

## Quick Start

### Installation

```bash
# Basic installation
pip install sevenrad-stills

# With watch mode (file monitoring)
pip install sevenrad-stills[td-linter-watch]

# With LSP server (editor integration)
pip install sevenrad-stills[td-linter-lsp]

# All features
pip install sevenrad-stills[td-linter-all]
```

### Basic Usage

```bash
# Lint a project
td-linter lint path/to/project.toe.dir

# Create a configuration file
td-linter init

# List available rules
td-linter rules
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

## Commands

| Command | Description |
|---------|-------------|
| `lint <path>` | Validate a `.toe.dir` project |
| `fix <path>` | Auto-fix violations |
| `watch <path>` | Continuously lint on file changes |
| `rules` | List available lint rules |
| `init` | Create a `td-linter.yaml` config file |
| `lsp` | Start the Language Server Protocol server |
| `version` | Show version information |

### Common Options

```bash
# Verbose output
td-linter lint project.toe.dir --verbose

# Select specific rules/categories
td-linter lint project.toe.dir --select S,C,G001

# Ignore rules/categories
td-linter lint project.toe.dir --ignore F

# Exit code 1 on warnings (for CI)
td-linter lint project.toe.dir --fail-on-warning

# Output as JSON
td-linter lint project.toe.dir --format json

# Output as SARIF (GitHub/GitLab integration)
td-linter lint project.toe.dir --format sarif
```

## Configuration

Create a `td-linter.yaml` file with `td-linter init`:

```yaml
version: "1.0.0"

# Extend a preset
extends: recommended

# Enable only these rules/categories
select: [S, C, T, R]

# Disable these rules/categories
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

## Rule Categories

| Code | Category | Description |
|------|----------|-------------|
| S | Syntax | File parsing and syntax validation |
| C | Connection | Operator connections, cycles, dangling inputs |
| T | Type | Type compatibility between operators |
| R | Reference | Operator references and dependencies |
| G | GLSL | Embedded GLSL shader validation |
| P | Python | Embedded Python expression validation |
| F | Performance | Performance optimization suggestions |

## Programmatic Usage

```python
from td_linter import lint, lint_and_check, Severity

# Basic linting
violations = lint("myproject.toe.dir")
for v in violations:
    print(f"{v.severity}: {v.message}")

# Check with pass/fail status
passed, violations = lint_and_check(
    "myproject.toe.dir",
    fail_on_warning=True
)
if not passed:
    sys.exit(1)

# With configuration
violations = lint(
    "myproject.toe.dir",
    config_path="td-linter.yaml",
    select=["S", "C"],
    ignore=["F003"]
)
```

## Editor Integration

td-linter includes an LSP server for real-time feedback in editors:

```bash
# Start LSP server (for editor integration)
td-linter lsp

# TCP mode for debugging
td-linter lsp --transport tcp --port 2087
```

**Supported editors:**
- VS Code
- Neovim (via nvim-lspconfig)
- Sublime Text (via LSP package)
- Emacs (via lsp-mode)

See the [LSP Integration Guide](../docs/touchdesigner/guides/td-linter-lsp-integration.md) for setup instructions.

## CI/CD Integration

### GitHub Actions

```yaml
- name: Lint TouchDesigner project
  run: |
    pip install sevenrad-stills
    td-linter lint project.toe.dir --format sarif > results.sarif

- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: results.sarif
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: td-linter
        name: td-linter
        entry: td-linter lint
        language: system
        files: '\.toe\.dir/'
        pass_filenames: false
```

## Watch Mode

Continuously lint during development:

```bash
# Basic watch
td-linter watch project.toe.dir

# With auto-fix
td-linter watch project.toe.dir --fix

# Adjust debounce delay
td-linter watch project.toe.dir --debounce 0.3
```

## Auto-Fix

Automatically fix certain violations:

```bash
# Preview fixes
td-linter fix project.toe.dir --dry-run

# Apply fixes
td-linter fix project.toe.dir

# Verbose output
td-linter fix project.toe.dir --verbose
```

Currently fixable rules:
- `G002`: Removes `#version` directives from GLSL shaders

## Writing Custom Rules

Create plugins to extend td-linter with custom rules:

```python
from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation

class NoEmptyContainers(LintRule):
    @property
    def rule_id(self) -> str:
        return "CUSTOM001"

    @property
    def name(self) -> str:
        return "no-empty-containers"

    @property
    def description(self) -> str:
        return "COMP containers should have children"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        for node_path in graph.nodes:
            data = graph.nodes[node_path]
            if data.get("family") == "COMP":
                if not list(graph.successors(node_path)):
                    yield Violation(
                        rule=self.rule_id,
                        message="Empty COMP container",
                        path=node_path,
                        severity="warning",
                    )
```

Load plugins in `td-linter.yaml`:

```yaml
plugins:
  - path: ./my_rules.py
```

See the [Plugin Guide](../docs/touchdesigner/guides/td-linter-plugins.md) for details.

## Documentation

- [Quickstart Guide](../docs/touchdesigner/guides/td-linter-quickstart.md) - Getting started
- [Watch Mode Guide](../docs/touchdesigner/guides/td-linter-watch-mode.md) - Continuous linting
- [Auto-Fix Guide](../docs/touchdesigner/guides/td-linter-auto-fix.md) - Automatic fixes
- [Plugin Guide](../docs/touchdesigner/guides/td-linter-plugins.md) - Custom rules
- [LSP Integration](../docs/touchdesigner/guides/td-linter-lsp-integration.md) - Editor setup
- [Rules Catalog](../docs/touchdesigner/reference/td-linter-rules-catalog.md) - All rules explained
- [API Reference](../docs/touchdesigner/reference/td-linter-api.md) - Python API

## Requirements

- Python 3.10+
- For GLSL validation: `glslangValidator` (from Vulkan SDK or standalone)

## License

See the main project license.
