---
title: Max Externals
parent: Reference
nav_order: 4
has_toc: true
---

# Max Externals: Linting & Development Guide

The `max-externals/` directory contains a Max 9 package with GPU shader effects ported from the Taichi pipeline. Static analysis tools validate help patchers and GenExpr shaders before runtime, catching configuration errors that would otherwise cause silent failures in Max.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Quick Start](#quick-start)
3. [Help Patcher Rules](#help-patcher-rules)
4. [GenExpr Shader Rules](#genexpr-shader-rules)
5. [Developer Guide](#developer-guide)

---

## Quick Reference

| File Type | Linter | Rules | Common Issues |
|-----------|--------|-------|---------------|
| `.maxhelp` | `lint_maxhelp.py` | 23 | Signal flow breaks, dial range mismatches |
| `.genjit` | `lint_genjit.py` | 15+ | Syntax errors, missing output assignment |

---

## Quick Start

### Running the Linters

```bash
cd max-externals

# Validate all help patchers (strict mode)
make test-help-lint

# Validate all GenExpr shaders
make test-lint

# Verbose output (help patchers only)
make test-help-lint-verbose

# Strict mode for GenExpr (warnings as errors)
make test-lint-strict
```

### Understanding Output

**ERROR** — Validation failed. Fix before committing.
```
ERROR [signal-flow] sr.corruption.maxhelp: No path from jit.movie to jit.pwindow
```

**WARNING** — Potential issue, but passes in normal mode.
```
WARNING [inlet-connections] sr.maskgen.maxhelp: jit.gl.pix inlet 1 not connected
```

### Strict Mode

In strict mode, warnings become errors:

```bash
make test-help-lint        # Strict mode (default for CI)
make test-help-lint-lenient # Warnings only
```

---

## Help Patcher Rules

The help patcher linter (`lint_maxhelp.py`) validates Max help files using graph-based analysis. Rules are grouped by category.

### Signal Flow Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `signal-flow` | ERROR | GPU pipeline must connect: `qmetro` → `jit.movie` → `jit.gl.pix` → `jit.pwindow` |
| `context-init` | ERROR | `jit.world` must initialize before `jit.movie` uses the context |
| `context-naming` | WARNING | Context names should use underscores (`sr_effect_ctx`, not `sr.ctx`) |

**Common Fix: Missing Signal Flow**

The linter reports no path from `jit.movie` to display:
```
ERROR [signal-flow] sr.effect.maxhelp: No path from jit.movie to jit.pwindow
```

Check that:
1. `jit.movie` has `@output_texture 1` attribute
2. `jit.gl.pix` is connected between movie and display
3. All patchlines exist (outlet 0 → inlet 0)

### Parameter Validation Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `dial-range` | ERROR | Dial output range must not exceed shader parameter bounds |
| `c-external-dial-range` | ERROR | Dial range must match C external parameter bounds |
| `dial-float-output` | ERROR | Dials with fractional multipliers need `floatoutput: 1` |
| `parameter-ui` | WARNING | All shader parameters need UI controls and connections |
| `param-init` | WARNING | Parameter controls should be initialized from loadbang |
| `dial-init` | WARNING | Dials feeding parameters need initialization |
| `dial-init-range` | WARNING | Dial initial value should be within valid range |
| `dial-decimals` | WARNING | Fractional values need appropriate decimal display |

**Common Fix: Dial Range Mismatch**

The linter reports dial exceeds parameter bounds:
```
ERROR [dial-range] sr.corruption.maxhelp: dial outputs 0.0-1.27 but 'intensity' bounds are 0.0-1.0
```

Fix by adjusting dial attributes:
```json
{
  "size": 100,
  "mult": 0.01,
  "floatoutput": 1
}
```

This produces output range `0.0` to `1.0` (100 × 0.01).

### UI & Layout Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `overlap` | ERROR/WARNING | UI elements shouldn't overlap (>25% = error, minor = warning) |
| `inlet-connections` | WARNING | Multi-inlet objects should have all inlets connected |
| `connection-type` | WARNING | Type mismatches between connected objects |
| `display-sink-type` | WARNING | Display sinks must receive valid matrix/texture |
| `metadata` | WARNING | Patchers should have `description` and `tags` fields |

### Other Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `structure` | ERROR | Missing `patcher` key or required fields |
| `json` | ERROR | Invalid JSON syntax |
| `file` | ERROR | File doesn't exist or can't be read |
| `security` | ERROR | Shader name too long or path traversal attempt |
| `shader-name` | WARNING | Invalid characters in shader name |
| `genexpr-syntax` | ERROR | GenExpr code syntax errors (via LSP) |
| `genjit-format` | ERROR | Referenced .genjit file has format errors |

**Common Fix: Overlapping Elements**

The linter reports overlapping UI boxes:
```
ERROR [overlap] sr.maskgen.maxhelp: 'dial' overlaps 'live.dial' by 45%
```

Adjust `patching_rect` coordinates to separate the elements:
```json
"patching_rect": [100.0, 200.0, 50.0, 50.0]  // [x, y, width, height]
```

---

## GenExpr Shader Rules

The GenExpr linter (`lint_genjit.py`) validates `.genjit` shader files for syntax and structure.

### Structure Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `structure` | ERROR | Top-level must be JSON object with `patcher` key |
| `patcher-fields` | ERROR | Required: `fileversion`, `appversion`, `boxes`, `lines` |
| `boxes` | ERROR | Each box needs `box` wrapper with required fields |
| `codebox` | ERROR | Must have exactly one codebox with non-empty `code` field |
| `connections` | ERROR | Input → Codebox → Output connectivity required |

### Code Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `codebox-output` | ERROR | GenExpr must assign to `out1` (or `out`) |
| `codebox-input` | WARNING | GenExpr should reference `in1` (input texture) |
| `glsl-reserved-words` | ERROR | Cannot use GLSL keywords as variables (`half`, `uniform`, `varying`) |
| `function-definitions` | ERROR | User-defined functions not supported in GenExpr |
| `delimiters` | ERROR | Matching parentheses, brackets, braces |
| `params` | ERROR | Format: `Param name(default, min, max)` |

**Common Fix: Missing Output Assignment**

The linter reports no output assignment:
```
ERROR [codebox-output] sr.effect.genjit: GenExpr code does not assign to 'out1'
```

Ensure your shader assigns the output:
```c
// Read input channels
r = in1.r;
g = in1.g;
b = in1.b;
a = in1.a;

// Apply effect and output
out1 = vec(r * intensity, g * intensity, b * intensity, a);
```

**Common Fix: GLSL Reserved Word**

The linter reports GLSL keyword usage:
```
ERROR [glsl-reserved-words] sr.effect.genjit: 'half' is GLSL reserved, use 'half_val'
```

Rename the variable (GenExpr uses implicit typing):
```c
// Before (error)
half = 0.5;

// After (valid)
half_val = 0.5;
```

---

## Developer Guide

### Adding New Lint Rules

**Help Patcher Rules** — Edit `max-externals/tools/lint_maxhelp.py`:

The linter uses a class-based approach. Add checks inside the `MaxhelpLinter` class:

```python
def _validate_my_rule(self) -> None:
    """Check for my custom rule."""
    for box_id, box in self.boxes.items():
        if problem_found:
            self.error("my-rule", "Description of issue", box_id)
```

Register the check in `validate_file()`:
```python
self._validate_my_rule()
```

**GenExpr Rules** — Edit `max-externals/tools/lint_genjit.py`. Uses `GenjitLinter` class with `self.error()` / `self.warning()` methods.

### C External Parameter Metadata

Parameters for C externals (not GenExpr) are defined in `tools/c_external_params.json`:

```json
{
  "sr.maskgen": {
    "gap_width": {"type": "float", "min": 0.001, "max": 0.5},
    "scan_period": {"type": "int", "min": 2, "max": 100}
  },
  "sr.tilegen": {
    "tile_count": {"type": "int", "min": 1, "max": 1000},
    "seed": {"type": "int", "min": null, "max": null}
  }
}
```

The help patcher linter uses this to validate dial ranges for C externals.

### Graph-Based Validation

The help patcher linter builds a directed graph using `networkx`:

- **Nodes**: `(box_id, "in"/"out", port_num)` for each box port
- **Edges**: Patchlines with type annotations (texture, matrix, bang)
- **Analysis**: `nx.has_path()` verifies signal flow connectivity

This enables validation of:
- Complete GPU pipelines (source → effect → display)
- Initialization order (loadbang → jit.world → jit.movie)
- Parameter routing (dial → message → jit.gl.pix)

### Running Tests

```bash
# Help patcher linter tests
make test-help-lint-test

# GenExpr linter tests
make test-lint-test

# Both test suites
cd max-externals && pytest tools/
```

Test files are in `max-externals/tools/test_lint_*.py`.
