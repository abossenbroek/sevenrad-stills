# TouchDesigner Migration: Linting Infrastructure

Complete guide to setting up GLSL and C++ linting for TouchDesigner operator development.

## Related Documents

- [00-IMPLEMENTATION-OVERVIEW.md](00-IMPLEMENTATION-OVERVIEW.md) - High-level roadmap and decisions
- [02-EFFECTS-AND-DEMOS.md](02-EFFECTS-AND-DEMOS.md) - GLSL effects, .tox structure, demo system

---

## Tools

| Tool | Purpose | License | Usage |
|------|---------|---------|-------|
| **glsl_analyzer** | LSP for real-time GLSL validation in editors | GPL-3.0 | CI containers only |
| **glslangValidator** | Khronos reference compiler for CI/CD | Apache-2.0 | Local + CI |
| **clang-tidy** | C++ static analysis for C++ TOPs | Apache-2.0 | Local + CI |
| **clang-format** | C++ code formatting | Apache-2.0 | Local + CI |

> **Note**: glsl_analyzer is GPL-3.0 licensed. Run only in CI containers to avoid license contamination of proprietary source.

---

## Installation

### macOS (Homebrew)

```bash
# GLSL tools
brew install glslang

# C++ tools
brew install llvm

# Add llvm to PATH (for clang-tidy, clang-format)
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

### glsl_analyzer (Optional - for editor LSP)

Download from [GitHub Releases](https://github.com/nolanderc/glsl_analyzer/releases):

```bash
# Download to ~/.local/bin
curl -L https://github.com/nolanderc/glsl_analyzer/releases/latest/download/glsl_analyzer-aarch64-macos.tar.gz | tar xz -C ~/.local/bin
chmod +x ~/.local/bin/glsl_analyzer
```

---

## Directory Structure

```
touchdesigner/
├── glsl/
│   ├── common/
│   │   └── tdCommon.glsl          # Shared utilities (random, sampling, color)
│   ├── effects/
│   │   ├── saturation.frag
│   │   ├── chromatic_aberration.frag
│   │   └── ... (14 effects)
│   └── test_fixtures/
│       ├── valid/                  # Should pass linting
│       │   ├── minimal_passthrough.frag
│       │   └── td_uniforms_usage.frag
│       └── invalid/                # Should fail linting
│           ├── syntax_error.frag
│           ├── missing_output.frag
│           └── wrong_version.frag
├── cpp/
│   ├── CMakeLists.txt
│   ├── .clang-tidy
│   ├── .clang-format
│   └── src/
└── scripts/
    └── validate_glsl.py            # GLSL validation wrapper
```

---

## GLSL Validation Script

Create `touchdesigner/scripts/validate_glsl.py`:

```python
#!/usr/bin/env python3
"""GLSL shader validation for TouchDesigner compatibility."""

import subprocess
import sys
import tempfile
from pathlib import Path

# TouchDesigner preamble - simulates TD's built-in declarations
TD_PREAMBLE = """
#version 330 core

// TouchDesigner built-in uniforms (simplified for validation)
uniform sampler2D sTD2DInputs[8];
uniform vec4 uTDOutputInfo;
uniform int uTDPass;

// Input from vertex shader
in vec2 vUV;

// Standard output
layout(location = 0) out vec4 fragColor;

// TouchDesigner helper stubs
vec4 TDOutputSwizzle(vec4 c) { return c; }

// --- USER SHADER BEGINS BELOW ---
"""

TD_COMPUTE_PREAMBLE = """
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// TouchDesigner compute shader stubs
uniform sampler2D sTD2DInputs[8];
layout(rgba32f) uniform image2D sTD2DOutputs[8];

// --- USER SHADER BEGINS BELOW ---
"""

def validate_shader(shader_path: Path) -> tuple[bool, str]:
    """Validate a GLSL shader file."""
    content = shader_path.read_text()

    # Check for forbidden #version directive
    if "#version" in content:
        return False, f"ERROR: {shader_path}: Contains #version directive (TouchDesigner auto-injects this)"

    # Determine shader type
    suffix = shader_path.suffix.lower()
    if suffix == ".comp":
        preamble = TD_COMPUTE_PREAMBLE
        stage = "comp"
    else:
        preamble = TD_PREAMBLE
        stage = "frag"

    # Create temp file with preamble
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(preamble)
        f.write(content)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['glslangValidator', '-S', stage, temp_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return True, f"OK: {shader_path}"
        else:
            errors = result.stdout + result.stderr
            return False, f"ERRORS in {shader_path}:\n{errors}"
    finally:
        Path(temp_path).unlink()

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_glsl.py <shader_file> [shader_file...]")
        sys.exit(1)

    all_passed = True
    for path in sys.argv[1:]:
        passed, message = validate_shader(Path(path))
        print(message)
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
```

---

## Pre-commit Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  # ... existing hooks ...

  # GLSL validation
  - repo: local
    hooks:
      - id: glsl-validate
        name: Validate GLSL shaders
        entry: python touchdesigner/scripts/validate_glsl.py
        language: python
        files: \.glsl$|\.frag$|\.vert$|\.comp$
        pass_filenames: true

  # C++ formatting and linting (if C++ TOPs are used)
  - repo: https://github.com/cpp-linter/cpp-linter-hooks
    rev: v1.1.11
    hooks:
      - id: clang-format
        args: [--style=file]
        files: ^touchdesigner/cpp/.*\.(cpp|h|hpp)$
      - id: clang-tidy
        args: [--config-file=touchdesigner/cpp/.clang-tidy]
        files: ^touchdesigner/cpp/.*\.(cpp|h|hpp)$
```

---

## CI/CD Workflow

Create `.github/workflows/touchdesigner-lint.yml`:

```yaml
name: TouchDesigner Lint

on:
  push:
    branches: [main, develop, feature/*]
    paths:
      - 'touchdesigner/**'
  pull_request:
    branches: [main, develop]
    paths:
      - 'touchdesigner/**'

jobs:
  glsl-validation:
    name: GLSL Shader Validation
    runs-on: ubuntu-24.04
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install glslang
        run: sudo apt-get update && sudo apt-get install -y glslang-tools

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Validate GLSL shaders
        run: |
          find touchdesigner/glsl/effects -name '*.frag' -o -name '*.comp' | \
            xargs python touchdesigner/scripts/validate_glsl.py

      - name: Test valid fixtures (should pass)
        run: |
          for shader in touchdesigner/glsl/test_fixtures/valid/*.frag; do
            python touchdesigner/scripts/validate_glsl.py "$shader"
          done

      - name: Test invalid fixtures (should fail)
        run: |
          for shader in touchdesigner/glsl/test_fixtures/invalid/*.frag; do
            if python touchdesigner/scripts/validate_glsl.py "$shader" 2>/dev/null; then
              echo "ERROR: $shader should have failed but passed!"
              exit 1
            fi
          done

  cpp-lint:
    name: C++ Static Analysis
    runs-on: ubuntu-24.04
    if: ${{ hashFiles('touchdesigner/cpp/src/**/*.cpp') != '' }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install LLVM/Clang tools
        run: sudo apt-get update && sudo apt-get install -y clang-tidy clang-format cmake

      - name: Generate compile_commands.json
        working-directory: touchdesigner/cpp
        run: |
          mkdir -p build && cd build
          cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

      - name: Run clang-tidy
        working-directory: touchdesigner/cpp
        run: |
          find src -name '*.cpp' | xargs clang-tidy -p build
```

---

## clang-tidy Configuration

Create `touchdesigner/cpp/.clang-tidy`:

```yaml
---
Checks: >
  -*,
  bugprone-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers

WarningsAsErrors: >
  bugprone-use-after-move,
  bugprone-undefined-memory-manipulation

HeaderFilterRegex: '.*'

CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-function-cognitive-complexity.Threshold
    value: 25
```

---

## clang-format Configuration

Create `touchdesigner/cpp/.clang-format`:

```yaml
---
Language: Cpp
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: Never
BreakBeforeBraces: Attach
```

---

## Editor Integration

### VSCode

Add to `.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.frag": "glsl",
    "*.vert": "glsl",
    "*.comp": "glsl",
    "*.glsl": "glsl"
  },
  "glsl-analyzer.serverPath": "glsl_analyzer",
  "C_Cpp.clang_format_style": "file",
  "C_Cpp.codeAnalysis.clangTidy.enabled": true,
  "[cpp]": {
    "editor.formatOnSave": true
  }
}
```

**Recommended Extensions**:
- `slevesque.shader` - GLSL syntax highlighting
- `dtoplak.vscode-glsllint` - GLSL linting
- `ms-vscode.cpptools` - C++ IntelliSense

### Neovim

Add to LSP configuration:

```lua
-- GLSL LSP
require('lspconfig').glsl_analyzer.setup{
  cmd = { "glsl_analyzer" },
  filetypes = { "glsl", "vert", "frag", "comp" },
}

-- C++ LSP with clang-tidy
require('lspconfig').clangd.setup{
  cmd = {
    "clangd",
    "--clang-tidy",
    "--header-insertion=iwyu",
  },
}
```

---

## Test Fixtures

### Valid Shaders (Should Pass)

`touchdesigner/glsl/test_fixtures/valid/minimal_passthrough.frag`:
```glsl
// Minimal passthrough shader
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    fragColor = TDOutputSwizzle(color);
}
```

`touchdesigner/glsl/test_fixtures/valid/td_uniforms_usage.frag`:
```glsl
// Shader using TD uniforms
uniform float uBrightness;

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    color.rgb *= uBrightness;
    fragColor = TDOutputSwizzle(color);
}
```

### Invalid Shaders (Should Fail)

`touchdesigner/glsl/test_fixtures/invalid/has_version.frag`:
```glsl
#version 330 core  // ERROR: TD auto-injects version

void main() {
    fragColor = vec4(1.0);
}
```

`touchdesigner/glsl/test_fixtures/invalid/syntax_error.frag`:
```glsl
// Missing semicolon
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st)
    fragColor = TDOutputSwizzle(color);
}
```

`touchdesigner/glsl/test_fixtures/invalid/undefined_uniform.frag`:
```glsl
// Uses undefined uniform
void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    color.rgb *= uUndefinedVar;  // ERROR
    fragColor = TDOutputSwizzle(color);
}
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `#version directive found` | Shader contains version | Remove `#version` line (TD injects it) |
| `'sTD2DInputs' undeclared` | Not using TD preamble | Run through validate_glsl.py |
| `glslangValidator not found` | Tool not installed | `brew install glslang` |
| `clang-tidy: compile_commands.json not found` | CMake not configured | Run `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` |

### Debugging

```bash
# Test a single shader manually
glslangValidator -S frag your_shader.frag

# Verbose validation
python touchdesigner/scripts/validate_glsl.py your_shader.frag

# Check GLSL version support
glslangValidator --version
```
