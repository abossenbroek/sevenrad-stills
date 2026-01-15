# TDL-054: Package Distribution

---
id: TDL-054
status: pending
priority: high
phase: 5
depends_on: [TDL-014]
blocks: []
---

## Problem Statement

The linter needs to be installable via pip. This requires proper packaging with pyproject.toml, dependencies declared, entry points configured, and publication to PyPI (or TestPyPI initially).

## Acceptance Criteria

- [ ] pyproject.toml properly configured
- [ ] Dependencies specified (typer, networkx, lark, etc.)
- [ ] Entry point creates `td-linter` command
- [ ] Package installs cleanly: `pip install td-linter`
- [ ] Package works after install (not just editable mode)
- [ ] Published to TestPyPI (or PyPI)
- [ ] README with installation and usage

## Files to Create

```
td_linter/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    └── td_linter/
        ├── __init__.py     # Version info
        └── ...
```

## Research Pointers

### Modern Python Packaging

- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- pyproject.toml replaces setup.py
- Use `build` and `twine` for publishing

### pyproject.toml Structure

```toml
[project]
name = "td-linter"
version = "1.0.0"
description = "Linter for TouchDesigner .toe.dir projects"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = [
    "typer[all]>=0.9.0",
    "networkx>=3.0",
    "lark>=1.1.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.scripts]
td-linter = "td_linter.cli:app"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Entry Points

```toml
[project.scripts]
td-linter = "td_linter.cli:app"
```

This makes `td-linter` available after install.

### Lark Distribution

Lark is pure Python with no native dependencies, making distribution straightforward:
- No compilation required on any platform
- Grammars are plain text files included in the package
- Works on Python 3.8+ without additional setup

Simply include grammar files in the package data.

### Building and Publishing

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ td-linter

# If OK, upload to PyPI
twine upload dist/*
```

### Version Management

Consider:
- Semantic versioning (1.0.0)
- Version in `__init__.py`
- Dynamic version from pyproject.toml or git tags

### README Content

- What is td-linter
- Installation: `pip install td-linter`
- Quick start: `td-linter lint myproject.toe.dir`
- Configuration: td-linter.yaml
- Link to full documentation

### Package Testing

Before publishing:
1. Build package
2. Install in clean virtualenv
3. Run CLI commands
4. Run test suite against installed package

```bash
# Create clean environment
python -m venv test_env
source test_env/bin/activate

# Install from local wheel
pip install dist/td_linter-1.0.0-py3-none-any.whl

# Test
td-linter --version
td-linter lint some_project.toe.dir
```

### License Selection

Choose appropriate license:
- MIT: Permissive, simple
- Apache 2.0: Patent protection
- GPL: Copyleft

## Definition of Done

All acceptance criteria checked. Package installable from PyPI/TestPyPI, works correctly.
