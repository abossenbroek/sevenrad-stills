# TDL-033: TouchDesigner Python Stubs

---
id: TDL-033
status: pending
priority: medium
phase: 3
depends_on: [TDL-032]
blocks: []
---

## Problem Statement

Python validators (TDL-032) and IDEs need type information about TouchDesigner's Python API. Creating stub files (`.pyi`) enables better undefined name detection and IDE autocompletion when developing TD scripts outside TouchDesigner.

## Acceptance Criteria

- [ ] Stub files cover core TD classes (Op, Par, etc.)
- [ ] Stub files cover TD special objects (me, op(), project)
- [ ] Stub files cover common TD functions (run, cook, debug)
- [ ] Stubs work with mypy/pyright for type checking
- [ ] Stubs work with IDE autocompletion
- [ ] Documented how to use stubs in external editors

## Files to Create

```
td_linter/
├── stubs/
│   ├── td.pyi            # Main TD module
│   ├── tdu.pyi           # TD utilities
│   └── _builtins.pyi     # op, me, project, etc.
└── docs/
    └── using_stubs.md
```

## Research Pointers

### Python Stub Files

- https://mypy.readthedocs.io/en/stable/stubs.html
- `.pyi` files contain type annotations without implementation
- Used by type checkers (mypy, pyright) and IDEs

### TouchDesigner API Documentation

- https://docs.derivative.ca/Python
- https://docs.derivative.ca/OP_Class
- https://docs.derivative.ca/Par_Class

Study these to understand the API surface to stub.

### Core Classes to Stub

**Op Class**:
```python
class Op:
    name: str
    path: str
    parent: "Op"
    def par(self, name: str) -> "Par": ...
    def op(self, path: str) -> "Op | None": ...
```

**Par Class**:
```python
class Par:
    val: Any
    eval: Any
    expr: str
    mode: ParMode
```

### Special Objects

```python
# Fake declarations for TD builtins
op: Callable[[str], Op | None]
me: Op
project: Project
absTime: AbsTime
```

### Stub Design Decisions

1. **Accuracy vs Coverage**: Perfect stubs are impossible without full TD source. Aim for "good enough" to catch common errors.

2. **Versioning**: TD API changes between versions. Consider version-specific stubs or union types.

3. **Dynamic Nature**: `op('name')` returns different types based on runtime. Use `Op | None` or generics.

### Usage Documentation

Document how users can leverage stubs:

**In IDE (VS Code/PyCharm)**:
- Add stubs directory to Python path
- Configure interpreter to include stubs

**With mypy**:
```bash
mypy --custom-typeshed-dir=path/to/stubs script.py
```

### Resources

- Existing TD community stubs? Search GitHub
- How other tools create stubs for proprietary APIs
- Consider generating stubs from TD runtime introspection

## Scope Limitation

This is medium priority. Basic undefined name detection (TDL-032) works without full stubs. Stubs enhance the experience but aren't required for basic linting.

## Definition of Done

All acceptance criteria checked. Stubs enable basic type checking and autocompletion.
