# TDL-032: Python AST Validator

---
id: TDL-032
status: pending
priority: high
phase: 3
depends_on: [TDL-030]
blocks: [TDL-033]
---

## Problem Statement

Python scripts in Execute DATs and callbacks should be syntax-checked before runtime. TouchDesigner provides special globals (`op`, `me`, `project`) that don't exist at lint time. The validator must catch syntax errors without false positives for TD builtins.

## Acceptance Criteria

- [ ] Syntax validation using `ast.parse()`
- [ ] Undefined name detection via AST analysis
- [ ] TD builtins whitelist prevents false positives
- [ ] Reports syntax errors with line numbers
- [ ] Handles .text file header (version line)
- [ ] Test cases for valid TD scripts and syntax errors

## Files to Create

```
td_linter/
├── embedded/
│   └── python_validator.py
└── tests/
    └── test_python_validator.py
```

## Research Pointers

### Python AST Module

- https://docs.python.org/3/library/ast.html
- `ast.parse(source)` - Returns AST or raises SyntaxError
- `ast.walk(tree)` - Iterate all nodes
- `ast.Name` - Variable reference node

### Syntax Validation

```python
try:
    tree = ast.parse(content, filename=path)
except SyntaxError as e:
    # Report: e.msg, e.lineno
```

### Undefined Name Detection

Approach:
1. Walk AST, collect all `ast.Name` nodes
2. Track defined names (function defs, assignments, imports)
3. Track used names (Name nodes in Load context)
4. Report: used - defined - builtins - td_builtins

### TD Builtins Whitelist (RF-003)

From the spec, comprehensive list:

**Core objects**:
`op`, `me`, `mod`, `ext`, `par`, `storage`, `fetch`, `store`, `parent`, `ipar`, `iop`, `ui`, `project`, `root`, `absTime`, `app`, `sysinfo`, `monitors`

**Common functions**:
`run`, `cook`, `debug`, `passive`, `var`, `vardict`

**Callbacks** (when defined, not errors):
`onCook`, `onPulse`, `onValueChange`, `onStart`, `onCreate`, `onExit`, etc.

**Modules**:
`td`, `tdu`, `TDF`, `TDJSON`, `TDStoreTools`, `TDFunctions`

### Callback Completeness Check

Execute DATs should implement standard callbacks. If some are defined, suggest missing ones as INFO (not ERROR).

Standard Execute DAT callbacks:
`onStart`, `onCreate`, `onExit`, `onFrameStart`, `onFrameEnd`, `onPlayStateChange`, `onDeviceChange`, `onProjectPreSave`, `onProjectPostSave`

### AST Analysis Patterns

**Finding undefined names**:
```python
for node in ast.walk(tree):
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            used.append((node.id, node.lineno))
```

**Finding function definitions**:
```python
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        defined.add(node.name)
```

### Edge Cases

- Empty scripts
- Scripts with only imports
- Scripts that define all callbacks vs partial
- Lambda functions
- Comprehensions (have their own scope)

## Warning vs Error

- **ERROR**: Syntax errors (will definitely fail)
- **WARNING**: Undefined names (might be TD builtins we missed)
- **INFO**: Missing callbacks (best practice, not error)

## Definition of Done

All acceptance criteria checked. Python syntax validated, TD builtins not flagged.
