# TDL-013: Python Bindings Integration

---
id: TDL-013
status: pending
priority: high
phase: 1
depends_on: [TDL-010, TDL-011]
blocks: [TDL-020, TDL-030]
---

## Problem Statement

The tree-sitter grammars produce native parsers. To use them in our Python-based linter, we need Python bindings. This involves building the grammar, generating shared libraries, and integrating with the `tree-sitter` Python package.

## Acceptance Criteria

- [ ] Grammars compile to shared library (.so/.dylib/.dll)
- [ ] Python can load compiled grammars
- [ ] Parse function returns AST nodes
- [ ] AST traversal helpers work
- [ ] Memory management is correct (no leaks)
- [ ] Works on macOS, Linux, and Windows

## Files to Create

```
td_linter/
├── parser.py              # Python wrapper for grammars
├── ast_utils.py           # AST traversal utilities
└── tests/
    ├── test_n_parser.py
    └── test_parm_parser.py
```

## Research Pointers

### Tree-sitter Python Package

- https://github.com/tree-sitter/py-tree-sitter
- `pip install tree-sitter`

Key classes:
- `Language` - Loads a compiled grammar
- `Parser` - Creates a parser instance
- `Tree` - The parse result
- `Node` - AST nodes

### Building the Grammar

```bash
# In tree-sitter-toedir/
npm install
npx tree-sitter generate
npx tree-sitter build
```

This produces `build/toedir.so` (or platform equivalent).

### Loading in Python

```python
from tree_sitter import Language, Parser

# Load compiled grammar
LANGUAGE = Language('path/to/toedir.so', 'toedir_node')

# Create parser
parser = Parser()
parser.set_language(LANGUAGE)

# Parse a file
tree = parser.parse(source_bytes)
```

### AST Traversal

Study tree-sitter's node API:
- `node.type` - Node type string
- `node.children` - Child nodes
- `node.text` - Source text as bytes
- `node.start_point`, `node.end_point` - Location

Consider: Should you use tree-sitter's cursor API or write recursive traversal?

### Two Grammars, One Package

You have two grammars (node and parm). Options:
1. Single .so with multiple languages
2. Two separate .so files

Research how tree-sitter handles multiple languages in one project.

### Platform Considerations

| Platform | Extension | Build System |
|----------|-----------|--------------|
| macOS | .dylib | clang |
| Linux | .so | gcc |
| Windows | .dll | MSVC |

Consider using `tree-sitter build` which handles cross-platform automatically.

### Error Handling

Tree-sitter provides error recovery:
- `node.has_error` - Did this subtree have errors?
- `node.is_missing` - Is this a missing node?
- `ERROR` node type for unparseable regions

Design how to surface these to users.

## Performance Considerations

- Parse on demand, not eagerly
- Consider caching parsed ASTs
- tree-sitter is incremental - can re-parse on edit

## Definition of Done

All acceptance criteria checked. Python can parse .n and .parm files into traversable ASTs.
