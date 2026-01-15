# TDL-013: Python Parser Integration

---
id: TDL-013
status: pending
priority: high
phase: 1
depends_on: [TDL-010, TDL-011]
blocks: [TDL-020, TDL-030]
---

## Problem Statement

The Lark grammars produce Python-native parsers. We need to integrate these parsers into the linter codebase, providing a clean API for parsing .n and .parm files and traversing the resulting parse trees.

## Acceptance Criteria

- [ ] Lark grammars integrated into td_linter package
- [ ] Python can load and use both grammars
- [ ] Parse function returns traversable parse trees
- [ ] AST traversal helpers work
- [ ] Error handling provides clear messages for parse failures
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

### Lark Python Package

- https://lark-parser.readthedocs.io/en/latest/
- `pip install lark`

Key classes:
- `Lark` - Creates a parser from a grammar
- `Tree` - The parse result (tree of nodes)
- `Token` - Terminal tokens in the tree
- `Transformer` - For tree transformation

### Loading the Grammar

```python
from lark import Lark
from pathlib import Path

# Load grammar from file
grammar_path = Path(__file__).parent / "grammars" / "node.lark"
node_parser = Lark(
    grammar_path.read_text(),
    start="source_file",
    parser="lalr",  # Fast parser for unambiguous grammars
)

# Parse a file
tree = node_parser.parse(source_text)
```

### AST Traversal

Study Lark's tree API:
- `tree.data` - Rule name that matched
- `tree.children` - Child nodes (Tree or Token)
- `tree.find_data(name)` - Find all subtrees with given rule
- `tree.find_pred(pred)` - Find subtrees matching predicate

Consider: Should you use Lark's Transformer or write recursive traversal?

### Two Grammars, One Package

You have two grammars (node and parm). Simply create two Lark parser instances:

```python
node_parser = Lark(node_grammar, start="source_file")
parm_parser = Lark(parm_grammar, start="source_file")
```

### Platform Considerations

Lark is pure Python, so it works on all platforms without compilation:

| Platform | Notes |
|----------|-------|
| macOS | Works out of the box |
| Linux | Works out of the box |
| Windows | Works out of the box |

No native compilation needed - simpler distribution than native parsers.

### Error Handling

Lark provides excellent error messages:

```python
try:
    tree = parser.parse(source)
except lark.exceptions.UnexpectedToken as e:
    print(f"Parse error at line {e.line}, column {e.column}")
    print(f"Expected: {e.expected}")
except lark.exceptions.UnexpectedCharacters as e:
    print(f"Unexpected character at line {e.line}, column {e.column}")
```

Design how to surface these to users with context.

## Performance Considerations

- Parse on demand, not eagerly
- Consider caching parsed ASTs
- Use LALR parser for speed (if grammar is unambiguous)
- Earley parser handles ambiguous grammars but is slower

## Definition of Done

All acceptance criteria checked. Python can parse .n and .parm files into traversable ASTs.
