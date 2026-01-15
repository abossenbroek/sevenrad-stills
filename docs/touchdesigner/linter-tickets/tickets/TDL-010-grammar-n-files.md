# TDL-010: Tree-sitter Grammar for .n Files

---
id: TDL-010
status: pending
priority: critical
phase: 1
depends_on: [TDL-001]
blocks: [TDL-013, TDL-020]
---

## Problem Statement

.n files define TouchDesigner operators (type, position, flags, connections). To validate them, we need a formal grammar that can parse any valid .n file into an AST. Without a grammar, we're reduced to regex hacks that break on edge cases.

## Acceptance Criteria

- [ ] Tree-sitter grammar parses all .n files in sample corpus
- [ ] Grammar handles all operator families (TOP, CHOP, SOP, DAT, COMP, MAT, POP)
- [ ] Grammar handles optional elements in any order (inputs, color, dock, view)
- [ ] Grammar uses semantic node types (tile_coord, input_index, mode_flag)
- [ ] Error recovery produces partial AST for malformed files
- [ ] Unit tests cover happy path and edge cases
- [ ] `tree-sitter parse` succeeds on corpus with no errors

## Files to Create

```
tree-sitter-toedir/
├── grammar.js              # The grammar definition
├── package.json            # npm package config
├── binding.gyp             # Native binding config
├── src/                    # Generated (by tree-sitter generate)
├── corpus/                 # Test corpus
│   ├── valid/              # Files that should parse
│   └── invalid/            # Files that should fail gracefully
└── README.md
```

## Research Pointers

### Tree-sitter Learning Path

1. **Official tutorial**: https://tree-sitter.github.io/tree-sitter/creating-parsers/
2. **Grammar writing guide**: https://tree-sitter.github.io/tree-sitter/creating-parsers/3-writing-the-grammar.html
3. **First grammar gist**: https://gist.github.com/Aerijo/df27228d70c633e088b0591b8857eeef

### .n File Structure (from spec)

```
TYPE:subtype
tile X Y W H
flags = [flag_list]
[inputs { ... }]
[color R G B [A]]
[dock ref_name]
[view ...]
end
```

Key observations:
- Header is fixed order: type, tile, flags
- Middle elements (inputs, color, dock, view) can appear in any order
- Footer is always `end`

### Grammar Design Decisions

1. **Permutation problem**: Optional middle elements in any order.
   - Solution: Use `repeat(choice(...))` pattern
   - Trade-off: Allows duplicates; semantic check needed

2. **Semantic types**: Instead of generic `integer`, use:
   - `tile_coord` for positions
   - `input_index` for input slots
   - `mode_flag` for flags
   - Why: Better error messages, range validation

3. **Whitespace handling**: Use `extras: $ => [/\s/]`

### Test Strategy

| Test Case | Description |
|-----------|-------------|
| minimal.n | Just type, tile, flags, end |
| full.n | All optional elements present |
| reordered.n | Optional elements in unusual order |
| complex_inputs.n | Multiple input connections |
| malformed.n | Missing end, invalid type |

### Common Grammar Pitfalls

- **Left recursion**: Tree-sitter handles it, but be aware
- **Ambiguity**: Multiple parse trees for same input
- **Greedy matching**: `repeat` can consume too much
- **Whitespace in wrong places**: Check `extras`

## Incremental Development

1. Start with minimal grammar (type, tile, flags, end)
2. Add one optional element at a time
3. Test against corpus after each addition
4. Handle edge cases discovered in samples

## Definition of Done

All acceptance criteria checked. Grammar parses entire sample corpus without errors.
