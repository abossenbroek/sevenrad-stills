# td-linter - TouchDesigner Project Validator

## Overview

`td-linter` validates expanded TouchDesigner `.toe.dir` projects before they are collapsed back to binary `.toe` files. It catches structural errors, broken references, and type mismatches that would otherwise only surface at runtime.

## Architecture

```
td_linter/
├── cli.py              # Typer CLI (lint, rules, init, version)
├── linter.py           # Main orchestration
├── grammars/           # Lark EBNF grammars
│   ├── n_file.lark     # .n (node definition) grammar
│   └── parm_file.lark  # .parm (parameter) grammar
├── parsers/            # File parsers using Lark
│   ├── n_parser.py     # Parses .n files → ParsedNFile
│   ├── parm_parser.py  # Parses .parm files → ParsedParmFile
│   └── toc_parser.py   # Parses .toc manifest files
├── graph/              # NetworkX graph model
│   ├── types.py        # OperatorFamily enum, compatibility rules
│   ├── model.py        # OperatorNode, Connection dataclasses
│   └── builder.py      # Builds DiGraph from parsed files
└── rules/              # Validation rules
    ├── base.py         # LintRule base class, Violation dataclass
    ├── no_invalid_cycles.py
    ├── no_dangling_inputs.py
    ├── valid_references.py
    └── type_compatibility.py
```

## TouchDesigner File Formats

### .n files (Node Definitions)
```
TOP:displace
tile 260 200 130 72
flags =  viewer 1 parlanguage 0
inputs
{
0 	moviefilein1
1 	chopto1
}
color 0.67 0.67 0.67
view -1 3 0 0 1 1 0 0
end
```

### .parm files (Parameters)
```
?
type 0 hermite
rough 0 0.25
tx 49 6531 absTime.frame*.6
autoexportroot 17 "" me.parent()
?
```

Mode flags: 0=constant, 17=string expr, 32=numeric, 49=expression

### .toc files (Manifest)
One path per line, special entries start with `.` (`.build`, `.start`, etc.)

## Key Patterns

### Lark Transformers
Each parser creates a **fresh transformer instance per parse** to avoid state carryover:
```python
def parse(self, file_path: Path) -> ParsedNFile:
    tree = self._parser.parse(content)
    transformer = NFileTransformer()  # Fresh instance!
    return transformer.transform(tree)
```

### NetworkX Graph
Operators are nodes, connections are directed edges:
```python
graph.add_node(node.path, operator=node, family=node.family.value)
graph.add_edge(source_path, target_path, input_index=idx)
```

Missing references create `MISSING:` placeholder nodes.

### Validation Rules
All rules inherit from `LintRule` and implement `check(graph) -> Iterator[Violation]`:
```python
class MyRule(LintRule):
    @property
    def id(self) -> str: return "my-rule-id"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        # Yield violations found
```

## Commands

```bash
# Lint a project
td-linter lint path/to/project.toe.dir

# List available rules
td-linter rules

# Create config file
td-linter init

# Verbose output
td-linter lint project.toe.dir -v
```

## Testing

```bash
# Run linter on fixtures
uv run td-linter lint docs/touchdesigner/fixtures/projects/reference_toe/example.toe.dir
uv run td-linter lint docs/touchdesigner/fixtures/projects/shader_test_harness.toe.dir

# Test parser directly
uv run python -c "
from td_linter.parsers.n_parser import NFileParser
parser = NFileParser()
result = parser.parse(Path('path/to/file.n'))
print(result.inputs)
"
```

## Operator Families

| Family | Description |
|--------|-------------|
| TOP | Texture Operators (images, video) |
| CHOP | Channel Operators (audio, control signals) |
| SOP | Surface Operators (3D geometry) |
| DAT | Data Operators (tables, text, scripts) |
| COMP | Component Operators (containers, UI) |
| MAT | Material Operators (shaders) |

## Valid Connections

- TOP → TOP, MAT
- CHOP → CHOP (supports native feedback)
- SOP → SOP, MAT
- DAT → DAT
- COMP → COMP

Cross-family requires converter operators (chopto, sopto, tochop, etc.)

## Feedback Loops

Valid cycles:
- All-CHOP cycles (native feedback support)
- Cycles containing: `feedback`, `feedbackchop`, `timemachine`, `delay`, `lag`

## Adding New Rules

1. Create `rules/my_new_rule.py`:
```python
from td_linter.rules.base import LintRule, Violation

class MyNewRule(LintRule):
    @property
    def id(self) -> str:
        return "my-rule-id"

    @property
    def description(self) -> str:
        return "What this rule checks"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        for node in graph.nodes:
            if something_wrong(node):
                yield Violation(
                    rule=self.id,
                    message="Description of problem",
                    path=node,
                )
```

2. Register in `linter.py`:
```python
from td_linter.rules.my_new_rule import MyNewRule

def get_all_rules() -> list[LintRule]:
    return [
        # ... existing rules
        MyNewRule(),
    ]
```

## Ruff Configuration

The package has specific ignores in `pyproject.toml`:
- `cli.py`: B008 (Typer pattern), ARG001 (future args)
- `parsers/*.py`: ANN401 (Lark uses Any), ARG002 (transformer methods)

## Related Files

- Fixtures: `docs/touchdesigner/fixtures/projects/`
- Format docs: `docs/touchdesigner/reference/toe_dir_format.yaml`
- Specification: `docs/touchdesigner/reference/linter_spec.md`
- Tickets: `docs/touchdesigner/linter-tickets/`
