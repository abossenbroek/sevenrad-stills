# td-linter Plugin Guide

td-linter supports custom rules through a plugin system. This guide explains how to write and use plugins.

## Plugin Architecture

Plugins are Python files containing `LintRule` subclasses. The plugin loader discovers and instantiates these rules.

## Writing a Custom Rule

### Basic Rule

Create a file `my_rules.py`:

```python
"""Custom td-linter rules."""

from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation


class NoEmptyContainers(LintRule):
    """Detect empty COMP containers."""

    @property
    def rule_id(self) -> str:
        return "CUSTOM001"

    @property
    def name(self) -> str:
        return "no-empty-containers"

    @property
    def description(self) -> str:
        return "COMP containers should have at least one child operator"

    @property
    def category(self) -> str:
        return "C"  # Connection category

    @property
    def severity(self) -> str:
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for empty containers."""
        for node_path in graph.nodes:
            node_data = graph.nodes[node_path]
            family = node_data.get("family", "")

            if family == "COMP":
                # Check if container has children
                children = list(graph.successors(node_path))
                if len(children) == 0:
                    yield Violation(
                        rule=self.rule_id,
                        message="Empty COMP container",
                        path=node_path,
                        severity=self.severity,
                    )
```

### Required Properties

| Property | Type | Description |
|----------|------|-------------|
| `rule_id` | str | Unique identifier (e.g., "CUSTOM001") |
| `name` | str | Human-readable name (lowercase, dashes) |
| `description` | str | What the rule checks |
| `category` | str | Category code (S, C, T, R, G, P, F) |
| `severity` | str | Default severity (error, warning, info) |

### The `check` Method

```python
def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
    """Analyze the operator graph and yield violations."""
```

The graph contains:
- **Nodes**: Operator paths as keys, with attributes:
  - `operator`: ParsedNFile object (may be None)
  - `family`: Operator family (TOP, CHOP, SOP, DAT, COMP, MAT)
- **Edges**: Connections between operators with `input_index` attribute

## Loading Plugins

### From Configuration File

Add to `td-linter.yaml`:

```yaml
plugins:
  - path: ./my_rules.py
  - module: my_package.rules
```

### From Command Line

Plugins specified in config are loaded automatically:

```bash
td-linter lint project.toe.dir --config config-with-plugins.yaml
```

## Plugin Loading Methods

### 1. File Path

Load from a local Python file:

```yaml
plugins:
  - path: ./custom_rules.py
  - path: /absolute/path/to/rules.py
```

### 2. Module Import

Load from an installed Python package:

```yaml
plugins:
  - module: my_company.td_rules
  - module: my_project.linter.custom
```

### 3. Entry Points

Plugins can be registered as Python entry points in `pyproject.toml`:

```toml
[project.entry-points."td_linter.rules"]
my_rules = "my_package.rules:MyCustomRule"
```

Entry point plugins are discovered automatically.

## Rule with Options

Rules can accept configuration options:

```python
class MaxOperatorDepth(LintRule):
    """Check operator nesting depth."""

    @property
    def rule_id(self) -> str:
        return "CUSTOM002"

    @property
    def name(self) -> str:
        return "max-operator-depth"

    @property
    def description(self) -> str:
        return "Operators should not be nested too deeply"

    @property
    def category(self) -> str:
        return "F"

    @property
    def severity(self) -> str:
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        # Get option with default value
        max_depth = self.get_option("max_depth", 10)

        for node_path in graph.nodes:
            depth = node_path.count("/")
            if depth > max_depth:
                yield Violation(
                    rule=self.rule_id,
                    message=f"Operator depth {depth} exceeds max {max_depth}",
                    path=node_path,
                    severity=self.severity,
                )
```

Configure options in YAML:

```yaml
plugins:
  - path: ./my_rules.py

rules:
  CUSTOM002:
    enabled: true
    options:
      max_depth: 15
```

## Rule with Auto-Fix

Rules can provide fixes:

```python
from td_linter.rules.base import LintRule, Violation, Fix, Replacement


class RemoveDebugPrints(LintRule):
    """Remove debug print statements from scripts."""

    @property
    def rule_id(self) -> str:
        return "CUSTOM003"

    @property
    def name(self) -> str:
        return "no-debug-prints"

    @property
    def description(self) -> str:
        return "Remove debug print statements"

    @property
    def category(self) -> str:
        return "P"

    @property
    def severity(self) -> str:
        return "warning"

    @property
    def fixable(self) -> bool:
        return True

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        for node_path in graph.nodes:
            node_data = graph.nodes[node_path]
            operator = node_data.get("operator")

            if operator and hasattr(operator, "text_content"):
                lines = operator.text_content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "print(" in line and "DEBUG" in line:
                        fix = Fix(
                            description="Remove debug print",
                            replacements=[
                                Replacement(
                                    file_path=operator.source_file,
                                    start_line=i,
                                    end_line=i,
                                    new_text="",
                                )
                            ],
                        )
                        yield Violation(
                            rule=self.rule_id,
                            message="Debug print statement found",
                            path=node_path,
                            severity=self.severity,
                            line=i,
                            fix=fix,
                        )
```

## Testing Plugins

Test your plugin locally:

```python
import networkx as nx
from my_rules import NoEmptyContainers

# Create test graph
graph = nx.DiGraph()
graph.add_node("/project/empty_comp", family="COMP", operator=None)

# Run rule
rule = NoEmptyContainers()
violations = list(rule.check(graph))

assert len(violations) == 1
assert violations[0].rule == "CUSTOM001"
```

## Error Handling

Plugin loading errors are reported but don't crash the linter:

```
Warning: Failed to load plugin ./broken_rules.py
  SyntaxError: invalid syntax (broken_rules.py, line 5)
```

## Best Practices

1. **Use descriptive rule IDs**: Prefix with your organization (e.g., "ACME001")
2. **Provide clear messages**: Include what's wrong and where
3. **Set appropriate severity**: error for breaking issues, warning for best practices
4. **Document options**: Explain what each option does
5. **Test thoroughly**: Include edge cases in your tests
6. **Handle missing data**: Check for None values in graph data

## Security Model

td-linter validates plugins before loading them using AST analysis.

### What's Checked

The plugin security validator scans for:
- **Dangerous imports**: `os`, `subprocess`, `sys`, `shutil`, `importlib`
- **Dangerous functions**: `eval`, `exec`, `compile`, `open`, `__import__`
- **Network access**: `socket`, `urllib`, `requests`, `http`
- **File operations**: Direct file I/O outside the validation context

### Why This Matters

Plugins run with full Python access. The security checks prevent:
- Arbitrary code execution
- File system access outside the project
- Network requests to external services
- System command execution

### If Your Plugin is Blocked

If your plugin uses a blocked construct legitimately:

1. **Refactor to avoid it** - Usually possible for most use cases
2. **Use built-in utilities** - td-linter provides safe alternatives
3. **Request allowlisting** - For known-safe patterns

```python
# BLOCKED: Direct file read
content = open(path).read()

# SAFE: Use the operator's stored content
content = node_data.get("operator").text_content
```

## Accessing Graph Data

The NetworkX graph contains rich data about each operator.

### Node Attributes

```python
def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
    for node_path in graph.nodes:
        data = graph.nodes[node_path]

        # Available attributes:
        family = data.get("family")       # "TOP", "CHOP", "SOP", etc.
        op_type = data.get("op_type")     # "noise", "moviefilein", etc.
        source_file = data.get("source_file")  # Path to .n file
        operator = data.get("operator")   # Full OperatorNode object

        # From OperatorNode:
        if operator:
            name = operator.name
            tile = operator.tile  # TilePosition(x, y, width, height)
            flags = operator.flags  # dict of flag values
            inputs = operator.inputs  # list of (index, ref_name)
            color = operator.color  # RGB tuple or None
```

### Edge Attributes

```python
def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
    for source, target, data in graph.edges(data=True):
        input_index = data.get("input_index")  # Which input slot
        is_missing = data.get("missing", False)  # Reference to non-existent op
```

### Graph Traversal

```python
import networkx as nx

def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
    # Get all predecessors (operators feeding into this one)
    for node in graph.nodes:
        inputs = list(graph.predecessors(node))
        outputs = list(graph.successors(node))

    # Find cycles
    cycles = list(nx.simple_cycles(graph))

    # Get all paths between two nodes
    paths = list(nx.all_simple_paths(graph, source, target))

    # Calculate depth (longest path from root)
    # Note: Only works for DAGs
    try:
        lengths = nx.single_source_shortest_path_length(graph, root)
    except nx.NetworkXError:
        pass  # Has cycles

    # Get strongly connected components
    sccs = list(nx.strongly_connected_components(graph))
```

## Debugging Plugins

### Print Debug Output

```python
class MyRule(LintRule):
    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        import sys
        print(f"DEBUG: Graph has {len(graph.nodes)} nodes", file=sys.stderr)

        for node in graph.nodes:
            data = graph.nodes[node]
            print(f"DEBUG: {node} -> {data.get('family')}", file=sys.stderr)
```

Run with stderr visible:
```bash
td-linter lint project.toe.dir 2>&1 | grep DEBUG
```

### Test in Isolation

```python
# test_my_rule.py
import networkx as nx
from my_rules import MyRule

def test_rule():
    graph = nx.DiGraph()
    graph.add_node("test/op1", family="TOP", op_type="noise")
    graph.add_node("test/op2", family="TOP", op_type="null")
    graph.add_edge("test/op1", "test/op2", input_index=0)

    rule = MyRule()
    violations = list(rule.check(graph))

    print(f"Found {len(violations)} violations:")
    for v in violations:
        print(f"  {v.rule}: {v.message}")

if __name__ == "__main__":
    test_rule()
```

### Check Plugin Loading

```bash
# Verify plugin loads without errors
python -c "
from pathlib import Path
exec(Path('my_rules.py').read_text())
print('Plugin loaded successfully')
"
```

## Publishing Your Plugin

### As a Python Package

1. Create package structure:
```
my-td-rules/
├── pyproject.toml
├── src/
│   └── my_td_rules/
│       ├── __init__.py
│       └── rules.py
└── README.md
```

2. Configure `pyproject.toml`:
```toml
[project]
name = "my-td-rules"
version = "1.0.0"

[project.entry-points."td_linter.rules"]
my_rules = "my_td_rules.rules"
```

3. Install and use:
```bash
pip install my-td-rules
# Plugin auto-discovered via entry points
td-linter lint project.toe.dir
```

### Sharing as a File

For simpler distribution:

1. Share the `.py` file
2. Users add to their config:
```yaml
plugins:
  - path: ./my_rules.py
```

### Documentation Template

Include with your plugin:

```markdown
# My Custom Rules

## Installation

pip install my-td-rules

## Rules

### MYCO001: my-custom-rule

**Severity**: warning

**Description**: Checks for [specific condition].

**Why**: Explains why this matters.

**Options**:
- `threshold` (int, default: 10): Maximum allowed value

**Configuration**:
```yaml
rules:
  MYCO001:
    enabled: true
    options:
      threshold: 15
```
```

## Performance Tips

### Minimize Graph Traversal

```python
# SLOW: Multiple full traversals
for node in graph.nodes:
    if has_issue_a(node):
        yield violation
for node in graph.nodes:
    if has_issue_b(node):
        yield violation

# FAST: Single traversal
for node in graph.nodes:
    if has_issue_a(node):
        yield violation_a
    if has_issue_b(node):
        yield violation_b
```

### Cache Computed Values

```python
class MyRule(LintRule):
    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        # Cache expensive computation
        top_operators = {
            n for n in graph.nodes
            if graph.nodes[n].get("family") == "TOP"
        }

        for node in top_operators:
            # Use cached set
            ...
```

### Use NetworkX Efficiently

```python
# SLOW: Check existence by iteration
exists = any(n == target for n in graph.nodes)

# FAST: Direct lookup
exists = target in graph.nodes

# SLOW: Get all edges then filter
edges = [(u, v) for u, v in graph.edges if u == source]

# FAST: Use built-in method
edges = list(graph.out_edges(source))
```

### Skip Unnecessary Work

```python
def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
    # Early exit if nothing to check
    if len(graph.nodes) == 0:
        return

    # Only check relevant operators
    relevant = [
        n for n in graph.nodes
        if graph.nodes[n].get("family") in ("TOP", "CHOP")
    ]

    for node in relevant:
        # Expensive check only on filtered set
        ...
```
