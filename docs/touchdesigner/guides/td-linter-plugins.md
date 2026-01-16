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
