# td-linter Python API Reference

This document describes the programmatic Python API for integrating td-linter into build scripts, CI pipelines, and custom tooling.

## Quick Example

```python
from td_linter import lint, lint_and_check, Severity

# Simple linting - works with .toe files or .toe.dir directories
violations = lint("myproject.toe")
for v in violations:
    print(f"{v.severity}: {v.message} at {v.path}")

# Check with pass/fail status
passed, violations = lint_and_check("myproject.toe", fail_on_warning=True)
if not passed:
    sys.exit(1)
```

## Core Functions

### lint()

Main function to lint a TouchDesigner project. Accepts both `.toe` files and `.toe.dir` directories.

```python
def lint(
    path: str | Path,
    config_path: str | Path | None = None,
    select: list[str] | None = None,
    ignore: list[str] | None = None,
    validate_expressions: bool = False,
    validate_embedded: bool = True,
    td_path: str | Path | None = None,
    keep_files_after_expand: bool = False,
) -> list[Violation]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to `.toe` file or `.toe.dir` directory |
| `config_path` | `str \| Path \| None` | `None` | Path to `td-linter.yaml` config |
| `select` | `list[str] \| None` | `None` | Rule IDs or category codes to enable |
| `ignore` | `list[str] \| None` | `None` | Rule IDs or category codes to disable |
| `validate_expressions` | `bool` | `False` | Validate Python expressions in `.parm` files |
| `validate_embedded` | `bool` | `True` | Validate embedded GLSL/Python code |
| `td_path` | `str \| Path \| None` | `None` | Path to TouchDesigner bin directory (for `.toe` files) |
| `keep_files_after_expand` | `bool` | `False` | Keep expanded `.toe.dir` after linting `.toe` files |

**Returns:** `list[Violation]` - List of violations found

**Raises:**
- `FileNotFoundError` - If path does not exist
- `TouchDesignerNotFoundError` - If linting `.toe` file and TouchDesigner cannot be found

**Example:**

```python
from td_linter import lint

# Lint a .toe file directly (auto-expands and cleans up)
violations = lint("myproject.toe")

# Lint with explicit TouchDesigner path
violations = lint("myproject.toe", td_path="/Applications/TouchDesigner.app/Contents/MacOS")

# Keep expanded files for debugging
violations = lint("myproject.toe", keep_files_after_expand=True)

# Lint a .toe.dir directory
violations = lint("myproject.toe.dir")

# With configuration
violations = lint(
    "myproject.toe",
    config_path="td-linter.yaml",
    select=["S", "C"],        # Only syntax and connection rules
    ignore=["F003"],          # Skip heavy texture chains rule
    validate_embedded=True,
)

# Filter by severity
errors = [v for v in violations if v.severity == "error"]
warnings = [v for v in violations if v.severity == "warning"]
```

### lint_and_check()

Convenience function that returns pass/fail status.

```python
def lint_and_check(
    path: str | Path,
    fail_on_warning: bool = False,
    **kwargs,
) -> tuple[bool, list[Violation]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to `.toe` file or `.toe.dir` directory |
| `fail_on_warning` | `bool` | `False` | If True, warnings also cause failure |
| `**kwargs` | | | Additional arguments passed to `lint()` |

**Returns:** `tuple[bool, list[Violation]]` - (passed, violations)

**Example:**

```python
from td_linter import lint_and_check
import sys

passed, violations = lint_and_check(
    "myproject.toe.dir",
    fail_on_warning=True,
    select=["S", "C", "T"],
)

if not passed:
    for v in violations:
        print(f"{v.severity.upper()}: {v.message}")
    sys.exit(1)

print("Linting passed!")
```

## TouchDesigner Tools

The `td_tools` module provides utilities for discovering TouchDesigner installations and working with `.toe` files.

### find_touchdesigner()

Find TouchDesigner installation directory containing `toeexpand`/`toecollapse`.

```python
from td_linter.td_tools import find_touchdesigner

def find_touchdesigner(td_path: Path | None = None) -> Path | None
```

**Search order:**
1. Explicit `td_path` argument (if provided)
2. `TOUCHDESIGNER_PATH` environment variable
3. Common installation locations for the current platform
4. PATH lookup for `toeexpand` executable

**Returns:** Path to directory containing toeexpand/toecollapse, or None if not found.

**Example:**

```python
from td_linter.td_tools import find_touchdesigner

# Auto-discover TouchDesigner
td_dir = find_touchdesigner()
if td_dir:
    print(f"Found TouchDesigner at: {td_dir}")
else:
    print("TouchDesigner not found")

# Use explicit path
td_dir = find_touchdesigner(Path("/Applications/TouchDesigner.app"))
```

### expand_toe()

Expand a `.toe` file to a `.toe.dir` directory.

```python
from td_linter.td_tools import expand_toe

def expand_toe(
    toe_path: Path,
    output_dir: Path | None = None,
    td_path: Path | None = None,
    timeout: int = 60,
) -> Path
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `toe_path` | `Path` | required | Path to the `.toe` file to expand |
| `output_dir` | `Path \| None` | `None` | Output directory (default: same as `.toe` file) |
| `td_path` | `Path \| None` | `None` | Path to TouchDesigner bin directory |
| `timeout` | `int` | `60` | Command timeout in seconds |

**Returns:** Path to the expanded `.toe.dir` directory.

**Raises:**
- `TouchDesignerNotFoundError` - If TouchDesigner cannot be found
- `ToeExpandError` - If expansion fails
- `FileNotFoundError` - If toe_path doesn't exist

**Example:**

```python
from td_linter.td_tools import expand_toe

# Expand a .toe file
toe_dir = expand_toe(Path("project.toe"))
print(f"Expanded to: {toe_dir}")

# Expand to specific location
toe_dir = expand_toe(
    Path("project.toe"),
    output_dir=Path("/tmp"),
    timeout=120,
)
```

### collapse_toe()

Collapse a `.toe.dir` directory back to a `.toe` file.

```python
from td_linter.td_tools import collapse_toe

def collapse_toe(
    toe_dir: Path,
    td_path: Path | None = None,
    timeout: int = 60,
) -> Path
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `toe_dir` | `Path` | required | Path to the `.toe.dir` directory |
| `td_path` | `Path \| None` | `None` | Path to TouchDesigner bin directory |
| `timeout` | `int` | `60` | Command timeout in seconds |

**Returns:** Path to the collapsed `.toe` file.

**Raises:**
- `TouchDesignerNotFoundError` - If TouchDesigner cannot be found
- `ToeCollapseError` - If collapse fails
- `FileNotFoundError` - If toe_dir doesn't exist

### Exception Classes

```python
from td_linter.td_tools import (
    TouchDesignerNotFoundError,
    ToeExpandError,
    ToeCollapseError,
)
```

| Exception | Description |
|-----------|-------------|
| `TouchDesignerNotFoundError` | TouchDesigner installation cannot be found |
| `ToeExpandError` | `toeexpand` command failed |
| `ToeCollapseError` | `toecollapse` command failed |

## Data Types

### Violation

Represents a single lint rule violation.

```python
@dataclass
class Violation:
    rule: str                           # Rule ID (e.g., "C001")
    message: str                        # Human-readable message
    path: str                           # Operator path
    severity: str = "error"             # "error", "warning", or "info"
    source_file: Path | None = None     # Source file path
    line: int | None = None             # Line number (1-indexed)
    context: dict[str, object] = {}     # Additional context
    fix: Fix | None = None              # Optional auto-fix
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `rule` | `str` | Rule ID (e.g., "C001", "no-invalid-cycles") |
| `message` | `str` | Human-readable description of the issue |
| `path` | `str` | Operator path where violation occurred |
| `severity` | `str` | "error", "warning", or "info" |
| `source_file` | `Path \| None` | Absolute path to source file |
| `line` | `int \| None` | Line number (1-indexed) |
| `context` | `dict` | Additional context (cycle path, etc.) |
| `fix` | `Fix \| None` | Auto-fix information if available |

### Severity

Severity levels for violations.

```python
class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
```

**Example:**

```python
from td_linter import lint, Severity

violations = lint("project.toe.dir")
errors = [v for v in violations if v.severity == Severity.ERROR.value]
```

### Fix

Represents an automatic fix for a violation.

```python
@dataclass
class Fix:
    description: str                    # What the fix does
    replacements: list[Replacement]     # Text changes to apply
```

### Replacement

A text replacement within a file.

```python
@dataclass
class Replacement:
    file_path: Path                     # File to modify
    start_line: int                     # Starting line (1-indexed)
    end_line: int                       # Ending line (1-indexed, exclusive)
    start_col: int | None = None        # Start column (0-indexed)
    end_col: int | None = None          # End column (0-indexed, exclusive)
    new_text: str = ""                  # Replacement text
    content_hash: str | None = None     # SHA-256 hash for verification
```

## Fix System

### FixApplier

Applies fixes from violations to files.

```python
class FixApplier:
    def __init__(
        self,
        dry_run: bool = False,
        project_root: Path | None = None,
        verify_hashes: bool = True,
    ) -> None: ...
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dry_run` | `bool` | `False` | Preview changes without modifying files |
| `project_root` | `Path \| None` | `None` | Restrict fixes to this directory (security) |
| `verify_hashes` | `bool` | `True` | Verify file content hasn't changed |

**Methods:**

#### apply()

```python
def apply(self, violations: Sequence[Violation]) -> FixResult
```

Apply fixes from violations. Returns `FixResult` with counts.

#### preview()

```python
def preview(self, violations: Sequence[Violation]) -> dict[Path, str]
```

Preview fixes without applying. Returns dict of file paths to new content.

### FixResult

Result of applying fixes.

```python
@dataclass
class FixResult:
    applied: list[Fix]                  # Successfully applied fixes
    failed: list[tuple[Fix, Exception]] # Fixes that failed
    skipped: list[tuple[Fix, str]]      # Skipped fixes with reason

    @property
    def success_count(self) -> int: ...
    @property
    def failure_count(self) -> int: ...
    @property
    def skipped_count(self) -> int: ...
    @property
    def total_count(self) -> int: ...
```

**Example:**

```python
from pathlib import Path
from td_linter import lint
from td_linter.fix import FixApplier

violations = lint("project.toe.dir")
fixable = [v for v in violations if v.fix is not None]

# Preview first
applier = FixApplier(dry_run=True)
previews = applier.preview(fixable)
for file_path, content in previews.items():
    print(f"Would modify: {file_path}")

# Apply for real
applier = FixApplier(
    dry_run=False,
    project_root=Path("project.toe.dir"),  # Security: restrict to project
)
result = applier.apply(fixable)

print(f"Applied: {result.success_count}")
print(f"Failed: {result.failure_count}")
print(f"Skipped: {result.skipped_count}")
```

## Watch Mode API

### TDLintWatcher

Basic file watcher with callback.

```python
class TDLintWatcher:
    def __init__(
        self,
        toe_dir: Path,
        on_change: Callable[[], None],
        debounce_delay: float = 0.5,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def is_running(self) -> bool: ...
```

**Example:**

```python
from pathlib import Path
from td_linter.watch import TDLintWatcher

def on_change():
    violations = lint("project.toe.dir")
    print(f"Found {len(violations)} violations")

watcher = TDLintWatcher(
    Path("project.toe.dir"),
    on_change=on_change,
    debounce_delay=0.5,
)

# Start watching
watcher.start()
print("Watching... press Ctrl+C to stop")

# Or use as context manager
with TDLintWatcher(Path("project.toe.dir"), on_change) as w:
    while w.is_running:
        time.sleep(1)
```

### EventBasedWatcher

Advanced watcher with event stream.

```python
class EventBasedWatcher:
    def __init__(
        self,
        toe_dir: Path,
        lint_callback: Callable[[Path], Sequence[Violation]],
        debounce_delay: float = 0.5,
    ) -> None: ...

    @property
    def events(self) -> EventStream: ...
    def start(self) -> None: ...
    def stop(self, reason: str = "user_interrupt") -> None: ...
    @property
    def is_running(self) -> bool: ...
```

**Event Types:**

| Event Type | Description | Attributes |
|------------|-------------|------------|
| `WatcherStartEvent` | Watcher started | `toe_dir` |
| `WatcherStopEvent` | Watcher stopped | `toe_dir`, `reason` |
| `FileChangeEvent` | File changed | `toe_dir`, `file_path` |
| `LintStartEvent` | Lint started | `toe_dir` |
| `LintCompleteEvent` | Lint finished | `toe_dir`, `violations`, `duration_ms` |
| `LintErrorEvent` | Lint failed | `toe_dir`, `error`, `error_message` |

**Example:**

```python
from pathlib import Path
from td_linter import lint
from td_linter.watch import EventBasedWatcher
from td_linter.events import EventType

def lint_project(toe_dir: Path):
    return lint(str(toe_dir))

watcher = EventBasedWatcher(
    Path("project.toe.dir"),
    lint_callback=lint_project,
)

# Subscribe to specific events
watcher.events.subscribe(
    lambda e: print(f"Lint complete: {e.error_count} errors"),
    event_types={EventType.LINT_COMPLETE}
)

watcher.events.subscribe(
    lambda e: print(f"Error: {e.error_message}"),
    event_types={EventType.LINT_ERROR}
)

watcher.start()
```

### run_event_based_watch()

Convenience function for event-based watching.

```python
def run_event_based_watch(
    toe_dir: Path,
    lint_callback: Callable[[Path], Sequence[Violation]],
    on_event: Callable[[WatchEvent], None] | None = None,
    debounce_delay: float = 0.5,
    check_interval: float = 1.0,
) -> None
```

## Rule System

### LintRule Base Class

Base class for creating custom rules.

```python
class LintRule(ABC):
    def __init__(self, options: dict[str, OptionValue] | None = None) -> None: ...

    @property
    @abstractmethod
    def rule_id(self) -> str: ...      # e.g., "CUSTOM001"

    @property
    @abstractmethod
    def name(self) -> str: ...          # e.g., "my-custom-rule"

    @property
    @abstractmethod
    def description(self) -> str: ...   # What the rule checks

    @property
    def category(self) -> str: ...      # Derived from rule_id prefix

    @property
    def severity(self) -> str: ...      # Default: "error"

    @property
    def fixable(self) -> bool: ...      # Default: False

    def get_option(self, key: str, default: T = None) -> T: ...

    @abstractmethod
    def check(self, graph: nx.DiGraph) -> Iterator[Violation]: ...
```

**Example Custom Rule:**

```python
from typing import Iterator
import networkx as nx
from td_linter.rules.base import LintRule, Violation

class NoEmptyContainers(LintRule):
    @property
    def rule_id(self) -> str:
        return "CUSTOM001"

    @property
    def name(self) -> str:
        return "no-empty-containers"

    @property
    def description(self) -> str:
        return "COMP containers should have at least one child"

    @property
    def severity(self) -> str:
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        max_depth = self.get_option("max_depth", 10)

        for node_path in graph.nodes:
            data = graph.nodes[node_path]
            if data.get("family") == "COMP":
                if not list(graph.successors(node_path)):
                    yield Violation(
                        rule=self.rule_id,
                        message="Empty COMP container",
                        path=node_path,
                        severity=self.severity,
                    )
```

## Configuration Models

### LintConfigModel

Pydantic model for validating configuration files.

```python
class LintConfigModel(BaseModel):
    version: str = "1.0.0"
    extends: PresetName | list[PresetName] | None = None
    select: list[str] | None = None
    ignore: list[str] | None = None
    rules: dict[str, RuleConfigModel] = {}
    plugins: list[dict[str, str]] | None = None
```

### RuleConfigModel

Configuration for a single rule.

```python
class RuleConfigModel(BaseModel):
    enabled: bool = True
    severity: Severity = Severity.ERROR
    options: dict[str, Any] = {}
```

### PresetName

Available configuration presets.

```python
class PresetName(str, Enum):
    RECOMMENDED = "recommended"
    STRICT = "strict"
    MINIMAL = "minimal"
    PEDANTIC = "pedantic"
```

## Complete Examples

### Build Script Integration

```python
#!/usr/bin/env python3
"""Lint TouchDesigner project as part of build."""

import sys
from pathlib import Path
from td_linter import lint_and_check, Severity

def main():
    project = Path("project.toe.dir")

    if not project.exists():
        print(f"Error: {project} not found")
        sys.exit(1)

    passed, violations = lint_and_check(
        project,
        fail_on_warning=True,
        select=["S", "C", "T", "R"],  # Skip GLSL/Python/Performance
    )

    # Print summary
    errors = sum(1 for v in violations if v.severity == "error")
    warnings = sum(1 for v in violations if v.severity == "warning")
    print(f"Lint results: {errors} errors, {warnings} warnings")

    # Print violations
    for v in violations:
        prefix = "ERROR" if v.severity == "error" else "WARN"
        print(f"  [{prefix}] {v.rule}: {v.message}")
        print(f"         at {v.path}")

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
```

### Pre-commit Hook Script

```python
#!/usr/bin/env python3
"""Pre-commit hook for td-linter."""

import subprocess
import sys
from pathlib import Path

from td_linter import lint

def find_toe_dirs() -> list[Path]:
    """Find changed .toe.dir directories."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
    )
    changed_files = result.stdout.strip().split("\n")

    toe_dirs = set()
    for f in changed_files:
        parts = Path(f).parts
        for i, part in enumerate(parts):
            if part.endswith(".toe.dir"):
                toe_dirs.add(Path(*parts[:i+1]))
                break

    return list(toe_dirs)

def main():
    toe_dirs = find_toe_dirs()
    if not toe_dirs:
        sys.exit(0)

    total_errors = 0
    for toe_dir in toe_dirs:
        if not toe_dir.exists():
            continue

        violations = lint(toe_dir)
        errors = [v for v in violations if v.severity == "error"]
        total_errors += len(errors)

        if errors:
            print(f"\n{toe_dir}:")
            for v in errors:
                print(f"  {v.rule}: {v.message}")

    if total_errors:
        print(f"\nFound {total_errors} error(s). Commit aborted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Continuous Watch with Auto-Fix

```python
#!/usr/bin/env python3
"""Watch project and auto-fix violations."""

import time
from pathlib import Path
from td_linter import lint
from td_linter.fix import FixApplier
from td_linter.watch import EventBasedWatcher
from td_linter.events import EventType, LintCompleteEvent

def lint_project(toe_dir: Path):
    return lint(str(toe_dir))

def on_lint_complete(event: LintCompleteEvent):
    print(f"Lint complete: {event.error_count} errors, {event.warning_count} warnings")

    # Auto-fix if there are fixable violations
    fixable = [v for v in event.violations if v.fix is not None]
    if fixable:
        applier = FixApplier(project_root=event.toe_dir)
        result = applier.apply(fixable)
        if result.success_count:
            print(f"Auto-fixed {result.success_count} issue(s)")

def main():
    project = Path("project.toe.dir")

    watcher = EventBasedWatcher(
        project,
        lint_callback=lint_project,
        debounce_delay=0.3,
    )

    watcher.events.subscribe(
        on_lint_complete,
        event_types={EventType.LINT_COMPLETE}
    )

    print(f"Watching {project}... (Ctrl+C to stop)")
    watcher.start()

    try:
        while watcher.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("\nStopped watching")

if __name__ == "__main__":
    main()
```
