# TDL-042: Built-in Rule Set

---
id: TDL-042
status: pending
priority: high
phase: 4
depends_on: [TDL-041]
blocks: []
---

## Problem Statement

The linter needs a comprehensive set of built-in rules. Each rule from Phases 2-3 should be wrapped in a Rule class that integrates with the rule system. The "recommended" preset should include rules that catch real issues without excessive noise.

## Acceptance Criteria

- [ ] All Phase 2 validators wrapped as rules
- [ ] All Phase 3 validators wrapped as rules
- [ ] Each rule has: id, severity, category, description
- [ ] Each rule produces Violation objects
- [ ] Rules are configurable via options
- [ ] "recommended" preset defined with sensible defaults
- [ ] 15+ rules in the built-in set

## Files to Create

```
td_linter/
├── rules/
│   ├── builtin/
│   │   ├── __init__.py
│   │   ├── syntax.py        # valid-n-file, valid-parm-file
│   │   ├── connection.py    # cycles, dangling
│   │   ├── type.py          # compatibility
│   │   ├── reference.py     # operator refs
│   │   ├── glsl.py          # GLSL rules
│   │   └── python.py        # Python rules
│   └── presets/
│       └── recommended.yaml
```

## Research Pointers

### Rule Interface

Design a common interface for all rules:

```python
class Rule(ABC):
    id: str
    severity: Severity
    category: str
    description: str

    @abstractmethod
    def check(self, context: LintContext) -> list[Violation]: ...
```

### LintContext

What does each rule need access to?
- Parsed ASTs (.n, .parm files)
- Network graph
- File system access
- Configuration options

Design a context object passed to all rules.

### Violation Structure

```python
@dataclass
class Violation:
    rule: str           # Rule ID
    severity: Severity
    message: str
    path: str           # File path
    line: int | None    # Line number if known
    context: dict       # Additional info
    fix: str | None     # How to fix
```

### Built-in Rules (from spec)

**Syntax**:
- `valid-n-file-syntax`
- `valid-parm-file-syntax`
- `toc-completeness`

**Connection**:
- `no-invalid-cycles`
- `no-dangling-inputs`

**Type**:
- `type-compatibility`

**Reference**:
- `valid-operator-reference`
- `valid-path-references`

**GLSL**:
- `glsl-no-version`
- `glsl-syntax`
- `glsl-td-output`

**Python**:
- `python-syntax`
- `python-undefined-name`
- `td-execute-dat-callbacks`

**Performance** (optional):
- `deep-nesting`
- `excessive-inputs`

**Style** (optional):
- `naming-convention`
- `tile-overlap`

### Rule Registration

Pattern: Rules register themselves or are discovered:

```python
# Option 1: Explicit registration
BUILTIN_RULES = [
    NoInvalidCyclesRule(),
    TypeCompatibilityRule(),
    ...
]

# Option 2: Discovery via base class
def discover_rules():
    return [cls() for cls in Rule.__subclasses__()]
```

### Recommended Preset

```yaml
# recommended.yaml
version: "1.0.0"
rules:
  # All syntax rules: ERROR
  - id: valid-n-file-syntax
    enabled: true
  - id: valid-parm-file-syntax
    enabled: true

  # Connection rules: ERROR
  - id: no-invalid-cycles
    enabled: true
  - id: no-dangling-inputs
    enabled: true

  # Style rules: disabled by default
  - id: naming-convention
    enabled: false
```

### Options Pattern

Rules may have configurable options:

```python
class DeepNestingRule(Rule):
    def check(self, context):
        max_depth = context.options.get('max_depth', 10)
        # ... use max_depth
```

Config:
```yaml
rules:
  - id: deep-nesting
    options:
      max_depth: 15
```

## Definition of Done

All acceptance criteria checked. 15+ rules wrapped and registered in recommended preset.
