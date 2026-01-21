# TDL-043: Rule Configuration API

---
id: TDL-043
status: pending
priority: medium
phase: 4
depends_on: [TDL-041]
blocks: []
---

## Problem Statement

Users need to query available rules, check which are enabled, and understand rule categories. The CLI `td-linter rules` command needs an API to list and filter rules. Programmatic users may also want to query rules.

## Acceptance Criteria

- [ ] `get_all_rules()` returns all registered rules
- [ ] `get_enabled_rules()` returns only enabled rules
- [ ] `get_rules_by_category(cat)` filters by category
- [ ] `get_rule(id)` returns specific rule or None
- [ ] Rules sortable by severity, category, name
- [ ] API usable by CLI and programmatic consumers

## Files to Create

```
td_linter/
├── rules/
│   └── api.py
└── tests/
    └── test_rules_api.py
```

## Research Pointers

### API Design

```python
class RuleRegistry:
    def __init__(self, config: Config):
        self.rules = load_rules(config)

    def all(self) -> list[RuleConfig]:
        """All registered rules."""
        return list(self.rules.values())

    def enabled(self) -> list[RuleConfig]:
        """Only enabled rules."""
        return [r for r in self.rules.values() if r.enabled]

    def by_category(self, category: str) -> list[RuleConfig]:
        """Rules in a category."""
        return [r for r in self.rules.values() if r.category == category]

    def get(self, rule_id: str) -> RuleConfig | None:
        """Get rule by ID."""
        return self.rules.get(rule_id)
```

### CLI Integration

```python
@app.command()
def rules(category: str | None = None):
    registry = RuleRegistry(config)

    if category:
        rules = registry.by_category(category)
    else:
        rules = registry.all()

    # Display table
    for rule in sorted(rules, key=lambda r: (r.category, r.id)):
        print(f"{rule.id}: {rule.severity} ({rule.category})")
```

### Categories (from spec)

| Category | Description |
|----------|-------------|
| syntax | File format issues |
| connection | Graph structure issues |
| type | Type compatibility |
| reference | Operator references |
| glsl | GLSL shader issues |
| python | Python script issues |
| performance | Performance concerns |
| style | Style/naming |

### Rule Documentation

Consider: Should rules provide their full description/examples via API?

```python
@dataclass
class RuleConfig:
    id: str
    name: str
    severity: Severity
    category: str
    description: str
    rationale: str | None = None
    examples: dict | None = None  # bad/good examples
    fix: str | None = None
```

### Filter Combinations

Support multiple filters:
```python
# Category AND enabled
registry.by_category('glsl').filter(lambda r: r.enabled)
```

Or provide combined method:
```python
def query(self, category=None, enabled=None, severity=None):
    rules = self.all()
    if category:
        rules = [r for r in rules if r.category == category]
    if enabled is not None:
        rules = [r for r in rules if r.enabled == enabled]
    if severity:
        rules = [r for r in rules if r.severity == severity]
    return rules
```

### Display Formatting

For CLI output, consider:
- Table format (rich.Table)
- JSON output for scripting
- Markdown output for documentation

## Definition of Done

All acceptance criteria checked. API supports CLI and programmatic rule queries.
