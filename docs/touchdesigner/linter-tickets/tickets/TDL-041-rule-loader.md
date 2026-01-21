# TDL-041: Rule Loader Implementation

---
id: TDL-041
status: pending
priority: high
phase: 4
depends_on: [TDL-040]
blocks: [TDL-042, TDL-043]
---

## Problem Statement

Rules are defined in YAML files. The loader must parse these files, resolve `extends` references to presets, merge configurations, and provide rules to the validator. Bad config should produce clear error messages, not cryptic failures.

## Acceptance Criteria

- [ ] Loads rules from YAML file
- [ ] Validates against schema (TDL-040)
- [ ] Resolves `extends` presets (recommended, strict, minimal)
- [ ] Merges user overrides onto preset defaults
- [ ] Handles missing config gracefully (use defaults)
- [ ] Clear error messages for invalid config
- [ ] Test cases for various config scenarios

## Files to Create

```
td_linter/
├── rules/
│   ├── loader.py
│   ├── presets/
│   │   ├── recommended.yaml
│   │   ├── strict.yaml
│   │   └── minimal.yaml
│   └── builtin.yaml         # All built-in rules
└── tests/
    └── test_rule_loader.py
```

## Research Pointers

### YAML Loading

- `pip install pyyaml`
- `yaml.safe_load()` - Use safe_load, never load()
- Handle file not found, parse errors

### Preset Resolution

```python
def load_config(path):
    config = load_yaml(path)

    # Resolve extends
    for preset in config.get('extends', []):
        preset_rules = load_preset(preset)
        merge_rules(base=preset_rules, override=config['rules'])

    return config
```

### Merge Strategy

When user config overrides preset:
- Rule with same ID: user wins
- User can disable preset rules: `enabled: false`
- User can change severity: `severity: warning`

```python
def merge_rules(base: list, override: list):
    # Build dict keyed by rule ID
    result = {r['id']: r for r in base}
    for rule in override:
        if rule['id'] in result:
            result[rule['id']].update(rule)
        else:
            result[rule['id']] = rule
    return list(result.values())
```

### RuleConfig Dataclass

```python
@dataclass
class RuleConfig:
    id: str
    name: str
    severity: Severity
    category: str
    description: str
    enabled: bool = True
    options: dict = field(default_factory=dict)
```

### Preset Design

**recommended**: Sensible defaults for most projects
- All error-level syntax rules enabled
- All error-level connection rules enabled
- Info-level style rules disabled

**strict**: All rules enabled at original severity

**minimal**: Only critical rules that prevent crashes

### Error Handling

```python
try:
    config = load_yaml(path)
except FileNotFoundError:
    logger.info("No config found, using defaults")
    return default_config()
except yaml.YAMLError as e:
    raise ConfigError(f"Invalid YAML: {e}")
```

### Config Discovery

Look for config in order:
1. Explicit `--config path`
2. `td-linter.yaml` in project root
3. `.td-linter.yaml` in project root
4. Default (recommended preset)

### Testing Scenarios

| Scenario | Expected |
|----------|----------|
| No config file | Use defaults |
| Empty config | Use defaults |
| extends only | Use preset |
| extends + overrides | Merge correctly |
| Invalid YAML | Clear error |
| Unknown rule ID | Warning |

## Definition of Done

All acceptance criteria checked. Loader handles all config scenarios gracefully.
