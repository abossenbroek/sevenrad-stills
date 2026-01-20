# TDL-040: YAML Rule Schema

---
id: TDL-040
status: pending
priority: high
phase: 4
depends_on: []
blocks: [TDL-041]
---

## Problem Statement

Rules need a formal schema so that configuration files can be validated. Without a schema, typos in config files silently fail. JSON Schema provides validation and IDE autocompletion for YAML configs.

## Acceptance Criteria

- [ ] JSON Schema defines rule structure
- [ ] Schema validates: id, name, severity, category, description
- [ ] Schema supports: enabled, options, examples, fix
- [ ] Schema supports: extends (presets), version
- [ ] Config files validate against schema
- [ ] VSCode/IDE autocompletion works with schema reference

## Files to Create

```
td_linter/
├── schemas/
│   └── td-linter-rules.schema.json
└── tests/
    └── test_config_validation.py
```

## Research Pointers

### JSON Schema Basics

- https://json-schema.org/learn/getting-started-step-by-step
- Draft-07 is widely supported
- VSCode and most YAML editors understand `$schema` reference

### Schema Structure

Top-level:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "rules"],
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "extends": { ... },
    "rules": { ... }
  }
}
```

### Rule Definition Schema

```json
"rule": {
  "type": "object",
  "required": ["id", "name", "severity", "category", "description"],
  "properties": {
    "id": { "type": "string", "pattern": "^[a-z][a-z0-9-]+$" },
    "severity": { "enum": ["error", "warning", "info"] },
    "category": { "enum": ["syntax", "connection", "type", ...] },
    "enabled": { "type": "boolean", "default": true },
    "options": { "type": "object" }
  }
}
```

### Extends Mechanism

```yaml
extends:
  - recommended
  - strict
```

Presets are named bundles of rule configurations. Schema should validate preset names.

### IDE Integration

Add schema reference to config files:
```yaml
# yaml-language-server: $schema=path/to/schema.json
version: "1.0.0"
```

### Validation Libraries

- `jsonschema` Python package
- `yamllint` for additional YAML validation
- Consider build-time schema validation

### Example Rule Definition

```yaml
rules:
  - id: no-invalid-cycles
    name: "No invalid cycles"
    severity: error
    category: connection
    description: |
      Detects cycles that would cause TouchDesigner to hang.
    rationale: |
      Cycles in TOP/SOP chains cause infinite loops.
    examples:
      bad:
        - "displace1 -> blur1 -> displace1"
      good:
        - "feedback1 -> delay1 -> feedback1"
    fix: "Add a feedback operator to create intentional loop"
    enabled: true
```

### Schema Testing

Test that:
- Valid configs pass validation
- Invalid configs (typos, wrong types) fail
- Missing required fields fail
- Unknown fields warn or fail

## Definition of Done

All acceptance criteria checked. Config files validate against schema, IDEs provide autocompletion.
