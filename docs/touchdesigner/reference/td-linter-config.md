# td-linter Configuration Reference

Complete reference for td-linter configuration options.

## Configuration File Locations

td-linter searches for configuration in this order:

1. `--config` CLI argument (if provided)
2. `td-linter.yaml` in current directory
3. `td-linter.yml` in current directory
4. `.td-linter.yaml` in current directory
5. `.td-linter.yml` in current directory

## Creating a Config File

```bash
# Create with recommended preset
td-linter init

# Create with specific preset
td-linter init --preset strict
```

## Complete Schema

```yaml
# Schema version (required)
version: "1.0.0"

# Extend a preset configuration
extends: recommended  # or: strict, minimal, pedantic

# Enable only these rules/categories
select:
  - S      # Category: all syntax rules
  - C001   # Specific rule
  - T      # Category: all type rules

# Disable these rules/categories
ignore:
  - F      # Category: all performance rules
  - G003   # Specific rule

# Per-rule configuration
rules:
  C001:
    enabled: true
    severity: error
    options: {}

  F001:
    enabled: true
    severity: warning
    options:
      max_depth: 15

  F003:
    enabled: false  # Completely disable

# Plugin configuration
plugins:
  - path: ./my_rules.py           # Load from file
  - module: my_package.rules      # Load from installed package
```

## Field Reference

### version

**Type:** `string` (semver format)
**Required:** Yes
**Default:** `"1.0.0"`

Schema version for the configuration file. Currently always `"1.0.0"`.

```yaml
version: "1.0.0"
```

### extends

**Type:** `string | list[string]`
**Required:** No
**Default:** None

Inherit from a preset configuration.

**Available presets:**

| Preset | Description | Rules Enabled |
|--------|-------------|---------------|
| `recommended` | Sensible defaults | S, C, T, R (G, P, F as warnings) |
| `strict` | All rules enabled | All rules as errors |
| `minimal` | Crash-prevention only | S001, S002, C001, T001 |
| `pedantic` | Maximum strictness | All rules, stricter thresholds |

```yaml
# Single preset
extends: recommended

# Multiple presets (later overrides earlier)
extends:
  - recommended
  - strict
```

### select

**Type:** `list[string]`
**Required:** No
**Default:** None (use all enabled rules)

Enable only specific rules or categories. Supports:
- Category codes: `S`, `C`, `T`, `R`, `G`, `P`, `F`
- Rule IDs: `S001`, `C002`, `G001`

```yaml
# Only syntax and connection rules
select: [S, C]

# Mix of categories and specific rules
select: [S, C001, T001, G]
```

### ignore

**Type:** `list[string]`
**Required:** No
**Default:** None

Disable specific rules or categories. Same format as `select`.

```yaml
# Disable all performance rules
ignore: [F]

# Disable specific rules
ignore: [F001, G003, P002]
```

### rules

**Type:** `dict[string, RuleConfig]`
**Required:** No
**Default:** `{}`

Per-rule configuration overrides.

**RuleConfig fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable/disable the rule |
| `severity` | `string` | varies | `error`, `warning`, or `info` |
| `options` | `dict` | `{}` | Rule-specific options |

```yaml
rules:
  # Change severity
  C002:
    severity: warning

  # Disable a rule
  F002:
    enabled: false

  # Configure options
  F001:
    enabled: true
    severity: warning
    options:
      max_depth: 20

  F003:
    options:
      max_chain_length: 12
```

### plugins

**Type:** `list[PluginConfig]`
**Required:** No
**Default:** None

Load custom rule plugins.

**Plugin formats:**

```yaml
plugins:
  # From file path (relative or absolute)
  - path: ./my_rules.py
  - path: /absolute/path/to/rules.py

  # From installed Python module
  - module: my_package.rules
  - module: company.td_linter.custom
```

## Rule Options

### F001: deep-nesting

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_depth` | `int` | `10` | Maximum allowed nesting depth |

```yaml
rules:
  F001:
    options:
      max_depth: 15
```

### F002: excessive-inputs

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_inputs` | `int` | `16` | Maximum inputs per operator |

```yaml
rules:
  F002:
    options:
      max_inputs: 24
```

### F003: heavy-texture-chains

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_chain_length` | `int` | `8` | Maximum TOP chain length without cache |

```yaml
rules:
  F003:
    options:
      max_chain_length: 10
```

## Preset Details

### recommended

Default for most projects. Balanced between strictness and practicality.

```yaml
# Equivalent to:
select: [S, C, T, R, G, P, F]
rules:
  G003:
    severity: warning
  P002:
    severity: warning
  F001:
    severity: warning
  F002:
    severity: warning
  F003:
    severity: warning
  F004:
    severity: warning
  F005:
    severity: info
```

### strict

All rules enabled as errors. For projects requiring maximum validation.

```yaml
# Equivalent to:
select: [S, C, T, R, G, P, F]
rules:
  # All rules severity: error
  F001:
    options:
      max_depth: 8
  F002:
    options:
      max_inputs: 12
  F003:
    options:
      max_chain_length: 6
```

### minimal

Only essential rules that prevent crashes or data loss.

```yaml
# Equivalent to:
select: [S001, S002, S003, C001, T001]
ignore: [C002, C003, C004, T002, R, G, P, F]
```

### pedantic

Maximum strictness for code review and quality gates.

```yaml
# Equivalent to strict, plus:
rules:
  F001:
    options:
      max_depth: 5
  F002:
    options:
      max_inputs: 8
  F003:
    options:
      max_chain_length: 4
  F005:
    severity: warning  # Cook-every-frame elevated
```

## Example Configurations

### Development (relaxed)

```yaml
version: "1.0.0"
extends: recommended

# Focus on structural issues during development
ignore: [F, G003]

rules:
  C002:
    severity: warning  # Dangling inputs as warning
```

### CI/CD (strict)

```yaml
version: "1.0.0"
extends: strict

# Ensure clean code before merge
rules:
  F001:
    options:
      max_depth: 10
```

### GLSL-Heavy Project

```yaml
version: "1.0.0"
extends: recommended

# Prioritize shader validation
select: [S, G]

rules:
  G001:
    severity: error
  G002:
    severity: error
  G003:
    severity: warning
```

### Large Project (performance-focused)

```yaml
version: "1.0.0"
extends: recommended

# Skip expensive validation, focus on structure
ignore: [G, P]

rules:
  F001:
    options:
      max_depth: 20  # Larger projects need more depth
  F003:
    enabled: false   # Skip texture chain analysis
```

### Plugin Project

```yaml
version: "1.0.0"
extends: recommended

plugins:
  - path: ./lint_rules/naming.py
  - path: ./lint_rules/studio_standards.py
  - module: acme_td_rules

rules:
  # Configure plugin rules
  ACME001:
    severity: warning
    options:
      prefix: "op_"
```

## CLI Overrides

Configuration can be overridden via CLI:

```bash
# Use specific config file
td-linter lint project.toe.dir --config strict.yaml

# Override select (replaces config)
td-linter lint project.toe.dir --select S,C

# Override ignore (adds to config)
td-linter lint project.toe.dir --ignore F

# Combined
td-linter lint project.toe.dir --select S,C,T --ignore C002
```

## Validation Errors

Common configuration errors:

**Invalid rule ID:**
```
Error: Invalid rule ID 'X001'. Must be category letter + 3 digits (e.g., S001)
```

**Invalid pattern:**
```
Error: Invalid pattern 'syntax'. Must be category code (S, C, T, R, P, G, F) or rule ID (e.g., S001)
```

**Invalid preset:**
```
Error: Invalid preset 'maximum'. Options: recommended, strict, minimal, pedantic
```

**Plugin not found:**
```
Warning: Failed to load plugin ./missing.py
  FileNotFoundError: [Errno 2] No such file or directory
```
