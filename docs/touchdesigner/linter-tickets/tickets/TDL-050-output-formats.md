# TDL-050: JSON/SARIF Output Formatters

---
id: TDL-050
status: pending
priority: high
phase: 5
depends_on: [TDL-014]
blocks: [TDL-052]
---

## Problem Statement

Different consumers need different output formats. Humans want colored terminal output. CI scripts want JSON. GitHub Code Scanning wants SARIF. The linter needs pluggable output formatters.

## Acceptance Criteria

- [ ] Text formatter: colored, grouped by file, human-readable
- [ ] JSON formatter: stable schema, machine-parseable
- [ ] SARIF formatter: valid SARIF 2.1.0, works with GitHub
- [ ] `--format` flag selects formatter
- [ ] `--no-color` disables terminal colors
- [ ] All formatters produce consistent information

## Files to Create

```
td_linter/
├── output/
│   ├── __init__.py
│   ├── text.py           # Human-readable
│   ├── json.py           # Machine-readable
│   └── sarif.py          # GitHub Code Scanning
└── tests/
    ├── test_text_output.py
    ├── test_json_output.py
    └── test_sarif_output.py
```

## Research Pointers

### Text Output Design

Use rich library for terminal formatting:
- Group violations by file
- Color by severity (red=error, yellow=warning, blue=info)
- Show rule ID, line number, message
- Summary at end

Example:
```
myproject.toe.dir/glsl1.n
  ERROR no-dangling-inputs:4: Input references non-existent: 'movie_in'

myproject.toe.dir/text_shader.text
  WARNING glsl-td-output:12: Should use TDOutputSwizzle()

Summary: 1 error, 1 warning, 0 info
```

### JSON Output Schema

Design a stable schema:
```json
{
  "version": "1.0.0",
  "violations": [
    {
      "rule": "no-dangling-inputs",
      "severity": "error",
      "message": "Input references non-existent: 'movie_in'",
      "path": "myproject.toe.dir/glsl1.n",
      "line": 4,
      "context": { "missing_ref": "movie_in" }
    }
  ],
  "summary": {
    "total": 1,
    "errors": 1,
    "warnings": 0,
    "info": 0
  }
}
```

### SARIF Format

- https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
- GitHub SARIF: https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/sarif-support-for-code-scanning

Required structure:
```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "td-linter",
        "version": "1.0.0"
      }
    },
    "results": [
      {
        "ruleId": "no-dangling-inputs",
        "level": "error",
        "message": { "text": "..." },
        "locations": [{
          "physicalLocation": {
            "artifactLocation": { "uri": "path/to/file" },
            "region": { "startLine": 4 }
          }
        }]
      }
    ]
  }]
}
```

### Severity Mapping

| Internal | Text | JSON | SARIF |
|----------|------|------|-------|
| ERROR | red | "error" | "error" |
| WARNING | yellow | "warning" | "warning" |
| INFO | blue | "info" | "note" |

### Formatter Interface

```python
class OutputFormatter(ABC):
    @abstractmethod
    def format(self, violations: list[Violation]) -> str: ...

class TextFormatter(OutputFormatter): ...
class JSONFormatter(OutputFormatter): ...
class SARIFFormatter(OutputFormatter): ...
```

### Testing

For each formatter:
- Empty violations list
- Single violation
- Multiple violations, multiple files
- All severity levels
- Special characters in messages

For SARIF specifically:
- Validate against SARIF schema
- Test with GitHub upload-sarif action

## Definition of Done

All acceptance criteria checked. All three formats produce valid, consistent output.
