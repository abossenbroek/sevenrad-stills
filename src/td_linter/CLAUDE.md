# td-linter - TouchDesigner Project Validator

## Overview

`td-linter` validates expanded TouchDesigner `.toe.dir` projects before they are collapsed back to binary `.toe` files. It catches structural errors, broken references, and type mismatches that would otherwise only surface at runtime.

## Validation Philosophy

**Ground truth**: If TouchDesigner opens a `.toe` file successfully, it's valid. Any linter errors on working TD projects are false positives in our linter—fix the linter, not the project.

## Architecture

See `docs/touchdesigner/reference/linter_spec.md` for full architecture.

Key directories: `grammars/`, `parsers/`, `graph/`, `rules/`, `embedded/`, `output/`

## Workflow

### Direct .toe File Linting (Recommended)

```bash
# Lint a .toe file directly (auto-expands, lints, cleans up)
uv run td-linter lint project.toe

# Keep expanded files for debugging
uv run td-linter lint project.toe --keep-files-after-expand

# Specify TouchDesigner path explicitly
uv run td-linter lint project.toe --td-path /Applications/TouchDesigner.app/Contents/MacOS
```

### Manual Workflow (Alternative)

```bash
# Expand binary .toe to .toe.dir (requires TouchDesigner installed)
toeexpand project.toe

# Lint the expanded project
uv run td-linter lint project.toe.dir

# Collapse back to binary (after fixes)
toecollapse project.toe.dir
```

## TouchDesigner Path Discovery

td-linter automatically finds TouchDesigner in this order:
1. `--td-path` CLI option
2. `TOUCHDESIGNER_PATH` environment variable
3. Common installation locations:
   - macOS: `/Applications/TouchDesigner*.app/Contents/MacOS/`
   - Windows: `C:\Program Files\Derivative\TouchDesigner*\bin\`
   - Linux: `/opt/TouchDesigner*/bin/`
4. `toeexpand`/`toecollapse` in PATH

## Discovering Grammar Gaps

Real-world files expose parser gaps that synthetic tests miss. When linting fails on a valid TD project:

1. Read the actual file that failed to parse
2. Find the unrecognized block/directive
3. Add to grammar (`.lark` file)
4. Add transformer handler (parser `.py` file)
5. Keep the real-world file as a regression fixture

## Embedded Code Validation

TD injects runtime context not present in raw code. For GLSL uniforms/structs:

1. **Check official docs first**: https://docs.derivative.ca/Write_a_GLSL_TOP
2. TD preamble stubs are in `embedded/glsl_validator.py`
3. For undocumented fields: use Info DAT in TD to inspect compiled shaders

False positives about "undefined" TD builtins → check docs, then update preamble.

## TouchDesigner File Formats

### .n files (Node Definitions)
```
TOP:switch
tile 1250 -420 130 72
flags =  viewer 1
inputs
{
0 	in2
}
exports
{
null1
}
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

## Commands

```bash
# Lint a .toe file (auto-expands)
uv run td-linter lint project.toe

# Lint a .toe.dir directory
uv run td-linter lint project.toe.dir

# List available rules
uv run td-linter rules

# Create config file
uv run td-linter init

# Verbose output
uv run td-linter lint project.toe -v

# Keep expanded files for debugging
uv run td-linter lint project.toe --keep-files-after-expand

# Specify TouchDesigner path
uv run td-linter lint project.toe --td-path /path/to/TouchDesigner/bin
```

## Testing

```bash
# Run unit tests
uv run pytest tests/unit/td_linter/ -v

# Lint fixtures directly
uv run td-linter lint docs/touchdesigner/fixtures/projects/reference_toe/example.toe.dir
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

2. Register in `linter.py`

## Related Files

- Fixtures: `docs/touchdesigner/fixtures/projects/`
- Format docs: `docs/touchdesigner/reference/toe_dir_format.yaml`
- Specification: `docs/touchdesigner/reference/linter_spec.md`
- Tickets: `docs/touchdesigner/linter-tickets/`
