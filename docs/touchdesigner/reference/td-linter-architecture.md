# td-linter Architecture Documentation

This document explains the internal architecture of td-linter for developers who want to contribute or understand the codebase.

## System Overview

```
                              ┌──────────────────────────────────────────┐
                              │              CLI Layer                   │
                              │            (cli.py / typer)              │
                              └────────────────────┬─────────────────────┘
                                                   │
            ┌──────────────────────────────────────┼──────────────────────────────────────┐
            │                                      │                                      │
            ▼                                      ▼                                      ▼
┌───────────────────────┐            ┌───────────────────────┐            ┌───────────────────────┐
│     Linter Core       │            │      Watch Mode       │            │     LSP Server        │
│    (linter.py)        │            │     (watch.py)        │            │  (lsp/server.py)      │
│                       │            │                       │            │                       │
│  Orchestrates the     │            │  File monitoring      │            │  Real-time editor     │
│  validation pipeline  │            │  with debouncing      │            │  integration          │
└───────────┬───────────┘            └───────────┬───────────┘            └───────────┬───────────┘
            │                                    │                                    │
            └────────────────────────────────────┴────────────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌─────────────────────┐            ┌─────────────────────────┐            ┌─────────────────────┐
│      Parsers        │            │      Graph Model        │            │   Embedded Code     │
│    (parsers/)       │            │       (graph/)          │            │    (embedded/)      │
│                     │            │                         │            │                     │
│  n_parser.py        │──────────▶│  builder.py             │            │  language_detector  │
│  parm_parser.py     │            │  model.py              │            │  glsl_validator     │
│  toc_parser.py      │            │  types.py              │            │  python_validator   │
└─────────────────────┘            └───────────┬─────────────┘            └─────────────────────┘
                                               │
                                               ▼
                                   ┌─────────────────────────┐
                                   │     Rules Engine        │
                                   │       (rules/)          │
                                   │                         │
                                   │  base.py (LintRule)     │
                                   │  registry.py            │
                                   │  loader.py              │
                                   │  builtin/               │
                                   └───────────┬─────────────┘
                                               │
                                               ▼
                                   ┌─────────────────────────┐
                                   │   Output Formatters     │
                                   │       (output/)         │
                                   │                         │
                                   │  text.py (default)      │
                                   │  json_formatter.py      │
                                   │  sarif.py               │
                                   └─────────────────────────┘
```

## Component Details

### 1. Parsers Module (`parsers/`)

The parsers module converts TouchDesigner text files into structured Python objects using Lark grammars.

#### n_parser.py - Node Definition Parser

Parses `.n` files that define operators.

**Input format:**
```
TOP:displace
tile 260 200 130 72
flags =  viewer 1 parlanguage 0
inputs
{
0 	moviefilein1
1 	chopto1
}
color 0.67 0.67 0.67
view -1 3 0 0 1 1 0 0
end
```

**Output:** `ParsedNFile` dataclass with:
- `family`: Operator family (TOP, CHOP, SOP, DAT, COMP, MAT)
- `op_type`: Operator type (displace, noise, moviefilein, etc.)
- `tile`: Position tuple (x, y, width, height)
- `flags`: Dictionary of flag values
- `inputs`: List of (index, reference_name) tuples
- `color`: RGB color tuple
- `has_errors`: Boolean indicating parse errors

**Key pattern:** Fresh transformer instance per parse to avoid state carryover:
```python
def parse(self, file_path: Path) -> ParsedNFile:
    tree = self._parser.parse(content)
    transformer = NFileTransformer()  # Fresh instance!
    return transformer.transform(tree)
```

#### parm_parser.py - Parameter Parser

Parses `.parm` files containing parameter values.

**Mode flags:**
| Mode | Type | Description |
|------|------|-------------|
| 0 | Constant | Numeric constant value |
| 17 | String | String with expression |
| 32 | Numeric | Numeric with expression |
| 49 | Expression | Python expression |

**Output:** `ParsedParmFile` with parameters dictionary mapping name to (mode, value).

#### toc_parser.py - Table of Contents Parser

Parses the `.toc` manifest file listing all project contents.

**Special entries:** Lines starting with `.` (`.build`, `.start`, etc.)

### 2. Graph Module (`graph/`)

Builds and models the operator network using NetworkX.

#### types.py - Type System

**OperatorFamily enum:**
```python
class OperatorFamily(str, Enum):
    TOP = "TOP"    # Texture operators
    CHOP = "CHOP"  # Channel operators
    SOP = "SOP"    # Surface operators
    DAT = "DAT"    # Data operators
    COMP = "COMP"  # Component operators
    MAT = "MAT"    # Material operators
```

**Type compatibility rules:**
```python
COMPATIBLE_CONNECTIONS = {
    OperatorFamily.TOP: {OperatorFamily.TOP, OperatorFamily.MAT},
    OperatorFamily.CHOP: {OperatorFamily.CHOP},
    OperatorFamily.SOP: {OperatorFamily.SOP, OperatorFamily.MAT},
    OperatorFamily.DAT: {OperatorFamily.DAT},
    OperatorFamily.COMP: {OperatorFamily.COMP},
}
```

**Feedback operators:** `feedback`, `feedbackchop`, `timemachine`, `delay`, `lag`

**Converter operators:** `chopto`, `sopto`, `tochop`, `todat`, `datexec`

#### model.py - Data Models

```python
@dataclass
class TilePosition:
    x: int
    y: int
    width: int
    height: int

@dataclass
class OperatorNode:
    name: str
    family: OperatorFamily
    op_type: str
    path: str                    # e.g., "local/noise1"
    tile: TilePosition
    source_file: Path
    flags: dict[str, int]
    inputs: list[tuple[int, str]]
    color: tuple[float, float, float] | None

@dataclass
class Connection:
    source: str                  # Source operator path
    target: str                  # Target operator path
    input_index: int             # Input slot on target
```

#### builder.py - Graph Construction

Two-phase graph building process:

**Phase 1:** Create all nodes from `.n` files
```python
for n_file in n_files:
    parsed = self._n_parser.parse(n_file)
    node = self._create_node(parsed, n_file, toe_dir)
    graph.add_node(
        node.path,
        operator=node,
        family=node.family.value,
        op_type=node.op_type,
        source_file=str(n_file),
    )
```

**Phase 2:** Create edges from inputs
```python
for input_idx, ref_name in parsed.inputs:
    source_path = self._resolve_reference(ref_name, parent_dir, graph)
    if source_path not in graph.nodes:
        # Create placeholder for missing reference
        missing_path = f"MISSING:{source_path}"
        graph.add_node(missing_path, family="MISSING", missing=True)
        source_path = missing_path
    graph.add_edge(source_path, node_path, input_index=input_idx)
```

**Missing reference handling:** Nodes prefixed with `MISSING:` are placeholders that allow validation rules to report dangling inputs without crashing.

### 3. Rules Module (`rules/`)

#### base.py - Rule Infrastructure

```python
@dataclass
class Violation:
    rule: str                    # Rule ID ("C001")
    message: str                 # Human-readable message
    path: str                    # Operator path
    severity: str = "error"
    source_file: Path | None = None
    line: int | None = None
    context: dict = field(default_factory=dict)
    fix: Fix | None = None       # Optional auto-fix

class LintRule(ABC):
    @property
    @abstractmethod
    def rule_id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self, graph: nx.DiGraph) -> Iterator[Violation]: ...
```

**Category codes:**
| Code | Category |
|------|----------|
| S | Syntax |
| C | Connection |
| T | Type |
| R | Reference |
| G | GLSL |
| P | Python |
| F | Performance |

#### registry.py - Rule Management

The `RuleRegistry` class manages rule instances, configuration, and filtering:

- `enabled()` - Returns list of enabled rules
- `select(patterns)` - Enable rules matching patterns
- `by_category(code)` - Get rules by category code
- `list_all_rules()` - List all rules with info

#### loader.py - Configuration Loading

Loads `td-linter.yaml` configuration:
1. Validates with Pydantic models
2. Applies preset inheritance
3. Merges rule-specific overrides

#### builtin/ - Built-in Rules

| Category | Rules |
|----------|-------|
| connection.py | NoInvalidCyclesRule, NoDanglingInputsRule |
| type_rules.py | TypeCompatibilityRule |
| reference.py | ValidOperatorReferencesRule |
| syntax.py | ValidNFileSyntaxRule, ValidParmSyntaxRule |
| glsl.py | GLSLSyntaxRule, NoGLSLVersionRule |
| python_rules.py | PythonSyntaxRule, SafePythonRule |
| performance.py | DeepNestingRule, ExcessiveInputsRule |

### 4. Embedded Code Module (`embedded/`)

#### language_detector.py

Detects GLSL vs Python in `.text` files:

```python
class LanguageDetector:
    def detect(self, content: str) -> DetectionResult:
        """Returns Language.GLSL, Language.PYTHON, or Language.UNKNOWN"""
```

**Detection heuristics:**
- GLSL indicators: `vec2/3/4`, `uniform`, `sampler2D`, `fragColor`, `TDOutputSwizzle`
- Python indicators: `def`, `import`, `op(`, `me.`, `project.`

#### glsl_validator.py

Validates GLSL shader code using `glslangValidator`:

1. Prepends TouchDesigner preamble (uniforms, in/out variables)
2. Writes to temp file
3. Runs `glslangValidator`
4. Parses output for errors
5. Adjusts line numbers (accounts for preamble)
6. Filters "undefined uniform" warnings (TD injects these)

#### python_validator.py

AST-based Python validation:

1. Parse with `ast.parse()`
2. Walk AST for undefined names
3. Check against TouchDesigner API stubs
4. Report syntax errors and warnings

### 5. Output Module (`output/`)

#### base.py - Formatter Interface

```python
class OutputFormatter(ABC):
    @abstractmethod
    def format(
        self,
        violations: list[Violation],
        project_path: str | None = None,
    ) -> str: ...
```

#### Implementations

- **text.py**: Human-readable with colors (Rich)
- **json_formatter.py**: Structured JSON
- **sarif.py**: SARIF format for CI/CD tools

### 6. Supporting Modules

#### fix.py - Auto-Fix System

```python
class FixApplier:
    def apply(self, violations: Sequence[Violation]) -> FixResult
    def preview(self, violations: Sequence[Violation]) -> dict[Path, str]
```

**Security features:**
- Path traversal protection (validates against project boundary)
- Content hash verification (ensures file hasn't changed)
- Symlink resolution (follows all symlinks before validation)

**Application order:** Bottom-up (highest line numbers first) to preserve line numbers.

#### watch.py - File Watching

```python
class TDLintWatcher:
    """Basic watcher with callback"""
    def start(self) -> None
    def stop(self) -> None

class EventBasedWatcher:
    """Advanced watcher with event stream"""
    @property
    def events(self) -> EventStream
```

**Debouncing:** Timer-based, configurable delay (default 0.5s).

**Monitored extensions:** `.n`, `.parm`, `.text`, `.toc`

#### plugins.py - Plugin System

Plugin loading from:
- File paths (`./my_rules.py`)
- Python modules (`my_package.rules`)
- Entry points (`td_linter.rules`)

**Security:** AST-based validation in `plugin_security.py` to prevent dangerous code.

#### cache.py - Performance

`BoundedLRUCache` with TTL-based expiration for:
- File content caching
- Graph node caching in LSP

## Data Flow

### Lint Flow

```
.toe.dir/
    │
    ▼
┌───────────────┐
│  TOC Parser   │───▶ Validate manifest entries
└───────────────┘
    │
    ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  .n files     │────▶│  N Parser     │────▶│ OperatorNodes │
└───────────────┘     └───────────────┘     └───────────────┘
    │                                               │
    ▼                                               ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  .parm files  │────▶│ Parm Parser   │────▶│ Parameters    │
└───────────────┘     └───────────────┘     └───────────────┘
    │                                               │
    ▼                                               ▼
┌───────────────┐                          ┌───────────────┐
│  .text files  │                          │ NetworkX      │
│  (GLSL/Python)│                          │ DiGraph       │
└───────────────┘                          └───────┬───────┘
    │                                              │
    ▼                                              ▼
┌───────────────┐                          ┌───────────────┐
│ Language      │                          │ Graph Rules   │
│ Detector      │                          │ (C, T, R)     │
└───────┬───────┘                          └───────┬───────┘
        │                                          │
   ┌────┴────┐                                     │
   ▼         ▼                                     ▼
┌──────┐ ┌──────┐                          ┌───────────────┐
│ GLSL │ │Python│                          │  Violations   │
│ Valid│ │Valid │──────────────────────────▶               │
└──────┘ └──────┘                          └───────┬───────┘
                                                   │
                                                   ▼
                                           ┌───────────────┐
                                           │  Formatter    │
                                           │ (text/json/   │
                                           │  sarif)       │
                                           └───────────────┘
```

### Watch Mode Flow

```
┌───────────────┐
│ TDLintWatcher │
└───────┬───────┘
        │
        ▼
┌───────────────┐      File events
│   watchdog    │◀───────────────── .n, .parm, .text, .toc changes
│   Observer    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Debounce    │◀──── 0.5s default
│    Timer      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Lint Callback│──────▶ Violations
└───────────────┘
```

### LSP Flow

```
┌─────────────┐          ┌─────────────────────┐
│   Editor    │◀────────▶│  LSP Protocol       │
│ (VS Code,   │   stdio  │  (stdio/tcp/ws)     │
│  Neovim)    │   tcp    │                     │
└─────────────┘   ws     └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ TDLintLanguageServer│
                         │  (pygls)            │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    textDocument/        textDocument/          textDocument/
       didOpen              didSave                didClose
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ find_toe_dir()  │   │ lint_toe_dir()  │   │ clear_diagnostics│
    └─────────────────┘   └────────┬────────┘   └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ violation_to_       │
                         │ diagnostic()        │
                         └────────┬────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ publish_diagnostics │──────▶ Editor
                         └─────────────────────┘
```

## Key Design Decisions

### Why Lark for Parsing?

TouchDesigner's file formats have quirky syntax (mixed indentation, special mode flags). Lark's EBNF grammars provide:
- Formal grammar definition (maintainable, testable)
- Automatic parse tree generation
- Transformer pattern for clean data extraction
- Better error messages than regex-based parsing

### Why NetworkX for Graph?

- Battle-tested cycle detection (`nx.simple_cycles()`)
- Efficient graph traversal algorithms
- Rich attribute support on nodes and edges
- Python-native, no external dependencies

### Why Pydantic for Config?

- Type-safe configuration with validation
- Clear error messages for invalid config
- Schema generation for documentation
- IDE support via type hints

## Security Considerations

1. **Plugin Security** (`plugin_security.py`)
   - AST-based analysis before loading
   - Blocks dangerous imports and constructs

2. **Path Traversal Prevention** (`fix.py`)
   - All paths validated against project boundary
   - Symlinks fully resolved before validation

3. **Config File Size Limits** (`config_models.py`)
   - Maximum 1MB config file (DoS prevention)

4. **URI Validation** (`lsp/uri_utils.py`)
   - Strict URI parsing for LSP

## Extension Points

### Adding New Rules

1. Create rule class in `rules/builtin/`:
```python
class MyRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        # Validation logic
        yield Violation(...)
```

2. Register in `rules/builtin/__init__.py`

3. Add to rules catalog documentation

### Adding New Parsers

1. Create Lark grammar in `grammars/`
2. Create parser class in `parsers/`
3. Create transformer to convert parse tree to dataclass
4. Integrate into `NetworkGraphBuilder`

### Adding New Validators

1. Create validator class in `embedded/`
2. Update `LanguageDetector` if new language
3. Integrate into `_validate_embedded_code()` in `linter.py`

## Testing Strategy

### Test Organization

```
tests/
├── unit/
│   └── td_linter/
│       ├── parsers/         # Parser unit tests
│       ├── embedded/        # Validator unit tests
│       ├── output/          # Formatter unit tests
│       └── test_*.py        # Module unit tests
└── integration/
    └── td_linter/
        ├── test_lint_flow.py     # Full pipeline tests
        └── test_cli_pipeline.py  # CLI integration tests
```

### Fixtures

Located in `docs/touchdesigner/fixtures/projects/`:
- Real `.toe.dir` projects for integration tests
- Both valid and invalid examples

## Performance Considerations

### Caching

- **LSP:** `.toe.dir` path lookup caching
- **FixApplier:** File content caching (1min TTL, 100 file max)
- **Watch mode:** Avoids re-parsing unchanged files

### Debouncing

Watch mode uses timer-based debouncing to batch rapid file changes before re-linting.

### Lazy Imports

Optional dependencies (`watchdog`, `pygls`) imported on demand to speed up CLI startup.
