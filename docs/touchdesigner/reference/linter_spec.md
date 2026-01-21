# TouchDesigner .toe.dir Linter Specification

Comprehensive validation framework for expanded TouchDesigner projects (`.toe.dir` directories).

**Version**: 1.1.0-draft
**Status**: Specification
**Last Updated**: 2026-01-14

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [File Format Analysis](#3-file-format-analysis)
4. [Lark Grammar Specification](#4-lark-grammar-specification)
5. [Graph Validation with NetworkX](#5-graph-validation-with-networkx)
6. [LSP Integration for Embedded Code](#6-lsp-integration-for-embedded-code)
7. [Rule Schema Definition](#7-rule-schema-definition)
8. [Example Rules](#8-example-rules)
9. [CLI Interface Design](#9-cli-interface-design)
10. [Integration with Existing Tools](#10-integration-with-existing-tools)
11. [Implementation Phases](#11-implementation-phases)
12. [Research Sources](#12-research-sources)
13. [Changelog](#13-changelog)

---

## 1. Executive Summary

### Problem Statement

Manual creation and editing of `.toe.dir` directories is error-prone:
- Invalid operator connections cause TouchDesigner to hang
- Malformed `.n` or `.parm` files cause cryptic errors
- No validation before `toecollapse` catches issues too late
- Embedded GLSL/Python code errors discovered only at runtime

### Solution

A comprehensive linting framework providing:
- **Syntax validation** via Lark grammars for `.n` and `.parm` files
- **Graph validation** via NetworkX for connection integrity
- **LSP integration** for embedded GLSL and Python code
- **Extensible rule system** via YAML configuration
- **Pre-toecollapse validation** to catch errors early

### Key Benefits

| Benefit | Impact |
|---------|--------|
| Early error detection | Prevent TD hangs and cryptic errors |
| Developer experience | Real-time feedback in editors |
| CI/CD integration | Automated quality gates |
| Knowledge capture | Rules encode best practices |

---

## 2. Architecture Overview

### System Architecture Diagram

```
                              +----------------------------------+
                              |       td-linter CLI              |
                              |   (Python 3.11+ / typer)         |
                              +----------------------------------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
              v                              v                              v
+---------------------------+  +---------------------------+  +---------------------------+
|    Syntax Validator       |  |    Graph Validator        |  |   Embedded Code Validator |
|    (Lark)                 |  |    (NetworkX)             |  |   (LSP Clients)           |
+---------------------------+  +---------------------------+  +---------------------------+
              |                              |                              |
              v                              v                              v
+---------------------------+  +---------------------------+  +---------------------------+
| Lark grammars             |  | DiGraph construction      |  | glslangValidator          |
| - grammar.js (.n files)   |  | - Cycle detection         |  | (GLSL shaders)            |
| - grammar.js (.parm files)|  | - Type compatibility      |  +---------------------------+
+---------------------------+  | - Dangling inputs         |  | Python AST + pylint       |
                               | - Reference validation    |  | (Execute DAT scripts)     |
                               +---------------------------+  +---------------------------+
                                             |
                                             v
                              +---------------------------+
                              |    Rule Engine            |
                              |    (YAML-based rules)     |
                              +---------------------------+
                                             |
                                             v
                              +---------------------------+
                              |    Report Generator       |
                              |    (JSON, SARIF, text)    |
                              +---------------------------+
```

### Component Interaction Flow

```
.toe.dir/
    |
    +-- .toc (manifest) -----> TOC Parser -----> File existence validation
    |
    +-- *.n (nodes) ---------> Lark parser -----> Syntax validation
    |                                    |
    |                                    +-----> NetworkX graph builder
    |                                                    |
    +-- *.parm (params) -----> Lark parser -----> Syntax validation
    |                                    |
    |                                    +-----> Parameter type validation
    |
    +-- *.text (code) -------> Language detection
                                    |
                                    +-- GLSL? --> glslangValidator + shader-language-server
                                    |
                                    +-- Python? --> AST parse + pylint + TD API stubs
```

### Data Flow

```
                    +-------------+
                    | .toe.dir    |
                    | directory   |
                    +------+------+
                           |
                           v
                    +------+------+
                    | File        |
                    | Discovery   |
                    +------+------+
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    +-----+-----+    +-----+-----+    +-----+-----+
    | .n Parser |    |.parm Parser|   |.text Router|
    +-----------+    +-----------+    +-----------+
          |                |                |
          v                v                v
    +-----+-----+    +-----+-----+    +-----+-----+
    | AST       |    | AST       |    | GLSL/Py   |
    | (node def)|    | (params)  |    | Validator |
    +-----------+    +-----------+    +-----------+
          |                |                |
          +----------------+----------------+
                           |
                           v
                    +------+------+
                    | Rule Engine |
                    | (violations)|
                    +------+------+
                           |
                           v
                    +------+------+
                    | Report      |
                    | Generator   |
                    +-------------+
```

---

## 3. File Format Analysis

### 3.1 Node Definition Files (`.n`)

Based on analysis of actual `.toe.dir` contents:

**Structure Pattern**:
```
TYPE:subtype
tile X Y W H
flags = [flag_list]
[inputs { ... }]
[color R G B [A]]
[dock ref_name]
[view ...]
end
```

**Observed Examples**:

```
# Container Component
COMP:container
tile 200 100 400 244
flags =  picked on current on viewer 1 parlanguage 0
color 0.56 0.56 0.56
end

# CHOP Operator
CHOP:noise
tile 50 30 130 90
flags =  viewer 1 parlanguage 0
color 0.67 0.67 0.67
view -1 5 -1 1 0.034375 -1.5 1.5 4 3 33176 1 0 1 *
end

# TOP with inputs
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

# DAT with dock reference
DAT:text
tile 10 -20 160 130
flags =  viewer 1 parlanguage 0
color 0.67 0.67 0.67
dock replicator1
view -1 8 0 1 1 1 0 0 0 0 1 1 0
end
```

**Operator Family Types**:
| Family | Description | Examples |
|--------|-------------|----------|
| `TOP` | Texture Operators | displace, glsl, moviefilein, null |
| `CHOP` | Channel Operators | noise, math, select |
| `SOP` | Surface Operators | box, sphere, in, out |
| `DAT` | Data Operators | text, table, execute |
| `COMP` | Components | container, geo, base |
| `MAT` | Materials | phong, pbr, constant |

### 3.2 Parameter Files (`.parm`)

**Structure Pattern**:
```
?
param_name mode value [expression]
param_name mode value
...
?
```

**Observed Examples**:

```
# Container parameters
?
pageindex 0 1
w 0 1280
h 0 720
top 0 ./out1
borderover 0 off
parentshortcut 0 Project
?

# CHOP noise parameters with expressions
?
type 0 hermite
rough 0 0.25
tx 49 6531 absTime.frame*.6
ty 32 0
tz 32 0
constraint 0 offset
constrmean 0 0.5
end 0 2
autoexportroot 17 "" me.parent()
?

# GLSL TOP parameters (operator references)
?
pixeldat 0 text_shader
computedat 0 glsl1_compute
?
```

**Parameter Mode Flags** (values requiring systematic verification):

> **RF-002 Note**: The mode flag values below are preliminary observations and require
> systematic verification. See Phase 0 task for creating `mode_flag_discovery.toe` to
> document all modes comprehensively. Mode meanings may vary by TouchDesigner version.

| Mode | Observed Meaning (Unverified) |
|------|-------------------------------|
| `0` | Constant value |
| `32` | Default/unchanged |
| `49` | Expression mode |
| `17` | String with expression |

**RF-002 Discovery Task**: Create `mode_flag_discovery.toe` project in Phase 0/1 to:
- Enumerate all parameter mode flags systematically
- Document mode behavior across TD versions (2022, 2023, 2024)
- Create test cases for each mode type
- Build automated extraction of mode meanings from TD Python API

### 3.3 TOC Manifest Files (`.toc`)

**Structure**: Line-delimited file listing all files in the `.toe.dir`:

```
.build
.start
.grps
.root
.parm
project1.n
project1.parm
project1.panel
project1/geo1.n
project1/geo1.parm
...
.application
```

**Special Entries**:
- `.build` - Build metadata
- `.start` - Startup configuration
- `.grps` - Group definitions
- `.root` - Root operator reference
- `.parm` - Root parameters
- `.application` - Application settings

### 3.4 Text Files (`.text`)

**Header Format**: First line indicates file format version:
```
2
*                  [optional comment/metadata]
[content follows]
```

**Content Types**:
1. **GLSL Shaders** (in Text DATs referenced by GLSL TOPs)
2. **Python Scripts** (in Execute DATs, callbacks)
3. **Plain Text** (comments, documentation)

---

## 4. Lark Grammar Specification

### 4.1 Grammar for `.n` Files

**File**: `td_linter/grammars/node.lark`

```javascript
// Lark grammar for TouchDesigner .n (node definition) files
//
// RF-001: Flexible ordering for optional middle elements
// TouchDesigner .n files have a fixed header (type, tile, flags) and footer (end),
// but the middle elements (inputs, color, dock, view) can appear in any order.
// This grammar uses a two-pass approach: first capture all middle elements,
// then validate ordering in semantic analysis if needed.
module.exports = grammar({
  name: 'toedir_node',

  extras: $ => [/\s/],

  rules: {
    // Root: a node definition
    source_file: $ => $.node_definition,

    // RF-001: Use permutation-friendly structure
    // Required order: type -> tile -> flags -> [optional_middle_elements in any order] -> end
    // The optional_middle_elements can appear in any order or be omitted
    node_definition: $ => seq(
      $.type_declaration,
      $.tile_declaration,
      $.flags_declaration,
      // Optional middle elements can appear in any order (0 or more of each)
      // Use repeat with choice to allow flexible ordering
      repeat($.optional_middle_element),
      'end'
    ),

    // RF-001: Group optional elements that can appear in flexible order
    optional_middle_element: $ => choice(
      $.inputs_block,
      $.color_declaration,
      $.dock_declaration,
      $.view_declaration
    ),

    // TYPE:subtype
    type_declaration: $ => seq(
      $.operator_family,
      ':',
      $.operator_type
    ),

    operator_family: $ => choice(
      'TOP', 'CHOP', 'SOP', 'DAT', 'COMP', 'MAT', 'POP'
    ),

    operator_type: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // tile X Y W H - uses semantic tile_coord type
    tile_declaration: $ => seq(
      'tile',
      $.tile_coord,  // x (typically -10000 to 10000)
      $.tile_coord,  // y (typically -10000 to 10000)
      $.tile_coord,  // width (typically 50 to 2000)
      $.tile_coord   // height (typically 50 to 2000)
    ),

    // flags = [flag_list]
    flags_declaration: $ => seq(
      'flags',
      '=',
      repeat($.flag_item)
    ),

    flag_item: $ => choice(
      seq($.flag_name, $.flag_value),
      $.flag_name
    ),

    flag_name: $ => choice(
      'picked', 'current', 'viewer', 'parlanguage',
      'activate', 'render', 'display', 'pickable',
      'on', 'off'
    ),

    flag_value: $ => choice(
      $.mode_flag,  // RF-004: Use semantic mode_flag type
      'on',
      'off'
    ),

    // inputs { 0 ref1 \n 1 ref2 }
    inputs_block: $ => seq(
      'inputs',
      '{',
      repeat($.input_connection),
      '}'
    ),

    input_connection: $ => seq(
      $.input_index,         // RF-004: Use semantic input_index type (0-99)
      $.operator_reference   // operator name
    ),

    operator_reference: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // color R G B [A]
    color_declaration: $ => seq(
      'color',
      $.float,  // R
      $.float,  // G
      $.float,  // B
      optional($.float)  // A
    ),

    // dock operator_name
    dock_declaration: $ => seq(
      'dock',
      $.operator_reference
    ),

    // view [many numeric values and optional string]
    view_declaration: $ => seq(
      'view',
      repeat(choice($.float, $.integer, $.string, '*'))
    ),

    // RF-004: Semantic number types
    // tile_coord: reasonable range for network editor positions (-10000 to 10000)
    tile_coord: $ => /-?[0-9]+/,

    // input_index: valid input slot indices (0-99 covers all TD operators)
    input_index: $ => /[0-9]{1,2}/,

    // mode_flag: parameter mode flags (0-255 observed range)
    mode_flag: $ => /[0-9]{1,3}/,

    // Generic primitives (for view and other flexible contexts)
    integer: $ => /-?[0-9]+/,
    float: $ => /-?[0-9]+(\.[0-9]+)?/,
    string: $ => /"[^"]*"/,

    // Comments (if any)
    comment: $ => seq('#', /.*/),
  }
});
```

### 4.2 Grammar for `.parm` Files

**File**: `td_linter/grammars/parm.lark`

```javascript
// Lark grammar for TouchDesigner .parm (parameter) files
//
// RF-004: Uses semantic number types for mode flags
// This allows validation of mode values against known TD modes
module.exports = grammar({
  name: 'toedir_parm',

  extras: $ => [/[ \t]/],

  rules: {
    source_file: $ => seq(
      $.parm_start,
      repeat($.parameter),
      $.parm_end
    ),

    parm_start: $ => '?',
    parm_end: $ => '?',

    parameter: $ => seq(
      $.param_name,
      $.param_mode,
      $.param_value,
      optional($.param_expression),
      /\n/
    ),

    param_name: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // RF-004: Semantic mode flag type
    // Mode flags indicate parameter type and expression state:
    // 0=constant, 17=string+expr, 32=default, 49=expression
    // Range 0-255 based on observed values (see RF-002 discovery task)
    param_mode: $ => $.mode_flag,

    // RF-004: Semantic type for parameter modes (0-255 observed range)
    mode_flag: $ => /[0-9]{1,3}/,

    param_value: $ => choice(
      $.number,
      $.string_value,
      $.path_reference,
      $.identifier
    ),

    // RF-004: Expression content (for mode 49/17 parameters)
    // Expressions are Python code evaluated at runtime
    param_expression: $ => /[^\n]+/,

    // Value types
    number: $ => /-?[0-9]+(\.[0-9]+)?/,
    string_value: $ => /"[^"]*"/,
    path_reference: $ => /\.\/[a-zA-Z0-9_\/]+/,
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
  }
});
```

**RF-004 Semantic Types Summary**:

| Type | Location | Description | Range |
|------|----------|-------------|-------|
| `tile_coord` | .n files (tile declaration) | Network editor position | -10000 to 10000 |
| `input_index` | .n files (input connections) | Operator input slot | 0-99 |
| `mode_flag` | .n/.parm files | Parameter mode flags | 0-255 |

These semantic types enable:
- Better error messages ("invalid input index" vs "invalid number")
- Range validation during semantic analysis
- Documentation of expected value ranges

### 4.3 Using the Lark Parser

```bash
# Install Lark
pip install lark

# Test grammar in Python
python -c "
from lark import Lark
grammar = open('td_linter/grammars/node.lark').read()
parser = Lark(grammar, start='source_file')
tree = parser.parse(open('test_file.n').read())
print(tree.pretty())
"
```

### 4.4 Python Bindings Usage

```python
from lark import Lark
from pathlib import Path

# Load the grammar
grammar_path = Path(__file__).parent / "grammars" / "node.lark"
parser = Lark(grammar_path.read_text(), start="source_file", parser="lalr")

def parse_node_file(path: str) -> dict:
    """Parse a .n file and return structured data."""
    with open(path, 'rb') as f:
        tree = parser.parse(f.read())

    root = tree.root_node

    # Extract data from AST
    result = {
        'family': None,
        'type': None,
        'tile': None,
        'inputs': [],
        'flags': {},
    }

    for child in root.children:
        if child.type == 'type_declaration':
            result['family'] = child.children[0].text.decode()
            result['type'] = child.children[2].text.decode()
        elif child.type == 'inputs_block':
            for conn in child.children:
                if conn.type == 'input_connection':
                    idx = int(conn.children[0].text.decode())
                    ref = conn.children[1].text.decode()
                    result['inputs'].append((idx, ref))

    return result
```

---

## 5. Graph Validation with NetworkX

### 5.1 Graph Model

TouchDesigner networks form **directed graphs** where:
- **Nodes** = Operators (TOP, CHOP, SOP, DAT, COMP, MAT)
- **Edges** = Connections (from output to input)

```python
import networkx as nx
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

class OperatorFamily(Enum):
    TOP = "TOP"
    CHOP = "CHOP"
    SOP = "SOP"
    DAT = "DAT"
    COMP = "COMP"
    MAT = "MAT"
    POP = "POP"

@dataclass
class OperatorNode:
    """Represents a TouchDesigner operator in the graph."""
    name: str
    family: OperatorFamily
    op_type: str
    path: str  # Full path in network (e.g., /project1/geo1/box1)
    tile: tuple[int, int, int, int]  # x, y, w, h

    @property
    def full_id(self) -> str:
        return f"{self.path}/{self.name}"

@dataclass
class Connection:
    """Represents a connection between operators."""
    source: str  # Source operator path
    target: str  # Target operator path
    input_index: int
```

### 5.2 Graph Construction Algorithm

```python
def build_network_graph(toe_dir: Path) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a .toe.dir directory.

    Returns:
        DiGraph with operator nodes and connection edges
    """
    G = nx.DiGraph()

    # Phase 1: Discover all operators
    operators = {}
    for n_file in toe_dir.rglob("*.n"):
        op_data = parse_node_file(n_file)
        op_path = str(n_file.relative_to(toe_dir).with_suffix(''))

        op_node = OperatorNode(
            name=n_file.stem,
            family=OperatorFamily(op_data['family']),
            op_type=op_data['type'],
            path=op_path,
            tile=tuple(op_data['tile']) if op_data['tile'] else (0,0,0,0),
        )

        operators[op_path] = op_node
        G.add_node(op_path, data=op_node)

    # Phase 2: Build connections from inputs
    for n_file in toe_dir.rglob("*.n"):
        op_data = parse_node_file(n_file)
        target_path = str(n_file.relative_to(toe_dir).with_suffix(''))
        parent_path = str(n_file.parent.relative_to(toe_dir))

        for idx, source_name in op_data.get('inputs', []):
            # Resolve relative reference to full path
            source_path = f"{parent_path}/{source_name}"

            if source_path in operators:
                G.add_edge(
                    source_path,
                    target_path,
                    input_index=idx
                )
            else:
                # Mark as dangling reference
                G.add_edge(
                    f"MISSING:{source_name}",
                    target_path,
                    input_index=idx,
                    missing=True
                )

    return G
```

### 5.3 Validation Algorithms

#### 5.3.1 Cycle Detection

Most TouchDesigner networks should be DAGs (Directed Acyclic Graphs). Cycles can cause infinite loops.

```python
# AG-001: Explicit feedback operator types that legitimize cycles
FEEDBACK_OPERATOR_TYPES = {
    'feedback',      # TOP feedback operator
    'timemachine',   # TOP time-based feedback
    'feedbackchop',  # CHOP feedback operator
    'delay',         # CHOP delay (can create intentional feedback)
    'lag',           # CHOP lag (can be part of feedback)
}

def find_cycles(G: nx.DiGraph) -> list[list[str]]:
    """
    Detect cycles in the operator network.

    Note: Some cycles are valid (feedback loops in CHOPs),
    but most indicate errors.
    """
    try:
        cycles = list(nx.simple_cycles(G))
        return cycles
    except nx.NetworkXNoCycle:
        return []

def check_feedback_property(parm_path: Path) -> bool:
    """
    AG-001: Check if operator has feedback parameter enabled in .parm file.

    Some operators have a 'feedback' parameter that enables legitimate
    feedback behavior without using a dedicated feedback operator.
    """
    if not parm_path.exists():
        return False

    parm_data = parse_parm_file(parm_path)
    # Check for feedback-enabling parameters
    feedback_params = ['feedback', 'feedbackmode', 'usefeedback']
    for param in feedback_params:
        if param in parm_data:
            value = parm_data[param]
            # Check if enabled (non-zero, 'on', or truthy string)
            if value and value not in ('0', 'off', 'Off', 'OFF'):
                return True
    return False

def validate_no_invalid_cycles(G: nx.DiGraph, toe_dir: Path) -> list[Violation]:
    """
    Check for cycles that are likely invalid.

    AG-001: Enhanced feedback detection including:
    - Explicit feedback operator types (FEEDBACK_OPERATOR_TYPES)
    - Operators with feedback parameter enabled in .parm files
    - CHOP-only cycles (traditionally allowed)

    Valid cycles:
    - Contains feedback/timemachine/feedbackchop operator
    - Contains operator with feedback parameter enabled
    - CHOP-only chains (explicit feedback loop)
    - Container hierarchies with cloning

    Invalid cycles:
    - TOP chains that loop back without feedback operator
    - SOP chains that loop back
    """
    violations = []
    cycles = find_cycles(G)

    for cycle in cycles:
        # Check if cycle involves only CHOPs (potentially valid)
        families = [G.nodes[n]['data'].family for n in cycle if n in G.nodes]

        # CHOP-only cycles are traditionally allowed
        if all(f == OperatorFamily.CHOP for f in families):
            continue

        # AG-001: Check for explicit feedback operator types
        has_feedback_op = any(
            G.nodes.get(n, {}).get('data', None) and
            G.nodes[n]['data'].op_type in FEEDBACK_OPERATOR_TYPES
            for n in cycle
        )

        if has_feedback_op:
            continue

        # AG-001: Check for feedback parameter enabled in .parm files
        has_feedback_param = False
        for node in cycle:
            parm_path = toe_dir / f"{node}.parm"
            if check_feedback_property(parm_path):
                has_feedback_param = True
                break

        if has_feedback_param:
            continue

        # No valid feedback mechanism found - report violation
        violations.append(Violation(
            rule='no-invalid-cycles',
            severity=Severity.ERROR,
            message=f"Invalid cycle detected (no feedback operator): {' -> '.join(cycle)}",
            path=cycle[0],
            context={
                'cycle': cycle,
                'hint': 'Add a feedback/timemachine operator or enable feedback parameter'
            }
        ))

    return violations
```

#### 5.3.2 Type Compatibility Validation

Operators can only connect to operators of compatible families (or through conversion operators).

> **CM-002 Note**: The operator lists below are incomplete. A comprehensive `td_operators.yaml`
> database should be created to document all operator types and their valid connections.
> This database should be auto-generated from TouchDesigner's Python API when possible.

```python
# CM-002: Plan to load from td_operators.yaml in the future
# For now, using hardcoded lists that need expansion

# Type compatibility matrix
# Direct connections allowed (without conversion operators)
COMPATIBLE_CONNECTIONS = {
    OperatorFamily.TOP: {OperatorFamily.TOP},
    OperatorFamily.CHOP: {OperatorFamily.CHOP},
    OperatorFamily.SOP: {OperatorFamily.SOP},
    OperatorFamily.DAT: {OperatorFamily.DAT},
    OperatorFamily.MAT: {OperatorFamily.MAT},
    OperatorFamily.POP: {OperatorFamily.POP},
    # COMPs can connect to many things through their internal networks
    OperatorFamily.COMP: {OperatorFamily.COMP, OperatorFamily.TOP},
}

# Conversion operators that bridge families
# CM-002: Expanded list - needs comprehensive audit from TD
CONVERSION_OPERATORS = {
    ('CHOP', 'TOP'): 'chopto',         # CHOP to TOP
    ('TOP', 'CHOP'): 'toptochop',      # TOP to CHOP
    ('SOP', 'CHOP'): 'soptochop',      # SOP to CHOP
    ('CHOP', 'SOP'): 'choptosop',      # CHOP to SOP
    ('DAT', 'CHOP'): 'dattochop',      # DAT to CHOP
    ('SOP', 'DAT'): 'soptodat',        # SOP to DAT
    ('DAT', 'SOP'): 'dattosop',        # DAT to SOP
    ('CHOP', 'DAT'): 'choptodat',      # CHOP to DAT
    ('TOP', 'DAT'): 'toptodat',        # TOP to DAT (for pixel data)
    ('SOP', 'MAT'): 'soptomat',        # SOP to MAT
    ('MAT', 'TOP'): 'mattop',          # MAT render to TOP
}

# CM-002: Additional operators that accept cross-family inputs
# These need special handling in type validation
SPECIAL_INPUT_OPERATORS = {
    'texturesampler': {'accepts': ['TOP', 'MAT']},  # Can sample textures from TOPs/MATs
    'geometryCOMP': {'accepts': ['SOP', 'MAT']},    # Takes SOP geometry and MAT materials
    'geomat': {'accepts': ['SOP']},                  # Material from geometry
    'soptomat': {'accepts': ['SOP']},                # SOP to material
    'glsl': {'accepts': ['TOP', 'DAT']},            # GLSL TOP accepts DAT for shader code
    'render': {'accepts': ['SOP', 'MAT', 'COMP']},  # Render TOP/SOP accepts multiple families
}

# CM-002 TODO: Create td_operators.yaml with structure:
# operators:
#   - name: chopto
#     family: TOP
#     accepts_inputs: [CHOP]
#     output_family: TOP
#     description: "Convert CHOP channels to TOP pixels"
#   - name: texturesampler
#     family: MAT
#     accepts_inputs: [TOP, MAT]
#     ...
#
# Generate this file using:
# for family in td.families:
#     for op_type in op(family).OPType:
#         # Extract input/output info

def validate_type_compatibility(G: nx.DiGraph) -> list[Violation]:
    """Check that all connections are type-compatible."""
    violations = []

    for source, target in G.edges():
        if source.startswith("MISSING:"):
            continue  # Handled by dangling input check

        source_data = G.nodes[source].get('data')
        target_data = G.nodes[target].get('data')

        if not source_data or not target_data:
            continue

        source_family = source_data.family
        target_family = target_data.family

        # Check if target is a conversion operator
        if target_data.op_type in CONVERSION_OPERATORS.values():
            continue  # Conversion operators handle cross-family

        if target_family not in COMPATIBLE_CONNECTIONS.get(source_family, set()):
            violations.append(Violation(
                rule='type-compatibility',
                severity=Severity.ERROR,
                message=f"Incompatible connection: {source_family.value} -> {target_family.value}",
                path=target,
                context={
                    'source': source,
                    'source_family': source_family.value,
                    'target_family': target_family.value,
                }
            ))

    return violations
```

#### 5.3.3 Dangling Input Detection

```python
def validate_no_dangling_inputs(G: nx.DiGraph) -> list[Violation]:
    """Check for inputs that reference non-existent operators."""
    violations = []

    for source, target, data in G.edges(data=True):
        if source.startswith("MISSING:"):
            ref_name = source.replace("MISSING:", "")
            violations.append(Violation(
                rule='no-dangling-inputs',
                severity=Severity.ERROR,
                message=f"Input references non-existent operator: '{ref_name}'",
                path=target,
                context={'missing_ref': ref_name}
            ))

    return violations
```

#### 5.3.4 Reference Validation

```python
def validate_operator_references(
    G: nx.DiGraph,
    toe_dir: Path
) -> list[Violation]:
    """
    Validate that all operator references in .parm files resolve.

    Checks:
    - DAT references (pixeldat, computedat in GLSL TOPs)
    - Path references (./relative or /absolute paths)
    - Clone references
    """
    violations = []

    for parm_file in toe_dir.rglob("*.parm"):
        parm_data = parse_parm_file(parm_file)
        op_path = str(parm_file.relative_to(toe_dir).with_suffix(''))
        parent_path = str(parm_file.parent.relative_to(toe_dir))

        for param_name, param_value in parm_data.items():
            # Check for operator reference parameters
            if param_name in ('pixeldat', 'computedat', 'top', 'chop', 'dat', 'sop'):
                if isinstance(param_value, str) and param_value:
                    # Resolve reference
                    if param_value.startswith('./'):
                        ref_path = f"{parent_path}/{param_value[2:]}"
                    elif param_value.startswith('/'):
                        ref_path = param_value[1:]
                    else:
                        ref_path = f"{parent_path}/{param_value}"

                    if ref_path not in G.nodes:
                        violations.append(Violation(
                            rule='valid-operator-reference',
                            severity=Severity.ERROR,
                            message=f"Parameter '{param_name}' references non-existent operator: '{param_value}'",
                            path=op_path,
                            context={'param': param_name, 'ref': param_value}
                        ))

    return violations
```

### 5.4 Graph Visualization (Debug)

```python
def visualize_network(G: nx.DiGraph, output_path: str):
    """Generate ASCII visualization of the network."""
    try:
        import matplotlib.pyplot as plt

        # Color nodes by family
        family_colors = {
            OperatorFamily.TOP: '#4CAF50',
            OperatorFamily.CHOP: '#2196F3',
            OperatorFamily.SOP: '#FF9800',
            OperatorFamily.DAT: '#9C27B0',
            OperatorFamily.COMP: '#607D8B',
            OperatorFamily.MAT: '#F44336',
        }

        colors = []
        for node in G.nodes():
            data = G.nodes[node].get('data')
            if data:
                colors.append(family_colors.get(data.family, '#999999'))
            else:
                colors.append('#FF0000')  # Missing nodes in red

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=colors, with_labels=True,
                node_size=500, font_size=8, arrows=True)
        plt.savefig(output_path)
        plt.close()
    except ImportError:
        # ASCII fallback
        print("Network Graph (ASCII):")
        for source, target in G.edges():
            print(f"  {source} --> {target}")
```

---

## 6. LSP Integration for Embedded Code

### 6.1 Architecture

```
.text file detected
        |
        v
+-------+-------+
|  Language     |
|  Detection    |
+-------+-------+
        |
   +----+----+
   |         |
   v         v
+--+--+   +--+--+
| GLSL |   | Python |
+--+--+   +--+--+
   |         |
   v         v
glslang     pylint/ast
Validator   + TD stubs
```

### 6.2 Language Detection

```python
from enum import Enum
import re

class EmbeddedLanguage(Enum):
    GLSL = "glsl"
    PYTHON = "python"
    UNKNOWN = "unknown"

def detect_language(text_file: Path, context: dict) -> EmbeddedLanguage:
    """
    Detect the language of a .text file.

    Uses:
    1. Context from parent .n file (DAT type)
    2. Content heuristics
    """
    content = text_file.read_text()

    # Skip version header (first line: "2" or similar)
    lines = content.split('\n')
    if lines and lines[0].strip().isdigit():
        content = '\n'.join(lines[1:])

    # Check context - what type of DAT is this?
    dat_type = context.get('dat_type', '')

    if dat_type in ('execute', 'script', 'callbacks'):
        return EmbeddedLanguage.PYTHON

    # Content heuristics
    glsl_indicators = [
        r'\bvec[234]\b',
        r'\bmat[234]\b',
        r'\buniform\b',
        r'\bsampler2D\b',
        r'\bfragColor\b',
        r'\bTDOutputSwizzle\b',
        r'\bsTD2DInputs\b',
        r'\bvoid\s+main\s*\(',
    ]

    python_indicators = [
        r'\bdef\s+\w+\s*\(',
        r'\bimport\s+',
        r'\bfrom\s+\w+\s+import\b',
        r'\bop\s*\(',
        r'\bme\.',
        r'\bproject\.',
    ]

    glsl_score = sum(1 for p in glsl_indicators if re.search(p, content))
    python_score = sum(1 for p in python_indicators if re.search(p, content))

    if glsl_score > python_score:
        return EmbeddedLanguage.GLSL
    elif python_score > glsl_score:
        return EmbeddedLanguage.PYTHON

    return EmbeddedLanguage.UNKNOWN
```

### 6.3 GLSL Validation

Integration with existing `validate_glsl.py` and shader-language-server.

> **AG-002 Note**: GLSL validation uses a preamble-agnostic approach to avoid version
> lock-in. The validator focuses on syntax errors only and filters out "undefined uniform"
> warnings since TouchDesigner injects its own preamble at runtime with different uniforms
> depending on the TD version and operator configuration.

```python
import subprocess
import tempfile
from pathlib import Path
import re

# AG-002: Minimal preamble for syntax validation only
# This preamble is intentionally minimal to avoid version-specific assumptions.
# TouchDesigner will inject its own complete preamble at runtime.
TD_MINIMAL_FRAGMENT_PREAMBLE = """
#version 330 core

// AG-002: Minimal declarations for syntax validation
// These may not match actual TD runtime - that's intentional
uniform sampler2D sTD2DInputs[8];
uniform vec4 uTDOutputInfo;
uniform int uTDPass;

in vec2 vUV;
layout(location = 0) out vec4 fragColor;

// Stub TD functions - actual implementation provided by TD
vec4 TDOutputSwizzle(vec4 c) { return c; }

// --- USER SHADER BEGINS BELOW ---
"""

TD_MINIMAL_COMPUTE_PREAMBLE = """
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// AG-002: Minimal declarations for syntax validation
uniform sampler2D sTD2DInputs[8];
layout(rgba32f) uniform image2D sTD2DOutputs[8];

// --- USER SHADER BEGINS BELOW ---
"""

# AG-002: Warnings to filter out (not actual errors)
GLSLANG_IGNORED_WARNINGS = [
    r"undefined uniform",
    r"undeclared identifier",
    r"use of undeclared",
    r"implicitly sized",
]

class GLSLValidator:
    """
    Validate GLSL shaders for TouchDesigner compatibility.

    AG-002: Uses preamble-agnostic validation approach:
    - Focuses on syntax errors only
    - Filters out undefined uniform/variable warnings
    - Does not lock to specific GLSL version
    - Documents limitations clearly
    """

    def __init__(self, glslang_path: str = "glslangValidator"):
        self.glslang_path = glslang_path

    def validate(self, text_file: Path, shader_type: str = "frag") -> list[Violation]:
        """
        Validate a GLSL shader file.

        AG-002: Preamble-agnostic validation
        - Uses minimal preamble for syntax checking
        - Filters glslang output to ignore undefined warnings
        - Focuses on syntax errors only

        Limitations (documented per AG-002):
        - Cannot validate TD-specific uniforms exist
        - Cannot verify correct TD function signatures
        - May miss runtime errors that depend on TD version

        Args:
            text_file: Path to the .text file containing GLSL
            shader_type: "frag" for fragment, "comp" for compute

        Returns:
            List of Violation objects (syntax errors only)
        """
        violations = []
        content = self._extract_shader_content(text_file)

        # Check for forbidden #version directive
        if "#version" in content:
            violations.append(Violation(
                rule='glsl-no-version',
                severity=Severity.ERROR,
                message="GLSL shader contains #version directive (TouchDesigner injects this)",
                path=str(text_file),
            ))

        # Select minimal preamble (AG-002)
        preamble = TD_MINIMAL_COMPUTE_PREAMBLE if shader_type == "comp" else TD_MINIMAL_FRAGMENT_PREAMBLE

        # Validate with glslangValidator
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{shader_type}', delete=False) as f:
            f.write(preamble)
            f.write(content)
            temp_path = f.name

        try:
            result = subprocess.run(
                [self.glslang_path, '-S', shader_type, temp_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Parse glslang errors, filtering per AG-002
                for error in self._parse_glslang_errors(result.stdout + result.stderr):
                    # AG-002: Skip ignored warnings (undefined uniforms, etc.)
                    if self._should_ignore_error(error['message']):
                        continue

                    # Adjust line numbers to account for preamble
                    preamble_lines = len(preamble.split('\n'))
                    adjusted_line = error['line'] - preamble_lines if error['line'] > preamble_lines else error['line']

                    violations.append(Violation(
                        rule='glsl-syntax',
                        severity=Severity.ERROR,
                        message=error['message'],
                        path=str(text_file),
                        line=adjusted_line,
                    ))
        finally:
            Path(temp_path).unlink()

        return violations

    def _should_ignore_error(self, message: str) -> bool:
        """
        AG-002: Check if error should be filtered out.

        Filters undefined uniform/variable warnings since TD
        injects its own preamble with different definitions.
        """
        message_lower = message.lower()
        for pattern in GLSLANG_IGNORED_WARNINGS:
            if re.search(pattern, message_lower):
                return True
        return False

    def _extract_shader_content(self, text_file: Path) -> str:
        """Extract shader code, skipping .text file header."""
        content = text_file.read_text()
        lines = content.split('\n')

        # Skip version header and optional metadata line
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        if lines and lines[0].startswith('*'):
            lines = lines[1:]

        return '\n'.join(lines)

    def _parse_glslang_errors(self, output: str) -> list[dict]:
        """Parse glslangValidator error output."""
        errors = []
        for line in output.split('\n'):
            # Format: "ERROR: 0:LINE: message"
            match = re.match(r'ERROR:\s*\d+:(\d+):\s*(.+)', line)
            if match:
                errors.append({
                    'line': int(match.group(1)),
                    'message': match.group(2).strip()
                })
        return errors
```

**AG-002 Validation Limitations**:
- Cannot validate that TD-specific uniforms (sTD2DInputs, uTDOutputInfo) are used correctly
- Cannot verify TD function signatures match runtime expectations
- May miss errors that depend on specific TouchDesigner version
- Recommend runtime testing in TD for complete validation

### 6.4 Python Validation

Validate Python code in Execute DATs and callback scripts.

```python
import ast
from typing import Optional

# TouchDesigner Python API stubs for validation
TD_PYTHON_STUBS = """
# TouchDesigner built-in objects and functions
class _Op:
    def __call__(self, path): ...
    def __getattr__(self, name): ...

class _Me:
    parent: object
    digits: int
    name: str
    path: str
    storage: dict

    def __getattr__(self, name): ...

class _Project:
    realTime: bool
    name: str
    folder: str

    def quit(self): ...
    def save(self, path=None): ...

class _AbsTime:
    frame: float
    seconds: float

op = _Op()
me = _Me()
project = _Project()
absTime = _AbsTime()

def run(expression, delayFrames=0, delayMilliSeconds=0): ...
"""

class PythonValidator:
    """Validate Python code in TouchDesigner DATs."""

    def __init__(self):
        self.td_globals = self._compile_td_stubs()

    def _compile_td_stubs(self) -> dict:
        """Compile TD stubs into a namespace for validation."""
        namespace = {}
        exec(compile(TD_PYTHON_STUBS, '<td_stubs>', 'exec'), namespace)
        return namespace

    def validate(self, text_file: Path) -> list[Violation]:
        """
        Validate Python code from a .text file.

        Checks:
        1. Syntax validity (ast.parse)
        2. Basic semantic checks (undefined names that aren't TD builtins)
        """
        violations = []
        content = self._extract_python_content(text_file)

        # Phase 1: Syntax validation
        try:
            tree = ast.parse(content, filename=str(text_file))
        except SyntaxError as e:
            violations.append(Violation(
                rule='python-syntax',
                severity=Severity.ERROR,
                message=f"Python syntax error: {e.msg}",
                path=str(text_file),
                line=e.lineno,
            ))
            return violations  # Can't continue without valid AST

        # Phase 2: Name resolution check
        undefined = self._find_undefined_names(tree)

        # RF-003: Comprehensive TD builtins set
        # Significantly expanded to cover all TD Python API elements
        td_builtins = {
            # Core modules
            'td', 'tdu', 'TDF', 'TDJSON', 'TDStoreTools', 'TDFunctions',

            # Special objects
            'op', 'me', 'mod', 'ext', 'par', 'storage', 'fetch', 'store',
            'parent', 'ipar', 'iop', 'ui', 'project', 'root', 'absTime',
            'app', 'sysinfo', 'monitors', 'panelexec', 'panel',

            # Common TD functions
            'run', 'cook', 'debug', 'passive', 'var', 'vardict',

            # Callback function names (not flagged as undefined when defined)
            'onCook', 'onPulse', 'onValueChange', 'onOffToOn', 'onOnToOff',
            'onStart', 'onCreate', 'onExit', 'onFrameStart', 'onFrameEnd',
            'onPlayStateChange', 'onDeviceChange', 'onProjectPreSave',
            'onProjectPostSave', 'onParValueChange', 'onParExprChange',
            'onParModeChange', 'onParPulse', 'onParValuePulse',

            # CHOP execute callbacks
            'onOffToOn', 'onOnToOff', 'whileOn', 'whileOff', 'onValueChange',

            # Panel callbacks
            'onSelect', 'onRollover', 'onFocus', 'onDrop',

            # DAT callbacks
            'onTableChange', 'onRowChange', 'onColChange', 'onCellChange',

            # Timer CHOP callbacks
            'onDone', 'onStart', 'onTimerPulse', 'onReady',

            # Web DAT callbacks
            'onReceive', 'onConnect', 'onDisconnect',

            # Keyboard/Mouse
            'onKey', 'onKeyUp', 'onKeyDown', 'onMouse', 'onMouseUp', 'onMouseDown',
            'onMouseMove', 'onMouseDrag', 'onMouseEnter', 'onMouseLeave',
            'onMouseWheel',

            # Other common globals
            'args', 'kwargs', 'channel', 'channels', 'samples', 'sampleIndex',
            'scriptOp', 'dat', 'chop', 'top', 'sop', 'mat', 'comp',
        }

        # RF-003 Note: This set should be updated for different TD versions
        # Consider loading from external config for version-specific builtins

        for name, line in undefined:
            if name not in td_builtins and name not in dir(__builtins__):
                violations.append(Violation(
                    rule='python-undefined-name',
                    severity=Severity.WARNING,
                    message=f"Potentially undefined name: '{name}'",
                    path=str(text_file),
                    line=line,
                ))

        # Phase 3: Check for common TouchDesigner patterns
        violations.extend(self._check_td_patterns(tree, text_file))

        return violations

    def _extract_python_content(self, text_file: Path) -> str:
        """Extract Python code, skipping .text file header."""
        content = text_file.read_text()
        lines = content.split('\n')

        # Skip version header
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        # Skip comment/metadata line starting with * or #
        while lines and (lines[0].startswith('*') or lines[0].startswith('#')):
            lines = lines[1:]

        return '\n'.join(lines)

    def _find_undefined_names(self, tree: ast.AST) -> list[tuple[str, int]]:
        """Find names used but not defined."""
        defined = set()
        used = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used.append((node.id, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)

        return [(name, line) for name, line in used if name not in defined]

    def _check_td_patterns(self, tree: ast.AST, text_file: Path) -> list[Violation]:
        """Check for TouchDesigner-specific patterns and anti-patterns."""
        violations = []

        # Look for Execute DAT callback signatures
        expected_callbacks = {
            'onStart', 'onCreate', 'onExit', 'onFrameStart',
            'onFrameEnd', 'onPlayStateChange', 'onDeviceChange',
            'onProjectPreSave', 'onProjectPostSave'
        }

        defined_functions = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        # If this looks like an Execute DAT (has some callbacks), check completeness
        if defined_functions & expected_callbacks:
            missing = expected_callbacks - defined_functions
            if missing:
                violations.append(Violation(
                    rule='td-execute-dat-callbacks',
                    severity=Severity.INFO,
                    message=f"Execute DAT may be missing callbacks: {', '.join(sorted(missing))}",
                    path=str(text_file),
                ))

        return violations
```

### 6.5 Expression Validation (AG-003)

Parameter expressions in .parm files (mode 49 and 17) contain Python expressions that
should be validated separately from full Python scripts.

```python
import ast
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ExtractedExpression:
    """An expression extracted from a .parm file."""
    param_name: str
    mode: int
    expression: str
    file_path: Path
    line_number: int

# AG-003: Expression mode flags that contain Python expressions
EXPRESSION_MODES = {
    49,  # Full expression mode
    17,  # String with expression
}

class ExpressionValidator:
    """
    AG-003: Extract and validate parameter expressions.

    TouchDesigner .parm files can contain Python expressions in parameters
    with mode 49 (expression) or 17 (string with expression). These need
    special validation since they're evaluated at runtime.
    """

    def __init__(self):
        # TD expression-specific globals
        self.td_expression_globals = {
            'me', 'op', 'parent', 'absTime', 'project',
            'tdu', 'math', 'random',
            # Common expression properties
            'frame', 'seconds', 'playing',
        }

    def extract_expressions(self, parm_file: Path) -> list[ExtractedExpression]:
        """
        AG-003: Extract all expressions from a .parm file.

        Looks for parameters with mode 49 or 17 and extracts their expressions.
        """
        expressions = []
        content = parm_file.read_text()
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Skip delimiters
            if line.strip() == '?':
                continue

            # Parse parameter line: param_name mode value [expression]
            parts = line.split(None, 3)  # Split on whitespace, max 4 parts
            if len(parts) < 3:
                continue

            param_name = parts[0]
            try:
                mode = int(parts[1])
            except ValueError:
                continue

            # Check if this is an expression mode
            if mode in EXPRESSION_MODES:
                # Expression is in the 4th part (or value contains expression for mode 17)
                if len(parts) >= 4:
                    expression = parts[3]
                elif mode == 49 and len(parts) >= 3:
                    # Mode 49: the value itself might be the expression
                    expression = parts[2]
                else:
                    continue

                expressions.append(ExtractedExpression(
                    param_name=param_name,
                    mode=mode,
                    expression=expression,
                    file_path=parm_file,
                    line_number=line_num,
                ))

        return expressions

    def validate_expression(self, expr: ExtractedExpression) -> list[Violation]:
        """
        AG-003: Validate a single expression.

        Performs:
        1. Syntax validation (ast.parse in 'eval' mode)
        2. Basic name resolution against TD globals
        """
        violations = []

        # Clean up expression (remove quotes if string mode)
        expression = expr.expression.strip()
        if expression.startswith('"') and expression.endswith('"'):
            expression = expression[1:-1]

        # Phase 1: Syntax validation
        try:
            ast.parse(expression, mode='eval')
        except SyntaxError as e:
            violations.append(Violation(
                rule='expression-syntax',
                severity=Severity.ERROR,
                message=f"Expression syntax error in '{expr.param_name}': {e.msg}",
                path=str(expr.file_path),
                line=expr.line_number,
                context={'expression': expression}
            ))
            return violations  # Can't continue with invalid syntax

        # Phase 2: Check for obviously undefined names
        try:
            tree = ast.parse(expression, mode='eval')
            names_used = {
                node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }

            for name in names_used:
                if name not in self.td_expression_globals and name not in dir(__builtins__):
                    violations.append(Violation(
                        rule='expression-undefined-name',
                        severity=Severity.WARNING,
                        message=f"Expression '{expr.param_name}' uses potentially undefined: '{name}'",
                        path=str(expr.file_path),
                        line=expr.line_number,
                    ))
        except Exception:
            pass  # AST analysis failed, skip

        return violations

    def validate_all(self, toe_dir: Path) -> list[Violation]:
        """
        AG-003: Batch validate all expressions in a .toe.dir.

        Use with --validate-expressions CLI flag.
        """
        violations = []

        for parm_file in toe_dir.rglob("*.parm"):
            expressions = self.extract_expressions(parm_file)
            for expr in expressions:
                violations.extend(self.validate_expression(expr))

        return violations
```

**AG-003 CLI Usage**:
```bash
# Validate all parameter expressions
td-linter lint myproject.toe.dir --validate-expressions

# This extracts expressions from .parm files and validates them:
# - Syntax check using ast.parse(mode='eval')
# - Name resolution against TD expression globals
# - Reports warnings for potentially undefined names
```

### 6.6 LSP Server Integration (Future)

For real-time editor integration, we can spawn LSP servers for embedded languages:

```python
class EmbeddedLSPManager:
    """
    Manage LSP servers for embedded languages in .text files.

    This enables real-time validation and code intelligence in editors
    that support the Language Server Protocol.
    """

    def __init__(self):
        self.glsl_server = None
        self.python_server = None

    def start_glsl_server(self):
        """
        Start shader-language-server for GLSL validation.

        Reference: https://github.com/antaalt/shader-language-server
        """
        # The shader-language-server provides LSP for GLSL
        # Can be configured to understand TouchDesigner's GLSL dialect
        pass

    def start_python_server(self):
        """
        Start pylsp with TouchDesigner stubs for Python validation.

        Configuration includes:
        - TD API stub files for type checking
        - Custom plugin for TD-specific checks
        """
        pass
```

---

## 7. Rule Schema Definition

### 7.1 YAML Schema

**File**: `td-linter-rules.schema.yaml`

```yaml
# JSON Schema for td-linter rule definitions
$schema: "http://json-schema.org/draft-07/schema#"
$id: "https://sevenrad.com/td-linter/rules.schema.json"
title: "TD Linter Rules"
description: "Schema for TouchDesigner .toe.dir linter rule definitions"

type: object
required:
  - version
  - rules

properties:
  version:
    type: string
    description: "Schema version"
    pattern: "^\\d+\\.\\d+\\.\\d+$"

  extends:
    type: array
    description: "Base rule sets to extend"
    items:
      type: string
      enum:
        - "recommended"
        - "strict"
        - "minimal"

  rules:
    type: array
    items:
      $ref: "#/definitions/rule"

definitions:
  rule:
    type: object
    required:
      - id
      - name
      - severity
      - category
      - description
    properties:
      id:
        type: string
        description: "Unique rule identifier"
        pattern: "^[a-z][a-z0-9-]+$"

      name:
        type: string
        description: "Human-readable rule name"

      severity:
        type: string
        enum:
          - error    # Must fix - will cause TD failures
          - warning  # Should fix - may cause issues
          - info     # Consider - best practice suggestion
        description: "Default severity level"

      category:
        type: string
        enum:
          - syntax      # File format/syntax issues
          - connection  # Graph/connection issues
          - type        # Type compatibility issues
          - reference   # Operator reference issues
          - glsl        # GLSL shader issues
          - python      # Python script issues
          - performance # Performance concerns
          - style       # Style/naming conventions
        description: "Rule category"

      description:
        type: string
        description: "Detailed explanation of the rule"

      rationale:
        type: string
        description: "Why this rule exists"

      examples:
        type: object
        properties:
          bad:
            type: array
            items:
              type: string
          good:
            type: array
            items:
              type: string

      fix:
        type: string
        description: "How to fix violations"

      enabled:
        type: boolean
        default: true
        description: "Whether rule is enabled by default"

      options:
        type: object
        description: "Rule-specific configuration options"
        additionalProperties: true
```

### 7.2 Example Rules Configuration

**File**: `td-linter-rules.yaml`

```yaml
version: "1.0.0"
extends:
  - recommended

rules:
  # === Syntax Rules ===

  - id: valid-n-file-syntax
    name: "Valid .n file syntax"
    severity: error
    category: syntax
    description: |
      Ensures .n files conform to the expected TouchDesigner node
      definition format with valid TYPE:subtype, tile, flags, etc.
    rationale: |
      Malformed .n files cause toecollapse to fail or produce
      corrupt .toe files that crash TouchDesigner.
    enabled: true

  - id: valid-parm-file-syntax
    name: "Valid .parm file syntax"
    severity: error
    category: syntax
    description: |
      Ensures .parm files have proper structure with ? delimiters
      and valid parameter definitions.
    enabled: true

  - id: toc-completeness
    name: "TOC manifest completeness"
    severity: error
    category: syntax
    description: |
      Verifies that all files in .toe.dir are listed in .toc manifest
      and all .toc entries exist as files.
    enabled: true

  # === Connection Rules ===

  - id: no-invalid-cycles
    name: "No invalid cycles"
    severity: error
    category: connection
    description: |
      Detects cycles in the operator graph that would cause
      infinite loops. Feedback CHOPs are explicitly allowed.
    rationale: |
      Cycles in TOP or SOP chains cause TouchDesigner to hang.
    examples:
      bad:
        - "displace1 -> blur1 -> displace1 (cycle in TOPs)"
      good:
        - "feedback1 -> delay1 -> feedback1 (explicit CHOP feedback)"
    enabled: true

  - id: no-dangling-inputs
    name: "No dangling inputs"
    severity: error
    category: connection
    description: |
      All input references in .n files must point to existing operators.
    fix: "Create the missing operator or remove the input reference."
    enabled: true

  - id: type-compatibility
    name: "Type compatible connections"
    severity: error
    category: type
    description: |
      Operators can only connect to operators of compatible families.
      TOPs connect to TOPs, CHOPs to CHOPs, etc.
    rationale: |
      Cross-family connections require explicit conversion operators.
    options:
      allow_conversion_operators: true
    enabled: true

  # === Reference Rules ===

  - id: valid-operator-reference
    name: "Valid operator references"
    severity: error
    category: reference
    description: |
      All operator references in .parm files (pixeldat, computedat,
      chop, dat, top, sop) must resolve to existing operators.
    enabled: true

  - id: valid-path-references
    name: "Valid path references"
    severity: warning
    category: reference
    description: |
      Relative (./) and absolute (/) path references in parameters
      must resolve to existing operators.
    enabled: true

  # === GLSL Rules ===

  - id: glsl-no-version
    name: "No #version directive"
    severity: error
    category: glsl
    description: |
      GLSL shaders must not include #version directives as
      TouchDesigner injects the appropriate version.
    fix: "Remove the #version line from your shader."
    enabled: true

  - id: glsl-syntax
    name: "Valid GLSL syntax"
    severity: error
    category: glsl
    description: |
      GLSL code must pass glslangValidator with TouchDesigner preamble.
    enabled: true

  - id: glsl-td-output
    name: "Proper TD output"
    severity: warning
    category: glsl
    description: |
      Fragment shaders should use TDOutputSwizzle() for output.
    examples:
      bad:
        - "fragColor = color;"
      good:
        - "fragColor = TDOutputSwizzle(color);"
    enabled: true

  # === Python Rules ===

  - id: python-syntax
    name: "Valid Python syntax"
    severity: error
    category: python
    description: |
      Python code in Execute DATs and callbacks must be syntactically valid.
    enabled: true

  - id: python-undefined-name
    name: "No undefined names"
    severity: warning
    category: python
    description: |
      Variables and functions should be defined before use.
      TouchDesigner builtins (op, me, project, etc.) are allowed.
    enabled: true

  - id: td-execute-dat-callbacks
    name: "Execute DAT callbacks"
    severity: info
    category: python
    description: |
      Execute DATs should implement standard callback functions.
    enabled: true

  # === Performance Rules ===

  - id: deep-nesting
    name: "Avoid deep nesting"
    severity: warning
    category: performance
    description: |
      Deeply nested operator hierarchies (>10 levels) can impact performance.
    options:
      max_depth: 10
    enabled: true

  - id: excessive-inputs
    name: "Excessive inputs"
    severity: warning
    category: performance
    description: |
      Operators with many inputs (>8) may indicate over-complexity.
    options:
      max_inputs: 8
    enabled: true

  # === Style Rules ===

  - id: naming-convention
    name: "Operator naming convention"
    severity: info
    category: style
    description: |
      Operator names should follow consistent conventions.
    options:
      pattern: "^[a-z][a-z0-9_]*$"
      allow_digits_suffix: true
    enabled: false  # Optional

  - id: tile-overlap
    name: "No tile overlap"
    severity: info
    category: style
    description: |
      Operator tiles should not overlap in the network editor.
    enabled: false  # Optional
```

### 7.3 Rule Configuration API

```python
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class RuleConfig:
    """Configuration for a single rule."""
    id: str
    name: str
    severity: Severity
    category: str
    description: str
    enabled: bool = True
    options: dict = None

class RuleLoader:
    """Load and manage linter rules from YAML configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.rules: dict[str, RuleConfig] = {}
        self.load_builtin_rules()
        if config_path:
            self.load_custom_rules(config_path)

    def load_builtin_rules(self):
        """Load the default built-in rules."""
        builtin_path = Path(__file__).parent / "rules" / "builtin.yaml"
        self._load_rules_file(builtin_path)

    def load_custom_rules(self, config_path: Path):
        """Load custom rules from a YAML file."""
        self._load_rules_file(config_path)

    def _load_rules_file(self, path: Path):
        """Parse and load rules from a YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        # Handle extends
        for base in config.get('extends', []):
            self._load_rule_preset(base)

        # Load rules
        for rule_data in config.get('rules', []):
            rule = RuleConfig(
                id=rule_data['id'],
                name=rule_data['name'],
                severity=Severity[rule_data['severity'].upper()],
                category=rule_data['category'],
                description=rule_data['description'],
                enabled=rule_data.get('enabled', True),
                options=rule_data.get('options', {}),
            )
            self.rules[rule.id] = rule

    def get_enabled_rules(self) -> list[RuleConfig]:
        """Return all enabled rules."""
        return [r for r in self.rules.values() if r.enabled]

    def get_rules_by_category(self, category: str) -> list[RuleConfig]:
        """Return rules in a specific category."""
        return [r for r in self.rules.values() if r.category == category]
```

---

## 8. Example Rules

### 8.1 Critical Rules (Must Fix)

#### Rule: `no-invalid-cycles`

```python
class NoInvalidCyclesRule:
    """Detect cycles that would cause TouchDesigner to hang."""

    id = "no-invalid-cycles"
    severity = Severity.ERROR
    category = "connection"

    def check(self, graph: nx.DiGraph, context: LintContext) -> list[Violation]:
        violations = []

        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXNoCycle:
            return []

        for cycle in cycles:
            # Check if all nodes in cycle are CHOPs (feedback allowed)
            families = []
            for node in cycle:
                node_data = graph.nodes.get(node, {}).get('data')
                if node_data:
                    families.append(node_data.family)

            # Allow CHOP feedback loops
            if all(f == OperatorFamily.CHOP for f in families):
                continue

            # Check for explicit feedback operators
            has_feedback = any(
                graph.nodes.get(n, {}).get('data', {}).get('op_type') == 'feedback'
                for n in cycle
            )

            if not has_feedback:
                violations.append(Violation(
                    rule=self.id,
                    severity=self.severity,
                    message=f"Invalid cycle: {' -> '.join(cycle)} -> {cycle[0]}",
                    path=cycle[0],
                    context={'cycle': cycle}
                ))

        return violations
```

#### Rule: `type-compatibility`

```python
class TypeCompatibilityRule:
    """Ensure connections are between compatible operator types."""

    id = "type-compatibility"
    severity = Severity.ERROR
    category = "type"

    COMPATIBLE = {
        'TOP': {'TOP'},
        'CHOP': {'CHOP'},
        'SOP': {'SOP'},
        'DAT': {'DAT'},
        'MAT': {'MAT'},
        'POP': {'POP'},
        'COMP': {'COMP', 'TOP'},  # COMPs have special output connectors
    }

    CONVERTERS = {
        'chopto', 'toptochop', 'soptochop', 'choptosop',
        'dattochop', 'choptodat', 'soptodat',
    }

    def check(self, graph: nx.DiGraph, context: LintContext) -> list[Violation]:
        violations = []

        for source, target in graph.edges():
            if source.startswith("MISSING:"):
                continue

            source_data = graph.nodes.get(source, {}).get('data')
            target_data = graph.nodes.get(target, {}).get('data')

            if not source_data or not target_data:
                continue

            # Skip converter operators
            if target_data.op_type in self.CONVERTERS:
                continue

            source_family = source_data.family.value
            target_family = target_data.family.value

            compatible = self.COMPATIBLE.get(source_family, set())
            if target_family not in compatible:
                violations.append(Violation(
                    rule=self.id,
                    severity=self.severity,
                    message=f"Incompatible: {source_family} cannot connect to {target_family}",
                    path=target,
                    context={
                        'source': source,
                        'source_family': source_family,
                        'target_family': target_family,
                    }
                ))

        return violations
```

### 8.2 Warning Rules (Should Fix)

#### Rule: `glsl-td-output`

```python
class GLSLTDOutputRule:
    """Check that GLSL shaders use TDOutputSwizzle for proper output."""

    id = "glsl-td-output"
    severity = Severity.WARNING
    category = "glsl"

    def check(self, text_file: Path, context: LintContext) -> list[Violation]:
        violations = []
        content = text_file.read_text()

        # Check if this is a fragment shader
        if 'fragColor' not in content and 'gl_FragColor' not in content:
            return []

        # Look for direct assignment without TDOutputSwizzle
        import re

        # Pattern: fragColor = something; (not TDOutputSwizzle)
        direct_assigns = re.findall(
            r'fragColor\s*=\s*(?!TDOutputSwizzle)(\w+)',
            content
        )

        if direct_assigns:
            violations.append(Violation(
                rule=self.id,
                severity=self.severity,
                message="Fragment output should use TDOutputSwizzle() for proper color handling",
                path=str(text_file),
                fix="Change 'fragColor = color;' to 'fragColor = TDOutputSwizzle(color);'"
            ))

        return violations
```

### 8.3 Info Rules (Best Practices)

#### Rule: `td-execute-dat-callbacks`

```python
class TDExecuteDATCallbacksRule:
    """Check that Execute DATs have standard callback functions."""

    id = "td-execute-dat-callbacks"
    severity = Severity.INFO
    category = "python"

    STANDARD_CALLBACKS = {
        'onStart', 'onCreate', 'onExit',
        'onFrameStart', 'onFrameEnd',
        'onPlayStateChange', 'onDeviceChange',
        'onProjectPreSave', 'onProjectPostSave',
    }

    def check(self, text_file: Path, context: LintContext) -> list[Violation]:
        violations = []

        # Only check Execute DATs
        if not context.get('is_execute_dat', False):
            return []

        content = text_file.read_text()

        import ast
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []  # Syntax errors caught by another rule

        defined = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        # If at least one callback is defined, check for completeness
        if defined & self.STANDARD_CALLBACKS:
            missing = self.STANDARD_CALLBACKS - defined
            if missing:
                violations.append(Violation(
                    rule=self.id,
                    severity=self.severity,
                    message=f"Execute DAT missing standard callbacks: {', '.join(sorted(missing))}",
                    path=str(text_file),
                ))

        return violations
```

---

## 9. CLI Interface Design

### 9.1 Command Structure

```
td-linter
    lint <path>           # Lint a .toe.dir directory
    check <path>          # Alias for lint
    fix <path>            # Auto-fix issues where possible
    init                  # Create default config file
    rules                 # List available rules
    version               # Show version
```

### 9.2 CLI Implementation

```python
#!/usr/bin/env python3
"""
TouchDesigner .toe.dir Linter CLI

Usage:
    td-linter lint <path> [--config FILE] [--format FORMAT] [--fix] [--pre-collapse]
    td-linter rules [--category CAT]
    td-linter init [--force]
    td-linter version

Options:
    -c, --config FILE     Configuration file [default: td-linter.yaml]
    -f, --format FORMAT   Output format: text, json, sarif [default: text]
    --fix                 Attempt to auto-fix issues
    --pre-collapse        CM-001: Relaxed validation for pre-collapse mode
    --validate-expressions AG-003: Extract and validate expressions
    --category CAT        Filter rules by category
    --force               Overwrite existing config
    -v, --verbose         Verbose output
    -q, --quiet           Only show errors
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="td-linter",
    help="Validate TouchDesigner .toe.dir directories",
    add_completion=False,
)
console = Console()

@app.command()
def lint(
    path: Path = typer.Argument(..., help="Path to .toe.dir directory"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration file"
    ),
    format: str = typer.Option(
        "text", "--format", "-f",
        help="Output format: text, json, sarif"
    ),
    fix: bool = typer.Option(
        False, "--fix",
        help="Attempt to auto-fix issues"
    ),
    pre_collapse: bool = typer.Option(
        False, "--pre-collapse",
        help="CM-001: Relaxed validation for pre-collapse mode (skips .application, .panel checks)"
    ),
    validate_expressions: bool = typer.Option(
        False, "--validate-expressions",
        help="AG-003: Extract and validate parameter expressions"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Verbose output"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Only show errors"
    ),
):
    """
    Lint a .toe.dir directory for issues.

    Validates syntax, connections, embedded code, and best practices.

    CM-001: Pre-collapse mode (--pre-collapse):
        When working with manually created .toe.dir that hasn't been collapsed yet,
        use --pre-collapse to skip validation of files that are auto-generated
        by toecollapse (.application, .panel files, etc.).

    AG-003: Expression validation (--validate-expressions):
        Extract parameter expressions (mode 49/17) and validate them separately.
    """
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    if not path.is_dir() or not path.suffix == '.dir':
        console.print(f"[red]Error:[/red] Expected .toe.dir directory: {path}")
        raise typer.Exit(1)

    # Load configuration
    rule_loader = RuleLoader(config)

    # CM-001: Configure relaxed mode for pre-collapse validation
    lint_options = {
        'pre_collapse': pre_collapse,
        'validate_expressions': validate_expressions,
    }

    # Run linting with options
    linter = ToeDirLinter(rule_loader, options=lint_options)
    violations = linter.lint(path)

    # CM-001: In pre-collapse mode, filter out certain rule violations
    if pre_collapse:
        pre_collapse_skip_rules = {
            'toc-completeness',  # .toc may be incomplete before collapse
            'valid-application-file',  # .application generated by collapse
            'valid-panel-file',  # .panel generated by collapse
        }
        violations = [v for v in violations if v.rule not in pre_collapse_skip_rules]

    # Filter by severity if quiet
    if quiet:
        violations = [v for v in violations if v.severity == Severity.ERROR]

    # Output results
    if format == "json":
        output_json(violations)
    elif format == "sarif":
        output_sarif(violations, path)
    else:
        output_text(violations, verbose)

    # Exit code
    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        raise typer.Exit(1)


@app.command()
def rules(
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Filter by category"
    ),
):
    """List all available linting rules."""
    loader = RuleLoader()

    table = Table(title="TD Linter Rules")
    table.add_column("ID", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Category")
    table.add_column("Description")
    table.add_column("Enabled", justify="center")

    rules_list = loader.rules.values()
    if category:
        rules_list = [r for r in rules_list if r.category == category]

    for rule in sorted(rules_list, key=lambda r: (r.category, r.id)):
        enabled = "[green]Yes[/green]" if rule.enabled else "[dim]No[/dim]"
        table.add_row(
            rule.id,
            rule.severity.name.lower(),
            rule.category,
            rule.description[:50] + "..." if len(rule.description) > 50 else rule.description,
            enabled
        )

    console.print(table)


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing config"
    ),
):
    """Create a default td-linter.yaml configuration file."""
    config_path = Path("td-linter.yaml")

    if config_path.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {config_path}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    default_config = """\
# TD Linter Configuration
# See https://docs.sevenrad.com/td-linter for full documentation

version: "1.0.0"

extends:
  - recommended

# Rule overrides
rules: []

# Paths to ignore
ignore:
  - "**/backup/**"
  - "**/*.backup.toe.dir"
"""

    config_path.write_text(default_config)
    console.print(f"[green]Created:[/green] {config_path}")


@app.command()
def version():
    """Show version information."""
    console.print("td-linter version 1.0.0")
    console.print("Python:", sys.version.split()[0])


def output_text(violations: list[Violation], verbose: bool):
    """Output violations in human-readable text format."""
    if not violations:
        console.print("[green]No issues found![/green]")
        return

    # Group by file
    by_file = {}
    for v in violations:
        by_file.setdefault(v.path, []).append(v)

    for path, file_violations in sorted(by_file.items()):
        console.print(f"\n[bold]{path}[/bold]")

        for v in sorted(file_violations, key=lambda x: x.line or 0):
            severity_color = {
                Severity.ERROR: "red",
                Severity.WARNING: "yellow",
                Severity.INFO: "blue",
            }.get(v.severity, "white")

            line_info = f":{v.line}" if v.line else ""
            console.print(
                f"  [{severity_color}]{v.severity.name}[/{severity_color}] "
                f"[dim]{v.rule}[/dim]{line_info}: {v.message}"
            )

            if verbose and v.fix:
                console.print(f"    [dim]Fix: {v.fix}[/dim]")

    # Summary
    errors = sum(1 for v in violations if v.severity == Severity.ERROR)
    warnings = sum(1 for v in violations if v.severity == Severity.WARNING)
    infos = sum(1 for v in violations if v.severity == Severity.INFO)

    console.print(f"\n[bold]Summary:[/bold] {errors} errors, {warnings} warnings, {infos} info")


def output_json(violations: list[Violation]):
    """Output violations in JSON format."""
    import json

    output = {
        "violations": [
            {
                "rule": v.rule,
                "severity": v.severity.name.lower(),
                "message": v.message,
                "path": v.path,
                "line": v.line,
                "context": v.context,
            }
            for v in violations
        ],
        "summary": {
            "total": len(violations),
            "errors": sum(1 for v in violations if v.severity == Severity.ERROR),
            "warnings": sum(1 for v in violations if v.severity == Severity.WARNING),
            "info": sum(1 for v in violations if v.severity == Severity.INFO),
        }
    }

    print(json.dumps(output, indent=2))


def output_sarif(violations: list[Violation], path: Path):
    """Output violations in SARIF format for GitHub integration."""
    import json

    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "td-linter",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/sevenrad/td-linter",
                }
            },
            "results": [
                {
                    "ruleId": v.rule,
                    "level": {
                        Severity.ERROR: "error",
                        Severity.WARNING: "warning",
                        Severity.INFO: "note",
                    }.get(v.severity, "note"),
                    "message": {"text": v.message},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": v.path},
                            "region": {"startLine": v.line or 1}
                        }
                    }]
                }
                for v in violations
            ]
        }]
    }

    print(json.dumps(sarif, indent=2))


if __name__ == "__main__":
    app()
```

### 9.3 Example Usage

```bash
# Lint a project
td-linter lint myproject.toe.dir

# Lint with custom config
td-linter lint myproject.toe.dir --config .td-linter.yaml

# Output JSON for CI
td-linter lint myproject.toe.dir --format json > lint-results.json

# Output SARIF for GitHub Code Scanning
td-linter lint myproject.toe.dir --format sarif > lint.sarif

# List all rules
td-linter rules

# List rules by category
td-linter rules --category glsl

# Create default config
td-linter init
```

---

## 10. Integration with Existing Tools

### 10.1 Integration with validate_glsl.py

The existing `validate_glsl.py` script provides GLSL validation. The linter integrates with it:

```python
class ExistingGLSLValidatorAdapter:
    """
    Adapter for the existing validate_glsl.py script.

    Location: docs/touchdesigner/scripts/validate_glsl.py
    """

    def __init__(self):
        self.script_path = Path(__file__).parent.parent / "scripts" / "validate_glsl.py"

    def validate(self, shader_path: Path) -> list[Violation]:
        """Run validation using existing script."""
        import subprocess

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(shader_path)],
            capture_output=True,
            text=True
        )

        violations = []
        if result.returncode != 0:
            # Parse output for errors
            for line in result.stdout.split('\n'):
                if line.startswith('ERROR:') or line.startswith('ERRORS'):
                    violations.append(Violation(
                        rule='glsl-validation',
                        severity=Severity.ERROR,
                        message=line,
                        path=str(shader_path),
                    ))

        return violations
```

### 10.2 Integration with build_test_harness.py

The linter can be run as part of the build process:

```python
# In build_test_harness.py, add pre-collapse validation

def lint_before_collapse(harness_dir: Path) -> bool:
    """Run linter before collapsing to .toe."""
    from td_linter import ToeDirLinter, RuleLoader

    loader = RuleLoader()
    linter = ToeDirLinter(loader)
    violations = linter.lint(harness_dir)

    errors = [v for v in violations if v.severity == Severity.ERROR]

    if errors:
        print(f"ERROR: {len(errors)} linting errors found. Fix before collapse.")
        for v in errors:
            print(f"  {v.path}: {v.message}")
        return False

    warnings = [v for v in violations if v.severity == Severity.WARNING]
    if warnings:
        print(f"WARNING: {len(warnings)} warnings found (continuing anyway)")

    return True


def collapse(verbose: bool = False) -> int:
    """Collapse .toe.dir to .toe binary format."""
    # Add linting step
    if not lint_before_collapse(HARNESS_DIR):
        return 1

    # ... rest of existing collapse logic
```

### 10.3 Pre-commit Hook Integration

```yaml
# .pre-commit-config.yaml

repos:
  - repo: local
    hooks:
      - id: td-linter
        name: Lint TouchDesigner .toe.dir
        entry: td-linter lint
        language: python
        files: \.toe\.dir/
        pass_filenames: false
        additional_dependencies:
          - td-linter
```

### 10.4 GitHub Actions Workflow

```yaml
# .github/workflows/td-lint.yml

name: TouchDesigner Lint

on:
  push:
    paths:
      - '**/*.toe.dir/**'
  pull_request:
    paths:
      - '**/*.toe.dir/**'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install td-linter
        run: pip install td-linter

      - name: Install glslang
        run: sudo apt-get install -y glslang-tools

      - name: Lint .toe.dir directories
        run: |
          for dir in $(find . -name '*.toe.dir' -type d); do
            echo "Linting: $dir"
            td-linter lint "$dir" --format sarif >> lint-results.sarif
          done

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: lint-results.sarif
```

---

## 11. Implementation Phases

### Phase 0: Format Discovery (CM-003)

**Goal**: Collect comprehensive .toe.dir samples and document format variations

> **CM-003 Note**: Before implementing the linter, we need to understand the full range
> of .toe.dir format variations across TD versions and project types. This discovery
> sprint will build the foundation for robust grammar and validation rules.

**Duration**: 1 week (before Phase 1)

**Tasks**:
- [ ] **Sample Collection** (Target: 50+ samples)
  - [ ] Search GitHub for public TouchDesigner projects with expanded .toe.dir
  - [ ] Collect samples from TD forums and community resources
  - [ ] Extract .toe.dir from various TD demo projects
  - [ ] Document source and TD version for each sample

- [ ] **Version Coverage**
  - [ ] Collect samples from TouchDesigner 2022.x
  - [ ] Collect samples from TouchDesigner 2023.x
  - [ ] Collect samples from TouchDesigner 2024.x
  - [ ] Document any format differences between versions

- [ ] **Mode Flag Discovery (RF-002)**
  - [ ] Create `mode_flag_discovery.toe` test project
  - [ ] Enumerate all parameter modes via TD Python API
  - [ ] Document mode meanings and behavior
  - [ ] Create test cases for each mode type

- [ ] **Operator Type Catalog (CM-002)**
  - [ ] Generate complete operator list via TD Python API
  - [ ] Document input/output compatibility for each operator
  - [ ] Create initial `td_operators.yaml` database
  - [ ] Note version-specific operators

- [ ] **Format Documentation**
  - [ ] Document any undocumented file types found in samples
  - [ ] Catalog all .n file variations (flags, view formats)
  - [ ] Catalog all .parm file variations (modes, expressions)
  - [ ] Document special files (.build, .start, .grps, etc.)

**Deliverables**:
- `samples/` directory with 50+ .toe.dir samples
- `samples/catalog.yaml` documenting each sample
- `mode_flag_discovery.toe` test project
- Initial `td_operators.yaml` database
- Format variations document

**Sample Collection Sources**:
- GitHub: `"toe.dir" OR "expanded toe" touchdesigner`
- TD Forum: Projects section
- Derivative.ca: Example projects
- Community tutorials with downloadable projects

### Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Basic syntax validation and CLI

- [ ] Project structure setup
- [ ] Lark grammar for `.n` files
- [ ] Lark grammar for `.parm` files
- [ ] Basic CLI with `lint` command
- [ ] Text output format
- [ ] Unit tests for parsers

**Deliverables**:
- `td-linter lint <path>` validates syntax
- Test suite for parsers

### Phase 2: Graph Validation (Week 3-4)

**Goal**: Connection and type validation

- [ ] NetworkX graph construction from `.n` files
- [ ] Cycle detection algorithm
- [ ] Type compatibility checking
- [ ] Dangling input detection
- [ ] Operator reference validation

**Deliverables**:
- Connection validation rules working
- Graph visualization (debug mode)

### Phase 3: Embedded Code Validation (Week 5-6)

**Goal**: GLSL and Python validation

- [ ] Language detection for `.text` files
- [ ] GLSL validation (integrate existing `validate_glsl.py`)
- [ ] Python AST validation
- [ ] TouchDesigner API stub generation
- [ ] Warning for undefined names

**Deliverables**:
- GLSL shaders validated before toecollapse
- Python scripts checked for basic issues

### Phase 4: Rule System (Week 7-8)

**Goal**: Extensible YAML-based rules

- [ ] YAML schema for rules
- [ ] Rule loading and configuration
- [ ] Built-in rule set (recommended)
- [ ] Rule enable/disable per-project
- [ ] Custom rule options

**Deliverables**:
- `td-linter.yaml` configuration working
- `td-linter rules` command

### Phase 5: Integration & Polish (Week 9-10)

**Goal**: Production-ready tool

- [ ] JSON output format
- [ ] SARIF output for GitHub
- [ ] Pre-commit hook
- [ ] GitHub Actions workflow
- [ ] Integration with `build_test_harness.py`
- [ ] Documentation
- [ ] Package for pip

**Deliverables**:
- Full CI/CD integration
- Published package

### Phase 6: Advanced Features (Future)

**Goal**: Enhanced capabilities

- [ ] Auto-fix for common issues
- [ ] LSP server for editor integration
- [ ] Watch mode for continuous linting
- [ ] Performance profiling rules
- [ ] Custom rule API (Python plugins)

---

## 12. Research Sources

### Lark Parser
- [Official Documentation](https://lark-parser.readthedocs.io/en/latest/)
- [Grammar Reference](https://lark-parser.readthedocs.io/en/latest/grammar.html)
- [Lark Examples](https://github.com/lark-parser/lark/tree/master/examples)
- [Lark Cheat Sheet](https://lark-parser.readthedocs.io/en/latest/grammar.html#cheatsheet)

### Visual Programming Linters
- [Visual Programming: From Unreal Engine Blueprints to Node-RED](https://domaindrivendesign.org/visual-programming-from-unreal-engine-blueprints-to-node-red/)
- [Unreal Engine Blueprints Documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/blueprints-visual-scripting-in-unreal-engine)
- [Fundamental Blueprint Practices](https://unrealcommunity.wiki/6100e8119c9d1a89e0c31a3d)

### NetworkX Graph Validation
- [NetworkX Algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html)
- [NetworkX Cycles](https://networkx.org/documentation/stable/reference/algorithms/cycles.html)
- [find_cycle Function](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.find_cycle.html)

### Shader Language Servers
- [How to write a language server (shader-language-server)](https://antaalt.github.io/2024/11/23/shader-language-server.html)
- [glsl-lsp on GitHub](https://github.com/KubaP/glsl-lsp)
- [shader-validator VS Code Extension](https://github.com/antaalt/shader-validator)

### TouchDesigner Documentation
- [Python in TouchDesigner](https://docs.derivative.ca/Python)
- [Working with OPs in Python](https://docs.derivative.ca/Working_with_OPs_in_Python)
- [Operator Families](https://docs.derivative.ca/Operator)
- [TouchDesigner MCP Server (API Reference)](https://github.com/bottobot/touchdesigner-mcp-server)

### Houdini HDA Validation
- [hou.HDAModule](https://www.sidefx.com/docs/houdini/hom/hou/HDAModule.html)
- [hou.HDADefinition](https://www.sidefx.com/docs/houdini/hom/hou/HDADefinition.html)
- [Python Script Locations](https://www.sidefx.com/docs/houdini/hom/locations.html)

### Python AST and Linters
- [Python AST Module](https://docs.python.org/3/library/ast.html)
- [Learn Python ASTs by building your own linter](https://deepsource.com/blog/python-asts-by-building-your-own-linter)
- [Pylint Documentation](https://python.org.il/en/presentations/pylint-python-static-code-analysis)
- [MegaLinter](https://github.com/oxsecurity/megalinter)

### YAML Configuration
- [yamllint Documentation](https://yamllint.readthedocs.io/en/stable/configuration.html)
- [LinkML Schema Linter](https://linkml.io/linkml/schemas/linter.html)

---

## Appendix A: Complete File Format Reference

### A.1 Operator Type List

| Family | Types (observed) |
|--------|-----------------|
| TOP | displace, glsl, moviefilein, null, blur, composite, transform |
| CHOP | noise, math, select, feedback, constant, filter, speed |
| SOP | box, sphere, in, out, merge, transform, copy |
| DAT | text, table, execute, script, select |
| COMP | container, geo, base, window |
| MAT | phong, pbr, constant, wireframe |

### A.2 Flag Reference

| Flag | Meaning |
|------|---------|
| `picked` | Currently selected in UI |
| `current` | Currently active operator |
| `viewer` | Has viewer open |
| `parlanguage` | Parameter language mode |
| `activate` | Is activated |
| `render` | Is rendered |
| `display` | Is displayed |
| `pickable` | Can be picked/selected |

### A.3 Parameter Mode Values

| Mode | Meaning |
|------|---------|
| 0 | Constant value |
| 17 | String with expression |
| 32 | Default/unchanged |
| 49 | Expression mode |

---

## Appendix B: Sample Validation Output

```
$ td-linter lint shader_test_harness.toe.dir

shader_test_harness.toe.dir/project1/glsl_under_test.n
  ERROR no-dangling-inputs:4: Input references non-existent operator: 'movie_in'

shader_test_harness.toe.dir/project1/text_shader.text
  WARNING glsl-td-output:12: Fragment output should use TDOutputSwizzle() for proper color handling
    Fix: Change 'fragColor = color;' to 'fragColor = TDOutputSwizzle(color);'

shader_test_harness.toe.dir/project1/execute_autotest.text
  INFO td-execute-dat-callbacks: Execute DAT may be missing callbacks: onProjectPreSave, onProjectPostSave

Summary: 1 errors, 1 warnings, 1 info
```

---

## 13. Changelog

### Version 1.1.0-draft (2026-01-14)

**Red Team Improvements**

This version incorporates fixes from red team review to improve robustness and completeness.

| Fix ID | Category | Description |
|--------|----------|-------------|
| RF-001 | Grammar | Lark grammar uses permutation rules for flexible ordering of optional .n file elements |
| RF-002 | Discovery | Parameter mode flags marked as requiring systematic verification; added mode_flag_discovery.toe task |
| RF-003 | Validation | Comprehensive TD builtins set expanded to include all modules, callbacks, and special objects |
| RF-004 | Grammar | Semantic number types (tile_coord, input_index, mode_flag) replace generic integer for better validation |
| AG-001 | Validation | Cycle detection enhanced with FEEDBACK_OPERATOR_TYPES and .parm feedback property checking |
| AG-002 | Validation | GLSL validation uses preamble-agnostic approach; filters undefined uniform warnings |
| AG-003 | Validation | New expression validation system for mode 49/17 parameters with --validate-expressions flag |
| CM-001 | CLI | Added --pre-collapse flag for relaxed validation before toecollapse |
| CM-002 | Database | Operator compatibility lists expanded; plan for td_operators.yaml database documented |
| CM-003 | Process | Added Phase 0: Format Discovery sprint with 50+ sample collection target |

**New Sections**:
- Section 6.5: Expression Validation (AG-003)
- Section 11 Phase 0: Format Discovery (CM-003)
- Section 13: Changelog

**Enhanced Sections**:
- Section 3.2: Parameter mode flags with verification notes
- Section 4.1: Grammar with permutation rules and semantic types
- Section 4.2: Grammar with semantic mode_flag type
- Section 5.3.1: Cycle detection with feedback operator handling
- Section 5.3.2: Type compatibility with expanded operators
- Section 6.3: GLSL validation with preamble-agnostic approach
- Section 6.4: Python validation with comprehensive builtins
- Section 9.2: CLI with --pre-collapse and --validate-expressions flags

### Version 1.0.0-draft (2026-01-14)

- Initial specification document
- Lark grammar for .n and .parm files
- NetworkX graph validation
- GLSL and Python embedded code validation
- YAML rule schema
- CLI design
- Integration patterns

---

*Document generated: 2026-01-14*
*Author: SevenRad Development Team*
*Version: 1.1.0-draft*
