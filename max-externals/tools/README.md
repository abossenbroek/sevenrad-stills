# GenJit Linter

Red-team validation linter for .genjit files that ensures they conform to Max/MSP's JSON patcher format.

## Overview

The `lint_genjit.py` script is an adversarial linter designed to catch issues that would cause Max to fail loading .genjit shader files. It validates both **GenExpr** and **GLSL** shader formats.

## Usage

### Basic Usage

```bash
# Lint a single file
python tools/lint_genjit.py code/sr.saturation.genjit

# Lint multiple files
python tools/lint_genjit.py code/*.genjit

# Strict mode (warnings become errors)
python tools/lint_genjit.py --strict code/sr.saturation.genjit

# Verbose mode (show detailed info)
python tools/lint_genjit.py --verbose code/*.genjit
```

### Make Targets

```bash
# Lint all .genjit files
make test-lint

# Lint with strict mode
make test-lint-strict

# Run linter test suite
make test-lint-test
```

## Validations

The linter performs comprehensive validation across multiple categories:

### 1. JSON Structure
- Valid JSON syntax
- Top-level `patcher` key exists
- Required patcher fields: `fileversion`, `appversion`, `boxes`, `lines`
- Correct data types for all fields

### 2. Required Objects
- At least one `in 1` input object (maxclass: newobj, text: "in 1")
- At least one `out 1` output object (maxclass: newobj, text: "out 1")
- Exactly one codebox object (maxclass: codebox)
- Codebox has non-empty `code` field

### 3. Shader Code Validation

#### GenExpr Shaders
- Code references `in1` (input texture)
- Code assigns to `out1` or `out` (output)
- Uses `sample()` for texture sampling

#### GLSL Shaders
- Code assigns to `gl_FragColor` (output)
- Code samples textures using `texture2DRect()`, `texture2D()`, or `texture()`
- Has valid GLSL structure (`<jit.gl.pix>` wrapper or `#version` directive)

### 4. Param Validation
- Param objects have format: `param <name> <value>`
- Param objects have `numinlets=0`, `numoutlets=1`
- Param names match parameters used in shader code

### 5. Patchline Integrity
- All patchline sources reference existing box IDs
- All patchline destinations reference existing box IDs
- Patchlines use correct format: `[box_id, outlet_num]`

### 6. Connection Graph Validation
- Input (`in 1`) connects to codebox inlet 0
- Codebox connects to output (`out 1`)
- Each param connects to a unique codebox inlet (1, 2, 3, ...)
- Params do not connect to codebox inlet 0 (reserved for input texture)

### 7. Box ID Validation
- Box IDs have correct format: `obj-N` (e.g., `obj-1`, `obj-2`)
- All referenced box IDs exist

## Examples

### Valid GenExpr Shader

```json
{
  "patcher": {
    "fileversion": 1,
    "appversion": {"major": 8, "minor": 0, "revision": 0},
    "boxes": [
      {
        "box": {
          "id": "obj-1",
          "maxclass": "newobj",
          "numinlets": 0,
          "numoutlets": 1,
          "text": "in 1"
        }
      },
      {
        "box": {
          "id": "obj-2",
          "maxclass": "newobj",
          "numinlets": 0,
          "numoutlets": 1,
          "text": "param factor 1.0"
        }
      },
      {
        "box": {
          "id": "obj-3",
          "maxclass": "codebox",
          "numinlets": 2,
          "numoutlets": 1,
          "code": "color = sample(in1, norm);\nout1 = color * factor;"
        }
      },
      {
        "box": {
          "id": "obj-4",
          "maxclass": "newobj",
          "numinlets": 1,
          "numoutlets": 0,
          "text": "out 1"
        }
      }
    ],
    "lines": [
      {"patchline": {"source": ["obj-1", 0], "destination": ["obj-3", 0]}},
      {"patchline": {"source": ["obj-2", 0], "destination": ["obj-3", 1]}},
      {"patchline": {"source": ["obj-3", 0], "destination": ["obj-4", 0]}}
    ]
  }
}
```

### Common Errors Caught

**1. Not JSON format**
```
ERROR: Invalid JSON syntax
ERROR: File appears to be GenExpr code, not Max JSON patcher format
```

**2. Missing output assignment**
```
ERROR: GenExpr code does not assign to 'out1' or 'out'
```

**3. Invalid patchline reference**
```
ERROR: Patchline 0 'source' references non-existent box: 'obj-999'
```

**4. Missing connection**
```
ERROR: No connection from input ('obj-1') to codebox ('obj-3')
```

**5. GLSL without output**
```
ERROR: GLSL shader does not assign to 'gl_FragColor'
```

## Output Format

### PASS Example
```
PASS: code/sr.saturation.genjit
  All validations passed ✓
```

### FAIL Example
```
FAIL: code/sr.broken.genjit
  [ERROR] Missing required codebox object
  [ERROR] No connection from input to codebox
  [WARNING] Codebox has 3 outlets, typically should be 1
```

### Exit Codes
- `0` - All validations passed
- `1` - One or more validations failed

## Testing

The linter includes a comprehensive test suite with 20 test cases covering:
- Valid GenExpr and GLSL shaders
- Missing required objects
- Invalid JSON formats
- Broken patchline connections
- Empty or missing code
- Invalid box ID formats

Run the test suite:
```bash
python tools/test_lint_genjit.py
```

Expected output:
```
Running genjit linter test suite...

✓ Valid GenExpr shader
✓ Not JSON - plain text
✓ GenExpr code instead of JSON
...
✓ Patchline with invalid format

============================================================
Tests passed: 20/20
✓ All tests passed!
```

## Integration

The linter is integrated into the project's test suite:

```bash
# Run all tests (includes linting)
make test

# Run only linter
make test-lint

# Run linter in strict mode
make test-lint-strict
```

## Troubleshooting

### False Positives

If the linter incorrectly flags valid code:

1. Use `--verbose` to see detailed validation steps
2. Check if the shader format (GenExpr vs GLSL) is detected correctly
3. Verify the JSON structure matches the Max patcher format

### False Negatives

If the linter passes invalid code:

1. Test manually in Max to confirm it fails
2. Add a test case to `test_lint_genjit.py`
3. Update validation rules in `lint_genjit.py`

## Reference

The linter validates against the official Max/MSP GenJit format as seen in:
```
/Applications/Max.app/Contents/Resources/C74/examples/jitter-examples/gen/pinch.genjit
```

## Development

To add new validations:

1. Add validation logic to the appropriate `_validate_*` method in `lint_genjit.py`
2. Add test cases to `test_lint_genjit.py`
3. Run the test suite to verify
4. Update this README

## License

Part of the SevenRad Max Externals package.
