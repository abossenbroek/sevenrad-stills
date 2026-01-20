# TouchDesigner Migration: Shader Development and Automated Testing Guide

Complete guide for developing GLSL shaders in TouchDesigner and implementing automated testing infrastructure for rapid iteration.

## Related Documents

- [00-IMPLEMENTATION-OVERVIEW.md](00-IMPLEMENTATION-OVERVIEW.md) - High-level roadmap and decisions
- [01-LINTING-INFRASTRUCTURE.md](01-LINTING-INFRASTRUCTURE.md) - Linting tools, CI/CD, editor integration
- [02-EFFECTS-AND-DEMOS.md](02-EFFECTS-AND-DEMOS.md) - GLSL effects, .tox structure, demo system
- [03-REMEDIATION-PLAN.md](03-REMEDIATION-PLAN.md) - Pre-implementation validation and infrastructure

---

## Quick Start

```bash
# 1. Write shader externally
vim docs/touchdesigner/glsl/effects/my_effect.frag

# 2. Validate before loading into TD
python docs/touchdesigner/scripts/validate_glsl.py my_effect.frag

# 3. Run automated tests (requires TD installed)
python docs/touchdesigner/scripts/td_test_runner.py \
  --config fixtures/config/my_effect_test.json \
  --output /tmp/my_effect_output
```

---

## Section 1: Hello World Shader Examples

### 1.1 Minimal Passthrough Shader

The simplest shader that reads input and outputs unchanged:

```glsl
// minimal_passthrough.frag
// NOTE: No #version directive - TouchDesigner auto-injects

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    fragColor = TDOutputSwizzle(color);
}
```

### 1.2 Brightness Adjustment (Hello World)

A simple effect with one custom uniform parameter:

**File:** `glsl/effects/hello_brightness.frag`

```glsl
// hello_brightness.frag
// Adjustable brightness effect - Hello World for TD GLSL development
//
// Uniforms:
//   uBrightness - Brightness multiplier (range: 0.0-2.0, default: 1.0)

uniform float uBrightness;

void main() {
    // Read input texture at current UV coordinate
    vec4 color = texture(sTD2DInputs[0], vUV.st);

    // Apply brightness adjustment (preserving alpha)
    color.rgb *= uBrightness;

    // Output with proper color space handling
    fragColor = TDOutputSwizzle(color);
}
```

### 1.3 Invert Effect

Color inversion with blend amount:

```glsl
// hello_invert.frag

uniform float uAmount;  // 0.0 = original, 1.0 = fully inverted

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    vec3 inverted = 1.0 - color.rgb;
    color.rgb = mix(color.rgb, inverted, uAmount);
    fragColor = TDOutputSwizzle(color);
}
```

### 1.4 Uniform-to-Parameter Mapping

How GLSL uniforms connect to TouchDesigner Custom Parameters:

| GLSL Declaration | TD Parameter Type | Setup |
|------------------|-------------------|-------|
| `uniform float uValue;` | Float slider | Vectors page: vec0name="uValue" |
| `uniform int uMode;` | Menu (Int) | Vectors page: int0name="uMode" |
| `uniform vec2 uOffset;` | XY (Float pair) | Vectors page: vec0name="uOffset" |
| `uniform vec4 uColor;` | Color (RGBA) | Colors page: color0name="uColor" |
| `uniform int uSeed;` | Int | Vectors page: int0name="uSeed" |

**TouchDesigner GLSL TOP Setup:**

1. Create **Text DAT** → set File parameter to shader path
2. Create **GLSL TOP** → set GLSL DAT to the Text DAT
3. Add Custom Parameters to GLSL TOP or parent component
4. Wire uniforms on GLSL TOP's Vectors/Colors pages

---

## Section 2: Development Workflow

### 2.1 Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Write shader in external IDE (VSCode/Neovim with GLSL LSP) │
│                            │                                    │
│                            ▼                                    │
│  2. Validate: python validate_glsl.py myshader.frag            │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              │                           │                     │
│              ▼                           ▼                     │
│         PASS                         FAIL                      │
│              │                           │                     │
│              ▼                           ▼                     │
│  3. Load in TD via Text DAT     Fix errors, go to step 2       │
│     (File sync enabled)                                        │
│              │                                                 │
│              ▼                                                 │
│  4. Test interactively in TD                                   │
│              │                                                 │
│              ▼                                                 │
│  5. Modify shader externally → TD auto-reloads                 │
│              │                                                 │
│              ▼                                                 │
│  6. Run automated tests for regression                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 IDE Configuration

**VSCode** (recommended):

```json
{
  "files.associations": {
    "*.frag": "glsl",
    "*.comp": "glsl"
  },
  "[glsl]": {
    "editor.tabSize": 4,
    "editor.formatOnSave": false
  }
}
```

**Neovim with nvim-lspconfig:**

```lua
require('lspconfig').glsl_analyzer.setup{}
```

### 2.3 Validation Commands

```bash
# Validate single shader
python docs/touchdesigner/scripts/validate_glsl.py myshader.frag

# Validate all effects
find docs/touchdesigner/glsl/effects -name '*.frag' | \
  xargs -I {} python docs/touchdesigner/scripts/validate_glsl.py {}

# Validate with verbose output
python docs/touchdesigner/scripts/validate_glsl.py -v myshader.frag
```

### 2.4 Text DAT File Sync Setup

In TouchDesigner:

1. Create **Text DAT**
2. Parameters:
   - **File**: `/path/to/myshader.frag`
   - **Sync to File**: ON
   - **Load on Start**: ON

When you save the file externally, TouchDesigner automatically reloads it.

---

## Section 3: TouchDesigner Automation Fundamentals

TouchDesigner lacks true headless mode, but automated testing is achievable through **Perform Mode**, **Execute DAT**, and **environment variables**.

### 3.1 Perform Mode

Perform Mode hides the editor UI and runs only the output window:

```bash
# Launch in Perform Mode (macOS)
/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner \
  myproject.toe
```

**Limitations:**
- Output window still displays
- Requires GPU context (HDMI dummy plug on headless Macs)
- Non-Commercial license: 1280×1280 max resolution

### 3.2 Execute DAT onStart() Callback

The Execute DAT's `onStart()` callback runs automatically when a project opens:

```python
# execute_autotest (Execute DAT)

import os

def onStart():
    """Called automatically when project opens."""
    test_mode = os.environ.get('SHADER_TEST_MODE', '0')
    if test_mode == '1':
        run_automated_tests()

def run_automated_tests():
    """Execute the automated test suite."""
    import json

    config_path = os.environ.get('TEST_CONFIG', '')
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/td_output')

    # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Run tests...
```

### 3.3 Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `SHADER_TEST_MODE` | Enable automated testing | `1` |
| `TEST_CONFIG` | Path to test configuration JSON | `/path/to/config.json` |
| `OUTPUT_DIR` | Directory for output files | `/tmp/td_output` |

**Setting from external script:**

```python
import os
import subprocess

os.environ['SHADER_TEST_MODE'] = '1'
os.environ['TEST_CONFIG'] = '/path/to/config.json'
os.environ['OUTPUT_DIR'] = '/tmp/output'

subprocess.run([
    '/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner',
    '/path/to/test_project.toe'
])
```

### 3.4 Frame Capture with TOP.save()

```python
# Capture current TOP output to file
def capture_frame(top_name, output_path):
    """Save TOP output to PNG file."""
    top = op(top_name)
    top.cook(force=True)  # Ensure latest render
    top.save(output_path)
    return output_path

# Usage
capture_frame('render_out', '/tmp/output/test_001.png')
```

### 3.5 Clean Exit with project.quit()

**Critical:** Use `delayFrames` to ensure file writes complete before quitting.

```python
# WRONG - files may not finish writing
project.quit()

# CORRECT - wait 60 frames before quitting
run('project.quit()', delayFrames=60)
```

---

## Section 4: Automated Testing Architecture

### 4.1 Test Project Structure

```
fixtures/projects/
├── shader_test_harness.toe        # Main test runner project
│   ├── execute_autotest           # Execute DAT - onStart() entry point
│   ├── test_runner                # Text DAT - Python test logic
│   ├── text_shader                # Text DAT - loads shader file
│   ├── glsl_under_test            # GLSL TOP - shader being tested
│   ├── movie_in                   # Movie File In TOP - test input
│   ├── render_out                 # Render TOP - final output
│   ├── info_compile               # Info DAT - compile status
│   └── null_output                # Null TOP - cooking endpoint
```

### 4.2 Execute DAT Test Runner

Complete implementation for `execute_autotest`:

```python
# execute_autotest (Execute DAT)
# Enable "Start" callback in parameters

import os
import json

def onStart():
    """Entry point for automated testing."""
    if os.environ.get('SHADER_TEST_MODE') != '1':
        return

    # Disable real-time for deterministic rendering
    project.realTime = False

    # Load configuration
    config_path = os.environ.get('TEST_CONFIG', '')
    if not config_path:
        print("ERROR: TEST_CONFIG not set")
        project.quit()
        return

    with open(config_path) as f:
        config = json.load(f)

    # Run tests
    output_dir = os.environ.get('OUTPUT_DIR', '/tmp/td_output')
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for test in config.get('tests', []):
        result = run_single_test(test, output_dir)
        results.append(result)

    # Write results
    write_results(results, output_dir)

    # Clean exit
    run('project.quit()', delayFrames=60)


def run_single_test(test, output_dir):
    """Execute a single test case."""
    name = test.get('name', 'unnamed')
    shader_path = test.get('shader', '')
    params = test.get('params', {})

    # Load shader
    shader_dat = op('text_shader')
    shader_dat.par.file = shader_path
    shader_dat.par.loadonstartpulse.pulse()

    # Set parameters
    glsl = op('glsl_under_test')
    for param_name, value in params.items():
        if hasattr(glsl.par, param_name):
            setattr(glsl.par, param_name, value)

    # Cook and capture
    glsl.cook(force=True)
    op('render_out').cook(force=True)

    # Check for compile errors
    errors = glsl.errors
    if errors:
        return {
            'name': name,
            'status': 'failed',
            'error': errors
        }

    # Save output
    output_path = f"{output_dir}/{name}.png"
    op('render_out').save(output_path)

    return {
        'name': name,
        'status': 'passed',
        'output': output_path,
        'params': params
    }


def write_results(results, output_dir):
    """Write results JSON and completion marker."""
    # Write results
    results_path = f"{output_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'tests': results,
            'total': len(results),
            'status': 'complete'
        }, f, indent=2)

    # Write completion marker
    marker_path = f"{output_dir}/complete.marker"
    with open(marker_path, 'w') as f:
        f.write('done')
```

### 4.3 Test Configuration JSON Format

**File:** `fixtures/config/hello_world_test.json`

```json
{
  "version": "1.0",
  "description": "Hello World shader test configuration",
  "tests": [
    {
      "name": "brightness_default",
      "shader": "glsl/effects/hello_brightness.frag",
      "params": {
        "uBrightness": 1.0
      }
    },
    {
      "name": "brightness_half",
      "shader": "glsl/effects/hello_brightness.frag",
      "params": {
        "uBrightness": 0.5
      }
    }
  ]
}
```

### 4.4 Results JSON Format

```json
{
  "tests": [
    {
      "name": "brightness_default",
      "status": "passed",
      "output": "/tmp/output/brightness_default.png",
      "params": {"uBrightness": 1.0}
    },
    {
      "name": "brightness_half",
      "status": "passed",
      "output": "/tmp/output/brightness_half.png",
      "params": {"uBrightness": 0.5}
    }
  ],
  "total": 2,
  "status": "complete"
}
```

---

## Section 5: External Orchestration

### 5.1 Python Orchestrator Script

**File:** `scripts/td_test_runner.py`

The orchestrator launches TouchDesigner, passes configuration via environment variables, waits for a completion marker, and reads results.

```bash
# Basic usage
python docs/touchdesigner/scripts/td_test_runner.py \
  --config fixtures/config/hello_world_test.json \
  --output /tmp/output

# With timeout and verbose output
python docs/touchdesigner/scripts/td_test_runner.py \
  --config fixtures/config/hello_world_test.json \
  --output /tmp/output \
  --timeout 180 \
  --verbose

# Output as JSON
python docs/touchdesigner/scripts/td_test_runner.py \
  --config fixtures/config/hello_world_test.json \
  --output /tmp/output \
  --json
```

### 5.2 pytest Integration

```python
# tests/test_td_shaders.py

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "docs/touchdesigner/scripts"))
from td_test_runner import run_td_tests

FIXTURES = Path(__file__).parent.parent / "docs/touchdesigner/fixtures"


class TestShaderRender:
    """Test shader rendering in TouchDesigner."""

    @pytest.fixture
    def td_output_dir(self, tmp_path):
        return tmp_path / "td_output"

    @pytest.mark.parametrize("brightness", [0.5, 1.0, 1.5, 2.0])
    def test_brightness_sweep(self, td_output_dir, brightness):
        """Test brightness shader at various values."""
        config = {
            "version": "1.0",
            "tests": [{
                "name": f"brightness_{brightness}",
                "shader": str(FIXTURES / "shaders/hello_brightness.frag"),
                "params": {"uBrightness": brightness}
            }]
        }

        config_path = td_output_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f)

        results = run_td_tests(str(config_path), str(td_output_dir))

        assert results.success, f"Test failed: {results.error}"
        assert results.total == 1
```

### 5.3 CI/CD Considerations

Due to TouchDesigner licensing (per TD-002), full render tests cannot run in CI. Recommended approach:

| Environment | What Runs | Tool |
|-------------|-----------|------|
| CI (GitHub Actions) | GLSL syntax validation | `validate_glsl.py` |
| CI (macOS runner) | GLSL→SPIRV→Metal compilation | `glslangValidator` + `spirv-cross` |
| Local (pre-PR) | Full TD render tests | `td_test_runner.py` |
| Local (pre-commit) | GLSL validation | pre-commit hook |

---

## Section 6: Parameter Sweep Testing

### 6.1 Single Parameter Sweep

```json
{
  "version": "1.0",
  "sweep": {
    "type": "single",
    "shader": "glsl/effects/hello_brightness.frag",
    "parameter": "uBrightness",
    "values": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
  }
}
```

### 6.2 Multi-Parameter Grid Sweep

```json
{
  "version": "1.0",
  "sweep": {
    "type": "grid",
    "shader": "glsl/effects/chromatic_aberration.frag",
    "parameters": {
      "uAmount": [0.01, 0.05, 0.1],
      "uAngle": [0.0, 45.0, 90.0, 180.0]
    }
  }
}
```

Generates 3 × 4 = 12 test cases.

### 6.3 Sweep Implementation (TD Python)

```python
def generate_sweep_tests(sweep_config):
    """Generate individual tests from sweep configuration."""
    from itertools import product

    tests = []

    if sweep_config['type'] == 'single':
        param = sweep_config['parameter']
        for value in sweep_config['values']:
            tests.append({
                'name': f"{param}_{value}",
                'shader': sweep_config['shader'],
                'params': {param: value}
            })

    elif sweep_config['type'] == 'grid':
        param_names = list(sweep_config['parameters'].keys())
        param_values = list(sweep_config['parameters'].values())

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            name = '_'.join(f"{k}{v}" for k, v in params.items())
            tests.append({
                'name': name,
                'shader': sweep_config['shader'],
                'params': params
            })

    return tests
```

### 6.4 Golden Image Modes

| Mode | Purpose |
|------|---------|
| `generate` | Save outputs as new golden reference images |
| `compare` | Compare outputs against existing golden images |

```json
{
  "version": "1.0",
  "mode": "compare",
  "golden_dir": "fixtures/expected/brightness/",
  "tests": [...]
}
```

---

## Section 7: Frame Capture and Comparison

### 7.1 Reliable Frame Capture

```python
def capture_frame_reliably(top_name, output_path, wait_frames=5):
    """
    Capture frame with reliability measures.

    Args:
        top_name: Name of TOP to capture
        output_path: Destination file path
        wait_frames: Frames to wait for cook completion
    """
    top = op(top_name)
    top.cook(force=True)

    def do_capture():
        top.save(output_path)

    run("do_capture()", delayFrames=wait_frames)
```

### 7.2 Perceptual Comparison (SSIM)

Use SSIM (Structural Similarity Index) instead of MD5 for robustness across GPU drivers:

```python
from skimage.metrics import structural_similarity as ssim
import cv2

def compare_images(actual_path, expected_path, threshold=0.99):
    """Compare images using SSIM."""
    actual = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)
    expected = cv2.imread(expected_path, cv2.IMREAD_GRAYSCALE)

    score, _ = ssim(actual, expected, full=True)

    return {
        'passed': score >= threshold,
        'ssim_score': score,
        'threshold': threshold
    }
```

---

## Section 8: Common Pitfalls and Solutions

### 8.1 No #version Directive

**Problem:** TouchDesigner auto-injects `#version` — including it causes double declaration.

```glsl
// WRONG - will fail
#version 330 core
void main() { ... }

// CORRECT - TD injects version
void main() { ... }
```

The `validate_glsl.py` script checks for this and rejects shaders with `#version`.

### 8.2 Missing TDOutputSwizzle()

**Problem:** Forgetting `TDOutputSwizzle()` may cause incorrect alpha or color space issues.

```glsl
// WRONG - potential issues
fragColor = color;

// CORRECT - proper output handling
fragColor = TDOutputSwizzle(color);
```

### 8.3 Apple Silicon Texture Indexing

**Problem:** Non-constant texture array indices may fail on some Metal GPU families.

```glsl
// POTENTIALLY PROBLEMATIC on Apple Silicon
uniform int uInputIndex;
vec4 color = texture(sTD2DInputs[uInputIndex], vUV);  // Dynamic index

// SAFE - constant index
vec4 color = texture(sTD2DInputs[0], vUV);
```

### 8.4 Frame-Perfect Rendering

**Problem:** Real-time mode causes frame timing inconsistencies in tests.

**Solution:** Disable real-time mode:

```python
def onStart():
    project.realTime = False  # CRITICAL for deterministic tests
```

### 8.5 Delayed Quit

**Problem:** `project.quit()` may execute before file writes complete.

**Solution:** Use `delayFrames`:

```python
# WRONG
project.quit()

# CORRECT - wait 60 frames
run('project.quit()', delayFrames=60)
```

### 8.6 macOS GLSL Version Limit

**Problem:** macOS supports max GLSL 4.1 (not 4.60 like Windows).

**Solution:** Target these versions:

| Shader Type | GLSL Version |
|-------------|--------------|
| Fragment | 330 core |
| Compute | 430 core |

### 8.7 Non-Commercial License Cap

**Problem:** TD Non-Commercial caps at 1280×1280 resolution.

**Solution:** Use smaller test images:

- 64×64 for fast unit tests
- 256×256 for visual quality tests
- Max 1280×720 for NC license

### 8.8 Headless Mac

**Problem:** Macs without display may not initialize GPU properly.

**Solution:** Use HDMI dummy plug ($10-15) or run via Screen Sharing.

### 8.9 TD Python Time API

**Problem:** Incorrect time API usage.

```python
# WRONG - does not exist
frame = absTime.frame

# CORRECT
frame = me.time.frame
seconds = absTime.seconds
```

---

## Section 9: Integration with Existing Infrastructure

### 9.1 Using validate_glsl.py

```bash
# Validate single shader
python docs/touchdesigner/scripts/validate_glsl.py myshader.frag

# Validate all effects
python docs/touchdesigner/scripts/validate_glsl.py \
  docs/touchdesigner/glsl/effects/*.frag
```

### 9.2 Creating Test Fixtures

Add fixtures to `glsl/test_fixtures/`:

```
test_fixtures/
├── valid/                    # Should pass validation
│   ├── minimal_passthrough.frag
│   └── my_new_effect.frag
└── invalid/                  # Should fail validation
    ├── has_version.frag
    └── syntax_error.frag
```

### 9.3 Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: glsl-validate
        name: Validate GLSL shaders
        entry: python docs/touchdesigner/scripts/validate_glsl.py
        language: python
        files: \.frag$|\.comp$
        pass_filenames: true
```

### 9.4 CI/CD Workflow

Add to `.github/workflows/touchdesigner-macos.yml`:

```yaml
jobs:
  glsl-validation:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: Install tools
        run: brew install glslang spirv-cross

      - name: Validate GLSL
        run: |
          python docs/touchdesigner/scripts/validate_glsl.py \
            docs/touchdesigner/glsl/effects/*.frag

      - name: Validate Metal compilation
        run: |
          for shader in docs/touchdesigner/glsl/effects/*.frag; do
            glslangValidator -V -S frag "$shader" -o /tmp/shader.spv
            spirv-cross --msl /tmp/shader.spv --output /tmp/shader.metal
            xcrun -sdk macosx metal -c /tmp/shader.metal -o /dev/null
          done
```

---

## Appendix A: Quick Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHADER_TEST_MODE` | Enable testing (`1`/`0`) | `0` |
| `TEST_CONFIG` | Config JSON path | Required |
| `OUTPUT_DIR` | Output directory | `/tmp/td_output` |

### TD Python API

| Operation | Code |
|-----------|------|
| Disable real-time | `project.realTime = False` |
| Force render | `op('top').cook(force=True)` |
| Save frame | `op('top').save('/path/out.png')` |
| Delayed quit | `run('project.quit()', delayFrames=60)` |
| Get time | `absTime.seconds` or `me.time.frame` |

### Shader Template

```glsl
// effect_name.frag
// Description
//
// Uniforms:
//   uParam1 - Description (range: 0.0-1.0)

uniform float uParam1;

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);

    // ... effect implementation ...

    fragColor = TDOutputSwizzle(color);
}
```

---

## Section 10: Version Control with toeexpand/toecollapse

TouchDesigner projects (`.toe` files) are binary archives. For git-friendly version control, use TD's `toeexpand` and `toecollapse` utilities to convert between binary and text formats.

### 10.1 What are toeexpand/toecollapse?

| Tool | Purpose | Command |
|------|---------|---------|
| `toeexpand` | Extract `.toe` → `.toe.dir` (text format) | `toeexpand project.toe` |
| `toecollapse` | Compress `.toe.dir` → `.toe` (binary) | `toecollapse project.toe.dir project.toe` |

**Location on macOS:**
```bash
/Applications/TouchDesigner.app/Contents/MacOS/toeexpand
/Applications/TouchDesigner.app/Contents/MacOS/toecollapse
```

### 10.2 The .toe.dir Format

When expanded, a TD project becomes a directory of text files:

```
shader_test_harness.toe.dir/
├── .build              # TD version metadata
├── .start              # Startup settings (realtime, cookrate)
├── .root               # Root terminator
├── .grps               # Groups
├── .parm               # Global parameters
├── .application        # Window layout
├── project1.n          # Container definition (type, position)
├── project1.parm       # Container parameters
└── project1/
    ├── execute_autotest.n      # Execute DAT definition
    ├── execute_autotest.parm   # Parameters (activestart=1)
    ├── execute_autotest.text   # Python code content
    ├── glsl_under_test.n       # GLSL TOP definition
    ├── glsl_under_test.parm    # GLSL TOP parameters
    └── ...
```

**File conventions:**
- `.n` files: Node definition (type, tile position, inputs, flags)
- `.parm` files: Parameter values
- `.text` files: Text/Python DAT content

### 10.3 Recommended Git Workflow

```gitignore
# .gitignore - track .toe.dir, ignore .toe binaries
docs/touchdesigner/fixtures/projects/*.toe
!docs/touchdesigner/fixtures/projects/*.toe.dir
*.toc
```

**Development workflow:**
```
1. Edit .toe.dir files directly (text editor)
   OR open .toe in TD, make changes, run toeexpand
        ↓
2. git add/commit .toe.dir changes
   (.toe is gitignored - never commit binaries)
        ↓
3. Before running tests: toecollapse rebuilds .toe
   (td_test_runner.py does this automatically)
```

### 10.4 Build Script

The `build_test_harness.py` script handles rebuild detection:

```bash
# Build only if .toe.dir has changes
python docs/touchdesigner/scripts/build_test_harness.py

# Force rebuild
python docs/touchdesigner/scripts/build_test_harness.py --force

# Expand .toe back to .toe.dir (after TD edits)
python docs/touchdesigner/scripts/build_test_harness.py --expand
```

The `td_test_runner.py` automatically rebuilds before tests:
```python
# This happens automatically when you run tests
ensure_toe_built(project_path, verbose=verbose)
```

### 10.5 Editing .toe.dir Files Directly

For simple changes, edit text files directly:

**Example: Change GLSL TOP resolution**
```bash
# Edit project1/glsl_under_test.parm
vim fixtures/projects/shader_test_harness.toe.dir/project1/glsl_under_test.parm
```

**Parameter file format:**
```
?
resolutionw 0 512    # Changed from 256
resolutionh 0 512    # Changed from 256
glsldat 0 ../text_shader
?
```

**Example: Update Execute DAT Python code**
```bash
vim fixtures/projects/shader_test_harness.toe.dir/project1/execute_autotest.text
```

### 10.6 Working with TD GUI + toeexpand

For complex changes, use TouchDesigner GUI then export:

```bash
# 1. Open existing .toe in TouchDesigner
open fixtures/projects/shader_test_harness.toe

# 2. Make changes in TD GUI, save

# 3. Export back to .toe.dir format
/Applications/TouchDesigner.app/Contents/MacOS/toeexpand \
  fixtures/projects/shader_test_harness.toe

# 4. Commit the updated .toe.dir
git add fixtures/projects/shader_test_harness.toe.dir/
git commit -m "Update test harness operators"
```

---

## Appendix B: Building the Test Harness .toe

**Recommended approach:** Use the pre-built `shader_test_harness.toe.dir` and run `build_test_harness.py` to generate the `.toe` file.

**Alternative:** Manual steps to create `shader_test_harness.toe` in TouchDesigner:

1. **Create new project** → Save as `fixtures/projects/shader_test_harness.toe`

2. **Add operators:**
   - Text DAT named `text_shader` → File parameter empty initially
   - GLSL TOP named `glsl_under_test` → GLSL DAT = `text_shader`
   - Movie File In TOP named `movie_in` → load test video
   - Connect: `movie_in` → `glsl_under_test`
   - Render TOP named `render_out` → input = `glsl_under_test`
   - Info DAT named `info_compile` → Operator = `glsl_under_test`
   - Null TOP named `null_output` → input = `render_out`

3. **Add Execute DAT:**
   - Name: `execute_autotest`
   - Enable "Start" callback
   - Paste test runner code from Section 4.2

4. **Configure GLSL TOP:**
   - Vectors page: Set up uniform bindings as needed
   - Output Resolution: Match input or fixed test size (e.g., 256×256)

5. **Save project**

---

## Sources

- [TouchDesigner GLSL TOP Documentation](https://docs.derivative.ca/GLSL_TOP)
- [TouchDesigner Python Reference](https://docs.derivative.ca/Python)
- [TouchDesigner Execute DAT](https://docs.derivative.ca/Execute_DAT)
- [GLSL 3.30 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.3.30.pdf)
