# SevenRad Max Effects Test Suite

## Overview

The test suite provides automated validation for all SevenRad Max 8 effects by comparing outputs against Python/Taichi reference implementations.

## Files

- **test_suite.maxpat** - Max 8 patcher with UI for manual and automated testing
- **test_runner.js** - JavaScript automation script for batch testing
- **manifest.json** - Master list of all 38 test cases across 11 effects
- **input/test_image.png** - Standard test input image (512x512)
- **reference/*.png** - Python-generated reference outputs
- **reference/*.json** - Parameter files for each test case
- **actual/** - Max-generated outputs (created when tests run)

## Test Cases

### Complete Test Matrix

| Effect | Test Cases | Type | Notes |
|--------|------------|------|-------|
| noise | 4 | GPU Single-pass | Gaussian, row, column modes |
| saturation | 4 | GPU Single-pass | HSV color space manipulation |
| chromatic | 4 | GPU Single-pass | RGB channel shifting |
| blur | 4 | GPU Two-pass | Horizontal + vertical separable Gaussian |
| blur_circular | 3 | GPU Single-pass | Circular disk convolution |
| motion | 4 | GPU Single-pass | Directional motion blur |
| corduroy | 3 | GPU Single-pass | Scanline brightness variation |
| bayer | 4 | GPU Two-pass | Mosaic + demosaic |
| bandswap | 3 | Hybrid CPU+GPU | Requires sr.tilegen external |
| downscale | 3 | GPU Single-pass | Pixelation effect |
| slcoff | 2 | Hybrid CPU+GPU | Requires sr.maskgen external |
| **TOTAL** | **38** | | |

### Not Yet Implemented

- **saltpepper** - No test cases defined (empty array in manifest)
- **corruption** - No test cases defined (empty array in manifest)

## Usage

### Method 1: Manual Testing (Recommended for Development)

1. Open **test_suite.maxpat** in Max 8
2. Click "Load test image" button (loads `input/test_image.png`)
3. Select effect from dropdown (e.g., "noise")
4. Select test case (e.g., "0 - Light Gaussian noise")
5. Click "Run Test" button
6. Output appears in preview window and saves to `actual/noise_000.png`
7. Repeat for all test cases

### Method 2: Automated Batch Testing (Not Fully Implemented)

The JavaScript automation is limited by Max's threading model and GPU context requirements. For true automation, use the Python approach:

```bash
# From max-externals/tests/ directory
python run_all_max_tests.py
```

This would require:
- Max application scripting via OSC or UDP
- Node.js bridge to control Max remotely
- Or manual iteration through the UI

### Method 3: Python Reference Generation

```bash
# Generate reference images (already done)
cd /path/to/sevenrad-stills
python max-externals/tests/generate_references.py
```

## Validation

After running tests, compare outputs:

```bash
cd max-externals/tests
python compare_outputs.py
```

### Pass Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| PSNR | > 40 dB | Peak Signal-to-Noise Ratio |
| SSIM | > 0.99 | Structural Similarity Index |
| Max Pixel Diff | ≤ 2 | No pixel differs by more than 2/255 |

### Expected Results

For a correctly implemented effect:
```
noise_000.png:
  PSNR: 45.2 dB ✓ PASS
  SSIM: 0.995 ✓ PASS
  Max diff: 1 ✓ PASS
```

## Test Case Examples

### noise_000.json
```json
{
  "effect": "noise",
  "case_id": 0,
  "description": "Light Gaussian noise",
  "params": {
    "mode": "gaussian",
    "amount": 0.1,
    "seed": 42
  }
}
```

### chromatic_002.json
```json
{
  "effect": "chromatic",
  "case_id": 2,
  "description": "Diagonal fringing",
  "params": {
    "shift_x": 3.0,
    "shift_y": 3.0
  }
}
```

### blur_001.json
```json
{
  "effect": "blur",
  "case_id": 1,
  "description": "Medium blur",
  "params": {
    "sigma": 5.0
  }
}
```

## Effect Implementation Notes

### Single-Pass Effects

Apply directly via `jit.gl.pix`:

```
[jit.matrix test_input]
    |
[jit.gl.pix @gen sr.noise @mode 0 @amount 0.1 @seed 42]
    |
[jit.matrix test_output]
    |
[write actual/noise_000.png]
```

### Two-Pass Effects

Chain two shaders:

```
[jit.matrix test_input]
    |
[jit.gl.pix @gen sr.blur.h @sigma 5.0]
    |
[jit.gl.pix @gen sr.blur.v @sigma 5.0]
    |
[jit.matrix test_output]
    |
[write actual/blur_001.png]
```

### Hybrid CPU+GPU Effects

Combine C external with shader:

```
[bang] → [sr.tilegen 5 0.05 0.2 42 512 512]
              |
              | (tile parameters)
              ↓
[jit.matrix test_input] → [jit.gl.pix @gen sr.bandswap]
                                |
                        [jit.matrix test_output]
```

## Troubleshooting

### Tests Fail with Low PSNR

**Possible causes:**
1. RNG constants don't match Python (see IMPLEMENTATION_PLAN.md)
2. Floating-point precision differences
3. Color space conversion errors (RGB ↔ HSV)
4. Incorrect parameter mapping (e.g., mode string vs int)

**Fix:**
- Verify PCG hash implementation in GenExpr
- Check parameter type conversions in test_runner.js
- Compare intermediate values with Python

### GPU Context Errors

**Error:** `jit.gl.pix: no OpenGL context`

**Fix:**
```
[jit.world @erase_color 0 0 0 1]  ← Add this before jit.gl.pix
```

### File Not Found Errors

**Error:** Cannot read `input/test_image.png`

**Fix:**
Ensure working directory is set to `max-externals/tests/`:
```
[cd /path/to/max-externals/tests]
```

Or use absolute paths in the patcher.

### JavaScript Automation Fails

JavaScript automation in Max has limitations:
- Cannot easily create GL contexts
- Matrix processing is CPU-bound
- No direct file I/O for JSON parsing

**Recommended approach:**
Use manual UI-based testing or external automation via OSC/UDP.

## File Naming Convention

All test files follow this pattern:

```
{effect}_{case_id}.{ext}

Examples:
noise_000.json          (parameters)
noise_000.png           (reference output)
actual/noise_000.png    (Max output)

blur_002.json
blur_002.png
actual/blur_002.png
```

Case IDs are zero-padded to 3 digits (000-999).

## Adding New Test Cases

1. **Update manifest.json:**
```json
"myeffect": [
  {
    "case_id": 0,
    "description": "Basic test",
    "output_path": "max-externals/tests/reference/myeffect_000.png",
    "params_path": "max-externals/tests/reference/myeffect_000.json"
  }
]
```

2. **Create parameter file:**
```bash
echo '{
  "effect": "myeffect",
  "case_id": 0,
  "description": "Basic test",
  "params": {
    "intensity": 0.5
  }
}' > reference/myeffect_000.json
```

3. **Generate reference:**
```python
# Add to generate_references.py
def test_myeffect_000():
    img = load_test_image()
    result = myeffect_taichi(img, intensity=0.5)
    result.write("reference/myeffect_000.png")
```

4. **Update test_suite.maxpat:**
- Add "myeffect" to effect dropdown
- Add test case descriptions to case dropdown

5. **Run test and validate**

## Integration with CI/CD

For automated testing in continuous integration:

```bash
#!/bin/bash
# ci_test.sh

# Generate references (if needed)
python generate_references.py

# Run Max tests (requires Max installation)
# This part is tricky - Max doesn't have good CLI automation
# Options:
# 1. Use MaxMSP's JavaScript API via node.js
# 2. Use OSC/UDP to trigger tests remotely
# 3. Manual testing during development

# Compare outputs
python compare_outputs.py > test_results.txt

# Check for failures
if grep -q "FAIL" test_results.txt; then
    echo "Tests failed!"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
```

## Performance Benchmarks

Expected processing times (M1 Mac, 512x512 image):

| Effect | Time | FPS |
|--------|------|-----|
| noise | ~2ms | 500 |
| saturation | ~3ms | 333 |
| chromatic | ~2ms | 500 |
| blur (both passes) | ~8ms | 125 |
| blur_circular | ~15ms | 66 |
| motion | ~5ms | 200 |
| corduroy | ~2ms | 500 |
| bayer (both passes) | ~6ms | 166 |
| downscale | ~2ms | 500 |

GPU shaders should process in real-time (>30 FPS) for 1920x1080.

## Known Limitations

1. **No saltpepper/corruption tests** - These effects aren't fully specified yet
2. **Manual iteration required** - Full automation is complex in Max
3. **GPU context required** - Can't run headless easily
4. **Platform-specific** - Tested on macOS with M1, may differ on Windows/Intel

## Future Enhancements

- [ ] Add automation via Max's TCP/UDP API
- [ ] Implement remaining effects (saltpepper, corruption)
- [ ] Add performance benchmarking to test suite
- [ ] Create GitHub Actions workflow (if Max licensing allows)
- [ ] Add visual diff viewer in Max patcher
- [ ] Generate HTML report from compare_outputs.py

## Contact

For issues with the test suite, see:
- IMPLEMENTATION_PLAN.md (effect specifications)
- Individual help patchers in max-externals/help/
- Python source code in src/sevenrad_stills/operations/
