# TouchDesigner Migration: Remediation Plan

Pre-implementation validation and infrastructure hardening based on red team analysis.

## Related Documents

- [00-IMPLEMENTATION-OVERVIEW.md](00-IMPLEMENTATION-OVERVIEW.md) - High-level roadmap and decisions
- [01-LINTING-INFRASTRUCTURE.md](01-LINTING-INFRASTRUCTURE.md) - Linting tools, CI/CD, editor integration
- [02-EFFECTS-AND-DEMOS.md](02-EFFECTS-AND-DEMOS.md) - GLSL effects, .tox structure, demo system

---

## Issue Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 3 | Blocking - must fix before Phase 2 |
| HIGH | 5 | Important - fix during infrastructure phase |
| MEDIUM | 3 | Address before production |

```
DEPENDENCY GRAPH
================

[1.1 TD Preamble] ----+
                      |
[1.2 macOS CI] -------+---> [2.1 PCG Hash] ---> [3.1 Perceptual Diff]
                      |            |
[1.3 Fixtures] -------+     [2.2 Bilinear] ---> [4.2 Video Tests]
                      |            |
                      +---> [2.3 HSV] -----+--> [Phase 2: Effects]
                      |                    |
                      +---> [3.2 MoltenVK] +
                      |
                      +---> [5.1 Compute Limits]
                      |
                      +---> [5.3 Benchmarks]
```

---

## PHASE 1: Infrastructure Foundation

### 1.1 Extract Real TD Preamble

**Fixes:** Critical #2 (synthetic preamble is guesswork)

**Problem:** Current `validate_glsl.py` TD_PREAMBLE is hand-crafted, not extracted from actual TouchDesigner runtime.

**Deliverable:** `touchdesigner/reference/td_preamble_2022.glsl`

**Steps:**
1. Create minimal passthrough shader in TD 2022.20000+
2. Use TD's shader inspection tools to extract injected code
3. Document all uniforms, inputs, outputs, macros
4. Update `validate_glsl.py` TD_PREAMBLE to match exactly
5. Create version-specific preambles if TD 2023.x/2024.x differ

**Acceptance Criteria:**
- Shader that passes updated preamble compiles in actual TD without modification
- All TD built-in uniforms documented with types and purposes

---

### 1.2 Set Up macOS CI Runner

**Fixes:** Critical #3 (no target platform validation)

**Problem:** CI runs on Linux but target platform is macOS Apple Silicon. Cannot validate MoltenVK translation or TD runtime behavior.

**Deliverable:** `.github/workflows/touchdesigner-macos.yml`

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) GitHub-hosted `macos-14` | M1 hardware, no maintenance | Limited to free tier minutes |
| B) Self-hosted M1/M2 runner | Full control, TD license | Requires hardware, maintenance |

**Implementation (Option A):**
```yaml
name: TouchDesigner macOS Validation

on:
  push:
    branches: [main, develop, feature/*]
    paths: ['touchdesigner/**']

jobs:
  macos-validation:
    runs-on: macos-14  # M1 Apple Silicon
    steps:
      - uses: actions/checkout@v4

      - name: Install TouchDesigner
        run: |
          # Download TD installer (requires license for headless)
          brew install --cask touchdesigner || true

      - name: Install shader tools
        run: |
          brew install glslang spirv-cross

      - name: Validate GLSL->SPIRV->Metal path
        run: |
          for shader in touchdesigner/glsl/effects/*.frag; do
            glslangValidator -V -S frag "$shader" -o /tmp/shader.spv
            spirv-cross --msl /tmp/shader.spv --output /tmp/shader.metal
            xcrun -sdk macosx metal -c /tmp/shader.metal -o /dev/null
          done

      - name: Headless render test
        run: |
          # Requires TD license and project setup
          /Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner \
            -p touchdesigner/fixtures/projects/render_test.toe \
            -e "op('/render').cook(); op('/render').save('/tmp/output.png')"

      - uses: actions/upload-artifact@v4
        with:
          name: render-output
          path: /tmp/output.png
```

**Acceptance Criteria:**
- CI renders test shader on macOS and uploads artifact
- GLSL->SPIRV->Metal pipeline validates all shaders

---

### 1.3 Create Validation Fixture Infrastructure

**Fixes:** High #6 (migration order ignores dependencies)

**Problem:** Effects would be migrated before core utilities are validated.

**Deliverable:** Directory structure with test inputs and expected outputs

```
touchdesigner/fixtures/
├── inputs/
│   ├── checkerboard_64x64.png    # Bilinear alignment test
│   ├── gradient_64x64.png        # Color conversion test
│   ├── noise_reference.png       # Visual RNG comparison
│   └── solid_colors/
│       ├── red.png
│       ├── green.png
│       ├── blue.png
│       ├── white.png
│       ├── black.png
│       └── gray50.png
├── expected/
│   ├── pcg_hash_values.json      # Known-good hash outputs
│   ├── bilinear_samples.json     # Fractional sample values
│   └── hsv_conversions.json      # RGB<->HSV test vectors
├── videos/
│   ├── static_gradient.mov       # No motion baseline
│   ├── moving_gradient.mov       # Slow pan for temporal test
│   ├── high_motion.mov           # Fast movement stress test
│   └── color_bars_60fps.mov      # Technical reference
└── projects/
    ├── pcg_hash_test.toe         # Outputs hash as RGBA color
    ├── bilinear_test.toe         # Sampling validation
    ├── hsv_roundtrip.toe         # Color conversion test
    └── minimal_passthrough.toe   # Baseline TD project
```

**Test Vectors (pcg_hash_values.json):**
```json
{
  "test_seeds": [0, 1, 42, 65536, 2147483647, 2147483648, 4294967295],
  "expected_hashes": {
    "0": 2707161783,
    "1": 1442695040,
    "42": 3276206029,
    "...": "..."
  },
  "note": "Generate with Taichi reference implementation"
}
```

---

## PHASE 2: Algorithm Validation

### 2.1 PCG Hash Equivalence Testing

**Fixes:** Critical #1 (arithmetic mismatch between Taichi u32 and GLSL uint)

**Problem:** GLSL PCG may produce different bit patterns than Taichi due to unsigned shift semantics and MoltenVK translation.

**Deliverable:** Verified PCG implementation or documented divergence with fallback

**Test Shader (pcg_hash_test.frag):**
```glsl
// Outputs 32-bit hash as RGBA (8 bits per channel)
uniform int uTestSeed;

void main() {
    uint hash = pcg_hash(uint(uTestSeed));
    fragColor = vec4(
        float((hash >> 24u) & 0xFFu) / 255.0,
        float((hash >> 16u) & 0xFFu) / 255.0,
        float((hash >> 8u) & 0xFFu) / 255.0,
        float(hash & 0xFFu) / 255.0
    );
}
```

**Test Implementation:**
```python
# tests/test_pcg_equivalence.py
import pytest
from PIL import Image
import numpy as np

TEST_SEEDS = [0, 1, 42, 2**16, 2**31-1, 2**31, 2**32-1]

def decode_hash_from_rgba(rgba: tuple) -> int:
    """Decode 32-bit uint from RGBA pixel."""
    r, g, b, a = rgba
    return (r << 24) | (g << 16) | (b << 8) | a

def taichi_pcg_hash(seed: int) -> int:
    """Reference implementation from Taichi."""
    import taichi as ti
    # ... call actual Taichi kernel
    pass

@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_pcg_hash_equivalence(seed, render_glsl_shader):
    """Compare pcg_hash output: Taichi vs GLSL."""
    # Render GLSL shader with seed
    output = render_glsl_shader("pcg_hash_test.frag", {"uTestSeed": seed})
    glsl_hash = decode_hash_from_rgba(output.getpixel((0, 0)))

    # Get Taichi reference
    taichi_hash = taichi_pcg_hash(seed)

    assert glsl_hash == taichi_hash, (
        f"PCG hash mismatch at seed {seed}: "
        f"Taichi={taichi_hash}, GLSL={glsl_hash}"
    )
```

**Fallback Strategy:**
```
IF GLSL PCG differs from Taichi:
    |
    +-> Option A: Adjust GLSL shifts to match Taichi
    |   (explicit uint casts, avoid signed intermediates)
    |
    +-> Option B: Accept divergence
    |   - Generate GLSL-specific expected outputs
    |   - Document that TD effects differ from Taichi
    |
    +-> Option C: Alternative RNG
        - xorshift128+ (known portable)
        - Update all effects to use new RNG
```

---

### 2.2 Bilinear Sampling Alignment

**Fixes:** High #4 (coordinate system mismatch, 0.5 pixel offset risk)

**Problem:** Taichi and GLSL may have different texel coordinate conventions.

**Test Pattern:**
```
64x64 checkerboard where:
- Even positions (x+y % 2 == 0): RGB(255, 0, 0) red
- Odd positions: RGB(0, 255, 0) green

Sampling at exact pixel centers should return pure red or green.
Sampling between pixels should return interpolated yellow-ish.
```

**Test Implementation:**
```python
# tests/test_bilinear_alignment.py

def test_bilinear_at_pixel_centers():
    """Sample at exact pixel centers should return pure pixel value."""
    checkerboard = load_fixture("checkerboard_64x64.png")

    for x in [0, 15, 31, 47, 63]:
        for y in [0, 15, 31, 47, 63]:
            # UV for pixel center
            uv = ((x + 0.5) / 64.0, (y + 0.5) / 64.0)

            expected = checkerboard.getpixel((x, y))
            glsl_result = render_bilinear_shader(uv)

            # Must match exactly at pixel centers
            assert glsl_result == expected, (
                f"Pixel center mismatch at ({x},{y}): "
                f"expected={expected}, got={glsl_result}"
            )

def test_bilinear_interpolation():
    """Sample between pixels should interpolate correctly."""
    # Sample exactly between 4 pixels
    uv = (1.0 / 64.0, 1.0 / 64.0)  # Between pixels (0,0), (1,0), (0,1), (1,1)

    taichi_result = taichi_bilinear_sample(uv)
    glsl_result = render_bilinear_shader(uv)

    # Allow 1/255 tolerance for interpolation
    for c in range(3):
        assert abs(taichi_result[c] - glsl_result[c]) <= 1, (
            f"Interpolation mismatch at uv={uv}"
        )
```

**Coordinate Fix (if needed):**
```glsl
// Current (possibly wrong):
vec2 pos = uv * vec2(size) - 0.5;

// Alternative A (OpenGL convention):
vec2 pos = uv * vec2(size);

// Alternative B (explicit half-pixel offset):
vec2 pos = (uv - 0.5/vec2(size)) * vec2(size);
```

---

### 2.3 HSV Conversion Validation

**Fixes:** Implicit assumption that color math is correct

**Test Vectors:**
```json
{
  "conversions": [
    {"rgb": [1.0, 0.0, 0.0], "hsv": [0.0, 1.0, 1.0], "name": "pure red"},
    {"rgb": [0.0, 1.0, 0.0], "hsv": [0.333, 1.0, 1.0], "name": "pure green"},
    {"rgb": [0.0, 0.0, 1.0], "hsv": [0.667, 1.0, 1.0], "name": "pure blue"},
    {"rgb": [1.0, 1.0, 1.0], "hsv": [0.0, 0.0, 1.0], "name": "white"},
    {"rgb": [0.0, 0.0, 0.0], "hsv": [0.0, 0.0, 0.0], "name": "black"},
    {"rgb": [0.5, 0.5, 0.5], "hsv": [0.0, 0.0, 0.5], "name": "gray 50%"},
    {"rgb": [1.0, 0.5, 0.0], "hsv": [0.083, 1.0, 1.0], "name": "orange"},
    {"rgb": [0.5, 0.0, 0.5], "hsv": [0.833, 1.0, 0.5], "name": "purple"}
  ],
  "tolerance": 0.004
}
```

**Acceptance:** Roundtrip RGB->HSV->RGB error < 1/255 per channel.

---

## PHASE 3: Testing Infrastructure Hardening

### 3.1 Replace MD5 with Perceptual Diff

**Fixes:** High #7 (MD5 fragile to floating-point variance)

**Problem:** Bit-identical comparison fails across GPU drivers due to floating-point rounding differences.

**Deliverable:** `touchdesigner/scripts/perceptual_diff.py`

```python
"""Perceptual image comparison using SSIM instead of MD5."""

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    passed: bool
    ssim_score: float
    max_pixel_diff: int
    pixels_exceeding_threshold: int

def compare_images(
    actual_path: str,
    expected_path: str,
    ssim_threshold: float = 0.99,
    max_pixel_diff: int = 2
) -> ComparisonResult:
    """
    Compare images using perceptual metrics.

    Args:
        actual_path: Path to rendered output
        expected_path: Path to expected reference
        ssim_threshold: Minimum SSIM score (0.99 = very similar)
        max_pixel_diff: Maximum allowed per-pixel difference

    Returns:
        ComparisonResult with pass/fail and metrics
    """
    actual = np.array(Image.open(actual_path).convert('RGB'))
    expected = np.array(Image.open(expected_path).convert('RGB'))

    # Structural similarity
    score = ssim(actual, expected, channel_axis=2, data_range=255)

    # Per-pixel analysis
    diff = np.abs(actual.astype(int) - expected.astype(int))
    max_diff = int(diff.max())
    exceeding = int((diff > max_pixel_diff).sum())

    passed = score >= ssim_threshold and max_diff <= max_pixel_diff * 2

    return ComparisonResult(
        passed=passed,
        ssim_score=score,
        max_pixel_diff=max_diff,
        pixels_exceeding_threshold=exceeding
    )
```

**pytest Fixture:**
```python
# conftest.py
import pytest

@pytest.fixture
def assert_renders_match():
    """Fixture for perceptual image comparison."""
    def _compare(actual, expected, ssim_threshold=0.99, max_diff=2):
        result = compare_images(actual, expected, ssim_threshold, max_diff)
        if not result.passed:
            pytest.fail(
                f"Render mismatch:\n"
                f"  SSIM: {result.ssim_score:.4f} (need >= {ssim_threshold})\n"
                f"  Max pixel diff: {result.max_pixel_diff} (limit: {max_diff*2})\n"
                f"  Pixels exceeding threshold: {result.pixels_exceeding_threshold}"
            )
    return _compare
```

---

### 3.2 Add MoltenVK/Metal Validation Layer

**Fixes:** High #5 (glslangValidator doesn't catch TD runtime errors)

**Problem:** GLSL that passes linting may fail when translated to Metal via MoltenVK.

**CI Addition:**
```yaml
- name: Validate GLSL -> SPIRV -> Metal translation
  run: |
    for shader in touchdesigner/glsl/effects/*.frag; do
      echo "Validating: $shader"

      # Step 1: GLSL -> SPIRV
      glslangValidator -V -S frag "$shader" -o /tmp/shader.spv

      # Step 2: SPIRV -> Metal Shading Language
      spirv-cross --msl /tmp/shader.spv --output /tmp/shader.metal

      # Step 3: Compile Metal (catches translation issues)
      xcrun -sdk macosx metal -c /tmp/shader.metal -o /dev/null

      echo "  PASS: $shader"
    done
```

**Known MoltenVK Edge Cases (test fixtures):**
```glsl
// touchdesigner/glsl/test_fixtures/moltenvk_edge_cases.frag

// TEST 1: Dynamic array indexing (problematic on Metal)
uniform int uDynamicIndex;
vec4 test_dynamic_array() {
    // May fail: Metal requires constant indices for some operations
    return texture(sTD2DInputs[uDynamicIndex], vUV);
}

// TEST 2: Integer overflow wrapping
uint test_overflow() {
    // Should wrap to 0, but Metal behavior may differ
    return 0xFFFFFFFFu + 1u;
}

// TEST 3: Bitshift on sign bit
uint test_sign_shift() {
    // Should be 1, not sign-extended
    return 0x80000000u >> 31u;
}
```

---

### 3.3 Document Known Linter Gaps

**Deliverable:** Section in this document (below)

#### What glslangValidator Does NOT Catch

| Gap | Description | Workaround |
|-----|-------------|------------|
| TD uniform conflicts | Using a name TD reserves internally | Check TD docs for reserved names |
| MoltenVK translation | Metal has different semantics | SPIRV-Cross validation in CI |
| Texture binding limits | TD may have different limits than spec | Test at runtime |
| Precision mismatches | highp/mediump/lowp behavior | Always use highp on desktop |
| Non-constant array index | Metal restrictions | Avoid or test specifically |

---

## PHASE 4: Temporal Strategy & Video Handling

### 4.1 Effect Temporal Behavior Specification

**Fixes:** High #8 (video demos without temporal strategy)

| Effect | Seed Behavior | Frame Dependency | Animate Parameter |
|--------|---------------|------------------|-------------------|
| saturation | N/A | None | - |
| chromatic_aberration | N/A | None | - |
| noise | Configurable | Optional | `Animatenoise` |
| salt_pepper | Configurable | Optional | `Animatenoise` |
| corduroy | Static | None | - |
| gaussian_blur | N/A | None | - |
| circular_blur | N/A | None | - |
| motion_blur | N/A | None | - |
| downscale | N/A | None | - |
| slc_off | Static | None | - |
| band_swap | Static | None | - |
| buffer_corruption | Configurable | Optional | `Animateglitch` |
| bayer_filter | N/A | None | - |

**Temporal Uniform Convention:**
```glsl
// Standard temporal uniforms for effects with RNG
uniform int uSeed;           // Base seed (user-controlled)
uniform int uAnimateNoise;   // 0 = static, 1 = per-frame
uniform float uTime;         // absTime.seconds from TD

int getEffectiveSeed() {
    if (uAnimateNoise > 0) {
        // Time-based variation (smooth animation)
        return uSeed + int(uTime * 1000.0);
    }
    return uSeed;  // Static pattern
}
```

**TD Python for Frame Counter:**
```python
# CORRECT way to get frame index in TouchDesigner
frame_index = me.time.frame  # Current timeline frame

# WRONG (does not exist)
# frame_index = absTime.frame
```

---

### 4.2 Video Test Suite

**Deliverable:** `tests/test_temporal_stability.py`

```python
"""Temporal stability tests for video processing."""

import pytest

class TestTemporalStability:

    def test_static_noise_no_flicker(self, render_video_effect):
        """Static seed should produce identical noise every frame."""
        frames = render_video_effect(
            "noise",
            params={"Seed": 42, "Animatenoise": 0},
            num_frames=10
        )

        # All frames should be identical
        for i in range(1, len(frames)):
            ssim = compare_frames(frames[0], frames[i])
            assert ssim > 0.9999, (
                f"Flicker detected: frame {i} differs from frame 0 "
                f"(SSIM={ssim:.4f})"
            )

    def test_animated_noise_varies(self, render_video_effect):
        """Animated seed should produce different frames."""
        frames = render_video_effect(
            "noise",
            params={"Seed": 42, "Animatenoise": 1},
            num_frames=10
        )

        # Frames should differ
        for i in range(1, len(frames)):
            ssim = compare_frames(frames[0], frames[i])
            assert ssim < 0.95, (
                f"Animation not working: frame {i} too similar "
                f"(SSIM={ssim:.4f})"
            )

    def test_blur_no_temporal_artifacts(self, render_video_effect):
        """Blur on moving video should be smooth."""
        frames = render_video_effect(
            "gaussian_blur",
            params={"Kernelsize": 10},
            input_video="moving_gradient.mov",
            num_frames=30
        )

        # Inter-frame change should be smooth
        for i in range(1, len(frames)):
            ssim = compare_frames(frames[i-1], frames[i])
            assert ssim > 0.8, (
                f"Temporal artifact between frames {i-1} and {i} "
                f"(SSIM={ssim:.4f})"
            )
```

---

## PHASE 5: Performance, Limits & Process

### 5.1 Compute Shader Resolution Limits

**Fixes:** Medium #9 (Apple Silicon limits not tested)

**Test Matrix:**

| Resolution | Workgroups (16x16) | Invocations | Target |
|------------|-------------------|-------------|--------|
| 1920x1080 | 120 x 68 | 2M | MUST pass |
| 3840x2160 | 240 x 135 | 8M | MUST pass |
| 7680x4320 | 480 x 270 | 33M | SHOULD pass |
| 8192x8192 | 512 x 512 | 67M | MAY pass |

**Test Implementation:**
```python
@pytest.mark.parametrize("width,height", [
    (1920, 1080),
    (3840, 2160),
    (7680, 4320),
    (8192, 8192),
])
def test_compute_at_resolution(width, height, render_compute_effect):
    """Verify compute shaders work at various resolutions."""
    result = render_compute_effect(
        "band_swap",
        params={"Bandheight": 16, "Seed": 42},
        resolution=(width, height)
    )
    assert result.success, f"Failed at {width}x{height}: {result.error}"
```

---

### 5.2 GLSL Feasibility Criteria & Rollback Strategy

**Fixes:** Medium #10 (no rollback if pure GLSL fails)

**GLSL Feasibility Checklist:**
```
Before attempting pure GLSL for any effect, verify:

[ ] No random-access writes (fragment shaders read-only)
[ ] No inter-pixel communication (each pixel independent)
[ ] No recursive algorithms
[ ] Loops bounded to <1024 iterations
[ ] No bitwise float manipulation (no reinterpret_cast)
[ ] No geometry shader requirements (Metal limitation)
```

**Escalation Path:**
```
GLSL Fragment attempt
    |
    +--[FAIL]--> Document reason
                    |
                    v
              GLSL Compute attempt
                    |
                    +--[FAIL]--> Document reason
                                    |
                                    v
                              C++ TOP implementation
                                    |
                                    +--[FAIL]--> Python Script TOP
                                                 (not real-time)
```

**Pre-Approved C++ Exceptions:**

| Effect | Reason | Decision |
|--------|--------|----------|
| buffer_corruption | XOR mode needs bitwise float ops | C++ approved |
| band_swap | Random tile access | Try compute first, C++ fallback |

---

### 5.3 Performance Benchmarking

**Fixes:** Medium #11 (no performance validation)

**Performance Targets:**

| Effect | 1080p | 4K | Notes |
|--------|-------|-----|-------|
| saturation | <2ms | <8ms | Simple |
| chromatic_aberration | <2ms | <8ms | Simple |
| noise | <2ms | <8ms | PCG is fast |
| gaussian_blur | <8ms | <32ms | Two passes |
| circular_blur | <8ms | <32ms | Many samples |
| buffer_corruption | <4ms | <16ms | Compute |

**60fps Budget:** 16.67ms per frame

**Benchmark Script:**
```python
# touchdesigner/scripts/benchmark.py

def benchmark_effect(effect: str, params: dict,
                     resolution: tuple = (1920, 1080),
                     num_frames: int = 100) -> dict:
    """Measure effect performance."""
    import time

    # Warm-up
    for _ in range(10):
        render_frame(effect, params, resolution)

    # Benchmark
    times = []
    for _ in range(num_frames):
        start = time.perf_counter()
        render_frame(effect, params, resolution)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg = sum(times) / len(times)
    return {
        "effect": effect,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "avg_ms": round(avg, 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "fps": round(1000 / avg, 1),
        "meets_60fps": avg < 16.67
    }
```

---

## Execution Checklist

### Gate Criteria (MUST pass before Phase 2)

- [ ] macOS CI runner operational and rendering test shaders
- [ ] TD preamble extracted and validate_glsl.py updated
- [ ] PCG hash validated (or divergence documented with new fixtures)
- [ ] Bilinear sampling aligned (or offset corrected)
- [ ] Perceptual diff replaces MD5 in all tests
- [ ] MoltenVK validation added to CI
- [ ] Temporal behavior defined for all 14 effects

### Verification Commands

```bash
# Run algorithm validation tests
pytest tests/test_pcg_equivalence.py tests/test_bilinear_alignment.py -v

# Run temporal stability tests
pytest tests/test_temporal_stability.py -v

# Run full CI locally
act -j macos-validation  # Requires 'act' tool

# Benchmark all effects
python touchdesigner/scripts/benchmark.py --all --output perf_baseline.json
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PCG hash doesn't match | Medium | High | Fallback to GLSL-specific fixtures |
| MoltenVK has unsupported feature | Low | High | Check Apple Metal docs first |
| TD headless not licensable in CI | Medium | Medium | Use self-hosted runner |
| Performance doesn't meet 60fps | Low | Medium | Optimize or reduce effect quality |
| macOS runner unavailable | Low | High | Fall back to manual testing |
