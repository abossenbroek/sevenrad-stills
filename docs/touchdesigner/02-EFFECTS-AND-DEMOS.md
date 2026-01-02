# TouchDesigner Migration: Effects and Demos

Complete guide to migrating Taichi effects to GLSL shaders, building .tox operators, and creating the demo system.

## Related Documents

- [00-IMPLEMENTATION-OVERVIEW.md](00-IMPLEMENTATION-OVERVIEW.md) - High-level roadmap and decisions
- [01-LINTING-INFRASTRUCTURE.md](01-LINTING-INFRASTRUCTURE.md) - Linting tools, CI/CD, editor integration

---

## Common GLSL Utilities (tdCommon.glsl)

Create `touchdesigner/glsl/common/tdCommon.glsl`:

```glsl
// tdCommon.glsl - Shared utilities for SevenRad TouchDesigner effects
// Version: 0.1

#define TD_SHADERS_VERSION 0.1
#define TD_GL_CORE 330

// ============================================================================
// PCG Random Number Generator
// Ported from: src/sevenrad_stills/operations/taichi_kernels/random.py
// ============================================================================

uint pcg_hash(uint input_state) {
    uint state = input_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns random float in [0, 1)
float rand_float(ivec2 pos, int seed) {
    uint h = pcg_hash(uint(pos.x) + pcg_hash(uint(pos.y) + pcg_hash(uint(seed))));
    return float(h) / 4294967296.0;
}

// Box-Muller transform for Gaussian noise
float rand_gaussian(ivec2 pos, int seed, float sigma) {
    float u1 = rand_float(pos, seed);
    float u2 = rand_float(pos, seed + 1);
    // Avoid log(0)
    u1 = max(u1, 1e-10);
    float z = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
    return z * sigma;
}

// ============================================================================
// Bilinear Sampling
// Ported from: src/sevenrad_stills/operations/taichi_kernels/sampling.py
// ============================================================================

vec4 bilinear_sample(sampler2D tex, vec2 uv, ivec2 size) {
    vec2 pos = uv * vec2(size) - 0.5;
    ivec2 p0 = ivec2(floor(pos));
    vec2 f = fract(pos);

    // Clamp to valid range
    ivec2 p00 = clamp(p0, ivec2(0), size - 1);
    ivec2 p10 = clamp(p0 + ivec2(1, 0), ivec2(0), size - 1);
    ivec2 p01 = clamp(p0 + ivec2(0, 1), ivec2(0), size - 1);
    ivec2 p11 = clamp(p0 + ivec2(1, 1), ivec2(0), size - 1);

    vec4 c00 = texelFetch(tex, p00, 0);
    vec4 c10 = texelFetch(tex, p10, 0);
    vec4 c01 = texelFetch(tex, p01, 0);
    vec4 c11 = texelFetch(tex, p11, 0);

    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// ============================================================================
// Color Space Conversion
// Ported from: src/sevenrad_stills/operations/saturation_taichi.py
// ============================================================================

vec3 rgb_to_hsv(vec3 rgb) {
    float cmax = max(max(rgb.r, rgb.g), rgb.b);
    float cmin = min(min(rgb.r, rgb.g), rgb.b);
    float delta = cmax - cmin;

    float h = 0.0;
    if (delta > 0.0) {
        if (cmax == rgb.r) {
            h = mod((rgb.g - rgb.b) / delta, 6.0);
        } else if (cmax == rgb.g) {
            h = (rgb.b - rgb.r) / delta + 2.0;
        } else {
            h = (rgb.r - rgb.g) / delta + 4.0;
        }
        h /= 6.0;
    }

    float s = (cmax > 0.0) ? delta / cmax : 0.0;
    float v = cmax;

    return vec3(h, s, v);
}

vec3 hsv_to_rgb(vec3 hsv) {
    float h = hsv.x * 6.0;
    float s = hsv.y;
    float v = hsv.z;

    float c = v * s;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = v - c;

    vec3 rgb;
    if (h < 1.0)      rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else              rgb = vec3(c, 0.0, x);

    return rgb + m;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Luminance calculation (Rec. 709)
float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Clamp to valid color range
vec3 saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

vec4 saturate(vec4 x) {
    return clamp(x, 0.0, 1.0);
}
```

---

## Effect Classification

| Effect | Shader Type | Passes | Complexity | Notes |
|--------|-------------|--------|------------|-------|
| saturation | Fragment | 1 | Simple | RGB↔HSV conversion |
| chromatic_aberration | Fragment | 1 | Simple | Per-channel UV offset |
| noise (3 modes) | Fragment | 1 | Simple | PCG hash, uniform/gaussian/salt-pepper |
| salt_pepper | Fragment | 1 | Simple | Dual RNG threshold |
| corduroy | Fragment | 1 | Medium | Per-scanline brightness multiplier |
| downscale | Fragment | 1-2 | Medium | Box filter resize |
| gaussian_blur | Fragment | 2 (H+V) | Medium | Separable convolution |
| circular_blur | Fragment | 1 | Medium | 2D disk kernel sampling |
| motion_blur | Fragment | 1 | Medium | Linear kernel along angle |
| slc_off | Compute | 1 | Medium | Wedge-shaped mask application |
| band_swap | Compute | 1 | Medium | Tile-based band permutation |
| buffer_corruption | Compute | 1 | Complex | 3 corruption modes (shift/swap/zero) |
| bayer_filter | Fragment | 2 | Complex | Mosaic + demosaic passes |

---

## Migration Order

### Phase 2a: Common Utilities
- Create `tdCommon.glsl` with versioned headers
- Test all utility functions with minimal shaders

### Phase 3a: Simple Effects
1. **saturation** - Direct HSV manipulation
2. **chromatic_aberration** - UV offset per channel
3. **noise** - 3 modes using PCG hash
4. **salt_pepper** - Binary threshold noise

### Phase 3b: Medium Effects
5. **corduroy** - Scanline-based brightness
6. **downscale** - Resolution reduction
7. **gaussian_blur** - Two-pass separable (requires chained GLSL TOPs)

### Phase 3c: Multi-pass Effects
8. **circular_blur** - Disk kernel accumulation
9. **motion_blur** - Directional blur
10. **bayer_filter** - Mosaic (pass 1) + demosaic (pass 2)

### Phase 3d: Compute Shaders
11. **slc_off** - Wedge mask (GLSL 430 compute)
12. **band_swap** - Tile permutation (GLSL 430 compute)
13. **buffer_corruption** - Multi-mode corruption (GLSL 430 compute)

---

## .tox Operator Structure

Each effect is packaged as a reusable .tox component:

```
sr_[effect].tox/
├── base/
│   ├── in1 (In TOP)              # Input image
│   ├── out1 (Out TOP)            # Output result
│   ├── glsl_effect (GLSL TOP)    # Main shader
│   ├── help (Text DAT)           # Markdown documentation
│   └── info (Info DAT)           # Compile status/errors
├── parameters/
│   └── [Custom Parameters]       # Effect controls
└── logic/
    └── parameter_callbacks.py    # Python callbacks (if needed)
```

### Parameter Pages

| Page | Purpose |
|------|---------|
| **Effect** | Primary effect parameters (Value, Amount, Sigma, etc.) + Seed |
| **Demo** | Preset buttons (Subtle/Moderate/Extreme), Compare mode toggle |
| **About** | Version string, Author, Help link |

---

## Parameter Naming Convention

| Taichi Parameter | TouchDesigner Parameter | Type | Notes |
|------------------|-------------------------|------|-------|
| mode | Mode | Menu | Options as menu items |
| value | Value | Float | 0-1 or effect-specific range |
| seed | Seed | Int | Default: 42 |
| sigma | Sigma | Float | Standard deviation |
| amount | Amount | Float | Effect intensity |
| shift_x | Shiftx | Float | No underscores in TD |
| shift_y | Shifty | Float | No underscores in TD |
| kernel_size | Kernelsize | Int | Blur radius |
| angle | Angle | Float | Degrees (0-360) |

---

## Example Shader: saturation.frag

```glsl
// saturation.frag - Saturation adjustment effect
// Requires: tdCommon.glsl (included via TD preamble)

uniform float uSaturation;  // 0 = grayscale, 1 = original, >1 = boosted
uniform int uMode;          // 0 = multiply, 1 = add, 2 = set

void main() {
    vec4 color = texture(sTD2DInputs[0], vUV.st);
    vec3 hsv = rgb_to_hsv(color.rgb);

    if (uMode == 0) {
        hsv.y *= uSaturation;      // Multiply
    } else if (uMode == 1) {
        hsv.y += uSaturation - 1.0; // Add (1.0 = no change)
    } else {
        hsv.y = uSaturation;       // Set absolute
    }

    hsv.y = clamp(hsv.y, 0.0, 1.0);
    color.rgb = hsv_to_rgb(hsv);

    fragColor = TDOutputSwizzle(color);
}
```

---

## Example Compute Shader: band_swap.comp

```glsl
// band_swap.comp - Tile-based band permutation
// GLSL 430 compute shader

layout(local_size_x = 16, local_size_y = 16) in;

uniform int uBandHeight;    // Height of each band in pixels
uniform int uSeed;          // Random seed for permutation
uniform int uSwapCount;     // Number of band swaps to perform

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(sTD2DOutputs[0]);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Determine which band this pixel belongs to
    int bandIndex = pos.y / uBandHeight;
    int totalBands = (size.y + uBandHeight - 1) / uBandHeight;

    // Calculate swapped band index using RNG
    int newBandIndex = bandIndex;
    for (int i = 0; i < uSwapCount; i++) {
        uint h = pcg_hash(uint(i) + pcg_hash(uint(uSeed)));
        int band1 = int(h % uint(totalBands));
        int band2 = int(pcg_hash(h) % uint(totalBands));

        if (newBandIndex == band1) newBandIndex = band2;
        else if (newBandIndex == band2) newBandIndex = band1;
    }

    // Read from swapped position
    ivec2 srcPos = ivec2(pos.x, newBandIndex * uBandHeight + (pos.y % uBandHeight));
    srcPos.y = clamp(srcPos.y, 0, size.y - 1);

    vec4 color = texelFetch(sTD2DInputs[0], srcPos, 0);
    imageStore(sTD2DOutputs[0], pos, color);
}
```

---

## Demo System

### Per-Operator Demo

Each .tox includes built-in demo functionality:

**Compare Modes**:
- Side-by-side (50/50 split)
- Wipe (draggable divider)
- Toggle (A/B flip)

**Presets**:
| Preset | Purpose |
|--------|---------|
| Subtle | Barely noticeable effect |
| Moderate | Typical usage |
| Extreme | Maximum/artistic effect |

**Help DAT Content**:
```markdown
# sr_saturation

Adjusts color saturation using HSV color space.

## Parameters

- **Mode**: multiply, add, or set
- **Value**: Saturation amount (0=gray, 1=original, 2=double)
- **Seed**: Random seed (unused for this effect)

## Taichi Equivalent

```yaml
- operation: saturation
  mode: multiply
  value: 0.5
```

## Tips

- Use multiply mode for natural-looking adjustments
- Set mode is useful for forcing specific saturation levels
```

---

### Master Demo Project (sr_demo.toe)

```
sr_demo.toe/
├── ui/
│   ├── effect_browser         # Category-based effect selection
│   │   ├── blur/
│   │   ├── color/
│   │   ├── noise/
│   │   ├── degradation/
│   │   └── satellite/
│   ├── param_panel            # Dynamic parameter UI
│   ├── help_viewer            # Rendered markdown help
│   └── preset_selector        # Load/save effect chains
├── sources/
│   ├── sample_video           # Built-in test footage
│   ├── sample_image           # Static test images
│   ├── color_bars             # Technical test pattern
│   └── webcam                 # Live input option
├── effects/
│   └── [all sr_*.tox files]   # Loaded on demand
├── chain/
│   ├── chain_builder          # Drag-and-drop effect chaining
│   ├── chain_renderer         # Processes the chain
│   └── preset_chains/
│       ├── vhs_aesthetic.json
│       ├── satellite_glitch.json
│       ├── vintage_lens.json
│       └── film_grain.json
└── comparison/
    ├── side_by_side           # Original vs processed
    ├── difference             # Pixel difference view
    └── histogram              # Before/after histograms
```

---

## Effect Categories

### Blur
- gaussian_blur (separable, two-pass)
- circular_blur (disk kernel)
- motion_blur (directional)

### Color
- saturation (HSV manipulation)
- chromatic_aberration (channel offset)

### Noise
- noise (uniform, gaussian, salt-pepper modes)
- salt_pepper (dedicated S&P noise)

### Degradation
- downscale (resolution reduction)
- buffer_corruption (glitch effects)

### Satellite
- slc_off (Landsat scan line corrector failure simulation)
- corduroy (scanline brightness variation)
- bayer_filter (CFA mosaic/demosaic)
- band_swap (tile permutation)

---

## Unit Render Tests

Store expected outputs for deterministic regression testing:

```
tests/fixtures/renders/
├── saturation/
│   ├── grayscale_seed42.png       # 64×64 reference
│   ├── saturated_2x_seed42.png
│   └── input.png                  # Test input image
├── noise/
│   ├── gaussian_sigma0.1_seed42.png
│   ├── uniform_0.2_seed42.png
│   └── input.png
└── ...
```

### Test Workflow

1. **Headless Render**: TouchDesigner batch mode renders effect
2. **MD5 Comparison**: pytest compares output hash to expected
3. **Tolerance Mode**: Optional pixel-diff threshold for floating-point variance

```python
# tests/test_td_renders.py
def test_saturation_grayscale():
    """Verify saturation=0 produces grayscale output."""
    output = render_effect("saturation", {"Value": 0.0})
    expected = load_fixture("saturation/grayscale_seed42.png")
    assert md5(output) == md5(expected)
```

---

## TouchDesigner-Specific Considerations

### Multi-Pass Effects

For effects requiring multiple passes (gaussian_blur, bayer_filter):

```
Chain of GLSL TOPs:
in1 → glsl_pass1 → glsl_pass2 → out1
```

Each pass uses the previous output as input via `sTD2DInputs[0]`.

### Uniform Binding

Parameters are exposed via Custom Parameters on the .tox component:
- Float → `uniform float uParamName;`
- Int → `uniform int uParamName;`
- Menu → `uniform int uParamName;` (index)
- Color → `uniform vec4 uParamName;`

### Resolution Handling

Access resolution via TouchDesigner uniforms:
```glsl
vec2 resolution = uTDOutputInfo.zw;  // Output width, height
ivec2 size = textureSize(sTD2DInputs[0], 0);  // Input size
```

### Alpha Handling

Always preserve alpha unless specifically modifying it:
```glsl
vec4 color = texture(sTD2DInputs[0], vUV.st);
// Modify color.rgb only
color.rgb = processed_rgb;
fragColor = TDOutputSwizzle(color);  // Preserves alpha
```

---

## Sources

- [TouchDesigner GLSL TOP](https://docs.derivative.ca/GLSL_TOP)
- [TouchDesigner Custom Parameters](https://docs.derivative.ca/Custom_Parameters)
- [TouchDesigner .tox Components](https://docs.derivative.ca/Component)
- [GLSL 3.30 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.3.30.pdf)
- [GLSL 4.30 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.30.pdf)
