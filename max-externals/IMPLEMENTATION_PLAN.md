# SevenRad Max 8 Effects - Implementation Plan

## Overview

Port 12 Taichi GPU image effects to Max 8 using a hybrid architecture:
- **GPU effects**: GenExpr shaders via `jit.gl.pix`
- **CPU utilities**: C externals for tile/mask generation

Each effect is independently implementable by a separate programmer.

---

## Project Structure

```
max-externals/
├── CMakeLists.txt                    # Build system
├── package-info.json                 # Max package metadata
├── source/
│   ├── common/
│   │   ├── sr_rng.h                  # PCG random number generator
│   │   └── sr_utils.h                # Shared utilities
│   ├── sr.tilegen/                   # CPU: random tile bounds generator
│   │   └── sr.tilegen.c
│   └── sr.maskgen/                   # CPU: SLC-off mask generator
│       └── sr.maskgen.c
├── code/                             # GPU GenExpr shaders
│   ├── sr.noise.genjit
│   ├── sr.saturation.genjit
│   ├── sr.chromatic.genjit
│   ├── sr.blur.h.genjit
│   ├── sr.blur.v.genjit
│   ├── sr.blur.circular.genjit
│   ├── sr.motion.genjit
│   ├── sr.saltpepper.genjit
│   ├── sr.corduroy.genjit
│   ├── sr.bayer.mosaic.genjit
│   ├── sr.bayer.demosaic.genjit
│   ├── sr.bandswap.genjit
│   ├── sr.slcoff.genjit
│   ├── sr.corruption.genjit
│   └── sr.downscale.genjit
├── help/                             # Max help patchers
│   └── sr.*.maxhelp
├── tests/
│   ├── input/                        # Test input images
│   │   └── test_512x512.png
│   ├── reference/                    # Python-generated reference images
│   ├── actual/                       # Max-generated test outputs
│   ├── generate_references.py        # Creates test images with known params
│   ├── compare_outputs.py            # Pixel comparison tool
│   └── test_suite.maxpat             # Max patcher that runs all tests
└── docs/
    └── effect_specs/                 # Per-effect specification documents
```

---

## Testing Framework

### Overview

The testing framework ensures each ported effect produces identical output to the Python/Taichi reference implementation.

### Workflow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ generate_refs.py │────▶│ reference/*.png  │     │ test_suite.maxpat│
│ (Python/Taichi)  │     │ + params.json    │     │ (runs Max effects)│
└──────────────────┘     └────────┬─────────┘     └────────┬─────────┘
                                  │                        │
                                  ▼                        ▼
                         ┌──────────────────┐     ┌──────────────────┐
                         │ expected/        │     │ actual/          │
                         │ noise_001.png    │     │ noise_001.png    │
                         └────────┬─────────┘     └────────┬─────────┘
                                  │                        │
                                  └───────────┬────────────┘
                                              ▼
                                  ┌──────────────────┐
                                  │ compare_outputs.py│
                                  │ PSNR > 40dB = PASS│
                                  └──────────────────┘
```

### Acceptance Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| PSNR | > 40 dB | Allows ±1 pixel value difference from rounding |
| SSIM | > 0.99 | Structural similarity |
| Max pixel diff | ≤ 2 | No pixel more than 2/255 off |

### Running Tests

```bash
# 1. Generate reference images (run once from project root)
cd ../  # sevenrad-stills project root
python max-externals/tests/generate_references.py

# 2. Open Max and run test_suite.maxpat
# This will process all test cases and save to actual/

# 3. Compare outputs
python max-externals/tests/compare_outputs.py
```

---

## Shared Components

### PCG Random Number Generator

All effects requiring randomness MUST use the same PCG hash function to match Python output.

**Constants (must match Python exactly):**
```
COORD_PRIME_X = 374761393
COORD_PRIME_Y = 668265263
PCG_MULT = 747796405
PCG_INC = 2891336453
PCG_FACTOR = 277803737
UINT32_MAX = 4294967296.0
```

**GenExpr Implementation:**
```genexpr
// Include in every shader that needs RNG
pcg_hash(x, y, seed) {
    state = int((y * 374761393 + x * 668265263 + seed) * 747796405 + 2891336453);
    word = int(((state >> ((state >> 28) + 4)) ^ state) * 277803737);
    return float((word >> 22) ^ word) / 4294967296.0;
}

rand_gaussian(x, y, seed, sigma) {
    u1 = max(pcg_hash(x, y, seed), 0.0001);
    u2 = pcg_hash(x, y, seed + 1);
    return sigma * sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}
```

---

## Effect Specifications

Each effect below contains everything a programmer needs to implement it independently.

---

### EFFECT 01: sr.noise

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.noise.genjit` |
| **Source** | `src/sevenrad_stills/operations/noise_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| mode | int | 0-2 | 0 | 0=gaussian, 1=row, 2=column |
| amount | float | 0.0-1.0 | 0.1 | Noise intensity (std dev for gaussian) |
| seed | int | 0-2^31 | 0 | Random seed for reproducibility |

#### Algorithm

```genexpr
Param mode(0);
Param amount(0.1);
Param seed(0);

px = int(norm.x * dim.x);
py = int(norm.y * dim.y);

if (mode == 0) {  // Gaussian - independent per channel
    noise_r = rand_gaussian(px, py, seed, amount);
    noise_g = rand_gaussian(px, py, seed + 1000, amount);
    noise_b = rand_gaussian(px, py, seed + 2000, amount);
} else if (mode == 1) {  // Row - same noise across entire row
    noise_r = noise_g = noise_b = (pcg_hash(0, py, seed) - 0.5) * amount * 2;
} else {  // Column - same noise across entire column
    noise_r = noise_g = noise_b = (pcg_hash(px, 0, seed) - 0.5) * amount * 2;
}

out = clamp(in1 + vec(noise_r, noise_g, noise_b, 0), 0, 1);
```

#### Test Cases

| ID | Parameters | Expected Behavior |
|----|------------|-------------------|
| noise_000 | mode=0, amount=0.1, seed=42 | Light Gaussian noise |
| noise_001 | mode=0, amount=0.5, seed=42 | Heavy Gaussian noise |
| noise_002 | mode=1, amount=0.2, seed=123 | Horizontal scanlines |
| noise_003 | mode=2, amount=0.3, seed=456 | Vertical artifacts |

#### Checklist

- [ ] Read source: `noise_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 02: sr.saturation

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.saturation.genjit` |
| **Source** | `src/sevenrad_stills/operations/saturation_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| factor | float | 0.0-3.0 | 1.0 | 0=grayscale, 1=original, >1=boosted |

#### Algorithm

```genexpr
Param factor(1.0);

r = in1.r; g = in1.g; b = in1.b;

// RGB to HSV
cmax = max(r, max(g, b));
cmin = min(r, min(g, b));
delta = cmax - cmin;

// Hue calculation (degrees)
h = 0;
if (delta > 0.00001) {
    if (cmax == r) {
        h = 60 * fmod((g - b) / delta + 6, 6);
    } else if (cmax == g) {
        h = 60 * ((b - r) / delta + 2);
    } else {
        h = 60 * ((r - g) / delta + 4);
    }
}

// Saturation and Value
s = (cmax > 0.00001) ? delta / cmax : 0;
v = cmax;

// Apply saturation factor
s = clamp(s * factor, 0, 1);

// HSV to RGB
c = v * s;
x = c * (1 - abs(fmod(h / 60, 2) - 1));
m = v - c;

sector = int(h / 60) % 6;
if (sector == 0) { r = c + m; g = x + m; b = m; }
else if (sector == 1) { r = x + m; g = c + m; b = m; }
else if (sector == 2) { r = m; g = c + m; b = x + m; }
else if (sector == 3) { r = m; g = x + m; b = c + m; }
else if (sector == 4) { r = x + m; g = m; b = c + m; }
else { r = c + m; g = m; b = x + m; }

out = vec(r, g, b, in1.a);
```

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| saturation_000 | factor=0.0 | Complete grayscale |
| saturation_001 | factor=0.5 | 50% desaturated |
| saturation_002 | factor=1.5 | Moderately boosted |
| saturation_003 | factor=2.0 | Heavily saturated |

#### Checklist

- [ ] Read source: `saturation_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 03: sr.chromatic

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.chromatic.genjit` |
| **Source** | `src/sevenrad_stills/operations/chromatic_aberration_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| shift_x | float | -50 to 50 | 0.0 | Horizontal pixel shift |
| shift_y | float | -50 to 50 | 0.0 | Vertical pixel shift |

#### Algorithm

```genexpr
Param shift_x(0);
Param shift_y(0);

// Convert pixel shift to normalized coordinates
sx = shift_x / dim.x;
sy = shift_y / dim.y;

// Red channel: positive shift
// Green channel: no shift (reference)
// Blue channel: negative shift
r = sample(in1, norm + vec(sx, sy)).r;
g = in1.g;
b = sample(in1, norm - vec(sx, sy)).b;

out = vec(r, g, b, in1.a);
```

**Note:** Uses bilinear interpolation via `sample()` for sub-pixel accuracy.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| chromatic_000 | shift_x=5, shift_y=0 | Horizontal RGB fringing |
| chromatic_001 | shift_x=0, shift_y=5 | Vertical RGB fringing |
| chromatic_002 | shift_x=3, shift_y=3 | Diagonal fringing |
| chromatic_003 | shift_x=10, shift_y=-5 | Asymmetric fringing |

#### Checklist

- [ ] Read source: `chromatic_aberration_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 04: sr.blur (Two-Pass Separable Gaussian)

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) - **TWO FILES** |
| **Files** | `code/sr.blur.h.genjit`, `code/sr.blur.v.genjit` |
| **Source** | `src/sevenrad_stills/operations/blur_gaussian_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| sigma | float | 0.0-50.0 | 1.0 | Gaussian standard deviation |

#### Algorithm - Horizontal Pass (sr.blur.h.genjit)

```genexpr
Param sigma(1.0);

radius = int(ceil(sigma * 3));
sum = vec(0, 0, 0, 0);
weight_sum = 0;

for (i = -radius; i <= radius; i += 1) {
    weight = exp(-(i * i) / (2 * sigma * sigma + 0.0001));
    offset = vec(float(i) / dim.x, 0);
    sum += sample(in1, norm + offset) * weight;
    weight_sum += weight;
}

out = sum / max(weight_sum, 0.0001);
```

#### Algorithm - Vertical Pass (sr.blur.v.genjit)

```genexpr
Param sigma(1.0);

radius = int(ceil(sigma * 3));
sum = vec(0, 0, 0, 0);
weight_sum = 0;

for (i = -radius; i <= radius; i += 1) {
    weight = exp(-(i * i) / (2 * sigma * sigma + 0.0001));
    offset = vec(0, float(i) / dim.y);
    sum += sample(in1, norm + offset) * weight;
    weight_sum += weight;
}

out = sum / max(weight_sum, 0.0001);
```

#### Max Patcher Usage

```
[jit.gl.pix @gen sr.blur.h] → [jit.gl.pix @gen sr.blur.v]
```

Both objects must receive the same `sigma` parameter.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| blur_000 | sigma=1.0 | Subtle blur |
| blur_001 | sigma=5.0 | Medium blur |
| blur_002 | sigma=15.0 | Heavy blur |
| blur_003 | sigma=0.0 | No change (passthrough) |

#### Checklist

- [ ] Read source: `blur_gaussian_taichi.py`
- [ ] Implement horizontal pass shader
- [ ] Implement vertical pass shader
- [ ] Create help patcher showing chain
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 05: sr.blur.circular

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.blur.circular.genjit` |
| **Source** | `src/sevenrad_stills/operations/blur_circular_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| radius | int | 0-30 | 5 | Circular disk radius in pixels |

#### Algorithm

```genexpr
Param radius(5);

sum = vec(0, 0, 0, 0);
count = 0;

// Iterate over bounding box, check if inside circle
for (y = -radius; y <= radius; y += 1) {
    for (x = -radius; x <= radius; x += 1) {
        if (x*x + y*y <= radius*radius) {
            offset = vec(float(x) / dim.x, float(y) / dim.y);
            sum += sample(in1, norm + offset);
            count += 1;
        }
    }
}

out = sum / max(float(count), 1.0);
```

**Note:** This is O(r²) per pixel - may be slow for large radii.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| blur_circular_000 | radius=3 | Subtle bokeh |
| blur_circular_001 | radius=10 | Medium bokeh |
| blur_circular_002 | radius=20 | Heavy bokeh |
| blur_circular_003 | radius=0 | No change |

#### Checklist

- [ ] Read source: `blur_circular_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 06: sr.motion

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.motion.genjit` |
| **Source** | `src/sevenrad_stills/operations/motion_blur_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| kernel_size | int | 1-100 | 15 | Blur length in pixels |
| angle | float | 0-360 | 0 | Direction in degrees (0=right, 90=up) |

#### Algorithm

```genexpr
Param kernel_size(15);
Param angle(0);

rad = angle * 3.14159265359 / 180.0;
dx = cos(rad) / dim.x;
dy = -sin(rad) / dim.y;  // Negative because Y is flipped

sum = vec(0, 0, 0, 0);
half = float(kernel_size - 1) / 2.0;

for (i = 0; i < kernel_size; i += 1) {
    t = float(i) - half;
    offset = vec(t * dx, t * dy);
    sum += sample(in1, norm + offset);
}

out = sum / float(kernel_size);
```

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| motion_000 | kernel_size=15, angle=0 | Horizontal blur (right) |
| motion_001 | kernel_size=15, angle=90 | Vertical blur (up) |
| motion_002 | kernel_size=30, angle=45 | Diagonal blur |
| motion_003 | kernel_size=1, angle=0 | No change |

#### Checklist

- [ ] Read source: `motion_blur_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 07: sr.saltpepper

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.saltpepper.genjit` |
| **Source** | `src/sevenrad_stills/operations/salt_pepper_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| amount | float | 0.0-1.0 | 0.05 | Fraction of pixels affected |
| salt_ratio | float | 0.0-1.0 | 0.5 | Ratio: 0=all pepper, 1=all salt |
| seed | int | any | 0 | Random seed |

#### Algorithm

```genexpr
Param amount(0.05);
Param salt_ratio(0.5);
Param seed(0);

px = int(norm.x * dim.x);
py = int(norm.y * dim.y);

r1 = pcg_hash(px, py, seed);      // Decide if affected
r2 = pcg_hash(px, py, seed + 1);  // Decide salt vs pepper

out = in1;
if (r1 < amount) {
    if (r2 < salt_ratio) {
        out = vec(1, 1, 1, in1.a);  // Salt (white)
    } else {
        out = vec(0, 0, 0, in1.a);  // Pepper (black)
    }
}
```

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| saltpepper_000 | amount=0.05, salt_ratio=0.5, seed=42 | 5% noise, balanced |
| saltpepper_001 | amount=0.2, salt_ratio=0.5, seed=42 | 20% noise, balanced |
| saltpepper_002 | amount=0.1, salt_ratio=0.0, seed=42 | All pepper (black) |
| saltpepper_003 | amount=0.1, salt_ratio=1.0, seed=42 | All salt (white) |

#### Checklist

- [ ] Read source: `salt_pepper_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 08: sr.corduroy

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.corduroy.genjit` |
| **Source** | `src/sevenrad_stills/operations/corduroy_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| orientation | int | 0-1 | 0 | 0=vertical stripes, 1=horizontal |
| strength | float | 0.0-1.0 | 0.3 | Stripe intensity variation |
| density | float | 0.0-1.0 | 0.2 | Fraction of lines affected |
| seed | int | any | 0 | Random seed |

#### Algorithm

```genexpr
Param orientation(0);
Param strength(0.3);
Param density(0.2);
Param seed(0);

px = int(norm.x * dim.x);
py = int(norm.y * dim.y);

// Select line index based on orientation
line_idx = (orientation == 0) ? px : py;

// Check if this line is affected
r = pcg_hash(line_idx, 0, seed);

out = in1;
if (r < density) {
    // Generate brightness multiplier for this line
    // Range: [1 - strength*0.2, 1 + strength*0.2]
    mult = 1.0 + (pcg_hash(line_idx, 1, seed) - 0.5) * strength * 0.4;
    out.rgb = clamp(in1.rgb * mult, 0, 1);
}
```

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| corduroy_000 | orientation=0, strength=0.3, density=0.2, seed=42 | Vertical stripes |
| corduroy_001 | orientation=1, strength=0.3, density=0.2, seed=42 | Horizontal stripes |
| corduroy_002 | orientation=0, strength=0.8, density=0.5, seed=123 | Heavy vertical |
| corduroy_003 | orientation=1, strength=0.1, density=0.8, seed=456 | Subtle horizontal |

#### Checklist

- [ ] Read source: `corduroy_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 09: sr.bayer (Two-Pass)

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) - **TWO FILES** |
| **Files** | `code/sr.bayer.mosaic.genjit`, `code/sr.bayer.demosaic.genjit` |
| **Source** | `src/sevenrad_stills/operations/bayer_filter_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| pattern | int | 0-3 | 0 | 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG |

#### Bayer Patterns

```
RGGB (0):    BGGR (1):    GRBG (2):    GBRG (3):
R G          B G          G R          G B
G B          G R          B G          R G
```

#### Algorithm - Mosaic Pass (sr.bayer.mosaic.genjit)

```genexpr
Param pattern(0);

px = int(norm.x * dim.x);
py = int(norm.y * dim.y);
cx = px % 2;
cy = py % 2;

// RGGB pattern (pattern == 0)
if (pattern == 0) {
    if (cx == 0 && cy == 0) {
        out = vec(in1.r, 0, 0, 1);      // R position
    } else if (cx == 1 && cy == 1) {
        out = vec(0, 0, in1.b, 1);      // B position
    } else {
        out = vec(0, in1.g, 0, 1);      // G position
    }
}
// Implement other patterns similarly...
```

#### Algorithm - Demosaic Pass (sr.bayer.demosaic.genjit)

Bilinear interpolation:
- At R/B positions: interpolate G from 4 cross neighbors, interpolate B/R from 4 diagonal neighbors
- At G positions: interpolate R and B from 2 horizontal or 2 vertical neighbors

#### Max Patcher Usage

```
[jit.gl.pix @gen sr.bayer.mosaic] → [jit.gl.pix @gen sr.bayer.demosaic]
```

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| bayer_000 | pattern=0 | RGGB mosaic/demosaic artifacts |
| bayer_001 | pattern=1 | BGGR mosaic/demosaic artifacts |
| bayer_002 | pattern=2 | GRBG mosaic/demosaic artifacts |
| bayer_003 | pattern=3 | GBRG mosaic/demosaic artifacts |

#### Checklist

- [ ] Read source: `bayer_filter_taichi.py`
- [ ] Implement mosaic shader
- [ ] Implement demosaic shader
- [ ] Create help patcher showing chain
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 10: sr.downscale

| Property | Value |
|----------|-------|
| **Type** | GPU shader (GenExpr) |
| **File** | `code/sr.downscale.genjit` |
| **Source** | `src/sevenrad_stills/operations/downscale_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| scale | float | 0.01-1.0 | 0.5 | Scale factor (0.5 = half resolution) |
| pixelate | int | 0-1 | 1 | 1=upscale back for pixelation effect |
| method | int | 0-1 | 1 | 0=nearest, 1=bilinear |

#### Algorithm

```genexpr
Param scale(0.5);
Param pixelate(1);
Param method(1);

// Quantize to lower resolution grid
low_x = floor(norm.x * dim.x * scale) / (dim.x * scale);
low_y = floor(norm.y * dim.y * scale) / (dim.y * scale);
low_coord = vec(low_x, low_y);

if (method == 0) {
    // Nearest neighbor - sample at quantized position
    out = sample(in1, low_coord);
} else {
    // Bilinear - still looks blocky due to quantization
    out = sample(in1, low_coord);
}
```

**Note:** When pixelate=0, the output should actually be smaller. This requires jit.gl.pix dimension handling or a separate approach.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| downscale_000 | scale=0.5, pixelate=1, method=1 | 2x pixelation |
| downscale_001 | scale=0.25, pixelate=1, method=1 | 4x pixelation |
| downscale_002 | scale=0.5, pixelate=1, method=0 | 2x pixelation (nearest) |
| downscale_003 | scale=0.1, pixelate=1, method=1 | 10x pixelation |

#### Checklist

- [ ] Read source: `downscale_taichi.py`
- [ ] Implement GenExpr shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 11: sr.bandswap (Hybrid)

| Property | Value |
|----------|-------|
| **Type** | Hybrid: CPU external + GPU shader |
| **Files** | `source/sr.tilegen/sr.tilegen.c`, `code/sr.bandswap.genjit` |
| **Source** | `src/sevenrad_stills/operations/band_swap_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### CPU External: sr.tilegen

Generates random tile positions for corruption effects.

**Inlets:**
1. Bang - generate new tiles
2. int - tile_count
3. float - tile_size_min (0-1)
4. float - tile_size_max (0-1)
5. int - seed
6. int - width
7. int - height

**Outlets:**
1. List of [x, y, w, h, permutation_index] for each tile

**Permutation Indices:**
- 0: RGB (no change)
- 1: GRB
- 2: BGR
- 3: BRG
- 4: GBR
- 5: RBG

#### GPU Shader: sr.bandswap.genjit

Receives tile texture/params, applies channel permutation inside tiles.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| bandswap_000 | tile_count=5, size=[0.05,0.2], seed=42 | 5 swapped tiles |
| bandswap_001 | tile_count=20, size=[0.02,0.1], seed=42 | 20 small tiles |
| bandswap_002 | tile_count=3, size=[0.1,0.3], seed=123 | 3 large tiles |

#### Checklist

- [ ] Read source: `band_swap_taichi.py`
- [ ] Implement sr.tilegen C external
- [ ] Implement GPU shader
- [ ] Create help patcher showing connection
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 12: sr.slcoff (Hybrid)

| Property | Value |
|----------|-------|
| **Type** | Hybrid: CPU external + GPU shader |
| **Files** | `source/sr.maskgen/sr.maskgen.c`, `code/sr.slcoff.genjit` |
| **Source** | `src/sevenrad_stills/operations/slc_off_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| gap_width | float | 0.0-0.5 | 0.1 | Max gap fraction at edges |
| scan_period | int | 2-100 | 16 | Rows per scan cycle |
| fill_mode | int | 0-2 | 0 | 0=black, 1=white, 2=mean |

#### CPU External: sr.maskgen

Generates the wedge-shaped gap mask as a jit.matrix (single-plane float).

**Algorithm:**
1. Calculate center row: `center_y = height / 2`
2. For each scan line (every scan_period rows):
   - Distance from center: `|y - center_y| / (height/2)`
   - Gap width at this row: `distance * gap_width * width`
   - Create diagonal wedge across scan_period rows
   - Diagonal offset: 0.3 pixels per row

#### GPU Shader: sr.slcoff.genjit

Receives mask texture, fills gaps with specified color.

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| slcoff_000 | gap_width=0.1, scan_period=16, fill=black | Landsat gaps |
| slcoff_001 | gap_width=0.2, scan_period=8, fill=white | Wide white gaps |
| slcoff_002 | gap_width=0.05, scan_period=32, fill=black | Subtle gaps |

#### Checklist

- [ ] Read source: `slc_off_taichi.py`
- [ ] Implement sr.maskgen C external
- [ ] Implement GPU shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

### EFFECT 13: sr.corruption (Hybrid)

| Property | Value |
|----------|-------|
| **Type** | Hybrid: CPU external + GPU shader |
| **Files** | `source/sr.tilegen/sr.tilegen.c` (reuse), `code/sr.corruption.genjit` |
| **Source** | `src/sevenrad_stills/operations/buffer_corruption_taichi.py` |
| **Programmer** | _Unassigned_ |
| **Status** | Not started |

#### Parameters

| Name | Type | Range | Default | Description |
|------|------|-------|---------|-------------|
| corruption_type | int | 0-2 | 0 | 0=XOR, 1=invert, 2=shuffle |
| tile_count | int | 1-20 | 5 | Number of corrupted tiles |
| severity | float | 0.0-1.0 | 0.5 | Corruption intensity |
| tile_size_min | float | 0.0-1.0 | 0.05 | Min tile size |
| tile_size_max | float | 0.0-1.0 | 0.2 | Max tile size |
| seed | int | any | 0 | Random seed |

#### Corruption Modes

**XOR (0):** `pixel = pixel XOR (random_mask * severity * 255)`
**Invert (1):** `pixel = pixel * (1-severity) + (1-pixel) * severity`
**Shuffle (2):** Random RGB channel permutation per tile

#### Test Cases

| ID | Parameters | Expected |
|----|------------|----------|
| corruption_000 | type=0, tiles=5, severity=0.5, seed=42 | XOR corruption |
| corruption_001 | type=1, tiles=5, severity=0.5, seed=42 | Inverted tiles |
| corruption_002 | type=2, tiles=5, severity=0.8, seed=42 | Channel shuffled |

#### Checklist

- [ ] Read source: `buffer_corruption_taichi.py`
- [ ] Extend sr.tilegen for corruption params
- [ ] Implement GPU shader
- [ ] Create help patcher
- [ ] Pass all test cases (PSNR > 40dB)
- [ ] Document deviations

---

## Build Instructions

### Prerequisites

1. Max 8.5+ installed
2. Xcode Command Line Tools (macOS): `xcode-select --install`
3. CMake 3.19+: `brew install cmake`
4. Max SDK 8.2.0: Download from [Cycling74](https://cycling74.com/downloads/sdk)

### Setup

```bash
# Clone Max SDK into max-externals/
cd max-externals
git clone https://github.com/Cycling74/max-sdk.git

# Create build directory
mkdir build && cd build

# Generate Xcode project
cmake -G Xcode ..

# Build
cmake --build . --config Release
```

### Package Installation

Copy entire `max-externals/` folder to:
```
~/Documents/Max 8/Packages/sevenrad/
```

Restart Max to load the package.

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create project structure
- [ ] Set up CMake build system
- [ ] Implement shared PCG RNG in GenExpr
- [ ] Create test framework scripts
- [ ] Generate reference images

### Phase 2: Simple GPU Effects (Week 2-3)
- [ ] sr.noise
- [ ] sr.saturation
- [ ] sr.saltpepper
- [ ] sr.corduroy
- [ ] sr.chromatic
- [ ] sr.downscale

### Phase 3: Convolution Effects (Week 3-4)
- [ ] sr.blur (two-pass)
- [ ] sr.blur.circular
- [ ] sr.motion

### Phase 4: Complex/Hybrid Effects (Week 4-5)
- [ ] sr.bayer (two-pass)
- [ ] sr.tilegen (CPU external)
- [ ] sr.bandswap
- [ ] sr.slcoff + sr.maskgen
- [ ] sr.corruption

### Phase 5: Polish (Week 5-6)
- [ ] Help patchers for all effects
- [ ] Example project patcher
- [ ] Documentation
- [ ] Final test validation

---

## Programmer Assignment Template

| Effect | Programmer | Start Date | Status | PSNR Score |
|--------|------------|------------|--------|------------|
| sr.noise | | | Not started | - |
| sr.saturation | | | Not started | - |
| sr.chromatic | | | Not started | - |
| sr.blur | | | Not started | - |
| sr.blur.circular | | | Not started | - |
| sr.motion | | | Not started | - |
| sr.saltpepper | | | Not started | - |
| sr.corduroy | | | Not started | - |
| sr.bayer | | | Not started | - |
| sr.downscale | | | Not started | - |
| sr.bandswap | | | Not started | - |
| sr.slcoff | | | Not started | - |
| sr.corruption | | | Not started | - |

---

## References

### Max SDK Documentation
- [Max SDK 8.2.0 MOP QuickStart](https://sdk.cdn.cycling74.com/max-sdk-8.2.0/chapter_jit_mopqs.html)
- [jit.gl.pix.codebox Reference](https://docs.cycling74.com/reference/jit.gl.pix.codebox)
- [Porting Shadertoy to jit.gl.pix](https://cycling74.com/tutorials/porting-shadertoy-tutorials-to-jit-gl-pix)

### Source Code
- [cv.jit GitHub](https://github.com/Cycling74/cv.jit) - Example Jitter externals

### Taichi Source Operations
All source files are in: `src/sevenrad_stills/operations/`
- `noise_taichi.py`
- `saturation_taichi.py`
- `chromatic_aberration_taichi.py`
- `blur_gaussian_taichi.py`
- `blur_circular_taichi.py`
- `motion_blur_taichi.py`
- `salt_pepper_taichi.py`
- `corduroy_taichi.py`
- `bayer_filter_taichi.py`
- `band_swap_taichi.py`
- `slc_off_taichi.py`
- `buffer_corruption_taichi.py`
- `downscale_taichi.py`

### Taichi Kernel Utilities
- `taichi_kernels/random.py` - PCG RNG implementation
- `taichi_kernels/sampling.py` - Bilinear interpolation
- `taichi_kernels/convolution.py` - Convolution utilities
