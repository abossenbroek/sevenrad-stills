# Tutorial: Degradr-Inspired Image Effects

This tutorial demonstrates degradation and optical effects inspired by the [degradr repository](https://github.com/nhauber99/degradr) (MIT licensed), adapted to use pure NumPy/SciPy/scikit-image instead of PyTorch. These operations simulate various forms of image degradation from analog media, digital sensors, and optical imperfections.

## Attribution

Original degradr implementation by nhauber99 (MIT License)
Adaptations: Converted from PyTorch to NumPy/SciPy for better PIL integration
See: `/Users/antonbossenbroek/Documents_local/photo/7rad/video/LICENSE_DEGRADR.txt`

## Prerequisites

- sevenrad-stills installed and configured
- YouTube video URL (replace placeholder URLs in examples)
- Basic familiarity with YAML pipeline system (see [docs/PIPELINE.md](../../PIPELINE.md))
- Understanding of [docs/FILTER_GUIDE.md](../../FILTER_GUIDE.md) recommended

## Tutorial Overview

| Tutorial | Operations | Aesthetic Goal | Difficulty |
|----------|-----------|----------------|------------|
| [vhs-scanlines](#tutorial-1-vhs-scanlines-effect) | chromatic_aberration, noise, blur_gaussian, compression | VHS tape artifacts and analog video degradation | Intermediate |
| [sensor-noise](#tutorial-2-digital-sensor-noise-effect) | bayer_filter, noise, compression | Digital camera sensor artifacts and high-ISO noise | Intermediate |
| [lens-artifacts](#tutorial-3-vintage-lens-artifacts-effect) | blur_circular, chromatic_aberration, saturation | Vintage lens bokeh and optical imperfections | Beginner |

## Available Degradr Operations

These operations are registered and available in pipelines:

### Blur Operations
- **`blur_gaussian`** - Gaussian blur with configurable sigma
  - Parameters: `sigma` (float, typically 0.5-5.0)
  - Uses: `scipy.ndimage.gaussian_filter`

- **`blur_circular`** - Circular/bokeh blur with custom kernel
  - Parameters: `radius` (int, typically 3-15)
  - Uses: `scipy.ndimage.convolve` with circular kernel

### Noise Operations
- **`noise`** - Three noise modes for different artifacts
  - Parameters: `mode` (gaussian/row/column), `amount` (0.0-1.0), `seed` (optional int)
  - Modes:
    - `gaussian`: Pixel-level random noise
    - `row`: Horizontal scan line artifacts (VHS effect)
    - `column`: Vertical banding artifacts

### Optical Effects
- **`chromatic_aberration`** - Color channel shifting
  - Parameters: `shift_x` (int), `shift_y` (int)
  - Red and blue channels shift in opposite directions
  - Simulates lens chromatic aberration

- **`bayer_filter`** - Digital sensor mosaic pattern
  - Parameters: `pattern` (RGGB, BGGR, GRBG, GBRG)
  - Creates and demosaics Bayer pattern
  - Simulates digital sensor artifacts

### Enhancement Operations
- **`saturation`** - Color saturation adjustment
  - Parameters: `factor` (float, typically 0.5-2.0)
  - < 1.0 desaturates, > 1.0 saturates

- **`compression`** - JPEG compression with optional gamma
  - Parameters: `quality` (1-95), `subsampling` (0-2), `gamma` (optional float)
  - Gamma correction reveals noise in shadows

- **`motion_blur`** - Directional blur
  - Parameters: `kernel_size` (int), `angle` (degrees)
  - Creates motion blur at specified angle

## Common Segment Configuration

All tutorials use the same video segment for consistency:

```yaml
segment:
  start: 192.0    # 3 minutes 12 seconds
  end: 195.0      # 3 minutes 15 seconds
  interval: 0.0667  # 15 frames per second = 45 total frames
```

**Why these settings?**
- **3 seconds**: Enough variety without overwhelming output
- **15 fps**: Good balance between sample size and processing time
- **45 frames**: Substantial dataset to see filter effects across multiple frames

---

## Tutorial 1: VHS Scanlines Effect

**Goal**: Recreate the iconic VHS tape aesthetic with scanline artifacts, color fringing, and analog degradation.

**Use Case**: Nostalgic 80s/90s aesthetics, retro video art, analog memory exploration.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/degradr-effects/vhs-scanlines.yaml
```

### Expected Results

**Output**: `tutorials/degradr-effects/vhs-scanlines/final/` containing 45 degraded frames

**Intermediate Steps** (preserved for inspection):
- `vhs_color_fringing/` - Chromatic aberration applied
- `vhs_scanlines/` - Horizontal noise added
- `vhs_softness/` - Gaussian blur applied

**Visual Characteristics**:
- Chromatic aberration creating color fringing at edges
- Horizontal scanline noise patterns throughout
- Soft, analog blur removing digital sharpness
- JPEG compression simulating VHS digitization
- Overall nostalgic VHS playback aesthetic

### Pipeline Breakdown

```yaml
steps:
  # Step 1: Chromatic aberration
  - name: "vhs_color_fringing"
    operation: "chromatic_aberration"
    params:
      shift_x: 3    # Horizontal color separation
      shift_y: 1    # Slight vertical shift

  # Step 2: Horizontal scanline noise
  - name: "vhs_scanlines"
    operation: "noise"
    params:
      mode: "row"      # Horizontal noise bands
      amount: 0.15     # Moderate visibility
      seed: 42         # Reproducible results

  # Step 3: Gaussian blur for softness
  - name: "vhs_softness"
    operation: "blur_gaussian"
    params:
      sigma: 1.2       # Light blur

  # Step 4: JPEG compression
  - name: "vhs_digitization"
    operation: "compression"
    params:
      quality: 40      # Medium-low quality
      subsampling: 2   # Heavy chroma subsampling (4:2:0)
```

**Parameter Explanation**:
- `shift_x: 3, shift_y: 1` - Creates color fringing from tape misalignment
- `mode: "row"` - Horizontal noise is signature VHS look
- `amount: 0.15` - Visible but not overwhelming scanlines
- `sigma: 1.2` - Removes digital sharpness, adds analog softness
- `quality: 40` - Typical of 90s VHS-to-digital captures
- `subsampling: 2` - VHS had poor color vs luminance resolution

### What You'll Learn

- Combining multiple degradation layers for cumulative effects
- Using row noise for scan line artifacts
- Creating period-specific analog aesthetics
- Balancing blur, noise, and compression

### Experimentation Ideas

- **Heavier VHS degradation**: `shift_x: 6`, `amount: 0.25`, `quality: 30`
- **Tracking issues**: Add `motion_blur` with `angle: 0`, `kernel_size: 5`
- **Tape stretch**: Increase `shift_x` to 8-10 for severe misalignment
- **Random noise**: Remove `seed` parameter for variation each run

---

## Tutorial 2: Digital Sensor Noise Effect

**Goal**: Simulate digital camera sensor artifacts including Bayer pattern demosaicing, sensor noise, and compressed output.

**Use Case**: Low-quality digital camera aesthetic, high-ISO photography, computational imaging exploration.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/degradr-effects/sensor-noise.yaml
```

### Expected Results

**Output**: `tutorials/degradr-effects/sensor-noise/final/` containing 45 noisy frames

**Intermediate Steps**:
- `sensor_mosaic/` - Bayer filter artifacts
- `sensor_grain/` - Gaussian noise added

**Visual Characteristics**:
- Bayer pattern color artifacts (moire, false colors)
- Fine-grained gaussian noise throughout
- Noise more visible in darker regions
- Loss of color accuracy from demosaicing
- "Digital" look vs analog VHS softness

### Pipeline Breakdown

```yaml
steps:
  # Step 1: Bayer filter pattern
  - name: "sensor_mosaic"
    operation: "bayer_filter"
    params:
      pattern: "RGGB"   # Most common pattern

  # Step 2: Gaussian sensor noise
  - name: "sensor_grain"
    operation: "noise"
    params:
      mode: "gaussian"  # Natural sensor noise distribution
      amount: 0.12      # Simulates ISO 3200-6400
      seed: 1337

  # Step 3: Compression
  - name: "sensor_compression"
    operation: "compression"
    params:
      quality: 55       # Moderate quality
      subsampling: 1    # 4:2:2 chroma subsampling
      # gamma: 1.2      # Optional: reveals shadow noise
```

**Parameter Explanation**:
- `pattern: "RGGB"` - Most common Bayer pattern (Red-Green-Green-Blue grid)
- `mode: "gaussian"` - Real sensors have Gaussian noise distribution
- `amount: 0.12` - Moderate noise like high-ISO photography
- `quality: 55` - On-sensor compression quality
- `subsampling: 1` - Better than VHS but still lossy
- `gamma: 1.2` (commented) - Brightens shadows, reveals noise

### What You'll Learn

- How digital sensors create color through demosaicing
- Gaussian noise distribution in sensor output
- Optional gamma correction for shadow detail
- Difference between analog (VHS) and digital degradation

### Experimentation Ideas

- **Cheap sensor**: `amount: 0.20`, `quality: 35`, `subsampling: 2`
- **Different Bayer alignment**: Try `pattern: "BGGR"` or `"GRBG"`
- **Shadow noise**: Uncomment `gamma: 1.2` or try higher values (1.5)
- **Extreme ISO**: `amount: 0.30` for very noisy sensor
- **Add motion**: Add `motion_blur` for motion + noise combination

---

## Tutorial 3: Vintage Lens Artifacts Effect

**Goal**: Simulate optical imperfections of vintage lenses with bokeh, chromatic aberration, and color rendering.

**Use Case**: Vintage photography aesthetic, dreamy portraits, romantic cinematography, analog warmth.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/degradr-effects/lens-artifacts.yaml
```

### Expected Results

**Output**: `tutorials/degradr-effects/lens-artifacts/final/` containing 45 vintage-look frames

**Intermediate Steps**:
- `vintage_bokeh/` - Circular blur applied
- `vintage_fringing/` - Chromatic aberration added

**Visual Characteristics**:
- Soft, dreamy circular bokeh blur
- Color fringing at high-contrast edges
- Enhanced color saturation (vintage film palette)
- Romantic, analog photography aesthetic
- Loss of digital sharpness

### Pipeline Breakdown

```yaml
steps:
  # Step 1: Circular bokeh blur
  - name: "vintage_bokeh"
    operation: "blur_circular"
    params:
      radius: 6         # Moderate soft-focus

  # Step 2: Chromatic aberration
  - name: "vintage_fringing"
    operation: "chromatic_aberration"
    params:
      shift_x: 4        # Moderate horizontal shift
      shift_y: 2        # Slight vertical component

  # Step 3: Color saturation
  - name: "vintage_color"
    operation: "saturation"
    params:
      factor: 1.15      # Slightly boosted colors
```

**Parameter Explanation**:
- `radius: 6` - Moderate bokeh for dreamy soft-focus effect
- `shift_x: 4, shift_y: 2` - Vintage lenses had chromatic aberration at frame edges
- `factor: 1.15` - Subtle saturation boost for warm, nostalgic palette
- Circular blur mimics lens aperture shape in out-of-focus areas

### What You'll Learn

- Creating optical effects with computational operations
- Simulating physical lens characteristics
- Combining blur, aberration, and color for cohesive aesthetic
- Simple but effective vintage photography look

### Experimentation Ideas

- **Heavy bokeh**: `radius: 10-12` for dreamy portrait effect
- **Strong aberration**: `shift_x: 6`, `shift_y: 3` for vintage wide-angle look
- **Faded vintage**: `factor: 0.75` for desaturated film aesthetic
- **Technicolor**: `factor: 1.4` for oversaturated vintage color
- **Film grain**: Add gaussian noise with `amount: 0.05` for texture
- **Photo print**: Add compression after saturation for print aesthetic

---

## Advanced Topics

### Combining Multiple Effects

All effects can be layered. Order matters:

```yaml
# Vintage film camera with sensor noise
steps:
  - operation: "bayer_filter"           # Sensor artifacts
  - operation: "noise"                  # Film grain
    params:
      mode: "gaussian"
      amount: 0.05
  - operation: "blur_circular"          # Lens softness
  - operation: "chromatic_aberration"   # Lens aberration
  - operation: "saturation"             # Film color
```

```yaml
# Extreme glitch aesthetic
steps:
  - operation: "bayer_filter"
  - operation: "chromatic_aberration"
  - operation: "noise"
    params:
      mode: "row"
      amount: 0.3
  - operation: "noise"
    params:
      mode: "column"
      amount: 0.2
  - operation: "compression"
    params:
      quality: 10
```

### The Repeat Parameter

Apply any operation multiple times:

```yaml
- name: "heavy_blur"
  operation: "blur_gaussian"
  repeat: 3           # Apply 3 times
  params:
    sigma: 1.5        # Same parameters each time
```

### Seeded vs Random Noise

```yaml
# Deterministic - same result every run
- operation: "noise"
  params:
    mode: "gaussian"
    amount: 0.1
    seed: 42          # Fixed seed

# Random - different result each run
- operation: "noise"
  params:
    mode: "gaussian"
    amount: 0.1
    # No seed parameter
```

### Parameter Ranges

**For subtle effects:**
- Chromatic aberration: `shift_x: 1-3`, `shift_y: 0-1`
- Noise amount: `0.03-0.08`
- Blur sigma: `0.5-1.5`
- Circular blur radius: `3-5`
- Saturation factor: `0.9-1.2`

**For extreme effects:**
- Chromatic aberration: `shift_x: 8-15`, `shift_y: 4-8`
- Noise amount: `0.3-0.5`
- Blur sigma: `3.0-8.0`
- Circular blur radius: `15-30`
- Saturation factor: `0.3-2.5`

See [docs/FILTER_GUIDE.md](../../FILTER_GUIDE.md) for comprehensive parameter documentation.

---

## Understanding Output Structure

All tutorials follow this non-destructive output structure:

```
tutorials/degradr-effects/
├── vhs-scanlines/
│   ├── intermediate/
│   │   ├── vhs_color_fringing/
│   │   │   └── vhs_color_fringing_*_step00.jpg (45 images)
│   │   ├── vhs_scanlines/
│   │   │   └── vhs_scanlines_*_step01.jpg (45 images)
│   │   └── vhs_softness/
│   │       └── vhs_softness_*_step02.jpg (45 images)
│   └── final/
│       └── vhs_digitization_*_step03.jpg (45 images)
│
├── sensor-noise/
│   ├── intermediate/
│   │   ├── sensor_mosaic/
│   │   └── sensor_grain/
│   └── final/
│       └── sensor_compression_*_step02.jpg (45 images)
│
└── lens-artifacts/
    ├── intermediate/
    │   ├── vintage_bokeh/
    │   └── vintage_fringing/
    └── final/
        └── vintage_color_*_step02.jpg (45 images)
```

**Key points:**
- Each operation step saves to its own directory
- Intermediate steps preserved for artistic exploration
- Final output always in `final/` directory
- Filenames include operation name and step number
- Original frames never modified
- Non-destructive workflow enables iteration

---

## Technical Implementation

### Algorithm Equivalences

| degradr (PyTorch) | sevenrad-stills (NumPy/SciPy) |
|-------------------|-------------------------------|
| `torch.nn.functional.conv2d` | `scipy.ndimage.convolve` |
| `torch.random.normal` | `numpy.random.default_rng().normal` |
| PyTorch tensor ops | NumPy array ops |
| Intel IPP demosaic | `skimage.color.demosaicing_CFA_Bayer_Malvar2004` |

### Why NumPy/SciPy Instead of PyTorch?

- **Smaller footprint**: PyTorch is 500MB+, NumPy/SciPy much lighter
- **No GPU needed**: Single-image processing doesn't benefit from GPU
- **Better PIL integration**: Native NumPy array compatibility
- **Simpler deployment**: No CUDA or platform-specific binaries
- **Identical results**: Same mathematical operations

### Operation Details

**Gaussian Blur** (`blur_gaussian`):
- Uses `scipy.ndimage.gaussian_filter`
- Sigma controls blur strength
- Preserves alpha channel for RGBA

**Circular Blur** (`blur_circular`):
- Generates circular kernel via `np.ogrid`
- Applies via `scipy.ndimage.convolve`
- Simulates lens aperture shape

**Noise** (`noise`):
- Uses `numpy.random.default_rng()` for modern RNG
- Gaussian mode: `rng.normal(loc=0, scale=amount)`
- Row/column modes: broadcast per-row/column values

**Chromatic Aberration** (`chromatic_aberration`):
- Uses `scipy.ndimage.shift` for channel displacement
- Red shifts one direction, blue opposite, green unchanged
- Simulates wavelength-dependent refraction

**Bayer Filter** (`bayer_filter`):
- Creates mosaic using specified pattern (RGGB, BGGR, GRBG, GBRG)
- Demosaics with `skimage.color.demosaicing_CFA_Bayer_Malvar2004`
- Malvar2004 is production-quality algorithm

---

## Troubleshooting

### Operation Not Found

**Problem**: `Operation 'chromatic_aberration' not found`

**Solutions**:
1. Ensure you're on `feature/degradr-effects` branch or later
2. Check operation names match exactly (use underscores)
3. Verify operations registered in `src/sevenrad_stills/operations/__init__.py`
4. Try `sevenrad list-operations` to see available operations

### Effects Too Subtle

**Problem**: Can't see the degradation effects

**Solutions**:
1. Increase parameter values (higher shift, amount, radius)
2. Check intermediate directories to see each step
3. Try extreme values first, then dial back
4. Compare `final/` directly with original frames

### Effects Too Extreme

**Problem**: Image is unrecognizable or over-degraded

**Solutions**:
1. Reduce parameter values
2. Remove some operations from pipeline
3. Try subtle ranges from "Parameter Ranges" section
4. Build up effects gradually, checking intermediate results

### Bayer Filter Creates Unexpected Colors

**Problem**: Strange color patterns or incorrect colors

**Solutions**:
1. Try different Bayer patterns: RGGB, BGGR, GRBG, GBRG
2. This is expected behavior - demosaicing creates artifacts
3. Reduce noise if combining with bayer filter
4. Try with higher quality source images

### Pipeline Runs Slowly

**Problem**: Processing takes too long

**Solutions**:
1. Reduce frame count: increase `interval` (e.g., `interval: 0.1` for 30 frames)
2. Parallel processing is automatic - check CPU usage
3. Circular blur with large radius is expensive - reduce `radius`
4. Gaussian blur with large sigma is expensive - reduce `sigma`

### YAML Syntax Errors

**Problem**: Pipeline fails to parse

**Solutions**:
1. Check indentation - use spaces, not tabs
2. Ensure proper list syntax with `-` for steps
3. Validate YAML with online validator
4. Compare to working examples in this directory

---

## Artistic Context

### Poetic Interpretation

These effects explore the materiality of image capture technologies:

**VHS Scanlines**: The magnetic tape physically degrades through use, creating horizontal artifacts as the tape stretches and the read head misaligns. Each playback is a palimpsest of loss.

**Sensor Noise**: Digital sensors quantize continuous light into discrete measurements. The Bayer mosaic requires computational guesswork (demosaicing) to reconstruct color. "Seeing" is algorithmic construction, not passive recording.

**Lens Artifacts**: Vintage lenses refract light imperfectly, bending different wavelengths at different angles. These "flaws" create unique aesthetic signatures - character, not errors.

### Rimbaud and de Groen

**Rimbaud's Alchemy**: "I is another." The image becomes estranged from itself through successive transformations. Degradation as systematic derangement of the senses.

**de Groen's Critique**: Clinical, exhaustive documentation of decay. Marketing language ("vintage aesthetic," "bokeh effect") commodifies material failure. Late-stage capitalism aestheticizes its own obsolescence.

### Project Goals

These tutorials serve the book's exploration of:
- **Digital vs analog materiality**: How different media degrade
- **Computational seeing**: Algorithms that construct rather than record
- **Poetic data visualization**: Abstraction through systematic transformation
- **Obsolescence aesthetics**: Nostalgia for failed technologies

---

## Next Steps

After completing these tutorials:

1. **Experiment with parameters** - Try extreme and subtle variations
2. **Combine effects creatively** - Layer multiple degradations
3. **Create custom pipelines** - Develop your own aesthetic signatures
4. **Document findings** - Note which combinations work best
5. **Explore other tutorials**:
   - [compression-filters](../compression-filters/) - Multi-generation compression effects
   - [FILTER_GUIDE.md](../../FILTER_GUIDE.md) - Comprehensive parameter documentation
   - [PIPELINE.md](../../PIPELINE.md) - Complete pipeline system guide

## Questions or Issues?

- Check [FILTER_GUIDE.md](../../FILTER_GUIDE.md) for parameter ranges
- Review [PIPELINE.md](../../PIPELINE.md) for pipeline syntax
- Examine existing YAMLs for working examples
- Report issues on project repository

---

**Happy experimenting with degradr-inspired effects!**

*"The image does not degrade - it transforms, revealing the layers of computation that constructed it in the first place."*
