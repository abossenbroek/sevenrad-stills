---
title: Saturation Operation
parent: Operations
nav_order: 3
has_toc: true
---

# Saturation Operation

The `saturation` operation adjusts image color saturation using either fixed values or random variation. Perfect for creating desaturated/muted aesthetics, enhancing vibrant colors, or adding variation across frame sequences.

## Quick Reference

| Mode | Purpose | Key Parameters |
|------|---------|----------------|
| `fixed` | Apply consistent saturation adjustment | value |
| `random` | Apply random saturation within range | range |

---

## Basic Usage

### Fixed Saturation

```yaml
- name: "boost_colors"
  operation: "saturation"
  params:
    mode: "fixed"
    value: 0.5      # 1.5x saturation (50% increase)
```

### Random Saturation

```yaml
- name: "varied_saturation"
  operation: "saturation"
  params:
    mode: "random"
    range: [-0.3, 0.3]    # Random variation ±30%
    seed: 42              # Optional: for reproducibility
```

---

## Parameters

### Mode: Fixed

Apply a consistent saturation multiplier to all frames.

**Parameters:**
- **mode** (str): Must be `"fixed"`
- **value** (float): Saturation adjustment value
  - Actual saturation factor = `1.0 + value`
  - Range: `-1.0` to `∞` (negative values desaturate, positive values boost)

**Value Ranges:**

| Value | Actual Factor | Effect |
|-------|---------------|--------|
| `-1.0` | `0.0` | Complete desaturation (grayscale) |
| `-0.8` | `0.2` | Heavy desaturation |
| `-0.5` | `0.5` | Moderate desaturation |
| `-0.2` | `0.8` | Subtle desaturation |
| `0.0` | `1.0` | No change (original) |
| `0.3` | `1.3` | Subtle boost |
| `0.5` | `1.5` | Moderate boost |
| `1.0` | `2.0` | Heavy boost |
| `2.0` | `3.0` | Extreme boost |

### Mode: Random

Apply random saturation variation within a specified range. Useful for adding natural variation across frame sequences.

**Parameters:**
- **mode** (str): Must be `"random"`
- **range** (list[float, float]): `[min_value, max_value]`
  - Both values follow same scale as fixed mode
  - `min_value` must be ≥ `-1.0`
  - `max_value` must be > `min_value`
- **seed** (int, optional): Random seed for reproducibility

**Example Ranges:**

| Range | Effect |
|-------|--------|
| `[-1.0, -0.5]` | Random desaturation (grayscale to moderate) |
| `[-0.3, 0.3]` | Subtle random variation around original |
| `[0.0, 0.5]` | Random boost (original to moderate) |
| `[0.5, 2.0]` | Heavy random boost |

---

## Examples

### Complete Desaturation (Grayscale)

```yaml
- name: "grayscale"
  operation: "saturation"
  params:
    mode: "fixed"
    value: -1.0     # Factor = 0.0 → complete grayscale
```

**Visual effect:** Full black and white conversion

### Heavy Desaturation (Muted Colors)

```yaml
- name: "muted_aesthetic"
  operation: "saturation"
  params:
    mode: "fixed"
    value: -0.6     # Factor = 0.4 → heavily muted
```

**Visual effect:** Washed-out, muted color palette. Perfect for documentary or melancholic aesthetic.

### Subtle Desaturation

```yaml
- name: "subtle_muted"
  operation: "saturation"
  params:
    mode: "fixed"
    value: -0.2     # Factor = 0.8 → slightly muted
```

**Visual effect:** Slight reduction in vibrancy, natural-looking desaturation.

### Moderate Color Boost

```yaml
- name: "vivid_colors"
  operation: "saturation"
  params:
    mode: "fixed"
    value: 0.5      # Factor = 1.5 → 50% more saturated
```

**Visual effect:** Vibrant, punchy colors without extreme oversaturation.

### Heavy Color Boost

```yaml
- name: "hyper_saturated"
  operation: "saturation"
  params:
    mode: "fixed"
    value: 1.5      # Factor = 2.5 → 150% more saturated
```

**Visual effect:** Extreme color intensity, surreal appearance.

### Random Subtle Variation

```yaml
- name: "natural_variation"
  operation: "saturation"
  params:
    mode: "random"
    range: [-0.15, 0.15]    # ±15% variation
    seed: 100
```

**Visual effect:** Each frame has slightly different saturation, mimicking natural lighting variation.

### Random Heavy Variation

```yaml
- name: "dramatic_variation"
  operation: "saturation"
  params:
    mode: "random"
    range: [-0.8, 1.0]      # Wide variation
    seed: 200
```

**Visual effect:** Some frames nearly grayscale, others highly saturated - dramatic contrast.

---

## Parameter Visualization

### Desaturation Scale

```
value: -1.0  →  Grayscale
value: -0.8  →  ░░░░░░░░░░ Heavy muted
value: -0.5  →  ░░░░░█████ Moderate muted
value: -0.2  →  ░░████████ Subtle muted
value:  0.0  →  ██████████ Original
```

### Saturation Boost Scale

```
value:  0.0  →  ██████████ Original
value:  0.3  →  ███████████░ Subtle boost
value:  0.5  →  ████████████░░ Moderate boost
value:  1.0  →  ██████████████░░░░ Heavy boost
value:  2.0  →  ████████████████████ Extreme boost
```

---

## Combining with Other Operations

### Desaturated VHS Aesthetic

```yaml
steps:
  - name: "desaturate"
    operation: "saturation"
    params:
      mode: "fixed"
      value: -0.4           # Muted colors

  - name: "scanlines"
    operation: "noise"
    params:
      mode: "row"
      amount: 0.08

  - name: "compress"
    operation: "compression"
    params:
      quality: 35
      subsampling: 2
```

**Effect:** Muted, washed-out VHS tape aesthetic.

### Hyper-Saturated Glitch Art

```yaml
steps:
  - name: "boost_colors"
    operation: "saturation"
    params:
      mode: "fixed"
      value: 1.2            # Heavy saturation boost

  - name: "pixelate"
    operation: "downscale"
    params:
      scale: 0.1
      upscale: true
      upscale_method: "nearest"

  - name: "compress"
    operation: "compression"
    params:
      quality: 5
      subsampling: 2
```

**Effect:** Extreme colors combined with heavy pixelation and compression artifacts.

### Vintage Film Look

```yaml
steps:
  - name: "subtle_desaturate"
    operation: "saturation"
    params:
      mode: "fixed"
      value: -0.3           # Slightly muted

  - name: "film_grain"
    operation: "noise"
    params:
      mode: "gaussian"
      amount: 0.04

  - name: "soft_glow"
    operation: "blur_gaussian"
    params:
      sigma: 0.8
```

**Effect:** Muted colors with grain and slight softness, reminiscent of vintage film.

### Random Saturation Sequence

```yaml
steps:
  - name: "varied_saturation"
    operation: "saturation"
    params:
      mode: "random"
      range: [-0.5, 0.8]    # Wide variation
      seed: 42

  - name: "chromatic_shift"
    operation: "chromatic_aberration"
    params:
      shift_x: 3
      shift_y: 0
```

**Effect:** Each frame has different saturation levels combined with color fringing.

---

## Use Cases

### Documentary/Cinematic Muted Aesthetic

**Goal:** Create muted, desaturated look for serious/contemplative mood.

```yaml
- name: "documentary_muted"
  operation: "saturation"
  params:
    mode: "fixed"
    value: -0.4           # 0.6x saturation
```

**Best for:** Documentary work, serious subjects, melancholic mood.

### Commercial/Advertising Vibrant Colors

**Goal:** Punchy, eye-catching colors.

```yaml
- name: "commercial_vibrant"
  operation: "saturation"
  params:
    mode: "fixed"
    value: 0.4            # 1.4x saturation
```

**Best for:** Marketing materials, product photography, attention-grabbing visuals.

### Artistic Grayscale

**Goal:** Full black and white conversion.

```yaml
- name: "black_and_white"
  operation: "saturation"
  params:
    mode: "fixed"
    value: -1.0           # 0.0x saturation (complete grayscale)
```

**Best for:** Artistic photography, noir aesthetic, focus on form and composition.

### Natural Variation Across Frames

**Goal:** Mimic natural lighting/exposure variation in video sequences.

```yaml
- name: "natural_variation"
  operation: "saturation"
  params:
    mode: "random"
    range: [-0.1, 0.1]    # Subtle ±10% variation
    seed: 12345
```

**Best for:** Creating organic feel in still sequences, avoiding mechanical uniformity.

---

## Technical Details

### Implementation

- Uses PIL's `ImageEnhance.Color` for saturation adjustment
- Saturation factor calculation: `factor = max(0.0, 1.0 + value)`
- Factor of `0.0` produces complete grayscale
- Factor of `1.0` preserves original saturation
- Factors > `1.0` boost saturation proportionally

### Random Mode Behavior

- Uses Python's `random.uniform()` for value generation
- Optional `seed` parameter ensures reproducibility
- Each frame gets independent random value within range
- Distribution is uniform (all values equally likely)

### Performance

- Very fast operation (PIL-native enhancement)
- Minimal memory overhead
- Suitable for batch processing large frame sequences

---

## Best Practices

1. **Start subtle**: Begin with values between `-0.3` and `0.3`
2. **Consider color relationships**: Heavy desaturation can flatten contrast
3. **Test with representative frames**: Saturation effects vary by image content
4. **Combine thoughtfully**: Saturation + compression can create unexpected color shifts
5. **Use random mode sparingly**: Too much variation can feel chaotic

---

## Common Mistakes

### Mistake 1: Using negative factors directly

❌ **Wrong:**
```yaml
params:
  mode: "fixed"
  value: -2.0    # Results in factor = -1.0, clamped to 0.0
```

✅ **Correct:**
```yaml
params:
  mode: "fixed"
  value: -1.0    # Minimum valid value (complete grayscale)
```

### Mistake 2: Invalid range order

❌ **Wrong:**
```yaml
params:
  mode: "random"
  range: [0.5, -0.5]    # max < min (invalid)
```

✅ **Correct:**
```yaml
params:
  mode: "random"
  range: [-0.5, 0.5]    # min < max
```

### Mistake 3: Forgetting range is relative

❌ **Wrong assumption:**
```yaml
# Thinking range: [0.5, 1.5] means "50% to 150% saturation"
```

✅ **Correct understanding:**
```yaml
params:
  mode: "random"
  range: [-0.5, 0.5]    # Actual factors: 0.5x to 1.5x saturation
```

---

## Next Steps

- **Try hands-on examples**: Experiment with different value/range combinations
- **Explore combinations**: [Compression Operations](compression/) | [Degradr Effects](degradr/)
- **Learn YAML system**: [Pipeline Documentation](../reference/pipeline/)

---

For comprehensive parameter guidance and visual examples across all operations, see the [Filter Guide](../reference/filter-guide/).
