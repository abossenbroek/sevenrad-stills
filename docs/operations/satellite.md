---
title: Satellite Malfunction Operations
parent: Operations
nav_order: 4
has_toc: true
---

# Satellite Malfunction Operations

Six operations that simulate realistic satellite sensor and software failures based on documented remote sensing malfunctions. These operations recreate authentic artifacts from cosmic ray hits, detector calibration drift, transmission errors, and hardware failures.

## Quick Reference

| Operation | Category | Primary Use | Key Parameters |
|-----------|----------|-------------|----------------|
| `salt_pepper` | Sensor Hardware | Cosmic ray hits on sensors | amount, salt_vs_pepper |
| `corduroy` | Sensor Hardware | Detector calibration drift | strength, orientation, density |
| `band_swap` | Software/Transmission | Packet header corruption | tile_count, permutation |
| `buffer_corruption` | Software/Transmission | Bitwise memory corruption | tile_count, corruption_type, severity |
| `compression_artifact` | Software/Transmission | On-board encoder failures | tile_count, quality |
| `slc_off` | Hardware Failure | Landsat 7 SLC failure | gap_width, scan_period, fill_mode |

---

## salt_pepper Operation

Simulates cosmic ray hits on individual sensor pixels, creating random white (salt) or black (pepper) pixels across the image. Mimics Single Event Effects (SEE) from high-energy particle impacts in space.

### Quick Example

```yaml
- name: "cosmic_ray_hits"
  operation: "salt_pepper"
  params:
    amount: 0.001          # 0.1% of pixels affected
    salt_vs_pepper: 0.5    # Equal salt and pepper
    seed: 42
```

### Parameters

**amount** (float, required): Fraction of pixels to corrupt (0.0 to 1.0)
- `0.0001` - Very sparse (1 in 10,000 pixels) - subtle cosmic ray simulation
- `0.001` - Sparse (1 in 1,000 pixels) - moderate cosmic ray activity
- `0.005` - Moderate (1 in 200 pixels) - high radiation environment
- `0.01` - Dense (1 in 100 pixels) - extreme radiation event
- `0.05` - Very dense (5% pixels) - sensor failure cascade
- `0.1` - Catastrophic failure (10% pixels) - realistic maximum
- `0.5` - Extreme degradation (50% pixels) - artistic effect
- `1.0` - Complete noise (100% pixels) - maximum allowed

**salt_vs_pepper** (float, required): Ratio of white to black pixels (0.0 to 1.0)
- `0.0` - All pepper (pure black pixels)
- `0.25` - 25% salt, 75% pepper
- `0.5` - Equal salt and pepper (most realistic)
- `0.75` - 75% salt, 25% pepper
- `1.0` - All salt (pure white pixels)

**seed** (int, optional): Random seed for reproducibility

### Technical Behavior

- Entire RGB pixels affected (not per-channel)
- Alpha channel preserved in RGBA mode
- Grayscale mode supported (single channel corruption)
- Uniform random distribution across image

### Use Cases

**Solar Storm Simulation**
```yaml
params:
  amount: 0.005
  salt_vs_pepper: 0.5
  seed: 100
```
Simulates elevated cosmic ray activity during solar particle events.

**Sensor Dead Pixels**
```yaml
params:
  amount: 0.0001
  salt_vs_pepper: 0.0   # All black (dead pixels)
```
Mimics stuck-at-zero detector elements.

**Hot Pixels**
```yaml
params:
  amount: 0.0002
  salt_vs_pepper: 1.0   # All white (hot pixels)
```
Simulates stuck-at-maximum detector elements.

---

## corduroy Operation

Simulates "corduroy" or "banding" artifacts from push-broom and whisk-broom scanners where individual detector elements have slightly different sensitivity due to calibration drift or manufacturing variations.

### Quick Example

```yaml
- name: "detector_striping"
  operation: "corduroy"
  params:
    strength: 0.3          # Moderate striping
    orientation: "vertical"
    density: 0.15          # 15% of lines affected
    seed: 42
```

### Parameters

**strength** (float, required): Striping intensity (0.0 to 1.0)
- Maps to multiplier range: `[1.0 - strength×0.2, 1.0 + strength×0.2]`
- `0.0` - No striping
- `0.2` - Subtle (±4% brightness variation)
- `0.5` - Moderate (±10% brightness variation)
- `0.8` - Strong (±16% brightness variation)
- `1.0` - Maximum (±20% brightness variation)

**orientation** (str, required): Line direction
- `"vertical"` - Vertical lines (typical for push-broom scanners)
- `"horizontal"` - Horizontal lines (typical for whisk-broom scanners)

**density** (float, required): Proportion of lines affected (0.0 to 1.0)
- `0.05` - 5% of lines (sparse detector issues)
- `0.1` - 10% of lines (moderate calibration drift)
- `0.2` - 20% of lines (significant calibration problems)
- `0.5` - 50% of lines (severe detector array degradation)
- `1.0` - All lines (complete calibration failure)

**seed** (int, optional): Random seed for reproducibility

### Technical Behavior

- Affects entire columns (vertical) or rows (horizontal)
- Each line gets independent random multiplier
- Multipliers sampled uniformly within strength range
- Values clipped to [0.0, 1.0] after multiplication

### Use Cases

**Early Mission Calibration**
```yaml
params:
  strength: 0.1
  orientation: "vertical"
  density: 0.05
```
Simulates minor calibration drift in new sensors.

**Long-Duration Mission Degradation**
```yaml
params:
  strength: 0.6
  orientation: "vertical"
  density: 0.25
```
Simulates significant detector aging after years in orbit.

**Thermal Stress Striping**
```yaml
params:
  strength: 0.4
  orientation: "horizontal"
  density: 0.15
  seed: 200
```
Simulates temperature-induced sensitivity variations.

---

## band_swap Operation

Simulates packet header corruption in satellite downlink where spectral band data is received correctly but misinterpreted due to corrupted metadata, creating rectangular regions with swapped color channels.

### Quick Example

```yaml
- name: "packet_corruption"
  operation: "band_swap"
  params:
    tile_count: 5
    permutation: "GRB"
    tile_size_range: [0.05, 0.2]
    seed: 42
```

### Parameters

**tile_count** (int, required): Number of corrupted tiles (1 to 50)
- `1-5` - Isolated errors (single packet corruption)
- `10-20` - Moderate errors (multiple packet loss)
- `30-50` - Severe errors (widespread transmission issues)

**permutation** (str, required): Channel swap pattern
- `"GRB"` - Green→Red, Red→Green, Blue→Blue
- `"BGR"` - Blue→Red, Green→Green, Red→Blue (full reversal)
- `"BRG"` - Blue→Red, Red→Green, Green→Blue (rotate right)
- `"GBR"` - Green→Red, Blue→Green, Red→Blue (rotate left)
- `"RBG"` - Red→Red, Blue→Green, Green→Blue (swap G/B only)

**tile_size_range** (list[float, float], optional): Tile size as fractions [min, max]
- Default: `[0.05, 0.2]` (5% to 20% of image dimensions)
- `[0.01, 0.05]` - Small tiles (localized corruption)
- `[0.1, 0.3]` - Medium tiles (packet-level corruption)
- `[0.2, 0.5]` - Large tiles (scene-level corruption)

**seed** (int, optional): Random seed for reproducibility

### Technical Behavior

- Only works on RGB/RGBA images (raises error for grayscale)
- Alpha channel preserved in RGBA mode
- Tiles placed randomly with uniform distribution
- Permutation applied instantly (not gradual)

### Use Cases

**Single Packet Corruption**
```yaml
params:
  tile_count: 3
  permutation: "BGR"
  tile_size_range: [0.1, 0.15]
```
Simulates isolated header errors in downlink stream.

**Widespread Transmission Issues**
```yaml
params:
  tile_count: 30
  permutation: "GRB"
  tile_size_range: [0.05, 0.2]
  seed: 100
```
Simulates severe communication link problems.

---

## buffer_corruption Operation

Simulates Single Event Upsets (SEUs) from cosmic rays flipping bits in on-board image buffer memory, creating localized rectangular "glitch blocks" with bitwise corruptions.

### Quick Example

```yaml
- name: "memory_corruption"
  operation: "buffer_corruption"
  params:
    tile_count: 5
    corruption_type: "xor"
    severity: 0.5
    tile_size_range: [0.05, 0.2]
    seed: 42
```

### Parameters

**tile_count** (int, required): Number of corrupted memory tiles (1 to 20)
- `1-3` - Isolated memory hits
- `5-10` - Moderate radiation exposure
- `15-20` - Severe cosmic ray event

**corruption_type** (str, required): Type of bitwise corruption
- `"xor"` - Bitwise XOR with random pattern (simulates bit flips)
- `"invert"` - Bitwise inversion (simulates register corruption)
- `"channel_shuffle"` - Random RGB permutation per tile (simulates pointer corruption)

**severity** (float, required): Corruption intensity (0.0 to 1.0)
- For `xor`: Controls magnitude of XOR mask (0-255 × severity)
- For `invert`: Controls blend between original and inverted (0% to 100%)
- For `channel_shuffle`: Probability of shuffling per tile (0% to 100%)
- `0.0` - No corruption
- `0.3` - Subtle artifacts
- `0.5` - Moderate corruption
- `0.8` - Severe corruption
- `1.0` - Maximum corruption

**tile_size_range** (list[float, float], optional): Tile size as fractions [min, max]
- Default: `[0.05, 0.2]`

**seed** (int, optional): Random seed for reproducibility

### Technical Behavior

- Works on RGB, RGBA, and grayscale modes
- Alpha channel preserved in RGBA
- XOR and invert work per-pixel
- Channel shuffle only effective on RGB/RGBA

### Use Cases

**Cosmic Ray SEU Event**
```yaml
params:
  tile_count: 3
  corruption_type: "xor"
  severity: 0.4
```
Simulates typical single-event upset from particle impact.

**Register Corruption**
```yaml
params:
  tile_count: 1
  corruption_type: "invert"
  severity: 1.0
  tile_size_range: [0.1, 0.1]
```
Simulates complete inversion in small memory region.

**Pointer Corruption**
```yaml
params:
  tile_count: 5
  corruption_type: "channel_shuffle"
  severity: 1.0
```
Simulates address pointer errors in multi-planar buffers.

---

## compression_artifact Operation

Simulates on-board JPEG encoder failures where specific memory regions get over-compressed due to rate control errors, buffer overflow, or hardware malfunctions.

### Quick Example

```yaml
- name: "encoder_failure"
  operation: "compression_artifact"
  params:
    tile_count: 5
    quality: 5             # Very low quality
    tile_size_range: [0.1, 0.25]
    seed: 42
```

### Parameters

**tile_count** (int, required): Number of corrupted encoder tiles (1 to 30)
- `1-5` - Isolated encoder errors
- `10-20` - Moderate encoder instability
- `25-30` - Severe encoder failure

**quality** (int, required): JPEG quality for corrupted tiles (1 to 20)
- Note: Range 1-20 (not standard 1-100) to force visible artifacts
- `1-5` - Extreme blocking and color bleeding
- `6-10` - Severe artifacts
- `11-15` - Moderate artifacts
- `16-20` - Subtle artifacts

**tile_size_range** (list[float, float], optional): Tile size as fractions [min, max]
- Default: `[0.05, 0.2]`

**seed** (int, optional): Random seed for reproducibility

### Technical Behavior

- Uses in-memory JPEG compression via io.BytesIO
- Each tile compressed independently
- Creates sharp boundaries between pristine and degraded regions
- Grayscale converted to RGB for compression, then back

### Use Cases

**Rate Control Failure**
```yaml
params:
  tile_count: 3
  quality: 8
  tile_size_range: [0.15, 0.25]
```
Simulates encoder running out of bandwidth mid-scene.

**Buffer Overflow**
```yaml
params:
  tile_count: 10
  quality: 3
  tile_size_range: [0.05, 0.15]
  seed: 100
```
Simulates memory constraints forcing quality reduction.

**Thermal Throttling**
```yaml
params:
  tile_count: 15
  quality: 10
  tile_size_range: [0.1, 0.2]
```
Simulates heat-induced encoder performance degradation.

---

## slc_off Operation

Simulates the May 31, 2003 Landsat 7 Scan Line Corrector (SLC) failure, creating characteristic wedge-shaped data gaps that widen from center to edges.

### Quick Example

```yaml
- name: "landsat_slc_off"
  operation: "slc_off"
  params:
    gap_width: 0.22        # Historical 22% maximum gap
    scan_period: 14        # Typical scan line spacing
    fill_mode: "mean"
    seed: 42
```

### Parameters

**gap_width** (float, required): Maximum gap width at edges (0.0 to 0.5)
- Fraction of image width
- Gaps increase linearly from center (0%) to edges (gap_width × 100%)
- `0.0` - No gaps (SLC functioning)
- `0.1` - 10% gaps at edges (minor SLC degradation)
- `0.22` - 22% gaps (historical Landsat 7 SLC-Off maximum)
- `0.35` - 35% gaps (severe failure)
- `0.5` - 50% gaps (catastrophic failure)

**scan_period** (int, required): Scan line spacing in rows (2 to 100)
- Gap frequency (every Nth row)
- `2-5` - High frequency (very visible striping)
- `10-15` - Moderate frequency (Landsat 7-like)
- `20-40` - Low frequency (sparse gaps)
- `50-100` - Very sparse (isolated gaps)

**fill_mode** (str, required): Gap fill strategy
- `"black"` - Fill with pure black (0, 0, 0)
- `"white"` - Fill with pure white (255, 255, 255)
- `"mean"` - Fill with row mean + small random variation (most realistic)

**seed** (int, optional): Random seed (used for mean fill variation only)

### Technical Behavior

- Creates diagonal wedge-shaped gaps in zig-zag pattern (wider at edges, none at center)
- Gaps alternate direction (left/right) on each scan line
- Diagonal offset simulates uncompensated forward satellite motion
- Gaps are geometric (deterministic except for mean fill variation)
- Works on RGB, RGBA, and grayscale
- Alpha channel preserved in RGBA

### Technical Details

The failure pattern accurately recreates Landsat 7 ETM+ SLC-Off geometry:
1. Calculate distance from image center (normalized 0..1)
2. Gap width = distance × gap_width × image_width
3. Apply diagonal gaps periodically (every scan_period rows)
4. Each scan line alternates direction (zig-zag pattern)
5. Diagonal offset of 0.3 pixels/row creates shallow angle typical of real failure
6. Gaps span multiple rows matching scan_period duration

### Use Cases

**Historical Landsat 7 SLC-Off**
```yaml
params:
  gap_width: 0.22
  scan_period: 14
  fill_mode: "black"
```
Accurately recreates May 31, 2003 failure characteristics.

**SLC Partial Degradation**
```yaml
params:
  gap_width: 0.10
  scan_period: 20
  fill_mode: "mean"
  seed: 100
```
Simulates early-stage SLC malfunction before complete failure.

**Catastrophic SLC Failure**
```yaml
params:
  gap_width: 0.4
  scan_period: 8
  fill_mode: "black"
```
Simulates severe geometric distortion beyond historical levels.

---

## Combining Operations

Satellite malfunctions often occur in combination. Realistic scenarios:

### Solar Storm Event

```yaml
steps:
  # Cosmic rays hit sensor
  - name: "particle_hits"
    operation: "salt_pepper"
    params:
      amount: 0.003
      salt_vs_pepper: 0.5

  # Radiation causes memory corruption
  - name: "memory_seu"
    operation: "buffer_corruption"
    params:
      tile_count: 5
      corruption_type: "xor"
      severity: 0.4
```

### Long-Duration Mission Degradation

```yaml
steps:
  # Detector calibration drift
  - name: "sensor_aging"
    operation: "corduroy"
    params:
      strength: 0.5
      orientation: "vertical"
      density: 0.2

  # Encoder wear
  - name: "compression_degradation"
    operation: "compression_artifact"
    params:
      tile_count: 8
      quality: 12
```

### Severe Transmission Failure

```yaml
steps:
  # Packet corruption
  - name: "band_misidentification"
    operation: "band_swap"
    params:
      tile_count: 15
      permutation: "BGR"

  # Encoder failure
  - name: "encoder_errors"
    operation: "compression_artifact"
    params:
      tile_count: 20
      quality: 5
```

### Landsat 7 Post-2003

```yaml
steps:
  # SLC failure
  - name: "slc_off_gaps"
    operation: "slc_off"
    params:
      gap_width: 0.22
      scan_period: 14
      fill_mode: "mean"

  # Continuing detector drift
  - name: "detector_striping"
    operation: "corduroy"
    params:
      strength: 0.3
      orientation: "vertical"
      density: 0.1
```

---

## Technical Implementation Notes

### Performance

All operations are optimized for batch processing:
- salt_pepper: O(n) where n = affected pixels
- corduroy: O(h×w) with NumPy broadcasting
- band_swap: O(tile_count × tile_area) with NumPy slicing
- buffer_corruption: O(tile_count × tile_area) with bitwise ops
- compression_artifact: O(tile_count × JPEG_encode_time)
- slc_off: O(h×w) with geometric calculation

### Color Space Handling

- RGB: All operations supported
- RGBA: Alpha channel always preserved
- Grayscale: All operations except band_swap (raises error)

### Reproducibility

All operations support `seed` parameter for deterministic output:
- salt_pepper: Pixel selection and salt/pepper choice
- corduroy: Line selection and multiplier values
- band_swap: Tile positions
- buffer_corruption: Tile positions and XOR masks/permutations
- compression_artifact: Tile positions
- slc_off: Mean fill variation only (geometry is deterministic)

---

## Best Practices

1. **Start subtle**: Use low amounts/densities to understand effects
2. **Match failure modes**: Choose operations that match your aesthetic goal
3. **Test on representative frames**: Effects vary by image content
4. **Use seeds for sequences**: Ensures consistent corruption across frames
5. **Combine realistically**: Study actual satellite failures for inspiration

---

## Common Mistakes

### Mistake 1: Using band_swap on grayscale

**Wrong:**
```yaml
# Grayscale image
- operation: "band_swap"
```

**Correct:**
```yaml
# RGB or RGBA image required
# Use buffer_corruption with channel_shuffle instead for grayscale
- operation: "buffer_corruption"
  params:
    corruption_type: "xor"  # Works on grayscale
```

### Mistake 2: Unrealistic salt_pepper for scientific simulation

**Not realistic for satellite simulation:**
```yaml
params:
  amount: 0.5    # 50% of pixels - artistic effect, not realistic
```

**Realistic for satellite simulation:**
```yaml
params:
  amount: 0.001  # 0.1% of pixels - typical cosmic ray rate
```

**Note:** High values (0.5-1.0) are valid for artistic/glitch effects, but don't represent real satellite failures.

### Mistake 3: Forgetting seeds for sequences

**Wrong:**
```yaml
# No seed - different corruption each frame
- operation: "buffer_corruption"
  params:
    tile_count: 5
```

**Correct:**
```yaml
# Same corruption pattern across sequence
- operation: "buffer_corruption"
  params:
    tile_count: 5
    seed: 42
```

---

## Next Steps

- **Try hands-on examples**: [Satellite Malfunctions Tutorial](../tutorials/satellite-malfunctions/)
- **Understand the science**: [Technical Explanation](../technical_notes/satellite-failures/)
- **Explore combinations**: [Filter Guide](../reference/filter-guide/)

---

For comprehensive technical background on these failure modes, see [Satellite Sensing Failures Explained](../technical_notes/satellite-failures/).
