---
title: Satellite Sensing Failures
parent: Technical Notes
nav_order: 2
has_toc: true
---

# Satellite Remote Sensing Failures: Technical Background

This document explains the real-world satellite sensor and software failures simulated by the satellite malfunction operations. Understanding these failure modes provides context for using the operations effectively and appreciating their basis in actual remote sensing challenges.

## Overview

Satellite remote sensing systems face unique environmental and operational challenges that lead to characteristic image artifacts. These failures fall into three categories:

1. **Sensor Hardware Failures** - Physical detector issues (cosmic rays, calibration drift)
2. **Software/Transmission Failures** - Data corruption during processing and downlink
3. **Mechanical Hardware Failures** - Physical component failures (scan mechanisms)

---

## Sensor Hardware Failures

### Salt & Pepper Noise: Cosmic Ray Hits

**Physical Mechanism**

In the space environment, high-energy particles (cosmic rays, solar protons) constantly bombard satellite electronics. When these particles strike a sensor's photodetector array:

1. **Direct ionization** - Particle passes through silicon, creating electron-hole pairs
2. **Charge collection** - These charges are misinterpreted as signal
3. **Single Event Effect (SEE)** - One particle hit → one corrupted pixel

**Characteristics**

- **Random distribution** - No spatial correlation (cosmic rays are isotropic)
- **Bimodal response** - Pixel reads either maximum (255) or minimum (0) value
- **Transient** - Only affects single frame, next frame is clean
- **Frequency** - Depends on orbital altitude and solar activity
  - Low Earth Orbit (LEO): ~0.01-0.1% pixels per frame during quiet periods
  - Solar storms: Can increase by 10-100× for hours or days
  - South Atlantic Anomaly: 5-10× higher rates

**Real-World Examples**

- Hubble Space Telescope: Visible cosmic ray hits in long exposures
- Mars rovers: Random bright pixels in Mastcam images
- Landsat: Occasional white/black pixel speckles in thermal bands

**Why Both Salt (White) and Pepper (Black)?**

The polarity depends on:
- Particle energy and angle
- Depth of penetration into detector
- Whether charge accumulates or depletes
Result: approximately equal probability of saturation high or low

### Corduroy Striping: Detector Calibration Drift

**Physical Mechanism**

Multispectral satellites use linear detector arrays (push-broom) or 2D arrays (whisk-broom) to build up images by scanning:

1. **Manufacturing variations** - Each detector element has slightly different sensitivity
2. **Initial calibration** - Ground calibration compensates for these differences
3. **Orbital degradation** - Over time, detectors age differently:
   - Radiation damage accumulates non-uniformly
   - Thermal cycling causes mechanical stress
   - Contamination affects some detectors more than others
4. **Calibration drift** - Original compensation no longer accurate

**Characteristics**

- **Stripe pattern** - Repeating lines perpendicular to scan direction
  - Push-broom: Vertical stripes (satellite moves forward, scans across-track)
  - Whisk-broom: Horizontal or diagonal stripes
- **Persistent** - Same pattern appears in every image
- **Brightness variation** - Typically ±5-20% brightness
- **Worsens over time** - Years 1-2: subtle, Years 5+: significant

**Mathematical Model**

If detector i has gain drift Δgᵢ:
```
I_observed(i) = I_true × (1 + Δgᵢ)
```

Where Δgᵢ varies slowly over detector array (correlated with manufacturing lot, position in array, thermal gradients).

**Real-World Examples**

- **Landsat 4/5 Thematic Mapper** - Visible vertical striping in bands 5 & 7 after 10+ years
- **MODIS Terra** - "Bowtie" striping visible in some bands, especially at scan edges
- **Aqua MODIS** - Detector 6 degradation created visible stripe in Band 6

**Correction Methods**

Ground processing applies:
1. **Periodic re-calibration** - Update gain/offset coefficients
2. **Destriping algorithms** - Histogram matching or Fourier filtering
3. **Replacement with backup detectors** - If available

---

## Software/Transmission Failures

### Band Swap: Packet Header Corruption

**Physical Mechanism**

Satellite downlink transmits image data as packets with metadata headers:

```
[Packet Header: Band ID, Position] [Pixel Data: Raw Values]
```

Bit flips in packet headers (from cosmic rays, transmission noise) can corrupt metadata while leaving data intact:

1. **Header corruption** - Band identifier bits flip (e.g., Band 1 → Band 3)
2. **Data received correctly** - Pixel values transmitted without error
3. **Misinterpretation** - Ground station assigns data to wrong spectral band
4. **Result** - Red channel data interpreted as Blue, etc.

**Characteristics**

- **Rectangular regions** - Size corresponds to packet length
- **Sharp boundaries** - Instant transition between correct and swapped regions
- **Spatial structure preserved** - Shapes and textures intact, only colors wrong
- **Deterministic per tile** - Same swap pattern within each corrupted packet

**Why Rectangular Tiles?**

Satellite data is packetized in fixed-size tiles for efficient transmission:
- Typical size: 256×256 to 512×512 pixels
- Each tile has independent header
- Header corruption affects entire tile uniformly

**Real-World Examples**

- **NOAA AVHRR** - Occasional band swaps in real-time transmission
- **MSG SEVIRI** - Rare channel mis-identification in rapid scan mode
- **Suomi-NPP VIIRS** - Documented cases of incorrect band labeling in early mission

**Probability**

Given cosmic ray SEU rate ~10⁻¹⁰ bits/second:
- Packet header: ~100 bits
- Data downlink rate: 100 Mbps
- Expected header corruption: ~1 packet per 10⁵ transmitted

### Buffer Corruption: Single Event Upsets in Memory

**Physical Mechanism**

On-board image buffers (DRAM/SRAM) hold raw image data before compression and transmission. Cosmic ray hits can flip bits in memory:

1. **Particle strike** - High-energy proton/heavy ion hits memory cell
2. **Charge deposition** - Creates electron-hole pairs
3. **Bit flip** - If charge exceeds threshold, bit flips (0→1 or 1→0)
4. **Data corruption** - Pixel values corrupted before encoding

**Types of Corruption**

**XOR Corruption** (Most common):
```
Original pixel: 10110011 (179)
XOR mask:       00001000 (bit 3 flipped)
Corrupted:      10111011 (187)
```

**Inversion** (Less common):
```
Original pixel: 10110011 (179)
Inverted:       01001100 (76)
```

**Pointer Corruption** (Multi-planar buffers):
```
RGB stored as: R[1024×1024], G[1024×1024], B[1024×1024]
If pointer corrupted: R data read as B, G data read as R, etc.
```

**Characteristics**

- **Localized regions** - Typically affect memory pages (4-64KB blocks)
- **Pixel-level corruption** - Individual values altered
- **Bitwise patterns** - Systematic bit flips create specific visual signatures
- **Varies by memory type**:
  - DRAM: Large regions (row/column corruption)
  - SRAM: Small regions (individual cells)

**Real-World Examples**

- **Mars rover cameras** - Occasional "glitch blocks" in Navcam/Hazcam images
- **Cassini ISS** - Random corruption in raw image buffers
- **Hubble WFC3** - Memory bit flips in electronics

**SEU Rates by Orbit**

- Low Earth Orbit (400-800 km): 10⁻⁶ - 10⁻⁵ upsets/bit/day
- Geostationary (36,000 km): 10⁻⁷ - 10⁻⁶ upsets/bit/day
- During solar storms: 100× - 1000× higher

### Compression Artifacts: On-board Encoder Failures

**Physical Mechanism**

Satellites use on-board JPEG/JPEG2000 compression to reduce data volume. Encoder failures create localized artifacts:

**Rate Control Failures**:
1. **Target bitrate** - Encoder must fit scene into fixed bandwidth
2. **Complexity variation** - Some image regions harder to compress
3. **Buffer overflow** - When complex region exceeds budget
4. **Emergency quality reduction** - Encoder drops quality mid-scene

**Hardware Malfunctions**:
1. **DCT engine errors** - Discrete Cosine Transform unit malfunction
2. **Quantization table corruption** - Encoding parameters corrupted
3. **Thermal throttling** - Overheating reduces clock speed → reduced quality

**Characteristics**

- **Tile-based** - JPEG compresses in 8×8 or 16×16 blocks
- **Variable quality** - Different regions at different compression levels
- **8×8 blocking** - Visible DCT block boundaries
- **Color bleeding** - Chroma subsampling artifacts

**Real-World Examples**

- **Landsat 7 ETM+** - Early mission compression artifacts from encoder bugs
- **Sentinel-2 MSI** - Rare JPEG2000 artifacts in L1C products
- **Commercial satellites** - Quality variations in rapid-revisit systems

**Compression Ratios**

- Target: 10:1 to 20:1 (high quality)
- Failure mode: 50:1 to 100:1 (severe artifacts)

---

## Mechanical Hardware Failures

### SLC-Off: Landsat 7 Scan Line Corrector Failure

**Physical Mechanism**

The Scan Line Corrector (SLC) was a mechanical mirror system on Landsat 7's Enhanced Thematic Mapper Plus (ETM+):

**Normal Operation** (Before May 31, 2003):
1. **Whisk-broom scanner** - Oscillating mirror scans perpendicular to flight
2. **Forward motion** - Satellite moves ~7 km/s
3. **SLC compensation** - Second mirror compensates for motion
4. **Result** - Straight, continuous scan lines

**After Failure** (May 31, 2003 onwards):
1. **SLC mirror stuck** - Compensation mechanism failed
2. **Uncompensated forward motion** - Satellite movement creates diagonal scan displacement
3. **Zig-zag gap pattern** - Diagonal gaps alternate direction (left/right) on successive scans
4. **Data gaps** - Missed regions between scan lines form diagonal wedges
5. **Wedge pattern** - Gaps widen with distance from nadir (subsatellite point)

**Geometry**

The gap width as a function of position:

```
gap_width(y) = gap_max × |y - y_center| / (y_max / 2)
```

Where:
- y = row position in image
- y_center = nadir (center) row
- gap_max ≈ 22% of scene width (historical value)

**Characteristics**

- **Diagonal gaps** - Uncompensated forward motion creates shallow diagonal wedges across image
- **Alternating directions** - Each scan line's gap tilts opposite direction (left/right zig-zag)
- **Symmetric wedges** - Gaps widen from center to edges
- **No gaps at center** - Nadir point has complete coverage
- **Maximum at edges** - ~22% data loss at top/bottom
- **Repeating pattern** - Gaps every ~14 scan lines (ETM+ geometry)
- **Persistent** - Same pattern in every scene since May 31, 2003

**Impact**

- **14% data loss** - Average across entire scene
- **22% maximum loss** - At scene edges
- **Continued operation** - Satellite still providing valuable data
- **Gap-filling algorithms** - Post-processing interpolates missing data

**Real-World Timeline**

- **Launch**: April 15, 1999
- **SLC failure**: May 31, 2003 (4 years into mission)
- **Continued operation**: Still operational as of 2024 (21+ years total)
- **Legacy**: 20+ years of SLC-Off data in archives

**Scientific Impact**

Despite gaps, Landsat 7 ETM+ SLC-Off data remains valuable:
- Long-term continuity with Landsat 4/5
- Higher resolution than Landsat 8/9 for some applications
- Gap-filling algorithms make data usable for many purposes

---

## Failure Mode Interactions

Real satellite systems experience multiple simultaneous failures:

### Radiation Environment Correlations

Solar particle events create correlated failures:
```
Solar Storm Event
├── Increased cosmic ray flux → salt_pepper
├── Memory upsets → buffer_corruption
├── Transmission errors → band_swap
└── Detector damage → corduroy (long-term)
```

### Age-Related Degradation

Long-duration missions accumulate failures:
```
Mission Years 1-3: Minimal degradation
├── salt_pepper: 0.0001 (baseline cosmic rays)
└── corduroy: strength 0.1 (minor drift)

Mission Years 5-10: Moderate degradation
├── salt_pepper: 0.0003 (detector damage)
├── corduroy: strength 0.4 (calibration drift)
└── compression_artifact: encoder wear

Mission Years 15+: Severe degradation
├── salt_pepper: 0.001 (accumulated damage)
├── corduroy: strength 0.8 (severe drift)
├── compression_artifact: frequent failures
└── Potential mechanical failures (SLC-Off scenario)
```

### Cascading Failures

One failure can trigger others:
```
Solar Particle Event
→ Memory corruption (buffer_corruption)
→ Corrupts compression tables
→ Encoder failure (compression_artifact)
→ Reduced data quality
```

---

## Orbital Considerations

Failure rates vary by orbit type:

### Low Earth Orbit (LEO)

**Altitude**: 400-800 km
**Examples**: Landsat, MODIS Terra/Aqua, Sentinel-2

**Radiation Environment**:
- Moderate cosmic ray flux
- South Atlantic Anomaly (SAA) passages: 5-10× higher rates
- Solar particle events: Significant impact

**Typical Failure Rates**:
- salt_pepper: 0.0001-0.001% per frame
- buffer_corruption: ~1 event per 10⁶ frames
- corduroy: Develops over 5-15 years

### Sun-Synchronous Orbit (SSO)

**Altitude**: 600-800 km
**Examples**: Most Earth observation satellites

**Characteristics**:
- Fixed local time (e.g., always 10:30 AM)
- Consistent solar illumination
- SAA passages every ~90 minutes

**Failure Patterns**:
- Regular SAA-induced spikes in salt_pepper
- Thermal cycling → accelerated detector drift

### Geostationary Orbit (GEO)

**Altitude**: 36,000 km
**Examples**: GOES, MSG, Himawari

**Radiation Environment**:
- Lower overall flux (outside Van Allen belts)
- Greater solar wind exposure
- Higher energy particles

**Typical Failure Rates**:
- Lower salt_pepper rate than LEO
- Higher energy SEU events (buffer_corruption)

---

## Mitigation Strategies

Understanding failures informs both simulation realism and appreciation of operational challenges:

### Radiation Hardening

**Detector Design**:
- Shielding (tungsten, tantalum)
- Redundant detectors
- Error correction codes (ECC)

**Memory Protection**:
- EDAC (Error Detection And Correction)
- Triple Modular Redundancy (TMR)
- Scrubbing (periodic memory refresh)

### Calibration Management

**On-orbit Calibration**:
- Solar calibrator targets
- Lunar observations (stable reference)
- Cross-calibration with other satellites

**Periodic Updates**:
- Quarterly coefficient updates
- Detector health monitoring
- Drift prediction models

### Data Quality Flags

Operational systems flag suspect data:
- Pixel quality masks
- SAA passage indicators
- Calibration confidence levels

---

## Using This Information

### For Realistic Simulation

Match parameters to mission phase:

**Early Mission** (Years 0-3):
```yaml
- operation: "salt_pepper"
  params:
    amount: 0.0001  # Baseline
- operation: "corduroy"
  params:
    strength: 0.1   # Minimal drift
```

**Mid-Mission** (Years 5-10):
```yaml
- operation: "salt_pepper"
  params:
    amount: 0.0005  # Some detector damage
- operation: "corduroy"
  params:
    strength: 0.4   # Noticeable drift
```

**Late Mission** (Years 15+):
```yaml
- operation: "salt_pepper"
  params:
    amount: 0.002   # Significant damage
- operation: "corduroy"
  params:
    strength: 0.7   # Severe drift
```

### For Aesthetic Goals

Use understanding to create intentional effects:
- **Documentary realism**: Match historical parameters (SLC-Off: 0.22)
- **Glitch art**: Exaggerate specific failures (band_swap tile_count: 50)
- **Scientific visualization**: Isolated failures to understand individual effects

---

## References and Further Reading

**Landsat 7 SLC-Off**:
- USGS: "Landsat 7 SLC-off Products: Background and Information" (2003)
- Storey et al., "Landsat 7 Scan Line Corrector-Off Gap-Filled Product Development" (2005)

**Radiation Effects**:
- Barth et al., "Space, Atmospheric, and Terrestrial Radiation Environments" (2003)
- Holmes-Siedle & Adams, "Handbook of Radiation Effects" (2002)

**Detector Calibration**:
- Xiong et al., "MODIS On-Orbit Calibration and Characterization" (2006)
- Chander et al., "Overview of Landsat Calibration History" (2009)

**Single Event Effects**:
- Normand, "Single Event Effects in Avionics" (1996)
- Petersen, "Single Event Effects in Aerospace" (2011)

---

## Conclusion

The six satellite malfunction operations simulate well-documented failure modes from decades of remote sensing operations. Understanding the physical mechanisms, orbital environments, and historical examples provides:

1. **Realistic parameter selection** - Match real-world failure rates
2. **Intentional aesthetic choices** - Know what you're simulating
3. **Scientific appreciation** - Recognize challenges of space-based imaging

These operations bridge artistic expression and technical accuracy, allowing creation of both realistic simulations and intentional glitch aesthetics grounded in actual satellite physics.

---

**Next Steps**: Apply this understanding in the [Satellite Malfunctions Tutorial](../tutorials/satellite-malfunctions/) and explore parameter options in the [Satellite Operations Reference](../operations/satellite/).
