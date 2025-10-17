# Extending the Pipeline System

This guide explains how to add new operations to the sevenrad-stills pipeline system while maintaining modularity, flexibility, and non-destructive processing.

## Design Principles

### 1. Non-Destructive Processing
Every operation preserves its input and creates new output files. The system maintains a complete history of transformations.

**Key Concept**: Like Photoshop's adjustment layers, each operation builds on top of previous results without modifying the original.

```yaml
steps:
  - name: "step1"           # Input: extracted frames
    operation: "compress"    # Output: compressed versions

  - name: "step2"           # Input: step1 output
    operation: "blur"        # Output: compressed + blurred

  - name: "step3"           # Input: step2 output
    operation: "downscale"   # Output: compressed + blurred + downscaled
```

**Output Structure**:
```
output/
├── intermediate/
│   ├── extracted/              # Original frames
│   ├── step1/                  # After compression
│   ├── step2/                  # After blur
│   └── step3/                  # After downscale
└── final/                      # Last step output
```

### 2. Chained Filenames
Each operation prepends its name to the filename, creating a visual history of transformations.

**Example**:
```
Original:                extracted_video_000001.jpg
After compress:          compress_extracted_video_000001_step00.jpg
After blur:              blur_compress_extracted_video_000001_step00_step01.jpg
After downscale:         downscale_blur_compress_extracted_video_000001_step00_step01_step02.jpg
```

This makes it immediately clear which operations were applied and in what order.

### 3. Pluggable Architecture
Operations are self-contained modules that follow a standard interface. Adding a new operation requires:

1. Create operation class
2. Register in `__init__.py`
3. Use in YAML pipelines

No changes to core pipeline logic required.

## Creating New Operations

### Step 1: Define the Operation Class

Create a new file in `src/sevenrad_stills/operations/`:

```python
# src/sevenrad_stills/operations/my_operation.py
"""My custom operation for doing X."""

from pathlib import Path
from PIL import Image
from .base import Operation

class MyOperation(Operation):
    """Apply my custom transformation.

    This operation does [describe what it does].

    Parameters:
        param1 (int): Description of param1 (range: min-max)
        param2 (str): Description of param2 (options: "a", "b", "c")
    """

    def __init__(self, param1: int = 50, param2: str = "default"):
        """Initialize operation with parameters.

        Args:
            param1: First parameter
            param2: Second parameter
        """
        super().__init__()
        self.param1 = self._validate_param1(param1)
        self.param2 = self._validate_param2(param2)

    def _validate_param1(self, value: int) -> int:
        """Validate param1 is in acceptable range."""
        if not 1 <= value <= 100:
            raise ValueError(f"param1 must be 1-100, got {value}")
        return value

    def _validate_param2(self, value: str) -> str:
        """Validate param2 is acceptable option."""
        valid = ["option_a", "option_b", "option_c"]
        if value not in valid:
            raise ValueError(f"param2 must be one of {valid}, got {value}")
        return value

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply the transformation to an image.

        This method must:
        1. Accept a PIL Image
        2. Return a modified PIL Image
        3. NOT modify the input image (create a copy if needed)

        Args:
            image: Input PIL Image

        Returns:
            Transformed PIL Image
        """
        # Create a copy to preserve input
        result = image.copy()

        # Apply your transformation
        # ... your processing logic here ...

        return result
```

### Step 2: Register the Operation

Add to `src/sevenrad_stills/operations/__init__.py`:

```python
from .my_operation import MyOperation

OPERATIONS = {
    "compression": CompressionOperation,
    "downscale": DownscaleOperation,
    "motion_blur": MotionBlurOperation,
    "multi_compress": MultiCompressOperation,
    "my_operation": MyOperation,  # Add your operation
}
```

### Step 3: Use in YAML Pipelines

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=VIDEO_ID"

segment:
  start: 10.0
  end: 13.0
  interval: 0.1

output:
  base_dir: "output"

steps:
  - name: "custom_step"
    operation: "my_operation"
    params:
      param1: 75
      param2: "option_a"
```

## Operation Design Patterns

### Pattern 1: Simple Transform
Single input → single output with parameters.

**Example**: Adjust brightness

```python
class BrightnessOperation(Operation):
    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = max(0.1, min(3.0, factor))

    def apply(self, image: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.factor)
```

**YAML Usage**:
```yaml
- name: "brighten"
  operation: "brightness"
  params:
    factor: 1.5  # 50% brighter
```

### Pattern 2: Parametric Transform
Multiple configurable parameters affecting behavior.

**Example**: Color shift with multiple controls

```python
class ColorShiftOperation(Operation):
    def __init__(self, hue: int = 0, saturation: float = 1.0, value: float = 1.0):
        super().__init__()
        self.hue = hue % 360
        self.saturation = max(0.0, min(2.0, saturation))
        self.value = max(0.0, min(2.0, value))

    def apply(self, image: Image.Image) -> Image.Image:
        # Convert RGB → HSV, adjust, convert back
        # ... implementation ...
        return result
```

**YAML Usage**:
```yaml
- name: "color_adjust"
  operation: "color_shift"
  params:
    hue: 30        # Shift hue by 30 degrees
    saturation: 1.2  # Increase saturation 20%
    value: 0.9     # Decrease brightness 10%
```

### Pattern 3: Iterative Transform
Apply operation multiple times (using `repeat` parameter).

**Example**: Progressive blur

```yaml
- name: "heavy_blur"
  operation: "gaussian_blur"
  repeat: 5      # Apply blur 5 times
  params:
    radius: 2    # Radius for each iteration
```

The pipeline automatically handles repetition - your operation just defines single application.

### Pattern 4: Conditional Parameters
Parameters that enable/disable sub-features.

**Example**: Downscale with optional upscale

```python
class DownscaleOperation(Operation):
    def __init__(
        self,
        scale: float = 0.5,
        upscale: bool = False,
        downscale_method: str = "lanczos",
        upscale_method: str = "lanczos"
    ):
        super().__init__()
        self.scale = scale
        self.upscale = upscale
        self.downscale_method = self._get_resample_filter(downscale_method)
        self.upscale_method = self._get_resample_filter(upscale_method)

    def apply(self, image: Image.Image) -> Image.Image:
        # Downscale
        new_size = (int(image.width * self.scale),
                   int(image.height * self.scale))
        result = image.resize(new_size, self.downscale_method)

        # Conditionally upscale back
        if self.upscale:
            result = result.resize(image.size, self.upscale_method)

        return result
```

**YAML Usage**:
```yaml
- name: "pixelate"
  operation: "downscale"
  params:
    scale: 0.1          # Reduce to 10%
    upscale: true       # Scale back to original size
    downscale_method: "bicubic"
    upscale_method: "nearest"  # Creates blocky pixels
```

### Pattern 5: Multi-Generation Transform
Internal iteration with changing parameters.

**Example**: Progressive quality decay

```python
class MultiCompressOperation(Operation):
    def __init__(
        self,
        iterations: int = 5,
        quality_start: int = 75,
        quality_end: int = 25,
        decay: str = "linear"
    ):
        super().__init__()
        self.iterations = iterations
        self.quality_start = quality_start
        self.quality_end = quality_end
        self.decay = decay

    def apply(self, image: Image.Image) -> Image.Image:
        result = image.copy()

        for i in range(self.iterations):
            # Calculate quality for this iteration
            quality = self._calculate_quality(i)

            # Apply compression
            result = self._compress_once(result, quality)

        return result

    def _calculate_quality(self, iteration: int) -> int:
        """Calculate quality using decay curve."""
        if self.decay == "linear":
            # Linear interpolation
            progress = iteration / max(1, self.iterations - 1)
            quality = self.quality_start - (self.quality_start - self.quality_end) * progress
        elif self.decay == "exponential":
            # Exponential decay (more degradation early)
            progress = (iteration / max(1, self.iterations - 1)) ** 2
            quality = self.quality_start - (self.quality_start - self.quality_end) * progress

        return int(quality)
```

**YAML Usage**:
```yaml
- name: "degrade"
  operation: "multi_compress"
  params:
    iterations: 10
    quality_start: 90
    quality_end: 10
    decay: "exponential"  # Heavy degradation early
```

## Stacking Operations (Photoshop-Style)

### Simple Stack
Operations applied in sequence, each building on the previous:

```yaml
steps:
  # Like Photoshop layers, bottom to top
  - name: "base_compress"
    operation: "compression"
    params:
      quality: 70

  - name: "add_blur"
    operation: "motion_blur"
    params:
      kernel_size: 5
      angle: 45

  - name: "color_grade"
    operation: "color_shift"
    params:
      saturation: 0.8
      value: 1.1

  - name: "final_sharpen"
    operation: "sharpen"
    params:
      strength: 1.5
```

**Output**: Each step creates intermediate results, final output has all effects.

### Branching (Future Enhancement)
Process same input multiple ways:

```yaml
steps:
  - name: "extract_base"
    operation: "extract"

  # Branch A: Heavy degradation
  - name: "branch_a_compress"
    operation: "compression"
    input: "extract_base"  # Explicitly reference input
    params:
      quality: 10

  - name: "branch_a_blur"
    operation: "motion_blur"
    input: "branch_a_compress"
    params:
      kernel_size: 20

  # Branch B: Subtle effects
  - name: "branch_b_compress"
    operation: "compression"
    input: "extract_base"  # Same input, different path
    params:
      quality: 85

  - name: "branch_b_color"
    operation: "color_shift"
    input: "branch_b_compress"
    params:
      saturation: 1.2
```

### Conditional Operations (Future Enhancement)
Apply operations based on image properties:

```yaml
steps:
  - name: "adaptive_compress"
    operation: "compression"
    condition:
      if: "width > 1920"  # Only compress large images
    params:
      quality: 80
```

## Best Practices

### 1. Parameter Validation
Always validate parameters in `__init__` to fail fast with clear errors:

```python
def __init__(self, quality: int = 75):
    super().__init__()
    if not 1 <= quality <= 100:
        raise ValueError(
            f"quality must be between 1 and 100, got {quality}"
        )
    self.quality = quality
```

### 2. Provide Sensible Defaults
Choose defaults that work for most use cases:

```python
def __init__(
    self,
    quality: int = 75,      # Good balance
    subsampling: int = 2    # Standard 4:2:0
):
    ...
```

### 3. Document Parameter Ranges
Use docstrings to document acceptable ranges and their effects:

```python
class CompressionOperation(Operation):
    """JPEG compression with quality control.

    Parameters:
        quality (int): JPEG quality level
            - Range: 1-100
            - 1-30: Heavy compression, visible artifacts
            - 31-70: Moderate compression, some artifacts
            - 71-95: Light compression, minimal artifacts
            - 96-100: Minimal compression, large files

        subsampling (int): Chroma subsampling mode
            - 0: No subsampling (4:4:4) - highest quality
            - 1: Moderate subsampling (4:2:2)
            - 2: Heavy subsampling (4:2:0) - most compression
    """
```

### 4. Use Type Hints
Enable static type checking and better IDE support:

```python
from typing import Optional
from PIL import Image
from pathlib import Path

def apply(self, image: Image.Image) -> Image.Image:
    ...

def _helper_method(self, value: int, multiplier: float = 1.0) -> int:
    ...
```

### 5. Make Operations Deterministic
Same input + same parameters = same output. Avoid randomness unless explicitly configurable:

```python
# Bad - produces different results each run
def apply(self, image: Image.Image) -> Image.Image:
    noise_amount = random.randint(0, 50)  # Unpredictable!
    ...

# Good - reproducible results
def __init__(self, noise_amount: int = 25, seed: Optional[int] = None):
    self.noise_amount = noise_amount
    self.seed = seed

def apply(self, image: Image.Image) -> Image.Image:
    if self.seed is not None:
        random.seed(self.seed)  # Reproducible with seed
    noise_amount = random.randint(0, self.noise_amount)
    ...
```

### 6. Keep Operations Focused
Each operation should do one thing well. Combine multiple simple operations rather than creating complex mega-operations:

```yaml
# Good - composable
steps:
  - name: "blur"
    operation: "gaussian_blur"
    params:
      radius: 3

  - name: "compress"
    operation: "compression"
    params:
      quality: 60

# Bad - monolithic
steps:
  - name: "blur_and_compress"  # Don't do this
    operation: "blur_compress_combo"
    params:
      blur_radius: 3
      compression_quality: 60
```

### 7. Preserve Image Mode
Handle different image modes (RGB, RGBA, L, etc.) appropriately:

```python
def apply(self, image: Image.Image) -> Image.Image:
    # Preserve original mode
    original_mode = image.mode

    # Convert if needed for processing
    if original_mode != "RGB":
        working_image = image.convert("RGB")
    else:
        working_image = image.copy()

    # ... process working_image ...

    # Convert back if needed
    if original_mode != "RGB":
        result = result.convert(original_mode)

    return result
```

## Testing New Operations

### Unit Test Template

```python
# tests/operations/test_my_operation.py
import pytest
from PIL import Image
from sevenrad_stills.operations.my_operation import MyOperation

def test_my_operation_basic():
    """Test basic functionality."""
    op = MyOperation(param1=50)

    # Create test image
    img = Image.new("RGB", (100, 100), color="red")

    # Apply operation
    result = op.apply(img)

    # Verify output
    assert isinstance(result, Image.Image)
    assert result.size == img.size
    assert result.mode == img.mode

def test_my_operation_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="param1 must be"):
        MyOperation(param1=150)  # Out of range

def test_my_operation_preserves_input():
    """Test that input image is not modified."""
    op = MyOperation()
    img = Image.new("RGB", (100, 100), color="red")
    original_pixels = img.copy()

    result = op.apply(img)

    # Input should be unchanged
    assert img.tobytes() == original_pixels.tobytes()
    # Output should be different
    assert result.tobytes() != img.tobytes()
```

### Integration Test Template

```python
# tests/integration/test_my_operation_pipeline.py
def test_my_operation_in_pipeline(tmp_path):
    """Test operation works in full pipeline."""
    yaml_config = tmp_path / "test_pipeline.yaml"
    yaml_config.write_text("""
source:
  youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

segment:
  start: 1.0
  end: 2.0
  interval: 1.0

output:
  base_dir: "{output_dir}"

steps:
  - name: "test_operation"
    operation: "my_operation"
    params:
      param1: 75
""".format(output_dir=str(tmp_path)))

    # Run pipeline
    # ... run and verify output ...
```

## Performance Considerations

### 1. Minimize Copies
Only copy when necessary to preserve input:

```python
# Efficient - modify in place when safe
def apply(self, image: Image.Image) -> Image.Image:
    result = image.copy()  # One copy
    # Work on result directly
    return result

# Inefficient - multiple unnecessary copies
def apply(self, image: Image.Image) -> Image.Image:
    temp1 = image.copy()
    temp2 = temp1.copy()  # Unnecessary
    result = temp2.copy()  # Unnecessary
    return result
```

### 2. Use Efficient Libraries
Leverage optimized libraries for heavy operations:

```python
import numpy as np
from PIL import Image

def apply(self, image: Image.Image) -> Image.Image:
    # Convert to numpy for efficient operations
    arr = np.array(image)

    # Fast vectorized operations
    arr = arr * 0.8  # Much faster than pixel-by-pixel

    # Convert back
    return Image.fromarray(arr.astype('uint8'))
```

### 3. Consider Parallel Processing
The pipeline handles parallel frame processing automatically. Your operation just needs to be thread-safe:

```python
# Thread-safe - no shared mutable state
class MyOperation(Operation):
    def __init__(self, param: int):
        self.param = param  # Immutable after init

    def apply(self, image: Image.Image) -> Image.Image:
        # Each call is independent
        result = image.copy()
        # ... process ...
        return result
```

## Examples of Operation Combinations

### Example 1: Vintage Film Look
```yaml
steps:
  - name: "grain"
    operation: "add_noise"
    params:
      amount: 15
      type: "gaussian"

  - name: "fade"
    operation: "color_shift"
    params:
      saturation: 0.7
      value: 0.9

  - name: "vignette"
    operation: "vignette"
    params:
      strength: 0.6

  - name: "aged_compress"
    operation: "compression"
    params:
      quality: 65
```

### Example 2: Digital Glitch Effect
```yaml
steps:
  - name: "pixelate"
    operation: "downscale"
    params:
      scale: 0.3
      upscale: true
      upscale_method: "nearest"

  - name: "color_corrupt"
    operation: "channel_shift"
    params:
      red_offset: 5
      blue_offset: -5

  - name: "heavy_compress"
    operation: "multi_compress"
    params:
      iterations: 8
      quality_start: 50
      quality_end: 5
      decay: "exponential"
```

### Example 3: Motion Blur Trail
```yaml
steps:
  - name: "blur_horizontal"
    operation: "motion_blur"
    params:
      kernel_size: 15
      angle: 0

  - name: "blur_diagonal"
    operation: "motion_blur"
    params:
      kernel_size: 10
      angle: 45

  - name: "slight_compress"
    operation: "compression"
    params:
      quality: 85
```

## Summary

The sevenrad-stills pipeline is designed like Photoshop's layer system:

1. **Non-Destructive**: Every operation preserves inputs
2. **Stackable**: Chain operations in any order
3. **Modular**: Each operation is self-contained
4. **Flexible**: Configure via YAML without code changes
5. **Traceable**: Chained filenames show transformation history

To add a new operation:
1. Create operation class with `apply(image) -> image`
2. Validate parameters in `__init__`
3. Register in `OPERATIONS` dict
4. Use in YAML pipelines

The system handles the rest: file management, chaining, parallel processing, and output organization.
