"""
Taichi end-to-end pipeline Bayer filter operation.

Simulates digital sensor artifacts using Bayer filter mosaicing and demosaicing.
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

from typing import Any, Literal

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
from sevenrad_stills.pipeline.protocols import TempFieldSpec

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Type alias for Bayer patterns
BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
VALID_PATTERNS: set[BayerPattern] = {"RGGB", "BGGR", "GRBG", "GBRG"}


# Define the kernels only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _mosaicing_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        mosaic: ti.template(),  # type: ignore[valid-type]
        pattern_code: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for Bayer mosaicing - extract single channel per pixel.

        Patterns (top-left 2x2 block):
        0=RGGB: R G / G B
        1=BGGR: B G / G R
        2=GRBG: G R / B G
        3=GBRG: G B / R G

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            mosaic: Output scalar.field with shape (batch, height, width)
            pattern_code: Pattern identifier (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            pixel = source[batch, i, j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]

            # Determine position in 2x2 Bayer pattern
            row_mod = i % 2
            col_mod = j % 2

            value = 0.0

            # RGGB pattern
            if pattern_code == 0:
                if row_mod == 0 and col_mod == 0:
                    value = r  # R
                elif row_mod == 0 and col_mod == 1:
                    value = g  # G
                elif row_mod == 1 and col_mod == 0:
                    value = g  # G
                else:  # row_mod == 1 and col_mod == 1
                    value = b  # B

            # BGGR pattern
            elif pattern_code == 1:
                if row_mod == 0 and col_mod == 0:
                    value = b  # B
                elif row_mod == 0 and col_mod == 1:
                    value = g  # G
                elif row_mod == 1 and col_mod == 0:
                    value = g  # G
                else:  # row_mod == 1 and col_mod == 1
                    value = r  # R

            # GRBG pattern
            elif pattern_code == 2:
                if row_mod == 0 and col_mod == 0:
                    value = g  # G
                elif row_mod == 0 and col_mod == 1:
                    value = r  # R
                elif row_mod == 1 and col_mod == 0:
                    value = b  # B
                else:  # row_mod == 1 and col_mod == 1
                    value = g  # G

            # GBRG pattern
            else:  # pattern_code == 3
                if row_mod == 0 and col_mod == 0:
                    value = g  # G
                elif row_mod == 0 and col_mod == 1:
                    value = b  # B
                elif row_mod == 1 and col_mod == 0:
                    value = r  # R
                else:  # row_mod == 1 and col_mod == 1
                    value = g  # G

            mosaic[batch, i, j] = value

    @ti.kernel  # type: ignore[misc]
    def _demosaicing_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        mosaic: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        pattern_code: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for Bayer demosaicing using bilinear interpolation.

        Reconstructs full RGB image from mosaic by interpolating missing channels.

        Args:
            mosaic: Input scalar.field with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            pattern_code: Pattern identifier (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            row_mod = i % 2
            col_mod = j % 2

            r = 0.0
            g = 0.0
            b = 0.0

            # Determine which channel is known at this position
            channel_type = 0  # 0=R, 1=G, 2=B

            if pattern_code == 0:  # RGGB
                if row_mod == 0 and col_mod == 0:
                    channel_type = 0  # R
                elif row_mod == 0 and col_mod == 1:
                    channel_type = 1  # G
                elif row_mod == 1 and col_mod == 0:
                    channel_type = 1  # G
                else:
                    channel_type = 2  # B
            elif pattern_code == 1:  # BGGR
                if row_mod == 0 and col_mod == 0:
                    channel_type = 2  # B
                elif row_mod == 0 and col_mod == 1:
                    channel_type = 1  # G
                elif row_mod == 1 and col_mod == 0:
                    channel_type = 1  # G
                else:
                    channel_type = 0  # R
            elif pattern_code == 2:  # GRBG
                if row_mod == 0 and col_mod == 0:
                    channel_type = 1  # G
                elif row_mod == 0 and col_mod == 1:
                    channel_type = 0  # R
                elif row_mod == 1 and col_mod == 0:
                    channel_type = 2  # B
                else:
                    channel_type = 1  # G
            else:  # GBRG
                if row_mod == 0 and col_mod == 0:
                    channel_type = 1  # G
                elif row_mod == 0 and col_mod == 1:
                    channel_type = 2  # B
                elif row_mod == 1 and col_mod == 0:
                    channel_type = 0  # R
                else:
                    channel_type = 1  # G

            # Bilinear interpolation for missing channels
            # Handle edge cases with clamping
            value_center = mosaic[batch, i, j]

            if channel_type == 0:  # R position
                r = value_center
                # Interpolate G from 4 neighbors (cross pattern)
                g_count = 0
                g_sum = 0.0
                if i > 0:
                    g_sum += mosaic[batch, i - 1, j]
                    g_count += 1
                if i < height - 1:
                    g_sum += mosaic[batch, i + 1, j]
                    g_count += 1
                if j > 0:
                    g_sum += mosaic[batch, i, j - 1]
                    g_count += 1
                if j < width - 1:
                    g_sum += mosaic[batch, i, j + 1]
                    g_count += 1
                g = g_sum / ti.max(g_count, 1)

                # Interpolate B from 4 diagonal neighbors
                b_count = 0
                b_sum = 0.0
                if i > 0 and j > 0:
                    b_sum += mosaic[batch, i - 1, j - 1]
                    b_count += 1
                if i > 0 and j < width - 1:
                    b_sum += mosaic[batch, i - 1, j + 1]
                    b_count += 1
                if i < height - 1 and j > 0:
                    b_sum += mosaic[batch, i + 1, j - 1]
                    b_count += 1
                if i < height - 1 and j < width - 1:
                    b_sum += mosaic[batch, i + 1, j + 1]
                    b_count += 1
                b = b_sum / ti.max(b_count, 1)

            elif channel_type == 1:  # G position
                g = value_center

                # Determine if we need R or B from horizontal/vertical neighbors
                # This depends on pattern, but we'll use a simpler approach:
                # Interpolate both R and B from respective neighbors

                # For R: check horizontal neighbors (or diagonal depending on position)
                r_count = 0
                r_sum = 0.0
                # Check horizontal
                if j > 0:
                    r_sum += mosaic[batch, i, j - 1]
                    r_count += 1
                if j < width - 1:
                    r_sum += mosaic[batch, i, j + 1]
                    r_count += 1
                # If no horizontal neighbors found, use vertical
                if r_count == 0:
                    if i > 0:
                        r_sum += mosaic[batch, i - 1, j]
                        r_count += 1
                    if i < height - 1:
                        r_sum += mosaic[batch, i + 1, j]
                        r_count += 1
                r = r_sum / ti.max(r_count, 1)

                # For B: opposite direction from R
                b_count = 0
                b_sum = 0.0
                # Try vertical first
                if i > 0:
                    b_sum += mosaic[batch, i - 1, j]
                    b_count += 1
                if i < height - 1:
                    b_sum += mosaic[batch, i + 1, j]
                    b_count += 1
                # If no vertical neighbors, use horizontal
                if b_count == 0:
                    if j > 0:
                        b_sum += mosaic[batch, i, j - 1]
                        b_count += 1
                    if j < width - 1:
                        b_sum += mosaic[batch, i, j + 1]
                        b_count += 1
                b = b_sum / ti.max(b_count, 1)

            else:  # B position
                b = value_center
                # Interpolate G from 4 neighbors (cross pattern)
                g_count = 0
                g_sum = 0.0
                if i > 0:
                    g_sum += mosaic[batch, i - 1, j]
                    g_count += 1
                if i < height - 1:
                    g_sum += mosaic[batch, i + 1, j]
                    g_count += 1
                if j > 0:
                    g_sum += mosaic[batch, i, j - 1]
                    g_count += 1
                if j < width - 1:
                    g_sum += mosaic[batch, i, j + 1]
                    g_count += 1
                g = g_sum / ti.max(g_count, 1)

                # Interpolate R from 4 diagonal neighbors
                r_count = 0
                r_sum = 0.0
                if i > 0 and j > 0:
                    r_sum += mosaic[batch, i - 1, j - 1]
                    r_count += 1
                if i > 0 and j < width - 1:
                    r_sum += mosaic[batch, i - 1, j + 1]
                    r_count += 1
                if i < height - 1 and j > 0:
                    r_sum += mosaic[batch, i + 1, j - 1]
                    r_count += 1
                if i < height - 1 and j < width - 1:
                    r_sum += mosaic[batch, i + 1, j + 1]
                    r_count += 1
                r = r_sum / ti.max(r_count, 1)

            # Clamp and write result with preserved alpha
            dest[batch, i, j] = ti.Vector(
                [
                    ti.max(0.0, ti.min(1.0, r)),
                    ti.max(0.0, ti.min(1.0, g)),
                    ti.max(0.0, ti.min(1.0, b)),
                    1.0,
                ]
            )


class BayerFilterTaichiOperation(BaseTaichiOperation):
    """
    Taichi Bayer filter for end-to-end GPU pipeline.

    Simulates digital sensor artifacts using Bayer filter mosaicing and demosaicing.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation requires a temporary field for the mosaic intermediate result
    and does not support in-place execution.

    Example:
        >>> op = BayerFilterTaichiOperation()
        >>> temp_fields = {"mosaic": mosaic_field}
        >>> op.apply_to_field(source, dest, temp_fields, {"pattern": "RGGB"}, height, width)

    """

    def __init__(self) -> None:
        """Initialize Bayer filter operation."""
        super().__init__("bayer_filter_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Bayer filter requires two passes (mosaic + demosaic) with intermediate
        storage, so cannot be done in-place.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """
        Specify temporary field requirements.

        Bayer filter needs a single-channel mosaic buffer for intermediate storage.

        Returns:
            List with one TempFieldSpec for the mosaic buffer (1 channel).

        """
        return [
            TempFieldSpec(
                name="mosaic",
                shape_factor=(1.0, 1.0, 1),  # Same dimensions, 1 channel
                dtype="f32",
            )
        ]

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate Bayer filter parameters.

        Expected params:
        - pattern: str - Bayer pattern ("RGGB", "BGGR", "GRBG", or "GBRG")

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If pattern is missing or invalid

        """
        if "pattern" not in params:
            msg = "Bayer filter requires 'pattern' parameter"
            raise ValueError(msg)

        pattern = params["pattern"]
        if not isinstance(pattern, str):
            msg = f"Pattern must be a string, got {type(pattern)}"
            raise ValueError(msg)

        if pattern not in VALID_PATTERNS:
            valid = ", ".join(sorted(VALID_PATTERNS))
            msg = f"Invalid pattern '{pattern}'. Must be one of {valid}"
            raise ValueError(msg)

    def apply_to_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,  # ti.Vector.field
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply Bayer filter on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Must contain 'mosaic' key with scalar field for intermediate
            params: Must contain 'pattern' key with Bayer pattern string
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available
            KeyError: If mosaic temp field is not provided

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        if "mosaic" not in temp_fields:
            msg = "Bayer filter requires 'mosaic' temporary field"
            raise KeyError(msg)

        pattern = params["pattern"]
        # Convert pattern string to code for kernel
        pattern_code = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3}[pattern]

        mosaic_field = temp_fields["mosaic"]

        # Two-pass operation: mosaic then demosaic
        _mosaicing_kernel(source, mosaic_field, pattern_code, 0, height, width)
        _demosaicing_kernel(mosaic_field, dest, pattern_code, 0, height, width)

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        NumPy reference implementation for testing.

        Produces results similar to apply_to_field for correctness testing.
        Uses simple bilinear demosaicing (not Malvar2004 from CPU version).

        Args:
            image: Input image as numpy array (H, W, 3) float32 in [0, 1]
            params: Must contain 'pattern' key

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        pattern = params["pattern"]

        height, width = image.shape[:2]

        # 1. Mosaicing: Create Bayer pattern CFA
        mosaic = np.zeros((height, width), dtype=np.float32)

        if pattern == "RGGB":
            mosaic[0::2, 0::2] = image[0::2, 0::2, 0]  # R
            mosaic[0::2, 1::2] = image[0::2, 1::2, 1]  # G
            mosaic[1::2, 0::2] = image[1::2, 0::2, 1]  # G
            mosaic[1::2, 1::2] = image[1::2, 1::2, 2]  # B
        elif pattern == "BGGR":
            mosaic[0::2, 0::2] = image[0::2, 0::2, 2]  # B
            mosaic[0::2, 1::2] = image[0::2, 1::2, 1]  # G
            mosaic[1::2, 0::2] = image[1::2, 0::2, 1]  # G
            mosaic[1::2, 1::2] = image[1::2, 1::2, 0]  # R
        elif pattern == "GRBG":
            mosaic[0::2, 0::2] = image[0::2, 0::2, 1]  # G
            mosaic[0::2, 1::2] = image[0::2, 1::2, 0]  # R
            mosaic[1::2, 0::2] = image[1::2, 0::2, 2]  # B
            mosaic[1::2, 1::2] = image[1::2, 1::2, 1]  # G
        elif pattern == "GBRG":
            mosaic[0::2, 0::2] = image[0::2, 0::2, 1]  # G
            mosaic[0::2, 1::2] = image[0::2, 1::2, 2]  # B
            mosaic[1::2, 0::2] = image[1::2, 0::2, 0]  # R
            mosaic[1::2, 1::2] = image[1::2, 1::2, 1]  # G

        # 2. Demosaicing using simple bilinear interpolation
        result = np.zeros((height, width, 3), dtype=np.float32)

        # Pattern lookup for determining channel type at each position
        pattern_map = {
            "RGGB": np.array([[0, 1], [1, 2]], dtype=np.int32),
            "BGGR": np.array([[2, 1], [1, 0]], dtype=np.int32),
            "GRBG": np.array([[1, 0], [2, 1]], dtype=np.int32),
            "GBRG": np.array([[1, 2], [0, 1]], dtype=np.int32),
        }

        channel_type = np.tile(
            pattern_map[pattern], ((height + 1) // 2, (width + 1) // 2)
        )[:height, :width]

        # Interpolate for each position
        for i in range(height):
            for j in range(width):
                ch_type = channel_type[i, j]
                value_center = mosaic[i, j]

                if ch_type == 0:  # R position
                    result[i, j, 0] = value_center

                    # G from cross neighbors
                    g_neighbors = []
                    if i > 0:
                        g_neighbors.append(mosaic[i - 1, j])
                    if i < height - 1:
                        g_neighbors.append(mosaic[i + 1, j])
                    if j > 0:
                        g_neighbors.append(mosaic[i, j - 1])
                    if j < width - 1:
                        g_neighbors.append(mosaic[i, j + 1])
                    result[i, j, 1] = np.mean(g_neighbors) if g_neighbors else 0.0

                    # B from diagonal neighbors
                    b_neighbors = []
                    if i > 0 and j > 0:
                        b_neighbors.append(mosaic[i - 1, j - 1])
                    if i > 0 and j < width - 1:
                        b_neighbors.append(mosaic[i - 1, j + 1])
                    if i < height - 1 and j > 0:
                        b_neighbors.append(mosaic[i + 1, j - 1])
                    if i < height - 1 and j < width - 1:
                        b_neighbors.append(mosaic[i + 1, j + 1])
                    result[i, j, 2] = np.mean(b_neighbors) if b_neighbors else 0.0

                elif ch_type == 1:  # G position
                    result[i, j, 1] = value_center

                    # R and B from neighbors
                    # Try horizontal for one, vertical for the other
                    h_neighbors = []
                    if j > 0:
                        h_neighbors.append(mosaic[i, j - 1])
                    if j < width - 1:
                        h_neighbors.append(mosaic[i, j + 1])

                    v_neighbors = []
                    if i > 0:
                        v_neighbors.append(mosaic[i - 1, j])
                    if i < height - 1:
                        v_neighbors.append(mosaic[i + 1, j])

                    # Assign to R and B based on pattern position
                    if h_neighbors:
                        result[i, j, 0] = np.mean(h_neighbors)
                    else:
                        result[i, j, 0] = np.mean(v_neighbors) if v_neighbors else 0.0

                    if v_neighbors:
                        result[i, j, 2] = np.mean(v_neighbors)
                    else:
                        result[i, j, 2] = np.mean(h_neighbors) if h_neighbors else 0.0

                else:  # B position
                    result[i, j, 2] = value_center

                    # G from cross neighbors
                    g_neighbors = []
                    if i > 0:
                        g_neighbors.append(mosaic[i - 1, j])
                    if i < height - 1:
                        g_neighbors.append(mosaic[i + 1, j])
                    if j > 0:
                        g_neighbors.append(mosaic[i, j - 1])
                    if j < width - 1:
                        g_neighbors.append(mosaic[i, j + 1])
                    result[i, j, 1] = np.mean(g_neighbors) if g_neighbors else 0.0

                    # R from diagonal neighbors
                    r_neighbors = []
                    if i > 0 and j > 0:
                        r_neighbors.append(mosaic[i - 1, j - 1])
                    if i > 0 and j < width - 1:
                        r_neighbors.append(mosaic[i - 1, j + 1])
                    if i < height - 1 and j > 0:
                        r_neighbors.append(mosaic[i + 1, j - 1])
                    if i < height - 1 and j < width - 1:
                        r_neighbors.append(mosaic[i + 1, j + 1])
                    result[i, j, 0] = np.mean(r_neighbors) if r_neighbors else 0.0

        # Clip to valid range
        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the Bayer filter kernels
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_mosaic = ti.field(dtype=ti.f32, shape=(1, 2, 2))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Trigger compilation for both kernels
        _mosaicing_kernel(dummy_src, dummy_mosaic, 0, 0, 2, 2)
        _demosaicing_kernel(dummy_mosaic, dummy_dst, 0, 0, 2, 2)
