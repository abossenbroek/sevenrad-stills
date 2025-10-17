"""
Integration tests for compression and degradation filters.

These tests verify that all filters work correctly together in realistic
scenarios. Marked as 'slow' because they process actual images through
multiple pipeline steps.
"""

from pathlib import Path

import pytest
from PIL import Image
from sevenrad_stills.operations import get_operation


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a complex test image with gradients and patterns."""
    img = Image.new("RGB", (200, 200))
    pixels = img.load()

    # Create a gradient background with some patterns
    for y in range(200):
        for x in range(200):
            # Gradient
            r = int(255 * (x / 200))
            g = int(255 * (y / 200))
            b = 128

            # Add some checkerboard pattern for compression artifacts
            if (x // 20 + y // 20) % 2 == 0:
                r = min(255, r + 30)
                g = min(255, g + 30)

            pixels[x, y] = (r, g, b)

    return img


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output_dir = tmp_path / "filter_test_output"
    output_dir.mkdir()
    return output_dir


class TestCompressionFiltersIntegration:
    """Integration tests for compression and degradation operations."""

    @pytest.mark.slow
    def test_compression_operation_integration(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test compression operation with various quality levels."""
        operation = get_operation("compression")

        # Test severe compression
        params = {"quality": 10, "subsampling": 2}
        result = operation.apply(test_image_rgb, params)
        output_path = temp_output_dir / "compression_severe.jpg"
        operation.save_image(result, output_path, quality=95)

        assert output_path.exists()
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Test high quality
        params_hq = {"quality": 90, "subsampling": 0}
        result_hq = operation.apply(test_image_rgb, params_hq)
        assert result_hq.size == test_image_rgb.size

    @pytest.mark.slow
    def test_downscale_operation_integration(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test downscale operation with pixelation."""
        operation = get_operation("downscale")

        # Test heavy pixelation
        params = {
            "scale": 0.15,
            "upscale": True,
            "downscale_method": "bicubic",
            "upscale_method": "nearest",
        }
        result = operation.apply(test_image_rgb, params)
        output_path = temp_output_dir / "downscale_heavy.jpg"
        operation.save_image(result, output_path, quality=95)

        assert output_path.exists()
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

    @pytest.mark.slow
    def test_motion_blur_operation_integration(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test motion blur operation with various settings."""
        operation = get_operation("motion_blur")

        # Test minimal blur
        params = {"kernel_size": 3, "angle": 0}
        result = operation.apply(test_image_rgb, params)
        output_path = temp_output_dir / "motion_blur_minimal.jpg"
        operation.save_image(result, output_path, quality=95)

        assert output_path.exists()
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Test moderate blur with angle
        params_moderate = {"kernel_size": 10, "angle": 45}
        result_moderate = operation.apply(test_image_rgb, params_moderate)
        assert result_moderate.size == test_image_rgb.size

    @pytest.mark.slow
    def test_multi_compress_operation_integration(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test multi-generation compression."""
        operation = get_operation("multi_compress")

        # Test with linear decay
        params = {
            "iterations": 5,
            "quality_start": 60,
            "quality_end": 20,
            "decay": "linear",
            "subsampling": 2,
        }
        result = operation.apply(test_image_rgb, params)
        output_path = temp_output_dir / "multi_compress_linear.jpg"
        operation.save_image(result, output_path, quality=95)

        assert output_path.exists()
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Test with exponential decay
        params_exp = {
            "iterations": 7,
            "quality_start": 70,
            "quality_end": 15,
            "decay": "exponential",
            "subsampling": 2,
        }
        result_exp = operation.apply(test_image_rgb, params_exp)
        assert result_exp.size == test_image_rgb.size

    @pytest.mark.slow
    def test_combined_degradation_pipeline(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test combining multiple operations in sequence."""
        # Step 1: Downscale
        downscale_op = get_operation("downscale")
        step1 = downscale_op.apply(
            test_image_rgb,
            {
                "scale": 0.25,
                "upscale": True,
                "upscale_method": "nearest",
            },
        )

        # Step 2: Compress
        compress_op = get_operation("compression")
        step2 = compress_op.apply(step1, {"quality": 20, "subsampling": 2})

        # Step 3: Minimal blur
        blur_op = get_operation("motion_blur")
        step3 = blur_op.apply(step2, {"kernel_size": 3, "angle": 0})

        # Step 4: Multi-compress
        multi_compress_op = get_operation("multi_compress")
        final = multi_compress_op.apply(
            step3,
            {
                "iterations": 5,
                "quality_start": 40,
                "quality_end": 15,
                "decay": "linear",
            },
        )

        # Save final result
        output_path = temp_output_dir / "combined_degradation.jpg"
        multi_compress_op.save_image(final, output_path, quality=95)

        assert output_path.exists()
        assert final.size == test_image_rgb.size
        assert final.mode == "RGB"

        # Verify all intermediate steps produced valid images
        assert step1.size == test_image_rgb.size
        assert step2.size == test_image_rgb.size
        assert step3.size == test_image_rgb.size

    @pytest.mark.slow
    def test_social_media_compression_simulation(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test simulating social media compression artifacts."""
        # Simulate multiple shares with compression
        multi_compress_op = get_operation("multi_compress")
        result = multi_compress_op.apply(
            test_image_rgb,
            {
                "iterations": 4,
                "quality_start": 75,
                "quality_end": 45,
                "decay": "linear",
                "subsampling": 2,
            },
        )

        output_path = temp_output_dir / "social_media_sim.jpg"
        multi_compress_op.save_image(result, output_path, quality=95)

        assert output_path.exists()
        assert result.size == test_image_rgb.size

    @pytest.mark.slow
    def test_glitch_art_aesthetic(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test creating glitch art aesthetic with extreme degradation."""
        # Extreme pixelation
        downscale_op = get_operation("downscale")
        step1 = downscale_op.apply(
            test_image_rgb,
            {
                "scale": 0.08,
                "upscale": True,
                "upscale_method": "nearest",
            },
        )

        # Severe compression
        compress_op = get_operation("compression")
        step2 = compress_op.apply(step1, {"quality": 5, "subsampling": 2})

        # Heavy multi-generation
        multi_compress_op = get_operation("multi_compress")
        final = multi_compress_op.apply(
            step2,
            {
                "iterations": 12,
                "quality_start": 30,
                "quality_end": 5,
                "decay": "exponential",
            },
        )

        output_path = temp_output_dir / "glitch_art.jpg"
        multi_compress_op.save_image(final, output_path, quality=95)

        assert output_path.exists()
        assert final.size == test_image_rgb.size

    @pytest.mark.slow
    def test_vhs_analog_degradation(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test simulating VHS/analog video degradation."""
        # Horizontal motion blur (scan lines)
        blur_op = get_operation("motion_blur")
        step1 = blur_op.apply(test_image_rgb, {"kernel_size": 7, "angle": 0})

        # Moderate compression
        compress_op = get_operation("compression")
        step2 = compress_op.apply(step1, {"quality": 35, "subsampling": 2})

        # Slight pixelation
        downscale_op = get_operation("downscale")
        final = downscale_op.apply(
            step2,
            {
                "scale": 0.5,
                "upscale": True,
                "downscale_method": "bilinear",
                "upscale_method": "bilinear",
            },
        )

        output_path = temp_output_dir / "vhs_analog.jpg"
        downscale_op.save_image(final, output_path, quality=95)

        assert output_path.exists()
        assert final.size == test_image_rgb.size

    @pytest.mark.slow
    def test_all_operations_registered(self) -> None:
        """Test that all new operations are properly registered."""
        # Verify all operations can be retrieved
        compression_op = get_operation("compression")
        downscale_op = get_operation("downscale")
        blur_op = get_operation("motion_blur")
        multi_compress_op = get_operation("multi_compress")

        # Verify they have correct names
        assert compression_op.name == "compression"
        assert downscale_op.name == "downscale"
        assert blur_op.name == "motion_blur"
        assert multi_compress_op.name == "multi_compress"

    @pytest.mark.slow
    def test_different_subsampling_modes(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test all subsampling modes produce different block artifacts."""
        compression_op = get_operation("compression")

        # Test all three subsampling modes
        for subsampling, name in [
            (0, "no_subsampling"),
            (1, "moderate_subsampling"),
            (2, "heavy_subsampling"),
        ]:
            result = compression_op.apply(
                test_image_rgb,
                {"quality": 30, "subsampling": subsampling},
            )
            output_path = temp_output_dir / f"subsampling_{name}.jpg"
            compression_op.save_image(result, output_path, quality=95)

            assert output_path.exists()
            assert result.size == test_image_rgb.size

    @pytest.mark.slow
    def test_minimal_to_extreme_blur_range(
        self, test_image_rgb: Image.Image, temp_output_dir: Path
    ) -> None:
        """Test full range of blur from minimal to extreme."""
        blur_op = get_operation("motion_blur")

        for kernel_size, name in [(2, "minimal"), (10, "moderate"), (30, "heavy")]:
            result = blur_op.apply(
                test_image_rgb,
                {"kernel_size": kernel_size, "angle": 45},
            )
            output_path = temp_output_dir / f"blur_{name}.jpg"
            blur_op.save_image(result, output_path, quality=95)

            assert output_path.exists()
            assert result.size == test_image_rgb.size
