"""
Integration tests for backend selection functionality.

Tests that backend configuration works correctly through the pipeline.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image
from sevenrad_stills.operations import (
    BackendNotAvailableError,
    get_backend_implementation,
)
from sevenrad_stills.pipeline.models import PipelineConfig
from sevenrad_stills.utils.exceptions import PipelineError


class TestBackendRegistration:
    """Test that backends are properly registered."""

    def test_cpu_backend_always_available(self) -> None:
        """All operations should have CPU backend."""
        operations = [
            "band_swap",
            "bayer_filter",
            "blur_circular",
            "blur_gaussian",
            "buffer_corruption",
            "chromatic_aberration",
            "compression",
            "compression_artifact",
            "corduroy",
            "downscale",
            "motion_blur",
            "multi_compress",
            "noise",
            "salt_pepper",
            "saturation",
            "slc_off",
        ]

        for op in operations:
            operation = get_backend_implementation(op, "cpu")
            assert operation is not None
            # Operation should have the base name in its name
            assert op in operation.name

    def test_gpu_backend_for_supported_operations(self) -> None:
        """GPU backend should be available for most operations."""
        gpu_operations = [
            "band_swap",
            "bayer_filter",
            "blur_circular",
            "blur_gaussian",
            "buffer_corruption",
            "chromatic_aberration",
            "compression",
            "compression_artifact",
            "corduroy",
            "downscale",
            "motion_blur",
            "noise",
            "salt_pepper",
            "saturation",
            "slc_off",
        ]

        for op in gpu_operations:
            operation = get_backend_implementation(op, "gpu")
            assert operation is not None
            # Operation should have the base name in its name
            assert op in operation.name

    @pytest.mark.mac
    def test_metal_backend_for_supported_operations(self) -> None:
        """Metal backend should be available for supported operations on macOS."""
        metal_operations = [
            "bayer_filter",
            "compression",
            "compression_artifact",
            "corduroy",
            "downscale",
            "motion_blur",
            "noise",
            "salt_pepper",
            "saturation",
            "slc_off",
        ]

        for op in metal_operations:
            operation = get_backend_implementation(op, "metal")
            assert operation is not None
            # Operation should have the base name in its name
            assert op in operation.name

    def test_unsupported_backend_raises_error(self) -> None:
        """Requesting unsupported backend should raise error."""
        with pytest.raises(BackendNotAvailableError) as exc_info:
            get_backend_implementation("multi_compress", "gpu")

        assert "multi_compress" in str(exc_info.value)
        assert "gpu" in str(exc_info.value)
        assert "cpu" in str(exc_info.value).lower()


class TestPipelineConfiguration:
    """Test backend configuration in pipeline models."""

    def test_default_backend_is_cpu(self, tmp_path: Path) -> None:
        """Pipeline should default to CPU backend if not specified."""
        # Test YAML would normally not include backend field
        config = PipelineConfig.model_validate(
            {
                "source": {"youtube_url": "https://www.youtube.com/watch?v=test"},
                "segment": {"start": 0.0, "end": 1.0, "interval": 0.5},
                "pipeline": {
                    "steps": [
                        {
                            "name": "test",
                            "operation": "saturation",
                            "params": {"factor": 1.5},
                        }
                    ]
                },
            }
        )

        assert config.backend == "cpu"

    def test_backend_can_be_set_to_gpu(self, tmp_path: Path) -> None:
        """Backend can be configured as GPU."""
        config = PipelineConfig.model_validate(
            {
                "source": {"youtube_url": "https://www.youtube.com/watch?v=test"},
                "backend": "gpu",
                "segment": {"start": 0.0, "end": 1.0, "interval": 0.5},
                "pipeline": {
                    "steps": [
                        {
                            "name": "test",
                            "operation": "saturation",
                            "params": {"factor": 1.5},
                        }
                    ]
                },
            }
        )

        assert config.backend == "gpu"

    @pytest.mark.mac
    def test_backend_can_be_set_to_metal(self, tmp_path: Path) -> None:
        """Backend can be configured as Metal."""
        config = PipelineConfig.model_validate(
            {
                "source": {"youtube_url": "https://www.youtube.com/watch?v=test"},
                "backend": "metal",
                "segment": {"start": 0.0, "end": 1.0, "interval": 0.5},
                "pipeline": {
                    "steps": [
                        {
                            "name": "test",
                            "operation": "saturation",
                            "params": {"factor": 1.5},
                        }
                    ]
                },
            }
        )

        assert config.backend == "metal"

    def test_invalid_backend_rejected(self, tmp_path: Path) -> None:
        """Invalid backend should be rejected by Pydantic."""
        with pytest.raises(
            ValueError, match=r"Input should be 'cpu', 'gpu' or 'metal'"
        ):
            PipelineConfig.model_validate(
                {
                    "source": {"youtube_url": "https://www.youtube.com/watch?v=test"},
                    "backend": "cuda",  # Invalid - only cpu/gpu/metal allowed
                    "segment": {"start": 0.0, "end": 1.0, "interval": 0.5},
                    "pipeline": {"steps": []},
                }
            )


class TestBackendExecution:
    """Test that operations execute with correct backend."""

    def test_cpu_backend_executes_saturation(self) -> None:
        """Test CPU backend execution with saturation operation."""
        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            test_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Get CPU operation
            operation = get_backend_implementation("saturation", "cpu")

            # Apply operation with correct parameters
            image = Image.open(tmp_path)
            result = operation.apply(image, {"mode": "fixed", "value": 1.5})

            assert result is not None
            assert result.size == (100, 100)
            assert result.mode == "RGB"
        finally:
            tmp_path.unlink()

    def test_gpu_backend_executes_saturation(self) -> None:
        """Test GPU backend execution with saturation operation."""
        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            test_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Get GPU operation
            operation = get_backend_implementation("saturation", "gpu")

            # Apply operation with correct parameters
            image = Image.open(tmp_path)
            result = operation.apply(image, {"mode": "fixed", "value": 1.5})

            assert result is not None
            assert result.size == (100, 100)
            assert result.mode == "RGB"
        finally:
            tmp_path.unlink()

    @pytest.mark.mac
    def test_metal_backend_executes_saturation(self) -> None:
        """Test Metal backend execution with saturation operation."""
        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            test_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Get Metal operation
            operation = get_backend_implementation("saturation", "metal")

            # Apply operation with correct parameters
            image = Image.open(tmp_path)
            result = operation.apply(image, {"mode": "fixed", "value": 1.5})

            assert result is not None
            assert result.size == (100, 100)
            assert result.mode == "RGB"
        finally:
            tmp_path.unlink()
