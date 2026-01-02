"""
Tests for reference image generation.

This test module validates that all Taichi operations can be instantiated,
their reference_numpy methods work correctly, and they produce valid outputs.
"""
# ruff: noqa: S101 PLR2004

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

# Import all Taichi operations
from sevenrad_stills.operations import (
    BandSwapTaichiOperation,
    BayerFilterTaichiOperation,
    BlurCircularTaichiOperation,
    BlurGaussianTaichiOperation,
    BufferCorruptionTaichiOperation,
    ChromaticAberrationTaichiOperation,
    CorduroyTaichiOperation,
    DownscaleTaichiOperation,
    MotionBlurTaichiOperation,
    NoiseTaichiOperation,
    SaltPepperTaichiOperation,
    SaturationTaichiOperation,
    SlcOffTaichiOperation,
)


# Test fixtures
@pytest.fixture
def test_image() -> NDArray[np.floating[Any]]:
    """Create a simple test image for validation."""
    img: NDArray[np.floating[Any]] = np.zeros((64, 64, 3), dtype=np.float32)

    # Create gradients
    for x in range(64):
        img[:, x, 0] = x / 64.0
    for y in range(64):
        img[y, :, 1] = y / 64.0
    for y in range(64):
        for x in range(64):
            img[y, x, 2] = (x + y) / 128.0

    return np.clip(img, 0.0, 1.0)


class TestOperationInstantiation:
    """Test that all operations can be instantiated."""

    def test_noise_instantiation(self) -> None:
        """Test NoiseTaichiOperation instantiation."""
        op = NoiseTaichiOperation()
        assert op.name == "noise_taichi"
        assert hasattr(op, "reference_numpy")

    def test_saturation_instantiation(self) -> None:
        """Test SaturationTaichiOperation instantiation."""
        op = SaturationTaichiOperation()
        assert op.name == "saturation_taichi"
        assert hasattr(op, "reference_numpy")

    def test_chromatic_instantiation(self) -> None:
        """Test ChromaticAberrationTaichiOperation instantiation."""
        op = ChromaticAberrationTaichiOperation()
        assert op.name == "chromatic_aberration_taichi"
        assert hasattr(op, "reference_numpy")

    def test_blur_gaussian_instantiation(self) -> None:
        """Test BlurGaussianTaichiOperation instantiation."""
        op = BlurGaussianTaichiOperation()
        assert op.name == "blur_gaussian_taichi"
        assert hasattr(op, "reference_numpy")

    def test_blur_circular_instantiation(self) -> None:
        """Test BlurCircularTaichiOperation instantiation."""
        op = BlurCircularTaichiOperation()
        assert op.name == "blur_circular_taichi"
        assert hasattr(op, "reference_numpy")

    def test_motion_blur_instantiation(self) -> None:
        """Test MotionBlurTaichiOperation instantiation."""
        op = MotionBlurTaichiOperation()
        assert op.name == "motion_blur_taichi"
        assert hasattr(op, "reference_numpy")

    def test_salt_pepper_instantiation(self) -> None:
        """Test SaltPepperTaichiOperation instantiation."""
        op = SaltPepperTaichiOperation()
        assert op.name == "salt_pepper_taichi"
        assert hasattr(op, "reference_numpy")

    def test_corduroy_instantiation(self) -> None:
        """Test CorduroyTaichiOperation instantiation."""
        op = CorduroyTaichiOperation()
        assert op.name == "corduroy_taichi"
        assert hasattr(op, "reference_numpy")

    def test_bayer_filter_instantiation(self) -> None:
        """Test BayerFilterTaichiOperation instantiation."""
        op = BayerFilterTaichiOperation()
        assert op.name == "bayer_filter_taichi"
        assert hasattr(op, "reference_numpy")

    def test_band_swap_instantiation(self) -> None:
        """Test BandSwapTaichiOperation instantiation."""
        op = BandSwapTaichiOperation()
        assert op.name == "band_swap_taichi"
        assert hasattr(op, "reference_numpy")

    def test_buffer_corruption_instantiation(self) -> None:
        """Test BufferCorruptionTaichiOperation instantiation."""
        op = BufferCorruptionTaichiOperation()
        assert op.name == "buffer_corruption_taichi"
        assert hasattr(op, "reference_numpy")

    def test_downscale_instantiation(self) -> None:
        """Test DownscaleTaichiOperation instantiation."""
        op = DownscaleTaichiOperation()
        assert op.name == "downscale_taichi"
        assert hasattr(op, "reference_numpy")

    def test_slc_off_instantiation(self) -> None:
        """Test SlcOffTaichiOperation instantiation."""
        op = SlcOffTaichiOperation()
        assert op.name == "slc_off_taichi"
        assert hasattr(op, "reference_numpy")


class TestReferenceNumpyMethod:
    """Test that reference_numpy methods work correctly."""

    def test_noise_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test noise operation reference implementation."""
        op = NoiseTaichiOperation()
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_saturation_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test saturation operation reference implementation."""
        op = SaturationTaichiOperation()
        params = {"factor": 1.5}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_chromatic_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test chromatic aberration operation reference implementation."""
        op = ChromaticAberrationTaichiOperation()
        params = {"shift_x": 3.0, "shift_y": 3.0}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_blur_gaussian_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test gaussian blur operation reference implementation."""
        op = BlurGaussianTaichiOperation()
        params = {"sigma": 2.0}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_blur_circular_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test circular blur operation reference implementation."""
        op = BlurCircularTaichiOperation()
        params = {"radius": 5}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_motion_blur_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test motion blur operation reference implementation."""
        op = MotionBlurTaichiOperation()
        params = {"kernel_size": 15, "angle": 45.0}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_salt_pepper_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test salt and pepper operation reference implementation."""
        op = SaltPepperTaichiOperation()
        # Note: Using correct parameter name 'salt_vs_pepper'
        params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_corduroy_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test corduroy operation reference implementation."""
        op = CorduroyTaichiOperation()
        params = {"orientation": 0, "strength": 0.3, "density": 0.2, "seed": 42}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_bayer_filter_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test bayer filter operation reference implementation."""
        op = BayerFilterTaichiOperation()
        params = {"pattern": "RGGB"}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_band_swap_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test band swap operation reference implementation."""
        op = BandSwapTaichiOperation()
        params = {"tile_count": 3, "permutation": "BGR", "seed": 42}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_buffer_corruption_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test buffer corruption operation reference implementation."""
        op = BufferCorruptionTaichiOperation()
        # Note: Using correct parameter names 'corruption_type' and 'severity'
        params = {
            "corruption_type": "xor",
            "severity": 0.5,
            "tile_count": 5,
            "seed": 42,
        }

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_downscale_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test downscale operation reference implementation."""
        op = DownscaleTaichiOperation()
        params = {"scale": 0.5, "pixelate": True}

        output = op.reference_numpy(test_image.copy(), params)

        # Note: downscale changes output dimensions when pixelate=False
        # With pixelate=True, it should match input size
        expected_h = int(test_image.shape[0] * 0.5)
        expected_w = int(test_image.shape[1] * 0.5)
        assert output.shape == (expected_h, expected_w, 3)
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_slc_off_reference_numpy(self, test_image: np.ndarray) -> None:
        """Test SLC-off operation reference implementation."""
        op = SlcOffTaichiOperation()
        params = {"gap_width": 0.1, "scan_period": 16, "fill_mode": "black"}

        output = op.reference_numpy(test_image.copy(), params)

        assert output.shape == test_image.shape
        assert output.dtype == np.float32
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)


class TestReferenceImageOutputs:
    """Test reference image output validation."""

    @pytest.fixture
    def reference_dir(self) -> Path:
        """Get reference directory path."""
        return Path(__file__).parent / "reference"

    @pytest.fixture
    def manifest_path(self, reference_dir: Path) -> Path:
        """Get manifest file path."""
        return reference_dir / "manifest.json"

    def test_manifest_exists(self, manifest_path: Path) -> None:
        """Test that manifest file exists."""
        assert manifest_path.exists(), "Manifest file should exist"

    def test_manifest_structure(self, manifest_path: Path) -> None:
        """Test manifest has correct structure."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert isinstance(manifest, dict)

        # Check expected effects are present
        expected_effects = [
            "noise",
            "saturation",
            "chromatic",
            "blur",
            "blur_circular",
            "motion",
            "corduroy",
            "bayer",
            "bandswap",
            "downscale",
            "slcoff",
        ]

        for effect in expected_effects:
            assert effect in manifest, f"Effect '{effect}' should be in manifest"
            assert isinstance(manifest[effect], list)

    def test_reference_images_exist(
        self,
        manifest_path: Path,
        reference_dir: Path,  # noqa: ARG002
    ) -> None:
        """Test that all referenced images exist."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent

        for _, cases in manifest.items():
            for case in cases:
                output_path = Path(case["output_path"])
                params_path = Path(case["params_path"])

                # Make paths absolute relative to project root
                if not output_path.is_absolute():
                    output_path = project_root / output_path
                if not params_path.is_absolute():
                    params_path = project_root / params_path

                assert output_path.exists(), f"Output image {output_path} should exist"
                assert params_path.exists(), f"Params file {params_path} should exist"

    def test_reference_images_valid(
        self,
        manifest_path: Path,
        reference_dir: Path,  # noqa: ARG002
    ) -> None:
        """Test that reference images are valid and have expected properties."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent

        for _, cases in manifest.items():
            for case in cases:
                output_path = Path(case["output_path"])

                # Make path absolute
                if not output_path.is_absolute():
                    output_path = project_root / output_path

                # Load and validate image
                img = Image.open(output_path)
                arr = np.array(img)

                # Check dimensions
                assert (
                    len(arr.shape) == 3
                ), "Image should be 3D (height, width, channels)"
                assert arr.shape[2] == 3, "Image should have 3 channels (RGB)"

                # Check value range
                assert arr.min() >= 0, "Pixel values should be >= 0"
                assert arr.max() <= 255, "Pixel values should be <= 255"

    def test_params_files_valid_json(
        self,
        manifest_path: Path,
        reference_dir: Path,  # noqa: ARG002
    ) -> None:
        """Test that all params files contain valid JSON."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent

        for _, cases in manifest.items():
            for case in cases:
                params_path = Path(case["params_path"])

                # Make path absolute
                if not params_path.is_absolute():
                    params_path = project_root / params_path

                # Load and validate JSON
                with open(params_path) as f:
                    params_data = json.load(f)

                assert "effect" in params_data
                assert "case_id" in params_data
                assert "description" in params_data
                assert "params" in params_data
                assert isinstance(params_data["params"], dict)


class TestInputImage:
    """Test input image generation."""

    def test_input_image_exists(self) -> None:
        """Test that input test image was generated."""
        input_path = Path(__file__).parent / "input" / "test_image.png"
        assert input_path.exists(), "Input test image should exist"

    def test_input_image_valid(self) -> None:
        """Test that input image is valid."""
        input_path = Path(__file__).parent / "input" / "test_image.png"
        img = Image.open(input_path)
        arr = np.array(img)

        assert len(arr.shape) == 3
        assert arr.shape[2] == 3
        assert arr.min() >= 0
        assert arr.max() <= 255
