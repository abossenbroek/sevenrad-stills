"""Unit tests for Metal Performance Shaders Gaussian blur operation."""

import platform

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian_mps import GaussianBlurMPSOperation

# Mark all tests in this module as Mac-specific
pytestmark = [
    pytest.mark.mac,
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="MPS tests only run on Mac (Metal backend)",
    ),
]


@pytest.fixture
def blur_mps_op() -> GaussianBlurMPSOperation:
    """Fixture providing a GaussianBlurMPSOperation instance."""
    return GaussianBlurMPSOperation()


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Fixture providing a small test RGB image."""
    array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Fixture providing a small test RGBA image."""
    array = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    return Image.fromarray(array, mode="RGBA")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Fixture providing a small test grayscale image."""
    array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    return Image.fromarray(array, mode="L")


def test_validate_params_missing_sigma(blur_mps_op: GaussianBlurMPSOperation) -> None:
    """Test validation fails when sigma is missing."""
    with pytest.raises(ValueError, match="requires a 'sigma' parameter"):
        blur_mps_op.validate_params({})


def test_validate_params_invalid_sigma_type(
    blur_mps_op: GaussianBlurMPSOperation,
) -> None:
    """Test validation fails when sigma is not a number."""
    with pytest.raises(ValueError, match="Sigma must be a number"):
        blur_mps_op.validate_params({"sigma": "not_a_number"})


def test_validate_params_negative_sigma(blur_mps_op: GaussianBlurMPSOperation) -> None:
    """Test validation fails when sigma is negative."""
    with pytest.raises(ValueError, match="Sigma must be non-negative"):
        blur_mps_op.validate_params({"sigma": -1.0})


def test_validate_params_valid_sigma(blur_mps_op: GaussianBlurMPSOperation) -> None:
    """Test validation passes with valid sigma."""
    blur_mps_op.validate_params({"sigma": 2.0})


def test_apply_zero_sigma_returns_copy(
    blur_mps_op: GaussianBlurMPSOperation, sample_rgb_image: Image.Image
) -> None:
    """Test that sigma=0 returns a copy of the original image."""
    result = blur_mps_op.apply(sample_rgb_image, {"sigma": 0.0})
    assert result is not sample_rgb_image
    assert np.array_equal(np.array(result), np.array(sample_rgb_image))


def test_apply_blurs_rgb_image(
    blur_mps_op: GaussianBlurMPSOperation, sample_rgb_image: Image.Image
) -> None:
    """Test that blur is applied to RGB image."""
    result = blur_mps_op.apply(sample_rgb_image, {"sigma": 2.0})
    assert result.mode == "RGB"
    assert result.size == sample_rgb_image.size

    # Blurred image should be different from original
    assert not np.array_equal(np.array(result), np.array(sample_rgb_image))


def test_apply_blurs_rgba_image(
    blur_mps_op: GaussianBlurMPSOperation, sample_rgba_image: Image.Image
) -> None:
    """Test that blur is applied to RGBA image with alpha preserved."""
    original_alpha = np.array(sample_rgba_image)[:, :, 3]

    result = blur_mps_op.apply(sample_rgba_image, {"sigma": 2.0})
    assert result.mode == "RGBA"
    assert result.size == sample_rgba_image.size

    # Alpha channel should be unchanged
    result_alpha = np.array(result)[:, :, 3]
    assert np.array_equal(result_alpha, original_alpha)

    # RGB channels should be blurred
    original_rgb = np.array(sample_rgba_image)[:, :, :3]
    result_rgb = np.array(result)[:, :, :3]
    assert not np.array_equal(result_rgb, original_rgb)


def test_apply_blurs_grayscale_image(
    blur_mps_op: GaussianBlurMPSOperation, sample_grayscale_image: Image.Image
) -> None:
    """Test that blur is applied to grayscale image."""
    result = blur_mps_op.apply(sample_grayscale_image, {"sigma": 2.0})
    assert result.mode == "L"
    assert result.size == sample_grayscale_image.size

    # Blurred image should be different from original
    assert not np.array_equal(np.array(result), np.array(sample_grayscale_image))


def test_consistency_across_calls(
    blur_mps_op: GaussianBlurMPSOperation, sample_rgb_image: Image.Image
) -> None:
    """Test that multiple calls with same params produce identical results."""
    params = {"sigma": 3.0}

    result1 = blur_mps_op.apply(sample_rgb_image, params)
    result2 = blur_mps_op.apply(sample_rgb_image, params)

    # Results should be identical (MPS is deterministic)
    np.testing.assert_array_equal(np.array(result1), np.array(result2))


def test_larger_sigma_more_blur(
    blur_mps_op: GaussianBlurMPSOperation, sample_rgb_image: Image.Image
) -> None:
    """Test that larger sigma produces more blur."""
    result_small = blur_mps_op.apply(sample_rgb_image, {"sigma": 1.0})
    result_large = blur_mps_op.apply(sample_rgb_image, {"sigma": 5.0})

    # Both should be different from original
    assert not np.array_equal(np.array(result_small), np.array(sample_rgb_image))
    assert not np.array_equal(np.array(result_large), np.array(sample_rgb_image))

    # Larger sigma should produce different result than smaller sigma
    assert not np.array_equal(np.array(result_small), np.array(result_large))
