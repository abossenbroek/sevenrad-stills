#!/usr/bin/env python3
"""
Generate reference images for Max effect validation.

This script creates test images using the Taichi operations with known parameters.
Each test case produces: (input_image, params, expected_output)

The generated reference images can be compared against Max output to verify
that the GenExpr shaders produce identical results (PSNR > 40dB).

Usage:
    python generate_references.py [--output-dir tests/reference]
"""
# ruff: noqa: T201 PLR2004

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sevenrad_stills.operations import (  # noqa: E402
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

# Test cases for each effect
# Each entry: {"params": {...}, "description": "..."}
TEST_CASES: dict[str, list[dict[str, Any]]] = {
    "noise": [
        {
            "params": {"mode": "gaussian", "amount": 0.1, "seed": 42},
            "description": "Light Gaussian noise",
        },
        {
            "params": {"mode": "gaussian", "amount": 0.5, "seed": 42},
            "description": "Heavy Gaussian noise",
        },
        {
            "params": {"mode": "row", "amount": 0.2, "seed": 123},
            "description": "Horizontal scanlines",
        },
        {
            "params": {"mode": "column", "amount": 0.3, "seed": 456},
            "description": "Vertical artifacts",
        },
    ],
    "saturation": [
        {"params": {"factor": 0.0}, "description": "Complete grayscale"},
        {"params": {"factor": 0.5}, "description": "50% desaturated"},
        {"params": {"factor": 1.5}, "description": "Moderately boosted"},
        {"params": {"factor": 2.0}, "description": "Heavily saturated"},
    ],
    "chromatic": [
        {
            "params": {"shift_x": 5.0, "shift_y": 0.0},
            "description": "Horizontal RGB fringing",
        },
        {
            "params": {"shift_x": 0.0, "shift_y": 5.0},
            "description": "Vertical RGB fringing",
        },
        {
            "params": {"shift_x": 3.0, "shift_y": 3.0},
            "description": "Diagonal fringing",
        },
        {
            "params": {"shift_x": 10.0, "shift_y": -5.0},
            "description": "Asymmetric fringing",
        },
    ],
    "blur": [
        {"params": {"sigma": 1.0}, "description": "Subtle blur"},
        {"params": {"sigma": 5.0}, "description": "Medium blur"},
        {"params": {"sigma": 15.0}, "description": "Heavy blur"},
        {"params": {"sigma": 0.0}, "description": "No change (passthrough)"},
    ],
    "blur_circular": [
        {"params": {"radius": 3}, "description": "Small disk blur"},
        {"params": {"radius": 7}, "description": "Medium disk blur"},
        {"params": {"radius": 15}, "description": "Large disk blur"},
    ],
    "motion": [
        {
            "params": {"kernel_size": 15, "angle": 0.0},
            "description": "Horizontal motion",
        },
        {
            "params": {"kernel_size": 15, "angle": 90.0},
            "description": "Vertical motion",
        },
        {
            "params": {"kernel_size": 25, "angle": 45.0},
            "description": "Diagonal motion",
        },
        {
            "params": {"kernel_size": 30, "angle": 135.0},
            "description": "Strong diagonal",
        },
    ],
    "saltpepper": [
        {
            "params": {"amount": 0.01, "salt_ratio": 0.5, "seed": 42},
            "description": "Light noise",
        },
        {
            "params": {"amount": 0.05, "salt_ratio": 0.5, "seed": 42},
            "description": "Medium noise",
        },
        {
            "params": {"amount": 0.1, "salt_ratio": 0.8, "seed": 42},
            "description": "More salt",
        },
        {
            "params": {"amount": 0.1, "salt_ratio": 0.2, "seed": 42},
            "description": "More pepper",
        },
    ],
    "corduroy": [
        {
            "params": {"orientation": 0, "strength": 0.3, "density": 0.2, "seed": 42},
            "description": "Vertical stripes",
        },
        {
            "params": {"orientation": 1, "strength": 0.3, "density": 0.2, "seed": 42},
            "description": "Horizontal stripes",
        },
        {
            "params": {"orientation": 0, "strength": 0.5, "density": 0.5, "seed": 123},
            "description": "Dense stripes",
        },
    ],
    "bayer": [
        {"params": {"pattern": "RGGB"}, "description": "RGGB pattern"},
        {"params": {"pattern": "BGGR"}, "description": "BGGR pattern"},
        {"params": {"pattern": "GRBG"}, "description": "GRBG pattern"},
        {"params": {"pattern": "GBRG"}, "description": "GBRG pattern"},
    ],
    "bandswap": [
        {
            "params": {"tile_count": 3, "permutation": "BGR", "seed": 42},
            "description": "BGR swap",
        },
        {
            "params": {"tile_count": 5, "permutation": "GRB", "seed": 123},
            "description": "GRB swap",
        },
        {
            "params": {"tile_count": 8, "permutation": "RBG", "seed": 456},
            "description": "RBG swap",
        },
    ],
    "corruption": [
        {
            "params": {"mode": "xor", "intensity": 0.5, "tile_count": 5, "seed": 42},
            "description": "XOR corruption",
        },
        {
            "params": {"mode": "invert", "intensity": 0.5, "tile_count": 5, "seed": 42},
            "description": "Invert corruption",
        },
        {
            "params": {
                "mode": "shuffle",
                "intensity": 0.5,
                "tile_count": 5,
                "seed": 42,
            },
            "description": "Shuffle corruption",
        },
    ],
    "downscale": [
        {
            "params": {"scale": 0.5, "pixelate": True},
            "description": "Half resolution pixelated",
        },
        {
            "params": {"scale": 0.25, "pixelate": True},
            "description": "Quarter resolution pixelated",
        },
        {"params": {"scale": 0.1, "pixelate": True}, "description": "Heavy pixelation"},
    ],
    "slcoff": [
        {
            "params": {"gap_width": 0.1, "scan_period": 16, "fill_mode": "black"},
            "description": "Black fill",
        },
        {
            "params": {"gap_width": 0.15, "scan_period": 8, "fill_mode": "white"},
            "description": "White fill",
        },
    ],
}

# Operation class mapping
OPERATIONS: dict[str, type[Any]] = {
    "noise": NoiseTaichiOperation,
    "saturation": SaturationTaichiOperation,
    "chromatic": ChromaticAberrationTaichiOperation,
    "blur": BlurGaussianTaichiOperation,
    "blur_circular": BlurCircularTaichiOperation,
    "motion": MotionBlurTaichiOperation,
    "saltpepper": SaltPepperTaichiOperation,
    "corduroy": CorduroyTaichiOperation,
    "bayer": BayerFilterTaichiOperation,
    "bandswap": BandSwapTaichiOperation,
    "corruption": BufferCorruptionTaichiOperation,
    "downscale": DownscaleTaichiOperation,
    "slcoff": SlcOffTaichiOperation,
}


def create_test_image(width: int = 512, height: int = 512) -> NDArray[np.floating[Any]]:
    """Create a colorful test image with gradients and patterns."""
    img: NDArray[np.floating[Any]] = np.zeros((height, width, 3), dtype=np.float32)

    # Horizontal gradient (red)
    for x in range(width):
        img[:, x, 0] = x / width

    # Vertical gradient (green)
    for y in range(height):
        img[y, :, 1] = y / height

    # Diagonal gradient (blue)
    for y in range(height):
        for x in range(width):
            img[y, x, 2] = (x + y) / (width + height)

    # Add some circular patterns
    cy, cx = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < min(width, height) // 4:
                # Bright center circle
                factor = 1.0 - dist / (min(width, height) // 4)
                img[y, x] = img[y, x] * 0.5 + 0.5 * factor

    return np.clip(img, 0.0, 1.0)


def generate_all_references(
    output_dir: Path, input_image: NDArray[np.floating[Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Generate reference images for all test cases."""
    results: dict[str, list[dict[str, Any]]] = {}

    for effect_name, cases in TEST_CASES.items():
        if effect_name not in OPERATIONS:
            print(f"  Skipping {effect_name}: operation not available")
            continue

        op_class = OPERATIONS[effect_name]

        try:
            op = op_class()
        except Exception as e:
            print(f"  Skipping {effect_name}: {e}")
            continue

        results[effect_name] = []

        for i, case in enumerate(cases):
            params = case["params"]
            description = case["description"]

            try:
                # Use NumPy reference implementation
                output = op.reference_numpy(input_image.copy(), params)

                # Save output image
                output_path = output_dir / f"{effect_name}_{i:03d}.png"
                output_uint8 = (output * 255).astype(np.uint8)
                Image.fromarray(output_uint8).save(output_path)

                # Save parameters
                params_path = output_dir / f"{effect_name}_{i:03d}.json"
                with open(params_path, "w") as f:
                    json.dump(
                        {
                            "effect": effect_name,
                            "case_id": i,
                            "description": description,
                            "params": params,
                        },
                        f,
                        indent=2,
                    )

                results[effect_name].append(
                    {
                        "case_id": i,
                        "description": description,
                        "output_path": str(output_path),
                        "params_path": str(params_path),
                    }
                )

                print(f"  Generated {effect_name}_{i:03d}: {description}")

            except Exception as e:
                print(f"  ERROR {effect_name}_{i:03d}: {e}")

    return results


def main() -> None:
    """Generate reference images from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate reference images for Max effect validation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "reference",
        help="Output directory for reference images",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="Custom input image (default: generated test pattern)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of generated test image (default: 512)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create input image
    if args.input_image and args.input_image.exists():
        print(f"Loading input image: {args.input_image}")
        input_pil = Image.open(args.input_image).convert("RGB")
        input_image = np.array(input_pil).astype(np.float32) / 255.0
    else:
        print(f"Generating test image ({args.size}x{args.size})")
        input_image = create_test_image(args.size, args.size)

        # Save test input
        input_path = args.output_dir.parent / "input" / "test_image.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_uint8 = (input_image * 255).astype(np.uint8)
        Image.fromarray(input_uint8).save(input_path)
        print(f"Saved test input to: {input_path}")

    print(f"\nGenerating reference images to: {args.output_dir}")
    results = generate_all_references(args.output_dir, input_image)

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")

    # Summary
    total = sum(len(cases) for cases in results.values())
    print(f"\nGenerated {total} reference images for {len(results)} effects")


if __name__ == "__main__":
    main()
