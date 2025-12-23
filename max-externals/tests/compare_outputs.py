#!/usr/bin/env python3
"""
Compare Max output images against Python reference images.

This script validates that the Max GenExpr shaders produce results
matching the Taichi/NumPy reference implementations.

Tolerance: PSNR > 40dB (allows for float→char rounding differences)

Usage:
    python compare_outputs.py tests/reference tests/actual [--threshold 40]
"""
# ruff: noqa: T201 PLR2004

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using simple PSNR calculation.")


def calculate_psnr(reference: NDArray[np.uint8], actual: NDArray[np.uint8]) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    if SKIMAGE_AVAILABLE:
        return float(peak_signal_noise_ratio(reference, actual, data_range=255))

    # Simple MSE-based PSNR
    mse = np.mean((reference.astype(np.float64) - actual.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(reference: NDArray[np.uint8], actual: NDArray[np.uint8]) -> float:
    """Calculate Structural Similarity Index between two images."""
    if not SKIMAGE_AVAILABLE:
        return 1.0  # Skip SSIM if skimage not available

    ndim_color = 3  # Number of dimensions for color images
    # Convert to grayscale for SSIM if color
    if len(reference.shape) == ndim_color:
        return float(
            structural_similarity(reference, actual, channel_axis=2, data_range=255)
        )
    return float(structural_similarity(reference, actual, data_range=255))


def calculate_max_diff(reference: NDArray[np.uint8], actual: NDArray[np.uint8]) -> int:
    """Calculate maximum pixel difference."""
    diff = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    return int(np.max(diff))


def compare_images(reference_path: Path, actual_path: Path) -> dict[str, Any]:
    """Compare two images and return metrics."""
    try:
        ref_img = np.array(Image.open(reference_path).convert("RGB"))
        act_img = np.array(Image.open(actual_path).convert("RGB"))
    except Exception as e:
        return {
            "error": str(e),
            "passed": False,
        }

    # Check dimensions match
    if ref_img.shape != act_img.shape:
        return {
            "error": f"Shape mismatch: {ref_img.shape} vs {act_img.shape}",
            "passed": False,
        }

    psnr = calculate_psnr(ref_img, act_img)
    ssim = calculate_ssim(ref_img, act_img)
    max_diff = calculate_max_diff(ref_img, act_img)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "max_pixel_diff": max_diff,
        "passed": True,  # Will be evaluated against thresholds later
    }


def run_comparison(
    reference_dir: Path,
    actual_dir: Path,
    psnr_threshold: float = 40.0,
    ssim_threshold: float = 0.99,
    max_diff_threshold: int = 2,
) -> dict[str, Any]:
    """Compare all images in reference directory against actual directory."""
    results: dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "missing": 0,
        "details": [],
    }

    # Find all reference images
    reference_images = sorted(reference_dir.glob("*.png"))

    for ref_path in reference_images:
        name = ref_path.stem
        act_path = actual_dir / ref_path.name

        # Load params if available
        params_path = reference_dir / f"{name}.json"
        params: dict[str, Any] = {}
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)

        if not act_path.exists():
            results["missing"] = int(results["missing"]) + 1
            results["details"].append(
                {
                    "name": name,
                    "status": "missing",
                    "description": params.get("description", ""),
                }
            )
            continue

        metrics = compare_images(ref_path, act_path)

        if "error" in metrics:
            results["failed"] = int(results["failed"]) + 1
            results["details"].append(
                {
                    "name": name,
                    "status": "error",
                    "error": metrics["error"],
                    "description": params.get("description", ""),
                }
            )
            continue

        # Check thresholds
        psnr_pass = metrics["psnr"] >= psnr_threshold or metrics["psnr"] == float("inf")
        ssim_pass = metrics["ssim"] >= ssim_threshold
        diff_pass = metrics["max_pixel_diff"] <= max_diff_threshold

        all_pass = psnr_pass and ssim_pass and diff_pass

        if all_pass:
            results["passed"] = int(results["passed"]) + 1
            status = "passed"
        else:
            results["failed"] = int(results["failed"]) + 1
            status = "failed"

        results["details"].append(
            {
                "name": name,
                "status": status,
                "psnr": round(metrics["psnr"], 2)
                if metrics["psnr"] != float("inf")
                else "inf",
                "ssim": round(metrics["ssim"], 4),
                "max_pixel_diff": metrics["max_pixel_diff"],
                "psnr_pass": psnr_pass,
                "ssim_pass": ssim_pass,
                "diff_pass": diff_pass,
                "description": params.get("description", ""),
            }
        )

    return results


def print_results(results: dict[str, Any]) -> bool:
    """Print comparison results in a readable format."""
    total = int(results["passed"]) + int(results["failed"]) + int(results["missing"])

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nSummary: {results['passed']}/{total} passed")
    print(f"  Passed:  {results['passed']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Missing: {results['missing']}")

    if results["details"]:
        print("\nDetails:")
        print("-" * 70)

        for detail in results["details"]:
            status_icon = {
                "passed": "✓",
                "failed": "✗",
                "missing": "?",
                "error": "!",
            }.get(detail["status"], "?")

            print(f"\n{status_icon} {detail['name']}")
            if detail.get("description"):
                print(f"  Description: {detail['description']}")

            if detail["status"] in ("passed", "failed"):
                psnr_mark = "✓" if detail.get("psnr_pass") else "✗"
                ssim_mark = "✓" if detail.get("ssim_pass") else "✗"
                diff_mark = "✓" if detail.get("diff_pass") else "✗"

                print(f"  PSNR: {detail['psnr']} dB {psnr_mark}")
                print(f"  SSIM: {detail['ssim']} {ssim_mark}")
                print(f"  Max pixel diff: {detail['max_pixel_diff']} {diff_mark}")

            elif detail["status"] == "missing":
                print("  Actual image not found")

            elif detail["status"] == "error":
                print(f"  Error: {detail.get('error', 'Unknown')}")

    print("\n" + "=" * 70)

    if results["failed"] > 0 or results["missing"] > 0:
        print("RESULT: FAIL")
        return False
    print("RESULT: PASS")
    return True


def main() -> None:
    """Run comparison from command line."""
    parser = argparse.ArgumentParser(
        description="Compare Max output against Python reference"
    )
    parser.add_argument(
        "reference_dir", type=Path, help="Directory with reference images"
    )
    parser.add_argument(
        "actual_dir", type=Path, help="Directory with Max output images"
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=40.0,
        help="Minimum PSNR in dB (default: 40)",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.99,
        help="Minimum SSIM (default: 0.99)",
    )
    parser.add_argument(
        "--max-diff", type=int, default=2, help="Maximum pixel difference (default: 2)"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if not args.reference_dir.exists():
        print(f"Error: Reference directory not found: {args.reference_dir}")
        sys.exit(1)

    if not args.actual_dir.exists():
        print(f"Error: Actual directory not found: {args.actual_dir}")
        sys.exit(1)

    results = run_comparison(
        args.reference_dir,
        args.actual_dir,
        psnr_threshold=args.psnr_threshold,
        ssim_threshold=args.ssim_threshold,
        max_diff_threshold=args.max_diff,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        success = print_results(results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
