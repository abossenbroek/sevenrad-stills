"""
Profiling script for buffer corruption GPU performance analysis.

Identifies bottlenecks in CPU preprocessing, GPU computation, and memory transfer.
"""

import time
from typing import Any

import numpy as np
from PIL import Image
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.buffer_corruption_gpu import (
    BufferCorruptionGPUOperation,
)


def profile_detailed(
    operation: BufferCorruptionGPUOperation,
    image: Image.Image,
    params: dict[str, Any],
    warmup: int = 2,
    runs: int = 10,
) -> dict[str, float]:
    """Profile GPU operation with detailed timing breakdown."""
    # Warmup
    for _ in range(warmup):
        _ = operation.apply(image, params)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = operation.apply(image, params)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def profile_cpu(
    operation: BufferCorruptionOperation,
    image: Image.Image,
    params: dict[str, Any],
    runs: int = 10,
) -> dict[str, float]:
    """Profile CPU operation."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = operation.apply(image, params)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def main() -> None:
    """Run comprehensive profiling."""
    print("=" * 80)
    print("Buffer Corruption GPU Performance Analysis")
    print("=" * 80)

    # Test configurations
    image_sizes = [
        (1920, 1080, "FHD"),
        (3840, 2160, "4K"),
        (7680, 4320, "8K"),
    ]

    corruption_types = ["xor", "invert", "channel_shuffle"]

    gpu_op = BufferCorruptionGPUOperation()
    cpu_op = BufferCorruptionOperation()

    for width, height, name in image_sizes:
        print(f"\n{'=' * 80}")
        print(f"Image Size: {name} ({width}x{height})")
        print(f"{'=' * 80}")

        image = Image.new("RGB", (width, height), color=(128, 128, 128))

        for corruption_type in corruption_types:
            params = {
                "tile_count": 20,
                "corruption_type": corruption_type,
                "severity": 0.8,
                "seed": 42,
            }

            print(f"\n{corruption_type.upper():^80}")
            print("-" * 80)

            # CPU baseline
            cpu_stats = profile_cpu(cpu_op, image, params, runs=5)
            print(
                f"CPU:  {cpu_stats['mean']*1000:6.2f}ms ± "
                f"{cpu_stats['std']*1000:5.2f}ms "
                f"(min: {cpu_stats['min']*1000:6.2f}ms, "
                f"max: {cpu_stats['max']*1000:6.2f}ms)"
            )

            # GPU
            gpu_stats = profile_detailed(gpu_op, image, params, warmup=2, runs=5)
            print(
                f"GPU:  {gpu_stats['mean']*1000:6.2f}ms ± "
                f"{gpu_stats['std']*1000:5.2f}ms "
                f"(min: {gpu_stats['min']*1000:6.2f}ms, "
                f"max: {gpu_stats['max']*1000:6.2f}ms)"
            )

            # Speedup
            speedup = cpu_stats["mean"] / gpu_stats["mean"]
            if speedup > 1.0:
                print(f"Speedup: {speedup:.2f}x FASTER ✓")
            else:
                print(f"Speedup: {speedup:.2f}x (SLOWER) ✗")

    print("\n" + "=" * 80)
    print("Profile Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
