"""Benchmark optimized GPU implementation against CPU and original GPU."""

import time
from typing import Any

import numpy as np
from PIL import Image
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.buffer_corruption_gpu import (
    BufferCorruptionGPUOperation,
)
from sevenrad_stills.operations.buffer_corruption_gpu_optimized import (
    BufferCorruptionGPUOptimizedOperation,
)


def benchmark(
    operation: Any,
    image: Image.Image,
    params: dict[str, Any],
    warmup: int = 2,
    runs: int = 10,
) -> dict[str, float]:
    """Benchmark an operation."""
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
    }


def main() -> None:
    """Run benchmark comparison."""
    print("\n" + "=" * 90)
    print("GPU OPTIMIZATION BENCHMARK - Optimized vs Original GPU vs CPU")
    print("=" * 90)

    cpu_op = BufferCorruptionOperation()
    gpu_op = BufferCorruptionGPUOperation()
    gpu_opt_op = BufferCorruptionGPUOptimizedOperation()

    # Test configurations
    configs = [
        (1920, 1080, "FHD"),
        (3840, 2160, "4K"),
        (7680, 4320, "8K"),
    ]

    corruption_types = ["xor", "invert", "channel_shuffle"]

    for width, height, name in configs:
        print(f"\n{name} ({width}x{height})")
        print("=" * 90)

        image = Image.new("RGB", (width, height), color=(128, 128, 128))

        for corruption_type in corruption_types:
            params = {
                "tile_count": 20,
                "corruption_type": corruption_type,
                "severity": 0.8,
                "seed": 42,
            }

            print(f"\n  {corruption_type.upper()}")
            print("  " + "-" * 86)

            # Benchmark all three
            cpu_stats = benchmark(cpu_op, image, params, warmup=1, runs=5)
            gpu_stats = benchmark(gpu_op, image, params, warmup=2, runs=5)
            gpu_opt_stats = benchmark(gpu_opt_op, image, params, warmup=2, runs=5)

            cpu_ms = cpu_stats["mean"] * 1000
            gpu_ms = gpu_stats["mean"] * 1000
            gpu_opt_ms = gpu_opt_stats["mean"] * 1000

            print(f"  CPU:              {cpu_ms:7.2f}ms")
            print(f"  GPU (original):   {gpu_ms:7.2f}ms  ({cpu_ms/gpu_ms:4.2f}x)")
            print(
                f"  GPU (optimized):  {gpu_opt_ms:7.2f}ms  ({cpu_ms/gpu_opt_ms:4.2f}x)"
            )

            # Highlight improvements
            improvement = gpu_ms / gpu_opt_ms if gpu_opt_ms > 0 else 0
            if improvement > 1.1:
                print(f"  → OPTIMIZED {improvement:.2f}x FASTER than original GPU! ✓")
            elif gpu_opt_ms < cpu_ms:
                speedup = cpu_ms / gpu_opt_ms
                print(f"  → {speedup:.2f}x faster than CPU ✓")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
