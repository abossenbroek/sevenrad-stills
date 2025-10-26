"""Benchmark CPU vs GPU band swap operations."""

import time
from typing import Any

import numpy as np
from PIL import Image
from sevenrad_stills.operations.band_swap import BandSwapOperation
from sevenrad_stills.operations.band_swap_gpu import BandSwapGPUOperation


def benchmark_operation(
    operation: Any,  # noqa: ANN401
    image: Image.Image,
    params: dict[str, Any],
    warmup: int = 2,
    iterations: int = 10,
) -> tuple[float, float]:
    """
    Benchmark an operation.

    Args:
        operation: The operation to benchmark
        image: Test image
        params: Operation parameters
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Tuple of (mean_time, std_time) in milliseconds

    """
    # Warmup
    for _ in range(warmup):
        operation.apply(image, params)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation.apply(image, params)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return float(np.mean(times)), float(np.std(times))


def main() -> None:
    """Run benchmarks."""
    print("Band Swap Operation Benchmark: CPU vs GPU")  # noqa: T201
    print("=" * 60)  # noqa: T201

    # Test configurations
    configs = [
        ("Small (512x512)", (512, 512), 10),
        ("Medium (1024x1024)", (1024, 1024), 10),
        ("Large (2048x2048)", (2048, 2048), 10),
        ("Very Large (4096x4096)", (4096, 4096), 5),
    ]

    permutations = ["GRB", "BGR", "RBG"]

    cpu_op = BandSwapOperation()
    gpu_op = BandSwapGPUOperation()

    for name, size, tile_count in configs:
        print(f"\n{name} Image, {tile_count} tiles")  # noqa: T201
        print("-" * 60)  # noqa: T201

        # Create test image
        test_image = Image.new("RGB", size, color=(100, 150, 200))

        for perm in permutations:
            params = {
                "tile_count": tile_count,
                "permutation": perm,
                "seed": 42,
            }

            # Benchmark CPU
            cpu_mean, cpu_std = benchmark_operation(cpu_op, test_image, params)

            # Benchmark GPU
            gpu_mean, gpu_std = benchmark_operation(gpu_op, test_image, params)

            speedup = cpu_mean / gpu_mean
            speedup_str = (
                f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
            )

            print(f"  {perm}:")  # noqa: T201
            print(f"    CPU: {cpu_mean:7.2f} ± {cpu_std:5.2f} ms")  # noqa: T201
            print(f"    GPU: {gpu_mean:7.2f} ± {gpu_std:5.2f} ms")  # noqa: T201
            print(f"    Speedup: {speedup_str}")  # noqa: T201

    print("\n" + "=" * 60)  # noqa: T201
    print("Benchmark complete!")  # noqa: T201


if __name__ == "__main__":
    main()
