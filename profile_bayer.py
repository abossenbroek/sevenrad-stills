"""Profile Bayer filter GPU implementation to identify bottlenecks."""
# ruff: noqa: T201

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from PIL import Image
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation


def create_test_image(size: tuple[int, int]) -> Image.Image:
    """Create a test RGB image."""
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def profile_size(size: tuple[int, int], runs: int = 10) -> None:
    """Profile a specific image size."""
    print(f"\n{'=' * 60}")
    print(f"Profiling {size[0]}x{size[1]} image ({runs} runs)")
    print(f"{'=' * 60}\n")

    operation = BayerFilterGPUOperation()
    image = create_test_image(size)
    params = {"pattern": "RGGB"}

    # Warmup
    for _ in range(2):
        operation.apply(image, params)

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    for _ in range(runs):
        operation.apply(image, params)
    end = time.perf_counter()

    profiler.disable()

    # Print timing summary
    avg_time = (end - start) / runs
    print(f"Average time per run: {avg_time:.4f}s\n")

    # Print profiling stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)  # Top 30 functions
    print(s.getvalue())


if __name__ == "__main__":
    # Profile different sizes
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    for size in sizes:
        profile_size(size, runs=5)
