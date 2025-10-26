"""Detailed profiling of apply() method internals."""
# ruff: noqa: T201, PLR0915

import time

import numpy as np
import taichi as ti
from PIL import Image
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation


def create_test_image(size: tuple[int, int]) -> Image.Image:
    """Create a test RGB image."""
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def profile_apply_internals(size: tuple[int, int], runs: int = 20) -> None:
    """Manually time each step of the apply() method."""
    print(f"\n{'=' * 70}")
    print(f"Detailed profiling for {size[0]}x{size[1]} ({runs} runs)")
    print(f"{'=' * 70}\n")

    operation = BayerFilterGPUOperation()
    image = create_test_image(size)
    params = {"pattern": "RGGB"}

    # Warmup
    for _ in range(3):
        operation.apply(image, params)

    # Detailed timing
    times: dict[str, list[float]] = {
        "total": [],
        "numpy_convert": [],
        "field_create": [],
        "data_to_gpu": [],
        "kernel_exec": [],
        "data_from_gpu": [],
        "numpy_postprocess": [],
    }

    for _ in range(runs):
        t0 = time.perf_counter()

        # NumPy conversion
        t1 = time.perf_counter()
        img_array = np.array(image, dtype=np.float32) / 255.0
        t2 = time.perf_counter()
        times["numpy_convert"].append(t2 - t1)

        height, width = img_array.shape[:2]

        # Field creation
        t3 = time.perf_counter()
        rgb_in = ti.field(dtype=ti.f32, shape=(height, width, 3))
        rgb_out = ti.field(dtype=ti.f32, shape=(height, width, 3))
        t4 = time.perf_counter()
        times["field_create"].append(t4 - t3)

        # Data to GPU
        t5 = time.perf_counter()
        rgb_in.from_numpy(img_array)
        t6 = time.perf_counter()
        times["data_to_gpu"].append(t6 - t5)

        # Kernel execution
        from sevenrad_stills.operations.bayer_filter_gpu import bayer_filter_fast

        pattern_id = 0  # RGGB
        t7 = time.perf_counter()
        bayer_filter_fast(rgb_in, rgb_out, pattern_id, height, width)
        ti.sync()  # Ensure kernel completes
        t8 = time.perf_counter()
        times["kernel_exec"].append(t8 - t7)

        # Data from GPU
        t9 = time.perf_counter()
        result_array = rgb_out.to_numpy()
        t10 = time.perf_counter()
        times["data_from_gpu"].append(t10 - t9)

        # NumPy postprocessing
        t11 = time.perf_counter()
        result_array_uint8 = np.clip(result_array * 255.0, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result_array_uint8)
        t12 = time.perf_counter()
        times["numpy_postprocess"].append(t12 - t11)

        t_end = time.perf_counter()
        times["total"].append(t_end - t0)

    # Calculate averages
    avg = {k: np.mean(v) * 1000 for k, v in times.items()}  # Convert to ms
    std = {k: np.std(v) * 1000 for k, v in times.items()}

    # Print results
    print(f"{'Step':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total'}")
    print("-" * 70)

    total_avg = avg["total"]
    for step in [
        "numpy_convert",
        "field_create",
        "data_to_gpu",
        "kernel_exec",
        "data_from_gpu",
        "numpy_postprocess",
    ]:
        pct = (avg[step] / total_avg) * 100
        print(f"{step:<25} {avg[step]:>10.2f}   {std[step]:>10.2f}   {pct:>5.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<25} {avg['total']:>10.2f}   {std['total']:>10.2f}   100.0%")

    # Calculate what Metal could save
    metal_saves = (
        avg["field_create"]  # Pre-allocated buffers
        + avg["data_to_gpu"]  # Zero-copy with unified memory
        + avg["data_from_gpu"]  # Zero-copy with unified memory
    )
    metal_overhead = avg["numpy_convert"] + avg["numpy_postprocess"]
    estimated_metal = avg["kernel_exec"] + metal_overhead

    print(f"\n{'Optimization Potential:':<25}")
    print(f"  Kernel execution:       {avg['kernel_exec']:>10.2f} ms (irreducible)")
    print(f"  NumPy overhead:         {metal_overhead:>10.2f} ms (could move to GPU)")
    print(f"  Transfer/alloc savings: {metal_saves:>10.2f} ms (eliminated by Metal)")
    print(f"  Estimated Metal time:   {estimated_metal:>10.2f} ms")
    print(f"  Potential speedup:      {total_avg / estimated_metal:>10.2f}x")


if __name__ == "__main__":
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    for size in sizes:
        profile_apply_internals(size, runs=20)
