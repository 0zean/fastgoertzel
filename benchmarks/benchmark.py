import time

import fastgoertzel
import numpy as np


def benchmark_goertzel():
    """Benchmark Goertzel performance."""
    sizes = [100, 1000, 10000, 100000, 1000000]

    print("Goertzel Algorithm Benchmark")
    print("-" * 40)
    print(f"{'Size':<10} {'Time (ms)':<15} {'Throughput (MS/s)':<15}")
    print("-" * 40)

    for size in sizes:
        signal = np.random.randn(size)
        frequency = 0.1

        # Warmup
        for _ in range(10):
            fastgoertzel.goertzel(signal, frequency)

        # Benchmark
        n_iterations = max(1, 10000 // size)
        start = time.perf_counter()
        for _ in range(n_iterations):
            amp, phase = fastgoertzel.goertzel(signal, frequency)
        elapsed = time.perf_counter() - start

        time_ms = (elapsed / n_iterations) * 1000
        throughput = size / (elapsed / n_iterations) / 1e6

        print(f"{size:<10} {time_ms:<15.3f} {throughput:<15.2f}")


if __name__ == "__main__":
    benchmark_goertzel()
