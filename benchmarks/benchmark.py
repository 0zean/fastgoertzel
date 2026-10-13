"""Reproducible benchmark harness for fastgoertzel.

Design goals:

* Every number is the *minimum* of ``repeat`` timing rounds, each round running
  the statement ``number`` times (chosen by ``timeit.autorange``).  The minimum
  is the least noisy estimator of the intrinsic cost; the median is reported
  alongside it so that noisy runs are visible.
* Every fastgoertzel number is reported next to the *same* measurement for a
  set of reference implementations, so that a speed-up is always "X times
  faster than <named baseline>" and never a bare multiplier.
* Thread count is controlled explicitly (``--threads``) and recorded, because
  ``goertzel_batch`` and ``goertzel_sliding_batch`` use OpenMP and their
  speed-ups depend almost entirely on core count.
* The machine, Python, NumPy and fastgoertzel versions are captured and written
  to the JSON report so that published numbers can state where they came from.
* Throughput is not latency.  The latency section times individual calls with
  ``time.perf_counter_ns`` and reports p50/p90/p99/p99.9/max, which is what a
  real-time or trading loop cares about; OpenMP region start-up shows up here
  as tail latency even when the mean looks fine.

Usage::

    python benchmarks/benchmark.py                 # full run, prints tables
    python benchmarks/benchmark.py --quick         # smaller sizes, fewer repeats
    python benchmarks/benchmark.py --threads 1     # pin OpenMP to one thread
    python benchmarks/benchmark.py --latency-only  # per-call latency percentiles only
    python benchmarks/benchmark.py --json out.json # also write machine-readable report

Do not quote a speed-up from this script without also quoting the baseline it
was measured against and the thread count.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import timeit
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Reference implementations
# --------------------------------------------------------------------------- #


def py_goertzel(xs: list[float], f: float) -> tuple[float, float]:
    """Idiomatic pure-Python Goertzel over a list of floats.

    This is the honest "native Python" baseline.  Indexing a NumPy array one
    element at a time in a Python loop is ~4x slower than this and should not
    be used as the baseline for a speed-up claim.
    """
    n = len(xs)
    k = int(f * n)
    w = 2.0 * math.pi * k / n
    cw, sw = math.cos(w), math.sin(w)
    c = 2.0 * cw
    s1 = s2 = 0.0
    for v in xs:
        s0 = v + c * s1 - s2
        s2 = s1
        s1 = s0
    re = cw * s1 - s2
    im = sw * s1
    return math.hypot(re, im) / (n / 2.0), math.atan2(im, re)


def np_single_bin(x: np.ndarray, f: float) -> tuple[float, float]:
    """Vectorised single-bin DFT with two real dot products (no complex temporaries)."""
    n = len(x)
    k = int(f * n)
    ang = (2.0 * math.pi * k / n) * np.arange(n)
    re = float(np.dot(x, np.cos(ang)))
    im = -float(np.dot(x, np.sin(ang)))
    return math.hypot(re, im) / (n / 2.0), math.atan2(im, re)


def np_rfft_bin(x: np.ndarray, f: float) -> tuple[float, float]:
    """Full real FFT, then pick one bin.  Computes every bin, so it is the
    natural baseline when a user needs more than a handful of frequencies."""
    n = len(x)
    k = int(f * n)
    X = np.fft.rfft(x)[k]
    return abs(X) / (n / 2.0), float(np.angle(X))


def np_sliding_rfft(x: np.ndarray, window: int, freqs: np.ndarray) -> np.ndarray:
    v = np.lib.stride_tricks.sliding_window_view(x, window)
    X = np.fft.rfft(v, axis=1)[:, (freqs * window).astype(int)]
    return np.stack([np.abs(X) / (window / 2.0), np.angle(X)], axis=-1)


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #


@dataclass
class Timing:
    name: str
    seconds_min: float
    seconds_median: float
    number: int
    repeat: int


def time_call(
    name: str, fn: Callable[[], object], repeat: int, max_number: int | None = None
) -> Timing:
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    if max_number is not None:
        number = max(1, min(number, max_number))
    rounds = [t / number for t in timer.repeat(repeat=repeat, number=number)]
    return Timing(name, min(rounds), statistics.median(rounds), number, repeat)


def fmt_s(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.3f} s "


def ratio(ref: Timing, fg: Timing) -> str:
    return f"{ref.seconds_min / fg.seconds_min:5.1f}x"


def environment(threads: str) -> dict[str, Any]:
    import fastgoertzel

    cpu = platform.processor()
    try:  # py-cpuinfo is pulled in by pytest-benchmark; optional here
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        cpu = info.get("brand_raw", cpu)
        flags = sorted(f for f in info.get("flags", []) if f.startswith(("avx", "fma", "sse4")))
    except Exception:
        flags = []
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "fastgoertzel": fastgoertzel.__version__,
        "cpu": cpu,
        "cpu_flags": flags,
        "logical_cores": os.cpu_count(),
        "omp_num_threads": threads,
    }


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #


def bench_single(
    sizes: list[int], repeat: int, with_python: bool, rng: np.random.Generator
) -> list[dict[str, Any]]:
    import fastgoertzel as fg

    f = 0.1
    rows: list[dict[str, Any]] = []
    print("=== goertzel(x, f): one frequency ===")
    print(
        f"{'N':>10} | {'fastgoertzel':>12} {'MS/s':>7} | {'pure Python':>12} {'x':>6} | {'NumPy 2x dot':>12} {'x':>6} | {'np.fft.rfft':>12} {'x':>6}"
    )
    for n in sizes:
        x = rng.standard_normal(n)
        t_fg = time_call("fastgoertzel", lambda: fg.goertzel(x, f), repeat)
        t_np = time_call("numpy_two_dots", lambda: np_single_bin(x, f), repeat)
        t_fft = time_call("numpy_rfft", lambda: np_rfft_bin(x, f), repeat)
        row = {
            "n": n,
            "fastgoertzel": asdict(t_fg),
            "numpy_two_dots": asdict(t_np),
            "numpy_rfft": asdict(t_fft),
        }
        line = f"{n:>10} | {fmt_s(t_fg.seconds_min):>12} {n / t_fg.seconds_min / 1e6:7.0f} |"
        if with_python and n <= 1_000_000:
            xl = x.tolist()
            t_py = time_call(
                "pure_python",
                lambda: py_goertzel(xl, f),
                repeat=min(repeat, 3),
                max_number=1,
            )
            row["pure_python"] = asdict(t_py)
            line += f" {fmt_s(t_py.seconds_min):>12} {ratio(t_py, t_fg):>6} |"
        else:
            line += f" {'-':>12} {'-':>6} |"
        line += f" {fmt_s(t_np.seconds_min):>12} {ratio(t_np, t_fg):>6} | {fmt_s(t_fft.seconds_min):>12} {ratio(t_fft, t_fg):>6}"
        print(line)
        rows.append(row)
    print(
        "    x = baseline time / fastgoertzel time (min of repeats); > 1 means fastgoertzel is faster\n"
    )
    return rows


def bench_batch(
    cases: list[tuple[int, int]], repeat: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    import fastgoertzel as fg

    rows: list[dict[str, Any]] = []
    print("=== goertzel_batch(x, freqs): N samples x M frequencies ===")
    print(
        f"{'N':>10} {'M':>4} | {'fg batch':>12} | {'M x goertzel':>12} {'x':>6} | {'np.fft.rfft':>12} {'x':>6} | {'M x NumPy dot':>13} {'x':>6}"
    )
    for n, m in cases:
        x = rng.standard_normal(n)
        freqs = (np.arange(m) + 1) / (m + 1) * 0.5
        t_b = time_call("fastgoertzel_batch", lambda: fg.goertzel_batch(x, freqs), repeat)
        t_s = time_call("m_x_goertzel", lambda: [fg.goertzel(x, v) for v in freqs], repeat)
        t_fft = time_call("numpy_rfft_all_bins", lambda: np.fft.rfft(x), repeat)
        t_np = time_call(
            "m_x_numpy_two_dots",
            lambda: [np_single_bin(x, v) for v in freqs],
            repeat=min(repeat, 3),
        )
        print(
            f"{n:>10} {m:>4} | {fmt_s(t_b.seconds_min):>12} | {fmt_s(t_s.seconds_min):>12} {ratio(t_s, t_b):>6} | {fmt_s(t_fft.seconds_min):>12} {ratio(t_fft, t_b):>6} | {fmt_s(t_np.seconds_min):>13} {ratio(t_np, t_b):>6}"
        )
        rows.append({"n": n, "m": m, **{t.name: asdict(t) for t in (t_b, t_s, t_fft, t_np)}})
    print()
    return rows


def bench_sliding(
    cases: list[tuple[int, int, int]], repeat: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    import fastgoertzel as fg

    rows: list[dict[str, Any]] = []
    print("=== goertzel_sliding_batch(x, W, freqs): N samples, window W, M frequencies ===")
    print(f"{'N':>8} {'W':>5} {'M':>4} | {'fg sliding':>12} | {'NumPy windows+rfft':>18} {'x':>6}")
    for n, w, m in cases:
        x = rng.standard_normal(n)
        freqs = (np.arange(m) + 1) / (m + 1) * 0.5
        t_s = time_call(
            "fastgoertzel_sliding",
            lambda: fg.goertzel_sliding_batch(x, w, freqs),
            repeat,
        )
        t_n = time_call("numpy_sliding_rfft", lambda: np_sliding_rfft(x, w, freqs), repeat)
        print(
            f"{n:>8} {w:>5} {m:>4} | {fmt_s(t_s.seconds_min):>12} | {fmt_s(t_n.seconds_min):>18} {ratio(t_n, t_s):>6}"
        )
        rows.append({"n": n, "w": w, "m": m, **{t.name: asdict(t) for t in (t_s, t_n)}})
    print()
    return rows


def bench_threads(thread_counts: list[int | None], repeat: int) -> list[dict[str, Any]]:
    """Re-run the OpenMP functions in subprocesses with OMP_NUM_THREADS pinned."""
    code = (
        "import numpy as np, fastgoertzel as fg, timeit\n"
        "x=np.random.default_rng(1).standard_normal(1_000_000); fr=(np.arange(32)+1)/33*0.5\n"
        f"t=timeit.Timer(lambda: fg.goertzel_batch(x, fr)); n,_=t.autorange(); b=min(t.repeat({repeat},n))/n\n"
        "y=x[:100_000]; fr2=(np.arange(8)+1)/9*0.5\n"
        f"t=timeit.Timer(lambda: fg.goertzel_sliding_batch(y,256,fr2)); n,_=t.autorange(); s=min(t.repeat({repeat},n))/n\n"
        "print(b, s)"
    )
    rows: list[dict[str, Any]] = []
    print("=== OpenMP scaling: batch (N=1e6, M=32) and sliding (N=1e5, W=256, M=8) ===")
    for th in thread_counts:
        env = dict(os.environ)
        env.pop("OMP_NUM_THREADS", None)
        if th is not None:
            env["OMP_NUM_THREADS"] = str(th)
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        ).stdout
        b, s = (float(v) for v in out.split())
        label = "default" if th is None else str(th)
        print(f"   OMP_NUM_THREADS={label:<8} batch {fmt_s(b)}   sliding {fmt_s(s)}")
        rows.append({"threads": label, "batch_seconds": b, "sliding_seconds": s})
    print()
    return rows


def latency_percentiles(
    fn: Callable[[], object], calls: int, warmup: int = 2_000
) -> dict[str, Any]:
    """Time `calls` individual invocations and return percentiles in microseconds."""
    for _ in range(warmup):
        fn()
    samples = np.empty(calls)
    clock = time.perf_counter_ns
    for i in range(calls):
        t0 = clock()
        fn()
        samples[i] = clock() - t0
    samples /= 1e3  # ns -> us
    p = np.percentile(samples, [50, 90, 99, 99.9])
    return {
        "calls": calls,
        "p50_us": float(p[0]),
        "p90_us": float(p[1]),
        "p99_us": float(p[2]),
        "p999_us": float(p[3]),
        "max_us": float(samples.max()),
        "mean_us": float(samples.mean()),
    }


def bench_latency(calls: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Per-call latency for the small inputs typical of a real-time loop.

    Each case is a single Python-level call; the numbers therefore include argument
    conversion and result construction, not only the kernel. The batch cases straddle
    the OpenMP work threshold on purpose: the larger one enters a parallel region on
    every call, and its tail (p99.9, max) shows the cost of doing so.
    """
    import fastgoertzel as fg

    cases: list[tuple[str, Callable[[], object]]] = []
    for n in (64, 256, 1_024, 4_096):
        x = rng.standard_normal(n)
        cases.append((f"goertzel        N={n:<6}", functools.partial(fg.goertzel, x, 0.1)))
    x256 = rng.standard_normal(256)
    x65k = rng.standard_normal(65_536)
    f8 = np.linspace(0.05, 0.45, 8)
    cases.append(("batch           N=256    M=8 (serial)", lambda: fg.goertzel_batch(x256, f8)))
    cases.append(("batch           N=65536  M=8 (OpenMP)", lambda: fg.goertzel_batch(x65k, f8)))
    x1k = rng.standard_normal(1_024)
    cases.append(
        ("sliding         N=1024 W=256 M=4", lambda: fg.goertzel_sliding_batch(x1k, 256, f8[:4]))
    )
    ref = rng.standard_normal(256)
    cases.append(("np.fft.rfft     N=256 (reference)", lambda: np.fft.rfft(ref)))
    cases.append(("NumPy 2x dot    N=256 (reference)", lambda: np_single_bin(ref, 0.1)))

    rows: list[dict[str, Any]] = []
    print(f"=== per-call latency, {calls} calls each (microseconds) ===")
    print(f"{'case':<40} {'p50':>8} {'p90':>8} {'p99':>8} {'p99.9':>8} {'max':>9} {'mean':>8}")
    for name, fn in cases:
        r = latency_percentiles(fn, calls)
        print(
            f"{name:<40} {r['p50_us']:8.2f} {r['p90_us']:8.2f} {r['p99_us']:8.2f} "
            f"{r['p999_us']:8.2f} {r['max_us']:9.1f} {r['mean_us']:8.2f}"
        )
        rows.append({"case": name.strip(), **r})
    print(
        "    Includes Python call overhead. Pin the process to one core and set "
        "OMP_NUM_THREADS=1 to see the kernel-only tail.\n"
    )
    return rows


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true", help="smaller sizes and fewer repeats")
    ap.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="timing rounds per measurement (default 7)",
    )
    ap.add_argument(
        "--threads", type=int, default=None, help="pin OMP_NUM_THREADS for this process"
    )
    ap.add_argument("--no-python", action="store_true", help="skip the slow pure-Python baseline")
    ap.add_argument("--no-threads", action="store_true", help="skip the OpenMP scaling sweep")
    ap.add_argument("--no-latency", action="store_true", help="skip the per-call latency section")
    ap.add_argument(
        "--latency-only", action="store_true", help="run only the per-call latency section"
    )
    ap.add_argument(
        "--calls",
        type=int,
        default=None,
        help="calls per latency case (default 100000, 10000 with --quick)",
    )
    ap.add_argument("--json", type=str, default=None, help="write a machine-readable report here")
    args = ap.parse_args()

    if args.threads is not None:
        # Must be set before the extension (and its OpenMP runtime) is first imported.
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    repeat = 3 if args.quick else args.repeat
    sizes = (
        [1_000, 100_000, 1_000_000]
        if args.quick
        else [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    )
    batch_cases = (
        [(100_000, 8), (1_000_000, 32)]
        if args.quick
        else [
            (1_000, 8),
            (100_000, 8),
            (1_000_000, 8),
            (1_000_000, 64),
            (10_000_000, 8),
        ]
    )
    sliding_cases = (
        [(10_000, 256, 8)]
        if args.quick
        else [
            (10_000, 256, 8),
            (10_000, 256, 64),
            (100_000, 256, 8),
            (100_000, 1024, 8),
        ]
    )
    threads = [1, 2, 4, None] if args.quick else [1, 2, 4, 8, 16, None]

    env = environment(os.environ.get("OMP_NUM_THREADS", "default"))
    print(" | ".join(f"{k}: {v}" for k, v in env.items() if k != "cpu_flags"))
    print()

    rng = np.random.default_rng(1)
    calls = args.calls or (10_000 if args.quick else 100_000)
    report: dict[str, Any] = {
        "environment": env,
        "methodology": (
            "throughput: timeit.autorange number, min and median of repeat rounds, ratios use "
            "minima; latency: perf_counter_ns per call after warm-up, percentiles in us"
        ),
    }
    if not args.latency_only:
        report["single"] = bench_single(sizes, repeat, not args.no_python, rng)
        report["batch"] = bench_batch(batch_cases, repeat, rng)
        report["sliding"] = bench_sliding(sliding_cases, repeat, rng)
    if not args.no_latency:
        report["latency"] = bench_latency(calls, rng)
    if not args.latency_only and not args.no_threads:
        report["threads"] = bench_threads(threads, min(repeat, 5))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
