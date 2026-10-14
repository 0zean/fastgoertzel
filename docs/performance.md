# Performance

## How the numbers are produced

`benchmarks/benchmark.py` measures every function against named baselines and writes a
JSON report; `benchmarks/render_tables.py` renders the tables below from that report
into the README, so they always describe one real run on one named machine. The rules
the harness follows:

1. **Every ratio names its baseline**: a pure-Python loop over a list, the NumPy
   two-real-dot single-bin DFT, `np.fft.rfft` for one bin or for all bins, or
   `sliding_window_view` + `rfft` for windows.
2. **Threads are pinned and reported.** The thread sweep runs each thread count in its
   own subprocess with `OMP_NUM_THREADS` set. A multi-threaded fastgoertzel number is
   never shown against a single-threaded baseline without saying so.
3. **Minimum of repeated rounds** is the headline; the median is kept in the JSON to
   expose noise.
4. **The machine is recorded**: CPU, core count, OS, Python, NumPy, fastgoertzel version
   and the kernel build selected.
5. **Per-call latency** is measured separately, as percentiles over 100,000 calls with
   Python overhead included, because that is what a real-time loop cares about.

Numbers on other machines will differ; run `make bench` and `make readme-tables` and
quote your own together with the baseline and thread count. `tests/test_performance.py`
guards against large regressions with machine-independent ratio checks.

## How to read them

- **One frequency** is single-threaded by design. The Goertzel recurrence is a
  dependency chain that costs a few cycles per sample however the loop is written;
  signals of 1,024 samples or more are therefore split into independent blocks of at
  least 512 samples (up to eight) whose results are combined exactly, which keeps the
  FMA units busy. The hot loops are compiled for baseline x86-64 and for AVX2+FMA and the
  build is chosen at import.
- **Several frequencies.** On one thread a batch costs the same per frequency as a
  single call: both run eight chains and are compute-bound once the signal is in cache.
  What the batch adds is OpenMP over lane groups, and one pass over the data per eight
  frequencies when the signal is larger than cache.
- **Sliding windows** use the sliding-DFT update (constant cost per hop, re-synchronised
  exactly every 256 windows), so the window length no longer matters. What remains per
  window in the `(value, phase)` form is mostly `sqrt` and `atan2`, which
  `goertzel_sliding_batch_complex` skips.
- **Per-call latency** includes Python call overhead. Calls that enter an OpenMP region
  carry a heavier tail: pass `threads=1` or keep the work below the 65,536
  operation threshold for predictable latency on small inputs.

## Measured

```{include} ../README.md
:start-after: <!-- benchmark-tables:start -->
:end-before: <!-- benchmark-tables:end -->
```
