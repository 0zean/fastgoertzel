# fastgoertzel

The Goertzel algorithm in C++ with Python bindings: evaluate the spectrum of a real signal
at the frequencies you care about, exactly, without computing a full FFT. One frequency,
many frequencies, sliding windows, or a streaming filter fed sample by sample.

```python
import numpy as np
import fastgoertzel as fg

fs = 44100.0
x = np.cos(2 * np.pi * 440 / fs * np.arange(4410) + 0.25)

amp, phase = fg.goertzel(x, 440, fs=fs)          # 440 Hz, evaluated exactly
X = fg.goertzel_batch_complex(x, [440, 880], fs=fs)
flt = fg.GoertzelFilter(440, fs=fs)
flt.push(x[:1000])
flt.push(x[1000:])
assert np.isclose(flt.result()[0], amp)
```

- **Exact frequencies.** Any frequency in `[0, fs)` is evaluated as the DTFT at that
  frequency; nothing is rounded to a DFT bin. `goertzel_bin` gives integer bins when
  that is what you want.
- **Fast.** About 2.5 billion samples per second for one frequency on a current desktop
  core, OpenMP across frequencies and windows, a constant-time sliding-window update, and
  per-call latency around a microsecond. See [Performance](performance.md).
- **Safe.** Every input is validated before the GIL is released; nothing you pass can
  crash the interpreter. Results do not depend on the thread count or on the CPU's
  instruction set beyond rounding.
- **Portable.** Wheels for Linux (x86_64, aarch64, i686), macOS (x86_64, arm64), and
  Windows, Python 3.10 to 3.14, compiled for baseline CPUs with an AVX2+FMA kernel
  selected at import. The C++ core is a single header usable without Python.

```{toctree}
:maxdepth: 2

installation
usage
api
performance
migration
changelog
contributing
```

Source and issues: [github.com/0zean/fastgoertzel](https://github.com/0zean/fastgoertzel).
Package: [pypi.org/project/fastgoertzel](https://pypi.org/project/fastgoertzel/).
Licensed under the GPL-3.0-or-later.
