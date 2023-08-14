![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/dark%20logo.png#gh-light-mode-only)
![fastgoertzel Logo](https://raw.githubusercontent.com/0zean/fastgoertzel/master/docs/_static/light%20logo.png#gh-dark-mode-only)

<!-- start here -->

fastgoertzel
============

A Python implementation of the Goertzel algorithm built using `Rust` for improved run time and efficiency on large datasets and large for loops.


## To-Do:

- [ ] Improved speed.
- [ ] Implement testing for speed comparison
- [ ] Fix issue with IIR implementation of Goertzel.
- [ ] Add support for sampling rate.

## Installation

You can install using two methods:

Using `pip install`:
```bash
$ pip install fastgoertzel
```

Using `maturin` after cloning repository:
```bash
$ git clone git://github.com/0zean/fastgoertzel.git
$ cd fastgoertzel
$ maturin develop
```

## Usage
```python
import numpy as np
import pandas as pd

import fastgoertzel as G


def wave(amp, freq, phase, x):
    return amp * np.sin(2*np.pi * freq * x + phase)


x = np.arange(0, 512)
y = wave(1, 1/128, 0, x)

amp, phase = G.goertzel(y, 1/128)
print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')

# Compared to max amplitude FFT output 
ft = np.fft.fft(y)
FFT = pd.DataFrame()
FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
FFT['phase'] = np.arctan2(ft.imag, ft.real)

max_ = FFT.iloc[FFT['amp'].idxmax()]
print(f'FFT amp: {max_["amp"]:.4f}, '
        f'phase: {max_["phase"]:.4f}, '
        f'freq: {max_["freq"]:.4f}')

```