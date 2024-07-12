import timeit

import fastgoertzel as G
import numpy as np


def wave(amp, freq, phase, x):
    return amp * np.sin(2*np.pi * freq * x + phase)


def py_goertzel(x, f):
    N = len(x)
    k = f * N
    w = 2 * np.pi * k / N
    cw = np.cos(w)
    c = 2 * cw
    sw = np.sin(w)
    z1, z2 = 0, 0

    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0

    ip = cw * z1 - z2
    qp = sw * z1
    
    amp = np.sqrt(ip**2 + qp**2) / (N / 2)
    phase = np.arctan2(qp, ip)
    return amp, phase


x_test = np.arange(0, 10**7)
y_test = wave(1, 1/128, 0, x_test)

time_py = timeit.timeit('py_goertzel(y_test, 1/128)', globals=globals(), number=10)

time_rust = timeit.timeit('G.goertzel(y_test, 1/128)', globals=globals(), number=10)

print(f"Time taken by py_goertzel: {time_py} seconds")
print(f"Time taken by fastgoertzel: {time_rust} seconds")

time_difference = time_py - time_rust
print(f"Difference in time: {time_difference} seconds")
print(f"fastgoertzel is {time_py / time_rust:.3} times faster")
