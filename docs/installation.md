# Installation

## From PyPI

```bash
pip install fastgoertzel
```

Wheels are published for:

| Platform | Architectures | OpenMP |
|---|---|---|
| Linux (manylinux, musllinux) | x86_64, aarch64, i686 | yes |
| macOS | x86_64, arm64 | no (see below) |
| Windows | AMD64 | yes, runtime bundled |

for CPython 3.10 to 3.14. A source distribution is published as well. The only runtime
dependency is NumPy (1.21.2 or later).

The hot loops in every x86 wheel exist twice, compiled for baseline x86-64 and for
AVX2+FMA, and the build is chosen when the module is imported. The wheel therefore runs
on any CPU and still uses wide instructions where they exist. `fastgoertzel._core._kernel`
reports the choice (`"avx2"` or `"baseline"`) and the environment variable
`FASTGOERTZEL_KERNEL=baseline|avx2|auto` overrides it. On arm64 there is one build.

## From source

A C++17 compiler is required (MSVC 2019+, GCC 9+, Clang 10+). CMake and Ninja are
fetched automatically when they are missing.

```bash
git clone https://github.com/0zean/fastgoertzel.git
cd fastgoertzel
pip install .
```

The build is driven by `CMakeLists.txt` through scikit-build-core. Options are passed
with the `CMAKE_ARGS` environment variable, for example
`CMAKE_ARGS="-DFASTGOERTZEL_NATIVE=ON" pip install .`:

| Option | Default | Effect |
|---|---|---|
| `-DFASTGOERTZEL_NATIVE=ON` | OFF | Tune for the build machine (`-march=native`). Never used for wheels. |
| `-DFASTGOERTZEL_OPENMP=OFF` | ON | Single-threaded build even when OpenMP is available. |
| `-DOpenMP_ROOT=$(brew --prefix libomp)` | unset | Enable OpenMP on macOS after `brew install libomp`. |
| `-DFASTGOERTZEL_WERROR=ON` | OFF | Warnings as errors (CI uses this). |

### OpenMP on macOS

Apple's compiler ships without OpenMP, so the macOS wheels are single-threaded: the batch
and sliding functions still work, and `threads=` is accepted and ignored. To get
multi-threaded batches on macOS, install `libomp` from Homebrew and build from source:

```bash
brew install libomp
CMAKE_ARGS="-DOpenMP_ROOT=$(brew --prefix libomp)" pip install --no-binary fastgoertzel fastgoertzel
```

`fastgoertzel._core._has_openmp` tells you whether the build you have uses OpenMP.

## The C++ core without Python

`include/fastgoertzel/goertzel.hpp` is a header-only library with no dependencies beyond
`<cmath>`. From CMake:

```cmake
include(FetchContent)
FetchContent_Declare(fastgoertzel
  GIT_REPOSITORY https://github.com/0zean/fastgoertzel.git
  GIT_TAG v2.0.0)
set(FASTGOERTZEL_BUILD_PYTHON OFF)
FetchContent_MakeAvailable(fastgoertzel)
target_link_libraries(my_app PRIVATE fastgoertzel::core)
```

```cpp
#include <fastgoertzel/goertzel.hpp>
namespace fgz = fastgoertzel;

// One shot: X(f) for a normalised frequency f in cycles per sample.
fgz::Complex X = fgz::dtft(x.data(), x.size(), 440.0 / 44100.0);
double amplitude = fgz::magnitude(X) * fgz::amplitude_scale(x.size(), false);

// Streaming: feed samples as they arrive, read the result at any time.
fgz::StreamingGoertzel flt(440.0 / 44100.0);
flt.push(chunk.data(), chunk.size());
flt.push(next_sample);
fgz::Complex so_far = flt.result();
```

The header validates nothing and never throws; the caller checks `n >= 1` and
`0 <= f < 1` (`fgz::is_valid_frequency`). It is the code the Python functions run, minus
the block decomposition, multi-frequency lanes, and OpenMP scheduling, which live in
`cpp/`. The C++ unit tests (`-DFASTGOERTZEL_BUILD_TESTS=ON`) show every function in use.
