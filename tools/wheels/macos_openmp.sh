#!/usr/bin/env bash
# Build the LLVM OpenMP runtime for the macOS wheel builds (cibuildwheel before-all).
#
# Apple's compiler ships no OpenMP runtime, and the prebuilt alternatives do not fit:
# Homebrew's libomp bottles are built for the runner's macOS release and carry that as their
# minimum version, which delocate would force onto the wheel (macosx_15_0); conda-forge's old
# tar.bz2 builds (12.0.1, 2021) aborted intermittently on x86_64 when several host threads
# started parallel regions at once. Building from the release sources takes about a minute
# on the runner, uses the same recipe as Homebrew's formula, and lets us choose the minimum
# macOS: 10.13 on x86_64 and 11.0 on arm64, the wheels' own deployment targets (set in
# pyproject.toml for x86_64; 11.0 is cibuildwheel's arm64 default).
#
# Result: $HOME/openmp/{include/omp.h,lib/libomp.dylib}. pyproject.toml points CMake at it
# with -DOpenMP_ROOT; CMakeLists.txt keeps its directory as the module's install rpath so
# that delocate can find the dylib (its install name is @rpath/libomp.dylib) and bundle it.
set -euo pipefail

llvm_version="19.1.7"
case "$(uname -m)" in
    arm64) target="11.0" ;;
    x86_64) target="10.13" ;;
    *)
        echo "macos_openmp.sh: unsupported architecture $(uname -m)" >&2
        exit 1
        ;;
esac

prefix="$HOME/openmp"
work="$(mktemp -d -t llvm-openmp)"
base="https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvm_version"
cd "$work"
curl -fsSL --retry 3 -o openmp.tar.xz "$base/openmp-$llvm_version.src.tar.xz"
curl -fsSL --retry 3 -o cmake.tar.xz "$base/cmake-$llvm_version.src.tar.xz"
tar -xf openmp.tar.xz
tar -xf cmake.tar.xz
mv "cmake-$llvm_version.src" cmake  # the openmp tree expects ../cmake/Modules

cmake -S "openmp-$llvm_version.src" -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$target" \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
cmake --build build --parallel "$(sysctl -n hw.ncpu)"
cmake --install build
cd /
rm -rf "$work"

ls -l "$prefix/include/omp.h" "$prefix/lib/libomp.dylib"
echo "minimum macOS of libomp.dylib (expect $target):"
otool -l "$prefix/lib/libomp.dylib" | grep -E -A3 "LC_VERSION_MIN_MACOSX|LC_BUILD_VERSION" | grep -E "version|minos" || true
