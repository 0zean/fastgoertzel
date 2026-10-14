#!/usr/bin/env bash
# Fetch a prebuilt LLVM OpenMP runtime for the macOS wheel builds (cibuildwheel before-all).
#
# Apple's compiler ships no OpenMP runtime. Homebrew's libomp bottles are built for the
# runner's macOS release and carry that release as their minimum version, which delocate
# would then force onto the wheel (macosx_15_0 instead of 10.13 / 11.0). The conda-forge
# build below targets macOS 10.9 (x86_64) and 11.0 (arm64), below cibuildwheel's default
# deployment targets, so the wheel keeps its normal platform tag. scikit-learn ships its
# macOS wheels the same way.
#
# Result: $HOME/openmp/{include/omp.h,lib/libomp.dylib}. pyproject.toml points CMake at it
# with -DOpenMP_ROOT; CMakeLists.txt keeps its directory as the module's install rpath so
# that delocate can find the dylib (its install name is @rpath/libomp.dylib) and bundle it.
set -euo pipefail

base="https://api.anaconda.org/download/conda-forge/llvm-openmp/12.0.1"
case "$(uname -m)" in
    arm64)
        url="$base/osx-arm64/llvm-openmp-12.0.1-hf3c4609_0.tar.bz2"
        md5="047713aa71d0dd2294d8cd5f502b23b8"
        ;;
    x86_64)
        url="$base/osx-64/llvm-openmp-12.0.1-hda6cdc1_0.tar.bz2"
        md5="2a5ef7d574ef02d7c500e5cddc4bf690"
        ;;
    *)
        echo "macos_openmp.sh: unsupported architecture $(uname -m)" >&2
        exit 1
        ;;
esac

prefix="$HOME/openmp"
archive="$(mktemp -t llvm-openmp).tar.bz2"
curl -fsSL --retry 3 -o "$archive" "$url"
actual="$(md5 -q "$archive")"
if [[ "$actual" != "$md5" ]]; then
    echo "macos_openmp.sh: checksum mismatch for $url: $actual != $md5" >&2
    exit 1
fi
mkdir -p "$prefix"
tar -xjf "$archive" -C "$prefix"
rm -f "$archive"

ls -l "$prefix/include/omp.h" "$prefix/lib/libomp.dylib"
echo "minimum macOS of libomp.dylib:"
otool -l "$prefix/lib/libomp.dylib" | grep -E -A3 "LC_VERSION_MIN_MACOSX|LC_BUILD_VERSION" | grep -E "version|minos" || true
