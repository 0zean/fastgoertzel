# SPDX-License-Identifier: GPL-3.0-or-later
"""Build script for the fastgoertzel C++ extension.

All package metadata lives in pyproject.toml (setuptools ignores metadata passed here
when a [project] table exists). This file only describes how to compile the extension.

Flag policy (PLAN.md, section 2.4):
  * no -march=native / /arch:AVX2 in wheels: they crash on CPUs without those features
    and the scalar kernel gains nothing from them;
  * no -ffast-math / /fp:fast: on GCC < 13 fast-math sets FTZ/DAZ for the whole process
    at import time, and it changes results;
  * FMA contraction is allowed (it only improves accuracy);
  * OpenMP where the toolchain provides it (GCC, Clang, MSVC); Apple Clang ships without
    it, so macOS builds are single-threaded unless FASTGOERTZEL_OPENMP_FLAGS is set.

Environment overrides for source builds:
  FASTGOERTZEL_NATIVE=1          add -march=native (never used for published wheels)
  FASTGOERTZEL_OPENMP_FLAGS=...  extra compile *and* link flags for OpenMP, e.g.
                                 "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
                                  -L/opt/homebrew/opt/libomp/lib -lomp" on macOS with Homebrew.
"""

import os
import re
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).parent


def read_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("could not find [project].version in pyproject.toml")
    return match.group(1)


def compiler_flags() -> tuple[list[str], list[str]]:
    compile_args: list[str] = []
    link_args: list[str] = []

    if sys.platform == "win32":
        compile_args += ["/O2", "/openmp", "/fp:contract"]
    else:
        compile_args += ["-O3", "-Wall", "-Wextra", "-Wpedantic", "-ffp-contract=fast"]
        if sys.platform != "darwin":
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

    if os.environ.get("FASTGOERTZEL_NATIVE") == "1" and sys.platform != "win32":
        compile_args.append("-march=native")

    extra_openmp = os.environ.get("FASTGOERTZEL_OPENMP_FLAGS")
    if extra_openmp:
        compile_args += extra_openmp.split()
        link_args += extra_openmp.split()

    return compile_args, link_args


extra_compile_args, extra_link_args = compiler_flags()

ext_modules = [
    Pybind11Extension(
        "fastgoertzel._fastgoertzel_core",
        ["src/python_bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        # Unquoted on purpose: python_bindings.cpp stringifies it.
        define_macros=[("VERSION_INFO", read_version())],
    ),
]

setup(
    packages=["fastgoertzel"],
    package_dir={"": "python"},
    package_data={"fastgoertzel": ["*.pyi", "py.typed"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
