import sys

import numpy as np
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

__version__ = "1.0.0"

system = sys.platform.lower()

extra_compile_args = []
extra_link_args = []

if system == "win32":
    extra_compile_args = [
        "/O2",                  # Optimize for speed
        "/arch:AVX2",           # Enable AVX2
        "/openmp:experimental", # OpenMP 4.0 support (MSVC 2022+)
        "/fp:fast",             # Fast floating-point math
        "/Qpar",                # Auto-parallelization
    ]
    extra_link_args = []

elif system == "darwin":
    # Apple Clang does NOT support OpenMP by default
    # If you install LLVM with OpenMP via Homebrew, you need to point to it explicitly
    extra_compile_args = [
        "-O3",
        "-march=native",
        "-mavx2",
        "-Wall",
        "-Wextra",
        "-ffast-math",
        "-funroll-loops",
        "-flto",
        "-ftree-vectorize",
    ]
    extra_link_args = ["-flto"]

    # Optional OpenMP support (only if user installed LLVM via Homebrew)
    # Uncomment these if you know your build env supports it:
    # extra_compile_args += ["-Xpreprocessor", "-fopenmp", "-I/usr/local/include"]
    # extra_link_args += ["-lomp", "-L/usr/local/lib"]

else:
    # Linux / other Unix-like systems
    extra_compile_args = [
        "-O3",
        "-march=native",
        "-mavx2",
        "-Wall",
        "-Wextra",
        "-ffast-math",
        "-funroll-loops",
        "-flto",
        "-ftree-vectorize",
        "-fopenmp",  # OpenMP on GCC/Clang
    ]
    extra_link_args = ["-fopenmp", "-flto"] 

ext_modules = [
    Pybind11Extension(
        "fastgoertzel._fastgoertzel_core",
        ["src/python_bindings.cpp", "src/goertzel.cpp"],
        include_dirs=[
            "src",
            np.get_include(),
        ],
        cxx_std=17,  # Use C++17 standard
        # Optimization flags
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="fastgoertzel",
    version=__version__,
    author="Nicholas Picini",
    author_email="pd1138@protonmail.com",
    url="https://github.com/0zean/fastgoertzel_cpp",
    description="High-performance Goertzel algorithm implementation",
    long_description=open("docs/pypi.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.2.6",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "ruff",
            "mypy",
            "sphinx",
            "bumpver",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)
