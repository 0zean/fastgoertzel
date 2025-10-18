import sys

import numpy as np
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

__version__ = "0.1.0"

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
        extra_compile_args=[
            "-O3",                # Maximum optimization
            "-march=native",      # Use CPU-specific instructions
            "-mavx2",             # Enable AVX2
            "-Wall",              # All warnings
            "-Wextra",            # Extra warnings
            "-ffast-math",        # Fast floating-point
            "-funroll-loops",     # Unroll loops
            "-flto",              # Link-time optimization
            "-ftree-vectorize",   # Auto-vectorization
            "-fopenmp",           # OpenMP parallelization 
        ]
        if sys.platform != "win32"
        else [
            "/O2",                  # Maximum optimization
            "/arch:AVX2",           # Enable AVX2
            "/openmp:experimental", # OpenMP parallelization
            "/fp:fast",             # Fast floating-point
            "/Qpar",                # Auto-parallelization
        ],
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy==2.2.6",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "black",
            "mypy",
            "sphinx",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)
