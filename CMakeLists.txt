# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.18...3.31)

project(fastgoertzel LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FASTGOERTZEL_OPENMP "Parallelise the batch functions with OpenMP when available" ON)
option(FASTGOERTZEL_NATIVE "Tune for the build machine (-march=native). Never for wheels." OFF)
option(FASTGOERTZEL_WERROR "Treat compiler warnings as errors (CI)" OFF)
option(FASTGOERTZEL_BUILD_PYTHON "Build the Python extension module" ON)
option(FASTGOERTZEL_BUILD_BENCHMARKS "Build the standalone C++ kernel benchmark" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Version: scikit-build-core passes the pyproject.toml version (the _FULL form keeps
# pre-release suffixes such as .dev0); standalone builds get "dev".
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(FASTGOERTZEL_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
elseif(DEFINED SKBUILD_PROJECT_VERSION)
  set(FASTGOERTZEL_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(FASTGOERTZEL_VERSION "dev")
endif()

# ---------------------------------------------------------------------------
# Header-only core, usable from C++ without Python
# ---------------------------------------------------------------------------
add_library(fastgoertzel_core INTERFACE)
add_library(fastgoertzel::core ALIAS fastgoertzel_core)
target_include_directories(fastgoertzel_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(fastgoertzel_core INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# Compiler flag policy (shared by every target below)
#   * no -march/-arch flags in wheels: they crash on CPUs lacking those features;
#   * no fast-math: it changes results and, on GCC < 13, sets FTZ/DAZ process-wide;
#   * FMA contraction allowed: it only improves accuracy and is where the speed is;
#   * warnings on everywhere, as errors when FASTGOERTZEL_WERROR is set.
# ---------------------------------------------------------------------------
add_library(fastgoertzel_flags INTERFACE)
if(MSVC)
  target_compile_options(fastgoertzel_flags INTERFACE /W4 /fp:contract /permissive-)
  if(FASTGOERTZEL_WERROR)
    target_compile_options(fastgoertzel_flags INTERFACE /WX)
  endif()
else()
  target_compile_options(fastgoertzel_flags INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -ffp-contract=fast)
  if(FASTGOERTZEL_WERROR)
    target_compile_options(fastgoertzel_flags INTERFACE -Werror)
  endif()
  if(FASTGOERTZEL_NATIVE)
    target_compile_options(fastgoertzel_flags INTERFACE -march=native)
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP (optional). On macOS: brew install libomp and configure with
#   -DOpenMP_ROOT=$(brew --prefix libomp)
# ---------------------------------------------------------------------------
set(FASTGOERTZEL_HAVE_OPENMP OFF)
if(FASTGOERTZEL_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(FASTGOERTZEL_HAVE_OPENMP ON)
  else()
    message(STATUS "fastgoertzel: OpenMP not found; batch functions will be single-threaded")
  endif()
endif()

# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
if(FASTGOERTZEL_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_core MODULE cpp/bindings.cpp)
  target_link_libraries(_core PRIVATE fastgoertzel::core fastgoertzel_flags)
  target_compile_definitions(_core PRIVATE VERSION_INFO=${FASTGOERTZEL_VERSION})
  if(FASTGOERTZEL_HAVE_OPENMP)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  endif()

  install(TARGETS _core LIBRARY DESTINATION fastgoertzel)
endif()

# ---------------------------------------------------------------------------
# Standalone C++ kernel benchmark (not shipped). Configure with
#   cmake -S . -B build-bench -DFASTGOERTZEL_BUILD_PYTHON=OFF -DFASTGOERTZEL_BUILD_BENCHMARKS=ON
# ---------------------------------------------------------------------------
if(FASTGOERTZEL_BUILD_BENCHMARKS)
  add_executable(kernel_bench benchmarks/cpp/kernel_bench.cpp)
  target_link_libraries(kernel_bench PRIVATE fastgoertzel::core fastgoertzel_flags)
endif()

message(STATUS "fastgoertzel ${FASTGOERTZEL_VERSION}: OpenMP=${FASTGOERTZEL_HAVE_OPENMP} "
               "native=${FASTGOERTZEL_NATIVE} werror=${FASTGOERTZEL_WERROR}")
