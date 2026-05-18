# Kokkos CUDA backend.
#
# This builds the Kokkos geometry code only, using a Kokkos installation that
# was configured with CUDA enabled. It does not build the direct CUDA kernels.
#
# Usage:
#   make cleanobj
#   make CONFIG=config/kokkos-cuda.mk
#
# This matches examples/kokkos_test/compile_kokkos.sh by default. Override
# KOKKOS_WRAPPER or KOKKOS_INSTALL from the command line if needed.
#
# If the wrapper does not find the host compiler, set:
#   export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)

USE_CUDA   = 0
USE_KOKKOS = 1
BACKEND    = kokkos-cuda

KOKKOS_WRAPPER ?= ${SCRATCH}/kokkos/bin/nvcc_wrapper
KOKKOS_INSTALL ?= ${SCRATCH}/kokkos-install

KOKKOS_CXX ?= ${KOKKOS_WRAPPER}
KOKKOS_CXXFLAGS ?= -std=c++20 --expt-extended-lambda -I${KOKKOS_INSTALL}/include
KOKKOS_LDFLAGS ?= -L${KOKKOS_INSTALL}/lib64 -L${KOKKOS_INSTALL}/lib -L/usr/lib64
KOKKOS_LIBS ?= -lkokkoscore -lcuda

# Kept here as documentation for the architecture you intend to target.
# The actual Kokkos CUDA architecture is usually baked into the Kokkos build.
CUDA_ARCH ?=
