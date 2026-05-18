# Kokkos CPU-threaded backend.
#
# This assumes Kokkos was built/installed with OpenMP enabled and is visible
# to your compiler/linker environment.
#
# Usage:
#   make cleanobj
#   make CONFIG=config/kokkos-openmp.mk
#
# Common overrides:
#   make CONFIG=config/kokkos-openmp.mk KOKKOS_CXX=/path/to/kokkos/bin/nvcc_wrapper
#   make CONFIG=config/kokkos-openmp.mk KOKKOS_CXXFLAGS="-I/path/to/kokkos/include"
#   make CONFIG=config/kokkos-openmp.mk KOKKOS_LDFLAGS="-L/path/to/kokkos/lib64"

USE_CUDA   = 0
USE_KOKKOS = 1
BACKEND    = kokkos-openmp

KOKKOS_INSTALL ?= ${SCRATCH}/kokkos-openmp-install

KOKKOS_CXX ?= $(CXXLINKER)
KOKKOS_CXXFLAGS ?= -std=c++20 -I${KOKKOS_INSTALL}/include
KOKKOS_LDFLAGS ?= -L${KOKKOS_INSTALL}/lib64 -L${KOKKOS_INSTALL}/lib
KOKKOS_LIBS ?= -lkokkoscore -lkokkoscontainers -lkokkosalgorithms
