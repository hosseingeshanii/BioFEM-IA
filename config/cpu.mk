# Plain CPU build: no direct CUDA and no Kokkos.
#
# Usage:
#   make cleanobj
#   make CONFIG=config/cpu.mk

USE_CUDA   = 0
USE_KOKKOS = 0
BACKEND    = cpu
