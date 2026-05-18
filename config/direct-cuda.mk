# Direct CUDA backend: builds src/cuda_kernel.cu and src/cuda_wrapper.cpp.
#
# Usage:
#   make cleanobj
#   make CONFIG=config/direct-cuda.mk

USE_CUDA   = 1
USE_KOKKOS = 0
BACKEND    = cuda

# Set these from the command line or your environment when needed:
#   make CONFIG=config/direct-cuda.mk CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80
CUDA_ARCH ?=
