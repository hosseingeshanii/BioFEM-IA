#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so the module environment stays in your shell:"
  echo "  source config/load-kokkos-cuda.sh"
  exit 1
fi

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5
ml CUDA/13.0.0

export NVCC_WRAPPER_DEFAULT_COMPILER="$(which g++)"

echo "Loaded Kokkos CUDA build environment."
