#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so the module environment stays in your shell:"
  echo "  source load-env.sh"
  exit 1
fi

# Pick the module set by SLURM cluster name (TAMU HPRC: grace, faster). Login and compute nodes
# of a cluster report the same name. Override with BIOFEM_CLUSTER=<name>.
BIOFEM_CLUSTER="${BIOFEM_CLUSTER:-$(scontrol show config 2>/dev/null | awk '/^ClusterName/{print $3}')}"

module purge
case "$BIOFEM_CLUSTER" in
  faster)
    # FASTER has no PETSc 3.22.5; 3.23.5 is built with GCC 14.2.0 + OpenMPI 5.0.7 (foss-2025a).
    ml GCC/14.2.0 OpenMPI/5.0.7
    ml PETSc/3.23.5
    ;;
  *)
    # Grace (default)
    ml GCC/13.2.0 OpenMPI/4.1.6
    ml PETSc/3.22.5
    ml CUDA/13.0.0
    ;;
esac

export NVCC_WRAPPER_DEFAULT_COMPILER="$(which g++)"

echo "Loaded BioFEM-IA build environment (${BIOFEM_CLUSTER:-grace})."
