#!/bin/bash
#SBATCH --job-name=dmplex_ntasks_32
#SBATCH --time=01:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=out_dmplex_ntasks_32
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "BENCH_NTASKS=${SLURM_NTASKS:-32}"
echo "BENCH_JOB_ID=${SLURM_JOB_ID:-local}"

mpirun -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' ./testt-dmplex
