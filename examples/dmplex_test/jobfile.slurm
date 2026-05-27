#!/bin/bash
#SBATCH --job-name=dmplex_test
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=out
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5

cd "$SLURM_SUBMIT_DIR" || exit 1

mpirun -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' ./testt-dmplex
