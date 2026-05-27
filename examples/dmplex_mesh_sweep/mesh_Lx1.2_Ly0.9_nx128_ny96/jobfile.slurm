#!/bin/bash
#SBATCH --job-name=dmplex_mesh_nx128_ny96_np32
#SBATCH --time=01:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=out_dmplex_nx128_ny96_np32
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "BENCH_BACKEND=dmplex_mpi"
echo "BENCH_MESH_NX=128"
echo "BENCH_MESH_NY=96"
echo "BENCH_NODES=12289"
echo "BENCH_ELEMENTS=24132"
echo "BENCH_NTASKS=${SLURM_NTASKS:-32}"
echo "BENCH_JOB_ID=${SLURM_JOB_ID:-local}"
/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \
  mpirun -np "${SLURM_NTASKS:-32}" -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' ./testt-dmplex
