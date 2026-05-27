#!/bin/bash
#SBATCH --job-name=dmplex_mesh_nx96_ny72_np32
#SBATCH --time=01:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=out_dmplex_nx96_ny72_np32
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "BENCH_BACKEND=dmplex_mpi"
echo "BENCH_MESH_NX=96"
echo "BENCH_MESH_NY=72"
echo "BENCH_NODES=6915"
echo "BENCH_ELEMENTS=13496"
echo "BENCH_NTASKS=${SLURM_NTASKS:-32}"
echo "BENCH_JOB_ID=${SLURM_JOB_ID:-local}"
/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \
  mpirun -np "${SLURM_NTASKS:-32}" -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' ./testt-dmplex
