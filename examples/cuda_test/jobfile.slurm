#!/bin/bash
#SBATCH --job-name=testt_1core
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=out
#SBATCH --account=132714077318

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5
module load CUDA/13.0.0

# Run from the submission directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# This OpenMPI build is not direct-launchable via srun on this cluster.
mpirun -np "${SLURM_NTASKS:-1}" ./testt
