#!/bin/bash
#SBATCH --job-name=testt_cpu_1core        # job name
#SBATCH --time=02:00:00               # wall time (HH:MM:SS)
#SBATCH --ntasks=1                    # total MPI tasks (== cores here)
#SBATCH --mem=100G                     # memory per node
##SBATCH --output=out.%j               # stdout+stderr go to out.<jobid>
#SBATCH --output=out               # stdout+stderr go to out.<jobid>
#SBATCH --account=132714077318        # charge SUs to this account/allocation
## #SBATCH --partition=short           # 2-hour queue; omit to let Slurm choose

# --- Environment setup (use your existing setup_env.sh if you like) ---
module purge
module load GCC/12.3.0 OpenMPI/4.1.5   # use your preferred toolchain/MPI
module load PETSc/3.20.3-debug       # if your app needs PETSc

# --- Go to your run directory (SCRATCH is recommended) ---
# cd "$SLURM_SUBMIT_DIR"

# --- Launch the job ---

# Option A (recommended on Slurm): use srun (automatically matches allocation)
# srun --mpi=pmix ./testt-cpu

# Option B: if you specifically want mpirun:
# mpirun  ./testt-cpu
mpirun   -mca pml ucx   -mca btl '^uct,ofi'   -mca mtl '^ofi'   ./testt-cpu