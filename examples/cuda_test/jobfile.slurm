#!/bin/bash
#SBATCH --job-name=cuda_stats
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=out_cuda_stats
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5
ml CUDA/13.0.0

ml Nsight-Systems >/dev/null 2>&1 || true
ml Nsight-Compute >/dev/null 2>&1 || true

# Run from the submission directory
cd "$SLURM_SUBMIT_DIR" || exit 1

# This OpenMPI build is not direct-launchable via srun on this cluster.
RUN_CMD=(mpirun -np "${SLURM_NTASKS:-1}" ./testt-cuda)
# Enable with: sbatch --export=ALL,CUDA_ALLOC_LOG=1 jobfile.slurm
if [[ "${CUDA_ALLOC_LOG:-0}" == "1" ]]; then
  RUN_CMD+=(-cuda_alloc_log)
fi
JOB_NAME="${SLURM_JOB_NAME}"
PROFILE_TOOL="${PROFILE_TOOL:-nsys}"

case "${PROFILE_TOOL}" in
  nsys)
    PROFILE_OUT="${PROFILE_OUT:-${JOB_NAME}_nsys_cuda_report}"
    nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --output="${PROFILE_OUT}" "${RUN_CMD[@]}"
    nsys stats --force-export=true --report cuda_gpu_trace,cuda_api_sum,cuda_gpu_kern_sum "${PROFILE_OUT}.nsys-rep" > "${PROFILE_OUT}_stats.txt"
    ;;
  ncu)
    PROFILE_OUT="${PROFILE_OUT:-${JOB_NAME}_ncu_cuda_report}"
    ncu --set full --target-processes all --export "${PROFILE_OUT}" "${RUN_CMD[@]}"
    ncu --import "${PROFILE_OUT}.ncu-rep" --page details > "${PROFILE_OUT}_details.txt"
    ;;
  none|"")
    "${RUN_CMD[@]}"
    ;;
  *)
    echo "Unsupported PROFILE_TOOL='${PROFILE_TOOL}'. Use none, nsys, or ncu." >&2
    exit 2
    ;;
esac
