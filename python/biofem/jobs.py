"""SLURM jobfile builders for run and sweep."""

# ---------------------------------------------------------------------------
# Jobfile builders
# ---------------------------------------------------------------------------
def _run_jobfile(
    *, backend, executable, run_dir, np, time, account, partition="",
    mpi_flags, load_env, gpu, petsc_opts="", slurm_output="",
) -> str:
    acct_line   = f"#SBATCH --account={account}" if account else ""
    part_line   = f"#SBATCH --partition={partition}" if partition else ""
    gpu_line    = "#SBATCH --gpus=1" if gpu else ""
    output_line = f"#SBATCH --output={slurm_output}" if slurm_output else f"#SBATCH --output=biofem_{backend}_%j.out"
    ntasks      = 1 if gpu else np
    env_block   = load_env if load_env else "# no load-env.sh found"
    exe_str     = f"{executable}{(' ' + petsc_opts) if petsc_opts else ''}"
    if gpu:
        mpi_cmd = f'/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \\\n  {exe_str}'
    else:
        mpi_extra = f" {mpi_flags}" if mpi_flags.strip() else ""
        mpi_cmd = (
            f'/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \\\n'
            f'  mpirun -np "${{SLURM_NTASKS:-{np}}}"{mpi_extra} {exe_str}'
        )
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name=biofem-{backend}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --mem=8G\n"
        f"{output_line}\n"
        f"{part_line}\n"
        f"{gpu_line}\n"
        f"{acct_line}\n"
        f"\nset -euo pipefail\n"
        f"\n{env_block}\n"
        f'\ncd "{run_dir}" || exit 1\n'
        f'\necho "BENCH_TIME_LIMIT={time}"\n'
        f"\n{mpi_cmd}\n"
    )


def _sweep_jobfile(
    *, backend, executable, case_dir, nx, ny, ntasks, nnodes, nelems,
    time, memory, account, partition="", mpi_flags, load_env, gpu,
) -> str:
    acct_line = f"#SBATCH --account={account}" if account else ""
    part_line = f"#SBATCH --partition={partition}" if partition else ""
    gpu_line  = "#SBATCH --gpus=1" if gpu else ""
    env_block = load_env if load_env else "# no load-env.sh found"
    if gpu:
        mpi_cmd = f'/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \\\n  {executable}'
    else:
        mpi_extra = f" {mpi_flags}" if mpi_flags.strip() else ""
        mpi_cmd = (
            f'/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \\\n'
            f'  mpirun -np "${{SLURM_NTASKS:-{ntasks}}}"{mpi_extra} {executable}'
        )
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={backend}_mesh_nx{nx}_ny{ny}_np{ntasks}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --mem={memory}\n"
        f"#SBATCH --output=out_{backend}_nx{nx}_ny{ny}_np{ntasks}\n"
        f"{part_line}\n"
        f"{gpu_line}\n"
        f"{acct_line}\n"
        f"\nset -euo pipefail\n"
        f"\n{env_block}\n"
        f'\ncd "{case_dir}" || exit 1\n'
        f'\necho "BENCH_BACKEND={backend}"\n'
        f'echo "BENCH_MESH_NX={nx}"\n'
        f'echo "BENCH_MESH_NY={ny}"\n'
        f'echo "BENCH_NODES={nnodes}"\n'
        f'echo "BENCH_ELEMENTS={nelems}"\n'
        f'echo "BENCH_NTASKS=${{SLURM_NTASKS:-{ntasks}}}"\n'
        f'echo "BENCH_JOB_ID=${{SLURM_JOB_ID:-local}}"\n'
        f'echo "BENCH_TIME_LIMIT={time}"\n'
        f"\n{mpi_cmd}\n"
    )
