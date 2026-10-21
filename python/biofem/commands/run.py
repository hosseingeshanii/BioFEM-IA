"""biofem run: run the solver in the current case directory (sbatch on a cluster)."""
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import BACKENDS, load_site_config, detect_cluster, _load_env_line
from ..jobs import _run_jobfile
from ..monitor import _run_with_monitor


def cmd_run(args):
    info = BACKENDS[args.backend]
    executable = info["exe"]
    if not executable.exists():
        print(f"[biofem] Executable not found: {executable}", file=sys.stderr)
        print(f"         Run: biofem build --backend {args.backend}", file=sys.stderr)
        sys.exit(1)

    run_dir = Path.cwd()
    if not (run_dir / "control.dat").exists():
        print("[biofem] No control.dat found in current directory.", file=sys.stderr)
        print("         cd into a case directory before running.", file=sys.stderr)
        sys.exit(1)

    cfg       = load_site_config()
    cluster   = detect_cluster()
    account   = args.account or cfg.get("slurm", "account", fallback="")
    partition = args.partition or cfg.get("slurm", "partition", fallback="")
    mpi_flags = cfg.get("mpi", "extra_flags", fallback="")
    load_env  = _load_env_line()

    # Build extra PETSc options from --monitor and --petsc-opts
    monitors = [m.strip() for m in args.monitor.split(",") if m.strip()]
    petsc_opts_parts = []
    if "residual" in monitors:
        petsc_opts_parts.append("-monitor_residual 1")
    if args.petsc_opts.strip():
        petsc_opts_parts.append(args.petsc_opts.strip())
    petsc_opts = " ".join(petsc_opts_parts)

    # Resolve the output directory, matching PETSc option precedence:
    # command-line (--petsc-opts) > control.dat > hardcoded default "output".
    def _parse_out_dir(tokens):
        for i, t in enumerate(tokens):
            if t == "-out_dir" and i + 1 < len(tokens):
                return tokens[i + 1]
        return None

    out_dir = (
        _parse_out_dir(petsc_opts.split())           # 1. command-line wins
        or _parse_out_dir(                            # 2. control.dat
            [tok for line in (run_dir / "control.dat").read_text(errors="replace").splitlines()
             for tok in line.split()
             if not line.lstrip().startswith("#")]
        )
        or "output"                                   # 3. default
    )

    # --local forces a direct terminal run regardless of SLURM availability.
    # Otherwise submit via sbatch when on a cluster and not in an allocation
    # (or when a GPU backend is requested, which always needs a dedicated node).
    if args.local:
        use_sbatch = False
    else:
        use_sbatch = cluster["has_slurm"] and (info["gpu"] or not cluster["in_allocation"])

    if use_sbatch:
        if not account:
            print("[biofem] Warning: no SLURM account set. Job may be rejected.", file=sys.stderr)
            print("         Pass --account YOUR_ACCOUNT, or set it in config/site.cfg", file=sys.stderr)
        if info["gpu"] and not partition:
            print("[biofem] Warning: no SLURM partition set for a GPU backend.", file=sys.stderr)
            print("         --gpus=1 alone may not route to a GPU partition on this cluster.", file=sys.stderr)
            print("         Pass --partition YOUR_GPU_PARTITION, or set it in config/site.cfg", file=sys.stderr)
        jf = _run_jobfile(
            backend=args.backend, executable=executable, run_dir=run_dir,
            np=args.np, time=args.time, account=account, partition=partition,
            mpi_flags=mpi_flags, load_env=load_env, gpu=info["gpu"],
            petsc_opts=petsc_opts, slurm_output=args.slurm_output,
        )
        if args.dry_run:
            print(jf)
            return
        with tempfile.NamedTemporaryFile("w", suffix=".slurm", delete=False, dir=run_dir) as f:
            f.write(jf)
            tmp = Path(f.name)
        want_monitor = "residual" in monitors
        sbatch_cmd = ["sbatch"]
        if args.wait or want_monitor:
            # --wait is required for live monitoring (we need sbatch to block
            # so we can poll residual.log while the job runs on the compute node)
            sbatch_cmd.append("--wait")
        sbatch_cmd.append(str(tmp))
        try:
            if want_monitor:
                # Pass the sbatch command into _run_with_monitor so it polls
                # output/residual.log (on the shared filesystem) while the job
                # runs — same live plot + final PNG save as for local runs.
                sbatch_shell = " ".join(str(s) for s in sbatch_cmd)
                _run_with_monitor(sbatch_shell, run_dir, args.port, out_dir)
            else:
                subprocess.run(sbatch_cmd, cwd=run_dir, check=True)
        finally:
            tmp.unlink(missing_ok=True)
        if want_monitor and not args.wait:
            # job already finished (--wait was added implicitly above), nothing extra to do
            pass
    else:
        # Build the mpi command parts
        mpi_extra     = mpi_flags.split() if mpi_flags.strip() else []
        petsc_opts_list = petsc_opts.split() if petsc_opts else []
        cmd_parts = ["mpirun", "-np", str(args.np)] + mpi_extra + [str(executable)] + petsc_opts_list
        cmd_str   = " ".join(cmd_parts)

        # Prepend environment setup so the solver sees the right modules
        if load_env:
            shell_cmd = f"{load_env} && {cmd_str}"
        else:
            shell_cmd = cmd_str

        if args.dry_run:
            print(shell_cmd)
            return

        print(f"[biofem] running locally: {cmd_str}")
        if "residual" in monitors:
            _run_with_monitor(shell_cmd, run_dir, args.port, out_dir)
        else:
            subprocess.run(["bash", "--login", "-c", shell_cmd], cwd=run_dir, check=True)
