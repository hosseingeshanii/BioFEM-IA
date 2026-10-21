"""biofem sweep: generate plate meshes and submit one SLURM job per (mesh, ntasks)."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import BACKENDS, REPO_ROOT, load_site_config, _load_env_line
from ..jobs import _sweep_jobfile
from ..mesh import (HAS_MESH_DEPS, _build_mesh, _boundary_node_lists,
                    _write_nlist, _write_elist, _write_blist)


FAIL_PATTERNS = ("PETSC ERROR", "Killed", "signal 9", "non-zero exit code")
_ITER1_RE     = re.compile(r"\[iter\s+1\]")


def _sweep_resources(nx: int, ny: int) -> tuple[str, str]:
    n = nx * ny
    if n <= 48  * 36:  return "00:30:00", "4G"
    if n <= 128 * 96:  return "01:00:00", "8G"
    if n <= 256 * 192: return "02:00:00", "16G"
    if n <= 384 * 288: return "04:00:00", "32G"
    if n <= 512 * 384: return "06:00:00", "64G"
    return "08:00:00", "128G"


def _out_failed(case_dir, backend, nx, ny, ntasks) -> bool:
    p = case_dir / f"out_{backend}_nx{nx}_ny{ny}_np{ntasks}"
    return p.exists() and any(pat in p.read_text(errors="replace") for pat in FAIL_PATTERNS)


def _out_done(case_dir, backend, nx, ny, ntasks) -> bool:
    p = case_dir / f"out_{backend}_nx{nx}_ny{ny}_np{ntasks}"
    return p.exists() and bool(_ITER1_RE.search(p.read_text(errors="replace")))


def cmd_sweep(args):
    if not HAS_MESH_DEPS:
        print("[biofem] numpy and scipy are required for sweep mesh generation.", file=sys.stderr)
        print("         Load the python/scipy module or: pip install numpy scipy", file=sys.stderr)
        sys.exit(1)

    info = BACKENDS[args.backend]
    executable = info["exe"]
    if not executable.exists():
        print(f"[biofem] Executable not found: {executable}", file=sys.stderr)
        print(f"         Run: biofem build --backend {args.backend}", file=sys.stderr)
        sys.exit(1)

    if not shutil.which("sbatch"):
        print("[biofem] sbatch not found. Sweep requires SLURM to submit jobs.", file=sys.stderr)
        print("         Use --dry-run to preview generated jobfiles.", file=sys.stderr)
        if not args.dry_run:
            sys.exit(1)

    cfg       = load_site_config()
    account   = args.account or cfg.get("slurm", "account", fallback="")
    partition = args.partition or cfg.get("slurm", "partition", fallback="")
    mpi_flags = cfg.get("mpi", "extra_flags", fallback="")
    load_env  = _load_env_line()

    if not account:
        print("[biofem] Warning: no SLURM account set. Jobs may be rejected.", file=sys.stderr)
        print("         Pass --account YOUR_ACCOUNT, or set it in config/site.cfg", file=sys.stderr)
    if info["gpu"] and not partition:
        print("[biofem] Warning: no SLURM partition set for a GPU backend.", file=sys.stderr)
        print("         --gpus=1 alone may not route to a GPU partition on this cluster.", file=sys.stderr)
        print("         Pass --partition YOUR_GPU_PARTITION, or set it in config/site.cfg", file=sys.stderr)

    nx_list = [int(x) for x in args.nx.split(",")]
    ny_list = [int(x) for x in args.ny.split(",")]
    if len(nx_list) != len(ny_list):
        print("[biofem] --nx and --ny must have the same number of values.", file=sys.stderr)
        sys.exit(1)

    ntasks_list = [int(x) for x in args.ntasks.split(",")]
    lx, ly      = args.lx, args.ly
    output_dir  = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    template_ctrl = REPO_ROOT / "examples" / "dmplex_test" / "control.dat"
    if not template_ctrl.exists():
        print(f"[biofem] Template control.dat not found: {template_ctrl}", file=sys.stderr)
        sys.exit(1)

    print(f"[biofem] sweep  backend={args.backend}  cases={len(nx_list)}  ntasks={ntasks_list}")
    print(f"[biofem] output {output_dir}")

    submitted = skipped = 0

    for nx, ny in zip(nx_list, ny_list):
        case_dir = output_dir / f"mesh_Lx{lx}_Ly{ly}_nx{nx}_ny{ny}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "input").mkdir(exist_ok=True)
        (case_dir / "output").mkdir(exist_ok=True)

        pts, tri = _build_mesh(lx, ly, nx, ny)
        nnodes, nelems = pts.shape[0], tri.shape[0]
        edges = _boundary_node_lists(pts, lx, ly)
        _write_nlist(case_dir / "input" / "nlist00", pts)
        _write_elist(case_dir / "input" / "elist00", tri)
        _write_blist(case_dir / "input" / "blist00", edges)
        shutil.copy2(template_ctrl, case_dir / "control.dat")

        time_limit, memory = _sweep_resources(nx, ny)

        for ntasks in ntasks_list:
            approx_elems = 2 * nx * ny
            if approx_elems // ntasks < args.min_elems_per_task:
                print(f"  skip  {case_dir.name}  np={ntasks}  (~{approx_elems // ntasks} elems/task)")
                skipped += 1
                continue
            if args.failed_only and not _out_failed(case_dir, args.backend, nx, ny, ntasks):
                skipped += 1
                continue
            if not args.rerun_completed and _out_done(case_dir, args.backend, nx, ny, ntasks):
                print(f"  done  {case_dir.name}  np={ntasks}  (skip)")
                skipped += 1
                continue

            jf = _sweep_jobfile(
                backend=args.backend, executable=executable,
                case_dir=case_dir, nx=nx, ny=ny, ntasks=ntasks,
                nnodes=nnodes, nelems=nelems,
                time=time_limit, memory=memory, account=account, partition=partition,
                mpi_flags=mpi_flags, load_env=load_env, gpu=info["gpu"],
            )
            jobfile = case_dir / f"jobfile_np{ntasks}.slurm"
            jobfile.write_text(jf)

            if args.dry_run:
                print(f"  dry   {case_dir.name}  np={ntasks}  {jobfile}")
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch", jobfile.name], cwd=case_dir, text=True, capture_output=True,
            )
            msg = result.stdout.strip() or result.stderr.strip()
            if result.returncode != 0:
                print(f"  ERROR {case_dir.name}  np={ntasks}  {msg}", file=sys.stderr)
                sys.exit(result.returncode)
            print(f"  sub   {case_dir.name}  np={ntasks}  {msg}")
            submitted += 1

    print(f"\n[biofem] {submitted} submitted, {skipped} skipped.")
