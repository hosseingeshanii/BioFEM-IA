"""biofem - BioFEM-IA solver CLI.

Subcommands:
  run          run solver (submits to SLURM when on a cluster)
  sweep        generate mesh cases and submit parameter sweep jobs
  post         post-process solver output (post field, ...)
  postprocess  alias of `post field`
"""
import argparse
import sys

from . import __version__
from .config import BACKENDS
from .commands.run import cmd_run
from .commands.sweep import cmd_sweep
from .post.field import cmd_postprocess


def _add_field_args(p_pp):
    p_pp.add_argument("--dir", action="append", required=True,
                       help="case directory containing control.dat and output_dir "
                            "(repeat for multiple cases to compare)")
    p_pp.add_argument("--field", required=True,
                       help="VTK field name, e.g. gamma, disp, Fint, Fext, Fdyn")
    p_pp.add_argument("--component", default="mag", choices=["mag", "x", "y", "z"],
                       help="for vector fields, which component to average (default: mag)")
    p_pp.add_argument("--out-dir", default="output_mu",
                       help="output subdirectory within each --dir (default: output_mu)")
    p_pp.add_argument("--prefix", default="auto", choices=["auto", "surface", "surfaceghost"],
                       help="which VTK file family to read (default: auto-detect)")
    p_pp.add_argument("--dt", type=float, default=None,
                       help="override timestep (default: read -dt from each case's control.dat)")
    p_pp.add_argument("--label", default="",
                       help="comma-separated legend labels, one per --dir (default: directory names)")
    p_pp.add_argument("--title", default="",
                       help="plot title (default: auto-generated from field name)")
    p_pp.add_argument("--out", default="",
                       help="output PNG path (default: postprocess_<field>.png)")
    p_pp.add_argument("--show", action="store_true",
                       help="also open an interactive window (in addition to saving)")


def main():
    parser = argparse.ArgumentParser(
        prog="biofem",
        description="BioFEM-IA solver CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Build first (from repo root):
  make build                  # dmplex backend (default)
  make build ARGS="-DBIOFEM_CUDA=ON"   # cuda backend
  make install                # symlink biofem into ~/.local/bin

Run:
  biofem run                                  # submit to SLURM (default)
  biofem run --local                          # run in this terminal now
  biofem run --np 8 --account myaccount       # SLURM, 8 tasks
  biofem run --backend cuda --dry-run         # preview jobfile
  biofem sweep --backend dmplex --nx 64,128,256 --ny 48,96,192 --ntasks 1,4,8,16
""",
    )
    parser.add_argument("--version", action="version", version=f"biofem {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # run
    p_run = sub.add_parser(
        "run", help="run solver in current case directory",
        description=(
            "Run the solver from the current directory (must contain control.dat).\n"
            "On a SLURM cluster the job is submitted via sbatch unless you are\n"
            "already inside an interactive allocation."
        ),
    )
    p_run.add_argument("--backend", default="dmplex", choices=list(BACKENDS),
                       help="solver backend (default: dmplex)")
    p_run.add_argument("--np", type=int, default=1,
                       help="MPI task count (default: 1)")
    p_run.add_argument("--time", default="01:00:00",
                       help="SLURM time limit (default: 01:00:00)")
    p_run.add_argument("--account", default="",
                       help="SLURM account (overrides site.cfg)")
    p_run.add_argument("--partition", default="",
                       help="SLURM partition (overrides site.cfg; needed for GPU backends on most clusters)")
    p_run.add_argument("--local", action="store_true",
                       help="run directly in this terminal instead of submitting to SLURM")
    p_run.add_argument("--monitor", default="",
                       help="comma-separated monitors to enable, e.g. residual")
    p_run.add_argument("--petsc-opts", default="",
                       help="extra PETSc flags appended after the executable, e.g. \"-out_dir out_n2\"")
    p_run.add_argument("--slurm-output", default="",
                       help="SLURM output file path (overrides default biofem_<backend>_%%j.out)")
    p_run.add_argument("--wait", action="store_true",
                       help="block until the submitted SLURM job completes (passes --wait to sbatch)")
    p_run.add_argument("--port", type=int, default=8988,
                       help="WebAgg port for live residual plot (--monitor residual, default: 8988)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="print jobfile or shell command without executing")

    # sweep
    p_sw = sub.add_parser(
        "sweep", help="generate mesh cases and submit parameter sweep",
        description=(
            "Generate rectangular Delaunay meshes for each (nx, ny) pair,\n"
            "write case directories under --output-dir, and submit one\n"
            "SLURM job per (mesh, ntasks) combination."
        ),
    )
    p_sw.add_argument("--backend", default="dmplex", choices=list(BACKENDS),
                      help="solver backend (default: dmplex)")
    p_sw.add_argument("--nx", default="64",
                      help="comma-separated nx values, e.g. 64,128,256")
    p_sw.add_argument("--ny", default="48",
                      help="comma-separated ny values (must match --nx length)")
    p_sw.add_argument("--ntasks", default="1",
                      help="comma-separated MPI task counts, e.g. 1,4,8,16")
    p_sw.add_argument("--lx", type=float, default=1.2, help="domain length x (default: 1.2)")
    p_sw.add_argument("--ly", type=float, default=0.9, help="domain length y (default: 0.9)")
    p_sw.add_argument("--output-dir", default="./sweep",
                      help="directory for generated case folders (default: ./sweep)")
    p_sw.add_argument("--account", default="",
                      help="SLURM account (overrides site.cfg)")
    p_sw.add_argument("--partition", default="",
                      help="SLURM partition (overrides site.cfg; needed for GPU backends on most clusters)")
    p_sw.add_argument("--min-elems-per-task", type=int, default=100,
                      help="skip (mesh, ntasks) with fewer elements/task (default: 100)")
    p_sw.add_argument("--failed-only", action="store_true",
                      help="only submit runs whose existing output shows a failure")
    p_sw.add_argument("--rerun-completed", action="store_true",
                      help="re-submit jobs that already completed successfully")
    p_sw.add_argument("--dry-run", action="store_true",
                      help="generate jobfiles but do not submit")

    # postprocess
    p_pp = sub.add_parser(
        "postprocess", help="plot avg node/element field vs time from VTK output",
        description=(
            "Read a node field (e.g. Fint, disp, Fext, Fdyn from surfaceghost*.vtk)\n"
            "or element field (e.g. gamma, thickness from surface*.vtk) across all\n"
            "timesteps, average it over all nodes/elements per timestep, and plot\n"
            "against time. Pass --dir multiple times to overlay several cases on\n"
            "the same plot for comparison."
        ),
    )
    _add_field_args(p_pp)

    # post: post-processing commands (field = the same as postprocess)
    p_post = sub.add_parser("post", help="post-process solver output (field, ...)")
    post_sub = p_post.add_subparsers(dest="post_command", metavar="POST_COMMAND")
    p_field = post_sub.add_parser("field", help="plot avg node/element field vs time from VTK output",
                                  description=p_pp.description)
    _add_field_args(p_field)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "post":
        if args.post_command is None:
            p_post.print_help()
            sys.exit(0)
        {"field": cmd_postprocess}[args.post_command](args)
        return
    {"run": cmd_run, "sweep": cmd_sweep, "postprocess": cmd_postprocess}[args.command](args)


if __name__ == "__main__":
    main()
