#!/usr/bin/env python3
"""Submit DMPlex MPI mesh-scaling jobs across mesh sizes and optional task counts."""
import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = {
    (16, 12),
    (32, 24),
    (48, 36),
    (64, 48),
    (96, 72),
    (128, 96),
    (160, 120),
    (256, 192),
    (384, 288),
    (512, 384),
    (640, 480),
}
DEFAULT_NTASKS = [1, 2, 4, 8, 16, 32]
CASE_RE = re.compile(r"mesh_Lx.*_nx(\d+)_ny(\d+)$")
FAIL_PATTERNS = ("PETSC ERROR", "Killed", "signal 9", "non-zero exit code")


def replace_directive(text, key, value):
    return re.sub(rf"^#SBATCH --{key}=.*$", f"#SBATCH --{key}={value}", text, flags=re.MULTILINE)


def slurm_resources(nx, ny):
    nodes = nx * ny
    if nodes <= 48 * 36:
        return "00:30:00", "4G"
    if nodes <= 128 * 96:
        return "01:00:00", "8G"
    if nodes <= 256 * 192:
        return "02:00:00", "16G"
    if nodes <= 384 * 288:
        return "04:00:00", "32G"
    if nodes <= 512 * 384:
        return "06:00:00", "64G"
    return "08:00:00", "128G"


def update_jobfile(path, nx, ny, ntasks):
    text = path.read_text()
    time_limit, memory = slurm_resources(nx, ny)
    text = replace_directive(text, "job-name", f"dmplex_mesh_nx{nx}_ny{ny}_np{ntasks}")
    text = replace_directive(text, "time", time_limit)
    text = replace_directive(text, "ntasks", str(ntasks))
    text = replace_directive(text, "mem", memory)
    text = replace_directive(text, "output", f"out_dmplex_nx{nx}_ny{ny}_np{ntasks}")
    text = re.sub(
        r'echo "BENCH_NTASKS=\$\{SLURM_NTASKS:-\d+\}"',
        f'echo "BENCH_NTASKS=${{SLURM_NTASKS:-{ntasks}}}"',
        text,
    )
    text = re.sub(
        r'mpirun -np "\$\{SLURM_NTASKS:-\d+\}"',
        f'mpirun -np "${{SLURM_NTASKS:-{ntasks}}}"',
        text,
    )
    path.write_text(text)


def failed_output(case_dir, nx, ny, ntasks):
    path = case_dir / f"out_dmplex_nx{nx}_ny{ny}_np{ntasks}"
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    return any(pattern in text for pattern in FAIL_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description="Submit DMPlex MPI mesh-scaling jobs.")
    parser.add_argument("--case", help="Only submit one case directory name.")
    parser.add_argument("--skip-case", nargs="+", default=[], help="Case directory names to skip.")
    parser.add_argument("--ntasks", nargs="+", type=int, default=DEFAULT_NTASKS,
                        help="MPI task counts to run per mesh (default: 1 2 4 8 16 32).")
    parser.add_argument("--all-cases", action="store_true",
                        help="Submit every mesh_* directory, including older generated cases.")
    parser.add_argument("--failed-only", action="store_true",
                        help="Only submit case/ntask runs whose existing output failed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cases = sorted(p for p in root.iterdir() if p.is_dir() and CASE_RE.match(p.name))
    if args.case:
        cases = [p for p in cases if p.name == args.case]
    elif not args.all_cases:
        cases = [p for p in cases if tuple(map(int, CASE_RE.match(p.name).groups())) in DEFAULT_CASES]
    if args.skip_case:
        skip_set = set(args.skip_case)
        cases = [p for p in cases if p.name not in skip_set]

    if not cases:
        raise SystemExit("No mesh case directories found. Run generate_mesh_cases.py first.")

    for case_dir in cases:
        match = CASE_RE.match(case_dir.name)
        nx, ny = int(match.group(1)), int(match.group(2))
        jobfile = case_dir / "jobfile.slurm"
        for ntasks in args.ntasks:
            if args.failed_only and not failed_output(case_dir, nx, ny, ntasks):
                continue
            if args.dry_run:
                print(f"DRY RUN: ntasks={ntasks}  cd {case_dir} && sbatch jobfile.slurm")
            else:
                update_jobfile(jobfile, nx, ny, ntasks)
                result = subprocess.run(
                    ["sbatch", "jobfile.slurm"], cwd=case_dir, text=True, capture_output=True
                )
                submit_text = result.stdout.strip() or result.stderr.strip()
                if result.returncode != 0 or "sbatch: error:" in submit_text:
                    print(f"{case_dir.name} ntasks={ntasks}: {submit_text}", file=sys.stderr)
                    raise SystemExit(result.returncode or 1)
                print(f"{case_dir.name} ntasks={ntasks}: {submit_text}")


if __name__ == "__main__":
    main()
