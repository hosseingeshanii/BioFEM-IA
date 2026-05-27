#!/usr/bin/env python3
"""Submit DMPlex MPI strong-scaling jobs for a range of task counts."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_NTASKS = [1, 2, 4, 8, 16, 32]


def replace_directive(text, key, value):
    return re.sub(rf"^#SBATCH --{key}=.*$", f"#SBATCH --{key}={value}", text, flags=re.MULTILINE)


def update_jobfile(path, ntasks):
    text = path.read_text()
    text = replace_directive(text, "job-name", f"dmplex_ntasks_{ntasks}")
    text = replace_directive(text, "ntasks", str(ntasks))
    text = replace_directive(text, "output", f"out_dmplex_ntasks_{ntasks}")
    text = re.sub(
        r'echo "BENCH_NTASKS=\$\{SLURM_NTASKS:-\d+\}"',
        f'echo "BENCH_NTASKS=${{SLURM_NTASKS:-{ntasks}}}"',
        text,
    )
    path.write_text(text)


def failed_output(root, ntasks):
    path = root / f"out_dmplex_ntasks_{ntasks}"
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    return any(p in text for p in ("PETSC ERROR", "Killed", "signal 9", "non-zero exit code"))


def main():
    parser = argparse.ArgumentParser(description="Submit DMPlex MPI strong-scaling jobs.")
    parser.add_argument("--ntasks", nargs="+", type=int, default=DEFAULT_NTASKS,
                        help="List of MPI task counts to sweep (default: 1 2 4 8 16 32).")
    parser.add_argument("--failed-only", action="store_true",
                        help="Only resubmit task counts whose previous output failed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    jobfile = root / "jobfile.slurm"
    if not jobfile.exists():
        raise SystemExit(f"jobfile.slurm not found in {root}")

    for ntasks in sorted(args.ntasks):
        if args.failed_only and not failed_output(root, ntasks):
            continue
        if args.dry_run:
            print(f"DRY RUN: ntasks={ntasks}  sbatch jobfile.slurm  (output: out_dmplex_ntasks_{ntasks})")
            continue
        update_jobfile(jobfile, ntasks)
        result = subprocess.run(["sbatch", "jobfile.slurm"], cwd=root, text=True, capture_output=True)
        submit_text = result.stdout.strip() or result.stderr.strip()
        if result.returncode != 0 or "sbatch: error:" in submit_text:
            print(f"ntasks={ntasks}: {submit_text}", file=sys.stderr)
            raise SystemExit(result.returncode or 1)
        print(f"ntasks={ntasks}: {submit_text}")


if __name__ == "__main__":
    main()
