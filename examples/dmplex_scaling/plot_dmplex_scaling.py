#!/usr/bin/env python3
"""Parse DMPlex strong-scaling output files and produce a CSV + plot.

Reads out_dmplex_ntasks_N files, extracts iter-1 wall time, computes
parallel efficiency vs ideal strong scaling, and writes:
  dmplex_scaling.csv
  dmplex_scaling.png  (or .svg if matplotlib is unavailable)
"""
import csv
import re
from pathlib import Path

OUT_RE = re.compile(r"out_dmplex_ntasks_(?P<ntasks>\d+)$")
ITER_RE = re.compile(r"\[iter\s+(?P<iter>\d+)\]\s+DMPlexGeomKernel:\s+wall=(?P<wall>\S+)\s+s")


def parse_output(path):
    m = OUT_RE.match(path.name)
    if not m:
        return None
    ntasks = int(m.group("ntasks"))
    row = {"ntasks": ntasks, "file": path.name, "status": "missing_iter_1"}

    failed = False
    for line in path.read_text(errors="replace").splitlines():
        if "PETSC ERROR" in line or "non-zero status" in line or "Killed" in line:
            failed = True
        im = ITER_RE.search(line)
        if im and int(im.group("iter")) == 1:
            row["iter1_wall_seconds"] = float(im.group("wall"))
            row["status"] = "ok"

    if failed and row["status"] != "ok":
        row["status"] = "error"
    return row


def collect_rows(root):
    rows = []
    for path in sorted(root.glob("out_dmplex_ntasks_*")):
        row = parse_output(path)
        if row:
            rows.append(row)
    return sorted(rows, key=lambda r: r["ntasks"])


def add_scaling_metrics(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return
    t1 = next((r["iter1_wall_seconds"] for r in ok if r["ntasks"] == 1), None)
    if t1 is None:
        t1 = ok[0]["iter1_wall_seconds"]
        n1 = ok[0]["ntasks"]
    else:
        n1 = 1
    for r in ok:
        r["ideal_wall_seconds"] = t1 * n1 / r["ntasks"]
        r["parallel_efficiency"] = r["ideal_wall_seconds"] / r["iter1_wall_seconds"]


def write_csv(rows, path):
    fields = ["ntasks", "iter1_wall_seconds", "ideal_wall_seconds",
              "parallel_efficiency", "status", "file"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_svg_plot(rows, path):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return False

    width, height = 900, 560
    left, right, top, bottom = 85, 40, 45, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    ntasks_vals = [r["ntasks"] for r in ok]
    actual_times = [r["iter1_wall_seconds"] for r in ok]
    ideal_times = [r.get("ideal_wall_seconds", r["iter1_wall_seconds"]) for r in ok]

    min_t = min(min(actual_times), min(ideal_times))
    max_t = max(max(actual_times), max(ideal_times))
    pad = (max_t - min_t) * 0.1 or max_t * 0.1 or 1e-6
    min_t = max(0.0, min_t - pad)
    max_t += pad

    min_n, max_n = min(ntasks_vals), max(ntasks_vals)

    def sx(n):
        if max_n == min_n:
            return left + plot_w / 2
        return left + (n - min_n) / (max_n - min_n) * plot_w

    def sy(v):
        if max_t == min_t:
            return top + plot_h / 2
        return top + plot_h - (v - min_t) / (max_t - min_t) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="85" y="28" font-family="Arial,sans-serif" font-size="18" font-weight="700">DMPlex MPI strong scaling — geom kernel iter 1</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827" stroke-width="1.2"/>',
    ]

    for n in ntasks_vals:
        x = sx(n)
        lines.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+6}" stroke="#111827"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">{n}</text>')

    for i in range(6):
        v = min_t + i * (max_t - min_t) / 5
        y = sy(v)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11">{v:.4f}</text>')

    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">MPI tasks (ntasks)</text>')
    lines.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">Wall time iter 1 (s)</text>')

    # Ideal scaling line (dashed)
    ideal_pts = " ".join(f"{sx(r['ntasks']):.1f},{sy(r.get('ideal_wall_seconds', r['iter1_wall_seconds'])):.1f}" for r in ok)
    lines.append(f'<polyline points="{ideal_pts}" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="6,4"/>')

    # Actual timing line
    actual_pts = " ".join(f"{sx(r['ntasks']):.1f},{sy(r['iter1_wall_seconds']):.1f}" for r in ok)
    lines.append(f'<polyline points="{actual_pts}" fill="none" stroke="#2563eb" stroke-width="2.2"/>')
    for r in ok:
        cx, cy = sx(r["ntasks"]), sy(r["iter1_wall_seconds"])
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#2563eb"/>')

    # Legend
    lx, ly = left + plot_w - 200, top + 16
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="#2563eb" stroke-width="2.2"/>')
    lines.append(f'<circle cx="{lx+12}" cy="{ly}" r="4" fill="#2563eb"/>')
    lines.append(f'<text x="{lx+32}" y="{ly+4}" font-family="Arial,sans-serif" font-size="12">actual</text>')
    lines.append(f'<line x1="{lx}" y1="{ly+22}" x2="{lx+24}" y2="{ly+22}" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="6,4"/>')
    lines.append(f'<text x="{lx+32}" y="{ly+26}" font-family="Arial,sans-serif" font-size="12">ideal (1/ntasks)</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")
    return True


def write_plot(rows, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = path.with_suffix(".svg")
        wrote = write_svg_plot(rows, svg_path)
        if wrote:
            print(f"matplotlib not available; wrote {svg_path.name} instead.")
        else:
            print("matplotlib not available; wrote CSV only.")
        return False

    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        print("No completed iter-1 timings found yet; wrote CSV only.")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ntasks = [r["ntasks"] for r in ok]
    actual = [r["iter1_wall_seconds"] for r in ok]
    ideal = [r.get("ideal_wall_seconds", r["iter1_wall_seconds"]) for r in ok]
    efficiency = [r.get("parallel_efficiency", 1.0) for r in ok]

    ax1.plot(ntasks, ideal, "--", color="gray", label="ideal (1/ntasks)")
    ax1.plot(ntasks, actual, "o-", color="#2563eb", label="actual")
    ax1.set_xlabel("MPI tasks")
    ax1.set_ylabel("Wall time iter 1 (s)")
    ax1.set_title("Strong scaling — wall time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1.0, linestyle="--", color="gray", label="ideal efficiency")
    ax2.plot(ntasks, efficiency, "s-", color="#dc2626", label="parallel efficiency")
    ax2.set_xlabel("MPI tasks")
    ax2.set_ylabel("Parallel efficiency  (T₁ / (N · Tₙ))")
    ax2.set_title("Strong scaling — parallel efficiency")
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("DMPlex MPI strong scaling — geom kernel iter 1", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Wrote {path.name}.")
    return True


def main():
    root = Path(__file__).resolve().parent
    rows = collect_rows(root)
    add_scaling_metrics(rows)

    csv_path = root / "dmplex_scaling.csv"
    png_path = root / "dmplex_scaling.png"
    write_csv(rows, csv_path)
    write_plot(rows, png_path)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    missing = sum(1 for r in rows if r.get("status") != "ok")
    print(f"Wrote {csv_path.name}: {ok} completed, {missing} incomplete/missing.")


if __name__ == "__main__":
    main()
