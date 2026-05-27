#!/usr/bin/env python3
"""Parse DMPlex MPI mesh-sweep outputs and produce a CSV + plot.

Reads out_dmplex_nx{NX}_ny{NY}_np{NP} files from each case subdirectory,
extracts iter-1 DMPlexGeomKernel wall time and total wall time, and writes:
  dmplex_mesh_scaling_iter1.csv
  dmplex_mesh_scaling_iter1.png  (or .svg if matplotlib is unavailable)
"""
import csv
import re
from pathlib import Path


CASE_RE = re.compile(r"mesh_Lx.*_nx(\d+)_ny(\d+)$")
OUT_RE = re.compile(r"out_dmplex_nx(\d+)_ny(\d+)_np(\d+)$")
ITER_RE = re.compile(r"\[iter\s+(?P<iter>\d+)\]\s+DMPlexGeomKernel:\s+wall=(?P<wall>\S+)\s+s")
BENCH_RE = re.compile(r"^BENCH_(\w+)=(.+)$", re.MULTILINE)

COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def parse_output(path):
    m = OUT_RE.match(path.name)
    if not m:
        return None
    nx, ny, ntasks = int(m.group(1)), int(m.group(2)), int(m.group(3))

    row = {
        "mesh": f"mesh_Lx1.2_Ly0.9_nx{nx}_ny{ny}",
        "nx": nx,
        "ny": ny,
        "ntasks": ntasks,
        "nodes": None,
        "elements": None,
        "iter1_wall_seconds": None,
        "wall_seconds": None,
        "job_id": None,
        "status": "missing_iter_1",
        "file": path.name,
    }

    text = path.read_text(errors="replace")

    for bm in BENCH_RE.finditer(text):
        key, val = bm.group(1), bm.group(2).strip()
        if key == "NODES":
            try:
                row["nodes"] = int(val)
            except ValueError:
                pass
        elif key == "ELEMENTS":
            try:
                row["elements"] = int(val)
            except ValueError:
                pass
        elif key == "WALL_SECONDS":
            try:
                row["wall_seconds"] = float(val)
            except ValueError:
                pass
        elif key == "JOB_ID":
            row["job_id"] = val

    failed = False
    for line in text.splitlines():
        if any(p in line for p in ("PETSC ERROR", "non-zero exit code", "Killed", "signal 9")):
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
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir() and CASE_RE.match(p.name)):
        for out_file in sorted(case_dir.glob("out_dmplex_nx*_ny*_np*")):
            row = parse_output(out_file)
            if row:
                rows.append(row)
    return sorted(rows, key=lambda r: (r["nx"] * r["ny"], r["ntasks"]))


def write_csv(rows, path):
    fields = [
        "mesh", "nx", "ny", "nodes", "elements", "ntasks",
        "iter1_wall_seconds", "wall_seconds", "status", "file", "job_id",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_svg_plot(rows, path):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("iter1_wall_seconds") is not None
          and r.get("elements") is not None]
    if not ok:
        return False

    series = {}
    for r in ok:
        series.setdefault(r["ntasks"], []).append(r)

    width, height = 980, 620
    left, right, top, bottom = 95, 220, 50, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_elems = [r["elements"] for r in ok]
    all_times = [r["iter1_wall_seconds"] for r in ok]
    min_e, max_e = min(all_elems), max(all_elems)
    min_t, max_t = min(all_times), max(all_times)
    pad = (max_t - min_t) * 0.08 or max_t * 0.08 or 1e-6
    min_t = max(0.0, min_t - pad)
    max_t += pad

    def sx(e):
        if max_e == min_e:
            return left + plot_w / 2
        return left + (e - min_e) / (max_e - min_e) * plot_w

    def sy(v):
        if max_t == min_t:
            return top + plot_h / 2
        return top + plot_h - (v - min_t) / (max_t - min_t) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial,sans-serif" font-size="18" font-weight="700">DMPlex MPI mesh scaling — geom kernel iter 1</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
    ]

    x_ticks = sorted(set(all_elems))
    if len(x_ticks) > 8:
        step = max(1, len(x_ticks) // 7)
        x_ticks = x_ticks[::step]
        if max_e not in x_ticks:
            x_ticks.append(max_e)

    for e in x_ticks:
        x = sx(e)
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">{e}</text>')

    for i in range(6):
        v = min_t + i * (max_t - min_t) / 5
        y = sy(v)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12">{v:.3g}</text>')

    for idx, (ntasks, series_rows) in enumerate(sorted(series.items())):
        series_rows.sort(key=lambda r: r["elements"])
        color = COLORS[idx % len(COLORS)]
        label = f"DMPlex MPI (np{ntasks})"
        points = [(sx(r["elements"]), sy(r["iter1_wall_seconds"])) for r in series_rows]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = top + 20 + idx * 22
        lines.append(f'<line x1="{left + plot_w + 20}" y1="{legend_y - 4:.1f}" x2="{left + plot_w + 50}" y2="{legend_y - 4:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{left + plot_w + 35}" cy="{legend_y - 4:.1f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{left + plot_w + 60}" y="{legend_y:.1f}" font-family="Arial,sans-serif" font-size="12">{label}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14">Mesh elements</text>')
    lines.append(f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="14">Wall time iter 1 (s)</text>')
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
            print("matplotlib not available; no valid data for plot.")
        return False

    ok = [r for r in rows if r.get("status") == "ok" and r.get("iter1_wall_seconds") is not None
          and r.get("elements") is not None]
    if not ok:
        print("No completed iter-1 timings found; wrote CSV only.")
        return False

    series = {}
    for r in ok:
        series.setdefault(r["ntasks"], []).append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    for ntasks, series_rows in sorted(series.items()):
        series_rows.sort(key=lambda r: r["elements"])
        ax.plot(
            [r["elements"] for r in series_rows],
            [r["iter1_wall_seconds"] for r in series_rows],
            "o-",
            label=f"DMPlex MPI (np{ntasks})",
        )

    ax.set_xlabel("Mesh elements")
    ax.set_ylabel("Wall time iter 1 (s)")
    ax.set_title("DMPlex MPI mesh scaling — geom kernel iter 1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Wrote {path.name}.")
    return True


def main():
    root = Path(__file__).resolve().parent
    rows = collect_rows(root)

    csv_path = root / "dmplex_mesh_scaling_iter1.csv"
    png_path = root / "dmplex_mesh_scaling_iter1.png"
    write_csv(rows, csv_path)
    write_plot(rows, png_path)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    other = len(rows) - ok
    print(f"Wrote {csv_path.name}: {ok} completed, {other} incomplete/missing.")


if __name__ == "__main__":
    main()
