#!/usr/bin/env python3
import csv
from pathlib import Path


MARKERS = {
    "direct_cuda": "square",
    "kokkos_cuda": "circle",
    "kokkos_openmp": "circle",
    "dmplex_mpi": "diamond",
}
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
    "#a16207",
]


def load_rows(path, backend):
    rows = []
    if not path.exists():
        return rows

    with path.open() as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {
                "backend": backend,
                "mesh": raw.get("mesh"),
                "nx": int(raw["nx"]) if raw.get("nx") else None,
                "ny": int(raw["ny"]) if raw.get("ny") else None,
                "elements": int(raw["elements"]) if raw.get("elements") else None,
                "cores": int(raw["cores"]) if raw.get("cores") else None,
                "gpus": int(raw["gpus"]) if raw.get("gpus") else None,
                "ntasks": int(raw["ntasks"]) if raw.get("ntasks") else None,
                "wall_seconds": float(raw["wall_seconds"]) if raw.get("wall_seconds") else None,
                "current_iter_kokkos_seconds": float(raw["current_iter_kokkos_seconds"]) if raw.get("current_iter_kokkos_seconds") else None,
                "current_iter_cuda_seconds": float(raw["current_iter_cuda_seconds"]) if raw.get("current_iter_cuda_seconds") else None,
                "iter1_wall_seconds": float(raw["iter1_wall_seconds"]) if raw.get("iter1_wall_seconds") else None,
                "status": raw.get("status"),
            }
            if backend == "direct_cuda":
                row["iter_seconds"] = row["current_iter_cuda_seconds"]
            elif backend == "dmplex_mpi":
                row["iter_seconds"] = row["iter1_wall_seconds"]
            else:
                row["iter_seconds"] = row["current_iter_kokkos_seconds"]

            suffix = ""
            if backend == "kokkos_openmp" and row["cores"] is not None:
                suffix = f" (c{row['cores']})"
            elif backend == "dmplex_mpi" and row["ntasks"] is not None:
                suffix = f" (np{row['ntasks']})"

            if backend == "dmplex_mpi":
                row["label"] = "DMPlex MPI" + suffix
            else:
                row["label"] = backend.replace("kokkos_", "Kokkos ").replace("cuda", "CUDA") + suffix
            rows.append(row)
    return rows


def plot_rows(rows, y_field, title, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = output_path.with_suffix(".svg")
        if write_svg_plot(rows, y_field, title, svg_path):
            print(f"matplotlib is not installed; wrote {svg_path.name} instead.")
            return True
        print(f"matplotlib is not installed; no valid rows for {output_path.name}.")
        return False

    rows = [r for r in rows if r.get(y_field) is not None and r.get("status") == "ok"]
    if not rows:
        print(f"No valid rows found for {y_field}; skipped {output_path.name}.")
        return False

    series = {}
    for row in rows:
        key = row["label"]
        series.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, series_rows in sorted(series.items()):
        series_rows.sort(key=lambda r: (r["elements"] or 0, r["nx"] or 0, r["ny"] or 0))
        backend = series_rows[0]["backend"]
        if backend == "direct_cuda":
            marker = "s"
        elif backend == "dmplex_mpi":
            marker = "D"
        else:
            marker = "o"
        ax.plot(
            [r["elements"] for r in series_rows],
            [r[y_field] for r in series_rows],
            marker=marker,
            label=label,
        )

    ax.set_xlabel("Mesh elements")
    ax.set_ylabel("Seconds")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return True


def svg_marker(lines, marker, x, y, color):
    if marker == "square":
        lines.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>')
    elif marker == "diamond":
        lines.append(f'<polygon points="{x:.1f},{y - 5:.1f} {x + 5:.1f},{y:.1f} {x:.1f},{y + 5:.1f} {x - 5:.1f},{y:.1f}" fill="{color}"/>')
    else:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')


def write_svg_plot(rows, y_field, title, output_path):
    rows = [r for r in rows if r.get(y_field) is not None and r.get("status") == "ok"]
    if not rows:
        return False

    series = {}
    for row in rows:
        series.setdefault(row["label"], []).append(row)

    width, height = 980, 620
    left, right, top, bottom = 95, 260, 50, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    elems = [int(r["elements"]) for r in rows if r.get("elements") is not None]
    values = [float(r[y_field]) for r in rows]
    min_elem, max_elem = min(elems), max(elems)
    min_value, max_value = min(values), max(values)
    y_pad = (max_value - min_value) * 0.08 or max_value * 0.08 or 1.0
    min_value = max(0.0, min_value - y_pad)
    max_value += y_pad

    def sx(elem):
        if max_elem == min_elem:
            return left + plot_w / 2
        return left + (elem - min_elem) / (max_elem - min_elem) * plot_w

    def sy(value):
        if max_value == min_value:
            return top + plot_h / 2
        return top + plot_h - (value - min_value) / (max_value - min_value) * plot_h

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>',
    ]

    x_ticks = sorted(set(elems))
    if len(x_ticks) > 8:
        step = max(1, len(x_ticks) // 7)
        x_ticks = x_ticks[::step]
        if max_elem not in x_ticks:
            x_ticks.append(max_elem)

    for elem in x_ticks:
        x = sx(elem)
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{elem}</text>')

    for i in range(6):
        value = min_value + i * (max_value - min_value) / 5
        y = sy(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{value:.3g}</text>')

    for idx, (label, series_rows) in enumerate(sorted(series.items())):
        series_rows.sort(key=lambda r: (r["elements"] or 0, r["nx"] or 0, r["ny"] or 0))
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS.get(series_rows[0]["backend"], "circle")
        points = [(sx(int(r["elements"])), sy(float(r[y_field]))) for r in series_rows]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            svg_marker(lines, marker, x, y, color)
        legend_y = top + 20 + idx * 20
        svg_marker(lines, marker, left + plot_w + 35, legend_y - 4, color)
        lines.append(f'<line x1="{left + plot_w + 20}" y1="{legend_y - 4:.1f}" x2="{left + plot_w + 50}" y2="{legend_y - 4:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + plot_w + 60}" y="{legend_y:.1f}" font-family="Arial, sans-serif" font-size="12">{label}</text>')

    y_label = "Seconds"
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Mesh elements</text>')
    lines.append(f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{y_label}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")
    return True


def main():
    root = Path(__file__).resolve().parent
    openmp_csv = root / "kokkos_openmp_mesh_sweep" / "mesh_core_scaling_iter1.csv"
    kokkos_cuda_csv = root / "kokkos_cuda_mesh_sweep" / "mesh_cuda_scaling_iter1.csv"
    direct_cuda_csv = root / "cuda_mesh_sweep" / "mesh_direct_cuda_scaling_iter1.csv"
    dmplex_csv = root / "dmplex_mesh_sweep" / "dmplex_mesh_scaling_iter1.csv"

    rows = []
    rows.extend(load_rows(openmp_csv, "kokkos_openmp"))
    rows.extend(load_rows(kokkos_cuda_csv, "kokkos_cuda"))
    rows.extend(load_rows(direct_cuda_csv, "direct_cuda"))
    rows.extend(load_rows(dmplex_csv, "dmplex_mpi"))

    if not rows:
        print("No input data found. Generate CSV results with the sweep plot scripts first.")
        return

    wall_path = root / "backend_walltime_comparison.png"
    iter_path = root / "backend_iter_comparison.png"

    wrote_wall = plot_rows(rows, "wall_seconds", "Backend wall-clock comparison across meshes", wall_path)
    wrote_iter = plot_rows(rows, "iter_seconds", "Backend per-iteration geometry kernel comparison across meshes", iter_path)

    if wrote_wall:
        print(f"Wrote {wall_path.name}")
    if wrote_iter:
        print(f"Wrote {iter_path.name}")


if __name__ == "__main__":
    main()
