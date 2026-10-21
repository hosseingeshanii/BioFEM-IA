"""biofem post field (alias: biofem postprocess): average one VTK field over nodes/elements vs time."""
import sys
from pathlib import Path

from .io import _vtk_read_field, _field_reduce, _get_dt, _vtk_series_for_prefix, file_time


def cmd_postprocess(args):
    import matplotlib
    if args.show:
        # Try an interactive backend so plt.show() actually opens a window;
        # falls back to headless Agg (plot still saved, --show becomes a
        # no-op with a warning) if none is usable, e.g. no DISPLAY on a
        # compute node.
        for backend in ("TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        else:
            matplotlib.use("Agg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = args.field
    component = args.component
    dirs = args.dir
    labels = args.label.split(",") if args.label else [Path(d).resolve().name for d in dirs]
    if len(labels) != len(dirs):
        print("error: --label count must match --dir count", file=sys.stderr)
        sys.exit(1)

    prefixes = ["surfaceghost", "surface"] if args.prefix == "auto" else [args.prefix]

    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for case_dir, label in zip(dirs, labels):
        dt = args.dt if args.dt is not None else _get_dt(case_dir)
        if dt is None:
            print(f"warning: could not determine dt for {case_dir} "
                  f"(no -dt in control.dat, and --dt not given); skipping", file=sys.stderr)
            continue

        series = []
        used_prefix = None
        for prefix in prefixes:
            candidate = _vtk_series_for_prefix(case_dir, args.out_dir, prefix)
            if not candidate:
                continue
            # probe the first file for the field before committing to this prefix
            probe = _vtk_read_field(candidate[0][1], field)
            if probe is not None:
                series = candidate
                used_prefix = prefix
                break
        if not series:
            print(f"warning: field '{field}' not found in any "
                  f"{'/'.join(prefixes)}00_*.vtk under {case_dir}/{args.out_dir}; skipping",
                  file=sys.stderr)
            continue

        ts, vals = [], []
        for ti, fp in series:
            parsed = _vtk_read_field(fp, field)
            if parsed is None:
                continue
            reduced = _field_reduce(parsed, component)
            if reduced is None:
                continue
            ts.append(file_time(used_prefix, ti, dt))   # surface file k = state after step k-1
            vals.append(reduced)

        if not ts:
            print(f"warning: no readable '{field}' values under {case_dir}", file=sys.stderr)
            continue

        ax.plot(ts, vals, marker=".", markersize=2, linewidth=1.2, label=f"{label} ({used_prefix})")
        any_data = True

    if not any_data:
        print("error: no data plotted for any --dir", file=sys.stderr)
        sys.exit(1)

    comp_suffix = "" if component == "mag" else f" ({component})"
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"avg {field}{comp_suffix}")
    ax.set_title(args.title or f"avg {field}{comp_suffix} vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(f"postprocess_{field}.png")
    fig.savefig(out_path, dpi=120)
    print(f"[biofem] postprocess plot saved -> {out_path}")

    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f"[biofem] --show requested but could not open a display ({e})", file=sys.stderr)
