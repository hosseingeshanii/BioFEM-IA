"""biofem post animate: animation (GIF + full-colour animated PNG) of one case.

Frames come from shell3d00_*.vtk (thick wall, -write_3d_shell 1) or surface00_*.vtk (mid-surface).
Each frame shows the chosen views (side: apex motion; base: from below the base into the cavity)
with the reference shape as a grey wireframe, coloured by |u| (cm) or the wall thickness ratio h/h0
(shell3d only). The title gives t, mean gamma, endocardial volume and mean h/h0; steps whose
nonlinear solve did not converge are marked. Rendering uses pvpython (ParaView).
Writes <out>.gif, <out>.png (APNG) and <out>_peak.png (frame with the largest mean gamma).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import style
from .io import (VTK, read_control, control_float, output_dir, output_files, file_step,
                 read_lv_volume, find_solver_log, parse_solver_log)
from .lv import TO_CM

RENDER_SCRIPT = Path(__file__).resolve().parent / "render" / "pv_views.py"
CELL_TYPE = {3: 5, 6: 13}          # triangle, wedge


def _write_vtk(path, pts, cells, fields):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 2.0\nbiofem post animate frame\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(pts)} float\n")
        np.savetxt(f, pts, fmt="%.6f")
        f.write(f"CELLS {len(cells)} {sum(len(c) + 1 for c in cells)}\n")
        for c in cells:
            f.write(f"{len(c)} " + " ".join(map(str, c)) + "\n")
        f.write(f"CELL_TYPES {len(cells)}\n")
        f.write("\n".join(str(CELL_TYPE[len(c)]) for c in cells) + "\n")
        f.write(f"POINT_DATA {len(pts)}\n")
        for name, vals in fields.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, vals, fmt="%.6f")


def _pvpython():
    exe = shutil.which("pvpython") or shutil.which("pvbatch")
    if exe is None:
        sys.exit("error: pvpython (ParaView) not found on PATH; load a ParaView module or install ParaView")
    return exe


def cmd_animate(args):
    plt = style.setup(False)
    import matplotlib
    from PIL import Image

    case = Path(args.dir)
    ctrl = read_control(case)
    out_path = output_dir(case, ctrl, args.out_dir or None)
    dt = control_float(ctrl, "-dt", 1.0)
    s = TO_CM[args.solver_unit]

    surf = dict(output_files(out_path, "surface"))
    shell = dict(output_files(out_path, "shell3d"))
    source = args.source
    if source == "auto":
        source = "shell3d" if shell else "surface"
    series = shell if source == "shell3d" else surf
    if not series:
        sys.exit(f"error: no {source}00_*.vtk in {out_path}")
    if args.field == "h_ratio" and source != "shell3d":
        sys.exit("error: --field h_ratio needs shell3d output (-write_3d_shell 1)")
    ks = sorted(k for k in series if k != 1)[::args.every]      # file 1 repeats file 0
    if sorted(series)[-1] not in ks:
        ks.append(sorted(series)[-1])

    bad_steps = set()
    log = find_solver_log(case)
    if log is not None:
        bad_steps = {st["step"] for st in parse_solver_log(log) if st["converged"] is False}
    vol = read_lv_volume(out_path)

    vmin, vmax = (float(x) for x in args.range.split(","))
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in ("side", "base"):
            sys.exit(f"error: unknown view '{v}' (use side, base)")

    work = Path(tempfile.mkdtemp(prefix="biofem_animate_"))
    try:
        # 1. frames in cm with the colour field
        ref = VTK(series[ks[0]])
        X0 = ref.points * s
        h0 = ref.point_data.get("thickness")
        info = {}
        for k in ks:
            v = VTK(series[k])
            X = v.points * s
            fields = {"u": np.linalg.norm(X - X0, axis=1)}
            hr = None
            if "thickness" in v.point_data and h0 is not None:
                fields["h_ratio"] = v.point_data["thickness"] / h0
                hr = float(fields["h_ratio"].mean())
            _write_vtk(work / f"in_{k:06d}.vtk", X, v.cells, fields)
            g = None
            if k in surf:
                gv = VTK(surf[k]).cell_data.get("gamma")
                g = float(gv.mean()) if gv is not None else None
            ti = file_step(source, k)
            info[k] = dict(t=max(ti, 0) * dt, step=ti, gamma=g, h=hr,
                           V=vol[k][1] * s ** 3 if k in vol else None)

        # 2. render with ParaView
        field = "u" if args.field == "u" else "h_ratio"
        # no X display (ssh / batch node): render offscreen, else VTK's X interactor segfaults
        pv = [_pvpython()] + ([] if os.environ.get("DISPLAY") else ["--force-offscreen-rendering"])
        r = subprocess.run(pv + [str(RENDER_SCRIPT), str(work), str(work / "png"), field,
                            str(vmin), str(vmax), ",".join(views)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("error: pvpython failed:\n" + r.stderr[-2000:])

        # 3. compose
        label = args.label or case.resolve().name
        frames = []
        nv = len(views)
        for k in ks:
            fig = plt.figure(figsize=(4.6 * nv + 1.0, 5.6), dpi=100)
            wv = 0.88 / nv
            for j, name in enumerate(views):
                ax = fig.add_axes([j * wv, 0.0, wv, 0.84])
                ax.imshow(Image.open(work / "png" / f"{name}_{k:06d}.png").convert("RGB"))
                ax.axis("off")
                ax.set_title({"side": "side view (apex)", "base": "from the base, into the cavity"}[name], fontsize=11, pad=2)
            cb = fig.add_axes([0.9, 0.15, 0.02, 0.58])
            fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin, vmax), "viridis"),
                         cax=cb).set_label("|u| (cm)" if field == "u" else "h / h0", fontsize=10)
            d = info[k]
            parts = [label, f"t = {d['t']:.2f} s"]
            if d["gamma"] is not None:
                parts.append(f"$\\gamma$ = {d['gamma']:.3f}")
            if d["V"] is not None:
                parts.append(f"$V_{{endo}}$ = {d['V']:.2f} ml")
            if d["h"] is not None:
                parts.append(f"wall $h/h_0$ = {d['h']:.3f}")
            flag = d["step"] in bad_steps
            fig.suptitle("   ".join(parts) + ("   [step not converged]" if flag else ""), fontsize=11, y=0.975,
                         color=style.FAIL if flag else style.INK)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)

        out = Path(args.out or f"{label}_animation.gif").with_suffix(".gif")
        idx = np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)
        w, h = frames[0].size
        mos = Image.new("RGB", (w * 4, h * 3))
        for j, i in enumerate(idx):
            mos.paste(frames[i], ((j % 4) * w, (j // 4) * h))
        pal = mos.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        dur = [600] + [int(1000 / args.fps)] * max(len(q) - 2, 0) + [1200]
        dur = dur[:len(q)]
        q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0)
        frames[0].save(out.with_suffix(".png"), save_all=True, append_images=frames[1:], duration=dur, loop=0)
        gk = [info[k]["gamma"] if info[k]["gamma"] is not None else -1 for k in ks]
        frames[int(np.argmax(gk))].save(out.with_name(out.stem + "_peak.png"))
        print(f"[biofem] animation saved -> {out} (GIF), {out.with_suffix('.png')} (animated PNG, full colour), "
              f"{out.with_name(out.stem + '_peak.png')} ({len(frames)} frames, source {source})")
    finally:
        shutil.rmtree(work, ignore_errors=True)
