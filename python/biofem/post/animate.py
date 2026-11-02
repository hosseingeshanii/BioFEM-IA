"""biofem post animate: animation (GIF + full-colour animated PNG) of one case.

Frames come from shell3d00_*.vtk (thick wall, -write_3d_shell 1) or surface00_*.vtk (mid-surface) of a shell case,
or from a 3D vertex run (run_info.txt + u_stepNNNN.txt, biofem-solid3d / pulse export; boundary surface of the mesh,
gamma from the ramp in run_info, V and h/h0 from the biofem post solid3d metrics).
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
    vertex_run = (case / "run_info.txt").exists()
    if vertex_run:                                     # 3D run: metres, boundary triangles of the volume mesh
        from .solid3d import read_run, metrics
        if args.field not in ("u", "twist"):
            sys.exit("error: 3D vertex runs: --field u or twist")
        vr = read_run(case, args.mesh)
        vm = metrics(vr)
        ks = sorted(vr["U"])[::args.every]
        if sorted(vr["U"])[-1] not in ks:
            ks.append(sorted(vr["U"])[-1])
        source, bad_steps, s = "3D vertex run", set(), 100.0
        gmax = float(vr["info"].get("gamma", 0.0))
        pc = {}                                        # Track P runs: cavity pressure per step (pv.csv)
        if (case / "pv.csv").exists():
            pv = np.genfromtxt(case / "pv.csv", delimiter=",", names=True)
            pcol = "p_kPa" if "p_kPa" in pv.dtype.names else "P_kPa"
            pc = {int(st): float(p) for st, p in zip(pv["step"], pv[pcol]) if st >= 0}
    else:
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

    auto_range = args.range.strip().lower() == "auto"
    if not auto_range:
        vmin, vmax = (float(x) for x in args.range.split(","))
    fmin, fmax = np.inf, -np.inf     # range of the coloured field over all frames (for --range auto)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in ("side", "base"):
            sys.exit(f"error: unknown view '{v}' (use side, base)")

    work = Path(tempfile.mkdtemp(prefix="biofem_animate_"))
    try:
        # 1. frames in cm with the colour field
        if vertex_run:
            X0 = vr["X0"] * s
            # the rim pins leave the base free to swing about the pinned point; remove the best-fit in-plane rigid
            # motion of the endocardial rim (lowest ring; the mu = const base is not a plane), Kabsch in x-y with
            # rotation about z, so |u| matches the shell's null-average BC
            zb = vr["X0"][:, 2].min()
            base = np.abs(vr["X0"][:, 2] - zb) < 1e-6 * np.ptp(vr["X0"][:, 2])
            def place(U):
                X = (vr["X0"] + U) * s
                P, Q = X0[base, :2], X[base, :2]
                cp, cq = P.mean(0), Q.mean(0)
                H = (Q - cq).T @ (P - cp)
                a = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])      # rotation taking Q onto P
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                X[:, :2] = (X[:, :2] - cq) @ R.T + cp
                return X
            # reference of the colours: end-diastole for Track P runs (u_step0000 = ED), else the unloaded mesh
            Xr = place(vr["U"][0])
            th0 = np.arctan2(Xr[:, 1], Xr[:, 0])
            info = {}
            for k in ks:
                X = place(vr["U"][k])
                # twist: rotation about the long axis relative to the reference, e3 = -z (apex -> base) as the torsion
                # metric; merid: sin(6 theta_ref), its zero contour = 12 material meridians (helices under torsion)
                tw = np.degrees(np.arctan2(-(X[:, 0] * Xr[:, 1] - X[:, 1] * Xr[:, 0]), X[:, 0] * Xr[:, 0] + X[:, 1] * Xr[:, 1]))
                fields = {"u": np.linalg.norm(X - Xr, axis=1), "twist": tw, "merid": np.sin(6.0 * th0)}
                _write_vtk(work / f"in_{k:06d}.vtk", X, vr["bnd"], fields)
                fld = fields[args.field]
                fmin, fmax = min(fmin, float(fld.min())), max(fmax, float(fld.max()))
                i = list(vm["k"]).index(k); t = k / vr["steps"]
                info[k] = dict(t=t, step=k, gamma=gmax * (1.0 - abs(2.0 * t - 1.0)), h=1.0 + vm["thick"][i] / 100.0, V=vm["V"][i],
                               p=pc.get(k))
        else:
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
                fld = fields.get("u" if args.field == "u" else "h_ratio")
                if fld is not None:
                    fmin, fmax = min(fmin, float(fld.min())), max(fmax, float(fld.max()))
                g = None
                if k in surf:
                    gv = VTK(surf[k]).cell_data.get("gamma")
                    g = float(gv.mean()) if gv is not None else None
                ti = file_step(source, k)
                info[k] = dict(t=max(ti, 0) * dt, step=ti, gamma=g, h=hr,
                               V=vol[k][1] * s ** 3 if k in vol else None)

        if auto_range:   # 0..max |u| (or min..max h/h0) over all frames, rounded outward to 2 significant digits
            def _nice(x, up):
                if x == 0 or not np.isfinite(x):
                    return 0.0
                e = 10.0 ** (np.floor(np.log10(abs(x))) - 1)
                return (np.ceil(x / e) if up else np.floor(x / e)) * e
            vmin = 0.0 if args.field == "u" else _nice(fmin, False)
            vmax = _nice(fmax, True)
            if args.field == "twist":                  # diverging: symmetric about 0
                vmax = _nice(max(abs(fmin), abs(fmax)), True); vmin = -vmax
            if vmax <= vmin:
                vmax = vmin + 1e-6
            print(f"[biofem] --range auto -> {vmin:g},{vmax:g}")

        # 2. render with ParaView
        field = args.field
        cmap = "coolwarm" if field == "twist" else "viridis"
        # no X display (ssh / batch node): render offscreen, else VTK's X interactor segfaults
        pv = [_pvpython()] + ([] if os.environ.get("DISPLAY") else ["--force-offscreen-rendering"])
        r = subprocess.run(pv + [str(RENDER_SCRIPT), str(work), str(work / "png"), field,
                            str(vmin), str(vmax), ",".join(views), cmap], capture_output=True, text=True)
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
            fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin, vmax), cmap), cax=cb).set_label(
                {"u": "|u| (cm)", "h_ratio": "h / h0", "twist": "rotation about the long axis (deg)"}[field], fontsize=10)
            d = info[k]
            parts = [label, f"t = {d['t']:.2f} s"]
            if d["gamma"] is not None:
                parts.append(f"$\\gamma$ = {d['gamma']:.3f}")
            if d.get("p") is not None:
                parts.append(f"$p_c$ = {d['p']:.2f} kPa")
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
