"""biofem post solid3d: LV metrics vs time for 3D runs in the "vertex run" format, with overlays.

A vertex run is a directory with
  run_info.txt          key=value lines: mesh (gmsh 2.2 .msh with physical surfaces 1 endo, 2 epi, 3 base), steps, gamma, source, ...
  u_stepNNNN.txt        one line per mesh vertex: X Y Z ux uy uz (reference position, displacement, metres)
written by biofem-solid3d (-problem lv) and, for the pulse reference, by pulse3d/export_vertex_u.py. Step k is at
t = k/steps of the gamma ramp (k = 0 is the reference). Vertices are matched to the mesh by position.

Metrics (definitions follow biofem post lv so shell runs can be overlaid; long axis z, apex at +z):
  - cavity volume V and EF(t) = (V0 - V)/V0: divergence theorem over the deformed endocardial triangles, origin on the base
    plane z_b (the endocardial rim keeps u_z = 0, so the planar cap adds nothing)
  - wall volume change (incompressibility check): sum of the deformed tetrahedra
  - mid-axis thickening: mean radius of the epicardial minus the endocardial vertices in the band |Z - z_mid| < --band,
    relative to the reference; z_mid halfway between the base plane and the mid-wall apex (as the shell's mid-surface band)
  - apex-base shortening of the epicardial apex vertex relative to the base plane (> 0: shortening)
  - torsion (Pezzuto & Ambrosi 2014): mean rotation of the epicardial vertices in the section at --torsion-sections[1]
    (apex) minus the one at [0] (base), fractions of the base-apex height, e3 from apex to base
--ref-shell overlays shell runs (biofem post lv), --ref a JSON of paper values (peak lines). With two or more --dir on the
same mesh, the last panel shows max |u_k - u_1| / max |u_1| of each run k against the first.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import style

PANELS = [("EF", "EF from the endocardial volume (%)", "EF"), ("thick", "mid-axis wall thickening (%)", "thickening"),
          ("short", "apex-base shortening (%)  (<0: elongation)", "shortening"), ("tors", "torsion, apex vs base (deg)", "torsion"),
          ("wall", "wall volume change (%)", None), ("du", "max |u - u_first| / max |u_first|", None)]


def read_msh22(path):
    """gmsh 2.2 ASCII: vertices (N,3), tetrahedra (M,4), triangles (K,3) and their physical tags (K,)."""
    lines = Path(path).read_text().split("\n")
    i = lines.index("$Nodes"); n = int(lines[i + 1])
    ids, X = {}, np.zeros((n, 3))
    for j in range(n):
        v = lines[i + 2 + j].split(); ids[int(v[0])] = j; X[j] = [float(c) for c in v[1:4]]
    i = lines.index("$Elements"); m = int(lines[i + 1])
    tets, tris, tags = [], [], []
    for j in range(m):
        v = [int(c) for c in lines[i + 2 + j].split()]
        etype, ntag = v[1], v[2]; nodes = [ids[c] for c in v[3 + ntag:]]
        if etype == 4: tets.append(nodes)
        elif etype == 2: tris.append(nodes); tags.append(v[3] if ntag else 0)
    return X, np.array(tets), np.array(tris), np.array(tags)


def read_run(run_dir, mesh_override=""):
    run_dir = Path(run_dir)
    info = dict(l.split("=", 1) for l in (run_dir / "run_info.txt").read_text().split("\n") if "=" in l)
    mesh = Path(mesh_override or info["mesh"])
    if not mesh.exists():
        raise FileNotFoundError(f"mesh {mesh} not found (use --mesh)")
    X0, tets, tris, tags = read_msh22(mesh)
    from scipy.spatial import cKDTree
    tree, steps = cKDTree(X0), int(info["steps"])
    U = {0: np.zeros_like(X0)}
    for f in sorted(run_dir.glob("u_step*.txt")):
        d = np.loadtxt(f)
        dist, order = tree.query(d[:, :3])
        if dist.max() > 1e-8 or len(np.unique(order)) != len(X0):
            raise ValueError(f"{f}: vertices do not match the mesh {mesh} (max distance {dist.max():.2e})")
        u = np.zeros_like(X0); u[order] = d[:, 3:6]
        U[int(f.stem[6:])] = u
    return dict(info=info, X0=X0, tets=tets, endo=tris[tags == 1], epi_v=np.unique(tris[tags == 2]),
                endo_v=np.unique(tris[tags == 1]), steps=steps, U=U)


def metrics(r, band_cm=0.4, sections=(0.10, 0.85)):
    X0, endo = r["X0"], r["endo"]
    ev, pv = r["endo_v"], r["epi_v"]
    zb = X0[ev, 2].min()                                          # endocardial rim = base plane
    za = 0.5 * (X0[ev, 2].max() + X0[pv, 2].max())                # mid-wall apex
    zmid, H, band = 0.5 * (zb + za), za - zb, band_cm / 100.0
    be, bp = np.abs(X0[ev, 2] - zmid) < band, np.abs(X0[pv, 2] - zmid) < band
    frac = (X0[pv, 2] - zb) / H
    sb, sa = np.abs(frac - sections[0]) < 0.04, np.abs(frac - sections[1]) < 0.04
    iapex = pv[np.argmax(X0[pv, 2])]
    T = r["tets"]
    def cavity(P):
        o = np.array([0.0, 0.0, zb]); A, B, C = P[endo[:, 0]] - o, P[endo[:, 1]] - o, P[endo[:, 2]] - o
        return abs(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)
    def wall(P):
        a, b, c, d = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]], P[T[:, 3]]
        return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6.0
    def theta(P, P0):
        x, X = P[:, :2], P0[:, :2]
        return np.degrees(np.arctan2(-(x[:, 0] * X[:, 1] - x[:, 1] * X[:, 0]), np.sum(x * X, axis=1)))   # e3 = -z
    rad = lambda P: np.hypot(P[:, 0], P[:, 1])
    V0, W0, L0 = cavity(X0), wall(X0), X0[iapex, 2] - zb
    h0 = rad(X0[pv][bp]).mean() - rad(X0[ev][be]).mean()
    out = {k: [] for k in ("t", "k", "V", "EF", "wall", "thick", "short", "tors")}
    for k in sorted(r["U"]):
        P = X0 + r["U"][k]
        th = theta(P[pv], X0[pv])
        out["t"].append(k / r["steps"]); out["k"].append(k)
        V = cavity(P); out["V"].append(V * 1e6); out["EF"].append(100 * (V0 - V) / V0)
        out["wall"].append(100 * (wall(P) / W0 - 1))
        out["thick"].append(100 * ((rad(P[pv][bp]).mean() - rad(P[ev][be]).mean()) / h0 - 1))
        out["short"].append(100 * (L0 - (P[iapex, 2] - zb)) / L0)
        out["tors"].append(th[sa].mean() - th[sb].mean())
    return {k: np.array(v) for k, v in out.items()}


def cmd_solid3d(args):
    plt = style.setup(args.show)
    dirs = args.dir or ["."]
    labels = style.labels_for(dirs, args.label)
    secs = tuple(float(v) for v in args.torsion_sections.split(","))
    runs, data = [], []
    for d, lab in zip(dirs, labels):
        r = read_run(d, args.mesh); runs.append(r)
        data.append((lab, metrics(r, args.band, secs)))
        print(f"[biofem] {lab}: {d} (source {r['info'].get('source', '?')}, {len(r['U']) - 1} saved steps)")
    shells = []
    if args.ref_shell:
        from .lv import compute
        for d in args.ref_shell:
            s = compute(d, None, "m", "+z", args.band, secs)
            s["EF"] = 100 * (s["V_endo"][0] - s["V_endo"]) / s["V_endo"][0]
            shells.append((f"shell {Path(d).resolve().name}", s))
    ref = json.loads(Path(args.ref).read_text()) if args.ref else {}

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, (lab, m) in enumerate(data):
        c = style.case_color(i)
        for ax, (key, title, _) in zip(axs.flat, PANELS):
            if key in m: ax.plot(m["t"], m[key], color=c, lw=1.8, label=lab)
            ax.set_title(title, fontsize=9)
    for j, (lab, s) in enumerate(shells):
        c = style.case_color(len(data) + j)
        for ax, (key, _, _) in zip(axs.flat, PANELS):
            sk = {"EF": "EF", "thick": "thick_radial", "short": "shortening", "tors": "torsion"}.get(key)
            if sk and sk in s: ax.plot(s["t"], s[sk], color=c, lw=1.5, ls="--", label=lab)
    du_rows = []
    if len(runs) > 1:                                   # displacement difference against the first run, same mesh
        base = runs[0]
        for i, (r, (lab, _)) in enumerate(zip(runs[1:], data[1:]), start=1):
            ks = [k for k in sorted(r["U"]) if k in base["U"] and k > 0]
            v = [np.abs(r["U"][k] - base["U"][k]).max() / max(np.abs(base["U"][k]).max(), 1e-300) for k in ks]
            rms = [np.sqrt(((r["U"][k] - base["U"][k]) ** 2).sum(1).mean()) / np.sqrt((base["U"][k] ** 2).sum(1).mean()) for k in ks]
            axs.flat[5].semilogy([k / base["steps"] for k in ks], v, color=style.case_color(i), lw=1.8, label=f"{lab}: max")
            axs.flat[5].semilogy([k / base["steps"] for k in ks], rms, color=style.case_color(i), lw=1.2, ls="--", label=f"{lab}: RMS")
            du_rows += [(lab, k, a, b) for k, a, b in zip(ks, v, rms)]
    for ax, (key, title, rk) in zip(axs.flat, PANELS):
        if rk and rk in ref:
            ax.axhline(ref[rk], color=style.INK2, lw=1, ls=":", label=f"{ref.get('label', 'ref')} peak {ref[rk]:g}")
        style.finish(ax)
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False, fontsize=7)
    if args.title: fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    out = Path(args.out or "solid3d_metrics.png")
    fig.savefig(out, dpi=130)
    print(f"[biofem] plot saved -> {out}")

    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh); cols = ["t", "k", "V", "EF", "wall", "thick", "short", "tors"]
        w.writerow(["case"] + cols)
        for lab, m in data:
            for j in range(len(m["t"])): w.writerow([lab] + [f"{m[c][j]:.6g}" for c in cols])
    if du_rows:
        with open(out.with_name(out.stem + "_du.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["case", "k", "max_rel", "rms_rel"])
            for row in du_rows: w.writerow([row[0], row[1], f"{row[2]:.6g}", f"{row[3]:.6g}"])
    print(f"[biofem] data saved -> {out.with_suffix('.csv')}")

    names = [("EF", "peak EF (%)", max), ("thick", "peak thickening (%)", max), ("short", "peak shortening (%)", max),
             ("short", "peak elongation (%)", lambda a: -min(a)), ("tors", "peak torsion (deg)", lambda a: a[np.argmax(np.abs(a))]),
             ("wall", "max |wall vol. change| (%)", lambda a: max(abs(a)))]
    width = 2 + max(14, *(len(l) for l, _ in data))
    print("\n" + f"{'':28s}" + "".join(f"{l:>{width}s}" for l, _ in data) + (f"{ref.get('label', 'ref'):>{width}s}" if ref else ""))
    for key, name, f in names:
        line = f"{name:28s}" + "".join(f"{f(m[key]):>{width}.4g}" for _, m in data)
        rk = {"EF": "EF", "thick": "thickening", "tors": "torsion"}.get(key) if "elongation" not in name else None
        if "shortening" in name: rk = "shortening"
        if ref: line += f"{ref[rk]:>{width}.4g}" if rk and rk in ref else f"{'':>{width}s}"
        print(line)
    if args.show: plt.show()
