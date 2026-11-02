"""biofem post solid3d: LV metrics vs time for 3D runs in the "vertex run" format, with overlays.

A vertex run is a directory with
  run_info.txt          key=value lines: mesh (gmsh 2.2 .msh with physical surfaces 1 endo, 2 epi, 3 base), steps, gamma, source, ...
  u_stepNNNN.txt        one line per mesh vertex: X Y Z ux uy uz (reference position, displacement, metres)
written by biofem-solid3d (-problem lv) and, for the pulse reference, by pulse3d/export_vertex_u.py. Step k is at
t = k/steps of the gamma ramp (k = 0 is the reference). Vertices are matched to the mesh by position.
Runs with cavity pressure (Track P, p_ed in run_info.txt) also write u_step0000.txt = end-diastole and pv.csv; their metrics
are relative to end-diastole (as Pezzuto), and a PV-loop figure <out>_pv.png is written.

Metrics (definitions follow biofem post lv so shell runs can be overlaid; long axis z, apex at +z):
  - cavity volume V and EF(t) = (V0 - V)/V0: divergence theorem over the deformed endocardial triangles, origin on the base
    plane z_b (the endocardial rim keeps u_z = 0, so the planar cap adds nothing)
  - wall volume change (incompressibility check): sum of the deformed tetrahedra
  - mid-axis thickening: mean radius of the epicardial minus the endocardial vertices in the band |Z - z_mid| < --band,
    relative to the reference; z_mid halfway between the base plane and the mid-wall apex (as the shell's mid-surface band)
  - apex-base shortening of the epicardial apex vertex relative to the base plane (> 0: shortening)
  - torsion (Pezzuto & Ambrosi 2014): mean rotation of the epicardial vertices in the section at --torsion-sections[1]
    (apex) minus the one at [0] (base), fractions of the base-apex height, e3 from apex to base
--ref-curves JSON (e.g. Pezzuto 2014 Fig. 6 digitized: panels radial_position_cm, torsion_case2_deg, I1e, I4fe, each
with endo / mid / epi lists of [s, value], s = phase coordinate 0 EDP, 1 AVO, 2 ESP, 3 MVO, 4 EDP) writes <out>_curves.png:
the same quantities of the runs on that phase axis (Track P runs: valve events from pv.csv), at the mid-axis section:
radial position of the endo / epi surfaces; torsion from the unloaded mesh (as the paper's Fig. 6) of the depth classes
sub-endo (psi < 0.15), mid (0.4..0.6), sub-epi (psi > 0.85), psi = transmural prolate depth, each at --torsion-sections of
its own base-apex height, points < 3 mm from the axis left out; I1 and I4f of Ce = Fa^-T C Fa^-1 averaged over the tets
of the band in the same classes (P1 F from the vertex values). It also writes <out>_phase.png: volume, pressure,
thickening and shortening from end-diastole on the same phase axis against the paper's time-dependent data.
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
    # match vertices by position (numpy only): hash the mesh vertices into 10 um cells, look each point up in its cell
    # and the 26 neighbours (no rounding-boundary misses); mesh vertices are >= ~0.1 mm apart
    cell = lambda P: np.floor(P / 1e-5).astype(np.int64)
    grid = {}
    for i, c in enumerate(map(tuple, cell(X0))):
        grid.setdefault(c, []).append(i)
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    def match(P):
        order = np.full(len(P), -1); dist = np.full(len(P), np.inf)
        for j, (p, c) in enumerate(zip(P, map(tuple, cell(P)))):
            for o in offs:
                for i in grid.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), ()):
                    dd = np.linalg.norm(X0[i] - p)
                    if dd < dist[j]: dist[j], order[j] = dd, i
        return dist, order
    steps = int(info["steps"])
    U = {0: np.zeros_like(X0)}
    for f in sorted(run_dir.glob("u_step*.txt")):
        d = np.loadtxt(f)
        dist, order = match(d[:, :3])
        if dist.max() > 1e-8 or len(np.unique(order)) != len(X0):
            raise ValueError(f"{f}: vertices do not match the mesh {mesh} (max distance {dist.max():.2e})")
        u = np.zeros_like(X0); u[order] = d[:, 3:6]
        U[int(f.stem[6:])] = u
    return dict(info=info, X0=X0, tets=tets, bnd=tris, endo=tris[tags == 1], epi_v=np.unique(tris[tags == 2]),
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
    R = X0 + r["U"][0]                                            # reference of the metrics: end-diastole with pressure
    V0, W0, L0 = cavity(R), wall(X0), R[iapex, 2] - zb
    h0 = rad(R[pv][bp]).mean() - rad(R[ev][be]).mean()
    out = {k: [] for k in ("t", "k", "V", "EF", "wall", "thick", "short", "tors")}
    for k in sorted(r["U"]):
        P = X0 + r["U"][k]
        th = theta(P[pv], R[pv])
        out["t"].append(k / r["steps"]); out["k"].append(k)
        V = cavity(P); out["V"].append(V * 1e6); out["EF"].append(100 * (V0 - V) / V0)
        out["wall"].append(100 * (wall(P) / W0 - 1))
        out["thick"].append(100 * ((rad(P[pv][bp]).mean() - rad(P[ev][be]).mean()) / h0 - 1))
        out["short"].append(100 * (L0 - (P[iapex, 2] - zb)) / L0)
        out["tors"].append(th[sa].mean() - th[sb].mean())
    return {k: np.array(v) for k, v in out.items()}


PROLATE = dict(d=0.0291, le=0.557, lp=1.00)          # Pezzuto geometry (biofem-solid3d prolate_frame defaults)


def prolate_depth_fiber(X, a_endo=-60.0, a_epi=60.0, d=PROLATE["d"], le=PROLATE["le"], lp=PROLATE["lp"]):
    """transmural depth psi (0 endo, 1 epi) and fiber f at reference points X, as biofem-solid3d prolate_frame"""
    r1 = np.linalg.norm(X - [0, 0, d], axis=1); r2 = np.linalg.norm(X + [0, 0, d], axis=1)
    lam = np.arccosh(np.maximum(1.0, (r1 + r2) / (2 * d))); mu = np.arccos(np.clip((r2 - r1) / (2 * d), -1, 1))
    th = np.arctan2(X[:, 1], X[:, 0]); psi = np.clip((lam - le) / (lp - le), 0, 1)
    al = np.radians(a_endo * (1 - psi) + a_epi * psi)
    em = np.c_[np.sinh(lam) * np.cos(mu) * np.cos(th), np.sinh(lam) * np.cos(mu) * np.sin(th), -np.cosh(lam) * np.sin(mu)]
    em /= np.maximum(np.linalg.norm(em, axis=1), 1e-30)[:, None]
    et = np.c_[-np.sin(th), np.cos(th), np.zeros_like(th)]
    f = np.cos(al)[:, None] * et + np.sin(al)[:, None] * em
    return psi, f / np.linalg.norm(f, axis=1)[:, None]


def phase_events(run_dir, steps):
    """t of EDP, AVO, ESP, MVO, EDP from pv.csv (midpoint between the steps around a valve event); None without pv.csv"""
    f = Path(run_dir) / "pv.csv"
    if not f.exists(): return None
    pv = np.genfromtxt(f, delimiter=",", names=True); pv = pv[pv["step"] >= 0]
    t, ph = pv["t"], pv["phase"].astype(int)
    def first(p): i = np.argmax(ph == p); return 0.5 * (t[i - 1] + t[i])
    return [0.0, first(2), 0.5, first(4), 1.0]


def curve_metrics(r, band_cm=0.4, sections=(0.10, 0.85)):
    """time series of the --ref-curves quantities (see the module doc)"""
    X0, T = r["X0"], r["tets"]; ev, pv_ = r["endo_v"], r["epi_v"]
    a_endo, a_epi = float(r["info"].get("alpha_endo", -60)), float(r["info"].get("alpha_epi", 60))
    gmax = float(r["info"].get("gamma", 0.2))
    zb = X0[ev, 2].min(); za = 0.5 * (X0[ev, 2].max() + X0[pv_, 2].max()); zmid, H = 0.5 * (zb + za), za - zb
    band = band_cm / 100.0
    geo = dict(d=float(r["info"].get("prolate_d", PROLATE["d"])), le=float(r["info"].get("lam_endo", PROLATE["le"])),
               lp=float(r["info"].get("lam_epi", PROLATE["lp"])))
    psi_v, _ = prolate_depth_fiber(X0, a_endo, a_epi, **geo)
    frac = (X0[:, 2] - zb) / H
    cls = {"endo": psi_v < 0.15, "mid": (psi_v > 0.4) & (psi_v < 0.6), "epi": psi_v > 0.85}
    # torsion sections per layer, at sections[0] / [1] of THAT layer's own base-apex height (the endocardium ends lower:
    # mid-wall fractions put its apex section among points next to the axis), points < 3 mm from the axis left out
    off = np.hypot(X0[:, 0], X0[:, 1]) > 0.003
    lsec = {}
    for c, m in cls.items():
        zl = X0[m, 2].max(); fl = (X0[:, 2] - zb) / (zl - zb)
        lsec[c] = (m & off & (np.abs(fl - sections[1]) < 0.04), m & off & (np.abs(fl - sections[0]) < 0.04))
    be, bp = ev[np.abs(X0[ev, 2] - zmid) < band], pv_[np.abs(X0[pv_, 2] - zmid) < band]
    C = X0[T].mean(1); psi_c, f_c = prolate_depth_fiber(C, a_endo, a_epi, **geo)
    tb = np.abs(C[:, 2] - zmid) < band
    tcls = {k: tb & m for k, m in {"endo": psi_c < 0.15, "mid": (psi_c > 0.4) & (psi_c < 0.6), "epi": psi_c > 0.85}.items()}
    Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], axis=2)
    Dinv = np.linalg.inv(Dm)
    rad = lambda P: np.hypot(P[:, 0], P[:, 1])
    def theta(P, P0):
        return np.degrees(np.arctan2(-(P[:, 0] * P0[:, 1] - P[:, 1] * P0[:, 0]), P[:, 0] * P0[:, 0] + P[:, 1] * P0[:, 1]))
    out = {"t": [], "rad_endo": [], "rad_epi": []}
    for k in ("tors", "I1e", "I4fe"):
        for c in cls: out[f"{k}_{c}"] = []
    for k in sorted(r["U"]):
        t = k / r["steps"]; g = gmax * (1 - abs(2 * t - 1)); P = X0 + r["U"][k]
        out["t"].append(t)
        out["rad_endo"].append(100 * rad(P[be]).mean()); out["rad_epi"].append(100 * rad(P[bp]).mean())
        th = theta(P, X0)
        for c, (a_, b_) in lsec.items(): out[f"tors_{c}"].append(th[a_].mean() - th[b_].mean())
        Ds = np.stack([P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]], P[T[:, 3]] - P[T[:, 0]]], axis=2)
        F = Ds @ Dinv
        A = (1 / (1 - g) - np.sqrt(1 - g)) * np.einsum("ni,nj->nij", f_c, f_c) + np.sqrt(1 - g) * np.eye(3)   # Fa^-1
        Fe = F @ A
        I1 = np.einsum("nij,nij->n", Fe, Fe); Ff = np.einsum("nij,nj->ni", F, f_c)
        I4 = np.einsum("ni,ni->n", Ff, Ff) / (1 - g) ** 2
        for c, m in tcls.items(): out[f"I1e_{c}"].append(I1[m].mean()); out[f"I4fe_{c}"].append(I4[m].mean())
    return {k: np.array(v) for k, v in out.items()}


def shell_curve_metrics(case_dir, band_cm=0.4, sections=(0.10, 0.85)):
    """--ref-curves quantities of a shell Track P run (biofem post lv files): radial position of the endo / epi layers
    (shell3d) at the mid-axis section and torsion of the endo, mid (surface) and epi layers relative to the unloaded
    mesh (file 0); t = (file - 1) dt (file 1 = cycle step 0). No elastic invariants (the shell does not write them)."""
    from .io import VTK, read_control, output_dir, output_files, control_float
    ctrl = read_control(case_dir); out = output_dir(Path(case_dir), ctrl, None); dt = control_float(ctrl, "-dt")
    surf = dict(output_files(out, "surface")); s3 = dict(output_files(out, "shell3d"))
    ks = sorted(k for k in surf if k in s3)
    M0 = VTK(surf[ks[0]]).points * 100.0; n = len(M0)
    zb, za = M0[:, 2].min(), M0[:, 2].max(); H = za - zb; zmid = 0.5 * (zb + za)
    band = np.abs(M0[:, 2] - zmid) < band_cm
    rad = lambda P: np.hypot(P[:, 0], P[:, 1])
    def layers(k):
        P = VTK(s3[k]).points * 100.0; A, B = P[:n], P[n:]
        return (A, B) if rad(A).mean() < rad(B).mean() else (B, A)
    E0, P0 = layers(ks[0])
    def sec(X):                                          # per-layer sections, as curve_metrics; not within 3 mm of the axis
        fl = (X[:, 2] - zb) / (X[:, 2].max() - zb); off = np.hypot(X[:, 0], X[:, 1]) > 0.3
        return off & (np.abs(fl - sections[1]) < 0.04), off & (np.abs(fl - sections[0]) < 0.04)
    secs_ = {"endo": sec(E0), "mid": sec(M0), "epi": sec(P0)}
    def theta(P, R):
        return np.degrees(np.arctan2(-(P[:, 0] * R[:, 1] - P[:, 1] * R[:, 0]), P[:, 0] * R[:, 0] + P[:, 1] * R[:, 1]))
    out_ = {"t": [], "rad_endo": [], "rad_epi": [], "tors_endo": [], "tors_mid": [], "tors_epi": []}
    for k in ks:
        if k < 1: continue
        E, P = layers(k); Mk = VTK(surf[k]).points * 100.0
        out_["t"].append((k - 1) * dt)
        out_["rad_endo"].append(rad(E[band]).mean()); out_["rad_epi"].append(rad(P[band]).mean())
        for L, X, X0 in (("endo", E, E0), ("mid", Mk, M0), ("epi", P, P0)):
            th = theta(X, X0); a_, b_ = secs_[L]; out_[f"tors_{L}"].append(th[a_].mean() - th[b_].mean())
    return {k: np.array(v) for k, v in out_.items()}, out


def plot_ref_curves(plt, runs, dirs, labels, ref, out, band, secs, shells=()):
    panels = [("radial_position_cm", "radial position, mid-axis section (cm)", ["endo", "epi"], "rad"),
              ("torsion_case2_deg", "torsion, apex vs base, from unloaded (deg)", ["endo", "mid", "epi"], "tors"),
              ("I1e", r"$I_1^e$, mid-axis section", ["endo", "mid", "epi"], "I1e"),
              ("I4fe", r"$I_{4f}^e$, mid-axis section", ["endo", "mid", "epi"], "I4fe")]
    ls = {"endo": "--", "mid": "-.", "epi": "-"}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    rows = []
    cases = [(lab, curve_metrics(r, band, secs), phase_events(d, r["steps"])) for r, d, lab in zip(runs, dirs, labels)]
    for d in shells:                                     # shell Track P runs
        m, od = shell_curve_metrics(d, band, secs)
        cases.append((f"shell {Path(d).resolve().name}", m, phase_events(od, None)))
    for i, (lab, m, ev) in enumerate(cases):
        if ev is None: continue
        s = np.interp(m["t"], ev, [0, 1, 2, 3, 4])
        for ax, (_, _, layers, key) in zip(axs, panels):
            for L in layers:
                if f"{key}_{L}" not in m: continue
                y = m[f"{key}_{L}"]
                ax.plot(s, y, color=style.case_color(i), lw=1.8, ls=ls[L], label=f"{lab} {L}" if ax is axs[0] or L == "epi" else None)
                rows += [(lab, key, L, a, b) for a, b in zip(s, y)]
    for ax, (pk, title, layers, _) in zip(axs, panels):
        for L in layers:
            c = np.array(ref["panels"].get(pk, {}).get(L, []))
            if len(c): ax.plot(c[:, 0], c[:, 1], color=style.INK, lw=1.2, ls=ls[L], alpha=0.8,
                               label=f"{ref.get('label', 'Pezzuto 2014')} {L}")
        ax.set_title(title, fontsize=9); style.finish(ax, "")
        ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["EDP", "AVO", "ESP", "MVO", "EDP"])
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    f = out.with_name(out.stem + "_curves.png"); fig.savefig(f, dpi=130)
    with open(f.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["case", "quantity", "layer", "s", "value"])
        for row in rows: w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}", f"{row[4]:.6g}"])
    print(f"[biofem] curves vs {ref.get('label', 'reference')} saved -> {f}")


def plot_phase_metrics(plt, runs, dirs, labels, shells, ref, out, band, secs):
    """<out>_phase.png: the cycle on the paper's phase axis (0 EDP, 1 AVO, 2 ESP, 3 MVO, 4 EDP; each run's own valve
    events from pv.csv), against the paper's time-dependent data only: cavity volume / pressure at its Fig. 4 points
    (constant segments where the protocol fixes them), wall thickening from its Fig. 6 endo / epi radii. Shortening: no
    curve published (text value only). Torsion: <out>_curves.png."""
    from .lv import compute
    # torsion: in <out>_curves.png (endo / mid / epi from the unloaded state, as the paper's Fig. 6)
    panels = [("V", "cavity volume (ml)"), ("p", "cavity pressure (kPa)"), ("thick", "mid-axis wall thickening vs ED (%)"),
              ("short", "apex-base shortening vs ED (%)  (<0: elongation)")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2)); ax = dict(zip([k for k, _ in panels], axs))
    rows, cases = [], []
    for r, d, lab in zip(runs, dirs, labels):
        ev = phase_events(d, r["steps"])
        if ev is None: continue
        m = metrics(r, band, secs); pv = np.genfromtxt(Path(d) / "pv.csv", delimiter=",", names=True); pv = pv[pv["step"] >= 0]
        cases.append((lab, ev, dict(t=m["t"], thick=m["thick"], tors=m["tors"], short=m["short"]),
                      dict(t=pv["t"], V=pv["V_ml"], p=pv["p_kPa" if "p_kPa" in pv.dtype.names else "P_kPa"])))
    for d in shells:
        sh = compute(d, None, "m", "+z", band, secs)
        from .io import read_control, output_dir
        od = output_dir(Path(d), read_control(Path(d)), None); ev = phase_events(od, None)
        if ev is None: continue
        pv = np.genfromtxt(od / "pv.csv", delimiter=",", names=True)
        i0 = sh.get("iref", 0)                           # drop the unloaded file 0 (also at t = 0)
        cut = lambda a: None if a is None else a[i0:]
        cases.append((f"shell {Path(d).resolve().name}", ev, dict(t=sh["t"][i0:], thick=cut(sh.get("thick_radial")),
                      tors=cut(sh["torsion"]), short=cut(sh["shortening"])), dict(t=pv["t"], V=pv["V_ml"], p=pv["p_kPa"])))
    for i, (lab, ev, m, pv) in enumerate(cases):
        c = style.case_color(i)
        for key, src in (("V", pv), ("p", pv), ("thick", m), ("short", m)):
            if src.get(key) is None: continue
            sx = np.interp(src["t"], ev, [0, 1, 2, 3, 4])
            ax[key].plot(sx, src[key], color=c, lw=1.8, label=lab)
            rows += [(lab, key, a, b) for a, b in zip(sx, src[key])]
    k = ref.get("key_points_fig4", {}); P = ref.get("panels", {}); lab = "Pezzuto 2014"
    if k:
        Vk = [k["EDV_ml"], k["EDV_ml"], k["ESV_ml"], k["ESV_ml"], k["EDV_ml"]]
        ax["V"].plot([0, 1], Vk[:2], color=style.INK, lw=1.2, ls=":"); ax["V"].plot([2, 3], Vk[2:4], color=style.INK, lw=1.2, ls=":")
        ax["V"].plot(range(5), Vk, "o", color=style.INK, ms=5, label=f"{lab} (Fig. 4 table; isovolumic exact)")
        ped, pao = k["p_ed_mmHg"] * 0.133322, k["p_ao_mmHg"] * 0.133322
        ax["p"].plot([1, 2], [pao, pao], color=style.INK, lw=1.2, ls=":"); ax["p"].plot([3, 4], [ped, ped], color=style.INK, lw=1.2, ls=":")
        ax["p"].plot(range(5), [ped, pao, pao, ped, ped], "o", color=style.INK, ms=5, label=f"{lab} (protocol; valve points)")
    rp = P.get("radial_position_cm", {})
    if rp.get("endo") and rp.get("epi"):
        en, ep = np.array(rp["endo"]), np.array(rp["epi"])
        sg = np.linspace(max(en[0, 0], ep[0, 0]), min(en[-1, 0], ep[-1, 0]), 200)
        w = np.interp(sg, ep[:, 0], ep[:, 1]) - np.interp(sg, en[:, 0], en[:, 1])
        ax["thick"].plot(sg, 100 * (w / w[0] - 1), color=style.INK, lw=1.4, label=f"{lab} (Fig. 6 radii)")
    ax["short"].plot([2], [4.0], "o", mfc="none", color=style.INK, ms=6, label=f"{lab}: text value ~4% (no curve)")
    for key, title in panels:
        a = ax[key]; a.set_title(title, fontsize=9); style.finish(a, "")
        a.set_xticks([0, 1, 2, 3, 4]); a.set_xticklabels(["EDP", "AVO", "ESP", "MVO", "EDP"]); a.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    f = out.with_name(out.stem + "_phase.png"); fig.savefig(f, dpi=130)
    with open(f.with_suffix(".csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["case", "quantity", "s", "value"])
        for row in rows: wr.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.6g}"])
    print(f"[biofem] cycle vs {lab} on the phase axis saved -> {f}")


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
            i0 = s.get("iref", 0)                       # end-diastole for Track P shell runs
            s["EF"] = 100 * (s["V_endo"][i0] - s["V_endo"]) / s["V_endo"][i0]
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
            if r["X0"].shape != base["X0"].shape or not np.allclose(r["X0"], base["X0"]): continue   # other mesh
            ks = [k for k in sorted(r["U"]) if k in base["U"] and k > 0]
            v = [np.abs(r["U"][k] - base["U"][k]).max() / max(np.abs(base["U"][k]).max(), 1e-300) for k in ks]
            rms = [np.sqrt(((r["U"][k] - base["U"][k]) ** 2).sum(1).mean()) / np.sqrt((base["U"][k] ** 2).sum(1).mean()) for k in ks]
            axs.flat[5].semilogy([k / base["steps"] for k in ks], v, color=style.case_color(i), lw=1.8, label=f"{lab}: max")
            axs.flat[5].semilogy([k / base["steps"] for k in ks], rms, color=style.case_color(i), lw=1.2, ls="--", label=f"{lab}: RMS")
            du_rows += [(lab, k, a, b) for k, a, b in zip(ks, v, rms)]
    if not du_rows:
        axs.flat[5].set_visible(False)                  # one 3D run: nothing to difference
    for ax, (key, title, rk) in zip(axs.flat, PANELS):
        if not ax.get_visible(): continue
        if rk and rk in ref:
            ax.axhline(ref[rk], color=style.INK2, lw=1, ls=":", label=f"{ref.get('label', 'ref')} peak {ref[rk]:g}")
        style.finish(ax)
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False, fontsize=7)
    if args.title: fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    out = Path(args.out or "solid3d_metrics.png")
    fig.savefig(out, dpi=130)
    print(f"[biofem] plot saved -> {out}")

    def read_pv(f):                                     # biofem-solid3d (p_kPa) or the pulse script (P_kPa, filling rows < 0)
        pv = np.genfromtxt(f, delimiter=",", names=True)
        pv = pv[pv["step"] >= 0]
        return {"t": pv["t"], "V_ml": pv["V_ml"], "p_kPa": pv["p_kPa" if "p_kPa" in pv.dtype.names else "P_kPa"]}
    pvs = [(lab, read_pv(Path(d) / "pv.csv")) for d, lab in zip(dirs, labels) if (Path(d) / "pv.csv").exists()]
    for d in args.ref_shell:                            # shell Track P runs: <case>/<out_dir>/pv.csv (out_dir from control.dat)
        from .io import read_control, output_dir
        f = output_dir(Path(d), read_control(Path(d)), None) / "pv.csv"
        if f.exists(): pvs.append((f"shell {Path(d).resolve().name}", read_pv(f)))
    if pvs:                                             # Track P: PV loop and pressure / volume vs time
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for i, (lab, pv) in enumerate(pvs):
            c = style.case_color(i)
            axs[0].plot(pv["V_ml"], pv["p_kPa"], color=c, lw=1.8, label=lab)
            axs[1].plot(pv["t"], pv["p_kPa"], color=c, lw=1.8, label=lab)
            axs[2].plot(pv["t"], pv["V_ml"], color=c, lw=1.8, label=lab)
            edv, esv = pv["V_ml"][0], pv["V_ml"].min()
            print(f"[biofem] {lab}: EDV {edv:.2f} ml, ESV {esv:.2f} ml, SV {edv - esv:.2f} ml, EF {100 * (edv - esv) / edv:.1f}%")
        for ax, (xl, yl) in zip(axs, [("cavity volume (ml)", "cavity pressure (kPa)"), ("t / T", "cavity pressure (kPa)"),
                                      ("t / T", "cavity volume (ml)")]):
            ax.set_ylabel(yl, fontsize=9); style.finish(ax, xl)
        kp = json.loads(Path(args.ref_curves).read_text()).get("key_points_fig4") if args.ref_curves else None
        if kp:                                          # EDP, AVO, ESP, MVO of the paper's PV loop (mmHg -> kPa)
            mm = 0.133322
            V = [kp["EDV_ml"], kp["EDV_ml"], kp["ESV_ml"], kp["ESV_ml"], kp["EDV_ml"]]
            P = [kp["p_ed_mmHg"] * mm, kp["p_ao_mmHg"] * mm, kp["p_ao_mmHg"] * mm, kp["p_ed_mmHg"] * mm, kp["p_ed_mmHg"] * mm]
            axs[0].plot(V, P, color=style.INK, lw=1.0, ls=":", marker="o", ms=4, label="Pezzuto 2014 (Fig. 4 points)")
        axs[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out.with_name(out.stem + "_pv.png"), dpi=130)
        print(f"[biofem] PV loop saved -> {out.with_name(out.stem + '_pv.png')}")

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

    if args.ref_curves:
        refc = json.loads(Path(args.ref_curves).read_text())
        plot_ref_curves(plt, runs, dirs, labels, refc, out, args.band, secs, shells=args.ref_shell)
        plot_phase_metrics(plt, runs, dirs, labels, args.ref_shell, refc, out, args.band, secs)

    names = [("EF", "peak EF (%)", max), ("thick", "peak thickening (%)", max), ("short", "peak shortening (%)", max),
             ("short", "peak elongation (%)", lambda a: -min(a)), ("tors", "peak torsion (deg)", lambda a: a[np.argmax(np.abs(a))]),
             ("wall", "max |wall vol. change| (%)", lambda a: max(abs(a)))]
    shell_cols = [(lab, {"EF": s["EF"], "thick": s["thick_radial"], "short": s["shortening"], "tors": s["torsion"]})
                  for lab, s in shells if "thick_radial" in s]
    cols = data + shell_cols
    width = 2 + max(14, *(len(l) for l, _ in cols))
    print("\n" + f"{'':28s}" + "".join(f"{l:>{width}s}" for l, _ in cols) + (f"{ref.get('label', 'ref'):>{width}s}" if ref else ""))
    for key, name, f in names:
        line = f"{name:28s}" + "".join(f"{f(m[key]):>{width}.4g}" if key in m else f"{'-':>{width}s}" for _, m in cols)
        rk = {"EF": "EF", "thick": "thickening", "tors": "torsion"}.get(key) if "elongation" not in name else None
        if "shortening" in name: rk = "shortening"
        if ref: line += f"{ref[rk]:>{width}.4g}" if rk and rk in ref else f"{'':>{width}s}"
        print(line)
    if args.show: plt.show()
