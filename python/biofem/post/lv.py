"""biofem post lv: left-ventricle metrics vs time.

From the output series (surface00_*.vtk, shell3d00_*.vtk if -write_3d_shell 1, lv_volume00.dat):
  - cavity volume (endocardial and mid-surface), EF = (Vmax - Vmin)/Vmax, SV = Vmax - Vmin
  - mean and max |u|, mean nodal rate |dx|/dt
  - wall thickening in a mid-axis band: radial (epi minus endo radius, shell3d) and normal
    (h/h0 from shell3d thickness, or the lambda3 cell field without shell3d)
  - apex-base length change (shortening > 0), base plane = reference rim
  - torsion as in Pezzuto & Ambrosi 2014 (p. 1587): rotation theta of a point about the long axis e3,
    sin(theta) = ((x ^ X) . e3) / (r R), with e3 pointing from apex to base and R, r the distances from
    the axis; torsion = theta(section just above the apex) - theta(section near the base). Here every
    epicardial point in each section is used (mean), not one point, so mesh asymmetry averages out
    (mid-surface without shell3d). Sections at --torsion-sections (fractions of base-apex height).
Geometry convention: long axis along z, apex at +z (or -z with --apex -z), base = the other end.
Lengths are reported in cm and volumes in ml; --solver-unit gives the solver's length unit.
Reference values (e.g. from a paper) come from --ref <json>, never from this code:
  {"label": "...", "EF": 56, "SV": 22.6, "thickening": 36, "shortening": 4, "torsion": 15}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import style
from .io import (VTK, read_control, control_float, output_dir, output_files, file_step,
                 read_lv_volume, find_solver_log, parse_solver_log)

TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1}
TORSION_SECTIONS = (0.10, 0.85)   # base section, apex section: fractions of the base-apex height
SECTION_HALFWIDTH = 0.04          # slab half-width, same units


def _unwrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _theta_pezzuto(P, P0, e3, axis_xy):
    """Rotation (deg) of each point about the long axis, Pezzuto's sign: sin(theta) = ((x ^ X).e3)/(r R).
    P, P0: (n,3) current/reference positions; e3 = +-z unit vector (apex -> base); axis through axis_xy."""
    x = P[:, :2] - axis_xy; X = P0[:, :2] - axis_xy
    cross_z = x[:, 0] * X[:, 1] - x[:, 1] * X[:, 0]          # (x ^ X) . z
    return np.degrees(np.arctan2(cross_z * e3[2], np.sum(x * X, axis=1)))


def compute(case_dir, out_override=None, solver_unit="m", apex="+z", band_cm=0.4, sections=TORSION_SECTIONS):
    ctrl = read_control(case_dir)
    out = output_dir(case_dir, ctrl, out_override)
    dt = control_float(ctrl, "-dt")
    if dt is None:
        raise ValueError(f"no -dt in {case_dir}/control.dat")
    s = TO_CM[solver_unit]
    sgn = 1.0 if apex == "+z" else -1.0

    surf = output_files(out, "surface")
    if not surf:
        raise FileNotFoundError(f"no surface00_*.vtk in {out}")
    ks = [k for k, _ in surf]
    t = np.array([max(file_step("surface", k), 0) * dt for k in ks])

    X, lam3, cells = [], [], None
    for _, p in surf:
        v = VTK(p)
        X.append(v.points * s)
        lam3.append(v.cell_data.get("lambda3"))
        cells = cells or v.cells
    X = np.array(X)
    z = sgn * X[0][:, 2]
    zb, za = z.min(), z.max(); H = za - zb
    zmid = 0.5 * (zb + za)
    U = np.linalg.norm(X - X[0], axis=2)
    dtk = np.diff(t)
    rate = np.r_[np.nan, np.where(dtk > 0, np.linalg.norm(np.diff(X, axis=0), axis=2).mean(1) / np.where(dtk > 0, dtk, 1), np.nan)]
    iapex = int(np.argmax(z))
    L = sgn * X[:, iapex, 2] - zb
    shortening = 100.0 * (L[0] - L) / L[0]

    res = dict(case=str(case_dir), t=t, k=np.array(ks), dt=dt,
               u_mean=U.mean(1), u_max=U.max(1), rate=rate, shortening=shortening)

    # thickening + torsion: shell3d if present, else mid-surface / lambda3
    s3 = dict(output_files(out, "shell3d"))
    band_nodes = np.abs(z - zmid) < band_cm
    frac = (z - zb) / H
    sec_base = np.abs(frac - sections[0]) < SECTION_HALFWIDTH
    sec_apex = np.abs(frac - sections[1]) < SECTION_HALFWIDTH
    e3 = np.array([0.0, 0.0, -sgn])                       # apex -> base
    axis_xy = X[0][:, :2].mean(0)
    if all(k in s3 for k in ks):
        n = X.shape[1]
        thick_r, thick_n, tors = [], [], []
        ref = None
        for k in ks:
            v = VTK(s3[k]); P = v.points * s; h = v.point_data["thickness"]
            A, B = P[:n], P[n:]
            if ref is None:
                endo_first = np.hypot(A[:, 0], A[:, 1]).mean() < np.hypot(B[:, 0], B[:, 1]).mean()
                ref = (A, B) if endo_first else (B, A)
                h0 = h[:n].copy()
            endo, epi = (A, B) if endo_first else (B, A)
            w = np.hypot(epi[:, 0], epi[:, 1]) - np.hypot(endo[:, 0], endo[:, 1])
            w0 = np.hypot(ref[1][:, 0], ref[1][:, 1]) - np.hypot(ref[0][:, 0], ref[0][:, 1])
            thick_r.append(100 * (w[band_nodes].mean() / w0[band_nodes].mean() - 1))
            thick_n.append(100 * (h[:n][band_nodes].mean() / h0[band_nodes].mean() - 1))
            th = _theta_pezzuto(epi, ref[1], e3, axis_xy)
            tors.append(th[sec_apex].mean() - th[sec_base].mean())
        res.update(thick_radial=np.array(thick_r), thick_normal=np.array(thick_n), torsion=np.array(tors), torsion_surface="epicardium")
    else:
        if lam3[0] is not None and cells:
            cz = np.array([z[c].mean() for c in cells]); cb = np.abs(cz - zmid) < band_cm
            res["thick_normal"] = np.array([100 * (l3[cb].mean() - 1) for l3 in lam3])
        tors = [(lambda th: th[sec_apex].mean() - th[sec_base].mean())(_theta_pezzuto(Xk, X[0], e3, axis_xy)) for Xk in X]
        res.update(torsion=np.array(tors), torsion_surface="mid-surface")

    vol = read_lv_volume(out)
    if vol:
        f = s ** 3                                     # length^3 -> cm^3 = ml
        res["V_mid"] = np.array([vol[k][0] * f if k in vol else np.nan for k in ks])
        res["V_endo"] = np.array([vol[k][1] * f if k in vol else np.nan for k in ks])

    log = find_solver_log(case_dir)
    res["not_converged_t"] = []
    if log is not None:
        res["not_converged_t"] = [st["step"] * dt for st in parse_solver_log(log) if st["converged"] is False]
    return res


def scalars(r):
    """Cycle summary of one case."""
    out = {}
    V = r.get("V_endo")
    if V is not None and np.isfinite(V).any():
        vmax, vmin = np.nanmax(V), np.nanmin(V)
        out.update(EDV=vmax, ESV=vmin, SV=vmax - vmin, EF=100 * (vmax - vmin) / vmax)
    for key, name in (("thick_radial", "thickening"), ("thick_normal", "thickening_normal")):
        if key in r:
            out[name] = np.nanmax(r[key])
    out["shortening"] = np.nanmax(r["shortening"]); out["elongation"] = -np.nanmin(r["shortening"])
    out["torsion"] = r["torsion"][np.nanargmax(np.abs(r["torsion"]))]
    out["u_max"] = np.nanmax(r["u_max"])
    out["not_converged_steps"] = len(r["not_converged_t"])
    return out


PANELS = [  # key, title, ref key
    ("V_endo", "endocardial volume (ml)", None),
    ("u_mean", "mean |u| (cm)", None),
    ("u_max", "max |u| (cm)", None),
    ("rate", "mean nodal rate |dx|/dt (cm/s)", None),
    ("thick", "wall thickening, mid-axis band (%)", "thickening"),
    ("shortening", "apex-base shortening (%)  (<0: elongation)", "shortening"),
    ("torsion", "torsion, apex vs base section (deg, Pezzuto def.)", "torsion"),
    ("V_mid", "mid-surface volume (ml)", None),
]


def cmd_lv(args):
    plt = style.setup(args.show)
    dirs = args.dir or ["."]
    labels = style.labels_for(dirs, args.label)
    ref = json.loads(Path(args.ref).read_text()) if args.ref else {}

    data = []
    for d, lab in zip(dirs, labels):
        try:
            data.append((lab, compute(d, args.out_dir, args.solver_unit, args.apex, args.band,
                                      tuple(float(v) for v in args.torsion_sections.split(",")))))
        except (FileNotFoundError, ValueError) as e:
            print(f"warning: {e}; skipping", file=sys.stderr)
    if not data:
        sys.exit("error: nothing to plot")

    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    single = len(data) == 1
    for i, (lab, r) in enumerate(data):
        c = style.case_color(i)
        for ax, (key, title, rk) in zip(axs.flat, PANELS):
            if single and i == 0:
                style.shade_steps(ax, r["not_converged_t"], r["dt"], label="not converged" if key == "V_endo" else None)
            if key == "thick":
                if "thick_radial" in r:
                    ax.plot(r["t"], r["thick_radial"], color=c, lw=1.8, label=f"{lab} radial" if not single else "radial (epi-endo)")
                if "thick_normal" in r:
                    ax.plot(r["t"], r["thick_normal"], color=c, lw=1.2, ls="--", label=f"{lab} normal" if not single else "normal (h/h0)")
            elif key in r:
                ax.plot(r["t"], r[key], color=c, lw=1.8, label=None if single else lab)
            ax.set_title(title, fontsize=9)
    for ax, (key, title, rk) in zip(axs.flat, PANELS):
        if rk and rk in ref:
            val = ref[rk]
            ax.axhline(val, color=style.INK2, lw=1, ls=":", label=f"{ref.get('label', 'ref')} peak {val:g}")
        style.finish(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7)
    if args.title:
        fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    out = Path(args.out or "lv_metrics.png")
    fig.savefig(out, dpi=130)
    print(f"[biofem] LV metrics plot saved -> {out}")

    cols = ["t", "V_endo", "V_mid", "u_mean", "u_max", "rate", "thick_radial", "thick_normal", "shortening", "torsion"]
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case"] + cols)
        for lab, r in data:
            for j in range(len(r["t"])):
                w.writerow([lab] + [f"{r[c][j]:.6g}" if c in r else "" for c in cols])
    print(f"[biofem] time series saved -> {csv_path}")

    keys = ["EF", "SV", "EDV", "ESV", "thickening", "thickening_normal", "shortening", "elongation", "torsion", "u_max", "not_converged_steps"]
    units = dict(EF="%", SV="ml", EDV="ml", ESV="ml", thickening="%", thickening_normal="%", shortening="%", elongation="%",
                 torsion="deg", u_max="cm", not_converged_steps="")
    rows = [(lab, scalars(r)) for lab, r in data]
    width = 2 + max(12, len(ref.get("label", "ref")), *(len(l) for l, _ in rows))
    print("\n" + f"{'':20s}" + "".join(f"{l:>{width}s}" for l, _ in rows) + (f"{ref.get('label', 'ref'):>{width}s}" if ref else ""))
    for k in keys:
        if not any(k in s for _, s in rows):
            continue
        line = f"{k + (' (' + units[k] + ')' if units[k] else ''):20s}"
        for _, s in rows:
            line += f"{s[k]:>{width}.3g}" if k in s else f"{'-':>{width}s}"
        if ref:
            line += f"{ref[k]:>{width}.3g}" if k in ref else f"{'':>{width}s}"
        print(line)
    for lab, r in data:
        if r.get("torsion_surface") == "mid-surface":
            print(f"note: {lab}: no shell3d output (-write_3d_shell 1); torsion from the mid-surface, no radial thickening")
    if args.show:
        plt.show()
