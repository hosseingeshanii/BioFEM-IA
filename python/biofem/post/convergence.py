"""biofem post convergence: nonlinear solver convergence per step, from the solver log.

Per step: ||R|| at the start and end of the step, the reduction end/start against
-fem_snes_rtol, Newton iterations, and (with -log_c33 1) the local plane-stress (C33) solve.
Steps whose SNES did not converge are marked: last iterate accepted (-accept_reduction) or
rolled back. Writes a PNG and a CSV (one row per step and case).
"""
import csv
import sys
from pathlib import Path

from . import style
from .io import read_control, control_float, find_solver_log, parse_solver_log


def load(case_dir, log=None):
    path = find_solver_log(case_dir, log)
    if path is None or not path.exists():
        raise FileNotFoundError(f"no solver log (*.out) in {case_dir}; pass --log")
    steps = [s for s in parse_solver_log(path) if s["fnorm"]]
    ctrl = read_control(case_dir)
    return dict(log=path, steps=steps, rtol=control_float(ctrl, "-fem_snes_rtol"),
                max_it=control_float(ctrl, "-fem_snes_max_it"), dt=control_float(ctrl, "-dt", 1.0))


def summary(d):
    st = d["steps"]
    active = [s for s in st if s["fnorm"][0] > 0 and len(s["fnorm"]) > 1]
    bad = [s for s in st if s["converged"] is False]
    acc = [s["step"] for s in bad if s["outcome"] == "accepted"]
    rb = [s["step"] for s in bad if s["outcome"] == "rolled back"]
    red = [s["fnorm"][-1] / s["fnorm"][0] for s in active]
    return dict(steps=len(st), not_converged=[s["step"] for s in bad], accepted=acc, rolled_back=rb,
                stagnated=sum(s["stagnated"] for s in st),
                reduction_median=sorted(red)[len(red) // 2] if red else float("nan"),
                reduction_worst=max(red) if red else float("nan"),
                newton_max=max(len(s["fnorm"]) - 1 for s in st) if st else 0,
                ksp_total=sum(s["ksp_its"] for s in st))


def cmd_convergence(args):
    plt = style.setup(args.show)
    dirs = args.dir or ["."]
    if args.log and len(dirs) > 1:
        sys.exit("error: --log only works with a single --dir")
    labels = style.labels_for(dirs, args.label)

    data = []
    for d, lab in zip(dirs, labels):
        try:
            data.append((lab, load(d, args.log)))
        except FileNotFoundError as e:
            print(f"warning: {e}; skipping", file=sys.stderr)
    if not data:
        sys.exit("error: nothing to plot")

    has_c33 = any(s["c33"] for _, d in data for s in d["steps"])
    fig, axs = plt.subplots(1, 4 if has_c33 else 3, figsize=(16 if has_c33 else 12, 3.6))
    single = len(data) == 1
    for i, (lab, d) in enumerate(data):
        c = style.case_color(i)
        st = [s for s in d["steps"] if len(s["fnorm"]) > 1]     # skip trivial steps (0 Newton its, e.g. step 0)
        t = [s["step"] * d["dt"] for s in st]
        f0 = [s["fnorm"][0] for s in st]
        fe = [s["fnorm"][-1] for s in st]
        red = [b / a if a > 0 else float("nan") for a, b in zip(f0, fe)]
        nit = [len(s["fnorm"]) - 1 for s in st]
        bad = [k for k, s in enumerate(st) if s["converged"] is False]
        good = [k for k in range(len(st)) if k not in set(bad)]
        if single:
            for ax in axs:
                style.shade_steps(ax, [t[k] for k in bad], d["dt"], label="not converged" if ax is axs[0] else None)
        pre = "" if single else f"{lab}: "
        axs[0].semilogy(t, f0, "o", ms=2.5, color=style.INK2 if single else c, alpha=1 if single else 0.35,
                        label=f"{pre}start of step")
        axs[0].semilogy([t[k] for k in good], [fe[k] for k in good], "o", ms=3, color=c, label=f"{pre}end, converged")
        axs[0].semilogy([t[k] for k in bad], [fe[k] for k in bad], "x", ms=5, color=style.FAIL if single else c,
                        label=f"{pre}end, not converged")
        axs[1].semilogy([t[k] for k in good], [red[k] for k in good], "o", ms=3, color=c, label=f"{pre}converged")
        axs[1].semilogy([t[k] for k in bad], [red[k] for k in bad], "x", ms=5, color=style.FAIL if single else c,
                        label=f"{pre}not converged")
        axs[2].plot(t, nit, "o-", ms=2.5, lw=1, color=c, label=lab)
        rbk = [k for k in bad if st[k]["outcome"] == "rolled back"]
        if rbk:
            axs[0].semilogy([t[k] for k in rbk], [fe[k] for k in rbk], "s", ms=6, mfc="none", color=style.FAIL if single else c,
                            label=f"{pre}rolled back")
        if has_c33:
            c3 = [(s["step"] * d["dt"], s["c33"]) for s in d["steps"] if s["c33"] and s["step"] > 0]
            if c3:
                tc = [a for a, _ in c3]
                axs[3].semilogy(tc, [b["last_dC33_rel_max"] for _, b in c3], "-", lw=1.5, color=c,
                                label=f"{pre}max |last dC33|/C33")
                axs[3].semilogy(tc, [b["S33_rel_max"] for _, b in c3], "--", lw=1.2, color=c,
                                label=f"{pre}max|S33| / max|S_in-plane|")
                ncl = [a for a, b in c3 if b.get("clamped_elems", 0) > 0]
                if ncl:
                    axs[3].plot(ncl, [min(b["last_dC33_rel_max"] for _, b in c3)] * len(ncl), "|", ms=8, color=style.FAIL,
                                label=f"{pre}det(C) guard active")
        if d["rtol"]:
            axs[1].axhline(d["rtol"], color=c if not single else style.INK2, lw=1, ls="--")
            if single:
                axs[1].text(t[0], d["rtol"] * 1.3, f"-fem_snes_rtol {d['rtol']:g}", fontsize=7.5, color=style.INK2)
    axs[0].set_title(r"$\|R\|_2$ per step", fontsize=9)
    axs[1].set_title("reduction in the step (end / start)", fontsize=9)
    axs[2].set_title("Newton iterations per step", fontsize=9)
    if has_c33:
        sub = next(s["c33"].get("subitr") for _, d in data for s in d["steps"] if s["c33"])
        axs[3].set_title(f"local C33 (plane-stress) solve, {sub:g} sub-iterations", fontsize=9)
    for ax in axs:
        style.finish(ax)
        ax.legend(frameon=False, fontsize=7)
    if single:
        lab, d = data[0]
        sm = summary(d)
        fig.suptitle(args.title or f"{lab}: {sm['steps'] - len(sm['not_converged'])}/{sm['steps']} steps converged", fontsize=10)
    elif args.title:
        fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    out = Path(args.out or "convergence.png")
    fig.savefig(out, dpi=130)
    print(f"[biofem] convergence plot saved -> {out}")

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "step", "t", "fnorm_start", "fnorm_end", "reduction", "newton_its", "ksp_its", "converged", "reason",
                    "outcome", "stagnated", "C33_mean", "C33_max", "last_dC33_rel_max", "S33_rel_max", "clamped_elems"])
        for lab, d in data:
            for s in d["steps"]:
                a, b = s["fnorm"][0], s["fnorm"][-1]
                w.writerow([lab, s["step"], f"{s['step'] * d['dt']:.6g}", f"{a:.4e}", f"{b:.4e}",
                            f"{b / a:.3e}" if a > 0 else "", len(s["fnorm"]) - 1, s["ksp_its"], s["converged"], s["reason"],
                            s["outcome"] or "", s["stagnated"]]
                           + [s["c33"].get(k, "") for k in ("C33_mean", "C33_max", "last_dC33_rel_max", "S33_rel_max", "clamped_elems")])
    print(f"[biofem] per-step table saved -> {csv_path}")

    for lab, d in data:
        sm = summary(d)
        nc = sm["not_converged"]
        print(f"{lab}: log {d['log'].name}; {sm['steps'] - len(nc)}/{sm['steps']} steps converged"
              + (f" (not converged: steps {nc}; iterate accepted {len(sm['accepted'])}, rolled back {len(sm['rolled_back'])},"
                 f" stagnation stops {sm['stagnated']})" if nc else "")
              + f"; reduction median {sm['reduction_median']:.1e}, worst {sm['reduction_worst']:.1e};"
              f" max Newton {sm['newton_max']}; GMRES total {sm['ksp_total']}")

    if args.show:
        plt.show()
