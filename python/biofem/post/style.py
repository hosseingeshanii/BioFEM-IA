"""One plot style for biofem post commands."""
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAIL = "#b3261e"                                   # non-converged steps
# categorical order for cases (fixed, never cycled beyond its length)
CASES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def setup(show=False):
    """Select a backend and apply the style; returns pyplot."""
    import matplotlib
    if show:
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
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def case_color(i):
    if i >= len(CASES):
        raise ValueError(f"at most {len(CASES)} cases per plot")
    return CASES[i]


def shade_steps(ax, times, dt, label=None):
    """Light red band over each time in `times` (e.g. non-converged steps)."""
    for j, t in enumerate(times):
        ax.axvspan(t - dt / 2, t + dt / 2, color=FAIL, alpha=0.10, lw=0, label=label if j == 0 else None)


def finish(ax, xlabel="t (s)"):
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel(xlabel)


def labels_for(dirs, label_arg):
    from pathlib import Path
    labels = label_arg.split(",") if label_arg else [Path(d).resolve().name for d in dirs]
    if len(labels) != len(dirs):
        raise SystemExit("error: --label count must match --dir count")
    return labels
