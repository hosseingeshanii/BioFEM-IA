"""Live SNES residual monitor used by biofem run --monitor residual."""
import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# residual monitor helpers  (used by cmd_run when --monitor residual is set)
# ---------------------------------------------------------------------------

# Embedded script run as a subprocess to own the tornado/WebAgg event loop.
# Main biofem process writes plot data to a JSON file; this script reads it.
_WEBAGG_DISPLAY_SCRIPT = r"""
import sys, json, time, threading
port      = int(sys.argv[1])
data_file = sys.argv[2]
import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams["webagg.port"] = port
mpl.rcParams["webagg.open_in_browser"] = False
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))

def _update():
    # Wait for tornado to start before first draw attempt.
    time.sleep(3)
    while True:
        try:
            with open(data_file) as f:
                d = json.load(f)
            rn = d.get("rn", [])
            ax.cla()
            if rn:
                ax.semilogy(range(len(rn)), rn, "b.-", linewidth=1.2, markersize=6)
            ax.set_xlabel("Newton iteration (cumulative)")
            ax.set_ylabel("||R||")
            ax.set_title("SNES convergence — all Newton iterations")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            # Schedule the canvas flush in tornado's own thread (thread-safe).
            try:
                from matplotlib.backends.backend_webagg import WebAggApplication
                iol = getattr(WebAggApplication, "ioloop", None)
                if iol is not None:
                    iol.add_callback(fig.canvas.draw)
                else:
                    fig.canvas.draw()
            except Exception:
                fig.canvas.draw()
        except Exception:
            pass
        time.sleep(2)

threading.Thread(target=_update, daemon=True).start()
plt.show()   # blocking — tornado owns this thread; subprocess is killed by parent on exit
"""

# Embedded script that opens a native matplotlib window (Tk/Qt) instead of
# WebAgg. Only usable when a real display is reachable (local X, or ssh -Y/-X
# forwarding) — VS Code's integrated terminal has no DISPLAY, so this is
# skipped there and WebAgg is used instead.
_GUI_DISPLAY_SCRIPT = r"""
import sys, json, time
data_file = sys.argv[1]
import matplotlib
fig = ax = None
errors = []
for backend in ("TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.show(block=False)
        break
    except Exception as e:
        errors.append(f"{backend}: {e}")
        fig = ax = None
        continue
if fig is None:
    print("no usable GUI backend / display —\n  " + "\n  ".join(errors), file=sys.stderr)
    sys.exit(1)   # caller falls back to WebAgg

while True:
    try:
        with open(data_file) as f:
            d = json.load(f)
        rn = d.get("rn", [])
        ax.cla()
        if rn:
            ax.semilogy(range(len(rn)), rn, "b.-", linewidth=1.2, markersize=6)
        ax.set_xlabel("Newton iteration (cumulative)")
        ax.set_ylabel("||R||")
        ax.set_title("SNES convergence — all Newton iterations")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw_idle()
    except Exception:
        pass
    try:
        plt.pause(2.0)   # pumps the GUI event loop while waiting
    except Exception:
        break   # window was closed by the user
"""

def _parse_residual_log(log_file):
    timesteps, snes_iters, rnorms = [], [], []
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    try:
                        timesteps.append(int(parts[0]))
                        snes_iters.append(int(parts[1]))
                        rnorms.append(float(parts[2]))
                    except ValueError:
                        pass
    except OSError:
        pass
    return timesteps, snes_iters, rnorms


def _draw_snes_plot(ax, rnorms):
    """Single-panel: every SNES Newton iteration's ||R||, across all time steps."""
    ax.cla()
    if rnorms:
        ax.semilogy(range(len(rnorms)), rnorms, "b.-", linewidth=1.2, markersize=6)
    ax.set_xlabel("Newton iteration (cumulative)")
    ax.set_ylabel("||R||")
    ax.set_title("SNES convergence — all Newton iterations")
    ax.grid(True, which="both", alpha=0.3)


def _run_with_monitor(shell_cmd, run_dir, port, out_dir="output"):
    """Run solver as Popen and live-plot SNES residuals.

    WebAgg runs in a completely separate subprocess so its blocking tornado
    event loop never touches the main process.  Main process: submits job,
    polls log, prints status, handles Ctrl+C — no matplotlib involved at all.
    Final PNG saved via standalone Agg figure.
    """
    import sys as _sys
    import json as _json
    import tempfile as _tmp
    import time as _time
    run_dir  = Path(run_dir)
    log_file = run_dir / out_dir / "residual.log"
    png_file = run_dir / out_dir / "residual_monitor.png"

    # ── launch a live-plot display subprocess ────────────────────────────────
    # Prefer a native GUI window (local X11, or ssh -X/-Y forwarding) since
    # that's usable outside a browser. Only try it when DISPLAY is actually
    # set — VS Code's integrated terminal has no DISPLAY, so this is skipped
    # there and we fall straight to WebAgg (a local server + browser tab).
    data_file    = Path(_tmp.mktemp(suffix="_biofem_plot.json"))
    display_proc = None
    use_webagg   = False
    use_gui      = False
    try:
        import matplotlib  # check availability only; do NOT import pyplot here
        data_file.write_text("{}")

        if os.environ.get("DISPLAY"):
            try:
                display_proc = subprocess.Popen(
                    [_sys.executable, "-c", _GUI_DISPLAY_SCRIPT, str(data_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                _time.sleep(1.5)   # give the GUI backend time to open the window
                if display_proc.poll() is not None:
                    err = display_proc.stderr.read().decode(errors="replace").strip()
                    raise RuntimeError(err or "GUI display process exited immediately")
                use_gui = True
                print("[biofem] live plot → native window")
            except Exception as e:
                print(f"[biofem] native window unavailable ({e}); falling back to WebAgg")
                if display_proc is not None:
                    display_proc.terminate()
                    display_proc = None

        if not use_gui:
            display_proc = subprocess.Popen(
                [_sys.executable, "-c", _WEBAGG_DISPLAY_SCRIPT, str(port), str(data_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,   # isolate from terminal Ctrl+C signal
            )
            _time.sleep(2.5)              # give tornado time to bind the port
            if display_proc.poll() is not None:
                err = display_proc.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(err or "display process exited immediately")
            use_webagg = True
            print(f"[biofem] live plot → http://localhost:{port}")
            print(f"[biofem] (VSCode: Ports tab → forward {port}, then open in browser)")
    except Exception as e:
        if display_proc is not None:
            display_proc.terminate()
            display_proc = None
        print(f"[biofem] live plot unavailable ({e})")
        print(f"[biofem] running without live plot — PNG saved to {png_file}")

    # ── start solver / sbatch ─────────────────────────────────────────────────
    if shell_cmd.startswith("sbatch "):
        proc = subprocess.Popen(shell_cmd.split(), cwd=run_dir)
        print(f"[biofem] job submitted, monitoring {log_file} ...")
    else:
        proc = subprocess.Popen(["bash", "--login", "-c", shell_cmd], cwd=run_dir)
        print(f"[biofem] solver started (PID {proc.pid})")

    # ── monitor loop ──────────────────────────────────────────────────────────
    last_size = -1
    try:
        while True:
            done = proc.poll() is not None

            if log_file.exists():
                sz = log_file.stat().st_size
                if sz != last_size:
                    last_size = sz
                    ts, si, rn = _parse_residual_log(log_file)
                    if rn:
                        print(f"\r[biofem] step={ts[-1]}  newton={si[-1]}  rnorm={rn[-1]:.3e}",
                              end="", flush=True)
                        if use_gui or use_webagg:
                            try:
                                data_file.write_text(_json.dumps({"rn": rn}))
                            except Exception:
                                pass

            _time.sleep(2.0)   # Ctrl+C interrupts cleanly here

            if done:
                break
    except KeyboardInterrupt:
        print("\n[biofem] interrupted — terminating solver ...")
        proc.terminate()
        proc.wait()

    # ── shut down display subprocess ──────────────────────────────────────────
    if display_proc is not None:
        try:
            display_proc.terminate()
            display_proc.wait(timeout=3)
        except Exception:
            pass
    try:
        data_file.unlink(missing_ok=True)
    except Exception:
        pass

    # ── final PNG via standalone Agg figure ───────────────────────────────────
    ts, si, rn = _parse_residual_log(log_file)
    if rn:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
            from matplotlib.figure import Figure as _Fig
            fig2 = _Fig(figsize=(10, 5))
            _FCA(fig2)
            ax2 = fig2.add_subplot(1, 1, 1)
            _draw_snes_plot(ax2, rn)
            fig2.tight_layout()
            fig2.savefig(png_file, dpi=100)
            print(f"\n[biofem] plot saved → {png_file}")
        except Exception as e:
            print(f"\n[biofem] PNG save failed: {e}")

    if proc.returncode not in (0, -15):  # -15 = SIGTERM (our own interrupt)
        raise subprocess.CalledProcessError(proc.returncode, shell_cmd)
