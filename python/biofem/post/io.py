"""Readers for solver output: legacy-ASCII VTK fields, output file series, control.dat."""
import re
from pathlib import Path

import numpy as np


def _vtk_read_field(path, field_name):
    """Parse a legacy-ASCII VTK file for a SCALARS or VECTORS block named
    `field_name`, in either the POINT_DATA or CELL_DATA section.

    Returns a list of floats (scalar field) or a list of (x, y, z) tuples
    (vector field), or None if the field isn't present in this file.
    Robust to whatever line-wrapping the writer used, since it just reads
    `count` whitespace-separated tokens after the header line rather than
    assuming one value/line.
    """
    with open(path) as f:
        tokens_iter = _WhitespaceTokenizer(f)
        count = None
        while True:
            line = tokens_iter.next_line()
            if line is None:
                return None
            stripped = line.strip()
            if stripped.startswith("POINT_DATA") or stripped.startswith("CELL_DATA"):
                count = int(stripped.split()[1])
                continue
            if stripped.startswith("SCALARS "):
                parts = stripped.split()
                name = parts[1]
                if name != field_name:
                    continue
                if count is None:
                    return None
                nxt = tokens_iter.next_line()
                if nxt is None or not nxt.strip().startswith("LOOKUP_TABLE"):
                    tokens_iter.pushback_line(nxt)
                vals = tokens_iter.next_tokens(count)
                return [float(v) for v in vals]
            if stripped.startswith("VECTORS "):
                parts = stripped.split()
                name = parts[1]
                if name != field_name:
                    continue
                if count is None:
                    return None
                vals = tokens_iter.next_tokens(count * 3)
                fvals = [float(v) for v in vals]
                return list(zip(fvals[0::3], fvals[1::3], fvals[2::3]))


class _WhitespaceTokenizer:
    """Line-oriented reader that can also pull a fixed number of
    whitespace-separated tokens spanning multiple lines."""

    def __init__(self, fh):
        self._fh = fh
        self._pending_line = None

    def next_line(self):
        if self._pending_line is not None:
            l, self._pending_line = self._pending_line, None
            return l
        return self._fh.readline() or None

    def pushback_line(self, line):
        self._pending_line = line

    def next_tokens(self, n):
        out = []
        while len(out) < n:
            line = self.next_line()
            if line is None:
                break
            out.extend(line.split())
        return out[:n]


def _field_reduce(values, component="mag"):
    """Reduce a parsed field (list of floats, or list of (x,y,z) tuples) to
    a single average value for one timestep."""
    if not values:
        return None
    if isinstance(values[0], tuple):
        if component == "mag":
            reduced = [ (x*x + y*y + z*z) ** 0.5 for (x, y, z) in values ]
        else:
            idx = {"x": 0, "y": 1, "z": 2}[component]
            reduced = [v[idx] for v in values]
    else:
        reduced = values
    return sum(reduced) / len(reduced)


def _get_dt(case_dir):
    control = Path(case_dir) / "control.dat"
    if not control.exists():
        return None
    for line in control.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-dt "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _vtk_series_for_prefix(case_dir, out_dir, prefix):
    pattern = f"{prefix}00_*.vtk"
    files = sorted(Path(case_dir, out_dir).glob(pattern))
    series = []
    for fp in files:
        m = re.search(r"_(\d+)\.vtk$", fp.name)
        if not m:
            continue
        series.append((int(m.group(1)), fp))
    return series


# ---------------------------------------------------------------------------
# Structured readers (numpy) used by biofem post convergence / lv
# ---------------------------------------------------------------------------
def file_step(prefix, k):
    """Solver step whose converged state output file number k holds.

    surface*/shell3d* files are written as Output(ti+1): file 0 is the reference
    (before step 0) and file k >= 1 holds the state after step ti = k-1.
    surfaceghost* files are written with ti itself. Returns -1 for the reference.
    """
    if prefix.startswith("surfaceghost"):
        return k
    return k - 1


def file_time(prefix, k, dt):
    """Time t = ti*dt of the state in output file k (reference: t = 0)."""
    return max(file_step(prefix, k), 0) * dt


def read_control(case_dir):
    """control.dat -> {option: value string or None for a bare flag}; last occurrence wins."""
    opts = {}
    path = Path(case_dir) / "control.dat"
    if not path.exists():
        return opts
    for line in path.read_text(errors="replace").splitlines():
        tok = line.split("#", 1)[0].split()
        i = 0
        while i < len(tok):
            if tok[i].startswith("-") and not _is_number(tok[i]):
                val = tok[i + 1] if i + 1 < len(tok) and not (tok[i + 1].startswith("-") and not _is_number(tok[i + 1])) else None
                opts[tok[i]] = val
                i += 2 if val is not None else 1
            else:
                i += 1
    return opts


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def control_float(ctrl, key, default=None):
    try:
        return float(ctrl[key])
    except (KeyError, TypeError, ValueError):
        return default


def output_dir(case_dir, ctrl=None, override=None):
    """Output directory of a case: override > -out_dir in control.dat > 'output'."""
    ctrl = read_control(case_dir) if ctrl is None else ctrl
    return Path(case_dir) / (override or ctrl.get("-out_dir") or "output")


def output_files(out_path, prefix):
    """[(k, path)] for <prefix>00_<k>.vtk in out_path, sorted by k."""
    return [(k, p) for k, p in _vtk_series_for_prefix(out_path, ".", prefix)]


class VTK:
    """Legacy-ASCII VTK file: points (n, 3), cells (list of index lists),
    point_data / cell_data {name: (n,) or (n, 3) array}."""

    def __init__(self, path):
        tok = Path(path).read_text(errors="replace").split()
        self.points = None
        self.cells = []
        self.point_data, self.cell_data = {}, {}
        section, count, i = None, 0, 0
        while i < len(tok):
            t = tok[i]
            if t == "POINTS":
                n = int(tok[i + 1]); i += 3
                self.points = np.array(tok[i:i + 3 * n], float).reshape(n, 3); i += 3 * n
            elif t == "CELLS":
                n = int(tok[i + 1]); i += 3
                for _ in range(n):
                    m = int(tok[i]); self.cells.append([int(x) for x in tok[i + 1:i + 1 + m]]); i += 1 + m
            elif t == "CELL_TYPES":
                i += 2 + int(tok[i + 1])
            elif t in ("POINT_DATA", "CELL_DATA"):
                section = self.point_data if t == "POINT_DATA" else self.cell_data
                count = int(tok[i + 1]); i += 2
            elif t == "SCALARS" and section is not None:
                name = tok[i + 1]; i += 3
                if i + 1 < len(tok) and _is_number(tok[i]) and tok[i + 1] == "LOOKUP_TABLE":
                    i += 1                                   # optional numComp
                if tok[i] == "LOOKUP_TABLE":
                    i += 2
                section[name] = np.array(tok[i:i + count], float); i += count
            elif t == "VECTORS" and section is not None:
                name = tok[i + 1]; i += 3
                section[name] = np.array(tok[i:i + 3 * count], float).reshape(count, 3); i += 3 * count
            else:
                i += 1


def read_lv_volume(out_path):
    """lv_volume00.dat -> {file number k: (V_midsurface, V_endo)} in solver length units^3."""
    path = Path(out_path) / "lv_volume00.dat"
    if not path.exists():
        return {}
    vol = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 3:
            vol[int(p[0])] = (float(p[1]), float(p[2]))
    return vol


def find_solver_log(case_dir, override=None):
    """The solver's stdout: --log if given, else the newest *.out in the case directory."""
    if override:
        return Path(override)
    logs = sorted(Path(case_dir).glob("*.out"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


_TIME_RE = re.compile(r"body:\d+ Time\((\d+)\)=(\S+)")
_SNES_RE = re.compile(r"^\s*(\d+) SNES Function norm (\S+)")
_KSP_RE  = re.compile(r"Linear fem_ solve .* iterations (\d+)")
_NL_RE   = re.compile(r"Nonlinear fem_ solve (converged|did not converge) due to (\S+) iterations (\d+)")


def parse_solver_log(path):
    """Per solver step: {'step', 't', 'fnorm' (list, one per Newton iteration), 'ksp_its' (total),
    'converged' (bool or None if no SNES summary line), 'reason'}.
    Needs -fem_snes_monitor, -fem_ksp_converged_reason, -fem_snes_converged_reason in control.dat."""
    steps, cur = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _TIME_RE.search(line)
            if m:
                cur = dict(step=int(m.group(1)), t=float(m.group(2)), fnorm=[], ksp_its=0, converged=None, reason="")
                steps.append(cur)
                continue
            if cur is None:
                continue
            m = _SNES_RE.match(line)
            if m:
                cur["fnorm"].append(float(m.group(2))); continue
            m = _KSP_RE.search(line)
            if m:
                cur["ksp_its"] += int(m.group(1)); continue
            m = _NL_RE.search(line)
            if m:
                cur["converged"] = m.group(1) == "converged"; cur["reason"] = m.group(2)
    return steps
