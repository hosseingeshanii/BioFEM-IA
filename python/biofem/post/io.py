"""Readers for solver output: legacy-ASCII VTK fields, output file series, control.dat."""
import re
from pathlib import Path


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
