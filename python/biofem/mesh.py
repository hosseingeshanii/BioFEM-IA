"""Rectangular Delaunay plate meshes (nlist/elist/blist) for biofem sweep."""
from pathlib import Path

# Lazy-guard numpy/scipy so build/run work without them installed.
try:
    import numpy as np
    from scipy.spatial import Delaunay as _Delaunay
    HAS_MESH_DEPS = True
except ImportError:
    HAS_MESH_DEPS = False


# ---------------------------------------------------------------------------
# Mesh generation (rectangular Delaunay with boundary-edge fix)
# ---------------------------------------------------------------------------
def _build_mesh(lx: float, ly: float, nx: int, ny: int):
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    tri  = _Delaunay(pts).simplices.astype(np.int64)
    return _fix_two_boundary_edge_tris(pts, tri, lx, ly)


def _on_boundary_edge(pts, i, j, lx, ly, tol=1e-10) -> bool:
    pi, pj = pts[i], pts[j]
    return any([
        abs(pi[0])      <= tol and abs(pj[0])      <= tol,
        abs(pi[0] - lx) <= tol and abs(pj[0] - lx) <= tol,
        abs(pi[1])      <= tol and abs(pj[1])      <= tol,
        abs(pi[1] - ly) <= tol and abs(pj[1] - ly) <= tol,
    ])


def _orient2(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _split_edge(pts, tri, u, v, midmap: dict):
    key = (min(u, v), max(u, v))
    if key not in midmap:
        midmap[key] = pts.shape[0]
        pts = np.vstack([pts, 0.5 * (pts[u] + pts[v])[None, :]])
    m = midmap[key]

    new_tris = []
    for t in tri:
        if u in t and v in t:
            r       = next(x for x in t if x != u and x != v)
            a, b, c = t
            sgn     = _orient2(pts[a], pts[b], pts[c])
            t1, t2  = (u, m, r), (m, v, r)
            if sgn != 0:
                if _orient2(pts[t1[0]], pts[t1[1]], pts[t1[2]]) * sgn < 0:
                    t1 = (u, r, m)
                if _orient2(pts[t2[0]], pts[t2[1]], pts[t2[2]]) * sgn < 0:
                    t2 = (m, r, v)
            new_tris += [t1, t2]
        else:
            new_tris.append(tuple(t))
    return pts, np.array(new_tris, dtype=np.int64)


def _fix_two_boundary_edge_tris(pts, tri, lx, ly, tol=1e-10, max_passes=1000):
    midmap: dict = {}
    for _ in range(max_passes):
        to_split = None
        for t in tri:
            a, b, c = t
            bedges = [(i, j) for i, j in [(a, b), (b, c), (c, a)]
                      if _on_boundary_edge(pts, i, j, lx, ly, tol)]
            if len(bedges) == 2:
                (u1, v1), (u2, v2) = bedges
                shared = next((x for x in (u1, v1) if x in (u2, v2)), None)
                if shared is None:
                    continue
                others = [x for x in (a, b, c) if x != shared]
                if len(others) == 2:
                    to_split = tuple(others)
                    break
        if to_split is None:
            return pts, tri
        pts, tri = _split_edge(pts, tri, *to_split, midmap)
    raise RuntimeError("Could not resolve two-boundary-edge triangles.")


def _boundary_node_lists(pts, lx: float, ly: float, tol=1e-10) -> list:
    ids = np.arange(1, pts.shape[0] + 1, dtype=int)

    def sorted_ids(mask, col, reverse=False):
        idx = np.where(mask)[0]
        order = np.argsort(pts[idx, col])
        if reverse:
            order = order[::-1]
        return ids[idx[order]].tolist()

    return [
        sorted_ids(np.abs(pts[:, 1])        <= tol, 0),          # bottom
        sorted_ids(np.abs(pts[:, 0] - lx)   <= tol, 1),          # right
        sorted_ids(np.abs(pts[:, 1] - ly)   <= tol, 0, True),    # top
        sorted_ids(np.abs(pts[:, 0])         <= tol, 1, True),    # left
    ]


def _write_nlist(path: Path, pts):
    with open(path, "w") as f:
        f.write(f"{pts.shape[0]}\n")
        for i, (x, y) in enumerate(pts, start=1):
            f.write(f"{i}\t0\t0\t{x:.6f} {y:.6f} {0.0:.6f}\n")


def _write_elist(path: Path, tri):
    with open(path, "w") as f:
        f.write(f"{tri.shape[0]}\n")
        for e, (a, b, c) in enumerate(tri, start=1):
            f.write(f"1\t1\t0\t0\t0\t0\t0\t0\t4\t0\t{e}\t{a+1}\t{b+1}\t{c+1}\t{c+1}\n")


def _write_blist(path: Path, edges: list):
    bset = set()
    for e in edges:
        bset.update(e)
    with open(path, "w") as f:
        f.write("4\n")
        f.write(f"{len(bset)}\n")
        for e in edges:
            f.write(f"{len(e)}\n")
        for e in edges:
            f.write(" ".join(map(str, e)) + "\n")
