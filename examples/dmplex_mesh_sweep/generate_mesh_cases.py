#!/usr/bin/env python3
"""Generate DMPlex MPI mesh-sweep case directories for 11 mesh sizes."""
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


# ================= USER PARAMETERS =================
Lx, Ly = 1.2, 0.9
CASES = [
    (16, 12),
    (32, 24),
    (48, 36),
    (64, 48),
    (96, 72),
    (128, 96),
    (160, 120),
    (256, 192),
    (384, 288),
    (512, 384),
    (640, 480),
]
DEFAULT_NTASKS = 1
tol = 1e-10
# ===================================================


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TEMPLATE_DIR = REPO_ROOT / "examples" / "dmplex_test"
EXECUTABLE_NAME = "testt-dmplex"
EXECUTABLE = REPO_ROOT / "build" / EXECUTABLE_NAME


# ---------- basic mesh ----------
def rect_points(Lx, Ly, nx, ny):
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([X.ravel(), Y.ravel()])


def delaunay_tris(pts):
    return Delaunay(pts).simplices.astype(np.int64)


# ---------- boundary logic ----------
def is_on_x0(p, tol):
    return abs(p[0] - 0.0) <= tol


def is_on_xL(p, Lx, tol):
    return abs(p[0] - Lx) <= tol


def is_on_y0(p, tol):
    return abs(p[1] - 0.0) <= tol


def is_on_yL(p, Ly, tol):
    return abs(p[1] - Ly) <= tol


def is_boundary_edge(pts, i, j, Lx, Ly, tol):
    pi, pj = pts[i], pts[j]
    if is_on_x0(pi, tol) and is_on_x0(pj, tol):
        return True
    if is_on_xL(pi, Lx, tol) and is_on_xL(pj, Lx, tol):
        return True
    if is_on_y0(pi, tol) and is_on_y0(pj, tol):
        return True
    if is_on_yL(pi, Ly, tol) and is_on_yL(pj, Ly, tol):
        return True
    return False


# ---------- conforming edge split ----------
def tri_has_edge(t, u, v):
    return (u in t) and (v in t)


def tri_third_vertex(t, u, v):
    for x in t:
        if x != u and x != v:
            return x
    raise RuntimeError("triangle does not have a third vertex?")


def orient2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def split_edge_conforming(pts, tri, u, v, midpoint_node_map):
    if u == v:
        return pts, tri

    key = (min(u, v), max(u, v))

    if key in midpoint_node_map:
        m = midpoint_node_map[key]
    else:
        mp = 0.5 * (pts[u] + pts[v])
        m = pts.shape[0]
        pts = np.vstack([pts, mp[None, :]])
        midpoint_node_map[key] = m

    new_tris = []
    for t in tri:
        if tri_has_edge(t, u, v):
            r = tri_third_vertex(t, u, v)

            a, b, c = t
            orig_sign = orient2(pts[a], pts[b], pts[c])

            t1 = (u, m, r)
            t2 = (m, v, r)

            if orig_sign != 0:
                s1 = orient2(pts[t1[0]], pts[t1[1]], pts[t1[2]])
                if s1 * orig_sign < 0:
                    t1 = (u, r, m)
                s2 = orient2(pts[t2[0]], pts[t2[1]], pts[t2[2]])
                if s2 * orig_sign < 0:
                    t2 = (m, r, v)

            new_tris.append(t1)
            new_tris.append(t2)
        else:
            new_tris.append(tuple(t))

    tri = np.array(new_tris, dtype=np.int64)
    return pts, tri


def fix_two_boundary_edge_triangles_conforming(pts, tri, Lx, Ly, tol=1e-10, max_passes=1000):
    pts = np.asarray(pts, float)
    tri = np.asarray(tri, np.int64)

    midpoint_node_map = {}

    for _pass in range(max_passes):
        to_split = None

        for t in tri:
            a, b, c = t
            edges = [(a, b), (b, c), (c, a)]
            bedges = [(i, j) for (i, j) in edges if is_boundary_edge(pts, i, j, Lx, Ly, tol)]

            if len(bedges) == 2:
                (u1, v1), (u2, v2) = bedges
                shared = None
                for x in (u1, v1):
                    if x == u2 or x == v2:
                        shared = x
                        break
                if shared is None:
                    continue

                others = [x for x in (a, b, c) if x != shared]
                if len(others) != 2:
                    continue
                p, q = others
                to_split = (p, q)
                break

        if to_split is None:
            return pts, tri

        p, q = to_split
        pts, tri = split_edge_conforming(pts, tri, p, q, midpoint_node_map)

    raise RuntimeError("Reached max_passes without resolving all 2-boundary-edge triangles.")


# ---------- boundary lists ----------
def boundary_node_lists(pts, Lx, Ly, tol=1e-10):
    pts = np.asarray(pts)
    ids = np.arange(1, pts.shape[0] + 1, dtype=int)

    on_bottom = np.abs(pts[:, 1] - 0.0) <= tol
    on_right = np.abs(pts[:, 0] - Lx) <= tol
    on_top = np.abs(pts[:, 1] - Ly) <= tol
    on_left = np.abs(pts[:, 0] - 0.0) <= tol

    e0_idx = np.where(on_bottom)[0]
    e0 = ids[e0_idx[np.argsort(pts[e0_idx, 0])]].tolist()

    e1_idx = np.where(on_right)[0]
    e1 = ids[e1_idx[np.argsort(pts[e1_idx, 1])]].tolist()

    e2_idx = np.where(on_top)[0]
    e2 = ids[e2_idx[np.argsort(-pts[e2_idx, 0])]].tolist()

    e3_idx = np.where(on_left)[0]
    e3 = ids[e3_idx[np.argsort(-pts[e3_idx, 1])]].tolist()

    return [e0, e1, e2, e3]


# ---------- writers ----------
def write_nlist(path, pts):
    with open(path, "w") as f:
        f.write(f"{pts.shape[0]}\n")
        for i, (x, y) in enumerate(pts, start=1):
            f.write(f"{i}\t0\t0\t{x:.6f} {y:.6f} {0.0:.6f}\n")


def write_elist(path, tri):
    with open(path, "w") as f:
        f.write(f"{tri.shape[0]}\n")
        for e, (a, b, c) in enumerate(tri, start=1):
            n1, n2, n3 = int(a) + 1, int(b) + 1, int(c) + 1
            f.write(f"1\t1\t0\t0\t0\t0\t0\t0\t4\t0\t{e}\t{n1}\t{n2}\t{n3}\t{n3}\n")


def write_blist(path, edges):
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


def write_vtk_legacy(path, pts, tri):
    pts3 = np.column_stack([pts, np.zeros(len(pts))])

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("surface mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {pts3.shape[0]} float\n")
        for x, y, z in pts3:
            f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"\nCELLS {tri.shape[0]} {tri.shape[0] * 4}\n")
        for i, j, k in tri:
            f.write(f"3 {int(i)} {int(j)} {int(k)}\n")

        f.write(f"\nCELL_TYPES {tri.shape[0]}\n")
        for _ in range(tri.shape[0]):
            f.write("5\n")


def slurm_resources(nx, ny):
    nodes = nx * ny
    if nodes <= 48 * 36:
        return "00:30:00", "4G"
    if nodes <= 128 * 96:
        return "01:00:00", "8G"
    if nodes <= 256 * 192:
        return "02:00:00", "16G"
    if nodes <= 384 * 288:
        return "04:00:00", "32G"
    if nodes <= 512 * 384:
        return "06:00:00", "64G"
    return "08:00:00", "128G"


def write_jobfile(path, nx, ny, nnodes, nelems, ntasks):
    name = f"dmplex_mesh_nx{nx}_ny{ny}_np{ntasks}"
    output = f"out_dmplex_nx{nx}_ny{ny}_np{ntasks}"
    time_limit, memory = slurm_resources(nx, ny)
    path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --time={time_limit}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --output={output}
#SBATCH --account=132714077318

set -euo pipefail

module purge
ml GCC/13.2.0 OpenMPI/4.1.6
ml PETSc/3.22.5

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "BENCH_BACKEND=dmplex_mpi"
echo "BENCH_MESH_NX={nx}"
echo "BENCH_MESH_NY={ny}"
echo "BENCH_NODES={nnodes}"
echo "BENCH_ELEMENTS={nelems}"
echo "BENCH_NTASKS=${{SLURM_NTASKS:-{ntasks}}}"
echo "BENCH_JOB_ID=${{SLURM_JOB_ID:-local}}"
/usr/bin/time -f "BENCH_WALL_SECONDS=%e" \\
  mpirun -np "${{SLURM_NTASKS:-{ntasks}}}" -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi' ./{EXECUTABLE_NAME}
"""
    )


def build_case(nx, ny):
    case_dir = SCRIPT_DIR / f"mesh_Lx{Lx}_Ly{Ly}_nx{nx}_ny{ny}"
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pts0 = rect_points(Lx, Ly, nx, ny)
    tri0 = delaunay_tris(pts0)
    pts, tri = fix_two_boundary_edge_triangles_conforming(pts0, tri0, Lx, Ly, tol=tol)
    edges = boundary_node_lists(pts, Lx, Ly, tol=tol)

    write_nlist(input_dir / "nlist00", pts)
    write_elist(input_dir / "elist00", tri)
    write_blist(input_dir / "blist00", edges)
    write_vtk_legacy(input_dir / f"surface00_nx{nx}_ny{ny}.vtk", pts, tri)

    shutil.copy2(TEMPLATE_DIR / "control.dat", case_dir / "control.dat")
    executable_link = case_dir / EXECUTABLE_NAME
    if executable_link.exists() or executable_link.is_symlink():
        executable_link.unlink()
    executable_link.symlink_to(f"../../../build/{EXECUTABLE_NAME}")
    write_jobfile(case_dir / "jobfile.slurm", nx, ny, pts.shape[0], tri.shape[0], DEFAULT_NTASKS)

    return case_dir, pts.shape[0], tri.shape[0]


def main():
    if not EXECUTABLE.exists():
        print(f"Warning: missing executable now, but creating links anyway: {EXECUTABLE}")

    for nx, ny in CASES:
        case_dir, nnodes, nelems = build_case(nx, ny)
        print(f"{case_dir.relative_to(REPO_ROOT)}: nodes={nnodes} elems={nelems}")


if __name__ == "__main__":
    main()
