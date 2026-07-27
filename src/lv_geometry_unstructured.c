/*
 * lv_geometry_unstructured.c
 *
 * Loads a pre-generated unstructured Delaunay triangulation of the LV
 * surface (see BioFEM-studies/lv_test/unstructured_mesh/lv_unstructured_mesh.py)
 * and assigns the same Streeter/Bayer fiber architecture and gamma taper
 * that CreateLVMesh (lv_geometry.c) uses for the structured ring mesh.
 *
 * Unlike the structured mesh, there is no explicit "apex cap": the point-
 * sampling + Delaunay approach in the Python generator maps the whole
 * truncated cap, including the apex, onto an ordinary flat disk, so the
 * apex is just one interior mesh vertex with normal valence, not a special-
 * cased fan region. n_elmt_base = n_elmt (no CST-fallback exclusion zone),
 * so every element goes through the regular Loop-subdivision IrrVer/Patch
 * path, which already handles arbitrary vertex valence.
 *
 * See include/lv_geometry.h for the full API.
 */

#include "variables.h"
#include "lv_geometry.h"
#include <math.h>
#include <stdio.h>

extern PetscInt   dof;
extern PetscReal  E, mu;

extern PetscErrorCode Create(IBMNodes *ibm, FE *fem, PetscInt ibi);
extern PetscErrorCode IrrVer(IBMNodes *ibm);
extern PetscErrorCode Patch (IBMNodes *ibm);

extern struct Cmpnts PLUS (struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts UNIT (struct Cmpnts v1);
extern PetscReal     DOT  (struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal     SIZE (struct Cmpnts v1);

/*
 * CreateLVMeshUnstructured — read the mesh file at -lv_unstructured_mesh_file,
 * populate ibm/fem exactly like CreateLVMesh does, and assign fiber
 * directions with the same Bayer 2012 Eq.(2)/(7) mid-wall formula and the
 * same near-apex gamma taper (see lv_geometry.c stage 7), just driven by
 * continuous angular distance from the apex instead of discrete ring index.
 *
 * Mesh file format (written by lv_unstructured_mesh.py):
 *   n_v
 *   x y z            (repeated n_v times)
 *   n_elmt
 *   i j k            (0-based, repeated n_elmt times)
 *   n_hull
 *   hull_node_idx    (repeated n_hull times)  -- true open mesh boundary (base rim)
 *   n_apex_pin
 *   apex_pin_idx     (repeated n_apex_pin times) -- 3-2-1 rigid-body pin nodes
 */
PetscErrorCode CreateLVMeshUnstructured(IBMNodes *ibm, FE *fem, const LVParams *p)
{
  PetscErrorCode ierr;
  char           filepath[512];
  PetscBool      has_path = PETSC_FALSE;
  FILE          *fd;

  PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-lv_unstructured_mesh_file",
                         filepath, sizeof(filepath), &has_path);
  if (!has_path) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
            "-lv_geom_unstructured 1 requires -lv_unstructured_mesh_file <path>");
  }

  fd = fopen(filepath, "r");
  if (!fd) {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
             "Cannot open -lv_unstructured_mesh_file '%s'", filepath);
  }

  PetscInt n_v, n_elmt, n_hull, n_apex_pin;
  if (fscanf(fd, "%d", &n_v) != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: n_v");

  /* Read into temporary buffers first -- Create() needs n_v AND n_elmt to
   * size its arrays, but the element count only appears after the node
   * block in the file, so we can't allocate ibm's own arrays until both
   * counts are known. */
  PetscReal *tx, *ty, *tz;
  ierr = PetscMalloc1(n_v, &tx); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_v, &ty); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_v, &tz); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n_v; i++) {
    double x, y, z;
    if (fscanf(fd, "%lf %lf %lf", &x, &y, &z) != 3)
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: node %d", (int)i);
    tx[i] = x; ty[i] = y; tz[i] = z;
  }

  if (fscanf(fd, "%d", &n_elmt) != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: n_elmt");
  PetscInt *te1, *te2, *te3;
  ierr = PetscMalloc1(n_elmt, &te1); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_elmt, &te2); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_elmt, &te3); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n_elmt; i++) {
    PetscInt a, b, c;
    if (fscanf(fd, "%d %d %d", &a, &b, &c) != 3)
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: element %d", (int)i);
    te1[i] = a; te2[i] = b; te3[i] = c;
  }

  ibm->n_v          = n_v;
  ibm->n_elmt       = n_elmt;
  ibm->n_elmt_base  = n_elmt;  /* every element goes through the regular subdivision path */
  ibm->n_edge       = 0;
  ibm->n_ghosts     = 0;
  ibm->ibi          = 0;

  PetscPrintf(PETSC_COMM_WORLD,
    "[lv-unstruct] n_v=%d  n_elmt=%d  (from %s)\n", (int)n_v, (int)n_elmt, filepath);

  PetscPrintf(PETSC_COMM_WORLD, "[lv-unstruct] stage 2: calling Create()\n");
  ierr = Create(ibm, fem, 0); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[lv-unstruct] Create() done\n");

  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->El[0][i]        = E;
    ibm->El[1][i]        = mu;
    ibm->E_epsilon[0][i] = 0.0;
    ibm->E_epsilon[1][i] = 0.0;
  }

  for (PetscInt i = 0; i < n_v; i++) {
    ibm->x_bp[i] = tx[i];  ibm->y_bp[i] = ty[i];  ibm->z_bp[i] = tz[i];
    ibm->x_bp0[i] = tx[i]; ibm->y_bp0[i] = ty[i]; ibm->z_bp0[i] = tz[i];
  }
  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->nv1[i] = te1[i]; ibm->nv2[i] = te2[i]; ibm->nv3[i] = te3[i];
  }
  ierr = PetscFree(tx); CHKERRQ(ierr); ierr = PetscFree(ty); CHKERRQ(ierr); ierr = PetscFree(tz); CHKERRQ(ierr);
  ierr = PetscFree(te1); CHKERRQ(ierr); ierr = PetscFree(te2); CHKERRQ(ierr); ierr = PetscFree(te3); CHKERRQ(ierr);

  if (fscanf(fd, "%d", &n_hull) != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: n_hull");
  PetscInt *hull_nodes;
  ierr = PetscMalloc1(PetscMax(1, n_hull), &hull_nodes); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n_hull; i++) fscanf(fd, "%d", &hull_nodes[i]);

  if (fscanf(fd, "%d", &n_apex_pin) != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "bad mesh file: n_apex_pin");
  if (n_apex_pin > 3) n_apex_pin = 3;
  ibm->n_apex_pin = n_apex_pin;
  for (PetscInt i = 0; i < n_apex_pin; i++) fscanf(fd, "%d", &ibm->apex_pin_nodes[i]);

  fclose(fd);

  /* ----------------------------------------------------------------
   * bnodes[]: base rim only (the true open mesh boundary) -- used by
   * IrrVer's boundary "ghost" adjustment (bcount = count+3), same purpose
   * as ring 0 / base ring in the structured mesh. The apex-pin nodes are
   * NOT put here: they're ordinary interior vertices we've chosen to
   * constrain, not topological boundary, and IrrVer would misclassify
   * their true valence if they were included (see ibm->n_apex_pin instead,
   * consumed directly in main.c FormFunctionFEM). */
  ibm->sum_n_bnodes = n_hull;
  ierr = PetscFree(ibm->bnodes); CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(1, n_hull), &(ibm->bnodes)); CHKERRQ(ierr);
  for (PetscInt i = 0; i < n_hull; i++) ibm->bnodes[i] = hull_nodes[i];
  ierr = PetscFree(hull_nodes); CHKERRQ(ierr);

  ierr = PetscFree(ibm->n_bnodes); CHKERRQ(ierr);
  ierr = PetscCalloc1(8, &ibm->n_bnodes); CHKERRQ(ierr);
  ibm->n_bnodes[0] = n_hull;   /* group 0: base rim (matches EdgeDirectionalFix(1,...) convention
                                 * loosely -- not currently fixed by main.c either way) */

  /* ----------------------------------------------------------------
   * Patch nodes nv4/nv5/nv6 -- identical algorithm to CreateLVMesh stage 5.
   * ---------------------------------------------------------------- */
  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->nv4[i] = 1000000; ibm->nv5[i] = 1000000; ibm->nv6[i] = 1000000;
  }
  for (PetscInt i = 0; i < n_elmt; i++) {
    PetscInt n1e = ibm->nv1[i], n2e = ibm->nv2[i], n3e = ibm->nv3[i];
    for (PetscInt j = 0; j < n_elmt; j++) {
      if (i == j) continue;
      PetscInt  n1pe = ibm->nv1[j], n2pe = ibm->nv2[j], n3pe = ibm->nv3[j];
      PetscInt  mn = 0; PetscReal cn = 0.0;
      if (n1e == n1pe || n1e == n2pe || n1e == n3pe) { mn++; cn += 3.5; }
      if (n2e == n1pe || n2e == n2pe || n2e == n3pe) { mn++; cn += 2.5; }
      if (n3e == n1pe || n3e == n2pe || n3e == n3pe) { mn++; cn += 1.5; }
      if (mn == 2) {
        PetscInt npe;
        if      (n1pe != n1e && n1pe != n2e && n1pe != n3e) npe = n1pe;
        else if (n2pe != n1e && n2pe != n2e && n2pe != n3e) npe = n2pe;
        else                                                  npe = n3pe;
        if      (cn == 4.0) ibm->nv4[i] = npe;
        else if (cn == 5.0) ibm->nv5[i] = npe;
        else                ibm->nv6[i] = npe;
      }
    }
  }
  PetscPrintf(PETSC_COMM_WORLD, "[lv-unstruct] patch nodes done\n");

  /* ----------------------------------------------------------------
   * Fiber directions -- identical Bayer 2012 Eq.(2)/(7) mid-wall formula to
   * CreateLVMesh stage 7, with the same near-apex gamma taper, just driven
   * by continuous theta = acos(z/a) instead of discrete ring index. The
   * spheroid semi-axes (a, b) are baked into the mesh file's coordinates by
   * the Python generator, but we still need them here for the normal /
   * taper formulas -- read the same -lv_a/-lv_b options CreateLVMesh uses.
   * ---------------------------------------------------------------- */
  PetscReal alpha = 0.5 * (p->alpha_endo + p->alpha_epi) * PETSC_PI / 180.0;
  struct Cmpnts e_down = {0.0, 0.0, -1.0};

  /* theta_pin: angular distance from the apex out to the pinned apex-pin
   * nodes (average over however many there are) -- the taper's zero point,
   * analogous to theta_ring0 in the structured mesh. theta_taper_width
   * reuses -lv_N_taper_rings, scaled by the mesh's average nearest-neighbor
   * angular spacing near the apex, since there's no discrete ring spacing
   * here to multiply directly. */
  PetscReal theta_pin = 0.0;
  for (PetscInt i = 0; i < ibm->n_apex_pin; i++) {
    PetscInt nb = ibm->apex_pin_nodes[i];
    PetscReal cz = ibm->z_bp0[nb] / p->a;
    if (cz > 1.0) cz = 1.0; if (cz < -1.0) cz = -1.0;
    theta_pin += acos(cz);
  }
  if (ibm->n_apex_pin > 0) theta_pin /= (PetscReal)ibm->n_apex_pin;

  /* Average element "radius" in theta near the apex, used as one taper-ring
   * equivalent, so -lv_N_taper_rings means roughly the same physical taper
   * width as it did for the structured mesh at similar point density. */
  PetscReal avg_theta_spacing = p->f_cut > 0
    ? (PetscReal)acos(1.0 - 2.0*p->f_cut) / PetscSqrtReal((PetscReal)n_elmt)
    : 0.1;
  PetscReal theta_taper_width = (PetscReal)p->N_taper_rings * avg_theta_spacing * 4.0;

  for (PetscInt ec = 0; ec < n_elmt; ec++) {
    PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];

    struct Cmpnts c;
    c.x = (ibm->x_bp[n1e] + ibm->x_bp[n2e] + ibm->x_bp[n3e]) / 3.0;
    c.y = (ibm->y_bp[n1e] + ibm->y_bp[n2e] + ibm->y_bp[n3e]) / 3.0;
    c.z = (ibm->z_bp[n1e] + ibm->z_bp[n2e] + ibm->z_bp[n3e]) / 3.0;

    struct Cmpnts grad_F = {c.x / (p->b * p->b), c.y / (p->b * p->b), c.z / (p->a * p->a)};
    struct Cmpnts e_n = UNIT(grad_F);

    PetscReal d = DOT(e_down, e_n);
    struct Cmpnts e_l_raw = {e_down.x - d*e_n.x, e_down.y - d*e_n.y, e_down.z - d*e_n.z};
    struct Cmpnts e_l;
    if (SIZE(e_l_raw) > 1.0e-10) {
      e_l = UNIT(e_l_raw);
    } else {
      struct Cmpnts e_x = {1.0, 0.0, 0.0};
      PetscReal d2 = DOT(e_x, e_n);
      struct Cmpnts raw = {e_x.x - d2*e_n.x, e_x.y - d2*e_n.y, e_x.z - d2*e_n.z};
      e_l = UNIT(raw);
    }
    struct Cmpnts e_c = CROSS(e_n, e_l);

    PetscReal cos_theta_c = c.z / p->a;
    if (cos_theta_c > 1.0) cos_theta_c = 1.0;
    if (cos_theta_c < -1.0) cos_theta_c = -1.0;
    PetscReal theta_c = acos(cos_theta_c);
    PetscReal taper = (theta_taper_width > 0.0)
      ? (theta_c - theta_pin) / theta_taper_width
      : 1.0;
    if (taper > 1.0) taper = 1.0;
    if (taper < 0.0) taper = 0.0;

    ibm->n_fib[ec].x = cos(alpha) * e_c.x + sin(alpha) * e_l.x;
    ibm->n_fib[ec].y = cos(alpha) * e_c.y + sin(alpha) * e_l.y;
    ibm->n_fib[ec].z = cos(alpha) * e_c.z + sin(alpha) * e_l.z;
    ibm->gamma_scale[ec] = taper;
  }

  PetscPrintf(PETSC_COMM_WORLD,
    "[lv-unstruct] fibers assigned: alpha_endo=%.1f alpha_epi=%.1f  "
    "apex_pin=%d nodes  theta_pin=%.2f deg  taper_width=%.2f deg\n",
    p->alpha_endo, p->alpha_epi, (int)ibm->n_apex_pin,
    theta_pin * 180.0 / PETSC_PI, theta_taper_width * 180.0 / PETSC_PI);

  PetscPrintf(PETSC_COMM_WORLD, "[lv-unstruct] subdivision surface topology (IrrVer + Patch)\n");
  ierr = IrrVer(ibm); CHKERRQ(ierr);
  ierr = Patch(ibm);  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[lv-unstruct] done\n");

  return 0;
}
