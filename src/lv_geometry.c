/*
 * lv_geometry.c
 *
 * Builds a triangulated truncated-prolate-spheroid left-ventricle surface,
 * populates the IBMNodes / FE data structures, and assigns Streeter helical
 * fiber directions to ibm->n_fib.
 *
 * Fiber model reference
 * ---------------------
 *   Streeter DD et al., Circ Res 24:339-347, 1969.
 *   Bayer JD et al., Ann Biomed Eng 40(10):2243-2254, 2012.
 *
 * See include/lv_geometry.h for the full API.
 */

#include "variables.h"
#include "lv_geometry.h"
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Globals mirrored from io.c / main.c                                 */
/* ------------------------------------------------------------------ */
extern PetscInt   dof;
extern PetscInt   ConstitutiveLawNonLinear, n_Fung_Coeffs, n_lin_model_coeffs;
extern PetscInt   ressmooth, inverse, Adam, par_jac;
extern PetscReal  char_length_x, char_length_y, char_length_z;

/* ------------------------------------------------------------------ */
/* io.c entry points we depend on                                      */
/* ------------------------------------------------------------------ */
extern PetscErrorCode Create(IBMNodes *ibm, FE *fem, PetscInt ibi);

/* Material parameters needed to initialize El/E_epsilon (set via PetscOptions) */
extern PetscReal E, mu;

/* ------------------------------------------------------------------ */
/* Vector math helpers from math.c                                     */
/* ------------------------------------------------------------------ */
extern struct Cmpnts PLUS (struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts UNIT (struct Cmpnts v1);
extern PetscReal     DOT  (struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal     SIZE (struct Cmpnts v1);
extern struct Cmpnts AMULT(PetscReal alpha, struct Cmpnts v1);

/* ------------------------------------------------------------------ */
/* Local helpers                                                        */
/* ------------------------------------------------------------------ */

/* 0-based index of node (k, j) in the ring layout:
 *   node 0          : apex
 *   nodes 1..N_phi  : ring k=0 (first ring below apex)
 *   nodes 1+k*Np .. : ring k                                          */
#define RING_NODE(k, j, Np)  (1 + (k)*(Np) + (j))

/* ------------------------------------------------------------------ */

PetscErrorCode LVParamsCreate(LVParams *p)
{
  p->a          = 4.5;
  p->b          = 2.5;
  p->f_cut      = 0.55;
  p->N_theta    = 16;
  p->N_phi      = 32;
  p->alpha_apex = 60.0;
  p->alpha_base = -60.0;

  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_a",          &p->a,          PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_b",          &p->b,          PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_f_cut",      &p->f_cut,      PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_theta",    &p->N_theta,    PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_phi",      &p->N_phi,      PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_apex", &p->alpha_apex, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_base", &p->alpha_base, PETSC_NULL);

  return 0;
}

/* ------------------------------------------------------------------ */

PetscErrorCode CreateLVMesh(IBMNodes *ibm, FE *fem, const LVParams *p)
{
  PetscReal a             = p->a;
  PetscReal b             = p->b;
  PetscReal f_cut         = p->f_cut;
  PetscInt  N_theta       = p->N_theta;
  PetscInt  N_phi         = p->N_phi;
  PetscReal alpha_apex_deg = p->alpha_apex;
  PetscReal alpha_base_deg = p->alpha_base;

  PetscErrorCode ierr;

  /* ----------------------------------------------------------------
   * 1.  Mesh dimensions
   * ---------------------------------------------------------------- */
  /* θ_cut: polar angle from apex at which the base plane cuts the spheroid.
   * Derived from f_cut (fraction of total axial height 2a kept):
   *   z_base = a - 2a·f_cut = a(1-2f_cut)
   *   θ_cut  = arccos(1 - 2·f_cut)                                   */
  PetscReal theta_cut  = acos(1.0 - 2.0 * f_cut);
  PetscReal theta_step = theta_cut / (PetscReal)N_theta;
  PetscReal z_apex     = a;
  PetscReal z_base     = a * cos(theta_cut);

  PetscInt n_v          = 1 + N_theta * N_phi;
  PetscInt n_elmt       = N_phi * (2 * N_theta - 1);  /* apex fan + quad strips */
  PetscInt n_edge       = 1;                           /* one open boundary loop */
  PetscInt n_ghosts     = 0;
  PetscInt sum_n_bnodes = N_phi;                       /* base ring */

  ibm->n_v          = n_v;
  ibm->n_elmt       = n_elmt;
  ibm->n_edge       = n_edge;
  ibm->n_ghosts     = n_ghosts;
  ibm->sum_n_bnodes = sum_n_bnodes;
  ibm->ibi          = 0;

  PetscPrintf(PETSC_COMM_WORLD,
    "LV mesh: a=%.4f  b=%.4f  f_cut=%.2f  theta_cut=%.2f deg  "
    "N_theta=%d  N_phi=%d  n_v=%d  n_elmt=%d\n",
    a, b, f_cut, theta_cut * 180.0 / PETSC_PI,
    (int)N_theta, (int)N_phi, (int)n_v, (int)n_elmt);

  /* ----------------------------------------------------------------
   * 2.  Allocate all standard IBMNodes / FE arrays
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 2: calling Create()\n");
  ierr = Create(ibm, fem, 0); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[lv] Create() done\n");

  /* Create() allocates ibm->n_bnodes[n_edge] but NOT ibm->bnodes.
   * That allocation lives in Input() — do it here instead.            */
  ierr = PetscMalloc(sum_n_bnodes * sizeof(PetscInt), &(ibm->bnodes)); CHKERRQ(ierr);

  /* Create() uses PetscMalloc for El/E_epsilon, leaving them uninitialised.
   * InitMaterial() is only called when ConstitutiveLawNonLinear=1.  For the
   * linear case StressLinear reads ibm->El[], so we must seed them here.  */
  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->El[0][i]         = E;
    ibm->El[1][i]         = mu;
    ibm->E_epsilon[0][i]  = 0.0;
    ibm->E_epsilon[1][i]  = 0.0;
  }

  /* EdgeDirectionalFix(3, ...) in FormFunctionFEM unconditionally walks edges
   * 0-3 of ibm->n_bnodes[].  Create() only PetscMalloc's n_edge=1 entries,
   * leaving indices 1-3 as uninitialised garbage.  Re-allocate with enough
   * zero-initialised slots so the out-of-range reads are safe no-ops.      */
  PetscFree(ibm->n_bnodes);
  ierr = PetscCalloc1(8, &ibm->n_bnodes); CHKERRQ(ierr); /* zero-init */
  ibm->n_bnodes[0] = N_phi;                               /* base ring */

  /* ----------------------------------------------------------------
   * 3.  Node coordinates
   *     Node 0 : apex (0, 0, a);  RING_NODE(k,j,N_phi) : ring k, col j
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 3: filling node coordinates\n");
  ibm->x_bp[0] = 0.0;  ibm->y_bp[0] = 0.0;  ibm->z_bp[0] = a;
  ibm->x_bp0[0] = 0.0; ibm->y_bp0[0] = 0.0; ibm->z_bp0[0] = a;

  for (PetscInt k = 0; k < N_theta; k++) {
    PetscReal theta = (k + 1) * theta_step;
    PetscReal sth   = sin(theta);
    PetscReal cth   = cos(theta);
    for (PetscInt j = 0; j < N_phi; j++) {
      PetscReal phi = 2.0 * PETSC_PI * (PetscReal)j / (PetscReal)N_phi;
      PetscInt  nc  = RING_NODE(k, j, N_phi);
      ibm->x_bp[nc]  = b * sth * cos(phi);
      ibm->y_bp[nc]  = b * sth * sin(phi);
      ibm->z_bp[nc]  = a * cth;
      ibm->x_bp0[nc] = ibm->x_bp[nc];
      ibm->y_bp0[nc] = ibm->y_bp[nc];
      ibm->z_bp0[nc] = ibm->z_bp[nc];
    }
  }

  /* ----------------------------------------------------------------
   * 4.  Element connectivity  (0-based node indices)
   *
   *   Apex fan  (N_phi triangles):
   *     apex -- ring0[j] -- ring0[j+1]
   *
   *   Quad strip between ring k and ring k+1 (2·N_phi triangles):
   *     split each latitude quad into two triangles with a shared
   *     diagonal from the lower-left to upper-right corner.
   * ---------------------------------------------------------------- */
  PetscInt ec = 0;

  /* Apex fan */
  for (PetscInt j = 0; j < N_phi; j++, ec++) {
    ibm->nv1[ec] = 0;
    ibm->nv2[ec] = RING_NODE(0,  j,            N_phi);
    ibm->nv3[ec] = RING_NODE(0, (j + 1) % N_phi, N_phi);
  }

  /* Quad strips: band between ring k (closer to apex) and ring k+1 */
  for (PetscInt k = 0; k < N_theta - 1; k++) {
    for (PetscInt j = 0; j < N_phi; j++) {
      PetscInt jn   = (j + 1) % N_phi;
      PetscInt n_bl = RING_NODE(k,     j,  N_phi);  /* bottom-left  */
      PetscInt n_br = RING_NODE(k,     jn, N_phi);  /* bottom-right */
      PetscInt n_tl = RING_NODE(k + 1, j,  N_phi);  /* top-left     */
      PetscInt n_tr = RING_NODE(k + 1, jn, N_phi);  /* top-right    */

      /* Triangle 1: bl – tl – br */
      ibm->nv1[ec] = n_bl;  ibm->nv2[ec] = n_tl;  ibm->nv3[ec] = n_br;  ec++;
      /* Triangle 2: tl – tr – br */
      ibm->nv1[ec] = n_tl;  ibm->nv2[ec] = n_tr;  ibm->nv3[ec] = n_br;  ec++;
    }
  }

  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 4: connectivity done (%d elements)\n", (int)ec);

  /* ----------------------------------------------------------------
   * 5.  Patch nodes  nv4/nv5/nv6
   *     Each element stores the "opposite" node of each of its three
   *     neighbours — used by the subdivision-surface bending stencil.
   *     Algorithm reproduced from io.c: Input().
   *     "1000000" flags a boundary element with no neighbour on that edge.
   * ---------------------------------------------------------------- */
  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->nv4[i] = 1000000;
    ibm->nv5[i] = 1000000;
    ibm->nv6[i] = 1000000;
  }

  for (PetscInt i = 0; i < n_elmt; i++) {
    PetscInt n1e = ibm->nv1[i], n2e = ibm->nv2[i], n3e = ibm->nv3[i];
    for (PetscInt j = 0; j < n_elmt; j++) {
      if (i == j) continue;
      PetscInt  n1pe = ibm->nv1[j], n2pe = ibm->nv2[j], n3pe = ibm->nv3[j];
      PetscInt  mn   = 0;
      PetscReal cn   = 0.0;
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

  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 5: patch nodes done\n");

  /* ----------------------------------------------------------------
   * 6.  Boundary edge info  (the open base ring)
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 6: boundary edge info\n");
  for (PetscInt j = 0; j < N_phi; j++) {
    ibm->bnodes[j] = RING_NODE(N_theta - 1, j, N_phi);
  }

  /* ----------------------------------------------------------------
   * 7.  Fiber directions  —  Streeter helical rule
   *
   *   At each element centroid c = (cx, cy, cz):
   *
   *   e_n = UNIT( cx/b², cy/b², cz/a² )          outward spheroid normal
   *   e_l = UNIT( ẑ₋  - (ẑ₋·e_n) e_n )           meridional apex→base
   *           where ẑ₋ = (0,0,-1)
   *   e_c = e_n × e_l                              circumferential
   *
   *   z* = (a - cz)/(a - z_base)  ∈ [0,1]
   *   α  = α_apex(1-z*) + α_base·z*               linear Streeter rule
   *
   *   f  = cos(α)·e_c + sin(α)·e_l                → ibm->n_fib[ec]
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 7: assigning fiber directions\n");

  PetscReal alpha_apex = alpha_apex_deg * PETSC_PI / 180.0;
  PetscReal alpha_base = alpha_base_deg * PETSC_PI / 180.0;

  /* Direction vector pointing from apex toward base along the z axis */
  struct Cmpnts e_down = {0.0, 0.0, -1.0};

  PetscReal dz = z_apex - z_base;  /* always > 0 */

  for (ec = 0; ec < n_elmt; ec++) {
    PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];

    /* Element centroid */
    struct Cmpnts c;
    c.x = (ibm->x_bp[n1e] + ibm->x_bp[n2e] + ibm->x_bp[n3e]) / 3.0;
    c.y = (ibm->y_bp[n1e] + ibm->y_bp[n2e] + ibm->y_bp[n3e]) / 3.0;
    c.z = (ibm->z_bp[n1e] + ibm->z_bp[n2e] + ibm->z_bp[n3e]) / 3.0;

    /* Outward surface normal: gradient of the spheroid level-set F=x²/b²+y²/b²+z²/a²-1 */
    struct Cmpnts grad_F;
    grad_F.x = c.x / (b * b);
    grad_F.y = c.y / (b * b);
    grad_F.z = c.z / (a * a);
    struct Cmpnts e_n = UNIT(grad_F);

    /* Meridional direction (apex→base): project e_down onto the tangent plane */
    PetscReal d     = DOT(e_down, e_n);
    struct Cmpnts e_l_raw;
    e_l_raw.x = e_down.x - d * e_n.x;
    e_l_raw.y = e_down.y - d * e_n.y;
    e_l_raw.z = e_down.z - d * e_n.z;

    struct Cmpnts e_l;
    if (SIZE(e_l_raw) > 1.0e-10) {
      e_l = UNIT(e_l_raw);
    } else {
      /* Degenerate at or very near the apex: fall back to a stable tangent.
       * Project (1,0,0) onto the tangent plane.                           */
      struct Cmpnts e_x = {1.0, 0.0, 0.0};
      PetscReal d2 = DOT(e_x, e_n);
      e_l_raw.x = e_x.x - d2 * e_n.x;
      e_l_raw.y = e_x.y - d2 * e_n.y;
      e_l_raw.z = e_x.z - d2 * e_n.z;
      e_l = UNIT(e_l_raw);
    }

    /* Circumferential direction (right-hand: outward-normal × meridional) */
    struct Cmpnts e_c = CROSS(e_n, e_l);

    /* Normalised longitudinal coordinate z* ∈ [0,1] */
    PetscReal zstar = (z_apex - c.z) / dz;
    if (zstar < 0.0) zstar = 0.0;
    if (zstar > 1.0) zstar = 1.0;

    /* Helix angle: linear interpolation from apex to base */
    PetscReal alpha = (1.0 - zstar) * alpha_apex + zstar * alpha_base;

    /* Fiber unit vector in the tangent plane */
    ibm->n_fib[ec].x = cos(alpha) * e_c.x + sin(alpha) * e_l.x;
    ibm->n_fib[ec].y = cos(alpha) * e_c.y + sin(alpha) * e_l.y;
    ibm->n_fib[ec].z = cos(alpha) * e_c.z + sin(alpha) * e_l.z;
  }

  PetscPrintf(PETSC_COMM_WORLD,
    "LV fibers assigned: α_apex=%.1f°  α_base=%.1f°  (Streeter 1969 / Bayer 2012)\n",
    alpha_apex_deg, alpha_base_deg);

  return 0;
}

/* ------------------------------------------------------------------ */

PetscErrorCode WriteLVFiberVTK(IBMNodes *ibm, const char *filepath)
{
  FILE *f = fopen(filepath, "w");
  if (!f) {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN,
             "Cannot open file: %s", filepath);
  }

  fprintf(f, "# vtk DataFile Version 2.0\n");
  fprintf(f, "LV surface with fiber directions\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n");

  /* Nodes */
  fprintf(f, "POINTS %d float\n", ibm->n_v);
  for (PetscInt i = 0; i < ibm->n_v; i++) {
    fprintf(f, "%f %f %f\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
  }

  /* Triangles */
  fprintf(f, "CELLS %d %d\n", ibm->n_elmt, 4 * ibm->n_elmt);
  for (PetscInt i = 0; i < ibm->n_elmt; i++) {
    fprintf(f, "3 %d %d %d\n", ibm->nv1[i], ibm->nv2[i], ibm->nv3[i]);
  }

  fprintf(f, "CELL_TYPES %d\n", ibm->n_elmt);
  for (PetscInt i = 0; i < ibm->n_elmt; i++) fprintf(f, "5\n");

  /* Cell-centred fiber direction vector */
  fprintf(f, "CELL_DATA %d\n", ibm->n_elmt);
  fprintf(f, "VECTORS fiber_direction float\n");
  for (PetscInt i = 0; i < ibm->n_elmt; i++) {
    fprintf(f, "%f %f %f\n",
            ibm->n_fib[i].x, ibm->n_fib[i].y, ibm->n_fib[i].z);
  }

  fclose(f);
  PetscPrintf(PETSC_COMM_WORLD, "LV fiber VTK written to: %s\n", filepath);
  return 0;
}
