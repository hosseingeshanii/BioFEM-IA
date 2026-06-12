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
/* io.c / bending.c entry points we depend on                         */
/* ------------------------------------------------------------------ */
extern PetscErrorCode Create(IBMNodes *ibm, FE *fem, PetscInt ibi);
extern PetscErrorCode IrrVer(IBMNodes *ibm);   /* classify irregular vertices   */
extern PetscErrorCode Patch (IBMNodes *ibm);   /* build subdivision patch table */

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

/* 0-based flat index of ring node (k, j):
 *   ring k=0 is the top (near-apex) open boundary ring.
 *   No apex node — both top and base are open boundaries so every
 *   interior vertex has valence 6 (regular subdivision surface).      */
#define RING_NODE(k, j, Np)  ((k)*(Np) + (j))

/* ------------------------------------------------------------------ */

PetscErrorCode LVParamsCreate(LVParams *p)
{
  p->a          = 4.5;
  p->b          = 2.5;
  p->f_cut      = 0.55;
  p->N_theta    = 16;
  p->N_phi      = 32;
  p->alpha_endo = 60.0;
  p->alpha_epi  = -60.0;

  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_a",          &p->a,          PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_b",          &p->b,          PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_f_cut",      &p->f_cut,      PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_theta",    &p->N_theta,    PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_phi",      &p->N_phi,      PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_endo", &p->alpha_endo, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_epi",  &p->alpha_epi,  PETSC_NULL);

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
  PetscReal alpha_endo_deg = p->alpha_endo;
  PetscReal alpha_epi_deg  = p->alpha_epi;

  PetscErrorCode ierr;

  /* ----------------------------------------------------------------
   * 1.  Mesh dimensions
   * ---------------------------------------------------------------- */
  /* theta_cut: polar angle where the base plane cuts the spheroid.
   *   z_base = a*cos(theta_cut),  theta_cut = acos(1 - 2*f_cut)
   * Ring k sits at theta = (k+1)*theta_step so ring 0 is one step below
   * the true apex.  Both top and base are open boundaries so every
   * interior vertex has valence 6 (all regular for Loop subdivision). */
  PetscReal theta_cut  = acos(1.0 - 2.0 * f_cut);
  PetscReal theta_step = theta_cut / (PetscReal)N_theta;
  /* PetscReal z_apex     = a; */   /* unused: apex cap removed */
  PetscReal z_base     = a * cos(theta_cut);

  PetscInt n_elmt_base  = (N_theta - 1) * 2 * N_phi;   /* quad strips only         */
  PetscInt n_v          = N_theta * N_phi;              /* rings only, no apex node */
  PetscInt n_elmt       = n_elmt_base;                  /* quad strips only         */
  /* NOTE: apex fan commented out — top boundary is open like the base.
   *   + N_phi;  simple apex fan */
  PetscInt n_edge       = 0;  /* base BC via EdgeDirectionalFix, no ghosts */
  PetscInt n_ghosts     = 0;
  /* Both apex ring AND base ring are boundary for IrrVer's ghost trick.
     Apex ring BCs are NOT imposed (EdgeDirectionalFix only uses edge_n=1 below). */
  PetscInt sum_n_bnodes = 2 * N_phi;

  ibm->n_v          = n_v;
  ibm->n_elmt       = n_elmt;
  ibm->n_elmt_base  = n_elmt_base;
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

  ierr = PetscMalloc(sum_n_bnodes * sizeof(PetscInt), &(ibm->bnodes)); CHKERRQ(ierr);

  for (PetscInt i = 0; i < n_elmt; i++) {
    ibm->El[0][i]         = E;
    ibm->El[1][i]         = mu;
    ibm->E_epsilon[0][i]  = 0.0;
    ibm->E_epsilon[1][i]  = 0.0;
  }

  /* EdgeDirectionalFix unconditionally reads n_bnodes[0..3]; zero-init 8 slots */
  PetscFree(ibm->n_bnodes);
  ierr = PetscCalloc1(8, &ibm->n_bnodes); CHKERRQ(ierr);
  /* n_bnodes[0]=N_phi tells IrrVer the apex ring is a boundary (bcount=3+3=6 → regular).
     No actual force BCs are applied to it — EdgeDirectionalFix only targets edge_n=1. */
  ibm->n_bnodes[0] = N_phi;
  ibm->n_bnodes[1] = N_phi;   /* base ring — fixed via EdgeDirectionalFix */

  /* ----------------------------------------------------------------
   * 3.  Node coordinates  —  RING_NODE(k,j,N_phi) = k*N_phi + j
   *     Ring k=0 is the first ring below the apex (θ = theta_step).
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 3: filling node coordinates\n");

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

  /* Apex node removed: top boundary is open. */
  /* PetscInt n_apex_node = N_theta * N_phi;
  ibm->x_bp[n_apex_node]  = 0.0;  ibm->x_bp0[n_apex_node] = 0.0;
  ibm->y_bp[n_apex_node]  = 0.0;  ibm->y_bp0[n_apex_node] = 0.0;
  ibm->z_bp[n_apex_node]  = z_apex; ibm->z_bp0[n_apex_node] = z_apex; */

  /* ----------------------------------------------------------------
   * 4.  Element connectivity  (0-based node indices)
   *
   *   Quad strips only — apex fan commented out.  Top boundary is open
   *   like the base so every interior vertex has valence 6 (regular).
   * ---------------------------------------------------------------- */
  PetscInt ec = 0;

  for (PetscInt k = 0; k < N_theta - 1; k++) {
    for (PetscInt j = 0; j < N_phi; j++) {
      PetscInt jn   = (j + 1) % N_phi;
      PetscInt n_bl = RING_NODE(k,     j,  N_phi);
      PetscInt n_br = RING_NODE(k,     jn, N_phi);
      PetscInt n_tl = RING_NODE(k + 1, j,  N_phi);
      PetscInt n_tr = RING_NODE(k + 1, jn, N_phi);

      ibm->nv1[ec] = n_bl;  ibm->nv2[ec] = n_tl;  ibm->nv3[ec] = n_br;  ec++;
      ibm->nv1[ec] = n_tl;  ibm->nv2[ec] = n_tr;  ibm->nv3[ec] = n_br;  ec++;
    }
  }

  /* Apex fan removed: top ring is open boundary.
  for (PetscInt j = 0; j < N_phi; j++) {
    PetscInt jn = (j + 1) % N_phi;
    ibm->nv1[ec] = RING_NODE(0, j,  N_phi);
    ibm->nv2[ec] = RING_NODE(0, jn, N_phi);
    ibm->nv3[ec] = n_apex_node;
    ec++;
  } */

  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 4: connectivity done (%d elements)\n", (int)ec);

  /* ----------------------------------------------------------------
   * 5.  Patch nodes  nv4/nv5/nv6
   *     Opposite node of each neighbouring triangle — 1000000 = none.
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
   * 6.  Boundary rings  —  top (ring 0) and base (ring N_theta-1)
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 6: boundary edge info\n");
  /* Both apex ring (edge_n=0) and base ring (edge_n=1) in bnodes[].
     IrrVer uses this to apply the ghost trick (bcount=count+3) for boundary nodes,
     keeping the apex ring classified as REGULAR despite the cap additions.
     No displacement BCs are applied to the apex ring — EdgeDirectionalFix targets edge_n=1 only. */
  for (PetscInt j = 0; j < N_phi; j++)
    ibm->bnodes[j] = RING_NODE(0, j, N_phi);              /* apex ring  (edge_n=0) */
  for (PetscInt j = 0; j < N_phi; j++)
    ibm->bnodes[N_phi + j] = RING_NODE(N_theta - 1, j, N_phi); /* base ring (edge_n=1) */

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
   *   α  = (α_endo + α_epi)/2   mid-wall value, constant (Bayer Eq.2, d=0.5)
   *
   *   f  = cos(α)·e_c + sin(α)·e_l                → ibm->n_fib[ec]
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 7: assigning fiber directions\n");

  /* Mid-wall helix angle: Bayer 2012 Eq.(2) evaluated at d=0.5 */
  PetscReal alpha = 0.5 * (alpha_endo_deg + alpha_epi_deg) * PETSC_PI / 180.0;

  /* Direction vector pointing from apex toward base along the z axis */
  struct Cmpnts e_down = {0.0, 0.0, -1.0};

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

    /* Fiber unit vector in the tangent plane — Bayer 2012 Eq.(7) */
    ibm->n_fib[ec].x = cos(alpha) * e_c.x + sin(alpha) * e_l.x;
    ibm->n_fib[ec].y = cos(alpha) * e_c.y + sin(alpha) * e_l.y;
    ibm->n_fib[ec].z = cos(alpha) * e_c.z + sin(alpha) * e_l.z;
  }

  PetscPrintf(PETSC_COMM_WORLD,
    "LV fibers assigned: α_endo=%.1f°  α_epi=%.1f°  α_mid=%.1f°  (Bayer 2012 Eq.2, d=0.5)\n",
    alpha_endo_deg, alpha_epi_deg, (alpha_endo_deg + alpha_epi_deg) * 0.5);

  /* ----------------------------------------------------------------
   * 8.  Subdivision surface topology
   *     IrrVer classifies each element as regular (ire=0) or irregular
   *     (ire=1) based on vertex valence, and fills irv/val.
   *     Patch builds the 16-slot Loop stencil table (ibm->patch[]).
   *     Both are needed by ElemUpdateGeomSubdivFromCoords_ in
   *     active_strain.c; Create() allocates the arrays but leaves
   *     them uninitialised.
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 8: subdivision surface topology (IrrVer + Patch)\n");
  ierr = IrrVer(ibm); CHKERRQ(ierr);
  ierr = Patch(ibm);  CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 8: done\n");

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
