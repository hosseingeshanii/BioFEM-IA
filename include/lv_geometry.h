#ifndef LV_GEOMETRY_H
#define LV_GEOMETRY_H

#include "variables.h"

/* -----------------------------------------------------------------------
 * LVParams — all geometry and fiber parameters for the analytic LV mesh.
 *
 * Populated by LVParamsCreate(), which sets defaults then reads overrides
 * from the PETSc options database (command line / control.dat).
 * ----------------------------------------------------------------------- */
typedef struct {
  PetscReal  a;             /* polar semi-axis, apex-to-centre [cm]              */
  PetscReal  b;             /* equatorial semi-axis [cm]                          */
  PetscReal  f_cut;         /* axial fraction of spheroid retained (0,1]          */
  PetscInt   N_theta;       /* number of latitude rings (excl. apex point)        */
  PetscInt   N_phi;         /* number of nodes per ring                           */
  PetscInt   N_apex_extra;  /* extra rings inserted near apex to shrink hole      */
  PetscReal  alpha_endo;    /* Streeter helix angle at endocardium [degrees]      */
  PetscReal  alpha_epi;     /* Streeter helix angle at epicardium [degrees]       */
} LVParams;

/*
 * LVParamsCreate — fill *p with defaults, then override from PETSc options.
 *
 *   -lv_a          [4.5]    polar semi-axis
 *   -lv_b          [2.5]    equatorial semi-axis
 *   -lv_f_cut      [0.55]   height fraction  (0.5 = hemisphere)
 *   -lv_N_theta       [16]   latitude rings
 *   -lv_N_phi         [32]   nodes per ring
 *   -lv_N_apex_extra  [0]    extra rings near apex to shrink the apex hole
 *   -lv_alpha_endo [60.0]   endocardial helix angle [deg]  (Bayer: 40°)
 *   -lv_alpha_epi  [-60.0]  epicardial helix angle  [deg]  (Bayer: -50°)
 *
 * The mid-wall fiber angle is α_mid = (α_endo + α_epi)/2, constant over the
 * surface, consistent with Bayer et al. 2012 Eq.(2) evaluated at d=0.5.
 */
PetscErrorCode LVParamsCreate(LVParams *p);

/*
 * CreateLVMesh — build a triangulated truncated-prolate-spheroid LV surface,
 * populate ibm and fem, and assign Streeter helical fiber directions to
 * ibm->n_fib.  Internally calls Create() from io.c, so the global `dof`
 * must already be set before calling this function.
 *
 * After this call:
 *   fem->ibm           is set and fully allocated
 *   ibm->x/y/z_bp[*]  node coordinates
 *   ibm->nv1/2/3[*]   triangle connectivity (0-based)
 *   ibm->nv4/5/6[*]   patch-node indices for subdivision bending
 *   ibm->n_bnodes[0]   = N_phi  (one open boundary ring at the base)
 *   ibm->bnodes[*]     base-ring node indices
 *   ibm->n_fib[ec]     unit fiber direction per element
 */
PetscErrorCode CreateLVMesh(IBMNodes *ibm, FE *fem, const LVParams *p);

/*
 * WriteLVFiberVTK — write the surface mesh + per-element fiber vectors to a
 * legacy VTK file for inspection in ParaView (Glyph filter on fiber_direction).
 */
PetscErrorCode WriteLVFiberVTK(IBMNodes *ibm, const char *filepath);

#endif /* LV_GEOMETRY_H */
