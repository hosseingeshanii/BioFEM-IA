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
  PetscInt   N_cap_final;   /* target coarse-ring width for the single apex-cap
                               reduction jump (see CreateLVMesh cap coarsening);
                               must evenly divide N_phi, nearest valid divisor
                               used otherwise                                    */
  PetscReal  alpha_apex;    /* helix angle at the apex (latitude endpoint) [degrees] */
  PetscReal  alpha_base;    /* helix angle at the base (latitude endpoint) [degrees] */
  PetscInt   N_taper_rings; /* rings beyond ring 0 over which active fiber
                               magnitude ramps 0 -> 1 (see stage 7 comment
                               in CreateLVMesh); 0 = hard on/off at ring 0   */
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
 *   -lv_N_cap_final   [4]    target width of the single coarse ring the whole
 *                            cap collapses ring 0 down to before fanning to
 *                            the apex point (see CreateLVMesh cap coarsening
 *                            comment) — must evenly divide N_phi; the
 *                            nearest valid divisor is used otherwise. Smaller
 *                            means fewer total cap elements but a bigger
 *                            single jump (worse element shape right at the
 *                            ring-0 attachment, which is not pinned)
 *   -lv_alpha_apex [60.0]   helix angle at the apex latitude [deg]
 *   -lv_alpha_base [-60.0]  helix angle at the base latitude [deg]
 *   (previously named -lv_alpha_endo/-lv_alpha_epi and read as a constant
 *   transmural mid-wall average; control.dat had already been setting
 *   -lv_alpha_apex/-lv_alpha_base, which those old option names never
 *   matched, so control.dat's values were silently ignored -- see the
 *   "unused options" warning every run printed. Renamed to match both
 *   control.dat and the actual latitude-interpolated usage below.)
 *   -lv_N_taper_rings [2]   rings beyond ring 0 (which is pinned, along with
 *                           the cap) over which active fiber magnitude
 *                           ramps linearly from 0 up to 1, instead of
 *                           switching straight from 0% to 100% activation
 *                           at the pinned/free boundary — that hard switch
 *                           right next to a rigid wall is what was tearing
 *                           elements just outside whatever was pinned.
 *
 * The helix angle varies linearly by latitude, α(z*) = (1-z*)·α_apex +
 * z*·α_base with z* the normalized apex(0)->base(1) axial coordinate
 * (docs/lv_geometry_theory.tex §3.6 Eq.4-5). Still a single constant
 * transmurally (one representative mid-wall surface per element, per
 * §3.3) — no depth (endo/epi) variation, only latitude variation.
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
 * CreateLVMeshUnstructured — load a pre-generated unstructured Delaunay
 * triangulation of the LV surface (see
 * BioFEM-studies/lv_test/unstructured_mesh/lv_unstructured_mesh.py and
 * src/lv_geometry_unstructured.c) and assign the same Bayer/Streeter fiber
 * architecture and near-apex gamma taper CreateLVMesh uses. Unlike
 * CreateLVMesh there is no explicit apex cap: the apex is one ordinary
 * mesh vertex. Requires -lv_unstructured_mesh_file <path>.
 *
 * After this call, in addition to everything CreateLVMesh sets:
 *   ibm->n_apex_pin       = 3 (or fewer)
 *   ibm->apex_pin_nodes[] = minimal ("3-2-1") rigid-body pin node indices,
 *                           consumed directly by main.c FormFunctionFEM
 */
PetscErrorCode CreateLVMeshUnstructured(IBMNodes *ibm, FE *fem, const LVParams *p);

/*
 * WriteLVFiberVTK — write the surface mesh + per-element fiber vectors to a
 * legacy VTK file for inspection in ParaView (Glyph filter on fiber_direction).
 */
PetscErrorCode WriteLVFiberVTK(IBMNodes *ibm, const char *filepath);

#endif /* LV_GEOMETRY_H */
