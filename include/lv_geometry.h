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
  PetscReal  alpha_apex;    /* helix angle at the apex (latitude endpoint) [degrees] --
                               only used when fiber_latitude_sweep=1, see below */
  PetscReal  alpha_base;    /* helix angle at the base (latitude endpoint) [degrees] --
                               only used when fiber_latitude_sweep=1, see below */
  PetscReal  alpha_midwall; /* DEFAULT fiber model: single constant helix angle [degrees],
                               representing the mid-wall crossing of the real transmural
                               (endo +alpha -> epi -alpha) rotation -- see Bayer et al. 2012
                               (Ann. Biomed. Eng. 40, LDRB algorithm) rule R2 and Goktepe,
                               Menzel & Kuhl 2014 (JMPS 72) Fig. 10. This shell has one
                               midsurface (no endo/epi split), so it cannot resolve the
                               transmural sweep directly; 0 deg (circumferential) is the
                               theoretically consistent single value for that midsurface. */
  PetscInt   fiber_latitude_sweep; /* 0 (default) = constant alpha_midwall everywhere.
                               1 = LEGACY: linearly interpolate alpha_apex -> alpha_base by
                               latitude instead. This was a mislabeling bug: alpha_apex/base
                               were renamed from alpha_endo/alpha_epi (a transmural pair) to
                               match control.dat's flag names, but the interpolation variable
                               stayed latitude (apex-to-base) instead of becoming depth
                               (endo-to-epi) -- so it swept the correct +-60deg range along
                               the wrong anatomical axis. Kept opt-in for comparison. */
  PetscInt   N_taper_rings; /* rings beyond ring 0 over which active fiber
                               magnitude ramps 0 -> 1 (see stage 7 comment
                               in CreateLVMesh); 0 = hard on/off at ring 0   */
  PetscInt   gamma_wave;         /* 0 (default) = every element's gamma ramp starts at t=0
                                    simultaneously. 1 = propagating-wavefront activation: each
                                    element's local ramp is delayed proportionally to its angular
                                    (theta) distance from the apex, modeling a depolarization front
                                    that reaches the apex first and spreads toward the base --
                                    see Barta et al. 1987 / Bovendeerd et al. 1992 Sec. "Activation
                                    sequence". Simplified to a single surface-propagation delay
                                    (no separate transmural leg, since this is a single-midsurface
                                    shell with no endo/epi split). */
  PetscReal  gamma_wave_delay_max; /* delay [same units as -dt] experienced by the latest-activated
                                    element (theta_c = theta_cut, i.e. the base rim) when
                                    gamma_wave=1; delay grows linearly with theta_c/theta_cut from
                                    0 at the apex. */
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
 *   -lv_alpha_midwall [0.0]  DEFAULT fiber model: single constant helix angle
 *                            [deg] used everywhere (see LVParams doc above).
 *   -lv_fiber_latitude_sweep [0]  1 = LEGACY: linearly blend -lv_alpha_apex
 *                            [60.0] -> -lv_alpha_base [-60.0] by latitude
 *                            instead. This was a mislabeling bug: these two
 *                            options were originally named -lv_alpha_endo/
 *                            -lv_alpha_epi and read as a constant transmural
 *                            mid-wall average (correct); they were renamed to
 *                            -lv_alpha_apex/-lv_alpha_base to match
 *                            control.dat's option names (which the old names
 *                            never matched, so control.dat's values were
 *                            silently ignored), but the interpolation
 *                            variable stayed latitude instead of becoming
 *                            transmural depth -- so the correct +-60deg
 *                            range from Bayer et al. 2012 (Ann. Biomed. Eng.
 *                            40, LDRB rule R2) / Goktepe, Menzel & Kuhl 2014
 *                            (JMPS 72, Fig. 10) ended up swept along the
 *                            wrong anatomical axis (apex-to-base instead of
 *                            endo-to-epi). Kept opt-in for A/B comparison,
 *                            and as a placeholder for a future true
 *                            per-thickness-quadrature-point transmural
 *                            implementation.
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
