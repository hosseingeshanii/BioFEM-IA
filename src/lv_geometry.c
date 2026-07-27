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

/* 0-based flat index of apex-cap center node i (0,1,2), stored right after
 * all ring nodes: index N_total*N_phi + i. (Superseded by the multi-level
 * cap coarsening rings below; kept only for the #if 0 reference blocks.) */
#define CAP_NODE(i, N_total, Np)  ((N_total)*(Np) + (i))

/* Max halving levels for apex cap coarsening (see CreateLVMesh) — 16 is far
 * more than any realistic N_phi (2^16) would ever need. */
#define LV_MAX_CAP_LEVELS 16

/* ------------------------------------------------------------------ */

PetscErrorCode LVParamsCreate(LVParams *p)
{
  p->a            = 4.5;
  p->b            = 2.5;
  p->f_cut        = 0.55;
  p->N_theta      = 16;
  p->N_phi        = 32;
  p->N_apex_extra = 0;
  p->N_cap_final  = 4;
  p->alpha_endo   = 60.0;
  p->alpha_epi    = -60.0;
  p->N_taper_rings = 2;

  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_a",             &p->a,            PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_b",             &p->b,            PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_f_cut",         &p->f_cut,        PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_theta",       &p->N_theta,      PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_phi",         &p->N_phi,        PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_apex_extra",  &p->N_apex_extra, PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_cap_final",   &p->N_cap_final,  PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_endo",    &p->alpha_endo,   PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lv_alpha_epi",     &p->alpha_epi,    PETSC_NULL);
  PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-lv_N_taper_rings", &p->N_taper_rings, PETSC_NULL);

  return 0;
}

/* ------------------------------------------------------------------ */

PetscErrorCode CreateLVMesh(IBMNodes *ibm, FE *fem, const LVParams *p)
{
  PetscReal a              = p->a;
  PetscReal b              = p->b;
  PetscReal f_cut          = p->f_cut;
  PetscInt  N_theta        = p->N_theta;
  PetscInt  N_phi          = p->N_phi;
  PetscInt  N_apex_extra   = p->N_apex_extra;
  PetscReal alpha_endo_deg = p->alpha_endo;
  PetscReal alpha_epi_deg  = p->alpha_epi;
  PetscInt  N_taper_rings  = p->N_taper_rings;

  /* Total rings = base mesh rings + extra apex rings */
  PetscInt  N_total        = N_theta + N_apex_extra;

#if 0
  /* --- superseded apex-cap design: fanned ring 0 (N_phi nodes) straight to
   * 3 interior nodes in a single jump. With N_phi as large as 32, that's a
   * ~10:1 collapse per zone — the ring-0 edges are short arcs near the
   * apex while the two edges to the far-off interior node are much longer,
   * so the resulting triangles are thin slivers, which is exactly what
   * made convergence hard near the cap. Kept here for reference; see the
   * replacement coarsening scheme below. */
  const PetscInt N_cap_nodes = 3;
#endif

  /* Apex cap coarsening: instead of collapsing all of ring 0 straight to a
   * handful of interior nodes (see #if 0 block above), do ONE "ring
   * reduction" band straight from ring 0 (N_phi nodes) down to a small
   * coarse ring of -lv_N_cap_final nodes, then close THAT small ring
   * directly by connecting its own nodes to each other — no separate apex
   * vertex at all. A fan converging on one shared center point is
   * inherently cone-shaped, and pinning only that one point while
   * everything nearby pulls on it under load is exactly what produced a
   * needle-like spike there; removing the point removes the singularity,
   * not just its motion. The whole cap is pinned rigid and carries no
   * active fiber force (see stage 6 / stage 7 below), so its own element
   * shape no longer drives solver conditioning — the one thing that still
   * matters is that the transition band's triangles touch ring 0, which is
   * NOT pinned. Segment i of the transition band spans fine-ring indices
   * [zone_bound[i], zone_bound[i+1]) (nearly-even split of N_phi across the
   * C coarse segments, allowing -lv_N_cap_final to be any value >= 3, not
   * just a divisor of N_phi); each segment gets K_i fan triangles (split as
   * evenly as possible between the two coarse nodes bounding it) plus 1
   * "closing" triangle for the coarse-ring edge itself, using the fine
   * ring's own shared split-point node as its third vertex. The coarse
   * ring's own C nodes are then closed with a plain (C-2)-triangle fan
   * from one of its own nodes — for C=3 that's exactly 1 triangle: "just
   * an element at the apex cap", all 3 of whose nodes are pinned. */
  PetscInt cap_level_n[LV_MAX_CAP_LEVELS];      /* node count of coarsening ring m (only m=0 used) */
  PetscInt cap_level_offset[LV_MAX_CAP_LEVELS]; /* flat node-index base of coarsening ring m */
  PetscInt n_cap_levels = 0;
  {
    PetscInt want = p->N_cap_final;
    if (want < 3) want = 3;
    if (want < N_phi) {
      cap_level_n[0] = want;
      n_cap_levels   = 1;
    }
  }
  /* Ring closed directly (no apex vertex): the single coarsening ring, or
   * ring 0 itself if N_phi was already too small to coarsen at all. */
  PetscInt N_final = (n_cap_levels > 0) ? cap_level_n[0] : N_phi;

  PetscInt n_cap_ring_nodes = (n_cap_levels > 0) ? cap_level_n[0] : 0;

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

  /* Cap element count: the transition band contributes F+C triangles total
   * (sum of each segment's K_i fan triangles, which sum to F, plus one
   * closing triangle per segment, C of them) if a coarsening ring exists;
   * the innermost ring (C nodes, or N_phi if there's no coarsening ring at
   * all) is then closed directly with a (ring_size - 2)-triangle fan among
   * its own nodes — no apex vertex. */
  PetscInt n_cap_elmt = N_final - 2;
  if (n_cap_levels > 0) n_cap_elmt += N_phi + N_final;

  PetscInt n_elmt_base  = (N_total - 1) * 2 * N_phi;   /* quad strips only         */
  {
    PetscInt off = N_total * N_phi;
    for (PetscInt m = 0; m < n_cap_levels; m++) {
      cap_level_offset[m] = off;
      off += cap_level_n[m];
    }
  }
  PetscInt n_v           = N_total * N_phi + n_cap_ring_nodes;
  /* n_elmt_base intentionally EXCLUDES the cap: IrrVer()/Patch() already
   * skip ec >= n_elmt_base for Loop-subdivision valence classification and
   * fall back to CST membrane elements there (see bending.c / active_strain.c) —
   * exactly the mechanism needed for the cap's irregular valence. */
  PetscInt n_elmt       = n_elmt_base + n_cap_elmt;
  PetscInt n_edge       = 0;  /* base BC via EdgeDirectionalFix, no ghosts */
  PetscInt n_ghosts     = 0;
  /* Apex ring, base ring, AND every cap-only node (all coarsening-ring
   * nodes — there's no separate apex vertex any more) are boundary groups —
   * see the n_bnodes[2] comment below for why the whole cap is pinned. */
  PetscInt sum_n_bnodes = 2 * N_phi + n_cap_ring_nodes;

  ibm->n_v          = n_v;
  ibm->n_elmt       = n_elmt;
  ibm->n_elmt_base  = n_elmt_base;
  ibm->n_edge       = n_edge;
  ibm->n_ghosts     = n_ghosts;
  ibm->sum_n_bnodes = sum_n_bnodes;
  ibm->ibi          = 0;

  PetscPrintf(PETSC_COMM_WORLD,
    "LV mesh: a=%.4f  b=%.4f  f_cut=%.2f  theta_cut=%.2f deg  "
    "N_theta=%d  N_apex_extra=%d  N_phi=%d  n_v=%d  n_elmt=%d\n",
    a, b, f_cut, theta_cut * 180.0 / PETSC_PI,
    (int)N_theta, (int)N_apex_extra, (int)N_phi, (int)n_v, (int)n_elmt);
  PetscPrintf(PETSC_COMM_WORLD,
    "LV apex cap: %d coarsening level(s) (N_phi=%d -> N_final=%d), "
    "no apex vertex, %d cap element(s)\n",
    (int)n_cap_levels, (int)N_phi, (int)N_final, (int)n_cap_elmt);

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
  /* group 0: apex ring — fixed via EdgeDirectionalFix(0,...) in main.c, and also
   *          tells IrrVer the apex ring is a boundary (bcount=count+3=6 → regular).
   * group 1: base ring — currently NOT fixed (main.c only calls edge_n=0 and 2).
   * group 2: the WHOLE cap (every coarsening-ring node, NOT ring 0) — fixed
   *          via EdgeDirectionalFix(2,...). There is no separate apex
   *          vertex to pin: earlier designs funneled the cap to one shared
   *          center point, and pinning only that point while everything
   *          nearby pulled on it under load produced a sharp needle-like
   *          spike. Removing the center vertex (see cap coarsening comment
   *          above) removes the singularity itself, not just its motion —
   *          ring 0 and everything outward (the actual LV wall) remains
   *          fully free. */
  ibm->n_bnodes[0] = N_phi;
  ibm->n_bnodes[1] = N_phi;
  ibm->n_bnodes[2] = n_cap_ring_nodes;

  /* ----------------------------------------------------------------
   * 3.  Node coordinates  —  RING_NODE(k,j,N_phi) = k*N_phi + j
   *     Ring k=0 is the first ring below the apex (θ = theta_step).
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 3: filling node coordinates\n");

  /* Two-segment theta spacing:
   *   rings 0..N_apex_extra-1  : squeezed into [0, theta_step] to shrink the apex hole
   *   rings N_apex_extra..N_total-1 : regular spacing theta_step..theta_cut */
  for (PetscInt k = 0; k < N_total; k++) {
    PetscReal theta;
    if (k < N_apex_extra)
      theta = (PetscReal)(k + 1) / (PetscReal)(N_apex_extra + 1) * theta_step;
    else
      theta = (PetscReal)(k - N_apex_extra + 1) * theta_step;
    PetscReal sth = sin(theta);
    PetscReal cth = cos(theta);
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

#if 0
  /* --- superseded: 3 apex-cap center nodes clustered halfway (in theta)
   * between the true apex and ring 0, regardless of how large N_phi was —
   * see the #if 0 block at the top of this function for why that made
   * poorly-shaped triangles. Kept here for reference. */
  {
    PetscReal theta_ring0 = (N_apex_extra > 0)
      ? theta_step / (PetscReal)(N_apex_extra + 1)
      : theta_step;
    PetscReal theta_c = 0.5 * theta_ring0;
    PetscReal sth_c   = sin(theta_c);
    PetscReal cth_c   = cos(theta_c);
    for (PetscInt i = 0; i < N_cap_nodes; i++) {
      PetscReal phi_c = 2.0 * PETSC_PI * ((PetscReal)i + 0.5) / (PetscReal)N_cap_nodes;
      PetscInt  nc    = CAP_NODE(i, N_total, N_phi);
      ibm->x_bp[nc]  = b * sth_c * cos(phi_c);
      ibm->y_bp[nc]  = b * sth_c * sin(phi_c);
      ibm->z_bp[nc]  = a * cth_c;
      ibm->x_bp0[nc] = ibm->x_bp[nc];
      ibm->y_bp0[nc] = ibm->y_bp[nc];
      ibm->z_bp0[nc] = ibm->z_bp[nc];
    }
  }
#endif

  /* Apex-cap coarsening ring node coordinates — no apex vertex. Coarsening
   * ring m sits at theta_m = theta_ring0 * (cap_level_n[m] / N_phi) —
   * proportional to its own node count relative to ring 0's — which keeps
   * circumferential node spacing (arc length ~ r*dphi, and r ~ theta near
   * the apex) approximately constant across every coarsening ring, so the
   * transition band doesn't introduce its own aspect-ratio jump on top of
   * the deliberate radial coarsening. */
  {
    PetscReal theta_ring0 = (N_apex_extra > 0)
      ? theta_step / (PetscReal)(N_apex_extra + 1)
      : theta_step;

    for (PetscInt m = 0; m < n_cap_levels; m++) {
      PetscInt  cnt   = cap_level_n[m];
      PetscReal theta = theta_ring0 * (PetscReal)cnt / (PetscReal)N_phi;
      PetscReal sth   = sin(theta);
      PetscReal cth   = cos(theta);
      for (PetscInt j = 0; j < cnt; j++) {
        PetscReal phi = 2.0 * PETSC_PI * (PetscReal)j / (PetscReal)cnt;
        PetscInt  nc  = cap_level_offset[m] + j;
        ibm->x_bp[nc]  = b * sth * cos(phi);
        ibm->y_bp[nc]  = b * sth * sin(phi);
        ibm->z_bp[nc]  = a * cth;
        ibm->x_bp0[nc] = ibm->x_bp[nc];
        ibm->y_bp0[nc] = ibm->y_bp[nc];
        ibm->z_bp0[nc] = ibm->z_bp[nc];
      }
    }
  }

  /* ----------------------------------------------------------------
   * 4.  Element connectivity  (0-based node indices)
   *
   *   Quad strips (base mesh) followed by the coarsened apex cap.
   * ---------------------------------------------------------------- */
  PetscInt ec = 0;

  for (PetscInt k = 0; k < N_total - 1; k++) {
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

  /* n_elmt_base excludes the cap by design (see comment at its definition
   * above) — ec must equal n_elmt_base exactly at this point. */
  if (ec != n_elmt_base) {
    SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "LV mesh: base element count mismatch (ec=%d, n_elmt_base=%d)",
             (int)ec, (int)n_elmt_base);
  }

#if 0
  /* --- superseded apex cap connectivity: ring-0 (N_phi nodes) split into
   * 3 zones, each fanned in one jump straight to 1 of 3 interior center
   * nodes. See the #if 0 block near the top of this function for why this
   * produced badly-shaped (sliver) triangles. Kept here for reference;
   * replaced by the multi-level ring-reduction cap below. */
  {
    PetscInt zone_bound[4];
    for (PetscInt i = 0; i <= N_cap_nodes; i++) {
      zone_bound[i] = (PetscInt)round((PetscReal)i * (PetscReal)N_phi
                                       / (PetscReal)N_cap_nodes);
    }

    for (PetscInt i = 0; i < N_cap_nodes; i++) {
      PetscInt c_i = CAP_NODE(i, N_total, N_phi);
      /* zone-interior fan: one triangle per ring-0 edge strictly inside the zone */
      for (PetscInt k = zone_bound[i]; k < zone_bound[i + 1] - 1; k++) {
        ibm->nv1[ec] = RING_NODE(0, k,     N_phi);
        ibm->nv2[ec] = RING_NODE(0, k + 1, N_phi);
        ibm->nv3[ec] = c_i;
        ec++;
      }
    }

    for (PetscInt i = 0; i < N_cap_nodes; i++) {
      PetscInt i_next   = (i + 1) % N_cap_nodes;
      PetscInt c_i      = CAP_NODE(i,      N_total, N_phi);
      PetscInt c_next   = CAP_NODE(i_next, N_total, N_phi);
      PetscInt r_last   = zone_bound[i + 1] - 1;             /* last node of zone i     */
      PetscInt r_seam   = zone_bound[i + 1] % N_phi;          /* first node of zone i+1  */

      /* seam bridge A: covers the ring-0 boundary edge (r_last, r_seam) */
      ibm->nv1[ec] = RING_NODE(0, r_last, N_phi);
      ibm->nv2[ec] = RING_NODE(0, r_seam, N_phi);
      ibm->nv3[ec] = c_i;
      ec++;

      /* seam bridge B: fills the gap up to the center triangle's edge */
      ibm->nv1[ec] = c_next;
      ibm->nv2[ec] = c_i;
      ibm->nv3[ec] = RING_NODE(0, r_seam, N_phi);
      ec++;
    }

    /* center triangle */
    ibm->nv1[ec] = CAP_NODE(0, N_total, N_phi);
    ibm->nv2[ec] = CAP_NODE(1, N_total, N_phi);
    ibm->nv3[ec] = CAP_NODE(2, N_total, N_phi);
    ec++;
  }
#endif

  /* ----------------------------------------------------------------
   * 4b.  Apex cap  —  a single ring-reduction transition band
   *      (N_phi -> N_final), closed by connecting the coarse ring's own
   *      nodes directly to each other — no apex vertex.
   *
   *   The transition band fully triangulates the annulus between ring 0
   *   (F=N_phi nodes) and the coarse ring (C=N_final nodes). Segment i
   *   spans fine-ring indices [zone_bound[i], zone_bound[i+1]) — a
   *   nearly-even split of F across the C segments, allowing C to be any
   *   value >= 3, not just a divisor of F — and gets K_i = zone_bound[i+1]
   *   - zone_bound[i] "fan" triangles covering its fine-ring edges (the
   *   first ~K_i/2 apexed at c(i), the rest at c(i+1)), plus 1 "closing"
   *   triangle covering the coarse-ring edge (c(i),c(i+1)) itself, using
   *   the fine ring's own split-point node as its third vertex. Every
   *   fine-ring edge and every coarse-ring edge ends up covered by exactly
   *   one triangle from this band; the coarse-ring edges get their second
   *   (closing) face from the self-fan below. The coarse ring is then
   *   closed directly — a fan from its own node 0 to nodes 2..C-1, C-2
   *   triangles, no new vertex. For C=3 that's exactly 1 triangle.
   * ---------------------------------------------------------------- */
  {
    PetscInt fine_offset = -1;   /* -1 sentinel: use RING_NODE(0,j) for ring 0 */
    PetscInt fine_count  = N_phi;

    for (PetscInt m = 0; m < n_cap_levels; m++) {
      PetscInt coarse_offset = cap_level_offset[m];
      PetscInt coarse_count  = cap_level_n[m];

      PetscInt *zone_bound;
      ierr = PetscMalloc1(coarse_count + 1, &zone_bound); CHKERRQ(ierr);
      for (PetscInt i = 0; i <= coarse_count; i++) {
        zone_bound[i] = (PetscInt)round((PetscReal)i * (PetscReal)fine_count
                                         / (PetscReal)coarse_count);
      }

      for (PetscInt i = 0; i < coarse_count; i++) {
        PetscInt i_next    = (i + 1) % coarse_count;
        PetscInt c_i       = coarse_offset + i;
        PetscInt c_next    = coarse_offset + i_next;
        PetscInt seg_start = zone_bound[i];
        PetscInt K         = zone_bound[i + 1] - seg_start;
        PetscInt split     = K / 2;   /* fan edges [0,split) -> c_i, [split,K) -> c_next */

        for (PetscInt j = 0; j < K; j++) {
          PetscInt fa_idx = seg_start + j;
          PetscInt fb_idx = (fa_idx + 1) % fine_count;
          PetscInt fa = (fine_offset < 0) ? RING_NODE(0, fa_idx, N_phi) : fine_offset + fa_idx;
          PetscInt fb = (fine_offset < 0) ? RING_NODE(0, fb_idx, N_phi) : fine_offset + fb_idx;
          PetscInt apex = (j < split) ? c_i : c_next;
          ibm->nv1[ec] = fa;  ibm->nv2[ec] = fb;  ibm->nv3[ec] = apex;  ec++;
        }

        /* closing triangle: coarse edge (c_i,c_next), apex at the fine
         * ring's split-point node shared by both fan halves above */
        PetscInt fs_idx = (seg_start + split) % fine_count;
        PetscInt fs = (fine_offset < 0) ? RING_NODE(0, fs_idx, N_phi) : fine_offset + fs_idx;
        ibm->nv1[ec] = fs;  ibm->nv2[ec] = c_next;  ibm->nv3[ec] = c_i;  ec++;
      }

      ierr = PetscFree(zone_bound); CHKERRQ(ierr);

      fine_offset = coarse_offset;
      fine_count  = coarse_count;
    }

    /* Close the innermost ring (fine_offset/fine_count, still ring 0 if no
     * coarsening ran) directly by fanning from its own node 0 — no apex
     * vertex, count-2 triangles. For count==3 this is exactly 1 triangle:
     * "just an element at the apex cap", all 3 of its nodes pinned. */
    {
      PetscInt count = fine_count;
      PetscInt n0 = (fine_offset < 0) ? RING_NODE(0, 0, N_phi) : fine_offset;
      for (PetscInt j = 1; j < count - 1; j++) {
        PetscInt na = (fine_offset < 0) ? RING_NODE(0, j,     N_phi) : fine_offset + j;
        PetscInt nb = (fine_offset < 0) ? RING_NODE(0, j + 1, N_phi) : fine_offset + (j + 1);
        ibm->nv1[ec] = n0;  ibm->nv2[ec] = na;  ibm->nv3[ec] = nb;  ec++;
      }
    }

    /* Orientation fix: for a star-shaped-about-origin surface like this
     * spheroid, the outward normal always has positive dot product with
     * the element centroid. Swap nv2/nv3 on any cap triangle that fails
     * this test so all cap elements share consistent outward winding with
     * the rest of the mesh, regardless of the manual ordering above. */
    for (PetscInt e = n_elmt_base; e < ec; e++) {
      PetscInt n1 = ibm->nv1[e], n2 = ibm->nv2[e], n3 = ibm->nv3[e];
      struct Cmpnts p1 = {ibm->x_bp[n1], ibm->y_bp[n1], ibm->z_bp[n1]};
      struct Cmpnts p2 = {ibm->x_bp[n2], ibm->y_bp[n2], ibm->z_bp[n2]};
      struct Cmpnts p3 = {ibm->x_bp[n3], ibm->y_bp[n3], ibm->z_bp[n3]};
      struct Cmpnts centroid = AMULT(1.0 / 3.0, PLUS(PLUS(p1, p2), p3));
      struct Cmpnts normal   = CROSS(MINUS(p2, p1), MINUS(p3, p1));
      if (DOT(normal, centroid) < 0.0) {
        ibm->nv2[e] = n3;
        ibm->nv3[e] = n2;
      }
    }
  }

  if (ec != n_elmt) {
    SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "LV mesh: total element count mismatch (ec=%d, n_elmt=%d)",
             (int)ec, (int)n_elmt);
  }

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
   * 6.  Boundary rings  —  top (ring 0), base (ring N_theta-1), whole cap
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 6: boundary edge info\n");
  /* Both apex ring (edge_n=0) and base ring (edge_n=1) in bnodes[].
     IrrVer uses this to apply the ghost trick (bcount=count+3) for boundary nodes,
     keeping the apex ring classified as REGULAR despite the cap additions.
     No displacement BCs are applied to the apex ring — EdgeDirectionalFix targets edge_n=1 only. */
  for (PetscInt j = 0; j < N_phi; j++)
    ibm->bnodes[j] = RING_NODE(0, j, N_phi);               /* apex ring  (edge_n=0) */
  for (PetscInt j = 0; j < N_phi; j++)
    ibm->bnodes[N_phi + j] = RING_NODE(N_total - 1, j, N_phi); /* base ring (edge_n=1) */
  /* Whole cap (edge_n=2): every coarsening-ring node — there's no separate
   * apex vertex — fixed via EdgeDirectionalFix(2,...) in main.c. */
  {
    PetscInt off = 2 * N_phi;
    for (PetscInt m = 0; m < n_cap_levels; m++) {
      for (PetscInt j = 0; j < cap_level_n[m]; j++) {
        ibm->bnodes[off++] = cap_level_offset[m] + j;
      }
    }
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
   *   α  = (α_endo + α_epi)/2   mid-wall value, constant (Bayer Eq.2, d=0.5)
   *
   *   f  = cos(α)·e_c + sin(α)·e_l                → ibm->n_fib[ec]
   * ---------------------------------------------------------------- */
  PetscPrintf(PETSC_COMM_WORLD, "[lv] stage 7: assigning fiber directions\n");

  /* Mid-wall helix angle: Bayer 2012 Eq.(2) evaluated at d=0.5 */
  PetscReal alpha = 0.5 * (alpha_endo_deg + alpha_epi_deg) * PETSC_PI / 180.0;

  /* Direction vector pointing from apex toward base along the z axis */
  struct Cmpnts e_down = {0.0, 0.0, -1.0};

  /* Apex cap: no active fiber contribution at all, via gamma_scale=0 (see
   * ElemActDefGrad in active_strain.c, which multiplies gamma by
   * ibm->gamma_scale[ec]) rather than zeroing n_fib itself — n_fib stays a
   * genuine unit direction everywhere so nothing downstream that assumes a
   * unit fiber vector breaks. The circumferential/meridional frame (e_c,
   * e_l) still has a coordinate singularity right at the true apex point
   * (like the "hairy ball" pole of a sphere), but since the cap carries no
   * active stress at all that singularity never matters mechanically.
   *
   * Ring 0 is free (not pinned — see main.c), but going straight from 0%
   * (cap) to 100% activation right at the cap/ring-0 boundary produced
   * tearing/spiking there: a hard on/off activation switch right next to a
   * rigid boundary concentrates all the strain into whatever is
   * immediately adjacent to it. Instead, ramp gamma itself (via
   * gamma_scale) linearly from 0 at theta_ring0 (the cap edge) up to 1 over
   * -lv_N_taper_rings ring spacings, so nearby elements pick up contraction
   * gradually. Elements fully beyond the taper zone are unaffected
   * (gamma_scale clamps to 1). */
  PetscReal theta_ring0_for_taper = (N_apex_extra > 0)
    ? theta_step / (PetscReal)(N_apex_extra + 1)
    : theta_step;
  PetscReal theta_taper_width = (PetscReal)N_taper_rings * theta_step;

  for (ec = 0; ec < n_elmt; ec++) {
    PetscInt n1e, n2e, n3e;
    struct Cmpnts c;

    if (ec >= n_elmt_base) {
      /* Cap elements: use the element's own vertex centroid for direction
       * (harmless — gamma_scale=0 makes the direction mechanically inert)
       * and fully deactivate via gamma_scale. */
      n1e = ibm->nv1[ec]; n2e = ibm->nv2[ec]; n3e = ibm->nv3[ec];
      c.x = (ibm->x_bp[n1e] + ibm->x_bp[n2e] + ibm->x_bp[n3e]) / 3.0;
      c.y = (ibm->y_bp[n1e] + ibm->y_bp[n2e] + ibm->y_bp[n3e]) / 3.0;
      c.z = (ibm->z_bp[n1e] + ibm->z_bp[n2e] + ibm->z_bp[n3e]) / 3.0;

      struct Cmpnts grad_F = {c.x / (b * b), c.y / (b * b), c.z / (a * a)};
      struct Cmpnts e_n = UNIT(grad_F);
      struct Cmpnts e_x = {1.0, 0.0, 0.0};
      PetscReal d2 = DOT(e_x, e_n);
      struct Cmpnts e_l_raw = {e_x.x - d2 * e_n.x, e_x.y - d2 * e_n.y, e_x.z - d2 * e_n.z};
      struct Cmpnts e_l = UNIT(e_l_raw);
      struct Cmpnts e_c = CROSS(e_n, e_l);

      ibm->n_fib[ec].x = cos(alpha) * e_c.x + sin(alpha) * e_l.x;
      ibm->n_fib[ec].y = cos(alpha) * e_c.y + sin(alpha) * e_l.y;
      ibm->n_fib[ec].z = cos(alpha) * e_c.z + sin(alpha) * e_l.z;
      ibm->gamma_scale[ec] = 0.0;
      continue;
    }

    n1e = ibm->nv1[ec]; n2e = ibm->nv2[ec]; n3e = ibm->nv3[ec];

    /* Element centroid */
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

    /* Ring-0 activation taper: theta from centroid z (c.z ~= a*cos(theta)
     * near the surface); clamp for centroids that land fractionally
     * outside [-1,1] due to averaging, then ramp linearly 0 (at
     * theta_ring0, the cap edge) -> 1 (theta_ring0 + taper width). Applied
     * to gamma_scale, not n_fib — see comment above the loop. */
    PetscReal cos_theta_c = c.z / a;
    if (cos_theta_c > 1.0)  cos_theta_c = 1.0;
    if (cos_theta_c < -1.0) cos_theta_c = -1.0;
    PetscReal theta_c = acos(cos_theta_c);
    PetscReal taper = (theta_taper_width > 0.0)
      ? (theta_c - theta_ring0_for_taper) / theta_taper_width
      : 1.0;
    if (taper > 1.0) taper = 1.0;
    if (taper < 0.0) taper = 0.0;

    /* Fiber vector in the tangent plane — Bayer 2012 Eq.(7) direction,
     * always unit magnitude; activation strength lives in gamma_scale. */
    ibm->n_fib[ec].x = cos(alpha) * e_c.x + sin(alpha) * e_l.x;
    ibm->n_fib[ec].y = cos(alpha) * e_c.y + sin(alpha) * e_l.y;
    ibm->n_fib[ec].z = cos(alpha) * e_c.z + sin(alpha) * e_l.z;
    ibm->gamma_scale[ec] = taper;
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
