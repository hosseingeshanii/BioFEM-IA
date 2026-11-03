/* biofem-solid3d: 3D active-strain solid, Taylor-Hood P2-P1 on DMPlex/PetscFE.
 * Formulation: research lv-vv/formulation/solid3d_formulation.pdf (v2); equation numbers below refer to it.
 *
 * -problem cube   free active cube (verification plan item 1): HO, uniform fiber e_x, rollers on x/y/z = min;
 *                 exact u = (Fa - I) X, p = a.
 * -problem plate  the shell's Nitti plate (research studies/2026-10-19_plate_nitti_verification): compressible neo-Hookean
 *                 (-mu, -K), diagonal Fa = diag(1 - g/Lx^2, 1 - g/Ly^2, 1/(a1 a2)) (Nitti eq. 44-47, -nitti_Lx/-nitti_Ly),
 *                 rollers on x/y/z = min, traction -traction [Pa] on x = max. Exact: homogeneous F = diag(l1, l2, l3).
 * -problem lv     Pezzuto LV on the gmsh prolate mesh (pulse3d/prolate_mesh.py, physical tags 1 endo, 2 epi, 3 base):
 *                 HO, fibers as an auxiliary P1 field from the prolate depth (same formula and P1 interpolation as the
 *                 pulse script), base u_z = 0, rim pins (endo rim theta=0: u_x = u_y = 0; epi rim theta=0: u_y = 0),
 *                 gamma linear 0 -> gamma -> 0 over -steps (Pezzuto ramp); vertex displacements written every -out_every steps.
 *                 -p_ed > 0: Track P, the Pezzuto PV loop (same protocol as pulse3d/lv3d_pv_loop.py): cavity pressure p_c as
 *                 a follower load on the endocardium (tag 1), filling 0 -> p_ed with gamma = 0 in -fill_steps, then over the
 *                 gamma ramp: isovolumic contraction (V = EDV, p_c unknown) until p_c >= p_ao, ejection at p_ao until gamma
 *                 peaks, isovolumic relaxation (V = ESV) until p_c <= p_ed, filling at p_ed. pv.csv in -out_dir.
 *                 -mean_bc 1: instead of the rim pins, Pezzuto's null average in-plane displacement and null average
 *                 rotation about z of the ventricle (MeanBC* below); the pins (default) are kept for the pulse comparison.
 * Mesh from options: -dm_plex_filename <file.msh> (or a PETSc box with a mesh generator).
 */
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "solid3d_ho.h"
#include "solid3d.h"

typedef enum { PROB_CUBE = 0, PROB_PLATE = 1, PROB_LV = 2 } Problem;
static const char *const ProblemNames[] = {"cube", "plate", "lv", "Problem", "PROB_", NULL};

/* PetscDS constants */
enum { C_LAW, C_A, C_B, C_AF, C_BF, C_AS, C_BS, C_AFS, C_BFS, C_TENSION, C_SHEET, C_MU, C_K,
       C_GAMMA, C_FA_DIAG, C_A1, C_A2, C_A3, C_FIBER_AUX, C_F0, C_S0 = C_F0 + 3, C_TRACTION = C_S0 + 3, C_PCAV, C_ZB, C_REG, C_N };

typedef struct { MatParams m; PetscReal A[9], f[3], s[3]; } KernelData;

static void kernel_data(const PetscScalar c[], const PetscInt aOff[], const PetscScalar a[], KernelData *kd)
{
  MatParams *m = &kd->m;
  m->law = (MatLaw)(PetscInt)PetscRealPart(c[C_LAW]);
  m->a = PetscRealPart(c[C_A]); m->b = PetscRealPart(c[C_B]); m->af = PetscRealPart(c[C_AF]); m->bf = PetscRealPart(c[C_BF]);
  m->as = PetscRealPart(c[C_AS]); m->bs = PetscRealPart(c[C_BS]); m->afs = PetscRealPart(c[C_AFS]); m->bfs = PetscRealPart(c[C_BFS]);
  m->tension_only = (PetscBool)(PetscRealPart(c[C_TENSION]) != 0.0); m->sheet = (PetscBool)(PetscRealPart(c[C_SHEET]) != 0.0);
  m->mu = PetscRealPart(c[C_MU]); m->K = PetscRealPart(c[C_K]);
  const PetscBool aux = (PetscBool)(PetscRealPart(c[C_FIBER_AUX]) != 0.0);
  for (PetscInt i = 0; i < 3; i++) {
    kd->f[i] = aux ? PetscRealPart(a[aOff[0] + i]) : PetscRealPart(c[C_F0 + i]);
    kd->s[i] = aux ? PetscRealPart(a[aOff[1] + i]) : PetscRealPart(c[C_S0 + i]);
  }
  if (aux) {   /* interpolated P1 vectors are not unit / orthogonal between vertices: f unit, s Gram-Schmidt against f */
    PetscReal nf = PetscSqrtReal(kd->f[0] * kd->f[0] + kd->f[1] * kd->f[1] + kd->f[2] * kd->f[2]), sf;
    for (PetscInt i = 0; i < 3; i++) kd->f[i] /= nf;
    sf = kd->s[0] * kd->f[0] + kd->s[1] * kd->f[1] + kd->s[2] * kd->f[2];
    for (PetscInt i = 0; i < 3; i++) kd->s[i] -= sf * kd->f[i];
    PetscReal ns = PetscSqrtReal(kd->s[0] * kd->s[0] + kd->s[1] * kd->s[1] + kd->s[2] * kd->s[2]);
    for (PetscInt i = 0; i < 3; i++) kd->s[i] /= ns;
  }
  if (PetscRealPart(c[C_FA_DIAG]) != 0.0) {
    for (PetscInt i = 0; i < 9; i++) kd->A[i] = 0.0;
    kd->A[0] = 1.0 / PetscRealPart(c[C_A1]); kd->A[4] = 1.0 / PetscRealPart(c[C_A2]); kd->A[8] = 1.0 / PetscRealPart(c[C_A3]);
  } else AS_InverseFiberFa(PetscRealPart(c[C_GAMMA]), kd->f, kd->A);
}

static void grad_to_F(const PetscScalar u_x[], const PetscInt uOff_x[], PetscReal F[9])
{
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt l = 0; l < 3; l++) F[k * 3 + l] = (k == l ? 1.0 : 0.0) + PetscRealPart(u_x[uOff_x[0] + k * 3 + l]);   /* (2.1) */
}

static void JFinvT(const PetscReal F[9], PetscReal M[9])   /* cofactor matrix = J F^-T */
{
  M[0] = F[4] * F[8] - F[5] * F[7]; M[1] = F[5] * F[6] - F[3] * F[8]; M[2] = F[3] * F[7] - F[4] * F[6];
  M[3] = F[2] * F[7] - F[1] * F[8]; M[4] = F[0] * F[8] - F[2] * F[6]; M[5] = F[1] * F[6] - F[0] * F[7];
  M[6] = F[1] * F[5] - F[2] * F[4]; M[7] = F[2] * F[3] - F[0] * F[5]; M[8] = F[0] * F[4] - F[1] * F[3];
}

#define KERNEL_ARGS PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], \
  const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], \
  const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t

/* R_u (3.8): f1 = P */
static void f1_u(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  KernelData kd; PetscReal F[9], P[9];
  kernel_data(constants, aOff, a, &kd); grad_to_F(u_x, uOff_x, F);
  AS_Piola(&kd.m, kd.A, kd.f, kd.s, F, PetscRealPart(u[uOff[1]]), P, NULL);
  for (PetscInt i = 0; i < 9; i++) f1[i] = P[i];
}
/* R_p (3.9): incompressible -(J - 1); compressible law: p = 0 (the field is unused) */
static void f0_p(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  if ((MatLaw)(PetscInt)PetscRealPart(constants[C_LAW]) != MAT_HO) { f0[0] = u[uOff[1]]; return; }
  PetscReal F[9]; grad_to_F(u_x, uOff_x, F);
  f0[0] = -(F[0] * (F[4] * F[8] - F[5] * F[7]) - F[1] * (F[3] * F[8] - F[5] * F[6]) + F[2] * (F[3] * F[7] - F[4] * F[6]) - 1.0);
}
/* traction on the natural boundary: -t . du (sign: R = int P:grad du - int t . du) */
static void f0_traction(KERNEL_ARGS, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -constants[C_TRACTION]; f0[1] = 0.0; f0[2] = 0.0;
}
/* cavity pressure p_c on the endocardium, a follower load (3.8): R_u += p_c int du . J F^-T N dA, N = outward normal of
   the wall (into the cavity). Tangent: d(J F^-T)_iJ/dF_kl = (M_iJ M_kl - M_il M_kJ)/J with M = J F^-T (as PETSc ex77). */
static void f0_pressure(KERNEL_ARGS, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal F[9], M[9]; const PetscReal pc = PetscRealPart(constants[C_PCAV]);
  grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  for (PetscInt i = 0; i < 3; i++) f0[i] = pc * (M[i * 3] * n[0] + M[i * 3 + 1] * n[1] + M[i * 3 + 2] * n[2]);
}
static void g1_pressure(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal F[9], M[9], m[3]; const PetscReal pc = PetscRealPart(constants[C_PCAV]);
  grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  const PetscReal J = F[0] * M[0] + F[1] * M[1] + F[2] * M[2];
  for (PetscInt i = 0; i < 3; i++) m[i] = M[i * 3] * n[0] + M[i * 3 + 1] * n[1] + M[i * 3 + 2] * n[2];
  for (PetscInt i = 0; i < 3; i++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++)
    g1[(i * 3 + k) * 3 + l] = pc / J * (m[i] * M[k * 3 + l] - M[i * 3 + l] * m[k]);
}
/* g3_uu = dP_ij/dF_kl at ((i*3 + k)*3 + j)*3 + l */
static void g3_uu(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  KernelData kd; PetscReal F[9], P[9], D[81];
  kernel_data(constants, aOff, a, &kd); grad_to_F(u_x, uOff_x, F);
  AS_Piola(&kd.m, kd.A, kd.f, kd.s, F, PetscRealPart(u[uOff[1]]), P, D);
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++)
    g3[((i * 3 + k) * 3 + j) * 3 + l] = D[((i * 3 + j) * 3 + k) * 3 + l];
}
/* g2_up = dP_ij/dp = -J F^-T_ij (incompressible only) */
static void g2_up(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscReal F[9], M[9]; grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  const PetscBool inc = (PetscBool)((MatLaw)(PetscInt)PetscRealPart(constants[C_LAW]) == MAT_HO);
  for (PetscInt i = 0; i < 9; i++) g2[i] = inc ? -M[i] : 0.0;
}
/* g1_pu = -J F^-T_kl (incompressible only) */
static void g1_pu(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal F[9], M[9]; grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  const PetscBool inc = (PetscBool)((MatLaw)(PetscInt)PetscRealPart(constants[C_LAW]) == MAT_HO);
  for (PetscInt i = 0; i < 9; i++) g1[i] = inc ? -M[i] : 0.0;
}
/* g0_pp = 1 for the compressible law (p = 0), 0 otherwise */
static void g0_pp(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = ((MatLaw)(PetscInt)PetscRealPart(constants[C_LAW]) == MAT_HO) ? 0.0 : 1.0;
}

static PetscErrorCode zero_fn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt i = 0; i < Nc; i++) u[i] = 0.0;
  return PETSC_SUCCESS;
}

/* ---------------- exact solutions ---------------- */
typedef struct { PetscReal Fa[9]; PetscReal p; } AffineExact;   /* u = (G - I) X, p const */
static PetscErrorCode exact_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const AffineExact *e = (const AffineExact *)ctx;
  for (PetscInt i = 0; i < 3; i++) {
    PetscReal c = 0.0;
    for (PetscInt j = 0; j < 3; j++) c += (e->Fa[i * 3 + j] - (i == j ? 1.0 : 0.0)) * x[j];
    u[i] = c;
  }
  return PETSC_SUCCESS;
}
static PetscErrorCode exact_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = ((const AffineExact *)ctx)->p;
  return PETSC_SUCCESS;
}

/* ---------------- LV fibers (prolate depth), as in pulse3d/lv3d_active_strain.py ---------------- */
typedef struct { PetscReal d, le, lp, a_endo, a_epi; } Prolate;
static void prolate_frame(const Prolate *g, const PetscReal x[], PetscReal f[3], PetscReal s[3])
{
  const PetscReal r1 = PetscSqrtReal(x[0] * x[0] + x[1] * x[1] + (x[2] - g->d) * (x[2] - g->d));
  const PetscReal r2 = PetscSqrtReal(x[0] * x[0] + x[1] * x[1] + (x[2] + g->d) * (x[2] + g->d));
  const PetscReal lam = PetscAcoshReal(PetscMax(1.0, (r1 + r2) / (2.0 * g->d)));
  const PetscReal mu = PetscAcosReal(PetscMin(1.0, PetscMax(-1.0, (r2 - r1) / (2.0 * g->d))));
  const PetscReal th = PetscAtan2Real(x[1], x[0]);
  const PetscReal ls = PetscMin(1.0, PetscMax(0.0, (lam - g->le) / (g->lp - g->le)));
  const PetscReal al = (g->a_endo * (1.0 - ls) + g->a_epi * ls) * PETSC_PI / 180.0;
  PetscReal el[3] = {PetscCoshReal(lam) * PetscSinReal(mu) * PetscCosReal(th), PetscCoshReal(lam) * PetscSinReal(mu) * PetscSinReal(th), PetscSinhReal(lam) * PetscCosReal(mu)};
  PetscReal em[3] = {PetscSinhReal(lam) * PetscCosReal(mu) * PetscCosReal(th), PetscSinhReal(lam) * PetscCosReal(mu) * PetscSinReal(th), -PetscCoshReal(lam) * PetscSinReal(mu)};
  const PetscReal et[3] = {-PetscSinReal(th), PetscCosReal(th), 0.0};
  PetscReal nl = PetscSqrtReal(el[0] * el[0] + el[1] * el[1] + el[2] * el[2]), nm = PetscSqrtReal(em[0] * em[0] + em[1] * em[1] + em[2] * em[2]);
  for (PetscInt i = 0; i < 3; i++) { el[i] /= nl; em[i] /= (nm > 0 ? nm : 1.0); f[i] = PetscCosReal(al) * et[i] + PetscSinReal(al) * em[i]; }
  PetscReal nf = PetscSqrtReal(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (PetscInt i = 0; i < 3; i++) { f[i] /= nf; s[i] = el[i]; }
}
static PetscErrorCode fiber_fn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal f[3], s[3]; prolate_frame((const Prolate *)ctx, x, f, s);
  for (PetscInt i = 0; i < 3; i++) u[i] = f[i];
  return PETSC_SUCCESS;
}
static PetscErrorCode sheet_fn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscReal f[3], s[3]; prolate_frame((const Prolate *)ctx, x, f, s);
  for (PetscInt i = 0; i < 3; i++) u[i] = s[i];
  return PETSC_SUCCESS;
}

/* ---------------- labels ---------------- */
static PetscErrorCode BoxExtents(DM dm, PetscReal lo[3], PetscReal hi[3])
{
  Vec coords; const PetscScalar *c; PetscInt n; PetscReal l[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL}, h[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm, &coords)); PetscCall(VecGetLocalSize(coords, &n)); PetscCall(VecGetArrayRead(coords, &c));
  for (PetscInt i = 0; i < n / 3; i++)
    for (PetscInt d = 0; d < 3; d++) { l[d] = PetscMin(l[d], PetscRealPart(c[3 * i + d])); h[d] = PetscMax(h[d], PetscRealPart(c[3 * i + d])); }
  PetscCall(VecRestoreArrayRead(coords, &c));
  PetscCallMPI(MPIU_Allreduce(l, lo, 3, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  PetscCallMPI(MPIU_Allreduce(h, hi, 3, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)dm)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* boundary faces with centroid on x_d = val -> label 'name' (value 1), closure completed */
static PetscErrorCode LabelPlaneFaces(DM dm, const char name[], PetscInt d, PetscReal val)
{
  PetscInt fStart, fEnd; DMLabel lab;
  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, name)); PetscCall(DMGetLabel(dm, name, &lab));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt f = fStart; f < fEnd; f++) {
    PetscInt support; PetscReal vol, cen[3], nrm[3];
    PetscCall(DMPlexGetSupportSize(dm, f, &support));
    if (support != 1) continue;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &vol, cen, nrm));
    if (PetscAbsReal(cen[d] - val) < 1e-9 * (1.0 + PetscAbsReal(val))) PetscCall(DMLabelSetValue(lab, f, 1));
  }
  PetscCall(DMPlexLabelComplete(dm, lab));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* label 'name' = closure of the "Face Sets" stratum 'value' (gmsh physical tag) */
static PetscErrorCode LabelFromFaceSets(DM dm, const char name[], PetscInt value)
{
  DMLabel fs, lab; IS is; const PetscInt *pts; PetscInt n = 0;
  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &fs));
  PetscCheck(fs, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "mesh has no 'Face Sets' label (gmsh physical surfaces)");
  PetscCall(DMCreateLabel(dm, name)); PetscCall(DMGetLabel(dm, name, &lab));
  PetscCall(DMLabelGetStratumIS(fs, value, &is));
  if (is) {
    PetscCall(ISGetLocalSize(is, &n)); PetscCall(ISGetIndices(is, &pts));
    for (PetscInt i = 0; i < n; i++) PetscCall(DMLabelSetValue(lab, pts[i], 1));
    PetscCall(ISRestoreIndices(is, &pts)); PetscCall(ISDestroy(&is));
  }
  PetscCall(DMPlexLabelComplete(dm, lab));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* label 'name' = the mesh vertex at point P (within tol), on every rank that has it */
static PetscErrorCode LabelVertexAt(DM dm, const char name[], const PetscReal P[3], PetscReal tol, PetscInt *nfound)
{
  PetscInt vStart, vEnd, found = 0; DMLabel lab; Vec coords; PetscSection cs; const PetscScalar *c; DM cdm;
  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, name)); PetscCall(DMGetLabel(dm, name, &lab));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetCoordinateDM(dm, &cdm)); PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMGetCoordinatesLocal(dm, &coords)); PetscCall(VecGetArrayRead(coords, &c));
  for (PetscInt v = vStart; v < vEnd; v++) {
    PetscInt off; PetscCall(PetscSectionGetOffset(cs, v, &off));
    const PetscReal dd = PetscSqrtReal(PetscSqr(PetscRealPart(c[off]) - P[0]) + PetscSqr(PetscRealPart(c[off + 1]) - P[1]) + PetscSqr(PetscRealPart(c[off + 2]) - P[2]));
    if (dd < tol) { PetscCall(DMLabelSetValue(lab, v, 1)); found++; }
  }
  PetscCall(VecRestoreArrayRead(coords, &c));
  PetscCallMPI(MPIU_Allreduce(&found, nfound, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* vertex positions and displacements (owned vertices) -> text file "x y z ux uy uz" */
static PetscErrorCode WriteVertexDisplacement(DM dm, Vec U, const char fname[])
{
  Vec Ul, coords; PetscSection s, cs; DM cdm; const PetscScalar *u, *c; PetscInt vStart, vEnd; FILE *fp;
  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(dm, &Ul)); PetscCall(VecSet(Ul, 0.0));
  PetscCall(DMGlobalToLocal(dm, U, INSERT_VALUES, Ul));
  PetscCall(DMGetLocalSection(dm, &s)); PetscCall(DMGetCoordinateDM(dm, &cdm)); PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(Ul, &u)); PetscCall(VecGetArrayRead(coords, &c));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, fname, "w", &fp));
  for (PetscInt v = vStart; v < vEnd; v++) {
    PetscInt gs, ge, off, coff;
    PetscCall(DMPlexGetPointGlobal(dm, v, &gs, &ge));
    if (gs < 0) continue;                                  /* not owned */
    PetscCall(PetscSectionGetFieldOffset(s, v, 0, &off)); PetscCall(PetscSectionGetOffset(cs, v, &coff));
    PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD, fp, "%.10e %.10e %.10e %.10e %.10e %.10e\n",
                                       (double)PetscRealPart(c[coff]), (double)PetscRealPart(c[coff + 1]), (double)PetscRealPart(c[coff + 2]),
                                       (double)PetscRealPart(u[off]), (double)PetscRealPart(u[off + 1]), (double)PetscRealPart(u[off + 2])));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, fp));
  PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));
  PetscCall(VecRestoreArrayRead(Ul, &u)); PetscCall(VecRestoreArrayRead(coords, &c));
  PetscCall(DMRestoreLocalVector(dm, &Ul));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ParaView output: displacement and pressure interpolated to P1 (their vertex values) and written as VTU by PETSc's
 * VTK viewer (parallel). The P1 copy of the DM is made once. */
/* -mean_bc: in-plane rigid modes at the dofs (translations; rotation about z with the CURRENT coordinates x + u, the
   tangent's null vector of a deformed body), and the regularising spring of the preconditioner matrix */
static void mode_tx(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = 1.0; f[1] = 0.0; f[2] = 0.0; }
static void mode_ty(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = 0.0; f[1] = 1.0; f[2] = 0.0; }
static void mode_rz(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = -(x[1] + PetscRealPart(u[uOff[0] + 1])); f[1] = x[0] + PetscRealPart(u[uOff[0]]); f[2] = 0.0; }
static void mode_rz0(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = -x[1]; f[1] = x[0]; f[2] = 0.0; }
static void mode_X(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = x[0]; f[1] = x[1]; f[2] = 0.0; }
static void mode_zero(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{ f[0] = 0.0; }
static void g0_uu_reg(KERNEL_ARGS, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt i = 0; i < 9; i++) g0[i] = 0.0;
  for (PetscInt i = 0; i < 3; i++) g0[i * 4] = constants[C_REG];
}

static void copy_u(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  for (PetscInt i = 0; i < 3; i++) f[i] = u[uOff[0] + i];
}
static void copy_p(KERNEL_ARGS, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  f[0] = u[uOff[1]];
}
static PetscErrorCode WriteVTU(DM dm, Vec U, const char fname[])
{
  /* P1 copy of (u, p) on the same mesh: the VTK viewer writes vertex fields only, and refuses two grids per file */
  static DM dmV = NULL;
  void (*funcs[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {copy_u, copy_p};
  Vec V; PetscViewer vw;
  PetscFunctionBeginUser;
  if (!dmV) {
    const char *names[2] = {"displacement", "pressure"};
    PetscSection sec;
    PetscCall(DMClone(dm, &dmV));
    for (PetscInt f = 0; f < 2; f++) {
      PetscFE fe;
      PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, f == 0 ? 3 : 1, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
      PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
      PetscCall(DMSetField(dmV, f, NULL, (PetscObject)fe)); PetscCall(PetscFEDestroy(&fe));
    }
    PetscCall(DMCreateDS(dmV));
    PetscCall(DMGetLocalSection(dmV, &sec));
    PetscCall(PetscSectionSetComponentName(sec, 1, 0, "p"));     /* scalar arrays are named <vec><field>.<component> */
  }
  PetscCall(DMGetGlobalVector(dmV, &V));
  PetscCall(PetscObjectSetName((PetscObject)V, ""));           /* arrays "displacement" and "pressure.p" */
  PetscCall(DMProjectField(dmV, 0.0, U, funcs, INSERT_VALUES, V));
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &vw));
  PetscCall(VecView(V, vw));
  PetscCall(PetscViewerDestroy(&vw));
  PetscCall(DMRestoreGlobalVector(dmV, &V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ---------------- -mean_bc: null average in-plane displacement and rotation (Pezzuto & Ambrosi 2014) ----------------
 * No pins: the base u_z = 0 leaves the in-plane translations and the rotation about z free (3 rigid modes). The
 * Lagrange multipliers of the 3 average conditions act as a uniform in-plane force and torque, i.e. the residual is
 * stationary only up to its rigid-mode components: those are projected out of the residual (MeanBCFunction), the modes
 * are the Jacobian's null space (MeanBCJacobian; KSP removes them, so every Newton update and hence u has zero mean
 * displacement / rotation), and the preconditioner matrix adds a tiny spring C_REG (LU of the singular Jacobian fails).
 * Means are dof sums (P2 nodes), the rotation one with the current coordinates. */
typedef struct { DM dm; Vec tx, ty, rz0, PX, q[3]; PetscInt nx; } MeanBC;
static MeanBC *mean_bc_ctx = NULL;                          /* set for -mean_bc: solves are re-gauged (MeanBCGauge_) */

static PetscErrorCode MeanBCModes_(MeanBC *mb, Vec X)
{
  void (*fr[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_rz, mode_zero};
  PetscScalar d;
  PetscFunctionBeginUser;
  PetscCall(VecCopy(mb->tx, mb->q[0])); PetscCall(VecCopy(mb->ty, mb->q[1]));
  PetscCall(DMProjectField(mb->dm, 0.0, X, fr, INSERT_VALUES, mb->q[2]));
  for (PetscInt i = 0; i < 3; i++) {                        /* Gram-Schmidt */
    for (PetscInt j = 0; j < i; j++) { PetscCall(VecDot(mb->q[i], mb->q[j], &d)); PetscCall(VecAXPY(mb->q[i], -d, mb->q[j])); }
    PetscCall(VecNormalize(mb->q[i], NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MeanBCLocal_(DM dm, Vec X, Vec *Xl)
{
  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(dm, Xl)); PetscCall(VecSet(*Xl, 0.0));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, *Xl, 0.0, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, *Xl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MeanBCFunction(SNES snes, Vec X, Vec F, void *ctx)
{
  MeanBC *mb = (MeanBC *)ctx; Vec Xl, Fl; PetscScalar d;
  PetscFunctionBeginUser;
  PetscCall(MeanBCLocal_(mb->dm, X, &Xl)); PetscCall(DMGetLocalVector(mb->dm, &Fl)); PetscCall(VecSet(Fl, 0.0));
  PetscCall(DMPlexSNESComputeResidualFEM(mb->dm, Xl, Fl, NULL));
  PetscCall(VecSet(F, 0.0)); PetscCall(DMLocalToGlobal(mb->dm, Fl, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(mb->dm, &Xl)); PetscCall(DMRestoreLocalVector(mb->dm, &Fl));
  PetscCall(MeanBCModes_(mb, X));
  for (PetscInt i = 0; i < 3; i++) { PetscCall(VecDot(F, mb->q[i], &d)); PetscCall(VecAXPY(F, -d, mb->q[i])); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MeanBCJacobian(SNES snes, Vec X, Mat J, Mat Jp, void *ctx)
{
  MeanBC *mb = (MeanBC *)ctx; Vec Xl; MatNullSpace nsp;
  PetscFunctionBeginUser;
  PetscCall(MeanBCLocal_(mb->dm, X, &Xl));
  PetscCall(DMPlexSNESComputeJacobianFEM(mb->dm, Xl, J, Jp, NULL));
  PetscCall(DMRestoreLocalVector(mb->dm, &Xl));
  PetscCall(MeanBCModes_(mb, X));
  {                                                         /* the null space locks its vectors read-only: own copies */
    Vec v[3];
    for (PetscInt i = 0; i < 3; i++) { PetscCall(VecDuplicate(mb->q[i], &v[i])); PetscCall(VecCopy(mb->q[i], v[i])); }
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)J), PETSC_FALSE, 3, v, &nsp));
    for (PetscInt i = 0; i < 3; i++) PetscCall(VecDestroy(&v[i]));
  }
  PetscCall(MatSetNullSpace(J, nsp)); PetscCall(MatNullSpaceDestroy(&nsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* after a converged solve: the finite in-plane rigid motion x' = Q(phi) x + t (x = X + u) that makes the mean in-plane
   displacement and the mean rotation sum_a (X_a x u_a)_z exactly zero (the Newton updates are not exactly orthogonal to
   the rigid modes: the regularised LU amplifies them). R(Q x + t) = Q R(x) here (base u_z = 0, follower pressure,
   material fibers), so equilibrium is unchanged. A = X + u and A_perp = e_z x A on the x, y displacement dofs. */
static PetscErrorCode MeanBCGauge_(MeanBC *mb, Vec U)
{
  void (*fr[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_rz, mode_zero};
  Vec A, Ap, W, Uxy; PetscScalar d; PetscReal phi = 0.0, tx = 0.0, ty = 0.0;
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(U, &A)); PetscCall(VecDuplicate(U, &Ap)); PetscCall(VecDuplicate(U, &W)); PetscCall(VecDuplicate(U, &Uxy));
  PetscCall(VecWAXPY(W, 1.0, mb->tx, mb->ty)); PetscCall(VecPointwiseMult(Uxy, U, W));   /* u on the x, y dofs */
  PetscCall(VecWAXPY(A, 1.0, mb->PX, Uxy));
  PetscCall(DMProjectField(mb->dm, 0.0, U, fr, INSERT_VALUES, Ap));
  /* c(phi) = rz0 . u'(phi) with t(phi) from the two translation conditions; Newton on phi (numerical derivative) */
  for (PetscInt it = 0; it < 20; it++) {
    PetscReal c[2];
    for (PetscInt j = 0; j < 2; j++) {
      const PetscReal ph = phi + (j ? 1e-7 : 0.0);
      PetscCall(VecAXPBYPCZ(W, PetscCosReal(ph), PetscSinReal(ph), 0.0, A, Ap));            /* Q(ph) x (in-plane) */
      PetscCall(VecAXPY(W, -1.0, mb->PX));                                                 /* - X */
      PetscCall(VecDot(W, mb->tx, &d)); tx = -PetscRealPart(d) / mb->nx;
      PetscCall(VecDot(W, mb->ty, &d)); ty = -PetscRealPart(d) / mb->nx;
      PetscCall(VecAXPY(W, tx, mb->tx)); PetscCall(VecAXPY(W, ty, mb->ty));
      PetscCall(VecDot(W, mb->rz0, &d)); c[j] = PetscRealPart(d);
    }
    if (PetscAbsReal(c[0]) < 1e-14 * mb->nx) break;
    phi -= c[0] * 1e-7 / (c[1] - c[0]);
  }
  PetscCall(VecAXPBYPCZ(W, PetscCosReal(phi), PetscSinReal(phi), 0.0, A, Ap));
  PetscCall(VecAXPY(W, -1.0, mb->PX));
  PetscCall(VecDot(W, mb->tx, &d)); tx = -PetscRealPart(d) / mb->nx; PetscCall(VecDot(W, mb->ty, &d)); ty = -PetscRealPart(d) / mb->nx;
  PetscCall(VecAXPY(W, tx, mb->tx)); PetscCall(VecAXPY(W, ty, mb->ty));
  PetscCall(VecAXPY(U, -1.0, Uxy)); PetscCall(VecAXPY(U, 1.0, W));                          /* new u on the x, y dofs */
  PetscCall(VecDestroy(&A)); PetscCall(VecDestroy(&Ap)); PetscCall(VecDestroy(&W)); PetscCall(VecDestroy(&Uxy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* mean in-plane displacement (dof means of u_x, u_y) and mean rotation about z (reference coordinates) of U */
static PetscErrorCode MeanBCReport_(MeanBC *mb, Vec U, PetscReal m[3])
{
  PetscScalar a, b, c, rr;
  PetscFunctionBeginUser;
  PetscCall(VecDot(U, mb->tx, &a)); PetscCall(VecDot(U, mb->ty, &b)); PetscCall(VecDot(U, mb->rz0, &c)); PetscCall(VecDot(mb->rz0, mb->rz0, &rr));
  m[0] = PetscRealPart(a) / mb->nx; m[1] = PetscRealPart(b) / mb->nx; m[2] = PetscRealPart(c) / PetscRealPart(rr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* reference position relative to x0 = (0, 0, z_b) on the displacement field (exact in P2 on straight tets) */
static PetscErrorCode xrel_fn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt i = 0; i < 3; i++) u[i] = x[i];
  u[2] -= *(const PetscReal *)ctx;
  return PETSC_SUCCESS;
}

/* cavity volume V = -1/3 int (x - x0) . J F^-T N dA over the endocardium (x0 = (0, 0, z_b) on the base plane; the endo
   rim keeps u_z = 0, so the flat cap closing the cavity adds nothing), and optionally b = dR/dp_c (the pressure load at
   p_c = 1; R is linear in p_c; dV/du = -b). With the local load b_a = int N_a J F^-T N dA over ALL dofs (PETSc assembles
   local residuals with ADD_ALL_VALUES), V = -1/3 sum_a (x - x0)_a . b_a exactly, since x - x0 = X + u - x0 is in the P2
   space. Rank sums count each face once (no overlap). */
static PetscErrorCode CavityVolume(DM dm, PetscDS ds, Vec U, const PetscScalar c[], PetscReal *V, Vec b)
{
  static Vec Xrel = NULL;
  Vec Ul, R0, R1; PetscScalar cc[C_N], dot; PetscReal loc;
  PetscFunctionBeginUser;
  if (!Xrel) {
    PetscReal zb = PetscRealPart(c[C_ZB]);
    PetscErrorCode (*funcs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {xrel_fn, zero_fn};
    void *ctxs[2] = {&zb, NULL};
    PetscCall(DMCreateLocalVector(dm, &Xrel));
    PetscCall(DMProjectFunctionLocal(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, Xrel));
  }
  PetscCall(DMGetLocalVector(dm, &Ul)); PetscCall(DMGetLocalVector(dm, &R0)); PetscCall(DMGetLocalVector(dm, &R1));
  PetscCall(VecSet(Ul, 0.0));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, Ul, 0.0, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocal(dm, U, INSERT_VALUES, Ul));
  for (PetscInt i = 0; i < C_N; i++) cc[i] = c[i];
  cc[C_PCAV] = 0.0; PetscCall(PetscDSSetConstants(ds, C_N, cc));
  PetscCall(VecSet(R0, 0.0)); PetscCall(DMPlexSNESComputeResidualFEM(dm, Ul, R0, NULL));
  cc[C_PCAV] = 1.0; PetscCall(PetscDSSetConstants(ds, C_N, cc));
  PetscCall(VecSet(R1, 0.0)); PetscCall(DMPlexSNESComputeResidualFEM(dm, Ul, R1, NULL));
  PetscCall(PetscDSSetConstants(ds, C_N, (PetscScalar *)c));
  PetscCall(VecAXPY(R1, -1.0, R0));                          /* local b (all dofs) */
  PetscCall(VecAXPY(Ul, 1.0, Xrel));                         /* x - x0 (the pressure-field entries of b are 0) */
  PetscCall(VecDot(Ul, R1, &dot)); loc = -PetscRealPart(dot) / 3.0;
  PetscCallMPI(MPIU_Allreduce(&loc, V, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  if (b) { PetscCall(VecSet(b, 0.0)); PetscCall(DMLocalToGlobal(dm, R1, ADD_VALUES, b)); }
  PetscCall(DMRestoreLocalVector(dm, &Ul)); PetscCall(DMRestoreLocalVector(dm, &R0)); PetscCall(DMRestoreLocalVector(dm, &R1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* isovolumic solve at the constants c: find p_c = c[C_PCAV] with V(u(p_c)) = Vstar. Outer Newton on p_c around full SNES
   solves: b = dR/dp_c (the pressure load at p_c = 1, R is linear in p_c), dV/du = -b, so with K z = b (the last SNES
   factorization) dV/dp_c = b . z. */
static PetscErrorCode SolveVolume(SNES snes, PetscDS ds, Vec U, PetscScalar c[], PetscReal Vstar, PetscReal vtol, PetscInt *its_out, PetscBool *ok)
{
  DM dm; Vec b, z; KSP ksp; PetscReal V = 0.0, dVdp;
  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(VecDuplicate(U, &b)); PetscCall(VecDuplicate(U, &z));
  *ok = PETSC_FALSE;
  for (PetscInt it = 0; it < 25; it++) {
    SNESConvergedReason r; PetscInt its;
    PetscCall(PetscDSSetConstants(ds, C_N, c));
    PetscCall(SNESSolve(snes, NULL, U));
    PetscCall(SNESGetConvergedReason(snes, &r)); PetscCall(SNESGetIterationNumber(snes, &its));
    *its_out += its;
    if (r <= 0) break;
    if (mean_bc_ctx) PetscCall(MeanBCGauge_(mean_bc_ctx, U));
    PetscCall(CavityVolume(dm, ds, U, c, &V, b));
    if (PetscAbsReal(V - Vstar) <= vtol * Vstar) { *ok = PETSC_TRUE; break; }
    const PetscReal pc = PetscRealPart(c[C_PCAV]);
    PetscCall(SNESGetKSP(snes, &ksp)); PetscCall(KSPSolve(ksp, b, z));
    PetscCall(VecDot(b, z, &dVdp));
    if (!(dVdp > 0.0)) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  isovolumic: dV/dp = %g <= 0\n", (double)dVdp)); break; }
    c[C_PCAV] = pc + (Vstar - V) / dVdp;
  }
  if (!*ok) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  isovolumic solve failed: V %.6e target %.6e, p_c %.1f Pa\n", (double)V, (double)Vstar, (double)PetscRealPart(c[C_PCAV])));
  PetscCall(VecDestroy(&b)); PetscCall(VecDestroy(&z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* one load step, with up to 'levels' halvings on failure (substeps interpolate the constants linearly). Vstar > 0: the
   cavity volume is held at Vstar and p_c is solved for (starting from c0's p_c) instead of interpolated. */
static PetscErrorCode SolveStep(SNES snes, PetscDS ds, Vec U, PetscScalar c0[], const PetscScalar c1[], PetscReal Vstar, PetscReal vtol, PetscInt levels, PetscInt *its_out, PetscBool *ok)
{
  Vec U0; PetscScalar c[C_N];
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(U, &U0)); PetscCall(VecCopy(U, U0));
  *ok = PETSC_FALSE; *its_out = 0;
  for (PetscInt lev = 0; lev <= levels && !*ok; lev++) {
    const PetscInt n = 1 << lev; PetscBool good = PETSC_TRUE;
    PetscScalar pc = c0[C_PCAV];
    PetscCall(VecCopy(U0, U));
    for (PetscInt j = 1; j <= n && good; j++) {
      for (PetscInt i = 0; i < C_N; i++) c[i] = c0[i] + (c1[i] - c0[i]) * (PetscReal)j / n;
      if (Vstar > 0.0) {
        c[C_PCAV] = pc;
        PetscCall(SolveVolume(snes, ds, U, c, Vstar, vtol, its_out, &good));
        pc = c[C_PCAV];
      } else {
        SNESConvergedReason r; PetscInt its;
        PetscCall(PetscDSSetConstants(ds, C_N, c));
        PetscCall(SNESSolve(snes, NULL, U));
        PetscCall(SNESGetConvergedReason(snes, &r)); PetscCall(SNESGetIterationNumber(snes, &its));
        *its_out += its; good = (PetscBool)(r > 0);
        if (good && mean_bc_ctx) PetscCall(MeanBCGauge_(mean_bc_ctx, U));
      }
    }
    if (good) *ok = PETSC_TRUE;
    else if (lev < levels) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  step failed with %d substep(s), retrying smaller\n", (int)n));
  }
  if (*ok) for (PetscInt i = 0; i < C_N; i++) c0[i] = c[i];
  else { PetscCall(VecCopy(U0, U)); PetscCall(PetscDSSetConstants(ds, C_N, c0)); }
  PetscCall(VecDestroy(&U0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -test_jacobian_dir: the follower-load tangent alone. With D(.) = (.)(p_c) - (.)(0): D(J v) against the central difference
   D(R(u + h v) - R(u - h v)) / 2h along a random v; the elastic part cancels exactly in D (the full -snes_test_jacobian
   is dense and too slow on LV meshes) */
static PetscErrorCode TestJacobianDir(SNES snes, PetscDS ds, Vec U, const PetscScalar c[], PetscReal pc, const char *where)
{
  Mat J; Vec v, Jv[2], FD[2][4], R, W; PetscRandom rnd; PetscReal nu, nv, err, nJv; PetscScalar cs[C_N];
  DM dm;
  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(VecDuplicate(U, &v)); PetscCall(VecDuplicate(U, &R)); PetscCall(VecDuplicate(U, &W));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd)); PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
  PetscCall(VecSetRandom(v, rnd)); PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(VecNorm(U, NORM_INFINITY, &nu)); PetscCall(VecNorm(v, NORM_INFINITY, &nv));
  PetscCall(VecScale(v, 1e-3 * PetscMax(nu, 1e-6) / nv));
  for (PetscInt i = 0; i < C_N; i++) cs[i] = c[i];
  for (PetscInt q = 0; q < 2; q++) {                       /* q = 0: p_c = 0, q = 1: p_c = pc */
    cs[C_PCAV] = q ? pc : 0.0; PetscCall(PetscDSSetConstants(ds, C_N, cs));
    PetscCall(VecDuplicate(U, &Jv[q]));
    PetscCall(SNESComputeJacobian(snes, U, J, J)); PetscCall(MatMult(J, v, Jv[q]));
    for (PetscInt e = 0; e < 4; e++) {
      const PetscReal h = PetscPowReal(10.0, -2.0 - 2.0 * e);
      PetscCall(VecDuplicate(U, &FD[q][e]));
      PetscCall(VecWAXPY(W, h, v, U)); PetscCall(SNESComputeFunction(snes, W, FD[q][e]));
      PetscCall(VecWAXPY(W, -h, v, U)); PetscCall(SNESComputeFunction(snes, W, R));
      PetscCall(VecAXPY(FD[q][e], -1.0, R)); PetscCall(VecScale(FD[q][e], 0.5 / h));
    }
  }
  PetscCall(VecAXPY(Jv[1], -1.0, Jv[0])); PetscCall(VecNorm(Jv[1], NORM_2, &nJv));
  for (PetscInt e = 0; e < 4; e++) {
    PetscCall(VecAXPY(FD[1][e], -1.0, FD[0][e])); PetscCall(VecAXPY(FD[1][e], -1.0, Jv[1])); PetscCall(VecNorm(FD[1][e], NORM_2, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test_jacobian_dir %s] follower load, h %.0e: |D(Jv) - D(FD)| / |D(Jv)| = %.3e (|D(Jv)| %.3e)\n",
                          where, (double)PetscPowReal(10.0, -2.0 - 2.0 * e), (double)(err / nJv), (double)nJv));
  }
  PetscCall(PetscDSSetConstants(ds, C_N, (PetscScalar *)c));
  PetscCall(MatDestroy(&J)); PetscCall(VecDestroy(&v)); PetscCall(VecDestroy(&R)); PetscCall(VecDestroy(&W));
  for (PetscInt q = 0; q < 2; q++) { PetscCall(VecDestroy(&Jv[q])); for (PetscInt e = 0; e < 4; e++) PetscCall(VecDestroy(&FD[q][e])); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* vertex file u_stepNNNN.txt and VTU solid3d_NNNN.vtu for step k */
static PetscErrorCode WriteStep(DM dm, Vec U, const char out_dir[], PetscInt k, PetscBool vertex, PetscBool vtk)
{
  char fn[PETSC_MAX_PATH_LEN];
  PetscFunctionBeginUser;
  if (vertex) { PetscCall(PetscSNPrintf(fn, sizeof(fn), "%s/u_step%04d.txt", out_dir, (int)k)); PetscCall(WriteVertexDisplacement(dm, U, fn)); }
  if (vtk) { PetscCall(PetscSNPrintf(fn, sizeof(fn), "%s/solid3d_%04d.vtu", out_dir, (int)k)); PetscCall(WriteVTU(dm, U, fn)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the 3D solver: called by the biofem executables for -model solid3d (main.c), or by the standalone main below */
PetscErrorCode Solid3dRun(void)
{
  DM dm; SNES snes; Vec U; PetscFE fe[2]; PetscDS ds;
  Problem prob = PROB_CUBE;
  PetscReal gmax = 0.2, traction = 9.0e5, nLx = 1.2, nLy = 0.9, p_ed = 0.0, p_ao = 1.0e4, vtol = 1e-8;
  PetscInt steps = 4, out_every = 5, levels = 3, vtk_every = -1, fill_steps = 20;
  PetscBool test_jdir = PETSC_FALSE, mean_bc = PETSC_FALSE;
  PetscReal mean_bc_reg = 1e-8;
  char out_dir[PETSC_MAX_PATH_LEN] = "solid3d_out";
  MatParams m = {MAT_HO, 236.2, 10.810, 20037.0, 14.154, 3724.5, 5.1645, 410.8, 11.300, PETSC_TRUE, PETSC_TRUE, 1.5e6, 7.445e7};
  Prolate geo = {0.0291, 0.557, 1.00, -60.0, 60.0};
  PetscReal mu_base_deg = 113.0;
  PetscScalar c0[C_N], c1[C_N];

  PetscFunctionBeginUser;
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "biofem-solid3d options", "");
  PetscCall(PetscOptionsEnum("-problem", "cube | plate | lv", "", ProblemNames, (PetscEnum)prob, (PetscEnum *)&prob, NULL));
  PetscCall(PetscOptionsReal("-gamma", "peak activation", "", gmax, &gmax, NULL));
  PetscCall(PetscOptionsInt("-steps", "load steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-substep_levels", "halvings tried on a failed step", "", levels, &levels, NULL));
  PetscCall(PetscOptionsInt("-out_every", "lv: write vertex displacements every N steps", "", out_every, &out_every, NULL));
  PetscCall(PetscOptionsString("-out_dir", "output directory", "", out_dir, out_dir, sizeof(out_dir), NULL));
  PetscCall(PetscOptionsInt("-vtk_every", "write <out_dir>/solid3d_NNNN.vtu (P1 displacement, pressure) every N steps (default: -out_every for lv, off otherwise; 0 = off)", "", vtk_every, &vtk_every, NULL));
  PetscCall(PetscOptionsBool("-ho_tension_only", "", "", m.tension_only, &m.tension_only, NULL));
  PetscCall(PetscOptionsBool("-ho_sheet", "", "", m.sheet, &m.sheet, NULL));
  PetscCall(PetscOptionsReal("-mu", "neo-Hookean mu (plate)", "", m.mu, &m.mu, NULL));
  PetscCall(PetscOptionsReal("-K", "neo-Hookean K (plate)", "", m.K, &m.K, NULL));
  PetscCall(PetscOptionsReal("-traction", "plate: traction on x = max [Pa]", "", traction, &traction, NULL));
  PetscCall(PetscOptionsReal("-nitti_Lx", "plate: Nitti patch Lx", "", nLx, &nLx, NULL));
  PetscCall(PetscOptionsReal("-nitti_Ly", "plate: Nitti patch Ly", "", nLy, &nLy, NULL));
  PetscCall(PetscOptionsReal("-prolate_d", "lv: focal length of the prolate geometry [m] (must match the mesh)", "", geo.d, &geo.d, NULL));
  PetscCall(PetscOptionsReal("-lam_endo", "lv: prolate lambda of the endocardium (must match the mesh)", "", geo.le, &geo.le, NULL));
  PetscCall(PetscOptionsReal("-lam_epi", "lv: prolate lambda of the epicardium (must match the mesh)", "", geo.lp, &geo.lp, NULL));
  PetscCall(PetscOptionsReal("-mu_base_deg", "lv: prolate mu of the base cut [deg] (must match the mesh)", "", mu_base_deg, &mu_base_deg, NULL));
  PetscCall(PetscOptionsReal("-alpha_endo", "lv: helix angle at the endocardium [deg]", "", geo.a_endo, &geo.a_endo, NULL));
  PetscCall(PetscOptionsReal("-alpha_epi", "lv: helix angle at the epicardium [deg]", "", geo.a_epi, &geo.a_epi, NULL));
  PetscCall(PetscOptionsReal("-p_ed", "lv: end-diastolic cavity pressure [Pa]; > 0 runs the PV loop (Track P)", "", p_ed, &p_ed, NULL));
  PetscCall(PetscOptionsReal("-p_ao", "lv: aortic pressure during ejection [Pa]", "", p_ao, &p_ao, NULL));
  PetscCall(PetscOptionsInt("-fill_steps", "lv: pressure steps of the passive filling 0 -> p_ed", "", fill_steps, &fill_steps, NULL));
  PetscCall(PetscOptionsBool("-mean_bc", "lv: null average in-plane displacement and rotation instead of the rim pins", "", mean_bc, &mean_bc, NULL));
  PetscCall(PetscOptionsReal("-mean_bc_reg", "lv -mean_bc: spring of the preconditioner matrix, relative to the HO a / d^2", "", mean_bc_reg, &mean_bc_reg, NULL));
  PetscCall(PetscOptionsBool("-test_jacobian_dir", "check J v against a central difference of R (lv: unloaded and at end-diastole)", "", test_jdir, &test_jdir, NULL));
  PetscCall(PetscOptionsReal("-volume_rtol", "lv: isovolumic phases, |V - V*| <= rtol V*", "", vtol, &vtol, NULL));
  PetscOptionsEnd();
  if (prob == PROB_PLATE) m.law = MAT_NH;
  if (vtk_every < 0) vtk_every = (prob == PROB_LV) ? out_every : 0;

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_dim", "3"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* fields: P2 displacement, P1 pressure, quadrature degree 4 */
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, 3, PETSC_TRUE, 2, 4, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "displacement"));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, 1, PETSC_TRUE, 1, 4, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject)fe[1], "pressure"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe[1]));

  /* boundary conditions */
  PetscReal lo[3], hi[3];
  PetscCall(BoxExtents(dm, lo, hi));
  if (prob == PROB_CUBE || prob == PROB_PLATE) {
    for (PetscInt d = 0; d < 3; d++) {
      char name[8]; PetscCall(PetscSNPrintf(name, sizeof(name), "sym%d", (int)d));
      PetscCall(LabelPlaneFaces(dm, name, d, lo[d]));
    }
    if (prob == PROB_PLATE) PetscCall(LabelPlaneFaces(dm, "loaded", 0, hi[0]));
  } else {
    const PetscReal mub = mu_base_deg * PETSC_PI / 180.0, zr = geo.d * PetscCoshReal(geo.le) * PetscCosReal(mub);
    const PetscReal Apt[3] = {geo.d * PetscSinhReal(geo.le) * PetscSinReal(mub), 0.0, zr};
    const PetscReal Bpt[3] = {geo.d * PetscSinhReal(geo.lp) * PetscSinReal(mub), 0.0, geo.d * PetscCoshReal(geo.lp) * PetscCosReal(mub)};
    PetscInt nA, nB;
    PetscCall(LabelFromFaceSets(dm, "base", 3));
    PetscCall(LabelFromFaceSets(dm, "endo", 1));
    PetscCall(LabelVertexAt(dm, "pinA", Apt, 1e-8, &nA));
    PetscCall(LabelVertexAt(dm, "pinB", Bpt, 1e-8, &nB));
    PetscCheck(nA && nB, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "rim pin vertices not found (A %d, B %d)", (int)nA, (int)nB);
  }
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, f1_u));
  PetscCall(PetscDSSetResidual(ds, 1, f0_p, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_up, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_pp, NULL, NULL, NULL));
  if (prob == PROB_LV && mean_bc) {                         /* preconditioner matrix = Jacobian + a tiny spring */
    PetscCall(PetscDSSetJacobianPreconditioner(ds, 0, 0, g0_uu_reg, NULL, NULL, g3_uu));
    PetscCall(PetscDSSetJacobianPreconditioner(ds, 0, 1, NULL, NULL, g2_up, NULL));
    PetscCall(PetscDSSetJacobianPreconditioner(ds, 1, 0, NULL, g1_pu, NULL, NULL));
    PetscCall(PetscDSSetJacobianPreconditioner(ds, 1, 1, g0_pp, NULL, NULL, NULL));
  }
  if (prob == PROB_CUBE || prob == PROB_PLATE) {
    for (PetscInt d = 0; d < 3; d++) {
      char name[8]; DMLabel lab; PetscInt one = 1, comp = d;
      PetscCall(PetscSNPrintf(name, sizeof(name), "sym%d", (int)d)); PetscCall(DMGetLabel(dm, name, &lab));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, name, lab, 1, &one, 0, 1, &comp, (void (*)(void))zero_fn, NULL, NULL, NULL));
    }
    if (prob == PROB_PLATE) {
      DMLabel lab; PetscInt one = 1; PetscInt comps[3] = {0, 1, 2};
      PetscCall(DMGetLabel(dm, "loaded", &lab));
      PetscInt bd; PetscWeakForm wf;
      PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "traction", lab, 1, &one, 0, 3, comps, NULL, NULL, NULL, &bd));
      /* attach the traction to THIS boundary's label (PetscDSSetBdResidual would act on every natural boundary face) */
      PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormSetIndexBdResidual(wf, lab, 1, 0, 0, 0, f0_traction, 0, NULL));
    }
  } else {
    DMLabel lab; PetscInt one = 1, cz = 2, cxy[2] = {0, 1}, cy = 1;
    PetscCall(DMGetLabel(dm, "base", &lab));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "base_uz", lab, 1, &one, 0, 1, &cz, (void (*)(void))zero_fn, NULL, NULL, NULL));
    if (!mean_bc) {                                          /* rim pins (default; the pulse reference uses them) */
      PetscCall(DMGetLabel(dm, "pinA", &lab));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "pinA", lab, 1, &one, 0, 2, cxy, (void (*)(void))zero_fn, NULL, NULL, NULL));
      PetscCall(DMGetLabel(dm, "pinB", &lab));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "pinB", lab, 1, &one, 0, 1, &cy, (void (*)(void))zero_fn, NULL, NULL, NULL));
    }
    if (p_ed > 0.0) {                                       /* cavity pressure on the endocardium */
      PetscInt bd, comps[3] = {0, 1, 2}; PetscWeakForm wf;
      PetscCall(DMGetLabel(dm, "endo", &lab));
      PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "cavity", lab, 1, &one, 0, 3, comps, NULL, NULL, NULL, &bd));
      PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormSetIndexBdResidual(wf, lab, 1, 0, 0, 0, f0_pressure, 0, NULL));
      PetscCall(PetscWeakFormSetIndexBdJacobian(wf, lab, 1, 0, 0, 0, 0, NULL, 0, g1_pressure, 0, NULL, 0, NULL));
      if (mean_bc) PetscCall(PetscWeakFormSetIndexBdJacobianPreconditioner(wf, lab, 1, 0, 0, 0, 0, NULL, 0, g1_pressure, 0, NULL, 0, NULL));
    }
  }

  /* fibers as an auxiliary P1 field (lv) */
  if (prob == PROB_LV) {
    DM dmAux; PetscFE fa[2]; Vec Aux;
    PetscCall(DMClone(dm, &dmAux));
    for (PetscInt i = 0; i < 2; i++) {
      PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, 3, PETSC_TRUE, 1, 4, &fa[i]));
      PetscCall(PetscFECopyQuadrature(fe[0], fa[i]));
      PetscCall(DMSetField(dmAux, i, NULL, (PetscObject)fa[i]));
    }
    PetscCall(DMCreateDS(dmAux));
    PetscCall(DMCreateLocalVector(dmAux, &Aux));
    PetscErrorCode (*funcs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {fiber_fn, sheet_fn};
    void *ctxs[2] = {&geo, &geo};
    PetscCall(DMProjectFunctionLocal(dmAux, 0.0, funcs, ctxs, INSERT_ALL_VALUES, Aux));
    PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, Aux));
    PetscCall(VecDestroy(&Aux)); PetscCall(PetscFEDestroy(&fa[0])); PetscCall(PetscFEDestroy(&fa[1])); PetscCall(DMDestroy(&dmAux));
  }

  /* constants at the start (gamma = 0 / no load) and per step */
  for (PetscInt i = 0; i < C_N; i++) c0[i] = 0.0;
  c0[C_LAW] = m.law; c0[C_A] = m.a; c0[C_B] = m.b; c0[C_AF] = m.af; c0[C_BF] = m.bf; c0[C_AS] = m.as; c0[C_BS] = m.bs;
  c0[C_AFS] = m.afs; c0[C_BFS] = m.bfs; c0[C_TENSION] = m.tension_only; c0[C_SHEET] = m.sheet; c0[C_MU] = m.mu; c0[C_K] = m.K;
  c0[C_FIBER_AUX] = (prob == PROB_LV); c0[C_FA_DIAG] = (prob == PROB_PLATE);
  c0[C_A1] = c0[C_A2] = c0[C_A3] = 1.0;
  c0[C_F0] = 1.0; c0[C_S0 + 2] = 1.0;                       /* uniform fiber e_x, sheet e_z (cube) */
  c0[C_ZB] = geo.d * PetscCoshReal(geo.le) * PetscCosReal(mu_base_deg * PETSC_PI / 180.0);   /* z of the endo rim */
  c0[C_REG] = mean_bc_reg * m.a / (geo.d * geo.d);
  PetscCall(PetscDSSetConstants(ds, C_N, c0));

  PetscCall(DMCreateGlobalVector(dm, &U)); PetscCall(PetscObjectSetName((PetscObject)U, "solution"));
  PetscCall(VecSet(U, 0.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes)); PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  MeanBC mb = {NULL, NULL, NULL, NULL, {NULL, NULL, NULL}, 0};
  Mat Jmb = NULL, Jpmb = NULL;
  if (prob == PROB_LV && mean_bc) {
    void (*ft[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_tx, mode_zero};
    void (*fy[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_ty, mode_zero};
    void (*fr[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_rz0, mode_zero};
    PetscScalar nx;
    mb.dm = dm;
    void (*fX[2])(KERNEL_ARGS, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {mode_X, mode_zero};
    PetscCall(VecDuplicate(U, &mb.tx)); PetscCall(VecDuplicate(U, &mb.ty)); PetscCall(VecDuplicate(U, &mb.rz0)); PetscCall(VecDuplicate(U, &mb.PX));
    PetscCall(DMProjectField(dm, 0.0, U, fX, INSERT_VALUES, mb.PX));
    for (PetscInt i = 0; i < 3; i++) PetscCall(VecDuplicate(U, &mb.q[i]));
    PetscCall(DMProjectField(dm, 0.0, U, ft, INSERT_VALUES, mb.tx)); PetscCall(DMProjectField(dm, 0.0, U, fy, INSERT_VALUES, mb.ty));
    PetscCall(DMProjectField(dm, 0.0, U, fr, INSERT_VALUES, mb.rz0));
    PetscCall(VecDot(mb.tx, mb.tx, &nx)); mb.nx = (PetscInt)PetscRealPart(nx);
    PetscCall(DMCreateMatrix(dm, &Jmb)); PetscCall(DMCreateMatrix(dm, &Jpmb));
    PetscCall(SNESSetFunction(snes, NULL, MeanBCFunction, &mb));
    PetscCall(SNESSetJacobian(snes, Jmb, Jpmb, MeanBCJacobian, &mb));
    mean_bc_ctx = &mb;
  }
  PetscCall(SNESSetFromOptions(snes));
  if (prob == PROB_LV && mean_bc) {                         /* the operator is singular: Krylov with the null space removed */
    KSP ksp; PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPSetType(ksp, KSPFGMRES)); PetscCall(KSPSetTolerances(ksp, 1e-10, 1e-30, PETSC_DEFAULT, 200));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[mean_bc] no rim pins; null average in-plane displacement and rotation; FGMRES on the "
                          "Jacobian with its rigid null space, preconditioner = the given PC of Jacobian + spring %.2e Pa/m^2\n", (double)c0[C_REG]));
  }
  if (prob == PROB_LV || vtk_every > 0) {                  /* rank 0 creates the output directory, the others wait */
    PetscMPIInt rank; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (!rank) {
      char fn[PETSC_MAX_PATH_LEN], mesh[PETSC_MAX_PATH_LEN] = ""; FILE *fp;
      PetscCall(PetscMkdir(out_dir));
      PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_filename", mesh, sizeof(mesh), NULL));
      PetscCall(PetscSNPrintf(fn, sizeof(fn), "%s/run_info.txt", out_dir));
      fp = fopen(fn, "w");
      PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot write %s", fn);
      /* read by biofem post solid3d: the vertex files u_stepNNNN.txt are at t = k/steps of the gamma ramp */
      fprintf(fp, "code=biofem-solid3d\nsource=%s\nproblem=%s\nmesh=%s\ngamma=%g\nsteps=%d\nout_every=%d\nalpha_endo=%g\nalpha_epi=%g\n",
              BIOFEM_GIT, ProblemNames[prob], mesh, (double)gmax, (int)steps, (int)out_every, (double)geo.a_endo, (double)geo.a_epi);
      fprintf(fp, "prolate_d=%g\nlam_endo=%g\nlam_epi=%g\nmu_base_deg=%g\n", (double)geo.d, (double)geo.le, (double)geo.lp, (double)mu_base_deg);
      if (prob == PROB_LV) fprintf(fp, "base_bc=%s\n", mean_bc ? "mean (null average in-plane displacement and rotation)" : "pins (endo rim theta=0: ux=uy=0; epi rim theta=0: uy=0)");
      /* p_ed > 0: u_step0000.txt is end-diastole (after filling); metrics are relative to it */
      if (prob == PROB_LV && p_ed > 0.0) fprintf(fp, "p_ed=%g\np_ao=%g\nfill_steps=%d\n", (double)p_ed, (double)p_ao, (int)fill_steps);
      fclose(fp);
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  /* plate: Nitti's diagonal Fa at gamma */
  const PetscReal a1 = 1.0 - gmax / (nLx * nLx), a2 = 1.0 - gmax / (nLy * nLy), a3 = 1.0 / (a1 * a2);
  PetscLogDouble t0; PetscCall(PetscTime(&t0));
  const PetscBool pv = (PetscBool)(prob == PROB_LV && p_ed > 0.0);
  const char *phase_names[5] = {"filling", "isovol-contraction", "ejection", "isovol-relaxation", "filling"};
  FILE *fpv = NULL; PetscReal V, EDV = 0.0, ESV = 0.0, Vfix = 0.0; PetscInt phase = 0;
  if (pv) {                                                  /* phase 0: passive filling 0 -> p_ed at gamma = 0 */
    char fn[PETSC_MAX_PATH_LEN]; PetscCall(PetscSNPrintf(fn, sizeof(fn), "%s/pv.csv", out_dir));
    PetscCall(PetscFOpen(PETSC_COMM_WORLD, fn, "w", &fpv));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fpv, "step,t,phase,gamma,p_kPa,V_ml,converged,newton_its\n"));
    PetscCall(CavityVolume(dm, ds, U, c0, &V, NULL));
    if (test_jdir) {                                        /* on a small random state, gamma 0.1 */
      Vec r; PetscRandom rnd; PetscScalar cs[C_N];
      for (PetscInt i = 0; i < C_N; i++) cs[i] = c0[i];
      cs[C_GAMMA] = 0.1;
      PetscCall(VecDuplicate(U, &r)); PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd)); PetscCall(PetscRandomSetInterval(rnd, -1e-4, 1e-4));
      PetscCall(VecSetRandom(r, rnd)); PetscCall(PetscRandomDestroy(&rnd));
      PetscCall(TestJacobianDir(snes, ds, r, cs, 5000.0, "random state"));
      PetscCall(VecDestroy(&r));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[filling] unloaded cavity volume %.3f ml\n", (double)(V * 1e6)));
    for (PetscInt j = 1; j <= fill_steps; j++) {
      PetscInt its; PetscBool ok;
      for (PetscInt i = 0; i < C_N; i++) c1[i] = c0[i];
      c1[C_PCAV] = p_ed * (PetscReal)j / fill_steps;
      PetscCall(SolveStep(snes, ds, U, c0, c1, 0.0, vtol, levels, &its, &ok));
      PetscCall(CavityVolume(dm, ds, U, c0, &V, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[filling %d/%d] p_c %.3f kPa: V %.3f ml, %s, %d Newton its\n", (int)j, (int)fill_steps,
                            (double)(PetscRealPart(c1[C_PCAV]) / 1000), (double)(V * 1e6), ok ? "converged" : "NOT converged", (int)its));
      PetscCheck(ok, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "filling step %d did not converge", (int)j);
    }
    EDV = V; Vfix = EDV; phase = 1;
    if (test_jdir) PetscCall(TestJacobianDir(snes, ds, U, c0, p_ed, "end-diastole"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "END-DIASTOLE: EDV %.3f ml at %.3f kPa\n", (double)(EDV * 1e6), (double)(p_ed / 1000)));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fpv, "0,0,0,0,%.6f,%.6f,1,0\n", (double)(p_ed / 1000), (double)(EDV * 1e6)));
    PetscCall(WriteStep(dm, U, out_dir, 0, PETSC_TRUE, (PetscBool)(vtk_every > 0)));
  }
  Vec Ustart; PetscScalar cstart[C_N];
  PetscCall(VecDuplicate(U, &Ustart));
  for (PetscInt k = 1; k <= steps; k++) {
    const PetscReal tt = (PetscReal)k / steps;
    for (PetscInt i = 0; i < C_N; i++) c1[i] = c0[i];
    if (prob == PROB_LV) {                                 /* Pezzuto ramp: linear up to T/2, back to 0 at T */
      c1[C_GAMMA] = gmax * (1.0 - PetscAbsReal(2.0 * tt - 1.0));
    } else {
      const PetscReal s = tt;
      c1[C_GAMMA] = gmax * s;
      c1[C_A1] = 1.0 + (a1 - 1.0) * s; c1[C_A2] = 1.0 + (a2 - 1.0) * s; c1[C_A3] = 1.0 + (a3 - 1.0) * s;
      c1[C_TRACTION] = traction * s;
    }
    PetscInt its = 0; PetscBool ok;
    if (pv && phase == 2 && tt > 0.5 + 1e-12) {           /* gamma starts to fall: the aortic valve closes */
      PetscCall(CavityVolume(dm, ds, U, c0, &ESV, NULL)); Vfix = ESV; phase = 3;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "END-SYSTOLE: ESV %.3f ml, EF %.2f%%\n", (double)(ESV * 1e6), (double)(100 * (EDV - ESV) / EDV)));
    }
    PetscCall(VecCopy(U, Ustart)); for (PetscInt i = 0; i < C_N; i++) cstart[i] = c0[i];
    for (;;) {                                             /* a valve event re-solves the step in the new phase */
      const PetscBool iso = (PetscBool)(pv && (phase == 1 || phase == 3));
      if (pv && !iso) c1[C_PCAV] = (phase == 2) ? p_ao : p_ed;
      PetscCall(SolveStep(snes, ds, U, c0, c1, iso ? Vfix : 0.0, vtol, levels, &its, &ok));
      if (!ok || !pv) break;
      const PetscReal pc = PetscRealPart(c0[C_PCAV]);
      PetscInt next = phase;
      if (phase == 1 && pc >= p_ao) next = 2;               /* aortic valve opens */
      if (phase == 3 && pc <= p_ed) next = 4;               /* mitral valve opens */
      if (next == phase) break;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s valve opens at t = %.3f (p_c %.3f kPa): step redone as %s\n", next == 2 ? "aortic" : "mitral",
                            (double)tt, (double)(pc / 1000), phase_names[next]));
      phase = next; PetscCall(VecCopy(Ustart, U)); for (PetscInt i = 0; i < C_N; i++) c0[i] = cstart[i];
      PetscCall(PetscDSSetConstants(ds, C_N, c0));
    }
    PetscLogDouble t1; PetscCall(PetscTime(&t1));
    if (pv) {
      PetscCall(CavityVolume(dm, ds, U, c0, &V, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[step %d %s] gamma %.5f: p_c %.3f kPa, V %.3f ml, %s, %d Newton its (%.1f s)\n", (int)k, phase_names[phase],
                            (double)PetscRealPart(c1[C_GAMMA]), (double)(PetscRealPart(c0[C_PCAV]) / 1000), (double)(V * 1e6),
                            ok ? "converged" : "NOT converged", (int)its, (double)(t1 - t0)));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fpv, "%d,%.6f,%d,%.6f,%.6f,%.6f,%d,%d\n", (int)k, (double)tt, (int)phase, (double)PetscRealPart(c1[C_GAMMA]),
                             (double)(PetscRealPart(c0[C_PCAV]) / 1000), (double)(V * 1e6), (int)ok, (int)its));
    } else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[step %d] gamma %.5f: %s, %d Newton its (%.1f s)\n", (int)k, (double)PetscRealPart(c1[C_GAMMA]),
                            ok ? "converged" : "NOT converged", (int)its, (double)(t1 - t0)));
    if (mb.dm && (k % out_every == 0 || k == steps)) {
      PetscReal mm[3]; PetscCall(MeanBCReport_(&mb, U, mm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  [mean_bc] mean u_x %.2e m, mean u_y %.2e m, mean rotation %.2e rad\n", (double)mm[0], (double)mm[1], (double)mm[2]));
    }
    if (!ok) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "stopping: step %d failed with %d substep levels\n", (int)k, (int)levels)); break; }
    PetscCall(WriteStep(dm, U, out_dir, k, (PetscBool)(prob == PROB_LV && (k % out_every == 0 || k == steps)),
                        (PetscBool)(vtk_every > 0 && (k % vtk_every == 0 || k == steps))));
  }
  PetscCall(VecDestroy(&Ustart));
  if (fpv) PetscCall(PetscFClose(PETSC_COMM_WORLD, fpv));

  /* checks */
  if (prob == PROB_CUBE || prob == PROB_PLATE) {
    AffineExact ex; Vec Uex, R; PetscReal nR, nE, nX;
    for (PetscInt i = 0; i < 9; i++) ex.Fa[i] = 0.0;
    if (prob == PROB_CUBE) {
      const PetscReal f0[3] = {1, 0, 0}, lf = 1.0 - gmax, lt = 1.0 / PetscSqrtReal(1.0 - gmax);
      for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) ex.Fa[i * 3 + j] = (lf - lt) * f0[i] * f0[j] + (i == j ? lt : 0.0);
      ex.p = m.a;
    } else {
      /* the shell study's analytical solution (studies/2026-10-19_plate_nitti_verification): stretches by Newton on (s, e) */
      PetscReal s = 1.2, e = 0.9;
      for (PetscInt it = 0; it < 100; it++) {
        const PetscReal Je = s * e * e;
        const PetscReal r1 = m.K * Je * (Je - 1.0) - m.mu * (1.0 - e * e), r2 = m.mu / a1 * (s - e * e / s) - traction;
        const PetscReal d11 = m.K * (2.0 * Je - 1.0) * e * e, d12 = m.K * (2.0 * Je - 1.0) * 2.0 * s * e + 2.0 * m.mu * e;
        const PetscReal d21 = m.mu / a1 * (1.0 + e * e / (s * s)), d22 = -m.mu / a1 * 2.0 * e / s, det = d11 * d22 - d12 * d21;
        s -= (r1 * d22 - r2 * d12) / det; e -= (d11 * r2 - d21 * r1) / det;
      }
      ex.Fa[0] = a1 * s; ex.Fa[4] = a2 * e; ex.Fa[8] = a3 * e; ex.p = 0.0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[plate] Nitti Fa = diag(%.4f, %.4f, %.4f); analytical stretches (%.5f, %.5f, %.5f) (shell study: 1.0387, 0.6868, 1.4064)\n",
                            (double)a1, (double)a2, (double)a3, (double)ex.Fa[0], (double)ex.Fa[4], (double)ex.Fa[8]));
    }
    PetscErrorCode (*funcs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {exact_u, exact_p};
    void *ctxs[2] = {&ex, &ex};
    PetscCall(VecDuplicate(U, &Uex)); PetscCall(VecDuplicate(U, &R));
    PetscCall(DMProjectFunction(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, Uex));
    PetscCall(SNESComputeFunction(snes, Uex, R)); PetscCall(VecNorm(R, NORM_INFINITY, &nR));
    PetscCall(VecNorm(Uex, NORM_INFINITY, &nX));
    PetscCall(VecAXPY(Uex, -1.0, U)); PetscCall(VecNorm(Uex, NORM_INFINITY, &nE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%s] residual at the exact solution ||R||_inf = %.3e; solution vs exact: max |error| = %.3e (max |exact| = %.3e)\n",
                          ProblemNames[prob], (double)nR, (double)nE, (double)nX));
    PetscCall(VecDestroy(&Uex)); PetscCall(VecDestroy(&R));
  }

  PetscCall(VecViewFromOptions(U, NULL, "-solution_view"));
  PetscCall(VecDestroy(&U));
  PetscCall(PetscFEDestroy(&fe[0])); PetscCall(PetscFEDestroy(&fe[1]));
  if (mb.dm) {
    PetscCall(VecDestroy(&mb.tx)); PetscCall(VecDestroy(&mb.ty)); PetscCall(VecDestroy(&mb.rz0)); PetscCall(VecDestroy(&mb.PX));
    mean_bc_ctx = NULL;
    for (PetscInt i = 0; i < 3; i++) PetscCall(VecDestroy(&mb.q[i]));
    PetscCall(MatDestroy(&Jmb)); PetscCall(MatDestroy(&Jpmb));
  }
  PetscCall(SNESDestroy(&snes)); PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef SOLID3D_STANDALONE
int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "biofem-solid3d: 3D active-strain solid, P2-P1"));
#ifndef BIOFEM_GIT
#define BIOFEM_GIT "unknown"
#endif
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[biofem-solid3d] source %s\n", BIOFEM_GIT));
  PetscCall(Solid3dRun());
  PetscCall(PetscFinalize());
  return 0;
}
#endif
