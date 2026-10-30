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
 * Mesh from options: -dm_plex_filename <file.msh> (or a PETSc box with a mesh generator).
 */
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "solid3d_ho.h"

typedef enum { PROB_CUBE = 0, PROB_PLATE = 1, PROB_LV = 2 } Problem;
static const char *const ProblemNames[] = {"cube", "plate", "lv", "Problem", "PROB_", NULL};

/* PetscDS constants */
enum { C_LAW, C_A, C_B, C_AF, C_BF, C_AS, C_BS, C_AFS, C_BFS, C_TENSION, C_SHEET, C_MU, C_K,
       C_GAMMA, C_FA_DIAG, C_A1, C_A2, C_A3, C_FIBER_AUX, C_F0, C_S0 = C_F0 + 3, C_TRACTION = C_S0 + 3, C_N };

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

/* one load step, with up to 'levels' halvings on failure (substeps interpolate the constants linearly) */
static PetscErrorCode SolveStep(SNES snes, PetscDS ds, Vec U, PetscScalar c0[], const PetscScalar c1[], PetscInt levels, PetscInt *its_out, PetscBool *ok)
{
  Vec U0; PetscScalar c[C_N];
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(U, &U0)); PetscCall(VecCopy(U, U0));
  *ok = PETSC_FALSE; *its_out = 0;
  for (PetscInt lev = 0; lev <= levels && !*ok; lev++) {
    const PetscInt n = 1 << lev; PetscBool good = PETSC_TRUE;
    PetscCall(VecCopy(U0, U));
    for (PetscInt j = 1; j <= n && good; j++) {
      for (PetscInt i = 0; i < C_N; i++) c[i] = c0[i] + (c1[i] - c0[i]) * (PetscReal)j / n;
      PetscCall(PetscDSSetConstants(ds, C_N, c));
      PetscCall(SNESSolve(snes, NULL, U));
      SNESConvergedReason r; PetscInt its;
      PetscCall(SNESGetConvergedReason(snes, &r)); PetscCall(SNESGetIterationNumber(snes, &its));
      *its_out += its; good = (PetscBool)(r > 0);
    }
    if (good) *ok = PETSC_TRUE;
    else if (lev < levels) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  step failed with %d substep(s), retrying smaller\n", (int)n));
  }
  if (*ok) for (PetscInt i = 0; i < C_N; i++) c0[i] = c1[i];
  else { PetscCall(VecCopy(U0, U)); PetscCall(PetscDSSetConstants(ds, C_N, c0)); }
  PetscCall(VecDestroy(&U0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm; SNES snes; Vec U; PetscFE fe[2]; PetscDS ds;
  Problem prob = PROB_CUBE;
  PetscReal gmax = 0.2, traction = 9.0e5, nLx = 1.2, nLy = 0.9;
  PetscInt steps = 4, out_every = 5, levels = 3;
  char out_dir[PETSC_MAX_PATH_LEN] = "solid3d_out";
  MatParams m = {MAT_HO, 236.2, 10.810, 20037.0, 14.154, 3724.5, 5.1645, 410.8, 11.300, PETSC_TRUE, PETSC_TRUE, 1.5e6, 7.445e7};
  Prolate geo = {0.0291, 0.557, 1.00, -60.0, 60.0};
  PetscReal mu_base_deg = 113.0;
  PetscScalar c0[C_N], c1[C_N];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "biofem-solid3d: 3D active-strain solid, P2-P1"));
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "biofem-solid3d options", "");
  PetscCall(PetscOptionsEnum("-problem", "cube | plate | lv", "", ProblemNames, (PetscEnum)prob, (PetscEnum *)&prob, NULL));
  PetscCall(PetscOptionsReal("-gamma", "peak activation", "", gmax, &gmax, NULL));
  PetscCall(PetscOptionsInt("-steps", "load steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-substep_levels", "halvings tried on a failed step", "", levels, &levels, NULL));
  PetscCall(PetscOptionsInt("-out_every", "lv: write vertex displacements every N steps", "", out_every, &out_every, NULL));
  PetscCall(PetscOptionsString("-out_dir", "output directory", "", out_dir, out_dir, sizeof(out_dir), NULL));
  PetscCall(PetscOptionsBool("-ho_tension_only", "", "", m.tension_only, &m.tension_only, NULL));
  PetscCall(PetscOptionsBool("-ho_sheet", "", "", m.sheet, &m.sheet, NULL));
  PetscCall(PetscOptionsReal("-mu", "neo-Hookean mu (plate)", "", m.mu, &m.mu, NULL));
  PetscCall(PetscOptionsReal("-K", "neo-Hookean K (plate)", "", m.K, &m.K, NULL));
  PetscCall(PetscOptionsReal("-traction", "plate: traction on x = max [Pa]", "", traction, &traction, NULL));
  PetscCall(PetscOptionsReal("-nitti_Lx", "plate: Nitti patch Lx", "", nLx, &nLx, NULL));
  PetscCall(PetscOptionsReal("-nitti_Ly", "plate: Nitti patch Ly", "", nLy, &nLy, NULL));
  PetscCall(PetscOptionsReal("-alpha_endo", "lv: helix angle at the endocardium [deg]", "", geo.a_endo, &geo.a_endo, NULL));
  PetscCall(PetscOptionsReal("-alpha_epi", "lv: helix angle at the epicardium [deg]", "", geo.a_epi, &geo.a_epi, NULL));
  PetscOptionsEnd();
  if (prob == PROB_PLATE) m.law = MAT_NH;

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
    PetscCall(DMGetLabel(dm, "pinA", &lab));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "pinA", lab, 1, &one, 0, 2, cxy, (void (*)(void))zero_fn, NULL, NULL, NULL));
    PetscCall(DMGetLabel(dm, "pinB", &lab));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "pinB", lab, 1, &one, 0, 1, &cy, (void (*)(void))zero_fn, NULL, NULL, NULL));
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
  PetscCall(PetscDSSetConstants(ds, C_N, c0));

  PetscCall(DMCreateGlobalVector(dm, &U)); PetscCall(PetscObjectSetName((PetscObject)U, "solution"));
  PetscCall(VecSet(U, 0.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes)); PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetFromOptions(snes));
  if (prob == PROB_LV) PetscCall(PetscMkdir(out_dir));

  /* plate: Nitti's diagonal Fa at gamma */
  const PetscReal a1 = 1.0 - gmax / (nLx * nLx), a2 = 1.0 - gmax / (nLy * nLy), a3 = 1.0 / (a1 * a2);
  PetscLogDouble t0; PetscCall(PetscTime(&t0));
  for (PetscInt k = 1; k <= steps; k++) {
    for (PetscInt i = 0; i < C_N; i++) c1[i] = c0[i];
    if (prob == PROB_LV) {                                 /* Pezzuto ramp: linear up to T/2, back to 0 at T */
      const PetscReal tt = (PetscReal)k / steps;
      c1[C_GAMMA] = gmax * (1.0 - PetscAbsReal(2.0 * tt - 1.0));
    } else {
      const PetscReal s = (PetscReal)k / steps;
      c1[C_GAMMA] = gmax * s;
      c1[C_A1] = 1.0 + (a1 - 1.0) * s; c1[C_A2] = 1.0 + (a2 - 1.0) * s; c1[C_A3] = 1.0 + (a3 - 1.0) * s;
      c1[C_TRACTION] = traction * s;
    }
    PetscInt its; PetscBool ok;
    PetscCall(SolveStep(snes, ds, U, c0, c1, levels, &its, &ok));
    PetscLogDouble t1; PetscCall(PetscTime(&t1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[step %d] gamma %.5f: %s, %d Newton its (%.1f s)\n", (int)k, (double)PetscRealPart(c1[C_GAMMA]),
                          ok ? "converged" : "NOT converged", (int)its, (double)(t1 - t0)));
    if (!ok) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "stopping: step %d failed with %d substep levels\n", (int)k, (int)levels)); break; }
    if (prob == PROB_LV && (k % out_every == 0 || k == steps)) {
      char fn[PETSC_MAX_PATH_LEN]; PetscCall(PetscSNPrintf(fn, sizeof(fn), "%s/u_step%04d.txt", out_dir, (int)k));
      PetscCall(WriteVertexDisplacement(dm, U, fn));
    }
  }

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
  PetscCall(SNESDestroy(&snes)); PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
