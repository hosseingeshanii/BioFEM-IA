/* biofem-solid3d: 3D active-strain Holzapfel-Ogden solid, Taylor-Hood P2-P1 on DMPlex/PetscFE.
 * Formulation: research lv-vv/formulation/solid3d_formulation.pdf (v2); equation numbers below refer to it.
 *
 * Milestone 1 (this version): the free active cube of the verification plan (sec. 8, item 1). Unit cube, uniform
 * fiber f0 = e_x, sheet s0 = e_z, rollers on x=0 (u_x=0), y=0 (u_y=0), z=0 (u_z=0). Exact solution u = (Fa - I) X,
 * p = a (stress-free). Checks: residual at the exact solution, Newton to it, and (with -snes_test_jacobian) the tangent.
 *
 * Example (local, small): biofem-solid3d -dm_plex_simplex 1 -dm_plex_box_faces 2,2,2 -gamma 0.2 -steps 4 \
 *   -snes_monitor -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full \
 *   -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type jacobi
 */
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "solid3d_ho.h"

/* PetscDS constants: HO parameters, gamma, switches, fiber and sheet (uniform in milestone 1) */
enum { C_A, C_B, C_AF, C_BF, C_AS, C_BS, C_AFS, C_BFS, C_GAMMA, C_TENSION, C_SHEET, C_F0, C_S0 = C_F0 + 3, C_N = C_S0 + 3 };

static void unpack(const PetscScalar c[], HOParams *m, PetscReal *gamma, PetscReal f[3], PetscReal s[3])
{
  m->a = PetscRealPart(c[C_A]); m->b = PetscRealPart(c[C_B]); m->af = PetscRealPart(c[C_AF]); m->bf = PetscRealPart(c[C_BF]);
  m->as = PetscRealPart(c[C_AS]); m->bs = PetscRealPart(c[C_BS]); m->afs = PetscRealPart(c[C_AFS]); m->bfs = PetscRealPart(c[C_BFS]);
  m->tension_only = (PetscBool)(PetscRealPart(c[C_TENSION]) != 0.0); m->sheet = (PetscBool)(PetscRealPart(c[C_SHEET]) != 0.0);
  *gamma = PetscRealPart(c[C_GAMMA]);
  for (PetscInt i = 0; i < 3; i++) { f[i] = PetscRealPart(c[C_F0 + i]); s[i] = PetscRealPart(c[C_S0 + i]); }
}

static void grad_to_F(const PetscScalar u_x[], const PetscInt uOff_x[], PetscReal F[9])
{
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt l = 0; l < 3; l++) F[k * 3 + l] = (k == l ? 1.0 : 0.0) + PetscRealPart(u_x[uOff_x[0] + k * 3 + l]);   /* (2.1) */
}

/* Residual R_u (3.8): f1 = P (multiplies grad of the u test function) */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                 const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants,
                 const PetscScalar constants[], PetscScalar f1[])
{
  HOParams m; PetscReal g, f[3], s[3], F[9], P[9];
  unpack(constants, &m, &g, f, s);
  grad_to_F(u_x, uOff_x, F);
  HO_Stress(&m, g, f, s, F, PetscRealPart(u[uOff[1]]), P, NULL);
  for (PetscInt i = 0; i < 9; i++) f1[i] = P[i];
}

/* Residual R_p (3.9): f0 = -(J - 1) */
static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                 const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                 const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants,
                 const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal F[9];
  grad_to_F(u_x, uOff_x, F);
  const PetscReal J = F[0] * (F[4] * F[8] - F[5] * F[7]) - F[1] * (F[3] * F[8] - F[5] * F[6]) + F[2] * (F[3] * F[7] - F[4] * F[6]);
  f0[0] = -(J - 1.0);
}

/* J F^-T, needed by the two coupling blocks (4.2)-(4.3) */
static void JFinvT(const PetscReal F[9], PetscReal M[9])
{
  /* cofactor matrix = J F^-T */
  M[0] = F[4] * F[8] - F[5] * F[7]; M[1] = F[5] * F[6] - F[3] * F[8]; M[2] = F[3] * F[7] - F[4] * F[6];
  M[3] = F[2] * F[7] - F[1] * F[8]; M[4] = F[0] * F[8] - F[2] * F[6]; M[5] = F[1] * F[6] - F[0] * F[7];
  M[6] = F[1] * F[5] - F[2] * F[4]; M[7] = F[2] * F[3] - F[0] * F[5]; M[8] = F[0] * F[4] - F[1] * F[3];
}

/* g3_uu = dP_ij/dF_kl, layout ((i*3 + k)*3 + j)*3 + l: test comp i, trial comp k, test deriv j, trial deriv l */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                  const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                  const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  HOParams m; PetscReal g, f[3], s[3], F[9], P[9], D[81];
  unpack(constants, &m, &g, f, s);
  grad_to_F(u_x, uOff_x, F);
  HO_Tangent(&m, g, f, s, F, PetscRealPart(u[uOff[1]]), P, D);
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++)
      for (PetscInt k = 0; k < 3; k++)
        for (PetscInt l = 0; l < 3; l++) g3[((i * 3 + k) * 3 + j) * 3 + l] = D[((i * 3 + j) * 3 + k) * 3 + l];
}

/* g2_up = dP_ij/dp = -J F^-T_ij, layout (i*1 + 0)*3 + j */
static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                  const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                  const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscReal F[9], M[9];
  grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  for (PetscInt i = 0; i < 9; i++) g2[i] = -M[i];
}

/* g1_pu = d(-(J-1))/dF_kl = -J F^-T_kl, layout (0*3 + k)*3 + l */
static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                  const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                  const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal F[9], M[9];
  grad_to_F(u_x, uOff_x, F); JFinvT(F, M);
  for (PetscInt i = 0; i < 9; i++) g1[i] = -M[i];
}

/* ---------------- exact solution of the free cube: u = (Fa - I) X, p = a ---------------- */
typedef struct { PetscReal Fa[9]; PetscReal a; } CubeExact;

static PetscErrorCode exact_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const CubeExact *e = (const CubeExact *)ctx;
  for (PetscInt i = 0; i < 3; i++) {
    PetscReal c = 0.0;
    for (PetscInt j = 0; j < 3; j++) c += (e->Fa[i * 3 + j] - (i == j ? 1.0 : 0.0)) * x[j];
    u[i] = c;
  }
  return PETSC_SUCCESS;
}
static PetscErrorCode exact_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = ((const CubeExact *)ctx)->a;
  return PETSC_SUCCESS;
}
static PetscErrorCode zero_fn(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt i = 0; i < Nc; i++) u[i] = 0.0;
  return PETSC_SUCCESS;
}

/* Label the faces of the box whose centroid lies on coordinate plane x_d = lo[d]: label "sym<d>", value 1 */
static PetscErrorCode LabelSymmetryFaces(DM dm)
{
  PetscInt fStart, fEnd;
  PetscReal lo[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL}, glo[3];
  Vec coords; const PetscScalar *c; PetscInt n;
  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetLocalSize(coords, &n));
  PetscCall(VecGetArrayRead(coords, &c));
  for (PetscInt i = 0; i < n / 3; i++)
    for (PetscInt d = 0; d < 3; d++) lo[d] = PetscMin(lo[d], PetscRealPart(c[3 * i + d]));
  PetscCall(VecRestoreArrayRead(coords, &c));
  PetscCallMPI(MPIU_Allreduce(lo, glo, 3, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt d = 0; d < 3; d++) {
    char name[8]; DMLabel lab;
    PetscCall(PetscSNPrintf(name, sizeof(name), "sym%d", (int)d));
    PetscCall(DMCreateLabel(dm, name));
    PetscCall(DMGetLabel(dm, name, &lab));
    for (PetscInt f = fStart; f < fEnd; f++) {
      PetscInt support; PetscReal vol, cen[3], nrm[3];
      PetscCall(DMPlexGetSupportSize(dm, f, &support));
      if (support != 1) continue;                       /* boundary faces only */
      PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &vol, cen, nrm));
      if (PetscAbsReal(cen[d] - glo[d]) < 1e-10) PetscCall(DMLabelSetValue(lab, f, 1));
    }
    PetscCall(DMPlexLabelComplete(dm, lab));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm; SNES snes; Vec U, Uex, R; PetscFE fe[2]; PetscDS ds;
  PetscReal gmax = 0.2; PetscInt steps = 4;
  PetscScalar cst[C_N];
  HOParams m = {236.2, 10.810, 20037.0, 14.154, 3724.5, 5.1645, 410.8, 11.300, PETSC_TRUE, PETSC_TRUE};   /* Wang 2013, Pa */
  PetscReal f0[3] = {1, 0, 0}, s0[3] = {0, 0, 1};
  CubeExact ex;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "biofem-solid3d: 3D active-strain HO solid, P2-P1"));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-gamma", &gmax, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-steps", &steps, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ho_tension_only", &m.tension_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ho_sheet", &m.sheet, NULL));

  /* mesh: box from options (-dm_plex_dim 3 -dm_plex_simplex 1 -dm_plex_box_faces n,n,n), version-robust */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_dim", "3"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(LabelSymmetryFaces(dm));

  /* fields: P2 displacement, P1 pressure, same quadrature (degree 4) */
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, 3, PETSC_TRUE, 2, 4, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "displacement"));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 3, 1, PETSC_TRUE, 1, 4, &fe[1]));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscObjectSetName((PetscObject)fe[1], "pressure"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe[1]));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, f1_u));
  PetscCall(PetscDSSetResidual(ds, 1, f0_p, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_up, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL, NULL));

  /* rollers on the three symmetry planes (also remove the rigid modes) */
  for (PetscInt d = 0; d < 3; d++) {
    char name[8]; DMLabel lab; PetscInt one = 1, comp = d;
    PetscCall(PetscSNPrintf(name, sizeof(name), "sym%d", (int)d));
    PetscCall(DMGetLabel(dm, name, &lab));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, name, lab, 1, &one, 0, 1, &comp, (void (*)(void))zero_fn, NULL, NULL, NULL));
  }

  PetscCall(DMCreateGlobalVector(dm, &U));
  PetscCall(PetscObjectSetName((PetscObject)U, "solution"));
  PetscCall(VecDuplicate(U, &Uex)); PetscCall(VecDuplicate(U, &R));
  PetscCall(VecSet(U, 0.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetFromOptions(snes));

  cst[C_A] = m.a; cst[C_B] = m.b; cst[C_AF] = m.af; cst[C_BF] = m.bf; cst[C_AS] = m.as; cst[C_BS] = m.bs; cst[C_AFS] = m.afs; cst[C_BFS] = m.bfs;
  cst[C_TENSION] = m.tension_only; cst[C_SHEET] = m.sheet;
  for (PetscInt i = 0; i < 3; i++) { cst[C_F0 + i] = f0[i]; cst[C_S0 + i] = s0[i]; }

  for (PetscInt k = 1; k <= steps; k++) {
    const PetscReal g = gmax * k / steps;
    cst[C_GAMMA] = g;
    PetscCall(PetscDSSetConstants(ds, C_N, cst));
    PetscCall(SNESSolve(snes, NULL, U));
    SNESConvergedReason reason; PetscInt its;
    PetscCall(SNESGetConvergedReason(snes, &reason)); PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[step %d] gamma %.4f: %s after %d Newton its\n", (int)k, (double)g,
                          reason > 0 ? "converged" : "NOT converged", (int)its));
  }

  /* exact solution at gamma_max: Fa (2.2) with f0 = e_x, p = a */
  {
    const PetscReal g = gmax, lf = 1.0 - g, lt = 1.0 / PetscSqrtReal(1.0 - g);
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++) ex.Fa[i * 3 + j] = (lf - lt) * f0[i] * f0[j] + (i == j ? lt : 0.0);
    ex.a = m.a;
    PetscErrorCode (*funcs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {exact_u, exact_p};
    void *ctxs[2] = {&ex, &ex};
    PetscCall(DMProjectFunction(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, Uex));
    PetscReal nR, nE, nX;
    PetscCall(SNESComputeFunction(snes, Uex, R)); PetscCall(VecNorm(R, NORM_INFINITY, &nR));
    PetscCall(VecNorm(Uex, NORM_INFINITY, &nX));
    PetscCall(VecAXPY(Uex, -1.0, U)); PetscCall(VecNorm(Uex, NORM_INFINITY, &nE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[cube] residual at the exact solution ||R||_inf = %.3e; Newton solution vs exact: max |error| = %.3e (max |exact| = %.3e)\n",
                          (double)nR, (double)nE, (double)nX));
  }

  PetscCall(VecViewFromOptions(U, NULL, "-solution_view"));
  PetscCall(VecDestroy(&U)); PetscCall(VecDestroy(&Uex)); PetscCall(VecDestroy(&R));
  PetscCall(PetscFEDestroy(&fe[0])); PetscCall(PetscFEDestroy(&fe[1]));
  PetscCall(SNESDestroy(&snes)); PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
