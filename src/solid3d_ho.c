/* Active-strain materials for the 3D solid, pointwise, Cartesian components.
 * Equation numbers refer to research lv-vv/formulation/solid3d_formulation.pdf (v2). */
#include <math.h>
#include "solid3d_ho.h"

#define IX(i, j) ((i) * 3 + (j))
#define I4(i, j, k, l) ((((i) * 3 + (j)) * 3 + (k)) * 3 + (l))

static PetscReal det3(const PetscReal M[9])
{
  return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

static void inv3(const PetscReal M[9], PetscReal Mi[9])
{
  const PetscReal d = det3(M);
  Mi[0] = (M[4] * M[8] - M[5] * M[7]) / d; Mi[1] = (M[2] * M[7] - M[1] * M[8]) / d; Mi[2] = (M[1] * M[5] - M[2] * M[4]) / d;
  Mi[3] = (M[5] * M[6] - M[3] * M[8]) / d; Mi[4] = (M[0] * M[8] - M[2] * M[6]) / d; Mi[5] = (M[2] * M[3] - M[0] * M[5]) / d;
  Mi[6] = (M[3] * M[7] - M[4] * M[6]) / d; Mi[7] = (M[1] * M[6] - M[0] * M[7]) / d; Mi[8] = (M[0] * M[4] - M[1] * M[3]) / d;
}

/* C = X^T Y Z for 3x3 (row-major) */
static void triple(const PetscReal X[9], const PetscReal Y[9], const PetscReal Z[9], PetscBool transX, PetscReal C[9])
{
  PetscReal T[9];
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += Y[IX(i, k)] * Z[IX(k, j)];
      T[IX(i, j)] = c;
    }
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += (transX ? X[IX(k, i)] : X[IX(i, k)]) * T[IX(k, j)];
      C[IX(i, j)] = c;
    }
}

void AS_InverseFiberFa(PetscReal gamma, const PetscReal f[3], PetscReal A[9])
{
  const PetscReal lf = 1.0 - gamma, lt = 1.0 / PetscSqrtReal(1.0 - gamma);   /* Fa = lf f x f + lt (I - f x f) (2.2) */
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) A[IX(i, j)] = (1.0 / lf - 1.0 / lt) * f[i] * f[j] + ((i == j) ? 1.0 / lt : 0.0);   /* (2.3) */
}

/* HO: S_W (3.2) and CC^W (4.5) at Ce. f, s are used as given (unit in exact arithmetic). */
static void elastic_HO(const MatParams *m, const PetscReal Ce[9], const PetscReal f[3], const PetscReal s[3],
                       PetscReal Se[9], PetscReal *CCe)
{
  PetscReal I1 = Ce[0] + Ce[4] + Ce[8], I4f = 0.0, I4s = 0.0, I8 = 0.0, H[9];
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      I4f += f[i] * Ce[IX(i, j)] * f[j]; I4s += s[i] * Ce[IX(i, j)] * s[j]; I8 += f[i] * Ce[IX(i, j)] * s[j];
      H[IX(i, j)] = 0.5 * (f[i] * s[j] + s[i] * f[j]);
    }
  const PetscBool on_f = (PetscBool)(!m->tension_only || I4f > 1.0);
  const PetscBool on_s = (PetscBool)(m->sheet && (!m->tension_only || I4s > 1.0));
  const PetscReal ef = on_f ? I4f - 1.0 : 0.0, es = on_s ? I4s - 1.0 : 0.0, i8 = m->sheet ? I8 : 0.0;
  const PetscReal E1 = PetscExpReal(m->b * (I1 - 3.0)), Ef = PetscExpReal(m->bf * ef * ef), Es = PetscExpReal(m->bs * es * es);
  const PetscReal Efs = PetscExpReal(m->bfs * i8 * i8);
  const PetscReal cf = 2.0 * m->af * ef * Ef, cs = 2.0 * m->as * es * Es, cfs = m->sheet ? 2.0 * m->afs * i8 * Efs : 0.0;
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++)
      Se[IX(i, j)] = ((i == j) ? m->a * E1 : 0.0) + cf * f[i] * f[j] + cs * s[i] * s[j] + cfs * H[IX(i, j)];
  if (!CCe) return;
  const PetscReal k1 = 2.0 * m->a * m->b * E1;
  const PetscReal kf = on_f ? 4.0 * m->af * Ef * (1.0 + 2.0 * m->bf * ef * ef) : 0.0;
  const PetscReal ks = on_s ? 4.0 * m->as * Es * (1.0 + 2.0 * m->bs * es * es) : 0.0;
  const PetscReal kfs = m->sheet ? 4.0 * m->afs * Efs * (1.0 + 2.0 * m->bfs * i8 * i8) : 0.0;
  for (PetscInt p = 0; p < 3; p++) for (PetscInt q = 0; q < 3; q++) for (PetscInt r = 0; r < 3; r++) for (PetscInt t = 0; t < 3; t++)
    CCe[I4(p, q, r, t)] = ((p == q && r == t) ? k1 : 0.0) + kf * f[p] * f[q] * f[r] * f[t] + ks * s[p] * s[q] * s[r] * s[t]
                          + kfs * H[IX(p, q)] * H[IX(r, t)];
}

/* Compressible neo-Hookean of the shell code (ElemElasStress / ElemElsTangMatTens), in Cartesian components:
 * S_e = mu (I - Ce^-1) + K (Je^2 - Je) Ce^-1,
 * CC_e = K Je (2 Je - 1) Ce^-1 x Ce^-1 - (K (Je^2 - Je) - mu) (Ce^-1_ik Ce^-1_jl + Ce^-1_il Ce^-1_jk). */
static void elastic_NH(const MatParams *m, const PetscReal Ce[9], PetscReal Se[9], PetscReal *CCe)
{
  PetscReal Ci[9];
  inv3(Ce, Ci);
  const PetscReal Je = PetscSqrtReal(det3(Ce));
  for (PetscInt i = 0; i < 9; i++) Se[i] = m->mu * ((i % 4 == 0) ? 1.0 - Ci[i] : -Ci[i]) + m->K * (Je * Je - Je) * Ci[i];
  if (!CCe) return;
  const PetscReal al = m->K * (Je * Je - Je) - m->mu, be = m->K * Je * (2.0 * Je - 1.0);
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++)
    CCe[I4(i, j, k, l)] = be * Ci[IX(i, j)] * Ci[IX(k, l)] - al * (Ci[IX(i, k)] * Ci[IX(j, l)] + Ci[IX(i, l)] * Ci[IX(j, k)]);
}

PetscReal AS_Piola(const MatParams *m, const PetscReal A[9], const PetscReal f[3], const PetscReal s[3],
                   const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal dPdF[81])
{
  PetscReal C[9], Ci[9], Ce[9], Se[9], S[9], CCe[81], T1[81], T2[81], CC[81];
  const PetscBool incompressible = (PetscBool)(m->law == MAT_HO);
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += F[IX(k, i)] * F[IX(k, j)];
      C[IX(i, j)] = c;                                                    /* C = F^T F (2.1) */
    }
  const PetscReal J = det3(F);
  inv3(C, Ci);
  triple(A, C, A, PETSC_TRUE, Ce);                                       /* Ce = A^T C A (2.4) */
  if (m->law == MAT_HO) elastic_HO(m, Ce, f, s, Se, dPdF ? CCe : NULL);
  else elastic_NH(m, Ce, Se, dPdF ? CCe : NULL);
  triple(A, Se, A, PETSC_FALSE, S);                                     /* A S_e A^T: A symmetric for both Fa types */
  if (incompressible)
    for (PetscInt i = 0; i < 9; i++) S[i] -= p * J * Ci[i];             /* (3.5) */
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += F[IX(i, k)] * S[IX(k, j)];
      P[IX(i, j)] = c;
    }
  if (!dPdF) return J;
  /* CC = A A A A CC_e, one index at a time (4.4, first term) */
  for (PetscInt i = 0; i < 3; i++) for (PetscInt q = 0; q < 3; q++) for (PetscInt r = 0; r < 3; r++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt a = 0; a < 3; a++) c += A[IX(i, a)] * CCe[I4(a, q, r, t)]; T1[I4(i, q, r, t)] = c; }
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt r = 0; r < 3; r++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt q = 0; q < 3; q++) c += A[IX(j, q)] * T1[I4(i, q, r, t)]; T2[I4(i, j, r, t)] = c; }
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt r = 0; r < 3; r++) c += A[IX(k, r)] * T2[I4(i, j, r, t)]; T1[I4(i, j, k, t)] = c; }
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++) {
    PetscReal c = 0.0; for (PetscInt t = 0; t < 3; t++) c += A[IX(l, t)] * T1[I4(i, j, k, t)]; CC[I4(i, j, k, l)] = c; }
  if (incompressible) {                                                  /* (4.4, second term) */
    const PetscReal pJ = p * J;
    for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++)
      CC[I4(i, j, k, l)] -= pJ * (Ci[IX(i, j)] * Ci[IX(k, l)] - Ci[IX(i, k)] * Ci[IX(j, l)] - Ci[IX(i, l)] * Ci[IX(j, k)]);
  }
  /* dP_ij/dF_kl = delta_ik S_lj + F_im F_kr CC_mjlr */
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++) {
    PetscReal c = (i == k) ? S[IX(l, j)] : 0.0;
    for (PetscInt mm = 0; mm < 3; mm++)
      for (PetscInt r = 0; r < 3; r++) c += F[IX(i, mm)] * F[IX(k, r)] * CC[I4(mm, j, l, r)];
    dPdF[I4(i, j, k, l)] = c;
  }
  return J;
}
