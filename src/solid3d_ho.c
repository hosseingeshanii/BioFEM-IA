/* Active-strain Holzapfel-Ogden material for the 3D solid, pointwise, Cartesian components.
 * Equation numbers refer to research lv-vv/formulation/solid3d_formulation.pdf (v2). */
#include <math.h>
#include "solid3d_ho.h"

#define IX(i, j) ((i) * 3 + (j))

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

/* Everything both entry points need: A = Fa^-1 (2.3), C, Ce = A C A (2.4), invariants, S_W (3.2), S (3.5). */
typedef struct {
  PetscReal A[9], C[9], Ci[9], J;
  PetscReal I4f_1, I4s_1, I8;        /* I4f - 1, I4s - 1 (before the tension-only cut), I8 */
  PetscReal E1, Ef, Es, Efs;
  PetscBool on_f, on_s;              /* fiber / sheet term active (H(I4 - 1) under tension_only) */
  PetscReal S[9];
} HOState;

static void ho_state(const HOParams *m, PetscReal gamma, const PetscReal f[3], const PetscReal s[3],
                     const PetscReal F[9], PetscReal p, HOState *st)
{
  const PetscReal lf = 1.0 - gamma, lt = 1.0 / PetscSqrtReal(1.0 - gamma);   /* Fa = lf f x f + lt (iI - f x f) */
  PetscReal Ce[9], tmp[9], SW[9], H[9];

  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++)
      st->A[IX(i, j)] = (1.0 / lf - 1.0 / lt) * f[i] * f[j] + ((i == j) ? 1.0 / lt : 0.0);          /* (2.3) */
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += F[IX(k, i)] * F[IX(k, j)];
      st->C[IX(i, j)] = c;
    }
  st->J = det3(F);
  inv3(st->C, st->Ci);
  /* Ce = A^T C A, A symmetric (2.4) */
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += st->C[IX(i, k)] * st->A[IX(k, j)];
      tmp[IX(i, j)] = c;
    }
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += st->A[IX(k, i)] * tmp[IX(k, j)];
      Ce[IX(i, j)] = c;
    }
  PetscReal I1 = Ce[0] + Ce[4] + Ce[8], I4f = 0.0, I4s = 0.0, I8 = 0.0;
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      I4f += f[i] * Ce[IX(i, j)] * f[j]; I4s += s[i] * Ce[IX(i, j)] * s[j]; I8 += f[i] * Ce[IX(i, j)] * s[j];
    }
  st->I4f_1 = I4f - 1.0; st->I4s_1 = I4s - 1.0; st->I8 = I8;
  st->on_f = (PetscBool)(!m->tension_only || st->I4f_1 > 0.0);
  st->on_s = (PetscBool)(m->sheet && (!m->tension_only || st->I4s_1 > 0.0));
  const PetscReal ef = st->on_f ? st->I4f_1 : 0.0, es = st->on_s ? st->I4s_1 : 0.0;
  st->E1 = PetscExpReal(m->b * (I1 - 3.0));
  st->Ef = PetscExpReal(m->bf * ef * ef);
  st->Es = PetscExpReal(m->bs * es * es);
  st->Efs = m->sheet ? PetscExpReal(m->bfs * I8 * I8) : 1.0;
  const PetscReal cf = 2.0 * m->af * ef * st->Ef, cs = 2.0 * m->as * es * st->Es, cfs = m->sheet ? 2.0 * m->afs * I8 * st->Efs : 0.0;
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      H[IX(i, j)] = 0.5 * (f[i] * s[j] + s[i] * f[j]);
      SW[IX(i, j)] = ((i == j) ? m->a * st->E1 : 0.0) + cf * f[i] * f[j] + cs * s[i] * s[j] + cfs * H[IX(i, j)];  /* (3.2) */
    }
  /* S = A S_W A - p J C^-1 (3.5) */
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += SW[IX(i, k)] * st->A[IX(k, j)];
      tmp[IX(i, j)] = c;
    }
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += st->A[IX(i, k)] * tmp[IX(k, j)];
      st->S[IX(i, j)] = c - p * st->J * st->Ci[IX(i, j)];
    }
}

static void first_piola(const PetscReal F[9], const PetscReal S[9], PetscReal P[9])
{
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) {
      PetscReal c = 0.0;
      for (PetscInt k = 0; k < 3; k++) c += F[IX(i, k)] * S[IX(k, j)];
      P[IX(i, j)] = c;
    }
}

void HO_Stress(const HOParams *m, PetscReal gamma, const PetscReal f0[3], const PetscReal s0[3],
               const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal *J)
{
  HOState st;
  ho_state(m, gamma, f0, s0, F, p, &st);
  first_piola(F, st.S, P);
  if (J) *J = st.J;
}

void HO_Tangent(const HOParams *m, PetscReal gamma, const PetscReal f[3], const PetscReal s[3],
                const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal dPdF[81])
{
  HOState st;
  PetscReal CW[81], T1[81], T2[81], CC[81], H[9];
  ho_state(m, gamma, f, s, F, p, &st);
  first_piola(F, st.S, P);

  /* C^W = 2 dS_W/dCe (4.5) */
  const PetscReal ef = st.on_f ? st.I4f_1 : 0.0, es = st.on_s ? st.I4s_1 : 0.0;
  const PetscReal kf = st.on_f ? 4.0 * m->af * st.Ef * (1.0 + 2.0 * m->bf * ef * ef) : 0.0;
  const PetscReal ks = st.on_s ? 4.0 * m->as * st.Es * (1.0 + 2.0 * m->bs * es * es) : 0.0;
  const PetscReal kfs = m->sheet ? 4.0 * m->afs * st.Efs * (1.0 + 2.0 * m->bfs * st.I8 * st.I8) : 0.0;
  const PetscReal k1 = 2.0 * m->a * m->b * st.E1;
  for (PetscInt i = 0; i < 3; i++)
    for (PetscInt j = 0; j < 3; j++) H[IX(i, j)] = 0.5 * (f[i] * s[j] + s[i] * f[j]);
  for (PetscInt p_ = 0; p_ < 3; p_++)
    for (PetscInt q = 0; q < 3; q++)
      for (PetscInt r = 0; r < 3; r++)
        for (PetscInt t = 0; t < 3; t++)
          CW[((p_ * 3 + q) * 3 + r) * 3 + t] = ((p_ == q && r == t) ? k1 : 0.0) + kf * f[p_] * f[q] * f[r] * f[t]
                                               + ks * s[p_] * s[q] * s[r] * s[t] + kfs * H[IX(p_, q)] * H[IX(r, t)];
  /* push-back C_IJKL = A_Ip A_Jq A_Kr A_Ls C^W_pqrs, one index at a time (4.4, first term) */
  const PetscReal *A = st.A;
  for (PetscInt iI = 0; iI < 3; iI++) for (PetscInt q = 0; q < 3; q++) for (PetscInt r = 0; r < 3; r++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt p_ = 0; p_ < 3; p_++) c += A[IX(iI, p_)] * CW[((p_ * 3 + q) * 3 + r) * 3 + t]; T1[((iI * 3 + q) * 3 + r) * 3 + t] = c; }
  for (PetscInt iI = 0; iI < 3; iI++) for (PetscInt Jj = 0; Jj < 3; Jj++) for (PetscInt r = 0; r < 3; r++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt q = 0; q < 3; q++) c += A[IX(Jj, q)] * T1[((iI * 3 + q) * 3 + r) * 3 + t]; T2[((iI * 3 + Jj) * 3 + r) * 3 + t] = c; }
  for (PetscInt iI = 0; iI < 3; iI++) for (PetscInt Jj = 0; Jj < 3; Jj++) for (PetscInt K = 0; K < 3; K++) for (PetscInt t = 0; t < 3; t++) {
    PetscReal c = 0.0; for (PetscInt r = 0; r < 3; r++) c += A[IX(K, r)] * T2[((iI * 3 + Jj) * 3 + r) * 3 + t]; T1[((iI * 3 + Jj) * 3 + K) * 3 + t] = c; }
  for (PetscInt iI = 0; iI < 3; iI++) for (PetscInt Jj = 0; Jj < 3; Jj++) for (PetscInt K = 0; K < 3; K++) for (PetscInt L = 0; L < 3; L++) {
    PetscReal c = 0.0; for (PetscInt t = 0; t < 3; t++) c += A[IX(L, t)] * T1[((iI * 3 + Jj) * 3 + K) * 3 + t]; CC[((iI * 3 + Jj) * 3 + K) * 3 + L] = c; }
  /* pressure part -p J (Ci_IJ Ci_KL - Ci_IK Ci_JL - Ci_IL Ci_JK) (4.4, second term) */
  const PetscReal *Ci = st.Ci, pJ = p * st.J;
  for (PetscInt iI = 0; iI < 3; iI++) for (PetscInt Jj = 0; Jj < 3; Jj++) for (PetscInt K = 0; K < 3; K++) for (PetscInt L = 0; L < 3; L++)
    CC[((iI * 3 + Jj) * 3 + K) * 3 + L] -= pJ * (Ci[IX(iI, Jj)] * Ci[IX(K, L)] - Ci[IX(iI, K)] * Ci[IX(Jj, L)] - Ci[IX(iI, L)] * Ci[IX(Jj, K)]);
  /* dP_ij/dF_kl = delta_ik S_lj + F_im F_kr CC_mjlr */
  for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) for (PetscInt k = 0; k < 3; k++) for (PetscInt l = 0; l < 3; l++) {
    PetscReal c = (i == k) ? st.S[IX(l, j)] : 0.0;
    for (PetscInt mm = 0; mm < 3; mm++)
      for (PetscInt r = 0; r < 3; r++) c += F[IX(i, mm)] * F[IX(k, r)] * CC[((mm * 3 + j) * 3 + l) * 3 + r];
    dPdF[((i * 3 + j) * 3 + k) * 3 + l] = c;
  }
}
