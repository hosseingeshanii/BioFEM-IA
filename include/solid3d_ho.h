#ifndef SOLID3D_HO_H
#define SOLID3D_HO_H
/* Active-strain Holzapfel-Ogden material for the 3D solid (biofem-solid3d), pointwise, Cartesian components.
 * Formulation: research lv-vv/formulation/solid3d_formulation.pdf (v2), eqs. (2.2)-(2.4), (3.1)-(3.5), (4.4)-(4.5).
 * Plain C, no PETSc objects: the same code can run inside PetscDS kernels on CPU or GPU. */
#include <petscsys.h>

typedef struct {
  PetscReal a, b, af, bf, as, bs, afs, bfs; /* Wang 2013 Table I, Pa */
  PetscBool tension_only;                    /* <I4-1> on the fiber and sheet terms */
  PetscBool sheet;                           /* sheet (I4s) and fiber-sheet (I8) terms */
} HOParams;

/* First Piola-Kirchhoff stress P = F S, S = A S_W(Ce) A^T - p J C^-1 (eq. 3.5); returns J = det F.
 * F, P: 3x3 row-major; f0, s0: unit fiber and sheet directions (orthogonal); gamma: activation. */
void HO_Stress(const HOParams *m, PetscReal gamma, const PetscReal f0[3], const PetscReal s0[3],
               const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal *J);

/* Consistent tangent dP_ij/dF_kl = delta_ik S_lj + F_im F_kr CC_mjlr, CC = 2 dS/dC (eqs. 4.4-4.5),
 * stored dPdF[((i*3+j)*3+k)*3+l]. Also returns P (as HO_Stress). */
void HO_Tangent(const HOParams *m, PetscReal gamma, const PetscReal f0[3], const PetscReal s0[3],
                const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal dPdF[81]);

#endif
