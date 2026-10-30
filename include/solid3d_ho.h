#ifndef SOLID3D_HO_H
#define SOLID3D_HO_H
/* Active-strain materials for the 3D solid (biofem-solid3d), pointwise, Cartesian components.
 * Formulation: research lv-vv/formulation/solid3d_formulation.pdf (v2), eqs. (2.2)-(2.4), (3.1)-(3.5), (4.4)-(4.5).
 * Plain C, no PETSc objects: the same code can run inside PetscDS kernels on CPU or GPU.
 *
 * Structure: an elastic law gives S_e = 2 dPsi/dCe and CC_e = 2 dS_e/dCe at Ce (no tissue pressure); the
 * active-strain wrapper pushes them back with A = Fa^-1 (S = A S_e A^T, CC = A A A A CC_e), adds the tissue pressure
 * term -p J C^-1 when the law is incompressible, and returns P = F S and dP/dF. */
#include <petscsys.h>

typedef enum { MAT_HO = 0, MAT_NH = 1 } MatLaw;

typedef struct {
  MatLaw    law;
  PetscReal a, b, af, bf, as, bs, afs, bfs; /* HO, Wang 2013 Table I [Pa] */
  PetscBool tension_only, sheet;            /* HO switches */
  PetscReal mu, K;                          /* compressible neo-Hookean of the shell code [Pa] */
} MatParams;

/* A = Fa^-1 of the isochoric fiber Fa (2.2)-(2.3) */
void AS_InverseFiberFa(PetscReal gamma, const PetscReal f0[3], PetscReal A[9]);

/* P = F S (3.5) and, if dPdF != NULL, dP_ij/dF_kl stored at ((i*3+j)*3+k)*3+l (4.4-4.5).
 * p (tissue pressure) is used only by incompressible laws (HO). f0, s0 only by HO. Returns J = det F. */
PetscReal AS_Piola(const MatParams *m, const PetscReal A[9], const PetscReal f0[3], const PetscReal s0[3],
                   const PetscReal F[9], PetscReal p, PetscReal P[9], PetscReal dPdF[81]);

#endif
