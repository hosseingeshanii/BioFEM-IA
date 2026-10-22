#ifndef JACOBIAN_PC_H
#define JACOBIAN_PC_H
/* Assembled finite-difference (colored) Jacobian used as the preconditioner of the matrix-free
 * Newton solve (-fem_jac_pc 1). See src/jacobian_pc.c. */
#include <petscsnes.h>

typedef struct {
  Mat            P;             /* assembled preconditioner matrix, sparsity from the element patches */
  MatFDColoring  fdc;           /* coloring-based finite-difference Jacobian */
  PetscInt       ncolors;       /* residual evaluations per Jacobian */
  PetscInt      *ghost_rows;    /* global ghost-DOF rows owned here (diagonal set to 1) */
  PetscInt       n_ghost_rows;
  PetscInt       n_builds;      /* Jacobians built so far */
  PetscLogDouble t_build;       /* seconds spent building them */
} FEJacPC;

PetscErrorCode FEM_JacPCCreate(FE *fem, FEJacPC *jp);
PetscErrorCode FEM_JacPCAttach(SNES snes, Mat Jmf, FEJacPC *jp);
PetscErrorCode FEM_JacPCDestroy(FEJacPC *jp);

#endif
