/* Assembled preconditioner for the matrix-free Newton solve (-fem_jac_pc 1).
 *
 * The operator stays matrix-free (MatCreateSNESMF). For the preconditioner, the tangent stiffness
 * P ~ K = dR/dx is assembled by finite differences with coloring (PETSc MatFDColoring):
 *
 *  - Sparsity: R at node i depends only on the control points of the element patches that contain i,
 *    so K_ij != 0 only if i and j are control points of a common element patch. Ghost patch nodes
 *    are replaced by their three source nodes (g = x1 + x2 - x3, GlobalGhostInit). Verified with
 *    -debug_jac_node (main.c): the nodes whose residual changes are exactly these.
 *  - Layout: same global DOF numbering as the solution Vec (ctx->ibm_to_global_dof0), 3x3 blocks.
 *  - Coloring: distance-2 (MatColoring), so each color's columns touch disjoint rows; one residual
 *    evaluation per color.
 *  - Ghost DOF rows: the residual is zero there (FEM_DMPlexGeomSyncGhostDOFs) and ghost x entries are
 *    never read, so their rows/columns are empty; a 1 is put on their diagonal so the factorization works.
 *
 * Options: -fem_jac_pc 1 enables it; the PC itself is chosen with -fem_pc_type (default here: bjacobi
 * with ILU on each block), lagging with -fem_snes_lag_jacobian / -fem_snes_lag_preconditioner.
 */
#include "variables.h"
#include "dmplex_geom.h"
#include "jacobian_pc.h"
#include <petscsnes.h>
#include <petsc/private/hashseti.h>

extern PetscInt dof;
extern PetscInt mean_bc_skip;
extern PetscInt lv_mean_bc;
extern PetscErrorCode FormFunctionFEM(SNES snes, Vec x, Vec R, void *ctx);

/* Real nodes an element's residual contribution depends on (its "virtual" patch list). */
static PetscInt ElemPatchRealNodes_(IBMNodes *ibm, PetscInt ec, PetscInt list[48])
{
  PetscInt n = 0;
  for (PetscInt i = 0; i < 16; i++) {
    PetscInt q = ibm->patch[16 * ec + i];
    if (q == 1000000 || q < 0) continue;
    if (q < ibm->n_v) list[n++] = q;
    else if (ibm->ghost_src && q < ibm->n_v + ibm->n_ghosts)
      for (PetscInt k = 0; k < 3; k++) list[n++] = ibm->ghost_src[3 * (q - ibm->n_v) + k];
  }
  return n;
}

PetscErrorCode FEM_JacPCCreate(FE *fem, FEJacPC *jp)
{
  PetscFunctionBeginUser;
  IBMNodes      *ibm  = fem->ibm;
  DMPlexGeomCtx *gctx = &fem->geom_ctx;
  const PetscInt nv = ibm->n_v, ng = ibm->n_ghosts, bs = dof;
  PetscMPIInt    rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(gctx->initialized, PETSC_COMM_WORLD, PETSC_ERR_ORDER, "-fem_jac_pc needs the DMPlex geometry context");
  PetscCheck(bs == 3, PETSC_COMM_WORLD, PETSC_ERR_SUP, "-fem_jac_pc assumes 3 DOFs per node");

  PetscInt nloc, N, lo, hi;
  PetscCall(VecGetLocalSize(fem->x, &nloc));
  PetscCall(VecGetSize(fem->x, &N));
  PetscCall(VecGetOwnershipRange(fem->x, &lo, &hi));
  const PetscInt mb = nloc / bs, blo = lo / bs, bhi = hi / bs;

  /* coupled node sets for the rows this rank owns (block-row index = global dof0 / 3) */
  PetscHSetI *rowset;
  PetscCall(PetscMalloc1(mb, &rowset));
  for (PetscInt r = 0; r < mb; r++) PetscCall(PetscHSetICreate(&rowset[r]));
  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++) {
    PetscInt list[48];
    PetscInt n = ElemPatchRealNodes_(ibm, ec, list);
    for (PetscInt a = 0; a < n; a++) {
      PetscInt brow = gctx->ibm_to_global_dof0[list[a]] / bs;
      if (brow < blo || brow >= bhi) continue;
      for (PetscInt b = 0; b < n; b++) PetscCall(PetscHSetIAdd(rowset[brow - blo], gctx->ibm_to_global_dof0[list[b]] / bs));
    }
  }
  /* every node couples to itself; ghost rows get only their diagonal */
  for (PetscInt v = 0; v < nv + ng; v++) {
    PetscInt brow = gctx->ibm_to_global_dof0[v] / bs;
    if (gctx->ibm_to_global_dof0[v] < 0 || brow < blo || brow >= bhi) continue;
    PetscCall(PetscHSetIAdd(rowset[brow - blo], brow));
  }

  PetscInt *dnz, *onz, maxrow = 0;
  PetscCall(PetscCalloc2(mb, &dnz, mb, &onz));
  for (PetscInt r = 0; r < mb; r++) {
    PetscInt sz; PetscCall(PetscHSetIGetSize(rowset[r], &sz));
    PetscInt *cols; PetscCall(PetscMalloc1(sz, &cols));
    PetscInt off = 0; PetscCall(PetscHSetIGetElems(rowset[r], &off, cols));
    for (PetscInt c = 0; c < sz; c++) { if (cols[c] >= blo && cols[c] < bhi) dnz[r]++; else onz[r]++; }
    maxrow = PetscMax(maxrow, sz);
    PetscCall(PetscFree(cols));
  }
  PetscCall(MatCreate(PETSC_COMM_WORLD, &jp->P));
  PetscCall(MatSetSizes(jp->P, nloc, nloc, N, N));
  PetscCall(MatSetType(jp->P, MATAIJ));
  PetscCall(MatSetBlockSize(jp->P, bs));
  PetscCall(MatXAIJSetPreallocation(jp->P, bs, dnz, onz, NULL, NULL));
  PetscCall(MatSetOption(jp->P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  {
    PetscScalar zeros[9] = {0};
    for (PetscInt r = 0; r < mb; r++) {
      PetscInt sz; PetscCall(PetscHSetIGetSize(rowset[r], &sz));
      PetscInt *cols; PetscCall(PetscMalloc1(sz, &cols));
      PetscInt off = 0; PetscCall(PetscHSetIGetElems(rowset[r], &off, cols));
      PetscInt brow = blo + r;
      for (PetscInt c = 0; c < sz; c++) PetscCall(MatSetValuesBlocked(jp->P, 1, &brow, 1, &cols[c], zeros, INSERT_VALUES));
      PetscCall(PetscFree(cols));
    }
  }
  PetscCall(MatAssemblyBegin(jp->P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jp->P, MAT_FINAL_ASSEMBLY));
  for (PetscInt r = 0; r < mb; r++) PetscCall(PetscHSetIDestroy(&rowset[r]));
  PetscCall(PetscFree(rowset));
  PetscCall(PetscFree2(dnz, onz));

  /* ghost DOF rows (owned by rank 0): global indices, for the diagonal fix */
  jp->n_ghost_rows = 0; jp->ghost_rows = NULL;
  if (ng > 0) {
    PetscCall(PetscMalloc1(bs * ng, &jp->ghost_rows));
    for (PetscInt g = 0; g < ng; g++) {
      PetscInt d0 = gctx->ibm_to_global_dof0[nv + g];
      if (d0 < lo || d0 >= hi) continue;
      for (PetscInt k = 0; k < bs; k++) jp->ghost_rows[jp->n_ghost_rows++] = d0 + k;
    }
  }

  /* distance-2 coloring + finite-difference Jacobian object */
  MatColoring   mc;
  ISColoring    isc;
  PetscCall(MatColoringCreate(jp->P, &mc));
  PetscCall(MatColoringSetDistance(mc, 2));
  PetscCall(MatColoringSetType(mc, MATCOLORINGJP));
  PetscCall(MatColoringSetFromOptions(mc));
  PetscCall(MatColoringApply(mc, &isc));
  PetscCall(MatColoringDestroy(&mc));
  PetscCall(ISColoringGetIS(isc, PETSC_USE_POINTER, &jp->ncolors, NULL));
  PetscCall(ISColoringRestoreIS(isc, PETSC_USE_POINTER, NULL));
  PetscCall(MatFDColoringCreate(jp->P, isc, &jp->fdc));
  PetscCall(MatFDColoringSetFunction(jp->fdc, (PetscErrorCode (*)(void))FormFunctionFEM, fem));
  PetscCall(MatFDColoringSetFromOptions(jp->fdc));
  PetscCall(MatFDColoringSetUp(jp->P, isc, jp->fdc));
  PetscCall(ISColoringDestroy(&isc));

  MatInfo info; PetscCall(MatGetInfo(jp->P, MAT_GLOBAL_SUM, &info));
  PetscInt maxrow_all = maxrow;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &maxrow_all, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[jacpc] P: %d x %d, %.0f nonzeros (%.1f per row), max %d coupled nodes per row, %d colors "
                        "(= residual evaluations per Jacobian)\n", (int)N, (int)N, (double)info.nz_used,
                        (double)(info.nz_used / N), (int)maxrow_all, (int)jp->ncolors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SNES Jacobian callback: J is the matrix-free operator, P the assembled preconditioner. */
static PetscErrorCode FEM_JacPCCompute_(SNES snes, Vec x, Mat J, Mat P, void *ctx)
{
  PetscFunctionBeginUser;
  FEJacPC *jp = (FEJacPC *)ctx;
  PetscLogDouble t0, t1;
  PetscCall(PetscTime(&t0));
  mean_bc_skip = 1;      /* the mean-BC penalty couples all DOFs: keep it out of the colored P */
  {
    /* The FD columns are (F(x + h e) - F(x))/h. The perturbed F are evaluated with the penalty skipped, so the base
     * F(x) must be too: SNES's own F(x) includes the penalty, which is nonzero once the means are (slightly) nonzero
     * after the first Newton update, and would add penalty(x)/h to every column of P. */
    Vec xc, F0; void *fctx; PetscErrorCode (*ff)(void);
    PetscCall(VecDuplicate(x, &xc)); PetscCall(VecCopy(x, xc)); PetscCall(VecDuplicate(x, &F0));
    PetscCall(MatFDColoringGetFunction(jp->fdc, &ff, &fctx));
    PetscCall(FormFunctionFEM(snes, xc, F0, fctx));
    PetscCall(MatFDColoringSetF(jp->fdc, F0));
    PetscCall(MatFDColoringApply(P, jp->fdc, xc, snes));
    PetscCall(VecDestroy(&xc)); PetscCall(VecDestroy(&F0));
    if (J != P) { PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY)); }   /* sets the MFFD base point */
  }
  mean_bc_skip = 0;
  PetscCall(MatSetOption(P, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  for (PetscInt i = 0; i < jp->n_ghost_rows; i++) PetscCall(MatSetValue(P, jp->ghost_rows[i], jp->ghost_rows[i], 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  /* -lv_mean_bc: the penalty is left out of the colored P, which is then singular in the 3 in-plane rigid modes
   * (x, y translation, rotation about z). A tiny shift on the x, y diagonal (-jacpc_rigid_shift_rel times the mean
   * |diagonal|, default 1e-6) makes P invertible without changing it on the elastic modes; the 3 rigid modes become
   * outlier eigenvalues of P^-1 J that GMRES removes in a few iterations. (Adding the penalty's own k*I instead is
   * wrong: k = 352 is ~35x the small-strain cross-fiber membrane stiffness of HO, ~3 a h.) Dofs are node-interleaved. */
  if (lv_mean_bc) {
    Vec dk; PetscInt lo, hi; PetscScalar *a; PetscReal rel = 1e-6, sum = 0.0, cnt = 0.0, shift;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-jacpc_rigid_shift_rel", &rel, NULL));
    PetscCall(MatCreateVecs(P, &dk, NULL));
    PetscCall(MatGetDiagonal(P, dk));
    PetscCall(VecGetOwnershipRange(dk, &lo, &hi));
    PetscCall(VecGetArray(dk, &a));
    for (PetscInt r = lo; r < hi; r++) if (r % dof < 2) { sum += PetscAbsScalar(a[r - lo]); cnt += 1.0; }
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    shift = rel * sum / PetscMax(cnt, 1.0);
    for (PetscInt r = lo; r < hi; r++) a[r - lo] = (r % dof < 2) ? shift : 0.0;
    PetscCall(VecRestoreArray(dk, &a));
    PetscCall(MatDiagonalSet(P, dk, ADD_VALUES));
    PetscCall(VecDestroy(&dk));
    if (jp->n_builds == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[jacpc] mean BC: rigid-mode shift %.3e on the x,y diagonal (rel %.1e of mean |diag|)\n", (double)shift, (double)rel));
  }
  PetscCall(PetscTime(&t1));
  jp->n_builds++; jp->t_build += t1 - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FEM_JacPCAttach(SNES snes, Mat Jmf, FEJacPC *jp)
{
  PetscFunctionBeginUser;
  PetscCall(SNESSetJacobian(snes, Jmf, jp->P, FEM_JacPCCompute_, jp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FEM_JacPCDestroy(FEJacPC *jp)
{
  PetscFunctionBeginUser;
  PetscCall(MatFDColoringDestroy(&jp->fdc));
  PetscCall(MatDestroy(&jp->P));
  PetscCall(PetscFree(jp->ghost_rows));
  PetscFunctionReturn(PETSC_SUCCESS);
}
