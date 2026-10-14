#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include "variables.h"

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode InitCudaWorkspace(void);
PetscErrorCode DestroyCudaWorkspace(void);
PetscErrorCode PrepareCudaHostWorkspace(FE *fem);
PetscErrorCode RunCudaElementCoordKernel(FE *fem);
PetscErrorCode RunCudaSubdivGeomKernel(FE *fem, PetscBool use_reference_coords);

/**
 * @brief Partitioned variant of RunCudaSubdivGeomKernel for the DMPlex+CUDA
 * path: operates on a per-rank local cell subset (from a DMPlexPatchLayout)
 * instead of the whole, undistributed ibm->n_elmt/ire/val/patch arrays.
 *
 * @param fem                FE struct (fem->ibm and fem->act_data must be
 *                           initialised, coordinates already SF-exchanged).
 * @param nLocalCells        Number of cells owned by this rank's layout.
 * @param localOrigCell      localOrigCell[lc] = original/global element
 *                           index for local cell lc (from
 *                           DMPlexPatchLayout::orig_cell). Cells with
 *                           orig_cell outside [0, ibm->n_elmt) (ghost cells)
 *                           are skipped for the copy-back into
 *                           fem->act_data.
 * @param localIre           Per-local-cell regular/irregular flag (from
 *                           DMPlexPatchLayout::ire).
 * @param localVal           Per-local-cell valence (from
 *                           DMPlexPatchLayout::val).
 * @param localPatch         Per-local-cell 16-slot patch stencil, using
 *                           original/global vertex indices (from
 *                           DMPlexPatchLayout::patch).
 * @param use_reference_coords  PETSC_TRUE for geom0 (reference), PETSC_FALSE
 *                           for geom (current).
 */
PetscErrorCode RunCudaSubdivGeomKernelPartitioned(
    FE *fem,
    PetscInt nLocalCells,
    const PetscInt *localOrigCell,
    const PetscInt *localIre,
    const PetscInt *localVal,
    const PetscInt *localPatch,
    PetscBool use_reference_coords);
PetscErrorCode RunCudaMetricTensorKernel(FE *fem);

/**
 * @brief Partitioned variant of RunCudaMetricTensorKernel for the DMPlex+CUDA
 * path: operates on a per-rank local cell subset instead of the whole,
 * undistributed ibm->n_elmt. Reads fem->act_data.elem_act_data[ec].geom/geom0
 * (already computed, e.g. by RunCudaSubdivGeomKernelPartitioned) and writes
 * g/g0/gm/gm0 back into the same elem_act_data entries.
 *
 * @param fem            FE struct (fem->ibm, fem->act_data, and each local
 *                       cell's geom/geom0 must already be populated).
 * @param nLocalCells    Number of cells owned by this rank's layout.
 * @param localOrigCell  localOrigCell[lc] = original/global element index
 *                       for local cell lc (from DMPlexPatchLayout::orig_cell).
 */
/* Stage-timer accumulators (see cuda_kernel.cu): 0 h2d s, 1 kernel s, 2 d2h s,
   3 host pack/unpack s, 4 h2d bytes, 5 d2h bytes. GetReset returns and zeroes them. */
void CudaStageTimerAdd(int slot, double v);
void CudaStageTimersGetReset(double *out);

PetscErrorCode RunCudaMetricTensorKernelPartitioned(
    FE *fem,
    PetscInt nLocalCells,
    const PetscInt *localOrigCell);

#ifdef __cplusplus
}
#endif

#endif
