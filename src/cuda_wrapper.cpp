#include "cuda_bridge.h"

#include <vector>

extern "C" PetscErrorCode LaunchCudaElementCoordKernelImpl(
    PetscInt n_elmt,
    const PetscInt *nv1,
    const PetscInt *nv2,
    const PetscInt *nv3,
    const PetscReal *x_bp0,
    const PetscReal *y_bp0,
    const PetscReal *z_bp0,
    PetscReal *element_coords_out);

extern "C" PetscErrorCode LaunchCudaSubdivGeomKernelImpl(
    PetscInt n_elmt,
    const PetscInt *ire,
    const PetscInt *val,
    const PetscInt *patch,
    const PetscReal *xb,
    const PetscReal *yb,
    const PetscReal *zb,
    PetscInt *meta_out,
    PetscInt *status_out,
    PetscReal *geom_out,
    PetscReal *irregular_out);

extern "C" PetscErrorCode RunCudaElementCoordKernel(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");

  IBMNodes *ibm = fem->ibm;
  std::vector<PetscReal> element_coords(static_cast<size_t>(ibm->n_elmt) * 9u, 0.0);

  PetscCall(LaunchCudaElementCoordKernelImpl(ibm->n_elmt,
                                             ibm->nv1,
                                             ibm->nv2,
                                             ibm->nv3,
                                             ibm->x_bp0,
                                             ibm->y_bp0,
                                             ibm->z_bp0,
                                             element_coords.data()));

  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode RunCudaSubdivGeomKernel(FE *fem, PetscBool use_reference_coords)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");

  IBMNodes *ibm = fem->ibm;
  const PetscReal *xb = use_reference_coords ? ibm->x_bp0 : ibm->x_bp;
  const PetscReal *yb = use_reference_coords ? ibm->y_bp0 : ibm->y_bp;
  const PetscReal *zb = use_reference_coords ? ibm->z_bp0 : ibm->z_bp;

  std::vector<PetscInt> meta(static_cast<size_t>(ibm->n_elmt) * 3u, 0);
  std::vector<PetscInt> status(static_cast<size_t>(ibm->n_elmt), 0);
  std::vector<PetscReal> geom(static_cast<size_t>(ibm->n_elmt) * 24u, 0.0);
  std::vector<PetscReal> irregular(static_cast<size_t>(ibm->n_elmt) * 80u, 0.0);

  PetscCall(LaunchCudaSubdivGeomKernelImpl(ibm->n_elmt,
                                           ibm->ire,
                                           ibm->val,
                                           ibm->patch,
                                           xb,
                                           yb,
                                           zb,
                                           meta.data(),
                                           status.data(),
                                           geom.data(),
                                           irregular.data()));

  for (PetscInt ec = 0; ec < ibm->n_elmt && ec < 3; ++ec) {
    const size_t meta_base = static_cast<size_t>(ec) * 3u;
    const size_t geom_base = static_cast<size_t>(ec) * 24u;
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA geom host] ec=%d status=%d irregular=%d v=%d nen=%d\n",
                (int)ec,
                (int)status[ec],
                (int)meta[meta_base + 0],
                (int)meta[meta_base + 1],
                (int)meta[meta_base + 2]);
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA geom host] ec=%d ndx21=(%.6e, %.6e, %.6e) ndx31=(%.6e, %.6e, %.6e)\n",
                (int)ec,
                (double)geom[geom_base + 0], (double)geom[geom_base + 1], (double)geom[geom_base + 2],
                (double)geom[geom_base + 3], (double)geom[geom_base + 4], (double)geom[geom_base + 5]);
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA geom host] ec=%d nn=(%.6e, %.6e, %.6e) gc1=(%.6e, %.6e, %.6e) gc2=(%.6e, %.6e, %.6e)\n",
                (int)ec,
                (double)geom[geom_base + 6],  (double)geom[geom_base + 7],  (double)geom[geom_base + 8],
                (double)geom[geom_base + 9],  (double)geom[geom_base + 10], (double)geom[geom_base + 11],
                (double)geom[geom_base + 12], (double)geom[geom_base + 13], (double)geom[geom_base + 14]);
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA geom host] ec=%d Aaa=(%.6e, %.6e, %.6e) Abb=(%.6e, %.6e, %.6e) Aab=(%.6e, %.6e, %.6e)\n",
                (int)ec,
                (double)geom[geom_base + 15], (double)geom[geom_base + 16], (double)geom[geom_base + 17],
                (double)geom[geom_base + 18], (double)geom[geom_base + 19], (double)geom[geom_base + 20],
                (double)geom[geom_base + 21], (double)geom[geom_base + 22], (double)geom[geom_base + 23]);
  }

  PetscFunctionReturn(0);
}
