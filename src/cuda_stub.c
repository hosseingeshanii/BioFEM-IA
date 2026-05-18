#include "cuda_bridge.h"

PetscErrorCode InitCudaWorkspace(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyCudaWorkspace(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PrepareCudaHostWorkspace(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
          "CUDA kernel support is not enabled in this build. Rebuild with USE_CUDA=1.");
  PetscFunctionReturn(0);
}

PetscErrorCode RunCudaElementCoordKernel(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
          "CUDA kernel support is not enabled in this build. Rebuild with USE_CUDA=1.");
  PetscFunctionReturn(0);
}

PetscErrorCode RunCudaSubdivGeomKernel(FE *fem, PetscBool use_reference_coords)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  (void)use_reference_coords;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
          "CUDA kernel support is not enabled in this build. Rebuild with USE_CUDA=1.");
  PetscFunctionReturn(0);
}

PetscErrorCode RunCudaMetricTensorKernel(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
          "CUDA kernel support is not enabled in this build. Rebuild with USE_CUDA=1.");
  PetscFunctionReturn(0);
}
