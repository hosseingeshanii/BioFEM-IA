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
PetscErrorCode RunCudaMetricTensorKernel(FE *fem);

#ifdef __cplusplus
}
#endif

#endif
