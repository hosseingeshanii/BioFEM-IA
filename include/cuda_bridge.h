#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include "variables.h"

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode RunCudaElementCoordKernel(FE *fem);
PetscErrorCode RunCudaSubdivGeomKernel(FE *fem, PetscBool use_reference_coords);

#ifdef __cplusplus
}
#endif

#endif
