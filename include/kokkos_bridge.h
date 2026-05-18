#ifndef KOKKOS_BRIDGE_H
#define KOKKOS_BRIDGE_H

#include "variables.h"

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode InitKokkosWorkspace(void);
PetscErrorCode DestroyKokkosWorkspace(void);
PetscErrorCode RunKokkosSubdivGeomKernel(FE *fem, PetscBool use_reference_coords);

#ifdef __cplusplus
}
#endif

#endif
