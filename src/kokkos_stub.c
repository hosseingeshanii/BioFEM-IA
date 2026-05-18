#include "kokkos_bridge.h"

PetscErrorCode InitKokkosWorkspace(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyKokkosWorkspace(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode RunKokkosSubdivGeomKernel(FE *fem, PetscBool use_reference_coords)
{
  (void)fem;
  (void)use_reference_coords;
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
          "Kokkos support was not compiled in; rebuild with USE_KOKKOS=1");
}
