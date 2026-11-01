#ifndef SOLID3D_H
#define SOLID3D_H
#include <petscsys.h>

/* 3D active-strain solid (Taylor-Hood P2-P1, DMPlex/PetscFE), src/solid3d.c. Reads its own options; called between
   PetscInitialize and PetscFinalize (biofem -model solid3d, or the standalone biofem-solid3d). */
PetscErrorCode Solid3dRun(void);

#endif
