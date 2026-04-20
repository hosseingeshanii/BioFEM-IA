#ifndef MANUFACTURED_ACTIVE_STRAIN_H
#define MANUFACTURED_ACTIVE_STRAIN_H

#include "variables.h"

extern PetscInt  newmark;
extern PetscReal manufactured_gamma0, manufactured_T;

PetscErrorCode GetManufacturedActiveStrainOptions(void);
PetscErrorCode ManufacturedActiveStrainKinematics(FE *fem, PetscReal t);
PetscErrorCode ManufacturedInitialKinematicsIn(PetscInt step, FE *fem);
PetscErrorCode GenerateManufacturedFExternalAndInitialKinematics(FE *fem, PetscInt ibi, const char *subdir);

#endif  /* MANUFACTURED_ACTIVE_STRAIN_H */
