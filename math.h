#ifndef MATH_H
#define MATH_H

#include "variables.h"  

struct Cmpnts PLUS(struct Cmpnts v1, struct Cmpnts v2);
struct Cmpnts MINUS(struct Cmpnts v1, struct Cmpnts v2);
struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
PetscReal DOT(struct Cmpnts v1, struct Cmpnts v2);
struct Cmpnts UNIT(struct Cmpnts v1);
PetscReal SIZE(struct Cmpnts v1);
struct Cmpnts AMULT(PetscReal alpha, struct Cmpnts v1);

PetscErrorCode INV(PetscReal T[3][3], PetscReal _Tinv[3][3]);
PetscErrorCode MATMULT(PetscReal A[][2], PetscReal B[][2], PetscReal C[][2]);
PetscErrorCode TRANS(PetscReal A[3][3], PetscReal _AT[3][3]);
PetscReal SIGN(PetscReal a);

#endif
