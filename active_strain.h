#ifndef ACTIVE_STRAIN_H
#define ACTIVE_STRAIN_H

typedef PetscErrorCode (*ElemFunc)(FE *fem, PetscInt ec);

PetscErrorCode SetGaussianQuadrature(FE *fem);

PetscErrorCode GetUserActParams(FE *fem);

PetscErrorCode ActDataAllocate(FE *fem);
PetscErrorCode ActDataDestroy(FE *fem);

PetscErrorCode InitActStrainProblem(FE *fem);

PetscErrorCode ElemUpdateGeom0Subdiv(FE *fem, PetscInt ec);
PetscErrorCode ElemUpdateGeomSubdiv(FE *fem, PetscInt ec);
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec);
PetscErrorCode ElemActDefGrad(FE *fem, PetscInt ec);
PetscErrorCode ElemCGDefTens(FE *fem, PetscInt ec);

PetscErrorCode ElemElasCGDefTens(FE *fem, PetscInt ec);
PetscErrorCode ElemElasStress(FE *fem, PetscInt ec);
PetscErrorCode ElemTotStress(FE *fem, PetscInt ec);
PetscErrorCode ElemElsTangMatTens(FE *fem, PetscInt ec);
PetscErrorCode ElemTotTangMatTens(FE *fem, PetscInt ec);
PetscErrorCode ModElemC33(FE *fem, PetscInt ec);
PetscErrorCode ElemUpdateG0(FE *fem, PetscInt ec);
PetscErrorCode ElemC33Solve(FE *fem, PetscInt ec);
PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec, PetscReal *Fb_out);
PetscErrorCode FInternalPreCalc(FE *fem);
PetscErrorCode FInternalAct(FE *fem);
PetscErrorCode UpdateElements(FE *fem, ElemFunc func);


#endif  /* ACTIVE_STRAIN_H */