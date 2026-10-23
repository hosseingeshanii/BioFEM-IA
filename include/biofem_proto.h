#ifndef BIOFEM_PROTO_H
#define BIOFEM_PROTO_H
/* Prototypes of functions defined in other translation units and called without a declaration
 * (main.c). GCC >= 14 rejects implicit declarations (FASTER: GCC 14.2.0); with them declared,
 * any argument/return mismatch is caught at compile time. Signatures copied from the compiled
 * sources (CMakeLists.txt BIOFEM_COMMON). */
#include "variables.h"

/* io.c */
PetscErrorCode AreaNormal(IBMNodes *ibm);
PetscErrorCode Create(IBMNodes *ibm, FE *fem, PetscInt ibi);
PetscErrorCode Dimension(IBMNodes *ibm, PetscInt ibi);
PetscErrorCode Input(IBMNodes *ibm, PetscInt ibi);
PetscErrorCode LocationIn(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);
PetscErrorCode LocationOut(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);
PetscErrorCode Output(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);
PetscErrorCode OutputGhost(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);
PetscErrorCode Write3DShellVTK(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);

/* main.c (defined after first use) */
PetscErrorCode ContactZ(FE *fem);
PetscErrorCode FDynamic(FE *fem);
PetscErrorCode FInternal(FE *fem);
PetscErrorCode FormRK(Vec R, Vec x, Vec xn, Vec y, Vec yn, PetscReal alpha, FE *fem);
PetscErrorCode Free(FE *fem);
PetscErrorCode Init(FE *fem, PetscInt ibi);
PetscErrorCode xAccVel(FE *fem);

/* material.c */
PetscErrorCode Damp(IBMNodes *ibm, PetscReal M[9], PetscReal _C[9]);
PetscErrorCode InitMaterial(IBMNodes *ibm);
PetscErrorCode Mass(IBMNodes *ibm, PetscInt ec, PetscReal _M[9]);
PetscErrorCode MassDamp(FE *fem);

/* external.c */
PetscErrorCode EdgeDirectionalFix(PetscInt edge_n, PetscInt dir, FE *fem, Vec R);
PetscErrorCode GhostDirectionalFix(IBMNodes *ibm, PetscInt edge_n, PetscInt dir);
PetscErrorCode NodeDirectionalFix(PetscInt nb, PetscInt dir, FE *fem, Vec R);

/* bending.c, membrane.c */
PetscErrorCode Fbending(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, struct Cmpnts x1,
                        struct Cmpnts x2, struct Cmpnts x3, PetscReal _Fb[42], FE *fem);
PetscErrorCode Fmembrane(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, struct Cmpnts x1,
                         struct Cmpnts x2, struct Cmpnts x3, PetscReal _Fm[9], FE *fem);
PetscErrorCode InitGhost(FE *fem);
PetscErrorCode PatchLoc(IBMNodes *ibm);

/* bcs.c, contact.c, inverse.c, manufactured_active_strain.c */
PetscErrorCode FExternal(FE *fem);
PetscErrorCode GhostLoc(FE *fem);
PetscErrorCode ModifyFbending(FE *fem);
PetscErrorCode Fcontact(FE *fem);
PetscErrorCode InvSolver(FE *fem);
PetscErrorCode GenerateManufacturedFExternalAndKinematicsAllTimeSteps(FE *fem, PetscInt ibi);

#endif
