#include  "variables.h"

extern const PetscInt   dof;
extern const PetscReal  h0;
extern PetscInt         ti;
extern PetscReal        dt;

extern PetscErrorCode  EdgeFix(PetscInt edge_n, FE *fem);
extern PetscErrorCode  NodeFix(PetscInt nb, FE *fem);
extern PetscErrorCode  EdgeSym(PetscInt edge_n, PetscInt dir, FE *fem);
extern PetscErrorCode  SurfaceConstNormalPressure(PetscReal P, FE *fem);
extern PetscErrorCode  SurfaceSinNormalPressure(PetscReal P, PetscReal char_length_x, FE *fem);
extern PetscErrorCode  SurfaceNormalPressure(PetscReal P, FE *fem);
extern PetscErrorCode  DistributedForce(PetscReal w, FE *fem);
extern PetscErrorCode  EdgeConstPressure(PetscInt edge_n, PetscReal P, PetscInt dir, FE *fem);
extern PetscErrorCode  NodeForce(PetscInt nv, PetscReal F, PetscInt dir, FE *fem);
extern PetscErrorCode  GhostFix(PetscInt edge_n, FE *fem);
extern PetscErrorCode  GhostFree(PetscInt edge_n, FE *fem);
extern PetscErrorCode  ModifyGhostFix(PetscInt edge_n, FE *fem);
extern PetscErrorCode  ModifyGhostFree(PetscInt edge_n, FE *fem);
extern PetscErrorCode  SurfaceGravity(PetscReal P, FE *fem);


PetscErrorCode GhostLoc(FE *fem) {
  
  //patch test
  /* GhostFree(0,fem); */
  /* GhostFree(1,fem); */
  /* GhostFree(2,fem); */
  /* GhostFree(3,fem); */
  
  //plate
  //GhostFix(0,fem);
  //GhostFix(1,fem);
  //GhostFix(2,fem); 
  //GhostFix(3,fem); 
  
  //roof
  /* GhostFree(0,fem); */
  /* GhostFix(1, fem); */
  /* GhostFree(2,fem); */
  /* GhostFix(3, fem); */

  //CantiStatic
  /* GhostFix(0, fem); */
  /* GhostFree(1, fem); */
  /* GhostFree(2, fem); */
  /* GhostFree(3, fem); */

  //bhv
  //GhostFree(0,fem);
  //GhostFix(1,fem);

  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode ModifyFbending(FE *fem) {
 
  //patch test
  /* ModifyGhostFree(0,fem); */
  /* ModifyGhostFree(1,fem); */
  /* ModifyGhostFree(2,fem); */
  /* ModifyGhostFree(3,fem); */
  
  //plate
  //ModifyGhostFix(0,fem); 
  //ModifyGhostFix(1,fem); 
  //ModifyGhostFix(2,fem); 
  //ModifyGhostFix(3,fem); 

  //roof
  /* ModifyGhostFree(0,fem); */
  /* ModifyGhostFix(1,fem); */
  /* ModifyGhostFree(2,fem); */
  /* ModifyGhostFix(3,fem); */

  //CantiStatic
  /* ModifyGhostFix(3, fem); */
  /* ModifyGhostFree(1, fem); */
  /* ModifyGhostFree(2, fem); */
  /* ModifyGhostFree(0, fem); */

  //bhv
  //ModifyGhostFree(0,fem);
  //ModifyGhostFix(1,fem);
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FExternal(FE *fem) {

  PetscReal  f, t;
  IBMNodes   *ibm=fem->ibm;
  PetscInt   ibi=ibm->ibi;
  
  /* //patch test */
  // EdgeFix(0, fem);
  // EdgeFix(1, fem);
  // EdgeFix(2, fem);
  // EdgeFix(3, fem);

  //rectangular_plate
  /*
  SurfaceSinNormalPressure(0.04, 0.05, fem);
  EdgeClamp(0, fem);
  EdgeClamp(1, fem);
  EdgeClamp(2, fem);
  EdgeClamp(3, fem);
  
  */
  
  //cylinder
  //NodeForce(35-1, 25000, 0, fem); //coarse
  //NodeForce(88-1, -25000, 0, fem);//coarse
  /* NodeForce(34-1, 40000, 0, fem); //fine */
  /* NodeForce(77-1, -40000, 0, fem);//fine */
  //EdgeFree(0, fem);
  //EdgeFree(1, fem);
  
  //hemi
  //coarse & fine
  /* NodeForce(244-1, 200, 0, fem); */
  /* NodeForce(81-1, -200, 0, fem); */
  /* NodeForce(90-1, 200, 1, fem); */
  /* NodeForce(163-1, -200, 1, fem); */
  //EdgeFree(0, fem);
  //EdgeFree(1, fem);

  //Cant_Static
  //SurfaceNormalPressure(25., fem); //FSI test
  //EdgeClamp(3, fem);
  //EdgeClamp(3, fem);
  /* f = (5.4e-5)/3.; */
  /* NodeForce(28-1, f, 2, fem); */
  /* NodeForce(56-1, f, 2, fem); */
  /* NodeForce(25-1, f, 2, fem); */
 

  //Cant_Manufacture
  /* DistributedForce(8., fem); */
  /* EdgeClamp(3, fem); */

  //Cant_Dynamic
  /* t=ti*dt; */
  /* f=1.e-5*cos(8.0*t)/3.; */
  /* EdgeClamp(3, fem); */
  /* NodeForce(28-1, f, 2, fem); */
  /* NodeForce(56-1, f, 2, fem); */
  /* NodeForce(25-1, f, 2, fem); */

  //Cant_base
  /* EdgeFree(0, fem); */
  /* EdgeFree(1, fem); */
  /* EdgeFree(2, fem); */
  //EdgeClamp(3, fem);

  //sphere_inflation
  //SurfaceNormalPressure(.4, fem);

  //bhv
  //SurfaceNormalPressure(15998.7, fem); //15998.7, 10665.8, 5332.9
  //CardiacPressure(fem);
  //EdgeFix(1, fem);
  
  //biax
  /* PetscReal ratio=1.0;  PetscInt step=3000; */
  /* if(             ti<step+1) ratio=0.025; */
  /* if(ti>step   && ti<2*step+1) ratio=0.05; */
  /* if(ti>2*step && ti<3*step+1) ratio=0.1; */
  /* if(ti>3*step && ti<4*step+1) ratio=0.2; */
  /* if(ti>4*step && ti<5*step+1) ratio=0.4; */
  /* if(ti>5*step && ti<6*step+1) ratio=0.7; */
  /* if(ti>6*step && ti<7*step+1) ratio=1.0; */

  // EdgeConstPressure(0, -6.e4, 1, fem);
  EdgeConstPressure(1, 2, 0, fem);
  // EdgeConstPressure(2, 6.e4, 1, fem);
  // EdgeConstPressure(3, -2.e4, 0, fem);
  EdgeFree(0, fem);
  
  /* EdgeFree(1, fem); */
  /* EdgeFree(2, fem); */
  /* EdgeFree(3, fem); */

  //sphere-plate
  /* SurfaceGravity(980.665, fem); */
  /* if (ibi==0)  EdgeFix(0, fem); */

  //cloth
  //SurfaceGravity(9.81, fem);
  //NodeFix(58, fem);  NodeFix(424, fem); NodeFix(801, fem);  NodeFix(808, fem);  NodeFix(363, fem);  NodeFix(1347, fem); NodeFix(1147, fem); //coarse
  //NodeFix(122, fem);  NodeFix(1558, fem); NodeFix(3089, fem);  NodeFix(3090, fem);  NodeFix(1433, fem);  NodeFix(4882, fem); NodeFix(4883, fem); //fine

  return(0);
}
