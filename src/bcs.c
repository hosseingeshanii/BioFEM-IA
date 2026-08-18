#include  "variables.h"

extern const PetscInt   dof;
extern const PetscReal  h0;
extern PetscInt         ti;
extern PetscInt         tisteps;
extern PetscInt         prescribed_force_field;
extern PetscReal        dt;

extern PetscErrorCode  EdgeFix(PetscInt edge_n, FE *fem);
extern PetscErrorCode  EdgeFree(PetscInt edge_n, FE *fem);
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
extern PetscErrorCode  FExternalPrescribedForceFieldIn(PetscInt step, FE *fem);


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

  if (prescribed_force_field) {
    PetscCall(FExternalPrescribedForceFieldIn(ti, fem));
  }

  /* Pinched-hemisphere benchmark (Simo, Rifai & Fox 1990; Asadi & Borazjani
   * 2023 Fig. A6): two pairs of equal-and-opposite point forces 2P at four
   * equatorial nodes, ramped linearly 0 -> hemi_pmax over the run so the
   * saved timestep sequence directly gives the load-displacement sweep.
   * Self-equilibrating (net force/moment = 0), so no kinematic BC needed.
   * Parallel-safe: NodeForce routes via VecSetValues + ibm_to_global_dof0,
   * followed by ONE collective VecAssemblyBegin/End covering all four
   * calls below (not one per call). */
  {
    PetscBool hemi_on = PETSC_FALSE;
    PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-hemi_pinch_test", &hemi_on, PETSC_NULL);
    if (hemi_on) {
      PetscInt  n_out1 = -1, n_in1 = -1, n_out2 = -1, n_in2 = -1;
      PetscReal pmax = 100.0;
      PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-hemi_pinch_node_out1", &n_out1, PETSC_NULL);
      PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-hemi_pinch_node_in1",  &n_in1,  PETSC_NULL);
      PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-hemi_pinch_node_out2", &n_out2, PETSC_NULL);
      PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-hemi_pinch_node_in2",  &n_in2,  PETSC_NULL);
      PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-hemi_pinch_pmax", &pmax, PETSC_NULL);
      PetscBool const_load = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-hemi_pinch_const", &const_load, PETSC_NULL);
      /* -hemi_pinch_const 1: hold the load fixed at pmax from the first
       * step, instead of ramping 0->pmax across the whole run (the
       * load-displacement-sweep design -- see the block comment above).
       * For a single-point constant-load check, not for reproducing
       * Fig. A6A's sweep curve. */
      PetscReal twoP = const_load ? 2.0 * pmax
                                   : 2.0 * pmax * (PetscReal)ti / (PetscReal)tisteps;
      if (n_out1 >= 0) { NodeForce(n_out1,  twoP, 0, fem); }
      if (n_in1  >= 0) { NodeForce(n_in1,  -twoP, 1, fem); }
      if (n_out2 >= 0) { NodeForce(n_out2, -twoP, 0, fem); }
      if (n_in2  >= 0) { NodeForce(n_in2,   twoP, 1, fem); }
      VecAssemblyBegin(fem->Fext);
      VecAssemblyEnd(fem->Fext);

      /* Edge 0 = polar hole (smaller, open), edge 1 = equator (larger,
       * open, carries the pinch loads) -- see hemisphere_mesh.py.
       *
       * Neither edge previously got any EdgeFree/EdgeFix call here, so
       * ghost DOFs on both boundaries had no Fint/Fext/Fdyn zeroing at
       * all: FDynamic's mass/damping assembly loops over n_elmt+2*n_ghosts
       * (bending.c GlobalGhostInit's ghost/bridging triangles included),
       * so ghost vertices *do* pick up a real nonzero Fdyn every step with
       * nothing here to clear it -- they were behaving as free, unpinned
       * extra unknowns instead of the passive geometry-only reflections
       * GlobalGhost() intends, which is a plausible source of drift on
       * top of any imperfect load self-equilibration.
       *
       * -hemi_pinch_fix_hole 1 additionally pins the hole edge outright
       * (removes the 6 rigid-body modes the nominally self-equilibrating
       * point loads leave free), on top of the ghost-force zeroing.
       * Equator always stays free -- that's where the loads act and the
       * benchmark's deformation is measured.
       *
       * EdgeFree(edge_n,...) ignores edge_n and zeroes every ghost's
       * Fint/Fext/Fdyn unconditionally (see external.c), so one call
       * handles both edges' ghosts regardless of fix_hole. EdgeFix(0,...)
       * additionally pins edge 0's boundary-node positions/xdd and
       * re-zeroes its own (now correctly edge-scoped) ghost block --
       * redundant with EdgeFree there, but that's harmless. */
      PetscBool fix_hole = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-hemi_pinch_fix_hole", &fix_hole, PETSC_NULL);
      EdgeFree(1, fem);
      if (fix_hole) { EdgeFix(0, fem); }
    }
  }

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
  // EdgeConstPressure(1, 1.0, 0, fem);
  // EdgeConstPressure(2, 6.e4, 1, fem);
  // EdgeConstPressure(3, -2.e4, 0, fem);
  // EdgeFree(0, fem);

  // Rectangular Plate Active Strain Test - time-varying pressure
  // {
  //   PetscReal startP = 1.e5;    /* starting pressure */
  //   PetscReal stepP  = 1.e5;    /* increase per interval */
  //   PetscInt  interval = 10;    /* timesteps between increases */
  //   PetscReal P = startP + (PetscReal)(ti/interval)*stepP;
  //   EdgeConstPressure(1, 9e5, 0, fem);
  // }
  // EdgeConstPressure(3, -9e5, 0, fem);

  // EdgeDirectionalFix(0, 1, fem);
  // EdgeDirectionalFix(3, 0, fem);

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
