static char help[] = "Hosein FEM Petsc 3.6.2 \n\n";

#include  "variables.h"

PetscReal  E=0.0, mu=0.0, rho=0.0, h0=0.0, dt=0.0, dampfactor=0.0, char_length_x=1.0, char_length_y=1.0, char_length_z=1.0;
PetscInt   dof=3, twod=0, damping=0, membrane=0, bending=0, outghost=0, ConstitutiveLawNonLinear=0;
PetscInt   timeinteg=0, nbody=1, contact=0, explicit=0;
PetscInt   ec, nc, ti, tiout, tistart=0, rstart_flg, tisteps=1, curvature=6, manufactured=0, residual=1;

PetscErrorCode  FormFunctionFEM(SNES snes, Vec x, Vec R, void *ctx);
extern PetscErrorCode InitVel(PetscInt edge_n, PetscReal w, FE *fem);
extern PetscErrorCode MoveBoundary(PetscInt edge_n, FE *fem);
extern PetscErrorCode  ConstantVel(PetscReal vel, PetscInt dir, FE *fem);

FE         *fem; 
IBMNodes   *ibm;
PetscMalloc(nbody*sizeof(FE), &fem);
PetscMalloc(nbody*sizeof(IBMNodes), &ibm);
   
int main(int argc, char **argv)
{  
  PetscInitialize(&argc, &argv, (char*)0, help);

  PetscOptionsInsertFile(PETSC_COMM_WORLD, PETSC_NULL, "control.dat", PETSC_TRUE);
  
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nbody", &nbody, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-E", &E, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-mu", &mu, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-rho", &rho, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-h0", &h0, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_x", &char_length_x, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_y", &char_length_y, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_z", &char_length_z, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tisteps", &tisteps, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tistart", &tistart, rstart_flg);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-dt", &dt, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-timeinteg", &timeinteg, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-explicit", &explicit, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-damping", &damping, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-dampfactor", &dampfactor, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tiout", &tiout, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-twod", &twod, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-membrane", &membrane, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-bending", &bending, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-contact", &contact, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-ConstitutiveLawNonLinear", &ConstitutiveLawNonLinear, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-outghost", &outghost, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-curvature", &curvature, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-manufactured", &manufactured, PETSC_NULL);
 
  PetscInt   ibi, k;
  PetscReal  tcyc=0.76, t, alpha[4];  alpha[0] = 0.25;  alpha[1] = 1./3.;  alpha[2] = 0.5;  alpha[3] = 1.0; 
  SNES       snes;  
  Mat        J; 
  
  
  
  for (ibi=0; ibi<nbody; ibi++) { 
    Dimension(&ibm[ibi], ibi);
    Create(&ibm[ibi], &fem[ibi], ibi);
    Input(&ibm[ibi], ibi);
    Init(&fem[ibi], ibi);
    ContactZ(&fem[ibi]);

    if (explicit)  {VecSet(fem[ibi].Mass, 0.0);  VecSet(fem[ibi].Dissip, 0.0);  MassDamp(&fem[ibi]);}
    if (tistart)  {LocationIn(&fem[ibi], tistart, ibi);}
  }
  if(residual){
    
    double E_epsilon[ibm->n_v + ibm->n_ghosts] = {};
    
    // Initialize Python
    initialize_python();
    
    int ITER_MLP=0;
    for (ibi=0; ibi<nbody; ibi++) {  
      for (int k; k<ELS_Itrs; k++){
        for (int i=0;i<ibm->n_v + ibm->n_ghosts; i++){
          ResidualCalc(E_epsilon, ibi);
        //PetscPrintf(PETSC_COMM_WORLD, "body:%d, Res on node 10=%le\n", ibi, (fem[ibi].Res)[10]);
        
        PetscScalar *RRes, *FFinter;
        VecGetArray(fem[ibi].Res, &RRes);
        //VecGetArray(fem[ibi].Fint, &FFinter);
        /*
        PetscPrintf(PETSC_COMM_SELF, "Residual!\n");
        for (i=0; i<ibm->n_v; i++){
          PetscPrintf(PETSC_COMM_SELF, "%le %le %le\n", RRes[i*dof], RRes[i*dof+1], RRes[i*dof+2]);
        }
        
        */
        
        // Converting petsc array to a normal c array to pass it to the python script
        /*
        double R_array[(ibm->n_v+ibm->n_ghosts)*dof+2];        
        for (int i = 0; i < ibm->n_v; i++) {
          R_array[i*dof] = (double)RRes[i*dof];
          R_array[i*dof+1] = (double)RRes[i*dof+1];
          R_array[i*dof+2] = (double)RRes[i*dof+2];
        }
        */
        
        
        // get the inputs features (coordinates XM, YM)
        //initial location

        /*
        for (int i=0; i<ibm->n_v+ibm->n_ghosts; i++){
          for (int j=0; j<dof; j++){
            inputs[j*2+i*2*dof] = ibm->x_bp0[i];
            inputs[j*2+1+i*2*dof] = ibm->y_bp0[i];
          }
        }
        */

        for (int j=0; j<3; j++){
            double point_res[1]={(double)RRes[i*dof+j]};
            double point_input[2] = {ibm->x_bp0[i], ibm->y_bp0[i]};
            update_weights_in_python(point_input, point_res, ibi, i, ibm->n_v + ibm->n_ghosts);
        }
        
        VecRestoreArray(fem[ibi].Res, &RRes); // Restore array
        VecRestoreArray(fem[ibi].Fint, &FFinter); // Restore array
     
        }
        
        //Prediction(); 
        //double res_sum = predict_output(fem[ibi].Res, &fem[ibi], dof);
        //double res_sum = 0.0;
        
        //predict_output(fem[ibi].Res, &fem[ibi], dof, &res_sum);
        //PetscPrintf(PETSC_COMM_SELF, "Predicted reidual sum: %lf\n", res_sum);        
      }
    }
    // Finalize Python
    finalize_python();
  }
  if (tistart) tistart++;
  if (tistart==0) tisteps ++;

  for (ti=tistart; ti<tistart+tisteps; ti++) {

    for (ibi=0; ibi<nbody; ibi++) {
      PetscPrintf(PETSC_COMM_WORLD, "body:%d Time(%d)=%le\n", ibi, ti, ti*dt);
      // if (contact==1) {ContactZ(&fem[ibi]);}   
      

      if (explicit) {
	//------------------Explicit RK Solver
	for (k=0; k<4; k++) {	  
	  FormRK(fem[ibi].Res, fem[ibi].x, fem[ibi].xn, fem[ibi].y, fem[ibi].yn, alpha[k], &fem[ibi]);
	}
	
      } else {  
	//------------------Implicit SNES solver
	/* Typical options
	   -fem_snes_mf
	   #-fem_snes_type tr
	   -fem_snes_max_it 20
	   -fem_snes_monitor
	   #-fem_ksp_type fgmres	 */
	Vec U;
        VecDuplicate(fem[ibi].x, &U);
        VecCopy(fem[ibi].x, U);
        //SNES
        SNESCreate(PETSC_COMM_SELF, &snes);
        SNESSetFunction(snes, fem[ibi].Res, FormFunctionFEM, (void *)&fem[ibi]);
        SNESAppendOptionsPrefix(snes, "fem_");
        SNESSetFromOptions(snes);
        MatCreateSNESMF(snes, &J);  //MatrixFree
        SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, (void *)&fem[ibi]);
        SNESSolve(snes, PETSC_NULL,U); //Cannot pass fem[ibi].x

        VecCopy(U, fem[ibi].x);
        SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J);
      }
      VecCopy(fem[ibi].x, fem[ibi].y); //for contact energy calculations
      xAccVel(&fem[ibi]);
      //VecCopy(fem[ibi].xd, fem[ibi].y);  

       if (contact==1) {ContactZ(&fem[ibi]);}   
    } //ibi

    t = dt*(ti);
    //  t = t - ((PetscInt)(t/tcyc))*tcyc;
    //if (contact==1 && t>0.03) {Fcontact(fem);} //dynamic_bhv
    //if (contact==2 && t>0.075) {Fcontact2(fem);} //dynamic_bhv
    if (contact) {Fcontact(fem);} //static_bhv
    
    for (ibi=0; ibi<nbody; ibi++) {
      Contact(&fem[ibi]);
      VecCopy(fem[ibi].xn, fem[ibi].xnm1);
      VecCopy(fem[ibi].x, fem[ibi].xn);

      PetscReal norm=0.0, normv=0.0, norma=0.0, normfint=0.; 
      VecCopy(fem[ibi].x, fem[ibi].dx);
      VecAXPY(fem[ibi].dx, -1., fem[ibi].xnm1);
      VecNorm(fem[ibi].dx, NORM_2, &norm);
      VecNorm(fem[ibi].xd, NORM_INFINITY, &normv);
      VecNorm(fem[ibi].xdd, NORM_INFINITY, &norma);
      VecNorm(fem[ibi].Fint, NORM_INFINITY, &normfint);
      PetscPrintf(PETSC_COMM_SELF, "body:%d Norm(x-xn)= %le Vel %f Acc %f Fint %f\n",ibi,norm,normv,norma, normfint);
    }

    //    if (contact) {Fcontact(fem);} //static_bhv
    // Printout the results
    //if (ti!=0 && ti == (ti/tiout)*tiout){
    if (ti == (ti/tiout)*tiout){
      for (ibi=0; ibi<nbody; ibi++) {
	Output(&fem[ibi], ti, ibi);
	//LocationOut(&fem[ibi], ti, ibi);
	if (outghost) {OutputGhost(&fem[ibi], ti, ibi);}
      }
    }
  }// ti
  
  //Finish UP
  for (ibi=0; ibi<nbody; ibi++) {
    LocationOut(&fem[ibi], ti-1, ibi);
    //Output(&fem[ibi], ti, ibi);
    if(outghost){OutputGhost(&fem[ibi], ti, ibi);}
    Free(&fem[ibi]);
  }

  PetscFinalize();
  
  return(0);
}


void compute_R(double* E_epsilon, int ibi, double* residuals) {
    int size = ibm[ibi].n_v + ibm[ibi].n_ghosts;

    // Call ResidualCalc function
    ResidualCalc(E_epsilon, ibi);
    
    // Access the PETSc vector
    PetscScalar *residual_array;
    VecGetArray(fem[ibi].Res, &residual_array);

    // Copy the residuals from the PETSc vector to the output array
    for (int i = 0; i < size; i++) {
        residuals[i] = residual_array[i];
    }

    // Restore the PETSc vector
    VecRestoreArray(fem[ibi].Res, &residual_array);
}



PetscErrorCode ResidualCalc(double* E_epsilon, int ibi){
  // We have to take x_n+1 and x_n from last saved iteration to calculate the strains
  
  IBMNodes   *ibm=fem[ibi].ibm;
  PetscInt   nv, ec;
  PetscReal  *xx;
  //---------Update the location
  VecGetArray(fem[ibi].x, &xx);
  for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
      ibm->x_bp[nv] = xx[nv*dof  ];
      ibm->y_bp[nv] = xx[nv*dof+1];
      ibm->z_bp[nv] = xx[nv*dof+2];
      if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}  //2d case
    /* } */
  }
  VecRestoreArray(fem[ibi].x, &xx);

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {     
      PatchLoc(ibm); 
      GhostLoc(fem);
    } else if (curvature==6) {
      //GlobalGhost(ibm);
    }
  }
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0);  VecSet(fem->Fdyn, 0.0);
  VecSet(fem[ibi].Res, 0.0);  VecSet(fem->FJ, 0.0);  
  for (nv=0; nv<ibm->n_v; nv++)  fem->IE[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;
  PetscPrintf(PETSC_COMM_SELF, "Check FINTERNAL!\n");
  FInternal(E_epsilon, fem);
  if(tisteps>1) {FDynamic(fem);}
  FExternal(fem);

  VecWAXPY(fem[ibi].Res,-1., fem->Fext, fem->Fint);
  VecAXPY(fem[ibi].Res,1., fem->Fdyn);
  return(0);

  }
//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FormRK(Vec R, Vec x, Vec xn, Vec y, Vec yn, PetscReal alpha, FE *fem) {

  //y=dx/dt  
  IBMNodes   *ibm=fem->ibm;
  PetscReal  *xx, *RR, *RRes, *FF;
  PetscInt   nv;
  Vec        w1, w2;

  VecDuplicate(R, &w1);
  VecDuplicate(R, &w2);

  //---------Update the location 
  VecGetArray(x, &xx);
  for (nv=0; nv<ibm->n_v+ibm->n_ghosts; nv++) {
    ibm->x_bp[nv] = xx[nv*dof  ];
    ibm->y_bp[nv] = xx[nv*dof+1];
    ibm->z_bp[nv] = xx[nv*dof+2];
    if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}  //2d case
  }
  VecRestoreArray(x, &xx);

  //---------------------------Patchtest---------------------------------------------------------------------
  /* PetscInt i, start, end, edge_n, edge; */
  /* for (edge_n=0; edge_n<ibm->n_edge; edge_n++){ */
  /*   start = 0; end = 0; */
  /*   for (edge=0; edge<edge_n+1; edge++) { */
  /*     end += ibm->n_bnodes[edge]; */
  /*   } */
  /*   start = end - ibm->n_bnodes[edge_n]; */

  /*   for (i=start; i<end; i++) { //moving boundary nodes */
  /*     nv = ibm->bnodes[i]; */
  /*     ibm->x_bp[nv] = 2.*ibm->x_bp0[nv]; */
  /*     ibm->y_bp[nv] = ibm->y_bp0[nv]; */
  /*     ibm->z_bp[nv] = ibm->z_bp0[nv]; */
  /*   } */
  /* } */

  /* for (nv=ibm->n_v; nv<ibm->n_v+ibm->n_ghosts; nv++) { //moving ghost nodes */
  /*   ibm->x_bp[nv] = 2.*ibm->x_bp0[nv]; */
  /*   ibm->y_bp[nv] = ibm->y_bp0[nv]; */
  /*   ibm->z_bp[nv] = ibm->z_bp0[nv]; */
  /* } */
  //---------------------------Patchtest---------------------------------------------------------------------

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {     
      PatchLoc(ibm); 
      GhostLoc(fem);
    } else if (curvature==6) {
      //GlobalGhost(ibm);
    }
  }
  
  //---------Compute Forces then Residual
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0); 
  VecSet(R, 0.0);  VecSet(fem->FJ, 0.0); 

  FInternal(fem);
  FExternal(fem);

  VecWAXPY(R, -1., fem->Fint, fem->Fext);
  
  VecPointwiseMult(w1, fem->Dissip,y);
  VecAXPY(R, -1., w1);
  VecPointwiseDivide(w2, R, fem->Mass);
  VecCopy(w2, R);

  VecWAXPY(x, alpha*dt, y, xn);
  VecWAXPY(y, alpha*dt, R, yn);

  //---------2d case
  if(twod){
    VecGetArray(x, &xx);
    VecGetArray(R, &RR); 
    for (nv=0; nv<ibm->n_v; nv++) {
      xx[nv*dof+2] = ibm->z_bp0[nv];  
      RR[nv*dof+2] = 0.0;
    }
  VecRestoreArray(R, &RR);
  VecRestoreArray(x, &xx);
  }
  
  //---------------------------Patchtest---------------------------------------------------------------------
  /* VecGetArray(x, &xx); */
  /* for (edge_n=0; edge_n<ibm->n_edge; edge_n++){ */
  /*   start = 0; end = 0; */
  /*   for (edge=0; edge<edge_n+1; edge++) { */
  /*     end += ibm->n_bnodes[edge]; */
  /*   } */
  /*   start = end - ibm->n_bnodes[edge_n]; */

  /*   for (i=start; i<end; i++) { */
  /*     nv = ibm->bnodes[i]; */
  /*     xx[nv*dof] = 2.*ibm->x_bp0[nv]; */
  /*     xx[nv*dof+1] = ibm->y_bp0[nv]; */
  /*     xx[nv*dof+2] = ibm->z_bp0[nv]; */
  /*   } */
  /* } */

  /* for (nv=ibm->n_v; nv<ibm->n_v+ibm->n_ghosts; nv++) { */
  /*   xx[nv*dof] = 2.*ibm->x_bp0[nv]; */
  /*   xx[nv*dof+1] = ibm->y_bp0[nv]; */
  /*   xx[nv*dof+2] = ibm->z_bp0[nv]; */
  /* } */

  /* VecRestoreArray(x, &xx); */
  //---------------------------Patchtest---------------------------------------------------------------------

  VecDestroy(&w1);  VecDestroy(&w2);
  
  return(0);
}

//-----------------------------------------------------------------------------------------------
PetscErrorCode FormFunctionFEM(SNES snes, Vec x, Vec R,void *ctx) {

  FE         *fem=(FE *)ctx;
  IBMNodes   *ibm=fem->ibm;
  PetscReal  *xx,*RR, *RRes,*FF;
  PetscInt   nv, ec;

 
  //---------Update the location
  VecGetArray(x, &xx);
  for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
    /* if (ibm->contact[nv]) { // added by Iman 10/12/22 to fix x for nodes in contact as BC */
    /*   xx[nv*dof  ] = ibm->x_bp[nv]; */
    /*   xx[nv*dof+1] = ibm->y_bp[nv]; */
    /*   xx[nv*dof+2] = ibm->z_bp[nv]; */
    /* } else { */
      ibm->x_bp[nv] = xx[nv*dof  ];
      ibm->y_bp[nv] = xx[nv*dof+1];
      ibm->z_bp[nv] = xx[nv*dof+2];
      if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}  //2d case
    /* } */
  }
  VecRestoreArray(x, &xx);

  VecCopy(x, fem->x);

  //for displacement BCs
  //MoveBoundary(3, fem);
  //

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {
      PatchLoc(ibm);
      GhostLoc(fem);
    } else if (curvature==6) {
      //GlobalGhost(ibm);
    }
  }

  //---------Compute Forces then Residual
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0);  VecSet(fem->Fdyn, 0.0);
  VecSet(R, 0.0);  VecSet(fem->FJ, 0.0);  
  for (nv=0; nv<ibm->n_v; nv++)  fem->IE[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;

  FInternal(fem);
  if(tisteps>1) {FDynamic(fem);}
  FExternal(fem);

  VecWAXPY(R,-1., fem->Fext, fem->Fint);
  VecAXPY(R,1., fem->Fdyn);

  //---------2d case
  if(twod){
    VecGetArray(fem->x, &xx);
    VecGetArray(R, &RR);
    for (nv=0; nv<ibm->n_v; nv++) {
      xx[nv*dof+2]=ibm->z_bp0[nv];
      RR[nv*dof+2]=0.0;
    }
    VecRestoreArray(R, &RR);
    VecRestoreArray(fem->x, &xx);
  }

  //for displacement BCs
  //MoveBoundary(3, fem);
  //
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FInternal(double* E_epsilon, FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscInt       i;
  PetscReal      Fm[9], Fb[42], Fint[9];
  struct Cmpnts  x1, x2, x3, X1, X2, X3;
  PetscInt       n1e, n2e, n3e, n4e, n5e, n6e, j;
  PetscReal      s1=0., s2=0., s3=0.;

  for (i=0; i<9; i++) {Fm[i]=0.0; Fint[i]=0.0;}
  for (i=0; i<42; i++) {Fb[i]=0.0;}

  PetscReal  *FF,*FFJ;
  VecGetArray(fem->Fint, &FF);
  VecGetArray(fem->FJ, &FFJ);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
    n4e=ibm->nv4[ec];n5e=ibm->nv5[ec];n6e=ibm->nv6[ec];
    
    //current location
    x1.x=ibm->x_bp[n1e]; x1.y=ibm->y_bp[n1e]; x1.z=ibm->z_bp[n1e];
    x2.x=ibm->x_bp[n2e]; x2.y=ibm->y_bp[n2e]; x2.z=ibm->z_bp[n2e];
    x3.x=ibm->x_bp[n3e]; x3.y=ibm->y_bp[n3e]; x3.z=ibm->z_bp[n3e];
    //initial location
    X1.x=ibm->x_bp0[n1e]; X1.y=ibm->y_bp0[n1e]; X1.z=ibm->z_bp0[n1e];
    X2.x=ibm->x_bp0[n2e]; X2.y=ibm->y_bp0[n2e]; X2.z=ibm->z_bp0[n2e];
    X3.x=ibm->x_bp0[n3e]; X3.y=ibm->y_bp0[n3e]; X3.z=ibm->z_bp0[n3e];

    if (curvature==1) {
      if (membrane) {
	Fmembrane(ec, X1, X2, X3, x1, x2, x3, Fm, fem);
      }
    }

    if (bending) {
      Fbending(E_epsilon, ec, X1, X2, X3, x1, x2, x3, Fb, fem);
    }

    if (curvature==1) { //Only bending     
      for (i=0; i<9; i++) {
	Fint[i]=Fm[i]+Fb[i];
      }  //end loop over nodes of each element
      
      FF[n1e*dof] +=Fint[0];
      FF[n1e*dof+1] +=Fint[1];
      FF[n1e*dof+2] +=Fint[2];
      
      FF[n2e*dof] +=Fint[3];
      FF[n2e*dof+1] +=Fint[4];
      FF[n2e*dof+2] +=Fint[5];
      
      FF[n3e*dof] +=Fint[6];
      FF[n3e*dof+1] +=Fint[7];
      FF[n3e*dof+2] +=Fint[8];     
  
      if(n4e !=1000000){ // Front node is inside domain
	FF[n4e*dof] +=Fb[9];
	FF[n4e*dof+1] +=Fb[10];
	FF[n4e*dof+2] +=Fb[11];
      }else{ // Front node is ghost
	FFJ[n1e*dof] +=Fb[9];
	FFJ[n1e*dof+1] +=Fb[10];
	FFJ[n1e*dof+2] +=Fb[11];
      }
      
      if(n5e !=1000000){// Front node is inside domain
	FF[n5e*dof] +=Fb[12];
	FF[n5e*dof+1] +=Fb[13];
	FF[n5e*dof+2] +=Fb[14];
      }else{ // Front node is ghost
	FFJ[n2e*dof] +=Fb[12];
	FFJ[n2e*dof+1] +=Fb[13];
	FFJ[n2e*dof+2] +=Fb[14];
      }
      
      if(n6e !=1000000){// Front node is inside domain
	FF[n6e*dof] +=Fb[15];
	FF[n6e*dof+1] +=Fb[16];
	FF[n6e*dof+2] +=Fb[17];
      }else{ // Front node is ghost
	FFJ[n3e*dof] +=Fb[15];
	FFJ[n3e*dof+1] +=Fb[16];
	FFJ[n3e*dof+2] +=Fb[17];
      }

    } else if (curvature==6) {
      
      PetscInt  node, v = ibm->val[ec];

      for (i=0; i<(v+6); i++) {
      	if (ibm->patch[16*ec+i]!=1000000) {
      	  node = ibm->patch[16*ec+i];

      	  FF[dof*node] += Fb[dof*i];
      	  FF[dof*node+1] += Fb[dof*i+1];
      	  FF[dof*node+2] += Fb[dof*i+2];
      	 
      	}
      }      
    }
  }//end loop over elements
  
  VecRestoreArray(fem->FJ, &FFJ); 
  VecRestoreArray(fem->Fint, &FF); 
  
  if (bending && curvature==1) {ModifyFbending(fem);}
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FDynamic(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscInt       i;
  PetscReal      M[9], C[9], Fd[9], x[9], xn[9], xnm1[9], xd[9], xdd[9];
  PetscInt       n1e, n2e, n3e;
  PetscReal      Gama=0.5, Beta=0.25;
  PetscReal      M1, C1, M2, C2, M3, C3;

  for (i=0; i<9; i++) {M[i]=0.0; C[i]=0.0; Fd[i]=0.0;}
  M1=1./(Beta*pow(dt,2));  M2=1./(Beta*dt);  M3=(1./(2*Beta))-1.;
  C1=Gama/(Beta*dt);  C2=(Gama/Beta)-1;  C3=dt*(Gama/(2.*Beta)-1);

  PetscReal  *xx,*xxn,*xxnm1,*xxd,*xxdd,*FF,facc,fvel;
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->xnm1, &xxnm1);
  VecGetArray(fem->x, &xx);
  VecGetArray(fem->Fdyn, &FF);

  for (ec=0; ec<ibm->n_elmt+2*ibm->n_ghosts; ec++) {
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
  
    x[0]=ibm->x_bp[n1e];  x[1]=ibm->y_bp[n1e]; x[2]=ibm->z_bp[n1e];
    x[3]=ibm->x_bp[n2e];  x[4]=ibm->y_bp[n2e]; x[5]=ibm->z_bp[n2e];
    x[6]=ibm->x_bp[n3e];  x[7]=ibm->y_bp[n3e]; x[8]=ibm->z_bp[n3e];
           
    xn[0]=xxn[n1e*dof];  xn[1]=xxn[n1e*dof+1];  xn[2]=xxn[n1e*dof+2];
    xn[3]=xxn[n2e*dof];  xn[4]=xxn[n2e*dof+1];  xn[5]=xxn[n2e*dof+2];
    xn[6]=xxn[n3e*dof];  xn[7]=xxn[n3e*dof+1];  xn[8]=xxn[n3e*dof+2];

    xnm1[0]=xxnm1[n1e*dof];  xnm1[1]=xxnm1[n1e*dof+1];  xnm1[2]=xxnm1[n1e*dof+2];
    xnm1[3]=xxnm1[n2e*dof];  xnm1[4]=xxnm1[n2e*dof+1];  xnm1[5]=xxnm1[n2e*dof+2];
    xnm1[6]=xxnm1[n3e*dof];  xnm1[7]=xxnm1[n3e*dof+1];  xnm1[8]=xxnm1[n3e*dof+2];

    xd[0]=xxd[n1e*dof];  xd[1]=xxd[n1e*dof+1];  xd[2]=xxd[n1e*dof+2];
    xd[3]=xxd[n2e*dof];  xd[4]=xxd[n2e*dof+1];  xd[5]=xxd[n2e*dof+2];
    xd[6]=xxd[n3e*dof];  xd[7]=xxd[n3e*dof+1];  xd[8]=xxd[n3e*dof+2];

    xdd[0]=xxdd[n1e*dof];  xdd[1]=xxdd[n1e*dof+1];  xdd[2]=xxdd[n1e*dof+2];
    xdd[3]=xxdd[n2e*dof];  xdd[4]=xxdd[n2e*dof+1];  xdd[5]=xxdd[n2e*dof+2];
    xdd[6]=xxdd[n3e*dof];  xdd[7]=xxdd[n3e*dof+1];  xdd[8]=xxdd[n3e*dof+2];
     
    Mass(ibm, ec, M);
    if(damping) {Damp(ibm, M,C);}

    if(timeinteg==0){ //Newmark constant average acceleration
      for (i=0; i<9; i++) {
	facc=M[i]*(M1*xn[i]+M2*xd[i]+M3*xdd[i]);
	fvel=C[i]*(C1*xn[i]+C2*xd[i]+C3*xdd[i]);
	Fd[i]=M[i]*M1*x[i]+C[i]*C1*x[i]-facc-fvel;
      }

    }else if (timeinteg==1){ //central
      for (i=0; i<9; i++) {
	Fd[i]=M[i]*(x[i]-2*xn[i]+xnm1[i])/(dt*dt)+C[i]*(x[i]-xn[i])/dt;
      }
    }
    //AddToVector
    
    FF[n1e*dof] +=Fd[0];
    FF[n1e*dof+1] +=Fd[1];
    FF[n1e*dof+2] +=Fd[2];
    
    FF[n2e*dof] +=Fd[3];
    FF[n2e*dof+1] +=Fd[4];
    FF[n2e*dof+2] +=Fd[5];

    FF[n3e*dof] +=Fd[6];
    FF[n3e*dof+1] +=Fd[7];
    FF[n3e*dof+2] +=Fd[8];
   
  }//end loop over elements

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->xnm1, &xxnm1);
  VecRestoreArray(fem->xn, &xxn);
  VecRestoreArray(fem->Fdyn, &FF);

  return(0);
} 

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode xAccVel(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscReal      M1, C1, M2, C2, M3, C3, M[9], C[9];
  PetscReal      *xx, *xxn, *xxd, *xxdd;
  PetscReal      xxddn1, xxddn2, xxddn3;
  PetscReal      Gama=0.5, Beta=0.25;
  PetscInt       nv;
  
  M1 = 1./(Beta*pow(dt,2));  M2 = 1./(Beta*dt);  M3 = (1./(2*Beta)) - 1.;
  // C1 = Gama/(Beta*dt);  C2 = (Gama/Beta) - 1;  C3 = dt*(Gama/(2.*Beta) - 1);
  C2=(1.-Gama)*dt; C3=Gama*dt;
  
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->x, &xx);

  for (nv=0; nv<ibm->n_v+ibm->n_ghosts; nv++) {
    xxddn1= xxdd[dof*nv  ];
    xxddn2= xxdd[dof*nv+1];
    xxddn3= xxdd[dof*nv+2];

    xxdd[dof*nv  ] = M1*(xx[nv*dof  ] - xxn[nv*dof  ]) - M2*xxd[nv*dof  ] - M3*xxdd[nv*dof  ];
    xxdd[dof*nv+1] = M1*(xx[nv*dof+1] - xxn[nv*dof+1]) - M2*xxd[nv*dof+1] - M3*xxdd[nv*dof+1];
    xxdd[dof*nv+2] = M1*(xx[nv*dof+2] - xxn[nv*dof+2]) - M2*xxd[nv*dof+2] - M3*xxdd[nv*dof+2];

    //  if (ibm->contact[nv]<1 || ibm->ibi==0) {   
    // Iman 10/25/22 wrong Newmark - fix
    // xdot(n+1) = xdot(n) + (1-gamma)xddot(n) + gamma xddot(n+1)
    /* xxd[dof*nv  ] = C1*(xx[nv*dof  ]-xxn[nv*dof  ]) - C2*xxd[nv*dof  ] - C3*xxdd[nv*dof  ]; */
    /* xxd[dof*nv+1] = C1*(xx[nv*dof+1]-xxn[nv*dof+1]) - C2*xxd[nv*dof+1] - C3*xxdd[nv*dof+1]; */
    /* xxd[dof*nv+2] = C1*(xx[nv*dof+2]-xxn[nv*dof+2]) - C2*xxd[nv*dof+2] - C3*xxdd[nv*dof+2];  */
    xxd[dof*nv  ] = xxd[nv*dof  ] + C2*xxddn1 + C3*xxdd[nv*dof  ];
    xxd[dof*nv+1] = xxd[nv*dof+1] + C2*xxddn2 + C3*xxdd[nv*dof+1];
    xxd[dof*nv+2] = xxd[nv*dof+2] + C2*xxddn3 + C3*xxdd[nv*dof+2];
    
    //  }     
  }

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->xn, &xxn);

  //
  if (ti==0) {
    //if (ibm->ibi==0)  ConstantVel(12.649, 0, fem);
    //if (ibm->ibi==1)  ConstantVel(-25.298, 0, fem);
   // if (ibm->ibi==0)  ConstantVel(20., 0, fem);
   // if (ibm->ibi==1)  ConstantVel(-20., 0, fem);
  }
  //

  if (explicit) {VecCopy(fem->y, fem->yn);}
  
  return(0);
}

//-------------------------------------------------------------------------------------------------
PetscErrorCode Init(FE *fem, PetscInt ibi) {
  
  IBMNodes   *ibm = fem->ibm;
  PetscReal  *xx, mass;
  PetscInt   nv, ec, n1e, n2e, n3e;

  if (curvature==6) {GlobalGhostInit(ibm);}

  AreaNormal(ibm);
  for (ec=0; ec<ibm->n_elmt + 2*ibm->n_ghosts; ec++) {
    ibm->dA0[ec] = ibm->dA[ec]; 
    ibm->Nf_x[ec] = ibm->nf_x[ec];  ibm->Nf_y[ec] = ibm->nf_y[ec];  ibm->Nf_z[ec] = ibm->nf_z[ec]; 
  }

  ibm->rho = rho;
  ibm->h0 = h0;
  //
  /* if (ibm->ibi==0) { */
  /*   //rho = 1.;  h0 = 0.003; */
  /*   rho = 0.05; */
  /* } else { */
  /*   //rho = 10.;  h0 = 0.03; */
  /*   rho = 0.15; */
  /* } */
  //

  for (nv=0; nv<ibm->n_v; nv++)  ibm->m[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++) {
    mass = 1./3.*rho*ibm->dA0[ec]*h0;
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
    ibm->m[n1e] += mass;  ibm->m[n2e] += mass;  ibm->m[n3e] += mass;
  }
  
  if (bending){InitGhost(fem);}

  VecGetArray(fem->x, &xx);
  for (nv=0; nv<ibm->n_v+ibm->n_ghosts; nv++) {
    xx[nv*dof] = ibm->x_bp0[nv];
    xx[nv*dof+1] = ibm->y_bp0[nv];
    xx[nv*dof+2] = ibm->z_bp0[nv];
  }

 for (ec=0; ec<ibm->n_elmt; ec++) {
    ibm->g1[dof*ec] = ibm->G1[dof*ec]; ibm->g1[dof*ec+1] = ibm->G1[dof*ec+1]; ibm->g1[dof*ec+2] = ibm->G1[dof*ec+2];
    ibm->g2[dof*ec] = ibm->G2[dof*ec]; ibm->g2[dof*ec+1] = ibm->G2[dof*ec+1]; ibm->g2[dof*ec+2] = ibm->G2[dof*ec+2];
    ibm->g3[dof*ec] = ibm->Nf_x[ec]; ibm->g3[dof*ec+1] = ibm->Nf_y[ec]; ibm->g3[dof*ec+2] = ibm->Nf_z[ec];

    ibm->g1n[dof*ec] = ibm->g1[dof*ec]; ibm->g1n[dof*ec+1] = ibm->g1[dof*ec+1]; ibm->g1n[dof*ec+2] = ibm->g1[dof*ec+2];
    ibm->g2n[dof*ec] = ibm->g2[dof*ec]; ibm->g2n[dof*ec+1] = ibm->g2[dof*ec+1]; ibm->g2n[dof*ec+2] = ibm->g2[dof*ec+2];
    ibm->g3n[dof*ec] = ibm->g3[dof*ec]; ibm->g3n[dof*ec+1] = ibm->g3[dof*ec+1]; ibm->g3n[dof*ec+2] = ibm->g3[dof*ec+2];
  }

  VecRestoreArray(fem->x, &xx);
  VecCopy(fem->x, fem->xn);
  VecCopy(fem->x, fem->xnm1);
  
  if (ConstitutiveLawNonLinear) {
    InitMaterial(ibm);
  }
  
  if (ti==0) InitVel(0, 0, fem);
  if (ti==0) InitVel(1, 0, fem);
  if (ti==0) InitVel(2, 0, fem);
  if (ti==0) InitVel(3, 0, fem);
  //if (ti==0 && manufactured) MoveBoundary(1, fem);
  Output(fem, 0, ibi);
  if(outghost){OutputGhost(fem, 0, ibi);}
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode Free(FE *fem) {

  IBMNodes  *ibm=fem->ibm;
  //Vec
  VecDestroy(&(fem->Res));  VecDestroy(&(fem->x));  VecDestroy(&(fem->xn));  VecDestroy(&(fem->xnm1));
  VecDestroy(&(fem->xd));  VecDestroy(&(fem->xdd));  VecDestroy(&(fem->y));  VecDestroy(&(fem->yn)); 
  VecDestroy(&(fem->Fext));  VecDestroy(&(fem->Fint));  VecDestroy(&(fem->Fdyn));
  VecDestroy(&(fem->disp));  VecDestroy(&(fem->FJ)); 
  VecDestroy(&(fem->Mass));  VecDestroy(&(fem->Dissip));  VecDestroy(&(fem->dx)); 
  //Malloc
  PetscFree(ibm->x_bp);  PetscFree(ibm->y_bp);  PetscFree(ibm->z_bp);
  PetscFree(ibm->x_bp0);  PetscFree(ibm->y_bp0);  PetscFree(ibm->z_bp0);
  PetscFree(ibm->nv1);  PetscFree(ibm->nv2);  PetscFree(ibm->nv3);
  PetscFree(ibm->nv4);  PetscFree(ibm->nv5);  PetscFree(ibm->nv6);
  PetscFree(ibm->n_bnodes);  PetscFree(ibm->bnodes);  PetscFree(ibm->kve0); 
  PetscFree(ibm->kve);  PetscFree(fem->StressM);  PetscFree(fem->StrainM);   PetscFree(fem->IE);  PetscFree(fem->CE);  PetscFree(fem->KE);  PetscFree(ibm->m); 
  PetscFree(fem->StressB);  PetscFree(fem->StrainB);
  PetscFree(ibm->ire);  PetscFree(ibm->irv);  PetscFree(ibm->val);  PetscFree(ibm->patch);  PetscFree(ibm->contact);
  PetscFree(ibm->G);  PetscFree(ibm->G1);  PetscFree(ibm->G2); PetscFree(ibm->g1); PetscFree(ibm->g2); PetscFree(ibm->g3); PetscFree(ibm->g1n); PetscFree(ibm->g2n); PetscFree(ibm->g3n);
  
  if(bending){
    PetscFree(ibm->belmtsedge);  PetscFree(ibm->belmts);  PetscFree(ibm->edgefrontnodes);
    PetscFree(ibm->edgefrontnodesI); 
    PetscFree(ibm->p4x);  PetscFree(ibm->p4y);  PetscFree(ibm->p4z);
    PetscFree(ibm->p5x);  PetscFree(ibm->p5y);  PetscFree(ibm->p5z);
    PetscFree(ibm->p6x);  PetscFree(ibm->p6y);  PetscFree(ibm->p6z);
    
    PetscFree(ibm->p4x0);  PetscFree(ibm->p4y0);  PetscFree(ibm->p4z0);
    PetscFree(ibm->p5x0);  PetscFree(ibm->p5y0);  PetscFree(ibm->p5z0);
    PetscFree(ibm->p6x0);  PetscFree(ibm->p6y0);  PetscFree(ibm->p6z0);
  }
  
  return(0);
}

//-------------------------------------------------------------------------------------
PetscErrorCode ContactZ(FE *fem) {

  IBMNodes  *ibm=fem->ibm;
  PetscInt  nv;

  for (nv=0; nv<ibm->n_v; nv++) {
    ibm->contact[nv] = 0;
  }

  return(0);
}

//-----------------------------------------
PetscErrorCode Contact(FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec, n1e, n2e, n3e;
  PetscReal  *xx, *yy, *fint, *fext, *xxn, *xxd, *xxdd;//, *omega, *gd1, *gd2, *gd3, *m, *KEr, KEomega, omg[2], omgc[2];
  PetscReal  IEsum, KEsum, CEsum;
  /* PetscReal      M1, M2, M3; */
  /* PetscReal      Gama=0.5, Beta=0.25; */
  /* M1=1./(Beta*pow(dt,2));  M2=1./(Beta*dt);  M3=(1./(2*Beta))-1.; */

  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &omega); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd1); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd2); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd3); */
  /* PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &m); */
  /* PetscMalloc(ibm->n_v*sizeof(PetscReal), &KEr); */

  /* VecSet(fem->Fcnt, 0.0);   */
  VecSet(fem->Fint, 0.0);  //VecSet(fem->Fext, 0.0);  //VecSet(fem->Fdyn, 0.0); */
  for (nv=0; nv<ibm->n_v; nv++)  {
    fem->IE[nv] = 0.;
    // KEr[nv] = 0.;
  }
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;

  FInternal(fem);
  /* //if(tisteps>1) {FDynamic(fem);} */
  /* FExternal(fem); */
  /* VecWAXPY(fem->Fcnt, -1., fem->Fext, fem->Fint); */
  /* //VecAXPY(fem->Fcnt, -1., fem->Fext); */
  /* //VecAXPY(fem->Fcnt, 1., fem->Fint); */
  /* VecAXPY(fem->Fcnt, 1., fem->Fdyn); */
  
  VecGetArray(fem->x, &xx);
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->y, &yy);
  VecGetArray(fem->Fint, &fint);
  VecGetArray(fem->Fext, &fext);

  /* for (ec=0; ec<ibm->n_elmt; ec++) { */
  /*   n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec]; */

  /*   gd1[dof*ec] = (ibm->g1[dof*ec] - ibm->g1n[dof*ec])/dt;  gd1[dof*ec+1] = (ibm->g1[dof*ec+1] - ibm->g1n[dof*ec+1])/dt;  gd1[dof*ec+2] = (ibm->g1[dof*ec+2] - ibm->g1n[dof*ec+2])/dt; */
  /*   gd2[dof*ec] = (ibm->g2[dof*ec] - ibm->g2n[dof*ec])/dt;  gd2[dof*ec+1] = (ibm->g2[dof*ec+1] - ibm->g2n[dof*ec+1])/dt;  gd2[dof*ec+2] = (ibm->g2[dof*ec+2] - ibm->g2n[dof*ec+2])/dt; */
  /*   gd3[dof*ec] = (ibm->g3[dof*ec] - ibm->g3n[dof*ec])/dt;  gd3[dof*ec+1] = (ibm->g3[dof*ec+1] - ibm->g3n[dof*ec+1])/dt;  gd3[dof*ec+2] = (ibm->g3[dof*ec+2] - ibm->g3n[dof*ec+2])/dt; */

  /*   omg[0] = -1*(ibm->g3[dof*ec]*gd1[dof*ec] + ibm->g3[dof*ec+1]*gd1[dof*ec+1] + ibm->g3[dof*ec+2]*gd1[dof*ec+2]); */
  /*   omg[1] = -1*(ibm->g3[dof*ec]*gd2[dof*ec] + ibm->g3[dof*ec+1]*gd2[dof*ec+1] + ibm->g3[dof*ec+2]*gd2[dof*ec+2]); */

  /*   omega[dof*ec] = omg[0]*ibm->g2[dof*ec] - omg[1]*ibm->g1[ec*dof]; */
  /*   omega[dof*ec+1] = omg[0]*ibm->g2[dof*ec+1] - omg[1]*ibm->g1[ec*dof+1]; */
  /*   omega[dof*ec+2] = omg[0]*ibm->g2[dof*ec+2] - omg[1]*ibm->g1[ec*dof+2]; */

  /*   omgc[0] = omega[dof*ec]*ibm->g2[dof*ec] + omega[dof*ec+1]*ibm->g2[dof*ec+1] + omega[dof*ec+2]*ibm->g2[dof*ec+2]; */
  /*   omgc[1] = -1*(omega[dof*ec]*ibm->g1[dof*ec] + omega[dof*ec+1]*ibm->g1[dof*ec+1] + omega[dof*ec+2]*ibm->g1[dof*ec+2]); */
  /*   omega[dof*ec+2] = -1*(ibm->g3[dof*ec]*gd3[dof*ec] + ibm->g3[dof*ec+1]*gd3[dof*ec+1] + ibm->g3[dof*ec+2]*gd3[dof*ec+2]); */

  /*   m[ec] = rho*ibm->dA0[ec]*h0; */

  /*   KEomega = h0*h0/24.*m[ec]*(omega[dof*ec]*omega[dof*ec] + omega[dof*ec+1]*omega[dof*ec+1] + omega[dof*ec+2]*omega[dof*ec+2]); */
  /*   KEomega = h0*h0/24.*m[ec]*(omg[0]*omgc[0] + omg[1]*omgc[1]); */

  /*   KEr[n1e] += KEomega/3.;  KEr[n2e] += KEomega/3.;  KEr[n3e] += KEomega/3.; */
  /* } */

  KEsum=0.; IEsum=0.; CEsum=0.;
  for (nv=0; nv<ibm->n_v; nv++) {
    if (ti==0) fem->CE[nv] = 0.;
    // set acceleration such that Mxdd=Fint
    if (ibm->contact[nv]>0 && ibm->ibi==1) {
    /*   xxdd[dof*nv  ] = M2*xxd[nv*dof  ]/8.;//fint[dof*nv  ]/ibm->m[nv]++M1*(xx[nv*dof  ]-xxn[nv*dof  ]); */
    /*   xxdd[dof*nv+1] = M2*xxd[nv*dof+1]/8.;//fint[dof*nv+1]/ibm->m[nv]++M1*(xx[nv*dof+1]-xxn[nv*dof+1]); */
    /*   xxdd[dof*nv+2] = M2*xxd[nv*dof+2]/8.;//fint[dof*nv+2]/ibm->m[nv]+//+M1*(xx[nv*dof+2]-xxn[nv*dof+2]); */

      PetscPrintf(PETSC_COMM_WORLD, "nv %d contact %d f_int %f %f %f acc %f %f %f\n",nv, ibm->contact[nv],fint[dof*nv  ], fint[dof*nv+1], fint[dof*nv+2],xxdd[dof*nv], xxdd[dof*nv+1], xxdd[dof*nv+2]);
    }

    //    fem->CE[nv] = (fint[nv*dof] - fext[nv*dof])*(xx[nv*dof] - yy[nv*dof]) + (fint[nv*dof+1] - fext[nv*dof+1])*(xx[nv*dof+1] - yy[nv*dof+1]) + (fint[nv*dof+2] - fext[nv*dof+2])*(xx[nv*dof+2] - yy[nv*dof+2]);
    //em->CE[nv] = (fint[nv*dof] - fext[nv*dof])*(xx[nv*dof] - xxn[nv*dof]) + (fint[nv*dof+1] - fext[nv*dof+1])*(xx[nv*dof+1] - xxn[nv*dof+1]) + (fint[nv*dof+2] - fext[nv*dof+2])*(xx[nv*dof+2] - xxn[nv*dof+2]);
    fem->KE[nv] = 0.5*ibm->m[nv]*(xxd[dof*nv]*xxd[dof*nv] + xxd[dof*nv+1]*xxd[dof*nv+1] +xxd[dof*nv+2]*xxd[dof*nv+2]);// + KEr[nv];
    KEsum +=fem->KE[nv];
    IEsum +=fem->IE[nv];
    //    CEsum +=    KEr[nv];
    //fem->KE[nv] = KEr[nv];
    //fem->IE[nv] = fint[nv*dof]*(xx[nv*dof] - ibm->x_bp0[nv]) + fint[nv*dof+1]*(xx[nv*dof+1] - ibm->y_bp0[nv]) + fint[nv*dof+2]*(xx[nv*dof+2] - ibm->z_bp0[nv]);
  }

  /* for (ec=0; ec<ibm->n_elmt; ec++) { */
  /*   ibm->g1n[dof*ec] = ibm->g1[dof*ec]; ibm->g1n[dof*ec+1] = ibm->g1[dof*ec+1]; ibm->g1n[dof*ec+2] = ibm->g1[dof*ec+2]; */
  /*   ibm->g2n[dof*ec] = ibm->g2[dof*ec]; ibm->g2n[dof*ec+1] = ibm->g2[dof*ec+1]; ibm->g2n[dof*ec+2] = ibm->g2[dof*ec+2]; */
  /*   ibm->g3n[dof*ec] = ibm->g3[dof*ec]; ibm->g3n[dof*ec+1] = ibm->g3[dof*ec+1]; ibm->g3n[dof*ec+2] = ibm->g3[dof*ec+2]; */
  /* } */

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xn, &xxn);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->y, &yy);
  VecRestoreArray(fem->Fint, &fint);
  VecRestoreArray(fem->Fext, &fext);

  FILE  *f;
  char  filen[80];
  PetscReal TEsum;
  TEsum = KEsum + IEsum;

  sprintf(filen, "energies%2.2d.dat", ibm->ibi);
  f = fopen(filen, "a"); // open file

  PetscFPrintf(PETSC_COMM_SELF, f, "%d %f %f %f %f\n", ti, IEsum, KEsum, TEsum, CEsum);
  
  fclose(f);
  //PetscFree(omega); PetscFree(gd1); PetscFree(gd2); PetscFree(gd3); PetscFree(m); PetscFree(KEr);

  return (0);
}
