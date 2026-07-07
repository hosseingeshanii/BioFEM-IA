#include  "variables.h"
#include  <stdio.h>

extern const PetscInt  dof, nbody;
extern PetscReal       E, rho, h0, dt;
extern PetscInt        ti, curvature;
extern char            in_dir[256];

extern struct Cmpnts  PLUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal  DOT(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  UNIT(struct Cmpnts v1);
extern PetscReal  SIZE(struct Cmpnts v1);
extern struct Cmpnts  AMULT(PetscReal alpha, struct Cmpnts v1);
extern PetscErrorCode  INV(PetscReal T[3][3], PetscReal _Tinv[3][3]);
// extern PetscErrorCode  MATMULT(PetscReal A[][2], PetscReal B[][2], PetscReal C[][2]);
extern PetscErrorCode  TRANS(PetscReal A[3][3], PetscReal _AT[3][3]);

 
PetscErrorCode NodeForce(PetscInt nv, PetscReal F, PetscInt dir, FE *fem) {

  PetscReal *FF;
  
  VecGetArray(fem->Fext, &FF);

  FF[nv*dof+dir]=F;

  VecRestoreArray(fem->Fext, &FF);

  return(0); 
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode FExternalPrescribedForceFieldIn(PetscInt step, FE *fem)
{
  IBMNodes    *ibm = fem->ibm;
  PetscViewer viewer;
  char        filen[512];
  FILE        *fd;

  snprintf(filen, sizeof(filen), "%s/fext%1.1d_%5.5d.dat", in_dir, ibm->ibi, step);

  fd = fopen(filen, "r");
  PetscCheck(fd != NULL, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
             "Unable to open prescribed external force file '%s'", filen);
  fclose(fd);

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(fem->Fext, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  return(0);
}

PetscErrorCode ConstantVel(PetscReal vel, PetscInt dir, FE *fem)
{

  IBMNodes  *ibm=fem->ibm;
  PetscInt  nc;
  PetscReal *xxd, *xxdd, *xxn, *xxnm1;
  VecGetArray(fem->xd, &xxd);
  /* VecGetArray(fem->xdd, &xxdd); */
  /* VecGetArray(fem->xn, &xxn); */
  /* VecGetArray(fem->x, &xxnm1); */

  for (nc=0; nc<ibm->n_v; nc++) {

    xxd[nc*dof+dir] = vel; //constant vel x
    //  if (nc==69) xxd[nc*dof+dir]=-vel;
    //  xxdd[nc*dof+dir] = 0.; //zero acceleration
    //    xxn[nc*dof+dir] -= vel*dt; //xn=x-vel*dt Note xn=x in Init subroutine
    //  xxnm1[nc*dof+dir] -= 2.*vel*dt;  //xnm1=xn-vel*dt=x-vel*2*dt
  }

  VecRestoreArray(fem->xd, &xxd);
  /* VecRestoreArray(fem->xdd, &xxdd); */
  /* VecRestoreArray(fem->xn, &xxn); */
  /* VecRestoreArray(fem->xnm1, &xxnm1); */

  return(0);

}

//------------------------------------------------------------------------------------------------------------

PetscErrorCode EdgeConstPressure(PetscInt edge_n, PetscReal P, PetscInt dir, FE *fem) {
  PetscFunctionBeginUser;

  IBMNodes *ibm=fem->ibm;
  PetscInt start=0, end=0, edge, i, *cbn, *sortbn, nb, ii;
  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];
  
  PetscMalloc(ibm->n_bnodes[edge_n]*sizeof(PetscInt), &cbn);
  PetscMalloc(ibm->n_bnodes[edge_n]*sizeof(PetscInt), &sortbn);
  ii = 0;
  for (i=start; i<end; i++) {
    nb = ibm->bnodes[i];
    cbn[ii] = (PetscInt)(100000.0*PetscSqrtReal(pow(ibm->x_bp0[nb], 2) + pow(ibm->y_bp0[nb], 2) + pow(ibm->z_bp0[nb], 2)));
    sortbn[ii] = nb;
    ii += 1;    
  }
  PetscSortIntWithArray(ibm->n_bnodes[edge_n], cbn, sortbn);

  PetscReal *RR, b, *FF;
  PetscInt nb1, nb2; 
  VecGetArray(fem->Fext, &FF);
 
  for (i=0; i<ibm->n_bnodes[edge_n]-1; i++) {
    nb1 = sortbn[i];  nb2 = sortbn[i+1];
    b = PetscSqrtReal(pow(ibm->x_bp0[nb2] - ibm->x_bp0[nb1], 2) + pow(ibm->y_bp0[nb2] - ibm->y_bp0[nb1], 2) + pow(ibm->z_bp0[nb2] - ibm->z_bp0[nb1], 2));
    
    FF[nb1*dof+dir] += b*h0*P/2.;
    FF[nb2*dof+dir] += b*h0*P/2.;
    //  PetscPrintf(PETSC_COMM_SELF, "edgeLengthFornodes(%d,%d): %le\n",nb1,nb2,b);
  }

  VecRestoreArray(fem->Fext, &FF);

  PetscFree(cbn);  PetscFree(sortbn);

  PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode EdgeFix(PetscInt edge_n, FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscReal  *FFint, *FFext, *FFdyn, *xx, *xdd;
  PetscInt   start=0, end=0, edge, nbc, nb;

  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);
  VecGetArray(fem->x, &xx);
  VecGetArray(fem->xd, &xdd);

  for (nbc=start; nbc<end; nbc++) { //fix boundary nodes
    nb=ibm->bnodes[nbc];

    ibm->x_bp[nb] = ibm->x_bp0[nb]; //for kinematic contact
    ibm->y_bp[nb] = ibm->y_bp0[nb];
    ibm->z_bp[nb] = ibm->z_bp0[nb];
    xx[nb*dof] = ibm->x_bp0[nb];
    xx[nb*dof+1] = ibm->y_bp0[nb];
    xx[nb*dof+2] = ibm->z_bp0[nb];
    ibm->contact[nb] = 0;
    xdd[nb*dof] = 0.;
    xdd[nb*dof+1] = 0.;
    xdd[nb*dof+2] = 0.;

    FFint[nb*dof] =0.0;
    FFint[nb*dof+1] =0.0;
    FFint[nb*dof+2] =0.0;
    
    FFext[nb*dof] =0.0;
    FFext[nb*dof+1] =0.0;
    FFext[nb*dof+2] =0.0;
    
    FFdyn[nb*dof] =0.0;
    FFdyn[nb*dof+1] =0.0;
    FFdyn[nb*dof+2] =0.0;

  }

  /* for (nb=ibm->n_v; nb<ibm->n_v+ibm->n_ghosts; nb++) { //fix all ghost nodes */
  /*   FFint[nb*dof] =0.0; */
  /*   FFint[nb*dof+1] =0.0; */
  /*   FFint[nb*dof+2] =0.0; */
    
  /*   FFext[nb*dof] =0.0; */
  /*   FFext[nb*dof+1] =0.0; */
  /*   FFext[nb*dof+2] =0.0; */
    
  /*   FFdyn[nb*dof] =0.0; */
  /*   FFdyn[nb*dof+1] =0.0; */
  /*   FFdyn[nb*dof+2] =0.0; */
    
  /* } */

  for (nb=ibm->n_v+ibm->n_bnodes[edge_n-1]-1; nb<ibm->n_v+ibm->n_ghosts; nb++) { //fix ghost nodes of BHV Note:number of ghost nodes are one shorter than boundary nodes on edges
    FFint[nb*dof] = 0.0;
    FFint[nb*dof+1] = 0.0;
    FFint[nb*dof+2] = 0.0;
    
    FFext[nb*dof] = 0.0;
    FFext[nb*dof+1] = 0.0;
    FFext[nb*dof+2] = 0.0;
    
    FFdyn[nb*dof] = 0.0;
    FFdyn[nb*dof+1] = 0.0;
    FFdyn[nb*dof+2] = 0.0;
    
  }

  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xdd);
  PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode EdgeFree(PetscInt edge_n, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *FFint, *FFext, *FFdyn;
  PetscInt  nb;
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);

  for (nb=ibm->n_v; nb<ibm->n_v+ibm->n_ghosts; nb++) { //excludes ghost nodes from solver 
    FFint[nb*dof] =0.0;
    FFint[nb*dof+1] =0.0;
    FFint[nb*dof+2] =0.0;
    
    FFext[nb*dof] =0.0;
    FFext[nb*dof+1] =0.0;
    FFext[nb*dof+2] =0.0;
    
    FFdyn[nb*dof] =0.0;
    FFdyn[nb*dof+1] =0.0;
    FFdyn[nb*dof+2] =0.0;
    
  }

  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode InitVel(PetscInt edge_n, PetscReal w, FE *fem) {

  IBMNodes      *ibm = fem->ibm;
  DMPlexGeomCtx *gctx = &fem->geom_ctx;
  PetscReal     *xdd;
  PetscInt       nb;
  PetscInt       start=0, end=0, edge, nbc;

  VecGetArray(fem->xd, &xdd);

  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];

  for (nbc=start; nbc<end; nbc++) { //fix boundary nodes
    nb = ibm->bnodes[nbc];
    PetscInt local_idx = gctx->initialized ? gctx->ibm_to_local_idx[nb] : nb;
    if (local_idx < 0) continue;
    xdd[local_idx*dof] = 0.0;
  }

  VecRestoreArray(fem->xd, &xdd);

  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode EdgeClamp(PetscInt edge_n, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *FFint, *FFext, *FFdyn;
  PetscInt start=0, end=0, edge, nbc, nb, ec;

  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);

   for (nbc=start; nbc<end; nbc++) { //fix boundary nodes */
     nb=ibm->bnodes[nbc]; 
    
     FFint[nb*dof] =0.0; 
     FFint[nb*dof+1] =0.0; 
     FFint[nb*dof+2] =0.0; 
    
     FFext[nb*dof] =0.0; 
     FFext[nb*dof+1] =0.0; 
     FFext[nb*dof+2] =0.0; 
    
     FFdyn[nb*dof] =0.0; 
     FFdyn[nb*dof+1] =0.0; 
     FFdyn[nb*dof+2] =0.0; 

   } 

   for (nb=ibm->n_v; nb<ibm->n_v+ibm->n_ghosts; nb++) { //fix all ghost nodes 
     FFint[nb*dof] =0.0; 
     FFint[nb*dof+1] =0.0; 
     FFint[nb*dof+2] =0.0; 
    
     FFext[nb*dof] =0.0; 
     FFext[nb*dof+1] =0.0; 
     FFext[nb*dof+2] =0.0; 
    
     FFdyn[nb*dof] =0.0; 
     FFdyn[nb*dof+1] =0.0; 
     FFdyn[nb*dof+2] =0.0; 
    
   } 

  /* for (nb=ibm->n_v+ibm->n_bnodes[edge_n]-1; nb<ibm->n_v+ibm->n_ghosts; nb++) { //fix ghost nodes of BHV */
  /*   FFint[nb*dof] =0.0; */
  /*   FFint[nb*dof+1] =0.0; */
  /*   FFint[nb*dof+2] =0.0; */
    
  /*   FFext[nb*dof] =0.0; */
  /*   FFext[nb*dof+1] =0.0; */
  /*   FFext[nb*dof+2] =0.0; */
    
  /*   FFdyn[nb*dof] =0.0; */
  /*   FFdyn[nb*dof+1] =0.0; */
  /*   FFdyn[nb*dof+2] =0.0; */
    
  /* } */

  /* if (curvature ==6) { //clamped edge */
  /*   for (ec=0; ec<ibm->n_ghosts; ec++) { */
  /*     if (ibm->belmtsedge[ec]==edge_n) { */
  /* 	nb = ibm->edgefrontnodes[ec];       */

  /* 	FFint[nb*dof] = 0.0; */
  /* 	FFint[nb*dof+1] = 0.0; */
  /* 	FFint[nb*dof+2] = 0.0; */
	
  /* 	FFext[nb*dof] = 0.0; */
  /* 	FFext[nb*dof+1] = 0.0; */
  /* 	FFext[nb*dof+2] = 0.0; */
	
  /* 	FFdyn[nb*dof] = 0.0; */
  /* 	FFdyn[nb*dof+1] = 0.0; */
  /* 	FFdyn[nb*dof+2] = 0.0; */
	
  /*     } */
  /*   } */
  /* } */

  if (curvature ==6) { //general clamped edge for curvature 6 */
     for (nbc=0; nbc<ibm->n_ghosts; nbc++) { 

		nb = ibm->edgefrontnodes[nbc]; 

		ibm->x_bp[nb] = ibm->x_bp0[nb]; 
		ibm->y_bp[nb] = ibm->y_bp0[nb]; 
		ibm->z_bp[nb] = ibm->z_bp0[nb]; 
      
		FFint[nb*dof] =0.0; 
		FFint[nb*dof+1] =0.0; 
		FFint[nb*dof+2] =0.0; 
   
		FFext[nb*dof] =0.0; 
		FFext[nb*dof+1] =0.0; 
		FFext[nb*dof+2] =0.0; 
      
		FFdyn[nb*dof] =0.0; 
		FFdyn[nb*dof+1] =0.0; 
		FFdyn[nb*dof+2] =0.0; 
      
     } 
   } 


  /*
  if (curvature==6) { //clamped edge for curvature 6 for canti
    for (nb=0; nb<ibm->n_v+ibm->n_ghosts; nb++) {
      if (ibm->x_bp0[nb]<0.0045) {//for canti x<0.0017
  	//if (ibm->x_bp0[nb]<0.042) {//for plate
  	ibm->x_bp[nb] = ibm->x_bp0[nb];
  	ibm->y_bp[nb] = ibm->y_bp0[nb];
  	ibm->z_bp[nb] = ibm->z_bp0[nb];
	
  	FFint[nb*dof] =0.0;
  	FFint[nb*dof+1] =0.0;
  	FFint[nb*dof+2] =0.0;
	
  	FFext[nb*dof] =0.0;
  	FFext[nb*dof+1] =0.0;
  	FFext[nb*dof+2] =0.0;
      
  	FFdyn[nb*dof] =0.0;
  	FFdyn[nb*dof+1] =0.0;
  	FFdyn[nb*dof+2] =0.0;
	
      }
    }
  }
  */
  

  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
 
  return(0);
} 

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode MoveBoundary(PetscInt edge_n, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *FFint, *FFext, *FFdyn, *xx;
  PetscInt start=0, end=0, edge, nbc, nb;

  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];
  
  VecGetArray(fem->x, &xx);

  /* for (nbc=start; nbc<end; nbc++) { //prescribed motion */
  /*   nb=ibm->bnodes[nbc]; */

  /*   ibm->z_bp[nb] = 0.0015875*sin(43.96*ti*dt); */
  /*   xx[nb*dof+2] = ibm->z_bp[nb]; */

  /* } */

  if (curvature ==6) { //clamped edge for curvature 6
    for (nb=0; nb<ibm->n_v+ibm->n_ghosts; nb++) {
      if (ibm->x_bp0[nb]>0.0017) {//for canti
	//if (ibm->x_bp0[nb]<0.042) {//for plate

  	ibm->z_bp[nb] = pow(ibm->x_bp[nb],2)/0.08*cos(4*ti*dt);
  	xx[nb*dof+2] = ibm->z_bp[nb];
 
      }
    }
  }

  VecRestoreArray(fem->x, &xx);
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode EdgeDiaph(PetscInt edge_n, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *FFint, *FFext, *FFdyn;
  PetscInt start=0, end=0, edge, nbc, nb;

  for (edge=0; edge<edge_n+1; edge++) {
    end+=ibm->n_bnodes[edge];
  }
  start=end-ibm->n_bnodes[edge_n];
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);

  for (nbc=start; nbc<end; nbc++) {
    nb=ibm->bnodes[nbc];
    
    FFint[nb*dof] =0.0;
    FFint[nb*dof+2] =0.0;
    
    FFext[nb*dof] =0.0;
    FFext[nb*dof+2] =0.0;
    
    FFdyn[nb*dof] =0.0;
    FFdyn[nb*dof+2] =0.0;

  }

  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode EdgeSym(PetscInt edge_n, PetscInt dir, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *FFint, *FFext, *FFdyn;
  PetscInt start=0, end=0, edge, nbc, nb;

  for (edge=0; edge<edge_n+1; edge++) {
    end+=ibm->n_bnodes[edge];
  }
  start=end-ibm->n_bnodes[edge_n];
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);

  for (nbc=start; nbc<end; nbc++) {
    nb=ibm->bnodes[nbc];

    if (dir==0) {    
      FFint[nb*dof] =0.0;    
      FFext[nb*dof] =0.0;   
      FFdyn[nb*dof] =0.0;

    } else if (dir==1) {
      FFint[nb*dof+1] =0.0;    
      FFext[nb*dof+1] =0.0;   
      FFdyn[nb*dof+1] =0.0;

    } else if (dir==2) {
      FFint[nb*dof+2] =0.0;    
      FFext[nb*dof+2] =0.0;   
      FFdyn[nb*dof+2] =0.0;
    }
  }

  for (nb=ibm->n_v; nb<ibm->n_v+ibm->n_ghosts; nb++) { //excludes ghost nodes from solver
    FFint[nb*dof] =0.0;
    FFint[nb*dof+1] =0.0;
    FFint[nb*dof+2] =0.0;
    
    FFext[nb*dof] =0.0;
    FFext[nb*dof+1] =0.0;
    FFext[nb*dof+2] =0.0;
    
    FFdyn[nb*dof] =0.0;
    FFdyn[nb*dof+1] =0.0;
    FFdyn[nb*dof+2] =0.0;
    
  }
  
  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfaceNormalPressure(PetscReal P, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscInt i, ec, n1e, n2e, n3e;
  PetscReal A0, *FF;
  struct Cmpnts x1, x2, x3, dx21, dx31, X1, X2, X3, dX21, dX31, n;
  PetscInt start, end, edge_n, edge, bn;

  VecGetArray(fem->Fext, &FF);


  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
       
    A0 = ibm->dA[ec];
    n.x = ibm->nf_x[ec];  n.y = ibm->nf_y[ec];  n.z = ibm->nf_z[ec];
    
    FF[n1e*dof] += n.x*P*A0/3.;
    FF[n1e*dof+1] += n.y*P*A0/3.;
    FF[n1e*dof+2] += n.z*P*A0/3.;
    
    FF[n2e*dof] += n.x*P*A0/3.;
    FF[n2e*dof+1] += n.y*P*A0/3.;
    FF[n2e*dof+2] += n.z*P*A0/3.;
    
    FF[n3e*dof] += n.x*P*A0/3.;
    FF[n3e*dof+1] += n.y*P*A0/3.;
    FF[n3e*dof+2] += n.z*P*A0/3.;
    
  }//end loop over elements

  VecRestoreArray(fem->Fext, &FF);
 
  return(0);
  
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode CardiacPressure(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscInt       i, ec, n1e, n2e, n3e;
  PetscReal      P, A0, *FF;
  struct Cmpnts  x1, x2, x3, dx21, dx31, X1, X2, X3, dX21, dX31, n;
  PetscInt       start, end, edge_n, edge, bn;

  VecGetArray(fem->Fext, &FF);

  //Reading the pressure
  FILE       *fd;
  char       string[128];
  PetscInt   n_p, ie;
  PetscReal  *t_array, *p_array, tcyc, t, ts, ps, te, pe;
  fd = fopen("pressure.dat", "r");
  fscanf(fd, "%i", &n_p);
  
  PetscMalloc(n_p*sizeof(PetscReal), &t_array);
  PetscMalloc(n_p*sizeof(PetscReal), &p_array);
  
  i=-1;
  fgets(string, 128, fd);// skip line one
  while (i+1<n_p) {
    i++;
    fscanf(fd, " %le, %le\n", &t_array[i], &p_array[i]);
  }//end while
  fclose(fd);
  //end reading 
  tcyc = t_array[n_p-1]; 
  t = dt*(ti);
  t = t-((PetscInt)(t/tcyc))*tcyc;
  
  for (i=0; i<n_p; i++){
    if (t_array[i]<=t){
      ts = t_array[i];  ps = p_array[i];  
      ie = i+1;
    }
    te = t_array[ie];  pe = p_array[ie];
  }
  
  P = ps + ((pe - ps)/(te - ts))*(t - ts);
  if (t==0) {P = p_array[0];}
  P = -P;
  //PetscPrintf(PETSC_COMM_SELF, "cyclic time:%le Pressure= %le \n",t,P);
  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
       
    A0 = ibm->dA[ec];
    n.x = ibm->nf_x[ec];  n.y = ibm->nf_y[ec];  n.z = ibm->nf_z[ec];
    
    FF[n1e*dof] += n.x*P*A0/3.;
    FF[n1e*dof+1] += n.y*P*A0/3.;
    FF[n1e*dof+2] += n.z*P*A0/3.;
    
    FF[n2e*dof] += n.x*P*A0/3.;
    FF[n2e*dof+1] += n.y*P*A0/3.;
    FF[n2e*dof+2] += n.z*P*A0/3.;
    
    FF[n3e*dof] += n.x*P*A0/3.;
    FF[n3e*dof+1] += n.y*P*A0/3.;
    FF[n3e*dof+2] += n.z*P*A0/3.;
    
  }//end loop over elements

  VecRestoreArray(fem->Fext, &FF);
 
  return(0);
  
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfaceConstNormalPressure(PetscReal P, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscInt i, ec, n1e, n2e, n3e;
  PetscReal A0, *FF;
  struct Cmpnts X1, X2, X3, dX21, dX31, N;

  VecGetArray(fem->Fext, &FF);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e=ibm->nv1[ec]; n2e=ibm->nv2[ec]; n3e=ibm->nv3[ec];
    
    A0=ibm->dA0[ec];
    N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
    
    FF[n1e*dof]+=N.x*P*A0/3.;
    FF[n1e*dof+1]+=N.y*P*A0/3.;
    FF[n1e*dof+2]+=N.z*P*A0/3.;
      
    FF[n2e*dof]+=N.x*P*A0/3.;
    FF[n2e*dof+1]+=N.y*P*A0/3.;
    FF[n2e*dof+2]+=N.z*P*A0/3.;
    
    FF[n3e*dof]+=N.x*P*A0/3.;
    FF[n3e*dof+1]+=N.y*P*A0/3.;
    FF[n3e*dof+2]+=N.z*P*A0/3.;
  }//end loop over elements
  
  VecRestoreArray(fem->Fext, &FF);
  
  return(0);
}


//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfaceSinNormalPressure(PetscReal P, PetscReal char_length_x, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscInt i, ec, n1e, n2e, n3e;
  PetscReal A0, *FF, PP, x0;
  struct Cmpnts X1, X2, X3, dX21, dX31, N;

  VecGetArray(fem->Fext, &FF);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e=ibm->nv1[ec]; n2e=ibm->nv2[ec]; n3e=ibm->nv3[ec];
    
  //initial location
    X1.x=ibm->x_bp0[n1e]; X1.y=ibm->y_bp0[n1e]; X1.z=ibm->z_bp0[n1e];
    X2.x=ibm->x_bp0[n2e]; X2.y=ibm->y_bp0[n2e]; X2.z=ibm->z_bp0[n2e];
    X3.x=ibm->x_bp0[n3e]; X3.y=ibm->y_bp0[n3e]; X3.z=ibm->z_bp0[n3e];
    
    x0=(X1.x+X2.x+X3.x)/3.; 
    PP = P*sin(x0*char_length_x*3.141592);
    
    A0=ibm->dA0[ec];
    N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
    
    FF[n1e*dof]+=N.x*PP*A0/3.;
    FF[n1e*dof+1]+=N.y*PP*A0/3.;
    FF[n1e*dof+2]+=N.z*PP*A0/3.;
      
    FF[n2e*dof]+=N.x*PP*A0/3.;
    FF[n2e*dof+1]+=N.y*PP*A0/3.;
    FF[n2e*dof+2]+=N.z*PP*A0/3.;
    
    FF[n3e*dof]+=N.x*PP*A0/3.;
    FF[n3e*dof+1]+=N.y*PP*A0/3.;
    FF[n3e*dof+2]+=N.z*PP*A0/3.;
  }//end loop over elements
  
  VecRestoreArray(fem->Fext, &FF);
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 


PetscErrorCode DistributedForce(PetscReal w, FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscInt   nc;
  PetscReal  *FF, x, t, a, L, II, b;

  t = ti*dt;
  L = 0.04;
  a = 0.5*L;
  b = 0.004;
  II = b*pow(h0,3)/12.;
  VecGetArray(fem->Fext, &FF);

  for (nc=0; nc<ibm->n_v; nc++) {
    x = ibm->x_bp[nc]; 
    FF[nc*dof] = 0.;
    FF[nc*dof+1] = 0.;
    FF[nc*dof+2] = a*(4*II*E/pow(L, 2) - rho*b*h0*pow(w, 2)*pow(x, 2))*cos(w*t);
    //if (nc==42 || nc==88 || nc==33 || nc==40 || nc==56) PetscPrintf(PETSC_COMM_SELF, "force of node %d is %le\n", nc, FF[nc*dof+2]);
  }
  
  VecRestoreArray(fem->Fext, &FF);
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfaceConstNormalPressure2(PetscReal P, FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscInt i, ec, n1e, n2e, n3e;
  PetscReal A0, *FF;
  struct Cmpnts X1, X2, X3, dX21, dX31, N;

  VecGetArray(fem->Fext, &FF);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n1e=ibm->nv1[ec]; n2e=ibm->nv2[ec]; n3e=ibm->nv3[ec];
    
    A0=ibm->dA0[ec];
    N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
    if (n1e==98-1){ 
      FF[n1e*dof]+=N.x*P*A0/3.;
      FF[n1e*dof+1]+=N.y*P*A0/3.;
      FF[n1e*dof+2]+=N.z*P*A0/3.;
    }
    if (n2e==98-1){ 
      FF[n2e*dof]+=N.x*P*A0/3.;
      FF[n2e*dof+1]+=N.y*P*A0/3.;
      FF[n2e*dof+2]+=N.z*P*A0/3.;
    }
    if (n3e==98-1){ 
      FF[n3e*dof]+=N.x*P*A0/3.;
      FF[n3e*dof+1]+=N.y*P*A0/3.;
      FF[n3e*dof+2]+=N.z*P*A0/3.;
    }
  }//end loop over elements
  
  VecRestoreArray(fem->Fext, &FF);
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode SurfaceGravity(PetscReal g,FE *fem)
{

  IBMNodes *ibm=fem->ibm;
  PetscInt i,ec,n1e,n2e,n3e;
  PetscReal A0,*FF, Atot = 0.;
  PetscInt start, end, edge_n, edge, bn;
  VecGetArray(fem->Fext, &FF);

  /* for (ec=0; ec<ibm->n_elmt; ec++) { */
  /*   Atot += ibm->dA0[ec]; */
  /* } */

  /* P = P*Atot/ibm->n_v; */
  /* for (i=0; i<ibm->n_v; i++) { */
  /*   FF[dof*i+2] = P/10*(ti+1); */
  /* } */

  /* if (ibm->ibi==0) { */
  /*   rho = 1.;  h0 = 0.003; */
  /* } else { */
  /*   rho = 10.;  h0 = 0.03; */
  /* } */

  for (ec=0; ec<ibm->n_elmt; ec++) {
    
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
    A0=ibm->dA0[ec];
    
    FF[n1e*dof+2] -= g*rho*h0*A0/3.;
    FF[n2e*dof+2] -= g*rho*h0*A0/3.;
    FF[n3e*dof+2] -= g*rho*h0*A0/3.;
    
  }

  /* for (edge_n=0;edge_n<ibm->n_edge;edge_n=edge_n+2){ */
  /*     //compute start&end */
  /*     start=0; end=0; */
  /*     for (edge=0;edge<edge_n+1;edge++) { */
  /* 	end+=ibm->n_bnodes[edge]; */
  /*     } */
  /*     start=end-ibm->n_bnodes[edge_n]; */

  /*     //nodes on each edge */
  /*     for (i=start;i<end;i++) { */
  /* 	bn=ibm->bnodes[i]; */
  /* 	FF[bn*dof+2] = 2*FF[bn*dof+2]; */
  /*     } */
  /* } */

  
  VecRestoreArray(fem->Fext, &FF);

  return(0);
  
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfaceNonlinearForce(FE *fem)
{

  IBMNodes *ibm=fem->ibm;
  PetscInt i;
  PetscReal *FF;
  struct Cmpnts X;

  PetscReal f,fx,t,XM;
  PetscReal b=0.004, L=0.04, w=8.0,a,II;
  a=0.01*b/L; II=(1./12.)*b*pow(h0,3.);  t=ti*dt;


  VecGetArray(fem->Fext, &FF);
 for (i=0; i<ibm->n_v; i++) {
     
    
      //initial location
      X.x=ibm->x_bp0[i]; X.y=ibm->y_bp0[i]; X.z=ibm->z_bp0[i];
      XM=X.x;

      // fx=a*((4.*I*E/pow(L,2.))-(rho*b*h0*pow(w,2.)*pow(XM,2.)))+
      //	a*((-4.*I*E*XM/pow(L,3.))+(rho*b*h0*pow(w,2.)*pow(XM,3.)/(3.*L))); //dynamic
       fx=a*((4.*II*E/pow(L,2.))); //Static

      // f=fx*sin(w*t);  //dynamic
       f=fx; //Static

      FF[i*dof+2]=f/5.0;

     
    }//end loop over nodes

 
  VecRestoreArray(fem->Fext, &FF);
 return(0);
}
//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode SurfacePartialConstNormalPressure(PetscReal P,FE *fem)
{

  IBMNodes *ibm=fem->ibm;
  PetscInt i,ec,n1e,n2e,n3e;
  PetscReal A0,*FF,z0,x0;
  struct Cmpnts X1,X2,X3,dX21,dX31,N;

  VecGetArray(fem->Fext, &FF);
 for (ec=0; ec<ibm->n_elmt; ec++) {
     n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
    
      //initial location
      X1.x=ibm->x_bp0[n1e]; X1.y=ibm->y_bp0[n1e]; X1.z=ibm->z_bp0[n1e];
      X2.x=ibm->x_bp0[n2e]; X2.y=ibm->y_bp0[n2e]; X2.z=ibm->z_bp0[n2e];
      X3.x=ibm->x_bp0[n3e]; X3.y=ibm->y_bp0[n3e]; X3.z=ibm->z_bp0[n3e];

      dX21=MINUS(X2,X1); dX31=MINUS(X3,X1);  //dX21:G1 , dX31:G2
       A0=ibm->dA0[ec];
       // A0=0.5*SIZE(CROSS(dX21,dX31));
      N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
      //    N=UNIT(CROSS(dX21,dX31));

      x0=(X1.x+X2.x+X3.x)/3.; z0=(X1.z+X2.z+X3.z)/3.;

      if(x0>-0.05 && x0<0.05 && z0>0.175 && z0<0.225){
      FF[n1e*dof]+=N.x*P*A0/3.;
      FF[n1e*dof+1]+=N.y*P*A0/3.;
      FF[n1e*dof+2]+=N.z*P*A0/3.;

      FF[n2e*dof]+=N.x*P*A0/3.;
      FF[n2e*dof+1]+=N.y*P*A0/3.;
      FF[n2e*dof+2]+=N.z*P*A0/3.;

      FF[n3e*dof]+=N.x*P*A0/3.;
      FF[n3e*dof+1]+=N.y*P*A0/3.;
      FF[n3e*dof+2]+=N.z*P*A0/3.;
      } 
    }//end loop over elements

 
  VecRestoreArray(fem->Fext, &FF);
 return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode GhostFix(PetscInt edge_n,FE *fem) {

  IBMNodes *ibm=fem->ibm;
  PetscInt n1e, n2e, n3e, ec, be;
  
  struct Cmpnts N,X1,X2,X3,x1,x2,x3,x4,x5,x6,a1,a2,a3,A1,A2,A3,e1,e2,e3,n1,n2,n3,h1,h2,h3;
  for (ec=0; ec<ibm->n_ghosts; ec++) { //Go through ghost nodes
    be=ibm->belmts[ec];
    n1e=ibm->nv1[be];n2e=ibm->nv2[be];n3e=ibm->nv3[be];
    
    //currentlocation
    x1.x=ibm->x_bp[n1e]; x1.y=ibm->y_bp[n1e]; x1.z=ibm->z_bp[n1e];
    x2.x=ibm->x_bp[n2e]; x2.y=ibm->y_bp[n2e]; x2.z=ibm->z_bp[n2e];
    x3.x=ibm->x_bp[n3e]; x3.y=ibm->y_bp[n3e]; x3.z=ibm->z_bp[n3e];
    //initial location
    X1.x=ibm->x_bp0[n1e]; X1.y=ibm->y_bp0[n1e]; X1.z=ibm->z_bp0[n1e];
    X2.x=ibm->x_bp0[n2e]; X2.y=ibm->y_bp0[n2e]; X2.z=ibm->z_bp0[n2e];
    X3.x=ibm->x_bp0[n3e]; X3.y=ibm->y_bp0[n3e]; X3.z=ibm->z_bp0[n3e];

    A1=MINUS(X3,X2) , A2=MINUS(X1,X3), A3=MINUS(X2,X1);
    a1=MINUS(x3,x2) , a2=MINUS(x1,x3), a3=MINUS(x2,x1);
    //  N=UNIT(CROSS(A3,AMULT(-1.,A2)));
    N.x=ibm->Nf_x[be]; N.y=ibm->Nf_y[be]; N.z=ibm->Nf_z[be];
    
    if(ibm->belmtsedge[ec]==edge_n){ //check if it is on right edge
      if(ibm->edgefrontnodesI[ec]==1){
	//I=1, J=4 , P=3
	e1=UNIT(A1);
	n1=CROSS(e1,N);
	h1=AMULT(DOT(n1,a3),n1);
	x4=PLUS(x1,AMULT(2.,h1));
	ibm->p4x[be]=x4.x; ibm->p4y[be]=x4.y; ibm->p4z[be]=x4.z; 
      }else if(ibm->edgefrontnodesI[ec]==2){
	//I=2, J=5 , P=1
  	e2=UNIT(A2);
	n2=CROSS(e2,N);
	h2=AMULT(DOT(n2,a1),n2);
	x5=PLUS(x2,AMULT(2.,h2));
	ibm->p5x[be]=x5.x; ibm->p5y[be]=x5.y; ibm->p5z[be]=x5.z; 
      }else if(ibm->edgefrontnodesI[ec]==3){
	//I=3, J=6 , P=2
	e3=UNIT(A3);
	n3=CROSS(e3,N);
	h3=AMULT(DOT(n3,a2),n3);
	x6=PLUS(x3,AMULT(2.,h3));
	ibm->p6x[be]=x6.x; ibm->p6y[be]=x6.y; ibm->p6z[be]=x6.z; 
      }
    }
  }

  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode ModifyGhostFix(PetscInt edge_n,FE *fem) {
  
  IBMNodes *ibm=fem->ibm;
  PetscInt n1e,n2e,n3e,ec,be,i,p;
  
  struct Cmpnts N,X1,X2,X3,x1,x2,x3,x4,x5,x6,a1,a2,a3,A1,A2,A3,e1,e2,e3,n1,n2,n3;
  PetscReal E[3][3],Fadd[3],FJ[3],sum;
  
  PetscReal *FF,*FFJ;
  VecGetArray(fem->Fint, &FF);
  VecGetArray(fem->FJ, &FFJ);
  
  for (ec=0; ec<ibm->n_ghosts; ec++) { //Go through ghost nodes
    be=ibm->belmts[ec];
    n1e=ibm->nv1[be];n2e=ibm->nv2[be];n3e=ibm->nv3[be];

    //currentlocation
    x1.x=ibm->x_bp[n1e]; x1.y=ibm->y_bp[n1e]; x1.z=ibm->z_bp[n1e];
    x2.x=ibm->x_bp[n2e]; x2.y=ibm->y_bp[n2e]; x2.z=ibm->z_bp[n2e];
    x3.x=ibm->x_bp[n3e]; x3.y=ibm->y_bp[n3e]; x3.z=ibm->z_bp[n3e];
    //initial location
    X1.x=ibm->x_bp0[n1e]; X1.y=ibm->y_bp0[n1e]; X1.z=ibm->z_bp0[n1e];
    X2.x=ibm->x_bp0[n2e]; X2.y=ibm->y_bp0[n2e]; X2.z=ibm->z_bp0[n2e];
    X3.x=ibm->x_bp0[n3e]; X3.y=ibm->y_bp0[n3e]; X3.z=ibm->z_bp0[n3e];

    A1=MINUS(X3,X2) , A2=MINUS(X1,X3), A3=MINUS(X2,X1);
    a1=MINUS(x3,x2) , a2=MINUS(x1,x3), a3=MINUS(x2,x1);
    // N=UNIT(CROSS(A3,AMULT(-1.,A2)));
    N.x=ibm->Nf_x[be]; N.y=ibm->Nf_y[be]; N.z=ibm->Nf_z[be];
    
    if(ibm->belmtsedge[ec]==edge_n){ //check if it is on right edge
      if(ibm->edgefrontnodesI[ec]==1){
	//I=1, J=4 , P=3
	e1=UNIT(A1);
	n1=CROSS(e1,N);
	
	E[0][0]=1-2*n1.x*n1.x; E[0][1]=-2*n1.x*n1.y;  E[0][2]=-2*n1.x*n1.z;
	E[1][0]=-2*n1.y*n1.x;  E[1][1]=1-2*n1.y*n1.y; E[1][2]=-2*n1.y*n1.z;
	E[2][0]=-2*n1.z*n1.x;  E[2][1]=-2*n1.z*n1.y;  E[2][2]=1-2*n1.z*n1.z;
	
	FJ[0]=FFJ[n1e*dof]; FJ[1]=FFJ[n1e*dof+1]; FJ[2]=FFJ[n1e*dof+2];
	
	for (i=0;i<3;i++){
	  sum=0.0;
	  for (p=0;p<3;p++){
	    sum+=E[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
	
	FF[n1e*dof] +=Fadd[0]; 	FF[n1e*dof+1] +=Fadd[1];   FF[n1e*dof+2] +=Fadd[2];
	
      }else if(ibm->edgefrontnodesI[ec]==2){
	//I=2, J=5 , P=1
  	e2=UNIT(A2);
	n2=CROSS(e2,N);
	
	E[0][0]=1-2*n2.x*n2.x; E[0][1]=-2*n2.x*n2.y;  E[0][2]=-2*n2.x*n2.z;
	E[1][0]=-2*n2.y*n2.x;  E[1][1]=1-2*n2.y*n2.y; E[1][2]=-2*n2.y*n2.z;
	E[2][0]=-2*n2.z*n2.x;  E[2][1]=-2*n2.z*n2.y;  E[2][2]=1-2*n2.z*n2.z;

	FJ[0]=FFJ[n2e*dof]; FJ[1]=FFJ[n2e*dof+1]; FJ[2]=FFJ[n2e*dof+2];

	for (i=0;i<3;i++){
	  sum=0.0;
	  for (p=0;p<3;p++){
	    sum+=E[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}

	FF[n2e*dof] +=Fadd[0]; 	FF[n2e*dof+1] +=Fadd[1];   FF[n2e*dof+2] +=Fadd[2];

      }else if(ibm->edgefrontnodesI[ec]==3){
	//I=3, J=6 , P=2
	e3=UNIT(A3);
	n3=CROSS(e3,N);

	E[0][0]=1-2*n3.x*n3.x; E[0][1]=-2*n3.x*n3.y;  E[0][2]=-2*n3.x*n3.z;
	E[1][0]=-2*n3.y*n3.x;  E[1][1]=1-2*n3.y*n3.y; E[1][2]=-2*n3.y*n3.z;
	E[2][0]=-2*n3.z*n3.x;  E[2][1]=-2*n3.z*n3.y;  E[2][2]=1-2*n3.z*n3.z;
	
	FJ[0]=FFJ[n3e*dof]; FJ[1]=FFJ[n3e*dof+1]; FJ[2]=FFJ[n3e*dof+2];
	
	for (i=0;i<3;i++){
	  sum=0.0;
	  for (p=0;p<3;p++){
	    sum+=E[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
	
	FF[n3e*dof] +=Fadd[0]; 	FF[n3e*dof+1] +=Fadd[1]; 	FF[n3e*dof+2] +=Fadd[2];
      }
    }
  }
  

  VecRestoreArray(fem->FJ, &FFJ); 
  VecRestoreArray(fem->Fint, &FF); 
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode GhostFree(PetscInt edge_n, FE *fem) {
  
  IBMNodes *ibm=fem->ibm;
  PetscInt n1e, n2e, n3e, ec, be, i;

  struct Cmpnts x1, x2, x3, x4, x5, x6, a1, a2, a3, e1, e2, e3, n1, n2, n3, h1, h2, h3, n;
  for (ec=0; ec<ibm->n_ghosts; ec++) { //Go through ghost nodes
    be=ibm->belmts[ec];
    n1e=ibm->nv1[be]; n2e=ibm->nv2[be]; n3e=ibm->nv3[be];
    
    //currentlocation
    x1.x=ibm->x_bp[n1e]; x1.y=ibm->y_bp[n1e]; x1.z=ibm->z_bp[n1e];
    x2.x=ibm->x_bp[n2e]; x2.y=ibm->y_bp[n2e]; x2.z=ibm->z_bp[n2e];
    x3.x=ibm->x_bp[n3e]; x3.y=ibm->y_bp[n3e]; x3.z=ibm->z_bp[n3e];

    a1=MINUS(x3, x2) , a2=MINUS(x1, x3), a3=MINUS(x2, x1);

    struct Cmpnts nt;
    n.x=ibm->nf_x[be]; n.y=ibm->nf_y[be]; n.z=ibm->nf_z[be];
    
    if(ibm->belmtsedge[ec]==edge_n){ //check if it is on right edge
      if(ibm->edgefrontnodesI[ec]==1){
  	//I=1, J=4 , P=3
  	e1=UNIT(a1);
  	n1=CROSS(e1, n);
  	h1=AMULT(DOT(n1, a3), n1);
  	x4=PLUS(x1,AMULT(2., h1));
  	ibm->p4x[be]=x4.x; ibm->p4y[be]=x4.y; ibm->p4z[be]=x4.z;
	ibm->p4z[be]=x4.x*x4.x;

      }else if(ibm->edgefrontnodesI[ec]==2){
  	//I=2, J=5 , P=1
  	e2=UNIT(a2);
  	n2=CROSS(e2, n);
  	h2=AMULT(DOT(n2, a1), n2);
  	x5=PLUS(x2,AMULT(2., h2));
  	ibm->p5x[be]=x5.x; ibm->p5y[be]=x5.y; ibm->p5z[be]=x5.z;
	ibm->p5z[be]=x5.x*x5.x;

      }else if(ibm->edgefrontnodesI[ec]==3){
  	//I=3, J=6 , P=2
  	e3=UNIT(a3);
  	n3=CROSS(e3, n);
  	h3=AMULT(DOT(n3, a2), n3);
  	x6=PLUS(x3,AMULT(2., h3));
  	ibm->p6x[be]=x6.x; ibm->p6y[be]=x6.y; ibm->p6z[be]=x6.z;
	ibm->p6z[be]=x6.x*x6.x;

      }
    }
    
  }

  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode ModifyGhostFree(PetscInt edge_n, FE *fem) {
  
  IBMNodes *ibm=fem->ibm;
  PetscInt n1e, n2e, n3e, ec, be, i, j, m, p;
  
  struct Cmpnts x1, x2, x3, a1, a2, a3, e1, e2, e3, n, n1, n2, n3;
  PetscReal A, c, nIDaP, sum, sum1, sum2;
  PetscReal EI[3][3], EP[3][3], EF[3][3], Er1[3][3], Er2[3][3], Er3[3][3], FJ[3], Fadd[3];

  PetscReal NM1[3][3], NM2[3][3], nITPaP[3][3], eINM1[3][3], eINM2[3][3], CI[3][3], nCI[3][3], nITPnI[3][3];
  PetscReal IMnTPn[3][3], M1[3][3], M2[3][3];
  struct Cmpnts dx21, dx31;

  PetscReal *FF, *FFJ;
  VecGetArray(fem->Fint, &FF);
  VecGetArray(fem->FJ, &FFJ);
  for (ec=0; ec<ibm->n_ghosts; ec++) { //Go through ghost nodes
    be=ibm->belmts[ec];
    n1e=ibm->nv1[be]; n2e=ibm->nv2[be]; n3e=ibm->nv3[be];

    // currentlocation
    x1.x=ibm->x_bp[n1e]; x1.y=ibm->y_bp[n1e]; x1.z=ibm->z_bp[n1e];
    x2.x=ibm->x_bp[n2e]; x2.y=ibm->y_bp[n2e]; x2.z=ibm->z_bp[n2e];
    x3.x=ibm->x_bp[n3e]; x3.y=ibm->y_bp[n3e]; x3.z=ibm->z_bp[n3e];

    dx21=MINUS(x2, x1); dx31=MINUS(x3, x1); //dx21:g1 , dx31:g2
    a1=MINUS(x3, x2) , a2=MINUS(x1, x3), a3=MINUS(x2, x1);
 
    A=ibm->dA[be]; 
    A=0.5*SIZE(CROSS(dx21, dx31)); 
    n.x=ibm->nf_x[be]; n.y=ibm->nf_y[be]; n.z=ibm->nf_z[be];
    //  n=UNIT(CROSS(dx21,dx31));
  
    IMnTPn[0][0]=1.-n.x*n.x;  IMnTPn[0][1]=-n.x*n.y;      IMnTPn[0][2]=-n.x*n.z;
    IMnTPn[1][0]=-n.y*n.x;    IMnTPn[1][1]=1.-n.y*n.y;    IMnTPn[1][2]=-n.y*n.z;
    IMnTPn[2][0]=-n.z*n.x;    IMnTPn[2][1]=-n.z*n.y;      IMnTPn[2][2]=1.-n.z*n.z;  

    M1[0][0]=0.;        M1[0][1]=-dx21.z;    M1[0][2]=dx21.y;
    M1[1][0]=dx21.z;    M1[1][1]=0.;         M1[1][2]=-dx21.x;
    M1[2][0]=-dx21.y;   M1[2][1]=dx21.x;     M1[2][2]=0.;

    M2[0][0]=0.;        M2[0][1]=-dx31.z;  M2[0][2]=dx31.y;
    M2[1][0]=dx31.z;    M2[1][1]=0.;       M2[1][2]=-dx31.x;
    M2[2][0]=-dx31.y;   M2[2][1]=dx31.x;   M2[2][2]=0.;

    for (i=0; i<3; i++){
      for (j=0; j<3; j++){
	sum1=0.0; sum2=0.0;
	for (m=0; m<3; m++){
	  sum1+=IMnTPn[i][m]*M1[m][j]/(2.*A);
	  sum2+=IMnTPn[i][m]*M2[m][j]/(2.*A);
	}
	NM1[i][j]=sum1; NM2[i][j]=sum2;
      }
    }
    
    if(ibm->belmtsedge[ec]==edge_n){ //check if it is on right edge
      //--------------------------------------I=1,  P=3 , F=2 , J=4 ----------------------------------------------------------------------
      if(ibm->edgefrontnodesI[ec]==1){
	//I=1, P=3 , F=2 , J=4 
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    EI[i][j]=0.0; EP[i][j]=0.0; EF[i][j]=0.0;
	  }
	}
	e1=UNIT(a1);
	n1=CROSS(e1, n);

	nIDaP=DOT(n1, a3); //nI.ap

	nITPaP[0][0]=n1.x*a3.x; nITPaP[0][1]=n1.x*a3.y; nITPaP[0][2]=n1.x*a3.z; //nI Tensor product ap
	nITPaP[1][0]=n1.y*a3.x; nITPaP[1][1]=n1.y*a3.y; nITPaP[1][2]=n1.y*a3.z;
	nITPaP[2][0]=n1.z*a3.x; nITPaP[2][1]=n1.z*a3.y; nITPaP[2][2]=n1.z*a3.z;

	eINM1[0][0]=e1.y*NM1[2][0]-e1.z*NM1[1][0]; eINM1[0][1]=e1.y*NM1[2][1]-e1.z*NM1[1][1];  eINM1[0][2]=e1.y*NM1[2][2]-e1.z*NM1[1][2];
	eINM1[1][0]=e1.z*NM1[0][0]-e1.x*NM1[2][0]; eINM1[1][1]=e1.z*NM1[0][1]-e1.x*NM1[2][1];  eINM1[1][2]=e1.z*NM1[0][2]-e1.x*NM1[2][2];
	eINM1[2][0]=e1.x*NM1[1][0]-e1.y*NM1[0][0]; eINM1[2][1]=e1.x*NM1[1][1]-e1.y*NM1[0][1];  eINM1[2][2]=e1.x*NM1[1][2]-e1.y*NM1[0][2];

	eINM2[0][0]=e1.y*NM2[2][0]-e1.z*NM2[1][0]; eINM2[0][1]=e1.y*NM2[2][1]-e1.z*NM2[1][1];  eINM2[0][2]=e1.y*NM2[2][2]-e1.z*NM2[1][2];
	eINM2[1][0]=e1.z*NM2[0][0]-e1.x*NM2[2][0]; eINM2[1][1]=e1.z*NM2[0][1]-e1.x*NM2[2][1];  eINM2[1][2]=e1.z*NM2[0][2]-e1.x*NM2[2][2];
	eINM2[2][0]=e1.x*NM2[1][0]-e1.y*NM2[0][0]; eINM2[2][1]=e1.x*NM2[1][1]-e1.y*NM2[0][1];  eINM2[2][2]=e1.x*NM2[1][2]-e1.y*NM2[0][2];

	nITPnI[0][0]=n1.x*n1.x;  nITPnI[0][1]=n1.x*n1.y;  nITPnI[0][2]=n1.x*n1.z;
	nITPnI[1][0]=n1.y*n1.x;  nITPnI[1][1]=n1.y*n1.y;  nITPnI[1][2]=n1.y*n1.z;
	nITPnI[2][0]=n1.z*n1.x;  nITPnI[2][1]=n1.z*n1.y;  nITPnI[2][2]=n1.z*n1.z;

	c=1./pow(SIZE(a1),2.); //function(eI,aI)
	CI[0][0]=c*(SIZE(a1)-e1.x*a1.x);   CI[0][1]=c*(-e1.x*a1.y);         CI[0][2]=c*(-e1.x*a1.z);
	CI[1][0]=c*(-e1.y*a1.x);           CI[1][1]=c*(SIZE(a1)-e1.y*a1.y); CI[1][2]=c*(-e1.y*a1.z);
	CI[2][0]=c*(-e1.z*a1.x);           CI[2][1]=c*(-e1.z*a1.y);         CI[2][2]=c*(SIZE(a1)-e1.z*a1.z);

	nCI[0][0]=n.z*CI[1][0]-n.y*CI[2][0]; nCI[0][1]=n.z*CI[1][1]-n.y*CI[2][1]; nCI[0][2]=n.z*CI[1][2]-n.y*CI[2][2];
	nCI[1][0]=n.x*CI[2][0]-n.z*CI[0][0]; nCI[1][1]=n.x*CI[2][1]-n.z*CI[0][1]; nCI[1][2]=n.x*CI[2][2]-n.z*CI[0][2];
	nCI[2][0]=n.y*CI[0][0]-n.x*CI[1][0]; nCI[2][1]=n.y*CI[0][1]-n.x*CI[1][1]; nCI[2][2]=n.y*CI[0][2]-n.x*CI[1][2];

	//EI
	EI[0][0]=1.-2.*nITPnI[0][0]; EI[0][1]=-2.*nITPnI[0][1];  EI[0][2]=-2.*nITPnI[0][2];
	EI[1][0]=-2.*nITPnI[1][0];  EI[1][1]=1.-2.*nITPnI[1][1]; EI[1][2]=-2.*nITPnI[1][2];
	EI[2][0]=-2.*nITPnI[2][0];  EI[2][1]=-2.*nITPnI[2][1];  EI[2][2]=1.-2.*nITPnI[2][2];
	//EP
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*nCI[m][j];
	    }
	    EP[i][j]=2*nIDaP*nCI[i][j]+sum;
	  }
	}
	//EF
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*nCI[m][j];
	    }
	    EF[i][j]=-2*nIDaP*nCI[i][j]+sum+2*nITPnI[i][j];
	  }
	}
	
	//Er1
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM1[m][j]+2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er1[i][j]=sum-2*nIDaP*eINM1[i][j]+2*nIDaP*eINM2[i][j];
	  }
	}
       	//Er2
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er2[i][j]=sum-2*nIDaP*eINM2[i][j];
	  }
	}
	//Er3
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*eINM1[m][j];
	    }
	    Er3[i][j]=sum+2*nIDaP*eINM1[i][j];
	  }
	}
	//I=1, P=3 , F=2
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    EI[i][j]+=Er1[i][j];
	    EF[i][j]+=Er2[i][j];
	    EP[i][j]+=Er3[i][j];
	  }
	}
	
	FJ[0]=FFJ[n1e*dof]; FJ[1]=FFJ[n1e*dof+1]; FJ[2]=FFJ[n1e*dof+2];

	//I=1
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EI[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}

	FF[n1e*dof] +=Fadd[0]; 	FF[n1e*dof+1] +=Fadd[1]; FF[n1e*dof+2] +=Fadd[2];

	//F=2
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EF[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n2e*dof] +=Fadd[0]; 	FF[n2e*dof+1] +=Fadd[1]; FF[n2e*dof+2] +=Fadd[2];


	//P=3
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EP[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n3e*dof] +=Fadd[0]; 	FF[n3e*dof+1] +=Fadd[1]; FF[n3e*dof+2] +=Fadd[2];
	
	//--------------------------------------I=2, P=1 , F=3 , J=5 ----------------------------------------------------------------------	
      }else if(ibm->edgefrontnodesI[ec]==2){
	//I=2, P=1 , F=3 ,J=5 
  	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    EI[i][j]=0.0; EP[i][j]=0.0; EF[i][j]=0.0;
	  }
	}
	e2=UNIT(a2);
	n2=CROSS(e2, n);

	nIDaP=DOT(n2, a1); //nI.ap

	nITPaP[0][0]=n2.x*a1.x; nITPaP[0][1]=n2.x*a1.y; nITPaP[0][2]=n2.x*a1.z; //nI Tensor product ap
	nITPaP[1][0]=n2.y*a1.x; nITPaP[1][1]=n2.y*a1.y; nITPaP[1][2]=n2.y*a1.z;
	nITPaP[2][0]=n2.z*a1.x; nITPaP[2][1]=n2.z*a1.y; nITPaP[2][2]=n2.z*a1.z;

	eINM1[0][0]=e2.y*NM1[2][0]-e2.z*NM1[1][0]; eINM1[0][1]=e2.y*NM1[2][1]-e2.z*NM1[1][1];  eINM1[0][2]=e2.y*NM1[2][2]-e2.z*NM1[1][2];
	eINM1[1][0]=e2.z*NM1[0][0]-e2.x*NM1[2][0]; eINM1[1][1]=e2.z*NM1[0][1]-e2.x*NM1[2][1];  eINM1[1][2]=e2.z*NM1[0][2]-e2.x*NM1[2][2];
	eINM1[2][0]=e2.x*NM1[1][0]-e2.y*NM1[0][0]; eINM1[2][1]=e2.x*NM1[1][1]-e2.y*NM1[0][1];  eINM1[2][2]=e2.x*NM1[1][2]-e2.y*NM1[0][2];

	eINM2[0][0]=e2.y*NM2[2][0]-e2.z*NM2[1][0]; eINM2[0][1]=e2.y*NM2[2][1]-e2.z*NM2[1][1];  eINM2[0][2]=e2.y*NM2[2][2]-e2.z*NM2[1][2];
	eINM2[1][0]=e2.z*NM2[0][0]-e2.x*NM2[2][0]; eINM2[1][1]=e2.z*NM2[0][1]-e2.x*NM2[2][1];  eINM2[1][2]=e2.z*NM2[0][2]-e2.x*NM2[2][2];
	eINM2[2][0]=e2.x*NM2[1][0]-e2.y*NM2[0][0]; eINM2[2][1]=e2.x*NM2[1][1]-e2.y*NM2[0][1];  eINM2[2][2]=e2.x*NM2[1][2]-e2.y*NM2[0][2];

	nITPnI[0][0]=n2.x*n2.x;  nITPnI[0][1]=n2.x*n2.y;  nITPnI[0][2]=n2.x*n2.z;
	nITPnI[1][0]=n2.y*n2.x;  nITPnI[1][1]=n2.y*n2.y;  nITPnI[1][2]=n2.y*n2.z;
	nITPnI[2][0]=n2.z*n2.x;  nITPnI[2][1]=n2.z*n2.y;  nITPnI[2][2]=n2.z*n2.z;

	c=1./pow(SIZE(a2),2); //function(eI,aI)
	CI[0][0]=c*(SIZE(a2)-e2.x*a2.x);   CI[0][1]=c*(-e2.x*a2.y);         CI[0][2]=c*(-e2.x*a2.z);
	CI[1][0]=c*(-e2.y*a2.x);           CI[1][1]=c*(SIZE(a2)-e2.y*a2.y); CI[1][2]=c*(-e2.y*a2.z);
	CI[2][0]=c*(-e2.z*a2.x);           CI[2][1]=c*(-e2.z*a2.y);         CI[2][2]=c*(SIZE(a2)-e2.z*a2.z);
	
	nCI[0][0]=n.z*CI[1][0]-n.y*CI[2][0]; nCI[0][1]=n.z*CI[1][1]-n.y*CI[2][1]; nCI[0][2]=n.z*CI[1][2]-n.y*CI[2][2];
	nCI[1][0]=n.x*CI[2][0]-n.z*CI[0][0]; nCI[1][1]=n.x*CI[2][1]-n.z*CI[0][1]; nCI[1][2]=n.x*CI[2][2]-n.z*CI[0][2];
	nCI[2][0]=n.y*CI[0][0]-n.x*CI[1][0]; nCI[2][1]=n.y*CI[0][1]-n.x*CI[1][1]; nCI[2][2]=n.y*CI[0][2]-n.x*CI[1][2];

	//EI
	EI[0][0]=1-2*nITPnI[0][0]; EI[0][1]=-2*nITPnI[0][1];  EI[0][2]=-2*nITPnI[0][2];
	EI[1][0]=-2*nITPnI[1][0];  EI[1][1]=1-2*nITPnI[1][1]; EI[1][2]=-2*nITPnI[1][2];
	EI[2][0]=-2*nITPnI[2][0];  EI[2][1]=-2*nITPnI[2][1];  EI[2][2]=1-2*nITPnI[2][2];
	//EP
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*nCI[m][j];
	    }
	    EP[i][j]=2*nIDaP*nCI[i][j]+sum;
	  }
	}
	//EF
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*nCI[m][j];
	    }
	    EF[i][j]=-2*nIDaP*nCI[i][j]+sum+2*nITPnI[i][j];
	  }
	}
	
	//Er1
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM1[m][j]+2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er1[i][j]=sum-2*nIDaP*eINM1[i][j]+2*nIDaP*eINM2[i][j];
	  }
	}
       	//Er2
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er2[i][j]=sum-2*nIDaP*eINM2[i][j];
	  }
	}
	//Er3
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*eINM1[m][j];
	    }
	    Er3[i][j]=sum+2*nIDaP*eINM1[i][j];
	  }
	}
       //I=2,  P=1 , F=3
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    EI[i][j]+=Er2[i][j];
	    EF[i][j]+=Er3[i][j];
	    EP[i][j]+=Er1[i][j];
	  }
	}

	FJ[0]=FFJ[n2e*dof]; FJ[1]=FFJ[n2e*dof+1]; FJ[2]=FFJ[n2e*dof+2];
	//I=2
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EI[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
	FF[n2e*dof] +=Fadd[0]; 	FF[n2e*dof+1] +=Fadd[1]; FF[n2e*dof+2] +=Fadd[2];

	//F=3
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EF[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n3e*dof] +=Fadd[0]; 	FF[n3e*dof+1] +=Fadd[1]; FF[n3e*dof+2] +=Fadd[2];

	//P=1
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EP[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n1e*dof] +=Fadd[0]; 	FF[n1e*dof+1] +=Fadd[1]; FF[n1e*dof+2] +=Fadd[2];

	//--------------------------------------I=3, P=2 , F=1 , J=6 ----------------------------------------------------------------------	
      }else if(ibm->edgefrontnodesI[ec]==3){
	//I=3,  P=2 , F=1 , J=6 
	e3=UNIT(a3);
	n3=CROSS(e3, n);

	nIDaP=DOT(n3, a2); //nI.ap

	nITPaP[0][0]=n3.x*a2.x; nITPaP[0][1]=n3.x*a2.y; nITPaP[0][2]=n3.x*a2.z; //nI Tensor product ap
	nITPaP[1][0]=n3.y*a2.x; nITPaP[1][1]=n3.y*a2.y; nITPaP[1][2]=n3.y*a2.z;
	nITPaP[2][0]=n3.z*a2.x; nITPaP[2][1]=n3.z*a2.y; nITPaP[2][2]=n3.z*a2.z;

	eINM1[0][0]=e3.y*NM1[2][0]-e3.z*NM1[1][0]; eINM1[0][1]=e3.y*NM1[2][1]-e3.z*NM1[1][1];  eINM1[0][2]=e3.y*NM1[2][2]-e3.z*NM1[1][2];
	eINM1[1][0]=e3.z*NM1[0][0]-e3.x*NM1[2][0]; eINM1[1][1]=e3.z*NM1[0][1]-e3.x*NM1[2][1];  eINM1[1][2]=e3.z*NM1[0][2]-e3.x*NM1[2][2];
	eINM1[2][0]=e3.x*NM1[1][0]-e3.y*NM1[0][0]; eINM1[2][1]=e3.x*NM1[1][1]-e3.y*NM1[0][1];  eINM1[2][2]=e3.x*NM1[1][2]-e3.y*NM1[0][2];

	eINM2[0][0]=e3.y*NM2[2][0]-e3.z*NM2[1][0]; eINM2[0][1]=e3.y*NM2[2][1]-e3.z*NM2[1][1];  eINM2[0][2]=e3.y*NM2[2][2]-e3.z*NM2[1][2];
	eINM2[1][0]=e3.z*NM2[0][0]-e3.x*NM2[2][0]; eINM2[1][1]=e3.z*NM2[0][1]-e3.x*NM2[2][1];  eINM2[1][2]=e3.z*NM2[0][2]-e3.x*NM2[2][2];
	eINM2[2][0]=e3.x*NM2[1][0]-e3.y*NM2[0][0]; eINM2[2][1]=e3.x*NM2[1][1]-e3.y*NM2[0][1];  eINM2[2][2]=e3.x*NM2[1][2]-e3.y*NM2[0][2];

	nITPnI[0][0]=n3.x*n3.x;  nITPnI[0][1]=n3.x*n3.y;  nITPnI[0][2]=n3.x*n3.z;
	nITPnI[1][0]=n3.y*n3.x;  nITPnI[1][1]=n3.y*n3.y;  nITPnI[1][2]=n3.y*n3.z;
	nITPnI[2][0]=n3.z*n3.x;  nITPnI[2][1]=n3.z*n3.y;  nITPnI[2][2]=n3.z*n3.z;

	c=1./pow(SIZE(a3),2); //function(eI,aI)
	CI[0][0]=c*(SIZE(a3)-e3.x*a3.x);   CI[0][1]=c*(-e3.x*a3.y);         CI[0][2]=c*(-e3.x*a3.z);
	CI[1][0]=c*(-e3.y*a3.x);           CI[1][1]=c*(SIZE(a3)-e3.y*a3.y); CI[1][2]=c*(-e3.y*a3.z);
	CI[2][0]=c*(-e3.z*a3.x);           CI[2][1]=c*(-e3.z*a3.y);         CI[2][2]=c*(SIZE(a3)-e3.z*a3.z);

	nCI[0][0]=n.z*CI[1][0]-n.y*CI[2][0]; nCI[0][1]=n.z*CI[1][1]-n.y*CI[2][1]; nCI[0][2]=n.z*CI[1][2]-n.y*CI[2][2];
	nCI[1][0]=n.x*CI[2][0]-n.z*CI[0][0]; nCI[1][1]=n.x*CI[2][1]-n.z*CI[0][1]; nCI[1][2]=n.x*CI[2][2]-n.z*CI[0][2];
	nCI[2][0]=n.y*CI[0][0]-n.x*CI[1][0]; nCI[2][1]=n.y*CI[0][1]-n.x*CI[1][1]; nCI[2][2]=n.y*CI[0][2]-n.x*CI[1][2];

	//EI
	EI[0][0]=1-2*nITPnI[0][0]; EI[0][1]=-2*nITPnI[0][1];  EI[0][2]=-2*nITPnI[0][2];
	EI[1][0]=-2*nITPnI[1][0];  EI[1][1]=1-2*nITPnI[1][1]; EI[1][2]=-2*nITPnI[1][2];
	EI[2][0]=-2*nITPnI[2][0];  EI[2][1]=-2*nITPnI[2][1];  EI[2][2]=1-2*nITPnI[2][2];
	//EP
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*nCI[m][j];
	    }
	    EP[i][j]=2*nIDaP*nCI[i][j]+sum;
	  }
	}
	//EF
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*nCI[m][j];
	    }
	    EF[i][j]=-2*nIDaP*nCI[i][j]+sum+2*nITPnI[i][j];
	  }
	}
	
	//Er1
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM1[m][j]+2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er1[i][j]=sum-2*nIDaP*eINM1[i][j]+2*nIDaP*eINM2[i][j];
	  }
	}
       	//Er2
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=-2*nITPaP[i][m]*eINM2[m][j];
	    }
	    Er2[i][j]=sum-2*nIDaP*eINM2[i][j];
	  }
	}
	//Er3
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    sum=0.0; 
	    for (m=0; m<3; m++){
	      sum+=2*nITPaP[i][m]*eINM1[m][j];
	    }
	    Er3[i][j]=sum+2*nIDaP*eINM1[i][j];
	  }
	}
	//I=3, P=2 , F=1
	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    EI[i][j]+=Er3[i][j];
	    EF[i][j]+=Er1[i][j];
	    EP[i][j]+=Er2[i][j];
	  }
	}

	FJ[0]=FFJ[n3e*dof]; FJ[1]=FFJ[n3e*dof+1]; FJ[2]=FFJ[n3e*dof+2];
	//I=3
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EI[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
	FF[n3e*dof] +=Fadd[0]; 	FF[n3e*dof+1] +=Fadd[1]; FF[n3e*dof+2] +=Fadd[2];

	//F=1
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EF[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n1e*dof] +=Fadd[0]; 	FF[n1e*dof+1] +=Fadd[1]; FF[n1e*dof+2] +=Fadd[2];

	//P=2
	for (i=0; i<3; i++){
	  sum=0.0;
	  for (p=0; p<3; p++){
	    sum+=EP[p][i]*FJ[p];
	  }
	  Fadd[i]=sum;
	}
       	FF[n2e*dof] +=Fadd[0]; 	FF[n2e*dof+1] +=Fadd[1]; FF[n2e*dof+2] +=Fadd[2];
	
      }//else if

    }
  }

  VecRestoreArray(fem->FJ, &FFJ);
  VecRestoreArray(fem->Fint, &FF);

  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode NodeFix(PetscInt nb, FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscReal  *FFint, *FFext, *FFdyn, *xx, *xdd;
  
  VecGetArray(fem->Fint, &FFint);
  VecGetArray(fem->Fext, &FFext);
  VecGetArray(fem->Fdyn, &FFdyn);
  VecGetArray(fem->x, &xx);


  ibm->x_bp[nb] = ibm->x_bp0[nb]; //for kinematic contact
  ibm->y_bp[nb] = ibm->y_bp0[nb];
  ibm->z_bp[nb] = ibm->z_bp0[nb];
  xx[nb*dof] = ibm->x_bp0[nb];
  xx[nb*dof+1] = ibm->y_bp0[nb];
  xx[nb*dof+2] = ibm->z_bp0[nb];
  ibm->contact[nb] = 0;

  FFint[nb*dof] =0.0;
  FFint[nb*dof+1] =0.0;
  FFint[nb*dof+2] =0.0;
    
  FFext[nb*dof] =0.0;
  FFext[nb*dof+1] =0.0;
  FFext[nb*dof+2] =0.0;
    
  FFdyn[nb*dof] =0.0;
  FFdyn[nb*dof+1] =0.0;
  FFdyn[nb*dof+2] =0.0;

  VecRestoreArray(fem->Fdyn, &FFdyn);
  VecRestoreArray(fem->Fint, &FFint);
  VecRestoreArray(fem->Fext, &FFext);
  VecRestoreArray(fem->x, &xx);
  
  return(0);
}


PetscErrorCode EdgeDirectionalFix(PetscInt edge_n, PetscInt dir, FE *fem, Vec R) {
  PetscFunctionBeginUser;

  IBMNodes      *ibm  = fem->ibm;
  DMPlexGeomCtx *gctx = &fem->geom_ctx;
  PetscErrorCode ierr;
  PetscReal     *RRes;
  PetscInt       start=0, end=0, edge, nbc, nb;

  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];

  ierr = VecGetArray(R, &RRes); CHKERRQ(ierr);

  for (nbc=start; nbc<end; nbc++) { //fix boundary nodes
    nb = ibm->bnodes[nbc];
    /* In parallel, only the rank that owns nb writes to R; others skip. */
    PetscInt li = gctx->initialized ? gctx->ibm_to_local_idx[nb] : nb;
    if (li < 0) continue;

    switch (dir) {
    case 0:
      RRes[li*dof  ] = ibm->x_bp[nb] - ibm->x_bp0[nb];
      break;
    case 1:
      RRes[li*dof+1] = ibm->y_bp[nb] - ibm->y_bp0[nb];
      break;
    case 2:
      RRes[li*dof+2] = ibm->z_bp[nb] - ibm->z_bp0[nb];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Direction must be between 0 and 2");
    }
  }

  ierr = VecRestoreArray(R, &RRes); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EdgeFreeR(FE *fem, Vec R) {

  IBMNodes *ibm=fem->ibm;
  PetscReal *RRes;
  PetscInt  nb;
  
  VecGetArray(R, &RRes);
  
  for (nb=ibm->n_v; nb<ibm->n_v+ibm->n_ghosts; nb++) { //excludes ghost nodes from solver 
    RRes[nb*dof] =0.0;
    RRes[nb*dof+1] =0.0;
    RRes[nb*dof+2] =0.0; 
  }

  VecRestoreArray(R, &RRes);
 
  return(0);
}

PetscErrorCode GhostDirectionalFix(IBMNodes *ibm, PetscInt edge_n, PetscInt dir) {

  PetscInt i, ec, nv1, nv2, nv3, catch, edge, start, end, count=0;
  // modify starting count based on edge_n to ensure correct indexing of ghost nodes
  for (PetscInt ed=0; ed<edge_n; ed++){  
    count = count + ibm->n_bnodes[ed] - 1; //total count of ghost nodes up to current edge_n
  }
  
    //compute start&end
    start = 0; end = 0;
    for (edge=0; edge<edge_n+1; edge++) {
      end += ibm->n_bnodes[edge];
    }
    start = end - ibm->n_bnodes[edge_n];

    for (i=start; i<end-1; i++) {
      
      nv1 = ibm->bnodes[i+1];
      nv2 = ibm->bnodes[i];
      
      for (ec=0; ec<ibm->n_elmt; ec++) {
      	catch = 0;
      	if (nv1==ibm->nv1[ec] || nv1==ibm->nv2[ec] || nv1==ibm->nv3[ec]) {catch++;}
      	if (nv2==ibm->nv1[ec] || nv2==ibm->nv2[ec] || nv2==ibm->nv3[ec]) {catch++;}
      	if (catch==2) {
      	  if (ibm->nv1[ec]!=nv1 && ibm->nv1[ec]!=nv2) {
      	    nv3 = ibm->nv1[ec];
      	  } else if (ibm->nv2[ec]!=nv1 && ibm->nv2[ec]!=nv2) {
      	    nv3 = ibm->nv2[ec];
      	  } else{
      	    nv3 = ibm->nv3[ec];
      	  }
      	}	
      }
      switch (dir)
      {
      case 0:        
        // ibm->x_bp[ibm->n_v+count] = ibm->x_bp0[ibm->n_v+count];
        ibm->x_bp[ibm->n_v+count] = ibm->x_bp0[ibm->n_v+count] - (ibm->x_bp[nv3] - ibm->x_bp0[nv3]);
        ibm->y_bp[ibm->n_v+count] = ibm->y_bp[nv2] + ibm->y_bp[nv1] - ibm->y_bp[nv3];
        ibm->z_bp[ibm->n_v+count] = ibm->z_bp[nv2] + ibm->z_bp[nv1] - ibm->z_bp[nv3];
        break;
      case 1:
        ibm->x_bp[ibm->n_v+count] = ibm->x_bp[nv2] + ibm->x_bp[nv1] - ibm->x_bp[nv3];        
        ibm->y_bp[ibm->n_v+count] = ibm->y_bp0[ibm->n_v+count] - (ibm->y_bp[nv3] - ibm->y_bp0[nv3]);
        ibm->z_bp[ibm->n_v+count] = ibm->z_bp[nv2] + ibm->z_bp[nv1] - ibm->z_bp[nv3];
        break;
      case 2:
        ibm->x_bp[ibm->n_v+count] = ibm->x_bp[nv2] + ibm->x_bp[nv1] - ibm->x_bp[nv3];
        ibm->y_bp[ibm->n_v+count] = ibm->y_bp[nv2] + ibm->y_bp[nv1] - ibm->y_bp[nv3];
        // ibm->z_bp[ibm->n_v+count] = - ibm->z_bp[nv1];        
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "Direction must be between 0 and 2");
      }
      
      count++;      
    }
  // }

  return(0);
}
