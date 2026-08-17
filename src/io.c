#include  "variables.h"
#include  "active_strain.h"
#include  <petscvec.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

extern PetscInt   dof, outghost, ConstitutiveLawNonLinear, contact, n_Fung_Coeffs, n_lin_model_coeffs;
extern PetscInt   legacy_restart_in;
extern PetscInt   lv_geom_process;

extern PetscReal  dt, char_length_x, char_length_y, char_length_z;
extern PetscReal  h0;

extern char in_dir[256];

extern PetscInt   ressmooth, inverse;
extern PetscInt   Adam;
extern PetscInt   par_jac;

extern PetscInt   muscle_activation;
extern PetscInt   manufactured_fexternal_export, manufactured;
extern struct Cmpnts PLUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts UNIT(struct Cmpnts v1);
extern PetscReal DOT(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal SIZE(struct Cmpnts v1);


PetscErrorCode Dimension(IBMNodes *ibm, PetscInt ibi) {
 
  char  string[128];
  FILE  *fd;
  char  filen[256];

  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] in_dir = '%s', ibi=%d\n", in_dir, ibi);

  ibm->ibi = ibi;
  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] in_dir dsdd= '%s', ibi=%d\n", in_dir, ibi);

  snprintf(filen, sizeof(filen), "%s/nlist%2.2d", in_dir, ibi);
  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] Reading file '%s'\n", filen);
  fd = fopen(filen, "r");
  // if (!fd) {
  //   // PetscPrintf(PETSC_COMM_SELF, "[Dimension] ERROR: Failed to open '%s'\n", filen);
  //   return PETSC_ERR_FILE_OPEN;
  // }
  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] Successfully opened '%s'\n", filen);
  fscanf(fd, "%i", &ibm->n_v);
  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] Number of vertices (n_v) = %d\n", ibm->n_v);
  fclose(fd);

  snprintf(filen, sizeof(filen), "%s/elist%2.2d", in_dir, ibi);
  // PetscPrintf(PETSC_COMM_SELF, "[Dimension] Reading file '%s'\n", filen);

  fd = fopen(filen, "r");
  // if (!fd) {
  //   // PetscPrintf(PETSC_COMM_SELF, "[Dimension] ERROR: Failed to open '%s'\n", filen);
  //   return PETSC_ERR_FILE_OPEN;
  // }
  fscanf(fd, "%i", &ibm->n_elmt);
  fclose(fd);
  
  snprintf(filen, sizeof(filen), "%s/blist%2.2d", in_dir, ibi);
  fd = fopen(filen, "r");
  // if (!fd) {
  //   // PetscPrintf(PETSC_COMM_SELF, "[Dimension] ERROR: Failed to open '%s'\n", filen);
  //   return PETSC_ERR_FILE_OPEN;
  // }
  fscanf(fd, "%i", &ibm->n_edge);

  fgets(string,128, fd);
  fscanf(fd, "%i", &ibm->n_ghosts);
  fclose(fd);

  return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode Create(IBMNodes *ibm, FE *fem, PetscInt ibi) {
 
  fem->ibm=ibm; //link ibm of fem to ibm
  
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->x_bp));
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->y_bp));
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->z_bp));
  
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->x_bp0));
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->y_bp0));
  PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->z_bp0));

  // if (muscle_activation){
  //   PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->x_bpi));
  //   PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->y_bpi));
  //   PetscMalloc((ibm->n_v + ibm->n_ghosts)*sizeof(PetscReal), &(ibm->z_bpi));    
  // }
  

  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscInt), &(ibm->nv1));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscInt), &(ibm->nv2));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscInt), &(ibm->nv3));
  
  PetscMalloc(ibm->n_elmt*sizeof(struct Cmpnts), &(ibm->n_fib));

  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->gamma_scale));
  for (PetscInt i = 0; i < ibm->n_elmt; i++) ibm->gamma_scale[i] = 1.0;
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->activation_delay));
  for (PetscInt i = 0; i < ibm->n_elmt; i++) ibm->activation_delay[i] = 0.0;

  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->thickness));
  for (PetscInt i = 0; i < ibm->n_elmt; i++) ibm->thickness[i] = h0;

  ibm->n_apex_pin = 0;

  PetscMalloc(ibm->n_edge*sizeof(PetscInt), &(ibm->n_bnodes));
  
  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->nv4));
  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->nv5));
  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->nv6));

  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->kve0));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->kve));
 
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(fem->StressM)); 
  PetscMalloc((dof + 2)*ibm->n_elmt*sizeof(PetscReal), &(fem->StrainM)); // 2 extra for adding principal in-plane strains 
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(fem->StressB)); 
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(fem->StrainB));
  PetscCalloc1(ibm->n_v, &(fem->IE));


  PetscCalloc1(ibm->n_v, &(fem->CE));
  PetscCalloc1(ibm->n_v, &(ibm->m));
  PetscCalloc1(ibm->n_v, &(fem->KE));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(fem->FC));

  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->dA0));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->dA)); 

  // Allocating linear model coeffiecients
  PetscMalloc(2*sizeof(PetscReal *), &(ibm->El));
  for(int i=0; i<2; i++){
    PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->El[i]));
  }
  PetscMalloc(2*sizeof(PetscReal *), &(ibm->E_epsilon));
  for(int i=0; i<2; i++){
    PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->E_epsilon[i]));
  }


  // PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->El)); 
  // PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->E_epsilon)); 
  
  
  // PetscMalloc(ibm->n_v*sizeof(PetscReal **), &(fem->dR_dE));
  // for(int i=0; i<ibm->n_v; i++){
  //   PetscMalloc(ibm->n_elmt*sizeof(PetscReal *), &(fem->dR_dE[i]));
  //   for (int j=0; j<ibm->n_elmt; j++){
  //     PetscMalloc(dof*sizeof(PetscReal), &(fem->dR_dE[i][j]));
  //   }
  // }
  
  if (inverse) {
    PetscMalloc(ibm->n_v*sizeof(PetscReal ***), &(fem->dR_dE));
    for(int i=0; i<ibm->n_v; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal **), &(fem->dR_dE[i]));
      for (int j=0; j<ibm->n_elmt; j++){
        PetscMalloc(dof*sizeof(PetscReal *), &(fem->dR_dE[i][j]));
        for (int k=0; k<dof; k++){
          PetscMalloc(2*sizeof(PetscReal), &(fem->dR_dE[i][j][k]));
        }
      }
    }
  }

  
  if (Adam){
    PetscInt n_coeffs;
    if (ConstitutiveLawNonLinear){
      n_coeffs = n_Fung_Coeffs;
    }
    else{
      n_coeffs = n_lin_model_coeffs;
    }
    PetscMalloc(n_coeffs*sizeof(PetscReal *), &(ibm->Adam_mestimate));
    for(int i=0; i<n_coeffs; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->Adam_mestimate[i]));
    }

    PetscMalloc(n_coeffs*sizeof(PetscReal *), &(ibm->Adam_vestimate));
    for(int i=0; i<n_coeffs; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->Adam_vestimate[i]));
    }
  }

  if (ConstitutiveLawNonLinear){

    // Allocating Fung's model coeffiecients
    PetscMalloc(n_Fung_Coeffs*sizeof(PetscReal *), &(ibm->Fung_coeffs));
    for(int i=0; i<n_Fung_Coeffs; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->Fung_coeffs[i]));
    }

    // Allocating smooth Fung's model coeffiecients
    PetscMalloc(n_Fung_Coeffs*sizeof(PetscReal *), &(ibm->Fung_coeffs_smth));
    for(int i=0; i<n_Fung_Coeffs; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->Fung_coeffs_smth[i]));
    }

    // Allocating 2d epsilon array of Fung model's coeffs, 
    // Epsilon arrays will be used in Computing Jacobian dR/dc_Fung
    PetscMalloc(n_Fung_Coeffs*sizeof(PetscReal *), &(ibm->Fung_epsilons));
    for(int i=0; i<n_Fung_Coeffs; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->Fung_epsilons[i]));
    }

    PetscMalloc(ibm->n_v*sizeof(PetscReal ***), &(fem->Jac_Fung));
    for(int i=0; i<ibm->n_v; i++){
      PetscMalloc(ibm->n_elmt*sizeof(PetscReal **), &(fem->Jac_Fung[i]));
      for (int j=0; j<ibm->n_elmt; j++){
        PetscMalloc(dof*sizeof(PetscReal *), &(fem->Jac_Fung[i][j]));
        for (int k=0; k<dof; k++){
          PetscMalloc(n_Fung_Coeffs*sizeof(PetscReal), &(fem->Jac_Fung[i][j][k]));
        }
      }
    }

  }

  if (inverse){    
    PetscInt n_coeffs;
    if (ConstitutiveLawNonLinear){
      n_coeffs = n_Fung_Coeffs;
    }
    else{
      n_coeffs = n_lin_model_coeffs;
    }    

    if (par_jac){
      MatCreate(PETSC_COMM_WORLD, &fem->Jacobian);
      MatSetSizes(fem->Jacobian, PETSC_DECIDE, PETSC_DECIDE, ibm->n_elmt * n_coeffs, ibm->n_v * dof);
      MatSetFromOptions(fem->Jacobian);    
      MatSetType(fem->Jacobian, MATDENSE);
      MatSetUp(fem->Jacobian);
      MatZeroEntries(fem->Jacobian);

      // PetscMPIInt rank;
      // MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

      // PetscInt M, N;
      // MatGetSize(fem->Jacobian, &M, &N);

      // if (rank == 0) {          
      //   MatCreateSeqDense(PETSC_COMM_SELF, ibm->n_elmt * n_coeffs, ibm->n_v * dof, NULL, &fem->J_seq);
      // }
    }
    
    if (ressmooth){
      
      PetscInt MAX_NUM_NEIGH_NODES = 10;

      PetscMalloc(ibm->n_v*sizeof(PetscInt *), &(ibm->neigh_nodes_ind));
      for(int i=0; i<ibm->n_v; i++){
        PetscMalloc(MAX_NUM_NEIGH_NODES*sizeof(PetscInt), &(ibm->neigh_nodes_ind[i]));
      }
  
    }
  }


  // if(muscle_activation){
  //   PetscErrorCode ierr;


  //   ierr = PetscMalloc(ibm->n_elmt * sizeof(ElemActData), &fem->act_data.elem_act_data); CHKERRQ(ierr);
    
  //   PetscInt nBasisVecs = 2;
  //   VecCreateSeq(PETSC_COMM_SELF, ibm->n_elmt * nBasisVecs * dof, &fem->act_data.g_e_target);
  //   VecSet(fem->act_data.g_e_target, 0.0); 

  //   for (PetscInt ec = 0; ec < ibm->n_elmt; ec++) {
  //     for (PetscInt i = 0; i < 3; i++) {
  //       for (PetscInt j = 0; j < 3; j++) {
  //           fem->act_data.elem_act_data[ec].Fa[i][j] = 0.0;
  //       }
  //     }
  //   }
    
  // }

  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->nf_x));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->Nf_x)); 
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->nf_y));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->Nf_y)); 
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->nf_z));
  PetscMalloc((ibm->n_elmt + 2*ibm->n_ghosts)*sizeof(PetscReal), &(ibm->Nf_z));

  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->ire));
  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->irv));
  PetscMalloc(ibm->n_elmt*sizeof(PetscInt), &(ibm->val));

  PetscCalloc1(ibm->n_v + ibm->n_ghosts, &(ibm->contact));

  PetscMalloc(16*ibm->n_elmt*sizeof(PetscInt), &(ibm->patch));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->G));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->G1));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->G2));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g1));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g2));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g3));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g1n));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g2n));
  PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(ibm->g3n));

  PetscMalloc(ibm->n_elmt*sizeof(struct Cmpnts), &(ibm->qvec));
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->radvec));

  return (0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
/* Create all PETSc Vecs for body fem.
 * Must be called after FEM_DMPlexGeomSetup() (when used) so that
 * geom_ctx.nOwnedVerts is available.
 *   - Parallel muscle-activation path: VecCreateMPI with per-rank owned local size.
 *   - Serial / non-DMPlex path: VecCreateSeq covering all nodes incl. ghost stencil. */
PetscErrorCode InitVecs(FE *fem)
{
  IBMNodes      *ibm = fem->ibm;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;

  if (ctx->initialized) {
    VecCreateMPI(PETSC_COMM_WORLD, dof * ctx->nOwnedVerts, dof * (ibm->n_v + ibm->n_ghosts), &fem->Res);
  } else {
    VecCreateSeq(PETSC_COMM_SELF, dof * (ibm->n_v + ibm->n_ghosts), &fem->Res);
  }

  VecDuplicate(fem->Res, &fem->x);
  VecDuplicate(fem->Res, &fem->xn);
  VecDuplicate(fem->Res, &fem->xnm1);
  VecDuplicate(fem->Res, &fem->xd);
  VecDuplicate(fem->Res, &fem->dx);
  VecDuplicate(fem->Res, &fem->xdd);
  VecDuplicate(fem->Res, &fem->y);
  VecDuplicate(fem->Res, &fem->yn);
  VecDuplicate(fem->Res, &fem->Fint);
  VecDuplicate(fem->Res, &fem->Fext);
  VecDuplicate(fem->Res, &fem->Fdyn);
  VecDuplicate(fem->Res, &fem->disp);
  VecDuplicate(fem->Res, &fem->FJ);
  VecDuplicate(fem->Res, &fem->Mass);
  VecDuplicate(fem->Res, &fem->Dissip);
  VecDuplicate(fem->Res, &fem->Fcnt);

  VecSet(fem->Res, 0.0);   VecSet(fem->x, 0.0);    VecSet(fem->xn, 0.0);   VecSet(fem->xnm1, 0.0);
  VecSet(fem->xd, 0.0);    VecSet(fem->xdd, 0.0);  VecSet(fem->y, 0.0);    VecSet(fem->yn, 0.0);
  VecSet(fem->Fint, 0.0);  VecSet(fem->Fext, 0.0); VecSet(fem->Fdyn, 0.0); VecSet(fem->FJ, 0.0);
  VecSet(fem->disp, 0.0);  VecSet(fem->Fcnt, 0.0);
  VecSet(fem->Mass, 0.0);  VecSet(fem->Dissip, 0.0);

  if (ressmooth) {
    VecDuplicate(fem->Res, &fem->Res_smth);
    VecSet(fem->Res_smth, 0.0);
  }

  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------

PetscErrorCode Input(IBMNodes *ibm, PetscInt ibi) {

  PetscInt  i, ii, nc, ec, n_elmt, n_v=0;
  char      string[128];
  FILE      *fd;
  char      filen[256];
  
  //--------------------------------------------Reading nodes list
  snprintf(filen, sizeof(filen), "%s/nlist%2.2d", in_dir, ibi);
  fd = fopen(filen, "r");
  fscanf(fd,"%i", &n_v);
  
  PetscReal  *x_bp, *y_bp, *z_bp, xd[3], theta=3.14/4.;
  PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
  PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
  PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
  
  i=-1;
  fgets(string,128, fd);// skip line one
  while (i+1<n_v) {
    i++;
    fscanf(fd, "%d %d  %d %le %le %le\n", &ii, &ii, &ii, &x_bp[i], &y_bp[i], &z_bp[i]);
    //fscanf(fd, "%d %f %f %f\n", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);
  }
  fclose(fd);
  //Transfer data to IBM
  ibm->n_v = n_v;
  xd[0] = 0.;  xd[1] = 0.;  xd[2] = 0.;

  //
  //if (ibi==0)  xd[0] = -1.01;
  //if (ibi==1)  xd[0] = 1.01;
  //

  for (nc=0; nc<n_v; nc++) { 
    ibm->x_bp[nc] = x_bp[nc]/char_length_x + xd[0];  ibm->y_bp[nc] = y_bp[nc]/char_length_y + xd[1];  ibm->z_bp[nc] = z_bp[nc]/char_length_z + xd[2];
    /* if (ibi==1) { //rotate 45deg */
    /*   ibm->x_bp[nc] = (x_bp[nc]*cos(theta)-y_bp[nc]*sin(theta))/char_length_x + xd[0];  ibm->y_bp[nc] = (x_bp[nc]*sin(theta)+y_bp[nc]*cos(theta))/char_length_y + xd[1];  ibm->z_bp[nc] = z_bp[nc]/char_length_z + xd[2]; */
    /* } */
    /* ibm->x_bp0[nc] = x_bp[nc]/char_length_x + xd[0];  ibm->y_bp0[nc] = y_bp[nc]/char_length_y + xd[1];  ibm->z_bp0[nc] = z_bp[nc]/char_length_z + xd[2]; */
    ibm->x_bp0[nc] = ibm->x_bp[nc];  ibm->y_bp0[nc] =  ibm->y_bp[nc];  ibm->z_bp0[nc] = ibm->z_bp[nc];  }

  PetscPrintf(PETSC_COMM_WORLD, "Number of nodes of list (body:%d) %d \n", ibm->ibi, ibm->n_v);
  
  //------------------------------------------Reading elements list
  snprintf(filen, sizeof(filen), "%s/elist%2.2d", in_dir, ibi);
  fd = fopen(filen, "r"); 
  fscanf(fd, "%i", &n_elmt);
  
  PetscInt  *nv1, *nv2, *nv3;  
  char  str[10]; 
  PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
  PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
  PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);
  
  i = 0;
  fgets(string, 128, fd);//skip one line
  while (i<n_elmt) {
    i++;
    fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", &ii, &ii, &ii, &ii, &ii, &ii, &ii, &ii, &ii, &ii, &ii, &nv1[i-1], &nv2[i-1], &nv3[i-1], &ii);
    //fscanf(fd, "%d %s %d %d %d %d\n", &ii, str, &nv1[i-1], &nv2[i-1], &nv3[i-1]);
    nv1[i-1] = nv1[i-1] - 1;  nv2[i-1] = nv2[i-1] - 1;  nv3[i-1] = nv3[i-1] - 1;
  }
  fclose(fd);
  //Transfer data to IBM
  ibm->n_elmt = n_elmt;
  for (ec=0; ec<n_elmt; ec++) {
    ibm->nv1[ec] = nv1[ec];  ibm->nv2[ec] = nv2[ec];  ibm->nv3[ec] = nv3[ec]; 
  }
  PetscPrintf(PETSC_COMM_WORLD, "Number of element of list(body:%d) %d \n", ibi, ibm->n_elmt);

  //--------------------------------------Reading Boundary nodes
  snprintf(filen, sizeof(filen), "%s/blist%2.2d", in_dir, ibi);
  fd = fopen(filen, "r");
  PetscInt  n_edge, *bnodes;
  fscanf(fd, "%i", &n_edge);
  
  PetscInt  *n_bnodes, sum_n_bnodes=0;
  
  PetscMalloc(n_edge*sizeof(PetscInt), &n_bnodes);
  i = -1;
  fgets(string,128, fd);// skip line one
  fscanf(fd, "%i", &(ibm->n_ghosts));
  fgets(string, 128, fd);// skip line two
  PetscPrintf(PETSC_COMM_WORLD, "Number of ghost nodes %d\n", ibm->n_ghosts);

  while (i+1<n_edge) {
    i++;
    fscanf(fd, "%d \n", &(n_bnodes[i]));
    sum_n_bnodes += n_bnodes[i];
  }
  
  PetscMalloc(sum_n_bnodes*sizeof(PetscInt), &bnodes);
  
  i = -1;
  while (!feof(fd)) {
    i++;
    fscanf(fd, "%d", &bnodes[i]);
  }
  fclose(fd);
  //Transfer data to IBM
  ibm->n_edge = n_edge;
  ibm->sum_n_bnodes = sum_n_bnodes;
  for (i=0; i<n_edge; i++) {ibm->n_bnodes[i] = n_bnodes[i];}
  
  PetscMalloc(ibm->sum_n_bnodes*sizeof(PetscInt), &(ibm->bnodes));
  for (i=0; i<sum_n_bnodes; i++) {ibm->bnodes[i] = bnodes[i]-1;}
  
  //------------------------------Form the patch nodes
  PetscInt  n1e, n2e, n3e, *nv4, *nv5, *nv6;

  PetscMalloc(n_elmt*sizeof(PetscInt), &nv4);
  PetscMalloc(n_elmt*sizeof(PetscInt), &nv5);
  PetscMalloc(n_elmt*sizeof(PetscInt), &nv6);
  for (i=0; i<n_elmt; i++) { // A milion means it does not have patch node (it is on boundary)
    nv4[i] = 1000000;  nv5[i] = 1000000;  nv6[i] = 1000000;
  }

  PetscInt   j=0, n1pe, n2pe, n3pe;
  PetscInt   mn, npe; //mn: mutual nodes counter , cn:column number
  PetscReal  cn;

  for (i=0; i<n_elmt; i++) {
    n1e = nv1[i];  n2e = nv2[i];  n3e = nv3[i];

    for (j=0; j<n_elmt; j++) {
      n1pe = nv1[j];  n2pe = nv2[j];  n3pe = nv3[j];
 
      mn = 0; cn = 0; npe = 0;
      if(n1e==n1pe || n1e==n2pe || n1e==n3pe){mn = mn+1;  cn = cn + 3.5;}
      if(n2e==n1pe || n2e==n2pe || n2e==n3pe){mn = mn+1;  cn = cn + 2.5;}
      if(n3e==n1pe || n3e==n2pe || n3e==n3pe){mn = mn+1;  cn = cn + 1.5;}
      
      if(mn==2){ //we catch the patch, now find the patch element number
  	if(n1pe!=n1e && n1pe!=n2e && n1pe!=n3e){
  	  npe = n1pe;
  	}else if(n2pe!=n1e && n2pe!=n2e && n2pe!=n3e){
  	  npe = n2pe;
  	}else{
  	  npe = n3pe;
  	}
  	// put it in right location
	if(cn==4.){
  	  nv4[i] = npe;
  	}else if(cn==5.){
  	  nv5[i] = npe;
  	}else{
          nv6[i] = npe;
  	}
      } //end if catch
    }// end neighbor elements check
  }// end patch find
   
  //Transfer data to IBM
  for (ec=0; ec<n_elmt; ec++) {
    ibm->nv4[ec] = nv4[ec];  ibm->nv5[ec] = nv5[ec];  ibm->nv6[ec] = nv6[ec];
  }

  PetscFree(x_bp);  PetscFree(y_bp);  PetscFree(z_bp);
  PetscFree(nv1);  PetscFree(nv2);  PetscFree(nv3);
  PetscFree(n_bnodes);  PetscFree(bnodes); 
  PetscFree(nv4);  PetscFree(nv5);  PetscFree(nv6);

  return(0);   
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode Output(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt  n_cells=3, i, nv;
  IBMNodes          *ibm  = fem->ibm;
  DMPlexGeomCtx     *gctx = &fem->geom_ctx;
  FILE      *f;
  char      filepath[80];

  /* --- Parallel Vec gather: all ranks must participate before rank-0 early return. ---
   * For each Vec, scatter locally-owned entries into an ibm-node-ordered buffer,
   * then MPI_Allreduce (SUM) so rank 0 has the full array for VTK output.        */
  PetscReal *g_Fint = NULL, *g_Fdyn = NULL, *g_xd = NULL, *g_Fcnt = NULL, *g_Fext = NULL;

  if (gctx->initialized) {
    PetscInt nbuf = ibm->n_v * dof;
    PetscMalloc1(nbuf, &g_Fint);
    PetscMalloc1(nbuf, &g_Fdyn);
    PetscMalloc1(nbuf, &g_xd);
    PetscMalloc1(nbuf, &g_Fcnt);
    PetscMalloc1(nbuf, &g_Fext);
    PetscMemzero(g_Fint,  nbuf * sizeof(PetscReal));
    PetscMemzero(g_Fdyn,  nbuf * sizeof(PetscReal));
    PetscMemzero(g_xd,    nbuf * sizeof(PetscReal));
    PetscMemzero(g_Fcnt,  nbuf * sizeof(PetscReal));
    PetscMemzero(g_Fext,  nbuf * sizeof(PetscReal));

#define GATHER_NODE_VEC(vec, buf) do { \
      const PetscReal *_a; \
      VecGetArrayRead((vec), &_a); \
      for (nv = 0; nv < ibm->n_v; nv++) { \
        PetscInt li = gctx->ibm_to_local_idx[nv]; \
        if (li < 0) continue; \
        (buf)[nv*dof  ] = _a[li*dof  ]; \
        (buf)[nv*dof+1] = _a[li*dof+1]; \
        (buf)[nv*dof+2] = _a[li*dof+2]; \
      } \
      VecRestoreArrayRead((vec), &_a); \
      MPI_Allreduce(MPI_IN_PLACE, (buf), nbuf, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD); \
    } while (0)

    GATHER_NODE_VEC(fem->Fint, g_Fint);
    GATHER_NODE_VEC(fem->Fdyn, g_Fdyn);
    GATHER_NODE_VEC(fem->xd,   g_xd);
    GATHER_NODE_VEC(fem->Fcnt, g_Fcnt);
    GATHER_NODE_VEC(fem->Fext, g_Fext);
#undef GATHER_NODE_VEC
  }

  if (rank != 0) {
    PetscFree(g_Fint); PetscFree(g_Fdyn);
    PetscFree(g_xd);   PetscFree(g_Fcnt);
    PetscFree(g_Fext);
    return 0;
  }


  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  // Create directory if it doesn't exist
  mkdir(dir, 0777);

  // Construct the file path
  snprintf(filepath, sizeof(filepath), "%s/surface%2.2d_%5.5d.vtk", dir, ibi, ti);

  f = fopen(filepath, "w"); // open file in specified directory
  if (!f) {
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", filepath);
  }


  //sprintf(filen, "surface%2.2d_%5.5d.vtk", ibi,ti);
  //f = fopen(filen, "w"); // open file
  

  PetscFPrintf(PETSC_COMM_WORLD, f, "# vtk DataFile Version 2.0\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "Surface Grid\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "ASCII\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "DATASET UNSTRUCTURED_GRID\n");
  
  PetscFPrintf(PETSC_COMM_WORLD, f, "POINTS  %d float\n",(ibm->n_v));
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", (ibm->x_bp[i]),(ibm->y_bp[i]),(ibm->z_bp[i]));
  }

  PetscFPrintf(PETSC_COMM_WORLD, f, "CELLS %d %d\n",ibm->n_elmt, (n_cells+1)*(ibm->n_elmt));
  for (i=0; i<ibm->n_elmt; i++) {
    PetscFPrintf(PETSC_COMM_WORLD,f, "%d  %d %d %d\n",n_cells,(ibm->nv1[i]),(ibm->nv2[i]),(ibm->nv3[i]));
  }
  
  PetscFPrintf(PETSC_COMM_WORLD, f, "CELL_TYPES %d\n",ibm->n_elmt);
  for (i=0; i<ibm->n_elmt; i++) {
    PetscFPrintf(PETSC_COMM_WORLD,f, "%d\n",5);
  }
  
  PetscFPrintf(PETSC_COMM_WORLD, f, "POINT_DATA %d\n", ibm->n_v);

  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS contact integer\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", ibm->contact[i]);
  }
  
  /*
  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS CE float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", fem->CE[i]);
  }
  */
  
  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS m float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->m[i]);
  }

  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS IE float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", fem->IE[i]);
  }
  /*
  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS dR_dE float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", fem->dR_dE[i][100]);
  }
  */
  
  /*
  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS KE float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", fem->KE[i]);
  }
  */
  
  
  /* displacement: compute directly from ibm arrays (Allreduced on all ranks). */
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS disp float\n");
  for (i=0; i<ibm->n_v; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
                 ibm->x_bp[i]-ibm->x_bp0[i],
                 ibm->y_bp[i]-ibm->y_bp0[i],
                 ibm->z_bp[i]-ibm->z_bp0[i]);
  }

  /* Force/velocity Vecs: serial path uses direct array; parallel path uses
   * pre-gathered ibm-node-ordered arrays from the Allreduce above.          */
  PetscReal *FF;
  PetscBool is_parallel = gctx->initialized;

  /* Write one VECTORS block.  buf != NULL → use gathered array (parallel);
   * buf == NULL → fall back to direct serial array read from vec.           */
#define OUTPUT_VEC(label, vec, buf) do { \
    PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS " label " float\n"); \
    PetscReal *_obuf = (PetscReal *)(buf); \
    if (is_parallel && _obuf != NULL) { \
      for (i = 0; i < ibm->n_v; i++) \
        PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", \
                     _obuf[i*dof], _obuf[i*dof+1], _obuf[i*dof+2]); \
    } else { \
      VecGetArray((vec), &FF); \
      for (i = 0; i < ibm->n_v; i++) \
        PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", \
                     FF[i*dof], FF[i*dof+1], FF[i*dof+2]); \
      VecRestoreArray((vec), &FF); \
    } \
  } while (0)

  OUTPUT_VEC("Fint", fem->Fint, g_Fint);
  OUTPUT_VEC("Fext", fem->Fext, g_Fext);
  OUTPUT_VEC("Fdyn", fem->Fdyn, g_Fdyn);
  OUTPUT_VEC("Fcnt", fem->Fcnt, g_Fcnt);
  OUTPUT_VEC("u",    fem->xd,   g_xd);

  if (manufactured && muscle_activation) {
    OUTPUT_VEC("xdd", fem->xdd, NULL);
  }

#undef OUTPUT_VEC

  /* Residual R = Fint - Fext + Fdyn, matching the assembly in
   * FormFunctionFEM (main.c) -- written so per-node convergence/loading can
   * be inspected directly in ParaView instead of only the scalar SNES norm. */
  if (muscle_activation) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Res float\n");
    if (is_parallel) {
      for (i = 0; i < ibm->n_v; i++)
        PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
                     g_Fint[i*dof]   - g_Fext[i*dof]   + g_Fdyn[i*dof],
                     g_Fint[i*dof+1] - g_Fext[i*dof+1] + g_Fdyn[i*dof+1],
                     g_Fint[i*dof+2] - g_Fext[i*dof+2] + g_Fdyn[i*dof+2]);
    } else {
      PetscReal *_fi, *_fe, *_fd;
      VecGetArray(fem->Fint, &_fi);
      VecGetArray(fem->Fext, &_fe);
      VecGetArray(fem->Fdyn, &_fd);
      for (i = 0; i < ibm->n_v; i++)
        PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
                     _fi[i*dof]   - _fe[i*dof]   + _fd[i*dof],
                     _fi[i*dof+1] - _fe[i*dof+1] + _fd[i*dof+1],
                     _fi[i*dof+2] - _fe[i*dof+2] + _fd[i*dof+2]);
      VecRestoreArray(fem->Fint, &_fi);
      VecRestoreArray(fem->Fext, &_fe);
      VecRestoreArray(fem->Fdyn, &_fd);
    }
  }

  PetscFree(g_Fint); PetscFree(g_Fdyn);
  PetscFree(g_xd);   PetscFree(g_Fcnt);
  PetscFree(g_Fext);

  PetscFPrintf(PETSC_COMM_WORLD, f, "CELL_DATA %d\n",ibm->n_elmt);
  
  // if (muscle_activation){

  
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "TENSORS Fa float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f %f %f %f %f %f %f\n",
  //     fem->act_data.elem_act_data[i].Fa[0][0], fem->act_data.elem_act_data[i].Fa[0][1], fem->act_data.elem_act_data[i].Fa[0][2], 
  //     fem->act_data.elem_act_data[i].Fa[1][0], fem->act_data.elem_act_data[i].Fa[1][1], fem->act_data.elem_act_data[i].Fa[1][2], 
  //     fem->act_data.elem_act_data[i].Fa[2][0], fem->act_data.elem_act_data[i].Fa[2][1], fem->act_data.elem_act_data[i].Fa[2][2]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS ge1 float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
  //     fem->act_data.elem_act_data[i].g_e[0].x, fem->act_data.elem_act_data[i].g_e[0].y, fem->act_data.elem_act_data[i].g_e[0].z);    
  //   }
  
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS ge2 float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
  //     fem->act_data.elem_act_data[i].g_e[1].x, fem->act_data.elem_act_data[i].g_e[1].y, fem->act_data.elem_act_data[i].g_e[1].z);    
  //   }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS G1 float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
  //     ibm->G1[i*dof], ibm->G1[i*dof+1], ibm->G1[i*dof+2]);    
  //   }
  
  //   PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS G2 float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
  //     ibm->G2[i*dof], ibm->G2[i*dof+1], ibm->G2[i*dof+2]);    
  //   }

  //   PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS FG1 float\n");
  //   PetscInt nBasisVecs = 2;
  //   Vec F;
  //   PetscReal  *farray;
  //   VecDuplicate(fem->act_data.g_e_target, &F);

  //   calculate_cov_basis(fem->x, F, fem);
  //   VecGetArray(F, &farray);
    
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",
  //     farray[dof*nBasisVecs*i + 0] ,farray[dof*nBasisVecs*i + 1] , farray[dof*nBasisVecs*i + 2] );    
  //   }
  //   VecRestoreArray(F, &farray);
  //   VecDestroy(&F);

  // }

  
  
  
  // ibm->G1[ec*dof]; g_cov[0].y = ibm->G1[ec*dof+1]; g_cov[0].z = ibm->G1[ec*dof+2];

  
  
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS kve float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",ibm->kve[i*dof], ibm->kve[i*dof+1], ibm->kve[i*dof+2]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS nf float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",ibm->nf_x[i], ibm->nf_y[i], ibm->nf_z[i]);
  // }
  
  if (lv_geom_process) {
    PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS nfib float\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",ibm->n_fib[i].x, ibm->n_fib[i].y, ibm->n_fib[i].z);
    }
  }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS StrainM float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",fem->StrainM[i*(dof+2)], fem->StrainM[i*(dof+2)+1], fem->StrainM[i*(dof+2)+2]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS PStrainM float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",fem->StrainM[i*(dof+2)+3], fem->StrainM[i*(dof+2)+4], 0.);
  // }
  
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS StrainB float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",fem->StrainB[i*dof], fem->StrainB[i*dof+1], fem->StrainB[i*dof+2]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS StressM float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",fem->StressM[i*dof], fem->StressM[i*dof+1], fem->StressM[i*dof+2]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS StressB float\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",fem->StressB[i*dof], fem->StressB[i*dof+1], fem->StressB[i*dof+2]);
  // }


  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS dA float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_elmt; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->dA[i]);
  }

  PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS dA0 float\n");
  PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  for (i=0; i<ibm->n_elmt; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->dA0[i]);
  }

  /* gamma = GammaOfTimeDelayed(t - activation_delay[ec]) * gamma_scale[ec] --
   * the actual scalar activation driving contraction in this element this
   * timestep, including both the propagating-wavefront delay (see
   * -lv_gamma_wave) and the static apex/base taper multiplier. Lets taper
   * on/off, wave delay, ramp shape, etc. be verified directly from the VTK
   * output. */
  if (muscle_activation) {
    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS gamma float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", GammaOfTimeCurrent(fem, i) * ibm->gamma_scale[i]);
    }
  }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS El float\n");
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->El[0][i]);
  // }

  // PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS nu float\n");
  // PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
  // for (i=0; i<ibm->n_elmt; i++) {
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->El[1][i]);
  // }

  if (ConstitutiveLawNonLinear){
    
    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS c_f float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[0][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A1 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[1][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A2 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[2][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A3 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[3][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A4 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[4][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A5 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[5][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS A6 float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[6][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS theta float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs[7][i]);
    }

    PetscFPrintf(PETSC_COMM_WORLD, f,  "SCALARS theta_smth float\n");
    PetscFPrintf(PETSC_COMM_WORLD, f,  "LOOKUP_TABLE default\n");
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%f \n", ibm->Fung_coeffs_smth[7][i]);
    }

  }
  
  fclose(f);

  /* sprintf(filen, "TipDisp.dat"); */
  /* f = fopen(filen, "a"); */
  /* PetscFPrintf(PETSC_COMM_WORLD, f, "%le  %le\n", ti*dt, ibm->z_bp[56-1]/0.04); */
  /* //PetscFPrintf(PETSC_COMM_WORLD, f, "%le  %le\n", ti*dt, ibm->z_bp[52-1]); */
  /* fclose(f); */
 
  return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
/* Write3DShellVTK — visualization-only 3D reconstruction of the shell wall:
 * one layer of triangular-prism (VTK_WEDGE) elements built by offsetting
 * each surface node inward/outward along its local normal by half the
 * node-averaged wall thickness (ibm->thickness[], updated once per
 * timestep by UpdateElementThickness() -- see active_strain.c). Does NOT
 * feed back into the mechanics; the shell simulation is unaffected. Same
 * technique (and same C prototype/gather assumptions) as
 * BioFEM-studies/lv_test/build_3d_shell.py, just run in-line during the
 * simulation instead of as an offline post-process, and using the real
 * per-element C33-derived thickness instead of that script's placeholder.
 *
 * Rank-0 only, like Output() -- assumes ibm->x_bp/y_bp/z_bp/thickness are
 * already globally valid on rank 0 (true after Output()'s own gather step
 * and UpdateElementThickness()'s Allreduce).                                */
PetscErrorCode Write3DShellVTK(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank != 0) return 0;

  IBMNodes *ibm = fem->ibm;
  PetscInt  n_v = ibm->n_v, n_elmt = ibm->n_elmt;

  /* Per-node normal: area-weighted sum of incident (already outward-
   * oriented) face normals, normalized. Same convention as the Python
   * reference implementation. */
  PetscReal *nx, *ny, *nz;
  PetscCalloc1(n_v, &nx); PetscCalloc1(n_v, &ny); PetscCalloc1(n_v, &nz);
  PetscReal *node_thick_sum;
  PetscInt  *node_thick_cnt;
  PetscCalloc1(n_v, &node_thick_sum);
  PetscCalloc1(n_v, &node_thick_cnt);

  for (PetscInt ec = 0; ec < n_elmt; ec++) {
    PetscInt n1 = ibm->nv1[ec], n2 = ibm->nv2[ec], n3 = ibm->nv3[ec];
    struct Cmpnts p0 = {ibm->x_bp[n1], ibm->y_bp[n1], ibm->z_bp[n1]};
    struct Cmpnts p1 = {ibm->x_bp[n2], ibm->y_bp[n2], ibm->z_bp[n2]};
    struct Cmpnts p2 = {ibm->x_bp[n3], ibm->y_bp[n3], ibm->z_bp[n3]};
    struct Cmpnts e1 = MINUS(p1, p0), e2 = MINUS(p2, p0);
    struct Cmpnts fn = CROSS(e1, e2);  /* area-weighted (not normalized) face normal */

    nx[n1] += fn.x; ny[n1] += fn.y; nz[n1] += fn.z;
    nx[n2] += fn.x; ny[n2] += fn.y; nz[n2] += fn.z;
    nx[n3] += fn.x; ny[n3] += fn.y; nz[n3] += fn.z;

    node_thick_sum[n1] += ibm->thickness[ec]; node_thick_cnt[n1]++;
    node_thick_sum[n2] += ibm->thickness[ec]; node_thick_cnt[n2]++;
    node_thick_sum[n3] += ibm->thickness[ec]; node_thick_cnt[n3]++;
  }

  PetscReal *node_thick;
  PetscMalloc1(n_v, &node_thick);
  for (PetscInt v = 0; v < n_v; v++) {
    PetscReal len = sqrt(nx[v]*nx[v] + ny[v]*ny[v] + nz[v]*nz[v]);
    if (len > 1.0e-14) { nx[v] /= len; ny[v] /= len; nz[v] /= len; }
    node_thick[v] = (node_thick_cnt[v] > 0) ? node_thick_sum[v] / (PetscReal)node_thick_cnt[v] : 0.0;
  }
  PetscFree(node_thick_sum); PetscFree(node_thick_cnt);

  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  mkdir(dir, 0777);
  char filepath[96];
  snprintf(filepath, sizeof(filepath), "%s/shell3d%2.2d_%5.5d.vtk", dir, (int)ibi, (int)ti);
  FILE *f = fopen(filepath, "w");
  if (!f) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", filepath); }

  fprintf(f, "# vtk DataFile Version 2.0\nLV 3D shell (visualization only)\nASCII\nDATASET UNSTRUCTURED_GRID\n");
  fprintf(f, "POINTS %d float\n", 2 * n_v);
  for (PetscInt v = 0; v < n_v; v++) {
    PetscReal h2 = 0.5 * node_thick[v];
    fprintf(f, "%f %f %f\n", ibm->x_bp[v] - h2*nx[v], ibm->y_bp[v] - h2*ny[v], ibm->z_bp[v] - h2*nz[v]);
  }
  for (PetscInt v = 0; v < n_v; v++) {
    PetscReal h2 = 0.5 * node_thick[v];
    fprintf(f, "%f %f %f\n", ibm->x_bp[v] + h2*nx[v], ibm->y_bp[v] + h2*ny[v], ibm->z_bp[v] + h2*nz[v]);
  }

  fprintf(f, "CELLS %d %d\n", n_elmt, 7 * n_elmt);
  for (PetscInt ec = 0; ec < n_elmt; ec++) {
    PetscInt n1 = ibm->nv1[ec], n2 = ibm->nv2[ec], n3 = ibm->nv3[ec];
    fprintf(f, "6 %d %d %d %d %d %d\n", (int)n1, (int)n2, (int)n3,
            (int)(n1 + n_v), (int)(n2 + n_v), (int)(n3 + n_v));
  }
  fprintf(f, "CELL_TYPES %d\n", n_elmt);
  for (PetscInt ec = 0; ec < n_elmt; ec++) fprintf(f, "13\n");  /* VTK_WEDGE */

  fprintf(f, "POINT_DATA %d\n", 2 * n_v);
  fprintf(f, "SCALARS layer float 1\nLOOKUP_TABLE default\n");
  for (PetscInt v = 0; v < n_v; v++) fprintf(f, "0.0\n");
  for (PetscInt v = 0; v < n_v; v++) fprintf(f, "1.0\n");
  fprintf(f, "SCALARS thickness float 1\nLOOKUP_TABLE default\n");
  for (PetscInt v = 0; v < n_v; v++) fprintf(f, "%f\n", node_thick[v]);
  for (PetscInt v = 0; v < n_v; v++) fprintf(f, "%f\n", node_thick[v]);

  fclose(f);
  PetscFree(nx); PetscFree(ny); PetscFree(nz); PetscFree(node_thick);
  PetscPrintf(PETSC_COMM_SELF, "3D shell VTK written to: %s\n", filepath);
  return 0;
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode OutputGhost(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt   n_cells=3, i;
  IBMNodes      *ibm  = fem->ibm;
  DMPlexGeomCtx *gctx = &fem->geom_ctx;
  PetscReal  x, y, z;
  PetscInt   ec, be, n1e, n2e, n3e, n_ghosts=0;
  PetscErrorCode ierr;

  n_ghosts = ibm->n_ghosts;

  /* Parallel Vec gather (same pattern as Output(), io.c:539): fem->Fint/
   * Fext/Fdyn are distributed Vecs -- VecGetArray only returns the calling
   * rank's OWNED slice, not the full n_v array. Reading it as if it were
   * the global array (the previous version of this function did) reads
   * whatever happens to sit at that offset in the local buffer once i
   * exceeds the local slice -- garbage, occasionally bit patterns that
   * print as NaN. Gather to every rank via Allreduce(SUM) over owned
   * entries only, same as Output() does, BEFORE the rank-0-only file I/O
   * below (all ranks must reach the collective). Ghost-node slots
   * (>= ibm->n_v) are never part of these Vecs at all -- written as an
   * explicit 0.0 further down, same as before. */
  PetscReal *g_Fint = NULL, *g_Fext = NULL, *g_Fdyn = NULL;
  if (gctx->initialized) {
    PetscInt nbuf = ibm->n_v * dof;
    ierr = PetscMalloc1(nbuf, &g_Fint); CHKERRQ(ierr);
    ierr = PetscMalloc1(nbuf, &g_Fext); CHKERRQ(ierr);
    ierr = PetscMalloc1(nbuf, &g_Fdyn); CHKERRQ(ierr);
    ierr = PetscMemzero(g_Fint, nbuf*sizeof(PetscReal)); CHKERRQ(ierr);
    ierr = PetscMemzero(g_Fext, nbuf*sizeof(PetscReal)); CHKERRQ(ierr);
    ierr = PetscMemzero(g_Fdyn, nbuf*sizeof(PetscReal)); CHKERRQ(ierr);

#define OUTGHOST_GATHER(vec, buf) do { \
      const PetscReal *_a; \
      PetscInt _nv; \
      ierr = VecGetArrayRead((vec), &_a); CHKERRQ(ierr); \
      for (_nv = 0; _nv < ibm->n_v; _nv++) { \
        PetscInt _li = gctx->ibm_to_local_idx[_nv]; \
        if (_li < 0) continue; \
        (buf)[_nv*dof  ] = _a[_li*dof  ]; \
        (buf)[_nv*dof+1] = _a[_li*dof+1]; \
        (buf)[_nv*dof+2] = _a[_li*dof+2]; \
      } \
      ierr = VecRestoreArrayRead((vec), &_a); CHKERRQ(ierr); \
      MPI_Allreduce(MPI_IN_PLACE, (buf), nbuf, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD); \
    } while (0)

    OUTGHOST_GATHER(fem->Fint, g_Fint);
    OUTGHOST_GATHER(fem->Fext, g_Fext);
    OUTGHOST_GATHER(fem->Fdyn, g_Fdyn);
#undef OUTGHOST_GATHER
  }

  if (rank != 0) {
    ierr = PetscFree(g_Fint); CHKERRQ(ierr);
    ierr = PetscFree(g_Fext); CHKERRQ(ierr);
    ierr = PetscFree(g_Fdyn); CHKERRQ(ierr);
    return 0;
  }

  FILE  *f;
  char  filen[80];

  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  // Create directory if it doesn't exist
  mkdir(dir, 0777);

  // Construct the file path
  snprintf(filen, sizeof(filen), "%s/surfaceghost%2.2d_%5.5d.vtk", dir, ibi, ti);
  f = fopen(filen, "w"); // open file in specified directory
  if (!f) {
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", filen);
  }

  PetscFPrintf(PETSC_COMM_WORLD, f, "# vtk DataFile Version 2.0\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "Surface Grid\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "ASCII\n");
  PetscFPrintf(PETSC_COMM_WORLD, f, "DATASET UNSTRUCTURED_GRID\n");
   
  PetscFPrintf(PETSC_COMM_WORLD, f, "POINTS  %d float\n",(ibm->n_v+n_ghosts));
  for (i=0; i<ibm->n_v+n_ghosts; i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", (ibm->x_bp[i]), (ibm->y_bp[i]), (ibm->z_bp[i]));
  }

  //add ghost nodes location
  /* for (i=ibm->n_v; i<(ibm->n_v+n_ghosts); i++) { */
  /*   ec = i - ibm->n_v; */
  /*   be = ibm->belmts[ec]; */
  /*   if(ibm->edgefrontnodesI[ec]==1){ */
  /*     x = ibm->p4x[be];  y = ibm->p4y[be];  z = ibm->p4z[be]; */
  /*   }else if(ibm->edgefrontnodesI[ec]==2){ */
  /*     x = ibm->p5x[be];  y = ibm->p5y[be];  z = ibm->p5z[be]; */
  /*   }else if(ibm->edgefrontnodesI[ec]==3){ */
  /*     x = ibm->p6x[be];  y = ibm->p6y[be];  z = ibm->p6z[be]; */
  /*   } */
    
  /*   PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", x, y, z); */
  /* } */

  PetscFPrintf(PETSC_COMM_WORLD, f, "CELLS %d %d\n", (ibm->n_elmt+2*n_ghosts), (n_cells+1)*(ibm->n_elmt+2*n_ghosts));
  for (i=0; i<ibm->n_elmt+2*n_ghosts; i++) {
    PetscFPrintf(PETSC_COMM_WORLD,f, "%d  %d %d %d\n", n_cells, (ibm->nv1[i]), (ibm->nv2[i]), (ibm->nv3[i]));
  }

  /* //add ghost elements  */
  /* for (i=ibm->n_elmt; i<(ibm->n_elmt+n_ghosts); i++) { */
  /*   ec = i-ibm->n_elmt; */
  /*   be = ibm->belmts[ec]; */
  /*   if(ibm->edgefrontnodesI[ec]==1){ */
  /*     n1e = ec+ibm->n_v;  n2e = ibm->nv2[be];  n3e = ibm->nv3[be]; */
  /*   }else if(ibm->edgefrontnodesI[ec]==2){ */
  /*     n1e = ibm->nv1[be];  n2e = ec+ibm->n_v;  n3e = ibm->nv3[be]; */
  /*   }else if(ibm->edgefrontnodesI[ec]==3){ */
  /*     n1e = ibm->nv1[be];  n2e = ibm->nv2[be];  n3e = ec+ibm->n_v; */
  /*   } */
  /*   PetscFPrintf(PETSC_COMM_WORLD,f, "%d  %d %d %d\n", n_cells, n1e, n2e, n3e); */
  /* } */

  PetscFPrintf(PETSC_COMM_WORLD, f, "CELL_TYPES %d\n",(ibm->n_elmt+2*n_ghosts));
  for (i=0; i<(ibm->n_elmt+2*n_ghosts); i++) {
    PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", 5);
  }

  PetscReal  *FF;
  PetscInt   nv;

  /* fem->disp/Fint/Fext/Fdyn are PETSc Vecs sized dof*ibm->n_v (real DOFs
   * only -- see Create()'s VecCreateMPI/VecCreateSeq calls, both of which
   * omit n_ghosts in the parallel path). Ghost nodes are not part of the
   * solved DOF space, so: disp is computed directly from ibm->x_bp/x_bp0
   * (never touching the Vec), and force fields are written as an explicit
   * 0 for the ghost range instead of reading/writing past the Vec's true
   * allocated size (which previously produced NaN/heap corruption). */
  PetscReal *dd;
  ierr = PetscMalloc1(3*(ibm->n_v+n_ghosts), &dd); CHKERRQ(ierr);
  for (nv=0; nv<ibm->n_v+n_ghosts; nv++) {
    dd[nv*dof] = ibm->x_bp[nv]-ibm->x_bp0[nv];
    dd[nv*dof+1] = ibm->y_bp[nv]-ibm->y_bp0[nv];
    dd[nv*dof+2] = ibm->z_bp[nv]-ibm->z_bp0[nv];
  }

  PetscFPrintf(PETSC_COMM_WORLD, f, "POINT_DATA %d\n", ibm->n_v+n_ghosts);

  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS disp float\n");
  for (i=0; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", dd[i*dof], dd[i*dof+1], dd[i*dof+2]);
  }
  ierr = PetscFree(dd); CHKERRQ(ierr);

  if (!gctx->initialized) { ierr = VecGetArray(fem->Fint, &FF); CHKERRQ(ierr); }
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fint float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscReal fx = gctx->initialized ? g_Fint[i*dof]   : FF[i*dof];
    PetscReal fy = gctx->initialized ? g_Fint[i*dof+1] : FF[i*dof+1];
    PetscReal fz = gctx->initialized ? g_Fint[i*dof+2] : FF[i*dof+2];
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", fx, fy, fz);
  }
  for (i=ibm->n_v; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "0.0 0.0 0.0\n");
  }
  if (!gctx->initialized) { ierr = VecRestoreArray(fem->Fint, &FF); CHKERRQ(ierr); }

  if (!gctx->initialized) { ierr = VecGetArray(fem->Fext, &FF); CHKERRQ(ierr); }
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fext float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscReal fx = gctx->initialized ? g_Fext[i*dof]   : FF[i*dof];
    PetscReal fy = gctx->initialized ? g_Fext[i*dof+1] : FF[i*dof+1];
    PetscReal fz = gctx->initialized ? g_Fext[i*dof+2] : FF[i*dof+2];
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", fx, fy, fz);
  }
  for (i=ibm->n_v; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "0.0 0.0 0.0\n");
  }
  if (!gctx->initialized) { ierr = VecRestoreArray(fem->Fext, &FF); CHKERRQ(ierr); }

  if (!gctx->initialized) { ierr = VecGetArray(fem->Fdyn, &FF); CHKERRQ(ierr); }
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fdyn float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscReal fx = gctx->initialized ? g_Fdyn[i*dof]   : FF[i*dof];
    PetscReal fy = gctx->initialized ? g_Fdyn[i*dof+1] : FF[i*dof+1];
    PetscReal fz = gctx->initialized ? g_Fdyn[i*dof+2] : FF[i*dof+2];
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", fx, fy, fz);
  }
  for (i=ibm->n_v; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "0.0 0.0 0.0\n");
  }
  if (!gctx->initialized) { ierr = VecRestoreArray(fem->Fdyn, &FF); CHKERRQ(ierr); }

  ierr = PetscFree(g_Fint); CHKERRQ(ierr);
  ierr = PetscFree(g_Fext); CHKERRQ(ierr);
  ierr = PetscFree(g_Fdyn); CHKERRQ(ierr);

  fclose(f);

  return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode LocationOut(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {

  IBMNodes      *ibm=fem->ibm;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  PetscViewer   viewer = NULL;
  char          filen[256];
  PetscInt      fd;

  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  // Create directory if it doesn't exist
  mkdir(dir, 0777);

  if (ctx->initialized) {
    /* Parallel DMPlex path. fem->x/xn/xd/xdd are laid out by the CURRENT
     * run's partition (ctx->ibm_to_local_idx), which DMPlexDistribute
     * reassigns from scratch every time --np changes. A plain VecView dumps
     * data in that partition-dependent global-DOF order, so a checkpoint
     * written at one --np silently loads back with values attached to the
     * wrong mesh vertices when read at a different --np (VecLoad has no
     * notion of "vertex identity", only flat global-index position) --
     * SNES then sees a physically nonsensical state and diverges from step
     * one (observed: SNES Function norm ~1e6 instead of the expected O(1)
     * when restarting a 64-rank checkpoint with 128 ranks).
     *
     * Fix: gather into a rank-0 buffer ordered by the ORIGINAL ibm vertex
     * index (ctx->ibm_to_local_idx is keyed by that index already). That
     * ordering comes from the mesh file and never changes with --np, so the
     * checkpoint becomes restartable at any process count. */
    PetscMPIInt  rank;
    PetscInt     nTotal = ibm->n_v + ibm->n_ghosts;
    PetscInt     nDof   = dof * nTotal;
    PetscReal   *local_buf, *global_buf = NULL;
    Vec          vecs[4] = { fem->x, fem->xn, fem->xd, fem->xdd };
    const char  *tag[4]  = { "x", "xn", "xd", "xdd" };

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscCalloc1(nDof, &local_buf);
    if (rank == 0) PetscMalloc1(nDof, &global_buf);

    for (int k = 0; k < 4; ++k) {
      PetscReal *xx;
      PetscMemzero(local_buf, nDof*sizeof(PetscReal));
      VecGetArray(vecs[k], &xx);
      for (PetscInt v = 0; v < nTotal; ++v) {
        PetscInt li = ctx->ibm_to_local_idx[v];
        if (li >= 0)
          for (PetscInt d = 0; d < dof; ++d)
            local_buf[v*dof + d] = xx[li*dof + d];
      }
      VecRestoreArray(vecs[k], &xx);

      MPI_Reduce(local_buf, global_buf, nDof, MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD);

      if (rank == 0) {
        snprintf(filen, sizeof(filen), "%s/%s%1.1d_%5.5d.dat", dir, tag[k], ibi, ti);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
        PetscViewerBinaryGetDescriptor(viewer, &fd);
        PetscBinaryWrite(fd, global_buf, nDof, PETSC_REAL);
        PetscViewerDestroy(&viewer);
      }
    }

    PetscFree(local_buf);
    if (rank == 0) PetscFree(global_buf);
  } else {
    /* Serial / non-DMPlex path: fem->x etc. are COMM_SELF Vecs, already
     * np-invariant by construction -- the original direct VecView is fine
     * here. */
    snprintf(filen, sizeof(filen), "%s/x%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(fem->x, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xn%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(fem->xn, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(fem->xd, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xdd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_WRITE, &viewer);
    VecView(fem->xdd, viewer);
    PetscViewerDestroy(&viewer);
  }

  if (contact) {
    snprintf(filen, sizeof(filen), "%s/fcnt%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
    VecView(fem->Fcnt, viewer);
    
    snprintf(filen, sizeof(filen), "%s/cnt%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
    PetscViewerBinaryGetDescriptor(viewer,&fd);
    PetscBinaryWrite(fd,ibm->contact,ibm->n_v,PETSC_INT);
  }
  
  PetscViewerDestroy(&viewer);
  
  return(0);
}

PetscErrorCode InverseOut(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {
  /*
    Description:
    stores the material properties fields from the inverse problem solver.
  */

  IBMNodes     *ibm=fem->ibm; 
  char         filen[80], filen_m[80], filen_v[80];
  PetscInt     fd;

  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  // Create directory if it doesn't exist
  mkdir(dir, 0777);
  
  
  if(ConstitutiveLawNonLinear){    
    
    for(int i=0; i<n_Fung_Coeffs; i++){
      //sprintf(filen, "Fung%1.1d_%5.5d.dat", i, ti);
      snprintf(filen, sizeof(filen), "%s/Fung_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);
      PetscViewer  viewer;
      PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
      //PetscRealView(ibm->n_elmt, ibm->Fung_coeffs[i], viewer);
      PetscViewerBinaryGetDescriptor(viewer,&fd);
      PetscBinaryWrite(fd, ibm->Fung_coeffs[i], ibm->n_elmt, PETSC_REAL);
      PetscViewerDestroy(&viewer);

      if (Adam){
        snprintf(filen_m, sizeof(filen_m), "%s/Adam_m_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);
        snprintf(filen_v, sizeof(filen_v), "%s/Adam_v_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);
        PetscViewer  viewer_m;
        PetscViewer  viewer_v;
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen_m, FILE_MODE_WRITE, &viewer_m);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen_v, FILE_MODE_WRITE, &viewer_v);
        PetscViewerBinaryGetDescriptor(viewer_m,&fd);
        PetscBinaryWrite(fd, ibm->Adam_mestimate[i], ibm->n_elmt, PETSC_REAL);
        PetscViewerBinaryGetDescriptor(viewer_v,&fd);
        PetscBinaryWrite(fd, ibm->Adam_vestimate[i], ibm->n_elmt, PETSC_REAL);
        PetscViewerDestroy(&viewer_m);
        PetscViewerDestroy(&viewer_v);        
      }
    }    

    
  }  
  else{
    for (int i=0; i<n_lin_model_coeffs; i++){
      snprintf(filen, sizeof(filen), "%s/EL%2.2d_%5.5d.dat", dir, i, ti);
      PetscViewer  viewer;
      PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);      
      PetscViewerBinaryGetDescriptor(viewer,&fd);
      PetscBinaryWrite(fd, ibm->El[i], ibm->n_elmt, PETSC_REAL);
      PetscViewerDestroy(&viewer);
    }
    
  }  
  
  return(0);
}


PetscErrorCode InverseIn(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {
  /*
    Description:
    reads the material properties fields to restart the inverse problem solver.

    Usage:
    InvRstart()
  */
  
  IBMNodes     *ibm=fem->ibm;
  PetscViewer  viewer;
  char         filen[80], filen_m[80], filen_v[80];
  PetscInt     fd;

  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";
  // Create directory if it doesn't exist
  // mkdir(dir, 0777);
  
  printf("dir = %s\n", dir);
  
  if(ConstitutiveLawNonLinear){    
    
    for(int i=0; i<n_Fung_Coeffs; i++){      
      PetscViewer  viewer;
      snprintf(filen, sizeof(filen), "%s/Fung_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);

      PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
      PetscViewerBinaryGetDescriptor(viewer,&fd);
      PetscBinaryRead(fd, ibm->Fung_coeffs[i], ibm->n_elmt, PETSC_NULL, PETSC_REAL);
      
      PetscViewerDestroy(&viewer);

      if (Adam){
        PetscViewer  viewer_m;
        PetscViewer  viewer_v;
  
        snprintf(filen_m, sizeof(filen_m), "%s/Adam_m_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);
        snprintf(filen_v, sizeof(filen_v), "%s/Adam_v_%1.1d_%2.2d_%5.5d.dat", dir, ibi, i, ti);
        
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen_m, FILE_MODE_READ, &viewer_m);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen_v, FILE_MODE_READ, &viewer_v);
        
        PetscViewerBinaryGetDescriptor(viewer_m,&fd);      
        PetscBinaryRead(fd, ibm->Adam_mestimate[i], ibm->n_elmt, PETSC_NULL, PETSC_REAL);
        PetscViewerBinaryGetDescriptor(viewer_v,&fd);
        PetscBinaryRead(fd, ibm->Adam_vestimate[i], ibm->n_elmt, PETSC_NULL, PETSC_REAL);
  
        PetscViewerDestroy(&viewer_m);
        PetscViewerDestroy(&viewer_v); 
      }
    }
    
  }
  else{
    for (int i=0; i<n_lin_model_coeffs; i++){
      PetscViewer  viewer;
      snprintf(filen, sizeof(filen), "%s/EL%2.2d_%5.5d.dat", dir, i, ti);

      PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
      PetscViewerBinaryGetDescriptor(viewer,&fd);
      PetscBinaryRead(fd, ibm->El[i], ibm->n_elmt, PETSC_NULL, PETSC_REAL);
      
      PetscViewerDestroy(&viewer);
    }
      
  }

  
  return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode LocationIn(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir) {

  IBMNodes      *ibm=fem->ibm;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  PetscViewer   viewer = NULL;
  char          filen[256];
  PetscInt      fd;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Use current directory if out_dir is NULL or empty
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";

  if (ctx->initialized && legacy_restart_in) {
    /* TEMPORARY one-shot path: this checkpoint predates the ibm-order gather
     * below (written by plain COMM_WORLD VecView, partition-order for the
     * --np it was written with -- only valid to load back at that exact
     * --np). Use this once, at that same --np, to pull it into fem->x/xn/
     * xd/xdd; the caller then re-saves via the (already-fixed) LocationOut
     * below to convert it to the --np-invariant ibm-order format. */
    snprintf(filen, sizeof(filen), "%s/x%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->x, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xn%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xn, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xd, viewer);
    PetscViewerDestroy(&viewer);

    snprintf(filen, sizeof(filen), "%s/xdd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xdd, viewer);
    PetscViewerDestroy(&viewer);
  } else if (ctx->initialized) {
    /* Mirror of the ibm-order gather in LocationOut: read the flat,
     * partition-invariant buffer on rank 0, broadcast it to everyone, then
     * each rank pulls out only the ibm vertices it owns under the CURRENT
     * --np's partition (ctx->ibm_to_local_idx / ctx->ownerRank are rebuilt
     * for this run's --np by FEM_DMPlexGeomSetup before LocationIn is ever
     * called -- see the call order in main.c). This makes restart correct
     * across any change in process count between the run that wrote the
     * checkpoint and this one. */
    PetscInt    nTotal = ibm->n_v + ibm->n_ghosts;
    PetscInt    nDof   = dof * nTotal;
    PetscReal  *global_buf;
    Vec         vecs[4] = { fem->x, fem->xn, fem->xd, fem->xdd };
    const char *tag[4]  = { "x", "xn", "xd", "xdd" };

    PetscMalloc1(nDof, &global_buf);

    for (int k = 0; k < 4; ++k) {
      if (rank == 0) {
        snprintf(filen, sizeof(filen), "%s/%s%1.1d_%5.5d.dat", dir, tag[k], ibi, ti);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
        PetscViewerBinaryGetDescriptor(viewer, &fd);
        PetscBinaryRead(fd, global_buf, nDof, PETSC_NULL, PETSC_REAL);
        PetscViewerDestroy(&viewer);
      }
      MPI_Bcast(global_buf, nDof, MPIU_REAL, 0, PETSC_COMM_WORLD);

      PetscReal *xx;
      VecGetArray(vecs[k], &xx);
      for (PetscInt v = 0; v < nTotal; ++v) {
        PetscInt li = ctx->ibm_to_local_idx[v];
        if (li >= 0)
          for (PetscInt d = 0; d < dof; ++d)
            xx[li*dof + d] = global_buf[v*dof + d];
      }
      VecRestoreArray(vecs[k], &xx);
    }

    PetscFree(global_buf);
  } else {
    /* Serial / non-DMPlex path: unchanged, already np-invariant. */
    snprintf(filen, sizeof(filen), "%s/x%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->x, viewer);

    snprintf(filen, sizeof(filen), "%s/xn%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xn,viewer);

    snprintf(filen, sizeof(filen), "%s/xd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xd,viewer);

    snprintf(filen, sizeof(filen), "%s/xdd%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->xdd,viewer);
  }

  if (contact) {
    snprintf(filen, sizeof(filen), "%s/fcnt%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
    VecLoad(fem->Fcnt,viewer);
    PetscViewerDestroy(&viewer);
    
    snprintf(filen, sizeof(filen), "%s/cnt%1.1d_%5.5d.dat", dir, ibi, ti);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
    PetscViewerBinaryGetDescriptor(viewer,&fd);
    PetscBinaryRead(fd,ibm->contact,ibm->n_v, PETSC_NULL, PETSC_INT);
    PetscViewerDestroy(&viewer);
  }

  PetscInt   nv, ec;
  PetscReal  *xx;
  //---------Update the location
  VecGetArray(fem->x, &xx);
  for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
      ibm->x_bp[nv] = xx[nv*dof  ];
      ibm->y_bp[nv] = xx[nv*dof+1];
      ibm->z_bp[nv] = xx[nv*dof+2];
  }
  VecRestoreArray(fem->x, &xx);

  // PetscPrintf(PETSC_COMM_SELF, "a updating loc xn fun ran for body:%d on rank %d\n", ibi, rank);

  PetscViewerDestroy(&viewer);
  
  return(0);
}




//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode AreaNormal(IBMNodes *ibm) {
  
  struct Cmpnts  x1, x2, x3, dx21, dx31, n, cross;
  PetscInt       ec, n1e, n2e, n3e;
  
  for (ec=0; ec<ibm->n_elmt + 2*ibm->n_ghosts; ec++) {
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
    
    //current location
    x1.x = ibm->x_bp[n1e];  x1.y = ibm->y_bp[n1e];  x1.z = ibm->z_bp[n1e];
    x2.x = ibm->x_bp[n2e];  x2.y = ibm->y_bp[n2e];  x2.z = ibm->z_bp[n2e];
    x3.x = ibm->x_bp[n3e];  x3.y = ibm->y_bp[n3e];  x3.z = ibm->z_bp[n3e];
    
    dx21 = MINUS(x2, x1);  dx31 = MINUS(x3, x1);
    cross = CROSS(dx21, dx31);
    
    n = UNIT(cross);
    ibm->dA[ec] = 0.5*SIZE(cross); 
    
    ibm->nf_x[ec] = n.x;  ibm->nf_y[ec] = n.y;  ibm->nf_z[ec] = n.z;
  }
  
  return(0);
}
