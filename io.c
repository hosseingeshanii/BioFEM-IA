#include  "variables.h"
#include  <petscvec.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

extern PetscInt   dof, outghost, ConstitutiveLawNonLinear, contact, n_Fung_Coeffs, n_lin_model_coeffs;  

extern PetscReal  dt, char_length_x, char_length_y, char_length_z;

extern char in_dir[256];

extern PetscInt   ressmooth, inverse;
extern PetscInt   Adam;
extern PetscInt   par_jac;

extern PetscInt   muscle_activation;
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

  VecCreateSeq(PETSC_COMM_SELF,dof*(ibm->n_v + ibm->n_ghosts), &(fem->Res));  
  VecDuplicate(fem->Res, &(fem->x));
  VecDuplicate(fem->Res, &(fem->xn));
  VecDuplicate(fem->Res, &(fem->xnm1));
  VecDuplicate(fem->Res, &(fem->xd));
  VecDuplicate(fem->Res, &(fem->dx));
  VecDuplicate(fem->Res, &(fem->xdd));
  VecDuplicate(fem->Res, &(fem->y));
  VecDuplicate(fem->Res, &(fem->yn));
  VecDuplicate(fem->Res, &(fem->Fint));
  VecDuplicate(fem->Res, &(fem->Fext));
  VecDuplicate(fem->Res, &(fem->Fdyn));
  VecDuplicate(fem->Res, &(fem->disp));
  VecDuplicate(fem->Res, &(fem->FJ));
  VecDuplicate(fem->Res, &(fem->Mass));  
  VecDuplicate(fem->Res, &(fem->Dissip));
  VecDuplicate(fem->Res, &(fem->Fcnt));
  

  //Initialize
  VecSet(fem->Res, 0.0);  VecSet(fem->x, 0.0);  VecSet(fem->xn, 0.0);  VecSet(fem->xnm1, 0.0);
  VecSet(fem->xd, 0.0);  VecSet(fem->xdd, 0.0);  VecSet(fem->y, 0.0);  VecSet(fem->yn, 0.0);
  VecSet(fem->Fint, 0.0);  VecSet(fem->Fext, 0.0);  VecSet(fem->Fdyn, 0.0);  VecSet(fem->FJ, 0.0); 
  VecSet(fem->disp, 0.0);  VecSet(fem->Fcnt, 0.0);
  VecSet(fem->Mass, 0.0);  VecSet(fem->Dissip, 0.0);

  if (ressmooth){
    VecDuplicate(fem->Res, &(fem->Res_smth));  
    VecSet(fem->Res_smth, 0.0);
  }
  
  
  // if (muscle_activation){
  //   VecDuplicate(fem->Res, &(fem->x_intmd));
  //   VecSet(fem->x_intmd, 0.0);
  // }
  return (0);
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

  PetscPrintf(PETSC_COMM_SELF, "Number of nodes of list (body:%d) %d \n", ibm->ibi, ibm->n_v);
  
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
  PetscPrintf(PETSC_COMM_SELF, "Number of element of list(body:%d) %d \n", ibi, ibm->n_elmt);

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
PetscErrorCode Output(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {

  PetscInt  n_cells=3, i;
  IBMNodes  *ibm=fem->ibm;  
  FILE      *f;
  char      filepath[80];


  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
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
  
  
  //compute displacement
  PetscReal  *dd,*FF; 
  PetscInt   nv;
  VecGetArray(fem->disp, &dd);
  for (nv=0; nv<ibm->n_v; nv++) {
    dd[nv*dof] = ibm->x_bp[nv]-ibm->x_bp0[nv];
    dd[nv*dof+1] = ibm->y_bp[nv]-ibm->y_bp0[nv];
    dd[nv*dof+2] = ibm->z_bp[nv]-ibm->z_bp0[nv];
  }
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS disp float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", dd[i*dof], dd[i*dof+1], dd[i*dof+2]);
  }
  VecRestoreArray(fem->disp, &dd);
  
  VecGetArray(fem->Fint, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fint float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
    // if (i == 100){
    //   PetscPrintf(PETSC_COMM_WORLD, "Fint at node 100 what: %f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
    // }
  }
  VecRestoreArray(fem->Fint, &FF);
  
  VecGetArray(fem->Fext, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fext float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fext, &FF);

  VecGetArray(fem->Fdyn, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fdyn float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fdyn, &FF);

  // VecGetArray(fem->Res, &FF);
  // PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Res float\n");
  // for (i=0; i<ibm->n_v; i++){
  //   PetscFPrintf(PETSC_COMM_WORLD, f, "%.10f %.10f %.10f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  // }
  // VecRestoreArray(fem->Res, &FF);

  VecGetArray(fem->Fcnt, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fcnt float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fcnt, &FF);

  VecGetArray(fem->xd, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS u float\n");
  for (i=0; i<ibm->n_v; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->xd, &FF);

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
  
  // if(ConstitutiveLawNonLinear){
  //   PetscFPrintf(PETSC_COMM_WORLD, f,  "VECTORS nfib float\n");
  //   for (i=0; i<ibm->n_elmt; i++) {
  //     PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n",ibm->n_fib[i].x, ibm->n_fib[i].y, ibm->n_fib[i].z);
  //   }
  // }

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
PetscErrorCode OutputGhost(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {     

  PetscInt   n_cells=3, i;
  IBMNodes   *ibm=fem->ibm;
  PetscReal  x, y, z;
  PetscInt   ec, be, n1e, n2e, n3e, n_ghosts=0;

  n_ghosts = ibm->n_ghosts;

  FILE  *f;
  char  filen[80];

  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
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

  PetscReal  *dd,*FF;
  PetscInt   nv; 

  VecGetArray(fem->disp, &dd);
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
  VecRestoreArray(fem->disp, &dd);
  
  VecGetArray(fem->Fint, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fint float\n");
  for (i=0; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fint, &FF);

  VecGetArray(fem->Fext, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fext float\n");
  for (i=0; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fext, &FF);

  VecGetArray(fem->Fdyn, &FF);
  PetscFPrintf(PETSC_COMM_WORLD, f, "VECTORS Fdyn float\n");
  for (i=0; i<ibm->n_v+n_ghosts; i++){
    PetscFPrintf(PETSC_COMM_WORLD, f, "%f %f %f\n", FF[i*dof], FF[i*dof+1], FF[i*dof+2]);
  }
  VecRestoreArray(fem->Fdyn, &FF);

  fclose(f);

  return(0);
}

//-----------------------------------------------------------------------------------------------------------------------------------
PetscErrorCode LocationOut(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {
  
  IBMNodes     *ibm=fem->ibm;
  PetscViewer  viewer;
  char         filen[256];
  PetscInt     fd;

  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
  // Create directory if it doesn't exist
  mkdir(dir, 0777);

  snprintf(filen, sizeof(filen), "%s/x%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
  VecView(fem->x, viewer);
  PetscViewerDestroy(&viewer);
  
  snprintf(filen, sizeof(filen), "%s/xn%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
  VecView(fem->xn, viewer);
  PetscViewerDestroy(&viewer);

  snprintf(filen, sizeof(filen), "%s/xd%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
  VecView(fem->xd, viewer);
  PetscViewerDestroy(&viewer);
  
  snprintf(filen, sizeof(filen), "%s/xdd%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer);
  VecView(fem->xdd, viewer);
  PetscViewerDestroy(&viewer);

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

PetscErrorCode InverseOut(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {
  /*
    Description:
    stores the material properties fields from the inverse problem solver.
  */

  IBMNodes     *ibm=fem->ibm; 
  char         filen[80], filen_m[80], filen_v[80];
  PetscInt     fd;

  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
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


PetscErrorCode InverseIn(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {
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

  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
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
PetscErrorCode LocationIn(FE *fem, PetscInt ti, PetscInt ibi, const char *subdir) {

  IBMNodes     *ibm=fem->ibm;
  PetscViewer  viewer;
  char         filen[256];
  PetscInt     fd;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Use current directory if subdir is NULL or empty
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";

  snprintf(filen, sizeof(filen), "%s/x%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
  // PetscPrintf(PETSC_COMM_SELF, "a PetscViewerBinaryOpen fun ran for body:%d on rank %d\n", ibi, rank);

  VecLoad(fem->x, viewer);
  // PetscPrintf(PETSC_COMM_SELF, "a VecLoad fun ran for body:%d on rank %d\n", ibi, rank);

  snprintf(filen, sizeof(filen), "%s/xn%1.1d_%5.5d.dat", dir, ibi, ti);
  // PetscPrintf(PETSC_COMM_SELF, "b PetscViewerBinaryOpen fun ran for body:%d on rank %d\n", ibi, rank);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
  // PetscPrintf(PETSC_COMM_SELF, "a PetscViewerBinaryOpen xn fun ran for body:%d on rank %d\n", ibi, rank);
  VecLoad(fem->xn,viewer);
  // PetscPrintf(PETSC_COMM_SELF, "a VecLoad xn fun ran for body:%d on rank %d\n", ibi, rank);

  snprintf(filen, sizeof(filen), "%s/xd%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
  VecLoad(fem->xd,viewer);

  snprintf(filen, sizeof(filen), "%s/xdd%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_READ, &viewer);
  VecLoad(fem->xdd,viewer);

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
