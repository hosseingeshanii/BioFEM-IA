#include  "variables.h"
#include <stdbool.h>  
#include <unistd.h> 


extern PetscInt   dof, bending, ConstitutiveLawNonLinear, curvature, manufactured, n_Fung_Coeffs, n_lin_model_coeffs, tisteps, twod, uniform_fiber_dir;
extern PetscReal  decay_factor, learning_rate;
extern PetscInt   nbody, tistart, epoch_start, n_epochs, epoch_output, uniform_fung, epoch_update_jacobian, Adam;
extern char subdir[];

extern PetscInt   ressmooth, fibersmooth;
extern PetscInt   res_smooth_itrs;
extern PetscReal  initial_elas, initial_poisson;
extern PetscInt   constrained_el;
extern PetscInt   par_jac, progress_bar_show;




PetscErrorCode reset_Fung_epsilon(FE *fem){
  /*
  
  Description:
    Reset Fung's model epsilon array to all zero

  Usage:
    update_Fung_epsilon
  */
 /*_________________________________________*/
  IBMNodes  *ibm=fem->ibm;  
  for (PetscInt i=0; i<n_Fung_Coeffs; i++){
      for (PetscInt ec=0; ec<ibm->n_elmt; ec++){
          ibm->Fung_epsilons[i][ec] = 0.0;    
      }
  }
  return(0);
}


PetscErrorCode update_Fung_epsilon(PetscInt elmt_indx,  PetscInt coeff_indx, PetscInt dir, PetscReal eps_mag, FE *fem){
    /*
    Description:
      This function sets the epsilon value of an element in the Fung_epsilon array

    Usage:
      Will be used in computing the residual's Jacobian 
    */
    
    IBMNodes  *ibm=fem->ibm;  
    reset_Fung_epsilon(fem);
    ibm->Fung_epsilons[coeff_indx][elmt_indx] = dir*eps_mag;
    return(0);
}


PetscErrorCode reset_Lin_E_epsilon(FE *fem){
  IBMNodes  *ibm=fem->ibm;      
  for (PetscInt i=0; i<2; i++){
    for (PetscInt ec=0; ec<ibm->n_elmt; ec++){
        ibm->E_epsilon[i][ec] = 0.0;    
    }          
  }
    return(0);
}


PetscErrorCode update_Lin_E_epsilon(PetscInt elmt_indx,  PetscInt coeff_indx, PetscInt dir, PetscReal eps_mag, FE *fem){
    IBMNodes  *ibm=fem->ibm;  
    reset_Lin_E_epsilon(fem);
    ibm->E_epsilon[coeff_indx][elmt_indx] = dir*eps_mag;
    return(0);
}


PetscErrorCode ResidualCalc(int ibi, FE* fem){

  /*
  Description:
    Compute residual based on last given nodal coordinates, external forces, and BCs. 

  Usage:
    updateFung, update_elasticity

  Note:
    Check the "GlobalGhost(ibm);" call is required or not (based on BCs)
  */

  // We have to take x_n+1 and x_n from last saved iteration to calculate the strains  
  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec;
  PetscReal  *xx;
  //---------Update the location
  VecGetArray(fem->x, &xx);
  for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
      ibm->x_bp[nv] = xx[nv*dof  ];
      ibm->y_bp[nv] = xx[nv*dof+1];
      ibm->z_bp[nv] = xx[nv*dof+2];
      if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}  //2d case
    // /* } */
  }
  VecRestoreArray(fem->x, &xx);

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {     
      PatchLoc(ibm); 
      GhostLoc(fem);
    } else if (curvature==6) {
      GlobalGhost(ibm);
    }
  }
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0);  VecSet(fem->Fdyn, 0.0);
  VecSet(fem->Res, 0.0);  VecSet(fem->FJ, 0.0);  
  for (nv=0; nv<ibm->n_v; nv++)  fem->IE[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;

    
  FInternal(fem);
  if(tisteps>1) {FDynamic(fem);}
  FExternal(fem);

  VecWAXPY(fem->Res,-1., fem->Fext, fem->Fint);
  VecAXPY(fem->Res,1., fem->Fdyn);
  return(0);

}


PetscErrorCode grad_R_wrt_E_field(PetscInt ibi, PetscInt dof_index, FE* fem, PetscReal epsilon){
  /*
    Compute Jacobian dR_k^i/dc_m^j 

    dR_dE[nv][ec][k]

      i : node index
      j : element index
      k : dof index
      m : 0 for elasticity and 1 for poisson 

    Usage:
      should be called before update_elasticity function
  */

  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec;
  PetscScalar *RRes;
  PetscReal **residual_array_plus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_plus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_plus[i]);  
  }
  PetscReal **residual_array_minus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_minus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_minus[i]);  
  }

  for (int m = 0; m < 2; m++){

  for(ec=0; ec<ibm->n_elmt; ec++){        
    
    // double *residual_array_plus = (double *)malloc(ibm->n_v * sizeof(double));
    update_Lin_E_epsilon(ec, m, 1.0, epsilon, fem);
    ResidualCalc(ibi, fem);

    VecGetArray(fem[ibi].Res, &RRes);
    for (int i = 0; i < ibm->n_v; ++i) {
      for (int k = 0; k < dof; k++){
        residual_array_plus[i][k] = RRes[i*dof+k];
      }        
    }
    VecRestoreArray(fem[ibi].Res, &RRes); 

    // double *residual_array_minus = (double *)malloc(ibm->n_v * sizeof(double));
    update_Lin_E_epsilon(ec, m, -1.0, epsilon, fem);
    ResidualCalc(ibi, fem);

    VecGetArray(fem[ibi].Res, &RRes);
    for (int i = 0; i < ibm->n_v; ++i) {
      for (int k = 0; k < dof; k++){
        residual_array_minus[i][k] = RRes[i*dof+k];
      }
    }
    VecRestoreArray(fem[ibi].Res, &RRes); 

    for(nv=0; nv<ibm->n_v; nv++){              
      for (int k = 0; k < dof; k++){          
        fem->dR_dE[nv][ec][k][m] = (residual_array_plus[nv][k] - residual_array_minus[nv][k])/(2*epsilon);          
        
      }      
    }      
    
    // for(nv=0; nv<ibm->n_v; nv++){    
      
    //   fem->dR_dE[nv][ec] = (residual_array_plus[nv] - residual_array_minus[nv])/(2*epsilon);
    
    // }
    
  
  }        
  }
  free(residual_array_plus);  
  free(residual_array_minus);      

  return(0);
}

PetscErrorCode SeqJacobianUpdate(PetscInt ibi, FE* fem){
  /*
    This function updates the Jacobian Sequential matrix (stored on rank 0) 
    from the last update of distributed fem->Jacobian dense matrix.
  */

  IBMNodes   *ibm=fem->ibm;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


}
PetscErrorCode Jacobian(PetscInt ibi, FE* fem, PetscReal epsilon){
  /*
    Parralel computation of Jacobian Mat    
    Jacobian matrix dimension: [m * n_elements] * [n_nodes * dof]
  */

  IBMNodes   *ibm=fem->ibm;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


  PetscErrorCode ierr;
  PetscInt row_start, row_end;
  PetscInt percent_step = 10; // update every 10%

  PetscScalar *RRes;
  PetscReal *residual_array_plus, *residual_array_minus;

  PetscMalloc((ibm->n_v * dof) * sizeof(PetscReal), &residual_array_plus); 
  PetscMalloc((ibm->n_v * dof) * sizeof(PetscReal), &residual_array_minus); 

  MatGetOwnershipRange(fem->Jacobian, &row_start, &row_end);
  PetscInt total_rows = row_end - row_start;

  PetscPrintf(PETSC_COMM_SELF, "body:%d on rank %d, row_start=%d, row_end = %d\n", ibi, rank, row_start, row_end);

  for (PetscInt i = row_start; i < row_end; i++) {
    PetscInt ei = i % ibm->n_elmt;  // element index
    PetscInt mi = i / ibm->n_elmt;  // variable index
      

    update_Fung_epsilon(ei, mi, 1.0, epsilon, fem);
    ResidualCalc(ibi, fem);    
    VecGetArray(fem->Res, &RRes);
    for (PetscInt c_i=0; c_i < (ibm->n_v * dof); c_i++){
      residual_array_plus[c_i] = RRes[c_i];
    }    
    VecRestoreArray(fem->Res, &RRes); 
    

    update_Fung_epsilon(ei, mi, -1.0, epsilon, fem);
    ResidualCalc(ibi, fem);
    VecGetArray(fem->Res, &RRes);    
    for (PetscInt c_i=0; c_i < (ibm->n_v * dof); c_i++){
      residual_array_minus[c_i] = RRes[c_i];
    }    
    VecRestoreArray(fem->Res, &RRes); 
    
    for (PetscInt c_i=0; c_i < (ibm->n_v * dof); c_i++){
      PetscReal val = (residual_array_plus[c_i] - residual_array_minus[c_i])/(2*epsilon);
      if (PetscIsNanReal(val)) {
        PetscPrintf(PETSC_COMM_SELF, "[Rank %d] NaN found at (%d, %d)\n", rank, i, c_i);
      }
      MatSetValue(fem->Jacobian, i, c_i, val, INSERT_VALUES);
    }
    PetscInt current = i - row_start;
    
    PetscReal progress = 100.0 * ((PetscReal)current / (PetscReal)total_rows);
    if ((!rank)){
      // printf("\r[%.1f%%] Completed row %d of %d from rank: %d\n", progress, i, row_end - 1, rank);
      if (progress_bar_show){
        print_progress_bar(progress);
      }
      
    }
    

  }
  MatAssemblyBegin(fem->Jacobian, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(fem->Jacobian, MAT_FINAL_ASSEMBLY);
  
  return(0);
}

void print_progress_bar(PetscReal progress) {
  const int width = 50; // width of the progress bar
  int pos = (int)(progress * width / 100.0);
  
  printf("\r[");
  for (int i = 0; i < width; ++i) {
      if (i < pos) printf("=");
      else if (i == pos) printf(">");
      else printf(" ");
  }
  printf("] %.1f%%", progress);
  fflush(stdout);
}

PetscErrorCode JacMatToArr(FE* fem){
  IBMNodes   *ibm=fem->ibm;
  PetscInt rstart, rend;    

  PetscInt n_coeffs;
  if (ConstitutiveLawNonLinear){
    n_coeffs = n_Fung_Coeffs;
  }
  else{
    n_coeffs = n_lin_model_coeffs;
  }

  // MatGetOwnershipRange(fem->Jacobian, &rstart, &rend);
  
  for (PetscInt row_i = 0; row_i < ibm->n_elmt * n_coeffs; row_i++){    
    PetscScalar *vals;    
    
    PetscInt ei = row_i % ibm->n_elmt;  // element index
    PetscInt mi = row_i / ibm->n_elmt;  // variable index

    MatGetRow(fem->J_Seq, row_i, NULL, NULL, &vals);

    for (PetscInt col_i = 0; col_i < ibm->n_v * dof; col_i++){
      
      PetscInt ni = col_i / dof;  // node index
      PetscInt dof_i = col_i % dof;  // dof index

      fem->Jac_Fung[ni][ei][dof_i][mi] = vals[col_i];
    }
    
    MatRestoreRow(fem->J_Seq, row_i, NULL, NULL, &vals);
  }
  

  return(0);
}


PetscErrorCode FungJacobian(PetscInt ibi, FE* fem, PetscReal epsilon){
  /*
    Compute Jacobian dR_k^i/dc_m^j // Jac_Fung[i][j][k][m]
      i : node index
      j : element index
      k : dof index
      m : Fung's model coeff index (0-6) (c, A_i) (membrane only)

    Usage:
      updateFung
  */
  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec;
  PetscScalar *RRes;

  PetscReal **residual_array_plus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_plus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_plus[i]);  
  }
  PetscReal **residual_array_minus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_minus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_minus[i]);  
  }
  // PetscPrintf(PETSC_COMM_WORLD, "a r malloc\n");

  for (int m = 0; m < n_Fung_Coeffs-1; m++){
    for(ec=0; ec<ibm->n_elmt; ec++){        
      
      update_Fung_epsilon(ec, m, 1.0, epsilon, fem);
      ResidualCalc(ibi, fem);
      VecGetArray(fem->Res, &RRes);
      for (int i = 0; i < ibm->n_v; ++i) {
        for (int k = 0; k < dof; k++){
          residual_array_plus[i][k] = RRes[i*dof+k];
        }        
      }
      VecRestoreArray(fem->Res, &RRes); 

      update_Fung_epsilon(ec, m, -1.0, epsilon, fem);    
      ResidualCalc(ibi, fem);
      VecGetArray(fem->Res, &RRes);
      for (int i = 0; i < ibm->n_v; ++i) {
        for (int k = 0; k < dof; k++){
          residual_array_minus[i][k] = RRes[i*dof+k];
        }
      }
      VecRestoreArray(fem->Res, &RRes); 
      
      for(nv=0; nv<ibm->n_v; nv++){              
        for (int k = 0; k < dof; k++){          
          fem->Jac_Fung[nv][ec][k][m] = (residual_array_plus[nv][k] - residual_array_minus[nv][k])/(2*epsilon);          
          if (isnan(fem->Jac_Fung[nv][ec][k][m])) {
            // printf("NaN detected at ibi=%d, node=%d, dof=%d\n", ibi, i, j);
            // exit(1); // Stop execution for debugging
          }  
        }      
      }      
    }    
  }


  reset_Fung_epsilon(fem);

  if (uniform_fiber_dir){
    for (int m = n_Fung_Coeffs-1; m < n_Fung_Coeffs; m++){
      for(ec=0; ec<ibm->n_elmt; ec++){              
        ibm->Fung_epsilons[m][ec] = epsilon;
      }

      ResidualCalc(ibi, fem);      
      VecGetArray(fem->Res, &RRes);
      for (int i = 0; i < ibm->n_v; ++i) {
        for (int k = 0; k < dof; k++){
          residual_array_plus[i][k] = RRes[i*dof+k];
        }        
      }
      VecRestoreArray(fem->Res, &RRes); 
      

      for(ec=0; ec<ibm->n_elmt; ec++){            
        ibm->Fung_epsilons[m][ec] = -epsilon;
      }

      ResidualCalc(ibi, fem);
      VecGetArray(fem->Res, &RRes);
      for (int i = 0; i < ibm->n_v; ++i) {
        for (int k = 0; k < dof; k++){
          residual_array_minus[i][k] = RRes[i*dof+k];
        }
      }
      VecRestoreArray(fem->Res, &RRes); 

      for(ec=0; ec<ibm->n_elmt; ec++){
        for(nv=0; nv<ibm->n_v; nv++){              
          for (int k = 0; k < dof; k++){          
            fem->Jac_Fung[nv][ec][k][m] = (residual_array_plus[nv][k] - residual_array_minus[nv][k])/(2*epsilon);
                  
          }      
        }    
      }  
    }
  }
  else{
    for (int m = n_Fung_Coeffs-1; m < n_Fung_Coeffs; m++){
      for(ec=0; ec<ibm->n_elmt; ec++){       
               
        update_Fung_epsilon(ec, m, 1.0, epsilon, fem);

        ResidualCalc(ibi, fem);
        
        VecGetArray(fem->Res, &RRes);
        for (int i = 0; i < ibm->n_v; ++i) {
          for (int k = 0; k < dof; k++){
            residual_array_plus[i][k] = RRes[i*dof+k];
          }        
        }
        VecRestoreArray(fem->Res, &RRes); 

        update_Fung_epsilon(ec, m, -1.0, epsilon, fem);    
        ResidualCalc(ibi, fem);
        
        VecGetArray(fem->Res, &RRes);
        for (int i = 0; i < ibm->n_v; ++i) {
          for (int k = 0; k < dof; k++){
            residual_array_minus[i][k] = RRes[i*dof+k];
          }
        }
        VecRestoreArray(fem->Res, &RRes); 
        
        for(nv=0; nv<ibm->n_v; nv++){              
          for (int k = 0; k < dof; k++){          
            fem->Jac_Fung[nv][ec][k][m] = (residual_array_plus[nv][k] - residual_array_minus[nv][k])/(2*epsilon);          
          }      
        }      
      }    
    }
  }
  
  reset_Fung_epsilon(fem);
      
  free(residual_array_plus);
  free(residual_array_minus);
    
  return(0);
}


PetscErrorCode FungUniJacobian(PetscInt ibi, FE* fem, PetscReal epsilon){

  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec;
  PetscScalar *RRes;

  PetscReal **residual_array_plus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_plus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_plus[i]);  
  }

  PetscReal **residual_array_minus;
  PetscMalloc(ibm->n_v * sizeof(PetscReal*), &residual_array_minus); 
  for (int i = 0; i < ibm->n_v; i++) {
      PetscMalloc(dof * sizeof(PetscReal), &residual_array_minus[i]);  
  }
    
  reset_Fung_epsilon(fem);


  for (int m = 0; m < n_Fung_Coeffs; m++){

    for(ec=0; ec<ibm->n_elmt; ec++){              
      ibm->Fung_epsilons[m][ec] = epsilon;
    }

    ResidualCalc(ibi, fem);      
    VecGetArray(fem->Res, &RRes);
    for (int i = 0; i < ibm->n_v; i++) {
      for (int k = 0; k < dof; k++){
        residual_array_plus[i][k] = RRes[i*dof+k];
      }        
    }
    VecRestoreArray(fem->Res, &RRes); 

    reset_Fung_epsilon(fem);

    for(ec=0; ec<ibm->n_elmt; ec++){            
      ibm->Fung_epsilons[m][ec] = -epsilon;
    }

    ResidualCalc(ibi, fem);    
    VecGetArray(fem->Res, &RRes);
    for (int i = 0; i < ibm->n_v; i++) {
      for (int k = 0; k < dof; k++){
        residual_array_minus[i][k] = RRes[i*dof+k];
      }
    }
    VecRestoreArray(fem->Res, &RRes); 
    reset_Fung_epsilon(fem);

    for(ec=0; ec<ibm->n_elmt; ec++){
      for(nv=0; nv<ibm->n_v; nv++){              
        for (int k = 0; k < dof; k++){          
          fem->Jac_Fung[nv][ec][k][m] = (residual_array_plus[nv][k] - residual_array_minus[nv][k])/(2*epsilon);
          if (residual_array_plus[nv][k]!=0.0){
              //printf("residual_array_plus[nv][k] %f, residual_array_min[nv][k] %f\n", residual_array_plus[nv][k], residual_array_minus[nv][k]);  
            if (fem->Jac_Fung[nv][ec][k][m] > 0.0){
              //printf("fem->Jac_Fung %f\n", fem->Jac_Fung[nv][ec][k][m]);  
            }
          }
          
          //printf("fem->Jac_Fung %f\n", fem->Jac_Fung[nv][ec][k][m]);
        }      
      }    
    }  
  }

  reset_Fung_epsilon(fem);
      
  free(residual_array_plus);
  free(residual_array_minus);

  return(0);

}


PetscErrorCode updateFung(PetscReal* learning_rate, PetscInt epoch, FE *fem){
  
  PetscInt   nv, ec, k, m;
  IBMNodes *ibm=fem->ibm;
  PetscScalar *RRes;
  PetscReal initial_learning_rate = learning_rate[7];
  PetscReal min_learning_rate = 1e-4;
  // PetscReal decay_rate = 0.001;

  VecGetArray(fem->Res, &RRes);

  for (ec=0; ec<ibm->n_elmt; ec++){
    for (nv=0; nv<ibm->n_v; nv++){
      for (k=0; k<dof; k++){
        for (m=0; m<n_Fung_Coeffs-1; m++){                  
          double lr = learning_rate[m];
          if (m == 7) {
              lr = fmax(min_learning_rate, initial_learning_rate * exp(-decay_factor * epoch));
              lr = initial_learning_rate * exp(-decay_factor * epoch);
          }
          ibm->Fung_coeffs[m][ec] = ibm->Fung_coeffs[m][ec] - 2*lr*RRes[nv*dof+k]*(fem->Jac_Fung[nv][ec][k][m]);
          if (m == 7 && ibm->Fung_coeffs[m][ec] > 1.0) {
              ibm->Fung_coeffs[m][ec] = 1.0;
          }
          /*
          if (m == 7 && ibm->Fung_coeffs[m][ec] < 0.0) {
              ibm->Fung_coeffs[m][ec] = 0.0;
          }
          */
        }                
      }
    }
  }  
  VecRestoreArray(fem->Res, &RRes);

  return(0);
}


PetscErrorCode updateFungAdam(PetscReal *learning_rate, FE *fem, PetscInt time_step) {
  PetscInt ec, k, m, nv;
  IBMNodes *ibm = fem->ibm;
  PetscScalar *RRes;
  PetscReal beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  // PetscReal decay_rate = 0.1;

  // Temporary variables for Adam optimizer estimates
  PetscReal m_estimate, v_estimate, m_hat, v_hat, g;

  VecGetArray(fem->Res, &RRes);

  for (ec = 0; ec < ibm->n_elmt; ec++) {
      for (m = 0; m < n_Fung_Coeffs ; m++) {

        if (time_step == 0){
          ibm->Adam_mestimate[m][ec] = 0.0;
          ibm->Adam_vestimate[m][ec] = 0.0;
        }
        m_estimate = ibm->Adam_mestimate[m][ec];
        v_estimate = ibm->Adam_vestimate[m][ec];
        g = 0.0;

          // Accumulate gradients over nv and k
          for (nv = 0; nv < ibm->n_v; nv++) {
              for (k = 0; k < dof; k++) {
                  g += RRes[nv * dof + k] * fem->Jac_Fung[nv][ec][k][m];
              }
          }

          if (g==0){
	    //printf("g is zero!\n");
            continue;
          }
          //PetscPrintf(PETSC_COMM_WORLD, "ec=%d, m=%d, g=%f\n", ec, m, g);

          // Compute m_estimate and v_estimate using accumulated gradient
          m_estimate = beta1 * m_estimate + (1 - beta1) * g;
          v_estimate = beta2 * v_estimate + (1 - beta2) * g * g;

          ibm->Adam_mestimate[m][ec] = m_estimate;
          ibm->Adam_vestimate[m][ec] = v_estimate;
          
          //PetscPrintf(PETSC_COMM_WORLD, "m_estimate (before correction) = %f, v_estimate (before correction) = %f\n", m_estimate, v_estimate);
          // Correct bias for m and v estimates
          m_hat = m_estimate / (1 - pow(beta1, time_step+1));
          v_hat = v_estimate / (1 - pow(beta2, time_step+1));
          //PetscPrintf(PETSC_COMM_WORLD, "m_hat = %f, v_hat = %f\n", m_hat, v_hat);
          // Update the coefficient
          double lr = learning_rate[m]/(1 + decay_factor * time_step);
          ibm->Fung_coeffs[m][ec] -= lr * m_hat / (sqrt(v_hat) + epsilon);
          double pi = 3.141592;
          if (m == 7) {
            if (ibm->Fung_coeffs[m][ec] > 0.0){
              ibm->Fung_coeffs[m][ec] = 0.0;
            }
            if (ibm->Fung_coeffs[m][ec] < -pi/2){
              ibm->Fung_coeffs[m][ec] = -pi/2;
            }                        
          }
                    
          /*
          if (fabs(grad[i]) > grad_clip_threshold) {
              grad[i] = copysign(grad_clip_threshold, grad[i]);
          }
          */
          
          
      }
            
  }

  VecRestoreArray(fem->Res, &RRes);
  return (0);
}


PetscErrorCode update_elasticity(PetscReal *learning_rate, FE *fem, PetscInt time_step){
  
  PetscInt   nv, ec, k, m;
  IBMNodes *ibm=fem->ibm;
  PetscScalar *RRes;


  VecGetArray(fem->Res, &RRes);

  if (Adam){
    PetscReal m_estimate, v_estimate, m_hat, v_hat, g;
    PetscReal beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    for (ec = 0; ec < ibm->n_elmt; ec++){
      for (m = 0; m < 2 ; m++) {
      
        if (time_step == 0){
          ibm->Adam_mestimate[m][ec] = 0.0;
          ibm->Adam_vestimate[m][ec] = 0.0;
        }
      m_estimate = ibm->Adam_mestimate[m][ec];
      v_estimate = ibm->Adam_vestimate[m][ec];
      g = 0.0;
      
      for (nv = 0; nv < ibm->n_v; nv++) {
        for (k = 0; k < dof; k++) {
          g += RRes[nv*dof+k]*(fem->dR_dE[nv][ec][k][m]);
        }
      }

      if (g==0){
        continue;
      }

      // Compute m_estimate and v_estimate using accumulated gradient
      m_estimate = beta1 * m_estimate + (1 - beta1) * g;
      v_estimate = beta2 * v_estimate + (1 - beta2) * g * g;

      ibm->Adam_mestimate[m][ec] = m_estimate;
      ibm->Adam_vestimate[m][ec] = v_estimate;

      // Correct bias for m and v estimates
      m_hat = m_estimate / (1 - pow(beta1, time_step+1));
      v_hat = v_estimate / (1 - pow(beta2, time_step+1));

      // Update the coefficient
      double lr = learning_rate[m]/(1 + decay_factor * time_step);
      ibm->El[m][ec] -= lr * m_hat / (sqrt(v_hat) + epsilon);

      if (ibm->El[m][ec] < 0.0){
        if (constrained_el){
           ibm->El[m][ec] = 0.0;
        }      
      }
      if (ibm->El[1][ec] > 0.99){
        ibm->El[1][ec] = 0.99;
      }
    }
  }
  }

  else{

    for (ec=0; ec<ibm->n_elmt; ec++){
      for (nv=0; nv<ibm->n_v; nv++){
        for (k=0; k<dof; k++){
          ibm->El[0][ec] = ibm->El[0][ec] - 2*learning_rate[0]*RRes[nv*dof+k]*(fem->dR_dE[nv][ec][k][0]);

          if (ibm->El[0][ec] < 0.0){
            // ibm->El[ec] = 0.0;
          }
        }      
      }
    }  

  }
  
  VecRestoreArray(fem->Res, &RRes);
  
  return(0);
}


PetscErrorCode InvSolver(FE *fem){  
  /*
  Description:
    Main solver for inverse problem solution finding. 
    Works for linear and non-linear materials (Fung model).
    "inverse" flag should be set as 1 to run.
  
  Note:
    Always check bcs.c, contact.c before running
  */

  PetscInt ibi, epoch;
  PetscReal best_loss = INFINITY;  // Track the best (lowest) loss
  PetscReal improvement_threshold = 1e-4; // Minimum change in loss to consider as improvement
  PetscInt epochs_no_improvement = 0; // Counter for epochs without improvement
  PetscInt patience = 5;             // Number of epochs with no improvement to wait before decaying
  PetscInt dof_index = 2;  

  PetscReal* lr_fung; 
  PetscReal* lr_lin; 

  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  /// Opening Inverese Problem results files
  FILE **fp = malloc(nbody * sizeof(FILE *));  // Allocate memory for file pointers

  char filepath[256];  
  const char *dir = (subdir && strlen(subdir) > 0) ? subdir : ".";
  
  for (ibi=0; ibi<nbody; ibi++){

    snprintf(filepath, sizeof(filepath), "%s/INV_R%2.2d.txt", dir, ibi);
    PetscFOpen(PETSC_COMM_WORLD, filepath, "a", &fp[ibi]);  
    
  }
  //________________________________________________________________________//


  /// learning rate files reading
  if (ConstitutiveLawNonLinear){
    lr_fung = (PetscReal*)malloc(n_Fung_Coeffs * sizeof(PetscReal)); 

    FILE* file = fopen("lr_fung.txt", "r");      

    for (int i = 0; i < n_Fung_Coeffs; i++) {
        if (fscanf(file, "%lf", &lr_fung[i]) != 1) {
            printf("Error reading value\n");
            free(lr_fung);
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    PetscPrintf(PETSC_COMM_WORLD, "Fung learning rate file successfully has being read\n");
  }
  else{
    lr_lin = (PetscReal*)malloc(2 * sizeof(PetscReal)); 

    FILE* file = fopen("lr_lin.txt", "r");      

    for (int i = 0; i < 2; i++) {
        if (fscanf(file, "%lf", &lr_lin[i]) != 1) {
            printf("Error reading value\n");
            free(lr_lin);
            fclose(file);
            return -1;
        }
    }
    fclose(file);
    PetscPrintf(PETSC_COMM_WORLD, "Linear model learning rate file successfully has being read\n");
  }
  //________________________________________________________________________//  



  /// Problem Initialization/Reading last saved epoch for inverse rstart, 
  //  Initial ResCalc and ResSmoother call, 
  if (1){
    for (ibi=0; ibi<nbody; ibi++){    

      IBMNodes   *ibm=fem[ibi].ibm;    

      if (epoch_start != 0){
        InverseIn(&fem[ibi], tistart+epoch_start, ibi, subdir);
      }
      else{
        if (ConstitutiveLawNonLinear){
          PetscReal* init_fung_coeffs; 

          init_fung_coeffs = (PetscReal*)malloc(n_Fung_Coeffs * sizeof(PetscReal)); 

          FILE* file = fopen("init_fung.txt", "r");      

          for (int i = 0; i < n_Fung_Coeffs; i++) {
              if (fscanf(file, "%lf", &init_fung_coeffs[i]) != 1) {
                  printf("Error reading value init_fung\n");
                  free(init_fung_coeffs);
                  fclose(file);
                  return -1;
              }
          }

          fclose(file);
          PetscPrintf(PETSC_COMM_WORLD, "Fung initial file successfully has being read\n");

          initFung(&fem[ibi], init_fung_coeffs);
        }
        else{
          initialize_elasticity(&fem[ibi], initial_elas, initial_poisson);  
        }
      }
      PetscPrintf(PETSC_COMM_WORLD, "Inverse model initialized\n");

  /// Initial Residual Calculation ///

      ResidualCalc(ibi, &fem[ibi]);
      // PetscScalar *RRes;
      // VecGetArray(fem[ibi].Res, &RRes);
      // for (int i=0; i<ibm->n_v; i++){  
      //   // printf("body=%d, node=%d, Initial Res : %f,%f,%f \n",ibi, i, RRes[i*dof+0], RRes[i*dof+1], RRes[i*dof+2]);            
      // }
  
      // VecRestoreArray(fem[ibi].Res, &RRes);
  
      if (ressmooth){
        PetscPrintf(PETSC_COMM_WORLD, "Residual smoothing option has being selected\n");
        NeighNodeFinder(&fem[ibi]);
      }
  
      if (ConstitutiveLawNonLinear) {                
        
        for (int m = 0; m < n_Fung_Coeffs; m++){
          for(int ec=0; ec<ibm->n_elmt; ec++){
            for(int nv=0; nv<ibm->n_v; nv++){
              for (int k = 0; k < dof; k++){          
                fem[ibi].Jac_Fung[nv][ec][k][m] = 0.0;          
              }      
            }  
          }
        }                                        
       
        
      }      
    }        
  
  }
  
      
  /// Main Update Loop
  for (epoch=epoch_start; epoch<n_epochs+epoch_start; epoch++){

    double total_loss = 0.0;

    if (1){

    // Residual Calculations for all bodies
      {for (ibi=0; ibi<nbody; ibi++){      
        IBMNodes   *ibm=fem[ibi].ibm;

        if (ConstitutiveLawNonLinear){
          reset_Fung_epsilon(&fem[ibi]);
        }
        
        ResidualCalc(ibi, &fem[ibi]);
  
        if (ressmooth){
          ResSmoother(&fem[ibi], res_smooth_itrs, 0.5);
          ResSmoothToRes(&fem[ibi]);
        }
  
        PetscScalar *RRes;
        VecGetArray(fem[ibi].Res, &RRes);   
  
        for (PetscInt i=0; i<ibm->n_v; i++){
          for (PetscInt j=0; j<3; j++){            
            double point_residual = RRes[i*dof+j];                                       
            total_loss += ((point_residual - 0.0) * (point_residual - 0.0)/ ibm->n_v);          
            
          }
          // printf("body=%d, node=%d,  Res : %f,%f,%f \n",ibi, i, RRes[i*dof+0], RRes[i*dof+1], RRes[i*dof+2]);
        }
        
        VecRestoreArray(fem[ibi].Res, &RRes);  
      }
      }         
    }
    
    // Jacobian Update for all bodies 
    if ( ( (epoch % (epoch_update_jacobian)==0 || (epoch==epoch_start)) && (epoch != n_epochs + epoch_start - 1) ) ){

      if (par_jac){

        if (epoch == 0){
          PetscPrintf(PETSC_COMM_WORLD, "Parallel Jacobian Computation\n");
        }
        PetscBarrier(PETSC_NULL);

        for (ibi=0; ibi<nbody; ibi++){     
          double t_start = MPI_Wtime();
 
          Jacobian(ibi, &fem[ibi], 1e-4);          
          PetscBarrier(PETSC_NULL);  

          /// Creating seq copies of MPI Jacobian on each rank
          MatCreateRedundantMatrix(fem[ibi].Jacobian, size, PETSC_COMM_SELF, MAT_INITIAL_MATRIX, &fem[ibi].J_Seq);
          
          double t_end = MPI_Wtime();
          PetscPrintf(PETSC_COMM_WORLD, "Elapsed time: %f seconds\n", t_end - t_start);

          JacMatToArr(&fem[ibi]);        
          PetscBarrier(PETSC_NULL);
          MatDestroy(&(fem[ibi].J_Seq));
          
          PetscPrintf(PETSC_COMM_WORLD, "Jac Mat transfered to Jac 4D array\n");
        }        
      }

      else{
        if (!rank){
          
          if (epoch == 0){
            PetscPrintf(PETSC_COMM_WORLD, "Single processor Jacobian Computation\n");
          }
          for (ibi=0; ibi<nbody; ibi++){ 
            if (ConstitutiveLawNonLinear) {   
            
              if (uniform_fung){
                PetscPrintf(PETSC_COMM_WORLD, "Spatially Uniform Jacobian\n");
                FungUniJacobian(ibi, &fem[ibi], 1e-1);
              }
              else {
                PetscPrintf(PETSC_COMM_WORLD, "Jacobian Computation\n");
                FungJacobian(ibi, &fem[ibi], 1e-4);
                PetscPrintf(PETSC_COMM_WORLD, "Jacobian Computation completed successfully\n");
              }
            }
  
            else{
              grad_R_wrt_E_field(ibi, dof_index, &fem[ibi], 1e-1);
            }          
          }
          
        }      
      }      
    }
    
    // PetscPrintf(PETSC_COMM_WORLD, "Before start updating ...\n");

    /// Updating the material propertis coeffecients
    if (1){
      for (ibi=0; ibi<nbody; ibi++){      
        IBMNodes   *ibm=fem[ibi].ibm;      
              
        if (ConstitutiveLawNonLinear) {
  
          if (Adam){          
            updateFungAdam(lr_fung, &fem[ibi], epoch);
            if (fibersmooth){
              FiberSmoother(fem, 2, 0.5);
              UpdateFiber_from_smth(fem);
            }                                                           
          }
  
          else{
            updateFung(lr_fung, epoch, &fem[ibi]);
          }
        }
  
        else {        
          update_elasticity(lr_lin, &fem[ibi], epoch);
        }
        
        if ((epoch%epoch_output==0) && (!rank)){
          Output(&fem[ibi], tistart+epoch, ibi, subdir);
          if (epoch == n_epochs+epoch_start-1){
            InverseOut(&fem[ibi], tistart+epoch, ibi, subdir);  
          }
        }
      } 
    
      
      if (ConstitutiveLawNonLinear) {
      
        printf("Epoch %d, Average Loss: %.10f, body: %d, theta: %f\n", epoch + 1, total_loss, 0, fem[0].ibm->Fung_coeffs[7][10]);
        
        for (ibi=0; ibi<nbody; ibi++){
          PetscReal *A_i;
          PetscMalloc((n_Fung_Coeffs)*sizeof(PetscReal), &(A_i));        
  
          // PetscReal A_i = 0.0, avg_nu = 0.0;
          for (int m = 0; m < n_Fung_Coeffs; m++){
            A_i[m] = 0.0;
            for(int ec=0; ec < fem[ibi].ibm->n_elmt; ec++){
              A_i[m] = A_i[m] + fem[ibi].ibm->Fung_coeffs[m][ec]/fem[ibi].ibm->n_elmt;
            }  
          }
          
          PetscFPrintf(PETSC_COMM_WORLD, fp[ibi], "%d %.10f %d %f %f %f %f %f %f %f %f\n", 
          epoch + 1, total_loss, ibi, A_i[0], A_i[1], A_i[2], A_i[3], A_i[4], A_i[5], A_i[6], A_i[7]);
  
          PetscFree(A_i);
        }
        
        // printf("Epoch %d, Average Loss: %.8f, body: %d, theta: %f\n", epoch + 1, total_loss, 1, fem[0].ibm->Fung_coeffs[7][10]);
        // printf("Epoch %d, Average Loss: %.8f, body: %d, theta: %f\n", epoch + 1, total_loss, 2, fem[2].ibm->Fung_coeffs[7][10]);
      }
      else{
        PetscReal avg_El = 0.0, avg_nu = 0.0;
        for(int ec=0; ec<fem[0].ibm->n_elmt; ec++){
          avg_El += (fem[0].ibm->El[0][ec]/fem[0].ibm->n_elmt);
          avg_nu += fem[0].ibm->El[1][ec]/fem[0].ibm->n_elmt;
        }
        
  
        PetscFPrintf(PETSC_COMM_WORLD, fp[0], "%d %.8f %d %f %f\n", 
          epoch + 1, total_loss, 0, avg_El, avg_nu);
  
        printf("Epoch %d, Average Loss: %.8f, body: %d, El: %f, nu: %f\n", epoch + 1, total_loss, 0, avg_El, avg_nu);
      }


    }    
  }
  // Close the file
  PetscPrintf(PETSC_COMM_WORLD, "Before closing the inverse result files ...\n");

  for (ibi=0; ibi<nbody; ibi++){
    PetscFClose(PETSC_COMM_WORLD, fp[ibi]);
  }

  PetscPrintf(PETSC_COMM_WORLD, "Before going out of InvSolver ...\n");    
  // }
  
  return (0);
}


PetscErrorCode FiberSmoother(FE *fem, PetscInt n_iters, PetscReal smth_factor){ 
  /*
  Description: 
    Apply laplace operator to the rotaion angle of the Fung model.
  Note:
    Works for inverse BHV 
  */

  PetscInt  ibi, i, j, mn;
  PetscInt  n1e, n2e, n3e, *nv4, *nv5, *nv6;
  PetscInt  n1pe, n2pe, n3pe;

  for (ibi=0; ibi<nbody; ibi++){    

    IBMNodes   *ibm=fem[ibi].ibm;

    PetscReal      *theta_old;
    PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &(theta_old));

    PetscInt n_elmt = ibm->n_elmt;

    // copy initial theta to a new array which stores the smooth values of theta
    for (i=0; i<n_elmt; i++) {
      ibm->Fung_coeffs_smth[7][i] = ibm->Fung_coeffs[7][i];
    }

    for (int m=0; m<n_iters; m++){

      //update theta_old
      for (i=0; i<n_elmt; i++) {
        theta_old[i] = ibm->Fung_coeffs_smth[7][i];
      }  

      for (i=0; i<n_elmt; i++) {
      
      PetscReal sum = 0.0;
      PetscInt n_cmn_faces = 0;

      n1e = ibm->nv1[i];  n2e = ibm->nv2[i];  n3e = ibm->nv3[i];
      
      for (j=0; j<n_elmt; j++) {
        
        n1pe = ibm->nv1[j];  n2pe = ibm->nv2[j];  n3pe = ibm->nv3[j];
  
        mn = 0;
        if(n1e==n1pe || n1e==n2pe || n1e==n3pe){mn = mn+1;}
        if(n2e==n1pe || n2e==n2pe || n2e==n3pe){mn = mn+1;}
        if(n3e==n1pe || n3e==n2pe || n3e==n3pe){mn = mn+1;}
        
        if(mn==2){ //we catched a common face
          n_cmn_faces += 1;
          sum += theta_old[j];
        }
      }

      ibm->Fung_coeffs_smth[7][i] = (ibm->Fung_coeffs[7][i] + smth_factor * sum)/(1 + smth_factor * n_cmn_faces);
    } 
  }

  PetscFree(theta_old);
  }

  return (0);
}


PetscErrorCode UpdateFiber_from_smth(FE *fem){
  PetscInt ibi, i, j, mn;
  PetscInt  n1e, n2e, n3e, *nv4, *nv5, *nv6;
  PetscInt  n1pe, n2pe, n3pe;

  for (ibi=0; ibi<nbody; ibi++){    
    
    IBMNodes   *ibm=fem[ibi].ibm;

    PetscInt n_elmt = ibm->n_elmt;

    // copy initial theta to a new array which stores the smooth values of theta

    for (i=0; i<n_elmt; i++) {
      ibm->Fung_coeffs[7][i] = ibm->Fung_coeffs_smth[7][i];
    }
  }

}


PetscErrorCode NeighNodeFinder(FE *fem){
  
  PetscInt MAX_NUM_NEIGH_NODES = 10;
  PetscInt INIT_VALUE = 1000000;

  void check_and_insert(PetscInt **elmt_idxs, PetscInt i, PetscInt value) {
    for (PetscInt m = 0; m < MAX_NUM_NEIGH_NODES; m++) {
        if (elmt_idxs[i][m] == value) return; // Value already exists
    }
    // Insert into the first available slot
    for (PetscInt m = 0; m < MAX_NUM_NEIGH_NODES; m++) {
        if (elmt_idxs[i][m] == INIT_VALUE) {
            elmt_idxs[i][m] = value;
            return;
        }
    }
  }


  IBMNodes   *ibm=fem->ibm;

  PetscInt  ibi, i, j, mn;
  PetscInt  n1e, n2e, n3e, *nv4, *nv5, *nv6;
  PetscInt  n1pe, n2pe, n3pe;

  PetscInt n_elmt = ibm->n_elmt;
  PetscInt n_v = ibm->n_v;
  
  for (PetscInt i = 0; i < n_v; i++) {      

    for (PetscInt m = 0; m < MAX_NUM_NEIGH_NODES; m++) {

        ibm->neigh_nodes_ind[i][m] = INIT_VALUE;
    }
  }

  
  for (i=0; i<n_v; i++) {  
          
    /// Search for neighbouring nodes   
    for (j=0; j<n_elmt; j++) {
      
      n1pe = ibm->nv1[j];  n2pe = ibm->nv2[j];  n3pe = ibm->nv3[j];

      mn = 0;
      if(i==n1pe || i==n2pe || i==n3pe){
        check_and_insert(ibm->neigh_nodes_ind, i, i);
        check_and_insert(ibm->neigh_nodes_ind, i, n1pe);
        check_and_insert(ibm->neigh_nodes_ind, i, n2pe);
        check_and_insert(ibm->neigh_nodes_ind, i, n3pe);          
        mn = mn+1;
      }                        
    }
    // printf("ibm->neigh_nodes_ind[%d] = [%d,%d,%d,%d] \n", i, ibm->neigh_nodes_ind[i][1], ibm->neigh_nodes_ind[i][2], ibm->neigh_nodes_ind[i][3],ibm->neigh_nodes_ind[i][4]);
  }
  
}


PetscErrorCode ResSmoother(FE *fem, PetscInt n_iters, PetscReal smth_factor){ 
  /*
  Description: 
    Apply laplace operator to each component of the Residual vector nodal values.

  */
  
  PetscInt MAX_NUM_NEIGH_NODES = 10;
  PetscInt INIT_VALUE = 1000000;
  
  IBMNodes  *ibm=fem->ibm;

  PetscInt  ibi, i, j, mn;
  PetscInt  n1e, n2e, n3e, *nv4, *nv5, *nv6;
  PetscInt  n1pe, n2pe, n3pe;

  PetscInt  n_elmt = ibm->n_elmt;
  PetscInt  n_v = ibm->n_v;
  
  PetscReal *res_old;
  PetscMalloc(dof*ibm->n_v*sizeof(PetscReal), &(res_old));          
  
  PetscScalar *RRes_smth;
  VecGetArray(fem->Res_smth, &RRes_smth);

  PetscScalar *RRes;
  VecGetArray(fem->Res, &RRes);

  for (int i = 0; i < ibm->n_v; ++i) {
    for (int k = 0; k < dof; k++){
      RRes_smth[i*dof+k] = RRes[i*dof+k];
    }        
  }

  for (int m=0; m<n_iters; m++){

    //update res_old
    for (i=0; i<n_v; i++) {
      for (int k = 0; k < dof; k++){
        res_old[i*dof+k] = RRes_smth[i*dof+k];
      }
    }  

    for (i=0; i<n_v; i++) {

      for (int k = 0; k < dof; k++){
        PetscReal sum = 0.0;
        PetscInt  connec_counter = 0;

        for (PetscInt m = 1; m < MAX_NUM_NEIGH_NODES; m++) {

          if (ibm->neigh_nodes_ind[i][m] != INIT_VALUE){
            sum += res_old[ibm->neigh_nodes_ind[i][m] * dof + k];
            connec_counter += 1;
          }
        }

        RRes_smth[i*dof+k] = (RRes[i*dof+k] + smth_factor * sum)/(1 + connec_counter);
      }          
    } 
  }

  VecRestoreArray(fem->Res_smth, &RRes_smth);
  VecRestoreArray(fem->Res, &RRes);

  PetscFree(res_old);
  

  return (0);
}


PetscErrorCode ResSmoothToRes(FE *fem){ 
  /*
  Description: 
    Update Residual from smoothened Residual after calling ResSmoother function

  */

  IBMNodes  *ibm=fem->ibm;

  PetscScalar *RRes_smth;
  PetscScalar *RRes;

  VecGetArray(fem->Res_smth, &RRes_smth);
  VecGetArray(fem->Res, &RRes);

  for (int i = 0; i < ibm->n_v; ++i) {
    for (int k = 0; k < dof; k++){
      RRes[i*dof+k] = RRes_smth[i*dof+k];
    }        
  }

  VecRestoreArray(fem->Res_smth, &RRes_smth);
  VecRestoreArray(fem->Res, &RRes);

}