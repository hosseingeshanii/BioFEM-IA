/**
 * @file muscle_activation.c
 * @brief Implementation of muscle activation on shell element using active-strain, active stress approaches.
 *
 *
 * References:
 *  - Nitti, A., et al. (2021). A curvilinear isogeometric framework for the purely elastic response of shell structures
 *    undergoing active strain patterns. *Computers & Structures*, 248, 106533.
 *  - Torre, M., et al. (2024). An efficient active-stress electromechanical isogeometric formulation for Kirchhoff–Love shells.
 *    *Computers & Structures*, 276, 106879.
 *
 * @author Hossein Geshani
 * @date 2025-09-22
 */


 /**
 * @brief High-Level Procedure for Muscle Activation Update
 * @details
 * 1. Calculate active part of deformation gradient tensor coefficients in undeformed 
 *    curvilinear basis (Fa_ij) for all elements.
 * 2. For element with index ec, find intermediate state of shell midplane x_e after activation. 
 * 3. For element with index ec, update intermediate state basis vectors ge.
 * 4. For element with index ec, update elastic strain and stress.
 * 5. Update internal nodal forces on the patch corresponding to element idx ec.
 * 6. Assemble nodal internal forces (looping through all elements).
 */




#include  "math.h"
#include  "variables.h"
#include <petsctao.h>
#include <petscviewer.h>

PetscInt           fiber_based_act_coeff, curv_based_act_coeffs, cart_fib_act; 
extern  PetscInt   dof, curvature, ConstitutiveLawNonLinear;
extern  PetscReal  E, mu, rho, h0;
extern  PetscReal  initial_elas, initial_poisson;

PetscInt           num_gaussian_quad_points;

#define LOG(msg) PetscPrintf(PETSC_COMM_WORLD, "[%s] %s\n", __func__, msg)

/*
 * update_user_act_params — implementation.
 * Uses PETSc options database to set muscle activation coefficients:
 *  - -muscle_act_gamma always,
 *  - -muscle_act_a_1 if fiber- or curvature-based activation is enabled,
 *  - -muscle_act_a_2 only if curvature-based activation is enabled.
 * See muscle_activation.h for the API contract.
 */
PetscErrorCode update_user_act_params(FE *fem){
    UserCtx  *userctx = &fem->userctx;    

    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-fiber_based_act_coeff", &fiber_based_act_coeff, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-curv_based_act_coeffs", &curv_based_act_coeffs, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-cart_fib_act", &cart_fib_act, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num_gaussian_quad_points", &num_gaussian_quad_points, PETSC_NULL);
    
    
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_gamma", &(userctx->muscle_act_params.gamma), PETSC_NULL);

    if (fiber_based_act_coeff){
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_a_1", &(userctx->muscle_act_params.a_1), PETSC_NULL);
    }
    else if (curv_based_act_coeffs){
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_a_1", &(userctx->muscle_act_params.a_1), PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_a_2", &(userctx->muscle_act_params.a_2), PETSC_NULL);
    }
    return 0;
}


/**
 * @brief High-Level Procedure for Muscle Activation Update
 * @details
 * 1. Calculate active part of deformation gradient tensor coefficients in undeformed 
 *    curvilinear basis (Fa_ij) for all elements.
 */
PetscErrorCode compute_act_Fa_element(FE *fem, PetscInt ec)
{
    
    IBMNodes *ibm = fem->ibm;
    UserCtx  userctx = fem->userctx;

    struct Cmpnts g_cov[3]; // Covariant basis vectors
    struct Cmpnts g_contra[3]; // Contravariant basiss vectors
    PetscReal G_cov[3][3], G_contra[3][3]; // Metric tensors
    

    // Load covariant basis vectors
    g_cov[0].x = ibm->G1[ec*dof]; g_cov[0].y = ibm->G1[ec*dof+1]; g_cov[0].z = ibm->G1[ec*dof+2];
    g_cov[1].x = ibm->G2[ec*dof]; g_cov[1].y = ibm->G2[ec*dof+1]; g_cov[1].z = ibm->G2[ec*dof+2];    
    g_cov[2].x = UNIT(CROSS(g_cov[0], g_cov[1])).x;
    g_cov[2].y = UNIT(CROSS(g_cov[0], g_cov[1])).y;
    g_cov[2].z = UNIT(CROSS(g_cov[0], g_cov[1])).z;

    
    // Step 1: Compute covariant metric tensor G_cov[i][j] = g_i · g_j
    for (PetscInt i = 0; i < 3; i++){

        for (PetscInt j = 0; j < 3; j++){
            G_cov[i][j] = DOT(g_cov[i], g_cov[j]);
            // printf("G_cov[i][j] = %.10f \n", G_cov[i][j]);
        }
    }
                     

    // Step 2: Invert G_cov to get G_contra
    PetscReal det = G_cov[0][0]*(G_cov[1][1]*G_cov[2][2] - G_cov[1][2]*G_cov[2][1])
                  - G_cov[0][1]*(G_cov[1][0]*G_cov[2][2] - G_cov[1][2]*G_cov[2][0])
                  + G_cov[0][2]*(G_cov[1][0]*G_cov[2][1] - G_cov[1][1]*G_cov[2][0]);

    if (fabs(det) < 1e-12) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Singular metric tensor");

   
    // Step 2: Invert G_cov to get G_contra using your math.c function
    PetscCall(INV(G_cov, G_contra));


    // Step 3: Compute contravariant basis vectors g^i = G^{ij} g_j
    for (PetscInt i = 0; i < 3; i++) {
        g_contra[i].x = g_contra[i].y = g_contra[i].z = 0.0;
        for (PetscInt j = 0; j < 3; j++) {
            g_contra[i].x += G_contra[i][j] * g_cov[j].x;
            g_contra[i].y += G_contra[i][j] * g_cov[j].y;
            g_contra[i].z += G_contra[i][j] * g_cov[j].z;
        }
    }

    PetscReal gamma = userctx.muscle_act_params.gamma;

    if (fiber_based_act_coeff) {

        // printf("in fiber_based_act_coeff \n ");
        struct Cmpnts nfib = ibm->n_fib[ec];
        PetscReal nfib_comp[3], nperp_comp[3];

        for (PetscInt i = 0; i < 3; i++) {
            nfib_comp[i] = DOT(nfib, g_contra[i]);
             
            nperp_comp[i] = DOT(g_cov[2], g_contra[i]); // thickness direction          
        }
        // printf("m3i = %f, m3j = %f, m3k = %f \n", nperp_comp[0], nperp_comp[1],nperp_comp[2]);
        // nperp_comp[i] = g_contra[i];

        PetscReal J_ap = 1.0 - gamma * (nfib_comp[0]*nfib_comp[0] + nfib_comp[1]*nfib_comp[1]);
        PetscReal gamma_n = 1.0 / J_ap - 1.0;

        for (PetscInt i = 0; i < 3; i++) {
            for (PetscInt j = 0; j < 3; j++) {
                
                fem->act_data.elem_act_data[ec].Fa[i][j] = G_contra[i][j]
                    - gamma * nfib_comp[i] * nfib_comp[j]
                    + gamma_n * nperp_comp[i] * nperp_comp[j];
            }
        }
        
    }
    else if (curv_based_act_coeffs) {
        PetscReal a_1 = userctx.muscle_act_params.a_1;
        PetscReal a_2 = userctx.muscle_act_params.a_2;
       
        PetscReal G11 = G_contra[0][0], G12 = G_contra[0][1], G22 = G_contra[1][1];
        PetscReal J_ap = (G11 - gamma*(a_1*G11*G11 + a_2*G12*G12)) *
                         (G22 - gamma*(a_1*G12*G12 + a_2*G22*G22)) -
                         pow(G12 - gamma*(a_1*G11*G12 + a_2*G12*G22), 2);
        
        PetscReal gamma_n = 1.0 / J_ap * (G11*G22 - G12*G12) - 1.0;
        
        for (PetscInt i = 0; i < 3; i++) {
            for (PetscInt j = 0; j < 3; j++) {
                fem->act_data.elem_act_data[ec].Fa[i][j] = G_contra[i][j]
                    - gamma * (a_1 * G_contra[0][i]*G_contra[0][j] + a_2 * G_contra[1][i]*G_contra[1][j])
                    + gamma_n * G_contra[2][i]*G_contra[2][j];

            }
        }
    }

    else if (cart_fib_act){
      // This approach uses the actual Fa matrix not its contravariant components as we had before
      // PetscReal Iden[3][3];
      PetscReal nfiber[3] = {0.0};
      
      nfiber[0] = ibm->n_fib[ec].x; nfiber[1] = ibm->n_fib[ec].y; nfiber[2] = ibm->n_fib[ec].z;

      for (PetscInt i = 0; i < 3; i++) {
        for (PetscInt j = 0; j < 3; j++) {
          
          if (i == j){
            // fem->act_data.elem_act_data[ec].Fa[i][j] = 1.0 - gamma * nfiber[i] * nfiber[j];
            // fem->act_data.elem_act_data[ec].Fa[i][j] = 1.0 - gamma * nfiber[i] * nfiber[j];
          }
          else {
            fem->act_data.elem_act_data[ec].Fa[i][j] = 0.0;
          }
          
        }
      }
      fem->act_data.elem_act_data[ec].Fa[0][0] = 1.0 - gamma; fem->act_data.elem_act_data[ec].Fa[1][1] = 1.0 - gamma;
      fem->act_data.elem_act_data[ec].Fa[2][2] = 1.0;

    }

    return 0;
}


/* update_act_Fa — implementation. See muscle_activation.h for the API contract. */
PetscErrorCode update_act_Fa(FE *fem)
{
    IBMNodes  *ibm=fem->ibm;          

    for (PetscInt ec = 0; ec < ibm->n_elmt; ec++) {
        PetscCall(compute_act_Fa_element(fem, ec));
    }

    return 0;
}

/**
 * @brief Compute intermediate deformed covariant basis vectors.
 *
 * Loads covariant basis vectors from IBM data, computes the metric tensor,
 * and updates g_e using the active deformation gradient for element ec.
 *
 * @param[in,out] fem FE structure.
 * @param[in]     ec  Element index.
 * @return PetscErrorCode 0 on success.
 */
PetscErrorCode compute_intmd_state_cov_basis_element(FE *fem, PetscInt ec){
    IBMNodes *ibm = fem->ibm;
    struct Cmpnts g_cov[3]; // Covariant basis vectors
    PetscReal G_cov[3][3]; // Metric tensors


    // Load covariant basis vectors
    g_cov[0].x = ibm->G1[ec*dof]; g_cov[0].y = ibm->G1[ec*dof+1]; g_cov[0].z = ibm->G1[ec*dof+2];
    g_cov[1].x = ibm->G2[ec*dof]; g_cov[1].y = ibm->G2[ec*dof+1]; g_cov[1].z = ibm->G2[ec*dof+2];    
    g_cov[2].x = UNIT(CROSS(g_cov[0], g_cov[1])).x;
    g_cov[2].y = UNIT(CROSS(g_cov[0], g_cov[1])).y;
    g_cov[2].z = UNIT(CROSS(g_cov[0], g_cov[1])).z;

    
    // Step 1: Compute covariant metric tensor G_cov[i][j] = g_i · g_j
    for (PetscInt i = 0; i < 3; i++){
        for (PetscInt j = 0; j < 3; j++){
            G_cov[i][j] = DOT(g_cov[i], g_cov[j]);
        }
    }
    

    for (PetscInt m = 0; m < 3; m++){
      
      fem->act_data.elem_act_data[ec].g_e[m].x = 0.0; 
      fem->act_data.elem_act_data[ec].g_e[m].y = 0.0; 
      fem->act_data.elem_act_data[ec].g_e[m].z = 0.0;

      for (PetscInt i = 0; i < 3; i++){
        for (PetscInt j = 0; j < 3; j++){
          
          if (curv_based_act_coeffs || fiber_based_act_coeff){
            PetscReal coeff = fem->act_data.elem_act_data[ec].Fa[i][j] * G_cov[j][m];

            fem->act_data.elem_act_data[ec].g_e[m].x += coeff * g_cov[i].x;
            fem->act_data.elem_act_data[ec].g_e[m].y += coeff * g_cov[i].y;
            fem->act_data.elem_act_data[ec].g_e[m].z += coeff * g_cov[i].z;
  
          }                    
        }
      }                

      if (cart_fib_act){
      fem->act_data.elem_act_data[ec].g_e[m].x = fem->act_data.elem_act_data[ec].Fa[0][0] * g_cov[m].x 
                                                + fem->act_data.elem_act_data[ec].Fa[0][1] * g_cov[m].y
                                                + fem->act_data.elem_act_data[ec].Fa[0][2] * g_cov[m].z;

      fem->act_data.elem_act_data[ec].g_e[m].y = fem->act_data.elem_act_data[ec].Fa[1][0] * g_cov[m].x 
                                                + fem->act_data.elem_act_data[ec].Fa[1][1] * g_cov[m].y
                                                + fem->act_data.elem_act_data[ec].Fa[1][2] * g_cov[m].z;

      fem->act_data.elem_act_data[ec].g_e[m].z = fem->act_data.elem_act_data[ec].Fa[2][0] * g_cov[m].x 
                                                + fem->act_data.elem_act_data[ec].Fa[2][1] * g_cov[m].y
                                                + fem->act_data.elem_act_data[ec].Fa[2][2] * g_cov[m].z;

      }
      
    }

    return 0;
}

/* update_intmd_state_cov_basis — implementation. See muscle_activation.h for the API contract. */
PetscErrorCode update_intmd_state_cov_basis(FE *fem){
  IBMNodes  *ibm=fem->ibm;          
  PetscReal *garray;
  PetscInt nBasisVecs = 2;

  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++) {        
    PetscCall(compute_intmd_state_cov_basis_element(fem, ec));

  }    

  VecGetArray(fem->act_data.g_e_target, &garray);
    

  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++){    
    for (PetscInt m = 0; m < nBasisVecs; m++){      
      
      garray[ec * nBasisVecs * dof + m * dof + 0] = fem->act_data.elem_act_data[ec].g_e[m].x;      
      garray[ec * nBasisVecs * dof + m * dof + 1] = fem->act_data.elem_act_data[ec].g_e[m].y;
      garray[ec * nBasisVecs * dof + m * dof + 2] = fem->act_data.elem_act_data[ec].g_e[m].z; 
      
    }    
  }

  VecRestoreArray(fem->act_data.g_e_target, &garray);

  return 0;
}

PetscErrorCode calculate_cov_basis(Vec x_guess, Vec F, void *ctx){

  FE          *fem = (FE*)ctx;
  IBMNodes    *ibm=fem->ibm;

  PetscReal   sum;
  PetscInt    i, j, m, n1e, n2e, n3e;

  
  // Update x_bp which has been always the array of deformed final configuration
  PetscReal   *xx, *farray;
  PetscInt    nv, ec;

  PetscInt nBasisVecs = 2;

  VecGetArray(x_guess, &xx);
  for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) { 
    ibm->x_bp[nv] = xx[nv*dof  ];
    ibm->y_bp[nv] = xx[nv*dof+1];
    ibm->z_bp[nv] = xx[nv*dof+2];
  }
  VecRestoreArray(x_guess, &xx);

  VecGetArray(F, &farray);

  for (ec=0; ec<ibm->n_elmt; ec++) {

    n1e=ibm->nv1[ec];  n2e=ibm->nv2[ec];  n3e=ibm->nv3[ec];    
    
    if (curvature==6) {

      PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12];
      struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, nn;
      struct Cmpnts  ndx[2];
      PetscInt       v=ibm->val[ec], node;
     
      Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
      Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
      
      Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
      Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;

      Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
      Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
      
      Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
      Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
      
      Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
      Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;

  if (ibm->ire[ec]==0) {

	for (i=0; i<12; i++) {

	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    x[i] = ibm->x_bp[node];  y[i] = ibm->y_bp[node];  z[i] = ibm->z_bp[node];
	  }
	}		

  Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;

	for (i=0; i<12; i++) {
	  Aaa.x += Nab[0][i]*x[i];
	  Aaa.y += Nab[0][i]*y[i];
	  Aaa.z += Nab[0][i]*z[i];

	  Abb.x += Nab[1][i]*x[i];
	  Abb.y += Nab[1][i]*y[i];
	  Abb.z += Nab[1][i]*z[i];

	  Aab.x += Nab[2][i]*x[i];
	  Aab.y += Nab[2][i]*y[i];
	  Aab.z += Nab[2][i]*z[i];
	}

	for (PetscInt m = 0; m < nBasisVecs; m++){
    ndx[m].x = 0.0; ndx[m].y = 0.;  ndx[m].z = 0.;
    for (i = 0; i < 12; i++) {
      ndx[m].x += Na[m][i]*x[i];
      ndx[m].y += Na[m][i]*y[i];
      ndx[m].z += Na[m][i]*z[i];
    }

    farray[dof*nBasisVecs*ec + dof * m + 0] = ndx[m].x;
    farray[dof*nBasisVecs*ec + dof * m + 1] = ndx[m].y;
    farray[dof*nBasisVecs*ec + dof * m + 2] = ndx[m].z;
    

    // Updating the element g_e. The target basis vectors g_e_target remain unchanged 

    fem->act_data.elem_act_data[ec].g_e[m].x = ndx[m].x;
    fem->act_data.elem_act_data[ec].g_e[m].y = ndx[m].y;
    fem->act_data.elem_act_data[ec].g_e[m].z = ndx[m].z;
    //----------------------------------------------------------------------//        
  }     
    // Updating the curvature.

  nn=UNIT(CROSS(ndx[0], ndx[1]));
  
  fem->act_data.elem_act_data[ec].k_e[0] = DOT(Aaa, nn);
  fem->act_data.elem_act_data[ec].k_e[1] = DOT(Abb, nn);
  fem->act_data.elem_act_data[ec].k_e[2] = DOT(Aab, nn);

  }else if (ibm->ire[ec]==1) {


	PetscReal w;
	PetscReal **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));
	for (i=0; i<(v+6); i++) {
	  X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
	}	
			
	for (i=0; i<(v+6); i++) {
	  for (j=0; j<3; j++) {
	    X0[i][j] = 0.;
	  }
	}
	
	for (i=0; i<(v+6); i++) {
	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    X0[i][0] = ibm->x_bp[node];  X0[i][1] = ibm->y_bp[node];  X0[i][2] = ibm->z_bp[node];
	  }
	}

	w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.));
	             			
	PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];	
	//-----B2=dU2/dU1 picking matrix
	for(i=0; i<12; i++) {
	  for (j=0; j<v+12; j++) {
	    B2[i][j] = 0.;
	  }
	}
	
	B2[0][v+9] = 1.;  
	B2[1][v+6] = 1.;  
	B2[2][v+4] = 1.;  
	B2[3][v+1] = 1.;  
	B2[4][v+2] = 1.;  
	B2[5][v+5] = 1.;  
	B2[6][v] = 1.;      
	B2[7][1] = 1.;     
	B2[8][v+3] = 1.; 
	B2[9][v-1] = 1.; 
	B2[10][0] = 1.;     
	B2[11][2] = 1.;     
	
	//------B1=dU1/dU0 Loop's mask
	for(i=0; i<(v+12); i++) {
	  for (j=0; j<(v+6); j++) {
	    B1[i][j] = 0.;
	  }
	}
		
	B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
	B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
	B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
        if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
        if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
        B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
	B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
	B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
	B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
	B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
	B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
	B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
	B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
	B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
	B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
	B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
	B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
	B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
	B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;
	
	for (i=0; i<12; i++) {
	  for (j=0; j<(v+6); j++) {
	    B3[i][j] = 0.;
	    for (m=0; m<(v+12); m++) {
	      B3[i][j] += B2[i][m]*B1[m][j];
	    }
	  }
	}

	PetscReal INa[2][v+6], INab[3][v+6];
	PetscReal sum1, sum2, sum3, sum4, sum5;

	for (j=0; j<(v+6); j++) {
	  sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
	  for (i=0; i<12; i++) {
	    sum1 += B3[i][j]*Na[0][i];
	    sum2 += B3[i][j]*Na[1][i];
	    sum3 += B3[i][j]*Nab[0][i];
	    sum4 += B3[i][j]*Nab[1][i];
	    sum5 += B3[i][j]*Nab[2][i];
	  }
	  INa[0][j] = -2*sum1;
	  INa[1][j] = -2*sum2;
	  INab[0][j] = 4*sum3;
	  INab[1][j] = 4*sum4;
	  INab[2][j] = 4*sum5;
	}
      
  Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
	
	for (i=0; i<(v+6); i++) {
	  Aaa.x += INab[0][i]*X0[i][0];
	  Aaa.y += INab[0][i]*X0[i][1];
	  Aaa.z += INab[0][i]*X0[i][2];
	  
	  Abb.x += INab[1][i]*X0[i][0];
	  Abb.y += INab[1][i]*X0[i][1];
	  Abb.z += INab[1][i]*X0[i][2];
	  
	  Aab.x += INab[2][i]*X0[i][0];
	  Aab.y += INab[2][i]*X0[i][1];
	  Aab.z += INab[2][i]*X0[i][2];
	}

	for (PetscInt m = 0; m < nBasisVecs; m++){
    ndx[m].x = 0.0; ndx[m].y = 0.0;  ndx[m].z = 0.0;
    for (i=0; i<(v+6); i++) {
      ndx[m].x += INa[m][i]*X0[i][0];
      ndx[m].y += INa[m][i]*X0[i][1];
      ndx[m].z += INa[m][i]*X0[i][2];	
    }
    farray[dof*nBasisVecs*ec + dof * m + 0] = ndx[m].x;
    farray[dof*nBasisVecs*ec + dof * m + 1] = ndx[m].y;
    farray[dof*nBasisVecs*ec + dof * m + 2] = ndx[m].z;
  }   
		
	nn=UNIT(CROSS(ndx[0], ndx[1]));
  
  fem->act_data.elem_act_data[ec].k_e[0] = DOT(Aaa, nn);
  fem->act_data.elem_act_data[ec].k_e[1] = DOT(Abb, nn);
  fem->act_data.elem_act_data[ec].k_e[2] = DOT(Aab, nn);


	for (i=0; i<(v+6); i++) {
	  free(X0[i]);
	}
	free(X0); 
	
      }     
    }
  }
  
  VecRestoreArray(F, &farray);

  return(0);
}

PetscErrorCode form_interm_state_func_grad(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx){
  FE *fem       = (FE*)ctx;
  IBMNodes *ibm = fem->ibm;

  Vec F, diff;
  PetscScalar dot;
  
  VecDuplicate(fem->act_data.g_e_target, &F);
  VecDuplicate(fem->act_data.g_e_target, &diff);

  GlobalGhost(ibm);

  calculate_cov_basis(x, F, fem);
  PetscInt iter, niters;
  TaoGetSolutionStatus(tao, &iter, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
  
  
  
  VecWAXPY(diff, -1.0, fem->act_data.g_e_target, F); // diff = F - target  
  VecDot(diff, diff, &dot);
  *f = 0.5 * dot;

  // Finite difference gradient 
  PetscReal eps = 1e-6;
  PetscInt n3;
  VecGetSize(x, &n3);

  PetscScalar *garray;
  VecZeroEntries(g);
  VecGetArray(g, &garray);

  for (PetscInt i = 0; i < n3; i++) {
    PetscScalar xi, fi_plus, fi_minus;
    VecGetValues(x, 1, &i, &xi);

    VecSetValue(x, i, xi + eps, INSERT_VALUES);
    VecAssemblyBegin(x); VecAssemblyEnd(x);

    GlobalGhost(ibm);

    calculate_cov_basis(x, F, fem);
    VecWAXPY(diff, -1.0, fem->act_data.g_e_target, F);
    VecDot(diff, diff, &fi_plus);

    VecSetValue(x, i, xi - eps, INSERT_VALUES);
    VecAssemblyBegin(x); VecAssemblyEnd(x);

    GlobalGhost(ibm);

    calculate_cov_basis(x, F, fem);
    VecWAXPY(diff, -1.0, fem->act_data.g_e_target, F);
    VecDot(diff, diff, &fi_minus);

    garray[i] = (0.5 * fi_plus - 0.5 * fi_minus) / eps;

    VecSetValue(x, i, xi, INSERT_VALUES);
  }
  
  

  VecRestoreArray(g, &garray);
  VecAssemblyBegin(g); VecAssemblyEnd(g);

  /// Apply BCs 
  apply_act_bcs(fem, 0, g);
  apply_act_bcs(fem, 3, g);

  VecDestroy(&F);
  VecDestroy(&diff);

  return 0;

}

PetscErrorCode MyTaoMonitor(Tao tao, void *ctx)
{
    PetscInt  iter;
    PetscReal f, gnorm, cnorm, xdiff;
    TaoConvergedReason reason;
    PetscMPIInt rank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    /* This retrieves the solver status, including f and gradient norm */
    TaoGetSolutionStatus(tao, &iter, &f, &gnorm, &cnorm, &xdiff, &reason);

    PetscPrintf(PETSC_COMM_WORLD,
        "iter = %D  f = %.6e  ||g|| = %.6e  ||x-xprev|| = %.6e\n",
        iter, (double)f, (double)gnorm, (double)xdiff);


    if (rank == 0) {
        FILE *fp = fopen("tao_log.txt", "a");
        if (fp) {
            fprintf(fp, "%5d  %.6e  %.6e  %.6e  %.6e\n",
                    iter, f, gnorm, cnorm, xdiff);
            fclose(fp);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "Warning: could not open tao_log.txt\n");
        }
    }
    return 0;
}

PetscErrorCode find_intmd_coords(FE *fem){
  Tao tao;
  TaoLineSearch ls;
  Vec F, diff;
  PetscViewer viewerF, viewerDiff, viewerTarget;

  PetscFunctionBegin;

  TaoCreate(PETSC_COMM_SELF, &tao);
  TaoSetType(tao, TAOLMVM);
  TaoSetObjectiveAndGradientRoutine(tao, form_interm_state_func_grad, fem);
  TaoSetSolution(tao, fem->x);

  // TaoGetLineSearch(tao, &ls);
  // TaoLineSearchSetFromOptions(ls);

  TaoAppendOptionsPrefix(tao, "muscle_act_");
  TaoSetFromOptions(tao);

  // Duplicate vectors
  VecDuplicate(fem->act_data.g_e_target, &F);
  VecDuplicate(fem->act_data.g_e_target, &diff);

  // Compute F
  calculate_cov_basis(fem->x, F, fem);

  // Compute diff = F - g_e_target
  VecWAXPY(diff, -1.0, fem->act_data.g_e_target, F);

  // --- Write vectors to files ---
  // PetscViewerASCIIOpen(PETSC_COMM_SELF, "F.txt", &viewerF);
  // VecView(F, viewerF);
  // PetscViewerDestroy(&viewerF);

  // PetscViewerASCIIOpen(PETSC_COMM_SELF, "diff.txt", &viewerDiff);
  // VecView(diff, viewerDiff);
  // PetscViewerDestroy(&viewerDiff);

  // PetscViewerASCIIOpen(PETSC_COMM_SELF, "g_e_target.txt", &viewerTarget);
  // VecView(fem->act_data.g_e_target, viewerTarget);
  // PetscViewerDestroy(&viewerTarget);

  // Optional: print initial diff to screen
  // PetscPrintf(PETSC_COMM_SELF, "\nInitial diff Vec:\n");
  // VecView(diff, PETSC_VIEWER_STDOUT_SELF);


  TaoSetMonitor(tao, MyTaoMonitor, NULL, NULL);

  TaoSolve(tao);

  TaoDestroy(&tao);
  VecDestroy(&F);
  VecDestroy(&diff);

  PetscFunctionReturn(0);
}

PetscErrorCode update_intmd_ibm_coords(FE *fem){

  IBMNodes   *ibm=fem->ibm;
  PetscReal  *xx, *xxi;

  VecGetArray(fem->x, &xx);
  VecGetArray(fem->x_intmd, &xxi);
  
  for (PetscInt nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
  
    ibm->x_bp[nv] = xx[nv*dof  ];
    ibm->y_bp[nv] = xx[nv*dof+1];
    ibm->z_bp[nv] = xx[nv*dof+2];

    // xxi[nv*dof]   = xx[nv*dof  ];
    // xxi[nv*dof+1] = xx[nv*dof+1];
    // xxi[nv*dof+2] = xx[nv*dof+2];

    ibm->x_bpi[nv] = xx[nv*dof  ];
    ibm->y_bpi[nv] = xx[nv*dof+1];
    ibm->z_bpi[nv] = xx[nv*dof+2];

  }

  VecRestoreArray(fem->x, &xx);  
  VecRestoreArray(fem->x_intmd, &xxi);
  
  return 0;
}

PetscErrorCode apply_act_bcs(FE *fem, PetscInt edge_n, Vec G){

  IBMNodes   *ibm=fem->ibm;
  PetscReal  *xx;
  PetscInt   start=0, end=0, edge, nbc, nb;
  PetscScalar  *g;
  VecGetArray(G, &g);
  
  for (edge=0; edge<edge_n+1; edge++) {
    end += ibm->n_bnodes[edge];
  }
  start = end - ibm->n_bnodes[edge_n];
    
  for (nbc=start; nbc<end; nbc++) { 
    nb=ibm->bnodes[nbc];

    if (edge_n == 0){      
      g[nb*dof + 1] = 0.0;  
    }

    if (edge_n == 3){      
      g[nb*dof + 0] = 0.0;  
    }        
    g[nb*dof + 2] = 0.0;
  }
  VecRestoreArray(G, &g);

  return 0;
}

PetscErrorCode init_act_fibers(FE *fem)
{
  IBMNodes *ibm = fem->ibm;
  struct Cmpnts act_fib;

  // Default values
  act_fib.x = 1.0;
  act_fib.y = 0.0;
  act_fib.z = 0.0;

  // Overwrite if provided in options
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-act_fib_x", &(act_fib.x), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-act_fib_y", &(act_fib.y), PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-act_fib_z", &(act_fib.z), PETSC_NULL);

  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++) {
    fem->act_data.elem_act_data[ec].act_fib.x = act_fib.x;
    fem->act_data.elem_act_data[ec].act_fib.y = act_fib.y;
    fem->act_data.elem_act_data[ec].act_fib.z = act_fib.z;
  }

  return 0;
}

PetscErrorCode compute_mid_surface_el_stress_after_act(FE* fem, PetscInt ec, PetscReal S_m[3], PetscReal S_b[3]){
  // theta3 is the through thickness variable
  // for each theta the total elastic stress would be S_e = S_m + theta3 S_b 
  // (membrane & bending contributions)
  // from fem->x_intmd to fem->x

  IBMNodes       *ibm=fem->ibm;
  
  PetscInt       i, j, m;
  struct Cmpnts  dx21, dx31, gc1, gc2;

  PetscReal      Eb[3];

  struct Cmpnts  X1, X2, X3;
  PetscInt       n1e, n2e, n3e;
  PetscReal      k[3]; 


  //  initial location
  X1.x=ibm->x_bpi[n1e]; X1.y=ibm->y_bpi[n1e]; X1.z=ibm->z_bpi[n1e];
  X2.x=ibm->x_bpi[n2e]; X2.y=ibm->y_bpi[n2e]; X2.z=ibm->z_bpi[n2e];
  X3.x=ibm->x_bpi[n3e]; X3.y=ibm->y_bpi[n3e]; X3.z=ibm->z_bpi[n3e];

  //  
  PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12], b1, b2, b3, nA0, nA;
  struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, ndX21, ndX31, nn, ng1, ng2, ng;
  PetscInt       v=ibm->val[ec], node, nob=1;
  
  Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
  Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
  
  Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
  Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;
      
  Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
  Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
  
  Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
  Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
    
  Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
  Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;
  //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch outside!\n");
  if (ibm->ire[ec]==0) { //-------------------------------------regular patch-----------------------------------------
  //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch inside!\n");
    for (i=0; i<12; i++) {

      if (ibm->patch[16*ec+i]!=1000000) {
        node = ibm->patch[16*ec+i];
        x[i] = ibm->x_bp[node];  y[i] = ibm->y_bp[node];  z[i] = ibm->z_bp[node];
      }

    }

    for (i=0; i<12; i++) {
      if (ibm->patch[16*ec+i]==1000000) {nob=0;}
    }

    Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
    Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
    Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
    
    for (i=0; i<12; i++) {
      Aaa.x += Nab[0][i]*x[i];
      Aaa.y += Nab[0][i]*y[i];
      Aaa.z += Nab[0][i]*z[i];

      Abb.x += Nab[1][i]*x[i];
      Abb.y += Nab[1][i]*y[i];
      Abb.z += Nab[1][i]*z[i];

      Aab.x += Nab[2][i]*x[i];
      Aab.y += Nab[2][i]*y[i];
      Aab.z += Nab[2][i]*z[i];
    }
      
    ndx21.x=0.; ndx21.y=0.; ndx21.z=0.;
    ndx31.x=0.; ndx31.y=0.; ndx31.z=0.;
    
    for (i=0; i<12; i++) {
      ndx21.x += Na[0][i]*x[i];
      ndx21.y += Na[0][i]*y[i];
      ndx21.z += Na[0][i]*z[i];

      ndx31.x += Na[1][i]*x[i];
      ndx31.y += Na[1][i]*y[i];
      ndx31.z += Na[1][i]*z[i];
    }
    
    nn=UNIT(CROSS(ndx21, ndx31));
    ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
    ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];
    nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
    nA = 0.5*SIZE(CROSS(ndx21, ndx31));

    gc1=CROSS(ndx31, nn);
    gc1=AMULT(1./DOT(ndx21, gc1), gc1);
    
    gc2=CROSS(nn, ndx21);
    gc2=AMULT(1./DOT(ndx31, gc2), gc2);

    k[0] = DOT(Aaa,nn);
    k[1] = DOT(Abb,nn);
    k[2] = DOT(Aab,nn);

    ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
    ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
    ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
    //---------------------Compute E and S_m for membrane ---------------------------------------
    PetscReal  g[3], Em[3];
    
    g[0] = DOT(ndx21,ndx21);
    g[1] = DOT(ndx31,ndx31);
    g[2] = DOT(ndx21,ndx31);
  
    Em[0] = 0.5*(g[0] - DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[0]));
    Em[1] = 0.5*(g[1] - DOT(fem->act_data.elem_act_data[ec].g_e[1], fem->act_data.elem_act_data[ec].g_e[1]));
    Em[2] = g[2] - DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[1]);

    // printf("Em[0] = %.10f, Em[1] = %.10f, Em[2] = %.10f \n", Em[0], Em[1], Em[2]);
    // printf("g[0] = %.8f, g[1] = %.8f, g[2] = %.8f \n", g[0], g[1], g[2]);
    // printf("ge[0] = %.8f, ge[1] = %.8f, ge[2] = %.8f \n", DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[0]),
    //  DOT(fem->act_data.elem_act_data[ec].g_e[1], fem->act_data.elem_act_data[ec].g_e[1]), 
    //  DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[1]));
        
    if (ConstitutiveLawNonLinear) {
      MembraneNonLinear(ec, X1, X2, X3, Em, S_m, 1, ibm);
    } else {
      StressLinear(ec, X1, X2, X3, Em, S_m, 1, ibm);
    }
    // printf("S_m[0] = %.10f, S_m[1] = %.10f, S_m[2] = %.10f \n", S_m[0], S_m[1], S_m[2]);

    //---------------------Compute E and S_b for bending ---------------------------------------
    Eb[0] = fem->act_data.elem_act_data[ec].k_e[0] - k[0];
    Eb[1] = fem->act_data.elem_act_data[ec].k_e[1] - k[1];
    Eb[2] = 2*(fem->act_data.elem_act_data[ec].k_e[2] - k[2]);
  
    if (ConstitutiveLawNonLinear) {
      BendingNonLinear(ec, X1, X2, X3, Eb, S_b, 1, ibm);
          } else {
      StressLinear(ec, X1, X2, X3, Eb, S_b, 1, ibm); 
    }
        
  
  } else if (ibm->ire[ec]==1) { //----------------------irregular patch---------------------------
    //PetscPrintf(PETSC_COMM_SELF, "Check if irregular patch inside!\n");
    for (i=0; i<(v+6); i++) {
      if (ibm->patch[16*ec+i]==1000000) {nob=0;}
    }
    
    PetscReal  w;	
    PetscReal  **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));

    for (i=0; i<(v+6); i++) {
      X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
    }      
    
    for (i=0; i<(v+6); i++) {
      for (j=0; j<3; j++) {
        X0[i][j] = 0.;
      }
    }     
    
    for (i=0; i<(v+6); i++) {
      if (ibm->patch[16*ec+i]!=1000000) {
        node = ibm->patch[16*ec+i];
        X0[i][0] = ibm->x_bp[node];  X0[i][1] = ibm->y_bp[node];  X0[i][2] = ibm->z_bp[node];
      }
    }
    
    w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.)); 
          
    PetscReal  B3[12][v+6], B2[12][v+12], B1[v+12][v+6];
    
    //-----B2=dU2/dU1
    for(i=0; i<12; i++) {
for (j=0; j<(v+12); j++) {
  B2[i][j] = 0.;
}
    }
    
    //------B1=dU1/dU0
    for(i=0; i<(v+12); i++) {
for (j=0; j<(v+6); j++) {
  B1[i][j] = 0.;
}
    }
    
    B2[0][v+9] = 1.;  
    B2[1][v+6] = 1.;  
    B2[2][v+4] = 1.;  
    B2[3][v+1] = 1.;  
    B2[4][v+2] = 1.;  
    B2[5][v+5] = 1.;  
    B2[6][v] = 1.;      
    B2[7][1] = 1.;     
    B2[8][v+3] = 1.; 
    B2[9][v-1] = 1.; 
    B2[10][0] = 1.;     
    B2[11][2] = 1.;     
              
    B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
    B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
    B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
    if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
    if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
    B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
    B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
    B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
    B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
    B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
    B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
    B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
    B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
    B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
    B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
    B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
    B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
    B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
    B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;    
          
    for (i=0; i<12; i++) {
      for (j=0; j<(v+6); j++) {
        B3[i][j] = 0.;
        for (m=0; m<(v+12); m++) {
          B3[i][j] += B2[i][m]*B1[m][j];
        }
      }
    }
                  
    PetscReal  INa[2][v+6], INab[3][v+6];
    PetscReal  sum1, sum2, sum3, sum4, sum5;
    
    for (j=0; j<(v+6); j++) {
      sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
      for (i=0; i<12; i++) {
        sum1 += B3[i][j]*Na[0][i];
        sum2 += B3[i][j]*Na[1][i];
        sum3 += B3[i][j]*Nab[0][i];
        sum4 += B3[i][j]*Nab[1][i];
        sum5 += B3[i][j]*Nab[2][i];
      }
      INa[0][j] = -2*sum1;
      INa[1][j] = -2*sum2;
      INab[0][j] = 4*sum3;
      INab[1][j] = 4*sum4;
      INab[2][j] = 4*sum5;
    }     

    Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
    Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
    Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;

    for (i=0; i<(v+6); i++) {
      Aaa.x += INab[0][i]*X0[i][0];
      Aaa.y += INab[0][i]*X0[i][1];
      Aaa.z += INab[0][i]*X0[i][2];

      Abb.x += INab[1][i]*X0[i][0];
      Abb.y += INab[1][i]*X0[i][1];
      Abb.z += INab[1][i]*X0[i][2];

      Aab.x += INab[2][i]*X0[i][0];
      Aab.y += INab[2][i]*X0[i][1];
      Aab.z += INab[2][i]*X0[i][2];
    }
      
    ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
    ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
    
    for (i=0; i<(v+6); i++) {
      ndx21.x += INa[0][i]*X0[i][0];
      ndx21.y += INa[0][i]*X0[i][1];
      ndx21.z += INa[0][i]*X0[i][2];

      ndx31.x += INa[1][i]*X0[i][0];
      ndx31.y += INa[1][i]*X0[i][1];
      ndx31.z += INa[1][i]*X0[i][2];
    }

    nn=UNIT(CROSS(ndx21,ndx31));
    ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
    ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];

    dx21.x = X0[1][0] - X0[0][0]; dx21.y = X0[1][1] - X0[0][1]; dx21.z = X0[1][2] - X0[0][2];
    dx31.x = X0[v][0] - X0[0][0]; dx31.y = X0[v][1] - X0[0][1]; dx31.z = X0[v][2] - X0[0][2];

    nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
    nA = 0.5*SIZE(CROSS(ndx21, ndx31));     

    k[0] = DOT(Aaa,nn);
    k[1] = DOT(Abb,nn);
    k[2] = DOT(Aab,nn);

    gc1=CROSS(ndx31, nn);
    gc1=AMULT(1./DOT(ndx21, gc1), gc1);
    
    gc2=CROSS(nn, ndx21);
    gc2=AMULT(1./DOT(ndx31, gc2), gc2); 
    
    ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
    ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
    ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
    //---------------------Compute E and S_m for membrane ---------------------------------------
    PetscReal  g[3], Em[3];
    
    g[0] = DOT(ndx21,ndx21);
    g[1] = DOT(ndx31,ndx31);
    g[2] = DOT(ndx21,ndx31);
    
    Em[0] = 0.5*(g[0] - DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[0]));
    Em[1] = 0.5*(g[1] - DOT(fem->act_data.elem_act_data[ec].g_e[1], fem->act_data.elem_act_data[ec].g_e[1]));
    Em[2] = g[2] - DOT(fem->act_data.elem_act_data[ec].g_e[0], fem->act_data.elem_act_data[ec].g_e[1]);
    

    if (ConstitutiveLawNonLinear) {
      MembraneNonLinear(ec, X1, X2, X3, Em, S_m, 1, ibm);      
    } else {      
      StressLinear(ec, X1, X2, X3, Em, S_m, 1, ibm);
    }
    
    //---------------------Compute E and S_b for bending ---------------------------------------

    Eb[0] = fem->act_data.elem_act_data[ec].k_e[0] - k[0];
    Eb[1] = fem->act_data.elem_act_data[ec].k_e[1] - k[1];
    Eb[2] = 2*(fem->act_data.elem_act_data[ec].k_e[2] - k[2]);
  
    if (ConstitutiveLawNonLinear) {
      BendingNonLinear(ec, X1, X2, X3, Eb, S_b, 1, ibm);
    } else {
      StressLinear(ec, X1, X2, X3, Eb, S_b, 1, ibm);
    } 
                    
    //--------------------------free----------------------   
    for (i=0; i<(v+6); i++) {
      free(X0[i]);
    }
    
    free(X0);
  }     

  return 0;
}

PetscErrorCode SetGaussianQuadrature(FE *fem)
{
    PetscFunctionBeginUser;

    // Initialize all to zero
    for (PetscInt i = 0; i < 5; i++) {
        fem->act_data.theta[i] = 0.0;
        fem->act_data.w[i] = 0.0;
    }

    switch (num_gaussian_quad_points) {
    case 1:
        fem->act_data.theta[0] = 0.0;
        fem->act_data.w[0] = 2.0;
        break;

    case 2:
        fem->act_data.theta[0] = -0.5773502691896257;  // ±1/sqrt(3)
        fem->act_data.theta[1] =  0.5773502691896257;
        fem->act_data.w[0] = fem->act_data.w[1] = 1.0;
        break;

    case 3:
        fem->act_data.theta[0] = -0.7745966692414834;
        fem->act_data.theta[1] =  0.0;
        fem->act_data.theta[2] =  0.7745966692414834;
        fem->act_data.w[0] = fem->act_data.w[2] = 0.5555555555555556;
        fem->act_data.w[1] = 0.8888888888888888;
        break;

    case 4:
        fem->act_data.theta[0] = -0.8611363115940526;
        fem->act_data.theta[1] = -0.3399810435848563;
        fem->act_data.theta[2] =  0.3399810435848563;
        fem->act_data.theta[3] =  0.8611363115940526;
        fem->act_data.w[0] = fem->act_data.w[3] = 0.3478548451374538;
        fem->act_data.w[1] = fem->act_data.w[2] = 0.6521451548625461;
        break;

    case 5:
        fem->act_data.theta[0] = -0.9061798459386640;
        fem->act_data.theta[1] = -0.5384693101056831;
        fem->act_data.theta[2] =  0.0;
        fem->act_data.theta[3] =  0.5384693101056831;
        fem->act_data.theta[4] =  0.9061798459386640;
        fem->act_data.w[0] = fem->act_data.w[4] = 0.2369268850561891;
        fem->act_data.w[1] = fem->act_data.w[3] = 0.4786286704993665;
        fem->act_data.w[2] = 0.5688888888888889;
        break;

    default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "Number of Gauss points must be between 1 and 5");
    }

    PetscFunctionReturn(0);
}


PetscErrorCode update_mid_surf_elastic_stress(FE* fem){

  PetscFunctionBeginUser;
  
  IBMNodes *ibm = fem->ibm;
  PetscReal S_m[3], S_b[3];
  PetscScalar theta[5], w[5];


  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++){
    compute_mid_surface_el_stress_after_act(fem, ec, S_m, S_b);
    // printf("S_m[0] = %f, S_m[1] = %f, S_m[2] = %f \n", S_m[0], S_m[1], S_m[2]);
    // Put S_m to S_m act_data
    for (PetscInt n = 0; n < num_gaussian_quad_points; n++){
      for (PetscInt i = 0; i < 3; i++){
        
        fem->act_data.elem_act_data[ec].S_m[n][i] = S_m[i];
        fem->act_data.elem_act_data[ec].S_b[n][i] = S_b[i] * (h0 * 0.5 * fem->act_data.theta[n]);
        fem->act_data.elem_act_data[ec].S_e[n][i] = S_m[i] + S_b[i] * (h0 * 0.5 * fem->act_data.theta[n]);
        // if (S_m[i] != 0.0){
        //   printf("S_m[%d][%d][%d] = %f ", ec, n, i, fem->act_data.elem_act_data[ec].S_m[n][i]);
        // }
        
      }
    }
    // printf(" \n ");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CompInitCovBasisVecs(FE *fem){
  // updates fem->act_data.elem_act_data[ec].g_0[i][0]
  PetscFunctionBeginUser;

  IBMNodes *ibm = fem->ibm;
  PetscInt  ec;

  for (ec=0; ec<ibm->n_elmt; ec++) {

    struct Cmpnts a_cov[3];
    // Load covariant basis vectors
    a_cov[0].x = ibm->G1[ec*dof]; a_cov[0].y = ibm->G1[ec*dof+1]; a_cov[0].z = ibm->G1[ec*dof+2];
    a_cov[1].x = ibm->G2[ec*dof]; a_cov[1].y = ibm->G2[ec*dof+1]; a_cov[1].z = ibm->G2[ec*dof+2];    
    a_cov[2].x = UNIT(CROSS(a_cov[0], a_cov[1])).x;
    a_cov[2].y = UNIT(CROSS(a_cov[0], a_cov[1])).y;
    a_cov[2].z = UNIT(CROSS(a_cov[0], a_cov[1])).z;

    PetscInt       i, j, m, p, q, n1e, n2e, n3e;
    struct Cmpnts  dX21, dX31, N, Gc1, Gc2;

    n1e=ibm->nv1[ec];  n2e=ibm->nv2[ec];  n3e=ibm->nv3[ec];    
    if (curvature==6) {

    PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12];
    struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, nn;
    PetscInt       v=ibm->val[ec], node;
    
    Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
    Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
    
    Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
    Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;

    Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
    Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
    
    Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
    Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
    
    Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
    Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;

  if (ibm->ire[ec]==0) {

	for (i=0; i<12; i++) {

	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    x[i] = ibm->x_bp0[node];  y[i] = ibm->y_bp0[node];  z[i] = ibm->z_bp0[node];
	  }
	}
		
		
	for (i=0; i<12; i++) {
	  if (ibm->patch[16*ec+i]==1000000) {
		PetscPrintf(PETSC_COMM_SELF, "Ghost node missing!\n");
		PetscPrintf(PETSC_COMM_SELF, "Element index is%d \n", ec);
		PetscPrintf(PETSC_COMM_SELF, "\n");
	}
	}
	
	Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;

	for (i=0; i<12; i++) {
	  Aaa.x += Nab[0][i]*x[i];
	  Aaa.y += Nab[0][i]*y[i];
	  Aaa.z += Nab[0][i]*z[i];

	  Abb.x += Nab[1][i]*x[i];
	  Abb.y += Nab[1][i]*y[i];
	  Abb.z += Nab[1][i]*z[i];

	  Aab.x += Nab[2][i]*x[i];
	  Aab.y += Nab[2][i]*y[i];
	  Aab.z += Nab[2][i]*z[i];
	}
       
	ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
	ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
	
	for (i=0; i<12; i++) {
	  ndx21.x += Na[0][i]*x[i];
	  ndx21.y += Na[0][i]*y[i];
	  ndx21.z += Na[0][i]*z[i];
	  
	  ndx31.x += Na[1][i]*x[i];
	  ndx31.y += Na[1][i]*y[i];
	  ndx31.z += Na[1][i]*z[i];
	}
	
  // a_3,1 = a_1,1 cross a_2 + a_1 cross a_1,2 
  // a_3,2 = a_1,2 cross a_2 + a_1 cross a_2,2 
  struct Cmpnts a_3_1, a_3_2;


  PetscReal size_a3 = SIZE(CROSS(ndx21, ndx31));
  a_3_1 = PLUS(CROSS(Aaa, ndx31), CROSS(ndx21, Aab));
  a_3_2 = PLUS(CROSS(Aab, ndx31), CROSS(ndx21, Abb));
  
  a_3_1.x = a_3_1.x / size_a3; a_3_1.y = a_3_1.y / size_a3; a_3_1.z = a_3_1.z / size_a3;
  a_3_2.x = a_3_2.x / size_a3; a_3_2.y = a_3_2.y / size_a3; a_3_2.z = a_3_2.z / size_a3;

  for (PetscInt i = 0; i < num_gaussian_quad_points; i++){
    struct Cmpnts b_3_1, b_3_2;
    b_3_1.x = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.x; 
    b_3_2.x = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.x; 
    b_3_1.y = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.y; 
    b_3_2.y = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.y; 
    b_3_1.z = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.z; 
    b_3_2.z = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.z; 

    fem->act_data.elem_act_data[ec].g_0[i][0] = PLUS(a_cov[0], b_3_1);
    fem->act_data.elem_act_data[ec].g_0[i][1] = PLUS(a_cov[1], b_3_2);
    fem->act_data.elem_act_data[ec].g_0[i][2] = a_cov[2]; 

//     printf("ec=%d, i=%d:\n", ec, i);
// printf("g_0[%d][0] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][0].x,
//        fem->act_data.elem_act_data[ec].g_0[i][0].y,
//        fem->act_data.elem_act_data[ec].g_0[i][0].z);

// printf("g_0[%d][1] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][1].x,
//        fem->act_data.elem_act_data[ec].g_0[i][1].y,
//        fem->act_data.elem_act_data[ec].g_0[i][1].z);

// printf("g_0[%d][2] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][2].x,
//        fem->act_data.elem_act_data[ec].g_0[i][2].y,
//        fem->act_data.elem_act_data[ec].g_0[i][2].z);

  }

  }

  else if (ibm->ire[ec]==1) {
	
	for (i=0; i<(v+6); i++) {
	  if (ibm->patch[16*ec+i]==1000000) {
		PetscPrintf(PETSC_COMM_SELF, "Ghost node missing!\n");
		PetscPrintf(PETSC_COMM_SELF, "Element index is%d \n", ec);
                PetscPrintf(PETSC_COMM_SELF, "\n");	
	}
	}

	PetscReal w;
	PetscReal **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));
	for (i=0; i<(v+6); i++) {
	  X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
	}	
			
	for (i=0; i<(v+6); i++) {
	  for (j=0; j<3; j++) {
	    X0[i][j] = 0.;
	  }
	}
	
	for (i=0; i<(v+6); i++) {
	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    X0[i][0] = ibm->x_bp0[node];  X0[i][1] = ibm->y_bp0[node];  X0[i][2] = ibm->z_bp0[node];
	  }
	}

	w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.));
	             			
	PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];	
	//-----B2=dU2/dU1 picking matrix
	for(i=0; i<12; i++) {
	  for (j=0; j<v+12; j++) {
	    B2[i][j] = 0.;
	  }
	}
	
	B2[0][v+9] = 1.;  
	B2[1][v+6] = 1.;  
	B2[2][v+4] = 1.;  
	B2[3][v+1] = 1.;  
	B2[4][v+2] = 1.;  
	B2[5][v+5] = 1.;  
	B2[6][v] = 1.;      
	B2[7][1] = 1.;     
	B2[8][v+3] = 1.; 
	B2[9][v-1] = 1.; 
	B2[10][0] = 1.;     
	B2[11][2] = 1.;     
	
	//------B1=dU1/dU0 Loop's mask
	for(i=0; i<(v+12); i++) {
	  for (j=0; j<(v+6); j++) {
	    B1[i][j] = 0.;
	  }
	}
		
	B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
	B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
	B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
        if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
        if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
        B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
	B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
	B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
	B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
	B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
	B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
	B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
	B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
	B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
	B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
	B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
	B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
	B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
	B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;
	
	for (i=0; i<12; i++) {
	  for (j=0; j<(v+6); j++) {
	    B3[i][j] = 0.;
	    for (m=0; m<(v+12); m++) {
	      B3[i][j] += B2[i][m]*B1[m][j];
	    }
	  }
	}

	PetscReal INa[2][v+6], INab[3][v+6];
	PetscReal sum1, sum2, sum3, sum4, sum5;

	for (j=0; j<(v+6); j++) {
	  sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
	  for (i=0; i<12; i++) {
	    sum1 += B3[i][j]*Na[0][i];
	    sum2 += B3[i][j]*Na[1][i];
	    sum3 += B3[i][j]*Nab[0][i];
	    sum4 += B3[i][j]*Nab[1][i];
	    sum5 += B3[i][j]*Nab[2][i];
	  }
	  INa[0][j] = -2*sum1;
	  INa[1][j] = -2*sum2;
	  INab[0][j] = 4*sum3;
	  INab[1][j] = 4*sum4;
	  INab[2][j] = 4*sum5;	  
	}
      
	Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
	
	for (i=0; i<(v+6); i++) {
	  Aaa.x += INab[0][i]*X0[i][0];
	  Aaa.y += INab[0][i]*X0[i][1];
	  Aaa.z += INab[0][i]*X0[i][2];
	  
	  Abb.x += INab[1][i]*X0[i][0];
	  Abb.y += INab[1][i]*X0[i][1];
	  Abb.z += INab[1][i]*X0[i][2];
	  
	  Aab.x += INab[2][i]*X0[i][0];
	  Aab.y += INab[2][i]*X0[i][1];
	  Aab.z += INab[2][i]*X0[i][2];
	}
	
	ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
	ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
	
	for (i=0; i<(v+6); i++) {
	  ndx21.x += INa[0][i]*X0[i][0];
	  ndx21.y += INa[0][i]*X0[i][1];
	  ndx21.z += INa[0][i]*X0[i][2];
	
	  ndx31.x += INa[1][i]*X0[i][0];
	  ndx31.y += INa[1][i]*X0[i][1];
	  ndx31.z += INa[1][i]*X0[i][2];
	}

  // a_3,1 = a_1,1 cross a_2 + a_1 cross a_1,2 
  // a_3,2 = a_1,2 cross a_2 + a_1 cross a_2,2 
  struct Cmpnts a_3_1, a_3_2;
  
  PetscReal size_a3 = SIZE(CROSS(ndx21, ndx31));
  a_3_1 = PLUS(CROSS(Aaa, ndx31), CROSS(ndx21, Aab));
  a_3_2 = PLUS(CROSS(Aab, ndx31), CROSS(ndx21, Abb));
  
  a_3_1.x = a_3_1.x / size_a3; a_3_1.y = a_3_1.y / size_a3; a_3_1.z = a_3_1.z / size_a3;
  a_3_2.x = a_3_2.x / size_a3; a_3_2.y = a_3_2.y / size_a3; a_3_2.z = a_3_2.z / size_a3;

  for (PetscInt i = 0; i < num_gaussian_quad_points; i++){
    struct Cmpnts b_3_1, b_3_2;
    b_3_1.x = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.x; 
    b_3_2.x = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.x; 
    b_3_1.y = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.y; 
    b_3_2.y = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.y; 
    b_3_1.z = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_1.z; 
    b_3_2.z = (h0 * 0.5 * fem->act_data.theta[i]) * a_3_2.z; 

    fem->act_data.elem_act_data[ec].g_0[i][0] = PLUS(a_cov[0], b_3_1);
    fem->act_data.elem_act_data[ec].g_0[i][1] = PLUS(a_cov[1], b_3_2);
    fem->act_data.elem_act_data[ec].g_0[i][2] = a_cov[2]; 

//     printf("ec=%d, i=%d:\n", ec, i);
// printf("g_0[%d][0] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][0].x,
//        fem->act_data.elem_act_data[ec].g_0[i][0].y,
//        fem->act_data.elem_act_data[ec].g_0[i][0].z);

// printf("g_0[%d][1] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][1].x,
//        fem->act_data.elem_act_data[ec].g_0[i][1].y,
//        fem->act_data.elem_act_data[ec].g_0[i][1].z);

// printf("g_0[%d][2] = (%f, %f, %f)\n", i,
//        fem->act_data.elem_act_data[ec].g_0[i][2].x,
//        fem->act_data.elem_act_data[ec].g_0[i][2].y,
//        fem->act_data.elem_act_data[ec].g_0[i][2].z);

  }
	
	// nn=UNIT(CROSS(ndx21, ndx31));
	      	
	// k0[0] = DOT(Aaa,nn);
	// k0[1] = DOT(Abb,nn);
	// k0[2] = DOT(Aab,nn);
	
	// ibm->kve0[ec*dof] = k0[0];
	// ibm->kve0[ec*dof+1] = k0[1];
	// ibm->kve0[ec*dof+2] = k0[2];

	
	for (i=0; i<(v+6); i++) {
	  free(X0[i]);
	}
	free(X0); 
	
      }     
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode compute_Fbar_el(FE *fem, PetscInt ec){

  PetscFunctionBeginUser;

  IBMNodes *ibm = fem->ibm;
  
  PetscInt i, j, k;

  for (PetscInt n = 0; n < num_gaussian_quad_points; n++){

    struct Cmpnts g_cov[3], g_contra[3], tmp;
    PetscReal G[3][3], Ginv[3][3], GT[3][3];
    PetscReal Fa[3][3], Finv[3][3]; 
    PetscReal Fcontravariant[3][3], Fbar[3][3];
    PetscReal temp[3][3];
  

    // ---- (1) Load covariant basis vectors
    g_cov[0] = fem->act_data.elem_act_data[ec].g_0[n][0]; 
    g_cov[1] = fem->act_data.elem_act_data[ec].g_0[n][1]; 
    g_cov[2] = fem->act_data.elem_act_data[ec].g_0[n][2]; 

    // ---- (2) Form G matrix (covariant basis as columns) ----
    G[0][0] = g_cov[0].x; G[1][0] = g_cov[0].y; G[2][0] = g_cov[0].z;
    G[0][1] = g_cov[1].x; G[1][1] = g_cov[1].y; G[2][1] = g_cov[1].z;
    G[0][2] = g_cov[2].x; G[1][2] = g_cov[2].y; G[2][2] = g_cov[2].z;

    // ---- (3) Compute contravariant basis vectors ----
    INV(G, Ginv);
    for (i = 0; i < 3; i++) {
        g_contra[i].x = Ginv[0][i];
        g_contra[i].y = Ginv[1][i];
        g_contra[i].z = Ginv[2][i];
    }

      // ---- (4) Get Fa in Cartesian ----
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            Fa[i][j] = fem->act_data.elem_act_data[ec].Fa[i][j];
    

    // Compute inverse of Fa
    INV(Fa, Finv);

    // ---- Compute F_pq = G^T * Finv * G ----

    // temp = Finv * G
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            temp[i][j] = 0.0;
            for (k=0; k<3; k++) temp[i][j] += Finv[i][k] * G[k][j];
        }
    }

    TRANS(G, GT);

    // Fbar = G^T * temp
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            Fbar[i][j] = 0.0;
            for (k=0; k<3; k++) Fbar[i][j] += GT[i][k] * temp[k][j];
        }
    }

    for (i = 0; i < 3; i++){
      for (j = 0; j < 3; j++){
        fem->act_data.elem_act_data[ec].Fbar[n][i][j] = Fbar[i][j];
        // printf(" Fbar[i][j] = %f \n", Fbar[i][j]);
      }
    }             
      // Fbar
  // PetscPrintf(PETSC_COMM_SELF, "Fbar (inverse) n = %d:\n", n);
  // for (i = 0; i < 3; i++)
  //     PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n",
  //         Fbar[i][0], Fbar[i][1], Fbar[i][2]);


  }

  

  

  

  
  // Print result (optional)
  // PetscPrintf(PETSC_COMM_SELF, "F_pq (Fa^-1 components):\n");
  // for (i=0; i<3; i++) {
  //     PetscPrintf(PETSC_COMM_SELF, "[%e %e %e]\n", Fbar[i][0], Fbar[i][1], Fbar[i][2]);
  // }

  // ---- (5) Compute Fcontravariant = g^i · (Fa * g^j) ----
  // for (i = 0; i < 3; i++) {
  //     for (j = 0; j < 3; j++) {
  //         PetscReal tmpv[3];
  //         // tmpv = Fa * g^j (g^j is column j of G)
  //         tmpv[0] = Fa[0][0]*G[0][j] + Fa[0][1]*G[1][j] + Fa[0][2]*G[2][j];
  //         tmpv[1] = Fa[1][0]*G[0][j] + Fa[1][1]*G[1][j] + Fa[1][2]*G[2][j];
  //         tmpv[2] = Fa[2][0]*G[0][j] + Fa[2][1]*G[1][j] + Fa[2][2]*G[2][j];

  //         // Fcontravariant[i][j] = g^i ⋅ tmpv
  //         Fcontravariant[i][j] = g_contra[i].x*tmpv[0] +
  //                                g_contra[i].y*tmpv[1] +
  //                                g_contra[i].z*tmpv[2];
  //     }
  // }

  // ---- (6) Invert Fcontravariant to get Fbar ----
  // INV(Fcontravariant, Fbar);

  
  // PetscPrintf(PETSC_COMM_SELF, "\n==== Element %d ====\n", ec);

  // // Covariant basis
  // for (i = 0; i < 3; i++) {
  //     PetscPrintf(PETSC_COMM_SELF, "g_cov[%d] = [%g %g %g]\n",
  //         i, g_cov[i].x, g_cov[i].y, g_cov[i].z);
  // }

  // // Contravariant basis
  // for (i = 0; i < 3; i++) {
  //     PetscPrintf(PETSC_COMM_SELF, "g_contra[%d] = [%g %g %g]\n",
  //         i, g_contra[i].x, g_contra[i].y, g_contra[i].z);
  // }

  // // G and Ginv
  // PetscPrintf(PETSC_COMM_SELF, "G matrix:\n");
  // for (i = 0; i < 3; i++)
  //     PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", G[i][0], G[i][1], G[i][2]);

  // PetscPrintf(PETSC_COMM_SELF, "Ginv matrix:\n");
  // for (i = 0; i < 3; i++)
  //     PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", Ginv[i][0], Ginv[i][1], Ginv[i][2]);

  // // Fa (Cartesian)
  // PetscPrintf(PETSC_COMM_SELF, "Fa (Cartesian):\n");
  // for (i = 0; i < 3; i++)
  //     PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n",
  //         Fa[i][0], Fa[i][1], Fa[i][2]);

  // // Fcontravariant
  // PetscPrintf(PETSC_COMM_SELF, "Fcontravariant (F_a^{ij}):\n");
  // for (i = 0; i < 3; i++)
  //     PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n",
  //         Fcontravariant[i][0], Fcontravariant[i][1], Fcontravariant[i][2]);



  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UpdateFbar(FE *fem){

  PetscFunctionBeginUser;
  
  IBMNodes *ibm = fem->ibm;
  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++){
    compute_Fbar_el(fem, ec);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode cal_total_stress_act_el(FE* fem, PetscInt ec){

  PetscFunctionBeginUser;
  IBMNodes *ibm = fem->ibm;

  for (PetscInt n = 0; n < num_gaussian_quad_points; n++){

    PetscReal Fa[3][3], G[3][3], Ginv[3][3];
    PetscReal Fcontravariant[3][3], Fbar[3][3];
    struct Cmpnts g_cov[3], g_contra[3], tmp;
    PetscReal g_cont[3][3]; 
    PetscInt i, j, k;
    PetscInt w, p, z, s;
    PetscReal S[3][3], S_e[3][3];

     // ---- (1) Load covariant basis vectors
    g_cov[0] = fem->act_data.elem_act_data[ec].g_0[n][0]; 
    g_cov[1] = fem->act_data.elem_act_data[ec].g_0[n][1]; 
    g_cov[2] = fem->act_data.elem_act_data[ec].g_0[n][2]; 

    // ---- (2) Form G matrix (covariant basis as columns) ----
    G[0][0] = g_cov[0].x; G[1][0] = g_cov[0].y; G[2][0] = g_cov[0].z;
    G[0][1] = g_cov[1].x; G[1][1] = g_cov[1].y; G[2][1] = g_cov[1].z;
    G[0][2] = g_cov[2].x; G[1][2] = g_cov[2].y; G[2][2] = g_cov[2].z;

    // ---- (3) Compute contravariant basis vectors ----
    INV(G, Ginv);
    for (i = 0; i < 3; i++) {
        g_contra[i].x = Ginv[0][i];
        g_contra[i].y = Ginv[1][i];
        g_contra[i].z = Ginv[2][i];
    }

    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        g_cont[i][j] = DOT(g_contra[i], g_contra[j]);
      }
    }

    for (i = 0; i < 3; i++){
      for (j = 0; j < 3; j++){
        Fbar[i][j] = fem->act_data.elem_act_data[ec].Fbar[n][i][j];
        S_e[i][j] = 0.0;          
      }
    }  
    S_e[0][0] = fem->act_data.elem_act_data[ec].S_e[n][0];
    S_e[0][1] = fem->act_data.elem_act_data[ec].S_e[n][2];
    S_e[1][0] = fem->act_data.elem_act_data[ec].S_e[n][2];
    S_e[1][1] = fem->act_data.elem_act_data[ec].S_e[n][1];


    for (i = 0; i < 3; i++){
      for (j = 0; j < 3; j++){
        S[i][j] = 0;
        for (w = 0; w < 3; w++){
          for (p = 0; p < 3; p++){
            for (z = 0; z < 3; z++){
              for (s = 0; s < 3; s++){
                S[i][j] += 0.5*Fbar[w][p]*Fbar[z][s]*g_cont[s][j]*(S_e[p][z]*g_cont[w][i] + S_e[p][i]*g_cont[w][z]);      
              }
            }
          }
        }      
        // printf("S[i][j] = %f, ", S[i][j]);
      }
      // printf("\n");
    }
    fem->act_data.elem_act_data[ec].S[n][0] = S[0][0];
    fem->act_data.elem_act_data[ec].S[n][2] = S[0][1];
    // fem->act_data.elem_act_data[ec].S[n][2] = S[1][0];
    fem->act_data.elem_act_data[ec].S[n][1] = S[1][1];


    /* ---- (7) Print everything for verification ---- */
  // if (ec == 50){
  //   PetscPrintf(PETSC_COMM_SELF, "\n=== Element %d : cal_total_stress_act_el debug ===\n", (int)ec);

  // PetscPrintf(PETSC_COMM_SELF, "Covariant basis g_cov (columns of G):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "g_cov[%d] = [%g %g %g]\n", i, g_cov[i].x, g_cov[i].y, g_cov[i].z);

  // PetscPrintf(PETSC_COMM_SELF, "G matrix (covariant basis as columns):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", G[i][0], G[i][1], G[i][2]);

  // PetscPrintf(PETSC_COMM_SELF, "Ginv matrix (columns are g^i):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", Ginv[i][0], Ginv[i][1], Ginv[i][2]);

  // PetscPrintf(PETSC_COMM_SELF, "Contravariant basis g_contra (columns of Ginv):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "g_contra[%d] = [%g %g %g]\n", i, g_contra[i].x, g_contra[i].y, g_contra[i].z);

  // PetscPrintf(PETSC_COMM_SELF, "g_cont (g^i · g^j):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", g_cont[i][0], g_cont[i][1], g_cont[i][2]);

  // PetscPrintf(PETSC_COMM_SELF, "Fbar (from fem->act_data.elem_act_data[ec].Fbar):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", Fbar[i][0], Fbar[i][1], Fbar[i][2]);

  // PetscPrintf(PETSC_COMM_SELF, "S_e (membrane 2nd-P-K in your storage mapping):\n");
  // for (i = 0; i < 3; ++i)
  //   PetscPrintf(PETSC_COMM_SELF, "[%g %g %g]\n", S_e[i][0], S_e[i][1], S_e[i][2]);


  //   PetscPrintf(PETSC_COMM_SELF, "Computed total S_ij, n = %d (result of eq 34 loop):\n", n);
  //   for (i = 0; i < 3; ++i) {
  //     for (j = 0; j < 3; ++j) PetscPrintf(PETSC_COMM_SELF, "%14.8e ", S[i][j]);
  //     PetscPrintf(PETSC_COMM_SELF, "\n");
  //   }
  // }

  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode UpdateTotalStressAftAct(FE* fem){
  // Updates fem->act_data.elem_act_data[ec].S[n][i]
  PetscFunctionBeginUser;
  
  IBMNodes *ibm = fem->ibm;

  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++){
    cal_total_stress_act_el(fem, ec);
  }

  PetscFunctionReturn(PETSC_SUCCESS);

}

PetscErrorCode FInternalAftActEl(FE* fem, PetscInt ec, PetscReal _Fb[42]){
  
  PetscFunctionBeginUser;

  IBMNodes       *ibm=fem->ibm;    
  PetscReal      sum, sum1, sum2;
  PetscInt       i, j, m, p, q;
  struct Cmpnts  dx21, dx31, n, gc1, gc2;
  struct Cmpnts  a4, a5, a6;
  struct Cmpnts  x4, x5, x6;
  PetscReal      Eb[3], S[3];

  PetscInt       n1e, n2e, n3e, n4e, n5e, n6e;
  struct Cmpnts  x1, x2, x3, X1, X2, X3;

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

 
  //--------------------Compute kv:curvature at current----------------------------  
  PetscReal      ze[3], k[3], A, A0, kcur[3]; 

  A=ibm->dA[ec]; A0=ibm->dA0[ec];    

  if (curvature==6) { //subdivision surface
    //PetscPrintf(PETSC_COMM_SELF, "Check IF in curva!\n");
    PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12], b1, b2, b3, nA0, nA;
    struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, ndX21, ndX31, nn, ng1, ng2, ng;
    PetscInt       v=ibm->val[ec], node, nob=1;
    
    Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
    Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
    
    Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
    Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;
       
    Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
    Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
    
    Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
    Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
      
    Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
    Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;
    //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch outside!\n");
    if (ibm->ire[ec]==0) { //-------------------------------------regular patch-----------------------------------------
    //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch inside!\n");
      for (i=0; i<12; i++) {
        if (ibm->patch[16*ec+i]!=1000000) {
          node = ibm->patch[16*ec+i];
          x[i] = ibm->x_bp[node];  y[i] = ibm->y_bp[node];  z[i] = ibm->z_bp[node];
        }
      }
	
      for (i=0; i<12; i++) {
        if (ibm->patch[16*ec+i]==1000000) {nob=0;}
      }

      Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
      Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
      Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
      
      for (i=0; i<12; i++) {
        Aaa.x += Nab[0][i]*x[i];
        Aaa.y += Nab[0][i]*y[i];
        Aaa.z += Nab[0][i]*z[i];
        
        Abb.x += Nab[1][i]*x[i];
        Abb.y += Nab[1][i]*y[i];
        Abb.z += Nab[1][i]*z[i];
        
        Aab.x += Nab[2][i]*x[i];
        Aab.y += Nab[2][i]*y[i];
        Aab.z += Nab[2][i]*z[i];
      }
       
      ndx21.x=0.; ndx21.y=0.; ndx21.z=0.;
      ndx31.x=0.; ndx31.y=0.; ndx31.z=0.;
      
      for (i=0; i<12; i++) {
        ndx21.x += Na[0][i]*x[i];
        ndx21.y += Na[0][i]*y[i];
        ndx21.z += Na[0][i]*z[i];
        
        ndx31.x += Na[1][i]*x[i];
        ndx31.y += Na[1][i]*y[i];
        ndx31.z += Na[1][i]*z[i];
      }
     
      nn=UNIT(CROSS(ndx21, ndx31));
      ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
      ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];
      nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
      nA = 0.5*SIZE(CROSS(ndx21, ndx31));
 
      gc1=CROSS(ndx31, nn);
      gc1=AMULT(1./DOT(ndx21, gc1), gc1);
      
      gc2=CROSS(nn, ndx21);
      gc2=AMULT(1./DOT(ndx31, gc2), gc2);

      ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
      ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
      ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
    
      //-----------------------------------Forming membrane matrix------------------------------------------
      PetscReal  Bm[3][36];

      for (i=0; i<12; i++) {
      	Bm[0][3*i] = Na[0][i]*ndx21.x;   Bm[0][3*i+1] = Na[0][i]*ndx21.y;   Bm[0][3*i+2] = Na[0][i]*ndx21.z;
      	Bm[1][3*i] = Na[1][i]*ndx31.x;   Bm[1][3*i+1] = Na[1][i]*ndx31.y;   Bm[1][3*i+2] = Na[1][i]*ndx31.z;
      	Bm[2][3*i] = Na[0][i]*ndx31.x + Na[1][i]*ndx21.x;   Bm[2][3*i+1] = Na[0][i]*ndx31.y + Na[1][i]*ndx21.y;   Bm[2][3*i+2] = Na[0][i]*ndx31.z + Na[1][i]*ndx21.z;
      }

      //-----------------------------------Forming bending  matrix------------------------------------------
      PetscReal  Bs[3][36];

      for (i=0; i<12; i++) {
      	ng1 = AMULT(Na[0][i], gc1);   ng2 = AMULT(Na[1][i], gc2);
      	ng = PLUS(ng1, ng2);
      	b1 = DOT(ng, Aaa);   b2 = DOT(ng, Abb);   b3 = DOT(ng, Aab);

      	Bs[0][3*i] = -(Nab[0][i] + b1)*nn.x;   Bs[0][3*i+1] = -(Nab[0][i] + b1)*nn.y;   Bs[0][3*i+2] = -(Nab[0][i] + b1)*nn.z;
      	Bs[1][3*i] = -(Nab[1][i] + b2)*nn.x;   Bs[1][3*i+1] = -(Nab[1][i] + b2)*nn.y;   Bs[1][3*i+2] = -(Nab[1][i] + b2)*nn.z;
      	Bs[2][3*i] = -2*(Nab[2][i] + b3)*nn.x;   Bs[2][3*i+1] = -2*(Nab[2][i] + b3)*nn.y;   Bs[2][3*i+2] = -2*(Nab[2][i] + b3)*nn.z;
      }

    //--------------------------------------calculating membrane force-----------------------------
    PetscReal  _Fm[36];      
    for (PetscInt n = 0; n < num_gaussian_quad_points; n++){

      // Read Total Stress
      for (PetscInt i = 0; i < 3; i++){
        S[i] = fem->act_data.elem_act_data[ec].S[n][i];  
      }

      for (i=0; i<36; i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){

          sum += Bm[m][i]*S[m]*A0*h0*0.5*fem->act_data.w[n];
          // printf("Bm[m][i] = %f, S[m] = %f \n", Bm[m][i], S[m]);
      	  // sum += Bm[m][i]*Sm[m]*A0*h0; 
      	}
      	_Fm[i] += sum;	
      }

        //--------------------------------------calculating bending force-----------------------------
      for (i=0; i<36; i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
      	  // sum += Bs[m][i]*S[m]*A0*pow(h0,3.)/12.;
          sum += Bs[m][i] * S[m] * A0 * h0/2. * fem->act_data.w[n] * (h0/2. * fem->act_data.theta[n]);
          // printf("Bs[m][i] = %f \n", Bs[m][i]);

      	}
      	_Fb[i] += sum + _Fm[i];
      }      

    }
    

    }

    else if (ibm->ire[ec]==1) { //----------------------irregular patch---------------------------
      //PetscPrintf(PETSC_COMM_SELF, "Check if irregular patch inside!\n");
      for (i=0; i<(v+6); i++) {
        if (ibm->patch[16*ec+i]==1000000) {nob=0;}
      }
      
      PetscReal  w;	
      PetscReal  **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));
      for (i=0; i<(v+6); i++) {
        X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
      }      
      
      for (i=0; i<(v+6); i++) {
        for (j=0; j<3; j++) {
          X0[i][j] = 0.;
        }
      }     
      
      for (i=0; i<(v+6); i++) {
        if (ibm->patch[16*ec+i]!=1000000) {
          node = ibm->patch[16*ec+i];
          X0[i][0] = ibm->x_bp[node];  X0[i][1] = ibm->y_bp[node];  X0[i][2] = ibm->z_bp[node];
        }
      }
      
      w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.)); 
            
      PetscReal  B3[12][v+6], B2[12][v+12], B1[v+12][v+6];
      
      //-----B2=dU2/dU1
      for(i=0; i<12; i++) {
        for (j=0; j<(v+12); j++) {
          B2[i][j] = 0.;
        }
      }
      
      //------B1=dU1/dU0
      for(i=0; i<(v+12); i++) {
        for (j=0; j<(v+6); j++) {
          B1[i][j] = 0.;
        }
      }
      
      B2[0][v+9] = 1.;  
      B2[1][v+6] = 1.;  
      B2[2][v+4] = 1.;  
      B2[3][v+1] = 1.;  
      B2[4][v+2] = 1.;  
      B2[5][v+5] = 1.;  
      B2[6][v] = 1.;      
      B2[7][1] = 1.;     
      B2[8][v+3] = 1.; 
      B2[9][v-1] = 1.; 
      B2[10][0] = 1.;     
      B2[11][2] = 1.;     
                
      B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
      B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
      B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
      if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
      if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
      B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
      B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
      B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
      B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
      B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
      B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
      B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
      B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
      B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
      B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
      B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
      B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
      B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
      B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;    
           
      for (i=0; i<12; i++) {
        for (j=0; j<(v+6); j++) {
          B3[i][j] = 0.;
          for (m=0; m<(v+12); m++) {
            B3[i][j] += B2[i][m]*B1[m][j];
          }
        }
      }
                   	
      PetscReal  INa[2][v+6], INab[3][v+6];
      PetscReal  sum1, sum2, sum3, sum4, sum5;
      
      for (j=0; j<(v+6); j++) {
        sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
        for (i=0; i<12; i++) {
          sum1 += B3[i][j]*Na[0][i];
          sum2 += B3[i][j]*Na[1][i];
          sum3 += B3[i][j]*Nab[0][i];
          sum4 += B3[i][j]*Nab[1][i];
          sum5 += B3[i][j]*Nab[2][i];
        }
        INa[0][j] = -2*sum1;
        INa[1][j] = -2*sum2;
        INab[0][j] = 4*sum3;
        INab[1][j] = 4*sum4;
        INab[2][j] = 4*sum5;
      }     

      Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
      Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
      Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
	
      for (i=0; i<(v+6); i++) {
        Aaa.x += INab[0][i]*X0[i][0];
        Aaa.y += INab[0][i]*X0[i][1];
        Aaa.z += INab[0][i]*X0[i][2];
        
        Abb.x += INab[1][i]*X0[i][0];
        Abb.y += INab[1][i]*X0[i][1];
        Abb.z += INab[1][i]*X0[i][2];
        
        Aab.x += INab[2][i]*X0[i][0];
        Aab.y += INab[2][i]*X0[i][1];
        Aab.z += INab[2][i]*X0[i][2];
      }
	     
      ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
      ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
      
      for (i=0; i<(v+6); i++) {
        ndx21.x += INa[0][i]*X0[i][0];
        ndx21.y += INa[0][i]*X0[i][1];
        ndx21.z += INa[0][i]*X0[i][2];
        
        ndx31.x += INa[1][i]*X0[i][0];
        ndx31.y += INa[1][i]*X0[i][1];
        ndx31.z += INa[1][i]*X0[i][2];
      }

      nn=UNIT(CROSS(ndx21,ndx31));
      ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
      ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];

      dx21.x = X0[1][0] - X0[0][0]; dx21.y = X0[1][1] - X0[0][1]; dx21.z = X0[1][2] - X0[0][2];
      dx31.x = X0[v][0] - X0[0][0]; dx31.y = X0[v][1] - X0[0][1]; dx31.z = X0[v][2] - X0[0][2];

      nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
      nA = 0.5*SIZE(CROSS(ndx21, ndx31));     


      gc1=CROSS(ndx31, nn);
      gc1=AMULT(1./DOT(ndx21, gc1), gc1);
      
      gc2=CROSS(nn, ndx21);
      gc2=AMULT(1./DOT(ndx31, gc2), gc2); 
      
      ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
      ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
      ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
     
      
      //-----------------------------------Forming membrane matrix------------------------------
      PetscReal  Bm[3][3*(v+6)];
      for (i=0; i<(v+6); i++) {
	
      	Bm[0][3*i] = INa[0][i]*ndx21.x;   Bm[0][3*i+1] = INa[0][i]*ndx21.y;   Bm[0][3*i+2] = INa[0][i]*ndx21.z;
      	Bm[1][3*i] = INa[1][i]*ndx31.x;   Bm[1][3*i+1] = INa[1][i]*ndx31.y;   Bm[1][3*i+2] = INa[1][i]*ndx31.z;
      	Bm[2][3*i] = INa[0][i]*ndx31.x + INa[1][i]*ndx21.x;   Bm[2][3*i+1] = INa[0][i]*ndx31.y + INa[1][i]*ndx21.y;   Bm[2][3*i+2] = INa[0][i]*ndx31.z + INa[1][i]*ndx21.z;
      }
      //-----------------------------------Forming bending matrix------------------------------------------
      PetscReal  B4[3][3*(v+6)];
      
      for (i=0; i<(v+6); i++) {
      	ng1 = AMULT(INa[0][i], gc1);   ng2 = AMULT(INa[1][i], gc2);
      	ng = PLUS(ng1, ng2);
      	b1 = DOT(ng, Aaa);   b2 = DOT(ng, Abb);   b3 = DOT(ng, Aab);
	
      	B4[0][3*i] = -(INab[0][i] + b1)*nn.x;   B4[0][3*i+1] = -(INab[0][i] + b1)*nn.y;   B4[0][3*i+2] = -(INab[0][i] + b1)*nn.z;
      	B4[1][3*i] = -(INab[1][i] + b2)*nn.x;   B4[1][3*i+1] = -(INab[1][i] + b2)*nn.y;   B4[1][3*i+2] = -(INab[1][i] + b2)*nn.z;
      	B4[2][3*i] = -2*(INab[2][i] + b3)*nn.x;   B4[2][3*i+1] = -2*(INab[2][i] + b3)*nn.y;   B4[2][3*i+2] = -2*(INab[2][i] + b3)*nn.z;
      }
             
      /* //--------------------------------------calculating membrane force----------------------------- */
     
      PetscReal  _Fm[42];
    for (PetscInt n = 0; n < num_gaussian_quad_points; n++){

      // Read Total Stress
      for (PetscInt i = 0; i < 3; i++){
        S[i] = fem->act_data.elem_act_data[ec].S[n][i];  
      }

      for (i=0; i<3*(v+6); i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
          sum += Bm[m][i]*S[m]*A0*h0*0.5*fem->act_data.w[n];

      	  // sum += Bm[m][i]*S[m]*A0*h0;
      	}
      	_Fm[i] += sum;  	
      }
         
      //--------------------------------------calculating bending force-----------------------------
      for (i=0; i<3*(v+6); i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
          sum += B4[m][i] * S[m] * A0 * h0/2. * fem->act_data.w[n] * (h0/2. * fem->act_data.theta[n]);
      	  // sum += B4[m][i]*S[m]*A0*pow(h0,3.)/12.;
          // printf("B4[m][i] = %f \n", B4[m][i]);
      	}
      	_Fb[i] += sum + _Fm[i];
      }
    }
          
    
      //--------------------------free----------------------   
      for (i=0; i<(v+6); i++) {
        free(X0[i]);
      }
     
      free(X0);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);

}

PetscErrorCode UpdateFInternalAftAct(FE* fem){

  PetscFunctionBeginUser;
  IBMNodes *ibm = fem->ibm;

  update_mid_surf_elastic_stress(fem);
  UpdateTotalStressAftAct(fem);

  PetscInt       i;
  PetscReal Fb[42];

  for (i=0; i<42; i++) {Fb[i]=0.0;}

  PetscReal  *FF,*FFJ;
  VecGetArray(fem->Fint, &FF);

  for (PetscInt ec = 0; ec < ibm->n_elmt; ec++){

    FInternalAftActEl(fem, ec, Fb);
    if (curvature==6) {
          
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
  }

  VecRestoreArray(fem->Fint, &FF); 

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode InitMuscleActProblem(FE *fem){
  // LOG("update_user_act_params");
  // PetscCall(update_user_act_params(fem));
  // update_act_Fa(fem);
  // update_intmd_state_cov_basis(fem);
  // // Output(fem, ti+1, ibi, out_dir);

  // // PetscViewer viewerF, viewerDiff, viewerTarget;
  // // PetscViewerASCIIOpen(PETSC_COMM_SELF, "g_e_targetttt.txt", &viewerTarget);
  // // VecView(fem[ibi].act_data.g_e_target, viewerTarget);
  // // PetscViewerDestroy(&viewerTarget);

  // find_intmd_coords(fem);
  // update_intmd_ibm_coords(fem);      
  // // Output(fem, ti+2, ibi, out_dir);
  

  // SetGaussianQuadrature(fem);
  // printf("a SetGaussianQuadrature\n");
  // CompInitCovBasisVecs(fem);
  // printf("a CompInitCovBasisVecs\n");
  // UpdateFbar(fem);
  // printf("a UpdateFbar\n");
  
  // // printf("JUST FOR TEST!!\n");
  // // Input(&ibm[ibi], ibi);
  // // printf("JUST FOR TEST!!\n");

  // initialize_elasticity(fem, initial_elas, initial_poisson);
  // printf("a initialize_elasticity\n");      

  PetscFunctionBeginUser;

    LOG("update_user_act_params");
    PetscCall(update_user_act_params(fem));

    LOG("update_act_Fa");
    PetscCall(update_act_Fa(fem));

    LOG("update_intmd_state_cov_basis");
    PetscCall(update_intmd_state_cov_basis(fem));

    LOG("find_intmd_coords");
    PetscCall(find_intmd_coords(fem));

    LOG("update_intmd_ibm_coords");
    PetscCall(update_intmd_ibm_coords(fem));

    LOG("SetGaussianQuadrature");
    PetscCall(SetGaussianQuadrature(fem));

    LOG("CompInitCovBasisVecs");
    PetscCall(CompInitCovBasisVecs(fem));

    LOG("UpdateFbar");
    PetscCall(UpdateFbar(fem));

    LOG("initialize_elasticity");
    PetscCall(initialize_elasticity(fem, initial_elas, initial_poisson));

    LOG("done");
    PetscFunctionReturn(0);
}