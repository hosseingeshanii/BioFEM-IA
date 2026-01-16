/**
 * @file active_strain.c
 * @brief Implementation of muscle activation on shell elements using the active-strain approach.
 *
 * References:
 *  - Nitti, A., et al. (2021). A curvilinear isogeometric framework for the purely elastic response
 *    of shell structures undergoing active strain patterns. *Computers & Structures*, 248, 106533.
 *
 *  - Torre, M., et al. (2024). An efficient active-stress electromechanical isogeometric formulation
 *    for Kirchhoff–Love shells. *Computers & Structures*, 276, 106879.
 *
 * @author
 *   Hossein Geshani
 * @date
 *   2025-09-22
 */

/*--------------------------------------------------------------------------------------------------
 *                                        Includes
 *-------------------------------------------------------------------------------------------------*/

#include <math.h>
#include <petsctao.h>
#include <petscviewer.h>

#include "variables.h" /* Project-local header */

/*--------------------------------------------------------------------------------------------------
 *                                        Typedefs & Macros
 *-------------------------------------------------------------------------------------------------*/

typedef PetscErrorCode (*ElemFunc)(FE *fem, PetscInt ec);

#define LOG(msg) PetscPrintf(PETSC_COMM_WORLD, "[%s] %s\n", __func__, msg)

/*--------------------------------------------------------------------------------------------------
 *                                        Global Variables
 *-------------------------------------------------------------------------------------------------*/

/* Activation-related coefficients */
PetscInt fiber_based_act_coeff;
PetscInt curv_based_act_coeffs;
PetscInt cart_fib_act;

/* External globals from the rest of the code */
extern PetscInt dof, curvature, ConstitutiveLawNonLinear;
extern PetscReal E, mu, rho, h0;
extern PetscReal initial_elas, initial_poisson;

/* Shape function derivatives at element center (subdivision surface) */
static const PetscReal Na_center[2][12] = {
  /* Na[0][i] */
  {-0.0247, -0.0309,  0.0000, -0.4815, -0.1852,  0.0247,
    0.4815,  0.0000, -0.0062,  0.0309,  0.1852,  0.0062},
  /* Na[1][i] */
  {-0.0309, -0.0247, -0.1852, -0.4815,  0.0000, -0.0062,
    0.0000,  0.4815,  0.0247,  0.0062,  0.1852,  0.0309}
};

static const PetscReal Nab_center[3][12] = {
  /* Nab[0][i] */
  { 0.1111,  0.2222, -0.2222, -0.2222,  0.4444,  0.1111,
   -0.2222, -0.8889,  0.0000,  0.2222,  0.4444,  0.0000},
  /* Nab[1][i] */
  { 0.2222,  0.1111,  0.4444, -0.2222, -0.2222,  0.0000,
   -0.8889, -0.2222,  0.1111,  0.0000,  0.4444,  0.2222},
  /* Nab[2][i] */
  { 0.1667,  0.1667, -0.1111,  0.2222, -0.1111, -0.0556,
   -0.4444, -0.4444, -0.0556,  0.0556,  0.5556,  0.0556}
};


PetscErrorCode ActDataAllocate(FE *fem)
{
    PetscErrorCode ierr;
    ActData *act = &fem->act_data;

    PetscInt nelem = fem->ibm->n_elmt;
    PetscInt n_qp = act->n_qp;

    /*------------------------------------------------------------*/
    /* Allocate element-level activation data                     */
    /*------------------------------------------------------------*/
    ierr = PetscMalloc1(nelem, &act->elem_act_data);
    CHKERRQ(ierr);

    for (PetscInt ec = 0; ec < nelem; ec++)
    {
        ElemActData *ead = &act->elem_act_data[ec];

        /* Kinematics */
        ierr = PetscMalloc1(n_qp, &ead->Fa);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->Fa_inv);
        CHKERRQ(ierr);

        ierr = PetscMalloc1(n_qp, &ead->C);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->C_inv);
        CHKERRQ(ierr);

        ierr = PetscMalloc1(n_qp, &ead->Ce);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->Ce_inv);
        CHKERRQ(ierr);

        /* Stress */
        ierr = PetscMalloc1(n_qp, &ead->Se);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->S);
        CHKERRQ(ierr);

        /* Metrics */
        ierr = PetscMalloc1(n_qp, &ead->gm);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->gm0);
        CHKERRQ(ierr);

        /* Tangents */
        ierr = PetscMalloc1(n_qp, &ead->CCe);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->CC);
        CHKERRQ(ierr);

        /* Bases */
        ierr = PetscMalloc1(n_qp, &ead->g);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->g0);
        CHKERRQ(ierr);

        /* NEW: geometry cache */
        ierr = PetscMalloc1(n_qp, &ead->geom); CHKERRQ(ierr);
        for (PetscInt qp = 0; qp < n_qp; qp++) {
        ead->geom[qp].nen = 0;
        ead->geom[qp].v = 0;
        ead->geom[qp].is_irregular = 0;
        ead->geom[qp].INa0  = NULL;
        ead->geom[qp].INa1  = NULL;
        ead->geom[qp].INab0 = NULL;
        ead->geom[qp].INab1 = NULL;
        ead->geom[qp].INab2 = NULL;
        }
    }

    return 0;
}

/* Call this from your ActDataDestroy loop per element. */
static PetscErrorCode ElemActDataGeomDestroy_(FE *fem, ElemActData *ead)
{
  PetscErrorCode ierr = 0;
  PetscInt n_qp = fem->act_data.n_qp;

  if (!ead || !ead->geom) return 0;

  for (PetscInt qp = 0; qp < n_qp; qp++) {
    ierr = PetscFree(ead->geom[qp].INa0);  CHKERRQ(ierr);
    ierr = PetscFree(ead->geom[qp].INa1);  CHKERRQ(ierr);
    ierr = PetscFree(ead->geom[qp].INab0); CHKERRQ(ierr);
    ierr = PetscFree(ead->geom[qp].INab1); CHKERRQ(ierr);
    ierr = PetscFree(ead->geom[qp].INab2); CHKERRQ(ierr);
  }
  ierr = PetscFree(ead->geom); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode ActDataDestroy(FE *fem)
{
    PetscErrorCode ierr;
    ActData *act = &fem->act_data;

    PetscInt nelem = fem->ibm->n_elmt;

    for (PetscInt ec = 0; ec < nelem; ec++)
    {
        ElemActData *ead = &act->elem_act_data[ec];

        /* NEW: free geom cache */
        ierr = ElemActDataGeomDestroy_(fem, ead); CHKERRQ(ierr);

        ierr = PetscFree(ead->Fa);  CHKERRQ(ierr);
        ierr = PetscFree(ead->Fa_inv);  CHKERRQ(ierr);
        ierr = PetscFree(ead->C);   CHKERRQ(ierr);
        ierr = PetscFree(ead->C_inv);   CHKERRQ(ierr);
        ierr = PetscFree(ead->Ce);  CHKERRQ(ierr);
        ierr = PetscFree(ead->Ce_inv);  CHKERRQ(ierr);

        ierr = PetscFree(ead->Se);  CHKERRQ(ierr);
        ierr = PetscFree(ead->S);   CHKERRQ(ierr);

        ierr = PetscFree(ead->gm);  CHKERRQ(ierr);
        ierr = PetscFree(ead->gm0); CHKERRQ(ierr);

        ierr = PetscFree(ead->CCe); CHKERRQ(ierr);
        ierr = PetscFree(ead->CC);  CHKERRQ(ierr);

        ierr = PetscFree(ead->g);   CHKERRQ(ierr);
        ierr = PetscFree(ead->g0);  CHKERRQ(ierr);
    }

    ierr = PetscFree(act->elem_act_data);   CHKERRQ(ierr);
    act->elem_act_data = NULL;

    return 0;
}

/*--------------------------------------------------------------------------------------------------
 *                           Element-Level Computation Routines
 *-------------------------------------------------------------------------------------------------*/

/* Deformation gradient */
PetscErrorCode ElemActDefGrad(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    IBMNodes *ibm = fem->ibm;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    PetscReal Fa_cart[3][3] = {{0.0}};
    PetscReal Finv_cart[3][3];
    PetscReal S[3][3], ST[3][3];
    PetscReal tmp[3][3];

    PetscReal gamma = fem->userctx.muscle_act_params.gamma;

    /*------------------------------------------------------------*/
    /* 1. Active deformation gradient in Cartesian basis          */
    /*------------------------------------------------------------*/
    if (cart_fib_act)
    {
        PetscReal nfiber[3] = {
            ibm->n_fib[ec].x,
            ibm->n_fib[ec].y,
            ibm->n_fib[ec].z};

        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                Fa_cart[i][j] =
                    (i == j) ? (1.0 - gamma * nfiber[i] * nfiber[j]) : 0.0;
            }
        }
    }
    else
    {
        Fa_cart[0][0] = 1.0 - gamma;
        Fa_cart[1][1] = 1.0 - gamma;
        Fa_cart[2][2] = 1.0;
    }

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        /*------------------------------------------------------------*/
        /* 2. Build S: columns = reference covariant basis g0         */
        /*------------------------------------------------------------*/
        for (PetscInt j = 0; j < 3; j++)
        {
            S[0][j] = ead->g0[qp].Cov[j].x;
            S[1][j] = ead->g0[qp].Cov[j].y;
            S[2][j] = ead->g0[qp].Cov[j].z;
        }
        TRANSPOSE(S, ST);
        /*------------------------------------------------------------*/
        /* 3. Fa_ij = S^T * Fa_cart * S                               */
        /*------------------------------------------------------------*/
        MATMULT(ST, Fa_cart, tmp);
        MATMULT(tmp, S, ead->Fa[qp].Cov);

        /*------------------------------------------------------------*/
        /* 4. Invert in Cartesian basis                               */
        /*------------------------------------------------------------*/
        ierr = INV(Fa_cart, Finv_cart);
        CHKERRQ(ierr);

        /*------------------------------------------------------------*/
        /* 5. F̄_ij = S^T * Fa_cart^{-1} * S                           */
        /*------------------------------------------------------------*/
        MATMULT(ST, Finv_cart, tmp);
        MATMULT(tmp, S, ead->Fa_inv[qp].Cov);

        /*------------------------------------------------------------*/
        /* 6. Raise indices                                           */
        /*------------------------------------------------------------*/
        ierr = RaiseIndices2(ead->gm0[qp].Cont,
                             ead->Fa[qp].Cov,
                             ead->Fa[qp].Cont);
        CHKERRQ(ierr);

        ierr = RaiseIndices2(ead->gm0[qp].Cont,
                             ead->Fa_inv[qp].Cov,
                             ead->Fa_inv[qp].Cont);
        CHKERRQ(ierr);
    }

    return ierr;
}

/* Cauchy-Green deformation tensor */
PetscErrorCode ElemCGDefTens(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {

        /* In-plane block: C_{αβ} = g_{αβ} */
        for (PetscInt i = 0; i < 2; i++)
        {
            for (PetscInt j = 0; j < 2; j++)
            {
                ead->C[qp].Cov[i][j] = ead->gm[qp].Cov[i][j];
                ead->C_inv[qp].Cont[i][j] = ead->gm[qp].Cont[i][j];
            }
        }

        /* Mixed components vanish (KL shell) */
        for (PetscInt a = 0; a < 2; a++)
        {
            ead->C[qp].Cov[a][2] = 0.0;
            ead->C[qp].Cov[2][a] = 0.0;

            ead->C_inv[qp].Cont[a][2] = 0.0;
            ead->C_inv[qp].Cont[2][a] = 0.0;
        }

        /* Thickness stretch initialization */
        ead->C[qp].Cov[2][2] = 1.0;
        ead->C_inv[qp].Cont[2][2] = 1.0;

        ierr = RaiseIndices2(ead->gm[qp].Cont,
                             ead->C[qp].Cov,
                             ead->C[qp].Cont);
        CHKERRQ(ierr);

        ierr = LowerIndices2(ead->gm[qp].Cov,
                             ead->C_inv[qp].Cont,
                             ead->C_inv[qp].Cov);
        CHKERRQ(ierr);
    }

    return 0;
}

/* Elastic-only Cauchy-Green deformation tensor */
PetscErrorCode ElemElasCGDefTens(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        /* Zero the tensor before accumulation */
        for (PetscInt i = 0; i < 3; i++)
            for (PetscInt j = 0; j < 3; j++)
                ead->Ce_inv[qp].Cont[i][j] = 0.0;

        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                for (PetscInt m = 0; m < 3; m++)
                {
                    for (PetscInt p = 0; p < 3; p++)
                    {
                        for (PetscInt q = 0; q < 3; q++)
                        {
                            for (PetscInt l = 0; l < 3; l++)
                            {
                                ead->Ce_inv[qp].Cont[i][j] +=
                                    ead->Fa[qp].Cont[i][m] *
                                    ead->gm0[qp].Cov[m][p] *
                                    ead->C_inv[qp].Cont[p][q] *
                                    ead->Fa[qp].Cont[l][j] *
                                    ead->gm0[qp].Cov[l][q];
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

/* Elastic stresses */
PetscErrorCode ElemElasStress(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    PetscReal detCe, detG0, Je;
    PetscReal mu = fem->userctx.mu; // shear modulus
    PetscReal K = fem->userctx.K;   // bulk modulus

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {

        /*------------------------------------------------------------------
         * Compute elastic Jacobian Je = sqrt(det(Ce)/det(g0))
         *-----------------------------------------------------------------*/
        detCe = DET3x3(ead->Ce[qp].Cov);  // determinant of elastic CG (covariant)
        detG0 = DET3x3(ead->gm0[qp].Cov); // determinant of reference metric
        Je = sqrt(detCe / detG0);

        /*------------------------------------------------------------------
         * Compute Neo-Hookean elastic second Piola–Kirchhoff stress
         *    S^ij = mu (g0^ij - Ce^ij) + K Ce^ij (Je^2 - Je)
         *-----------------------------------------------------------------*/
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                ead->Se[qp].Cont[i][j] =
                    mu * (ead->gm0[qp].Cont[i][j] - ead->Ce[qp].Cont[i][j]) + K * ead->Ce[qp].Cont[i][j] * (Je * Je - Je);
            }
        }
    }

    return ierr;
}

/* Total stresses including active contributions */
PetscErrorCode ElemTotStress(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                for (PetscInt w = 0; w < 3; w++)
                {
                    for (PetscInt p = 0; p < 3; p++)
                    {
                        for (PetscInt z = 0; z < 3; z++)
                        {
                            for (PetscInt s = 0; s < 3; s++)
                            {
                                
                                ead->S[qp].Cont[i][j] = 1 / 2 * ead->Fa_inv[qp].Cont[w][p] * ead->Fa_inv[qp].Cont[z][s] * ead->gm0[qp].Cont[s][j] * (ead->Se[qp].Cont[p][z] * ead->gm0[qp].Cont[w][i] + ead->Se[qp].Cont[p][i] * ead->gm0[qp].Cont[w][z])
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

/* Elastic tangent matrix */
PetscErrorCode ElemElsTangMatTens(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)    
    {
        /*------------------------------------------------------------------
        * Compute elastic Jacobian Je = sqrt(det(Ce)/det(g0))
        *-----------------------------------------------------------------*/
        PetscReal detCe, detG0;
        detCe = DET3x3(ead->Ce[qp].Cov);  // determinant of elastic CG (covariant)
        detG0 = DET3x3(ead->gm0[qp].Cov); // determinant of reference metric
        PetscReal Je = sqrt(detCe / detG0);

        PetscReal (*Ce_bar)[3] = ead->Ce_inv[qp].Cont;

        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                for (PetscInt k = 0; k < 3; k++)
                {
                    for (PetscInt l = 0; l < 3; l++)
                    {   
                        //CHECK! the formulation
                        ead->CCe[qp].Cont[i][j][k][l] = 
                        mu * (Ce_bar[i][k] * Ce_bar[j][l] + Ce_bar[i][l] * Ce_bar[j][k])
                        + 2 * K * (Ce_bar[i][j] * (Je * Je - 0.5 * Je) * 
                        // Ce_bar[i][k] * 
                        Ce_bar[k][l]
                        - 0.5 * (Je * Je - Je) * (Ce_bar[i][k] * Ce_bar[j][l] + Ce_bar[i][l] * Ce_bar[j][k]));

                    }
                }
            }
        }

    }
    return 0;
}

/* Total tangent matrix including activation effects */
PetscErrorCode ElemTotTangMatTens(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData   *ead  = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        /* (Optional) If you want to reset CC each qp before accumulating, uncomment:
        for (PetscInt i=0;i<3;i++)
        for (PetscInt j=0;j<3;j++)
        for (PetscInt k=0;k<3;k++)
        for (PetscInt l=0;l<3;l++)
            ead->CC[qp].Cont[i][j][k][l] = 0.0;
        */

        for (PetscInt i = 0; i < 3; i++)
        for (PetscInt j = 0; j < 3; j++)
        for (PetscInt k = 0; k < 3; k++)
        for (PetscInt l = 0; l < 3; l++)
        {
            for (PetscInt g = 0; g < 3; g++)
            for (PetscInt h = 0; h < 3; h++)
            for (PetscInt m = 0; m < 3; m++)
            for (PetscInt n = 0; n < 3; n++)
            for (PetscInt p = 0; p < 3; p++)
            for (PetscInt s = 0; s < 3; s++)
            for (PetscInt w = 0; w < 3; w++)
            for (PetscInt z = 0; z < 3; z++)
            {
                ead->CC[qp].Cont[i][j][k][l] +=
                    0.25
                  * ead->Fa_inv[qp].Cov[w][p]
                  * ead->Fa_inv[qp].Cov[z][s]
                  * ead->Fa_inv[qp].Cov[g][h]
                  * ead->Fa_inv[qp].Cov[n][m]
                  * ead->gm0[qp].Cont[s][j]
                  * ead->gm0[qp].Cont[m][l]
                  * (
                        ead->gm0[qp].Cont[g][k]
                      * (
                            ead->CCe[qp].Cont[h][n][p][z] * ead->gm0[qp].Cont[w][i]
                          + ead->CCe[qp].Cont[h][n][p][i] * ead->gm0[qp].Cont[w][z]
                        )
                      + ead->gm0[qp].Cont[g][n]
                      * (
                            ead->CCe[qp].Cont[h][k][p][z] * ead->gm0[qp].Cont[w][i]
                          + ead->CCe[qp].Cont[h][k][p][i] * ead->gm0[qp].Cont[w][z]
                        )
                    );
            }
        }
    }

    return ierr;
}


/* Modification of element tensor component C33 (one Newton step per call)
   Goal: drive S33 -> 0 by updating C33 using DeltaC33 = -2*S33 / CC3333.
*/
PetscErrorCode ModElemC33(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData   *ead  = &fem->act_data.elem_act_data[ec];

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        /* Make sure S and CC are up-to-date BEFORE the update.
           If the calling code already computed them right before this function,
           you can remove these calls.
        */
        /* PetscCall(ElemTotStress(fem, ec));      // fills ead->S[qp]
           PetscCall(ElemTotTangMatTens(fem, ec)); // fills ead->CC[qp] */

        PetscReal S33    = ead->S[qp].Cont[2][2];
        PetscReal CC3333 = ead->CC[qp].Cont[2][2][2][2];

        /* Guard: avoid division by zero / tiny tangent */
        PetscReal denom_tol = 1e-14; /* tune as needed for your scaling */
        PetscCheck(PetscAbsReal(CC3333) > denom_tol,
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "CC3333 too small (%.6e) at qp=%" PetscInt_FMT, (double)CC3333, qp);

        /* One Newton update step */
        PetscReal DeltaC33 = -2.0 * S33 / CC3333;

        /* Update C33 (covariant) */
        ead->C[qp].Cov[2][2] += DeltaC33;
        
        /* Recompute stress/tangent after update so outer loop sees new S33.
           If your outer iteration recomputes these anyway, you can omit these.
        */
        /* PetscCall(ElemTotStress(fem, ec));
           PetscCall(ElemTotTangMatTens(fem, ec)); */
    }

    return ierr;
}


PetscErrorCode UpdateFint(FE *fem)
{
    h/2. * A0 * Bm * qp_w[qp] * S[qp] + h/2. * A0 * Bb * qp_w[qp] * S[qp] * (h/2. * qp_xi[qp]);
}


/* Transfer reference covariant basis vectors (G1,G2) from IBMNodes storage
   (computed by Kve0()) into element activation data ead->g0[qp].Cov[]. */
PetscErrorCode ElemUpdateG0(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    IBMNodes      *ibm  = fem->ibm;
    ElemActData   *ead  = &fem->act_data.elem_act_data[ec];

    const PetscInt dof = fem->dof; /* expected to be 3 here */

    /* Load reference covariant tangents from ibm arrays */
    Cmpnts G1, G2, N;

    G1.x = ibm->G1[dof*ec + 0];
    G1.y = ibm->G1[dof*ec + 1];
    G1.z = ibm->G1[dof*ec + 2];

    G2.x = ibm->G2[dof*ec + 0];
    G2.y = ibm->G2[dof*ec + 1];
    G2.z = ibm->G2[dof*ec + 2];

    /* Build reference normal */
    N = UNIT(CROSS(G1, G2));

    /* Push into all quadrature points (if your g0 is qp-dependent later, change here) */
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        ead->g0[qp].Cov[0] = G1;
        ead->g0[qp].Cov[1] = G2;
        ead->g0[qp].Cov[2] = N;
    }

    return ierr;
}


/* active_strain.c */

#include "variables.h"
#include <petscsys.h>
#include <math.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
   Shape function derivatives at element center (subdivision surface)
   (kept file-scope so ElemUpdateG and ElemUpdFint can share them)
   -------------------------------------------------------------------------- */
static const PetscReal Na_center[2][12] = {
  {-0.0247, -0.0309,  0.0000, -0.4815, -0.1852,  0.0247,
    0.4815,  0.0000, -0.0062,  0.0309,  0.1852,  0.0062},
  {-0.0309, -0.0247, -0.1852, -0.4815,  0.0000, -0.0062,
    0.0000,  0.4815,  0.0247,  0.0062,  0.1852,  0.0309}
};

static const PetscReal Nab_center[3][12] = {
  { 0.1111,  0.2222, -0.2222, -0.2222,  0.4444,  0.1111,
   -0.2222, -0.8889,  0.0000,  0.2222,  0.4444,  0.0000},
  { 0.2222,  0.1111,  0.4444, -0.2222, -0.2222,  0.0000,
   -0.8889, -0.2222,  0.1111,  0.0000,  0.4444,  0.2222},
  { 0.1667,  0.1667, -0.1111,  0.2222, -0.1111, -0.0556,
   -0.4444, -0.4444, -0.0556,  0.0556,  0.5556,  0.0556}
};

/* --------------------------------------------------------------------------
   ElemUpdateG:
   - updates curvilinear covariant basis g_i (stored in ead->g[qp].Cov[])
   - and contravariant basis g^i (stored in ead->g[qp].Cont[])
   - for ALL quadrature points qp (loop on qp)
   - Uses current coordinates ibm->x_bp/y_bp/z_bp (NOT reference)
   - Minimal changes: this is basically your Kve-style geometry part,
     just storing into ead instead of ibm->G1/G2.
   -------------------------------------------------------------------------- */
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec)
{
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ibm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");
  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->g, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->g is NULL");
  PetscCheck(fem->act_data.n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "n_qp must be > 0");

  /* Note: Your Na_center/Nab_center are “at element center”.
     So qp loop repeats the same geometry unless you later change Na/Nab to depend on qp.
     But you asked explicitly to loop qp, so we do that and store per qp. */
  for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {

    /* ------------------------------------------------------------
       REGULAR PATCH
       ------------------------------------------------------------ */
    if (ibm->ire[ec] == 0) {

      PetscReal x[12], y[12], z[12];
      PetscInt  node, nob = 1;

      for (PetscInt i = 0; i < 12; i++) {
        if (ibm->patch[16*ec + i] != 1000000) {
          node = ibm->patch[16*ec + i];
          x[i] = ibm->x_bp[node];
          y[i] = ibm->y_bp[node];
          z[i] = ibm->z_bp[node];
        } else {
          nob = 0;
        }
      }
      PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Regular patch missing control point ec=%" PetscInt_FMT, ec);

      /* second derivatives (used later in fint; we compute Aaa/Abb/Aab here
         only if you later decide to store them; for basis we only need ndx21/ndx31/nn) */
      struct Cmpnts ndx21 = {0,0,0};
      struct Cmpnts ndx31 = {0,0,0};

      for (PetscInt i = 0; i < 12; i++) {
        ndx21.x += Na_center[0][i]*x[i];
        ndx21.y += Na_center[0][i]*y[i];
        ndx21.z += Na_center[0][i]*z[i];

        ndx31.x += Na_center[1][i]*x[i];
        ndx31.y += Na_center[1][i]*y[i];
        ndx31.z += Na_center[1][i]*z[i];
      }

      struct Cmpnts nn = UNIT(CROSS(ndx21, ndx31));

      /* contravariant base vectors gc1, gc2 (your code names) */
      struct Cmpnts gc1 = CROSS(ndx31, nn);
      gc1 = AMULT(1.0 / DOT(ndx21, gc1), gc1);

      struct Cmpnts gc2 = CROSS(nn, ndx21);
      gc2 = AMULT(1.0 / DOT(ndx31, gc2), gc2);

      /* Store into ead->g[qp] */
      ead->g[qp].Cov[0]  = ndx21;
      ead->g[qp].Cov[1]  = ndx31;
      ead->g[qp].Cov[2]  = nn;

      ead->g[qp].Cont[0] = gc1;
      ead->g[qp].Cont[1] = gc2;
      ead->g[qp].Cont[2] = nn;   /* common choice; if you prefer a true g^3, change later */

      continue;
    }

    /* ------------------------------------------------------------
       IRREGULAR PATCH
       ------------------------------------------------------------ */
    if (ibm->ire[ec] == 1) {

      PetscInt v   = ibm->val[ec];
      PetscInt node, nob = 1;

      for (PetscInt i = 0; i < (v+6); i++) {
        if (ibm->patch[16*ec + i] == 1000000) { nob = 0; }
      }
      PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Irregular patch missing control point ec=%" PetscInt_FMT, ec);

      /* Build X0 (current coordinates at control points) exactly like your code */
      PetscReal **X0 = (PetscReal**)malloc((v+6) * sizeof(PetscReal*));
      PetscCheck(X0, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0");
      for (PetscInt i = 0; i < (v+6); i++) {
        X0[i] = (PetscReal*)malloc(3 * sizeof(PetscReal));
        PetscCheck(X0[i], PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0[i]");
        X0[i][0] = X0[i][1] = X0[i][2] = 0.0;
      }

      for (PetscInt i = 0; i < (v+6); i++) {
        node = ibm->patch[16*ec + i];
        X0[i][0] = ibm->x_bp[node];
        X0[i][1] = ibm->y_bp[node];
        X0[i][2] = ibm->z_bp[node];
      }

      PetscReal w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v), 2.));

      PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];

      /* B2 init */
      for (PetscInt i = 0; i < 12; i++)
        for (PetscInt j = 0; j < (v+12); j++)
          B2[i][j] = 0.;

      /* B1 init */
      for (PetscInt i = 0; i < (v+12); i++)
        for (PetscInt j = 0; j < (v+6); j++)
          B1[i][j] = 0.;

      /* B2 entries (your code) */
      B2[0][v+9]  = 1.;
      B2[1][v+6]  = 1.;
      B2[2][v+4]  = 1.;
      B2[3][v+1]  = 1.;
      B2[4][v+2]  = 1.;
      B2[5][v+5]  = 1.;
      B2[6][v]    = 1.;
      B2[7][1]    = 1.;
      B2[8][v+3]  = 1.;
      B2[9][v-1]  = 1.;
      B2[10][0]   = 1.;
      B2[11][2]   = 1.;

      /* B1 entries (your code) */
      {
        PetscInt j;
        B1[0][0] = 1 - v*w;  for (j=0; j<v; j++) {B1[0][1+j] = w;}
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
      }

      /* B3 = B2 * B1 */
      for (PetscInt i = 0; i < 12; i++) {
        for (PetscInt j = 0; j < (v+6); j++) {
          B3[i][j] = 0.;
          for (PetscInt m = 0; m < (v+12); m++) {
            B3[i][j] += B2[i][m] * B1[m][j];
          }
        }
      }

      /* INa / INab (derived from Na_center / Nab_center) */
      PetscReal INa[2][v+6], INab[3][v+6];
      for (PetscInt j = 0; j < (v+6); j++) {
        PetscReal sum1=0., sum2=0., sum3=0., sum4=0., sum5=0.;
        for (PetscInt i = 0; i < 12; i++) {
          sum1 += B3[i][j]*Na_center[0][i];
          sum2 += B3[i][j]*Na_center[1][i];
          sum3 += B3[i][j]*Nab_center[0][i];
          sum4 += B3[i][j]*Nab_center[1][i];
          sum5 += B3[i][j]*Nab_center[2][i];
        }
        INa[0][j]  = -2*sum1;
        INa[1][j]  = -2*sum2;
        INab[0][j] =  4*sum3;
        INab[1][j] =  4*sum4;
        INab[2][j] =  4*sum5;
      }

      /* First derivatives at element center for irregular patch */
      struct Cmpnts ndx21 = {0,0,0};
      struct Cmpnts ndx31 = {0,0,0};
      for (PetscInt i = 0; i < (v+6); i++) {
        ndx21.x += INa[0][i]*X0[i][0];
        ndx21.y += INa[0][i]*X0[i][1];
        ndx21.z += INa[0][i]*X0[i][2];

        ndx31.x += INa[1][i]*X0[i][0];
        ndx31.y += INa[1][i]*X0[i][1];
        ndx31.z += INa[1][i]*X0[i][2];
      }

      struct Cmpnts nn = UNIT(CROSS(ndx21, ndx31));

      struct Cmpnts gc1 = CROSS(ndx31, nn);
      gc1 = AMULT(1.0 / DOT(ndx21, gc1), gc1);

      struct Cmpnts gc2 = CROSS(nn, ndx21);
      gc2 = AMULT(1.0 / DOT(ndx31, gc2), gc2);

      /* Store */
      ead->g[qp].Cov[0]  = ndx21;
      ead->g[qp].Cov[1]  = ndx31;
      ead->g[qp].Cov[2]  = nn;

      ead->g[qp].Cont[0] = gc1;
      ead->g[qp].Cont[1] = gc2;
      ead->g[qp].Cont[2] = nn;   /* same note as above */

      /* Free */
      for (PetscInt i = 0; i < (v+6); i++) free(X0[i]);
      free(X0);

      continue;
    }

    /* If neither 0 nor 1 */
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
            "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
            ibm->ire[ec], ec);
  }

  return ierr;
}

PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec,
                          const PetscReal Sm[3],   /* membrane stress resultants (your Sm[m]) */
                          const PetscReal S[3],    /* bending stress resultants (your S[m]) */
                          const PetscReal Em[3],   /* membrane strains (for IE) */
                          const PetscReal Eb[3],   /* bending strains  (for IE) */
                          PetscReal A0, PetscReal h0,
                          PetscReal *Fb_out        /* output: element force vector (size 36 for reg, 3*(v+6) for irreg) */
                          )
{
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  const PetscInt dof = fem->dof; /* expect 3 */

  PetscCheck(Fb_out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Fb_out is NULL");
  PetscCheck(ead->g, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->g is NULL");

  /* You asked: max qps small; not parallel now.
     Note: Your Na/Nab are center-based so qp repeats; still loop qp if you want integration later. */
  for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {

    /* Grab basis computed in ElemUpdateG */
    const struct Cmpnts ndx21 = ead->g[qp].Cov[0];
    const struct Cmpnts ndx31 = ead->g[qp].Cov[1];
    const struct Cmpnts nn    = ead->g[qp].Cov[2];

    const struct Cmpnts gc1   = ead->g[qp].Cont[0];
    const struct Cmpnts gc2   = ead->g[qp].Cont[1];

    /* ------------------------- REGULAR PATCH ------------------------- */
    if (ibm->ire[ec] == 0) {

      /* Build x/y/z for Aaa, Abb, Aab only (needed for bending b1,b2,b3) */
      PetscReal x[12], y[12], z[12];
      PetscInt  node, nob = 1;

      for (PetscInt i = 0; i < 12; i++) {
        if (ibm->patch[16*ec + i] != 1000000) {
          node = ibm->patch[16*ec + i];
          x[i] = ibm->x_bp[node];
          y[i] = ibm->y_bp[node];
          z[i] = ibm->z_bp[node];
        } else {
          nob = 0;
        }
      }
      PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Regular patch missing control point ec=%" PetscInt_FMT, ec);

      struct Cmpnts Aaa = {0,0,0}, Abb = {0,0,0}, Aab = {0,0,0};

      for (PetscInt i = 0; i < 12; i++) {
        Aaa.x += Nab_center[0][i]*x[i];
        Aaa.y += Nab_center[0][i]*y[i];
        Aaa.z += Nab_center[0][i]*z[i];

        Abb.x += Nab_center[1][i]*x[i];
        Abb.y += Nab_center[1][i]*y[i];
        Abb.z += Nab_center[1][i]*z[i];

        Aab.x += Nab_center[2][i]*x[i];
        Aab.y += Nab_center[2][i]*y[i];
        Aab.z += Nab_center[2][i]*z[i];
      }

      /* Membrane matrix */
      PetscReal Bm[3][36];
      for (PetscInt i = 0; i < 12; i++) {
        Bm[0][3*i+0] = Na_center[0][i]*ndx21.x;   Bm[0][3*i+1] = Na_center[0][i]*ndx21.y;   Bm[0][3*i+2] = Na_center[0][i]*ndx21.z;
        Bm[1][3*i+0] = Na_center[1][i]*ndx31.x;   Bm[1][3*i+1] = Na_center[1][i]*ndx31.y;   Bm[1][3*i+2] = Na_center[1][i]*ndx31.z;
        Bm[2][3*i+0] = Na_center[0][i]*ndx31.x + Na_center[1][i]*ndx21.x;
        Bm[2][3*i+1] = Na_center[0][i]*ndx31.y + Na_center[1][i]*ndx21.y;
        Bm[2][3*i+2] = Na_center[0][i]*ndx31.z + Na_center[1][i]*ndx21.z;
      }

      /* Bending matrix */
      PetscReal Bs[3][36];
      for (PetscInt i = 0; i < 12; i++) {
        struct Cmpnts ng1 = AMULT(Na_center[0][i], gc1);
        struct Cmpnts ng2 = AMULT(Na_center[1][i], gc2);
        struct Cmpnts ng  = PLUS(ng1, ng2);

        PetscReal b1 = DOT(ng, Aaa);
        PetscReal b2 = DOT(ng, Abb);
        PetscReal b3 = DOT(ng, Aab);

        Bs[0][3*i+0] = -(Nab_center[0][i] + b1)*nn.x;   Bs[0][3*i+1] = -(Nab_center[0][i] + b1)*nn.y;   Bs[0][3*i+2] = -(Nab_center[0][i] + b1)*nn.z;
        Bs[1][3*i+0] = -(Nab_center[1][i] + b2)*nn.x;   Bs[1][3*i+1] = -(Nab_center[1][i] + b2)*nn.y;   Bs[1][3*i+2] = -(Nab_center[1][i] + b2)*nn.z;
        Bs[2][3*i+0] = -2.0*(Nab_center[2][i] + b3)*nn.x; Bs[2][3*i+1] = -2.0*(Nab_center[2][i] + b3)*nn.y; Bs[2][3*i+2] = -2.0*(Nab_center[2][i] + b3)*nn.z;
      }

      /* membrane force */
      PetscReal Fm[36];
      for (PetscInt ii = 0; ii < 36; ii++) {
        PetscReal sum = 0.0;
        for (PetscInt m = 0; m < 3; m++) sum += Bm[m][ii] * Sm[m] * A0 * h0;
        Fm[ii] = sum;
      }

      /* bending + membrane */
      for (PetscInt ii = 0; ii < 36; ii++) {
        PetscReal sum = 0.0;
        for (PetscInt m = 0; m < 3; m++) sum += Bs[m][ii] * S[m] * A0 * PetscPowReal(h0, 3.0) / 12.0;
        Fb_out[ii] += sum + Fm[ii];
      }

      /* IE/FC bookkeeping exactly like your code (optional to keep here) */
      {
        PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];
        PetscReal sum = 0.;
        for (PetscInt i = 0; i < 3; i++) {
          sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*PetscPowReal(h0,3.0)/12.*Eb[i]*S[i]);
          fem->FC[dof*ec + i] = Sm[i]*A0*h0 + A0*PetscPowReal(h0,2.0)/2.*S[i];
        }
        fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;
      }

      continue;
    }

    /* ------------------------- IRREGULAR PATCH ------------------------- */
    if (ibm->ire[ec] == 1) {

      const PetscInt v = ibm->val[ec];
      PetscInt node, nob = 1;

      for (PetscInt i = 0; i < (v+6); i++) {
        if (ibm->patch[16*ec+i] == 1000000) { nob = 0; }
      }
      PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Irregular patch missing control point ec=%" PetscInt_FMT, ec);

      /* Rebuild X0 just to compute Aaa/Abb/Aab and INa/INab (same as your original).
         Note: we are NOT recomputing ndx21/ndx31/nn/gc1/gc2: we reuse them from ElemUpdateG. */
      PetscReal **X0 = (PetscReal**)malloc((v+6)*sizeof(PetscReal*));
      PetscCheck(X0, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0");
      for (PetscInt i = 0; i < (v+6); i++) {
        X0[i] = (PetscReal*)malloc(3*sizeof(PetscReal));
        PetscCheck(X0[i], PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0[i]");
        node = ibm->patch[16*ec + i];
        X0[i][0] = ibm->x_bp[node];
        X0[i][1] = ibm->y_bp[node];
        X0[i][2] = ibm->z_bp[node];
      }

      PetscReal w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v), 2.));

      PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];

      for (PetscInt i = 0; i < 12; i++)
        for (PetscInt j = 0; j < (v+12); j++)
          B2[i][j] = 0.;

      for (PetscInt i = 0; i < (v+12); i++)
        for (PetscInt j = 0; j < (v+6); j++)
          B1[i][j] = 0.;

      /* B2 */
      B2[0][v+9]  = 1.;  B2[1][v+6]  = 1.;  B2[2][v+4]  = 1.;  B2[3][v+1]  = 1.;
      B2[4][v+2]  = 1.;  B2[5][v+5]  = 1.;  B2[6][v]    = 1.;  B2[7][1]    = 1.;
      B2[8][v+3]  = 1.;  B2[9][v-1]  = 1.;  B2[10][0]   = 1.;  B2[11][2]   = 1.;

      /* B1 (same as in ElemUpdateG; keep identical) */
      {
        PetscInt j;
        B1[0][0] = 1 - v*w;  for (j=0; j<v; j++) {B1[0][1+j] = w;}
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
      }

      for (PetscInt i = 0; i < 12; i++) {
        for (PetscInt j = 0; j < (v+6); j++) {
          B3[i][j] = 0.;
          for (PetscInt m = 0; m < (v+12); m++) {
            B3[i][j] += B2[i][m]*B1[m][j];
          }
        }
      }

      PetscReal INa[2][v+6], INab[3][v+6];
      for (PetscInt j = 0; j < (v+6); j++) {
        PetscReal sum1=0., sum2=0., sum3=0., sum4=0., sum5=0.;
        for (PetscInt i = 0; i < 12; i++) {
          sum1 += B3[i][j]*Na_center[0][i];
          sum2 += B3[i][j]*Na_center[1][i];
          sum3 += B3[i][j]*Nab_center[0][i];
          sum4 += B3[i][j]*Nab_center[1][i];
          sum5 += B3[i][j]*Nab_center[2][i];
        }
        INa[0][j]  = -2*sum1;
        INa[1][j]  = -2*sum2;
        INab[0][j] =  4*sum3;
        INab[1][j] =  4*sum4;
        INab[2][j] =  4*sum5;
      }

      /* Aaa/Abb/Aab from INab */
      struct Cmpnts Aaa = {0,0,0}, Abb = {0,0,0}, Aab = {0,0,0};
      for (PetscInt i = 0; i < (v+6); i++) {
        Aaa.x += INab[0][i]*X0[i][0];  Aaa.y += INab[0][i]*X0[i][1];  Aaa.z += INab[0][i]*X0[i][2];
        Abb.x += INab[1][i]*X0[i][0];  Abb.y += INab[1][i]*X0[i][1];  Abb.z += INab[1][i]*X0[i][2];
        Aab.x += INab[2][i]*X0[i][0];  Aab.y += INab[2][i]*X0[i][1];  Aab.z += INab[2][i]*X0[i][2];
      }

      /* Membrane matrix */
      PetscReal Bm[3][3*(v+6)];
      for (PetscInt i = 0; i < (v+6); i++) {
        Bm[0][3*i+0] = INa[0][i]*ndx21.x;   Bm[0][3*i+1] = INa[0][i]*ndx21.y;   Bm[0][3*i+2] = INa[0][i]*ndx21.z;
        Bm[1][3*i+0] = INa[1][i]*ndx31.x;   Bm[1][3*i+1] = INa[1][i]*ndx31.y;   Bm[1][3*i+2] = INa[1][i]*ndx31.z;
        Bm[2][3*i+0] = INa[0][i]*ndx31.x + INa[1][i]*ndx21.x;
        Bm[2][3*i+1] = INa[0][i]*ndx31.y + INa[1][i]*ndx21.y;
        Bm[2][3*i+2] = INa[0][i]*ndx31.z + INa[1][i]*ndx21.z;
      }

      /* Bending matrix (you called it B4) */
      PetscReal B4[3][3*(v+6)];
      for (PetscInt i = 0; i < (v+6); i++) {
        struct Cmpnts ng1 = AMULT(INa[0][i], gc1);
        struct Cmpnts ng2 = AMULT(INa[1][i], gc2);
        struct Cmpnts ng  = PLUS(ng1, ng2);

        PetscReal b1 = DOT(ng, Aaa);
        PetscReal b2 = DOT(ng, Abb);
        PetscReal b3 = DOT(ng, Aab);

        B4[0][3*i+0] = -(INab[0][i] + b1)*nn.x;   B4[0][3*i+1] = -(INab[0][i] + b1)*nn.y;   B4[0][3*i+2] = -(INab[0][i] + b1)*nn.z;
        B4[1][3*i+0] = -(INab[1][i] + b2)*nn.x;   B4[1][3*i+1] = -(INab[1][i] + b2)*nn.y;   B4[1][3*i+2] = -(INab[1][i] + b2)*nn.z;
        B4[2][3*i+0] = -2.0*(INab[2][i] + b3)*nn.x; B4[2][3*i+1] = -2.0*(INab[2][i] + b3)*nn.y; B4[2][3*i+2] = -2.0*(INab[2][i] + b3)*nn.z;
      }

      /* membrane force */
      PetscReal Fm[3*(v+6)];
      for (PetscInt ii = 0; ii < 3*(v+6); ii++) {
        PetscReal sum = 0.0;
        for (PetscInt m = 0; m < 3; m++) sum += Bm[m][ii] * Sm[m] * A0 * h0;
        Fm[ii] = sum;
      }

      for (PetscInt ii = 0; ii < 3*(v+6); ii++) {
        PetscReal sum = 0.0;
        for (PetscInt m = 0; m < 3; m++) sum += B4[m][ii] * S[m] * A0 * PetscPowReal(h0, 3.0) / 12.0;
        Fb_out[ii] += sum + Fm[ii];
      }

      /* IE/FC bookkeeping (same idea; keep if you want) */
      {
        PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];
        PetscReal sum = 0.;
        for (PetscInt i = 0; i < 3; i++) {
          sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*PetscPowReal(h0,3.0)/12.*Eb[i]*S[i]);
          fem->FC[dof*ec + i] = Sm[i]*A0*h0 + A0*PetscPowReal(h0,2.0)/2.*S[i];
        }
        fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;
      }

      for (PetscInt i = 0; i < (v+6); i++) free(X0[i]);
      free(X0);

      continue;
    }

    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
            "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
            ibm->ire[ec], ec);
  }

  return ierr;
}


// PetscErrorCode Init(FE *fem, PetscInt ec)
// {
    
// }
/*--------------------------------------------------------------------------------------------------
 *                                 Utility: Generic Element Loop
 *-------------------------------------------------------------------------------------------------*/

/**
 * @brief Apply a given element-level routine to all elements.
 *
 * @param fem      Pointer to finite element structure.
 * @param n_elems  Number of elements.
 * @param func     Element routine to apply.
 */
PetscErrorCode UpdateElements(FE *fem, PetscInt n_elems, ElemFunc func)
{
    PetscErrorCode ierr;

    for (PetscInt ec = 0; ec < n_elems; ec++)
    {
        ierr = func(fem, ec);
        if (ierr)
            return ierr; /* Standard PETSc error propagation */
    }

    return 0;
}
