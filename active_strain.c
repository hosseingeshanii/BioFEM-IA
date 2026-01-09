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
    }

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

        ierr = PetscFree(ead->Fa);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->Fa_inv);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->C);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->C_inv);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->Ce);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->Ce_inv);
        CHKERRQ(ierr);

        ierr = PetscFree(ead->Se);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->S);
        CHKERRQ(ierr);

        ierr = PetscFree(ead->gm);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->gm0);
        CHKERRQ(ierr);

        ierr = PetscFree(ead->CCe);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->CC);
        CHKERRQ(ierr);

        ierr = PetscFree(ead->g);
        CHKERRQ(ierr);
        ierr = PetscFree(ead->g0);
        CHKERRQ(ierr);
    }

    ierr = PetscFree(act->elem_act_data);
    CHKERRQ(ierr);

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
    // TODO
    return 0;
}

/* Total tangent matrix including activation effects */
PetscErrorCode ElemTotTangMatTens(FE *fem, PetscInt ec)
{
    // TODO
    return 0;
}

/* Modification of element tensor component C33 */
PetscErrorCode ModElemC33(FE *fem, PetscInt ec)
{
    // TODO
    return 0;
}

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
