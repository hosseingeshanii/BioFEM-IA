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
    ierr = PetscMalloc1(nelem, &act->elem_act_data);    CHKERRQ(ierr);

    for (PetscInt ec = 0; ec < nelem; ec++)
    {
        ElemActData *ead = &act->elem_act_data[ec];

        /* Kinematics */
        ierr = PetscMalloc1(n_qp, &ead->Fa);    CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->Fa_inv);    CHKERRQ(ierr);

        ierr = PetscMalloc1(n_qp, &ead->C); CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->C_inv); CHKERRQ(ierr);

        ierr = PetscMalloc1(n_qp, &ead->Ce);    CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->Ce_inv);    CHKERRQ(ierr);

        /* Stress */
        ierr = PetscMalloc1(n_qp, &ead->Se);    CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->S); CHKERRQ(ierr);

        /* Metrics */
        ierr = PetscMalloc1(n_qp, &ead->gm);    CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->gm0);   CHKERRQ(ierr);

        /* Tangents */
        ierr = PetscMalloc1(n_qp, &ead->CCe);   CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->CC);    CHKERRQ(ierr);

        /* Bases */
        ierr = PetscMalloc1(n_qp, &ead->g); CHKERRQ(ierr);
        ierr = PetscMalloc1(n_qp, &ead->g0);    CHKERRQ(ierr);

        /* NEW: single midsurface geometry cache (size 1) */
        ierr = PetscMalloc1(1, &ead->geom);       CHKERRQ(ierr);

        ead->geom[0].nen = 0;
        ead->geom[0].v = 0;
        ead->geom[0].is_irregular = 0;

        ead->geom[0].INa0  = NULL;
        ead->geom[0].INa1  = NULL;
        ead->geom[0].INab0 = NULL;
        ead->geom[0].INab1 = NULL;
        ead->geom[0].INab2 = NULL;
    }

    return 0;
}

/* Call this from your ActDataDestroy loop per element. */
static PetscErrorCode ElemActDataGeomDestroy_(ElemActData *ead)
{
  PetscErrorCode ierr = 0;

  if (!ead || !ead->geom) return 0;

  ierr = PetscFree(ead->geom[0].INa0);  CHKERRQ(ierr);
  ierr = PetscFree(ead->geom[0].INa1);  CHKERRQ(ierr);
  ierr = PetscFree(ead->geom[0].INab0); CHKERRQ(ierr);
  ierr = PetscFree(ead->geom[0].INab1); CHKERRQ(ierr);
  ierr = PetscFree(ead->geom[0].INab2); CHKERRQ(ierr);

  ierr = PetscFree(ead->geom);          CHKERRQ(ierr);
  ead->geom = NULL;

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
        ierr = ElemActDataGeomDestroy_(ead); CHKERRQ(ierr);

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

/* Helper: ensure irregular arrays are allocated to length nen */
static PetscErrorCode SubdivGeomEnsureIrregularArrays_(SubdivGeomQP *G, PetscInt nen)
{
  PetscErrorCode ierr = 0;

  /* If already allocated but wrong size, free and reallocate.
     We do not store size separately, so use G->nen as the size indicator. */
  if (G->INa0 && G->nen != nen) {
    ierr = PetscFree(G->INa0);  CHKERRQ(ierr);
    ierr = PetscFree(G->INa1);  CHKERRQ(ierr);
    ierr = PetscFree(G->INab0); CHKERRQ(ierr);
    ierr = PetscFree(G->INab1); CHKERRQ(ierr);
    ierr = PetscFree(G->INab2); CHKERRQ(ierr);
    G->INa0 = G->INa1 = G->INab0 = G->INab1 = G->INab2 = NULL;
  }

  if (!G->INa0) {
    ierr = PetscMalloc1(nen, &G->INa0);  CHKERRQ(ierr);
    ierr = PetscMalloc1(nen, &G->INa1);  CHKERRQ(ierr);
    ierr = PetscMalloc1(nen, &G->INab0); CHKERRQ(ierr);
    ierr = PetscMalloc1(nen, &G->INab1); CHKERRQ(ierr);
    ierr = PetscMalloc1(nen, &G->INab2); CHKERRQ(ierr);
  }

  return 0;
}

/* The main geometry update.
   Call this once per element per configuration (before stresses or fint). */
PetscErrorCode ElemUpdateGeomSubdiv(FE *fem, PetscInt ec)
{
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ibm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");
  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->geom, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");

  SubdivGeomQP *G = &ead->geom[0];  /* midsurface cache */

  /* ============================================================
     REGULAR PATCH
     ============================================================ */
  if (ibm->ire[ec] == 0) {
    PetscReal x[12], y[12], z[12];
    PetscInt node;
    PetscInt nob = 1;

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

    /* Aaa, Abb, Aab from Nab_center */
    G->Aaa.x = G->Aaa.y = G->Aaa.z = 0.0;
    G->Abb.x = G->Abb.y = G->Abb.z = 0.0;
    G->Aab.x = G->Aab.y = G->Aab.z = 0.0;

    for (PetscInt i = 0; i < 12; i++) {
      G->Aaa.x += Nab_center[0][i]*x[i];
      G->Aaa.y += Nab_center[0][i]*y[i];
      G->Aaa.z += Nab_center[0][i]*z[i];

      G->Abb.x += Nab_center[1][i]*x[i];
      G->Abb.y += Nab_center[1][i]*y[i];
      G->Abb.z += Nab_center[1][i]*z[i];

      G->Aab.x += Nab_center[2][i]*x[i];
      G->Aab.y += Nab_center[2][i]*y[i];
      G->Aab.z += Nab_center[2][i]*z[i];
    }

    /* ndx21, ndx31 from Na_center */
    G->ndx21.x = G->ndx21.y = G->ndx21.z = 0.0;
    G->ndx31.x = G->ndx31.y = G->ndx31.z = 0.0;

    for (PetscInt i = 0; i < 12; i++) {
      G->ndx21.x += Na_center[0][i]*x[i];
      G->ndx21.y += Na_center[0][i]*y[i];
      G->ndx21.z += Na_center[0][i]*z[i];

      G->ndx31.x += Na_center[1][i]*x[i];
      G->ndx31.y += Na_center[1][i]*y[i];
      G->ndx31.z += Na_center[1][i]*z[i];
    }

    G->nn = UNIT(CROSS(G->ndx21, G->ndx31));

    G->gc1 = CROSS(G->ndx31, G->nn);
    G->gc1 = AMULT(1.0 / DOT(G->ndx21, G->gc1), G->gc1);

    G->gc2 = CROSS(G->nn, G->ndx21);
    G->gc2 = AMULT(1.0 / DOT(G->ndx31, G->gc2), G->gc2);

    G->is_irregular = 0;
    G->v = 0;
    G->nen = 12;

    /* regular: ensure no stale irregular arrays */
    if (G->INa0) {
      ierr = PetscFree(G->INa0);  CHKERRQ(ierr);
      ierr = PetscFree(G->INa1);  CHKERRQ(ierr);
      ierr = PetscFree(G->INab0); CHKERRQ(ierr);
      ierr = PetscFree(G->INab1); CHKERRQ(ierr);
      ierr = PetscFree(G->INab2); CHKERRQ(ierr);
      G->INa0 = G->INa1 = G->INab0 = G->INab1 = G->INab2 = NULL;
    }

    return 0;
  }

  /* ============================================================
     IRREGULAR PATCH
     ============================================================ */
  if (ibm->ire[ec] == 1) {

    const PetscInt v   = ibm->val[ec];
    const PetscInt nen = v + 6;

    PetscInt nob = 1;
    for (PetscInt i = 0; i < nen; i++) {
      if (ibm->patch[16*ec + i] == 1000000) nob = 0;
    }
    PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Irregular patch missing control point ec=%" PetscInt_FMT, ec);

    /* X0 */
    PetscReal **X0 = NULL;
    ierr = PetscMalloc1(nen, &X0); CHKERRQ(ierr);
    for (PetscInt i = 0; i < nen; i++) {
      ierr = PetscMalloc1(3, &X0[i]); CHKERRQ(ierr);
      PetscInt node = ibm->patch[16*ec + i];
      X0[i][0] = ibm->x_bp[node];
      X0[i][1] = ibm->y_bp[node];
      X0[i][2] = ibm->z_bp[node];
    }

    PetscReal w = (1.0/(PetscReal)v) *
                  (0.625 - PetscPowReal(0.375 + 0.25*PetscCosReal(2*PETSC_PI/(PetscReal)v), 2.0));

    /* NOTE: you previously used VLA-style arrays. Keep them if your compiler supports it.
       Here I keep your earlier “<=64” fixed stack assumption since v<=32. */
    PetscCheck(v+12 <= 64, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "v too large for fixed stack arrays; v=%" PetscInt_FMT, v);

    PetscReal B2[12][64], B1[64][64], B3[12][64];

    for (PetscInt i = 0; i < 12; i++)
      for (PetscInt j = 0; j < (v+12); j++)
        B2[i][j] = 0.0;

    for (PetscInt i = 0; i < (v+12); i++)
      for (PetscInt j = 0; j < nen; j++)
        B1[i][j] = 0.0;

    /* B2 entries */
    B2[0][v+9]  = 1.;  B2[1][v+6]  = 1.;  B2[2][v+4]  = 1.;  B2[3][v+1]  = 1.;
    B2[4][v+2]  = 1.;  B2[5][v+5]  = 1.;  B2[6][v]    = 1.;  B2[7][1]    = 1.;
    B2[8][v+3]  = 1.;  B2[9][v-1]  = 1.;  B2[10][0]   = 1.;  B2[11][2]   = 1.;

    /* B1 entries (unchanged block) */
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

    /* B3 = B2*B1 */
    for (PetscInt i = 0; i < 12; i++) {
      for (PetscInt j = 0; j < nen; j++) {
        PetscReal s = 0.0;
        for (PetscInt m = 0; m < (v+12); m++) s += B2[i][m] * B1[m][j];
        B3[i][j] = s;
      }
    }

    /* IN arrays (stack) */
    PetscReal INa0[64], INa1[64], INab0[64], INab1[64], INab2[64];
    for (PetscInt j = 0; j < nen; j++) {
      PetscReal s1=0., s2=0., s3=0., s4=0., s5=0.;
      for (PetscInt i = 0; i < 12; i++) {
        s1 += B3[i][j]*Na_center[0][i];
        s2 += B3[i][j]*Na_center[1][i];
        s3 += B3[i][j]*Nab_center[0][i];
        s4 += B3[i][j]*Nab_center[1][i];
        s5 += B3[i][j]*Nab_center[2][i];
      }
      INa0[j]  = -2.0*s1;
      INa1[j]  = -2.0*s2;
      INab0[j] =  4.0*s3;
      INab1[j] =  4.0*s4;
      INab2[j] =  4.0*s5;
    }

    /* Aaa/Abb/Aab */
    G->Aaa.x = G->Aaa.y = G->Aaa.z = 0.0;
    G->Abb.x = G->Abb.y = G->Abb.z = 0.0;
    G->Aab.x = G->Aab.y = G->Aab.z = 0.0;

    for (PetscInt i = 0; i < nen; i++) {
      G->Aaa.x += INab0[i]*X0[i][0];  G->Aaa.y += INab0[i]*X0[i][1];  G->Aaa.z += INab0[i]*X0[i][2];
      G->Abb.x += INab1[i]*X0[i][0];  G->Abb.y += INab1[i]*X0[i][1];  G->Abb.z += INab1[i]*X0[i][2];
      G->Aab.x += INab2[i]*X0[i][0];  G->Aab.y += INab2[i]*X0[i][1];  G->Aab.z += INab2[i]*X0[i][2];
    }

    /* ndx21/ndx31 */
    G->ndx21.x = G->ndx21.y = G->ndx21.z = 0.0;
    G->ndx31.x = G->ndx31.y = G->ndx31.z = 0.0;

    for (PetscInt i = 0; i < nen; i++) {
      G->ndx21.x += INa0[i]*X0[i][0];
      G->ndx21.y += INa0[i]*X0[i][1];
      G->ndx21.z += INa0[i]*X0[i][2];

      G->ndx31.x += INa1[i]*X0[i][0];
      G->ndx31.y += INa1[i]*X0[i][1];
      G->ndx31.z += INa1[i]*X0[i][2];
    }

    G->nn = UNIT(CROSS(G->ndx21, G->ndx31));

    G->gc1 = CROSS(G->ndx31, G->nn);
    G->gc1 = AMULT(1.0 / DOT(G->ndx21, G->gc1), G->gc1);

    G->gc2 = CROSS(G->nn, G->ndx21);
    G->gc2 = AMULT(1.0 / DOT(G->ndx31, G->gc2), G->gc2);

    /* store irregular arrays in cache */
    ierr = SubdivGeomEnsureIrregularArrays_(G, nen); CHKERRQ(ierr);
    for (PetscInt i = 0; i < nen; i++) {
      G->INa0[i]  = INa0[i];
      G->INa1[i]  = INa1[i];
      G->INab0[i] = INab0[i];
      G->INab1[i] = INab1[i];
      G->INab2[i] = INab2[i];
    }

    G->is_irregular = 1;
    G->v = v;
    G->nen = nen;

    for (PetscInt i = 0; i < nen; i++) { ierr = PetscFree(X0[i]); CHKERRQ(ierr); }
    ierr = PetscFree(X0); CHKERRQ(ierr);

    return 0;
  }

  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
          "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
          ibm->ire[ec], ec);
}

/* ============================================================================
   4) ElemUpdateG: now ONLY copies from cached geom -> ead->g
   ============================================================================
*/
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec)
{
  PetscErrorCode ierr = 0;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ead->geom, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");
  PetscCheck(ead->g,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->g is NULL");

  for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
    const SubdivGeomQP *G = &ead->geom[qp];

    ead->g[qp].Cov[0]  = G->ndx21;
    ead->g[qp].Cov[1]  = G->ndx31;
    ead->g[qp].Cov[2]  = G->nn;

    ead->g[qp].Cont[0] = G->gc1;
    ead->g[qp].Cont[1] = G->gc2;
    ead->g[qp].Cont[2] = G->nn; /* as before; adjust if you want true g^3 */
  }

  return ierr;
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
