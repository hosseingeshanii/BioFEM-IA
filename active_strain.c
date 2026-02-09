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
#include <petscsys.h>
#include <stdlib.h>


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


PetscErrorCode GetUserActParams(FE *fem){
    // UserCtx  *userctx = &fem->userctx;    

    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num_gaussian_quad_points", &(fem->act_data.n_qp), PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-C33_subitr_nums", &(fem->act_data.C33_subitr_nums), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_gamma", &(fem->act_data.muscle_act_params.gamma), PETSC_NULL);    
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-bulk_modulus", &(fem->act_data.K), PETSC_NULL);

    fem->act_data.mu = mu;
    return 0;
}

PetscErrorCode ActDataAllocate(FE *fem)
{
  PetscErrorCode ierr;
  ActData *act = &fem->act_data;
  PetscInt nelem = fem->ibm->n_elmt;
  PetscInt n_qp  = act->n_qp;

  /* Always start from a known state */
  /* If ActData can be re-allocated, consider calling ActDataDestroy(fem) first */
  act->theta = NULL;
  act->w = NULL;
  act->elem_act_data = NULL;

  ierr = PetscMalloc1(n_qp, &act->theta); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_qp, &act->w);     CHKERRQ(ierr);

  ierr = PetscMalloc1(nelem, &act->elem_act_data); CHKERRQ(ierr);
  /* Ensure all pointers inside each ElemActData start as NULL */
  ierr = PetscMemzero(act->elem_act_data, nelem * sizeof(*act->elem_act_data)); CHKERRQ(ierr);

  for (PetscInt ec = 0; ec < nelem; ec++) {
    ElemActData *ead = &act->elem_act_data[ec];

    /* Kinematics */
    ierr = PetscMalloc1(n_qp, &ead->Fa);     CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->Fa_inv); CHKERRQ(ierr);

    ierr = PetscMalloc1(n_qp, &ead->C);     CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->C_inv); CHKERRQ(ierr);

    ierr = PetscMalloc1(n_qp, &ead->Ce);     CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->Ce_inv); CHKERRQ(ierr);

    /* Stress */
    ierr = PetscMalloc1(n_qp, &ead->Se); CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->S);  CHKERRQ(ierr);

    /* Metrics */
    ierr = PetscMalloc1(n_qp, &ead->gm);  CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->gm0); CHKERRQ(ierr);

    /* Tangents */
    ierr = PetscMalloc1(n_qp, &ead->CCe); CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->CC);  CHKERRQ(ierr);

    /* Bases */
    ierr = PetscMalloc1(n_qp, &ead->g);  CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->g0); CHKERRQ(ierr);

    /* Midsurface geometry cache (size 1) */
    ierr = PetscMalloc1(1, &ead->geom);  CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &ead->geom0); CHKERRQ(ierr);

    /* Must ensure internal pointers inside SubdivGeomQP are NULL */
    ierr = PetscMemzero(&ead->geom[0],  sizeof(SubdivGeomQP)); CHKERRQ(ierr);
    ierr = PetscMemzero(&ead->geom0[0], sizeof(SubdivGeomQP)); CHKERRQ(ierr);
  }

  return 0;
}


PetscErrorCode SetGaussianQuadrature(FE *fem)
{
    PetscFunctionBeginUser;

    // Initialize all to zero
    for (PetscInt i = 0; i < fem->act_data.n_qp; i++) {
        fem->act_data.theta[i] = 0.0;
        fem->act_data.w[i] = 0.0;
    }

    switch (fem->act_data.n_qp) {
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

static PetscErrorCode SubdivGeomDestroy_(SubdivGeomQP *G)
{
  PetscErrorCode ierr = 0;
  if (!G) return 0;

  ierr = PetscFree(G->INa0);  CHKERRQ(ierr);
  ierr = PetscFree(G->INa1);  CHKERRQ(ierr);
  ierr = PetscFree(G->INab0); CHKERRQ(ierr);
  ierr = PetscFree(G->INab1); CHKERRQ(ierr);
  ierr = PetscFree(G->INab2); CHKERRQ(ierr);

  PetscMemzero(G, sizeof(*G));
  return 0;
}

static PetscErrorCode ElemActDataGeomDestroy_(ElemActData *ead)
{
  PetscErrorCode ierr;
  if (!ead) return 0;

  noted:
  if (ead->geom) {
    ierr = SubdivGeomDestroy_(&ead->geom[0]); CHKERRQ(ierr);
    ierr = PetscFree(ead->geom); CHKERRQ(ierr);
    ead->geom = NULL;
  }
  if (ead->geom0) {
    ierr = SubdivGeomDestroy_(&ead->geom0[0]); CHKERRQ(ierr);
    ierr = PetscFree(ead->geom0); CHKERRQ(ierr);
    ead->geom0 = NULL;
  }
  return 0;
}


PetscErrorCode ActDataDestroy(FE *fem)
{
  PetscErrorCode ierr;
  ActData *act = &fem->act_data;
  PetscInt nelem = fem->ibm->n_elmt;

  /* If elem_act_data was never allocated, just free global arrays */
  if (!act->elem_act_data) {
    ierr = PetscFree(act->theta); CHKERRQ(ierr);
    ierr = PetscFree(act->w);     CHKERRQ(ierr);
    act->theta = NULL;
    act->w = NULL;
    return 0;
  }

  for (PetscInt ec = 0; ec < nelem; ec++) {
    ElemActData *ead = &act->elem_act_data[ec];

    ierr = ElemActDataGeomDestroy_(ead); CHKERRQ(ierr);

    ierr = PetscFree(ead->Fa);     CHKERRQ(ierr);
    ierr = PetscFree(ead->Fa_inv); CHKERRQ(ierr);
    ierr = PetscFree(ead->C);      CHKERRQ(ierr);
    ierr = PetscFree(ead->C_inv);  CHKERRQ(ierr);
    ierr = PetscFree(ead->Ce);     CHKERRQ(ierr);
    ierr = PetscFree(ead->Ce_inv); CHKERRQ(ierr);

    ierr = PetscFree(ead->Se); CHKERRQ(ierr);
    ierr = PetscFree(ead->S);  CHKERRQ(ierr);

    ierr = PetscFree(ead->gm);  CHKERRQ(ierr);
    ierr = PetscFree(ead->gm0); CHKERRQ(ierr);

    ierr = PetscFree(ead->CCe); CHKERRQ(ierr);
    ierr = PetscFree(ead->CC);  CHKERRQ(ierr);

    ierr = PetscFree(ead->g);  CHKERRQ(ierr);
    ierr = PetscFree(ead->g0); CHKERRQ(ierr);

    /* Optional: clear ead (helps catch use-after-free) */
    ierr = PetscMemzero(ead, sizeof(*ead)); CHKERRQ(ierr);
  }

  ierr = PetscFree(act->elem_act_data); CHKERRQ(ierr);
  ierr = PetscFree(act->theta);         CHKERRQ(ierr);
  ierr = PetscFree(act->w);             CHKERRQ(ierr);

  act->elem_act_data = NULL;
  act->theta = NULL;
  act->w = NULL;

  return 0;
}


static void PrintCmpnts_(const char *name, Cmpnts a)
{
PetscPrintf(PETSC_COMM_SELF, " %-8s = (% .6e, % .6e, % .6e)\n", name, a.x, a.y, a.z);
}


static PetscErrorCode PrintSubdivGeomQP_(const char *label, PetscInt ec, const SubdivGeomQP *G)
{
PetscErrorCode ierr = 0;
PetscCheck(G, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "G is NULL");


PetscPrintf(PETSC_COMM_SELF,
"\n==== %s: ec=%" PetscInt_FMT " ====\n"
" is_irregular=%" PetscInt_FMT " v=%" PetscInt_FMT " nen=%" PetscInt_FMT "\n",
label, ec, G->is_irregular, G->v, G->nen);


PrintCmpnts_("ndx21", G->ndx21);
PrintCmpnts_("ndx31", G->ndx31);
PrintCmpnts_("nn", G->nn);
PrintCmpnts_("gc1", G->gc1);
PrintCmpnts_("gc2", G->gc2);
PrintCmpnts_("Aaa", G->Aaa);
PrintCmpnts_("Abb", G->Abb);
PrintCmpnts_("Aab", G->Aab);


if (G->is_irregular) {
PetscCheck(G->nen > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Irregular but nen<=0");
PetscCheck(G->INa0 && G->INa1 && G->INab0 && G->INab1 && G->INab2,
PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
"Irregular but one or more IN* arrays are NULL");


/* print a small sample to avoid huge output */
const PetscInt nshow = PetscMin((PetscInt)6, G->nen);
PetscPrintf(PETSC_COMM_SELF, " IN* sample (first %" PetscInt_FMT " entries):\n", nshow);
for (PetscInt i = 0; i < nshow; i++) {
PetscPrintf(PETSC_COMM_SELF,
" i=%" PetscInt_FMT
" INa0=% .6e INa1=% .6e INab0=% .6e INab1=% .6e INab2=% .6e\n",
i, G->INa0[i], G->INa1[i], G->INab0[i], G->INab1[i], G->INab2[i]);
}
} else {
/* regular should not carry irregular arrays */
if (G->INa0 || G->INa1 || G->INab0 || G->INab1 || G->INab2) {
PetscPrintf(PETSC_COMM_SELF,
" WARNING: regular patch but IN* arrays are not NULL (stale?)\n");
}
}


return ierr;
}

PetscErrorCode DebugPrintGeomForElement(FE *fem, PetscInt ec)
{
  PetscErrorCode ierr = 0;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->geom0, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom0 is NULL");
  PetscCheck(ead->geom,  PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");

  ierr = PrintSubdivGeomQP_("GEOM0 (reference) geom0[0]", ec, &ead->geom0[0]); CHKERRQ(ierr);
  ierr = PrintSubdivGeomQP_("GEOM  (current)   geom[0]",  ec, &ead->geom[0]);  CHKERRQ(ierr);

  return ierr;
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
static PetscErrorCode ElemUpdateGeomSubdivFromCoords_(
    FE *fem, PetscInt ec,
    const PetscReal *xb, const PetscReal *yb, const PetscReal *zb,
    SubdivGeomQP *G)
{
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ibm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");
  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->geom, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");

//   SubdivGeomQP *G = &ead->geom[0];  /* midsurface cache */

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
        x[i] = xb[node];
        y[i] = yb[node];
        z[i] = zb[node];
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
      X0[i][0] = xb[node];
      X0[i][1] = yb[node];
      X0[i][2] = zb[node];
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


PetscErrorCode ElemUpdateGeomSubdiv(FE *fem, PetscInt ec)   /* current */
{
  IBMNodes *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];
  return ElemUpdateGeomSubdivFromCoords_(fem, ec, ibm->x_bp,  ibm->y_bp,  ibm->z_bp,  &ead->geom[0]);
}

PetscErrorCode ElemUpdateGeom0Subdiv(FE *fem, PetscInt ec)  /* reference */
{
  IBMNodes *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];
  return ElemUpdateGeomSubdivFromCoords_(fem, ec, ibm->x_bp0, ibm->y_bp0, ibm->z_bp0, &ead->geom0[0]);
}


/* Compute a3,1 and a3,2 given midsurface derivatives and tangents. */
static inline void Compute_a3_alpha_(
    const struct Cmpnts a1, const struct Cmpnts a2,
    const struct Cmpnts Aaa, const struct Cmpnts Aab, const struct Cmpnts Abb,
    struct Cmpnts *a3_1, struct Cmpnts *a3_2)
{
  /* size_a3 = ||a1 x a2|| */
  PetscReal size_a3 = SIZE(CROSS(a1, a2));

  /* Unnormalized:  a3,1 ~ (a1,1 x a2 + a1 x a2,1) with a2,1 = Aab */
  *a3_1 = PLUS(CROSS(Aaa, a2), CROSS(a1, Aab));

  /* Unnormalized:  a3,2 ~ (a1,2 x a2 + a1 x a2,2) with a1,2 = Aab, a2,2 = Abb */
  *a3_2 = PLUS(CROSS(Aab, a2), CROSS(a1, Abb));

  /* Normalize by ||a1 x a2|| */
  a3_1->x /= size_a3;  a3_1->y /= size_a3;  a3_1->z /= size_a3;
  a3_2->x /= size_a3;  a3_2->y /= size_a3;  a3_2->z /= size_a3;
}


/* Update ead->g and ead->g0 for all qp using theta[qp]. */
/* Helper: Compute metric tensor from covariant basis vectors
 * g_ij = g_i . g_j
 */
static PetscErrorCode ComputeMetricTensor(const Cmpnts g_cov[3], 
                                          PetscReal gCov[3][3])
{
  for (PetscInt i = 0; i < 3; i++) {
    for (PetscInt j = 0; j < 3; j++) {
      gCov[i][j] = DOT(g_cov[i], g_cov[j]);
    }
  }
  return 0;
}

/* Helper: Compute contravariant basis vectors from metric tensor inverse
 * g^i = g^ij * g_j  (raised indices)
 */
static PetscErrorCode ComputeContravariantBasis(const PetscReal gInv[3][3],
                                                const Cmpnts g_cov[3],
                                                Cmpnts g_cont[3])
{
  for (PetscInt i = 0; i < 3; i++) {
    g_cont[i].x = 0.0;
    g_cont[i].y = 0.0;
    g_cont[i].z = 0.0;
    for (PetscInt j = 0; j < 3; j++) {
      g_cont[i].x += gInv[i][j] * g_cov[j].x;
      g_cont[i].y += gInv[i][j] * g_cov[j].y;
      g_cont[i].z += gInv[i][j] * g_cov[j].z;
    }
  }
  return 0;
}

/* --- small helpers --- */


static void PrintMat3_(const char *name, const PetscReal A[3][3])
{
PetscPrintf(PETSC_COMM_SELF, " %s =\n", name);
for (PetscInt i = 0; i < 3; i++) {
PetscPrintf(PETSC_COMM_SELF, " [% .6e % .6e % .6e]\n", A[i][0], A[i][1], A[i][2]);
}
}


static PetscErrorCode PrintElemG_(FE *fem, PetscInt ec, PetscInt qp)
{
PetscErrorCode ierr = 0;
ActData *act = &fem->act_data;
ElemActData *ead = &act->elem_act_data[ec];


PetscCheck(ec >= 0 && ec < fem->ibm->n_elmt, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
"ec out of range: ec=%" PetscInt_FMT, ec);
PetscCheck(qp >= 0 && qp < act->n_qp, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
"qp out of range: qp=%" PetscInt_FMT, qp);


PetscPrintf(PETSC_COMM_SELF,
"\n============================================================\n"
"ElemUpdateG dump: ec=%" PetscInt_FMT " qp=%" PetscInt_FMT "\n"
" ire=%" PetscInt_FMT " val(v)=%" PetscInt_FMT " theta=% .6e\n"
"============================================================\n",
ec, qp, fem->ibm->ire[ec], fem->ibm->val[ec], (double)PetscRealPart(act->theta[qp]));


/* --- midsurface cached geometry (optional sanity check) --- */
if (ead->geom && ead->geom0) {
const SubdivGeomQP *Gc = &ead->geom[0];
const SubdivGeomQP *G0 = &ead->geom0[0];


PetscPrintf(PETSC_COMM_SELF, "\n-- midsurface cache (geom[0]) current --\n");
PrintCmpnts_("a1=ndx21", Gc->ndx21);
PrintCmpnts_("a2=ndx31", Gc->ndx31);
PrintCmpnts_("a3=nn", Gc->nn);
PrintCmpnts_("gc1", Gc->gc1);
PrintCmpnts_("gc2", Gc->gc2);


PetscPrintf(PETSC_COMM_SELF, "\n-- midsurface cache (geom0[0]) reference --\n");
PrintCmpnts_("a1_0", G0->ndx21);
PrintCmpnts_("a2_0", G0->ndx31);
PrintCmpnts_("a3_0", G0->nn);
PrintCmpnts_("gc1_0", G0->gc1);
PrintCmpnts_("gc2_0", G0->gc2);
}


/* --- CURRENT configuration: basis + metrics --- */
PetscPrintf(PETSC_COMM_SELF, "\n-- CURRENT (g, gm) at qp --\n");
PrintCmpnts_("g1 (Cov)", ead->g[qp].Cov[0]);
PrintCmpnts_("g2 (Cov)", ead->g[qp].Cov[1]);
PrintCmpnts_("g3 (Cov)", ead->g[qp].Cov[2]);


PrintCmpnts_("g^1 (Cont)", ead->g[qp].Cont[0]);
PrintCmpnts_("g^2 (Cont)", ead->g[qp].Cont[1]);
PrintCmpnts_("g^3 (Cont)", ead->g[qp].Cont[2]);


PrintMat3_("gm_ij (Cov)", ead->gm[qp].Cov);
PrintMat3_("gm^ij (Cont)", ead->gm[qp].Cont);


/* --- REFERENCE configuration: basis + metrics --- */
PetscPrintf(PETSC_COMM_SELF, "\n-- REFERENCE (g0, gm0) at qp --\n");
PrintCmpnts_("g0_1 (Cov)", ead->g0[qp].Cov[0]);
PrintCmpnts_("g0_2 (Cov)", ead->g0[qp].Cov[1]);
PrintCmpnts_("g0_3 (Cov)", ead->g0[qp].Cov[2]);


PrintCmpnts_("g0^1 (Cont)", ead->g0[qp].Cont[0]);
PrintCmpnts_("g0^2 (Cont)", ead->g0[qp].Cont[1]);
PrintCmpnts_("g0^3 (Cont)", ead->g0[qp].Cont[2]);


PrintMat3_("gm0_ij (Cov)", ead->gm0[qp].Cov);
PrintMat3_("gm0^ij (Cont)", ead->gm0[qp].Cont);


return ierr;
}


/* Update ead->g, ead->g0, ead->gm, ead->gm0 for all qp using theta[qp]. */
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec)
{
  PetscErrorCode ierr = 0;
  ActData     *act = &fem->act_data;
  ElemActData *ead = &act->elem_act_data[ec];

  PetscCheck(act->n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "n_qp must be > 0");
  PetscCheck(act->theta,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "act->theta is NULL");
  PetscCheck(ead->g,       PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->g is NULL");
  PetscCheck(ead->g0,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->g0 is NULL");
  PetscCheck(ead->gm,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->gm is NULL (metric tensor)");
  PetscCheck(ead->gm0,     PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->gm0 is NULL (reference metric tensor)");
  PetscCheck(ead->geom,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->geom is NULL (need midsurface cache)");
  PetscCheck(ead->geom0,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,  "ead->geom0 is NULL (need reference midsurface cache)");

  /* thickness half-scale */
  const PetscReal hhalf = 0.5 * h0;

  /* -------- CURRENT midsurface data -------- */
  const SubdivGeomQP *Gc = &ead->geom[0];   /* current midsurface */
  const struct Cmpnts a1c  = Gc->ndx21;
  const struct Cmpnts a2c  = Gc->ndx31;
  const struct Cmpnts a3c  = Gc->nn;        /* already UNIT(CROSS(a1,a2)) in geom */

  struct Cmpnts a3c_1, a3c_2;
  Compute_a3_alpha_(a1c, a2c, Gc->Aaa, Gc->Aab, Gc->Abb, &a3c_1, &a3c_2);

  /* -------- REFERENCE midsurface data -------- */
  const SubdivGeomQP *G0 = &ead->geom0[0];  /* reference midsurface */
  const struct Cmpnts a10 = G0->ndx21;
  const struct Cmpnts a20 = G0->ndx31;
  const struct Cmpnts a30 = G0->nn;

  struct Cmpnts a30_1, a30_2;
  Compute_a3_alpha_(a10, a20, G0->Aaa, G0->Aab, G0->Abb, &a30_1, &a30_2);

  /* -------- Fill for all thickness qp -------- */
  for (PetscInt qp = 0; qp < act->n_qp; qp++) {

    const PetscReal theta3 = hhalf * PetscRealPart(act->theta[qp]);

    /* -------- CURRENT configuration -------- */
    /* Covariant basis vectors: g_alpha = a_alpha + theta3 * a3,alpha ; g_3 = a3 */
    ead->g[qp].Cov[0] = PLUS(a1c,  AMULT(theta3, a3c_1));
    ead->g[qp].Cov[1] = PLUS(a2c,  AMULT(theta3, a3c_2));
    ead->g[qp].Cov[2] = a3c;

    /* Compute covariant metric tensor: g_ij = g_i . g_j */
    ierr = ComputeMetricTensor(ead->g[qp].Cov, ead->gm[qp].Cov); CHKERRQ(ierr);

    /* Compute contravariant metric tensor: g^ij = (g_ij)^{-1} */
    PetscReal gInv[3][3];
    ierr = INV(ead->gm[qp].Cov, gInv); CHKERRQ(ierr);
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        ead->gm[qp].Cont[i][j] = gInv[i][j];
      }
    }

    /* Compute contravariant basis vectors: g^i = g^ij * g_j */
    ierr = ComputeContravariantBasis(gInv, ead->g[qp].Cov, ead->g[qp].Cont); CHKERRQ(ierr);

    /* -------- REFERENCE configuration -------- */
    /* Covariant basis vectors: g0_alpha = a0_alpha + theta3 * a0_3,alpha ; g0_3 = a0_3 */
    ead->g0[qp].Cov[0] = PLUS(a10, AMULT(theta3, a30_1));
    ead->g0[qp].Cov[1] = PLUS(a20, AMULT(theta3, a30_2));
    ead->g0[qp].Cov[2] = a30;

    /* Compute covariant metric tensor: g0_ij = g0_i . g0_j */
    ierr = ComputeMetricTensor(ead->g0[qp].Cov, ead->gm0[qp].Cov); CHKERRQ(ierr);

    /* Compute contravariant metric tensor: g0^ij = (g0_ij)^{-1} */
    PetscReal g0Inv[3][3];
    ierr = INV(ead->gm0[qp].Cov, g0Inv); CHKERRQ(ierr);
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        ead->gm0[qp].Cont[i][j] = g0Inv[i][j];
      }
    }

    /* Compute contravariant basis vectors: g0^i = g0^ij * g0_j */
    ierr = ComputeContravariantBasis(g0Inv, ead->g0[qp].Cov, ead->g0[qp].Cont); CHKERRQ(ierr);
  }

  return ierr;
}


/*--------------------------------------------------------------------------------------------------
 *                           Element-Level Computation Routines
 *-------------------------------------------------------------------------------------------------*/

/* Helper function to print 3x3 matrix */
static inline void PrintMat3x3(const char *label, PetscReal M[3][3])
{
   PetscPrintf(PETSC_COMM_WORLD, "%s\n", label);
    for (PetscInt i = 0; i < 3; i++) {
       PetscPrintf(PETSC_COMM_WORLD, "  [%12.6f  %12.6f  %12.6f]\n",
                   (double)M[i][0], (double)M[i][1], (double)M[i][2]);
    }
}

/* Helper: Print 2D tensor (both covariant and contravariant) */
static inline void PrintElem2DTens(const char *label, const Elem2DTens *T)
{
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", label);
    PetscPrintf(PETSC_COMM_WORLD, "  Covariant:\n");
    for (PetscInt i = 0; i < 3; i++) {
        PetscPrintf(PETSC_COMM_WORLD, "    [%12.6f  %12.6f  %12.6f]\n",
                    (double)T->Cov[i][0], (double)T->Cov[i][1], (double)T->Cov[i][2]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "  Contravariant:\n");
    for (PetscInt i = 0; i < 3; i++) {
        PetscPrintf(PETSC_COMM_WORLD, "    [%12.6f  %12.6f  %12.6f]\n",
                    (double)T->Cont[i][0], (double)T->Cont[i][1], (double)T->Cont[i][2]);
    }
}

/* Print C and C_inv for a specific element ec */
static PetscErrorCode PrintElemC(FE *fem, PetscInt ec)
{
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n========== Element %d: Right Cauchy-Green Tensor ==========\n", ec);
    
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "\n--- QP %d ---\n", qp);
        PrintElem2DTens("C (Right Cauchy-Green tensor)", &ead->C[qp]);
        PrintElem2DTens("C_inv (Inverse)", &ead->C_inv[qp]);
    }
    
    return 0;
}

/* Print elastic Cauchy-Green tensor Ce and Ce_inv for element ec */
static PetscErrorCode PrintElemCe(FE *fem, PetscInt ec)
{
    if (ec != 100) return 0;
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElasCGDefTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        PrintElem2DTens("  Ce (Elastic Cauchy-Green tensor)", &ead->Ce[qp]);
        PrintElem2DTens("  Ce_inv (Inverse)", &ead->Ce_inv[qp]);
    }
    
    return 0;
}

/* Print elastic stress Se for element ec */
static PetscErrorCode PrintElemSe(FE *fem, PetscInt ec)
{
    if (ec != 100) return 0;
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElasStress ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        PrintElem2DTens("  Se (Elastic 2nd Piola-Kirchhoff stress)", &ead->Se[qp]);
    }
    
    return 0;
}

/* Print total stress S for element ec */
static PetscErrorCode PrintElemS(FE *fem, PetscInt ec)
{
    if (ec != 100) return 0;
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    // PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemTotStress ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        // PrintElem2DTens("  S (Total 2nd Piola-Kirchhoff stress)", &ead->S[qp]);
        PetscPrintf(PETSC_COMM_WORLD, "  S^[2][2] = %f\n", (double)ead->S[qp].Cont[2][2]);
    }
    
    return 0;
}

/* Print elastic material tangent CCe for element ec */
static PetscErrorCode PrintElemCCe(FE *fem, PetscInt ec)
{
    if (ec != 100) return 0;
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElsTangMatTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d: CCe[2][2][2][2] = %f\n", qp, (double)ead->CCe[qp].Cont[2][2][2][2]);
    }
    
    return 0;
}

/* Print total material tangent CC for element ec */
static PetscErrorCode PrintElemCC(FE *fem, PetscInt ec)
{
    if (ec != 100) return 0;
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemTotTangMatTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d: CC[2][2][2][2] = %f\n", qp, (double)ead->CC[qp].Cont[2][2][2][2]);
    }
    
    return 0;
}

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

    PetscReal gamma = fem->act_data.muscle_act_params.gamma;

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
        // PrintMat3x3("Fa_cart", Fa_cart);
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        if (ec == 100) {
            // PetscPrintf(PETSC_COMM_WORLD, "\n========== EC %d, QP %d ==========\n", ec, qp);
            
            // /* Print metric tensors */
            // PrintElem2DTens("gm0 (reference metric tensor)", &ead->gm0[qp]);
            // PrintElem2DTens("gm (current metric tensor)", &ead->gm[qp]);
        }
        
        /*------------------------------------------------------------*/
        /* 2. Build S: columns = reference covariant basis g0         */
        /*------------------------------------------------------------*/
        for (PetscInt j = 0; j < 3; j++)
        {
            S[0][j] = ead->g0[qp].Cov[j].x;
            S[1][j] = ead->g0[qp].Cov[j].y;
            S[2][j] = ead->g0[qp].Cov[j].z;
        }
        TRANS(S, ST);
        // PrintMat3x3("S", S);
        /*------------------------------------------------------------*/
        /* 3. Fa_ij = S^T * Fa_cart * S                               */
        /*------------------------------------------------------------*/
        MATMULT(ST, Fa_cart, tmp);
        MATMULT(tmp, S, ead->Fa[qp].Cov);

        // PrintMat3x3("ead->Fa[qp].Cov", ead->Fa[qp].Cov);

        /*------------------------------------------------------------*/
        /* 4. Invert in Cartesian basis                               */
        /*------------------------------------------------------------*/
        ierr = INV(Fa_cart, Finv_cart);
        CHKERRQ(ierr);

        // PrintMat3x3("Finv_cart", Finv_cart);

        /*------------------------------------------------------------*/
        /* 5. F̄_ij = S^T * Fa_cart^{-1} * S                           */
        /*------------------------------------------------------------*/
        MATMULT(ST, Finv_cart, tmp);
        MATMULT(tmp, S, ead->Fa_inv[qp].Cov);

        // PrintMat3x3("ead->Fa_inv[qp].Cov", ead->Fa_inv[qp].Cov);

        /*------------------------------------------------------------*/
        /* 6. Raise indices                                           */
        /*------------------------------------------------------------*/
        ierr = RaiseIndices2(ead->gm0[qp].Cont,
                             ead->Fa[qp].Cov,
                             ead->Fa[qp].Cont);
        CHKERRQ(ierr);
        
        if (ec == 100) {
            /* Print Fa after raising indices */
            // PrintElem2DTens("Fa (active deformation gradient)", &ead->Fa[qp]);
        }

        ierr = RaiseIndices2(ead->gm0[qp].Cont,
                             ead->Fa_inv[qp].Cov,
                             ead->Fa_inv[qp].Cont);
        CHKERRQ(ierr);
        
        if (ec == 100) {
            // /* Print Fa_inv after raising indices */
            // PrintElem2DTens("Fa_inv (inverse active deformation gradient)", &ead->Fa_inv[qp]);
        }
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

        // PrintMat3x3("C_cov before raise", ead->C[qp].Cov);
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
        
        /* Derive Ce.Cov from Ce_inv.Cont by lowering indices and inverting */
        /* Step 1: Lower indices to get Ce_inv.Cov from Ce_inv.Cont */
        ierr = LowerIndices2(ead->gm0[qp].Cov, ead->Ce_inv[qp].Cont, ead->Ce_inv[qp].Cov);
        CHKERRQ(ierr);
        
        /* Step 2: Invert to get Ce.Cov from Ce_inv.Cov */
        ierr = INV(ead->Ce_inv[qp].Cov, ead->Ce[qp].Cont);
        CHKERRQ(ierr);
        
        /* Step 3: Raise indices  to get Ce.Cont from Ce.Cov */
        ierr = LowerIndices2(ead->gm0[qp].Cov, ead->Ce[qp].Cont, ead->Ce[qp].Cov);
        CHKERRQ(ierr);
    }

    return 0;
}

/* Elastic stresses */
PetscErrorCode ElemElasStress(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    PetscReal detCe, detG0, Je;
    PetscReal mu = fem->act_data.mu; // shear modulus
    PetscReal K = fem->act_data.K;   // bulk modulus

    // if (ec == 100) {
    //     PetscPrintf(PETSC_COMM_WORLD, "\n========== ElemElasStress (EC=100) ==========\n");
    //     PetscPrintf(PETSC_COMM_WORLD, "mu = %f, K = %f\n", (double)mu, (double)K);
    // }

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {

        /*------------------------------------------------------------------
         * Compute elastic Jacobian Je = sqrt(det(Ce)/det(g0))
         *-----------------------------------------------------------------*/
        detCe = DET3x3(ead->Ce[qp].Cov);  // determinant of elastic CG (covariant)
        PetscReal detCecont = DET3x3(ead->Ce[qp].Cont);  // determinant of elastic CG (covariant)
        detG0 = DET3x3(ead->gm0[qp].Cov); // determinant of reference metric
        Je = sqrt(detCe / detG0);

        // if (ec == 100) {
        //     PetscPrintf(PETSC_COMM_SELF, "\nQP %d:\n", qp);
        //     // PetscPrintf(PETSC_COMM_SELF, "detCe = %f, detG0 = %f, detCecont = %f\n", (double)detCe, (double)detG0, detCecont);
        //     PetscPrintf(PETSC_COMM_SELF, "  Je = %f\n", (double)Je);
        // }

        /*------------------------------------------------------------------
         * Compute Neo-Hookean elastic second Piola–Kirchhoff stress
         *    S^ij = mu (g0^ij - Ce^ij) + K Ce^ij (Je^2 - Je)
         *-----------------------------------------------------------------*/
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                PetscReal term1 = mu * (ead->gm0[qp].Cont[i][j] - ead->Ce_inv[qp].Cont[i][j]);
                PetscReal term2 = K * ead->Ce_inv[qp].Cont[i][j] * (Je * Je - Je);
                
                ead->Se[qp].Cont[i][j] = term1 + term2;
                
                // if (ec == 100 && i < 2 && j < 2) {  /* Print only first 2x2 block to avoid clutter */
                //     PetscPrintf(PETSC_COMM_WORLD, "  [%d][%d]: term1 = %f, term2 = %f, Se[%d][%d] = %f\n",
                //                 i, j, (double)term1, (double)term2, i, j, (double)ead->Se[qp].Cont[i][j]);
                // }
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
              ead->S[qp].Cont[i][j] = 0.0;
          }
        }
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
                                
                                ead->S[qp].Cont[i][j] += 0.5 * ead->Fa_inv[qp].Cov[w][p] * ead->Fa_inv[qp].Cov[z][s] * ead->gm0[qp].Cont[s][j] * (ead->Se[qp].Cont[p][z] * ead->gm0[qp].Cont[w][i] + ead->Se[qp].Cont[p][i] * ead->gm0[qp].Cont[w][z]);
                            }
                        }
                    }
                }
            }
        }
        // PrintMat3x3(" Total S", ead->S[qp].Cont);
    }
    return 0;
}

/* Elastic tangent matrix */
PetscErrorCode ElemElsTangMatTens(FE *fem, PetscInt ec)
{
    PetscErrorCode ierr = 0;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscReal K = fem->act_data.K;
    PetscReal mu = fem->act_data.mu; // shear modulus

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)    
    {
        /*------------------------------------------------------------------
        * Compute elastic Jacobian Je = sqrt(det(Ce)/det(g0))
        *-----------------------------------------------------------------*/
        PetscReal detCe, detG0;
        detCe = DET3x3(ead->Ce[qp].Cov);  // determinant of elastic CG (covariant)
        detG0 = DET3x3(ead->gm0[qp].Cov); // determinant of reference metric
        PetscReal Je = sqrt(detCe / detG0);
        const PetscReal Je2 = Je * Je;

        const PetscReal alpha = K * (Je2 - Je) - mu;                 // alpha = K(J^2 - J) - mu
        const PetscReal beta  = 0.5 * K * Je * (2.0 * Je - 1.0);     // beta  = (1/2)K J(2J-1)



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
                        // ead->CCe[qp].Cont[i][j][k][l] = 
                        // mu * (Ce_bar[i][k] * Ce_bar[l][j] + Ce_bar[i][l] * Ce_bar[k][j])
                        // + 2 * K * (Ce_bar[i][j] * (Je * Je - 0.5 * Je) * Ce_bar[k][l]
                        // - 0.5 * (Je * Je - Je) * (Ce_bar[i][k] * Ce_bar[j][l] + Ce_bar[i][l] * Ce_bar[j][k]));
                      // ead->CCe[qp].Cont[i][j][k][l] =
                      //   beta * Ce_bar[i][j] * Ce_bar[k][l]
                      // - 0.5 * alpha * ( Ce_bar[i][k] * Ce_bar[l][j]
                      //                 + Ce_bar[i][l] * Ce_bar[k][j] );
                      ead->CCe[qp].Cont[i][j][k][l] =
                        (2.0 * beta) * Ce_bar[i][j] * Ce_bar[k][l]
                      - alpha * ( Ce_bar[i][k] * Ce_bar[l][j]
                                + Ce_bar[i][l] * Ce_bar[k][j] );
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
        
        */
        for (PetscInt i=0;i<3;i++)
        for (PetscInt j=0;j<3;j++)
        for (PetscInt k=0;k<3;k++)
        for (PetscInt l=0;l<3;l++)
            ead->CC[qp].Cont[i][j][k][l] = 0.0;

        for (PetscInt i = 2; i < 3; i++)
        for (PetscInt j = 2; j < 3; j++)
        for (PetscInt k = 2; k < 3; k++)
        for (PetscInt l = 2; l < 3; l++)
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
PetscErrorCode ModElemC33(FE *fem, PetscInt ec, PetscReal *delta)
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

        // PetscPrintf(PETSC_COMM_SELF, "S33 before update at qp=%" PetscInt_FMT ": %.6e\n", qp, (double)S33);
        // PetscPrintf(PETSC_COMM_SELF, "CC3333 before update at qp=%" PetscInt_FMT ": %.6e\n", qp, (double)CC3333);
        /* One Newton update step */

        PetscReal DeltaC33 = -1.0 * S33 / CC3333;
        *delta  = DeltaC33;
        /* Update C33 (covariant) */
        // PetscPrintf(PETSC_COMM_SELF, "before updating C33 = %f\n", (double)ead->C[qp].Cov[2][2]);
        ead->C[qp].Cov[2][2] += DeltaC33;
        // PetscPrintf(PETSC_COMM_SELF, "after updating C33 = %f\n", (double)ead->C[qp].Cov[2][2]);
        // PetscPrintf(PETSC_COMM_SELF, "after updating C33 = %f\n", (double)ead->C[qp].Cov[2][2]);

        /* Recompute stress/tangent after update so outer loop sees new S33.
           If your outer iteration recomputes these anyway, you can omit these.
        */
        /* PetscCall(ElemTotStress(fem, ec));
           PetscCall(ElemTotTangMatTens(fem, ec)); */
    }

    return ierr;
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


#undef __FUNCT__
#define __FUNCT__ "ElemUpdFint"
PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec, PetscReal *Fb_out)
{
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = NULL;
  ElemActData *ead = NULL;

  PetscCheck(fem,    PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem is NULL");
  PetscCheck(Fb_out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Fb_out is NULL");

  ibm = fem->ibm;
  PetscCheck(ibm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");

  ead = &fem->act_data.elem_act_data[ec];
  PetscCheck(ead,         PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->S,      PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->S is NULL");
  PetscCheck(ead->geom,   PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");
  PetscCheck(fem->act_data.w,     PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "act_data.w is NULL");
  PetscCheck(fem->act_data.theta, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "act_data.theta is NULL");
  PetscCheck(fem->act_data.n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "n_qp must be > 0");

  /* We use midsurface geometry only. */
  const SubdivGeomQP *G = &ead->geom[0];

  /* Surface bases (midsurface) */
  const struct Cmpnts ndx21 = G->ndx21;  /* a1 */
  const struct Cmpnts ndx31 = G->ndx31;  /* a2 */
  const struct Cmpnts nn    = G->nn;     /* a3 (unit normal) */
  const struct Cmpnts gc1   = G->gc1;    /* a^1 */
  const struct Cmpnts gc2   = G->gc2;    /* a^2 */

  /* Second derivatives / curvature-like vectors */
  const struct Cmpnts Aaa = G->Aaa;      /* a1,1  */
  const struct Cmpnts Abb = G->Abb;      /* a2,2  */
  const struct Cmpnts Aab = G->Aab;      /* a1,2  */

  /* Convenience scalars */
  PetscReal A0 = ibm->dA0[ec];
  const PetscReal half_h0 = 0.5 * h0;
  const PetscReal pref0   = 0.5 * h0 * A0; 

  /* -------------------------------------------------------------------
     Loop through thickness quadrature points qp
     ------------------------------------------------------------------- */
  for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {

    const PetscReal w_qp  = PetscRealPart(fem->act_data.w[qp]);      /* weight */
    const PetscReal xi_qp = PetscRealPart(fem->act_data.theta[qp]);  /* through-thickness coord */

    /* Build S-vector from contravariant stress at this qp */
    PetscReal Sv[3];
    Sv[0] = ead->S[qp].Cont[0][0];
    Sv[1] = ead->S[qp].Cont[1][1];
    Sv[2] = ead->S[qp].Cont[0][1];

    /* ============================================================
       REGULAR PATCH (12 control points -> 36 dofs)
       ============================================================ */
    if (ibm->ire[ec] == 0) {

      PetscReal Bm[3][36];
      PetscReal Bs[3][36];

      /* Membrane Bm */
      for (PetscInt a = 0; a < 12; a++) {
        Bm[0][3*a+0] = Na_center[0][a]*ndx21.x;   Bm[0][3*a+1] = Na_center[0][a]*ndx21.y;   Bm[0][3*a+2] = Na_center[0][a]*ndx21.z;
        Bm[1][3*a+0] = Na_center[1][a]*ndx31.x;   Bm[1][3*a+1] = Na_center[1][a]*ndx31.y;   Bm[1][3*a+2] = Na_center[1][a]*ndx31.z;

        Bm[2][3*a+0] = Na_center[0][a]*ndx31.x + Na_center[1][a]*ndx21.x;
        Bm[2][3*a+1] = Na_center[0][a]*ndx31.y + Na_center[1][a]*ndx21.y;
        Bm[2][3*a+2] = Na_center[0][a]*ndx31.z + Na_center[1][a]*ndx21.z;
      }

      /* Bending Bs */
      for (PetscInt a = 0; a < 12; a++) {

        /* ng = Na0 * a^1 + Na1 * a^2 */
        const struct Cmpnts ng1 = AMULT(Na_center[0][a], gc1);
        const struct Cmpnts ng2 = AMULT(Na_center[1][a], gc2);
        const struct Cmpnts ng  = PLUS(ng1, ng2);

        const PetscReal b1 = DOT(ng, Aaa);
        const PetscReal b2 = DOT(ng, Abb);
        const PetscReal b3 = DOT(ng, Aab);

        Bs[0][3*a+0] = -(Nab_center[0][a] + b1)*nn.x;    Bs[0][3*a+1] = -(Nab_center[0][a] + b1)*nn.y;    Bs[0][3*a+2] = -(Nab_center[0][a] + b1)*nn.z;
        Bs[1][3*a+0] = -(Nab_center[1][a] + b2)*nn.x;    Bs[1][3*a+1] = -(Nab_center[1][a] + b2)*nn.y;    Bs[1][3*a+2] = -(Nab_center[1][a] + b2)*nn.z;
        Bs[2][3*a+0] = -2.0*(Nab_center[2][a] + b3)*nn.x; Bs[2][3*a+1] = -2.0*(Nab_center[2][a] + b3)*nn.y; Bs[2][3*a+2] = -2.0*(Nab_center[2][a] + b3)*nn.z;
      }

      /* Assemble contribution */
      for (PetscInt ii = 0; ii < 36; ii++) {
        PetscReal mem = 0.0, bend = 0.0;
        for (PetscInt m = 0; m < 3; m++) {
          mem  += Bm[m][ii] * Sv[m];
          bend += Bs[m][ii] * Sv[m];
        }

        Fb_out[ii] += pref0 * w_qp * ( mem + bend * (half_h0 * xi_qp) );
      }

      continue;
    }

    /* ============================================================
       IRREGULAR PATCH (nen = v+6 control points -> 3*nen dofs)
       ============================================================ */
    if (ibm->ire[ec] == 1) {

      PetscCheck(G->is_irregular, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "geom says not irregular but ibm->ire[ec]==1 (ec=%" PetscInt_FMT ")", ec);

      const PetscInt nen = G->nen;
      PetscCheck(nen > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid nen=%" PetscInt_FMT, nen);

      PetscCheck(G->INa0 && G->INa1 && G->INab0 && G->INab1 && G->INab2,
                 PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Irregular IN arrays not allocated in geom cache");

      /* We form Bm and Bs on the fly to avoid giant stack arrays. */
      for (PetscInt a = 0; a < nen; a++) {

        /* Membrane rows for this control point a */
        PetscReal Bm0x = G->INa0[a]*ndx21.x;
        PetscReal Bm0y = G->INa0[a]*ndx21.y;
        PetscReal Bm0z = G->INa0[a]*ndx21.z;

        PetscReal Bm1x = G->INa1[a]*ndx31.x;
        PetscReal Bm1y = G->INa1[a]*ndx31.y;
        PetscReal Bm1z = G->INa1[a]*ndx31.z;

        PetscReal Bm2x = G->INa0[a]*ndx31.x + G->INa1[a]*ndx21.x;
        PetscReal Bm2y = G->INa0[a]*ndx31.y + G->INa1[a]*ndx21.y;
        PetscReal Bm2z = G->INa0[a]*ndx31.z + G->INa1[a]*ndx21.z;

        /* Bending coefficients b1,b2,b3 */
        const struct Cmpnts ng1 = AMULT(G->INa0[a], gc1);
        const struct Cmpnts ng2 = AMULT(G->INa1[a], gc2);
        const struct Cmpnts ng  = PLUS(ng1, ng2);

        const PetscReal b1 = DOT(ng, Aaa);
        const PetscReal b2 = DOT(ng, Abb);
        const PetscReal b3 = DOT(ng, Aab);

        /* Bending rows */
        PetscReal Bs0 = -(G->INab0[a] + b1);
        PetscReal Bs1 = -(G->INab1[a] + b2);
        PetscReal Bs2 = -2.0*(G->INab2[a] + b3);

        /* For each DOF component (x,y,z) for control point a */
        const PetscInt base = 3*a;

        /* x */
        {
          PetscReal mem  = Bm0x*Sv[0] + Bm1x*Sv[1] + Bm2x*Sv[2];
          PetscReal bend = (Bs0*nn.x)*Sv[0] + (Bs1*nn.x)*Sv[1] + (Bs2*nn.x)*Sv[2];
          Fb_out[base+0] += pref0 * w_qp * ( mem + bend * (half_h0 * xi_qp) );
        }
        /* y */
        {
          PetscReal mem  = Bm0y*Sv[0] + Bm1y*Sv[1] + Bm2y*Sv[2];
          PetscReal bend = (Bs0*nn.y)*Sv[0] + (Bs1*nn.y)*Sv[1] + (Bs2*nn.y)*Sv[2];
          Fb_out[base+1] += pref0 * w_qp * ( mem + bend * (half_h0 * xi_qp) );
        }
        /* z */
        {
          PetscReal mem  = Bm0z*Sv[0] + Bm1z*Sv[1] + Bm2z*Sv[2];
          PetscReal bend = (Bs0*nn.z)*Sv[0] + (Bs1*nn.z)*Sv[1] + (Bs2*nn.z)*Sv[2];
          Fb_out[base+2] += pref0 * w_qp * ( mem + bend * (half_h0 * xi_qp) );
        }
      }

      continue;
    }

    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
            "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
            ibm->ire[ec], ec);
  }

  return ierr;
}

PetscErrorCode InitActStrainProblem(FE *fem){
  PetscFunctionBeginUser;

  PetscCall(GetUserActParams(fem));
  PetscCall(ActDataAllocate(fem));
  PetscCall(SetGaussianQuadrature(fem));
  
  return 0;
}

PetscErrorCode ElemC33Solve(FE *fem, PetscInt ec) {
  PetscFunctionBeginUser;

  ElemActData *ead = NULL;

  PetscErrorCode ierr = 0;
  PetscReal delta;

  ead = &fem->act_data.elem_act_data[ec];

  for (PetscInt sub_itr = 0; sub_itr < fem->act_data.C33_subitr_nums; sub_itr++){
    
    ElemElasCGDefTens(fem, ec);
    // PrintElemCe(fem, ec);
    
    ElemElasStress(fem, ec);  
    // PrintElemSe(fem, ec);
    
    ElemTotStress(fem, ec);  
    // PrintElemS(fem, ec);
    
    ElemElsTangMatTens(fem, ec);  
    // PrintElemCCe(fem, ec);
    
    ElemTotTangMatTens(fem, ec); 
    // PrintElemCC(fem, ec); 
    // if (ec == 10)
    // {
    //   // PetscPrintf(PETSC_COMM_SELF, "deltaC33 = %f at sub_itr = %d \n", delta, sub_itr);
    //   // PetscPrintf(PETSC_COMM_SELF, "sub_itr = %d \n", sub_itr);
    // }
    // PetscPrintf(PETSC_COMM_SELF, "after ElemTotTangMatTens \n");
    // delta = 0.0;
        // PrintMat3x3(" ead->C[0].Cov be C33 Update", ead->C[0].Cov);
    // if (ec == 100)
    // PetscPrintf(PETSC_COMM_SELF, "before ModElemC33 on elem = %d \n", ec);
        // PrintElemS(fem, ec);

    ModElemC33(fem, ec, &delta);        

    if (ec == 100)
    {
      // PetscPrintf(PETSC_COMM_SELF, "ModElemC33 completed on elem = %d \n", ec);
      // PetscPrintf(PETSC_COMM_SELF, "deltaC33 = %f at sub_itr = %d \n", delta, sub_itr);
      // PrintElemS(fem, ec);
    }

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
      ierr = RaiseIndices2(ead->gm0[qp].Cont,
                            ead->C[qp].Cov,
                            ead->C[qp].Cont);
      CHKERRQ(ierr);

      ierr = INV(ead->C[qp].Cov, ead->C_inv[qp].Cont);
      CHKERRQ(ierr);                           

      ierr = LowerIndices2(ead->gm0[qp].Cov,
                            ead->C_inv[qp].Cont,
                            ead->C_inv[qp].Cov);
      CHKERRQ(ierr);                           
    }

    // if (ec == 100)
    // {
    //   PrintElemCe(fem, ec);
    // }
        
    // ElemElasCGDefTens(fem, ec);

    // if (ec == 100)
    // {
    //   PrintElemCe(fem, ec);
    // }

    // ElemElasStress(fem, ec);  
    // ElemTotStress(fem, ec);  
    if (ec == 10)
    {
    // PrintMat3x3(" Total S after C33 Update", ead->S[0].Cont);
      // PetscPrintf(PETSC_COMM_SELF, "sub_itr = %d \n", sub_itr);
    }
    
    // PrintMat3x3(" ead->C[0].Cov after C33 Update", ead->C[0].Cov);
    
    // ElemElsTangMatTens(fem, ec);  
    // ElemTotTangMatTens(fem, ec); 

  }
  
  return ierr;
}

PetscErrorCode FInternalPreCalc(FE *fem) {
  PetscFunctionBeginUser;
  
  PetscErrorCode ierr = 0;

  UpdateElements(fem, ElemUpdateGeom0Subdiv);  
  UpdateElements(fem, ElemUpdateGeomSubdiv);  
  
  // ierr = DebugPrintGeomForElement(fem, 100); CHKERRQ(ierr);

  UpdateElements(fem, ElemUpdateG);  
  // PetscInt ec = 100;
  // PetscInt qp = 0; /* choose qp you want to inspect */
  // PrintElemG_(fem, ec, qp);

  UpdateElements(fem, ElemActDefGrad);  
  UpdateElements(fem, ElemCGDefTens);  
  // PrintElemC(fem, 100);  /* uncomment to print element 100's C tensor */
  
  UpdateElements(fem, ElemC33Solve);
  return 0;
}

PetscErrorCode FInternalAct(FE *fem){
  PetscErrorCode ierr = 0;

  IBMNodes       *ibm=fem->ibm;
  PetscInt       i, ec;
  // PetscReal      Fm[9], Fb[42], Fint[9];

  // for (i=0; i<9; i++) {Fm[i]=0.0; Fint[i]=0.0;}
  // for (i=0; i<42; i++) {Fb[i]=0.0;}

  FInternalPreCalc(fem);
  // PetscPrintf(PETSC_COMM_SELF, "FInternalPreCalc completed\n");
  PetscReal  *FF;
  VecGetArray(fem->Fint, &FF);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    PetscInt  node, v = ibm->val[ec];
    PetscInt nloc = dof*(v+6);

    PetscReal *Fb;
    ierr = PetscCalloc1(nloc, &Fb); CHKERRQ(ierr);   // zeroed

    ElemUpdFint(fem, ec, Fb);


    /* Debug: print Fb for element 100 */
    // if (ec == 100) {
    //   PetscPrintf(PETSC_COMM_SELF, "Fb for ec=100 (dof=%" PetscInt_FMT ", v=%" PetscInt_FMT "):\n", dof, v);
    //   PetscInt nFb = dof * (v + 6);
    //   for (PetscInt idx = 0; idx < nFb && idx < (PetscInt)(sizeof(Fb)/sizeof(Fb[0])); idx++) {
    //     PetscPrintf(PETSC_COMM_SELF, " Fb[%3" PetscInt_FMT "] = % .12e\n", idx, (double)Fb[idx]);
    //   }
    // }

    for (i=0; i<(v+6); i++) {
      if (ibm->patch[16*ec+i]!=1000000) {
        node = ibm->patch[16*ec+i];

        FF[dof*node] += Fb[dof*i];
        FF[dof*node+1] += Fb[dof*i+1];
        FF[dof*node+2] += Fb[dof*i+2];
        
      //   if (ec == 100) {
      //   PetscPrintf(PETSC_COMM_SELF, "ec=100 node %" PetscInt_FMT "  FF_after  = [% .12e, % .12e, % .12e]\n",
      //               node,
      //               (double)FF[dof*node], (double)FF[dof*node+1], (double)FF[dof*node+2]);
      // }
        
      }
    } 
  }

  VecRestoreArray(fem->Fint, &FF); 
}

// PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec,
//                           const PetscReal Sm[3],   /* membrane stress resultants (your Sm[m]) */
//                           const PetscReal S[3],    /* bending stress resultants (your S[m]) */
//                           const PetscReal Em[3],   /* membrane strains (for IE) */
//                           const PetscReal Eb[3],   /* bending strains  (for IE) */
//                           PetscReal A0, PetscReal h0,
//                           PetscReal *Fb_out        /* output: element force vector (size 36 for reg, 3*(v+6) for irreg) */
//                           )
// {
//   PetscErrorCode ierr = 0;
//   IBMNodes    *ibm = fem->ibm;
//   ElemActData *ead = &fem->act_data.elem_act_data[ec];

//   const PetscInt dof = fem->dof; /* expect 3 */

//   PetscCheck(Fb_out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Fb_out is NULL");
//   PetscCheck(ead->g, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->g is NULL");

//   /* You asked: max qps small; not parallel now.
//      Note: Your Na/Nab are center-based so qp repeats; still loop qp if you want integration later. */
//   for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {

//     /* Grab basis computed in ElemUpdateG */
//     const struct Cmpnts ndx21 = ead->g[qp].Cov[0];
//     const struct Cmpnts ndx31 = ead->g[qp].Cov[1];
//     const struct Cmpnts nn    = ead->g[qp].Cov[2];

//     const struct Cmpnts gc1   = ead->g[qp].Cont[0];
//     const struct Cmpnts gc2   = ead->g[qp].Cont[1];

//     /* ------------------------- REGULAR PATCH ------------------------- */
//     if (ibm->ire[ec] == 0) {

//       /* Build x/y/z for Aaa, Abb, Aab only (needed for bending b1,b2,b3) */
//       PetscReal x[12], y[12], z[12];
//       PetscInt  node, nob = 1;

//       for (PetscInt i = 0; i < 12; i++) {
//         if (ibm->patch[16*ec + i] != 1000000) {
//           node = ibm->patch[16*ec + i];
//           x[i] = ibm->x_bp[node];
//           y[i] = ibm->y_bp[node];
//           z[i] = ibm->z_bp[node];
//         } else {
//           nob = 0;
//         }
//       }
//       PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
//                  "Regular patch missing control point ec=%" PetscInt_FMT, ec);

//       struct Cmpnts Aaa = {0,0,0}, Abb = {0,0,0}, Aab = {0,0,0};

//       for (PetscInt i = 0; i < 12; i++) {
//         Aaa.x += Nab_center[0][i]*x[i];
//         Aaa.y += Nab_center[0][i]*y[i];
//         Aaa.z += Nab_center[0][i]*z[i];

//         Abb.x += Nab_center[1][i]*x[i];
//         Abb.y += Nab_center[1][i]*y[i];
//         Abb.z += Nab_center[1][i]*z[i];

//         Aab.x += Nab_center[2][i]*x[i];
//         Aab.y += Nab_center[2][i]*y[i];
//         Aab.z += Nab_center[2][i]*z[i];
//       }

//       /* Membrane matrix */
//       PetscReal Bm[3][36];
//       for (PetscInt i = 0; i < 12; i++) {
//         Bm[0][3*i+0] = Na_center[0][i]*ndx21.x;   Bm[0][3*i+1] = Na_center[0][i]*ndx21.y;   Bm[0][3*i+2] = Na_center[0][i]*ndx21.z;
//         Bm[1][3*i+0] = Na_center[1][i]*ndx31.x;   Bm[1][3*i+1] = Na_center[1][i]*ndx31.y;   Bm[1][3*i+2] = Na_center[1][i]*ndx31.z;
//         Bm[2][3*i+0] = Na_center[0][i]*ndx31.x + Na_center[1][i]*ndx21.x;
//         Bm[2][3*i+1] = Na_center[0][i]*ndx31.y + Na_center[1][i]*ndx21.y;
//         Bm[2][3*i+2] = Na_center[0][i]*ndx31.z + Na_center[1][i]*ndx21.z;
//       }

//       /* Bending matrix */
//       PetscReal Bs[3][36];
//       for (PetscInt i = 0; i < 12; i++) {
//         struct Cmpnts ng1 = AMULT(Na_center[0][i], gc1);
//         struct Cmpnts ng2 = AMULT(Na_center[1][i], gc2);
//         struct Cmpnts ng  = PLUS(ng1, ng2);

//         PetscReal b1 = DOT(ng, Aaa);
//         PetscReal b2 = DOT(ng, Abb);
//         PetscReal b3 = DOT(ng, Aab);

//         Bs[0][3*i+0] = -(Nab_center[0][i] + b1)*nn.x;   Bs[0][3*i+1] = -(Nab_center[0][i] + b1)*nn.y;   Bs[0][3*i+2] = -(Nab_center[0][i] + b1)*nn.z;
//         Bs[1][3*i+0] = -(Nab_center[1][i] + b2)*nn.x;   Bs[1][3*i+1] = -(Nab_center[1][i] + b2)*nn.y;   Bs[1][3*i+2] = -(Nab_center[1][i] + b2)*nn.z;
//         Bs[2][3*i+0] = -2.0*(Nab_center[2][i] + b3)*nn.x; Bs[2][3*i+1] = -2.0*(Nab_center[2][i] + b3)*nn.y; Bs[2][3*i+2] = -2.0*(Nab_center[2][i] + b3)*nn.z;
//       }

//       /* membrane force */
//       PetscReal Fm[36];
//       for (PetscInt ii = 0; ii < 36; ii++) {
//         PetscReal sum = 0.0;
//         for (PetscInt m = 0; m < 3; m++) sum += Bm[m][ii] * Sm[m] * A0 * h0;
//         Fm[ii] = sum;
//       }

//       /* bending + membrane */
//       for (PetscInt ii = 0; ii < 36; ii++) {
//         PetscReal sum = 0.0;
//         for (PetscInt m = 0; m < 3; m++) sum += Bs[m][ii] * S[m] * A0 * PetscPowReal(h0, 3.0) / 12.0;
//         Fb_out[ii] += sum + Fm[ii];
//       }

//       /* IE/FC bookkeeping exactly like your code (optional to keep here) */
//       {
//         PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];
//         PetscReal sum = 0.;
//         for (PetscInt i = 0; i < 3; i++) {
//           sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*PetscPowReal(h0,3.0)/12.*Eb[i]*S[i]);
//           fem->FC[dof*ec + i] = Sm[i]*A0*h0 + A0*PetscPowReal(h0,2.0)/2.*S[i];
//         }
//         fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;
//       }

//       continue;
//     }

//     /* ------------------------- IRREGULAR PATCH ------------------------- */
//     if (ibm->ire[ec] == 1) {

//       const PetscInt v = ibm->val[ec];
//       PetscInt node, nob = 1;

//       for (PetscInt i = 0; i < (v+6); i++) {
//         if (ibm->patch[16*ec+i] == 1000000) { nob = 0; }
//       }
//       PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
//                  "Irregular patch missing control point ec=%" PetscInt_FMT, ec);

//       /* Rebuild X0 just to compute Aaa/Abb/Aab and INa/INab (same as your original).
//          Note: we are NOT recomputing ndx21/ndx31/nn/gc1/gc2: we reuse them from ElemUpdateG. */
//       PetscReal **X0 = (PetscReal**)malloc((v+6)*sizeof(PetscReal*));
//       PetscCheck(X0, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0");
//       for (PetscInt i = 0; i < (v+6); i++) {
//         X0[i] = (PetscReal*)malloc(3*sizeof(PetscReal));
//         PetscCheck(X0[i], PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed X0[i]");
//         node = ibm->patch[16*ec + i];
//         X0[i][0] = ibm->x_bp[node];
//         X0[i][1] = ibm->y_bp[node];
//         X0[i][2] = ibm->z_bp[node];
//       }

//       PetscReal w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v), 2.));

//       PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];

//       for (PetscInt i = 0; i < 12; i++)
//         for (PetscInt j = 0; j < (v+12); j++)
//           B2[i][j] = 0.;

//       for (PetscInt i = 0; i < (v+12); i++)
//         for (PetscInt j = 0; j < (v+6); j++)
//           B1[i][j] = 0.;

//       /* B2 */
//       B2[0][v+9]  = 1.;  B2[1][v+6]  = 1.;  B2[2][v+4]  = 1.;  B2[3][v+1]  = 1.;
//       B2[4][v+2]  = 1.;  B2[5][v+5]  = 1.;  B2[6][v]    = 1.;  B2[7][1]    = 1.;
//       B2[8][v+3]  = 1.;  B2[9][v-1]  = 1.;  B2[10][0]   = 1.;  B2[11][2]   = 1.;

//       /* B1 (same as in ElemUpdateG; keep identical) */
//       {
//         PetscInt j;
//         B1[0][0] = 1 - v*w;  for (j=0; j<v; j++) {B1[0][1+j] = w;}
//         B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
//         B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
//         if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
//         if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
//         B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
//         B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
//         B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
//         B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
//         B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
//         B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
//         B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
//         B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
//         B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
//         B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
//         B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
//         B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
//         B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
//         B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;
//       }

//       for (PetscInt i = 0; i < 12; i++) {
//         for (PetscInt j = 0; j < (v+6); j++) {
//           B3[i][j] = 0.;
//           for (PetscInt m = 0; m < (v+12); m++) {
//             B3[i][j] += B2[i][m]*B1[m][j];
//           }
//         }
//       }

//       PetscReal INa[2][v+6], INab[3][v+6];
//       for (PetscInt j = 0; j < (v+6); j++) {
//         PetscReal sum1=0., sum2=0., sum3=0., sum4=0., sum5=0.;
//         for (PetscInt i = 0; i < 12; i++) {
//           sum1 += B3[i][j]*Na_center[0][i];
//           sum2 += B3[i][j]*Na_center[1][i];
//           sum3 += B3[i][j]*Nab_center[0][i];
//           sum4 += B3[i][j]*Nab_center[1][i];
//           sum5 += B3[i][j]*Nab_center[2][i];
//         }
//         INa[0][j]  = -2*sum1;
//         INa[1][j]  = -2*sum2;
//         INab[0][j] =  4*sum3;
//         INab[1][j] =  4*sum4;
//         INab[2][j] =  4*sum5;
//       }

//       /* Aaa/Abb/Aab from INab */
//       struct Cmpnts Aaa = {0,0,0}, Abb = {0,0,0}, Aab = {0,0,0};
//       for (PetscInt i = 0; i < (v+6); i++) {
//         Aaa.x += INab[0][i]*X0[i][0];  Aaa.y += INab[0][i]*X0[i][1];  Aaa.z += INab[0][i]*X0[i][2];
//         Abb.x += INab[1][i]*X0[i][0];  Abb.y += INab[1][i]*X0[i][1];  Abb.z += INab[1][i]*X0[i][2];
//         Aab.x += INab[2][i]*X0[i][0];  Aab.y += INab[2][i]*X0[i][1];  Aab.z += INab[2][i]*X0[i][2];
//       }

//       /* Membrane matrix */
//       PetscReal Bm[3][3*(v+6)];
//       for (PetscInt i = 0; i < (v+6); i++) {
//         Bm[0][3*i+0] = INa[0][i]*ndx21.x;   Bm[0][3*i+1] = INa[0][i]*ndx21.y;   Bm[0][3*i+2] = INa[0][i]*ndx21.z;
//         Bm[1][3*i+0] = INa[1][i]*ndx31.x;   Bm[1][3*i+1] = INa[1][i]*ndx31.y;   Bm[1][3*i+2] = INa[1][i]*ndx31.z;
//         Bm[2][3*i+0] = INa[0][i]*ndx31.x + INa[1][i]*ndx21.x;
//         Bm[2][3*i+1] = INa[0][i]*ndx31.y + INa[1][i]*ndx21.y;
//         Bm[2][3*i+2] = INa[0][i]*ndx31.z + INa[1][i]*ndx21.z;
//       }

//       /* Bending matrix (you called it B4) */
//       PetscReal B4[3][3*(v+6)];
//       for (PetscInt i = 0; i < (v+6); i++) {
//         struct Cmpnts ng1 = AMULT(INa[0][i], gc1);
//         struct Cmpnts ng2 = AMULT(INa[1][i], gc2);
//         struct Cmpnts ng  = PLUS(ng1, ng2);

//         PetscReal b1 = DOT(ng, Aaa);
//         PetscReal b2 = DOT(ng, Abb);
//         PetscReal b3 = DOT(ng, Aab);

//         B4[0][3*i+0] = -(INab[0][i] + b1)*nn.x;   B4[0][3*i+1] = -(INab[0][i] + b1)*nn.y;   B4[0][3*i+2] = -(INab[0][i] + b1)*nn.z;
//         B4[1][3*i+0] = -(INab[1][i] + b2)*nn.x;   B4[1][3*i+1] = -(INab[1][i] + b2)*nn.y;   B4[1][3*i+2] = -(INab[1][i] + b2)*nn.z;
//         B4[2][3*i+0] = -2.0*(INab[2][i] + b3)*nn.x; B4[2][3*i+1] = -2.0*(INab[2][i] + b3)*nn.y; B4[2][3*i+2] = -2.0*(INab[2][i] + b3)*nn.z;
//       }

//       /* membrane force */
//       PetscReal Fm[3*(v+6)];
//       for (PetscInt ii = 0; ii < 3*(v+6); ii++) {
//         PetscReal sum = 0.0;
//         for (PetscInt m = 0; m < 3; m++) sum += Bm[m][ii] * Sm[m] * A0 * h0;
//         Fm[ii] = sum;
//       }

//       for (PetscInt ii = 0; ii < 3*(v+6); ii++) {
//         PetscReal sum = 0.0;
//         for (PetscInt m = 0; m < 3; m++) sum += B4[m][ii] * S[m] * A0 * PetscPowReal(h0, 3.0) / 12.0;
//         Fb_out[ii] += sum + Fm[ii];
//       }

//       /* IE/FC bookkeeping (same idea; keep if you want) */
//       {
//         PetscInt n1e = ibm->nv1[ec], n2e = ibm->nv2[ec], n3e = ibm->nv3[ec];
//         PetscReal sum = 0.;
//         for (PetscInt i = 0; i < 3; i++) {
//           sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*PetscPowReal(h0,3.0)/12.*Eb[i]*S[i]);
//           fem->FC[dof*ec + i] = Sm[i]*A0*h0 + A0*PetscPowReal(h0,2.0)/2.*S[i];
//         }
//         fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;
//       }

//       for (PetscInt i = 0; i < (v+6); i++) free(X0[i]);
//       free(X0);

//       continue;
//     }

//     SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
//             "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
//             ibm->ire[ec], ec);
//   }

//   return ierr;
// }


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
PetscErrorCode UpdateElements(FE *fem, ElemFunc func)
{
    PetscErrorCode ierr;
    PetscInt n_elems = fem->ibm->n_elmt;
    for (PetscInt ec = 0; ec < n_elems; ec++)
    {
        ierr = func(fem, ec);
        if (ierr)
            return ierr; /* Standard PETSc error propagation */
    }

    return 0;
}
