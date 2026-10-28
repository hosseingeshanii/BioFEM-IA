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
#include "manufactured_active_strain.h"
#include "cuda_bridge.h"
#include "dmplex_geom.h"
#include <petscsys.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


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
PetscInt  fa_pezzuto = 0;      /* -fa_pezzuto 1: isochoric Fa of Pezzuto & Ambrosi 2014 eq. 11 (with -cart_fib_act 1) */
PetscInt  fa_nitti_plate = 0;  /* -fa_nitti_plate 1: Nitti eq. 44-47 Fa on a flat Lx x Ly patch */
PetscReal fa_nitti_a1 = 1.0, fa_nitti_a2 = 1.0, fa_nitti_Lx = 1.0, fa_nitti_Ly = 1.0;
/* -material_model 1: Holzapfel-Ogden (Wang et al. 2013, IJNMBE 29:83, eq. 3 / Table I, Pa), evaluated at Ce,
 * with s0 = a3 (sheet through the wall). -ho_incompressible 0: exactly incompressible shell, p eliminated
 * point by point (option iii; Kiendl et al. 2015 sec. 5.1 extended to active strain). See ElemHOStress_. */
PetscInt  material_model = 0;      /* 0 = neo-Hookean (default), 1 = Holzapfel-Ogden */
PetscInt  ho_incompressible = 0;   /* 0 = option (iii); (i)/(ii) not implemented */
PetscInt  ho_tension_only = 1;     /* fiber/sheet terms only for I4 > 1 */
PetscInt  ho_sheet = 1;            /* sheet (I4s) and fiber-sheet (I8fs) terms */
PetscReal ho_a = 236.2, ho_b = 10.810, ho_af = 20037.0, ho_bf = 14.154,
          ho_as = 3724.5, ho_bs = 5.1645, ho_afs = 410.8, ho_bfs = 11.300;
PetscReal t_solve = -1.0;   /* >= 0: the time used for gamma(t) instead of ti*dt (main.c -adaptive_substeps) */
PetscInt ghost_force_map = 1; /* -ghost_force_map 0 disables mapping Fint on ghost nodes back to real nodes (old behaviour) */
static PetscInt cuda_metric_tensor = 0;

/* External globals from the rest of the code */
extern PetscInt dof, curvature, ConstitutiveLawNonLinear;
extern PetscInt ti, manufactured, manufactured_fexternal_export, prescribed_force_field;
extern PetscInt tistart, tisteps;
extern PetscInt dmplex_geom_process, cuda_process, kokkos_process;
extern PetscReal E, mu, rho, h0, dt;
extern PetscReal initial_elas, initial_poisson;
extern PetscErrorCode FDynamic(FE *fem);
extern PetscErrorCode xAccVel(FE *fem);
extern PetscErrorCode LocationOut(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);

extern char           in_dir[256];
extern char           out_dir[256];



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
    PetscFunctionBeginUser;
    // UserCtx  *userctx = &fem->userctx;    

    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-num_gaussian_quad_points", &(fem->act_data.n_qp), PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-C33_subitr_nums", &(fem->act_data.C33_subitr_nums), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fiber_transmural_twist_deg", &(fem->act_data.fiber_twist_deg), PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-cart_fib_act", &cart_fib_act, PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-ghost_force_map", &ghost_force_map, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-fa_pezzuto", &fa_pezzuto, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-fa_nitti_plate", &fa_nitti_plate, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fa_nitti_a1", &fa_nitti_a1, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fa_nitti_a2", &fa_nitti_a2, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fa_nitti_Lx", &fa_nitti_Lx, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fa_nitti_Ly", &fa_nitti_Ly, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-muscle_act_gamma", &(fem->act_data.muscle_act_params.gamma), PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-gamma_ramp_type",  &(fem->act_data.muscle_act_params.gamma_ramp_type), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gamma_T",          &(fem->act_data.muscle_act_params.gamma_T), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gamma_tau_up",     &(fem->act_data.muscle_act_params.gamma_tau_up), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gamma_tau_down",   &(fem->act_data.muscle_act_params.gamma_tau_down), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gamma_apd",        &(fem->act_data.muscle_act_params.gamma_apd), PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-manufactured_gamma0", &manufactured_gamma0, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-manufactured_T", &manufactured_T, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-bulk_modulus", &(fem->act_data.K), PETSC_NULL);
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-cuda_metric_tensor", &cuda_metric_tensor, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-material_model", &material_model, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-ho_incompressible", &ho_incompressible, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-ho_tension_only", &ho_tension_only, PETSC_NULL);
    PetscOptionsGetInt (PETSC_NULL, PETSC_NULL, "-ho_sheet", &ho_sheet, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_a",   &ho_a,   PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_b",   &ho_b,   PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_af",  &ho_af,  PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_bf",  &ho_bf,  PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_as",  &ho_as,  PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_bs",  &ho_bs,  PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_afs", &ho_afs, PETSC_NULL);
    PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-ho_bfs", &ho_bfs, PETSC_NULL);
    if (material_model == 1) {
        PetscCheck(cart_fib_act, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-material_model 1 (Holzapfel-Ogden) needs fibers: -cart_fib_act 1");
        PetscCheck(ho_incompressible == 0, PETSC_COMM_WORLD, PETSC_ERR_SUP, "-ho_incompressible %d not implemented; only 0 (exactly incompressible, option iii)", (int)ho_incompressible);
        PetscPrintf(PETSC_COMM_WORLD, "[material] Holzapfel-Ogden, exactly incompressible (C33 from J=1, p from S33=0): "
                    "a=%g b=%g af=%g bf=%g as=%g bs=%g afs=%g bfs=%g tension_only=%d sheet=%d (-mu, -bulk_modulus, -C33_subitr_nums unused)\n",
                    (double)ho_a, (double)ho_b, (double)ho_af, (double)ho_bf, (double)ho_as, (double)ho_bs, (double)ho_afs, (double)ho_bfs,
                    (int)ho_tension_only, (int)ho_sheet);
    }

    fem->act_data.mu = mu;
    PetscFunctionReturn(0);
}

PetscErrorCode ActDataAllocate(FE *fem)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr;
  ActData *act = &fem->act_data;
  PetscInt nelem = fem->ibm->n_elmt;
  PetscInt n_qp  = act->n_qp;

  
  act->theta = NULL;
  act->w = NULL;
  act->elem_act_data = NULL;

  ierr = PetscMalloc1(n_qp, &act->theta); CHKERRQ(ierr);
  ierr = PetscMalloc1(n_qp, &act->w);     CHKERRQ(ierr);

  ierr = PetscMalloc1(nelem, &act->elem_act_data); CHKERRQ(ierr);
  /* Ensure all pointers inside each ElemActData start as NULL */
  ierr = PetscMemzero(act->elem_act_data, nelem * sizeof(*act->elem_act_data)); CHKERRQ(ierr);

  /* The geometry-kernel-only benchmark paths (RunDMPlexGeomProcesses[Cuda],
     RunCudaProcesses, RunKokkosProcesses) run standalone and exit -- they
     never reach ElemC33Solve/the real elasticity solve, so they only ever
     touch geom/geom0/gm/gm0/g/g0 below (ElemUpdateG, called from
     RunLocalGeometry_, writes g/g0 as well as gm/gm0). The other 8 per-qp
     arrays (Fa, Fa_inv, C, C_inv, Ce, Ce_inv, Se, S, CCe, CC) are solver-only
     state that was previously allocated unconditionally for every element
     regardless of which path was running -- at mesh sizes in the millions of
     elements that is the dominant cost of Init() (confirmed via
     PetscMemoryGetCurrentUsage tracing: ~6GB/rank of a ~7GB total, for a mesh
     whose actual replicated ibm-> arrays only account for ~0.2GB). Real
     solves (none of these three flags set) still get every array, unchanged. */
  PetscBool skip_solver_arrays = (dmplex_geom_process || cuda_process || kokkos_process)
                                 ? PETSC_TRUE : PETSC_FALSE;

  for (PetscInt ec = 0; ec < nelem; ec++) {
    ElemActData *ead = &act->elem_act_data[ec];

    if (!skip_solver_arrays) {
      /* Kinematics */
      ierr = PetscMalloc1(n_qp, &ead->fib);    CHKERRQ(ierr);
      ead->fib_set = PETSC_FALSE;
      ierr = PetscMalloc1(n_qp, &ead->Fa);     CHKERRQ(ierr);
      ierr = PetscMalloc1(n_qp, &ead->Fa_inv); CHKERRQ(ierr);

      ierr = PetscMalloc1(n_qp, &ead->C);     CHKERRQ(ierr);
      ierr = PetscMalloc1(n_qp, &ead->C_inv); CHKERRQ(ierr);

      ierr = PetscMalloc1(n_qp, &ead->Ce);     CHKERRQ(ierr);
      ierr = PetscMalloc1(n_qp, &ead->Ce_inv); CHKERRQ(ierr);

      /* Stress */
      ierr = PetscMalloc1(n_qp, &ead->Se); CHKERRQ(ierr);
      ierr = PetscMalloc1(n_qp, &ead->S);  CHKERRQ(ierr);

      /* Tangents */
      ierr = PetscMalloc1(n_qp, &ead->CCe); CHKERRQ(ierr);
      ierr = PetscMalloc1(n_qp, &ead->CC);  CHKERRQ(ierr);
    }

    /* Bases + Metrics -- needed by the geometry-kernel benchmarks too
       (ElemUpdateG writes g/g0/gm/gm0 together; the CUDA metric-tensor
       kernel writes the same fields), so always allocated. */
    ierr = PetscMalloc1(n_qp, &ead->g);   CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->g0);  CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->gm);  CHKERRQ(ierr);
    ierr = PetscMalloc1(n_qp, &ead->gm0); CHKERRQ(ierr);

    /* Midsurface geometry cache (size 1) -- always needed. */
    ierr = PetscMalloc1(1, &ead->geom);  CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &ead->geom0); CHKERRQ(ierr);

    /* Must ensure internal pointers inside SubdivGeomQP are NULL */
    ierr = PetscMemzero(&ead->geom[0],  sizeof(SubdivGeomQP)); CHKERRQ(ierr);
    ierr = PetscMemzero(&ead->geom0[0], sizeof(SubdivGeomQP)); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
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

    case 6:
        fem->act_data.theta[0] = -0.9324695142031521;
        fem->act_data.theta[1] = -0.6612093864662645;
        fem->act_data.theta[2] = -0.2386191860831969;
        fem->act_data.theta[3] =  0.2386191860831969;
        fem->act_data.theta[4] =  0.6612093864662645;
        fem->act_data.theta[5] =  0.9324695142031521;
        fem->act_data.w[0] = fem->act_data.w[5] = 0.1713244923791704;
        fem->act_data.w[1] = fem->act_data.w[4] = 0.3607615730481386;
        fem->act_data.w[2] = fem->act_data.w[3] = 0.4679139345726910;
        break;

    case 7:
        fem->act_data.theta[0] = -0.9491079123427585;
        fem->act_data.theta[1] = -0.7415311855993945;
        fem->act_data.theta[2] = -0.4058451513773972;
        fem->act_data.theta[3] =  0.0;
        fem->act_data.theta[4] =  0.4058451513773972;
        fem->act_data.theta[5] =  0.7415311855993945;
        fem->act_data.theta[6] =  0.9491079123427585;
        fem->act_data.w[0] = fem->act_data.w[6] = 0.1294849661688697;
        fem->act_data.w[1] = fem->act_data.w[5] = 0.2797053914892766;
        fem->act_data.w[2] = fem->act_data.w[4] = 0.3818300505051189;
        fem->act_data.w[3] = 0.4179591836734694;
        break;

    default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "Number of Gauss points must be between 1 and 7");
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode SubdivGeomDestroy_(SubdivGeomQP *G)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr = 0;
  if (!G) PetscFunctionReturn(0);

  ierr = PetscFree(G->INa0);  CHKERRQ(ierr);
  ierr = PetscFree(G->INa1);  CHKERRQ(ierr);
  ierr = PetscFree(G->INab0); CHKERRQ(ierr);
  ierr = PetscFree(G->INab1); CHKERRQ(ierr);
  ierr = PetscFree(G->INab2); CHKERRQ(ierr);

  PetscMemzero(G, sizeof(*G));
  PetscFunctionReturn(0);
}

static PetscErrorCode ElemActDataGeomDestroy_(ElemActData *ead)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr;
  if (!ead) PetscFunctionReturn(0);

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
  PetscFunctionReturn(0);
}


PetscErrorCode ActDataDestroy(FE *fem)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr;
  ActData *act = &fem->act_data;
  PetscInt nelem = fem->ibm->n_elmt;

  /* If elem_act_data was never allocated, just free global arrays */
  if (!act->elem_act_data) {
    ierr = PetscFree(act->theta); CHKERRQ(ierr);
    ierr = PetscFree(act->w);     CHKERRQ(ierr);
    act->theta = NULL;
    act->w = NULL;
    PetscFunctionReturn(0);
  }

  for (PetscInt ec = 0; ec < nelem; ec++) {
    ElemActData *ead = &act->elem_act_data[ec];

    ierr = ElemActDataGeomDestroy_(ead); CHKERRQ(ierr);

    ierr = PetscFree(ead->fib);    CHKERRQ(ierr);
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

  PetscFunctionReturn(0);
}


static void PrintCmpnts_(const char *name, Cmpnts a)
{
PetscPrintf(PETSC_COMM_SELF, " %-8s = (% .6e, % .6e, % .6e)\n", name, a.x, a.y, a.z);
}


static PetscErrorCode PrintSubdivGeomQP_(const char *label, PetscInt ec, const SubdivGeomQP *G)
{
PetscFunctionBeginUser;
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


PetscFunctionReturn(ierr);
}

static PetscErrorCode DebugPrintGeomForElement(FE *fem, PetscInt ec)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr = 0;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->geom0, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom0 is NULL");
  PetscCheck(ead->geom,  PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");

  ierr = PrintSubdivGeomQP_("GEOM0 (reference) geom0[0]", ec, &ead->geom0[0]); CHKERRQ(ierr);
  ierr = PrintSubdivGeomQP_("GEOM  (current)   geom[0]",  ec, &ead->geom[0]);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Helper: ensure irregular arrays are allocated to length nen */
static PetscErrorCode SubdivGeomEnsureIrregularArrays_(SubdivGeomQP *G, PetscInt nen)
{
  PetscFunctionBeginUser;
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

  PetscFunctionReturn(0);
}

/* The main geometry update.
   Call this once per element per configuration (before stresses or fint). */
static PetscErrorCode ElemUpdateGeomSubdivFromCoords_(
    FE *fem, PetscInt ec,
    const PetscReal *xb, const PetscReal *yb, const PetscReal *zb,
    SubdivGeomQP *G)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr = 0;
  IBMNodes    *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];

  PetscCheck(ibm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");
  PetscCheck(ead, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead is NULL");
  PetscCheck(ead->geom, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ead->geom is NULL");

//   SubdivGeomQP *G = &ead->geom[0];  /* midsurface cache */

  /* Cap elements (ec >= n_elmt_base) close the apex topologically but their
     12-node subdivision stencils include topologically inconsistent neighbors
     (ring-1 nodes mixed with inner-cap nodes), producing garbage gm0.  Skip
     subdivision entirely and use the CST triangle geometry instead. */
  if (ibm->n_elmt_base > 0 && ec >= ibm->n_elmt_base) {
    const PetscInt n1 = ibm->nv1[ec], n2 = ibm->nv2[ec], n3 = ibm->nv3[ec];
    G->ndx21.x = xb[n2]-xb[n1];  G->ndx21.y = yb[n2]-yb[n1];  G->ndx21.z = zb[n2]-zb[n1];
    G->ndx31.x = xb[n3]-xb[n1];  G->ndx31.y = yb[n3]-yb[n1];  G->ndx31.z = zb[n3]-zb[n1];
    G->nn  = UNIT(CROSS(G->ndx21, G->ndx31));
    G->gc1 = CROSS(G->ndx31, G->nn);
    G->gc1 = AMULT(1.0 / DOT(G->ndx21, G->gc1), G->gc1);
    G->gc2 = CROSS(G->nn, G->ndx21);
    G->gc2 = AMULT(1.0 / DOT(G->ndx31, G->gc2), G->gc2);
    G->Aaa.x = G->Aaa.y = G->Aaa.z = 0.0;
    G->Abb.x = G->Abb.y = G->Abb.z = 0.0;
    G->Aab.x = G->Aab.y = G->Aab.z = 0.0;
    G->is_irregular = 0;
    G->v   = 0;
    G->nen = 12;
    if (G->INa0) {
      ierr = PetscFree(G->INa0);  CHKERRQ(ierr);
      ierr = PetscFree(G->INa1);  CHKERRQ(ierr);
      ierr = PetscFree(G->INab0); CHKERRQ(ierr);
      ierr = PetscFree(G->INab1); CHKERRQ(ierr);
      ierr = PetscFree(G->INab2); CHKERRQ(ierr);
      G->INa0 = G->INa1 = G->INab0 = G->INab1 = G->INab2 = NULL;
    }
    PetscFunctionReturn(0);
  }

  /* ============================================================
     REGULAR PATCH
     ============================================================ */
  if (ibm->ire[ec] == 0) {
    PetscReal x[12] = {0}, y[12] = {0}, z[12] = {0};
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

    /* CST flat/zero-curvature fallback removed -- boundary elements now get
       real ghost-node stencil neighbors (lv_geometry_unstructured.c +
       Patch()'s outer-ring fixup in bending.c), so nob should always be 1.
       Loud error instead of silently substituting wrong geometry if a gap
       remains (e.g. an element touching 2 boundary edges). */
    PetscCheck(nob, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "ElemGeom ec=%d: regular-patch stencil still incomplete after "
               "ghost-node resolution", (int)ec);

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

    PetscFunctionReturn(0);
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

    if (!nob) {
      /* CST/zero-force fallback removed -- see the regular-patch branch
         above for why (ghost nodes should now resolve every slot). Loud
         error instead of silently dropping this element's force. */
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "ElemGeom ec=%d: irregular-patch stencil still incomplete "
               "after ghost-node resolution", (int)ec);
      PetscFunctionReturn(0);
    }

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

    PetscFunctionReturn(0);
  }

  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
          "Unknown ibm->ire[ec]=%" PetscInt_FMT " for ec=%" PetscInt_FMT,
          ibm->ire[ec], ec);
}


PetscErrorCode ElemUpdateGeomSubdiv(FE *fem, PetscInt ec)   /* current */
{
  PetscFunctionBeginUser;
  IBMNodes *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];
  PetscFunctionReturn(ElemUpdateGeomSubdivFromCoords_(fem, ec, ibm->x_bp,  ibm->y_bp,  ibm->z_bp,  &ead->geom[0]));
}

PetscErrorCode ElemUpdateGeom0Subdiv(FE *fem, PetscInt ec)  /* reference */
{
  PetscFunctionBeginUser;
  IBMNodes *ibm = fem->ibm;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];
  PetscFunctionReturn(ElemUpdateGeomSubdivFromCoords_(fem, ec, ibm->x_bp0, ibm->y_bp0, ibm->z_bp0, &ead->geom0[0]));
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

  /* a3 = (a1 x a2)/|a1 x a2| is a unit vector, so a3,alpha must be tangent:
   *   a3,alpha = (I - a3 (x) a3) (a1 x a2),alpha / |a1 x a2|.
   * Without removing the normal part, g_alpha = a_alpha + theta3 a3,alpha gets a component
   * along a3, so g0_alpha3 != 0 off the midsurface while C_alpha3 = 0 (Kirchhoff-Love):
   * the reference state is strained for theta3 != 0 (n_qp > 1). No effect at theta3 = 0. */
  const struct Cmpnts n = AMULT(1.0 / size_a3, CROSS(a1, a2));
  const PetscReal d1 = DOT(*a3_1, n), d2 = DOT(*a3_2, n);
  a3_1->x -= d1 * n.x;  a3_1->y -= d1 * n.y;  a3_1->z -= d1 * n.z;
  a3_2->x -= d2 * n.x;  a3_2->y -= d2 * n.y;  a3_2->z -= d2 * n.z;

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
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < 3; i++) {
    for (PetscInt j = 0; j < 3; j++) {
      gCov[i][j] = DOT(g_cov[i], g_cov[j]);
    }
  }
  PetscFunctionReturn(0);
}

/* Helper: Compute contravariant basis vectors from metric tensor inverse
 * g^i = g^ij * g_j  (raised indices)
 */
static PetscErrorCode ComputeContravariantBasis(const PetscReal gInv[3][3],
                                                const Cmpnts g_cov[3],
                                                Cmpnts g_cont[3])
{
  PetscFunctionBeginUser;
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
  PetscFunctionReturn(0);
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
PetscFunctionBeginUser;
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


PetscFunctionReturn(ierr);
}


/* Update ead->g, ead->g0, ead->gm, ead->gm0 for all qp using theta[qp]. */
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec)
{
  PetscFunctionBeginUser;
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
  const PetscReal hhalf = 0.5 * fem->ibm->h0_e[ec];   /* reference thickness of this element */

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

  PetscFunctionReturn(ierr);
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
    PetscFunctionBeginUser;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n========== Element %d: Right Cauchy-Green Tensor ==========\n", ec);
    
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "\n--- QP %d ---\n", qp);
        PrintElem2DTens("C (Right Cauchy-Green tensor)", &ead->C[qp]);
        PrintElem2DTens("C_inv (Inverse)", &ead->C_inv[qp]);
    }
    
    PetscFunctionReturn(0);
}

/* Print elastic Cauchy-Green tensor Ce and Ce_inv for element ec */
static PetscErrorCode PrintElemCe(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    if (ec != 100) PetscFunctionReturn(0);
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElasCGDefTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        PrintElem2DTens("  Ce (Elastic Cauchy-Green tensor)", &ead->Ce[qp]);
        PrintElem2DTens("  Ce_inv (Inverse)", &ead->Ce_inv[qp]);
    }
    
    PetscFunctionReturn(0);
}

/* Print elastic stress Se for element ec */
static PetscErrorCode PrintElemSe(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    if (ec != 100) PetscFunctionReturn(0);
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElasStress ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        PrintElem2DTens("  Se (Elastic 2nd Piola-Kirchhoff stress)", &ead->Se[qp]);
    }
    
    PetscFunctionReturn(0);
}

/* Print total stress S for element ec */
static PetscErrorCode PrintElemS(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    if (ec != 100) PetscFunctionReturn(0);
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    // PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemTotStress ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d:\n", qp);
        // PrintElem2DTens("  S (Total 2nd Piola-Kirchhoff stress)", &ead->S[qp]);
        PetscPrintf(PETSC_COMM_WORLD, "  S^[2][2] = %f\n", (double)ead->S[qp].Cont[2][2]);
    }
    
    PetscFunctionReturn(0);
}

/* Print elastic material tangent CCe for element ec */
static PetscErrorCode PrintElemCCe(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    if (ec != 100) PetscFunctionReturn(0);
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemElsTangMatTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d: CCe[2][2][2][2] = %f\n", qp, (double)ead->CCe[qp].Cont[2][2][2][2]);
    }
    
    PetscFunctionReturn(0);
}

/* Print total material tangent CC for element ec */
static PetscErrorCode PrintElemCC(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    if (ec != 100) PetscFunctionReturn(0);
    
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    
    PetscPrintf(PETSC_COMM_WORLD, "\n--- After ElemTotTangMatTens ---\n");
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
        PetscPrintf(PETSC_COMM_WORLD, "QP %d: CC[2][2][2][2] = %f\n", qp, (double)ead->CC[qp].Cont[2][2][2][2]);
    }
    
    PetscFunctionReturn(0);
}

/**
 * Return the active-strain gamma at the current time step.
 *
 * gamma_ramp_type = 0 (default): constant gamma.
 * gamma_ramp_type = 1: full sin^2 cardiac cycle (systole + diastole).
 *   gamma(t) = gamma_peak * sin^2(pi * t / T)
 *   T = gamma_T if > 0, otherwise tisteps * dt (full simulation = one cycle).
 *   0 at t=0, peak at t=T/2, 0 again at t=T.
 * gamma_ramp_type = 2: half sin^2 ramp — systole only (monotonic 0 → peak).
 *   gamma(t) = gamma_peak * sin^2(pi/2 * t / T)
 *   T = gamma_T if > 0, otherwise tisteps * dt.
 *   Reaches gamma_peak at t=T; no return path, no unloading bifurcation.
 * gamma_ramp_type = 3: action-potential-like double-logistic pulse. Modeled
 *   after the shape of a prescribed transmembrane potential v(t) (fast
 *   depolarization upstroke, plateau, slower repolarization decay) run
 *   through a saturating strain-activation map, in the spirit of Nobile,
 *   Quarteroni & Ruiz-Baier (2012), Int. J. Numer. Meth. Biomed. Engng
 *   28:52-71, eq. (2.8)-(2.10) -- gamma there is an active-STRAIN quantity
 *   driven by v (and, in their richer variant, by an ODE in calcium), not
 *   an active-stress tension, matching this solver's Fa = I - gamma(f⊗f)
 *   formulation. Here v(t) itself is a simple prescribed pulse (no PDE),
 *   giving a real contraction-then-relaxation cycle instead of the type 1/2
 *   sin^2 ramps: sin^2 is symmetric about its peak by construction, whereas
 *   a real cardiac twitch rises much faster than it relaxes.
 *     phi_up(t)   = 1 / (1 + exp(-(t - 4*tau_up)   / tau_up))
 *     phi_down(t) = 1 / (1 + exp( (t - (4*tau_up+APD)) / tau_down))
 *     gamma(t) = gamma_peak * phi_up(t) * phi_down(t)
 *   The 4*tau_up offset just keeps phi_up ~0 at t=0 (so activation doesn't
 *   start already-partway-up); APD is measured from the upstroke's
 *   inflection point. gamma_tau_up/gamma_tau_down/gamma_apd control shape;
 *   gamma_T is reused as the cycle length only insofar as callers may want
 *   to repeat this pulse periodically (not done automatically here -- one
 *   pulse per run, matching how types 1/2 are one ramp per run too).
 */
/* delay > 0 shifts the ramp's local clock backward -- element activates
 * "late" by that much simulated time, modeling a propagating depolarization
 * wavefront (see -lv_gamma_wave in lv_geometry_unstructured.c) instead of
 * every element ramping up from t=0 simultaneously. delay=0 recovers the
 * original instantaneous-everywhere behavior exactly. */
static PetscReal GammaOfTimeDelayed(const MuscleActParams *p, PetscReal delay)
{
    if (p->gamma_ramp_type == 0) return p->gamma;
    PetscReal t = ((t_solve >= 0.0) ? t_solve : (PetscReal)ti * dt) - delay;
    if (p->gamma_ramp_type == 3) {
        PetscReal tau_up   = (p->gamma_tau_up   > 0.0) ? p->gamma_tau_up   : 0.02;
        PetscReal tau_down = (p->gamma_tau_down > 0.0) ? p->gamma_tau_down : 0.08;
        PetscReal apd      = (p->gamma_apd      > 0.0) ? p->gamma_apd      : 0.30;
        PetscReal t_up     = 4.0 * tau_up;
        PetscReal t_down   = t_up + apd;
        PetscReal phi_up   = 1.0 / (1.0 + PetscExpReal(-(t - t_up)   / tau_up));
        PetscReal phi_down = 1.0 / (1.0 + PetscExpReal( (t - t_down) / tau_down));
        return p->gamma * phi_up * phi_down;
    }
    PetscReal T = (p->gamma_T > 0.0) ? p->gamma_T : (PetscReal)tisteps * dt;
    if (t <= 0.0) return 0.0;
    if (p->gamma_ramp_type == 1) {
        PetscReal s = PetscSinReal(PETSC_PI * t / T);
        return p->gamma * s * s;
    }
    if (p->gamma_ramp_type == 4) {
        /* linear up to the peak at T/2, linear back to 0 at T (Pezzuto & Ambrosi 2014) */
        PetscReal x = t / T;
        if (x >= 1.0) return 0.0;
        return p->gamma * (1.0 - PetscAbsReal(2.0 * x - 1.0));
    }
    if (p->gamma_ramp_type == 5) {
        /* linear up to the peak at T, then held */
        return p->gamma * PetscMin(t / T, 1.0);
    }
    /* type 2: half-sin^2 — smooth monotonic ramp, systole only */
    PetscReal s = PetscSinReal(PETSC_PI * 0.5 * t / T);
    return p->gamma * s * s;
}

static PetscReal GammaOfTime(const MuscleActParams *p)
{
    return GammaOfTimeDelayed(p, 0.0);
}

/* Public wrapper around GammaOfTimeDelayed for output/diagnostic use (e.g.
 * io.c's Output(), so VTK files can report the actual per-element scalar
 * gamma driving activation this timestep, including the wavefront delay). */
PetscReal GammaOfTimeCurrent(FE *fem, PetscInt ec)
{
    PetscReal delay = fem->ibm->activation_delay ? fem->ibm->activation_delay[ec] : 0.0;
    return GammaOfTimeDelayed(&fem->act_data.muscle_act_params, delay);
}

/* Fill ead->fib[qp] (Cartesian unit fiber per thickness qp) for element ec, once.
 * Uniform (default): every qp gets the mid-wall fiber ibm->n_fib[ec].
 * -fiber_transmural_twist_deg T: rotate the mid-wall fiber about the outward wall normal e_n
 *   (ibm->n_fib_out[ec], set by the LV generators) so that the helix angle is
 *   alpha(xi) = alpha_mid - (T/2) xi,  xi = +1 at the epicardium, -1 at the endocardium,
 *   with xi = theta[qp] * sign(a3 . e_n) (theta in [-1,1] along the shell director a3).
 *   With e_c = e_n x e_l, f x e_n = e_l for f = e_c, so
 *   f(xi) = cos(d) f_mid + sin(d) (f_mid x e_n), d = alpha(xi) - alpha_mid. */
static PetscErrorCode ElemQPFibers_(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    IBMNodes    *ibm = fem->ibm;
    ActData     *act = &fem->act_data;
    ElemActData *ead = &act->elem_act_data[ec];
    PetscReal f[3] = {ibm->n_fib[ec].x, ibm->n_fib[ec].y, ibm->n_fib[ec].z};
    PetscReal nf = PetscSqrtReal(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    PetscCheck(nf > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "zero fiber vector at element %d", (int)ec);
    for (PetscInt i = 0; i < 3; i++) f[i] /= nf;

    PetscReal T = act->fiber_twist_deg * PETSC_PI / 180.0;
    PetscReal en[3] = {ibm->n_fib_out[ec].x, ibm->n_fib_out[ec].y, ibm->n_fib_out[ec].z};
    if (T != 0.0) {
        PetscReal ne = PetscSqrtReal(en[0]*en[0] + en[1]*en[1] + en[2]*en[2]);
        PetscCheck(ne > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                   "-fiber_transmural_twist_deg needs the outward fiber-frame normal (LV meshes only); none at element %d", (int)ec);
        for (PetscInt i = 0; i < 3; i++) en[i] /= ne;
    }
    const PetscReal fxn[3] = {f[1]*en[2] - f[2]*en[1], f[2]*en[0] - f[0]*en[2], f[0]*en[1] - f[1]*en[0]};
    for (PetscInt qp = 0; qp < act->n_qp; qp++) {
        PetscReal d = 0.0;
        if (T != 0.0) {
            const struct Cmpnts a3 = ead->g0[qp].Cov[2];
            PetscReal s  = (a3.x*en[0] + a3.y*en[1] + a3.z*en[2] >= 0.0) ? 1.0 : -1.0;
            PetscReal xi = s * act->theta[qp];
            d = -0.5 * T * xi;
        }
        for (PetscInt i = 0; i < 3; i++) ead->fib[qp][i] = PetscCosReal(d) * f[i] + PetscSinReal(d) * fxn[i];
    }
    ead->fib_set = PETSC_TRUE;

    if (ec == 0 && T != 0.0) {
        PetscPrintf(PETSC_COMM_SELF, "[fiber] element 0, twist %.1f deg: angle from the mid-wall fiber per qp (theta):",
                    (double)act->fiber_twist_deg);
        for (PetscInt qp = 0; qp < act->n_qp; qp++) {
            PetscReal c = ead->fib[qp][0]*f[0] + ead->fib[qp][1]*f[1] + ead->fib[qp][2]*f[2];
            PetscReal sgn = (ead->fib[qp][0]*fxn[0] + ead->fib[qp][1]*fxn[1] + ead->fib[qp][2]*fxn[2]) >= 0.0 ? 1.0 : -1.0;
            PetscPrintf(PETSC_COMM_SELF, " %+.2f(%+.3f)", (double)(sgn * PetscAcosReal(PetscMin(1.0, PetscMax(-1.0, c))) * 180.0 / PETSC_PI),
                        (double)act->theta[qp]);
        }
        PetscPrintf(PETSC_COMM_SELF, "\n");
    }
    PetscFunctionReturn(0);
}

/* Deformation gradient */
PetscErrorCode ElemActDefGrad(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
    PetscErrorCode ierr = 0;
    IBMNodes *ibm = fem->ibm;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    PetscReal Fa_cart[3][3] = {{0.0}};
    PetscReal Finv_cart[3][3];
    PetscReal S[3][3], ST[3][3];
    PetscReal tmp[3][3];

    PetscReal delay = ibm->activation_delay ? ibm->activation_delay[ec] : 0.0;
    PetscReal gamma = GammaOfTimeDelayed(&fem->act_data.muscle_act_params, delay);
    /* Per-element activation scale (e.g. LV apex-cap taper, see
     * lv_geometry.c); defaults to 1.0 for mesh types that don't set it. */
    gamma *= ibm->gamma_scale[ec];

    /* Print gamma once per timestep to confirm ramp. */
    if (ec == 0) {
        static PetscInt last_ti = -1;
        if (ti != last_ti) {
            last_ti = ti;
            PetscPrintf(PETSC_COMM_WORLD, "[gamma] ti=%d  t=%.4f  gamma=%.6f\n",
                        (int)ti, (double)(ti * dt), (double)gamma);
        }
    }

    /*------------------------------------------------------------*/
    /* 1. Active deformation gradient in Cartesian basis          */
    /*------------------------------------------------------------*/
    if (cart_fib_act)
    {
        /* Fiber-based F_a is built per thickness qp below, from ead->fib[qp]. */
        if (!ead->fib_set) { ierr = ElemQPFibers_(fem, ec); CHKERRQ(ierr); }
    }
    else
    {
        if (manufactured) {
            PetscReal t = ti * dt;
            PetscReal lambda;

            gamma  = manufactured_gamma0 * PetscPowReal(PetscSinReal(PETSC_PI * t / manufactured_T), 2.0);
            lambda = 1.0 - gamma;
            (void)lambda;

            Fa_cart[0][0] = lambda;
            Fa_cart[1][1] = 1.0;
            Fa_cart[2][2] = 1.0;
        } else if (fa_nitti_plate) {
            /* Nitti et al. 2021 eq. (44)-(47), single flat patch parametrized on [0,1]^2:
             * Fa = I - gamma (a1 g^1 x g^1 + a2 g^2 x g^2) + gamma_n g^3 x g^3 with
             * contravariant patch vectors |g^1| = 1/Lx, |g^2| = 1/Ly (plate along x, y),
             * gamma_n from isochoric Fa (eq. 46). */
            Fa_cart[0][0] = 1.0 - gamma * fa_nitti_a1 / (fa_nitti_Lx * fa_nitti_Lx);
            Fa_cart[1][1] = 1.0 - gamma * fa_nitti_a2 / (fa_nitti_Ly * fa_nitti_Ly);
            Fa_cart[2][2] = 1.0/(Fa_cart[0][0]*Fa_cart[1][1]);
        } else {
            /* in-plane isotropic, isochoric: diag(1-gamma, 1-gamma, (1-gamma)^-2) */
            Fa_cart[0][0] = 1.0 - gamma;
            Fa_cart[1][1] = 1.0 - gamma;
            Fa_cart[2][2] = 1.0/(Fa_cart[0][0]*Fa_cart[1][1]);
        }
        
    }
        // PrintMat3x3("Fa_cart", Fa_cart);
    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++)
    {
        if (cart_fib_act) {
            /* 1b. Fiber-based active deformation gradient at this qp (Cartesian), fiber ead->fib[qp] */
            const PetscReal *nfiber = ead->fib[qp];
            if (fa_pezzuto) {
                /* Pezzuto & Ambrosi 2014 eq. (11), isochoric (det Fa = 1):
                 * Fa = (1-gamma) f x f + (1-gamma)^(-1/2) (I - f x f), f unit. */
                const PetscReal lf = 1.0 - gamma, lt = 1.0 / PetscSqrtReal(1.0 - gamma);
                for (PetscInt i = 0; i < 3; i++)
                    for (PetscInt j = 0; j < 3; j++)
                        Fa_cart[i][j] = lt * (i == j ? 1.0 : 0.0) + (lf - lt) * nfiber[i] * nfiber[j];
            } else {
                for (PetscInt i = 0; i < 3; i++)
                    for (PetscInt j = 0; j < 3; j++)
                        Fa_cart[i][j] = (i == j ? 1.0 : 0.0) - gamma * nfiber[i] * nfiber[j];
            }
        }
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

    PetscFunctionReturn(ierr);
}

/* Cauchy-Green deformation tensor */
PetscErrorCode ElemCGDefTens(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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

    PetscFunctionReturn(0);
}

/* Elastic-only Cauchy-Green deformation tensor */
PetscErrorCode ElemElasCGDefTens(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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

    PetscFunctionReturn(0);
}

/* Elastic stresses */
PetscErrorCode ElemElasStress(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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

    PetscFunctionReturn(ierr);
}

/* Total stresses including active contributions */
PetscErrorCode ElemTotStress(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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
        /* OLD (Nitti et al. 2021, CMAME 382:113877, eq. A.8) -- kept for reference, do not use.
         * Its 2nd term comes from (A.4) contracting Fa^-1 with slot 4 of Is instead of slot 2;
         * exact only at gamma = 0, gives a non-symmetric S otherwise.
         * Correct: S = Fa^-1 Se Fa^-T (e.g. Pezzuto & Ambrosi 2014, IJNMBE 30, p.1582).
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
        */

        /*------------------------------------------------------------------
         * S = Fa^-1 Se Fa^-T  in contravariant components:
         *    A^i_p  = g0^{iw} Fbar_wp            (mixed comps of Fa^-1)
         *    S^{ij} = A^i_p Se^{pq} A^j_q
         * Valid for any (also non-symmetric) Fa.
         *-----------------------------------------------------------------*/
        PetscReal A[3][3];
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt p = 0; p < 3; p++)
            {
                A[i][p] = 0.0;
                for (PetscInt w = 0; w < 3; w++)
                {
                    A[i][p] += ead->gm0[qp].Cont[i][w] * ead->Fa_inv[qp].Cov[w][p];
                }
            }
        }
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt j = 0; j < 3; j++)
            {
                for (PetscInt p = 0; p < 3; p++)
                {
                    for (PetscInt q = 0; q < 3; q++)
                    {
                        ead->S[qp].Cont[i][j] += A[i][p] * ead->Se[qp].Cont[p][q] * A[j][q];
                    }
                }
            }
        }
        // PrintMat3x3(" Total S", ead->S[qp].Cont);
    }
    PetscFunctionReturn(0);
}

/* Elastic tangent matrix */
PetscErrorCode ElemElsTangMatTens(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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
        detCe = DET3x3(ead->Ce[qp].Cov);
        detG0 = DET3x3(ead->gm0[qp].Cov);
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
    PetscFunctionReturn(0);
}

/* Total tangent matrix including activation effects */
PetscErrorCode ElemTotTangMatTens(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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

        /* OLD (Nitti et al. 2021, CMAME 382:113877, eq. B.6) -- kept for reference, do not use.
         * Built from the same faulty dCe/dC as (A.4)/(A.5), used twice; wrong for gamma > 0
         * (1.5% vs finite difference at gamma = 0.2) and not even the derivative of (A.8).
         * Correct: one Fa^-1 (mixed comps) per index of CCe, below.
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
        */

        /*------------------------------------------------------------------
         * CC = 2 dS/dC with S = Fa^-1 Se Fa^-T, contravariant components:
         *    A^i_p      = g0^{iw} Fbar_wp            (mixed comps of Fa^-1)
         *    CC^{ijkl}  = A^i_p A^j_q A^k_r A^l_s CCe^{pqrs}
         * Only CC^{3333} is needed (C33 Newton in ModElemC33), so the
         * i,j,k,l ranges stay restricted to 2 as before.
         *-----------------------------------------------------------------*/
        PetscReal A[3][3];
        for (PetscInt i = 0; i < 3; i++)
        {
            for (PetscInt p = 0; p < 3; p++)
            {
                A[i][p] = 0.0;
                for (PetscInt w = 0; w < 3; w++)
                {
                    A[i][p] += ead->gm0[qp].Cont[i][w] * ead->Fa_inv[qp].Cov[w][p];
                }
            }
        }
        for (PetscInt i = 2; i < 3; i++)
        for (PetscInt j = 2; j < 3; j++)
        for (PetscInt k = 2; k < 3; k++)
        for (PetscInt l = 2; l < 3; l++)
        {
            for (PetscInt p = 0; p < 3; p++)
            for (PetscInt q = 0; q < 3; q++)
            for (PetscInt r = 0; r < 3; r++)
            for (PetscInt s = 0; s < 3; s++)
            {
                ead->CC[qp].Cont[i][j][k][l] +=
                    A[i][p] * A[j][q] * A[k][r] * A[l][s] * ead->CCe[qp].Cont[p][q][r][s];
            }
        }
    }

    PetscFunctionReturn(ierr);
}


/* Modification of element tensor component C33 (one Newton step per call)
   Goal: drive S33 -> 0 by updating C33 using DeltaC33 = -2*S33 / CC3333.
*/
PetscErrorCode ModElemC33(FE *fem, PetscInt ec, PetscReal *delta)
{
    PetscFunctionBeginUser;
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
        PetscReal denom_tol = 1e-14;
        if (PetscAbsReal(CC3333) <= denom_tol || CC3333 != CC3333) {
            PetscPrintf(PETSC_COMM_SELF,
                "[ModElemC33] ec=%d qp=%d: CC3333=%.6e S33=%.6e\n",
                (int)ec, (int)qp, (double)CC3333, (double)S33);
        }
        PetscCheck(PetscAbsReal(CC3333) > denom_tol,
                   PETSC_COMM_SELF, PETSC_ERR_FP,
                   "CC3333 too small (%.6e) at qp=%" PetscInt_FMT, (double)CC3333, qp);

        // PetscPrintf(PETSC_COMM_SELF, "S33 before update at qp=%" PetscInt_FMT ": %.6e\n", qp, (double)S33);
        // PetscPrintf(PETSC_COMM_SELF, "CC3333 before update at qp=%" PetscInt_FMT ": %.6e\n", qp, (double)CC3333);
        /* One Newton update step */

        PetscReal DeltaC33 = -1.0 * S33 / CC3333;

        /* Guard: clamp step so det(C) stays positive.
         * With thick shells or boundary elements, gm0 can have large off-diagonal
         * [0][2] terms that make det(C) sensitive to C33.  If the full Newton step
         * would drive det(C) <= 0, scale it back to half the zero-crossing. */
        {
            PetscReal C33_old = ead->C[qp].Cov[2][2];
            ead->C[qp].Cov[2][2] = C33_old + DeltaC33;  /* trial */
            if (DET3x3(ead->C[qp].Cov) <= 0.0) {
                /* binary search: find max |alpha| in [0,1] keeping det > 0 */
                PetscReal lo = 0.0, hi = 1.0;
                for (int bk = 0; bk < 16; bk++) {
                    PetscReal mid = 0.5 * (lo + hi);
                    ead->C[qp].Cov[2][2] = C33_old + mid * DeltaC33;
                    if (DET3x3(ead->C[qp].Cov) > 0.0) lo = mid; else hi = mid;
                }
                DeltaC33 *= lo;
                ead->C[qp].Cov[2][2] = C33_old + DeltaC33;
                ead->c33_clamped++;
            }
        }

        *delta  = DeltaC33;
        // PetscPrintf(PETSC_COMM_SELF, "after updating C33 = %f\n", (double)ead->C[qp].Cov[2][2]);
        // PetscPrintf(PETSC_COMM_SELF, "after updating C33 = %f\n", (double)ead->C[qp].Cov[2][2]);

        /* Recompute stress/tangent after update so outer loop sees new S33.
           If your outer iteration recomputes these anyway, you can omit these.
        */
        /* PetscCall(ElemTotStress(fem, ec));
           PetscCall(ElemTotTangMatTens(fem, ec)); */
    }

    PetscFunctionReturn(ierr);
}


/* Transfer reference covariant basis vectors (G1,G2) from IBMNodes storage
   (computed by Kve0()) into element activation data ead->g0[qp].Cov[]. */
PetscErrorCode ElemUpdateG0(FE *fem, PetscInt ec)
{
    PetscFunctionBeginUser;
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

    PetscFunctionReturn(ierr);
}


#undef __FUNCT__
#define __FUNCT__ "ElemUpdFint"
PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec, PetscReal *Fb_out)
{
  PetscFunctionBeginUser;
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
  const PetscReal half_h0 = 0.5 * ibm->h0_e[ec];
  const PetscReal pref0   = 0.5 * ibm->h0_e[ec] * A0; 

  /* LV apex cap elements are topologically irregular, but their subdivision
     geometry is intentionally replaced with a flat triangle in
     ElemUpdateGeomSubdivFromCoords_().  Assemble a matching CST membrane
     contribution into the patch slots that hold nv1/nv2/nv3. */
  if (ibm->n_elmt_base > 0 && ec >= ibm->n_elmt_base) {
    const PetscInt tri_nodes[3] = {ibm->nv1[ec], ibm->nv2[ec], ibm->nv3[ec]};
    const PetscReal dN[3][2] = {{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}};
    PetscInt tri_slot[3] = {-1, -1, -1};
    const PetscInt nslot = ibm->val[ec] + 6;

    for (PetscInt a = 0; a < 3; a++) {
      for (PetscInt i = 0; i < nslot; i++) {
        if (ibm->patch[16*ec + i] == tri_nodes[a]) {
          tri_slot[a] = i;
          break;
        }
      }
      PetscCheck(tri_slot[a] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "LV cap element %" PetscInt_FMT " vertex %" PetscInt_FMT
                 " is missing from patch[]", ec, tri_nodes[a]);
    }

    for (PetscInt qp = 0; qp < fem->act_data.n_qp; qp++) {
      const PetscReal w_qp = PetscRealPart(fem->act_data.w[qp]);
      PetscReal Sv[3];
      Sv[0] = ead->S[qp].Cont[0][0];
      Sv[1] = ead->S[qp].Cont[1][1];
      Sv[2] = ead->S[qp].Cont[0][1];

      for (PetscInt a = 0; a < 3; a++) {
        const PetscInt base = 3 * tri_slot[a];
        const PetscReal Na0 = dN[a][0];
        const PetscReal Na1 = dN[a][1];

        const PetscReal Bm0x = Na0 * ndx21.x;
        const PetscReal Bm0y = Na0 * ndx21.y;
        const PetscReal Bm0z = Na0 * ndx21.z;
        const PetscReal Bm1x = Na1 * ndx31.x;
        const PetscReal Bm1y = Na1 * ndx31.y;
        const PetscReal Bm1z = Na1 * ndx31.z;
        const PetscReal Bm2x = Na0 * ndx31.x + Na1 * ndx21.x;
        const PetscReal Bm2y = Na0 * ndx31.y + Na1 * ndx21.y;
        const PetscReal Bm2z = Na0 * ndx31.z + Na1 * ndx21.z;

        Fb_out[base+0] += pref0 * w_qp * (Bm0x*Sv[0] + Bm1x*Sv[1] + Bm2x*Sv[2]);
        Fb_out[base+1] += pref0 * w_qp * (Bm0y*Sv[0] + Bm1y*Sv[1] + Bm2y*Sv[2]);
        Fb_out[base+2] += pref0 * w_qp * (Bm0z*Sv[0] + Bm1z*Sv[1] + Bm2z*Sv[2]);
      }
    }

    PetscFunctionReturn(0);
  }

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

  PetscFunctionReturn(ierr);
}

/* Pre-allocate irregular geometry arrays for all elements.
   This ensures arrays are ready before CUDA kernels or geometry updates. */
static PetscErrorCode EnsureAllGeometryIrregularArrays_(FE *fem)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr = 0;
  IBMNodes *ibm = fem->ibm;

  PetscCheck(ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "fem->ibm is NULL");

  for (PetscInt ec = 0; ec < ibm->n_elmt; ++ec) {
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscCheck(ead != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Element %" PetscInt_FMT " act data must not be NULL", ec);

    /* Check if element is irregular */
    if (ibm->ire[ec] == 1) {
      const PetscInt v = ibm->val[ec];
      const PetscInt nen = v + 6;

      /* Ensure irregular arrays are allocated for both geom and geom0 */
      ierr = SubdivGeomEnsureIrregularArrays_(&ead->geom[0], nen); CHKERRQ(ierr);
      ierr = SubdivGeomEnsureIrregularArrays_(&ead->geom0[0], nen); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode InitActStrainProblem(FE *fem, PetscInt ibi){
  PetscFunctionBeginUser;

  if (manufactured){
    PetscCall(GetManufacturedActiveStrainOptions());
  }

  PetscCall(GetUserActParams(fem));
  PetscCall(ActDataAllocate(fem));
  PetscCall(SetGaussianQuadrature(fem));

  /* Pre-allocate irregular geometry arrays for all elements */
  PetscCall(EnsureAllGeometryIrregularArrays_(fem));

  if (prescribed_force_field && !manufactured_fexternal_export) {
    // PetscCall(ManufacturedInitialKinematicsIn(tistart, fem));
    PetscCall(ManufacturedKinematicsAndFExternalSetInitial(fem, ibi, tistart));
  }
  
  PetscFunctionReturn(0);
}

/* Holzapfel-Ogden, exactly incompressible shell (option iii), all thickness qp of element ec.
 * Needs C_ab (ElemCGDefTens) and Fa (ElemActDefGrad). Components are on the reference basis g0_i (g0_3 = a3).
 *  1. Thickness from J = 1 (Kiendl 2015 eqs. 22-24; det Fa = 1 so also Je = 1):
 *        C_33 = det(G_ab)/det(g_ab),  C_a3 = 0.
 *  2. Base elastic stress S_W = 2 dW/dCe (Wang 2013 eq. 3), f^i = g0^i . f (tangent part), s^i = delta^i_3, H = (f x s + s x f)/2:
 *        S_W^ij = a E1 G^ij + 2 af <I4f-1> Ef f^i f^j + 2 as <I4s-1> Es s^i s^j + 2 afs I8 Efs H^ij,
 *        E1 = exp(b(I1-3)), Ef = exp(bf(I4f-1)^2), Es = exp(bs(I4s-1)^2), Efs = exp(bfs I8^2).
 *  3. Push-back, then p from plane stress on the total stress (Psi_e = W(Ce) - p(Je-1)):
 *        Sbar^ij = A^i_p S_W^pq A^j_q,  p = Sbar^33 / Cbar^33,  S^ij = Sbar^ij - p Cbar^ij  (Cbar = C^-1),
 *     so S^33 = 0 exactly; Se^ij = S_W^ij - p Je Ce^-1,ij is stored for output. */
static PetscErrorCode ElemHOStress_(FE *fem, PetscInt ec)
{
  PetscFunctionBeginUser;
  ElemActData *ead = &fem->act_data.elem_act_data[ec];
  const PetscInt n_qp = fem->act_data.n_qp;

  for (PetscInt qp = 0; qp < n_qp; qp++) {
    const PetscReal detg = ead->gm[qp].Cov[0][0]*ead->gm[qp].Cov[1][1] - ead->gm[qp].Cov[0][1]*ead->gm[qp].Cov[1][0];
    const PetscReal detG = ead->gm0[qp].Cov[0][0]*ead->gm0[qp].Cov[1][1] - ead->gm0[qp].Cov[0][1]*ead->gm0[qp].Cov[1][0];
    PetscCheck(detg > 0.0 && detG > 0.0, PETSC_COMM_SELF, PETSC_ERR_FP, "non-positive area metric at element %d qp %d", (int)ec, (int)qp);
    for (PetscInt a = 0; a < 2; a++) { ead->C[qp].Cov[a][2] = 0.0; ead->C[qp].Cov[2][a] = 0.0; }
    ead->C[qp].Cov[2][2] = detG / detg;
    PetscCall(RaiseIndices2(ead->gm0[qp].Cont, ead->C[qp].Cov, ead->C[qp].Cont));
    PetscCall(INV(ead->C[qp].Cov, ead->C_inv[qp].Cont));
    PetscCall(LowerIndices2(ead->gm0[qp].Cov, ead->C_inv[qp].Cont, ead->C_inv[qp].Cov));
  }
  PetscCall(ElemElasCGDefTens(fem, ec));

  for (PetscInt qp = 0; qp < n_qp; qp++) {
    PetscReal (*Ce)[3] = ead->Ce[qp].Cov, (*G)[3] = ead->gm0[qp].Cont;
    const PetscReal *fc = ead->fib[qp];
    PetscReal f[3], sv[3] = {0.0, 0.0, 1.0};
    for (PetscInt i = 0; i < 3; i++)
      f[i] = ead->g0[qp].Cont[i].x*fc[0] + ead->g0[qp].Cont[i].y*fc[1] + ead->g0[qp].Cont[i].z*fc[2];
    /* Mesh fibers can have a small normal component (~0.5 deg on the LV); keep f tangent so f . s = 0 and
     * I8 = 0 in the reference state (else the fs term prestresses it), and unit: G_ab f^a f^b = 1. */
    f[2] = 0.0;
    {
      const PetscReal nf = PetscSqrtReal(ead->gm0[qp].Cov[0][0]*f[0]*f[0] + 2.0*ead->gm0[qp].Cov[0][1]*f[0]*f[1] + ead->gm0[qp].Cov[1][1]*f[1]*f[1]);
      f[0] /= nf; f[1] /= nf;
    }

    PetscReal I1 = 0.0, I4f = 0.0, I4s = Ce[2][2], I8 = 0.0;
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++) { I1 += Ce[i][j]*G[i][j]; I4f += Ce[i][j]*f[i]*f[j]; I8 += Ce[i][j]*f[i]*sv[j]; }

    const PetscReal c1  = ho_a * PetscExpReal(ho_b * (I1 - 3.0));
    const PetscReal ef  = (ho_tension_only && I4f < 1.0) ? 0.0 : I4f - 1.0;
    const PetscReal cf  = 2.0 * ho_af * ef * PetscExpReal(ho_bf * ef * ef);
    const PetscReal es  = (ho_tension_only && I4s < 1.0) ? 0.0 : I4s - 1.0;
    const PetscReal cs  = ho_sheet ? 2.0 * ho_as * es * PetscExpReal(ho_bs * es * es) : 0.0;
    const PetscReal cfs = ho_sheet ? 2.0 * ho_afs * I8 * PetscExpReal(ho_bfs * I8 * I8) : 0.0;

    PetscReal SW[3][3];
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++)
        SW[i][j] = c1*G[i][j] + cf*f[i]*f[j] + cs*sv[i]*sv[j] + 0.5*cfs*(f[i]*sv[j] + sv[i]*f[j]);

    /* A^i_p = g0^{iw} Fbar_wp (mixed comps of Fa^-1), as in ElemTotStress */
    PetscReal A[3][3], Sb[3][3];
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt p = 0; p < 3; p++) {
        A[i][p] = 0.0;
        for (PetscInt w = 0; w < 3; w++) A[i][p] += G[i][w] * ead->Fa_inv[qp].Cov[w][p];
      }
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++) {
        Sb[i][j] = 0.0;
        for (PetscInt p = 0; p < 3; p++)
          for (PetscInt q = 0; q < 3; q++) Sb[i][j] += A[i][p] * SW[p][q] * A[j][q];
      }

    const PetscReal pr = Sb[2][2] / ead->C_inv[qp].Cont[2][2];
    const PetscReal Je = PetscSqrtReal(DET3x3(ead->Ce[qp].Cov) / DET3x3(ead->gm0[qp].Cov));
    for (PetscInt i = 0; i < 3; i++)
      for (PetscInt j = 0; j < 3; j++) {
        ead->S[qp].Cont[i][j]  = Sb[i][j] - pr * ead->C_inv[qp].Cont[i][j];
        ead->Se[qp].Cont[i][j] = SW[i][j] - pr * Je * ead->Ce_inv[qp].Cont[i][j];
      }

    /* -debug_ho_nan 1: report quadrature points whose stress is not finite (at most 20 lines per process), with
     * where they are (element, thickness position theta, reference centroid height z0) and what made them blow up. */
    {
      static PetscInt dbg_nan = -1, n_printed = 0;
      if (dbg_nan < 0) { dbg_nan = 0; PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-debug_ho_nan", &dbg_nan, PETSC_NULL); }
      PetscBool bad = PETSC_FALSE;
      for (PetscInt i = 0; i < 3; i++) for (PetscInt j = 0; j < 3; j++) if (!PetscIsNormalReal(ead->S[qp].Cont[i][j]) && ead->S[qp].Cont[i][j] != 0.0) bad = PETSC_TRUE;
      if (dbg_nan && bad && n_printed < 20) {
        IBMNodes *ib = fem->ibm;
        const PetscReal z0 = (ib->z_bp0[ib->nv1[ec]] + ib->z_bp0[ib->nv2[ec]] + ib->z_bp0[ib->nv3[ec]]) / 3.0;
        const PetscReal detg = ead->gm[qp].Cov[0][0]*ead->gm[qp].Cov[1][1] - ead->gm[qp].Cov[0][1]*ead->gm[qp].Cov[1][0];
        const PetscReal detG = ead->gm0[qp].Cov[0][0]*ead->gm0[qp].Cov[1][1] - ead->gm0[qp].Cov[0][1]*ead->gm0[qp].Cov[1][0];
        PetscPrintf(PETSC_COMM_SELF, "[HOnan] ti=%d ec=%d qp=%d theta=%+.3f z0=%.4e  I1=%.4e I4f=%.4e I4s=%.4e I8=%.2e C33=%.4e Jo=sqrt(detg/detG)=%.4e p=%.3e S11=%.3e\n",
                    (int)ti, (int)ec, (int)qp, (double)PetscRealPart(fem->act_data.theta[qp]), (double)z0, (double)I1, (double)I4f, (double)I4s, (double)I8,
                    (double)ead->C[qp].Cov[2][2], (double)PetscSqrtReal(PetscMax(detg, 0.0) / detG), (double)pr, (double)ead->S[qp].Cont[0][0]);
        n_printed++;
      }
    }

    /* -debug_ho_elem <ec>: full state of this qp for an outside check (pezzuto_lv/checks/ho_code_check.py) */
    {
      PetscInt dbg = -1;
      PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-debug_ho_elem", &dbg, PETSC_NULL);
      if (dbg == ec) {
        const PetscReal delay = fem->ibm->activation_delay ? fem->ibm->activation_delay[ec] : 0.0;
        const PetscReal gam = GammaOfTimeDelayed(&fem->act_data.muscle_act_params, delay) * fem->ibm->gamma_scale[ec];
        PetscPrintf(PETSC_COMM_SELF, "[HOdbg] ec=%d qp=%d gamma=%.17g fib=%.17g,%.17g,%.17g", (int)ec, (int)qp, (double)gam, (double)fc[0], (double)fc[1], (double)fc[2]);
        for (PetscInt k = 0; k < 3; k++)
          PetscPrintf(PETSC_COMM_SELF, " G%d=%.17g,%.17g,%.17g", (int)k, (double)ead->g0[qp].Cov[k].x, (double)ead->g0[qp].Cov[k].y, (double)ead->g0[qp].Cov[k].z);
        for (PetscInt k = 0; k < 3; k++)
          PetscPrintf(PETSC_COMM_SELF, " g%d=%.17g,%.17g,%.17g", (int)k, (double)ead->g[qp].Cov[k].x, (double)ead->g[qp].Cov[k].y, (double)ead->g[qp].Cov[k].z);
        PetscPrintf(PETSC_COMM_SELF, " S=%.17g,%.17g,%.17g,%.17g p=%.17g C33=%.17g\n", (double)ead->S[qp].Cont[0][0], (double)ead->S[qp].Cont[1][1],
                    (double)ead->S[qp].Cont[0][1], (double)ead->S[qp].Cont[2][2], (double)pr, (double)ead->C[qp].Cov[2][2]);
      }
    }

    if (ec == 0 && qp == 0) {
      static PetscInt last_ti = -1;
      if (ti != last_ti) {
        last_ti = ti;
        PetscPrintf(PETSC_COMM_SELF, "[HO] ti=%d elem 0 qp 0: C33=%.6f Je=%.3e I1=%.6f I4f=%.6f I4s=%.6f I8=%.2e p=%.4e S11=%.4e S22=%.4e S33=%.1e\n",
                    (int)ti, (double)ead->C[qp].Cov[2][2], (double)(Je - 1.0), (double)I1, (double)I4f, (double)I4s, (double)I8, (double)pr,
                    (double)ead->S[qp].Cont[0][0], (double)ead->S[qp].Cont[1][1], (double)ead->S[qp].Cont[2][2]);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ElemC33Solve(FE *fem, PetscInt ec) {
  PetscFunctionBeginUser;

  if (material_model == 1) {   /* closed form, no local Newton */
    fem->act_data.elem_act_data[ec].c33_clamped = 0;
    fem->act_data.elem_act_data[ec].c33_last_delta = 0.0;
    PetscCall(ElemHOStress_(fem, ec));
    PetscFunctionReturn(0);
  }

  ElemActData *ead = NULL;

  PetscErrorCode ierr = 0;
  PetscReal delta;

  ead = &fem->act_data.elem_act_data[ec];
  ead->c33_clamped = 0;
  delta = 0.0;

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

    // if (ec == 100)
    // {
    //   // PetscPrintf(PETSC_COMM_SELF, "ModElemC33 completed on elem = %d \n", ec);
    //   // PetscPrintf(PETSC_COMM_SELF, "deltaC33 = %f at sub_itr = %d \n", delta, sub_itr);
    //   // PrintElemS(fem, ec);
    // }

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
    // if (ec == 10)
    // {
    // // PrintMat3x3(" Total S after C33 Update", ead->S[0].Cont);
    //   // PetscPrintf(PETSC_COMM_SELF, "sub_itr = %d \n", sub_itr);
    // }
    
    // PrintMat3x3(" ead->C[0].Cov after C33 Update", ead->C[0].Cov);
    
    // ElemElsTangMatTens(fem, ec);  
    // ElemTotTangMatTens(fem, ec); 

  }

  ead->c33_last_delta = PetscAbsReal(delta);

  /* TEMP diagnostic (-debug_c33_elem <ec>): C33 / S33 after the sub-iterations. */
  {
    PetscInt dbg = -1;
    PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-debug_c33_elem", &dbg, PETSC_NULL);
    if (dbg == ec) {
      PetscPrintf(PETSC_COMM_SELF, "[C33dbg] ec=%d C33=%.10e S33=%.3e S11=%.6e S22=%.6e S12=%.3e C11=%.6e C22=%.6e g0_11=%.6e g0_33=%.6e\n",
                  (int)ec, (double)ead->C[0].Cov[2][2], (double)ead->S[0].Cont[2][2], (double)ead->S[0].Cont[0][0],
                  (double)ead->S[0].Cont[1][1], (double)ead->S[0].Cont[0][1], (double)ead->C[0].Cov[0][0],
                  (double)ead->C[0].Cov[1][1], (double)ead->gm0[0].Cov[0][0], (double)ead->gm0[0].Cov[2][2]);
    }
  }
  
  PetscFunctionReturn(0);
}

/* -log_c33: one line per step on how well the local plane-stress (C33) solve did, over all
 * locally owned elements (state of the most recent residual evaluation):
 *   C33 mean/max, max |last DeltaC33|/C33 (size of the last local update = remaining error),
 *   max|S^33| / max|S^ab| over all elements (S from before the last local update; 0 if there is
 *   no in-plane stress), elements whose update was cut by the det(C)>0 guard. */
PetscErrorCode LogC33Stats(FE *fem, PetscInt step)
{
  PetscFunctionBeginUser;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  PetscReal mx[4] = {0.0, 0.0, 0.0, 0.0};   /* C33 max, rel. last delta max, max |S33|, max in-plane |S| */
  PetscReal sm[3] = {0.0, 0.0, 0.0};   /* C33 sum, elements, clamped elements */
  for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
    const PetscInt ec = ctx->layout.orig_cell[lc];
    if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscReal C33 = ead->C[0].Cov[2][2];
    PetscReal Sin = PetscMax(PetscMax(PetscAbsReal(ead->S[0].Cont[0][0]), PetscAbsReal(ead->S[0].Cont[1][1])),
                             PetscAbsReal(ead->S[0].Cont[0][1]));
    mx[0] = PetscMax(mx[0], C33);
    mx[1] = PetscMax(mx[1], ead->c33_last_delta / PetscMax(PetscAbsReal(C33), 1e-300));
    mx[2] = PetscMax(mx[2], PetscAbsReal(ead->S[0].Cont[2][2]));
    mx[3] = PetscMax(mx[3], Sin);
    sm[0] += C33; sm[1] += 1.0; sm[2] += (ead->c33_clamped > 0) ? 1.0 : 0.0;
  }
  MPI_Allreduce(MPI_IN_PLACE, mx, 4, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sm, 3, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,
              "[C33] step=%d C33_mean=%.6f C33_max=%.6f last_dC33_rel_max=%.3e S33_rel_max=%.3e clamped_elems=%d subitr=%d\n",
              (int)step, (double)(sm[0] / PetscMax(sm[1], 1.0)), (double)mx[0], (double)mx[1],
              (double)(mx[3] > 0.0 ? mx[2] / mx[3] : 0.0),
              (int)sm[2], (int)fem->act_data.C33_subitr_nums);
  PetscFunctionReturn(0);
}

PetscErrorCode FInternalPreCalc(FE *fem) {
  PetscFunctionBeginUser;

  PetscErrorCode ierr = 0;

  /* Geometry via persistent DMPlex context (set up once in main after Init). */
  PetscCall(FEM_DMPlexGeomUpdate(fem));
  // UpdateElements(fem, ElemUpdateGeom0Subdiv);
  // UpdateElements(fem, ElemUpdateGeomSubdiv);

  // ierr = DebugPrintGeomForElement(fem, 100); CHKERRQ(ierr);

  /* Run per-element physics pipeline in parallel over locally owned cells. */
  {
    DMPlexGeomCtx *ctx = &fem->geom_ctx;
    for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
      const PetscInt ec = ctx->layout.orig_cell[lc];
      if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
      PetscCall(ElemUpdateG(fem, ec));
      PetscCall(ElemActDefGrad(fem, ec));
      PetscCall(ElemCGDefTens(fem, ec));
      PetscCall(ElemC33Solve(fem, ec));
    }
  }

  /* Debug stats (all-element loop; disabled in parallel — only local cells have valid C).
  {
    IBMNodes *ibm = fem->ibm;
    PetscInt n_elmt = ibm->n_elmt;
    PetscInt n_qp = fem->act_data.n_qp;
    PetscReal sum_C33 = 0.0;
    PetscInt total_qp = 0;
    for (PetscInt ec = 0; ec < n_elmt; ec++) {
      ElemActData *ead = &fem->act_data.elem_act_data[ec];
      for (PetscInt qp = 0; qp < n_qp; qp++) { sum_C33 += ead->C[qp].Cov[2][2]; total_qp++; }
    }
    PetscReal avg_C33 = (total_qp > 0) ? sum_C33 / total_qp : 0.0;
    PetscPrintf(PETSC_COMM_SELF, "Average C33: %e (over %d QPs in %d elements)\n", (double)avg_C33, total_qp, n_elmt);
  } */
  PetscFunctionReturn(0);
}

/* UpdateElementThickness — call ONCE per converged timestep (not per Newton
 * iteration, unlike FInternalPreCalc/ElemC33Solve above, which re-solve C33
 * every residual evaluation).
 *
 * ead->C[qp].Cov[2][2] is built from ead->geom vs ead->geom0 (see
 * ElemUpdateG above) -- geom0 is the REFERENCE configuration cached once at
 * t=0, not the previous timestep -- so C33 is the standard, CUMULATIVE
 * right-Cauchy-Green component C33 = lambda3^2, the through-thickness
 * stretch *squared*, relative to t=0, not an incremental step-to-step
 * ratio. An earlier version of this function compounded it multiplicatively
 * onto the running thickness each timestep (thickness *= C33), which
 * double-counts the already-cumulative quantity every step and collapsed
 * thickness toward zero within ~10 steps (measured: 0.30 -> 0.23 by step 5,
 * with per-step C33 itself steadily *decreasing*, compounding the error
 * further). Corrected to the non-compounding relation, recomputed fresh
 * from h0 every call instead of building on the previous thickness. */
PetscErrorCode UpdateElementThickness(FE *fem)
{
  PetscFunctionBeginUser;
  IBMNodes      *ibm = fem->ibm;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  PetscInt       n_qp = fem->act_data.n_qp;

  PetscReal *local_thickness;
  PetscCall(PetscCalloc1(ibm->n_elmt, &local_thickness));

  for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
    const PetscInt ec = ctx->layout.orig_cell[lc];
    if (ec < 0 || ec >= ibm->n_elmt) continue;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscReal sum_C33 = 0.0;
    for (PetscInt qp = 0; qp < n_qp; qp++) sum_C33 += ead->C[qp].Cov[2][2];
    PetscReal C33 = (n_qp > 0) ? sum_C33 / (PetscReal)n_qp : 1.0;
    PetscReal lambda3 = (C33 > 0.0) ? PetscSqrtReal(C33) : 1.0;
    local_thickness[ec] = ibm->h0_e[ec] * lambda3;
  }

  PetscCallMPI(MPIU_Allreduce(local_thickness, ibm->thickness, ibm->n_elmt, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  PetscCall(PetscFree(local_thickness));

  PetscFunctionReturn(0);
}

PetscErrorCode FInternalAct(FE *fem){
  PetscFunctionBeginUser;
  PetscErrorCode ierr = 0;

  IBMNodes      *ibm = fem->ibm;
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  PetscInt       i;

  FInternalPreCalc(fem);

  /* Assemble Fint via VecSetValues so PETSc routes off-process contributions
   * to the correct rank during VecAssemblyBegin/End. */
  for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
    const PetscInt ec = ctx->layout.orig_cell[lc];
    if (ec < 0 || ec >= ibm->n_elmt) continue;
    PetscInt v    = ibm->val[ec];
    /* Regular elements (val==0) have 12 subdivision control points; irregular
     * elements have val+6 control points.  ElemUpdFint always writes 12*dof
     * entries for regular elements, so allocate accordingly. */
    PetscInt nen  = (ibm->ire[ec] == 0) ? 12 : (v + 6);
    PetscInt nloc = dof * nen;

    PetscReal *Fb;
    ierr = PetscCalloc1(nloc, &Fb); CHKERRQ(ierr);
    ElemUpdFint(fem, ec, Fb);

    for (i=0; i<nen; i++) {
      PetscInt node = ibm->patch[16*ec+i];
      if (node == 1000000 || node < 0) continue;
      if (node >= ibm->n_v) {
        /* Ghost node: x_g = x[s1] + x[s2] - x[s3] (GlobalGhostInit), so by
         * the chain rule its force goes +1 to s1, +1 to s2, -1 to s3. */
        if (!ghost_force_map || !ibm->ghost_src || node >= ibm->n_v + ibm->n_ghosts) continue;
        const PetscInt  gc = node - ibm->n_v;
        const PetscReal sg[3] = {1.0, 1.0, -1.0};
        for (PetscInt k = 0; k < 3; k++) {
          PetscInt gdof0 = ctx->ibm_to_global_dof0[ibm->ghost_src[3*gc+k]];
          if (gdof0 < 0) continue;
          PetscInt  dofs[3] = {gdof0, gdof0+1, gdof0+2};
          PetscReal vals[3] = {sg[k]*Fb[dof*i], sg[k]*Fb[dof*i+1], sg[k]*Fb[dof*i+2]};
          ierr = VecSetValues(fem->Fint, 3, dofs, vals, ADD_VALUES); CHKERRQ(ierr);
        }
        continue;
      }
      PetscInt gdof0 = ctx->ibm_to_global_dof0[node];
      if (gdof0 < 0) continue;
      PetscInt  dofs[3] = {gdof0, gdof0+1, gdof0+2};
      PetscReal vals[3] = {Fb[dof*i], Fb[dof*i+1], Fb[dof*i+2]};
      ierr = VecSetValues(fem->Fint, 3, dofs, vals, ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = PetscFree(Fb); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(fem->Fint); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(fem->Fint);   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------
 *                                 Utility: Generic Element Loop
 *-------------------------------------------------------------------------------------------------*/

/**
 * @brief Apply a given element-level routine to all elements.
 *
 * @param fem   Pointer to finite element structure.
 * @param func  Element routine to apply.
 */
PetscErrorCode UpdateElements(FE *fem, ElemFunc func)
{
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    PetscInt n_elems = fem->ibm->n_elmt;
    for (PetscInt ec = 0; ec < n_elems; ec++)
    {
        ierr = func(fem, ec);
        if (ierr)
            PetscFunctionReturn(ierr); /* Standard PETSc error propagation */
    }

    PetscFunctionReturn(0);
}
