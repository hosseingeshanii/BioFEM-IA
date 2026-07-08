static char help[] = "Hosein FEM Petsc 3.6.2 \n\n";

#include "variables.h"
#include "math.h"
#include <stdio.h>
#include <petscsystypes.h>  // Correct PETSc header for PetscBool type
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdbool.h>  
#include <stdlib.h>
#include "active_strain.h"
#include "cuda_bridge.h"
#include "kokkos_bridge.h"
#include "cuda_processes.h"
#include "kokkos_processes.h"
#include "manufactured_active_strain.h"
#include "dmplex_geom.h"
#include "lv_geometry.h"

PetscReal  E=0.0, mu=0.0, rho=0.0, h0=0.0, dt=0.0, dampfactor=0.0, char_length_x=1.0, char_length_y=1.0, char_length_z=1.0;
PetscInt   dof=3, twod=0, damping=0, membrane=0, bending=0, outghost=0, ConstitutiveLawNonLinear=0;
PetscInt   timeinteg=0, nbody=1, contact=0, explicit=0;
PetscInt   ec, nc, ti, tiout, tistart=0, rstart_flg, tisteps=1, curvature=6, manufactured=0, inverse=1, dR_dE_flag=0;
PetscInt   n_epochs=100, init_flag=1, epoch_start=0, epoch_output = 100;
PetscReal  initial_elas = 10.0, initial_poisson = 0.2;
PetscInt   constrained_el = 0;
PetscInt   par_jac = 0, progress_bar_show = 0; 
PetscReal  fib_smth_factor = 0.001, res_smth_factor = 0.001;

PetscInt   muscle_activation = 0; 
PetscInt   manufactured_fexternal_export = 0;
PetscInt   prescribed_force_field = 0;
PetscInt   cuda_process = 0;
PetscInt   kokkos_process = 0;
PetscInt   dmplex_geom_process = 0;
PetscInt   lv_geom_process  = 0;     /* use analytic LV mesh instead of file input */






char out_dir[256] = {"./"};
char in_dir[256] = {"./"};
PetscReal  learning_rate = 0.001;

// Adam optimizer options 
PetscInt   Adam = 0;
PetscReal  BETA1=0.9, BETA2=0.999, EPSILON=1e-8;

// Linear model options
PetscInt   n_lin_model_coeffs = 2; // Elasticity, Poisson ratio
// Fung's model options
PetscInt   n_Fung_Coeffs = 7 + 2;
PetscInt   uniform_fung = 0;
PetscInt   uniform_fiber_dir = 0;

// Optimizer's general options
PetscInt   epoch_update_jacobian = 100;
PetscInt   epoch;


PetscReal decay_factor = 0.5;    // Factor by which to reduce the learning rate
PetscInt  patience = 5;             // Number of epochs with no improvement to wait before decaying
PetscReal improvement_threshold = 1e-4; // Minimum change in loss to consider as improvement
PetscReal best_loss = PETSC_MAX_REAL;  // Track the best (lowest) loss
PetscInt  epochs_no_improvement = 0; // Counter for epochs without improvement

PetscInt fibersmooth = 0;
PetscInt ressmooth = 0;
PetscInt res_smooth_itrs = 2;

PetscInt   monitor_residual = 0;
FILE      *res_log_fp       = NULL;

PetscErrorCode  FormFunctionFEM(SNES snes, Vec x, Vec R, void *ctx);
extern PetscErrorCode InitVecs(FE *fem);
extern PetscErrorCode InitVel(PetscInt edge_n, PetscReal w, FE *fem);
extern PetscErrorCode MoveBoundary(PetscInt edge_n, FE *fem);
extern PetscErrorCode  ConstantVel(PetscReal vel, PetscInt dir, FE *fem);
extern PetscErrorCode initialize_elasticity(FE *fem, PetscReal initial_elasticity_value, PetscReal initial_poisson);
extern PetscErrorCode update_elasticity(PetscReal learning_rate, FE *fem);
extern PetscErrorCode ResidualCalc(int ibi, FE* fem);

extern PetscErrorCode initFung(FE *fem, PetscReal initial_value);
extern PetscErrorCode updateFung(PetscReal *learning_rate, PetscInt epoch, FE *fem);
extern PetscErrorCode updateFungAdam(PetscReal *learning_rate, FE *fem, PetscInt time_step);


extern PetscErrorCode FungJacobian(PetscInt ibi, FE* fem, PetscReal epsilon);
extern PetscErrorCode FungUniJacobian(PetscInt ibi, FE* fem, PetscReal epsilon);
extern PetscErrorCode IrrVer(IBMNodes *ibm);
extern PetscErrorCode Patch(IBMNodes *ibm);


// FE         *fem; 
// IBMNodes   *ibm;
   
// PyObject *pModule = NULL;

static void CleanupAndFinalize(FE *fem, IBMNodes *ibm, PetscInt n_initialized) {
  PetscInt ibi;
  for (ibi = 0; ibi < n_initialized; ibi++) {
    FEM_DMPlexGeomDestroy(&fem[ibi]);
    Free(&fem[ibi]);
  }
  if (cuda_process) DestroyCudaWorkspace();
  if (kokkos_process) DestroyKokkosWorkspace();
  PetscFree(fem);
  PetscFree(ibm);
  PetscBarrier(PETSC_NULL);
  PetscFinalize();
}

static PetscErrorCode ResidualMonitor(SNES snes, PetscInt its, PetscReal rnorm, void *ctx)
{
  if (res_log_fp) {
    fprintf(res_log_fp, "%d %d %.6e\n", (int)ti, (int)its, (double)rnorm);
    fflush(res_log_fp);
  }
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  /* Pass "control.dat" as the options file so PETSc loads it BEFORE argv.
   * PetscInitialize order: file → PETSC_OPTIONS env → argv (command line).
   * This means command-line flags override control.dat, not the other way round. */
  PetscInitialize(&argc, &argv, "control.dat", help);

  /* Also load control.dat from -in_dir if provided (e.g. when running from a
   * different directory).  These are inserted AFTER argv so in_dir file wins
   * over the cwd control.dat but still loses to explicit command-line flags. */
  PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-in_dir", in_dir, 255, PETSC_NULL);
  if (in_dir[0] != '\0') {
    char control_path[512];
    snprintf(control_path, sizeof(control_path), "%s/%s", in_dir, "control.dat");
    if (access(control_path, F_OK) == 0) {
      PetscOptionsInsertFile(PETSC_COMM_WORLD, PETSC_NULL, control_path, PETSC_TRUE);
    }
  }
  
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nbody", &nbody, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-E", &E, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-mu", &mu, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-rho", &rho, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-h0", &h0, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_x", &char_length_x, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_y", &char_length_y, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-char_length_z", &char_length_z, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tisteps", &tisteps, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tistart", &tistart, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-dt", &dt, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-timeinteg", &timeinteg, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-explicit", &explicit, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-damping", &damping, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-dampfactor", &dampfactor, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tiout", &tiout, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-twod", &twod, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-membrane", &membrane, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-bending", &bending, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-contact", &contact, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-ConstitutiveLawNonLinear", &ConstitutiveLawNonLinear, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-outghost", &outghost, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-curvature", &curvature, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-manufactured", &manufactured, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-inverse", &inverse, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-fibersmooth", &fibersmooth, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-ressmooth", &ressmooth, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-res_smooth_itrs", &res_smooth_itrs, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-n_epochs", &n_epochs, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-init_flag", &init_flag, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-epoch_start", &epoch_start, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-learning_rate", &learning_rate, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-fib_smth_factor", &fib_smth_factor, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-initial_elas", &initial_elas, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-initial_poisson", &initial_poisson, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-patience", &patience, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-decay_factor", &decay_factor, PETSC_NULL);
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-improvement_threshold", &improvement_threshold, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-epoch_output", &epoch_output, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-Adam", &Adam, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-constrained_el", &constrained_el, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-uniform_fung", &uniform_fung, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-uniform_fiber_dir", &uniform_fiber_dir, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-epoch_update_jacobian", &epoch_update_jacobian, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-par_jac", &par_jac, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-progress_bar_show", &progress_bar_show, PETSC_NULL);

  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-muscle_activation", &muscle_activation, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-cuda_process", &cuda_process, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-kokkos_process", &kokkos_process, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-dmplex_geom_process", &dmplex_geom_process, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-lv_geom_process", &lv_geom_process, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-monitor_residual", &monitor_residual, PETSC_NULL);

  PetscCheck(!(cuda_process && kokkos_process), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "-cuda_process and -kokkos_process are mutually exclusive; choose one preprocessing backend");
  PetscCheck(!((cuda_process || kokkos_process) && dmplex_geom_process), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "-dmplex_geom_process is mutually exclusive with -cuda_process and -kokkos_process");
#if defined(BIOFEM_BACKEND_CUDA)
  PetscCheck(!kokkos_process, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "This executable was built for direct CUDA; use -cuda_process 1 -kokkos_process 0");
#elif defined(BIOFEM_BACKEND_KOKKOS)
  PetscCheck(!cuda_process, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "This executable was built for Kokkos; use -kokkos_process 1 -cuda_process 0");
#else
  PetscCheck(!cuda_process, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "This executable was built without direct CUDA; rebuild with USE_CUDA=1");
  PetscCheck(!kokkos_process, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "This executable was built without Kokkos; rebuild with USE_KOKKOS=1");
#endif
  
  PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-out_dir", out_dir, 255, PETSC_NULL);
  PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-in_dir", in_dir, 255, PETSC_NULL);
  
  PetscInt   ibi, k;
  PetscErrorCode ierr;
  PetscReal  tcyc=0.76, t, alpha[4];  alpha[0] = 0.25;  alpha[1] = 1./3.;  alpha[2] = 0.5;  alpha[3] = 1.0; 
  SNES       snes;  
  Mat        J = NULL; 
  FE         *fem; 
  IBMNodes   *ibm;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) mkdir(out_dir, 0755);

  PetscCalloc1(nbody, &fem);
  PetscCalloc1(nbody, &ibm);
    
  for (ibi=0; ibi<nbody; ibi++) {
    /* Link ibm pointer on ALL ranks so Free() is safe even when Init is skipped */
    fem[ibi].ibm = &ibm[ibi];

    PetscPrintf(PETSC_COMM_WORLD, "Initializing body %d \n", ibi);

    /* Phase 1: Mesh topology + C arrays (no Vecs created here). */
    if (lv_geom_process) {
      /* LV mesh: analytic + deterministic — run on ALL ranks. */
      LVParams lv_p;
      ierr = LVParamsCreate(&lv_p); CHKERRQ(ierr);
      if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF, "[lv] a=%.2f b=%.2f f_cut=%.2f N_theta=%d N_phi=%d alpha_endo=%.1f alpha_epi=%.1f\n",
                    lv_p.a, lv_p.b, lv_p.f_cut, (int)lv_p.N_theta, (int)lv_p.N_phi,
                    lv_p.alpha_endo, lv_p.alpha_epi);
      }
      ierr = CreateLVMesh(&ibm[ibi], &fem[ibi], &lv_p); CHKERRQ(ierr);
      if (rank == 0) {
        char lv_vtk_path[512];
        snprintf(lv_vtk_path, sizeof(lv_vtk_path), "%s/lv_fiber_%02d.vtk", out_dir, (int)ibi);
        ierr = WriteLVFiberVTK(&ibm[ibi], lv_vtk_path); CHKERRQ(ierr);
      }
    } else {
      /* Standard mesh: read from files. */
      Dimension(&ibm[ibi], ibi);
      Create(&ibm[ibi], &fem[ibi], ibi);
      Input(&ibm[ibi], ibi);
    }
    if (rank == 0) PetscPrintf(PETSC_COMM_SELF, "Dimension of body %d is %d \n", ibi, ibm[ibi].n_v);

    /* Phase 2: Active-strain data + DMPlex context.
     * InitActStrainProblem must precede FEM_DMPlexGeomSetup because Setup calls
     * ElemUpdateGeom0Subdiv which reads fem->act_data.elem_act_data. */
    if (dmplex_geom_process || muscle_activation) {
      ierr = InitActStrainProblem(&fem[ibi], ibi); CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD, "Active-Strain geometry/basis storage initialized.\n");
      if (muscle_activation) {
        ierr = FEM_DMPlexGeomSetup(&fem[ibi], PETSC_COMM_WORLD); CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD, "DMPlex geometry context initialized.\n");
      }
    }

    /* Phase 3: Vec creation — parallel (VecCreateMPI) when geom_ctx.initialized,
     * serial (VecCreateSeq) otherwise. */
    ierr = InitVecs(&fem[ibi]); CHKERRQ(ierr);

    /* Phase 4: Build ibm→global-DOF map for parallel Fint assembly.
     * Requires Vec ownership ranges from InitVecs. */
    if (muscle_activation) {
      ierr = FEM_DMPlexGeomBuildNodeMap(&fem[ibi]); CHKERRQ(ierr);
    }

    /* Phase 5: Initialize Vec values from reference coordinates. */
    Init(&fem[ibi], ibi);
    if (rank == 0) PetscPrintf(PETSC_COMM_SELF, "Init finished \n");

    if (explicit)  {VecSet(fem[ibi].Mass, 0.0);  VecSet(fem[ibi].Dissip, 0.0);  MassDamp(&fem[ibi]);}
    if (tistart)  {
      LocationIn(&fem[ibi], tistart, ibi, out_dir);
    }
  }

  if (dmplex_geom_process) {
    ierr = RunDMPlexGeomProcesses(fem); CHKERRQ(ierr);
    CleanupAndFinalize(fem, ibm, nbody);
    return 0;
  }

  if (kokkos_process) {

    ierr = RunKokkosProcesses(fem); CHKERRQ(ierr);

    CleanupAndFinalize(fem, ibm, nbody);
    return 0;
  }

  if (cuda_process) {

    ierr = RunCudaProcesses(fem); CHKERRQ(ierr);
    /* exit early after processing initialized bodies only */
    CleanupAndFinalize(fem, ibm, nbody);
    return 0;

  }



  if (manufactured_fexternal_export) {
    PetscCheck(muscle_activation && manufactured, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "-manufactured_fexternal_export requires both -manufactured 1 and -muscle_activation 1");

    for (ibi = 0; ibi < nbody; ibi++) {
      PetscCall(GenerateManufacturedFExternalAndKinematicsAllTimeSteps(&fem[ibi], ibi));
    }
    CleanupAndFinalize(fem, ibm, nbody);
    return 0;
  }

  
  if(inverse){        
    InvSolver(fem);    

    PetscBarrier(PETSC_NULL);  
  }
  
  if (tistart) tistart++;
  PetscPrintf(PETSC_COMM_WORLD, "Starting time-stepping loop from tistart = %d for %d steps \n", tistart, tisteps);

  if (monitor_residual && rank == 0) {
    char res_log_path[512];
    snprintf(res_log_path, sizeof(res_log_path), "%s/residual.log", out_dir);
    res_log_fp = fopen(res_log_path, "w");
    if (res_log_fp) {
      fprintf(res_log_fp, "# timestep snes_iter rnorm\n");
      fflush(res_log_fp);
      PetscPrintf(PETSC_COMM_WORLD, "[monitor] writing residuals to %s\n", res_log_path);
    }
  }

  // if (tistart==0) tisteps ++;

  for (ti=tistart; ti<tistart+tisteps; ti++) {

  for (ibi=0; ibi<nbody; ibi++) {
    PetscPrintf(PETSC_COMM_WORLD, "body:%d Time(%d)=%le\n", ibi, ti, ti*dt);
    // if (contact==1) {ContactZ(&fem[ibi]);}   
      

    if (explicit) {
      //------------------Explicit RK Solver
      for (k=0; k<4; k++) {	  
      FormRK(fem[ibi].Res, fem[ibi].x, fem[ibi].xn, fem[ibi].y, fem[ibi].yn, alpha[k], &fem[ibi]);
      }
	
      } else { 
	//------------------Implicit SNES solver
	/* Typical options
	   -fem_snes_mf
	   #-fem_snes_type tr
	   -fem_snes_max_it 20
	   -fem_snes_monitor
	   #-fem_ksp_type fgmres	 */

        Vec U;
        PetscErrorCode ierr;
        SNESConvergedReason reason;
        VecDuplicate(fem[ibi].x, &U);
        VecCopy(fem[ibi].x, U);

        /* Trace vertex 20 (first free ring) at key stages.
         * ibm->x_bp is replicated on all ranks after each FormFunctionFEM Allreduce. */
#define TRACE_V 20
        {
          IBMNodes *tibm = fem[ibi].ibm;
          if (!rank) PetscPrintf(PETSC_COMM_SELF,
            "[TRACE ti=%d] BEFORE SNES: x_bp[%d]=(%g,%g,%g)  disp_z=%g\n",
            ti, TRACE_V,
            tibm->x_bp[TRACE_V], tibm->y_bp[TRACE_V], tibm->z_bp[TRACE_V],
            tibm->z_bp[TRACE_V]-tibm->z_bp0[TRACE_V]);
        }

        //SNES
        SNESCreate(PETSC_COMM_WORLD, &snes);
        SNESSetFunction(snes, fem[ibi].Res, FormFunctionFEM, (void *)&fem[ibi]);
        SNESAppendOptionsPrefix(snes, "fem_");
        MatCreateSNESMF(snes, &J);  //MatrixFree
        // SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, (void *)&fem[ibi]);
        SNESSetJacobian(snes, J, J, NULL, (void *)&fem[ibi]);
        SNESSetFromOptions(snes);
        if (monitor_residual && res_log_fp) {
          SNESMonitorSet(snes, ResidualMonitor, NULL, NULL);
        }

	        ierr = SNESSolve(snes, PETSC_NULL, U);
	        if (ierr) {
	          PetscPrintf(PETSC_COMM_WORLD,
	                      "SNESSolve failed with ierr=%d for body %d at step %d\n",
	                      (int)ierr, (int)ibi, (int)ti);
	          SNESDestroy(&snes);
	          MatDestroy(&J);
	          VecDestroy(&U);
	          PetscFinalize();
	          return (int)ierr;
	        }

	        ierr = SNESGetConvergedReason(snes, &reason);
	        if (ierr) {
	          PetscPrintf(PETSC_COMM_WORLD,
	                      "SNESGetConvergedReason failed with ierr=%d for body %d at step %d\n",
	                      (int)ierr, (int)ibi, (int)ti);
	          SNESDestroy(&snes);
	          MatDestroy(&J);
	          VecDestroy(&U);
	          PetscFinalize();
	          return (int)ierr;
	        }

        {
          IBMNodes *tibm = fem[ibi].ibm;
          if (!rank) PetscPrintf(PETSC_COMM_SELF,
            "[TRACE ti=%d] AFTER  SNES: x_bp[%d]=(%g,%g,%g)  disp_z=%g  reason=%d\n",
            ti, TRACE_V,
            tibm->x_bp[TRACE_V], tibm->y_bp[TRACE_V], tibm->z_bp[TRACE_V],
            tibm->z_bp[TRACE_V]-tibm->z_bp0[TRACE_V], (int)reason);
        }

	        if (reason >= 0) {
	          VecCopy(U, fem[ibi].x);
	        } else {
	          PetscPrintf(PETSC_COMM_WORLD,
	                      "Skipping solution update because SNES diverged (reason=%d) for body %d at step %d\n",
	                      (int)reason, (int)ibi, (int)ti);
	          /* ibm->x_bp was dirtied by FormFunctionFEM during failed SNES iterations.
	           * Restore it from fem->x so Output and xAccVel use the correct state. */
	          {
	            DMPlexGeomCtx *gctx_ = &fem[ibi].geom_ctx;
	            IBMNodes      *ibm_  = fem[ibi].ibm;
	            if (gctx_->initialized) {
	              const PetscReal *fx;
	              PetscMPIInt fmyrank_;
	              MPI_Comm_rank(PETSC_COMM_WORLD, &fmyrank_);
	              PetscMemzero(ibm_->x_bp, ibm_->n_v * sizeof(PetscReal));
	              PetscMemzero(ibm_->y_bp, ibm_->n_v * sizeof(PetscReal));
	              PetscMemzero(ibm_->z_bp, ibm_->n_v * sizeof(PetscReal));
	              VecGetArrayRead(fem[ibi].x, &fx);
	              PetscInt oi_ = 0;
	              for (PetscInt lv = 0; lv < gctx_->nLocalVerts; ++lv) {
	                PetscInt v = gctx_->origVert[lv];
	                if (v < 0 || v >= ibm_->n_v || gctx_->ownerRank[v] != fmyrank_) continue;
	                ibm_->x_bp[v] = fx[oi_*dof  ];
	                ibm_->y_bp[v] = fx[oi_*dof+1];
	                ibm_->z_bp[v] = fx[oi_*dof+2];
	                oi_++;
	              }
	              VecRestoreArrayRead(fem[ibi].x, &fx);
	              MPI_Allreduce(MPI_IN_PLACE, ibm_->x_bp, ibm_->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
	              MPI_Allreduce(MPI_IN_PLACE, ibm_->y_bp, ibm_->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
	              MPI_Allreduce(MPI_IN_PLACE, ibm_->z_bp, ibm_->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
	            }
	          }
	        }

        {
          IBMNodes *tibm = fem[ibi].ibm;
          if (!rank) PetscPrintf(PETSC_COMM_SELF,
            "[TRACE ti=%d] AFTER UPDATE: x_bp[%d]=(%g,%g,%g)  disp_z=%g\n",
            ti, TRACE_V,
            tibm->x_bp[TRACE_V], tibm->y_bp[TRACE_V], tibm->z_bp[TRACE_V],
            tibm->z_bp[TRACE_V]-tibm->z_bp0[TRACE_V]);
        }

        /* Only destroy if SNESSolve didn't corrupt the object */
        if (snes != NULL) {
          PetscErrorCode destroy_ierr = SNESDestroy(&snes);
          if (destroy_ierr != 0) {
            PetscPrintf(PETSC_COMM_WORLD, " WARNING: SNESDestroy failed with ierr = %d\n", destroy_ierr);
          } else {
            // PetscPrintf(PETSC_COMM_SELF, " after SNESDestroy\n");
          }
        }

        /* Destroy the Jacobian matrix created by MatCreateSNESMF */
        MatDestroy(&J);

        VecDestroy(&U);

      }
      VecCopy(fem[ibi].x, fem[ibi].y); //for contact energy calculations
      xAccVel(&fem[ibi]);
      //VecCopy(fem[ibi].xd, fem[ibi].y);  

       if (contact==1) {ContactZ(&fem[ibi]);}   

       
    } //ibi

    t = dt*(ti);
    //  t = t - ((PetscInt)(t/tcyc))*tcyc;
    //if (contact==1 && t>0.03) {Fcontact(fem);} //dynamic_bhv
    //if (contact==2 && t>0.075) {Fcontact2(fem);} //dynamic_bhv
    if (contact) {Fcontact(fem);} //static_bhv
    
    for (ibi=0; ibi<nbody; ibi++) {

      // Contact(&fem[ibi]);
      VecCopy(fem[ibi].xn, fem[ibi].xnm1);
      VecCopy(fem[ibi].x, fem[ibi].xn);

// {
//           PetscInt node = 117; /* change as needed */
//           PetscReal *FF = NULL;
//           PetscInt n_v = ibm->n_v;
//           if (node >= 0 && node < n_v) {
//             VecGetArray(fem[ibi].Fint, &FF);
//             PetscPrintf(PETSC_COMM_SELF, "Fint af cont at node %d: %le %le %le\n",
//                         node, FF[dof*node], FF[dof*node+1], FF[dof*node+2]);
//             VecRestoreArray(fem[ibi].Fint, &FF);
//           } else {
//             PetscPrintf(PETSC_COMM_SELF, "requested node %d out of range (n_v=%d)\n", node, n_v);
//           }
//         }

      PetscReal norm=0.0, normv=0.0, norma=0.0, normfint=0.; 
      VecCopy(fem[ibi].x, fem[ibi].dx);
      VecAXPY(fem[ibi].dx, -1., fem[ibi].xnm1);
      VecNorm(fem[ibi].dx, NORM_2, &norm);
      VecNorm(fem[ibi].xd, NORM_INFINITY, &normv);
      VecNorm(fem[ibi].xdd, NORM_INFINITY, &norma);
      VecNorm(fem[ibi].Fint, NORM_INFINITY, &normfint);

      PetscPrintf(PETSC_COMM_WORLD, "body:%d Norm(x-xn)= %le Vel %f Acc %f Fint %f\n",ibi,norm,normv,norma, normfint);
    }

    //    if (contact) {Fcontact(fem);} //static_bhv
    // Printout the results
    if (ti == (ti/tiout)*tiout){
      for (ibi=0; ibi<nbody; ibi++) {
        /* ti+1: file 0 = reference (Init), file 1..N = post-SNES deformed states */
        Output(&fem[ibi], ti+1, ibi, out_dir);
        if (outghost) {OutputGhost(&fem[ibi], ti, ibi, out_dir);}
      }
    }
  }// ti

  if (res_log_fp) { fclose(res_log_fp); res_log_fp = NULL; }

  /* Finish UP: output final state and cleanup on rank 0 only */
  if (rank == 0) {
    for (ibi=0; ibi<nbody; ibi++) {
      LocationOut(&fem[ibi], ti+1, ibi, out_dir);
      if(outghost){OutputGhost(&fem[ibi], ti, ibi, out_dir);}
    }
  }
  CleanupAndFinalize(fem, ibm, nbody);
  return(0);
}






//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FormRK(Vec R, Vec x, Vec xn, Vec y, Vec yn, PetscReal alpha, FE *fem) {

  //y=dx/dt  
  IBMNodes   *ibm=fem->ibm;
  PetscErrorCode ierr;
  PetscReal  *xx, *RR, *RRes, *FF;
  PetscInt   nv;
  Vec        w1, w2;

  VecDuplicate(R, &w1);
  VecDuplicate(R, &w2);

  //---------Update the location 
  VecGetArray(x, &xx);
  for (nv=0; nv<ibm->n_v+ibm->n_ghosts; nv++) {
    ibm->x_bp[nv] = xx[nv*dof  ];
    ibm->y_bp[nv] = xx[nv*dof+1];
    ibm->z_bp[nv] = xx[nv*dof+2];
    if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}  //2d case
  }
  VecRestoreArray(x, &xx);

  //---------------------------Patchtest---------------------------------------------------------------------
  /* PetscInt i, start, end, edge_n, edge; */
  /* for (edge_n=0; edge_n<ibm->n_edge; edge_n++){ */
  /*   start = 0; end = 0; */
  /*   for (edge=0; edge<edge_n+1; edge++) { */
  /*     end += ibm->n_bnodes[edge]; */
  /*   } */
  /*   start = end - ibm->n_bnodes[edge_n]; */

  /*   for (i=start; i<end; i++) { //moving boundary nodes */
  /*     nv = ibm->bnodes[i]; */
  /*     ibm->x_bp[nv] = 2.*ibm->x_bp0[nv]; */
  /*     ibm->y_bp[nv] = ibm->y_bp0[nv]; */
  /*     ibm->z_bp[nv] = ibm->z_bp0[nv]; */
  /*   } */
  /* } */

  /* for (nv=ibm->n_v; nv<ibm->n_v+ibm->n_ghosts; nv++) { //moving ghost nodes */
  /*   ibm->x_bp[nv] = 2.*ibm->x_bp0[nv]; */
  /*   ibm->y_bp[nv] = ibm->y_bp0[nv]; */
  /*   ibm->z_bp[nv] = ibm->z_bp0[nv]; */
  /* } */
  //---------------------------Patchtest---------------------------------------------------------------------

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {     
      PatchLoc(ibm); 
      GhostLoc(fem);
    } else if (curvature==6) {
      //GlobalGhost(ibm);
    }
  }
  
  //---------Compute Forces then Residual
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0); 
  VecSet(R, 0.0);  VecSet(fem->FJ, 0.0); 

  FInternal(fem);
  ierr = FExternal(fem); CHKERRQ(ierr);

  VecWAXPY(R, -1., fem->Fint, fem->Fext);
  
  VecPointwiseMult(w1, fem->Dissip,y);
  VecAXPY(R, -1., w1);
  VecPointwiseDivide(w2, R, fem->Mass);
  VecCopy(w2, R);

  VecWAXPY(x, alpha*dt, y, xn);
  VecWAXPY(y, alpha*dt, R, yn);

  //---------2d case
  if(twod){
    VecGetArray(x, &xx);
    VecGetArray(R, &RR); 
    for (nv=0; nv<ibm->n_v; nv++) {
      xx[nv*dof+2] = ibm->z_bp0[nv];  
      RR[nv*dof+2] = 0.0;
    }
  VecRestoreArray(R, &RR);
  VecRestoreArray(x, &xx);
  }
  
  //---------------------------Patchtest---------------------------------------------------------------------
  /* VecGetArray(x, &xx); */
  /* for (edge_n=0; edge_n<ibm->n_edge; edge_n++){ */
  /*   start = 0; end = 0; */
  /*   for (edge=0; edge<edge_n+1; edge++) { */
  /*     end += ibm->n_bnodes[edge]; */
  /*   } */
  /*   start = end - ibm->n_bnodes[edge_n]; */

  /*   for (i=start; i<end; i++) { */
  /*     nv = ibm->bnodes[i]; */
  /*     xx[nv*dof] = 2.*ibm->x_bp0[nv]; */
  /*     xx[nv*dof+1] = ibm->y_bp0[nv]; */
  /*     xx[nv*dof+2] = ibm->z_bp0[nv]; */
  /*   } */
  /* } */

  /* for (nv=ibm->n_v; nv<ibm->n_v+ibm->n_ghosts; nv++) { */
  /*   xx[nv*dof] = 2.*ibm->x_bp0[nv]; */
  /*   xx[nv*dof+1] = ibm->y_bp0[nv]; */
  /*   xx[nv*dof+2] = ibm->z_bp0[nv]; */
  /* } */

  /* VecRestoreArray(x, &xx); */
  //---------------------------Patchtest---------------------------------------------------------------------

  VecDestroy(&w1);  VecDestroy(&w2);
  
  return(0);
}

//-----------------------------------------------------------------------------------------------
PetscErrorCode FormFunctionFEM(SNES snes, Vec x, Vec R, void *ctx) {

  FE         *fem=(FE *)ctx;
  IBMNodes   *ibm=fem->ibm;
  Mat        Jmat = NULL;
  PetscErrorCode ierr;
  const PetscReal *xx;
  PetscReal  *RR, *RRes,*FF;
  PetscInt   nv, ec;
  //---------Update the location
  {
    DMPlexGeomCtx *gctx = &fem->geom_ctx;
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
    if (gctx->initialized) {
      /* Parallel path: each rank updates its owned vertices, then Allreduce gives
       * all ranks the complete ibm->x_bp[0..n_v-1]. */
      PetscMPIInt fmyrank;
      MPI_Comm_rank(PETSC_COMM_WORLD, &fmyrank);
      PetscMemzero(ibm->x_bp, ibm->n_v * sizeof(PetscReal));
      PetscMemzero(ibm->y_bp, ibm->n_v * sizeof(PetscReal));
      PetscMemzero(ibm->z_bp, ibm->n_v * sizeof(PetscReal));
      {
        PetscInt oi = 0;
        for (PetscInt lv = 0; lv < gctx->nLocalVerts; ++lv) {
          PetscInt v = gctx->origVert[lv];
          if (v < 0 || v >= ibm->n_v || gctx->ownerRank[v] != fmyrank) continue;
          ibm->x_bp[v] = xx[oi*dof  ];
          ibm->y_bp[v] = xx[oi*dof+1];
          ibm->z_bp[v] = xx[oi*dof+2];
          if (twod) ibm->z_bp[v] = ibm->z_bp0[v];
          oi++;
        }
      }
      ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
      MPI_Allreduce(MPI_IN_PLACE, ibm->x_bp, ibm->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, ibm->y_bp, ibm->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, ibm->z_bp, ibm->n_v, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD);
    } else {
      /* Serial path: Vec covers all nodes including ghost stencil nodes. */
      for (nv=0; nv<ibm->n_v + ibm->n_ghosts; nv++) {
        ibm->x_bp[nv] = xx[nv*dof  ];
        ibm->y_bp[nv] = xx[nv*dof+1];
        ibm->z_bp[nv] = xx[nv*dof+2];
        if(twod){ibm->z_bp[nv] = ibm->z_bp0[nv];}
      }
      ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
    }
  }

  VecCopy(x, fem->x);

  //for displacement BCs
  //MoveBoundary(3, fem);
  //

  AreaNormal(ibm);
  if (bending){
    if (curvature==1) {
      PatchLoc(ibm);
      GhostLoc(fem);
    } else if (curvature==6) {
      // EdgeDirectionalFix(0, 1, fem, R);
      // EdgeDirectionalFix(3, 0, fem, R);
      // EdgeFreeR(fem, R);
      GlobalGhost(ibm);
      GhostDirectionalFix(ibm, 0, 1);
      GhostDirectionalFix(ibm, 3, 0);
    }
  }

  //---------Compute Forces then Residual
  VecSet(fem->Fext, 0.0);  VecSet(fem->Fint, 0.0);  VecSet(fem->Fdyn, 0.0);
  VecSet(R, 0.0);  VecSet(fem->FJ, 0.0);
  for (nv=0; nv<ibm->n_v; nv++)  fem->IE[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;

  if (muscle_activation){
    ierr = FInternalAct(fem); CHKERRQ(ierr);
  }
  else{
    FInternal(fem);
  }
  
  if(tisteps>1) {FDynamic(fem);}
  ierr = FExternal(fem); CHKERRQ(ierr);

  VecWAXPY(R,-1., fem->Fext, fem->Fint);
  VecAXPY(R,1., fem->Fdyn);

  //---------2d case
  // if(twod){
  //   VecGetArray(fem->x, &xx);
  //   VecGetArray(R, &RR);
  //   for (nv=0; nv<ibm->n_v; nv++) {
  //     xx[nv*dof+2]=ibm->z_bp0[nv];
  //     RR[nv*dof+2]=0.0;
  //   }
  //   VecRestoreArray(R, &RR);
  //   VecRestoreArray(fem->x, &xx);
  // }

  /* Fix the LV apex ring (edge_n=0) in all 3 directions; base (edge_n=1) is free. */
  ierr = EdgeDirectionalFix(0, 0, fem, R); CHKERRQ(ierr);
  ierr = EdgeDirectionalFix(0, 1, fem, R); CHKERRQ(ierr);
  ierr = EdgeDirectionalFix(0, 2, fem, R); CHKERRQ(ierr);
  ierr = EdgeFreeR(fem, R); CHKERRQ(ierr);  /* no-op for LV (n_ghosts=0) */
  
  // GlobalGhost(ibm);

  PetscReal fext_inf = 0.0, scale = 1.0;
  ierr = VecNorm(fem->Fext, NORM_INFINITY, &fext_inf); CHKERRQ(ierr);

  /* Avoid dividing by ~0 when load is tiny */
  scale = 1.0 / PetscMax(fext_inf, 1.0);   /* choose 1.0 as floor (units) */

  ierr = VecScale(R, scale); CHKERRQ(ierr);

  //for displacement BCs
  //
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FInternal(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscInt       i, ec;
  PetscReal      Fm[9], Fb[42], Fint[9];
  struct Cmpnts  x1, x2, x3, X1, X2, X3;
  PetscInt       n1e, n2e, n3e, n4e, n5e, n6e, j;
  PetscReal      s1=0., s2=0., s3=0.;

  for (i=0; i<9; i++) {Fm[i]=0.0; Fint[i]=0.0;}
  for (i=0; i<42; i++) {Fb[i]=0.0;}

  PetscReal  *FF,*FFJ;
  VecGetArray(fem->Fint, &FF);
  VecGetArray(fem->FJ, &FFJ);
  
  //PetscPrintf(PETSC_COMM_SELF, "CHECK size inside FINTERNAL = %d\n", ibm->n_v);

  for (ec=0; ec<ibm->n_elmt; ec++) {
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

    if (curvature==1) {
      if (membrane) {
	Fmembrane(ec, X1, X2, X3, x1, x2, x3, Fm, fem);
      }
    }
    //PetscPrintf(PETSC_COMM_SELF, "Check before Fbending %d\n", bending);

    if (bending) {
      // if (muscle_activation){
      //   ElemUpdFint(fem, ec, Fb);
      // }
      // else{
        Fbending(ec, X1, X2, X3, x1, x2, x3, Fb, fem);
      // }
      
    }
    //PetscPrintf(PETSC_COMM_SELF, "Check after Fbending!\n");
    
    if (curvature==1) { //Only bending     
      for (i=0; i<9; i++) {
	Fint[i]=Fm[i]+Fb[i];
      }  //end loop over nodes of each element
      
      FF[n1e*dof] +=Fint[0];
      FF[n1e*dof+1] +=Fint[1];
      FF[n1e*dof+2] +=Fint[2];
      
      FF[n2e*dof] +=Fint[3];
      FF[n2e*dof+1] +=Fint[4];
      FF[n2e*dof+2] +=Fint[5];
      
      FF[n3e*dof] +=Fint[6];
      FF[n3e*dof+1] +=Fint[7];
      FF[n3e*dof+2] +=Fint[8];     
  
      if(n4e !=1000000){ // Front node is inside domain
	FF[n4e*dof] +=Fb[9];
	FF[n4e*dof+1] +=Fb[10];
	FF[n4e*dof+2] +=Fb[11];
      }else{ // Front node is ghost
	FFJ[n1e*dof] +=Fb[9];
	FFJ[n1e*dof+1] +=Fb[10];
	FFJ[n1e*dof+2] +=Fb[11];
      }
      
      if(n5e !=1000000){// Front node is inside domain
	FF[n5e*dof] +=Fb[12];
	FF[n5e*dof+1] +=Fb[13];
	FF[n5e*dof+2] +=Fb[14];
      }else{ // Front node is ghost
	FFJ[n2e*dof] +=Fb[12];
	FFJ[n2e*dof+1] +=Fb[13];
	FFJ[n2e*dof+2] +=Fb[14];
      }
      
      if(n6e !=1000000){// Front node is inside domain
	FF[n6e*dof] +=Fb[15];
	FF[n6e*dof+1] +=Fb[16];
	FF[n6e*dof+2] +=Fb[17];
      }else{ // Front node is ghost
	FFJ[n3e*dof] +=Fb[15];
	FFJ[n3e*dof+1] +=Fb[16];
	FFJ[n3e*dof+2] +=Fb[17];
      }

    } else if (curvature==6) {
      
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
  }//end loop over elements
  
  VecRestoreArray(fem->FJ, &FFJ); 
  VecRestoreArray(fem->Fint, &FF); 
  
  if (bending && curvature==1) {ModifyFbending(fem);}
 
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode FDynamic(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  DMPlexGeomCtx  *gctx = &fem->geom_ctx;

  PetscInt       i;
  PetscReal      M[9], C[9], Fd[9], x[9], xn[9], xnm1[9], xd[9], xdd[9];
  PetscInt       n1e, n2e, n3e;
  PetscReal      Gama=0.5, Beta=0.25;
  PetscReal      M1, C1, M2, C2, M3, C3;

  for (i=0; i<9; i++) {M[i]=0.0; C[i]=0.0; Fd[i]=0.0;}
  M1=1./(Beta*pow(dt,2));  M2=1./(Beta*dt);  M3=(1./(2*Beta))-1.;
  C1=Gama/(Beta*dt);  C2=(Gama/Beta)-1;  C3=dt*(Gama/(2.*Beta)-1);

  if (gctx->initialized) {
    /* Parallel path.  x (current position) is already fully replicated in
     * ibm->x_bp/y_bp/z_bp by FormFunctionFEM before FDynamic runs.  xn/xnm1/
     * xd/xdd live only in distributed Vecs, so replicate them the same way
     * (zero, fill owned entries, Allreduce-sum) — mirrors the ibm->x_bp sync
     * in FormFunctionFEM.  Fdyn contributions are written with VecSetValues
     * against the global DOF map, mirroring FInternalAct's Fint assembly, so
     * PETSc routes off-process contributions during VecAssemblyBegin/End. */
    PetscErrorCode ierr;
    PetscInt       nv = ibm->n_v;
    PetscReal      *xn_bp, *yn_bp, *zn_bp, *xnm1_bp, *ynm1_bp, *znm1_bp;
    PetscReal      *xd_bp, *yd_bp, *zd_bp, *xdd_bp, *ydd_bp, *zdd_bp;
    PetscMPIInt    myrank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);

    PetscCall(PetscMalloc1(nv, &xn_bp));   PetscCall(PetscMalloc1(nv, &yn_bp));   PetscCall(PetscMalloc1(nv, &zn_bp));
    PetscCall(PetscMalloc1(nv, &xnm1_bp)); PetscCall(PetscMalloc1(nv, &ynm1_bp)); PetscCall(PetscMalloc1(nv, &znm1_bp));
    PetscCall(PetscMalloc1(nv, &xd_bp));   PetscCall(PetscMalloc1(nv, &yd_bp));   PetscCall(PetscMalloc1(nv, &zd_bp));
    PetscCall(PetscMalloc1(nv, &xdd_bp));  PetscCall(PetscMalloc1(nv, &ydd_bp));  PetscCall(PetscMalloc1(nv, &zdd_bp));

#define FDYN_GATHER(vec, bx, by, bz) do { \
    const PetscReal *_a; \
    PetscMemzero((bx), nv * sizeof(PetscReal)); \
    PetscMemzero((by), nv * sizeof(PetscReal)); \
    PetscMemzero((bz), nv * sizeof(PetscReal)); \
    VecGetArrayRead((vec), &_a); \
    { \
      PetscInt oi = 0; \
      for (PetscInt lv = 0; lv < gctx->nLocalVerts; ++lv) { \
        PetscInt v = gctx->origVert[lv]; \
        if (v < 0 || v >= nv || gctx->ownerRank[v] != myrank) continue; \
        (bx)[v] = _a[oi*dof  ]; \
        (by)[v] = _a[oi*dof+1]; \
        (bz)[v] = _a[oi*dof+2]; \
        oi++; \
      } \
    } \
    VecRestoreArrayRead((vec), &_a); \
    MPI_Allreduce(MPI_IN_PLACE, (bx), nv, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD); \
    MPI_Allreduce(MPI_IN_PLACE, (by), nv, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD); \
    MPI_Allreduce(MPI_IN_PLACE, (bz), nv, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD); \
  } while (0)

    FDYN_GATHER(fem->xn,   xn_bp,   yn_bp,   zn_bp);
    FDYN_GATHER(fem->xnm1, xnm1_bp, ynm1_bp, znm1_bp);
    FDYN_GATHER(fem->xd,   xd_bp,   yd_bp,   zd_bp);
    FDYN_GATHER(fem->xdd,  xdd_bp,  ydd_bp,  zdd_bp);
#undef FDYN_GATHER

    for (PetscInt lc = 0; lc < gctx->layout.nLocalCells; ++lc) {
      const PetscInt ec = gctx->layout.orig_cell[lc];
      if (ec < 0 || ec >= ibm->n_elmt) continue;
      PetscReal facc, fvel;

      n1e=ibm->nv1[ec]; n2e=ibm->nv2[ec]; n3e=ibm->nv3[ec];

      x[0]=ibm->x_bp[n1e];  x[1]=ibm->y_bp[n1e]; x[2]=ibm->z_bp[n1e];
      x[3]=ibm->x_bp[n2e];  x[4]=ibm->y_bp[n2e]; x[5]=ibm->z_bp[n2e];
      x[6]=ibm->x_bp[n3e];  x[7]=ibm->y_bp[n3e]; x[8]=ibm->z_bp[n3e];

      xn[0]=xn_bp[n1e];  xn[1]=yn_bp[n1e];  xn[2]=zn_bp[n1e];
      xn[3]=xn_bp[n2e];  xn[4]=yn_bp[n2e];  xn[5]=zn_bp[n2e];
      xn[6]=xn_bp[n3e];  xn[7]=yn_bp[n3e];  xn[8]=zn_bp[n3e];

      xnm1[0]=xnm1_bp[n1e];  xnm1[1]=ynm1_bp[n1e];  xnm1[2]=znm1_bp[n1e];
      xnm1[3]=xnm1_bp[n2e];  xnm1[4]=ynm1_bp[n2e];  xnm1[5]=znm1_bp[n2e];
      xnm1[6]=xnm1_bp[n3e];  xnm1[7]=ynm1_bp[n3e];  xnm1[8]=znm1_bp[n3e];

      xd[0]=xd_bp[n1e];  xd[1]=yd_bp[n1e];  xd[2]=zd_bp[n1e];
      xd[3]=xd_bp[n2e];  xd[4]=yd_bp[n2e];  xd[5]=zd_bp[n2e];
      xd[6]=xd_bp[n3e];  xd[7]=yd_bp[n3e];  xd[8]=zd_bp[n3e];

      xdd[0]=xdd_bp[n1e];  xdd[1]=ydd_bp[n1e];  xdd[2]=zdd_bp[n1e];
      xdd[3]=xdd_bp[n2e];  xdd[4]=ydd_bp[n2e];  xdd[5]=zdd_bp[n2e];
      xdd[6]=xdd_bp[n3e];  xdd[7]=ydd_bp[n3e];  xdd[8]=zdd_bp[n3e];

      Mass(ibm, ec, M);
      if (damping) { Damp(ibm, M, C); }

      if (timeinteg==0) { //Newmark constant average acceleration
        for (i=0; i<9; i++) {
          facc=M[i]*(M1*xn[i]+M2*xd[i]+M3*xdd[i]);
          fvel=C[i]*(C1*xn[i]+C2*xd[i]+C3*xdd[i]);
          Fd[i]=M[i]*M1*x[i]+C[i]*C1*x[i]-facc-fvel;
        }
      } else if (timeinteg==1) { //central
        for (i=0; i<9; i++) {
          Fd[i]=M[i]*(x[i]-2*xn[i]+xnm1[i])/(dt*dt)+C[i]*(x[i]-xn[i])/dt;
        }
      }

      {
        PetscInt nodes[3] = {n1e, n2e, n3e};
        for (PetscInt k=0; k<3; k++) {
          PetscInt gdof0 = gctx->ibm_to_global_dof0[nodes[k]];
          if (gdof0 < 0) continue;
          PetscInt  dofs[3] = {gdof0, gdof0+1, gdof0+2};
          PetscReal vals[3] = {Fd[3*k], Fd[3*k+1], Fd[3*k+2]};
          ierr = VecSetValues(fem->Fdyn, 3, dofs, vals, ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }//end loop over local elements

    ierr = VecAssemblyBegin(fem->Fdyn); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(fem->Fdyn);   CHKERRQ(ierr);

    PetscCall(PetscFree(xn_bp));   PetscCall(PetscFree(yn_bp));   PetscCall(PetscFree(zn_bp));
    PetscCall(PetscFree(xnm1_bp)); PetscCall(PetscFree(ynm1_bp)); PetscCall(PetscFree(znm1_bp));
    PetscCall(PetscFree(xd_bp));   PetscCall(PetscFree(yd_bp));   PetscCall(PetscFree(zd_bp));
    PetscCall(PetscFree(xdd_bp));  PetscCall(PetscFree(ydd_bp));  PetscCall(PetscFree(zdd_bp));

    return 0;
  }

  PetscReal  *xx,*xxn,*xxnm1,*xxd,*xxdd,*FF,facc,fvel;
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->xnm1, &xxnm1);
  VecGetArray(fem->x, &xx);
  VecGetArray(fem->Fdyn, &FF);

  for (ec=0; ec<ibm->n_elmt+2*ibm->n_ghosts; ec++) {
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
  
    x[0]=ibm->x_bp[n1e];  x[1]=ibm->y_bp[n1e]; x[2]=ibm->z_bp[n1e];
    x[3]=ibm->x_bp[n2e];  x[4]=ibm->y_bp[n2e]; x[5]=ibm->z_bp[n2e];
    x[6]=ibm->x_bp[n3e];  x[7]=ibm->y_bp[n3e]; x[8]=ibm->z_bp[n3e];
           
    xn[0]=xxn[n1e*dof];  xn[1]=xxn[n1e*dof+1];  xn[2]=xxn[n1e*dof+2];
    xn[3]=xxn[n2e*dof];  xn[4]=xxn[n2e*dof+1];  xn[5]=xxn[n2e*dof+2];
    xn[6]=xxn[n3e*dof];  xn[7]=xxn[n3e*dof+1];  xn[8]=xxn[n3e*dof+2];

    xnm1[0]=xxnm1[n1e*dof];  xnm1[1]=xxnm1[n1e*dof+1];  xnm1[2]=xxnm1[n1e*dof+2];
    xnm1[3]=xxnm1[n2e*dof];  xnm1[4]=xxnm1[n2e*dof+1];  xnm1[5]=xxnm1[n2e*dof+2];
    xnm1[6]=xxnm1[n3e*dof];  xnm1[7]=xxnm1[n3e*dof+1];  xnm1[8]=xxnm1[n3e*dof+2];

    xd[0]=xxd[n1e*dof];  xd[1]=xxd[n1e*dof+1];  xd[2]=xxd[n1e*dof+2];
    xd[3]=xxd[n2e*dof];  xd[4]=xxd[n2e*dof+1];  xd[5]=xxd[n2e*dof+2];
    xd[6]=xxd[n3e*dof];  xd[7]=xxd[n3e*dof+1];  xd[8]=xxd[n3e*dof+2];

    xdd[0]=xxdd[n1e*dof];  xdd[1]=xxdd[n1e*dof+1];  xdd[2]=xxdd[n1e*dof+2];
    xdd[3]=xxdd[n2e*dof];  xdd[4]=xxdd[n2e*dof+1];  xdd[5]=xxdd[n2e*dof+2];
    xdd[6]=xxdd[n3e*dof];  xdd[7]=xxdd[n3e*dof+1];  xdd[8]=xxdd[n3e*dof+2];
     
    Mass(ibm, ec, M);
    if(damping) {Damp(ibm, M,C);}

    if(timeinteg==0){ //Newmark constant average acceleration
      for (i=0; i<9; i++) {
	facc=M[i]*(M1*xn[i]+M2*xd[i]+M3*xdd[i]);
	fvel=C[i]*(C1*xn[i]+C2*xd[i]+C3*xdd[i]);
	Fd[i]=M[i]*M1*x[i]+C[i]*C1*x[i]-facc-fvel;
      }

    }else if (timeinteg==1){ //central
      for (i=0; i<9; i++) {
	Fd[i]=M[i]*(x[i]-2*xn[i]+xnm1[i])/(dt*dt)+C[i]*(x[i]-xn[i])/dt;
      }
    }
    //AddToVector
    
    FF[n1e*dof] +=Fd[0];
    FF[n1e*dof+1] +=Fd[1];
    FF[n1e*dof+2] +=Fd[2];
    
    FF[n2e*dof] +=Fd[3];
    FF[n2e*dof+1] +=Fd[4];
    FF[n2e*dof+2] +=Fd[5];

    FF[n3e*dof] +=Fd[6];
    FF[n3e*dof+1] +=Fd[7];
    FF[n3e*dof+2] +=Fd[8];
   
  }//end loop over elements

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->xnm1, &xxnm1);
  VecRestoreArray(fem->xn, &xxn);
  VecRestoreArray(fem->Fdyn, &FF);

  return(0);
} 

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode xAccVel(FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscReal      M1, C1, M2, C2, M3, C3, M[9], C[9];
  PetscReal      *xx, *xxn, *xxd, *xxdd;
  PetscReal      xxddn1, xxddn2, xxddn3;
  PetscReal      xxdn1, xxdn2, xxdn3;
  PetscReal      Gama=0.5, Beta=0.25;
  PetscInt       nv;
  
  M1 = 1./(Beta*pow(dt,2));  M2 = 1./(Beta*dt);  M3 = (1./(2*Beta)) - 1.;
  // C1 = Gama/(Beta*dt);  C2 = (Gama/Beta) - 1;  C3 = dt*(Gama/(2.*Beta) - 1);
  C2=(1.-Gama)*dt; C3=Gama*dt;
  
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->x, &xx);

  {
    DMPlexGeomCtx *gctx = &fem->geom_ctx;
    PetscInt n_iter = gctx->initialized ? ibm->n_v : ibm->n_v + ibm->n_ghosts;
    for (nv=0; nv<n_iter; nv++) {
      /* In parallel, use local Vec index li; serial: li == nv. */
      PetscInt li = (gctx->initialized) ? gctx->ibm_to_local_idx[nv] : nv;
      if (li < 0) continue;

      xxdn1 = xxd[dof*li  ];
      xxdn2 = xxd[dof*li+1];
      xxdn3 = xxd[dof*li+2];
      xxddn1= xxdd[dof*li  ];
      xxddn2= xxdd[dof*li+1];
      xxddn3= xxdd[dof*li+2];

      xxdd[dof*li  ] = M1*(xx[li*dof  ] - xxn[li*dof  ]) - M2*xxd[li*dof  ] - M3*xxdd[li*dof  ];
      xxdd[dof*li+1] = M1*(xx[li*dof+1] - xxn[li*dof+1]) - M2*xxd[li*dof+1] - M3*xxdd[li*dof+1];
      xxdd[dof*li+2] = M1*(xx[li*dof+2] - xxn[li*dof+2]) - M2*xxd[li*dof+2] - M3*xxdd[li*dof+2];

      xxd[dof*li  ] = xxd[li*dof  ] + C2*xxddn1 + C3*xxdd[li*dof  ];
      xxd[dof*li+1] = xxd[li*dof+1] + C2*xxddn2 + C3*xxdd[li*dof+1];
      xxd[dof*li+2] = xxd[li*dof+2] + C2*xxddn3 + C3*xxdd[li*dof+2];

    }
  }

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->xn, &xxn);

  //
  if (ti==0) {
    //if (ibm->ibi==0)  ConstantVel(12.649, 0, fem);
    //if (ibm->ibi==1)  ConstantVel(-25.298, 0, fem);
   // if (ibm->ibi==0)  ConstantVel(20., 0, fem);
   // if (ibm->ibi==1)  ConstantVel(-20., 0, fem);
  }
  //

  if (explicit) {VecCopy(fem->y, fem->yn);}
  
  return(0);
}

//-------------------------------------------------------------------------------------------------
PetscErrorCode Init(FE *fem, PetscInt ibi) {
  
  IBMNodes   *ibm = fem->ibm;
  PetscReal  *xx, mass;
  PetscInt   nv, ec, n1e, n2e, n3e;

  if (curvature==6) {GlobalGhostInit(ibm);}

  AreaNormal(ibm);
  for (ec=0; ec<ibm->n_elmt + 2*ibm->n_ghosts; ec++) {
    ibm->dA0[ec] = ibm->dA[ec]; 
    ibm->Nf_x[ec] = ibm->nf_x[ec];  ibm->Nf_y[ec] = ibm->nf_y[ec];  ibm->Nf_z[ec] = ibm->nf_z[ec]; 
  }

  ibm->rho = rho;
  ibm->h0 = h0;
  //
  /* if (ibm->ibi==0) { */
  /*   //rho = 1.;  h0 = 0.003; */
  /*   rho = 0.05; */
  /* } else { */
  /*   //rho = 10.;  h0 = 0.03; */
  /*   rho = 0.15; */
  /* } */
  //

  for (nv=0; nv<ibm->n_v; nv++)  ibm->m[nv] = 0.;
  for (ec=0; ec<ibm->n_elmt; ec++) {
    mass = 1./3.*rho*ibm->dA0[ec]*h0;
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
    ibm->m[n1e] += mass;  ibm->m[n2e] += mass;  ibm->m[n3e] += mass;
  }

  /* PatchLoc inside InitGhost reads ibm->x_bp to set p4x/p4x0 etc.
   * Seed x_bp from x_bp0 (reference) so the reference curvature kve0
   * is computed from the correct geometry, not from uninitialized memory. */
  for (nv = 0; nv < ibm->n_v + ibm->n_ghosts; nv++) {
    ibm->x_bp[nv] = ibm->x_bp0[nv];
    ibm->y_bp[nv] = ibm->y_bp0[nv];
    ibm->z_bp[nv] = ibm->z_bp0[nv];
  }
  if (bending){InitGhost(fem);}

  VecGetArray(fem->x, &xx);
  {
    DMPlexGeomCtx *gctx = &fem->geom_ctx;
    if (gctx->initialized) {
      /* Parallel Vec: local array indexed by owned-vertex sequential index oi. */
      PetscMPIInt imyrank;
      MPI_Comm_rank(PETSC_COMM_WORLD, &imyrank);
      PetscInt oi = 0;
      for (PetscInt lv = 0; lv < gctx->nLocalVerts; ++lv) {
        PetscInt v = gctx->origVert[lv];
        if (v < 0 || v >= ibm->n_v || gctx->ownerRank[v] != imyrank) continue;
        xx[oi*dof  ] = ibm->x_bp0[v];
        xx[oi*dof+1] = ibm->y_bp0[v];
        xx[oi*dof+2] = ibm->z_bp0[v];
        oi++;
      }
    } else {
      for (nv=0; nv<ibm->n_v+ibm->n_ghosts; nv++) {
        xx[nv*dof  ] = ibm->x_bp0[nv];
        xx[nv*dof+1] = ibm->y_bp0[nv];
        xx[nv*dof+2] = ibm->z_bp0[nv];
      }
    }
  }

 for (ec=0; ec<ibm->n_elmt; ec++) {
    ibm->g1[dof*ec] = ibm->G1[dof*ec]; ibm->g1[dof*ec+1] = ibm->G1[dof*ec+1]; ibm->g1[dof*ec+2] = ibm->G1[dof*ec+2];
    ibm->g2[dof*ec] = ibm->G2[dof*ec]; ibm->g2[dof*ec+1] = ibm->G2[dof*ec+1]; ibm->g2[dof*ec+2] = ibm->G2[dof*ec+2];
    ibm->g3[dof*ec] = ibm->Nf_x[ec]; ibm->g3[dof*ec+1] = ibm->Nf_y[ec]; ibm->g3[dof*ec+2] = ibm->Nf_z[ec];

    ibm->g1n[dof*ec] = ibm->g1[dof*ec]; ibm->g1n[dof*ec+1] = ibm->g1[dof*ec+1]; ibm->g1n[dof*ec+2] = ibm->g1[dof*ec+2];
    ibm->g2n[dof*ec] = ibm->g2[dof*ec]; ibm->g2n[dof*ec+1] = ibm->g2[dof*ec+1]; ibm->g2n[dof*ec+2] = ibm->g2[dof*ec+2];
    ibm->g3n[dof*ec] = ibm->g3[dof*ec]; ibm->g3n[dof*ec+1] = ibm->g3[dof*ec+1]; ibm->g3n[dof*ec+2] = ibm->g3[dof*ec+2];
  }

  VecRestoreArray(fem->x, &xx);
  VecCopy(fem->x, fem->xn);
  VecCopy(fem->x, fem->xnm1);
  // printf("CHECK b InitMaterial\n");
  if (ConstitutiveLawNonLinear) {
    InitMaterial(ibm);
  }

  if (ti==0) InitVel(0, 0, fem);
  if (ti==0) InitVel(1, 0, fem);
  if (ti==0) InitVel(2, 0, fem);
  if (ti==0) InitVel(3, 0, fem);
  //if (ti==0 && manufactured) MoveBoundary(1, fem);
  // printf("CHECK b Output\n");
  Output(fem, 0, ibi, out_dir);
  
  if(outghost){OutputGhost(fem, 0, ibi, out_dir);}
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode Free(FE *fem) {

  IBMNodes  *ibm=fem->ibm;
  //Vec
  VecDestroy(&(fem->Res));  VecDestroy(&(fem->x));  VecDestroy(&(fem->xn));  VecDestroy(&(fem->xnm1));
  VecDestroy(&(fem->xd));  VecDestroy(&(fem->xdd));  VecDestroy(&(fem->y));  VecDestroy(&(fem->yn)); 
  VecDestroy(&(fem->Fext));  VecDestroy(&(fem->Fint));  VecDestroy(&(fem->Fdyn)); VecDestroy(&(fem->Fcnt)); 
  VecDestroy(&(fem->disp));  VecDestroy(&(fem->FJ)); 
  VecDestroy(&(fem->Mass));  VecDestroy(&(fem->Dissip));  VecDestroy(&(fem->dx)); 


  if (ressmooth){
    VecDestroy(&(fem->Res_smth));
  }
  if (par_jac){
    MatDestroy(&(fem->Jacobian));    
    // MatDestroy(&(fem->J_Seq));    
  }
  
  //Malloc
  PetscFree(ibm->x_bp);  PetscFree(ibm->y_bp);  PetscFree(ibm->z_bp);
  PetscFree(ibm->x_bp0);  PetscFree(ibm->y_bp0);  PetscFree(ibm->z_bp0);
  PetscFree(ibm->nv1);  PetscFree(ibm->nv2);  PetscFree(ibm->nv3);
  PetscFree(ibm->nv4);  PetscFree(ibm->nv5);  PetscFree(ibm->nv6);
  PetscFree(ibm->n_bnodes);  PetscFree(ibm->bnodes);  PetscFree(ibm->n_fib);  PetscFree(ibm->kve0); 
  PetscFree(ibm->kve);  PetscFree(fem->StressM);  PetscFree(fem->StrainM);   PetscFree(fem->IE);  PetscFree(fem->CE);  PetscFree(fem->KE);  PetscFree(ibm->m); 
  PetscFree(fem->FC); 
  PetscFree(fem->StressB);  PetscFree(fem->StrainB);
  PetscFree(ibm->ire);  PetscFree(ibm->irv);  PetscFree(ibm->val);  PetscFree(ibm->patch);  PetscFree(ibm->contact);
  PetscFree(ibm->G);  PetscFree(ibm->G1);  PetscFree(ibm->G2); PetscFree(ibm->g1); PetscFree(ibm->g2); PetscFree(ibm->g3); PetscFree(ibm->g1n); PetscFree(ibm->g2n); PetscFree(ibm->g3n);

  PetscFree(ibm->nf_x); PetscFree(ibm->Nf_x); PetscFree(ibm->nf_y); PetscFree(ibm->Nf_y); PetscFree(ibm->nf_z); PetscFree(ibm->Nf_z); 

  PetscFree(ibm->qvec); PetscFree(ibm->radvec); 
  PetscFree(ibm->dA0); PetscFree(ibm->dA); 

  for (int i=0; i<2; i++){
    PetscFree(ibm->El[i]);
    PetscFree(ibm->E_epsilon[i]); 
  }
  PetscFree(ibm->El);    
  PetscFree(ibm->E_epsilon); 

  if (Adam){
    PetscInt n_coeffs;
  if (ConstitutiveLawNonLinear){
      n_coeffs = n_Fung_Coeffs;
    }
    else{
      n_coeffs = n_lin_model_coeffs;
    }
    
    for (int i=0; i<n_coeffs; i++){
      PetscFree(ibm->Adam_mestimate[i]);
    }
    PetscFree(ibm->Adam_mestimate);

    for (int i=0; i<n_coeffs; i++){
      PetscFree(ibm->Adam_vestimate[i]);
    }
    PetscFree(ibm->Adam_vestimate);
  }

  if (ConstitutiveLawNonLinear){
    

    for (int i=0; i<n_Fung_Coeffs; i++){
      PetscFree(ibm->Fung_epsilons[i]);
    }
    PetscFree(ibm->Fung_epsilons);

    for (int i=0; i<n_Fung_Coeffs; i++){
      PetscFree(ibm->Fung_coeffs[i]);
    }
    PetscFree(ibm->Fung_coeffs);

    for (int i=0; i<n_Fung_Coeffs; i++){
      PetscFree(ibm->Fung_coeffs_smth[i]);
    }
    PetscFree(ibm->Fung_coeffs_smth);

    for (int i = 0; i < ibm->n_v; i++) {
      for (int j = 0; j < ibm->n_elmt; j++) {
        for (int k = 0; k < dof; k++) {
            PetscFree(fem->Jac_Fung[i][j][k]); 
        }
        PetscFree(fem->Jac_Fung[i][j]);
      }
      PetscFree(fem->Jac_Fung[i]);
    }
    PetscFree(fem->Jac_Fung);
  }
  if (inverse){
    if (ressmooth){
  for (int i=0; i<ibm->n_v; i++){
        PetscFree(ibm->neigh_nodes_ind[i]);
      }
      PetscFree(ibm->neigh_nodes_ind);
    }
  }

  // for (int i=0; i<ibm->n_v; i++){
  //   PetscFree(fem->dR_dE[i]);
  // }
  // PetscFree(fem->dR_dE);

  if (inverse) {
    for (int i = 0; i < ibm->n_v; i++) {
      for (int j = 0; j < ibm->n_elmt; j++) {
        for (int k=0; k<dof; k++){
          PetscFree(fem->dR_dE[i][j][k]);
        }        
        PetscFree(fem->dR_dE[i][j]);  
      }
      PetscFree(fem->dR_dE[i]);
    }
    PetscFree(fem->dR_dE);
  }

  if(bending){
    PetscFree(ibm->belmtsedge);  PetscFree(ibm->belmts);  PetscFree(ibm->edgefrontnodes);
    PetscFree(ibm->edgefrontnodesI); 
    PetscFree(ibm->p4x);  PetscFree(ibm->p4y);  PetscFree(ibm->p4z);
    PetscFree(ibm->p5x);  PetscFree(ibm->p5y);  PetscFree(ibm->p5z);
    PetscFree(ibm->p6x);  PetscFree(ibm->p6y);  PetscFree(ibm->p6z);
    
    PetscFree(ibm->p4x0);  PetscFree(ibm->p4y0);  PetscFree(ibm->p4z0);
    PetscFree(ibm->p5x0);  PetscFree(ibm->p5y0);  PetscFree(ibm->p5z0);
    PetscFree(ibm->p6x0);  PetscFree(ibm->p6y0);  PetscFree(ibm->p6z0);
  }

  if(muscle_activation || dmplex_geom_process){
    ActDataDestroy(fem); // Just to check no memory leak in allocation
    // PetscPrintf(PETSC_COMM_SELF, "After ActDataDestroy\n");
  }
  
  return(0);
}

//-------------------------------------------------------------------------------------
PetscErrorCode ContactZ(FE *fem) {

  IBMNodes  *ibm=fem->ibm;
  PetscInt  nv;

  for (nv=0; nv<ibm->n_v; nv++) {
    ibm->contact[nv] = 0;
  }

  return(0);
}

//-----------------------------------------
PetscErrorCode Contact(FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscInt   nv, ec, n1e, n2e, n3e;
  PetscReal  *xx, *yy, *fint, *fext, *xxn, *xxd, *xxdd;//, *omega, *gd1, *gd2, *gd3, *m, *KEr, KEomega, omg[2], omgc[2];
  PetscReal  IEsum, KEsum, CEsum;
  /* PetscReal      M1, M2, M3; */
  /* PetscReal      Gama=0.5, Beta=0.25; */
  /* M1=1./(Beta*pow(dt,2));  M2=1./(Beta*dt);  M3=(1./(2*Beta))-1.; */

  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &omega); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd1); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd2); */
  /* PetscMalloc(dof*ibm->n_elmt*sizeof(PetscReal), &gd3); */
  /* PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &m); */
  /* PetscMalloc(ibm->n_v*sizeof(PetscReal), &KEr); */

  /* VecSet(fem->Fcnt, 0.0);   */
  VecSet(fem->Fint, 0.0);  //VecSet(fem->Fext, 0.0);  //VecSet(fem->Fdyn, 0.0); */
  for (nv=0; nv<ibm->n_v; nv++)  {
    fem->IE[nv] = 0.;
    // KEr[nv] = 0.;
  }
  for (ec=0; ec<ibm->n_elmt; ec++)  fem->FC[ec] = 0.;

  FInternal(fem);
  /* //if(tisteps>1) {FDynamic(fem);} */
  /* FExternal(fem); */
  /* VecWAXPY(fem->Fcnt, -1., fem->Fext, fem->Fint); */
  /* //VecAXPY(fem->Fcnt, -1., fem->Fext); */
  /* //VecAXPY(fem->Fcnt, 1., fem->Fint); */
  /* VecAXPY(fem->Fcnt, 1., fem->Fdyn); */
  
  VecGetArray(fem->x, &xx);
  VecGetArray(fem->xd, &xxd);
  VecGetArray(fem->xn, &xxn);
  VecGetArray(fem->xdd, &xxdd);
  VecGetArray(fem->y, &yy);
  VecGetArray(fem->Fint, &fint);
  VecGetArray(fem->Fext, &fext);

  /* for (ec=0; ec<ibm->n_elmt; ec++) { */
  /*   n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec]; */

  /*   gd1[dof*ec] = (ibm->g1[dof*ec] - ibm->g1n[dof*ec])/dt;  gd1[dof*ec+1] = (ibm->g1[dof*ec+1] - ibm->g1n[dof*ec+1])/dt;  gd1[dof*ec+2] = (ibm->g1[dof*ec+2] - ibm->g1n[dof*ec+2])/dt; */
  /*   gd2[dof*ec] = (ibm->g2[dof*ec] - ibm->g2n[dof*ec])/dt;  gd2[dof*ec+1] = (ibm->g2[dof*ec+1] - ibm->g2n[dof*ec+1])/dt;  gd2[dof*ec+2] = (ibm->g2[dof*ec+2] - ibm->g2n[dof*ec+2])/dt; */
  /*   gd3[dof*ec] = (ibm->g3[dof*ec] - ibm->g3n[dof*ec])/dt;  gd3[dof*ec+1] = (ibm->g3[dof*ec+1] - ibm->g3n[dof*ec+1])/dt;  gd3[dof*ec+2] = (ibm->g3[dof*ec+2] - ibm->g3n[dof*ec+2])/dt; */

  /*   omg[0] = -1*(ibm->g3[dof*ec]*gd1[dof*ec] + ibm->g3[dof*ec+1]*gd1[dof*ec+1] + ibm->g3[dof*ec+2]*gd1[dof*ec+2]); */
  /*   omg[1] = -1*(ibm->g3[dof*ec]*gd2[dof*ec] + ibm->g3[dof*ec+1]*gd2[dof*ec+1] + ibm->g3[dof*ec+2]*gd2[dof*ec+2]); */

  /*   omega[dof*ec] = omg[0]*ibm->g2[dof*ec] - omg[1]*ibm->g1[ec*dof]; */
  /*   omega[dof*ec+1] = omg[0]*ibm->g2[dof*ec+1] - omg[1]*ibm->g1[ec*dof+1]; */
  /*   omega[dof*ec+2] = omg[0]*ibm->g2[dof*ec+2] - omg[1]*ibm->g1[ec*dof+2]; */

  /*   omgc[0] = omega[dof*ec]*ibm->g2[dof*ec] + omega[dof*ec+1]*ibm->g2[dof*ec+1] + omega[dof*ec+2]*ibm->g2[dof*ec+2]; */
  /*   omgc[1] = -1*(omega[dof*ec]*ibm->g1[dof*ec] + omega[dof*ec+1]*ibm->g1[dof*ec+1] + omega[dof*ec+2]*ibm->g1[dof*ec+2]); */
  /*   omega[dof*ec+2] = -1*(ibm->g3[dof*ec]*gd3[dof*ec] + ibm->g3[dof*ec+1]*gd3[dof*ec+1] + ibm->g3[dof*ec+2]*gd3[dof*ec+2]); */

  /*   m[ec] = rho*ibm->dA0[ec]*h0; */

  /*   KEomega = h0*h0/24.*m[ec]*(omega[dof*ec]*omega[dof*ec] + omega[dof*ec+1]*omega[dof*ec+1] + omega[dof*ec+2]*omega[dof*ec+2]); */
  /*   KEomega = h0*h0/24.*m[ec]*(omg[0]*omgc[0] + omg[1]*omgc[1]); */

  /*   KEr[n1e] += KEomega/3.;  KEr[n2e] += KEomega/3.;  KEr[n3e] += KEomega/3.; */
  /* } */

  KEsum=0.; IEsum=0.; CEsum=0.;
  for (nv=0; nv<ibm->n_v; nv++) {
    if (ti==0) fem->CE[nv] = 0.;
    // set acceleration such that Mxdd=Fint
    if (ibm->contact[nv]>0 && ibm->ibi==1) {
    /*   xxdd[dof*nv  ] = M2*xxd[nv*dof  ]/8.;//fint[dof*nv  ]/ibm->m[nv]++M1*(xx[nv*dof  ]-xxn[nv*dof  ]); */
    /*   xxdd[dof*nv+1] = M2*xxd[nv*dof+1]/8.;//fint[dof*nv+1]/ibm->m[nv]++M1*(xx[nv*dof+1]-xxn[nv*dof+1]); */
    /*   xxdd[dof*nv+2] = M2*xxd[nv*dof+2]/8.;//fint[dof*nv+2]/ibm->m[nv]+//+M1*(xx[nv*dof+2]-xxn[nv*dof+2]); */

      PetscPrintf(PETSC_COMM_WORLD, "nv %d contact %d f_int %f %f %f acc %f %f %f\n",nv, ibm->contact[nv],fint[dof*nv  ], fint[dof*nv+1], fint[dof*nv+2],xxdd[dof*nv], xxdd[dof*nv+1], xxdd[dof*nv+2]);
    }

    //    fem->CE[nv] = (fint[nv*dof] - fext[nv*dof])*(xx[nv*dof] - yy[nv*dof]) + (fint[nv*dof+1] - fext[nv*dof+1])*(xx[nv*dof+1] - yy[nv*dof+1]) + (fint[nv*dof+2] - fext[nv*dof+2])*(xx[nv*dof+2] - yy[nv*dof+2]);
    //em->CE[nv] = (fint[nv*dof] - fext[nv*dof])*(xx[nv*dof] - xxn[nv*dof]) + (fint[nv*dof+1] - fext[nv*dof+1])*(xx[nv*dof+1] - xxn[nv*dof+1]) + (fint[nv*dof+2] - fext[nv*dof+2])*(xx[nv*dof+2] - xxn[nv*dof+2]);
    fem->KE[nv] = 0.5*ibm->m[nv]*(xxd[dof*nv]*xxd[dof*nv] + xxd[dof*nv+1]*xxd[dof*nv+1] +xxd[dof*nv+2]*xxd[dof*nv+2]);// + KEr[nv];
    KEsum +=fem->KE[nv];
    IEsum +=fem->IE[nv];
    //    CEsum +=    KEr[nv];
    //fem->KE[nv] = KEr[nv];
    //fem->IE[nv] = fint[nv*dof]*(xx[nv*dof] - ibm->x_bp0[nv]) + fint[nv*dof+1]*(xx[nv*dof+1] - ibm->y_bp0[nv]) + fint[nv*dof+2]*(xx[nv*dof+2] - ibm->z_bp0[nv]);
  }

  /* for (ec=0; ec<ibm->n_elmt; ec++) { */
  /*   ibm->g1n[dof*ec] = ibm->g1[dof*ec]; ibm->g1n[dof*ec+1] = ibm->g1[dof*ec+1]; ibm->g1n[dof*ec+2] = ibm->g1[dof*ec+2]; */
  /*   ibm->g2n[dof*ec] = ibm->g2[dof*ec]; ibm->g2n[dof*ec+1] = ibm->g2[dof*ec+1]; ibm->g2n[dof*ec+2] = ibm->g2[dof*ec+2]; */
  /*   ibm->g3n[dof*ec] = ibm->g3[dof*ec]; ibm->g3n[dof*ec+1] = ibm->g3[dof*ec+1]; ibm->g3n[dof*ec+2] = ibm->g3[dof*ec+2]; */
  /* } */

  VecRestoreArray(fem->x, &xx);
  VecRestoreArray(fem->xd, &xxd);
  VecRestoreArray(fem->xn, &xxn);
  VecRestoreArray(fem->xdd, &xxdd);
  VecRestoreArray(fem->y, &yy);
  VecRestoreArray(fem->Fint, &fint);
  VecRestoreArray(fem->Fext, &fext);

  FILE  *f;
  char  filen[80];
  PetscReal TEsum;
  TEsum = KEsum + IEsum;

  sprintf(filen, "energies%2.2d.dat", ibm->ibi);
  f = fopen(filen, "a"); // open file

  PetscFPrintf(PETSC_COMM_SELF, f, "%d %f %f %f %f\n", ti, IEsum, KEsum, TEsum, CEsum);
  
  fclose(f);
  //PetscFree(omega); PetscFree(gd1); PetscFree(gd2); PetscFree(gd3); PetscFree(m); PetscFree(KEr);

  return (0);
}
