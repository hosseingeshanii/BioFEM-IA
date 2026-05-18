#include <stddef.h>

#include "cuda_processes.h"

#include "active_strain.h"
#include "cuda_bridge.h"

extern PetscInt nbody;
extern PetscInt muscle_activation;

typedef struct {
  ElemVec g;
  ElemVec g0;
  Elem2DTens gm;
  Elem2DTens gm0;
} MetricSnapshotEntry;

static PetscReal MaxCmpntsDiff_(Cmpnts a, Cmpnts b)
{
  PetscReal max_diff = PetscAbsReal(a.x - b.x);
  max_diff = PetscMax(max_diff, PetscAbsReal(a.y - b.y));
  max_diff = PetscMax(max_diff, PetscAbsReal(a.z - b.z));
  return max_diff;
}

static PetscReal MaxMat3Diff_(const PetscReal A[3][3], const PetscReal B[3][3])
{
  PetscReal max_diff = 0.0;
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      max_diff = PetscMax(max_diff, PetscAbsReal(A[i][j] - B[i][j]));
    }
  }
  return max_diff;
}

static PetscReal TimingSpeedup_(double cpu_time, double cuda_time)
{
  if (cuda_time <= 0.0) return 0.0;
  return (PetscReal)(cpu_time / cuda_time);
}

static void PrintMat3WithLabel_(const char *label, const PetscReal A[3][3])
{
  PetscPrintf(PETSC_COMM_SELF,
              "    %s = [[% .6e % .6e % .6e] [% .6e % .6e % .6e] [% .6e % .6e % .6e]]\n",
              label,
              (double)A[0][0], (double)A[0][1], (double)A[0][2],
              (double)A[1][0], (double)A[1][1], (double)A[1][2],
              (double)A[2][0], (double)A[2][1], (double)A[2][2]);
}

static PetscErrorCode SnapshotMetricState_(FE *fem,
                                           PetscInt n_compare_elems,
                                           MetricSnapshotEntry **snapshot_out)
{
  PetscFunctionBeginUser;

  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(snapshot_out != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "snapshot_out must not be NULL");

  const PetscInt n_qp = fem->act_data.n_qp;
  const size_t n_entries = (size_t)n_compare_elems * (size_t)n_qp;
  MetricSnapshotEntry *snapshot = NULL;

  PetscCall(PetscMalloc1(n_entries, &snapshot));

  for (PetscInt ec = 0; ec < n_compare_elems; ++ec) {
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    for (PetscInt qp = 0; qp < n_qp; ++qp) {
      const size_t idx = (size_t)ec * (size_t)n_qp + (size_t)qp;
      snapshot[idx].g = ead->g[qp];
      snapshot[idx].g0 = ead->g0[qp];
      snapshot[idx].gm = ead->gm[qp];
      snapshot[idx].gm0 = ead->gm0[qp];
    }
  }

  *snapshot_out = snapshot;
  PetscFunctionReturn(0);
}

static PetscErrorCode CompareMetricState_(FE *fem,
                                          PetscInt body_id,
                                          PetscInt n_compare_elems,
                                          const MetricSnapshotEntry *cpu_snapshot)
{
  PetscFunctionBeginUser;

  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(cpu_snapshot != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "cpu_snapshot must not be NULL");

  const PetscInt n_qp = fem->act_data.n_qp;
  const PetscReal tol = 1.0e-9;

  PetscPrintf(PETSC_COMM_SELF,
              "\nCPU vs CUDA metric comparison for body %d (first %" PetscInt_FMT " elements)\n",
              (int)body_id, n_compare_elems);

  for (PetscInt ec = 0; ec < n_compare_elems; ++ec) {
    PetscReal max_g = 0.0, max_g0 = 0.0, max_gm = 0.0, max_gm0 = 0.0;
    for (PetscInt qp = 0; qp < n_qp; ++qp) {
      const size_t idx = (size_t)ec * (size_t)n_qp + (size_t)qp;
      ElemActData *ead = &fem->act_data.elem_act_data[ec];

      for (PetscInt i = 0; i < 3; ++i) {
        max_g = PetscMax(max_g, MaxCmpntsDiff_(cpu_snapshot[idx].g.Cov[i], ead->g[qp].Cov[i]));
        max_g = PetscMax(max_g, MaxCmpntsDiff_(cpu_snapshot[idx].g.Cont[i], ead->g[qp].Cont[i]));
        max_g0 = PetscMax(max_g0, MaxCmpntsDiff_(cpu_snapshot[idx].g0.Cov[i], ead->g0[qp].Cov[i]));
        max_g0 = PetscMax(max_g0, MaxCmpntsDiff_(cpu_snapshot[idx].g0.Cont[i], ead->g0[qp].Cont[i]));
      }

      max_gm = PetscMax(max_gm, MaxMat3Diff_(cpu_snapshot[idx].gm.Cov, ead->gm[qp].Cov));
      max_gm = PetscMax(max_gm, MaxMat3Diff_(cpu_snapshot[idx].gm.Cont, ead->gm[qp].Cont));
      max_gm0 = PetscMax(max_gm0, MaxMat3Diff_(cpu_snapshot[idx].gm0.Cov, ead->gm0[qp].Cov));
      max_gm0 = PetscMax(max_gm0, MaxMat3Diff_(cpu_snapshot[idx].gm0.Cont, ead->gm0[qp].Cont));
    }

    PetscPrintf(PETSC_COMM_SELF,
                "  elem %" PetscInt_FMT ": max|g|=% .3e max|g0|=% .3e max|gm|=% .3e max|gm0|=% .3e -> %s\n",
                ec, (double)max_g, (double)max_g0, (double)max_gm, (double)max_gm0,
                (max_g <= tol && max_g0 <= tol && max_gm <= tol && max_gm0 <= tol) ? "MATCH" : "DIFF");
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PrintMetricSampleValues_(FE *fem,
                                               PetscInt body_id,
                                               PetscInt n_compare_elems,
                                               const MetricSnapshotEntry *cpu_snapshot)
{
  PetscFunctionBeginUser;

  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(cpu_snapshot != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "cpu_snapshot must not be NULL");

  const PetscInt n_qp = fem->act_data.n_qp;
  const PetscInt n_sample_elems = PetscMin((PetscInt)2, n_compare_elems);

  PetscPrintf(PETSC_COMM_SELF,
              "\nCPU vs CUDA sample metric values for body %d\n",
              (int)body_id);

  for (PetscInt ec = 0; ec < n_sample_elems; ++ec) {
    const PetscInt qp = 0;
    const size_t idx = (size_t)ec * (size_t)n_qp + (size_t)qp;
    ElemActData *ead = &fem->act_data.elem_act_data[ec];

    PetscPrintf(PETSC_COMM_SELF,
                "  elem %" PetscInt_FMT ", qp %" PetscInt_FMT "\n",
                ec, qp);
    PrintMat3WithLabel_("CPU gm.Cov", cpu_snapshot[idx].gm.Cov);
    PrintMat3WithLabel_("CUDA gm.Cov", ead->gm[qp].Cov);
    PrintMat3WithLabel_("CPU gm0.Cov", cpu_snapshot[idx].gm0.Cov);
    PrintMat3WithLabel_("CUDA gm0.Cov", ead->gm0[qp].Cov);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PrintCudaPreprocessTiming_(PetscInt body_id,
                                                 double cpu_geom0_time,
                                                 double cpu_geom_time,
                                                 double cpu_metric_time,
                                                 double cuda_coord_time,
                                                 double cuda_geom0_time,
                                                 double cuda_geom_time,
                                                 double cuda_metric_time)
{
  PetscFunctionBeginUser;

  const double cpu_geom_total = cpu_geom0_time + cpu_geom_time;
  const double cpu_total = cpu_geom_total + cpu_metric_time;
  const double cuda_geom_total = cuda_geom0_time + cuda_geom_time;
  const double cuda_total = cuda_coord_time + cuda_geom_total + cuda_metric_time;

  PetscPrintf(PETSC_COMM_SELF,
              "\nCUDA preprocess timing for body %d (host end-to-end)\n"
              "  task                         CPU (s)      CUDA (s)     CPU/CUDA\n"
              "  geom0/reference          % .6e  % .6e  % .3f x\n"
              "  geom/current            % .6e  % .6e  % .3f x\n"
              "  geom total              % .6e  % .6e  % .3f x\n"
              "  metric tensor           % .6e  % .6e  % .3f x\n"
              "  CUDA coord copy         % .6e  % .6e  % .3f x\n"
              "  preprocess total        % .6e  % .6e  % .3f x\n",
              (int)body_id,
              cpu_geom0_time, cuda_geom0_time, (double)TimingSpeedup_(cpu_geom0_time, cuda_geom0_time),
              cpu_geom_time, cuda_geom_time, (double)TimingSpeedup_(cpu_geom_time, cuda_geom_time),
              cpu_geom_total, cuda_geom_total, (double)TimingSpeedup_(cpu_geom_total, cuda_geom_total),
              cpu_metric_time, cuda_metric_time, (double)TimingSpeedup_(cpu_metric_time, cuda_metric_time),
              0.0, cuda_coord_time, 0.0,
              cpu_total, cuda_total, (double)TimingSpeedup_(cpu_total, cuda_total));

  PetscFunctionReturn(0);
}

PetscErrorCode RunCudaProcesses(FE *fem)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(muscle_activation, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "-cuda_process requires -muscle_activation 1 so metric tensor data is allocated");

  ierr = InitCudaWorkspace(); CHKERRQ(ierr);

  for (PetscInt ibi = 0; ibi < nbody; ibi++) {
    const PetscInt n_compare_elems = PetscMin((PetscInt)3, fem[ibi].ibm->n_elmt);
    MetricSnapshotEntry *cpu_snapshot = NULL;
    double t0, t1;
    double cpu_geom0_time, cpu_geom_time, cpu_metric_time;
    double cuda_geom0_time, cuda_geom_time, cuda_metric_time;

    t0 = MPI_Wtime();
    ierr = UpdateElements(&fem[ibi], ElemUpdateGeom0Subdiv); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cpu_geom0_time = t1 - t0;

    t0 = MPI_Wtime();
    ierr = UpdateElements(&fem[ibi], ElemUpdateGeomSubdiv); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cpu_geom_time = t1 - t0;

    t0 = MPI_Wtime();
    ierr = UpdateElements(&fem[ibi], ElemUpdateG); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cpu_metric_time = t1 - t0;

    ierr = SnapshotMetricState_(&fem[ibi], n_compare_elems, &cpu_snapshot); CHKERRQ(ierr);

    ierr = PrepareCudaHostWorkspace(&fem[ibi]); CHKERRQ(ierr);

    // Run multiple iterations to measure allocation reuse timing
    const PetscInt n_subdiv_iterations = 5;
    double cuda_geom0_times[5] = {0};
    double cuda_geom_times[5] = {0};

    for (PetscInt iter = 0; iter < n_subdiv_iterations; ++iter) {
      if (iter == 0) {
        t0 = MPI_Wtime();
        ierr = RunCudaSubdivGeomKernel(&fem[ibi], PETSC_TRUE); CHKERRQ(ierr);
        t1 = MPI_Wtime();
        cuda_geom0_times[iter] = t1 - t0;
      }

      t0 = MPI_Wtime();
      ierr = RunCudaSubdivGeomKernel(&fem[ibi], PETSC_FALSE); CHKERRQ(ierr);
      t1 = MPI_Wtime();
      cuda_geom_times[iter] = t1 - t0;
    }

    cuda_geom0_time = cuda_geom0_times[0];

    double cuda_geom_reuse_total = 0.0;
    PetscInt n_cuda_geom_reuse = 0;
    for (PetscInt iter = 1; iter < n_subdiv_iterations; ++iter) {
      cuda_geom_reuse_total += cuda_geom_times[iter];
      ++n_cuda_geom_reuse;
    }
    cuda_geom_time = (n_cuda_geom_reuse > 0) ? cuda_geom_reuse_total / (double)n_cuda_geom_reuse : cuda_geom_times[0];

    // Print timing comparison for each iteration
    for (PetscInt iter = 0; iter < n_subdiv_iterations; ++iter) {
      if (iter == 0) {
        PetscPrintf(PETSC_COMM_SELF,
                    "[iter %" PetscInt_FMT "] SubdivGeomKernel (reference): CUDA=%.6e  CPU=%.6e  speedup=%.3f x (includes allocation)\n",
                    iter, cuda_geom0_times[iter], cpu_geom0_time,
                    (double)TimingSpeedup_(cpu_geom0_time, cuda_geom0_times[iter]));
      }

      PetscPrintf(PETSC_COMM_SELF,
                  "[iter %" PetscInt_FMT "] SubdivGeomKernel (current): CUDA=%.6e  CPU=%.6e  speedup=%.3f x%s\n",
                  iter, cuda_geom_times[iter], cpu_geom_time,
                  (double)TimingSpeedup_(cpu_geom_time, cuda_geom_times[iter]),
                  (iter == 0) ? " (same outer iter as reference)" : " (reuse)");
    }

    t0 = MPI_Wtime();
    ierr = RunCudaMetricTensorKernel(&fem[ibi]); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cuda_metric_time = t1 - t0;
    PetscPrintf(PETSC_COMM_SELF,
                "CUDA metric-tensor kernel finished successfully for body %d.\n",
                ibi);

    ierr = PrintCudaPreprocessTiming_(ibi,
                                      cpu_geom0_time,
                                      cpu_geom_time,
                                      cpu_metric_time,
                                      0.0,  // cuda_coord_time (removed - was redundant with SubdivGeomKernel)
                                      cuda_geom0_time,
                                      cuda_geom_time,
                                      cuda_metric_time); CHKERRQ(ierr);
    ierr = CompareMetricState_(&fem[ibi], ibi, n_compare_elems, cpu_snapshot); CHKERRQ(ierr);
    ierr = PrintMetricSampleValues_(&fem[ibi], ibi, n_compare_elems, cpu_snapshot); CHKERRQ(ierr);
    ierr = PetscFree(cpu_snapshot); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
