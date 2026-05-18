#include <stddef.h>

#include "kokkos_processes.h"

#include "active_strain.h"
#include "kokkos_bridge.h"

extern PetscInt nbody;
extern PetscInt muscle_activation;

typedef struct {
  SubdivGeomQP geom;
} GeomSnapshotEntry;

static PetscReal MaxCmpntsDiff_(Cmpnts a, Cmpnts b)
{
  PetscReal max_diff = PetscAbsReal(a.x - b.x);
  max_diff = PetscMax(max_diff, PetscAbsReal(a.y - b.y));
  max_diff = PetscMax(max_diff, PetscAbsReal(a.z - b.z));
  return max_diff;
}

static PetscReal TimingSpeedup_(double cpu_time, double kokkos_time)
{
  if (kokkos_time <= 0.0) return 0.0;
  return (PetscReal)(cpu_time / kokkos_time);
}

static PetscReal IntDiff_(PetscInt a, PetscInt b)
{
  return (PetscReal)((a > b) ? (a - b) : (b - a));
}

static PetscErrorCode CopyIrregularArray_(PetscReal **dst, const PetscReal *src, PetscInt n)
{
  PetscFunctionBeginUser;
  *dst = NULL;
  if (src && n > 0) {
    PetscCall(PetscMalloc1(n, dst));
    PetscCall(PetscMemcpy(*dst, src, (size_t)n * sizeof(PetscReal)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SnapshotGeom_(const SubdivGeomQP *src, GeomSnapshotEntry *dst)
{
  PetscFunctionBeginUser;
  PetscCheck(src != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "src must not be NULL");
  PetscCheck(dst != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "dst must not be NULL");

  PetscCall(PetscMemzero(&dst->geom, sizeof(dst->geom)));
  dst->geom = *src;
  dst->geom.INa0 = NULL;
  dst->geom.INa1 = NULL;
  dst->geom.INab0 = NULL;
  dst->geom.INab1 = NULL;
  dst->geom.INab2 = NULL;

  if (src->is_irregular) {
    PetscCall(CopyIrregularArray_(&dst->geom.INa0, src->INa0, src->nen));
    PetscCall(CopyIrregularArray_(&dst->geom.INa1, src->INa1, src->nen));
    PetscCall(CopyIrregularArray_(&dst->geom.INab0, src->INab0, src->nen));
    PetscCall(CopyIrregularArray_(&dst->geom.INab1, src->INab1, src->nen));
    PetscCall(CopyIrregularArray_(&dst->geom.INab2, src->INab2, src->nen));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyGeomSnapshot_(GeomSnapshotEntry *entry)
{
  PetscFunctionBeginUser;
  if (!entry) PetscFunctionReturn(0);
  PetscCall(PetscFree(entry->geom.INa0));
  PetscCall(PetscFree(entry->geom.INa1));
  PetscCall(PetscFree(entry->geom.INab0));
  PetscCall(PetscFree(entry->geom.INab1));
  PetscCall(PetscFree(entry->geom.INab2));
  PetscFunctionReturn(0);
}

static PetscErrorCode SnapshotGeomState_(FE *fem,
                                         PetscBool use_reference_coords,
                                         PetscInt n_compare_elems,
                                         GeomSnapshotEntry **snapshot_out)
{
  PetscFunctionBeginUser;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(snapshot_out != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "snapshot_out must not be NULL");

  GeomSnapshotEntry *snapshot = NULL;
  PetscCall(PetscMalloc1(n_compare_elems, &snapshot));

  for (PetscInt ec = 0; ec < n_compare_elems; ++ec) {
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    SubdivGeomQP *geom = use_reference_coords ? &ead->geom0[0] : &ead->geom[0];
    PetscCall(SnapshotGeom_(geom, &snapshot[ec]));
  }

  *snapshot_out = snapshot;
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyGeomStateSnapshot_(GeomSnapshotEntry *snapshot,
                                                PetscInt n_compare_elems)
{
  PetscFunctionBeginUser;
  if (!snapshot) PetscFunctionReturn(0);
  for (PetscInt ec = 0; ec < n_compare_elems; ++ec) {
    PetscCall(DestroyGeomSnapshot_(&snapshot[ec]));
  }
  PetscCall(PetscFree(snapshot));
  PetscFunctionReturn(0);
}

static PetscReal MaxIrregularArrayDiff_(const PetscReal *cpu,
                                        const PetscReal *kokkos,
                                        PetscInt n)
{
  PetscReal max_diff = 0.0;
  if (!cpu || !kokkos || n <= 0) return max_diff;
  for (PetscInt i = 0; i < n; ++i) {
    max_diff = PetscMax(max_diff, PetscAbsReal(cpu[i] - kokkos[i]));
  }
  return max_diff;
}

static PetscReal MaxGeomDiff_(const SubdivGeomQP *cpu, const SubdivGeomQP *kokkos)
{
  PetscReal max_diff = 0.0;
  max_diff = PetscMax(max_diff, IntDiff_(cpu->is_irregular, kokkos->is_irregular));
  max_diff = PetscMax(max_diff, IntDiff_(cpu->v, kokkos->v));
  max_diff = PetscMax(max_diff, IntDiff_(cpu->nen, kokkos->nen));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->ndx21, kokkos->ndx21));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->ndx31, kokkos->ndx31));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->nn, kokkos->nn));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->gc1, kokkos->gc1));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->gc2, kokkos->gc2));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->Aaa, kokkos->Aaa));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->Abb, kokkos->Abb));
  max_diff = PetscMax(max_diff, MaxCmpntsDiff_(cpu->Aab, kokkos->Aab));

  if (cpu->is_irregular && kokkos->is_irregular && cpu->nen == kokkos->nen) {
    max_diff = PetscMax(max_diff, MaxIrregularArrayDiff_(cpu->INa0, kokkos->INa0, cpu->nen));
    max_diff = PetscMax(max_diff, MaxIrregularArrayDiff_(cpu->INa1, kokkos->INa1, cpu->nen));
    max_diff = PetscMax(max_diff, MaxIrregularArrayDiff_(cpu->INab0, kokkos->INab0, cpu->nen));
    max_diff = PetscMax(max_diff, MaxIrregularArrayDiff_(cpu->INab1, kokkos->INab1, cpu->nen));
    max_diff = PetscMax(max_diff, MaxIrregularArrayDiff_(cpu->INab2, kokkos->INab2, cpu->nen));
  }

  return max_diff;
}

static PetscErrorCode CompareGeomState_(FE *fem,
                                        PetscInt body_id,
                                        const char *label,
                                        PetscBool use_reference_coords,
                                        PetscInt n_compare_elems,
                                        const GeomSnapshotEntry *cpu_snapshot)
{
  PetscFunctionBeginUser;
  const PetscReal tol = 1.0e-9;

  PetscPrintf(PETSC_COMM_SELF,
              "\nCPU vs Kokkos %s geometry comparison for body %d (first %" PetscInt_FMT " elements)\n",
              label, (int)body_id, n_compare_elems);

  for (PetscInt ec = 0; ec < n_compare_elems; ++ec) {
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    const SubdivGeomQP *kokkos_geom = use_reference_coords ? &ead->geom0[0] : &ead->geom[0];
    const PetscReal max_geom = MaxGeomDiff_(&cpu_snapshot[ec].geom, kokkos_geom);

    PetscPrintf(PETSC_COMM_SELF,
                "  elem %" PetscInt_FMT ": max|geom|=% .3e -> %s\n",
                ec, (double)max_geom, (max_geom <= tol) ? "MATCH" : "DIFF");
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PrintKokkosGeomTiming_(PetscInt body_id,
                                             double cpu_geom0_time,
                                             double cpu_geom_time,
                                             double kokkos_geom0_time,
                                             double kokkos_geom_time)
{
  PetscFunctionBeginUser;

  const double cpu_total = cpu_geom0_time + cpu_geom_time;
  const double kokkos_total = kokkos_geom0_time + kokkos_geom_time;

  PetscPrintf(PETSC_COMM_SELF,
              "\nKokkos geometry timing for body %d (host end-to-end)\n"
              "  task                         CPU (s)    Kokkos (s)   CPU/Kokkos\n"
              "  geom0/reference          % .6e  % .6e  % .3f x\n"
              "  geom/current            % .6e  % .6e  % .3f x\n"
              "  geom total              % .6e  % .6e  % .3f x\n",
              (int)body_id,
              cpu_geom0_time, kokkos_geom0_time, (double)TimingSpeedup_(cpu_geom0_time, kokkos_geom0_time),
              cpu_geom_time, kokkos_geom_time, (double)TimingSpeedup_(cpu_geom_time, kokkos_geom_time),
              cpu_total, kokkos_total, (double)TimingSpeedup_(cpu_total, kokkos_total));

  PetscFunctionReturn(0);
}

PetscErrorCode RunKokkosProcesses(FE *fem)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(muscle_activation, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "-kokkos_process requires -muscle_activation 1 so subdivision geometry data is allocated");

  ierr = InitKokkosWorkspace(); CHKERRQ(ierr);

  for (PetscInt ibi = 0; ibi < nbody; ibi++) {
    const PetscInt n_compare_elems = PetscMin((PetscInt)3, fem[ibi].ibm->n_elmt);
    GeomSnapshotEntry *cpu_geom0_snapshot = NULL;
    GeomSnapshotEntry *cpu_geom_snapshot = NULL;
    double t0, t1;
    double cpu_geom0_time, cpu_geom_time;
    double kokkos_geom0_time, kokkos_geom_time;

    t0 = MPI_Wtime();
    ierr = UpdateElements(&fem[ibi], ElemUpdateGeom0Subdiv); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cpu_geom0_time = t1 - t0;

    t0 = MPI_Wtime();
    ierr = UpdateElements(&fem[ibi], ElemUpdateGeomSubdiv); CHKERRQ(ierr);
    t1 = MPI_Wtime();
    cpu_geom_time = t1 - t0;

    ierr = SnapshotGeomState_(&fem[ibi], PETSC_TRUE, n_compare_elems, &cpu_geom0_snapshot); CHKERRQ(ierr);
    ierr = SnapshotGeomState_(&fem[ibi], PETSC_FALSE, n_compare_elems, &cpu_geom_snapshot); CHKERRQ(ierr);

    const PetscInt n_subdiv_iterations = 5;
    double kokkos_geom0_times[5] = {0};
    double kokkos_geom_times[5] = {0};

    for (PetscInt iter = 0; iter < n_subdiv_iterations; ++iter) {
      if (iter == 0) {
        t0 = MPI_Wtime();
        ierr = RunKokkosSubdivGeomKernel(&fem[ibi], PETSC_TRUE); CHKERRQ(ierr);
        t1 = MPI_Wtime();
        kokkos_geom0_times[iter] = t1 - t0;
      }

      t0 = MPI_Wtime();
      ierr = RunKokkosSubdivGeomKernel(&fem[ibi], PETSC_FALSE); CHKERRQ(ierr);
      t1 = MPI_Wtime();
      kokkos_geom_times[iter] = t1 - t0;
    }

    kokkos_geom0_time = kokkos_geom0_times[0];

    double kokkos_geom_repeat_total = 0.0;
    PetscInt n_kokkos_geom_repeat = 0;
    for (PetscInt iter = 1; iter < n_subdiv_iterations; ++iter) {
      kokkos_geom_repeat_total += kokkos_geom_times[iter];
      ++n_kokkos_geom_repeat;
    }
    kokkos_geom_time = (n_kokkos_geom_repeat > 0) ? kokkos_geom_repeat_total / (double)n_kokkos_geom_repeat : kokkos_geom_times[0];

    for (PetscInt iter = 0; iter < n_subdiv_iterations; ++iter) {
      if (iter == 0) {
        PetscPrintf(PETSC_COMM_SELF,
                    "[iter %" PetscInt_FMT "] SubdivGeomKernel (reference): Kokkos=%.6e  CPU=%.6e  speedup=%.3f x (includes allocation)\n",
                    iter, kokkos_geom0_times[iter], cpu_geom0_time,
                    (double)TimingSpeedup_(cpu_geom0_time, kokkos_geom0_times[iter]));
      }

      PetscPrintf(PETSC_COMM_SELF,
                  "[iter %" PetscInt_FMT "] SubdivGeomKernel (current): Kokkos=%.6e  CPU=%.6e  speedup=%.3f x%s\n",
                  iter, kokkos_geom_times[iter], cpu_geom_time,
                  (double)TimingSpeedup_(cpu_geom_time, kokkos_geom_times[iter]),
                  (iter == 0) ? " (same outer iter as reference)" : " (repeat)");
    }

    ierr = PrintKokkosGeomTiming_(ibi,
                                  cpu_geom0_time,
                                  cpu_geom_time,
                                  kokkos_geom0_time,
                                  kokkos_geom_time); CHKERRQ(ierr);
    ierr = CompareGeomState_(&fem[ibi], ibi, "reference", PETSC_TRUE, n_compare_elems, cpu_geom0_snapshot); CHKERRQ(ierr);
    ierr = CompareGeomState_(&fem[ibi], ibi, "current", PETSC_FALSE, n_compare_elems, cpu_geom_snapshot); CHKERRQ(ierr);
    ierr = DestroyGeomStateSnapshot_(cpu_geom0_snapshot, n_compare_elems); CHKERRQ(ierr);
    ierr = DestroyGeomStateSnapshot_(cpu_geom_snapshot, n_compare_elems); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
