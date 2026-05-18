#include "cuda_bridge.h"

#include <vector>

extern "C" PetscErrorCode InitCudaWorkspaceImpl(void);
extern "C" PetscErrorCode DestroyCudaWorkspaceImpl(void);

extern "C" PetscErrorCode LaunchCudaElementCoordKernelImpl(
    PetscInt n_elmt,
    const PetscInt *nv1,
    const PetscInt *nv2,
    const PetscInt *nv3,
    const PetscReal *x_bp0,
    const PetscReal *y_bp0,
    const PetscReal *z_bp0,
    PetscReal *element_coords_out);

extern "C" PetscErrorCode LaunchCudaSubdivGeomKernelImpl(
    PetscInt n_elmt,
    const PetscInt *ire,
    const PetscInt *val,
    const PetscInt *patch,
    const PetscReal *xb,
    const PetscReal *yb,
    const PetscReal *zb,
    PetscInt *meta_out,
    PetscInt *status_out,
    PetscReal *geom_out,
    PetscReal *irregular_out);

extern "C" PetscErrorCode LaunchCudaMetricTensorKernelImpl(
    PetscInt n_elmt,
    PetscInt n_qp,
    PetscReal h0,
    const PetscReal *theta,
    const PetscReal *geom_current,
    const PetscReal *geom_reference,
    PetscInt *status_out,
    PetscReal *basis_current_out,
    PetscReal *basis_reference_out,
    PetscReal *metric_current_out,
    PetscReal *metric_reference_out);

static constexpr PetscInt kCudaGeomStrideCpp_ = 24;
static constexpr PetscInt kCudaBasisStrideCpp_ = 18;
static constexpr PetscInt kCudaMetricStrideCpp_ = 18;

class CudaHostWorkspace_ {
public:
  std::vector<PetscReal> element_coords;

  std::vector<PetscInt> geom_meta;
  std::vector<PetscInt> geom_status;
  std::vector<PetscReal> geom;
  std::vector<PetscReal> irregular;

  std::vector<PetscReal> theta;
  std::vector<PetscReal> metric_geom_current;
  std::vector<PetscReal> metric_geom_reference;
  std::vector<PetscInt> metric_status;
  std::vector<PetscReal> basis_current;
  std::vector<PetscReal> basis_reference;
  std::vector<PetscReal> metric_current;
  std::vector<PetscReal> metric_reference;
};

static CudaHostWorkspace_ g_cuda_host_workspace_;

template <typename T>
static inline void ResizeHostBuffer_(std::vector<T> &buffer, size_t size)
{
  buffer.resize(size);
}

template <typename T>
static PetscErrorCode CheckHostBufferSize_(const std::vector<T> &buffer, size_t size, const char *name)
{
  PetscFunctionBegin;
  PetscCheck(buffer.size() >= size, PETSC_COMM_SELF, PETSC_ERR_ORDER,
             "CUDA host buffer %s is not prepared for %zu entries; call PrepareCudaHostWorkspace() first",
             name, size);
  PetscFunctionReturn(0);
}

template <typename T>
static inline void ReleaseHostBuffer_(std::vector<T> &buffer)
{
  std::vector<T>().swap(buffer);
}

static void ClearCudaHostWorkspace_(void)
{
  ReleaseHostBuffer_(g_cuda_host_workspace_.element_coords);
  ReleaseHostBuffer_(g_cuda_host_workspace_.geom_meta);
  ReleaseHostBuffer_(g_cuda_host_workspace_.geom_status);
  ReleaseHostBuffer_(g_cuda_host_workspace_.geom);
  ReleaseHostBuffer_(g_cuda_host_workspace_.irregular);
  ReleaseHostBuffer_(g_cuda_host_workspace_.theta);
  ReleaseHostBuffer_(g_cuda_host_workspace_.metric_geom_current);
  ReleaseHostBuffer_(g_cuda_host_workspace_.metric_geom_reference);
  ReleaseHostBuffer_(g_cuda_host_workspace_.metric_status);
  ReleaseHostBuffer_(g_cuda_host_workspace_.basis_current);
  ReleaseHostBuffer_(g_cuda_host_workspace_.basis_reference);
  ReleaseHostBuffer_(g_cuda_host_workspace_.metric_current);
  ReleaseHostBuffer_(g_cuda_host_workspace_.metric_reference);
}

static inline void PackCmpnts_(PetscReal *dst, const Cmpnts &src)
{
  dst[0] = src.x;
  dst[1] = src.y;
  dst[2] = src.z;
}

static inline Cmpnts UnpackCmpnts_(const PetscReal *src)
{
  Cmpnts out;
  out.x = src[0];
  out.y = src[1];
  out.z = src[2];
  return out;
}

extern "C" PetscErrorCode InitCudaWorkspace(void)
{
  return InitCudaWorkspaceImpl();
}

extern "C" PetscErrorCode DestroyCudaWorkspace(void)
{
  ClearCudaHostWorkspace_();
  return DestroyCudaWorkspaceImpl();
}

extern "C" PetscErrorCode PrepareCudaHostWorkspace(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");
  PetscCall(InitCudaWorkspace());

  IBMNodes *ibm = fem->ibm;
  ActData *act = &fem->act_data;
  PetscCheck(act->n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "n_qp must be > 0");

  const size_t n_elem = static_cast<size_t>(ibm->n_elmt);
  const size_t n_qp = static_cast<size_t>(act->n_qp);
  const size_t n_entries = n_elem * n_qp;

  ResizeHostBuffer_(g_cuda_host_workspace_.element_coords, n_elem * 9u);

  ResizeHostBuffer_(g_cuda_host_workspace_.geom_meta, n_elem * 3u);
  ResizeHostBuffer_(g_cuda_host_workspace_.geom_status, n_elem);
  ResizeHostBuffer_(g_cuda_host_workspace_.geom, n_elem * kCudaGeomStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.irregular, n_elem * 80u);

  ResizeHostBuffer_(g_cuda_host_workspace_.theta, n_qp);
  ResizeHostBuffer_(g_cuda_host_workspace_.metric_geom_current, n_elem * kCudaGeomStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.metric_geom_reference, n_elem * kCudaGeomStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.metric_status, n_entries);
  ResizeHostBuffer_(g_cuda_host_workspace_.basis_current, n_entries * kCudaBasisStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.basis_reference, n_entries * kCudaBasisStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.metric_current, n_entries * kCudaMetricStrideCpp_);
  ResizeHostBuffer_(g_cuda_host_workspace_.metric_reference, n_entries * kCudaMetricStrideCpp_);

  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode RunCudaElementCoordKernel(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");
  PetscCall(InitCudaWorkspace());

  IBMNodes *ibm = fem->ibm;
  const size_t n_elem = static_cast<size_t>(ibm->n_elmt);
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.element_coords, n_elem * 9u, "element_coords"));
  PetscReal *element_coords = g_cuda_host_workspace_.element_coords.data();

  PetscCall(LaunchCudaElementCoordKernelImpl(ibm->n_elmt,
                                             ibm->nv1,
                                             ibm->nv2,
                                             ibm->nv3,
                                             ibm->x_bp0,
                                             ibm->y_bp0,
                                             ibm->z_bp0,
                                             element_coords));

  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode RunCudaMetricTensorKernel(FE *fem)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");
  PetscCall(InitCudaWorkspace());

  IBMNodes *ibm = fem->ibm;
  ActData *act = &fem->act_data;
  PetscCheck(act->n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "n_qp must be > 0");
  PetscCheck(act->theta != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "act->theta must not be NULL");
  PetscCheck(act->elem_act_data != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "act->elem_act_data must not be NULL");

  const size_t n_elem = static_cast<size_t>(ibm->n_elmt);
  const size_t n_qp = static_cast<size_t>(act->n_qp);
  const size_t n_entries = n_elem * n_qp;

  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.theta, n_qp, "theta"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.metric_geom_current,
                                 n_elem * kCudaGeomStrideCpp_, "metric_geom_current"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.metric_geom_reference,
                                 n_elem * kCudaGeomStrideCpp_, "metric_geom_reference"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.metric_status, n_entries, "metric_status"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.basis_current,
                                 n_entries * kCudaBasisStrideCpp_, "basis_current"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.basis_reference,
                                 n_entries * kCudaBasisStrideCpp_, "basis_reference"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.metric_current,
                                 n_entries * kCudaMetricStrideCpp_, "metric_current"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.metric_reference,
                                 n_entries * kCudaMetricStrideCpp_, "metric_reference"));

  PetscReal *theta = g_cuda_host_workspace_.theta.data();
  PetscReal *geom_current = g_cuda_host_workspace_.metric_geom_current.data();
  PetscReal *geom_reference = g_cuda_host_workspace_.metric_geom_reference.data();
  PetscInt *status = g_cuda_host_workspace_.metric_status.data();
  PetscReal *basis_current = g_cuda_host_workspace_.basis_current.data();
  PetscReal *basis_reference = g_cuda_host_workspace_.basis_reference.data();
  PetscReal *metric_current = g_cuda_host_workspace_.metric_current.data();
  PetscReal *metric_reference = g_cuda_host_workspace_.metric_reference.data();

  for (PetscInt qp = 0; qp < act->n_qp; ++qp) {
    theta[qp] = PetscRealPart(act->theta[qp]);
  }

  for (PetscInt ec = 0; ec < ibm->n_elmt; ++ec) {
    ElemActData *ead = &act->elem_act_data[ec];
    PetscCheck(ead->geom != NULL && ead->geom0 != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Element %" PetscInt_FMT " geometry caches must not be NULL", ec);
    PetscCheck(ead->g != NULL && ead->g0 != NULL && ead->gm != NULL && ead->gm0 != NULL,
               PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Element %" PetscInt_FMT " metric/basis storage must not be NULL", ec);

    const size_t base = static_cast<size_t>(ec) * kCudaGeomStrideCpp_;
    const SubdivGeomQP &Gc = ead->geom[0];
    const SubdivGeomQP &G0 = ead->geom0[0];

    PackCmpnts_(&geom_current[base + 0], Gc.ndx21);
    PackCmpnts_(&geom_current[base + 3], Gc.ndx31);
    PackCmpnts_(&geom_current[base + 6], Gc.nn);
    PackCmpnts_(&geom_current[base + 9], Gc.gc1);
    PackCmpnts_(&geom_current[base + 12], Gc.gc2);
    PackCmpnts_(&geom_current[base + 15], Gc.Aaa);
    PackCmpnts_(&geom_current[base + 18], Gc.Abb);
    PackCmpnts_(&geom_current[base + 21], Gc.Aab);

    PackCmpnts_(&geom_reference[base + 0], G0.ndx21);
    PackCmpnts_(&geom_reference[base + 3], G0.ndx31);
    PackCmpnts_(&geom_reference[base + 6], G0.nn);
    PackCmpnts_(&geom_reference[base + 9], G0.gc1);
    PackCmpnts_(&geom_reference[base + 12], G0.gc2);
    PackCmpnts_(&geom_reference[base + 15], G0.Aaa);
    PackCmpnts_(&geom_reference[base + 18], G0.Abb);
    PackCmpnts_(&geom_reference[base + 21], G0.Aab);
  }

  PetscCall(LaunchCudaMetricTensorKernelImpl(ibm->n_elmt,
                                             act->n_qp,
                                             fem->h0,
                                             theta,
                                             geom_current,
                                             geom_reference,
                                             status,
                                             basis_current,
                                             basis_reference,
                                             metric_current,
                                             metric_reference));

  for (PetscInt ec = 0; ec < ibm->n_elmt; ++ec) {
    ElemActData *ead = &act->elem_act_data[ec];
    for (PetscInt qp = 0; qp < act->n_qp; ++qp) {
      const size_t entry = static_cast<size_t>(ec) * n_qp + static_cast<size_t>(qp);
      PetscCheck(status[entry] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                 "CUDA metric kernel failed for element %" PetscInt_FMT ", qp %" PetscInt_FMT
                 " with status %d",
                 ec, qp, (int)status[entry]);

      const size_t basis_base = entry * kCudaBasisStrideCpp_;
      const size_t metric_base = entry * kCudaMetricStrideCpp_;

      ead->g[qp].Cov[0] = UnpackCmpnts_(&basis_current[basis_base + 0]);
      ead->g[qp].Cov[1] = UnpackCmpnts_(&basis_current[basis_base + 3]);
      ead->g[qp].Cov[2] = UnpackCmpnts_(&basis_current[basis_base + 6]);
      ead->g[qp].Cont[0] = UnpackCmpnts_(&basis_current[basis_base + 9]);
      ead->g[qp].Cont[1] = UnpackCmpnts_(&basis_current[basis_base + 12]);
      ead->g[qp].Cont[2] = UnpackCmpnts_(&basis_current[basis_base + 15]);

      ead->g0[qp].Cov[0] = UnpackCmpnts_(&basis_reference[basis_base + 0]);
      ead->g0[qp].Cov[1] = UnpackCmpnts_(&basis_reference[basis_base + 3]);
      ead->g0[qp].Cov[2] = UnpackCmpnts_(&basis_reference[basis_base + 6]);
      ead->g0[qp].Cont[0] = UnpackCmpnts_(&basis_reference[basis_base + 9]);
      ead->g0[qp].Cont[1] = UnpackCmpnts_(&basis_reference[basis_base + 12]);
      ead->g0[qp].Cont[2] = UnpackCmpnts_(&basis_reference[basis_base + 15]);

      for (PetscInt i = 0; i < 3; ++i) {
        for (PetscInt j = 0; j < 3; ++j) {
          const size_t ij = static_cast<size_t>(3 * i + j);
          ead->gm[qp].Cov[i][j] = metric_current[metric_base + ij];
          ead->gm[qp].Cont[i][j] = metric_current[metric_base + 9 + ij];
          ead->gm0[qp].Cov[i][j] = metric_reference[metric_base + ij];
          ead->gm0[qp].Cont[i][j] = metric_reference[metric_base + 9 + ij];
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode RunCudaSubdivGeomKernel(FE *fem, PetscBool use_reference_coords)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");
  PetscCall(InitCudaWorkspace());

  IBMNodes *ibm = fem->ibm;
  const PetscReal *xb = use_reference_coords ? ibm->x_bp0 : ibm->x_bp;
  const PetscReal *yb = use_reference_coords ? ibm->y_bp0 : ibm->y_bp;
  const PetscReal *zb = use_reference_coords ? ibm->z_bp0 : ibm->z_bp;

  const size_t n_elem = static_cast<size_t>(ibm->n_elmt);
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.geom_meta, n_elem * 3u, "geom_meta"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.geom_status, n_elem, "geom_status"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.geom, n_elem * kCudaGeomStrideCpp_, "geom"));
  PetscCall(CheckHostBufferSize_(g_cuda_host_workspace_.irregular, n_elem * 80u, "irregular"));

  PetscInt *meta = g_cuda_host_workspace_.geom_meta.data();
  PetscInt *status = g_cuda_host_workspace_.geom_status.data();
  PetscReal *geom = g_cuda_host_workspace_.geom.data();
  PetscReal *irregular = g_cuda_host_workspace_.irregular.data();

  PetscCall(LaunchCudaSubdivGeomKernelImpl(ibm->n_elmt,
                                           ibm->ire,
                                           ibm->val,
                                           ibm->patch,
                                           xb,
                                           yb,
                                           zb,
                                           meta,
                                           status,
                                           geom,
                                           irregular));

  // Copy geometry data from device to host structures
  for (PetscInt ec = 0; ec < ibm->n_elmt; ++ec) {
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscCheck(ead != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Element %" PetscInt_FMT " act data must not be NULL", ec);

    const size_t meta_base = static_cast<size_t>(ec) * 3u;
    const size_t geom_base = static_cast<size_t>(ec) * 24u;
    const size_t irregular_base = static_cast<size_t>(ec) * 80u;

    PetscInt is_irregular = meta[meta_base + 0];
    PetscInt v = meta[meta_base + 1];
    PetscInt nen = meta[meta_base + 2];

    // Select target geometry based on use_reference_coords
    SubdivGeomQP *target_geom = use_reference_coords ? &ead->geom0[0] : &ead->geom[0];

    // Set metadata
    target_geom->is_irregular = is_irregular;
    target_geom->v = v;
    target_geom->nen = nen;

    // Unpack geometry vectors
    target_geom->ndx21 = UnpackCmpnts_(&geom[geom_base + 0]);
    target_geom->ndx31 = UnpackCmpnts_(&geom[geom_base + 3]);
    target_geom->nn = UnpackCmpnts_(&geom[geom_base + 6]);
    target_geom->gc1 = UnpackCmpnts_(&geom[geom_base + 9]);
    target_geom->gc2 = UnpackCmpnts_(&geom[geom_base + 12]);
    target_geom->Aaa = UnpackCmpnts_(&geom[geom_base + 15]);
    target_geom->Abb = UnpackCmpnts_(&geom[geom_base + 18]);
    target_geom->Aab = UnpackCmpnts_(&geom[geom_base + 21]);

    // Handle irregular arrays if needed
    if (is_irregular) {
      // Allocate irregular arrays if not already allocated
      if (target_geom->INa0 == NULL) {
        PetscCall(PetscMalloc1(nen, &target_geom->INa0));
        PetscCall(PetscMalloc1(nen, &target_geom->INa1));
        PetscCall(PetscMalloc1(nen, &target_geom->INab0));
        PetscCall(PetscMalloc1(nen, &target_geom->INab1));
        PetscCall(PetscMalloc1(nen, &target_geom->INab2));
      }
      // Copy irregular arrays (each takes nen elements, total 5*nen from the 80-element buffer)
      PetscCall(PetscMemcpy(target_geom->INa0, &irregular[irregular_base], (size_t)nen * sizeof(PetscReal)));
      PetscCall(PetscMemcpy(target_geom->INa1, &irregular[irregular_base + nen], (size_t)nen * sizeof(PetscReal)));
      PetscCall(PetscMemcpy(target_geom->INab0, &irregular[irregular_base + 2*nen], (size_t)nen * sizeof(PetscReal)));
      PetscCall(PetscMemcpy(target_geom->INab1, &irregular[irregular_base + 3*nen], (size_t)nen * sizeof(PetscReal)));
      PetscCall(PetscMemcpy(target_geom->INab2, &irregular[irregular_base + 4*nen], (size_t)nen * sizeof(PetscReal)));
    }
  }

  PetscFunctionReturn(0);
}
