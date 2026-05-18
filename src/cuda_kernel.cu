#include "cuda_bridge.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static constexpr PetscInt kCudaPatchWidth_ = 16;
static constexpr PetscInt kCudaRegularNen_ = 12;
static constexpr PetscInt kCudaIrregularStride_ = 5 * kCudaPatchWidth_;
static constexpr PetscInt kCudaGeomStride_ = 24;
static constexpr PetscInt kCudaBasisStride_ = 18;
static constexpr PetscInt kCudaMetricStride_ = 18;
static constexpr int kCudaGeomThreadsPerElement_ = 256;

class CudaWorkspace_ {
public:
  PetscInt coord_capacity = 0;
  PetscInt elem_capacity = 0;
  PetscInt metric_entry_capacity = 0;
  PetscBool alloc_log = PETSC_FALSE;

  PetscInt *d_nv1 = NULL;
  PetscInt *d_nv2 = NULL;
  PetscInt *d_nv3 = NULL;
  PetscReal *d_x = NULL;
  PetscReal *d_y = NULL;
  PetscReal *d_z = NULL;
  PetscReal *d_element_coords_out = NULL;

  PetscInt *d_ire = NULL;
  PetscInt *d_val = NULL;
  PetscInt *d_patch = NULL;
  PetscInt *d_meta_out = NULL;
  PetscInt *d_status_elem_out = NULL;
  PetscReal *d_geom_out = NULL;
  PetscReal *d_irregular_out = NULL;

  PetscReal *d_theta = NULL;
  PetscReal *d_geom_current = NULL;
  PetscReal *d_geom_reference = NULL;
  PetscInt *d_status_metric_out = NULL;
  PetscReal *d_basis_current = NULL;
  PetscReal *d_basis_reference = NULL;
  PetscReal *d_metric_current = NULL;
  PetscReal *d_metric_reference = NULL;
};

static CudaWorkspace_ *g_cuda_workspace_ = NULL;

__device__ __constant__ PetscReal Na_center_dev_[2][12] = {
  {-0.0247, -0.0309,  0.0000, -0.4815, -0.1852,  0.0247,
    0.4815,  0.0000, -0.0062,  0.0309,  0.1852,  0.0062},
  {-0.0309, -0.0247, -0.1852, -0.4815,  0.0000, -0.0062,
    0.0000,  0.4815,  0.0247,  0.0062,  0.1852,  0.0309}
};

__device__ __constant__ PetscReal Nab_center_dev_[3][12] = {
  { 0.1111,  0.2222, -0.2222, -0.2222,  0.4444,  0.1111,
   -0.2222, -0.8889,  0.0000,  0.2222,  0.4444,  0.0000},
  { 0.2222,  0.1111,  0.4444, -0.2222, -0.2222,  0.0000,
   -0.8889, -0.2222,  0.1111,  0.0000,  0.4444,  0.2222},
  { 0.1667,  0.1667, -0.1111,  0.2222, -0.1111, -0.0556,
   -0.4444, -0.4444, -0.0556,  0.0556,  0.5556,  0.0556}
};

__device__ static inline Cmpnts MakeCmpnts_(PetscReal x, PetscReal y, PetscReal z)
{
  Cmpnts out;
  out.x = x;
  out.y = y;
  out.z = z;
  return out;
}

__device__ static inline Cmpnts Cross_(const Cmpnts a, const Cmpnts b)
{
  return MakeCmpnts_(a.y * b.z - b.y * a.z,
                     -a.x * b.z + b.x * a.z,
                     a.x * b.y - b.x * a.y);
}

__device__ static inline PetscReal Dot_(const Cmpnts a, const Cmpnts b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static inline Cmpnts Scale_(const PetscReal alpha, const Cmpnts a)
{
  return MakeCmpnts_(alpha * a.x, alpha * a.y, alpha * a.z);
}

__device__ static inline PetscReal Norm_(const Cmpnts a)
{
  return sqrt(Dot_(a, a));
}

__device__ static inline Cmpnts Add_(const Cmpnts a, const Cmpnts b)
{
  return MakeCmpnts_(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static inline void StoreCmpnts_(PetscReal *geom_out, PetscInt offset, const Cmpnts a)
{
  geom_out[offset + 0] = a.x;
  geom_out[offset + 1] = a.y;
  geom_out[offset + 2] = a.z;
}

__device__ static inline Cmpnts LoadCmpnts_(const PetscReal *geom_in, PetscInt offset)
{
  return MakeCmpnts_(geom_in[offset + 0], geom_in[offset + 1], geom_in[offset + 2]);
}

__device__ static inline PetscReal GeomCoeff_(PetscInt scalar, PetscReal na0, PetscReal na1,
                                              PetscReal nab0, PetscReal nab1, PetscReal nab2)
{
  if (scalar < 3) return nab0;
  if (scalar < 6) return nab1;
  if (scalar < 9) return nab2;
  if (scalar < 12) return na0;
  return na1;
}

__device__ static inline PetscReal GeomCoord_(PetscInt scalar, PetscReal x, PetscReal y, PetscReal z)
{
  const PetscInt component = scalar % 3;
  if (component == 0) return x;
  if (component == 1) return y;
  return z;
}

__device__ static inline void StoreMat3_(PetscReal *out, PetscInt offset, const PetscReal A[3][3])
{
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      out[offset + 3 * i + j] = A[i][j];
    }
  }
}

__device__ static inline void ComputeA3Alpha_(const Cmpnts a1,
                                              const Cmpnts a2,
                                              const Cmpnts Aaa,
                                              const Cmpnts Aab,
                                              const Cmpnts Abb,
                                              Cmpnts *a3_1,
                                              Cmpnts *a3_2,
                                              PetscInt *status)
{
  const PetscReal size_a3 = Norm_(Cross_(a1, a2));
  if (size_a3 <= 1.0e-30) {
    *status = 10;
    *a3_1 = MakeCmpnts_(0.0, 0.0, 0.0);
    *a3_2 = MakeCmpnts_(0.0, 0.0, 0.0);
    return;
  }

  *a3_1 = Scale_(1.0 / size_a3, Add_(Cross_(Aaa, a2), Cross_(a1, Aab)));
  *a3_2 = Scale_(1.0 / size_a3, Add_(Cross_(Aab, a2), Cross_(a1, Abb)));
  *status = 0;
}

__device__ static inline void ComputeMetricTensor_(const Cmpnts g_cov[3], PetscReal g_cov_out[3][3])
{
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      g_cov_out[i][j] = Dot_(g_cov[i], g_cov[j]);
    }
  }
}

__device__ static inline PetscBool Invert3x3_(const PetscReal A[3][3], PetscReal Ainv[3][3])
{
  const PetscReal a00 = A[0][0], a01 = A[0][1], a02 = A[0][2];
  const PetscReal a10 = A[1][0], a11 = A[1][1], a12 = A[1][2];
  const PetscReal a20 = A[2][0], a21 = A[2][1], a22 = A[2][2];

  const PetscReal c00 =  a11 * a22 - a12 * a21;
  const PetscReal c01 = -(a10 * a22 - a12 * a20);
  const PetscReal c02 =  a10 * a21 - a11 * a20;
  const PetscReal c10 = -(a01 * a22 - a02 * a21);
  const PetscReal c11 =  a00 * a22 - a02 * a20;
  const PetscReal c12 = -(a00 * a21 - a01 * a20);
  const PetscReal c20 =  a01 * a12 - a02 * a11;
  const PetscReal c21 = -(a00 * a12 - a02 * a10);
  const PetscReal c22 =  a00 * a11 - a01 * a10;

  const PetscReal det = a00 * c00 + a01 * c01 + a02 * c02;
  if (fabs(det) <= 1.0e-30) {
    return PETSC_FALSE;
  }

  const PetscReal inv_det = 1.0 / det;
  Ainv[0][0] = c00 * inv_det; Ainv[0][1] = c10 * inv_det; Ainv[0][2] = c20 * inv_det;
  Ainv[1][0] = c01 * inv_det; Ainv[1][1] = c11 * inv_det; Ainv[1][2] = c21 * inv_det;
  Ainv[2][0] = c02 * inv_det; Ainv[2][1] = c12 * inv_det; Ainv[2][2] = c22 * inv_det;
  return PETSC_TRUE;
}

__device__ static inline void ComputeContravariantBasis_(const PetscReal g_inv[3][3],
                                                         const Cmpnts g_cov[3],
                                                         Cmpnts g_cont[3])
{
  for (PetscInt i = 0; i < 3; ++i) {
    g_cont[i] = MakeCmpnts_(0.0, 0.0, 0.0);
    for (PetscInt j = 0; j < 3; ++j) {
      g_cont[i].x += g_inv[i][j] * g_cov[j].x;
      g_cont[i].y += g_inv[i][j] * g_cov[j].y;
      g_cont[i].z += g_inv[i][j] * g_cov[j].z;
    }
  }
}

__device__ static inline PetscInt ComputeMetricStateFromGeom_(const PetscReal *geom_in,
                                                              PetscReal theta3,
                                                              PetscReal *basis_out,
                                                              PetscReal *metric_out)
{
  const Cmpnts a1 = LoadCmpnts_(geom_in, 0);
  const Cmpnts a2 = LoadCmpnts_(geom_in, 3);
  const Cmpnts a3 = LoadCmpnts_(geom_in, 6);
  const Cmpnts Aaa = LoadCmpnts_(geom_in, 15);
  const Cmpnts Abb = LoadCmpnts_(geom_in, 18);
  const Cmpnts Aab = LoadCmpnts_(geom_in, 21);

  Cmpnts a3_1, a3_2;
  PetscInt status = 0;
  ComputeA3Alpha_(a1, a2, Aaa, Aab, Abb, &a3_1, &a3_2, &status);
  if (status != 0) {
    return status;
  }

  Cmpnts g_cov[3];
  g_cov[0] = Add_(a1, Scale_(theta3, a3_1));
  g_cov[1] = Add_(a2, Scale_(theta3, a3_2));
  g_cov[2] = a3;

  PetscReal g_cov_metric[3][3];
  PetscReal g_inv_metric[3][3];
  ComputeMetricTensor_(g_cov, g_cov_metric);
  if (!Invert3x3_(g_cov_metric, g_inv_metric)) {
    return 11;
  }

  Cmpnts g_cont[3];
  ComputeContravariantBasis_(g_inv_metric, g_cov, g_cont);

  StoreCmpnts_(basis_out, 0, g_cov[0]);
  StoreCmpnts_(basis_out, 3, g_cov[1]);
  StoreCmpnts_(basis_out, 6, g_cov[2]);
  StoreCmpnts_(basis_out, 9, g_cont[0]);
  StoreCmpnts_(basis_out, 12, g_cont[1]);
  StoreCmpnts_(basis_out, 15, g_cont[2]);

  StoreMat3_(metric_out, 0, g_cov_metric);
  StoreMat3_(metric_out, 9, g_inv_metric);
  return 0;
}

static PetscErrorCode CudaCheck_(cudaError_t code, const char *what)
{
  PetscFunctionBegin;
  PetscCheck(code == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_LIB,
             "%s failed: %s", what, cudaGetErrorString(code));
  PetscFunctionReturn(0);
}

static PetscErrorCode CudaMallocTimed_(void **ptr, size_t bytes, const char *what, PetscBool log_enabled,
                                       PetscLogDouble *elapsed)
{
  PetscLogDouble t0 = 0.0;
  PetscLogDouble t1 = 0.0;
  cudaError_t code;

  PetscFunctionBegin;
  if (log_enabled) {
    PetscCall(PetscTime(&t0));
  }
  code = cudaMalloc(ptr, bytes);
  if (log_enabled) {
    PetscCall(PetscTime(&t1));
  }
  if (log_enabled && elapsed != NULL) {
    *elapsed += t1 - t0;
  }
  PetscCall(CudaCheck_(code, what));
  PetscFunctionReturn(0);
}

static PetscErrorCode CudaFreePtr_(void **ptr)
{
  PetscFunctionBegin;
  if (*ptr != NULL) {
    PetscCall(CudaCheck_(cudaFree(*ptr), "cudaFree"));
    *ptr = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ConfigureCudaWorkspaceOptions_(CudaWorkspace_ *ws)
{
  PetscFunctionBegin;
  PetscCheck(ws != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "CUDA workspace must not be NULL");
  PetscCall(PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-cuda_alloc_log", &ws->alloc_log, PETSC_NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode GetCudaWorkspace_(CudaWorkspace_ **workspace)
{
  PetscFunctionBegin;
  if (g_cuda_workspace_ == NULL) {
    g_cuda_workspace_ = new CudaWorkspace_();
  }
  *workspace = g_cuda_workspace_;
  PetscFunctionReturn(0);
}

static PetscErrorCode EnsureCoordCapacity_(CudaWorkspace_ *ws, PetscInt coord_capacity)
{
  PetscBool alloc_log = PETSC_FALSE;
  PetscInt previous_capacity;

  PetscFunctionBegin;
  PetscCheck(ws != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "CUDA workspace must not be NULL");
  PetscCheck(coord_capacity >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "coord_capacity must be non-negative");
  alloc_log = ws->alloc_log;
  if (coord_capacity <= ws->coord_capacity) {
    if (alloc_log) {
      PetscPrintf(PETSC_COMM_SELF,
                  "[CUDA alloc] coord reuse requested=%" PetscInt_FMT " current=%" PetscInt_FMT "\n",
                  coord_capacity,
                  ws->coord_capacity);
    }
    PetscFunctionReturn(0);
  }

  previous_capacity = ws->coord_capacity;
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_x)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_y)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_z)));

  const size_t coord_bytes = static_cast<size_t>(coord_capacity) * sizeof(PetscReal);
  PetscLogDouble malloc_time = 0.0;
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_x), coord_bytes, "cudaMalloc(d_x)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_y), coord_bytes, "cudaMalloc(d_y)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_z), coord_bytes, "cudaMalloc(d_z)", alloc_log, &malloc_time));
  ws->coord_capacity = coord_capacity;
  if (alloc_log) {
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA alloc] coord malloc requested=%" PetscInt_FMT
                " previous=%" PetscInt_FMT " calls=3 bytes=%zu time=% .6e s\n",
                coord_capacity,
                previous_capacity,
                3u * coord_bytes,
                (double)malloc_time);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EnsureElementCapacity_(CudaWorkspace_ *ws, PetscInt elem_capacity)
{
  PetscBool alloc_log = PETSC_FALSE;
  PetscInt previous_capacity;

  PetscFunctionBegin;
  PetscCheck(ws != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "CUDA workspace must not be NULL");
  PetscCheck(elem_capacity >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "elem_capacity must be non-negative");
  alloc_log = ws->alloc_log;
  if (elem_capacity <= ws->elem_capacity) {
    if (alloc_log) {
      PetscPrintf(PETSC_COMM_SELF,
                  "[CUDA alloc] element reuse requested=%" PetscInt_FMT " current=%" PetscInt_FMT "\n",
                  elem_capacity,
                  ws->elem_capacity);
    }
    PetscFunctionReturn(0);
  }

  previous_capacity = ws->elem_capacity;
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv1)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv2)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv3)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_element_coords_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_ire)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_val)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_patch)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_meta_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_status_elem_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_irregular_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_reference)));

  const size_t elem_bytes = static_cast<size_t>(elem_capacity) * sizeof(PetscInt);
  const size_t patch_bytes = static_cast<size_t>(elem_capacity) * kCudaPatchWidth_ * sizeof(PetscInt);
  const size_t meta_bytes = static_cast<size_t>(elem_capacity) * 3u * sizeof(PetscInt);
  const size_t geom_bytes = static_cast<size_t>(elem_capacity) * kCudaGeomStride_ * sizeof(PetscReal);
  const size_t irregular_bytes = static_cast<size_t>(elem_capacity) * kCudaIrregularStride_ * sizeof(PetscReal);
  const size_t elem_coord_bytes = static_cast<size_t>(elem_capacity) * 9u * sizeof(PetscReal);

  PetscLogDouble malloc_time = 0.0;
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_nv1), elem_bytes, "cudaMalloc(d_nv1)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_nv2), elem_bytes, "cudaMalloc(d_nv2)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_nv3), elem_bytes, "cudaMalloc(d_nv3)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_element_coords_out), elem_coord_bytes,
                             "cudaMalloc(d_element_coords_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_ire), elem_bytes, "cudaMalloc(d_ire)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_val), elem_bytes, "cudaMalloc(d_val)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_patch), patch_bytes, "cudaMalloc(d_patch)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_meta_out), meta_bytes, "cudaMalloc(d_meta_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_status_elem_out), elem_bytes,
                             "cudaMalloc(d_status_elem_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_geom_out), geom_bytes, "cudaMalloc(d_geom_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_irregular_out), irregular_bytes,
                             "cudaMalloc(d_irregular_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_geom_current), geom_bytes,
                             "cudaMalloc(d_geom_current)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_geom_reference), geom_bytes,
                             "cudaMalloc(d_geom_reference)", alloc_log, &malloc_time));

  ws->elem_capacity = elem_capacity;
  const size_t malloc_bytes = 6u * elem_bytes + elem_coord_bytes + patch_bytes + meta_bytes +
                              3u * geom_bytes + irregular_bytes;
  if (alloc_log) {
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA alloc] element malloc requested=%" PetscInt_FMT
                " previous=%" PetscInt_FMT " calls=13 bytes=%zu time=% .6e s\n",
                elem_capacity,
                previous_capacity,
                malloc_bytes,
                (double)malloc_time);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EnsureMetricEntryCapacity_(CudaWorkspace_ *ws, PetscInt metric_entry_capacity)
{
  PetscBool alloc_log = PETSC_FALSE;
  PetscInt previous_capacity;

  PetscFunctionBegin;
  PetscCheck(ws != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "CUDA workspace must not be NULL");
  PetscCheck(metric_entry_capacity >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "metric_entry_capacity must be non-negative");
  alloc_log = ws->alloc_log;
  if (metric_entry_capacity <= ws->metric_entry_capacity) {
    if (alloc_log) {
      PetscPrintf(PETSC_COMM_SELF,
                  "[CUDA alloc] metric reuse requested=%" PetscInt_FMT " current=%" PetscInt_FMT "\n",
                  metric_entry_capacity,
                  ws->metric_entry_capacity);
    }
    PetscFunctionReturn(0);
  }

  previous_capacity = ws->metric_entry_capacity;
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_theta)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_status_metric_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_basis_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_basis_reference)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_metric_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_metric_reference)));

  const size_t qp_bytes = static_cast<size_t>(metric_entry_capacity) * sizeof(PetscReal);
  const size_t status_bytes = static_cast<size_t>(metric_entry_capacity) * sizeof(PetscInt);
  const size_t basis_bytes = static_cast<size_t>(metric_entry_capacity) * kCudaBasisStride_ * sizeof(PetscReal);
  const size_t metric_bytes = static_cast<size_t>(metric_entry_capacity) * kCudaMetricStride_ * sizeof(PetscReal);

  PetscLogDouble malloc_time = 0.0;
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_theta), qp_bytes, "cudaMalloc(d_theta)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_status_metric_out), status_bytes,
                             "cudaMalloc(d_status_metric_out)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_basis_current), basis_bytes,
                             "cudaMalloc(d_basis_current)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_basis_reference), basis_bytes,
                             "cudaMalloc(d_basis_reference)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_metric_current), metric_bytes,
                             "cudaMalloc(d_metric_current)", alloc_log, &malloc_time));
  PetscCall(CudaMallocTimed_(reinterpret_cast<void **>(&ws->d_metric_reference), metric_bytes,
                             "cudaMalloc(d_metric_reference)", alloc_log, &malloc_time));

  ws->metric_entry_capacity = metric_entry_capacity;
  const size_t malloc_bytes = qp_bytes + status_bytes + 2u * basis_bytes + 2u * metric_bytes;
  if (alloc_log) {
    PetscPrintf(PETSC_COMM_SELF,
                "[CUDA alloc] metric malloc requested=%" PetscInt_FMT
                " previous=%" PetscInt_FMT " calls=6 bytes=%zu time=% .6e s\n",
                metric_entry_capacity,
                previous_capacity,
                malloc_bytes,
                (double)malloc_time);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyCudaWorkspace_(CudaWorkspace_ *ws)
{
  PetscFunctionBegin;
  if (ws == NULL) {
    PetscFunctionReturn(0);
  }

  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv1)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv2)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_nv3)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_x)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_y)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_z)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_element_coords_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_ire)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_val)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_patch)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_meta_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_status_elem_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_irregular_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_theta)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_geom_reference)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_status_metric_out)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_basis_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_basis_reference)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_metric_current)));
  PetscCall(CudaFreePtr_(reinterpret_cast<void **>(&ws->d_metric_reference)));

  ws->coord_capacity = 0;
  ws->elem_capacity = 0;
  ws->metric_entry_capacity = 0;
  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode InitCudaWorkspaceImpl(void)
{
  PetscFunctionBegin;
  if (g_cuda_workspace_ == NULL) {
    g_cuda_workspace_ = new CudaWorkspace_();
  }
  PetscCall(ConfigureCudaWorkspaceOptions_(g_cuda_workspace_));
  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode DestroyCudaWorkspaceImpl(void)
{
  PetscFunctionBegin;
  PetscCall(DestroyCudaWorkspace_(g_cuda_workspace_));
  delete g_cuda_workspace_;
  g_cuda_workspace_ = NULL;
  PetscFunctionReturn(0);
}

__global__ static void ElementCoordCopyKernel_(PetscInt n_elmt,
                                               const PetscInt *nv1,
                                               const PetscInt *nv2,
                                               const PetscInt *nv3,
                                               const PetscReal *x_bp0,
                                               const PetscReal *y_bp0,
                                               const PetscReal *z_bp0,
                                               PetscReal *element_coords_out)
{
  const PetscInt ec = static_cast<PetscInt>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ec >= n_elmt) {
    return;
  }

  const PetscInt n1 = nv1[ec];
  const PetscInt n2 = nv2[ec];
  const PetscInt n3 = nv3[ec];
  const PetscInt base = 9 * ec;

  element_coords_out[base + 0] = x_bp0[n1] + 1.0;
  element_coords_out[base + 1] = y_bp0[n1] + 1.0;
  element_coords_out[base + 2] = z_bp0[n1] + 1.0;

  element_coords_out[base + 3] = x_bp0[n2] + 1.0;
  element_coords_out[base + 4] = y_bp0[n2] + 1.0;
  element_coords_out[base + 5] = z_bp0[n2] + 1.0;

  element_coords_out[base + 6] = x_bp0[n3] + 1.0;
  element_coords_out[base + 7] = y_bp0[n3] + 1.0;
  element_coords_out[base + 8] = z_bp0[n3] + 1.0;

  if (ec < 3) {
    printf("[CUDA kernel] ec=%d nodes=(%d,%d,%d)\n",
           static_cast<int>(ec),
           static_cast<int>(n1),
           static_cast<int>(n2),
           static_cast<int>(n3));
    printf("[CUDA kernel] ec=%d out=((%.6e, %.6e, %.6e), (%.6e, %.6e, %.6e), (%.6e, %.6e, %.6e))\n",
           static_cast<int>(ec),
           static_cast<double>(element_coords_out[base + 0]),
           static_cast<double>(element_coords_out[base + 1]),
           static_cast<double>(element_coords_out[base + 2]),
           static_cast<double>(element_coords_out[base + 3]),
           static_cast<double>(element_coords_out[base + 4]),
           static_cast<double>(element_coords_out[base + 5]),
           static_cast<double>(element_coords_out[base + 6]),
           static_cast<double>(element_coords_out[base + 7]),
           static_cast<double>(element_coords_out[base + 8]));
  }
}

__global__ static void SubdivGeomKernel_(PetscInt n_elmt,
                                         const PetscInt *ire,
                                         const PetscInt *val,
                                         const PetscInt *patch,
                                         const PetscReal *xb,
                                         const PetscReal *yb,
                                         const PetscReal *zb,
                                         PetscInt *meta_out,
                                         PetscInt *status_out,
                                         PetscReal *geom_out,
                                         PetscReal *irregular_out)
{
  const PetscInt ec = static_cast<PetscInt>(blockIdx.x);
  const PetscInt tid = static_cast<PetscInt>(threadIdx.x);
  if (ec >= n_elmt) {
    return;
  }

  const PetscInt meta_base = 3 * ec;
  const PetscInt geom_base = kCudaGeomStride_ * ec;
  const PetscInt irregular_base = kCudaIrregularStride_ * ec;
  const PetscInt patch_base = kCudaPatchWidth_ * ec;

  __shared__ PetscInt status_s;
  __shared__ PetscReal geom_part_s[15][kCudaPatchWidth_];
  __shared__ PetscReal geom_sum_s[15];
  __shared__ PetscReal coord_s[3][kCudaPatchWidth_];
  __shared__ PetscReal B1_s[22][kCudaPatchWidth_];
  __shared__ PetscReal IN_s[5][kCudaPatchWidth_];

  if (tid == 0) {
    meta_out[meta_base + 0] = 0;
    meta_out[meta_base + 1] = 0;
    meta_out[meta_base + 2] = 0;
    status_out[ec] = 0;
    status_s = 0;
  }
  for (PetscInt i = tid; i < kCudaGeomStride_; i += blockDim.x) {
    geom_out[geom_base + i] = 0.0;
  }
  for (PetscInt i = tid; i < kCudaIrregularStride_; i += blockDim.x) {
    irregular_out[irregular_base + i] = 0.0;
  }
  for (PetscInt i = tid; i < 15 * kCudaPatchWidth_; i += blockDim.x) {
    geom_part_s[i / kCudaPatchWidth_][i % kCudaPatchWidth_] = 0.0;
  }
  for (PetscInt i = tid; i < 15; i += blockDim.x) {
    geom_sum_s[i] = 0.0;
  }
  __syncthreads();

  if (ire[ec] == 0) {
    for (PetscInt work = tid; work < 15 * kCudaRegularNen_; work += blockDim.x) {
      const PetscInt scalar = work / kCudaRegularNen_;
      const PetscInt node_id = work % kCudaRegularNen_;
      const PetscInt node = patch[patch_base + node_id];
      if (node == 1000000) {
        atomicExch(&status_s, 2);
      } else {
        const PetscReal x = xb[node];
        const PetscReal y = yb[node];
        const PetscReal z = zb[node];
        const PetscReal na0 = Na_center_dev_[0][node_id];
        const PetscReal na1 = Na_center_dev_[1][node_id];
        const PetscReal nab0 = Nab_center_dev_[0][node_id];
        const PetscReal nab1 = Nab_center_dev_[1][node_id];
        const PetscReal nab2 = Nab_center_dev_[2][node_id];

        geom_part_s[scalar][node_id] = GeomCoeff_(scalar, na0, na1, nab0, nab1, nab2) *
                                       GeomCoord_(scalar, x, y, z);
      }
    }
    __syncthreads();
    if (tid < 15) {
      PetscReal sum = 0.0;
      for (PetscInt i = 0; i < kCudaRegularNen_; ++i) {
        sum += geom_part_s[tid][i];
      }
      geom_sum_s[tid] = sum;
    }
    __syncthreads();

    if (tid != 0) {
      return;
    }
    if (status_s != 0) {
      status_out[ec] = status_s;
      return;
    }

    const Cmpnts Aaa = MakeCmpnts_(geom_sum_s[0], geom_sum_s[1], geom_sum_s[2]);
    const Cmpnts Abb = MakeCmpnts_(geom_sum_s[3], geom_sum_s[4], geom_sum_s[5]);
    const Cmpnts Aab = MakeCmpnts_(geom_sum_s[6], geom_sum_s[7], geom_sum_s[8]);
    const Cmpnts ndx21 = MakeCmpnts_(geom_sum_s[9], geom_sum_s[10], geom_sum_s[11]);
    const Cmpnts ndx31 = MakeCmpnts_(geom_sum_s[12], geom_sum_s[13], geom_sum_s[14]);
    const Cmpnts nn_cross = Cross_(ndx21, ndx31);
    const PetscReal nn_norm = Norm_(nn_cross);
    if (nn_norm <= 0.0) {
      status_out[ec] = 4;
      return;
    }
    const Cmpnts nn = Scale_(1.0 / nn_norm, nn_cross);

    Cmpnts gc1 = Cross_(ndx31, nn);
    const PetscReal gc1_denom = Dot_(ndx21, gc1);
    if (fabs(gc1_denom) <= 1.0e-30) {
      status_out[ec] = 5;
      return;
    }
    gc1 = Scale_(1.0 / gc1_denom, gc1);

    Cmpnts gc2 = Cross_(nn, ndx21);
    const PetscReal gc2_denom = Dot_(ndx31, gc2);
    if (fabs(gc2_denom) <= 1.0e-30) {
      status_out[ec] = 6;
      return;
    }
    gc2 = Scale_(1.0 / gc2_denom, gc2);

    meta_out[meta_base + 2] = kCudaRegularNen_;

    StoreCmpnts_(geom_out, geom_base + 0, ndx21);
    StoreCmpnts_(geom_out, geom_base + 3, ndx31);
    StoreCmpnts_(geom_out, geom_base + 6, nn);
    StoreCmpnts_(geom_out, geom_base + 9, gc1);
    StoreCmpnts_(geom_out, geom_base + 12, gc2);
    StoreCmpnts_(geom_out, geom_base + 15, Aaa);
    StoreCmpnts_(geom_out, geom_base + 18, Abb);
    StoreCmpnts_(geom_out, geom_base + 21, Aab);

    // if (ec < 3) {
    //   printf("[CUDA geom kernel] ec=%d regular ndx21=(%.6e, %.6e, %.6e) nn=(%.6e, %.6e, %.6e)\n",
    //          (int)ec,
    //          (double)ndx21.x, (double)ndx21.y, (double)ndx21.z,
    //          (double)nn.x, (double)nn.y, (double)nn.z);
    // }
    return;
  }

  if (ire[ec] == 1) {
    const PetscInt v = val[ec];
    const PetscInt nen = v + 6;
    if (nen > kCudaPatchWidth_ || v + 12 > 22) {
      if (tid == 0) {
        status_out[ec] = 3;
      }
      return;
    }
    for (PetscInt i = tid; i < nen; i += blockDim.x) {
      const PetscInt node = patch[patch_base + i];
      if (node == 1000000) {
        atomicExch(&status_s, 2);
      } else {
        coord_s[0][i] = xb[node];
        coord_s[1][i] = yb[node];
        coord_s[2][i] = zb[node];
      }
    }
    for (PetscInt i = tid; i < 22 * kCudaPatchWidth_; i += blockDim.x) {
      B1_s[i / kCudaPatchWidth_][i % kCudaPatchWidth_] = 0.0;
    }
    __syncthreads();
    if (status_s != 0) {
      if (tid == 0) {
        status_out[ec] = status_s;
      }
      return;
    }

    const PetscReal angle = 2.0 * PETSC_PI / (PetscReal)v;
    const PetscReal c = cos(angle);
    const PetscReal w = (1.0 / (PetscReal)v) * (0.625 - pow(0.375 + 0.25 * c, 2.0));

    if (tid == 0) {
      B1_s[0][0] = 1 - v*w;  for (PetscInt j = 0; j < v; ++j) { B1_s[0][1+j] = w; }
      B1_s[1][0] = 0.375;  B1_s[1][1] = 0.375;  B1_s[1][2] = 0.125;  B1_s[1][v] = 0.125;
      B1_s[2][0] = 0.375;  B1_s[2][1] = 0.125;  B1_s[2][2] = 0.375;  B1_s[2][3] = 0.125;
      if (v > 5) { B1_s[v-4][0] = 0.375;  B1_s[v-4][v-5] = 0.125;  B1_s[v-4][v-4] = 0.375;  B1_s[v-4][v-3] = 0.125; }
      if (v > 4) { B1_s[v-3][0] = 0.375;  B1_s[v-3][v-4] = 0.125;  B1_s[v-3][v-3] = 0.375;  B1_s[v-3][v-2] = 0.125; }
      B1_s[v-2][0] = 0.375;  B1_s[v-2][v-3] = 0.125;  B1_s[v-2][v-2] = 0.375;  B1_s[v-2][v-1] = 0.125;
      B1_s[v-1][0] = 0.375;  B1_s[v-1][v-2] = 0.125;  B1_s[v-1][v-1] = 0.375;  B1_s[v-1][v] = 0.125;
      B1_s[v][0] = 0.375;  B1_s[v][1] = 0.125;  B1_s[v][v-1] = 0.125;  B1_s[v][v] = 0.375;
      B1_s[v+1][0] = 0.125;  B1_s[v+1][1] = 0.375;  B1_s[v+1][v] = 0.375;  B1_s[v+1][v+1] = 0.125;
      B1_s[v+2][0] = 0.0625;  B1_s[v+2][1] = 0.625;  B1_s[v+2][2] = 0.0625;  B1_s[v+2][v] = 0.0625;  B1_s[v+2][v+1] = 0.0625;  B1_s[v+2][v+2] = 0.0625;  B1_s[v+2][v+3] = 0.0625;
      B1_s[v+3][0] = 0.125;  B1_s[v+3][1] = 0.375;  B1_s[v+3][2] = 0.375;  B1_s[v+3][v+3] = 0.125;
      B1_s[v+4][0] = 0.0625;  B1_s[v+4][1] = 0.0625;  B1_s[v+4][v-1] = 0.0625;  B1_s[v+4][v] = 0.625;  B1_s[v+4][v+1] = 0.0625;  B1_s[v+4][v+4] = 0.0625;  B1_s[v+4][v+5] = 0.0625;
      B1_s[v+5][0] = 0.125;  B1_s[v+5][v-1] = 0.375;  B1_s[v+5][v] = 0.375;  B1_s[v+5][v+5] = 0.125;
      B1_s[v+6][1] = 0.375;  B1_s[v+6][v] = 0.125;  B1_s[v+6][v+1] = 0.375;  B1_s[v+6][v+2] =  0.125;
      B1_s[v+7][1] = 0.375;  B1_s[v+7][v+1] =  0.125;  B1_s[v+7][v+2] =  0.375;  B1_s[v+7][v+3] = 0.125;
      B1_s[v+8][1] = 0.375;  B1_s[v+8][2] = 0.125;  B1_s[v+8][v+2] = 0.125;  B1_s[v+8][v+3] = 0.375;
      B1_s[v+9][1] = 0.125;  B1_s[v+9][v] = 0.375;  B1_s[v+9][v+1] = 0.375;  B1_s[v+9][v+4] = 0.125;
      B1_s[v+10][v] = 0.375;  B1_s[v+10][v+1] = 0.125;  B1_s[v+10][v+4] = 0.375;  B1_s[v+10][v+5] = 0.125;
      B1_s[v+11][v-1] = 0.125;  B1_s[v+11][v] = 0.375;  B1_s[v+11][v+4] = 0.125;  B1_s[v+11][v+5] = 0.375;
    }
    __syncthreads();

    const PetscInt rows[12] = {v+9, v+6, v+4, v+1, v+2, v+5, v, 1, v+3, v-1, 0, 2};
    for (PetscInt work = tid; work < 5 * nen; work += blockDim.x) {
      const PetscInt which = work / nen;
      const PetscInt j = work % nen;
      PetscReal sum = 0.0;
      for (PetscInt i = 0; i < 12; ++i) {
        const PetscReal b3 = B1_s[rows[i]][j];
        if (which == 0) {
          sum += b3 * Na_center_dev_[0][i];
        } else if (which == 1) {
          sum += b3 * Na_center_dev_[1][i];
        } else if (which == 2) {
          sum += b3 * Nab_center_dev_[0][i];
        } else if (which == 3) {
          sum += b3 * Nab_center_dev_[1][i];
        } else {
          sum += b3 * Nab_center_dev_[2][i];
        }
      }
      const PetscReal scale = (which < 2) ? -2.0 : 4.0;
      IN_s[which][j] = scale * sum;
      irregular_out[irregular_base + which * kCudaPatchWidth_ + j] = IN_s[which][j];
    }
    __syncthreads();

    for (PetscInt work = tid; work < 15 * nen; work += blockDim.x) {
      const PetscInt scalar = work / nen;
      const PetscInt node_id = work % nen;
      const PetscReal x = coord_s[0][node_id];
      const PetscReal y = coord_s[1][node_id];
      const PetscReal z = coord_s[2][node_id];
      const PetscInt in_index = (scalar < 3) ? 2 : ((scalar < 6) ? 3 : ((scalar < 9) ? 4 : ((scalar < 12) ? 0 : 1)));
      geom_part_s[scalar][node_id] = IN_s[in_index][node_id] * GeomCoord_(scalar, x, y, z);
    }
    __syncthreads();
    if (tid < 15) {
      PetscReal sum = 0.0;
      for (PetscInt i = 0; i < nen; ++i) {
        sum += geom_part_s[tid][i];
      }
      geom_sum_s[tid] = sum;
    }
    __syncthreads();

    if (tid != 0) {
      return;
    }
    const Cmpnts Aaa = MakeCmpnts_(geom_sum_s[0], geom_sum_s[1], geom_sum_s[2]);
    const Cmpnts Abb = MakeCmpnts_(geom_sum_s[3], geom_sum_s[4], geom_sum_s[5]);
    const Cmpnts Aab = MakeCmpnts_(geom_sum_s[6], geom_sum_s[7], geom_sum_s[8]);
    const Cmpnts ndx21 = MakeCmpnts_(geom_sum_s[9], geom_sum_s[10], geom_sum_s[11]);
    const Cmpnts ndx31 = MakeCmpnts_(geom_sum_s[12], geom_sum_s[13], geom_sum_s[14]);
    const Cmpnts nn_cross = Cross_(ndx21, ndx31);
    const PetscReal nn_norm = Norm_(nn_cross);
    if (nn_norm <= 0.0) {
      status_out[ec] = 4;
      return;
    }
    const Cmpnts nn = Scale_(1.0 / nn_norm, nn_cross);

    Cmpnts gc1 = Cross_(ndx31, nn);
    const PetscReal gc1_denom = Dot_(ndx21, gc1);
    if (fabs(gc1_denom) <= 1.0e-30) {
      status_out[ec] = 5;
      return;
    }
    gc1 = Scale_(1.0 / gc1_denom, gc1);

    Cmpnts gc2 = Cross_(nn, ndx21);
    const PetscReal gc2_denom = Dot_(ndx31, gc2);
    if (fabs(gc2_denom) <= 1.0e-30) {
      status_out[ec] = 6;
      return;
    }
    gc2 = Scale_(1.0 / gc2_denom, gc2);

    meta_out[meta_base + 0] = 1;
    meta_out[meta_base + 1] = v;
    meta_out[meta_base + 2] = nen;

    StoreCmpnts_(geom_out, geom_base + 0, ndx21);
    StoreCmpnts_(geom_out, geom_base + 3, ndx31);
    StoreCmpnts_(geom_out, geom_base + 6, nn);
    StoreCmpnts_(geom_out, geom_base + 9, gc1);
    StoreCmpnts_(geom_out, geom_base + 12, gc2);
    StoreCmpnts_(geom_out, geom_base + 15, Aaa);
    StoreCmpnts_(geom_out, geom_base + 18, Abb);
    StoreCmpnts_(geom_out, geom_base + 21, Aab);

    // if (ec < 3) {
    //   printf("[CUDA geom kernel] ec=%d irregular v=%d nen=%d ndx21=(%.6e, %.6e, %.6e)\n",
    //          (int)ec, (int)v, (int)nen,
    //          (double)ndx21.x, (double)ndx21.y, (double)ndx21.z);
    // }
    return;
  }

  if (tid == 0) {
    status_out[ec] = 1;
  }
}

__global__ static void MetricTensorKernel_(PetscInt n_elmt,
                                           PetscInt n_qp,
                                           PetscReal h0,
                                           const PetscReal *theta,
                                           const PetscReal *geom_current,
                                           const PetscReal *geom_reference,
                                           PetscInt *status_out,
                                           PetscReal *basis_current_out,
                                           PetscReal *basis_reference_out,
                                           PetscReal *metric_current_out,
                                           PetscReal *metric_reference_out)
{
  const PetscInt idx = static_cast<PetscInt>(blockIdx.x * blockDim.x + threadIdx.x);
  const PetscInt total = n_elmt * n_qp;
  if (idx >= total) {
    return;
  }

  const PetscInt ec = idx / n_qp;
  const PetscInt qp = idx % n_qp;
  const PetscReal theta3 = 0.5 * h0 * theta[qp];
  const PetscInt geom_base = kCudaGeomStride_ * ec;
  const PetscInt basis_base = kCudaBasisStride_ * idx;
  const PetscInt metric_base = kCudaMetricStride_ * idx;

  PetscInt status = ComputeMetricStateFromGeom_(&geom_current[geom_base],
                                                theta3,
                                                &basis_current_out[basis_base],
                                                &metric_current_out[metric_base]);
  if (status == 0) {
    status = ComputeMetricStateFromGeom_(&geom_reference[geom_base],
                                         theta3,
                                         &basis_reference_out[basis_base],
                                         &metric_reference_out[metric_base]);
    if (status != 0) {
      status += 100;
    }
  }
  status_out[idx] = status;
}

extern "C" PetscErrorCode LaunchCudaElementCoordKernelImpl(
    PetscInt n_elmt,
    const PetscInt *nv1,
    const PetscInt *nv2,
    const PetscInt *nv3,
    const PetscReal *x_bp0,
    const PetscReal *y_bp0,
    const PetscReal *z_bp0,
    PetscReal *element_coords_out)
{
  CudaWorkspace_ *ws = NULL;
  PetscInt max_node_index = -1;

  PetscFunctionBegin;
  PetscCheck(n_elmt >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n_elmt must be non-negative");
  PetscCheck(nv1 != NULL && nv2 != NULL && nv3 != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Element connectivity arrays must not be NULL");
  PetscCheck(x_bp0 != NULL && y_bp0 != NULL && z_bp0 != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Coordinate arrays must not be NULL");
  PetscCheck(element_coords_out != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Output array must not be NULL");
  if (n_elmt == 0) {
    PetscFunctionReturn(0);
  }

  for (PetscInt ec = 0; ec < n_elmt; ++ec) {
    if (nv1[ec] > max_node_index) max_node_index = nv1[ec];
    if (nv2[ec] > max_node_index) max_node_index = nv2[ec];
    if (nv3[ec] > max_node_index) max_node_index = nv3[ec];
  }

  const size_t conn_bytes = static_cast<size_t>(n_elmt) * sizeof(PetscInt);
  const size_t coord_count = static_cast<size_t>(max_node_index + 1);
  const size_t coord_bytes = coord_count * sizeof(PetscReal);
  const size_t elem_coord_bytes = static_cast<size_t>(n_elmt) * 9u * sizeof(PetscReal);
  PetscCall(GetCudaWorkspace_(&ws));
  PetscCall(EnsureElementCapacity_(ws, n_elmt));
  PetscCall(EnsureCoordCapacity_(ws, max_node_index + 1));

  PetscPrintf(PETSC_COMM_SELF,
              "[CUDA host] n_elmt=%d max_node_index=%d coord_count=%d\n",
              (int)n_elmt,
              (int)max_node_index,
              (int)coord_count);

  PetscCall(CudaCheck_(cudaMemcpy(ws->d_nv1, nv1, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv1)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_nv2, nv2, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv2)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_nv3, nv3, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv3)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_x, x_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(x_bp0)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_y, y_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(y_bp0)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_z, z_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(z_bp0)"));

  if (n_elmt > 0) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((n_elmt + threads_per_block - 1) / threads_per_block);
    ElementCoordCopyKernel_<<<blocks, threads_per_block>>>(n_elmt,
                                                           ws->d_nv1,
                                                           ws->d_nv2,
                                                           ws->d_nv3,
                                                           ws->d_x,
                                                           ws->d_y,
                                                           ws->d_z,
                                                           ws->d_element_coords_out);
    PetscCall(CudaCheck_(cudaGetLastError(), "kernel launch"));
    PetscCall(CudaCheck_(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
    PetscCall(CudaCheck_(cudaMemcpy(element_coords_out, ws->d_element_coords_out, elem_coord_bytes,
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(element_coords_out)"));

    for (PetscInt ec = 0; ec < n_elmt && ec < 3; ++ec) {
      const PetscInt base = 9 * ec;
      PetscPrintf(PETSC_COMM_SELF,
                  "[CUDA host] ec=%d out=((%.6e, %.6e, %.6e), (%.6e, %.6e, %.6e), (%.6e, %.6e, %.6e))\n",
                  (int)ec,
                  (double)element_coords_out[base + 0],
                  (double)element_coords_out[base + 1],
                  (double)element_coords_out[base + 2],
                  (double)element_coords_out[base + 3],
                  (double)element_coords_out[base + 4],
                  (double)element_coords_out[base + 5],
                  (double)element_coords_out[base + 6],
                  (double)element_coords_out[base + 7],
                  (double)element_coords_out[base + 8]);
    }
  }

  PetscFunctionReturn(0);
}

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
    PetscReal *irregular_out)
{
  CudaWorkspace_ *ws = NULL;
  PetscInt max_node_index = -1;

  PetscFunctionBegin;
  PetscCheck(n_elmt >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n_elmt must be non-negative");
  PetscCheck(ire != NULL && val != NULL && patch != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Subdivision metadata arrays must not be NULL");
  PetscCheck(xb != NULL && yb != NULL && zb != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Coordinate arrays must not be NULL");
  PetscCheck(meta_out != NULL && status_out != NULL && geom_out != NULL && irregular_out != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Output arrays must not be NULL");
  if (n_elmt == 0) {
    PetscFunctionReturn(0);
  }

  for (PetscInt ec = 0; ec < n_elmt; ++ec) {
    const PetscInt nen = (ire[ec] == 0) ? kCudaRegularNen_ : (val[ec] + 6);
    PetscCheck(nen <= kCudaPatchWidth_, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "Element %" PetscInt_FMT " needs nen=%" PetscInt_FMT " but CUDA path supports up to %d patch nodes",
               ec, nen, (int)kCudaPatchWidth_);
    for (PetscInt i = 0; i < nen; ++i) {
      const PetscInt node = patch[kCudaPatchWidth_ * ec + i];
      if (node != 1000000 && node > max_node_index) {
        max_node_index = node;
      }
    }
  }

  PetscCheck(max_node_index >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No valid patch nodes were found");

  const size_t elem_bytes = static_cast<size_t>(n_elmt) * sizeof(PetscInt);
  const size_t patch_bytes = static_cast<size_t>(n_elmt) * kCudaPatchWidth_ * sizeof(PetscInt);
  const size_t coord_bytes = static_cast<size_t>(max_node_index + 1) * sizeof(PetscReal);
  const size_t meta_bytes = static_cast<size_t>(n_elmt) * 3u * sizeof(PetscInt);
  const size_t geom_bytes = static_cast<size_t>(n_elmt) * kCudaGeomStride_ * sizeof(PetscReal);
  const size_t irregular_bytes = static_cast<size_t>(n_elmt) * kCudaIrregularStride_ * sizeof(PetscReal);
  PetscCall(GetCudaWorkspace_(&ws));
  PetscCall(EnsureElementCapacity_(ws, n_elmt));
  PetscCall(EnsureCoordCapacity_(ws, max_node_index + 1));

  PetscCall(CudaCheck_(cudaMemcpy(ws->d_ire, ire, elem_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ire)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_val, val, elem_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(val)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_patch, patch, patch_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(patch)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_x, xb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(xb)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_y, yb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(yb)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_z, zb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(zb)"));

  {
    const int threads_per_block = kCudaGeomThreadsPerElement_;
    const int blocks = static_cast<int>(n_elmt);
    SubdivGeomKernel_<<<blocks, threads_per_block>>>(n_elmt,
                                                     ws->d_ire,
                                                     ws->d_val,
                                                     ws->d_patch,
                                                     ws->d_x,
                                                     ws->d_y,
                                                     ws->d_z,
                                                     ws->d_meta_out,
                                                     ws->d_status_elem_out,
                                                     ws->d_geom_out,
                                                     ws->d_irregular_out);
    PetscCall(CudaCheck_(cudaGetLastError(), "SubdivGeomKernel_ launch"));
    PetscCall(CudaCheck_(cudaDeviceSynchronize(), "cudaDeviceSynchronize(SubdivGeomKernel_)"));
  }

  PetscCall(CudaCheck_(cudaMemcpy(meta_out, ws->d_meta_out, meta_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(meta_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(status_out, ws->d_status_elem_out, elem_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(status_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(geom_out, ws->d_geom_out, geom_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(geom_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(irregular_out, ws->d_irregular_out, irregular_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(irregular_out)"));

  PetscFunctionReturn(0);
}

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
    PetscReal *metric_reference_out)
{
  CudaWorkspace_ *ws = NULL;

  PetscFunctionBegin;
  PetscCheck(n_elmt >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n_elmt must be non-negative");
  PetscCheck(n_qp > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "n_qp must be positive");
  PetscCheck(theta != NULL && geom_current != NULL && geom_reference != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Kernel input arrays must not be NULL");
  PetscCheck(status_out != NULL && basis_current_out != NULL && basis_reference_out != NULL &&
             metric_current_out != NULL && metric_reference_out != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Kernel output arrays must not be NULL");
  if (n_elmt == 0) {
    PetscFunctionReturn(0);
  }

  const size_t qp_bytes = static_cast<size_t>(n_qp) * sizeof(PetscReal);
  const size_t geom_bytes = static_cast<size_t>(n_elmt) * kCudaGeomStride_ * sizeof(PetscReal);
  const size_t total_entries = static_cast<size_t>(n_elmt) * static_cast<size_t>(n_qp);
  const size_t status_bytes = total_entries * sizeof(PetscInt);
  const size_t basis_bytes = total_entries * kCudaBasisStride_ * sizeof(PetscReal);
  const size_t metric_bytes = total_entries * kCudaMetricStride_ * sizeof(PetscReal);
  const PetscInt metric_entry_capacity = n_elmt * n_qp;
  PetscCall(GetCudaWorkspace_(&ws));
  PetscCall(EnsureElementCapacity_(ws, n_elmt));
  PetscCall(EnsureMetricEntryCapacity_(ws, metric_entry_capacity));

  PetscCall(CudaCheck_(cudaMemcpy(ws->d_theta, theta, qp_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(theta)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_geom_current, geom_current, geom_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(geom_current)"));
  PetscCall(CudaCheck_(cudaMemcpy(ws->d_geom_reference, geom_reference, geom_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(geom_reference)"));

  {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((total_entries + threads_per_block - 1) / threads_per_block);
    MetricTensorKernel_<<<blocks, threads_per_block>>>(n_elmt,
                                                       n_qp,
                                                       h0,
                                                       ws->d_theta,
                                                       ws->d_geom_current,
                                                       ws->d_geom_reference,
                                                       ws->d_status_metric_out,
                                                       ws->d_basis_current,
                                                       ws->d_basis_reference,
                                                       ws->d_metric_current,
                                                       ws->d_metric_reference);
    PetscCall(CudaCheck_(cudaGetLastError(), "MetricTensorKernel_ launch"));
    PetscCall(CudaCheck_(cudaDeviceSynchronize(), "cudaDeviceSynchronize(MetricTensorKernel_)"));
  }

  PetscCall(CudaCheck_(cudaMemcpy(status_out, ws->d_status_metric_out, status_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(status_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(basis_current_out, ws->d_basis_current, basis_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(basis_current_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(basis_reference_out, ws->d_basis_reference, basis_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(basis_reference_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(metric_current_out, ws->d_metric_current, metric_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(metric_current_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(metric_reference_out, ws->d_metric_reference, metric_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(metric_reference_out)"));

  PetscFunctionReturn(0);
}
