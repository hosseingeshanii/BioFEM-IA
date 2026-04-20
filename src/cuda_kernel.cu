#include "cuda_bridge.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static constexpr PetscInt kCudaPatchWidth_ = 16;
static constexpr PetscInt kCudaRegularNen_ = 12;
static constexpr PetscInt kCudaIrregularStride_ = 5 * kCudaPatchWidth_;
static constexpr PetscInt kCudaGeomStride_ = 24;

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

__device__ static inline void StoreCmpnts_(PetscReal *geom_out, PetscInt offset, const Cmpnts a)
{
  geom_out[offset + 0] = a.x;
  geom_out[offset + 1] = a.y;
  geom_out[offset + 2] = a.z;
}

static PetscErrorCode CudaCheck_(cudaError_t code, const char *what)
{
  PetscFunctionBegin;
  PetscCheck(code == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_LIB,
             "%s failed: %s", what, cudaGetErrorString(code));
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
  const PetscInt ec = static_cast<PetscInt>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ec >= n_elmt) {
    return;
  }

  const PetscInt meta_base = 3 * ec;
  const PetscInt geom_base = kCudaGeomStride_ * ec;
  const PetscInt irregular_base = kCudaIrregularStride_ * ec;
  const PetscInt patch_base = kCudaPatchWidth_ * ec;

  meta_out[meta_base + 0] = 0;
  meta_out[meta_base + 1] = 0;
  meta_out[meta_base + 2] = 0;
  status_out[ec] = 0;

  for (PetscInt i = 0; i < kCudaGeomStride_; ++i) {
    geom_out[geom_base + i] = 0.0;
  }
  for (PetscInt i = 0; i < kCudaIrregularStride_; ++i) {
    irregular_out[irregular_base + i] = 0.0;
  }

  if (ire[ec] == 0) {
    PetscReal x[kCudaRegularNen_], y[kCudaRegularNen_], z[kCudaRegularNen_];

    for (PetscInt i = 0; i < kCudaRegularNen_; ++i) {
      const PetscInt node = patch[patch_base + i];
      if (node == 1000000) {
        status_out[ec] = 2;
        return;
      }
      x[i] = xb[node];
      y[i] = yb[node];
      z[i] = zb[node];
    }

    Cmpnts Aaa = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts Abb = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts Aab = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts ndx21 = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts ndx31 = MakeCmpnts_(0.0, 0.0, 0.0);

    for (PetscInt i = 0; i < kCudaRegularNen_; ++i) {
      const PetscReal na0 = Na_center_dev_[0][i];
      const PetscReal na1 = Na_center_dev_[1][i];
      const PetscReal nab0 = Nab_center_dev_[0][i];
      const PetscReal nab1 = Nab_center_dev_[1][i];
      const PetscReal nab2 = Nab_center_dev_[2][i];

      Aaa.x += nab0 * x[i]; Aaa.y += nab0 * y[i]; Aaa.z += nab0 * z[i];
      Abb.x += nab1 * x[i]; Abb.y += nab1 * y[i]; Abb.z += nab1 * z[i];
      Aab.x += nab2 * x[i]; Aab.y += nab2 * y[i]; Aab.z += nab2 * z[i];

      ndx21.x += na0 * x[i]; ndx21.y += na0 * y[i]; ndx21.z += na0 * z[i];
      ndx31.x += na1 * x[i]; ndx31.y += na1 * y[i]; ndx31.z += na1 * z[i];
    }

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

    if (ec < 3) {
      printf("[CUDA geom kernel] ec=%d regular ndx21=(%.6e, %.6e, %.6e) nn=(%.6e, %.6e, %.6e)\n",
             (int)ec,
             (double)ndx21.x, (double)ndx21.y, (double)ndx21.z,
             (double)nn.x, (double)nn.y, (double)nn.z);
    }
    return;
  }

  if (ire[ec] == 1) {
    const PetscInt v = val[ec];
    const PetscInt nen = v + 6;
    if (nen > kCudaPatchWidth_ || v + 12 > 22) {
      status_out[ec] = 3;
      return;
    }

    PetscReal x[16], y[16], z[16];
    for (PetscInt i = 0; i < nen; ++i) {
      const PetscInt node = patch[patch_base + i];
      if (node == 1000000) {
        status_out[ec] = 2;
        return;
      }
      x[i] = xb[node];
      y[i] = yb[node];
      z[i] = zb[node];
    }

    const PetscReal angle = 2.0 * PETSC_PI / (PetscReal)v;
    const PetscReal c = cos(angle);
    const PetscReal w = (1.0 / (PetscReal)v) * (0.625 - pow(0.375 + 0.25 * c, 2.0));

    PetscReal B2[12][22], B1[22][16], B3[12][16];
    for (PetscInt i = 0; i < 12; ++i) {
      for (PetscInt j = 0; j < 22; ++j) {
        B2[i][j] = 0.0;
      }
    }
    for (PetscInt i = 0; i < 22; ++i) {
      for (PetscInt j = 0; j < 16; ++j) {
        B1[i][j] = 0.0;
      }
    }

    B2[0][v+9]  = 1.;  B2[1][v+6]  = 1.;  B2[2][v+4]  = 1.;  B2[3][v+1]  = 1.;
    B2[4][v+2]  = 1.;  B2[5][v+5]  = 1.;  B2[6][v]    = 1.;  B2[7][1]    = 1.;
    B2[8][v+3]  = 1.;  B2[9][v-1]  = 1.;  B2[10][0]   = 1.;  B2[11][2]   = 1.;

    B1[0][0] = 1 - v*w;  for (PetscInt j = 0; j < v; ++j) { B1[0][1+j] = w; }
    B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
    B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
    if (v > 5) { B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125; }
    if (v > 4) { B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125; }
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

    for (PetscInt i = 0; i < 12; ++i) {
      for (PetscInt j = 0; j < nen; ++j) {
        PetscReal sum = 0.0;
        for (PetscInt m = 0; m < v + 12; ++m) {
          sum += B2[i][m] * B1[m][j];
        }
        B3[i][j] = sum;
      }
    }

    PetscReal INa0[16], INa1[16], INab0[16], INab1[16], INab2[16];
    for (PetscInt j = 0; j < nen; ++j) {
      PetscReal s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
      for (PetscInt i = 0; i < 12; ++i) {
        s1 += B3[i][j] * Na_center_dev_[0][i];
        s2 += B3[i][j] * Na_center_dev_[1][i];
        s3 += B3[i][j] * Nab_center_dev_[0][i];
        s4 += B3[i][j] * Nab_center_dev_[1][i];
        s5 += B3[i][j] * Nab_center_dev_[2][i];
      }
      INa0[j] = -2.0 * s1;
      INa1[j] = -2.0 * s2;
      INab0[j] = 4.0 * s3;
      INab1[j] = 4.0 * s4;
      INab2[j] = 4.0 * s5;
    }

    Cmpnts Aaa = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts Abb = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts Aab = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts ndx21 = MakeCmpnts_(0.0, 0.0, 0.0);
    Cmpnts ndx31 = MakeCmpnts_(0.0, 0.0, 0.0);

    for (PetscInt i = 0; i < nen; ++i) {
      Aaa.x += INab0[i] * x[i]; Aaa.y += INab0[i] * y[i]; Aaa.z += INab0[i] * z[i];
      Abb.x += INab1[i] * x[i]; Abb.y += INab1[i] * y[i]; Abb.z += INab1[i] * z[i];
      Aab.x += INab2[i] * x[i]; Aab.y += INab2[i] * y[i]; Aab.z += INab2[i] * z[i];

      ndx21.x += INa0[i] * x[i]; ndx21.y += INa0[i] * y[i]; ndx21.z += INa0[i] * z[i];
      ndx31.x += INa1[i] * x[i]; ndx31.y += INa1[i] * y[i]; ndx31.z += INa1[i] * z[i];

      irregular_out[irregular_base + 0 * kCudaPatchWidth_ + i] = INa0[i];
      irregular_out[irregular_base + 1 * kCudaPatchWidth_ + i] = INa1[i];
      irregular_out[irregular_base + 2 * kCudaPatchWidth_ + i] = INab0[i];
      irregular_out[irregular_base + 3 * kCudaPatchWidth_ + i] = INab1[i];
      irregular_out[irregular_base + 4 * kCudaPatchWidth_ + i] = INab2[i];
    }

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

    if (ec < 3) {
      printf("[CUDA geom kernel] ec=%d irregular v=%d nen=%d ndx21=(%.6e, %.6e, %.6e)\n",
             (int)ec, (int)v, (int)nen,
             (double)ndx21.x, (double)ndx21.y, (double)ndx21.z);
    }
    return;
  }

  status_out[ec] = 1;
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
  PetscInt *d_nv1 = NULL, *d_nv2 = NULL, *d_nv3 = NULL;
  PetscReal *d_x_bp0 = NULL, *d_y_bp0 = NULL, *d_z_bp0 = NULL, *d_element_coords_out = NULL;
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

  PetscPrintf(PETSC_COMM_SELF,
              "[CUDA host] n_elmt=%d max_node_index=%d coord_count=%d\n",
              (int)n_elmt,
              (int)max_node_index,
              (int)coord_count);

  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_nv1), conn_bytes), "cudaMalloc(d_nv1)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_nv2), conn_bytes), "cudaMalloc(d_nv2)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_nv3), conn_bytes), "cudaMalloc(d_nv3)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_x_bp0), coord_bytes), "cudaMalloc(d_x_bp0)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_y_bp0), coord_bytes), "cudaMalloc(d_y_bp0)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_z_bp0), coord_bytes), "cudaMalloc(d_z_bp0)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_element_coords_out), elem_coord_bytes),
                       "cudaMalloc(d_element_coords_out)"));

  PetscCall(CudaCheck_(cudaMemcpy(d_nv1, nv1, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv1)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_nv2, nv2, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv2)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_nv3, nv3, conn_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nv3)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_x_bp0, x_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(x_bp0)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_y_bp0, y_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(y_bp0)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_z_bp0, z_bp0, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(z_bp0)"));

  if (n_elmt > 0) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((n_elmt + threads_per_block - 1) / threads_per_block);
    ElementCoordCopyKernel_<<<blocks, threads_per_block>>>(n_elmt,
                                                           d_nv1,
                                                           d_nv2,
                                                           d_nv3,
                                                           d_x_bp0,
                                                           d_y_bp0,
                                                           d_z_bp0,
                                                           d_element_coords_out);
    PetscCall(CudaCheck_(cudaGetLastError(), "kernel launch"));
    PetscCall(CudaCheck_(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
    PetscCall(CudaCheck_(cudaMemcpy(element_coords_out, d_element_coords_out, elem_coord_bytes,
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

  cudaFree(d_nv1);
  cudaFree(d_nv2);
  cudaFree(d_nv3);
  cudaFree(d_x_bp0);
  cudaFree(d_y_bp0);
  cudaFree(d_z_bp0);
  cudaFree(d_element_coords_out);

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
  PetscInt *d_ire = NULL, *d_val = NULL, *d_patch = NULL;
  PetscInt *d_meta_out = NULL, *d_status_out = NULL;
  PetscReal *d_xb = NULL, *d_yb = NULL, *d_zb = NULL;
  PetscReal *d_geom_out = NULL, *d_irregular_out = NULL;
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

  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_ire), elem_bytes), "cudaMalloc(d_ire)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_val), elem_bytes), "cudaMalloc(d_val)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_patch), patch_bytes), "cudaMalloc(d_patch)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_xb), coord_bytes), "cudaMalloc(d_xb)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_yb), coord_bytes), "cudaMalloc(d_yb)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_zb), coord_bytes), "cudaMalloc(d_zb)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_meta_out), meta_bytes), "cudaMalloc(d_meta_out)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_status_out), elem_bytes), "cudaMalloc(d_status_out)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_geom_out), geom_bytes), "cudaMalloc(d_geom_out)"));
  PetscCall(CudaCheck_(cudaMalloc(reinterpret_cast<void **>(&d_irregular_out), irregular_bytes),
                       "cudaMalloc(d_irregular_out)"));

  PetscCall(CudaCheck_(cudaMemcpy(d_ire, ire, elem_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ire)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_val, val, elem_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(val)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_patch, patch, patch_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(patch)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_xb, xb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(xb)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_yb, yb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(yb)"));
  PetscCall(CudaCheck_(cudaMemcpy(d_zb, zb, coord_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(zb)"));

  {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((n_elmt + threads_per_block - 1) / threads_per_block);
    SubdivGeomKernel_<<<blocks, threads_per_block>>>(n_elmt,
                                                     d_ire,
                                                     d_val,
                                                     d_patch,
                                                     d_xb,
                                                     d_yb,
                                                     d_zb,
                                                     d_meta_out,
                                                     d_status_out,
                                                     d_geom_out,
                                                     d_irregular_out);
    PetscCall(CudaCheck_(cudaGetLastError(), "SubdivGeomKernel_ launch"));
    PetscCall(CudaCheck_(cudaDeviceSynchronize(), "cudaDeviceSynchronize(SubdivGeomKernel_)"));
  }

  PetscCall(CudaCheck_(cudaMemcpy(meta_out, d_meta_out, meta_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(meta_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(status_out, d_status_out, elem_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(status_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(geom_out, d_geom_out, geom_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(geom_out)"));
  PetscCall(CudaCheck_(cudaMemcpy(irregular_out, d_irregular_out, irregular_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(irregular_out)"));

  cudaFree(d_ire);
  cudaFree(d_val);
  cudaFree(d_patch);
  cudaFree(d_xb);
  cudaFree(d_yb);
  cudaFree(d_zb);
  cudaFree(d_meta_out);
  cudaFree(d_status_out);
  cudaFree(d_geom_out);
  cudaFree(d_irregular_out);

  PetscFunctionReturn(0);
}
