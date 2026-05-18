#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kokkos_bridge.h"

static constexpr PetscInt kKokkosPatchWidth_ = 16;
static constexpr PetscInt kKokkosRegularNen_ = 12;
static constexpr PetscInt kKokkosGeomStride_ = 24;
static constexpr PetscInt kKokkosIrregularStride_ = 5 * kKokkosPatchWidth_;

using KokkosExecSpace_ = Kokkos::DefaultExecutionSpace;
using KokkosMemSpace_ = typename KokkosExecSpace_::memory_space;
using KokkosIntView_ = Kokkos::View<PetscInt *, KokkosMemSpace_>;
using KokkosRealView_ = Kokkos::View<PetscReal *, KokkosMemSpace_>;
using KokkosHostIntView_ = Kokkos::View<const PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using KokkosHostRealView_ = Kokkos::View<const PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct KokkosGeomWorkspace_ {
  FE *owner = nullptr;
  IBMNodes *ibm = nullptr;
  PetscInt n_elmt = 0;
  PetscInt max_node_index = -1;
  bool topology_ready = false;
  bool reference_coords_ready = false;

  KokkosIntView_ ire_d;
  KokkosIntView_ val_d;
  KokkosIntView_ patch_d;
  KokkosRealView_ x_d;
  KokkosRealView_ y_d;
  KokkosRealView_ z_d;
  KokkosRealView_ x0_d;
  KokkosRealView_ y0_d;
  KokkosRealView_ z0_d;
  KokkosIntView_ meta_d;
  KokkosIntView_ status_d;
  KokkosRealView_ geom_d;
  KokkosRealView_ irregular_d;

  std::vector<PetscInt> meta_h;
  std::vector<PetscInt> status_h;
  std::vector<PetscReal> geom_h;
  std::vector<PetscReal> irregular_h;
};

static std::vector<KokkosGeomWorkspace_> kokkos_geom_workspaces_;

static bool KokkosGeomTimingEnabled_()
{
  static int enabled = -1;
  if (enabled < 0) {
    const char *value = std::getenv("KOKKOS_GEOM_TIMING");
    enabled = (value && std::strcmp(value, "0") != 0) ? 1 : 0;
  }
  return enabled != 0;
}

struct KokkosCmpnts_ {
  PetscReal x, y, z;
};

KOKKOS_INLINE_FUNCTION
static KokkosCmpnts_ MakeCmpnts_(PetscReal x, PetscReal y, PetscReal z)
{
  KokkosCmpnts_ out = {x, y, z};
  return out;
}

KOKKOS_INLINE_FUNCTION
static KokkosCmpnts_ Cross_(const KokkosCmpnts_ a, const KokkosCmpnts_ b)
{
  return MakeCmpnts_(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

KOKKOS_INLINE_FUNCTION
static PetscReal Dot_(const KokkosCmpnts_ a, const KokkosCmpnts_ b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

KOKKOS_INLINE_FUNCTION
static PetscReal Norm_(const KokkosCmpnts_ a)
{
  return Kokkos::sqrt(Dot_(a, a));
}

KOKKOS_INLINE_FUNCTION
static KokkosCmpnts_ Scale_(const PetscReal s, const KokkosCmpnts_ a)
{
  return MakeCmpnts_(s * a.x, s * a.y, s * a.z);
}

KOKKOS_INLINE_FUNCTION
static void StoreCmpnts_(const Kokkos::View<PetscReal *> geom, const PetscInt offset, const KokkosCmpnts_ v)
{
  geom(offset + 0) = v.x;
  geom(offset + 1) = v.y;
  geom(offset + 2) = v.z;
}

KOKKOS_INLINE_FUNCTION
static PetscReal GeomCoord_(const PetscInt scalar, const PetscReal x, const PetscReal y, const PetscReal z)
{
  const PetscInt component = scalar % 3;
  return (component == 0) ? x : ((component == 1) ? y : z);
}

KOKKOS_INLINE_FUNCTION
static void FinishGeom_(const PetscInt ec,
                        const PetscInt meta_base,
                        const PetscInt geom_base,
                        const PetscInt is_irregular,
                        const PetscInt v,
                        const PetscInt nen,
                        const PetscReal geom_sum[15],
                        const Kokkos::View<PetscInt *> meta_out,
                        const Kokkos::View<PetscInt *> status_out,
                        const Kokkos::View<PetscReal *> geom_out)
{
  const KokkosCmpnts_ Aaa = MakeCmpnts_(geom_sum[0], geom_sum[1], geom_sum[2]);
  const KokkosCmpnts_ Abb = MakeCmpnts_(geom_sum[3], geom_sum[4], geom_sum[5]);
  const KokkosCmpnts_ Aab = MakeCmpnts_(geom_sum[6], geom_sum[7], geom_sum[8]);
  const KokkosCmpnts_ ndx21 = MakeCmpnts_(geom_sum[9], geom_sum[10], geom_sum[11]);
  const KokkosCmpnts_ ndx31 = MakeCmpnts_(geom_sum[12], geom_sum[13], geom_sum[14]);

  const KokkosCmpnts_ nn_cross = Cross_(ndx21, ndx31);
  const PetscReal nn_norm = Norm_(nn_cross);
  if (nn_norm <= 0.0) {
    status_out(ec) = 4;
    return;
  }
  const KokkosCmpnts_ nn = Scale_(1.0 / nn_norm, nn_cross);

  KokkosCmpnts_ gc1 = Cross_(ndx31, nn);
  const PetscReal gc1_denom = Dot_(ndx21, gc1);
  if (Kokkos::abs(gc1_denom) <= 1.0e-30) {
    status_out(ec) = 5;
    return;
  }
  gc1 = Scale_(1.0 / gc1_denom, gc1);

  KokkosCmpnts_ gc2 = Cross_(nn, ndx21);
  const PetscReal gc2_denom = Dot_(ndx31, gc2);
  if (Kokkos::abs(gc2_denom) <= 1.0e-30) {
    status_out(ec) = 6;
    return;
  }
  gc2 = Scale_(1.0 / gc2_denom, gc2);

  meta_out(meta_base + 0) = is_irregular;
  meta_out(meta_base + 1) = v;
  meta_out(meta_base + 2) = nen;

  StoreCmpnts_(geom_out, geom_base + 0, ndx21);
  StoreCmpnts_(geom_out, geom_base + 3, ndx31);
  StoreCmpnts_(geom_out, geom_base + 6, nn);
  StoreCmpnts_(geom_out, geom_base + 9, gc1);
  StoreCmpnts_(geom_out, geom_base + 12, gc2);
  StoreCmpnts_(geom_out, geom_base + 15, Aaa);
  StoreCmpnts_(geom_out, geom_base + 18, Abb);
	  StoreCmpnts_(geom_out, geom_base + 21, Aab);
}

static KokkosGeomWorkspace_ *FindKokkosGeomWorkspace_(FE *fem, IBMNodes *ibm)
{
  for (auto &workspace : kokkos_geom_workspaces_) {
    if (workspace.owner == fem && workspace.ibm == ibm) return &workspace;
  }

  kokkos_geom_workspaces_.push_back(KokkosGeomWorkspace_{});
  KokkosGeomWorkspace_ &workspace = kokkos_geom_workspaces_.back();
  workspace.owner = fem;
  workspace.ibm = ibm;
  return &workspace;
}

static PetscErrorCode PrepareKokkosGeomWorkspace_(KokkosGeomWorkspace_ *workspace, IBMNodes *ibm)
{
  PetscFunctionBegin;
  Kokkos::Profiling::ScopedRegion region("BioFEM::KokkosGeom::prepare_workspace");
  PetscCheck(workspace != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Kokkos geometry workspace must not be NULL");
  PetscCheck(ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");

  const PetscInt n_elmt = ibm->n_elmt;
  if (workspace->topology_ready && workspace->n_elmt == n_elmt) PetscFunctionReturn(0);

  PetscInt max_node_index = -1;
  for (PetscInt ec = 0; ec < n_elmt; ++ec) {
    const PetscInt nen = (ibm->ire[ec] == 0) ? kKokkosRegularNen_ : (ibm->val[ec] + 6);
    PetscCheck(nen <= kKokkosPatchWidth_, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "Element %" PetscInt_FMT " needs nen=%" PetscInt_FMT " but Kokkos path supports up to %d patch nodes",
               ec, nen, (int)kKokkosPatchWidth_);
    for (PetscInt i = 0; i < nen; ++i) {
      const PetscInt node = ibm->patch[kKokkosPatchWidth_ * ec + i];
      if (node != 1000000 && node > max_node_index) max_node_index = node;
    }
  }
  PetscCheck(max_node_index >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No valid patch nodes were found");

  workspace->n_elmt = n_elmt;
  workspace->max_node_index = max_node_index;
  workspace->topology_ready = true;
  workspace->reference_coords_ready = false;

  workspace->ire_d = KokkosIntView_("kokkos_ire", n_elmt);
  workspace->val_d = KokkosIntView_("kokkos_val", n_elmt);
  workspace->patch_d = KokkosIntView_("kokkos_patch", (size_t)n_elmt * kKokkosPatchWidth_);
  workspace->x_d = KokkosRealView_("kokkos_x", max_node_index + 1);
  workspace->y_d = KokkosRealView_("kokkos_y", max_node_index + 1);
  workspace->z_d = KokkosRealView_("kokkos_z", max_node_index + 1);
  workspace->x0_d = KokkosRealView_("kokkos_x0", max_node_index + 1);
  workspace->y0_d = KokkosRealView_("kokkos_y0", max_node_index + 1);
  workspace->z0_d = KokkosRealView_("kokkos_z0", max_node_index + 1);
  workspace->meta_d = KokkosIntView_("kokkos_geom_meta", (size_t)n_elmt * 3u);
  workspace->status_d = KokkosIntView_("kokkos_geom_status", n_elmt);
  workspace->geom_d = KokkosRealView_("kokkos_geom", (size_t)n_elmt * kKokkosGeomStride_);
  workspace->irregular_d = KokkosRealView_("kokkos_irregular", (size_t)n_elmt * kKokkosIrregularStride_);

  workspace->meta_h.resize((size_t)n_elmt * 3u);
  workspace->status_h.resize(n_elmt);
  workspace->geom_h.resize((size_t)n_elmt * kKokkosGeomStride_);
  workspace->irregular_h.resize((size_t)n_elmt * kKokkosIrregularStride_);

  Kokkos::deep_copy(workspace->ire_d, KokkosHostIntView_(ibm->ire, n_elmt));
  Kokkos::deep_copy(workspace->val_d, KokkosHostIntView_(ibm->val, n_elmt));
  Kokkos::deep_copy(workspace->patch_d, KokkosHostIntView_(ibm->patch, (size_t)n_elmt * kKokkosPatchWidth_));

  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode InitKokkosWorkspace(void)
{
  PetscFunctionBegin;
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
  }
  {
    static bool printed_execution_space = false;
    if (!printed_execution_space) {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      PetscPrintf(PETSC_COMM_WORLD,
                  "Kokkos default execution space: %s\n",
                  ExecSpace().name());
      printed_execution_space = true;
    }
  }
  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode DestroyKokkosWorkspace(void)
{
  PetscFunctionBegin;
  kokkos_geom_workspaces_.clear();
  if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::finalize();
  }
  PetscFunctionReturn(0);
}

extern "C" PetscErrorCode RunKokkosSubdivGeomKernel(FE *fem, PetscBool use_reference_coords)
{
  PetscFunctionBegin;
  PetscCheck(fem != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "FE pointer must not be NULL");
  PetscCheck(fem->ibm != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "IBM pointer must not be NULL");
  PetscCall(InitKokkosWorkspace());

  IBMNodes *ibm = fem->ibm;
  const PetscInt n_elmt = ibm->n_elmt;
  const PetscReal *xb = use_reference_coords ? ibm->x_bp0 : ibm->x_bp;
  const PetscReal *yb = use_reference_coords ? ibm->y_bp0 : ibm->y_bp;
  const PetscReal *zb = use_reference_coords ? ibm->z_bp0 : ibm->z_bp;

  PetscCheck(ibm->ire != NULL && ibm->val != NULL && ibm->patch != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Subdivision metadata arrays must not be NULL");
  PetscCheck(xb != NULL && yb != NULL && zb != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Coordinate arrays must not be NULL");

  if (n_elmt == 0) PetscFunctionReturn(0);

  KokkosGeomWorkspace_ *workspace = FindKokkosGeomWorkspace_(fem, ibm);
  PetscCall(PrepareKokkosGeomWorkspace_(workspace, ibm));

  if (use_reference_coords) {
    if (!workspace->reference_coords_ready) {
      Kokkos::deep_copy(workspace->x0_d, KokkosHostRealView_(xb, workspace->max_node_index + 1));
      Kokkos::deep_copy(workspace->y0_d, KokkosHostRealView_(yb, workspace->max_node_index + 1));
      Kokkos::deep_copy(workspace->z0_d, KokkosHostRealView_(zb, workspace->max_node_index + 1));
      workspace->reference_coords_ready = true;
    }
  } else {
    Kokkos::deep_copy(workspace->x_d, KokkosHostRealView_(xb, workspace->max_node_index + 1));
    Kokkos::deep_copy(workspace->y_d, KokkosHostRealView_(yb, workspace->max_node_index + 1));
    Kokkos::deep_copy(workspace->z_d, KokkosHostRealView_(zb, workspace->max_node_index + 1));
  }

  const KokkosIntView_ ire_d = workspace->ire_d;
  const KokkosIntView_ val_d = workspace->val_d;
  const KokkosIntView_ patch_d = workspace->patch_d;
  const KokkosRealView_ x_d = use_reference_coords ? workspace->x0_d : workspace->x_d;
  const KokkosRealView_ y_d = use_reference_coords ? workspace->y0_d : workspace->y_d;
  const KokkosRealView_ z_d = use_reference_coords ? workspace->z0_d : workspace->z_d;
  const KokkosIntView_ meta_d = workspace->meta_d;
  const KokkosIntView_ status_d = workspace->status_d;
  const KokkosRealView_ geom_d = workspace->geom_d;
  const KokkosRealView_ irregular_d = workspace->irregular_d;

  Kokkos::deep_copy(status_d, (PetscInt)0);

  Kokkos::parallel_for("SubdivGeomKokkos",
                       Kokkos::RangePolicy<KokkosExecSpace_>(0, n_elmt),
                       KOKKOS_LAMBDA(const PetscInt ec) {
    constexpr PetscReal Na[2][12] = {
      {-0.0247, -0.0309,  0.0000, -0.4815, -0.1852,  0.0247,
        0.4815,  0.0000, -0.0062,  0.0309,  0.1852,  0.0062},
      {-0.0309, -0.0247, -0.1852, -0.4815,  0.0000, -0.0062,
        0.0000,  0.4815,  0.0247,  0.0062,  0.1852,  0.0309}
    };
    constexpr PetscReal Nab[3][12] = {
      { 0.1111,  0.2222, -0.2222, -0.2222,  0.4444,  0.1111,
       -0.2222, -0.8889,  0.0000,  0.2222,  0.4444,  0.0000},
      { 0.2222,  0.1111,  0.4444, -0.2222, -0.2222,  0.0000,
       -0.8889, -0.2222,  0.1111,  0.0000,  0.4444,  0.2222},
      { 0.1667,  0.1667, -0.1111,  0.2222, -0.1111, -0.0556,
       -0.4444, -0.4444, -0.0556,  0.0556,  0.5556,  0.0556}
    };

    const PetscInt meta_base = 3 * ec;
    const PetscInt geom_base = kKokkosGeomStride_ * ec;
    const PetscInt irregular_base = kKokkosIrregularStride_ * ec;
    const PetscInt patch_base = kKokkosPatchWidth_ * ec;
    PetscReal geom_sum[15];
    for (PetscInt i = 0; i < 15; ++i) geom_sum[i] = 0.0;

    if (ire_d(ec) == 0) {
      for (PetscInt i = 0; i < kKokkosRegularNen_; ++i) {
        const PetscInt node = patch_d(patch_base + i);
        if (node == 1000000) {
          status_d(ec) = 2;
          return;
        }
        const PetscReal x = x_d(node);
        const PetscReal y = y_d(node);
        const PetscReal z = z_d(node);
        const PetscReal coeffs[5] = {Nab[0][i], Nab[1][i], Nab[2][i], Na[0][i], Na[1][i]};
        for (PetscInt group = 0; group < 5; ++group) {
          const PetscReal c = coeffs[group];
          geom_sum[3 * group + 0] += c * x;
          geom_sum[3 * group + 1] += c * y;
          geom_sum[3 * group + 2] += c * z;
        }
      }
      FinishGeom_(ec, meta_base, geom_base, 0, 0, kKokkosRegularNen_, geom_sum,
                  meta_d, status_d, geom_d);
      return;
    }

    if (ire_d(ec) == 1) {
      const PetscInt v = val_d(ec);
      const PetscInt nen = v + 6;
      if (nen > kKokkosPatchWidth_ || v + 12 > 22) {
        status_d(ec) = 3;
        return;
      }

      PetscReal coord[3][kKokkosPatchWidth_];
      for (PetscInt i = 0; i < nen; ++i) {
        const PetscInt node = patch_d(patch_base + i);
        if (node == 1000000) {
          status_d(ec) = 2;
          return;
        }
        coord[0][i] = x_d(node);
        coord[1][i] = y_d(node);
        coord[2][i] = z_d(node);
      }

      PetscReal B1[22][kKokkosPatchWidth_];
      for (PetscInt i = 0; i < 22; ++i)
        for (PetscInt j = 0; j < kKokkosPatchWidth_; ++j)
          B1[i][j] = 0.0;

      const PetscReal angle = 2.0 * PETSC_PI / (PetscReal)v;
      const PetscReal c = Kokkos::cos(angle);
      const PetscReal w = (1.0 / (PetscReal)v) * (0.625 - Kokkos::pow(0.375 + 0.25 * c, 2.0));

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

      const PetscInt rows[12] = {v+9, v+6, v+4, v+1, v+2, v+5, v, 1, v+3, v-1, 0, 2};
      PetscReal IN[5][kKokkosPatchWidth_];
      for (PetscInt which = 0; which < 5; ++which) {
        for (PetscInt j = 0; j < nen; ++j) {
          PetscReal sum = 0.0;
          for (PetscInt i = 0; i < 12; ++i) {
            const PetscReal b3 = B1[rows[i]][j];
            if (which == 0) sum += b3 * Na[0][i];
            else if (which == 1) sum += b3 * Na[1][i];
            else if (which == 2) sum += b3 * Nab[0][i];
            else if (which == 3) sum += b3 * Nab[1][i];
            else sum += b3 * Nab[2][i];
          }
          IN[which][j] = ((which < 2) ? -2.0 : 4.0) * sum;
          irregular_d(irregular_base + which * kKokkosPatchWidth_ + j) = IN[which][j];
        }
      }

      for (PetscInt scalar = 0; scalar < 15; ++scalar) {
        const PetscInt in_index = (scalar < 3) ? 2 : ((scalar < 6) ? 3 : ((scalar < 9) ? 4 : ((scalar < 12) ? 0 : 1)));
        for (PetscInt i = 0; i < nen; ++i) {
          geom_sum[scalar] += IN[in_index][i] *
                              GeomCoord_(scalar, coord[0][i], coord[1][i], coord[2][i]);
        }
      }

      FinishGeom_(ec, meta_base, geom_base, 1, v, nen, geom_sum, meta_d, status_d, geom_d);
      return;
    }

    status_d(ec) = 1;
  });
  Kokkos::fence();

  std::vector<PetscInt> &meta = workspace->meta_h;
  std::vector<PetscInt> &status = workspace->status_h;
  std::vector<PetscReal> &geom = workspace->geom_h;
  std::vector<PetscReal> &irregular = workspace->irregular_h;
  Kokkos::deep_copy(Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(meta.data(), meta.size()), meta_d);
  Kokkos::deep_copy(Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(status.data(), status.size()), status_d);
  Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(geom.data(), geom.size()), geom_d);
  Kokkos::deep_copy(Kokkos::View<PetscReal *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(irregular.data(), irregular.size()), irregular_d);

  auto unpack = [](const PetscReal *src) {
    Cmpnts out;
    out.x = src[0];
    out.y = src[1];
    out.z = src[2];
    return out;
  };

  for (PetscInt ec = 0; ec < n_elmt; ++ec) {
    PetscCheck(status[ec] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Kokkos subdivision geometry failed for element %" PetscInt_FMT " with status %" PetscInt_FMT,
               ec, status[ec]);
    ElemActData *ead = &fem->act_data.elem_act_data[ec];
    PetscCheck(ead != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
               "Element %" PetscInt_FMT " act data must not be NULL", ec);

    const size_t meta_base = (size_t)ec * 3u;
	    const size_t geom_base = (size_t)ec * kKokkosGeomStride_;
	    const size_t irregular_base = (size_t)ec * kKokkosIrregularStride_;
	    SubdivGeomQP *target_geom = use_reference_coords ? &ead->geom0[0] : &ead->geom[0];
	    const PetscInt previous_is_irregular = target_geom->is_irregular;
	    const PetscInt previous_nen = target_geom->nen;
	    const PetscBool can_reuse_irregular_arrays =
	      (PetscBool)(previous_is_irregular && previous_nen == meta[meta_base + 2] &&
	                  target_geom->INa0 && target_geom->INa1 && target_geom->INab0 &&
	                  target_geom->INab1 && target_geom->INab2);

	    target_geom->is_irregular = meta[meta_base + 0];
	    target_geom->v = meta[meta_base + 1];
	    target_geom->nen = meta[meta_base + 2];
    target_geom->ndx21 = unpack(&geom[geom_base + 0]);
    target_geom->ndx31 = unpack(&geom[geom_base + 3]);
    target_geom->nn = unpack(&geom[geom_base + 6]);
    target_geom->gc1 = unpack(&geom[geom_base + 9]);
    target_geom->gc2 = unpack(&geom[geom_base + 12]);
    target_geom->Aaa = unpack(&geom[geom_base + 15]);
    target_geom->Abb = unpack(&geom[geom_base + 18]);
    target_geom->Aab = unpack(&geom[geom_base + 21]);

	    if (target_geom->is_irregular) {
	      const PetscInt nen = target_geom->nen;
	      if (!can_reuse_irregular_arrays) {
	        PetscCall(PetscFree(target_geom->INa0));
	        PetscCall(PetscFree(target_geom->INa1));
	        PetscCall(PetscFree(target_geom->INab0));
	        PetscCall(PetscFree(target_geom->INab1));
	        PetscCall(PetscFree(target_geom->INab2));
	        PetscCall(PetscMalloc1(nen, &target_geom->INa0));
	        PetscCall(PetscMalloc1(nen, &target_geom->INa1));
	        PetscCall(PetscMalloc1(nen, &target_geom->INab0));
	        PetscCall(PetscMalloc1(nen, &target_geom->INab1));
	        PetscCall(PetscMalloc1(nen, &target_geom->INab2));
	      }
	      PetscCall(PetscMemcpy(target_geom->INa0, &irregular[irregular_base + 0 * kKokkosPatchWidth_], (size_t)nen * sizeof(PetscReal)));
	      PetscCall(PetscMemcpy(target_geom->INa1, &irregular[irregular_base + 1 * kKokkosPatchWidth_], (size_t)nen * sizeof(PetscReal)));
	      PetscCall(PetscMemcpy(target_geom->INab0, &irregular[irregular_base + 2 * kKokkosPatchWidth_], (size_t)nen * sizeof(PetscReal)));
	      PetscCall(PetscMemcpy(target_geom->INab1, &irregular[irregular_base + 3 * kKokkosPatchWidth_], (size_t)nen * sizeof(PetscReal)));
	      PetscCall(PetscMemcpy(target_geom->INab2, &irregular[irregular_base + 4 * kKokkosPatchWidth_], (size_t)nen * sizeof(PetscReal)));
	    } else if (previous_is_irregular || target_geom->INa0 || target_geom->INa1 ||
	               target_geom->INab0 || target_geom->INab1 || target_geom->INab2) {
	      PetscCall(PetscFree(target_geom->INa0));
	      PetscCall(PetscFree(target_geom->INa1));
	      PetscCall(PetscFree(target_geom->INab0));
      PetscCall(PetscFree(target_geom->INab1));
      PetscCall(PetscFree(target_geom->INab2));
    }
  }

  PetscFunctionReturn(0);
}
