/**
 * @file dmplex_geom.c
 * @brief Distributed geometry kernel for Loop subdivision surfaces using PETSc DMPlex.
 *
 * @details
 * This module implements a parallel geometry computation pipeline for triangulated
 * immersed-boundary surfaces discretized with Loop subdivision stencils.  The mesh
 * topology is distributed across MPI ranks with PETSc's DMPlex infrastructure; patch
 * stencils that cross partition boundaries are serviced by a PetscSF coordinate
 * broadcast set up once at initialization.
 *
 * <b>Terminology</b>
 *
 * - <em>Original index</em>: the vertex or cell index as it appears in the serial
 *   IBMNodes arrays (x_bp, y_bp, z_bp, nv1/nv2/nv3, patch, …).  These indices are
 *   stable across the full run and are stored in DMLabels so they survive distribution.
 *
 * - <em>DM local point index</em>: the index used inside a distributed DM's chart
 *   [pStart, pEnd).  Cells, edges, and vertices share one flat namespace.  These
 *   indices are local to each rank and bear no relation to original indices without
 *   consulting the label.
 *
 * - <em>Star Forest (PetscSF)</em>: PETSc's abstraction for one-to-many MPI patterns.
 *   A graph of <em>roots</em> (authoritative data on the owning rank) and
 *   <em>leaves</em> (remote copies).  PetscSFBcast sends root → leaves;
 *   PetscSFReduce accumulates leaves → root.  The SF built here has nroots = ibm->n_v
 *   (the full original vertex array) and nleaves = nRemoteNodes (cross-rank vertices
 *   needed by this rank's patch stencils).
 *
 * <b>One-time setup pipeline (RunDMPlexGeomProcesses)</b>
 * @code
 *   CreateDistributedDMPlex_()        // distribute mesh topology
 *   BuildOriginalVertexOwnerTable_()  // build ownerRank[0..n_v)
 *   DMPlexPatchLayoutCreate_()        // copy patch stencils to local layout
 *   BuildPatchCoordinateExchange_()   // build SF for cross-rank coord fetch
 * @endcode
 *
 * <b>Per-iteration kernel</b>
 * @code
 *   UpdatePatchRemoteCoordinates_()   // PetscSFBcast + write-back to ibm->x/y/z_bp
 *   RunLocalGeometry_()               // evaluate subdivision geometry locally
 * @endcode
 */

#include <petscdmplex.h>
#include <petscsf.h>
#include <petsclog.h>
#include <string.h>
#include "dmplex_geom.h"
#include "active_strain.h"
#include "cuda_bridge.h"

/* PetscLogEvent handles for -log_view profiling of the geometry-kernel path.
 * Split into "exchange" (PetscSF coordinate broadcast, i.e. cross-rank
 * communication) and "compute" (per-cell geometry evaluation) so -log_view
 * output can separate communication cost from compute cost for the MPI
 * scaling study, without changing any of the existing MPI_Wtime timing. */
static PetscLogEvent gDMPlexGeomExchangeEvent_ = 0;
static PetscLogEvent gDMPlexGeomComputeEvent_  = 0;
static PetscBool     gDMPlexGeomEventsRegistered_ = PETSC_FALSE;

static PetscErrorCode EnsureDMPlexGeomLogEvents_(void)
{
  PetscFunctionBegin;
  if (!gDMPlexGeomEventsRegistered_) {
    PetscCall(PetscLogEventRegister("DMPlexGeomExchange", 0, &gDMPlexGeomExchangeEvent_));
    PetscCall(PetscLogEventRegister("DMPlexGeomCompute", 0, &gDMPlexGeomComputeEvent_));
    gDMPlexGeomEventsRegistered_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/** Sentinel stored in DMPlexPatchLayout::patch to mark a missing stencil slot. */
#define DMPLEX_GEOM_PATCH_MISSING 1000000

extern PetscReal h0;
extern PetscInt  dof;

/* DMPlexPatchLayout, DMPlexPatchExchange, DMPlexGeomCtx are defined in
   dmplex_types.h (included transitively via variables.h → dmplex_geom.h). */

/* -------------------------------------------------------------------------
 * Destructor helpers
 * ------------------------------------------------------------------------- */

/**
 * @brief Free all heap memory owned by a DMPlexPatchLayout and zero the struct.
 * @param[in,out] layout Layout to destroy.  Tolerates a NULL pointer.
 */
static PetscErrorCode DMPlexPatchLayoutDestroy_(DMPlexPatchLayout *layout)
{
  PetscFunctionBeginUser;
  if (!layout) PetscFunctionReturn(0);
  PetscCall(PetscFree(layout->orig_cell));
  PetscCall(PetscFree(layout->ire));
  PetscCall(PetscFree(layout->irv));
  PetscCall(PetscFree(layout->val));
  PetscCall(PetscFree(layout->patch));
  PetscCall(DMDestroy(&layout->dm));
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  PetscFunctionReturn(0);
}

/**
 * @brief Free all heap memory owned by a DMPlexPatchExchange and zero the struct.
 * @param[in,out] exchange Exchange to destroy.  Tolerates a NULL pointer.
 */
static PetscErrorCode DMPlexPatchExchangeDestroy_(DMPlexPatchExchange *exchange)
{
  PetscFunctionBeginUser;
  if (!exchange) PetscFunctionReturn(0);
  PetscCall(PetscSFDestroy(&exchange->sf));
  PetscCall(PetscFree(exchange->orig_node));
  PetscCall(PetscFree(exchange->owner_rank));
  PetscCall(PetscFree(exchange->owner_index));
  PetscCall(PetscFree(exchange->x_remote));
  PetscCall(PetscFree(exchange->y_remote));
  PetscCall(PetscFree(exchange->z_remote));
  PetscCall(PetscMemzero(exchange, sizeof(*exchange)));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------
 * Setup pipeline
 * ------------------------------------------------------------------------- */

/**
 * @brief Construct a serial DMPlex from the IBMNodes connectivity on rank 0.
 *
 * @details
 * Only rank 0 populates the cell and coordinate arrays; all other ranks pass
 * empty arrays.  DMPlexCreateFromCellListPetsc is called with
 * @c interpolate=PETSC_FALSE so no edges are created; only cells and vertices
 * are stored.  This avoids PETSc's O(N^{3+}) serial edge-interpolation path
 * that dominates cost for large meshes.
 *
 * @param[in]  ibm       IBM surface mesh (read on rank 0 only).
 * @param[out] dm        New serial DMPlex (caller must DMDestroy).
 * @param[out] cStartOut First cell point index in the serial DM chart.
 * @param[out] vStartOut First vertex point index in the serial DM chart.
 */
static PetscErrorCode BuildSerialDMPlex_(IBMNodes *ibm, DM *dm,
                                         PetscInt *cStartOut, PetscInt *vStartOut,
                                         MPI_Comm comm)
{
  PetscMPIInt rank;
  PetscInt    dim = 2, spaceDim = 3, corners = 3;
  PetscInt    nCells = 0, nVerts = 0;
  PetscInt   *cells = NULL;
  PetscReal  *coords = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    nCells = ibm->n_elmt;
    nVerts = ibm->n_v;
    PetscCall(PetscMalloc1(corners * nCells, &cells));
    PetscCall(PetscMalloc1(spaceDim * nVerts, &coords));
    for (PetscInt ec = 0; ec < nCells; ++ec) {
      cells[corners * ec + 0] = ibm->nv1[ec];
      cells[corners * ec + 1] = ibm->nv2[ec];
      cells[corners * ec + 2] = ibm->nv3[ec];
    }
    for (PetscInt v = 0; v < nVerts; ++v) {
      coords[spaceDim * v + 0] = ibm->x_bp[v];
      coords[spaceDim * v + 1] = ibm->y_bp[v];
      coords[spaceDim * v + 2] = ibm->z_bp[v];
    }
  }

  /* PETSC_FALSE: do NOT interpolate edges on the serial mesh.
     Edge interpolation is O(N^{3+}) for large meshes; we distribute first
     and let per-rank interpolation (if needed) happen on small subsets. */
  PetscCall(DMPlexCreateFromCellListPetsc(comm, dim, nCells, nVerts, corners,
                                          PETSC_FALSE, cells, spaceDim, coords, dm));
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree(coords));

  /* Record chart offsets on rank 0 (other ranks have empty DMs). */
  {
    PetscInt cS = 0, cE = 0, vS = 0, vE = 0;
    PetscCall(DMPlexGetHeightStratum(*dm, 0, &cS, &cE));
    PetscCall(DMPlexGetDepthStratum(*dm, 0, &vS, &vE));
    /* Broadcast from rank 0 so every rank knows the serial offsets. */
    PetscCallMPI(MPI_Bcast(&cS, 1, MPIU_INT, 0, comm));
    PetscCallMPI(MPI_Bcast(&vS, 1, MPIU_INT, 0, comm));
    *cStartOut = cS;
    *vStartOut = vS;
  }
  PetscFunctionReturn(0);
}

/**
 * @brief Distribute the serial DMPlex and return per-rank original-index arrays.
 *
 * @details
 * Builds a serial (uninterpolated) DMPlex on rank 0, distributes it with
 * overlap=0, and extracts per-rank @c origCell[] / @c origVert[] arrays directly
 * from the migration Star Forest.  No DMLabels are used: label stratification
 * was the previous O(N^2) bottleneck.
 *
 * After this call every rank holds:
 *   - A distributed DM (no ghost cells; overlap=0)
 *   - origCell[lc]: original IBMNodes cell index for local cell lc, or -1
 *   - origVert[lv]: original IBMNodes vertex index for local vertex lv, or -1
 *
 * @param[in]  ibm         IBM surface mesh.
 * @param[out] dmDist      Distributed DM (caller must DMDestroy).
 * @param[out] origCellOut Per-rank original cell index array (size cEnd-cStart).
 * @param[out] origVertOut Per-rank original vertex index array (size vEnd-vStart).
 */
static PetscErrorCode CreateDistributedDMPlex_(IBMNodes *ibm,
                                               DM *dmDist,
                                               PetscInt **origCellOut,
                                               PetscInt **origVertOut,
                                               MPI_Comm comm)
{
  DM      dm = NULL, distributed = NULL;
  PetscSF migrationSF = NULL;
  PetscInt serial_cStart = 0, serial_vStart = 0;
  double  t0, t_build, t_dist;

  PetscFunctionBeginUser;
  t0 = MPI_Wtime();
  PetscCall(BuildSerialDMPlex_(ibm, &dm, &serial_cStart, &serial_vStart, comm));
  t_build = MPI_Wtime() - t0;

  /* overlap=0: no ghost cells.  DMPlex with uninterpolated serial DM supports
     overlap=0 distribution without requiring pre-existing edge topology.      */
  t0 = MPI_Wtime();
  PetscCall(DMPlexDistribute(dm, 0, &migrationSF, &distributed));
  t_dist = MPI_Wtime() - t0;

  if (distributed) {
    PetscCall(DMDestroy(&dm));
    dm = distributed;
  }
  *dmDist = dm;

  /* Build origCell[] and origVert[] from the migration SF.
     migrationSF leaf i: local point ilocal[i] came from serial point iremote[i].index.
     For np=1, migrationSF is NULL and the local DM is the serial DM. */
  PetscInt cStart, cEnd, vStart, vEnd;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));

  PetscInt *origCell = NULL, *origVert = NULL;
  PetscCall(PetscMalloc1(PetscMax(1, cEnd - cStart), &origCell));
  PetscCall(PetscMalloc1(PetscMax(1, vEnd - vStart), &origVert));
  for (PetscInt lc = 0; lc < cEnd - cStart; ++lc) origCell[lc] = -1;
  for (PetscInt lv = 0; lv < vEnd - vStart; ++lv) origVert[lv] = -1;

  if (migrationSF) {
    PetscInt           nroots, nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscCall(PetscSFGetGraph(migrationSF, &nroots, &nleaves, &ilocal, &iremote));
    for (PetscInt i = 0; i < nleaves; ++i) {
      const PetscInt p    = ilocal ? ilocal[i] : i;
      const PetscInt orig = iremote[i].index;
      if (p >= cStart && p < cEnd) {
        origCell[p - cStart] = orig - serial_cStart;
      } else if (p >= vStart && p < vEnd) {
        origVert[p - vStart] = orig - serial_vStart;
      }
    }
    PetscCall(PetscSFDestroy(&migrationSF));
  } else {
    /* np=1: distributed == NULL, local DM = serial DM, identity mapping. */
    for (PetscInt lc = 0; lc < cEnd - cStart; ++lc) origCell[lc] = lc;
    for (PetscInt lv = 0; lv < vEnd - vStart; ++lv) origVert[lv] = lv;
  }

  *origCellOut = origCell;
  *origVertOut = origVert;

  PetscCall(PetscPrintf(comm,
                        "  CreateDistributedDMPlex breakdown:\n"
                        "    BuildSerialDMPlex (no interpolation) % .6e s\n"
                        "    DMPlexDistribute (overlap=0)         % .6e s\n",
                        t_build, t_dist));
  PetscFunctionReturn(0);
}

/**
 * @brief Build a global table mapping each original vertex index to its owning MPI rank.
 *
 * @details
 * Uses the pre-built @p origVert array (from the migration SF) and the DM's
 * point SF to determine which local vertices are ghost copies.  Non-ghost
 * vertices are owned by this rank; ghost vertices are owned by another.
 *
 *   1. Mark ghost vertices using DMGetPointSF (leaves = ghost copies).
 *   2. For each non-ghost local vertex, record localOwner[origVert[lv]] = myRank.
 *   3. MPI_Allreduce(MPI_MAX): the owning rank's entry beats every -1.
 *
 * @param[in]  dm            Distributed DM.
 * @param[in]  ibm           IBM mesh (ibm->n_v used as array size).
 * @param[in]  origVert      origVert[lv] = original vertex index for local vertex lv.
 *                           Size: vEnd - vStart.  -1 for vertices with no original.
 * @param[out] ownerRankOut  Allocated array of size ibm->n_v.  Caller must PetscFree.
 */
static PetscErrorCode BuildOriginalVertexOwnerTable_(DM dm, IBMNodes *ibm,
                                                     const PetscInt *origVert,
                                                     PetscMPIInt **ownerRankOut,
                                                     MPI_Comm comm)
{
  PetscMPIInt  rank;
  PetscInt     pStart, pEnd, vStart, vEnd;
  PetscBool   *isGhost = NULL;
  PetscMPIInt *localOwner = NULL;
  PetscMPIInt *ownerRank = NULL;
  PetscSF      pointSF = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscCalloc1(pEnd - pStart, &isGhost));

  /* Mark ghost points using the DM's point SF. */
  PetscCall(DMGetPointSF(dm, &pointSF));
  if (pointSF) {
    PetscInt           nroots, nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscCall(PetscSFGetGraph(pointSF, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves >= 0) {
      for (PetscInt i = 0; i < nleaves; ++i) {
        const PetscInt p = ilocal ? ilocal[i] : i;
        if (p >= pStart && p < pEnd) isGhost[p - pStart] = PETSC_TRUE;
      }
    }
  }

  PetscCall(PetscMalloc1(ibm->n_v, &localOwner));
  PetscCall(PetscMalloc1(ibm->n_v, &ownerRank));
  for (PetscInt i = 0; i < ibm->n_v; ++i) localOwner[i] = -1;

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt orig = origVert[v - vStart];
    if (orig >= 0 && orig < ibm->n_v && !isGhost[v - pStart]) {
      localOwner[orig] = rank;
    }
  }

  PetscCallMPI(MPI_Allreduce(localOwner, ownerRank, ibm->n_v, MPI_INT, MPI_MAX, comm));
  PetscCall(PetscFree(localOwner));
  PetscCall(PetscFree(isGhost));
  *ownerRankOut = ownerRank;
  PetscFunctionReturn(0);
}

/**
 * @brief Build the DMPlexPatchExchange SF for cross-rank coordinate fetching.
 *
 * @details
 * Each cell's Loop subdivision stencil references up to 16 vertices by original
 * index.  Some of those vertices are owned by other MPI ranks.  This function
 * determines exactly which remote vertices this rank needs, deduplicates them,
 * and builds a PetscSF so that UpdatePatchRemoteCoordinates_() can broadcast
 * updated coordinates each geometry iteration in O(nRemoteNodes) communication.
 *
 * <b>Complexity note</b>: A naive implementation would check locality by scanning
 * all local vertices for each patch node query (O(N^2/P) total).  Instead, two
 * boolean lookup arrays indexed by original vertex ID are built in one O(N/P) walk
 * before the loop:
 *   - @c isLocalOrig[n_v]: true iff original vertex i is a non-ghost vertex on
 *     this rank.  Replaces the former per-query linear scan.
 *   - @c alreadyAdded[n_v]: deduplication flag; set when a remote node is
 *     registered, replacing the former O(nRemoteNodes) linear scan.
 * The main loop therefore runs in O(nLocalCells * max_patch_size).
 *
 * <b>SF layout</b>:
 *   - nroots  = ibm->n_v (full original vertex array; each rank is a potential root)
 *   - nleaves = nRemoteNodes (vertices this rank fetches from others)
 *   - remote[i] = {ownerRank[orig_node[i]], orig_node[i]} (root address)
 *
 * @param[in]  layout    Local patch layout (provides dm, nLocalCells, patch array).
 * @param[in]  ibm       IBM mesh (n_v, coordinate arrays used as SF root data).
 * @param[in]  origVert  origVert[lv]: original IBMNodes vertex index for local vertex lv.
 * @param[in]  ownerRank Global owner table from BuildOriginalVertexOwnerTable_().
 * @param[out] exchange  Initialised exchange struct (caller must call
 *                       DMPlexPatchExchangeDestroy_ when done).
 */
static PetscErrorCode BuildPatchCoordinateExchange_(const DMPlexPatchLayout *layout,
                                                    IBMNodes *ibm,
                                                    const PetscInt *origVert,
                                                    const PetscMPIInt *ownerRank,
                                                    DMPlexPatchExchange *exchange,
                                                    MPI_Comm comm)
{
  PetscInt     maxNeeded;
  PetscSFNode *remote       = NULL;
  PetscBool   *isLocalOrig  = NULL;
  PetscBool   *alreadyAdded = NULL;
  PetscBool   *isGhost      = NULL;
  PetscSF      pointSF      = NULL;
  PetscInt     pStart, pEnd, vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(exchange, sizeof(*exchange)));

  /* Build isLocalOrig[n_v] from origVert[] + DMGetPointSF ghost marks.
     One O(nLocalVerts) walk; no DMLabel lookups. */
  PetscCall(DMPlexGetChart(layout->dm, &pStart, &pEnd));
  PetscCall(PetscCalloc1(pEnd - pStart, &isGhost));
  PetscCall(DMGetPointSF(layout->dm, &pointSF));
  if (pointSF) {
    PetscInt           nroots, nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscCall(PetscSFGetGraph(pointSF, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves >= 0) {
      for (PetscInt i = 0; i < nleaves; ++i) {
        const PetscInt p = ilocal ? ilocal[i] : i;
        if (p >= pStart && p < pEnd) isGhost[p - pStart] = PETSC_TRUE;
      }
    }
  }
  PetscCall(PetscCalloc1(ibm->n_v, &isLocalOrig));
  PetscCall(DMPlexGetDepthStratum(layout->dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt orig = origVert[v - vStart];
    if (orig >= 0 && orig < ibm->n_v && !isGhost[v - pStart]) isLocalOrig[orig] = PETSC_TRUE;
  }
  PetscCall(PetscFree(isGhost));

  /* alreadyAdded[n_v]: dedup flag replacing O(nRemoteNodes) IntArrayContains_ scan. */
  PetscCall(PetscCalloc1(ibm->n_v, &alreadyAdded));

  maxNeeded = 16 * layout->nLocalCells;
  PetscCall(PetscMalloc1(PetscMax(1, maxNeeded), &exchange->orig_node));
  PetscCall(PetscMalloc1(PetscMax(1, maxNeeded), &exchange->owner_rank));
  PetscCall(PetscMalloc1(PetscMax(1, maxNeeded), &exchange->owner_index));

  for (PetscInt lc = 0; lc < layout->nLocalCells; ++lc) {
    const PetscInt nen = (layout->ire[lc] == 0) ? 12 : layout->val[lc] + 6;
    for (PetscInt i = 0; i < nen; ++i) {
      const PetscInt node = layout->patch[16 * lc + i];
      if (node == DMPLEX_GEOM_PATCH_MISSING) continue;
      /* Ghost/auxiliary nodes (>= n_v) live in the local ibm arrays on every rank. */
      if (node >= ibm->n_v) continue;
      if (isLocalOrig[node]) continue;
      if (alreadyAdded[node]) continue;

      PetscCheck(node >= 0 && node < ibm->n_v, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                 "Patch node %" PetscInt_FMT " is outside vertex range [0,%" PetscInt_FMT ")",
                 node, ibm->n_v);
      PetscCheck(ownerRank[node] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Could not find owning rank for patch node %" PetscInt_FMT, node);

      alreadyAdded[node] = PETSC_TRUE;
      exchange->orig_node[exchange->nRemoteNodes] = node;
      exchange->owner_rank[exchange->nRemoteNodes] = ownerRank[node];
      exchange->owner_index[exchange->nRemoteNodes] = node;
      ++exchange->nRemoteNodes;
    }
  }

  PetscCall(PetscFree(isLocalOrig));
  PetscCall(PetscFree(alreadyAdded));

  PetscCall(PetscMalloc1(PetscMax(1, exchange->nRemoteNodes), &exchange->x_remote));
  PetscCall(PetscMalloc1(PetscMax(1, exchange->nRemoteNodes), &exchange->y_remote));
  PetscCall(PetscMalloc1(PetscMax(1, exchange->nRemoteNodes), &exchange->z_remote));

  PetscCall(PetscMalloc1(PetscMax(1, exchange->nRemoteNodes), &remote));
  for (PetscInt i = 0; i < exchange->nRemoteNodes; ++i) {
    remote[i].rank  = exchange->owner_rank[i];
    remote[i].index = exchange->owner_index[i];
  }

  PetscCall(PetscSFCreate(comm, &exchange->sf));
  PetscCall(PetscSFSetGraph(exchange->sf, ibm->n_v, exchange->nRemoteNodes,
                            NULL, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(exchange->sf));
  PetscCall(PetscFree(remote));

  {
    PetscInt totalRemote = 0;
    PetscCallMPI(MPIU_Allreduce(&exchange->nRemoteNodes, &totalRemote, 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscPrintf(comm,
                          "DMPLex patch coordinate exchange: %" PetscInt_FMT
                          " remote patch-node coordinate slots across ranks.\n",
                          totalRemote));
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------
 * Per-iteration helpers
 * ------------------------------------------------------------------------- */

/**
 * @brief Broadcast updated vertex coordinates from owning ranks to all ranks that need them.
 *
 * @details
 * Executes three PetscSFBcast calls (one per coordinate component) over the
 * exchange SF, then writes the received values back into ibm->x/y/z_bp at the
 * appropriate original vertex indices.  After this call every rank's ibm
 * coordinate arrays are consistent for all vertices referenced by its patch stencils.
 *
 * The three Bcast calls are intentionally sequential (Begin+End per component)
 * rather than overlapped, for simplicity; they could be pipelined if latency
 * becomes a bottleneck.
 *
 * @param[in,out] ibm      IBM mesh whose x/y/z_bp arrays are updated in-place.
 * @param[in,out] exchange Exchange struct containing the SF and receive buffers.
 */
static PetscErrorCode UpdatePatchRemoteCoordinates_(IBMNodes *ibm, DMPlexPatchExchange *exchange)
{
  PetscFunctionBeginUser;
  if (!exchange || !exchange->sf) PetscFunctionReturn(0);

  PetscCall(PetscSFBcastBegin(exchange->sf, MPIU_REAL, ibm->x_bp, exchange->x_remote, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(exchange->sf, MPIU_REAL, ibm->x_bp, exchange->x_remote, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(exchange->sf, MPIU_REAL, ibm->y_bp, exchange->y_remote, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(exchange->sf, MPIU_REAL, ibm->y_bp, exchange->y_remote, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(exchange->sf, MPIU_REAL, ibm->z_bp, exchange->z_remote, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(exchange->sf, MPIU_REAL, ibm->z_bp, exchange->z_remote, MPI_REPLACE));

  for (PetscInt i = 0; i < exchange->nRemoteNodes; ++i) {
    const PetscInt node = exchange->orig_node[i];
    ibm->x_bp[node] = exchange->x_remote[i];
    ibm->y_bp[node] = exchange->y_remote[i];
    ibm->z_bp[node] = exchange->z_remote[i];
  }

  PetscFunctionReturn(0);
}

/**
 * @brief Copy Loop subdivision patch stencil data from IBMNodes to a local layout.
 *
 * @details
 * Uses the pre-built @p origCell array (from the migration SF) to map each
 * local cell to its original IBMNodes index in O(nLocalCells) without any
 * DMLabel lookups.
 *
 * @param[in]  dm       Distributed DM.
 * @param[in]  ibm      IBM mesh (source of stencil data).
 * @param[in]  origCell origCell[lc] = original cell index for local cell lc; -1 if unknown.
 * @param[out] layout   Initialised layout (caller must call DMPlexPatchLayoutDestroy_).
 */
static PetscErrorCode DMPlexPatchLayoutCreate_(DM dm, IBMNodes *ibm,
                                               const PetscInt *origCell,
                                               DMPlexPatchLayout *layout)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  PetscCall(PetscObjectReference((PetscObject)dm));
  layout->dm = dm;

  PetscCall(DMPlexGetHeightStratum(dm, 0, &layout->cStart, &layout->cEnd));
  layout->nLocalCells = layout->cEnd - layout->cStart;
  PetscCall(PetscMalloc1(layout->nLocalCells, &layout->orig_cell));
  PetscCall(PetscMalloc1(layout->nLocalCells, &layout->ire));
  PetscCall(PetscMalloc1(layout->nLocalCells, &layout->irv));
  PetscCall(PetscMalloc1(layout->nLocalCells, &layout->val));
  PetscCall(PetscMalloc1(16 * layout->nLocalCells, &layout->patch));

  for (PetscInt lc = 0; lc < layout->nLocalCells; ++lc) {
    const PetscInt ec = origCell[lc];
    layout->orig_cell[lc] = ec;
    if (ec >= 0 && ec < ibm->n_elmt) {
      layout->ire[lc] = ibm->ire[ec];
      layout->irv[lc] = ibm->irv[ec];
      layout->val[lc] = ibm->val[ec];
      for (PetscInt i = 0; i < 16; ++i) layout->patch[16 * lc + i] = ibm->patch[16 * ec + i];
    } else {
      layout->ire[lc] = 0;
      layout->irv[lc] = 0;
      layout->val[lc] = 0;
      for (PetscInt i = 0; i < 16; ++i) layout->patch[16 * lc + i] = DMPLEX_GEOM_PATCH_MISSING;
    }
  }
  PetscFunctionReturn(0);
}

/**
 * @brief Diagnostic: count and report cells whose patch stencils are fully local.
 *
 * @details
 * A patch is "complete" if every referenced original vertex (excluding
 * ghost/auxiliary nodes with index >= nVertices and MISSING slots) is a
 * non-ghost vertex on this rank.  An incomplete patch requires cross-rank
 * coordinate communication managed by the DMPlexPatchExchange.
 *
 * Counts are summed across all ranks with MPI_Allreduce and printed once.
 *
 * @param[in] layout    Local patch layout.
 * @param[in] origVert  origVert[lv]: original IBMNodes vertex index for local vertex lv.
 * @param[in] nVertices Total number of original vertices (ibm->n_v).
 */
static PetscErrorCode ReportPatchHalo_(const DMPlexPatchLayout *layout,
                                       const PetscInt *origVert,
                                       PetscInt nVertices)
{
  PetscInt  localComplete = 0, localIncomplete = 0;
  PetscInt  totalComplete = 0, totalIncomplete = 0;
  PetscBool *isLocalOrig  = NULL;
  PetscBool *isGhost      = NULL;
  PetscSF    pointSF      = NULL;
  PetscInt   pStart, pEnd, vStart, vEnd;

  PetscFunctionBeginUser;

  PetscCall(DMPlexGetChart(layout->dm, &pStart, &pEnd));
  PetscCall(PetscCalloc1(pEnd - pStart, &isGhost));
  PetscCall(DMGetPointSF(layout->dm, &pointSF));
  if (pointSF) {
    PetscInt           nroots, nleaves;
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscCall(PetscSFGetGraph(pointSF, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves >= 0) {
      for (PetscInt i = 0; i < nleaves; ++i) {
        const PetscInt p = ilocal ? ilocal[i] : i;
        if (p >= pStart && p < pEnd) isGhost[p - pStart] = PETSC_TRUE;
      }
    }
  }
  PetscCall(PetscCalloc1(nVertices, &isLocalOrig));
  PetscCall(DMPlexGetDepthStratum(layout->dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt orig = origVert[v - vStart];
    if (orig >= 0 && orig < nVertices && !isGhost[v - pStart]) isLocalOrig[orig] = PETSC_TRUE;
  }
  PetscCall(PetscFree(isGhost));

  for (PetscInt lc = 0; lc < layout->nLocalCells; ++lc) {
    PetscBool complete = PETSC_TRUE;
    PetscInt  nen = (layout->ire[lc] == 0) ? 12 : layout->val[lc] + 6;
    for (PetscInt i = 0; i < nen; ++i) {
      PetscInt node = layout->patch[16 * lc + i];
      if (node == DMPLEX_GEOM_PATCH_MISSING) {
        complete = PETSC_FALSE;
        continue;
      }
      /* Ghost/auxiliary nodes (>= nVertices) are always available locally. */
      if (node >= nVertices) continue;
      if (!isLocalOrig[node]) complete = PETSC_FALSE;
    }
    if (complete) ++localComplete;
    else ++localIncomplete;
  }
  PetscCall(PetscFree(isLocalOrig));

  PetscCallMPI(MPIU_Allreduce(&localComplete, &totalComplete, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(&localIncomplete, &totalIncomplete, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "DMPLex patch halo check: complete local patches=%" PetscInt_FMT
                        ", incomplete local patches=%" PetscInt_FMT "\n",
                        totalComplete, totalIncomplete));
  PetscFunctionReturn(0);
}

/**
 * @brief Synchronise remote coordinates and evaluate the geometry kernel for all local cells.
 *
 * @details
 * First calls UpdatePatchRemoteCoordinates_() to bring ibm coordinate arrays
 * up to date for all cross-rank patch nodes, then loops over every owned cell
 * and calls the three subdivision geometry update routines:
 *   - ElemUpdateGeom0Subdiv: reference-configuration geometry
 *   - ElemUpdateGeomSubdiv:  current-configuration geometry
 *   - ElemUpdateG:           metric tensor update
 *
 * Ghost cells (orig_cell < 0 or >= n_elmt) are skipped.
 * The total number of processed cells is reported via MPI_Allreduce.
 *
 * @param[in,out] fem      FE struct array; fem->ibm coordinates are updated in-place.
 * @param[in]     layout   Local patch layout (provides orig_cell mapping).
 * @param[in,out] exchange Coordinate exchange (SF broadcast executed here).
 */
static PetscErrorCode RunLocalGeometry_(FE *fem, const DMPlexPatchLayout *layout, DMPlexPatchExchange *exchange,
                                        PetscBool geom_only)
{
  PetscInt localCells = 0, totalCells = 0;
  /* geom_only restricts the timed loop to subdivision geometry
     (ElemUpdateGeom0Subdiv/ElemUpdateGeomSubdiv), skipping the metric/basis
     update (ElemUpdateG), so this CPU path's timing is scope-matched to the
     CUDA/Kokkos backends' "SubdivGeomKernel" timers, which measure geometry
     only. This is an explicit parameter, not a global PetscOptions read,
     because this same function is also the CPU reference used for
     correctness verification elsewhere (RunDMPlexGeomProcessesCuda), where
     geom+metric must both run regardless of what a benchmark run's control
     file sets -- a global flag would silently break that verification. */

  PetscFunctionBeginUser;
  PetscCall(EnsureDMPlexGeomLogEvents_());
  PetscCall(PetscLogEventBegin(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  PetscCall(UpdatePatchRemoteCoordinates_(fem->ibm, exchange));
  PetscCall(PetscLogEventEnd(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  PetscCall(PetscLogEventBegin(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  for (PetscInt lc = 0; lc < layout->nLocalCells; ++lc) {
    const PetscInt ec = layout->orig_cell[lc];
    if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
    PetscCall(ElemUpdateGeom0Subdiv(fem, ec));
    PetscCall(ElemUpdateGeomSubdiv(fem, ec));
    if (!geom_only) PetscCall(ElemUpdateG(fem, ec));
    ++localCells;
  }
  PetscCall(PetscLogEventEnd(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  PetscCallMPI(MPIU_Allreduce(&localCells, &totalCells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "DMPLex geometry process updated %" PetscInt_FMT
                        " local cell instances across the distributed plex.\n",
                        totalCells));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------
 * Public entry point
 * ------------------------------------------------------------------------- */

/**
 * @brief Entry point: run the distributed DMPlex geometry pipeline for all bodies.
 *
 * @details
 * For each IBM body, this function performs the full one-time setup (DMPlex
 * distribution, owner table, patch layout, coordinate exchange SF) and then
 * executes the geometry kernel @c n_repeat times to measure cold and warm
 * wall-clock times.  Setup and kernel times are reported to stdout.
 *
 * Runtime parameters (via PetscOptions):
 *   - @c -dmplex_geom_overlap N  Ghost overlap layers (default 2).
 *   - @c -nbody N                Number of IBM bodies (default 1).
 *
 * @param[in,out] fem  Array of FE structs, one per body.  fem[ibi].ibm must be
 *                     populated before calling this function.
 */
PetscErrorCode RunDMPlexGeomProcesses(FE *fem)
{
  PetscInt           nbody = 1;
  DMPlexPatchLayout  layout;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nbody", &nbody, PETSC_NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Running experimental DMPLex subdivision geometry path (overlap=0, no serial interpolation).\n"));

  for (PetscInt ibi = 0; ibi < nbody; ++ibi) {
    DM              dm = NULL;
    PetscMPIInt    *ownerRank = NULL;
    PetscInt       *origCell = NULL, *origVert = NULL;
    DMPlexPatchExchange exchange;
    double          t0;
    double          setup_time;
    const PetscInt  n_repeat = 3;
    double          geom_times[3] = {0};

    double t_dmplex, t_owner, t_layout, t_exchange;

    t0 = MPI_Wtime();
    PetscCall(CreateDistributedDMPlex_(fem[ibi].ibm, &dm, &origCell, &origVert, PETSC_COMM_WORLD));
    t_dmplex = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildOriginalVertexOwnerTable_(dm, fem[ibi].ibm, origVert, &ownerRank, PETSC_COMM_WORLD));
    t_owner = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(DMPlexPatchLayoutCreate_(dm, fem[ibi].ibm, origCell, &layout));
    t_layout = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildPatchCoordinateExchange_(&layout, fem[ibi].ibm, origVert, ownerRank, &exchange, PETSC_COMM_WORLD));
    t_exchange = MPI_Wtime() - t0;

    setup_time = t_dmplex + t_owner + t_layout + t_exchange;

    PetscCall(ReportPatchHalo_(&layout, origVert, fem[ibi].ibm->n_v));

    {
      /* Benchmark-only: restrict the timed loop below to geometry, skipping
         the metric/basis update, so this backend's number is scope-matched
         to the CUDA/Kokkos "SubdivGeomKernel" (geometry-only) timers used
         elsewhere for cross-backend comparison. Off by default -- a real
         solve needs both. */
      PetscBool geom_only = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-dmplex_geom_only", &geom_only, PETSC_NULL));

      for (PetscInt iter = 0; iter < n_repeat; ++iter) {
        t0 = MPI_Wtime();
        PetscCall(RunLocalGeometry_(&fem[ibi], &layout, &exchange, geom_only));
        geom_times[iter] = MPI_Wtime() - t0;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[iter %" PetscInt_FMT "] DMPlexGeomKernel: wall=%.6e s%s\n",
                              iter, geom_times[iter],
                              (iter == 0) ? " (cold)" : " (repeat)"));
      }
    }

    double geom_repeat_total = 0.0;
    for (PetscInt iter = 1; iter < n_repeat; ++iter) geom_repeat_total += geom_times[iter];
    double geom_avg_repeat = (n_repeat > 1) ? geom_repeat_total / (double)(n_repeat - 1) : geom_times[0];

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "\nDMPlex geometry timing for body %d (MPI wall-clock)\n"
                          "  task                         time (s)\n"
                          "  setup (SF + DMPlex)       % .6e\n"
                          "    CreateDistributedDMPlex % .6e\n"
                          "    BuildOwnerTable         % .6e\n"
                          "    PatchLayoutCreate       % .6e\n"
                          "    BuildCoordExchange      % .6e\n"
                          "  geom kernel iter-0         % .6e\n"
                          "  geom kernel avg repeat     % .6e\n",
                          (int)ibi,
                          setup_time,
                          t_dmplex, t_owner, t_layout, t_exchange,
                          geom_times[0],
                          geom_avg_repeat));

    PetscCall(DMPlexPatchExchangeDestroy_(&exchange));
    PetscCall(DMPlexPatchLayoutDestroy_(&layout));
    PetscCall(PetscFree(ownerRank));
    PetscCall(PetscFree(origCell));
    PetscCall(PetscFree(origVert));
    PetscCall(DMDestroy(&dm));
  }
  PetscFunctionReturn(0);
}

/* Correctness-verification helpers: compare a handful of SubdivGeomQP fixed
 * fields (ndx21/ndx31/nn/gc1/gc2/Aaa/Abb/Aab) between the GPU-computed and
 * CPU-computed geometry for the same element, used at the end of
 * RunDMPlexGeomProcessesCuda to sanity-check RunCudaSubdivGeomKernelPartitioned
 * against the established CPU reference path. Does not compare the
 * irregular-element INa/INab pointer-array fields (not covered here). */
/* Tracks both the max absolute difference and the max magnitude of the two
   operands seen, so callers can apply a combined absolute+relative
   tolerance (diff <= atol + rtol*scale) instead of a bare absolute one.
   Needed because CPU and GPU compute mathematically-equivalent but not
   bit-identical results even for correct code -- e.g. INV() (math.c)
   expands the 3x3 determinant along the first column while the CUDA
   Invert3x3_ expands along the first row; both are valid cofactor
   expansions, but floating-point addition isn't associative, so the two
   sum to a tiny (~1e-16 relative) different determinant, which then
   propagates into every entry of the inverted metric tensor. A pure
   absolute tolerance can't distinguish that from a real bug. */
static void UpdateAbsDiffAndScaleCmpnts_(Cmpnts a, Cmpnts b, PetscReal *diff, PetscReal *scale)
{
  *diff  = PetscMax(*diff,  PetscAbsReal(a.x - b.x));
  *diff  = PetscMax(*diff,  PetscAbsReal(a.y - b.y));
  *diff  = PetscMax(*diff,  PetscAbsReal(a.z - b.z));
  *scale = PetscMax(*scale, PetscMax(PetscAbsReal(a.x), PetscAbsReal(b.x)));
  *scale = PetscMax(*scale, PetscMax(PetscAbsReal(a.y), PetscAbsReal(b.y)));
  *scale = PetscMax(*scale, PetscMax(PetscAbsReal(a.z), PetscAbsReal(b.z)));
}

static void MaxAbsDiffAndScaleGeomQP_(const SubdivGeomQP *a, const SubdivGeomQP *b,
                                      PetscReal *diff, PetscReal *scale)
{
  *diff = 0.0; *scale = 0.0;
  UpdateAbsDiffAndScaleCmpnts_(a->ndx21, b->ndx21, diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->ndx31, b->ndx31, diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->nn,    b->nn,    diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->gc1,   b->gc1,   diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->gc2,   b->gc2,   diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->Aaa,   b->Aaa,   diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->Abb,   b->Abb,   diff, scale);
  UpdateAbsDiffAndScaleCmpnts_(a->Aab,   b->Aab,   diff, scale);
}

static void MaxAbsDiffAndScaleElem2DTens_(const Elem2DTens *a, const Elem2DTens *b,
                                          PetscReal *diff, PetscReal *scale)
{
  *diff = 0.0; *scale = 0.0;
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      *diff  = PetscMax(*diff,  PetscAbsReal(a->Cov[i][j] - b->Cov[i][j]));
      *diff  = PetscMax(*diff,  PetscAbsReal(a->Cont[i][j] - b->Cont[i][j]));
      *scale = PetscMax(*scale, PetscMax(PetscAbsReal(a->Cov[i][j]), PetscAbsReal(b->Cov[i][j])));
      *scale = PetscMax(*scale, PetscMax(PetscAbsReal(a->Cont[i][j]), PetscAbsReal(b->Cont[i][j])));
    }
  }
}

/**
 * @brief Per-rank local geometry update for the partitioned DMPlex+CUDA path:
 * PetscSF coordinate exchange on the CPU (same as RunLocalGeometry_), then
 * hand this rank's local cell partition to its own bound GPU for the actual
 * geometry-kernel compute (RunCudaSubdivGeomKernelPartitioned), instead of
 * RunLocalGeometry_'s CPU ElemUpdateGeom0Subdiv/ElemUpdateGeomSubdiv loop.
 *
 * Also computes the metric tensor on the GPU (RunCudaMetricTensorKernelPartitioned),
 * matching RunLocalGeometry_'s ElemUpdateG call -- both geometry and metric
 * stages now run on this rank's bound GPU.
 *
 * @param[in,out] fem      FE struct; fem->ibm coordinates are updated in-place.
 * @param[in]     layout   Local patch layout (this rank's cell partition).
 * @param[in,out] exchange Coordinate exchange (SF broadcast executed here).
 */
/* Seconds spent in the host MPI coordinate exchange / the whole GPU stage,
   accumulated over calls; the benchmark loop reads and zeroes them. */
static double gCudaExchangeSec_ = 0.0, gCudaGpuStageSec_ = 0.0;

static PetscErrorCode RunLocalGeometryCuda_(FE *fem, const DMPlexPatchLayout *layout, DMPlexPatchExchange *exchange)
{
  PetscFunctionBeginUser;
  PetscCall(EnsureDMPlexGeomLogEvents_());
  PetscCall(PetscLogEventBegin(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  {
    const double t_ex0 = MPI_Wtime();
    PetscCall(UpdatePatchRemoteCoordinates_(fem->ibm, exchange));
    gCudaExchangeSec_ += MPI_Wtime() - t_ex0;
  }
  PetscCall(PetscLogEventEnd(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));

  const double t_gpu0 = MPI_Wtime();
  PetscCall(PetscLogEventBegin(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  PetscCall(RunCudaSubdivGeomKernelPartitioned(fem, layout->nLocalCells, layout->orig_cell,
                                               layout->ire, layout->val, layout->patch, PETSC_TRUE));
  PetscCall(RunCudaSubdivGeomKernelPartitioned(fem, layout->nLocalCells, layout->orig_cell,
                                               layout->ire, layout->val, layout->patch, PETSC_FALSE));
  PetscCall(RunCudaMetricTensorKernelPartitioned(fem, layout->nLocalCells, layout->orig_cell));
  PetscCall(PetscLogEventEnd(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  gCudaGpuStageSec_ += MPI_Wtime() - t_gpu0;
  PetscFunctionReturn(0);
}

/**
 * @brief Entry point: run the distributed DMPlex geometry pipeline for all
 * bodies, with each rank's local cell partition evaluated on its own bound
 * GPU instead of on the CPU.
 *
 * @details
 * Identical one-time setup to RunDMPlexGeomProcesses (DMPlex distribution,
 * owner table, patch layout, coordinate exchange SF -- all CPU/host-side,
 * unchanged) and the same n_repeat timing loop structure; only the per-
 * iteration geometry evaluation differs (RunLocalGeometryCuda_ instead of
 * RunLocalGeometry_). GPU device binding for this rank happens automatically
 * on the first CUDA call inside RunLocalGeometryCuda_ (see
 * BindCudaDeviceToRank_ in cuda_kernel.cu).
 *
 * Selected via `-cuda_process 1 -dmplex_geom_process 1` together.
 *
 * @param[in,out] fem  Array of FE structs, one per body.  fem[ibi].ibm must be
 *                     populated before calling this function.
 */
PetscErrorCode RunDMPlexGeomProcessesCuda(FE *fem)
{
  PetscInt           nbody = 1;
  DMPlexPatchLayout  layout;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nbody", &nbody, PETSC_NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Running experimental DMPLex+CUDA subdivision geometry path "
                        "(CPU setup/exchange, GPU compute; overlap=0, no serial interpolation).\n"));

  for (PetscInt ibi = 0; ibi < nbody; ++ibi) {
    DM              dm = NULL;
    PetscMPIInt    *ownerRank = NULL;
    PetscInt       *origCell = NULL, *origVert = NULL;
    DMPlexPatchExchange exchange;
    double          t0;
    double          setup_time;
    /* n_repeat=5, with the first TWO iterations (not just one) treated as
     * warm-up before averaging: iter 0 pays cudaMalloc allocation cost
     * (Section~\ref{sec:profiling}: cudaMalloc dominates at small scale),
     * but iter 1 can *still* be anomalously slow even with allocation
     * reused -- observed directly in the 2-GPU verification run
     * (Section~\ref{sec:multigpu}): iter 1 took 13.7ms, iter 2 took 0.43ms,
     * a 30x gap with no allocation difference between them, consistent
     * with CUDA's lazy JIT compilation of the kernel on its first real
     * launch and/or GPU clock ramp-up from idle. Averaging iter 1 together
     * with genuinely steady-state iterations (as the old n_repeat=3 code
     * did) silently contaminates the "steady-state" number with that
     * warm-up cost. */
    const PetscInt  n_repeat = 5;
    const PetscInt  n_warmup = 2;
    double          geom_times[5] = {0};

    double t_dmplex, t_owner, t_layout, t_exchange;

    t0 = MPI_Wtime();
    PetscCall(CreateDistributedDMPlex_(fem[ibi].ibm, &dm, &origCell, &origVert, PETSC_COMM_WORLD));
    t_dmplex = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildOriginalVertexOwnerTable_(dm, fem[ibi].ibm, origVert, &ownerRank, PETSC_COMM_WORLD));
    t_owner = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(DMPlexPatchLayoutCreate_(dm, fem[ibi].ibm, origCell, &layout));
    t_layout = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildPatchCoordinateExchange_(&layout, fem[ibi].ibm, origVert, ownerRank, &exchange, PETSC_COMM_WORLD));
    t_exchange = MPI_Wtime() - t0;

    setup_time = t_dmplex + t_owner + t_layout + t_exchange;

    PetscCall(ReportPatchHalo_(&layout, origVert, fem[ibi].ibm->n_v));

    /* Sizes/allocates the shared CUDA host-staging buffers (geom_meta,
     * geom_status, geom, irregular) to ibm->n_elmt * act_data.n_qp --
     * RunCudaSubdivGeomKernelPartitioned only checks these are big enough,
     * it never allocates them itself (same contract as the plain
     * RunCudaSubdivGeomKernel, which RunCudaProcesses() in cuda_processes.c
     * also calls this before using). */
    PetscCall(PrepareCudaHostWorkspace(&fem[ibi]));

    double stage_tmp[6];
    for (PetscInt iter = 0; iter < n_repeat; ++iter) {
      if (iter == n_warmup) {
        CudaStageTimersGetReset(stage_tmp);
        gCudaExchangeSec_ = 0.0; gCudaGpuStageSec_ = 0.0;
      }
      t0 = MPI_Wtime();
      PetscCall(RunLocalGeometryCuda_(&fem[ibi], &layout, &exchange));
      geom_times[iter] = MPI_Wtime() - t0;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "[iter %" PetscInt_FMT "] DMPlexGeomKernelCuda: wall=%.6e s%s\n",
                            iter, geom_times[iter],
                            (iter == 0) ? " (cold)" : (iter < n_warmup) ? " (warmup)" : " (steady)"));
    }

    {
      /* Per-iteration stage breakdown over the post-warmup iterations, max over
         ranks: exchange = host MPI coordinate exchange; the GPU stage is split
         into host->device copy, kernel, device->host copy and host pack/unpack
         (the residual of gpu_stage is launch/sync/allocation overhead). */
      const double nit = (double)(n_repeat - n_warmup);
      double loc[8], mx[8];
      CudaStageTimersGetReset(stage_tmp);
      loc[0] = gCudaExchangeSec_ / nit;
      loc[1] = gCudaGpuStageSec_ / nit;
      loc[2] = stage_tmp[0] / nit;
      loc[3] = stage_tmp[1] / nit;
      loc[4] = stage_tmp[2] / nit;
      loc[5] = stage_tmp[3] / nit;
      loc[6] = stage_tmp[4] / nit / 1e6;
      loc[7] = stage_tmp[5] / nit / 1e6;
      PetscCallMPI(MPI_Allreduce(loc, mx, 8, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "\nGPU stage breakdown per iteration (max over ranks, s):\n"
                            "  exchange (host MPI)      % .6e\n"
                            "  gpu stage total          % .6e\n"
                            "    host->device copy      % .6e   (%.1f MB)\n"
                            "    kernels                % .6e\n"
                            "    device->host copy      % .6e   (%.1f MB)\n"
                            "    host pack/unpack       % .6e\n",
                            mx[0], mx[1], mx[2], mx[6], mx[3], mx[4], mx[7], mx[5]));
    }

    double geom_repeat_total = 0.0;
    for (PetscInt iter = n_warmup; iter < n_repeat; ++iter) geom_repeat_total += geom_times[iter];
    double geom_avg_repeat = (n_repeat > n_warmup) ? geom_repeat_total / (double)(n_repeat - n_warmup) : geom_times[n_repeat - 1];

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "\nDMPlex+CUDA geometry timing for body %d (MPI wall-clock, includes CPU<->GPU "
                          "memcpy but not one-time setup below)\n"
                          "  task                         time (s)\n"
                          "  setup (SF + DMPlex)       % .6e\n"
                          "    CreateDistributedDMPlex % .6e\n"
                          "    BuildOwnerTable         % .6e\n"
                          "    PatchLayoutCreate       % .6e\n"
                          "    BuildCoordExchange      % .6e\n"
                          "  geom kernel iter-0         % .6e\n"
                          "  geom kernel avg repeat     % .6e\n",
                          (int)ibi,
                          setup_time,
                          t_dmplex, t_owner, t_layout, t_exchange,
                          geom_times[0],
                          geom_avg_repeat));

    /* Correctness verification: fem[ibi].act_data.elem_act_data[ec].geom0/geom/
     * gm0/gm still hold this rank's GPU-computed results from the last
     * iteration above. Snapshot a handful of locally-owned cells' fixed
     * geometry and metric-tensor (qp=0) fields, then re-run the CPU path
     * (RunLocalGeometry_) on the SAME layout/exchange -- which overwrites
     * those same act_data entries with CPU-computed values -- and diff.
     * This checks RunCudaSubdivGeomKernelPartitioned's and
     * RunCudaMetricTensorKernelPartitioned's actual numerical output against
     * the established CPU reference, not just that they ran without
     * crashing or that the element counts happened to split correctly. */
    {
      const PetscInt n_check = PetscMin(3, layout.nLocalCells);
      SubdivGeomQP saved_geom0[3], saved_geom[3];
      Elem2DTens saved_gm0[3], saved_gm[3];
      PetscInt saved_ec[3];
      PetscInt n_saved = 0;

      for (PetscInt lc = 0; lc < layout.nLocalCells && n_saved < n_check; ++lc) {
        const PetscInt ec = layout.orig_cell[lc];
        if (ec < 0 || ec >= fem[ibi].ibm->n_elmt) continue;
        saved_ec[n_saved] = ec;
        saved_geom0[n_saved] = fem[ibi].act_data.elem_act_data[ec].geom0[0];
        saved_geom[n_saved]  = fem[ibi].act_data.elem_act_data[ec].geom[0];
        saved_gm0[n_saved]   = fem[ibi].act_data.elem_act_data[ec].gm0[0];
        saved_gm[n_saved]    = fem[ibi].act_data.elem_act_data[ec].gm[0];
        ++n_saved;
      }

      if (n_saved > 0) {
        PetscCall(RunLocalGeometry_(&fem[ibi], &layout, &exchange, PETSC_FALSE));

        for (PetscInt k = 0; k < n_saved; ++k) {
          const PetscInt ec = saved_ec[k];
          const SubdivGeomQP *cpu_geom0 = &fem[ibi].act_data.elem_act_data[ec].geom0[0];
          const SubdivGeomQP *cpu_geom  = &fem[ibi].act_data.elem_act_data[ec].geom[0];
          const Elem2DTens *cpu_gm0 = &fem[ibi].act_data.elem_act_data[ec].gm0[0];
          const Elem2DTens *cpu_gm  = &fem[ibi].act_data.elem_act_data[ec].gm[0];
          PetscReal diff0, scale0, diff, scale, gmdiff0, gmscale0, gmdiff, gmscale;
          MaxAbsDiffAndScaleGeomQP_(&saved_geom0[k], cpu_geom0, &diff0, &scale0);
          MaxAbsDiffAndScaleGeomQP_(&saved_geom[k],  cpu_geom,  &diff,  &scale);
          MaxAbsDiffAndScaleElem2DTens_(&saved_gm0[k], cpu_gm0, &gmdiff0, &gmscale0);
          MaxAbsDiffAndScaleElem2DTens_(&saved_gm[k],  cpu_gm,  &gmdiff,  &gmscale);

          /* Combined absolute+relative tolerance (diff <= atol + rtol*scale):
             a pure absolute 1e-9 threshold flagged CPU/GPU pairs as
             MISMATCH purely from the two independently-valid (but not
             bit-identical) 3x3-inverse formulas above -- see the comment on
             UpdateAbsDiffAndScaleCmpnts_. atol=1e-9 still catches a genuine
             near-zero-vs-nonzero bug; rtol=1e-6 comfortably covers the
             observed ~1e-7 worst-case amplified FP noise while remaining
             far tighter than any real logic-error magnitude would produce. */
          const PetscReal atol = 1e-9, rtol = 1e-6;
          PetscBool ok0   = (diff0   <= atol + rtol*scale0)   ? PETSC_TRUE : PETSC_FALSE;
          PetscBool ok    = (diff    <= atol + rtol*scale)    ? PETSC_TRUE : PETSC_FALSE;
          PetscBool gmok0 = (gmdiff0 <= atol + rtol*gmscale0) ? PETSC_TRUE : PETSC_FALSE;
          PetscBool gmok  = (gmdiff  <= atol + rtol*gmscale)  ? PETSC_TRUE : PETSC_FALSE;
          PetscCall(PetscPrintf(PETSC_COMM_SELF,
                                "[DMPlex+CUDA verify] rank-local ec=%" PetscInt_FMT
                                " max|geom0 diff|=%.3e max|geom diff|=%.3e"
                                " max|gm0 diff|=%.3e max|gm diff|=%.3e -> %s\n",
                                ec, (double)diff0, (double)diff, (double)gmdiff0, (double)gmdiff,
                                (ok0 && ok && gmok0 && gmok) ? "MATCH" : "MISMATCH"));
        }
      }
    }

    PetscCall(DMPlexPatchExchangeDestroy_(&exchange));
    PetscCall(DMPlexPatchLayoutDestroy_(&layout));
    PetscCall(PetscFree(ownerRank));
    PetscCall(PetscFree(origCell));
    PetscCall(PetscFree(origVert));
    PetscCall(DMDestroy(&dm));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------
 * Public persistent-context API
 * ------------------------------------------------------------------------- */

/**
 * @brief One-time setup of the distributed DMPlex geometry context.
 *
 * Distributes the IBM surface mesh with DMPlex (overlap=0), builds the
 * per-rank ownership table, local patch layout, and PetscSF coordinate
 * exchange.  Also computes reference-configuration geometry for all locally
 * owned cells.  Must be called before FEM_DMPlexGeomUpdate() or
 * FEM_DMPlexGeomBuildNodeMap().
 *
 * @param fem  FE structure whose ibm field is already populated.
 * @param comm MPI communicator (usually PETSC_COMM_WORLD).
 */
PetscErrorCode FEM_DMPlexGeomSetup(FE *fem, MPI_Comm comm)
{
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  DM             dm  = NULL;
  PetscInt      *origCell = NULL, *origVert = NULL;
  PetscMPIInt   *ownerRank = NULL;
  PetscInt       vStart, vEnd;

  PetscFunctionBeginUser;
  if (ctx->initialized) PetscFunctionReturn(0);  /* idempotent */

  ctx->comm = comm;

  PetscCall(CreateDistributedDMPlex_(fem->ibm, &dm, &origCell, &origVert, comm));
  PetscCall(BuildOriginalVertexOwnerTable_(dm, fem->ibm, origVert, &ownerRank, comm));
  PetscCall(DMPlexPatchLayoutCreate_(dm, fem->ibm, origCell, &ctx->layout));
  PetscCall(BuildPatchCoordinateExchange_(&ctx->layout, fem->ibm, origVert, ownerRank,
                                          &ctx->exchange, comm));

  /* origCell is now copied into ctx->layout.orig_cell — release the raw array. */
  PetscCall(PetscFree(origCell));

  /* Release our DM reference: ctx->layout.dm holds the remaining reference. */
  PetscCall(DMDestroy(&dm));

  /* Persist origVert (for Route B coord update) and ownerRank (for diagnostics). */
  PetscCall(DMPlexGetDepthStratum(ctx->layout.dm, 0, &vStart, &vEnd));
  ctx->nLocalVerts = vEnd - vStart;
  ctx->origVert    = origVert;
  ctx->ownerRank   = ownerRank;

  /* Build ibm_to_local_idx[v]: sequential local Vec index for owned vertex v, else -1.
     Also sets nOwnedVerts (Vec local size) in the same pass.

     Ghost/auxiliary nodes (indices [n_v, n_v+n_ghosts)) are not part of the
     DMPlex mesh -- they're synthetic mirror-reflected points computed
     locally from real node data (lv_geometry_unstructured.c). They still
     need real DOF slots in the solved Vecs (Res/x/Fint/...), each owned by
     exactly one rank; rank 0 owns all of them (small, fixed count, avoids
     any distributed-ownership bookkeeping for a purely derived quantity).
     Appended contiguously right after rank 0's real owned vertices so
     FEM_DMPlexGeomBuildNodeMap's oi counter (built the same way) lines up. */
  {
    PetscMPIInt myrank;
    PetscCallMPI(MPI_Comm_rank(comm, &myrank));
    PetscInt nTotal = fem->ibm->n_v + fem->ibm->n_ghosts;
    PetscCall(PetscMalloc1(nTotal, &ctx->ibm_to_local_idx));
    for (PetscInt v = 0; v < nTotal; ++v) ctx->ibm_to_local_idx[v] = -1;
    ctx->nOwnedVerts = 0;
    for (PetscInt lv = 0; lv < ctx->nLocalVerts; ++lv) {
      const PetscInt v = origVert[lv];
      if (v >= 0 && v < fem->ibm->n_v && ownerRank[v] == myrank)
        ctx->ibm_to_local_idx[v] = ctx->nOwnedVerts++;
    }
    if (myrank == 0) {
      for (PetscInt gc = 0; gc < fem->ibm->n_ghosts; ++gc)
        ctx->ibm_to_local_idx[fem->ibm->n_v + gc] = ctx->nOwnedVerts++;
    }
  }

  /* Compute reference-configuration geometry once — it never changes. */
  PetscCall(EnsureDMPlexGeomLogEvents_());
  PetscCall(PetscLogEventBegin(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  PetscCall(UpdatePatchRemoteCoordinates_(fem->ibm, &ctx->exchange));
  PetscCall(PetscLogEventEnd(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  PetscCall(PetscLogEventBegin(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
    const PetscInt ec = ctx->layout.orig_cell[lc];
    if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
    PetscCall(ElemUpdateGeom0Subdiv(fem, ec));
  }
  PetscCall(PetscLogEventEnd(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));

  ctx->initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/**
 * @brief Per-Newton-step geometry update.
 *
 * Syncs updated vertex coordinates to all ranks that need them (via
 * PetscSFBcast), then recomputes current-configuration geometry for all
 * locally owned cells.  Must be called every Newton step after coordinates
 * in ibm->x/y/z_bp have been updated on owning ranks.
 *
 * @param fem  FE structure with an initialised geom_ctx.
 */
PetscErrorCode FEM_DMPlexGeomUpdate(FE *fem)
{
  DMPlexGeomCtx *ctx = &fem->geom_ctx;

  PetscFunctionBeginUser;
  PetscCheck(ctx->initialized, ctx->comm, PETSC_ERR_ORDER,
             "FEM_DMPlexGeomSetup must be called before FEM_DMPlexGeomUpdate");

  PetscCall(EnsureDMPlexGeomLogEvents_());

  /* Sync current coordinates from owning ranks to all stencil nodes. */
  PetscCall(PetscLogEventBegin(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));
  PetscCall(UpdatePatchRemoteCoordinates_(fem->ibm, &ctx->exchange));
  PetscCall(PetscLogEventEnd(gDMPlexGeomExchangeEvent_, 0, 0, 0, 0));

  /* Evaluate current-configuration geometry for locally owned cells. */
  PetscCall(PetscLogEventBegin(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));
  for (PetscInt lc = 0; lc < ctx->layout.nLocalCells; ++lc) {
    const PetscInt ec = ctx->layout.orig_cell[lc];
    if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
    PetscCall(ElemUpdateGeomSubdiv(fem, ec));
  }
  PetscCall(PetscLogEventEnd(gDMPlexGeomComputeEvent_, 0, 0, 0, 0));

  PetscFunctionReturn(0);
}

/**
 * @brief Build the global DOF index map for parallel force assembly.
 *
 * Populates ctx->ibm_to_global_dof0[v]: the first global DOF index of
 * vertex v in any MPI Vec (e.g. Fint).  Must be called after InitVecs()
 * so that Vec ownership ranges are known.  Result is identical on all
 * ranks (built via MPI_Allreduce MAX).
 *
 * Used by FInternalAct() to route VecSetValues calls to the correct
 * global indices when assembling off-process force contributions.
 *
 * @param fem  FE structure with initialised geom_ctx and created Vecs.
 */
PetscErrorCode FEM_DMPlexGeomBuildNodeMap(FE *fem)
{
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  IBMNodes      *ibm = fem->ibm;
  PetscInt       lo, hi;

  PetscFunctionBeginUser;
  PetscCheck(ctx->initialized, ctx->comm, PETSC_ERR_ORDER,
             "FEM_DMPlexGeomSetup must be called before FEM_DMPlexGeomBuildNodeMap");

  PetscInt nTotal = ibm->n_v + ibm->n_ghosts;
  PetscCall(PetscMalloc1(nTotal, &ctx->ibm_to_global_dof0));
  for (PetscInt v = 0; v < nTotal; ++v) ctx->ibm_to_global_dof0[v] = -1;

  /* Each rank fills the global Vec DOF0 for its owned vertices only.
     oi is the sequential owned-vertex index matching the Vec local layout
     (must walk vertices in the SAME order as FEM_DMPlexGeomSetup's
     ibm_to_local_idx pass -- real owned vertices first, then (rank 0 only)
     ghost/auxiliary nodes [n_v, n_v+n_ghosts) appended contiguously). */
  {
    PetscMPIInt myrank;
    PetscCallMPI(MPI_Comm_rank(ctx->comm, &myrank));
    PetscCall(VecGetOwnershipRange(fem->Fint, &lo, &hi));
    PetscInt oi = 0;
    for (PetscInt lv = 0; lv < ctx->nLocalVerts; ++lv) {
      const PetscInt v = ctx->origVert[lv];
      if (v < 0 || v >= ibm->n_v || ctx->ownerRank[v] != myrank) continue;
      ctx->ibm_to_global_dof0[v] = lo + oi * dof;
      oi++;
    }
    if (myrank == 0) {
      for (PetscInt gc = 0; gc < ibm->n_ghosts; ++gc) {
        ctx->ibm_to_global_dof0[ibm->n_v + gc] = lo + oi * dof;
        oi++;
      }
    }
  }

  /* Allreduce(MAX): owned entries (>= 0) win over unowned entries (-1). */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, ctx->ibm_to_global_dof0, nTotal,
                              MPIU_INT, MPI_MAX, ctx->comm));

  PetscFunctionReturn(0);
}

/**
 * @brief Zero the residual at ghost/auxiliary node DOF rows.
 *
 * Ghost nodes (indices [n_v, n_v+n_ghosts)) are not independent unknowns --
 * their position is always the mirror-reflection of a real node, already
 * computed into ibm->x_bp/y_bp/z_bp by GlobalGhost() (called after every
 * AreaNormal()) and used directly by all geometry evaluation. This forces
 * the residual @p R to zero at their DOF rows, so SNES/KSP never try to
 * treat a ghost DOF as a real unknown to solve for.
 *
 * Does NOT touch @p x. An earlier version also wrote the ghost position
 * into x's DOF slot (to keep the solved Vec a complete record of the whole
 * mesh, matching the original ChimeraFEM ghost-node design) via
 * VecGetArray(x, ...) -- but x is the SNES solution iterate passed into
 * FormFunction, which PETSc treats as read-only during that call (enforced
 * by a read lock in some PETSc versions/builds, not in others -- this
 * passed silently locally on PETSc 3.21 but errored on Grace's 3.22.5:
 * "Vector ... was locked for read-only access"). Since every real geometry
 * computation reads ibm->x_bp (not x) for ghost positions, x's ghost slot
 * was purely cosmetic bookkeeping, not load-bearing -- dropped rather than
 * worked around.
 *
 * Only rank 0 does anything here (it owns every ghost DOF -- see the
 * ibm_to_local_idx pass in FEM_DMPlexGeomSetup); other ranks return
 * immediately. Must be called by every rank (VecGetArray/VecRestoreArray
 * on a distributed Vec are not collective, but this keeps the call site
 * uniform and safe if that ever changes).
 *
 * @param fem  FE structure with initialised geom_ctx.
 * @param R    Residual Vec (zeroed at ghost DOF rows here).
 */
PetscErrorCode FEM_DMPlexGeomSyncGhostDOFs(FE *fem, Vec R)
{
  DMPlexGeomCtx *ctx = &fem->geom_ctx;
  IBMNodes      *ibm = fem->ibm;
  PetscMPIInt    myrank;

  PetscFunctionBeginUser;
  if (ibm->n_ghosts == 0) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(ctx->comm, &myrank));
  if (myrank != 0) PetscFunctionReturn(0);

  PetscReal *RR;
  PetscCall(VecGetArray(R, &RR));
  for (PetscInt gc = 0; gc < ibm->n_ghosts; ++gc) {
    PetscInt gnode = ibm->n_v + gc;
    PetscInt li    = ctx->ibm_to_local_idx[gnode];
    PetscCheck(li >= 0, ctx->comm, PETSC_ERR_PLIB,
               "ghost node %" PetscInt_FMT " has no local Vec slot on rank 0", gnode);
    RR[li*dof  ] = 0.0;
    RR[li*dof+1] = 0.0;
    RR[li*dof+2] = 0.0;
  }
  PetscCall(VecRestoreArray(R, &RR));

  PetscFunctionReturn(0);
}

/**
 * @brief Free all memory owned by the DMPlex geometry context.
 *
 * Destroys the distributed DM, PetscSF, and all per-rank arrays.
 * Safe to call on an uninitialised context (no-op).
 *
 * @param fem  FE structure whose geom_ctx will be cleared.
 */
PetscErrorCode FEM_DMPlexGeomDestroy(FE *fem)
{
  DMPlexGeomCtx *ctx = &fem->geom_ctx;

  PetscFunctionBeginUser;
  if (!ctx->initialized) PetscFunctionReturn(0);

  PetscCall(DMPlexPatchExchangeDestroy_(&ctx->exchange));
  PetscCall(DMPlexPatchLayoutDestroy_(&ctx->layout));
  PetscCall(PetscFree(ctx->origVert));
  PetscCall(PetscFree(ctx->ownerRank));
  PetscCall(PetscFree(ctx->ibm_to_local_idx));
  PetscCall(PetscFree(ctx->ibm_to_global_dof0));
  PetscCall(PetscMemzero(ctx, sizeof(*ctx)));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------
 * Legacy entry point (deprecated — rebuilds DM+SF every call)
 * ------------------------------------------------------------------------- */

PetscErrorCode RunDMPlexGeomSubdiv(FE *fem, MPI_Comm comm)
{
  PetscFunctionBeginUser;
  /* Setup computes geom0 (reference, one-time) and syncs coords for geom (current). */
  PetscCall(FEM_DMPlexGeomSetup(fem, comm));
  PetscCall(FEM_DMPlexGeomUpdate(fem));
  PetscCall(FEM_DMPlexGeomDestroy(fem));
  PetscFunctionReturn(0);
}
