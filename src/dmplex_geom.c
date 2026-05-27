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
#include <string.h>
#include "dmplex_geom.h"
#include "active_strain.h"

/** Sentinel stored in DMPlexPatchLayout::patch to mark a missing stencil slot. */
#define DMPLEX_GEOM_PATCH_MISSING 1000000

extern PetscReal h0;

/* -------------------------------------------------------------------------
 * Internal data structures
 * ------------------------------------------------------------------------- */

/**
 * @brief Per-rank local copy of Loop subdivision patch stencil data.
 *
 * @details
 * Built once from the global IBMNodes arrays after DMPlex distribution.
 * All arrays are indexed by local cell index lc in [0, nLocalCells).
 * The DM reference is retained to allow label queries in downstream functions.
 */
typedef struct {
  DM        dm;          /**< Distributed DM; ref-counted, destroyed with layout. */
  PetscInt *orig_cell;   /**< orig_cell[lc]: original IBMNodes cell index for local cell lc.
                              -1 for ghost cells whose stencil is unavailable. */
  PetscInt *ire;         /**< ire[lc]: Loop subdivision flag (0 = regular, 1 = irregular). */
  PetscInt *irv;         /**< irv[lc]: irregular-vertex flag for the stencil. */
  PetscInt *val;         /**< val[lc]: valence of the irregular vertex (used when ire==1). */
  PetscInt *patch;       /**< patch[16*lc + i]: original vertex index of the i-th patch node
                              for local cell lc.  Up to 12 entries for regular cells
                              (ire==0) and val+6 for irregular cells.  Unused slots hold
                              DMPLEX_GEOM_PATCH_MISSING. */
  PetscInt  cStart;      /**< First cell point index in the DM chart (height stratum 0). */
  PetscInt  cEnd;        /**< One past the last cell point index. */
  PetscInt  nLocalCells; /**< Number of locally owned cells: cEnd - cStart. */
} DMPlexPatchLayout;

/**
 * @brief MPI coordinate exchange for cross-rank patch stencil nodes.
 *
 * @details
 * Holds the PetscSF and receive buffers used to fetch vertex coordinates that
 * belong to other ranks but are referenced by this rank's patch stencils.
 * The SF is configured with ibm->n_v roots per rank (the full x/y/z_bp arrays)
 * and nRemoteNodes leaves (one per unique cross-rank vertex needed here).
 *
 * Data flow each geometry iteration:
 * @code
 *   PetscSFBcast(sf, ibm->x_bp  →  x_remote)
 *   PetscSFBcast(sf, ibm->y_bp  →  y_remote)
 *   PetscSFBcast(sf, ibm->z_bp  →  z_remote)
 *   ibm->x_bp[orig_node[i]] = x_remote[i]   // write-back
 * @endcode
 */
typedef struct {
  PetscSF      sf;            /**< Star Forest: nroots=ibm->n_v, nleaves=nRemoteNodes. */
  PetscInt     nRemoteNodes;  /**< Number of unique remote vertices needed by this rank. */
  PetscInt    *orig_node;     /**< orig_node[i]: original vertex index of remote node i.
                                   Size: nRemoteNodes. */
  PetscMPIInt *owner_rank;    /**< owner_rank[i]: MPI rank that owns orig_node[i].
                                   Size: nRemoteNodes. */
  PetscInt    *owner_index;   /**< owner_index[i]: root slot on owner_rank[i], equal to
                                   orig_node[i] since original indices index into x_bp.
                                   Size: nRemoteNodes. */
  PetscReal   *x_remote;      /**< Receive buffer for x coordinates. Size: nRemoteNodes. */
  PetscReal   *y_remote;      /**< Receive buffer for y coordinates. Size: nRemoteNodes. */
  PetscReal   *z_remote;      /**< Receive buffer for z coordinates. Size: nRemoteNodes. */
} DMPlexPatchExchange;

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
 * empty arrays.  DMPlexCreateFromCellListPetsc is called collectively and
 * internally interpolates edges from the triangle connectivity, producing a
 * complete topological DAG (vertices → edges → faces).
 *
 * This function produces a <em>serial</em> (undistributed) DM on every rank,
 * with the full mesh residing only on rank 0.  Call CreateDistributedDMPlex_()
 * to obtain the distributed version.
 *
 * @param[in]  ibm  IBM surface mesh (read on rank 0 only).
 * @param[out] dm   New serial DMPlex (caller must DMDestroy).
 */
static PetscErrorCode BuildSerialDMPlex_(IBMNodes *ibm, DM *dm)
{
  PetscMPIInt rank;
  PetscInt    dim = 2, spaceDim = 3, corners = 3;
  PetscInt    nCells = 0, nVerts = 0;
  PetscInt   *cells = NULL;
  PetscReal  *coords = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

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

  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, nCells, nVerts, corners,
                                          PETSC_TRUE, cells, spaceDim, coords, dm));
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree(coords));
  PetscFunctionReturn(0);
}

/**
 * @brief Stamp every cell and vertex in a serial DM with its original index.
 *
 * @details
 * Creates two integer DMLabels on the serial (pre-distribution) DM:
 *   - @c "biofem_orig_cell":   maps DM cell point p → original cell index (p - cStart).
 *   - @c "biofem_orig_vertex": maps DM vertex point p → original vertex index (p - vStart).
 *
 * Only rank 0 performs the labelling because the serial DM holds the full mesh
 * only on rank 0.  After DMPlexDistribute() the labels migrate with the points,
 * so every distributed point retains its original index across ranks.
 *
 * @param[in,out] dm Serial DM (before distribution).
 */
static PetscErrorCode LabelOriginalPoints_(DM dm)
{
  PetscMPIInt rank;
  PetscInt    cStart, cEnd, vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMCreateLabel(dm, "biofem_orig_cell"));
  PetscCall(DMCreateLabel(dm, "biofem_orig_vertex"));
  if (rank == 0) {
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    for (PetscInt c = cStart; c < cEnd; ++c) {
      PetscCall(DMSetLabelValue(dm, "biofem_orig_cell", c, c - cStart));
    }
    for (PetscInt v = vStart; v < vEnd; ++v) {
      PetscCall(DMSetLabelValue(dm, "biofem_orig_vertex", v, v - vStart));
    }
  }
  PetscFunctionReturn(0);
}

/**
 * @brief Build, label, and distribute a DMPlex across all MPI ranks.
 *
 * @details
 * Combines BuildSerialDMPlex_() and LabelOriginalPoints_() with
 * DMPlexDistribute() to produce a distributed DM with @p overlap ghost layers.
 * The migration Star Forest is discarded; only the distributed DM is returned.
 *
 * After this call:
 *   - Each rank owns a contiguous subdomain of triangles plus @p overlap layers
 *     of ghost cells needed to evaluate boundary stencils.
 *   - Every distributed point carries its original index in the biofem labels,
 *     enabling mapping back to IBMNodes arrays on any rank.
 *
 * @param[in]  ibm      IBM surface mesh (read on rank 0 only).
 * @param[in]  overlap  Number of ghost cell layers for the distributed partition.
 * @param[out] dmDist   Distributed DM (caller must DMDestroy).
 */
static PetscErrorCode CreateDistributedDMPlex_(IBMNodes *ibm, PetscInt overlap, DM *dmDist)
{
  DM      dm = NULL, distributed = NULL;
  PetscSF migrationSF = NULL;

  PetscFunctionBeginUser;
  PetscCall(BuildSerialDMPlex_(ibm, &dm));
  PetscCall(LabelOriginalPoints_(dm));
  PetscCall(DMPlexDistribute(dm, overlap, &migrationSF, &distributed));
  if (distributed) {
    PetscCall(DMDestroy(&dm));
    dm = distributed;
  }
  PetscCall(PetscSFDestroy(&migrationSF));
  *dmDist = dm;
  PetscFunctionReturn(0);
}

/**
 * @brief Build a global table mapping each original vertex index to its owning MPI rank.
 *
 * @details
 * After distribution, the same original vertex may appear as a ghost on multiple
 * ranks, but it is <em>owned</em> by exactly one.  This function determines
 * ownership by:
 *
 *   1. Reading the DM's point Star Forest (DMGetPointSF) to identify ghost points.
 *      In a PetscSF the <em>leaves</em> are the ghost copies; marking them gives
 *      an O(nLeaves) ghost lookup table indexed by local chart position.
 *
 *   2. Iterating over every local vertex (DMPlexGetDepthStratum, depth=0),
 *      reading its original index from the @c "biofem_orig_vertex" label, and
 *      recording @c localOwner[origIdx] = myRank for non-ghost vertices.
 *
 *   3. Calling MPI_Allreduce(MPI_MAX) so every rank receives the same
 *      @c ownerRank[0..n_v) table.  Non-owners contribute -1; MPI_MAX selects
 *      the one rank that contributed its rank ID.
 *
 * @param[in]  dm            Distributed DM.
 * @param[in]  ibm           IBM mesh (only ibm->n_v is read).
 * @param[out] ownerRankOut  Allocated array of size ibm->n_v; ownerRank[i] is the
 *                           MPI rank that owns original vertex i.  Caller must PetscFree.
 */
static PetscErrorCode BuildOriginalVertexOwnerTable_(DM dm, IBMNodes *ibm, PetscMPIInt **ownerRankOut)
{
  PetscMPIInt  rank;
  PetscInt     pStart, pEnd, vStart, vEnd;
  PetscBool   *isGhost = NULL;
  PetscMPIInt *localOwner = NULL;
  PetscMPIInt *ownerRank = NULL;
  PetscSF      pointSF = NULL;
  DMLabel      vertexLabel = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscCalloc1(pEnd - pStart, &isGhost));

  /* Mark ghost points using the DM's point SF.
     SF leaves are the ghost copies; ilocal[i] is the local chart index of leaf i.
     If ilocal is NULL the leaves are packed at positions 0,1,2,... */
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

  PetscCall(DMGetLabel(dm, "biofem_orig_vertex", &vertexLabel));
  PetscCheck(vertexLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing biofem_orig_vertex label");
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt orig = -1;
    PetscCall(DMLabelGetValue(vertexLabel, v, &orig));
    if (orig >= 0 && orig < ibm->n_v && !isGhost[v - pStart]) {
      localOwner[orig] = rank;
    }
  }

  /* MPI_MAX collapses the per-rank tables: the owning rank's entry (>= 0) beats
     every non-owner's -1, giving a consistent global table on all ranks. */
  PetscCallMPI(MPI_Allreduce(localOwner, ownerRank, ibm->n_v, MPI_INT, MPI_MAX, PETSC_COMM_WORLD));
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
 * @param[in]  ownerRank Global owner table from BuildOriginalVertexOwnerTable_().
 * @param[out] exchange  Initialised exchange struct (caller must call
 *                       DMPlexPatchExchangeDestroy_ when done).
 */
static PetscErrorCode BuildPatchCoordinateExchange_(const DMPlexPatchLayout *layout,
                                                    IBMNodes *ibm,
                                                    const PetscMPIInt *ownerRank,
                                                    DMPlexPatchExchange *exchange)
{
  PetscInt     maxNeeded;
  PetscSFNode *remote      = NULL;
  PetscBool   *isLocalOrig = NULL;
  PetscBool   *alreadyAdded = NULL;
  PetscBool   *isGhost     = NULL;
  PetscSF      pointSF     = NULL;
  DMLabel      vertexLabel = NULL;
  PetscInt     pStart, pEnd, vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(exchange, sizeof(*exchange)));

  /* Build isLocalOrig[n_v]: true if original vertex i is a non-ghost local vertex
     on this rank. One O(nLocalVerts) walk replaces O(nLocalVerts) per patch-node call. */
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
  PetscCall(DMGetLabel(layout->dm, "biofem_orig_vertex", &vertexLabel));
  PetscCheck(vertexLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing biofem_orig_vertex label");
  PetscCall(DMPlexGetDepthStratum(layout->dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt orig = -1;
    PetscCall(DMLabelGetValue(vertexLabel, v, &orig));
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

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &exchange->sf));
  PetscCall(PetscSFSetGraph(exchange->sf, ibm->n_v, exchange->nRemoteNodes,
                            NULL, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES));
  PetscCall(PetscSFSetUp(exchange->sf));
  PetscCall(PetscFree(remote));

  {
    PetscInt totalRemote = 0;
    PetscCallMPI(MPIU_Allreduce(&exchange->nRemoteNodes, &totalRemote, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
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
 * After DMPlex distribution each rank owns a subset of cells.  This function
 * maps every distributed cell back to its original IBMNodes index via the
 * @c "biofem_orig_cell" label, then copies ire/irv/val and the 16-slot patch
 * array into the local DMPlexPatchLayout.
 *
 * Ghost cells (whose label value is -1 or outside [0, n_elmt)) receive
 * DMPLEX_GEOM_PATCH_MISSING in all patch slots and zeroed stencil flags;
 * they are skipped in the geometry kernel.
 *
 * @param[in]  dm     Distributed DM.
 * @param[in]  ibm    IBM mesh (source of stencil data).
 * @param[out] layout Initialised layout (caller must call DMPlexPatchLayoutDestroy_).
 */
static PetscErrorCode DMPlexPatchLayoutCreate_(DM dm, IBMNodes *ibm, DMPlexPatchLayout *layout)
{
  DMLabel cellLabel = NULL;

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

  PetscCall(DMGetLabel(dm, "biofem_orig_cell", &cellLabel));
  PetscCheck(cellLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing biofem_orig_cell label");

  for (PetscInt c = layout->cStart; c < layout->cEnd; ++c) {
    const PetscInt lc = c - layout->cStart;
    PetscInt       ec = -1;
    PetscCall(DMLabelGetValue(cellLabel, c, &ec));
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
 * @param[in] nVertices Total number of original vertices (ibm->n_v).
 */
static PetscErrorCode ReportPatchHalo_(const DMPlexPatchLayout *layout, PetscInt nVertices)
{
  PetscInt  localComplete = 0, localIncomplete = 0;
  PetscInt  totalComplete = 0, totalIncomplete = 0;
  PetscBool *isLocalOrig  = NULL;
  PetscBool *isGhost      = NULL;
  PetscSF    pointSF      = NULL;
  DMLabel    vertexLabel  = NULL;
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
  PetscCall(DMGetLabel(layout->dm, "biofem_orig_vertex", &vertexLabel));
  PetscCheck(vertexLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing biofem_orig_vertex label");
  PetscCall(DMPlexGetDepthStratum(layout->dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt orig = -1;
    PetscCall(DMLabelGetValue(vertexLabel, v, &orig));
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
static PetscErrorCode RunLocalGeometry_(FE *fem, const DMPlexPatchLayout *layout, DMPlexPatchExchange *exchange)
{
  PetscInt localCells = 0, totalCells = 0;

  PetscFunctionBeginUser;
  PetscCall(UpdatePatchRemoteCoordinates_(fem->ibm, exchange));
  for (PetscInt lc = 0; lc < layout->nLocalCells; ++lc) {
    const PetscInt ec = layout->orig_cell[lc];
    if (ec < 0 || ec >= fem->ibm->n_elmt) continue;
    PetscCall(ElemUpdateGeom0Subdiv(fem, ec));
    PetscCall(ElemUpdateGeomSubdiv(fem, ec));
    PetscCall(ElemUpdateG(fem, ec));
    ++localCells;
  }
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
 *   - @c -dmplex_geom_overlap <int>  Ghost overlap layers (default 2).
 *   - @c -nbody <int>                Number of IBM bodies (default 1).
 *
 * @param[in,out] fem  Array of FE structs, one per body.  fem[ibi].ibm must be
 *                     populated before calling this function.
 */
PetscErrorCode RunDMPlexGeomProcesses(FE *fem)
{
  PetscInt           overlap = 2;
  PetscInt           nbody = 1;
  DMPlexPatchLayout  layout;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-dmplex_geom_overlap", &overlap, PETSC_NULL));
  PetscCall(PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-nbody", &nbody, PETSC_NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Running experimental DMPLex subdivision geometry path with overlap=%" PetscInt_FMT ".\n",
                        overlap));

  for (PetscInt ibi = 0; ibi < nbody; ++ibi) {
    DM              dm = NULL;
    PetscMPIInt    *ownerRank = NULL;
    DMPlexPatchExchange exchange;
    double          t0, t1;
    double          setup_time;
    const PetscInt  n_repeat = 3;
    double          geom_times[3] = {0};

    t0 = MPI_Wtime();
    PetscCall(CreateDistributedDMPlex_(fem[ibi].ibm, overlap, &dm));
    PetscCall(BuildOriginalVertexOwnerTable_(dm, fem[ibi].ibm, &ownerRank));
    PetscCall(DMPlexPatchLayoutCreate_(dm, fem[ibi].ibm, &layout));
    PetscCall(BuildPatchCoordinateExchange_(&layout, fem[ibi].ibm, ownerRank, &exchange));
    t1 = MPI_Wtime();
    setup_time = t1 - t0;

    PetscCall(ReportPatchHalo_(&layout, fem[ibi].ibm->n_v));

    for (PetscInt iter = 0; iter < n_repeat; ++iter) {
      t0 = MPI_Wtime();
      PetscCall(RunLocalGeometry_(&fem[ibi], &layout, &exchange));
      t1 = MPI_Wtime();
      geom_times[iter] = t1 - t0;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "[iter %" PetscInt_FMT "] DMPlexGeomKernel: wall=%.6e s%s\n",
                            iter, geom_times[iter],
                            (iter == 0) ? " (cold)" : " (repeat)"));
    }

    double geom_repeat_total = 0.0;
    for (PetscInt iter = 1; iter < n_repeat; ++iter) geom_repeat_total += geom_times[iter];
    double geom_avg_repeat = (n_repeat > 1) ? geom_repeat_total / (double)(n_repeat - 1) : geom_times[0];

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "\nDMPlex geometry timing for body %d (MPI wall-clock)\n"
                          "  task                         time (s)\n"
                          "  setup (SF + DMPlex)       % .6e\n"
                          "  geom kernel iter-0         % .6e\n"
                          "  geom kernel avg repeat     % .6e\n",
                          (int)ibi,
                          setup_time,
                          geom_times[0],
                          geom_avg_repeat));

    PetscCall(DMPlexPatchExchangeDestroy_(&exchange));
    PetscCall(DMPlexPatchLayoutDestroy_(&layout));
    PetscCall(PetscFree(ownerRank));
    PetscCall(DMDestroy(&dm));
  }
  PetscFunctionReturn(0);
}
