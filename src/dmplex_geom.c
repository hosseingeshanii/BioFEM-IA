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
                                         PetscInt *cStartOut, PetscInt *vStartOut)
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

  /* PETSC_FALSE: do NOT interpolate edges on the serial mesh.
     Edge interpolation is O(N^{3+}) for large meshes; we distribute first
     and let per-rank interpolation (if needed) happen on small subsets. */
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, nCells, nVerts, corners,
                                          PETSC_FALSE, cells, spaceDim, coords, dm));
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree(coords));

  /* Record chart offsets on rank 0 (other ranks have empty DMs). */
  {
    PetscInt cS = 0, cE = 0, vS = 0, vE = 0;
    PetscCall(DMPlexGetHeightStratum(*dm, 0, &cS, &cE));
    PetscCall(DMPlexGetDepthStratum(*dm, 0, &vS, &vE));
    /* Broadcast from rank 0 so every rank knows the serial offsets. */
    PetscCallMPI(MPI_Bcast(&cS, 1, MPIU_INT, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Bcast(&vS, 1, MPIU_INT, 0, PETSC_COMM_WORLD));
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
                                               PetscInt **origVertOut)
{
  DM      dm = NULL, distributed = NULL;
  PetscSF migrationSF = NULL;
  PetscInt serial_cStart = 0, serial_vStart = 0;
  double  t0, t_build, t_dist;

  PetscFunctionBeginUser;
  t0 = MPI_Wtime();
  PetscCall(BuildSerialDMPlex_(ibm, &dm, &serial_cStart, &serial_vStart));
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
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
                                                     PetscMPIInt **ownerRankOut)
{
  PetscMPIInt  rank;
  PetscInt     pStart, pEnd, vStart, vEnd;
  PetscBool   *isGhost = NULL;
  PetscMPIInt *localOwner = NULL;
  PetscMPIInt *ownerRank = NULL;
  PetscSF      pointSF = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
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
                                                    const PetscInt *origVert,
                                                    const PetscMPIInt *ownerRank,
                                                    DMPlexPatchExchange *exchange)
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
    PetscCall(CreateDistributedDMPlex_(fem[ibi].ibm, &dm, &origCell, &origVert));
    t_dmplex = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildOriginalVertexOwnerTable_(dm, fem[ibi].ibm, origVert, &ownerRank));
    t_owner = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(DMPlexPatchLayoutCreate_(dm, fem[ibi].ibm, origCell, &layout));
    t_layout = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    PetscCall(BuildPatchCoordinateExchange_(&layout, fem[ibi].ibm, origVert, ownerRank, &exchange));
    t_exchange = MPI_Wtime() - t0;

    setup_time = t_dmplex + t_owner + t_layout + t_exchange;

    PetscCall(ReportPatchHalo_(&layout, origVert, fem[ibi].ibm->n_v));

    for (PetscInt iter = 0; iter < n_repeat; ++iter) {
      t0 = MPI_Wtime();
      PetscCall(RunLocalGeometry_(&fem[ibi], &layout, &exchange));
      geom_times[iter] = MPI_Wtime() - t0;

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
