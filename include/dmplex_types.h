#ifndef DMPLEX_TYPES_H
#define DMPLEX_TYPES_H

#include <petscdmplex.h>
#include <petscsf.h>

/**
 * @file dmplex_types.h
 * @brief Persistent DMPlex geometry context for Loop subdivision surfaces.
 *
 * These types are kept in a separate header (no dependency on variables.h) so
 * that variables.h can embed DMPlexGeomCtx in the FE struct without a circular
 * include.
 */

/**
 * @brief Per-rank local copy of Loop subdivision patch stencil data.
 *
 * Built once from the global IBMNodes arrays after DMPlex distribution.
 * All arrays are indexed by local cell index lc in [0, nLocalCells).
 * The DM reference is retained so that downstream functions can query it.
 */
typedef struct {
  DM        dm;          /**< Distributed DM; ref-counted, destroyed with layout. */
  PetscInt *orig_cell;   /**< orig_cell[lc]: original IBMNodes cell index for local cell lc.
                              -1 for ghost cells whose stencil is unavailable. */
  PetscInt *ire;         /**< ire[lc]: Loop subdivision flag (0=regular, 1=irregular). */
  PetscInt *irv;         /**< irv[lc]: irregular-vertex flag. */
  PetscInt *val;         /**< val[lc]: valence of the irregular vertex (used when ire==1). */
  PetscInt *patch;       /**< patch[16*lc+i]: original vertex index of the i-th patch node.
                              Up to 12 entries for regular (ire==0) and val+6 for irregular.
                              Unused slots hold DMPLEX_GEOM_PATCH_MISSING (1000000). */
  PetscInt  cStart;      /**< First cell point in the DM chart (height stratum 0). */
  PetscInt  cEnd;        /**< One past the last cell point. */
  PetscInt  nLocalCells; /**< Number of locally owned cells: cEnd - cStart. */
} DMPlexPatchLayout;

/**
 * @brief MPI coordinate exchange for cross-rank patch stencil nodes.
 *
 * Holds the PetscSF and receive buffers used to fetch vertex coordinates that
 * belong to other MPI ranks but are referenced by this rank's patch stencils.
 *
 * SF layout: nroots = ibm->n_v (full original vertex array; each rank serves as
 * potential root for all n_v entries), nleaves = nRemoteNodes.
 *
 * Data flow each geometry iteration:
 * @code
 *   PetscSFBcast(sf, ibm->x_bp → x_remote)   // fetch from owning ranks
 *   ibm->x_bp[orig_node[i]] = x_remote[i]     // write-back
 * @endcode
 */
typedef struct {
  PetscSF      sf;            /**< Star Forest: nroots=ibm->n_v, nleaves=nRemoteNodes. */
  PetscInt     nRemoteNodes;  /**< Number of unique remote vertices needed by this rank. */
  PetscInt    *orig_node;     /**< orig_node[i]: original vertex index of remote node i. */
  PetscMPIInt *owner_rank;    /**< owner_rank[i]: MPI rank that owns orig_node[i]. */
  PetscInt    *owner_index;   /**< owner_index[i]: root slot on owner_rank[i] (= orig_node[i]). */
  PetscReal   *x_remote;      /**< Receive buffer for x coordinates. Size: nRemoteNodes. */
  PetscReal   *y_remote;      /**< Receive buffer for y coordinates. Size: nRemoteNodes. */
  PetscReal   *z_remote;      /**< Receive buffer for z coordinates. Size: nRemoteNodes. */
} DMPlexPatchExchange;

/**
 * @brief Persistent DMPlex geometry context stored in FE.
 *
 * Built once by FEM_DMPlexGeomSetup() during initialization.  Holds the
 * distributed DM, per-rank stencil layout, coordinate-exchange SF, and the
 * local-to-original vertex mapping needed for parallel coordinate updates.
 *
 * Destroyed by FEM_DMPlexGeomDestroy().
 */
typedef struct {
  DMPlexPatchLayout   layout;      /**< Per-rank stencil layout (owns the distributed DM). */
  DMPlexPatchExchange exchange;    /**< PetscSF for cross-rank coordinate synchronization. */
  PetscInt           *origVert;    /**< origVert[lv]: original ibm vertex index for local vertex lv.
                                        Size: nLocalVerts.  Needed in Route B to map parallel
                                        Vec local DOFs back to ibm->x_bp global indices. */
  PetscInt            nLocalVerts;  /**< Number of local vertices: vEnd - vStart (includes ghost copies). */
  PetscInt            nOwnedVerts;  /**< Vertices owned by this rank (subset of nLocalVerts).
                                         Used as Vec local size in parallel SNES path. */
  PetscInt           *ibm_to_local_idx; /**< ibm_to_local_idx[v]: sequential local Vec index (0..nOwnedVerts-1)
                                             for ibm vertex v if owned by this rank, or -1 otherwise.
                                             Size: ibm->n_v.  Enables parallel-safe VecGetArray access. */
  PetscMPIInt        *ownerRank;    /**< ownerRank[v]: MPI rank owning original vertex v.
                                         Size: ibm->n_v. */
  PetscInt           *ibm_to_global_dof0; /**< ibm_to_global_dof0[v]: first global Vec DOF index
                                               for ibm vertex v.  Size: ibm->n_v.
                                               Built by FEM_DMPlexGeomBuildNodeMap() after InitVecs().
                                               -1 before that call. */
  PetscBool           initialized; /**< PETSC_TRUE after FEM_DMPlexGeomSetup completes. */
  MPI_Comm            comm;        /**< Communicator used during setup. */
} DMPlexGeomCtx;

#endif /* DMPLEX_TYPES_H */
