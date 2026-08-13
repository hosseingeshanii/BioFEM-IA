#ifndef DMPLEX_GEOM_H
#define DMPLEX_GEOM_H

#include "variables.h"

/**
 * @brief One-time DMPlex geometry setup.
 *
 * Builds the distributed DM, patch stencil layout, and coordinate-exchange SF
 * and stores them in fem->geom_ctx.  Also evaluates reference-configuration
 * geometry (geom0) for all locally owned cells — this never changes and is
 * computed only here.
 *
 * Idempotent: if fem->geom_ctx.initialized is already PETSC_TRUE the function
 * returns immediately without rebuilding anything.
 *
 * @param fem   FE struct whose geom_ctx will be populated.
 * @param comm  MPI communicator for the distributed DM (use PETSC_COMM_WORLD
 *              for parallel geometry, PETSC_COMM_SELF for serial/rank-0-only).
 */
PetscErrorCode FEM_DMPlexGeomSetup(FE *fem, MPI_Comm comm);

/**
 * @brief Per-SNES-iteration geometry update.
 *
 * Synchronizes current vertex coordinates across MPI ranks via the stored
 * PetscSF, then evaluates current-configuration geometry (geom) for all
 * locally owned cells.  Must be called after ibm->x_bp has been updated for
 * this rank's owned vertices.
 *
 * Requires FEM_DMPlexGeomSetup to have been called first.
 */
PetscErrorCode FEM_DMPlexGeomUpdate(FE *fem);

/**
 * @brief Destroy the persistent DMPlex geometry context in fem->geom_ctx.
 *
 * Frees the DM, PetscSF, all layout/exchange arrays, and origVert/ownerRank.
 * Safe to call on an uninitialized context (no-op).
 */
PetscErrorCode FEM_DMPlexGeomDestroy(FE *fem);

/**
 * @brief Build ibm_to_global_dof0 mapping in fem->geom_ctx.
 *
 * Maps each ibm vertex index v to its first global Vec DOF index in the
 * parallel Vec created by InitVecs().  Must be called after InitVecs() so
 * that Vec ownership ranges are available.
 *
 * @param fem  FE struct with geom_ctx.initialized and a valid fem->Fint Vec.
 */
PetscErrorCode FEM_DMPlexGeomBuildNodeMap(FE *fem);

/**
 * @brief Sync ghost/auxiliary node DOFs (mirror-reflected boundary points)
 * into the solved Vecs: writes their position (from ibm->x_bp, already
 * updated by GlobalGhost()) into @p x and zeroes @p R there, so
 * SNES never tries to solve for them as independent unknowns. No-op if
 * ibm->n_ghosts == 0. Must be called every residual evaluation, after
 * force/residual assembly and before VecNorm.
 *
 * @param fem  FE struct with initialised geom_ctx.
 * @param x    Current solution Vec.
 * @param R    Residual Vec.
 */
PetscErrorCode FEM_DMPlexGeomSyncGhostDOFs(FE *fem, Vec x, Vec R);

/* ---- Legacy / benchmark entry points ---- */

/**
 * @brief Benchmark: run the full distributed DMPlex geometry pipeline for all bodies.
 *
 * Performs one-time setup and repeats the geometry kernel n times for timing.
 * Does NOT use fem->geom_ctx — manages its own local setup and teardown.
 */
PetscErrorCode RunDMPlexGeomProcesses(FE *fem);

/**
 * @brief Drop-in geometry call: setup + update + destroy in one shot.
 *
 * Wraps FEM_DMPlexGeomSetup / FEM_DMPlexGeomUpdate / FEM_DMPlexGeomDestroy for
 * call sites that do not persist the context across iterations.  The comm
 * argument controls parallelism: PETSC_COMM_SELF = serial (rank-0 only),
 * PETSC_COMM_WORLD = fully distributed.
 *
 * @deprecated Prefer storing the context in FE and calling Setup once at init,
 *             then Update each iteration.  This function rebuilds the DM and SF
 *             on every call which is expensive for many timesteps.
 */
PetscErrorCode RunDMPlexGeomSubdiv(FE *fem, MPI_Comm comm);

#endif /* DMPLEX_GEOM_H */
