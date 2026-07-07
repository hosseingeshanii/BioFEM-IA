# BioFEM-IA Parallelism Architecture

This document tracks which data lives on which MPI ranks, which functions execute
where, and what MPI/PetscSF communication happens.  It covers the **current state**
and the **Route B target** (fully parallel SNES).

---

## 1. Key data structures

### IBMNodes  (`include/variables.h`)

The raw surface mesh.  All geometric and topological arrays live here.

| Field group | Contents | Current rank ownership |
|---|---|---|
| `n_v`, `n_elmt`, `n_edge` | mesh sizes | **all ranks** |
| `nv1/nv2/nv3` | triangle connectivity | **all ranks** |
| `x_bp0/y_bp0/z_bp0` | reference coordinates | **all ranks** |
| `x_bp/y_bp/z_bp` | current coordinates | **rank 0 only** (updated in `FormFunctionFEM`) |
| `x_bpi/y_bpi/z_bpi` | intermediate (post-activation) coords | rank 0 only |
| `ire/irv/val/patch` | Loop subdivision stencil | **all ranks** |
| `n_fib` | fiber direction vectors | rank 0 only |
| `dA/dA0` | area elements | rank 0 only |
| `n_ghosts` | ghost node count beyond `n_v` | rank 0 only |

**Why all ranks have topology**: `Dimension`, `Create`, `Input` (standard mesh path)
are called outside any `if (rank == 0)` guard.  Every rank independently reads the
mesh files.  The `ibm` struct is **replicated** — not partitioned — across ranks.

---

### FE  (`include/variables.h`)

The finite-element context.  One per body.  Embeds `IBMNodes *ibm` and `ActData act_data`.

| Field | Contents | Current rank ownership |
|---|---|---|
| `x`, `Res`, `Fint`, `Fext`, `Fdyn` | PETSc Vecs (size `n_v * dof`) | **rank 0 only** (`PETSC_COMM_SELF`) |
| `xn`, `xnm1`, `xd`, `xdd`, `dx`, `y`, `yn` | time-integration Vecs | rank 0 only |
| `Mass`, `Dissip` | explicit-scheme Vecs | rank 0 only |
| `Jacobian`, `J_Seq` | tangent matrices | rank 0 only |
| `StrainM/StressM/...` | scalar energy arrays | rank 0 only |
| `act_data` | active-strain per-element cache | rank 0 only (muscle_activation path) |
| `ibm` | pointer to IBMNodes | all ranks (pointer valid, fields as above) |

---

### ActData / ElemActData  (`include/act_variables.h`)

Per-element active-strain tensors and cached geometry.  Lives inside `FE.act_data`.

| Field | Contents | When populated | Current rank |
|---|---|---|---|
| `n_qp`, `theta`, `w` | quadrature scheme | `InitActStrainProblem` | rank 0 |
| `muscle_act_params` | gamma, a_1, a_2 | `InitActStrainProblem` | rank 0 |
| `elem_act_data[ec]` | per-element structs | `InitActStrainProblem` (allocates), then per-iteration | rank 0 |
| `elem_act_data[ec].geom0` | reference midsurface geometry (SubdivGeomQP) | `ElemUpdateGeom0Subdiv` — reference config, **never changes after first computation** | rank 0 |
| `elem_act_data[ec].geom` | current midsurface geometry (SubdivGeomQP) | `ElemUpdateGeomSubdiv` — updated every SNES iteration | rank 0 |
| `elem_act_data[ec].gm0` | reference metric tensor | `ElemUpdateG` | rank 0 |
| `elem_act_data[ec].gm` | current metric tensor | `ElemUpdateG` | rank 0 |
| `elem_act_data[ec].Fa` | active deformation gradient | `ElemActDefGrad` | rank 0 |
| `elem_act_data[ec].C` | right Cauchy-Green tensor | `ElemCGDefTens` | rank 0 |
| `elem_act_data[ec].S` | 2nd Piola-Kirchhoff stress | `ElemTotStress` | rank 0 |

---

### DMPlexPatchLayout / DMPlexPatchExchange  (`src/dmplex_geom.c`, internal)

Temporary structs built inside `RunDMPlexGeomSubdiv` / `RunDMPlexGeomProcesses`.
Currently **rebuilt and destroyed on every call**.  In Route B these must be
persistent in `FE`.

| Struct | Contents | Scope |
|---|---|---|
| `DMPlexPatchLayout` | local copy of `ire/val/patch` for owned cells; `orig_cell[lc]` mapping | per-rank local |
| `DMPlexPatchExchange` | PetscSF for cross-rank coordinate fetch; receive buffers `x/y/z_remote` | per-rank local |

---

## 2. Current execution model

```
MPI_Init
  ALL ranks: Dimension / Create / Input   →  ibm replicated on all ranks

  rank 0 only:  Init(fem)                 →  PETSc Vecs created (PETSC_COMM_SELF)
  rank 0 only:  InitActStrainProblem      →  act_data allocated

  if (dmplex_geom_process):
    ALL ranks: InitActStrainProblem       →  act_data on all ranks (benchmark only)
    ALL ranks: RunDMPlexGeomProcesses     →  distributed benchmark, then exit

  if (muscle_activation):

    rank 0 only: TIME LOOP  ────────────────────────────────────────────────────┐
      rank 0:   SNESCreate(PETSC_COMM_SELF, &snes)                              │
      rank 0:   SNESSolve(snes, U)                                              │
        rank 0:   FormFunctionFEM                                                │
          rank 0:   x_bp[nv] = xx[nv*dof]   (all n_v nodes from serial Vec)    │
          rank 0:   AreaNormal, PatchLoc, GhostLoc                              │
          rank 0:   FInternalAct                                                 │
            rank 0:   FInternalPreCalc                                           │
              rank 0:   RunDMPlexGeomSubdiv(fem, PETSC_COMM_SELF)               │
                          ↳ builds DM with all n_elmt cells on rank 0           │
                          ↳ UpdatePatchRemoteCoordinates_ (no-op, 1 rank)       │
                          ↳ ElemUpdateGeom0Subdiv + ElemUpdateGeomSubdiv        │
                            for all elements on rank 0                           │
              rank 0:   UpdateElements(ElemUpdateG)        [all elements]        │
              rank 0:   UpdateElements(ElemActDefGrad)     [all elements]        │
              rank 0:   UpdateElements(ElemCGDefTens)      [all elements]        │
              rank 0:   UpdateElements(ElemC33Solve)       [all elements]        │
            rank 0:   ElemUpdFint + direct FF[node] += scatter                  │
                                                                                 │
  non-rank-0: IDLE after init ─────────────────────────────────────────────────┘
```

**Non-rank-0 ranks do nothing after initialization in the muscle_activation path.**
They hold a full copy of `ibm` topology but have no `act_data`, no PETSc Vecs, and
never enter the time loop.

---

## 3. Route B target: fully parallel SNES

### The core idea

- Remove the `if (rank == 0)` guard on the time loop.
- Change `SNESCreate(PETSC_COMM_SELF, ...)` → `SNESCreate(PETSC_COMM_WORLD, ...)`.
- `fem->x` becomes a **distributed parallel Vec** — each rank owns a contiguous
  block of `n_owned_verts * dof` DOFs.
- Every rank participates in `FormFunctionFEM` and all functions below it.
- `RunDMPlexGeomSubdiv(fem, PETSC_COMM_WORLD)` naturally has all ranks present.

### Data ownership in Route B

| Data | Route B ownership |
|---|---|
| `ibm` topology | still replicated (unchanged) |
| `ibm->x_bp0` (ref coords) | still replicated (read from file on all ranks) |
| `ibm->x_bp` (current coords) | each rank stores full array; only owned-vertex entries are current; SF fills the rest |
| `fem->x` Vec | distributed: rank `r` owns `n_r * dof` entries |
| `act_data` / `elem_act_data` | replicated on all ranks (init on all ranks) |
| DMPlex layout + exchange | per-rank local subset; **persistent in FE, not rebuilt each iteration** |

### Route B execution flow

```
ALL ranks: Dimension / Create / Input     →  ibm replicated (unchanged)
ALL ranks: Init(fem)                      →  parallel Vecs (PETSC_COMM_WORLD)
ALL ranks: InitActStrainProblem           →  act_data on all ranks

TIME LOOP — all ranks ─────────────────────────────────────────────────────────┐
  ALL ranks: SNESCreate(PETSC_COMM_WORLD, &snes)                               │
  ALL ranks: SNESSolve                                                          │
    ALL ranks: FormFunctionFEM                                                   │
      ALL ranks: x_bp[owned_node] = xx[local_idx * dof]   ← LOCAL Vec array   │
      ALL ranks: AreaNormal (replicated ibm, each rank does same work)          │
      ALL ranks: FInternalAct                                                    │
        ALL ranks: FInternalPreCalc                                              │
          ALL ranks: RunDMPlexGeomSubdiv(fem, PETSC_COMM_WORLD)                 │
            UpdatePatchRemoteCoordinates_   ← PetscSFBcast fills remote x_bp   │
            ElemUpdateGeom0Subdiv(ec)  ┐                                        │
            ElemUpdateGeomSubdiv(ec)   ├ for LOCAL cells only (nLocalCells)     │
            ElemUpdateG(ec)            ┘                                        │
          ALL ranks: (remaining UpdateElements over local cells only)           │
        ALL ranks: ElemUpdFint → VecSetValues → VecAssemblyBegin/End           │
│──────────────────────────────────────────────────────────────────────────────┘
```

### How coordinate synchronization works in Route B

```
SNES Vec x (distributed)
   rank 0: xx[0..n0*dof-1]  →  ibm->x_bp[v0..v0+n0-1]   (owned vertices)
   rank 1: xx[0..n1*dof-1]  →  ibm->x_bp[v1..v1+n1-1]   (owned vertices)
   rank 2: ...

           UpdatePatchRemoteCoordinates_  (inside RunDMPlexGeomSubdiv)
           PetscSFBcast: reads ibm->x_bp[v] from ownerRank[v]
                         writes to x_remote[i] on ranks that need vertex v
           write-back:   ibm->x_bp[remote_node] = x_remote[i]

After PetscSFBcast: each rank's ibm->x_bp is complete for all vertices
referenced by its patch stencils.  No full MPI_Bcast of x_bp needed.
```

The PetscSF was designed for exactly this: the `nroots = ibm->n_v` root layout
means each rank provides its owned vertex coordinates; the SF moves only the
cross-rank entries that patch stencils actually need.

---

## 4. Function-by-function rank table

### Initialization (main.c)

| Function | Current | Route B |
|---|---|---|
| `Dimension / Create / Input` | all ranks | all ranks (unchanged) |
| `Init(fem)` | rank 0 | all ranks |
| `InitActStrainProblem` | rank 0 (muscle) / all (dmplex benchmark) | all ranks |

### Time loop (main.c)

| Function | Current | Route B |
|---|---|---|
| `SNESCreate` | rank 0, `PETSC_COMM_SELF` | all ranks, `PETSC_COMM_WORLD` |
| `SNESSolve` | rank 0 | all ranks |
| `FormFunctionFEM` | rank 0 | all ranks |
| `AreaNormal`, `PatchLoc`, `GhostLoc` | rank 0 | all ranks (replicated ibm, same result) |
| `FInternalAct` | rank 0 | all ranks |

### FInternalPreCalc (active_strain.c)

| Function | Current | Route B |
|---|---|---|
| `RunDMPlexGeomSubdiv` | rank 0, `PETSC_COMM_SELF` | all ranks, `PETSC_COMM_WORLD` |
| `ElemUpdateGeom0Subdiv(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| `ElemUpdateGeomSubdiv(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| `ElemUpdateG(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| `ElemActDefGrad(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| `ElemCGDefTens(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| `ElemC33Solve(ec)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |

### FInternalAct assembly (active_strain.c)

| Operation | Current | Route B |
|---|---|---|
| `ElemUpdFint(ec, Fb)` | rank 0, all `n_elmt` ec | all ranks, LOCAL cells only |
| scatter into `Fint` | `FF[node] += Fb[i]` raw array | `VecSetValues` + `VecAssemblyBegin/End` |

### DMPlex setup (dmplex_geom.c)

| Function | Called by | Current | Route B |
|---|---|---|---|
| `BuildSerialDMPlex_` | `CreateDistributedDMPlex_` | all ranks; reads ibm on rank 0 only | unchanged |
| `DMPlexDistribute` | `CreateDistributedDMPlex_` | all ranks | unchanged |
| `BuildOriginalVertexOwnerTable_` | `RunDMPlexGeomSubdiv` | all ranks | unchanged |
| `DMPlexPatchLayoutCreate_` | `RunDMPlexGeomSubdiv` | all ranks | unchanged |
| `BuildPatchCoordinateExchange_` | `RunDMPlexGeomSubdiv` | all ranks | unchanged |
| `UpdatePatchRemoteCoordinates_` | `RunDMPlexGeomSubdiv` | all ranks; no-op (1 rank) | all ranks; real cross-rank Bcast |
| DMPlex setup lifecycle | rebuilt every SNES call | **persistent in FE** (built once at init) |

---

## 5. What changes in code for Route B

### main.c

1. `Init(fem)` — remove `if (rank == 0)` guard; create Vecs with `PETSC_COMM_WORLD`
2. `InitActStrainProblem` for `muscle_activation` — change `muscle_activation && rank == 0`
   to `muscle_activation` (all ranks)
3. Time loop — remove `if (rank == 0)` at line 349
4. `SNESCreate(PETSC_COMM_SELF, ...)` → `SNESCreate(PETSC_COMM_WORLD, ...)`
5. `FormFunctionFEM` coord update loop — change from reading all `n_v` nodes to reading
   LOCAL nodes only (use `origVert[lv]` mapping from DMPlex setup)

### active_strain.c

6. `FInternalPreCalc` — change `RunDMPlexGeomSubdiv(fem, PETSC_COMM_SELF)` →
   `RunDMPlexGeomSubdiv(fem, PETSC_COMM_WORLD)`; use persistent layout/exchange from FE
7. `UpdateElements` calls in `FInternalPreCalc` — loop over local cells only
   (not `0..n_elmt`)
8. `FInternalAct` assembly loop — replace raw `FF[node] += Fb[i]` with
   `VecSetValues` + `VecAssemblyBegin/End`

### dmplex_geom.c / include

9. Move `DMPlexPatchLayout` and `DMPlexPatchExchange` to a public struct stored in `FE`
   (currently private to `dmplex_geom.c`)
10. Add a one-time setup function (e.g. `RunDMPlexGeomSetup`) that builds and stores
    the layout + exchange in `FE`; call it from `InitActStrainProblem`
11. Per-iteration function uses the stored layout + exchange without rebuilding
12. `geom0` (reference geometry) can be computed once during setup and never again;
    only `geom` (current) needs updating each SNES call

---

## 6. Important invariants to preserve

- `BuildSerialDMPlex_` must only read `ibm` on rank 0 — this is correct and must
  not change.  Other ranks pass empty arrays to `DMPlexCreateFromCellListPetsc`.

- The DMPlex partition drives BOTH which elements each rank computes AND which DOFs
  each rank owns in `fem->x`.  These must be consistent so that `origVert[lv]` gives
  the correct index into `ibm->x_bp`.

- `geom0` never changes after the first call to `ElemUpdateGeom0Subdiv`.  Once
  Route B is in place, move its computation into the one-time setup, not the
  per-SNES-iteration path.

- `ibm->x_bp0` (reference coords) is replicated and never modified.  The PetscSF
  for coordinate exchange only operates on `x_bp` (current).

- Ghost nodes (`nv > ibm->n_v`) in patch stencils are flagged by the condition
  `node >= ibm->n_v` in `BuildPatchCoordinateExchange_` and are always available
  locally — they are not subject to cross-rank communication.
