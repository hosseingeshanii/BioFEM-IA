#ifndef ACTIVE_STRAIN_H
#define ACTIVE_STRAIN_H

/** @brief Element-level callback signature used by generic update loops. */
typedef PetscErrorCode (*ElemFunc)(FE *fem, PetscInt ec);

/** @brief Set 1D Gauss-Legendre points and weights for active-strain integration. */
PetscErrorCode SetGaussianQuadrature(FE *fem);

/** @brief Read active-strain runtime options from PETSc command-line parameters. */
PetscErrorCode GetUserActParams(FE *fem);

/** @brief Allocate active-strain arrays for all elements and quadrature points. */
PetscErrorCode ActDataAllocate(FE *fem);

/** @brief Free all active-strain data allocated in ActDataAllocate(). */
PetscErrorCode ActDataDestroy(FE *fem);

/** @brief Initialize active-strain module (options, storage, and quadrature). */
PetscErrorCode InitActStrainProblem(FE *fem, PetscInt ibi);

/** @brief Update reference midsurface geometry cache for one element. */
PetscErrorCode ElemUpdateGeom0Subdiv(FE *fem, PetscInt ec);

/** @brief Update current midsurface geometry cache for one element. */
PetscErrorCode ElemUpdateGeomSubdiv(FE *fem, PetscInt ec);

/** @brief Build current covariant/contravariant basis vectors for one element. */
PetscErrorCode ElemUpdateG(FE *fem, PetscInt ec);

/** @brief Compute active deformation gradient for one element. */
PetscErrorCode ElemActDefGrad(FE *fem, PetscInt ec);

/** @brief Compute right Cauchy-Green tensor and its inverse for one element. */
PetscErrorCode ElemCGDefTens(FE *fem, PetscInt ec);

/** @brief Compute elastic right Cauchy-Green tensor for one element. */
PetscErrorCode ElemElasCGDefTens(FE *fem, PetscInt ec);

/** @brief Compute elastic second Piola-Kirchhoff stress for one element. */
PetscErrorCode ElemElasStress(FE *fem, PetscInt ec);

/** @brief Assemble total (active + elastic) stress for one element. */
PetscErrorCode ElemTotStress(FE *fem, PetscInt ec);

/** @brief Compute elastic tangent tensor for one element. */
PetscErrorCode ElemElsTangMatTens(FE *fem, PetscInt ec);

/** @brief Compute total tangent tensor for one element. */
PetscErrorCode ElemTotTangMatTens(FE *fem, PetscInt ec);

/** @brief Update through-thickness C33 contribution for one element. */
PetscErrorCode ModElemC33(FE *fem, PetscInt ec, PetscReal *delta);

/** @brief Build reference covariant/contravariant basis vectors for one element. */
PetscErrorCode ElemUpdateG0(FE *fem, PetscInt ec);

/** @brief Solve local thickness equation (C33) for one element. */
PetscErrorCode ElemC33Solve(FE *fem, PetscInt ec);

/** @brief Assemble element internal force vector into Fb_out. */
PetscErrorCode ElemUpdFint(FE *fem, PetscInt ec, PetscReal *Fb_out);

/** @brief Run per-element precomputations before global internal-force assembly. */
PetscErrorCode FInternalPreCalc(FE *fem);

/** @brief Assemble global internal force vector for active-strain mechanics. */
PetscErrorCode FInternalAct(FE *fem);

/** @brief Apply an element callback to all mesh elements. */
PetscErrorCode UpdateElements(FE *fem, ElemFunc func);


#endif  /* ACTIVE_STRAIN_H */
