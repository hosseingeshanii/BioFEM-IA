#ifndef ACT_VARIABLES_H
#define ACT_VARIABLES_H

#include <petscsys.h>
#include "math.h"

/** @file act_variables.h
 *  @brief Data containers for active-strain kinematics, stress, and material data.
 *
 *  Naming convention:
 *  - Cov: covariant components
 *  - Cont: contravariant components
 *  - bar: intermediate (activated) configuration
 *  - e: elastic part
 *  - 0: reference configuration
 */

/** @brief Second-order tensor in covariant and contravariant forms. */
typedef struct {
    PetscReal Cov[3][3];   /**< Covariant components. */
    PetscReal Cont[3][3];  /**< Contravariant components. */
} Elem2DTens;

/** @brief Fourth-order tensor (for example, material tangent). */
typedef struct {
    PetscReal Cont[3][3][3][3];  /**< Contravariant components. */
} Elem4DTens;

/** @brief Vector in covariant and contravariant forms. */
typedef struct {
    Cmpnts Cov[3];   /**< Covariant components. */
    Cmpnts Cont[3];  /**< Contravariant components. */
} ElemVec;

/** @brief Cached subdivision-surface geometry quantities at one quadrature point. */
typedef struct {
  PetscInt  nen;          /**< Number of element nodes: 12 for regular, v+6 for irregular. */
  PetscInt  v;            /**< Valence for irregular elements, 0 for regular. */
  PetscInt  is_irregular; /**< Flag: 0 regular, 1 irregular. */

  Cmpnts ndx21, ndx31, nn; /**< First derivatives and surface normal. */

  Cmpnts gc1, gc2; /**< Contravariant in-surface basis vectors. */

  Cmpnts Aaa, Abb, Aab; /**< Second derivative combinations used in bending terms. */

  PetscReal *INa0;   /**< For irregular elements: INa[0][i], length nen. */
  PetscReal *INa1;   /**< For irregular elements: INa[1][i], length nen. */
  PetscReal *INab0;  /**< For irregular elements: INab[0][i], length nen. */
  PetscReal *INab1;  /**< For irregular elements: INab[1][i], length nen. */
  PetscReal *INab2;  /**< For irregular elements: INab[2][i], length nen. */
} SubdivGeomQP;

/** @brief Element-level active-strain tensors, stresses, and cached basis/geometry. */
typedef struct {
    Elem2DTens   *Fa, *Fa_inv; /**< Active deformation gradient and inverse. */
    Elem2DTens   *C, *C_inv;   /**< Total right Cauchy-Green tensor and inverse. */
    Elem2DTens   *Ce, *Ce_inv; /**< Elastic right Cauchy-Green tensor and inverse. */

    Elem2DTens   *Se; /**< Elastic second Piola-Kirchhoff stress. */
    Elem2DTens   *S;  /**< Total second Piola-Kirchhoff stress. */

    Elem2DTens   *gm;  /**< Metric tensor in current configuration. */
    Elem2DTens   *gm0; /**< Metric tensor in reference configuration. */

    Elem4DTens   *CCe; /**< Elastic material tangent. */
    Elem4DTens   *CC;  /**< Total material tangent. */

    ElemVec      *g;  /**< Basis vectors in current configuration. */
    ElemVec      *g0; /**< Basis vectors in reference configuration. */

    SubdivGeomQP *geom;  /**< Size-1 midsurface cache for current configuration. */
    SubdivGeomQP *geom0; /**< Size-1 midsurface cache for reference configuration. */

} ElemActData;

/** @brief Active-muscle model parameters. */
typedef struct {
  PetscReal  gamma; /**< Activation magnitude. */
  PetscReal  a_1;   /**< Fiber-direction anisotropy parameter 1. */
  PetscReal  a_2;   /**< Fiber-direction anisotropy parameter 2. */
} MuscleActParams;

/** @brief Global active-strain data stored in FE context. */
typedef struct {
    PetscInt        n_qp;              /**< Number of through-thickness quadrature points. */
    PetscScalar     *theta;            /**< Quadrature locations. */
    PetscScalar     *w;                /**< Quadrature weights. */
    PetscReal       mu;                /**< Shear modulus. */
    PetscReal       K;                 /**< Bulk modulus. */
    PetscInt        C33_subitr_nums;   /**< Iteration count for local C33 solve. */
    MuscleActParams muscle_act_params; /**< Muscle-activation parameters. */
    ElemActData     *elem_act_data;    /**< Per-element activation cache. */
} ActData;

#endif
