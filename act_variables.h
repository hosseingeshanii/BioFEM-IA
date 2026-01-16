#ifndef ACT_VARIABLES_H
#define ACT_VARIABLES_H


#include <petscsys.h>   /* PetscInt, PetscReal, PetscScalar */
#include "types.h"

/*------------------------------------------------------------------------------
 *  Naming convention:
 *    Cov   : covariant components
 *    Cont  : contravariant components
 *    bar   : intermediate (activated) configuration
 *    e     : elastic part
 *    0     : reference configuration
 *----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 *  2nd-order tensor stored in covariant and contravariant forms
 *----------------------------------------------------------------------------*/
typedef struct {
    PetscReal Cov[3][3];
    PetscReal Cont[3][3];
} Elem2DTens;

/*------------------------------------------------------------------------------
 *  4th-order tensor (e.g. tangent material)
 *----------------------------------------------------------------------------*/
typedef struct {
    PetscReal Cont[3][3][3][3];
} Elem4DTens;

/*------------------------------------------------------------------------------
 *  Vector stored in covariant and contravariant forms
 *----------------------------------------------------------------------------*/
typedef struct {
    Cmpnts Cov[3];
    Cmpnts Cont[3];
} ElemVec;

typedef struct {
    Cmpnts ndx21, ndx31, nn;
    Cmpnts gc1, gc2;
    Cmpnts Aaa, Abb, Aab;
    PetscReal k[3];
    PetscReal nA;
    PetscInt  v;    /* valence, useful for irregular */
    PetscInt  nob;  /* no ghost missing */
} SubdivGeomQP;


/*------------------------------------------------------------------------------
 *  Element-level activation data
 *----------------------------------------------------------------------------*/
typedef struct {
    /* Kinematics */
    Elem2DTens   *Fa, *Fa_inv;      /* Active deformation gradient */
    Elem2DTens   *C, *C_inv;       /* Total right Cauchy–Green tensor */
    Elem2DTens   *Ce, *Ce_inv;      /* Elastic right Cauchy–Green tensor */

    /* Stress */
    Elem2DTens   *Se;      /* Elastic second Piola–Kirchhoff stress */
    Elem2DTens   *S;       /* Total second Piola–Kirchhoff stress */

    /* Metric tensors */
    Elem2DTens   *gm;      /* Current configuration metric */
    Elem2DTens   *gm0;     /* Reference configuration metric */

    /* Material tangents */
    Elem4DTens   *CCe;     /* Elastic material tangent */
    Elem4DTens   *CC;      /* Total material tangent */

    /* Basis vectors */
    ElemVec      *g;       /* Current basis vectors */
    ElemVec      *g0;      /* Reference basis vectors */

    SubdivGeomQP *geom;    /* size n_qp */

} ElemActData;

/*------------------------------------------------------------------------------
 *  Activation-related global data
 *----------------------------------------------------------------------------*/
typedef struct {
    PetscInt       n_qp;             /* Number of quadrature points */
    PetscScalar   *theta;            /* Quadrature locations */
    PetscScalar   *w;                /* Quadrature weights */

    ElemActData   *elem_act_data;
} ActData;

#endif 
