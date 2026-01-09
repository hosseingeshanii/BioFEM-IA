#ifndef VARIABLES_H
#define VARIABLES_H

#include  <petscsnes.h>
#include  <math.h>
#include  <petscvec.h>
#include  "act_variables.h"

typedef struct {
  PetscInt     Node;
  struct node  *next;
} node;

typedef struct {
  node  *head;
} List;


typedef struct {
  PetscReal  x, y;
} Cpt2D;

typedef struct {
  PetscReal  c, A1, A2, A3, A4, A5, A6;  //Fung Model Coeff
} FungC;

typedef struct {
  PetscReal  gamma, a_1, a_2; 
} MuscleActParams;

typedef struct {
  MuscleActParams muscle_act_params;
} UserCtx;

// typedef struct {

//   struct Cmpnts g_e[3];     // Intermediate covariant basis vectors after muscle activation for each element
//   struct Cmpnts g_0[5][3];  // undeformed configuration basis vecs for 5 points through the thickness

//   struct Cmpnts act_fib;
//   PetscReal     Fa[3][3];   // Active deformation gradient per element  
//   PetscReal     Fbar[5][3][3];   // Inverse Active deformation gradient tensor per element 
//   PetscReal     k_e[3];          // Intermediate curvature after muscle activation for each element                            
//   PetscReal     S_m[5][3];  // S_m: membrane elastic stress,  
//   PetscReal     S_b[5][3];  // S_b: bending elastic stress,                                                 
//   PetscReal     S_e[5][3];  // S_e: elastic stress,
//   PetscReal     S[5][3];    // S: total stress in final configuration    
//                             // up to 5 points through the thickness we can keep stress components. 

// } ElemActData;

// typedef struct {
//   ElemActData   *elem_act_data; 
//   Vec           g_e_target;     // Vec form of g_e used in tao solver  
//   PetscScalar   theta[5], w[5];
// } ActData;



typedef struct {
  PetscInt       n_v, n_elmt, n_edge, sum_n_bnodes, n_ghosts, ibi;
  PetscReal      *x_bp, *y_bp, *z_bp, *x_bp0, *y_bp0, *z_bp0;

  PetscReal      *x_bpi, *y_bpi, *z_bpi;    // intermediate coords when muscle activation is on.

  PetscReal      *p4x, *p4y, *p4z, *p5x, *p5y, *p5z, *p6x, *p6y, *p6z;
  PetscReal      *p4x0, *p4y0, *p4z0, *p5x0, *p5y0, *p5z0, *p6x0, *p6y0, *p6z0;
  PetscReal      *kve0, *kve;
  PetscInt       *nv1, *nv2, *nv3, *nv4, *nv5, *nv6;
  struct Cmpnts  *n_fib;
  PetscInt       *bnodes, *n_bnodes, *belmts, *edgefrontnodes, *edgefrontnodesI, *belmtsedge;
  PetscReal	     *nf_x, *nf_y, *nf_z, *Nf_x, *Nf_y, *Nf_z;  // Normal direction
  PetscReal      *dA, *dA0, *m, rho, h0; 
  
  PetscReal      **El;
  PetscReal      **E_epsilon;
  
  PetscReal      **Fung_epsilons;  
  PetscReal      **Fung_coeffs;
  PetscReal      **Fung_coeffs_smth;      //smoothed fiber direction vector stored in this array
  
  PetscInt       **neigh_nodes_ind;       //2D array of size [n_v*10] containing index of all neighbouring nodes around a node 

  PetscReal      **Adam_mestimate;  // Adam optimizer first moment estimate
  PetscReal      **Adam_vestimate;  // Adam optimizer second moment estimate

  PetscInt       *ire, *irv, *val, *patch; //for subdivision surface method
  PetscReal      *G, *G1, *G2, *g1, *g2, *g3, *g1n, *g2n, *g3n;
  struct Cmpnts  *qvec;
  PetscReal      *radvec; 
  PetscInt	     *contact; 
  Vec            vis; 
} IBMNodes;

typedef struct {
  Vec        Res, x, Fext, Fint, Fdyn, disp, FJ, dis, Fcnt;
  Vec        x_intmd;     // intermediate configuration after activation.
  Vec        Res_smth;    //smooth residual 
  Vec        xn, xnm1, xd, xdd, dx, y, yn;  //y,yn are for Runge Kutta y=dx/dt
  PetscReal  *StrainM, *StressM, *StrainB, *StressB, *IE, *CE, *FC, *KE;
  Vec        Mass, Dissip;
  // Added
  PetscReal  E, mu, rho, h0, dt, dampfactor, char_length_x, char_length_y, char_length_z;
  PetscInt   dof, twod, damping, membrane, bending, outghost, ConstitutiveLawNonLinear;
  PetscInt   timeinteg, nbody, contact, explicit;
  PetscInt   ec, nc, ti, tiout, tistart, rstart_flg, tisteps, curvature, manufactured, residual;
  PetscInt   ITER;
  PetscReal  ****dR_dE;
  PetscReal  ****Jac_Fung;
  Mat        Jacobian; 
  Mat        J_Seq;
  ActData    act_data;
  IBMNodes   *ibm;
  UserCtx    userctx;
} FE;

#endif