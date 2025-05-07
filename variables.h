#include  <petscsnes.h>
#include  <math.h>
#include  <petscvec.h>
#include <Python.h>


typedef struct {
  PetscInt     Node;
  struct node  *next;
} node;

typedef struct {
  node  *head;
} List;

struct  Cmpnts{
  PetscReal  x, y, z;
};

typedef struct {
  PetscReal  x, y;
} Cpt2D;

typedef struct {
  PetscReal  c, A1, A2, A3, A4, A5, A6;  //Fung Model Coeff
} FungC;


typedef struct {
  PetscInt       n_v, n_elmt, n_edge, sum_n_bnodes, n_ghosts, ibi;
  PetscReal      *x_bp, *y_bp, *z_bp, *x_bp0, *y_bp0, *z_bp0;
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
  PetscInt	 *contact; 
  Vec        vis; 
} IBMNodes;

typedef struct {
  Vec        Res, x, Fext, Fint, Fdyn, disp, FJ, dis, Fcnt;
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
  PyObject   *pModule;
  PetscReal  ****dR_dE;
  PetscReal  ****Jac_Fung;
  Mat        Jacobian; 
  Mat        J_Seq;
  IBMNodes   *ibm;
} FE;
