#include  "variables.h"

extern PetscReal  E, mu, rho, h0;
extern PetscInt   dof, ConstitutiveLawNonLinear, curvature;

extern struct Cmpnts PLUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal DOT(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts UNIT(struct Cmpnts v1);
extern PetscReal SIZE(struct Cmpnts v1);
extern struct Cmpnts AMULT(PetscReal alpha, struct Cmpnts v1);
extern PetscErrorCode INV(PetscReal T[3][3], PetscReal _Tinv[3][3]);
extern PetscErrorCode MATMULT(PetscReal A[][2], PetscReal B[][2], PetscReal C[][2]);
extern PetscErrorCode StressLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm);
extern PetscErrorCode StressNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], IBMNodes *ibm);
extern PetscErrorCode CalcCurvStressStrainxyz(PetscInt ec, PetscReal k[3], PetscReal strain[3], PetscReal stress[3], PetscInt method, PetscInt mb, FE *fem);


PetscErrorCode Fmembrane(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, struct Cmpnts x1, struct Cmpnts x2, struct Cmpnts x3, PetscReal _Fm[9], FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  PetscReal      sum, A0;
  int            i, j, m;
  struct Cmpnts  dx21, dx31, dX21, dX31;
  
  dx21 = MINUS(x2,x1);  dx31 = MINUS(x3,x1); //dx21:g1 , dx31:g2
  dX21 = MINUS(X2,X1);  dX31 = MINUS(X3,X1);  //dX21:G1 , dX31:G2
  //Compute E
  PetscReal  Em[3], S[3];
  PetscReal  g11, g12, g22, G11, G12, G22;
  g11 = DOT(dx21, dx21);  g12 = DOT(dx21, dx31);   g22 = DOT(dx31, dx31); 
  G11 = DOT(dX21, dX21);  G12 = DOT(dX21, dX31);   G22 = DOT(dX31, dX31); 
  
  Em[0] = 0.5*(g11 - G11);  Em[1] = 0.5*(g22 - G22);  Em[2] = g12 - G12;
  
  //compute B for membrane
  PetscReal  Bm[3][9];
  Bm[0][0] = -dx21.x;  Bm[0][1] = -dx21.y;  Bm[0][2] = -dx21.z;  Bm[0][3] = dx21.x;  Bm[0][4] = dx21.y;  Bm[0][5] = dx21.z;  Bm[0][6] = 0.;  Bm[0][7] = 0.;  Bm[0][8] = 0.;
  Bm[1][0] = -dx31.x;  Bm[1][1] = -dx31.y;  Bm[1][2] = -dx31.z;  Bm[1][3] = 0.;  Bm[1][4] = 0.;  Bm[1][5] = 0.;  Bm[1][6] = dx31.x;  Bm[1][7] = dx31.y;  Bm[1][8] = dx31.z;
  Bm[2][0] = -(dx21.x + dx31.x);  Bm[2][1] = -(dx21.y + dx31.y);  Bm[2][2] = -(dx21.z + dx31.z);  Bm[2][3] = dx31.x;  Bm[2][4] = dx31.y;  Bm[2][5] = dx31.z;  Bm[2][6] = dx21.x;  Bm[2][7] = dx21.y;  Bm[2][8] = dx21.z;

  A0 = ibm->dA0[ec];
  
  if(ConstitutiveLawNonLinear) {
    MembraneNonLinear(ec, X1, X2, X3, Em, S, ibm);   
  }else{
    StressLinear(ec, X1, X2, X3, Em, S, 0, ibm);
  }

  CalcCurvStressStrainxyz(ec, Em, Em, S, 0, 0, fem);
  
  for (i=0; i<9; i++) {
    sum = 0.0;
    for (m=0; m<3; m++) {
      sum += Bm[m][i]*S[m]*A0*h0;
    }
    _Fm[i] = sum;
  }
  
  return(0);
}

