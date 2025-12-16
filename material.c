#include  "variables.h"
#include <math.h>

extern PetscReal  E, mu, rho, h0, dampfactor, c1, c2;
extern PetscInt   dof, ConstitutiveLawNonLinear, n_Fung_Coeffs;
extern PetscInt   dR_dE_flag;
extern PetscInt   ITER;
extern PetscInt   epoch, epoch_output;
extern PetscReal  BETA1, BETA2, EPSILON;
extern PetscInt   ressmooth, fibersmooth;


extern struct Cmpnts  PLUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal  DOT(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  UNIT(struct Cmpnts v1);
extern PetscReal  SIZE(struct Cmpnts v1);
extern PetscErrorCode  INV(PetscReal T[3][3], PetscReal _Tinv[3][3]);
extern PetscErrorCode  MATMULT(PetscReal A[][2], PetscReal B[][2], PetscReal C[][2]);
extern struct Cmpnts  AMULT(PetscReal alpha, struct Cmpnts v1); 
extern PetscErrorCode TRANS(PetscReal A[3][3], PetscReal _AT[3][3]);
extern PetscReal  SIGN(PetscReal a);

 
PetscErrorCode InitMaterial(IBMNodes *ibm) {

  PetscInt  ec, i, j;
  
  // For LV
  struct Cmpnts  Axis, N, Midd, tcyl;  //Axis:Cylinder axis , Midd: middle of element , tcyl:tangent vector to cylinder
  PetscReal      pi=3.14159 , Teta; // Teta: angle of elemet in respect to the y=0 in xy surface
  PetscReal      R[3][3], nfibR[3], ttcyl[3], angle ;  //angle:angle of fiber direction respect to free edge
  PetscInt       n1e, n2e, n3e;
  PetscReal      e=0.03;
  
  Axis.x = 0.;  Axis.y = 0.;  Axis.z = 1;

  for (ec=0; ec<ibm->n_elmt; ec++) {

    /*
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
    N.x = ibm->nf_x[ec];  N.y = ibm->nf_y[ec];  N.z = ibm->nf_z[ec];
    //compute the middle of element in xy plane
    Midd.x = (ibm->x_bp0[n1e] + ibm->x_bp0[n2e] + ibm->x_bp0[n3e])/3.;
    Midd.y = (ibm->y_bp0[n1e] + ibm->y_bp0[n2e] + ibm->y_bp0[n3e])/3.;    
    
    Teta = atan(PetscAbsReal(Midd.y/Midd.x));
    if(Midd.x<0. && Midd.y>0.) {Teta = pi - Teta;}
    if(Midd.x<0. && Midd.y<0.) {Teta = pi + Teta;}
    if(Midd.x>0. && Midd.y<0.) {Teta = 2.*pi - Teta;}
    
    if((Teta>0. && Teta<pi/3.) || (Teta>2.*pi/3. && Teta<pi) || (Teta>4.*pi/3. && Teta<5.*pi/3.) ) {angle = pi - (45./180.)*pi;}
    if((Teta>pi/3. && Teta<2*pi/3.) || (Teta>pi && Teta<4.*pi/3.) || (Teta>5.*pi/3. && Teta<2.*pi) ) {angle = (45./180.)*pi;}
    //if((Teta>2.*pi - e) || (Teta>2.*pi/3. - e && Teta<2.*pi/3. + e) || (Teta>4.*pi/3. - e && Teta<4.*pi/3. + e) ) {angle = pi/2.;}
    
    //angle = -45./180.*pi; //constant fiber direction
    //  FormRotationMatrixAround Normal to element (N) in current mesh it is outward
    R[0][0] = cos(angle) + N.x*N.x*(1 - cos(angle));
    R[0][1] = N.x*N.y*(1 - cos(angle)) - N.z*sin(angle);
    R[0][2] = N.x*N.z*(1 - cos(angle)) + N.y*sin(angle);
    
    R[1][0] = N.x*N.y*(1 - cos(angle)) + N.z*sin(angle);
    R[1][1] = cos(angle) + N.y*N.y*(1 - cos(angle));
    R[1][2] = N.y*N.z*(1 - cos(angle)) - N.x*sin(angle);
    
    R[2][0] = N.z*N.x*(1 - cos(angle)) - N.y*sin(angle);
    R[2][1] = N.z*N.y*(1 - cos(angle)) + N.x*sin(angle);
    R[2][2] = cos(angle) + N.z*N.z*(1 - cos(angle));
    
    tcyl = CROSS(N, Axis);
    ttcyl[0] = tcyl.x;  ttcyl[1] = tcyl.y;  ttcyl[2] = tcyl.z;
   
    //Rotate Vector
    for (i=0; i<3; i++){
      nfibR[i] = 0.;
      for (j=0; j<3; j++){
	nfibR[i] += R[i][j]*ttcyl[j];
      }
    }
    
    //ibm->n_fib[ec].x = nfibR[0]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));
    //ibm->n_fib[ec].y = nfibR[1]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));
    //ibm->n_fib[ec].z = nfibR[2]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));   
    ibm->n_fib[ec].x = nfibR[0];
    ibm->n_fib[ec].y = nfibR[1];
    ibm->n_fib[ec].z = nfibR[2];
    */

    //for biaxial tests
    ibm->n_fib[ec].x = 0.7173;
    ibm->n_fib[ec].y = 0.7173;
    ibm->n_fib[ec].z = 0.0;
  }
  
  return(0);
}

//---------------------------------------------------------------------------------------  
PetscErrorCode Mass(IBMNodes *ibm,PetscInt ec,PetscReal _M[9]) {

  PetscReal  A0; 
  PetscInt   i;

 /* if (ibm->ibi==0) { */
 /*   //rho = 1.;  h0 = 0.003; */
 /*   rho = 0.05; */
 /*  } else { */
 /*   //rho = 10.;  h0 = 0.03; */
 /*   rho = 0.15; */
 /*  } */

  A0=ibm->dA0[ec];
  
  for (i=0; i<9; i++){
    _M[i]=rho*h0*A0/3;
  }      
 
  return(0);
}    
 
//---------------------------------------------------------------------------------------  
PetscErrorCode Damp(IBMNodes *ibm, PetscReal M[9],PetscReal _C[9]) {

  PetscInt i;

  /* if (ibm->ibi==0) { */
  /*   rho = 1.;  h0 = 0.003; */
  /* } else { */
  /*   rho = 10.;  h0 = 0.03; */
  /* } */

  for (i=0; i<9; i++){
    _C[i]=dampfactor*M[i];
    //_C[i]=dampfactor; 
  }
 
  return(0);     
} 

//---------------------------------------------------------------------------------------  
PetscErrorCode MassDamp(FE *fem) {

  IBMNodes   *ibm=fem->ibm;
  PetscReal  A0, M, C;
  PetscReal  *DDissip, *MMass;
  PetscInt   n1e, n2e, n3e, ec;
 
  VecGetArray(fem->Mass, &MMass);
  VecGetArray(fem->Dissip, &DDissip);
  
  for (ec=0; ec<ibm->n_elmt + 2*ibm->n_ghosts; ec++) {
    A0 = ibm->dA0[ec];
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];

    M = rho*h0*A0/3.;
      
    MMass[n1e*dof] += M;
    MMass[n1e*dof+1] += M;
    MMass[n1e*dof+2] += M;
    
    MMass[n2e*dof] += M;
    MMass[n2e*dof+1] += M;
    MMass[n2e*dof+2] += M;

    MMass[n3e*dof] += M;
    MMass[n3e*dof+1] += M;
    MMass[n3e*dof+2] += M;
 
    //C = dampfactor*M;
    C = dampfactor;

    DDissip[n1e*dof] += C;
    DDissip[n1e*dof+1] += C;
    DDissip[n1e*dof+2] += C;
    
    DDissip[n2e*dof] += C;
    DDissip[n2e*dof+1] += C;
    DDissip[n2e*dof+2] += C;

    DDissip[n3e*dof] += C;
    DDissip[n3e*dof+1] += C;
    DDissip[n3e*dof+2] += C;

  }

  VecRestoreArray(fem->Mass, &MMass);
  VecRestoreArray(fem->Dissip, &DDissip);

  return(0);     
}    


PetscErrorCode initFung(FE *fem, PetscReal *initial_value){
  IBMNodes *ibm=fem->ibm;
  for (PetscInt ec=0; ec<ibm->n_elmt; ec++){
    for (int m = 0; m < n_Fung_Coeffs; m++){
      ibm->Fung_coeffs[m][ec] = initial_value[m];
      ibm->Fung_coeffs_smth[m][ec] = initial_value[m];
      if (m==8){
        ibm->Fung_coeffs[m][ec] = 1.0;
      }
      if (m==9){
        ibm->Fung_coeffs[m][ec] = 0.0;
      }
      // if (m==7){
      //   // ibm->Fung_coeffs[m][ec] = -45./180.*3.141592;
      //   // ibm->Fung_coeffs[m][ec] = -45.0/180.*3.141592;
      //   ibm->Fung_coeffs[m][ec] = 0.0;
      // }
    }    
  }  

  return(0);     
}


PetscErrorCode initialize_elasticity(FE *fem, PetscReal initial_elasticity_value, PetscReal initial_poisson){
  IBMNodes *ibm=fem->ibm;
  for (PetscInt ec=0; ec<ibm->n_elmt; ec++){
    ibm->El[0][ec] = initial_elasticity_value;
    ibm->E_epsilon[0][ec] = 0.0;    
    ibm->El[1][ec] = initial_poisson;
    ibm->E_epsilon[1][ec] = 0.0;    
  }

  return(0);
}


//--------------------------------------------------------------------------------------- 
PetscErrorCode StressLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) {
 
  PetscReal      Q[3][3], Dloc[3][3], Strainloc[3], c, sum;
  PetscInt       i, j, m, n;
  PetscReal      q[3][3], xz, xe, yz, ye;
  struct Cmpnts  dX21, dX31, V1, V2, N, diff1, diff2;
  
  //PetscPrintf(PETSC_COMM_SELF, "Check in SL!\n");

  if (method==0) {
    dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2
    N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec];
  } else if (method==1) { 
    dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2];
    dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2];
    N = UNIT(CROSS(dX21, dX31));
  }  
  V1 = UNIT(dX21);
  V2 = CROSS(N, V1);

  xz = DOT(V1, dX21);  xe = DOT(V1, dX31);
  yz = DOT(V2, dX21);  ye = DOT(V2, dX31);

  q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe;
  q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye;
  q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz;

  INV(q, Q);

  // Define Nonlinear E
  //PetscReal p1,p2,p3,XM,ENon,a,I;
  //PetscReal F=1.e-5,w=8.0,L=1.0,b=0.004;
  //a=0.5/L; I=(1./12.)*b*pow(h0,3);

  /*
  PetscReal ENon[3], epsilon[3];
  PetscReal Eavg;
  PetscReal XM, YM;
  XM=(X1.x+X2.x+X3.x)/3.;
  YM=(X1.y+X2.y+X3.y)/3.;
  
  //p1=(rho*b*h0*pow(w,2))/(2.*I); p2=F/(2.*a*I); p3=E;
  //ENon=(p1/12.)*pow(XM,4.)+(p2/2.)*pow(XM,2.)+p3;
  //ENon = E*(1+0.5*(XM+YM));
  
  PetscInt       n1e, n2e, n3e;
  n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];    
  */
  
  
  
    //c = (E*Eavg)/(1. - pow(mu, 2.));
  // printf("ibm->El[ec] = %f \n", ibm->El[1][ec]);
  PetscReal mu_m = ibm->El[1][ec] + ibm->E_epsilon[1][ec];
  c = (ibm->El[0][ec] + ibm->E_epsilon[0][ec])/(1. - pow(mu_m, 2.));
  

  Dloc[0][0] = c;     Dloc[0][1] = c*mu_m;  Dloc[0][2] = 0.;
  Dloc[1][0] = c*mu_m;  Dloc[1][1] = c;     Dloc[1][2] = 0.;
  Dloc[2][0] = 0.;    Dloc[2][1] = 0.;    Dloc[2][2] = c*(1. - mu_m)/2.;

  //Form local strain
  for (i=0; i<3; i++){
    sum = 0.0;
    for (n=0; n<3; n++){
      sum += Q[n][i]*Strain[n];
    }
    Strainloc[i] = sum;
  }

  for (i=0; i<3; i++){
    sum = 0.0;
    for (m=0; m<3; m++){
      for (n=0; n<3; n++){
  	sum += Q[i][m]*Dloc[m][n]*Strainloc[n];
      }
    }
    _S[i] = sum;
  }

  return(0);
}


//--------------------------------------------------------------------------------------- 
PetscErrorCode StressManufactured(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) {
 
  PetscReal      Q[3][3], Dloc[3][3], Strainloc[3], c, sum;
  PetscInt       i, j, m, n;
  PetscReal      q[3][3], xz, xe, yz, ye;
  struct Cmpnts  dX21, dX31, V1, V2, N, diff1, diff2;

  if (method==0) {
    dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2
    N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec];
  } else if (method==1) { 
    dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2];
    dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2];
    N = UNIT(CROSS(dX21, dX31));
  }  
  V1 = UNIT(dX21);
  V2 = CROSS(N, V1);

  xz = DOT(V1, dX21);  xe = DOT(V1, dX31);
  yz = DOT(V2, dX21);  ye = DOT(V2, dX31);

  q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe;
  q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye;
  q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz;

  INV(q, Q);

  PetscReal x1, x2, x3, centx, L;
  x1 = ibm->x_bp[ibm->nv1[ec]];  x2 = ibm->x_bp[ibm->nv2[ec]];  x3 = ibm->x_bp[ibm->nv3[ec]];
  centx = (x1 + x2 + x3)/3.;
  L = 0.04;

  c = E*(1 + pow((centx/L),2))/(1. - pow(mu, 2.));
  Dloc[0][0] = c;     Dloc[0][1] = c*mu;  Dloc[0][2] = 0.;
  Dloc[1][0] = c*mu;  Dloc[1][1] = c;     Dloc[1][2] = 0.;
  Dloc[2][0] = 0.;    Dloc[2][1] = 0.;    Dloc[2][2] = c*(1. - mu)/2.;

  //Form local strain
  for (i=0; i<3; i++){
    sum = 0.0;
    for (n=0; n<3; n++){
      sum += Q[n][i]*Strain[n];
    }
    Strainloc[i] = sum;
  }

  for (i=0; i<3; i++){
    sum = 0.0;
    for (m=0; m<3; m++){
      for (n=0; n<3; n++){
  	sum += Q[i][m]*Dloc[m][n]*Strainloc[n];
      }
    }
    _S[i] = sum;
  }

  return(0);
}

//---------------------------------------------------------------------------------------  
PetscErrorCode CalcCurvStressStrainxyz(PetscInt ec, PetscReal k[3], PetscReal strain[3], PetscReal stress[3], PetscInt method, PetscInt mb, FE *fem) {

  IBMNodes       *ibm=fem->ibm;
  struct Cmpnts  X1, X2, X3, dX21, dX31;
  PetscInt       i, j, m, n1e, n2e, n3e;
  PetscReal      sum, cart[3];
  PetscReal      q[3][3], Q[3][3], xz, xe, yz, ye;
  
  if(method==0) {
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
    X1.x = ibm->x_bp0[n1e];  X1.y = ibm->y_bp0[n1e];  X1.z = ibm->z_bp0[n1e];
    X2.x = ibm->x_bp0[n2e];  X2.y = ibm->y_bp0[n2e];  X2.z = ibm->z_bp0[n2e];
    X3.x = ibm->x_bp0[n3e];  X3.y = ibm->y_bp0[n3e];  X3.z = ibm->z_bp0[n3e];
    dX21 = MINUS(X2, X1);
    dX31 = MINUS(X3, X1);

  } else if(method==1) {
    dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2];
    dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2];
  }

  xz = dX21.x;  xe = dX31.x;
  yz = dX21.y;  ye = dX31.y;
  
  q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe; //Transformation based on stress
  q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye;  
  q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz; 
  INV(q, Q);

  //curvilinear to global curvature
  for (i=0; i<3; i++) {
    sum = 0.;
    for (j=0; j<3; j++) {
      sum += Q[j][i]*k[j];
    }
    cart[i] = sum; 
  }

  if (mb==1) {
    ibm->kve[ec*dof] = cart[0];   ibm->kve[ec*dof+1] = cart[1];   ibm->kve[ec*dof+2] = cart[2];   
  }	

  //curvilinear to global strain
  for (i=0; i<3; i++) {
    sum = 0.;
    for (j=0; j<3; j++) {
      sum += Q[j][i]*strain[j]; 
    }
    cart[i] = sum;
  }

  if (mb==0) {
    fem->StrainM[ec*(dof+2)] = cart[0];   fem->StrainM[ec*(dof+2)+1] = cart[1];   fem->StrainM[ec*(dof+2)+2] = cart[2];  
    fem->StrainM[ec*(dof+2)+3] = (cart[0] + cart[1])/2. + sqrt(pow((cart[0] - cart[1])/2., 2) + pow(cart[2]/2., 2)); //in-plane max Principal Strain 
    fem->StrainM[ec*(dof+2)+4] = (cart[0] + cart[1])/2. - sqrt(pow((cart[0] - cart[1])/2., 2) + pow(cart[2]/2., 2)); //in-plane min Principal Strain 
  } else if (mb==1) {
    fem->StrainB[ec*dof] = cart[0];   fem->StrainB[ec*dof+1] = cart[1];   fem->StrainB[ec*dof+2] = cart[2];  
  }

  //curvilinear to global stress
  for (i=0; i<3; i++) {
    sum = 0.;
    for (j=0; j<3; j++) {
      sum += q[i][j]*stress[j];
    }
    cart[i] = sum;
  }

  if (mb==0) {
    fem->StressM[ec*dof] = cart[0];   fem->StressM[ec*dof+1] = cart[1];   fem->StressM[ec*dof+2] = cart[2]; 
  } else if (mb==1) {
    fem->StressB[ec*dof] = cart[0];   fem->StressB[ec*dof+1] = cart[1];   fem->StressB[ec*dof+2] = cart[2]; 
  }
  
  return(0);
}


PetscErrorCode BHVFiberDirUpdate(PetscInt ec, PetscReal angle, IBMNodes *ibm){

  /*
  Description:
    update the fiber direction vector in the undeformed configuration. finds the angle theta
    R(theta)*f_0 (f_0 = cross(N, axis))

  Note: 
    
  */
  struct Cmpnts  Axis, N, Midd, tcyl;  //Axis:Cylinder axis , Midd: middle of element , tcyl:tangent vector to cylinder
  PetscReal      pi=3.14159 , Teta; // Teta: angle of elemet in respect to the y=0 in xy surface
  PetscReal      R[3][3], nfibR[3], ttcyl[3];  //angle:angle of fiber direction respect to free edge
  PetscInt       n1e, n2e, n3e;
  PetscReal      e=0.03;

  struct Cmpnts  x1, x2, x3, dx21, dx31, cross;

  // angle = -45./180.*pi;

  Axis.x = 0.;  Axis.y = 0.;  Axis.z = 1;
  
    
    n1e = ibm->nv1[ec];  n2e = ibm->nv2[ec];  n3e = ibm->nv3[ec];
        
    //current location
    x1.x = ibm->x_bp0[n1e];  x1.y = ibm->y_bp0[n1e];  x1.z = ibm->z_bp0[n1e];
    x2.x = ibm->x_bp0[n2e];  x2.y = ibm->y_bp0[n2e];  x2.z = ibm->z_bp0[n2e];
    x3.x = ibm->x_bp0[n3e];  x3.y = ibm->y_bp0[n3e];  x3.z = ibm->z_bp0[n3e];
    
    dx21 = MINUS(x2, x1);  dx31 = MINUS(x3, x1);
    cross = CROSS(dx21, dx31);
    
    N = UNIT(cross);

    //  FormRotationMatrixAround Normal to element (N) in current mesh it is outward
    R[0][0] = cos(angle) + N.x*N.x*(1 - cos(angle));
    R[0][1] = N.x*N.y*(1 - cos(angle)) - N.z*sin(angle);
    R[0][2] = N.x*N.z*(1 - cos(angle)) + N.y*sin(angle);
    
    R[1][0] = N.x*N.y*(1 - cos(angle)) + N.z*sin(angle);
    R[1][1] = cos(angle) + N.y*N.y*(1 - cos(angle));
    R[1][2] = N.y*N.z*(1 - cos(angle)) - N.x*sin(angle);
    
    R[2][0] = N.z*N.x*(1 - cos(angle)) - N.y*sin(angle);
    R[2][1] = N.z*N.y*(1 - cos(angle)) + N.x*sin(angle);
    R[2][2] = cos(angle) + N.z*N.z*(1 - cos(angle));
    
    tcyl = CROSS(N, Axis);
    ttcyl[0] = tcyl.x;  ttcyl[1] = tcyl.y;  ttcyl[2] = tcyl.z;
   
    //Rotate Vector
    for (int i=0; i<3; i++){
      nfibR[i] = 0.;
      for (int j=0; j<3; j++){
	nfibR[i] += R[i][j]*ttcyl[j];
      }
    }
    
    ibm->n_fib[ec].x = nfibR[0]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));
    ibm->n_fib[ec].y = nfibR[1]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));
    ibm->n_fib[ec].z = nfibR[2]/sqrt(pow(nfibR[0],2) + pow(nfibR[1],2) + pow(nfibR[2],2));   
    // ibm->n_fib[ec].x = nfibR[0];
    // ibm->n_fib[ec].y = nfibR[1];
    // ibm->n_fib[ec].z = nfibR[2];    
  

}

//---------------------------------------------------------------------------------------
PetscErrorCode MembraneNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) {

  PetscReal      q[3][3], xz, xe, ye, yz, Q[3][3], Strainloc[3], sum;
  PetscInt       i, j, m, n, p;
  struct Cmpnts  dX21, dX31, V1, V2, N;

  if (method==0) {
    dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2
    N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec];
  } else if (method==1) {
    dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2];
    dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2];
    N = UNIT(CROSS(dX21, dX31));
  }

  V1 = UNIT(dX21);
  V2 = CROSS(N, V1);

  xz = DOT(V1, dX21);  xe = DOT(V1, dX31);
  yz = DOT(V2, dX21);  ye = DOT(V2, dX31);

  q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe;
  q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye;
  q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz;

  INV(q, Q);
 
  // Find theta: angle between the G1 dir and fiber direction
  PetscReal      cosalpha, costheta, theta;
  struct Cmpnts  n_fib, Nf;
  

  // For the bhv inverse problem 
  // PetscReal fib_theta = ibm->Fung_coeffs[7][ec] + ibm->Fung_epsilons[7][ec];
  // BHVFiberDirUpdate(ec, fib_theta, ibm);
  // n_fib.x = ibm->n_fib[ec].x;
  // n_fib.y = ibm->n_fib[ec].y;
  // n_fib.z = ibm->n_fib[ec].z;


  /// For patch test
  if (fibersmooth){    
    n_fib.x = ibm->Fung_coeffs_smth[7][ec] + ibm->Fung_epsilons[7][ec];          
  }
  else{
    n_fib.x = ibm->Fung_coeffs[7][ec] + ibm->Fung_epsilons[7][ec];
  }

  if (n_fib.x > 1.0){
    n_fib.x = 1.0;
  }
  n_fib.y = pow(1 - n_fib.x*n_fib.x, 0.5);
  n_fib.z = 0.0;

  ibm->n_fib[ec].x = n_fib.x;
  ibm->n_fib[ec].y = n_fib.y;
  ibm->n_fib[ec].z = n_fib.z;
  

  //----------------------------------------------------------------------//
  

  // For the genenral 3D fiber direction search
  /*
  PetscReal fib_theta = ibm->Fung_coeffs[7][ec] + ibm->Fung_epsilons[7][ec];
  PetscReal fib_phi = ibm->Fung_coeffs[8][ec] + ibm->Fung_epsilons[8][ec];

  n_fib.z = cos(fib_phi);
  n_fib.x = sin(fib_phi) * cos(fib_theta);
  n_fib.y = sin(fib_phi) * sin(fib_theta);
  */
  
  
  // n_fib.y = pow(1 - n_fib.x*n_fib.x, 0.5);
  // n_fib.z = 0.0;

  // ibm->n_fib[ec].x = n_fib.x;
  // ibm->n_fib[ec].y = n_fib.y;
  // ibm->n_fib[ec].z = n_fib.z;


  
  costheta = DOT(dX21, n_fib)/(SIZE(dX21)*SIZE(n_fib));
  if (costheta>1.) costheta = 1.;
  if (costheta<-1.) costheta = -1.;
  theta = acos(costheta);  
  Nf = CROSS(dX21, n_fib);
  cosalpha = DOT(Nf, N);
  if (cosalpha<0.) theta =- theta;

  // FormRotationMatrix
  PetscReal  R[3][3];
  R[0][0] = cos(theta)*cos(theta);  R[0][1] = sin(theta)*sin(theta);   R[0][2] = 2*sin(theta)*cos(theta);
  R[1][0] = sin(theta)*sin(theta);  R[1][1] = cos(theta)*cos(theta);   R[1][2] =-2*sin(theta)*cos(theta);
  R[2][0] = -sin(theta)*cos(theta);  R[2][1] = sin(theta)*cos(theta);   R[2][2] = cos(theta)*cos(theta)-sin(theta)*sin(theta);
 
  //Form local strain
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (n=0; n<3; n++) {
      sum += Q[n][i]*Strain[n];
    }
    Strainloc[i] = sum;
  }
  Strainloc[2] = Strainloc[2]/2.;

  //Compute Strain in fiber direction Ef=R Q E
  PetscReal  Ef[3];
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (m=0; m<3; m++) {
  	sum += R[i][m]*Strainloc[m];
    }
    Ef[i] = sum;
  }

  //  Tissue Constitutive Law according to Fung exponential form
  //PetscReal c=9.7e3, A1=49.558, A2=5.2871, A3=-3.124, A4=16.031 ,A5=-.004, A6=-0.02; // Values for BHV hammer et al, ABME 2011
  PetscReal  c=14.42e3, A1=61.27, A2=70.37, A3=5.11, A4=14.2, A5=3.1, A6=2.01; //Kim et al 2008 annuals of biomedical engineering

  c = ibm->Fung_coeffs[0][ec] + ibm->Fung_epsilons[0][ec];
  A1 = ibm->Fung_coeffs[1][ec] + ibm->Fung_epsilons[1][ec]; 
  A2 = ibm->Fung_coeffs[2][ec] + ibm->Fung_epsilons[2][ec];  
  A3 = ibm->Fung_coeffs[3][ec] + ibm->Fung_epsilons[3][ec]; 
  A4 = ibm->Fung_coeffs[4][ec] + ibm->Fung_epsilons[4][ec]; 
  A5 = ibm->Fung_coeffs[5][ec] + ibm->Fung_epsilons[5][ec]; 
  A6 = ibm->Fung_coeffs[6][ec] + ibm->Fung_epsilons[6][ec]; 


  //PetscReal  c=5.18e3, A1=96.25, A2=82.61, A3=12.48, A4=32.3, A5=15.22, A6=14.41;
  //PetscReal  c=6.72e3, A1=80.41, A2=181.86, A3=-17.43, A4=30.4, A5=2.89, A6=3.83;
  PetscReal  Qf, Sf[3]; //f:fiber direction
  Qf = A1*Ef[0]*Ef[0]+A2*Ef[1]*Ef[1]+2*A3*Ef[0]*Ef[1]+
    A4*Ef[2]*Ef[2]+2*A5*Ef[0]*Ef[2]+2*A6*Ef[1]*Ef[2];
  
  Sf[0] = c*(A1*Ef[0]+A3*Ef[1]+A5*Ef[2])*exp(Qf);
  Sf[1] = c*(A3*Ef[0]+A2*Ef[1]+A6*Ef[2])*exp(Qf);
  Sf[2] = c*(A5*Ef[0]+A6*Ef[1]+A4*Ef[2])*exp(Qf);

  // Convert Sf to Scur Scur=QT Rinv Sf
  PetscReal  Rinv[3][3];
  INV(R, Rinv);
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (m=0; m<3; m++) {
      for (p=0; p<3; p++) {
  	sum += Q[i][m]*Rinv[m][p]*Sf[p];
      }
    }
    _S[i] = sum;
  }
  
  return(0);
}

//---------------------------------------------------------------------------------------
PetscErrorCode BendingNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) {
  
  PetscReal      q[3][3], xz, xe, ye, yz, Q[3][3], Strainloc[3], sum;
  PetscInt       i, j, m, n, p;
  struct Cmpnts  dX21, dX31, V1, V2, N;

  if (method==0) {
    dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2
    N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec];
  } else if (method==1) {
    dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2];
    dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2];
    N = UNIT(CROSS(dX21, dX31));
  }

  V1 = UNIT(dX21);
  V2 = CROSS(N, V1);

  xz = DOT(V1, dX21);  xe = DOT(V1, dX31);
  yz = DOT(V2, dX21);  ye = DOT(V2, dX31);

  q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe;
  q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye;
  q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz;

  INV(q, Q);
 
  // Find theta: angle between the G1 dir and fiber direction
  PetscReal      cosalpha, costheta, theta;
  struct Cmpnts  n_fib, Nf;
  n_fib.x = ibm->n_fib[ec].x;  n_fib.y = ibm->n_fib[ec].y;  n_fib.z = ibm->n_fib[ec].z;
  costheta = DOT(dX21, n_fib)/(SIZE(dX21)*SIZE(n_fib));
  if (costheta>1.) costheta = 1.;
  if (costheta<-1.) costheta = -1.;
  theta = acos(costheta);
  Nf = CROSS(dX21, n_fib);
  cosalpha = DOT(Nf, N);
  if (cosalpha<0.) theta =- theta;

  // FormRotationMatrix
  PetscReal  R[3][3];
  R[0][0] = cos(theta)*cos(theta);  R[0][1] = sin(theta)*sin(theta);   R[0][2] = 2*sin(theta)*cos(theta);
  R[1][0] = sin(theta)*sin(theta);  R[1][1] = cos(theta)*cos(theta);   R[1][2] =-2*sin(theta)*cos(theta);
  R[2][0] = -sin(theta)*cos(theta);  R[2][1] = sin(theta)*cos(theta);   R[2][2] = cos(theta)*cos(theta)-sin(theta)*sin(theta);
 
  //Form local strain
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (n=0; n<3; n++) {
      sum += Q[n][i]*Strain[n];
    }
    Strainloc[i] = sum;
  }
  Strainloc[2] = Strainloc[2]/2.;

  //Compute Strain in fiber direction Ef=R Q E
  PetscReal  Ef[3];
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (m=0; m<3; m++) {
  	sum += R[i][m]*Strainloc[m];
    }
    Ef[i] = sum;
  }

  //Ef[0]=Strainloc[0]; Ef[1]=Strainloc[1]; Ef[2]=Strainloc[2];

  //  Tissue Constitutive Law according to Fung exponential form
  PetscReal  a1=443000, a2=620000, b1=4302, b2=6023; //Kim et al 2008 annuals of biomedical engineering
  PetscReal  Sf[3]; //f:fiber direction
  
  Sf[0] = a1*Ef[0] + b1*SIGN(Ef[0])*Ef[0]*Ef[0] + 0.25*(a1 + a2)*Ef[1];
  Sf[1] = 0.25*(a1 + a2)*Ef[0] + a2*Ef[1] + b2*SIGN(Ef[1])*Ef[1]*Ef[1];
  Sf[2] = 0.25*(a1 + a2)*Ef[2];

  // Convert Sf to Scur Scur=QT Rinv Sf
  PetscReal  Rinv[3][3];
  INV(R, Rinv);
  for (i=0; i<3; i++) {
    sum = 0.0;
    for (m=0; m<3; m++) {
      for (p=0; p<3; p++) {
  	sum += Q[i][m]*Rinv[m][p]*Sf[p];
      }
    }
    _S[i] = sum;
  }


  return(0);
}

//---------------------------------------------------------------------------------------  
/* PetscErrorCode MembraneNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) { */

/*   PetscReal      Q[3][3], Dloc[3][3], Strainloc[3], sum; */
/*   PetscInt       i, j, m, n; */
/*   PetscReal      q[3][3], xz, xe, yz, ye; */
/*   struct Cmpnts  dX21, dX31, V1, V2, N, diff1, diff2; */

/*   if (method==0) { */
/*     dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2 */
/*     N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec]; */
/*   } else if (method==1) { */
/*     dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2]; */
/*     dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2]; */
/*     N = UNIT(CROSS(dX21, dX31)); */
/*   } */
/*   V1 = UNIT(dX21); */
/*   V2 = CROSS(N, V1); */

/*   xz = DOT(V1, dX21);  xe = DOT(V1, dX31); */
/*   yz = DOT(V2, dX21);  ye = DOT(V2, dX31); */

/*   q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe; */
/*   q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye; */
/*   q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz; */

/*   INV(q, Q); */

/*   //Form local strain */
/*   for (i=0; i<3; i++){ */
/*     sum = 0.0; */
/*     for (n=0; n<3; n++){ */
/*       sum += Q[n][i]*Strain[n]; */
/*     } */
/*     Strainloc[i] = sum; */
/*   } */
/*   Strainloc[2] = Strainloc[2]/2.; */
/*   //PetscReal c=9.7e3, A1=49.558, A2=5.2871, A3=-3.124, A4=16.031 ,A5=-.004, A6=-0.02; // Values for BHV hammer et al, ABME 2011 */
/*   PetscReal  c=14.42e3, A1=61.27, A2=70.37, A3=5.11, A4=14.2, A5=3.1, A6=2.01;//Kim et al 2008 annuals of biomedical engineering */
/*   //A5=0.; A6=0.; A1=A2; */
/*   PetscReal  Qf, Sf[3]; //f:fiber direction */
/*   Qf = A1*Strainloc[0]*Strainloc[0]+A2*Strainloc[1]*Strainloc[1]+2*A3*Strainloc[0]*Strainloc[1]+ */
/*     A4*Strainloc[2]*Strainloc[2]+2*A5*Strainloc[0]*Strainloc[2]+2*A6*Strainloc[1]*Strainloc[2]; */
  
/*   /\*   Sf[0] = c*(A1*Ef[0]+A3*Ef[1]+A5*Ef[2])*exp(Qf); *\/ */
/*   /\*   Sf[1] = c*(A3*Ef[0]+A2*Ef[1]+A6*Ef[2])*exp(Qf); *\/ */
/*   /\*   Sf[2] = c*(A5*Ef[0]+A6*Ef[1]+A4*Ef[2])*exp(Qf); *\/ */
/*   c=c*exp(Qf); */
/*   Dloc[0][0] = c*A1;     Dloc[0][1] = c*A3;  Dloc[0][2] = c*A5; */
/*   Dloc[1][0] = c*A3;  Dloc[1][1] = c*A2;     Dloc[1][2] = c*A6; */
/*   Dloc[2][0] = c*A5;    Dloc[2][1] = c*A6;    Dloc[2][2] = c*A4; */

/*   for (i=0; i<3; i++){ */
/*     sum = 0.0; */
/*     for (m=0; m<3; m++){ */
/*       for (n=0; n<3; n++){ */
/*   	sum += Q[i][m]*Dloc[m][n]*Strainloc[n]; */
/*       } */
/*     } */
/*     _S[i] = sum; */
/*   } */

/*   return(0); */
/* } */

/* //--------------------------------------------------------------------------------------- */
/* PetscErrorCode BendingNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm) { */

/*   PetscReal      Q[3][3], Dloc[3][3], Strainloc[3], c, sum; */
/*   PetscInt       i, j, m, n; */
/*   PetscReal      q[3][3], xz, xe, yz, ye; */
/*   struct Cmpnts  dX21, dX31, V1, V2, N, diff1, diff2; */

/*   if (method==0) { */
/*     dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1);  //dX21:G1 , dX31:G2 */
/*     N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec]; */
/*   } else if (method==1) { */
/*     dX21.x = ibm->G1[ec*dof];  dX21.y = ibm->G1[ec*dof+1];  dX21.z = ibm->G1[ec*dof+2]; */
/*     dX31.x = ibm->G2[ec*dof];  dX31.y = ibm->G2[ec*dof+1];  dX31.z = ibm->G2[ec*dof+2]; */
/*     N = UNIT(CROSS(dX21, dX31)); */
/*   } */
/*   V1 = UNIT(dX21); */
/*   V2 = CROSS(N, V1); */

/*   xz = DOT(V1, dX21);  xe = DOT(V1, dX31); */
/*   yz = DOT(V2, dX21);  ye = DOT(V2, dX31); */

/*   q[0][0] = xz*xz;  q[0][1] = xe*xe;  q[0][2] = 2*xz*xe; */
/*   q[1][0] = yz*yz;  q[1][1] = ye*ye;  q[1][2] = 2*yz*ye; */
/*   q[2][0] = xz*yz;  q[2][1] = xe*ye;  q[2][2] = xz*ye + xe*yz; */

/*   INV(q, Q); */

/*   //Form local strain */
/*   for (i=0; i<3; i++){ */
/*     sum = 0.0; */
/*     for (n=0; n<3; n++){ */
/*       sum += Q[n][i]*Strain[n]; */
/*     } */
/*     Strainloc[i] = sum; */
/*   } */
/*   Strainloc[2] = Strainloc[2]/2.; */

/*   PetscReal  a1=443000, a2=620000, b1=4302, b2=6023; //Kim et al 2008 annuals of biomedical engineering */
/*   PetscReal  Sf[3]; //f:fiber direction */
  
/*   Sf[0] = a1*Strainloc[0] + b1*SIGN(Strainloc[0])*Strainloc[0]*Strainloc[0] + 0.25*(a1 + a2)*Strainloc[1]; */
/*   Sf[1] = 0.25*(a1 + a2)*Strainloc[0] + a2*Strainloc[1] + b2*SIGN(Strainloc[1])*Strainloc[1]*Strainloc[1]; */
/*   Sf[2] = 0.25*(a1 + a2)*Strainloc[2]; */

/*   for (i=0; i<3; i++){ */
/*     sum = 0.0; */
/*     for (m=0; m<3; m++){ */
/*       sum += Q[i][m]*Sf[m]; */
/*     } */
/*     _S[i] = sum; */
/*   } */

/*   return(0); */
/* } */
