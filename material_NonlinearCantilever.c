#include "variables.h"

extern  PetscReal E,mu,rho,h0,dampfactor;
extern PetscInt dof,ConstitutiveLawNonLinear;
extern struct Cmpnts PLUS(struct Cmpnts v1,struct Cmpnts v2);
extern struct Cmpnts MINUS(struct Cmpnts v1,struct Cmpnts v2);
extern struct Cmpnts CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal DOT(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts UNIT(struct Cmpnts v1);
extern PetscReal SIZE(struct Cmpnts v1);
extern PetscErrorCode INV(PetscReal T[3][3],PetscReal _Tinv[3][3]);
extern PetscErrorCode MATMULT(PetscReal A[][2],PetscReal B[][2], PetscReal C[][2]);

//---------------------------------------------------------------------------------------  
PetscErrorCode InitMaterial(IBMNodes *ibm)
{
  PetscInt ec, i, j;
// n_fib direction
  /* for (ec=0;ec<ibm->n_elmt;ec++) { */
/*     ibm->n_fib[ec].x=0.707106781; */
/*     ibm->n_fib[ec].y=0.707106781; */
/*     ibm->n_fib[ec].z=0.0; */
/*   } */

 /*  for (ec=0;ec<ibm->n_elmt;ec++) { */
/*     ibm->n_fib[ec].x=1.0; */
/*     ibm->n_fib[ec].y=0.0; */
/*     ibm->n_fib[ec].z=0.0; */
/*   } */

  // For LV
  struct Cmpnts   Axis,N,Midd,tcyl;  //Axis:Cylinder axis , Midd: middle of element , tcyl:tangent vector to cylinder
  PetscReal pi=3.14159 , Teta; // Teta: angle of elemet in respect to the y=0 in xy surface
  PetscReal R[3][3],nfibR[3], ttcyl[3],angle ;  //angle:angle of fiber direction respect to free edge
  PetscInt n1e,n2e,n3e;

 
   Axis.x=0.; Axis.y=0.; Axis.z=1;

   for (ec=0;ec<ibm->n_elmt; ec++) {
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];
    N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
   //compute the middle of element in xy plane
    Midd.x=(ibm->x_bp0[n1e]+ibm->x_bp0[n2e]+ibm->x_bp0[n3e])/3.;
    Midd.y=(ibm->y_bp0[n1e]+ibm->y_bp0[n2e]+ibm->y_bp0[n3e])/3.;

   
  
    Teta=atan( PetscAbsReal(Midd.y/Midd.x));
   if(Midd.x<0. && Midd.y>0.){Teta=pi-Teta;}
   if(Midd.x<0. && Midd.y<0.){Teta=pi+Teta;}
   if(Midd.x>0. && Midd.y<0.){Teta=2.*pi-Teta;}

 
   angle=-(25./180.)*pi;
   if( (Teta>0. && Teta<pi/3.) || (Teta>2.*pi/3. && Teta<pi) || (Teta>4.*pi/3. && Teta<5.*pi/3.) ){angle=-pi+(25./180.)*pi;}
  
   
//  FormRotationMatrixAround Normal to element (N) in curent mesh it is outward
   R[0][0]= cos(angle)+N.x*N.x*(1-cos(angle));
   R[0][1]= N.x*N.y*(1-cos(angle))-N.z*sin(angle);
   R[0][2]= N.x*N.z*(1-cos(angle))+N.y*sin(angle);

   R[1][0]= N.x*N.y*(1-cos(angle))+N.z*sin(angle);
   R[1][1]= cos(angle)+N.y*N.y*(1-cos(angle));
   R[1][2]= N.y*N.z*(1-cos(angle))-N.x*sin(angle);

   R[2][0]= N.z*N.x*(1-cos(angle))-N.y*sin(angle);
   R[2][1]= N.z*N.y*(1-cos(angle))+N.x*sin(angle);
   R[2][2]= cos(angle)+N.z*N.z*(1-cos(angle));

   tcyl=CROSS(N,Axis);
   ttcyl[0]=tcyl.x; ttcyl[1]=tcyl.y; ttcyl[2]=tcyl.z;

   
   //Rotate Vector
   for (i=0; i<3;i++){
     nfibR[i]=0.;
     for (j=0;j<3;j++){
       nfibR[i]+=R[i][j]*ttcyl[j];
     }
   }

   ibm->n_fib[ec].x=nfibR[0];
   ibm->n_fib[ec].y=nfibR[1];
   ibm->n_fib[ec].z=nfibR[2];
   
  }
 
 return(0);
}
//---------------------------------------------------------------------------------------  
PetscErrorCode Mass(IBMNodes *ibm,PetscInt ec,PetscReal _M[9])
{
  PetscReal A0; PetscInt i;
  A0=ibm->dA0[ec];

  for (i=0; i<9; i++){
    _M[i]=rho*h0*A0/3;
  }      
 return(0);
 }     
//---------------------------------------------------------------------------------------  
PetscErrorCode Damp(PetscReal M[9],PetscReal _C[9])
{
  PetscInt i;
  for (i=0; i<9; i++){
     _C[i]=dampfactor*M[i]; //1000.0 for fiber ,  beam bending=2*pi
  }
 return(0);     
} 
//---------------------------------------------------------------------------------------  
PetscErrorCode MassDamp(FE *fem)
{
  IBMNodes *ibm=fem->ibm;
  PetscReal A0,M,C;
  PetscReal *DDissip,*MMass;
  PetscInt n1e,n2e,n3e,ec;
 

  VecGetArray(fem->Mass, &MMass);
  VecGetArray(fem->Dissip, &DDissip);

  for (ec=0; ec<ibm->n_elmt; ec++) {
    A0=ibm->dA0[ec];
    n1e=ibm->nv1[ec];n2e=ibm->nv2[ec];n3e=ibm->nv3[ec];

    M=rho*h0*A0/3.;
      
    MMass[n1e*dof] +=M;
    MMass[n1e*dof+1] +=M;
    MMass[n1e*dof+2] +=M;
    
    MMass[n2e*dof] +=M;
    MMass[n2e*dof+1] +=M;
    MMass[n2e*dof+2] +=M;

    MMass[n3e*dof] +=M;
    MMass[n3e*dof+1] +=M;
    MMass[n3e*dof+2] +=M;
 
    C=dampfactor*M;

    DDissip[n1e*dof] +=C;
    DDissip[n1e*dof+1] +=C;
    DDissip[n1e*dof+2] +=C;
    
    DDissip[n2e*dof] +=C;
    DDissip[n2e*dof+1] +=C;
    DDissip[n2e*dof+2] +=C;

    DDissip[n3e*dof] +=C;
    DDissip[n3e*dof+1] +=C;
    DDissip[n3e*dof+2] +=C;

  }

  VecRestoreArray(fem->Mass, &MMass);
  VecRestoreArray(fem->Dissip, &DDissip);

 return(0);     
}         
//--------------------------------------------------------------------------------------- 
PetscErrorCode StressLinear(PetscInt ec,struct Cmpnts X1,struct Cmpnts X2,struct Cmpnts X3, PetscReal Strain[3],PetscReal _S[3],IBMNodes *ibm)
{
 
  PetscReal Q[3][3],Dloc[3][3],Strainloc[3],J11,J12,J22,Q11,Q12,Q22,c,sum;
  PetscInt i,j,m,n;
  
  struct Cmpnts dX21,dX31,V1,V2,N;
  dX21=MINUS(X2,X1); dX31=MINUS(X3,X1);  //dX21:G1 , dX31:G2
  V1=UNIT(dX21);
  N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
  V2=CROSS(N,V1);
  J11=DOT(dX21,V1); J12=DOT(dX31,V1); J22=DOT(dX31,V2);
  Q11=1./J11; Q12=-J12/(J11*J22); Q22=1./J22;
  Q[0][0]=pow(Q11,2.);   Q[0][1]=0.;                   Q[0][2]=0.;
  Q[1][0]=pow(Q12,2.);   Q[1][1]=pow(Q22,2.);          Q[1][2]=Q12*Q22;
  Q[2][0]=2*Q11*Q12;     Q[2][1]=0.;                   Q[2][2]=Q11*Q22;


  // Define Nonlinear E
  PetscReal p1,p2,p3,XM,ENon,a,I;
  PetscReal F=1.e-5,w=8.0,L=0.04,b=0.004;
  a=0.5/L; I=(1./12.)*b*pow(h0,3);
  XM=(X1.x+X2.x+X3.x)/3.;
  p1=(rho*b*h0*pow(w,2))/(2.*I); p2=F/(2.*a*I); p3=E;

  ENon=(p1/12.)*pow(XM,4.)+(p2/2.)*pow(XM,2.)+p3;


  c=ENon/(1.- pow(mu,2.));
  Dloc[0][0]=c;     Dloc[0][1]=c*mu;  Dloc[0][2]=0.;
  Dloc[1][0]=c*mu;  Dloc[1][1]=c;     Dloc[1][2]=0.;
  Dloc[2][0]=0.;    Dloc[2][1]=0.;    Dloc[2][2]=c*(1.-mu);



  //Form local strain to get ride of 2 constant
   for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=Q[i][n]*Strain[n];
    }
    Strainloc[i]=sum;
  }


  Strainloc[2]=Strainloc[2]/2.0;

  for (i=0;i<3;i++){
       sum=0.0;
      for (m=0;m<3;m++){
	for (n=0;n<3;n++){
	   sum+=Q[m][i]*Dloc[m][n]*Strainloc[n];
	}
      }
      _S[i]=sum;
  }

   
 return(0);
}
//---------------------------------------------------------------------------------------  
PetscErrorCode CalcStressStrainMxyz(PetscInt ec,struct Cmpnts x1,struct Cmpnts x2,struct Cmpnts x3, PetscReal Strain[3],FE *fem)
{
  IBMNodes *ibm=fem->ibm;
  PetscReal Q[3][3],Dloc[3][3],Strainloc[3],J11,J12,J22,Q11,Q12,Q22,sum;
  PetscInt i,j,m,n;
  
  struct Cmpnts dx21,dx31,v1,v2,nf;
  dx21=MINUS(x2,x1); dx31=MINUS(x3,x1);  
  v1=UNIT(dx21);
  nf.x=ibm->nf_x[ec]; nf.y=ibm->nf_y[ec]; nf.z=ibm->nf_z[ec];
  v2=CROSS(nf,v1);
  J11=DOT(dx21,v1); J12=DOT(dx31,v1); J22=DOT(dx31,v2);
  Q11=1./J11; Q12=-J12/(J11*J22); Q22=1./J22;
  Q[0][0]=pow(Q11,2.);   Q[0][1]=0.;                   Q[0][2]=0.;
  Q[1][0]=pow(Q12,2.);   Q[1][1]=pow(Q22,2.);          Q[1][2]=Q12*Q22;
  Q[2][0]=2*Q11*Q12;     Q[2][1]=0.;                   Q[2][2]=Q11*Q22;

  //Form local strain to get ride of 2 constant
   for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=Q[i][n]*Strain[n];
    }
    Strainloc[i]=sum;
  }



  //------------------------------------------------------------------------------------------
   PetscReal Stressxyz[3],Strainxyz[3],R[3][3],c,s;
  c=PetscAbsReal(dx21.x/SIZE(dx21));
  s=PetscSqrtReal(1-c*c);
  if(dx21.x>=0.0 && dx21.y>=0.0){;}
  if(dx21.x<=0.0 && dx21.y>=0.0){c=-c;}
  if(dx21.x<=0.0 && dx21.y<=0.0){c=-c; s=-s;}
  if(dx21.x>=0.0 && dx21.y<=0.0){s=-s;}

   R[0][0]= c*c;    R[0][1]= s*s;    R[0][2]= 2.*s*c;
   R[1][0]= s*s;    R[1][1]= c*c;    R[1][2]= -2.*s*c;
   R[2][0]= -s*c;   R[2][1]= s*c;     R[2][2]= c*c-s*s;

 for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=R[i][n]*Strainloc[n];
    }
    Strainxyz[i]=sum;
    Stressxyz[i]=E*sum;
  }

   fem->StrainM[ec*dof]=Strainxyz[0];   fem->StrainM[ec*dof+1]=Strainxyz[1];    fem->StrainM[ec*dof+2]=Strainxyz[2];
   fem->StressM[ec*dof]=Stressxyz[0];   fem->StressM[ec*dof+1]=Stressxyz[1];   fem->StressM[ec*dof+2]=Stressxyz[2];

  //----------------------------------------------------------------------------------------- 


  
 return(0);
}
//---------------------------------------------------------------------------------------  
PetscErrorCode CalcStressStrainBxyz(PetscInt ec,struct Cmpnts x1,struct Cmpnts x2,struct Cmpnts x3, PetscReal Strain[3],FE *fem)
{
  IBMNodes *ibm=fem->ibm;
  PetscReal Q[3][3],Dloc[3][3],Strainloc[3],J11,J12,J22,Q11,Q12,Q22,sum;
  PetscInt i,j,m,n;
  
  struct Cmpnts dx21,dx31,v1,v2,nf;
  dx21=MINUS(x2,x1); dx31=MINUS(x3,x1);  
  v1=UNIT(dx21);
  nf.x=ibm->nf_x[ec]; nf.y=ibm->nf_y[ec]; nf.z=ibm->nf_z[ec];
  v2=CROSS(nf,v1);
  J11=DOT(dx21,v1); J12=DOT(dx31,v1); J22=DOT(dx31,v2);
  Q11=1./J11; Q12=-J12/(J11*J22); Q22=1./J22;
  Q[0][0]=pow(Q11,2.);   Q[0][1]=0.;                   Q[0][2]=0.;
  Q[1][0]=pow(Q12,2.);   Q[1][1]=pow(Q22,2.);          Q[1][2]=Q12*Q22;
  Q[2][0]=2*Q11*Q12;     Q[2][1]=0.;                   Q[2][2]=Q11*Q22;

  //Form local strain to get ride of 2 constant
   for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=Q[i][n]*Strain[n];
    }
    Strainloc[i]=sum;
  }



  //------------------------------------------------------------------------------------------
   PetscReal Stressxyz[3],Strainxyz[3],R[3][3],c,s;
  c=PetscAbsReal(dx21.x/SIZE(dx21));
  s=PetscSqrtReal(1-c*c);
  if(dx21.x>=0.0 && dx21.y>=0.0){;}
  if(dx21.x<=0.0 && dx21.y>=0.0){c=-c;}
  if(dx21.x<=0.0 && dx21.y<=0.0){c=-c; s=-s;}
  if(dx21.x>=0.0 && dx21.y<=0.0){s=-s;}

   R[0][0]= c*c;    R[0][1]= s*s;    R[0][2]= 2.*s*c;
   R[1][0]= s*s;    R[1][1]= c*c;    R[1][2]= -2.*s*c;
   R[2][0]= -s*c;   R[2][1]= s*c;     R[2][2]= c*c-s*s;

 for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=R[i][n]*Strainloc[n];
    }
    Strainxyz[i]=sum;
    Stressxyz[i]=E*sum;
  }

   fem->StrainB[ec*dof]=Strainxyz[0];   fem->StrainB[ec*dof+1]=Strainxyz[1];    fem->StrainB[ec*dof+2]=Strainxyz[2];
   fem->StressB[ec*dof]=Stressxyz[0];    fem->StressB[ec*dof+1]=Stressxyz[1];   fem->StressB[ec*dof+2]=Stressxyz[2];

  //----------------------------------------------------------------------------------------- 

 return(0);
}
//---------------------------------------------------------------------------------------  
PetscErrorCode StressNonLinear(PetscInt ec,struct Cmpnts X1,struct Cmpnts X2,struct Cmpnts X3, PetscReal Strain[3],PetscReal _S[3],IBMNodes *ibm)
{

  PetscReal Q[3][3],Strainloc[3],J11,J12,J22,Q11,Q12,Q22,sum;
  PetscInt i,j,m,n,p;
  
  struct Cmpnts dX21,dX31,V1,V2,N;
  dX21=MINUS(X2,X1); dX31=MINUS(X3,X1);  //dX21:G1 , dX31:G2
  V1=UNIT(dX21);
  N.x=ibm->Nf_x[ec]; N.y=ibm->Nf_y[ec]; N.z=ibm->Nf_z[ec];
  V2=CROSS(N,V1);
  J11=DOT(dX21,V1); J12=DOT(dX31,V1); J22=DOT(dX31,V2);
  Q11=1./J11; Q12=-J12/(J11*J22); Q22=1./J22;
  Q[0][0]=pow(Q11,2.);   Q[0][1]=0.;                   Q[0][2]=0.;
  Q[1][0]=pow(Q12,2.);   Q[1][1]=pow(Q22,2.);          Q[1][2]=Q12*Q22;
  Q[2][0]=2*Q11*Q12;     Q[2][1]=0.;                   Q[2][2]=Q11*Q22;
 

// Find theta: angle between the G1 dir and fiber direction
  PetscReal cosalpha,costheta,theta;
  struct Cmpnts n_fib,Nf;
  n_fib.x=ibm->n_fib[ec].x;  n_fib.y=ibm->n_fib[ec].y;  n_fib.z=ibm->n_fib[ec].z;
  costheta = DOT(dX21, n_fib)/(SIZE(dX21)*SIZE(n_fib));
  if (costheta>1.) costheta=1.;
  if (costheta<-1.) costheta=-1.;
  theta=acos(costheta);
  Nf=CROSS(dX21,n_fib);
  cosalpha=DOT(Nf,N);
  if (cosalpha<0.) theta=-theta;

  // FormRotationMatrix 
  PetscReal R[3][3];
  R[0][0]= cos(theta)*cos(theta);  R[0][1]= sin(theta)*sin(theta);   R[0][2]= 2*sin(theta)*cos(theta);
  R[1][0]= sin(theta)*sin(theta);  R[1][1]= cos(theta)*cos(theta);   R[1][2]=-2*sin(theta)*cos(theta);
  R[2][0]=-sin(theta)*cos(theta);  R[2][1]= sin(theta)*cos(theta);   R[2][2]= cos(theta)*cos(theta)-sin(theta)*sin(theta);
 
//Form local strain to get ride of 2 constant
  for (i=0;i<3;i++){
    sum=0.0;
    for (n=0;n<3;n++){
      sum+=Q[i][n]*Strain[n];
    }
    Strainloc[i]=sum;
  }


  Strainloc[2]=Strainloc[2]/2.0;

  //Compute Strain in fiber direction Ef=R Q E
  PetscReal Ef[3];
  for (i=0;i<3;i++){
    sum=0.0;
    for (m=0;m<3;m++){
  	sum+=R[i][m]*Strainloc[m];
    }
    Ef[i]=sum;
  }

 //  Tissue Constitutive Law according to Fung exponential form
    PetscReal c=9.7,A1=49.558,A2=5.2871, A3=-3.124, A4=16.031 ,A5=-.004, A6=-0.02; // Values for BHV hammer et al, ABME 2011

   PetscReal Qf,Sf[3]; //f:fiber direction
  Qf=A1*Ef[0]*Ef[0]+A2*Ef[1]*Ef[1]+2*A3*Ef[0]*Ef[1]+
    A4*Ef[2]*Ef[2]+2*A5*Ef[0]*Ef[2]+2*A6*Ef[1]*Ef[2];
 
  Sf[0]= c*(A1*Ef[0]+A3*Ef[1]+A5*Ef[2])*exp(Qf);
  Sf[1]= c*(A3*Ef[0]+A2*Ef[1]+A6*Ef[2])*exp(Qf);
  Sf[2]= c*(A5*Ef[0]+A6*Ef[1]+A4*Ef[2])*exp(Qf);

  //PetscPrintf(PETSC_COMM_SELF, "ec:%d Ef= %le,%le,%le \n",ec,Ef[0],Ef[1],Ef[2]);
 


  // Convert Sf to Scur Scur=QT Rinv Sf
  PetscReal Rinv[3][3];
  INV(R,Rinv);
  for (i=0;i<3;i++){
    sum=0.0;
    for (m=0;m<3;m++){
      for (p=0;p<3;p++){
	sum+=Q[m][i]*Rinv[m][p]*Sf[p];
      }
    }
    _S[i]=sum;
  }
  
 return(0);
}
