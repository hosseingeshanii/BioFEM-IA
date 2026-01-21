#include  "variables.h"

extern PetscReal  E, mu, rho, h0;
extern PetscInt   dof, bending, ConstitutiveLawNonLinear, curvature, manufactured;

extern struct Cmpnts  PLUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal  DOT(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  UNIT(struct Cmpnts v1);
extern PetscReal  SIZE(struct Cmpnts v1);
extern struct Cmpnts  AMULT(PetscReal alpha, struct Cmpnts v1);
extern PetscErrorCode  INV(PetscReal T[3][3], PetscReal _Tinv[3][3]);
extern PetscErrorCode MATMULT(const PetscReal A[3][3], const PetscReal B[3][3], PetscReal C[3][3]);
extern PetscErrorCode  TRANS(PetscReal A[3][3], PetscReal _AT[3][3]);
extern PetscErrorCode  StressLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm);
extern PetscErrorCode  StressManufactured(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm);
extern PetscErrorCode  MembraneNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm);
extern PetscErrorCode  BendingNonLinear(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, PetscReal Strain[3], PetscReal _S[3], PetscInt method, IBMNodes *ibm);
extern PetscErrorCode  CalcCurvStressStrainxyz(PetscInt ec, PetscReal k[3], PetscReal strain[3], PetscReal stress[3], PetscInt method, PetscInt mb, FE *fem);


PetscErrorCode Fbending(PetscInt ec, struct Cmpnts X1, struct Cmpnts X2, struct Cmpnts X3, struct Cmpnts x1, struct Cmpnts x2, struct Cmpnts x3, PetscReal _Fb[42], FE *fem) {
  
  IBMNodes       *ibm=fem->ibm;
  PetscReal      sum, sum1, sum2;
  PetscInt       i, j, m, p, q;
  struct Cmpnts  dx21, dx31, n, gc1, gc2;
  struct Cmpnts  a4, a5, a6;
  struct Cmpnts  x4, x5, x6;
  PetscReal      Eb[3], S[3];

  //PetscPrintf(PETSC_COMM_SELF, "CHECK size inside Fbending = %d\n", ibm->n_v);

  //gc=contravariant

  /* if (ibm->ibi==0) { */
  /*   //rho = 1.;  h0 = 0.003; */
  /*   rho = 0.05; */
  /* } else { */
  /*   //rho = 10.;  h0 = 0.03; */
  /*   rho = 0.15; */
  /* } */
  
  x4.x=ibm->p4x[ec]; x4.y=ibm->p4y[ec]; x4.z=ibm->p4z[ec];
  x5.x=ibm->p5x[ec]; x5.y=ibm->p5y[ec]; x5.z=ibm->p5z[ec];
  x6.x=ibm->p6x[ec]; x6.y=ibm->p6y[ec]; x6.z=ibm->p6z[ec];
 
  dx21=MINUS(x2, x1); dx31=MINUS(x3, x1); //dx21:g1 , dx31:g2
  a4=MINUS(x4, x2);   a5=MINUS(x5, x3);  a6=MINUS(x6, x1);
  
  //--------------------Compute kv:curvature at current----------------------------  
  PetscReal      ze[3], k[3], A, A0, kcur[3]; 
  PetscReal      T[3][3], Tinv[3][3]; 
  PetscReal      csi4, eta4, z4, csi5, eta5, z5, csi6, eta6, z6; 
  struct Cmpnts  dX21, dX31;
  
  dX21=MINUS(X2,X1); dX31=MINUS(X3,X1);
  A=ibm->dA[ec]; A0=ibm->dA0[ec];
  n.x=ibm->nf_x[ec]; n.y=ibm->nf_y[ec]; n.z=ibm->nf_z[ec];

  gc1=CROSS(dx31, n);
  gc1=AMULT(1./DOT(dx21, gc1), gc1);
  
  gc2=CROSS(n, dx21);
  gc2=AMULT(1./DOT(dx31, gc2), gc2);
  
  csi4=1.+DOT(a4, gc1);  eta4=DOT(a4, gc2);    z4=DOT(a4, n);
  csi5=DOT(a5, gc1);     eta5=1.+DOT(a5, gc2); z5=DOT(a5, n);
  csi6=DOT(a6, gc1);     eta6=DOT(a6, gc2);    z6=DOT(a6, n);
  
  T[0][0]=pow(csi4, 2.)-csi4;   T[0][1]=pow(eta4, 2.)-eta4; T[0][2]=csi4*eta4;
  T[1][0]=pow(csi5, 2.)-csi5;   T[1][1]=pow(eta5, 2.)-eta5; T[1][2]=csi5*eta5;
  T[2][0]=pow(csi6, 2.)-csi6;   T[2][1]=pow(eta6, 2.)-eta6; T[2][2]=csi6*eta6;
  
  INV(T, Tinv);
  
  if (curvature == 1) {//Patch Fitting Method
    ze[0]=z4; ze[1]=z5; ze[2]=z6;
    
    for (i=0; i<3; i++){
      sum=0.0;
      for (m=0; m<3; m++){
	sum+=2.*Tinv[i][m]*ze[m];
      }
      k[i]=sum;
    }    

    //---------------------Compute E and S for bending ---------------------------------------
    Eb[0]=ibm->kve0[ec*dof]-k[0];
    Eb[1]=ibm->kve0[ec*dof+1]-k[1];
    Eb[2]=ibm->kve0[ec*dof+2]-k[2];
    //PetscPrintf(PETSC_COMM_SELF, "Check before StressLinear!\n");
    StressLinear(ec, X1, X2, X3, Eb, S, 0, ibm);    
    CalcCurvStressStrainxyz(ec, k, Eb, S, 0, 1, fem);
 
    //---------------------Compute Bb ----------------------------------------------------
    //-------------dTinv z=Bbar1 dz-----------------------------------
    PetscReal Bbar1[3][6],dTTinvz[3][6],v1,v2,v3;  
    
    v1=k[0]/2.; v2=k[1]/2.; v3=k[2]/2.; //v=Tinv ze=curvature/2
    
    dTTinvz[0][0]=(2.*csi4-1)*v1+eta4*v3; dTTinvz[0][1]=(2.*eta4-1)*v2+csi4*v3; dTTinvz[0][2]=0.; dTTinvz[0][3]=0.; dTTinvz[0][4]=0.; dTTinvz[0][5]=0.;
    dTTinvz[1][0]=0.; dTTinvz[1][1]=0.; dTTinvz[1][2]=(2.*csi5-1)*v1+eta5*v3; dTTinvz[1][3]=(2.*eta5-1)*v2+csi5*v3; dTTinvz[1][4]=0; dTTinvz[1][5]=0;
    dTTinvz[2][0]=0.; dTTinvz[2][1]=0.; dTTinvz[2][2]=0.; dTTinvz[2][3]=0.; dTTinvz[2][4]=(2.*csi6-1)*v1+eta6*v3; dTTinvz[2][5]=(2.*eta6-1)*v2+csi6*v3;
    
    for (i=0;i<3;i++){
      for (j=0;j<6;j++){
	sum=0.0;
	for (p=0;p<3;p++){
	  sum+=-Tinv[i][p]*dTTinvz[p][j];
	}
	Bbar1[i][j]=sum;
      }
    }
    
    //-------------dz=H3 da------------------------------------
    PetscReal IMnTPn[3][3],H3[3][18],M1[3][3],M2[3][3];
    PetscReal HH1[3],HH2[3],HH3[3],HH4[3],HH5[3],HH6[3],aa4[3],aa5[3],aa6[3];
    
    IMnTPn[0][0]=1.-n.x*n.x;  IMnTPn[0][1]=-n.x*n.y;      IMnTPn[0][2]=-n.x*n.z;
    IMnTPn[1][0]=-n.y*n.x;    IMnTPn[1][1]=1.-n.y*n.y;    IMnTPn[1][2]=-n.y*n.z;
    IMnTPn[2][0]=-n.z*n.x;    IMnTPn[2][1]=-n.z*n.y;      IMnTPn[2][2]=1.-n.z*n.z;  
    
    M1[0][0]=0.;        M1[0][1]=-dx21.z;    M1[0][2]=dx21.y;
    M1[1][0]=dx21.z;    M1[1][1]=0.;         M1[1][2]=-dx21.x;
    M1[2][0]=-dx21.y;   M1[2][1]=dx21.x;     M1[2][2]=0.;
    
    M2[0][0]=0.;        M2[0][1]=-dx31.z;  M2[0][2]=dx31.y;
    M2[1][0]=dx31.z;    M2[1][1]=0.;       M2[1][2]=-dx31.x;
    M2[2][0]=-dx31.y;   M2[2][1]=dx31.x;   M2[2][2]=0.;
    
    
    aa4[0]=a4.x;  aa4[1]=a4.y;  aa4[2]=a4.z;
    aa5[0]=a5.x;  aa5[1]=a5.y;  aa5[2]=a5.z;
    aa6[0]=a6.x;  aa6[1]=a6.y;  aa6[2]=a6.z;
    
    for (i=0;i<3;i++){
      HH1[i]=0.0; HH2[i]=0.0; HH3[i]=0.0; 
      HH4[i]=0.0; HH5[i]=0.0; HH6[i]=0.0;
      for (m=0;m<3;m++){
	for (p=0;p<3;p++){
	  HH1[i]+=-aa4[m]*IMnTPn[m][p]*M1[p][i]/(2.*A);  HH2[i]+=-aa4[m]*IMnTPn[m][p]*M2[p][i]/(2.*A);
	  HH3[i]+=-aa5[m]*IMnTPn[m][p]*M1[p][i]/(2.*A);  HH4[i]+=-aa5[m]*IMnTPn[m][p]*M2[p][i]/(2.*A);
	  HH5[i]+=-aa6[m]*IMnTPn[m][p]*M1[p][i]/(2.*A);  HH6[i]+=-aa6[m]*IMnTPn[m][p]*M2[p][i]/(2.*A);
	}
      }
    }
    
    H3[0][0]=0.; H3[0][1]=0.; H3[0][2]=0.;     H3[0][3]=HH1[0]; H3[0][4]=HH1[1]; H3[0][5]=HH1[2]; 
    H3[0][6]=HH2[0]; H3[0][7]=HH2[1]; H3[0][8]=HH2[2];    H3[0][9]=n.x; H3[0][10]=n.y; H3[0][11]=n.z; 
    H3[0][12]=0.; H3[0][13]=0.; H3[0][14]=0.;     H3[0][15]= 0.; H3[0][16]=0.; H3[0][17]=0.;
    
    H3[1][0]=0.; H3[1][1]=0.; H3[1][2]=0.;    H3[1][3]=HH3[0]; H3[1][4]=HH3[1]; H3[1][5]=HH3[2];
    H3[1][6]= HH4[0]; H3[1][7]=HH4[1]; H3[1][8]=HH4[2];    H3[1][9]=0.; H3[1][10]=0.; H3[1][11]=0.;
    H3[1][12]= n.x; H3[1][13]=n.y; H3[1][14]=n.z;   H3[1][15]=0.; H3[1][16]=0.; H3[1][17]=0.;
    
    H3[2][0]=0.; H3[2][1]=0.; H3[2][2]=0.;    H3[2][3]= HH5[0]; H3[2][4]=HH5[1]; H3[2][5]=HH5[2];
    H3[2][6]=HH6[0]; H3[2][7]=HH6[1]; H3[2][8]=HH6[2];    H3[2][9]=0.; H3[2][10]=0.; H3[2][11]=0.;
    H3[2][12]=0.; H3[2][13]=0.; H3[2][14]=0.;     H3[2][15]=n.x; H3[2][16]=n.y; H3[2][17]=n.z;
    
    //-------------dcsi=H1 da -----------------------------------
    PetscReal H1[6][18],NM1[3][3],NM2[3][3],ContraM[3][3],ContraMinv[3][3],CC1[3][6],CC2[3][6],DD1[3][6],DD2[3][6];
    
    for (i=0;i<3;i++){
      for (j=0;j<3;j++){
	sum1=0.0; sum2=0.0;
	for (m=0;m<3;m++){
	  sum1+=IMnTPn[i][m]*M1[m][j]/(2.*A);
	  sum2+=IMnTPn[i][m]*M2[m][j]/(2.*A);
	}
	NM1[i][j]=sum1; NM2[i][j]=sum2;
      }
    }
    
    ContraM[0][0]=dx21.x; ContraM[0][1]=dx21.y; ContraM[0][2]=dx21.z;
    ContraM[1][0]=dx31.x; ContraM[1][1]=dx31.y; ContraM[1][2]=dx31.z;
    ContraM[2][0]=n.x;    ContraM[2][1]=n.y;    ContraM[2][2]=n.z;
    
    INV(ContraM,ContraMinv);
    
    CC1[0][0]=0.;    CC1[0][1]=0.;    CC1[0][2]=0.;    CC1[0][3]=-gc1.x;  CC1[0][4]=-gc1.y;    CC1[0][5]=-gc1.z;
    CC1[1][0]=gc1.x; CC1[1][1]=gc1.y; CC1[1][2]=gc1.z; CC1[1][3]=0.;      CC1[1][4]=0.;        CC1[1][5]=0.;   
    CC1[2][0]=gc1.x*NM1[0][0]+gc1.y*NM1[1][0]+gc1.z*NM1[2][0];    
    CC1[2][1]=gc1.x*NM1[0][1]+gc1.y*NM1[1][1]+gc1.z*NM1[2][1];  
    CC1[2][2]=gc1.x*NM1[0][2]+gc1.y*NM1[1][2]+gc1.z*NM1[2][2];  
    CC1[2][3]=gc1.x*NM2[0][0]+gc1.y*NM2[1][0]+gc1.z*NM2[2][0]; 
    CC1[2][4]=gc1.x*NM2[0][1]+gc1.y*NM2[1][1]+gc1.z*NM2[2][1]; 
    CC1[2][5]=gc1.x*NM2[0][2]+gc1.y*NM2[1][2]+gc1.z*NM2[2][2]; 
    
    CC2[0][0]=0.;    CC2[0][1]=0.;    CC2[0][2]=0.;    CC2[0][3]=-gc2.x;  CC2[0][4]=-gc2.y;    CC2[0][5]=-gc2.z;
    CC2[1][0]=gc2.x; CC2[1][1]=gc2.y; CC2[1][2]=gc2.z; CC2[1][3]=0.;      CC2[1][4]=0.;        CC2[1][5]=0.;   
    CC2[2][0]=gc2.x*NM1[0][0]+gc2.y*NM1[1][0]+gc2.z*NM1[2][0];    
    CC2[2][1]=gc2.x*NM1[0][1]+gc2.y*NM1[1][1]+gc2.z*NM1[2][1];  
    CC2[2][2]=gc2.x*NM1[0][2]+gc2.y*NM1[1][2]+gc2.z*NM1[2][2];  
    CC2[2][3]=gc2.x*NM2[0][0]+gc2.y*NM2[1][0]+gc2.z*NM2[2][0]; 
    CC2[2][4]=gc2.x*NM2[0][1]+gc2.y*NM2[1][1]+gc2.z*NM2[2][1]; 
    CC2[2][5]=gc2.x*NM2[0][2]+gc2.y*NM2[1][2]+gc2.z*NM2[2][2];
    
    for (i=0;i<3;i++){
      for (j=0;j<6;j++){
	sum1=0.0;sum2=0.0;
	for (m=0;m<3;m++){
	  sum1+=ContraMinv[i][m]*CC1[m][j];
	  sum2+=ContraMinv[i][m]*CC2[m][j];
	}
	DD1[i][j]=sum1; DD2[i][j]=sum2;
      }
    }
    //line 0
    H1[0][0]=0.; H1[0][1]=0.; H1[0][2]=0.;      
    H1[0][3]=aa4[0]*DD1[0][0]+aa4[1]*DD1[1][0]+aa4[2]*DD1[2][0]; H1[0][4]=aa4[0]*DD1[0][1]+aa4[1]*DD1[1][1]+aa4[2]*DD1[2][1]; H1[0][5]=aa4[0]*DD1[0][2]+aa4[1]*DD1[1][2]+aa4[2]*DD1[2][2];
    H1[0][6]=aa4[0]*DD1[0][3]+aa4[1]*DD1[1][3]+aa4[2]*DD1[2][3]; H1[0][7]=aa4[0]*DD1[0][4]+aa4[1]*DD1[1][4]+aa4[2]*DD1[2][4]; H1[0][8]=aa4[0]*DD1[0][5]+aa4[1]*DD1[1][5]+aa4[2]*DD1[2][5];   
    H1[0][9]=gc1.x; H1[0][10]=gc1.y; H1[0][11]=gc1.z;  H1[0][12]=0.; H1[0][13]=0.; H1[0][14]=0.;     H1[0][15]=0.; H1[0][16]=0.; H1[0][17]=0.;
    //line 1
    H1[1][0]= 0.; H1[1][1]=0.; H1[1][2]=0.;     
    H1[1][3]=aa4[0]*DD2[0][0]+aa4[1]*DD2[1][0]+aa4[2]*DD2[2][0]; H1[1][4]=aa4[0]*DD2[0][1]+aa4[1]*DD2[1][1]+aa4[2]*DD2[2][1]; H1[1][5]=aa4[0]*DD2[0][2]+aa4[1]*DD2[1][2]+aa4[2]*DD2[2][2];
    H1[1][6]=aa4[0]*DD2[0][3]+aa4[1]*DD2[1][3]+aa4[2]*DD2[2][3]; H1[1][7]=aa4[0]*DD2[0][4]+aa4[1]*DD2[1][4]+aa4[2]*DD2[2][4]; H1[1][8]=aa4[0]*DD2[0][5]+aa4[1]*DD2[1][5]+aa4[2]*DD2[2][5];    
    H1[1][9]= gc2.x; H1[1][10]=gc2.y; H1[1][11]=gc2.z;  H1[1][12]=0.; H1[1][13]=0.; H1[1][14]=0.; H1[1][15]=0.; H1[1][16]=0.; H1[1][17]=0.;
    //line 2
    H1[2][0]=0.; H1[2][1]=0.; H1[2][2]=0.;     
    H1[2][3]=aa5[0]*DD1[0][0]+aa5[1]*DD1[1][0]+aa5[2]*DD1[2][0]; H1[2][4]=aa5[0]*DD1[0][1]+aa5[1]*DD1[1][1]+aa5[2]*DD1[2][1]; H1[2][5]=aa5[0]*DD1[0][2]+aa5[1]*DD1[1][2]+aa5[2]*DD1[2][2];
    H1[2][6]=aa5[0]*DD1[0][3]+aa5[1]*DD1[1][3]+aa5[2]*DD1[2][3]; H1[2][7]=aa5[0]*DD1[0][4]+aa5[1]*DD1[1][4]+aa5[2]*DD1[2][4]; H1[2][8]=aa5[0]*DD1[0][5]+aa5[1]*DD1[1][5]+aa5[2]*DD1[2][5];    
    H1[2][9]=0.; H1[2][10]=0.; H1[2][11]=0.; H1[2][12]= gc1.x; H1[2][13]=gc1.y; H1[2][14]=gc1.z; H1[2][15]=0.; H1[2][16]=0.; H1[2][17]=0.;
    //line 3
    H1[3][0]=0.; H1[3][1]=0.; H1[3][2]=0.;     
    H1[3][3]=aa5[0]*DD2[0][0]+aa5[1]*DD2[1][0]+aa5[2]*DD2[2][0]; H1[3][4]=aa5[0]*DD2[0][1]+aa5[1]*DD2[1][1]+aa5[2]*DD2[2][1]; H1[3][5]=aa5[0]*DD2[0][2]+aa5[1]*DD2[1][2]+aa5[2]*DD2[2][2];
    H1[3][6]=aa5[0]*DD2[0][3]+aa5[1]*DD2[1][3]+aa5[2]*DD2[2][3]; H1[3][7]=aa5[0]*DD2[0][4]+aa5[1]*DD2[1][4]+aa5[2]*DD2[2][4]; H1[3][8]=aa5[0]*DD2[0][5]+aa5[1]*DD2[1][5]+aa5[2]*DD2[2][5];    
    H1[3][9]= 0.; H1[3][10]=0.; H1[3][11]=0.;  H1[3][12]=gc2.x; H1[3][13]=gc2.y; H1[3][14]=gc2.z;      H1[3][15]=0.; H1[3][16]=0.; H1[3][17]= 0.;
    //line 4
    H1[4][0]=0.; H1[4][1]=0.; H1[4][2]=0.;      
    H1[4][3]=aa6[0]*DD1[0][0]+aa6[1]*DD1[1][0]+aa6[2]*DD1[2][0]; H1[4][4]=aa6[0]*DD1[0][1]+aa6[1]*DD1[1][1]+aa6[2]*DD1[2][1]; H1[4][5]=aa6[0]*DD1[0][2]+aa6[1]*DD1[1][2]+aa6[2]*DD1[2][2];
    H1[4][6]=aa6[0]*DD1[0][3]+aa6[1]*DD1[1][3]+aa6[2]*DD1[2][3]; H1[4][7]=aa6[0]*DD1[0][4]+aa6[1]*DD1[1][4]+aa6[2]*DD1[2][4]; H1[4][8]=aa6[0]*DD1[0][5]+aa6[1]*DD1[1][5]+aa6[2]*DD1[2][5];     
    H1[4][9]= 0.; H1[4][10]=0.; H1[4][11]=0.;  H1[4][12]=0.; H1[4][13]=0.; H1[4][14]=0.;     H1[4][15]=gc1.x; H1[4][16]=gc1.y; H1[4][17]=gc1.z;
    //line5
    H1[5][0]=0.; H1[5][1]=0.; H1[5][2]=0.;      
    H1[5][3]=aa6[0]*DD2[0][0]+aa6[1]*DD2[1][0]+aa6[2]*DD2[2][0]; H1[5][4]=aa6[0]*DD2[0][1]+aa6[1]*DD2[1][1]+aa6[2]*DD2[2][1]; H1[5][5]=aa6[0]*DD2[0][2]+aa6[1]*DD2[1][2]+aa6[2]*DD2[2][2];
    H1[5][6]=aa6[0]*DD2[0][3]+aa6[1]*DD2[1][3]+aa6[2]*DD2[2][3]; H1[5][7]=aa6[0]*DD2[0][4]+aa6[1]*DD2[1][4]+aa6[2]*DD2[2][4]; H1[5][8]=aa6[0]*DD2[0][5]+aa6[1]*DD2[1][5]+aa6[2]*DD2[2][5];     
    H1[5][9]=0.;  H1[5][10]=0.; H1[5][11]=0.;  H1[5][12]=0.; H1[5][13]=0.; H1[5][14]=0.;      H1[5][15]= gc2.x; H1[5][16]=gc2.y; H1[5][17]=gc2.z;
    
    //-------------da=H2 du -----------------------------------
    
    PetscReal H2[18][18],c;      
    
    for (i=0;i<18;i++){
      for (j=0;j<18;j++){H2[i][j]=0.0;}}
    
    i=0;j=3;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=0;j=6;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    
    i=3;j=6;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=3;j=0;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;          
    
    i=6;j=0;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=6;j=3;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;            
    
    i=9;j=3;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=9;j=9;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;             
    
    i=12;j=6;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=12;j=12;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;            
    
    i=15;j=0;c=-1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c;
    i=15;j=15;c=1.; 
    H2[i][j]=c;  H2[i+1][j+1]=c;  H2[i+2][j+2]=c; 
    
    //-------------Bb=-2[Bbar1 H1 H2+Tinv H3 H2]-----------------------------------
    PetscReal Bb[3][18];
    
    for (i=0;i<3;i++){
      for (j=0;j<18;j++){
	sum1=0.0;
	for (m=0;m<6;m++){
	  for (p=0;p<18;p++){
	    sum1+=Bbar1[i][m]*H1[m][p]*H2[p][j];
	  }
	}
	sum2=0.0;
	for (m=0;m<3;m++){
	  for (p=0;p<18;p++){
	    sum2+=Tinv[i][m]*H3[m][p]*H2[p][j];
	  }
	}
	Bb[i][j]=-2.0*(sum1+sum2); 
      }
    }
    
    for (i=0; i<18; i++){
      sum=0.0;
      for (m=0;m<3;m++){
	sum+=Bb[m][i]*S[m]*A0*pow(h0,3.)/12.;
      }
      _Fb[i]=sum;
    }
       
  } else if (curvature==6) { //subdivision surface
    //PetscPrintf(PETSC_COMM_SELF, "Check IF in curva!\n");
    PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12], b1, b2, b3, nA0, nA;
    struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, ndX21, ndX31, nn, ng1, ng2, ng;
    PetscInt       v=ibm->val[ec], node, nob=1;
    
    Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
    Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
    
    Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
    Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;
       
    Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
    Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
    
    Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
    Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
      
    Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
    Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;
    //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch outside!\n");
    if (ibm->ire[ec]==0) { //-------------------------------------regular patch-----------------------------------------
    //PetscPrintf(PETSC_COMM_SELF, "Check if regular patch inside!\n");
      for (i=0; i<12; i++) {
	if (ibm->patch[16*ec+i]!=1000000) {
	  node = ibm->patch[16*ec+i];
	  x[i] = ibm->x_bp[node];  y[i] = ibm->y_bp[node];  z[i] = ibm->z_bp[node];
	}
      }
	
      for (i=0; i<12; i++) {
	if (ibm->patch[16*ec+i]==1000000) {nob=0;}
      }

      Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
      Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
      Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
      
      for (i=0; i<12; i++) {
	Aaa.x += Nab[0][i]*x[i];
	Aaa.y += Nab[0][i]*y[i];
	Aaa.z += Nab[0][i]*z[i];
	
	Abb.x += Nab[1][i]*x[i];
	Abb.y += Nab[1][i]*y[i];
	Abb.z += Nab[1][i]*z[i];
	
	Aab.x += Nab[2][i]*x[i];
	Aab.y += Nab[2][i]*y[i];
	Aab.z += Nab[2][i]*z[i];
      }
       
      ndx21.x=0.; ndx21.y=0.; ndx21.z=0.;
      ndx31.x=0.; ndx31.y=0.; ndx31.z=0.;
      
      for (i=0; i<12; i++) {
	ndx21.x += Na[0][i]*x[i];
	ndx21.y += Na[0][i]*y[i];
	ndx21.z += Na[0][i]*z[i];
	
	ndx31.x += Na[1][i]*x[i];
	ndx31.y += Na[1][i]*y[i];
	ndx31.z += Na[1][i]*z[i];
      }
     
      nn=UNIT(CROSS(ndx21, ndx31));
      ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
      ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];
      nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
      nA = 0.5*SIZE(CROSS(ndx21, ndx31));
 
      gc1=CROSS(ndx31, nn);
      gc1=AMULT(1./DOT(ndx21, gc1), gc1);
      
      gc2=CROSS(nn, ndx21);
      gc2=AMULT(1./DOT(ndx31, gc2), gc2);

      k[0] = DOT(Aaa,nn);
      k[1] = DOT(Abb,nn);
      k[2] = DOT(Aab,nn);

      ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
      ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
      ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
      //---------------------Compute E and S for membrane ---------------------------------------
      PetscReal  g[3], Em[3], Sm[3];
      
      g[0] = DOT(ndx21,ndx21);
      g[1] = DOT(ndx31,ndx31);
      g[2] = DOT(ndx21,ndx31);
      
      Em[0] = 0.5*(g[0] - ibm->G[dof*ec]);
      Em[1] = 0.5*(g[1] - ibm->G[dof*ec+1]);
      Em[2] = g[2] - ibm->G[dof*ec+2];
      //PetscPrintf(PETSC_COMM_SELF, "Check before IF C Law!\n");

      if (ConstitutiveLawNonLinear) {
	MembraneNonLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
	//StressLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
      } else if (manufactured) {
	StressManufactured(ec, X1, X2, X3, Em, Sm, 1, ibm);
      } else {
        //PetscPrintf(PETSC_COMM_SELF, "Check before SL!\n");
	StressLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
      }

      CalcCurvStressStrainxyz(ec, k, Em, Sm, 1, 0, fem);
      
      //---------------------Compute E and S for bending ---------------------------------------
      Eb[0] = ibm->kve0[ec*dof] - k[0];
      Eb[1] = ibm->kve0[ec*dof+1] - k[1];
      Eb[2] = 2*(ibm->kve0[ec*dof+2] - k[2]);
    
      if (ConstitutiveLawNonLinear) {
	BendingNonLinear(ec, X1, X2, X3, Eb, S, 1, ibm);
	//StressLinear(ec, X1, X2, X3, Eb, S, 1, ibm);
      } else if (manufactured) {
	StressManufactured(ec, X1, X2, X3, Eb, S, 1, ibm);
      } else {
	StressLinear(ec, X1, X2, X3, Eb, S, 1, ibm); 
      }

      CalcCurvStressStrainxyz(ec, k, Eb, S, 1, 1, fem);
      
      //-----------------------------------Forming membrane matrix------------------------------------------
      PetscReal  Bm[3][36];

      for (i=0; i<12; i++) {

      	Bm[0][3*i] = Na[0][i]*ndx21.x;   Bm[0][3*i+1] = Na[0][i]*ndx21.y;   Bm[0][3*i+2] = Na[0][i]*ndx21.z;
      	Bm[1][3*i] = Na[1][i]*ndx31.x;   Bm[1][3*i+1] = Na[1][i]*ndx31.y;   Bm[1][3*i+2] = Na[1][i]*ndx31.z;
      	Bm[2][3*i] = Na[0][i]*ndx31.x + Na[1][i]*ndx21.x;   Bm[2][3*i+1] = Na[0][i]*ndx31.y + Na[1][i]*ndx21.y;   Bm[2][3*i+2] = Na[0][i]*ndx31.z + Na[1][i]*ndx21.z;
      }

      //-----------------------------------Forming bending  matrix------------------------------------------
      PetscReal  Bs[3][36];

      for (i=0; i<12; i++) {
      	ng1 = AMULT(Na[0][i], gc1);   ng2 = AMULT(Na[1][i], gc2);
      	ng = PLUS(ng1, ng2);
      	b1 = DOT(ng, Aaa);   b2 = DOT(ng, Abb);   b3 = DOT(ng, Aab);

      	Bs[0][3*i] = -(Nab[0][i] + b1)*nn.x;   Bs[0][3*i+1] = -(Nab[0][i] + b1)*nn.y;   Bs[0][3*i+2] = -(Nab[0][i] + b1)*nn.z;
      	Bs[1][3*i] = -(Nab[1][i] + b2)*nn.x;   Bs[1][3*i+1] = -(Nab[1][i] + b2)*nn.y;   Bs[1][3*i+2] = -(Nab[1][i] + b2)*nn.z;
      	Bs[2][3*i] = -2*(Nab[2][i] + b3)*nn.x;   Bs[2][3*i+1] = -2*(Nab[2][i] + b3)*nn.y;   Bs[2][3*i+2] = -2*(Nab[2][i] + b3)*nn.z;
      }

      //--------------------------------------calculating membrane force-----------------------------
      PetscReal  _Fm[36];

      for (i=0; i<36; i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
      	  sum += Bm[m][i]*Sm[m]*A0*h0;
      	}
      	_Fm[i] = sum;	
      }
           
      //--------------------------------------calculating bending force-----------------------------
      for (i=0; i<36; i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
      	  sum += Bs[m][i]*S[m]*A0*pow(h0,3.)/12.;
      	}
      	_Fb[i] = sum + _Fm[i];
      }      

      //---------------------------------------calculating internal energy--------------------------
      PetscInt  n1e=ibm->nv1[ec], n2e=ibm->nv2[ec], n3e=ibm->nv3[ec];
      sum = 0.;
      for (i=0; i<3; i++) {
	sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*pow(h0,3.)/12.*Eb[i]*S[i]);
	fem->FC[dof*ec+i] = Sm[i]*A0*h0 + A0*pow(h0,2.)/2.*S[i];
      }
      fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;

    } else if (ibm->ire[ec]==1) { //----------------------irregular patch---------------------------
      //PetscPrintf(PETSC_COMM_SELF, "Check if irregular patch inside!\n");
      for (i=0; i<(v+6); i++) {
	if (ibm->patch[16*ec+i]==1000000) {nob=0;}
      }
      
      PetscReal  w;	
      PetscReal  **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));
      for (i=0; i<(v+6); i++) {
	X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
      }      
      
      for (i=0; i<(v+6); i++) {
	for (j=0; j<3; j++) {
	  X0[i][j] = 0.;
	}
      }     
      
      for (i=0; i<(v+6); i++) {
	if (ibm->patch[16*ec+i]!=1000000) {
	  node = ibm->patch[16*ec+i];
	  X0[i][0] = ibm->x_bp[node];  X0[i][1] = ibm->y_bp[node];  X0[i][2] = ibm->z_bp[node];
	}
      }
      
      w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.)); 
            
      PetscReal  B3[12][v+6], B2[12][v+12], B1[v+12][v+6];
      
      //-----B2=dU2/dU1
      for(i=0; i<12; i++) {
	for (j=0; j<(v+12); j++) {
	  B2[i][j] = 0.;
	}
      }
      
      //------B1=dU1/dU0
      for(i=0; i<(v+12); i++) {
	for (j=0; j<(v+6); j++) {
	  B1[i][j] = 0.;
	}
      }
      
      B2[0][v+9] = 1.;  
      B2[1][v+6] = 1.;  
      B2[2][v+4] = 1.;  
      B2[3][v+1] = 1.;  
      B2[4][v+2] = 1.;  
      B2[5][v+5] = 1.;  
      B2[6][v] = 1.;      
      B2[7][1] = 1.;     
      B2[8][v+3] = 1.; 
      B2[9][v-1] = 1.; 
      B2[10][0] = 1.;     
      B2[11][2] = 1.;     
                
      B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
      B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
      B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
      if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
      if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
      B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
      B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
      B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
      B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
      B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
      B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
      B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
      B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
      B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
      B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
      B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
      B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
      B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
      B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;    
           
      for (i=0; i<12; i++) {
	for (j=0; j<(v+6); j++) {
	  B3[i][j] = 0.;
	  for (m=0; m<(v+12); m++) {
	    B3[i][j] += B2[i][m]*B1[m][j];
	  }
	}
      }
                   	
      PetscReal  INa[2][v+6], INab[3][v+6];
      PetscReal  sum1, sum2, sum3, sum4, sum5;
      
      for (j=0; j<(v+6); j++) {
	sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
	for (i=0; i<12; i++) {
	  sum1 += B3[i][j]*Na[0][i];
	  sum2 += B3[i][j]*Na[1][i];
	  sum3 += B3[i][j]*Nab[0][i];
	  sum4 += B3[i][j]*Nab[1][i];
	  sum5 += B3[i][j]*Nab[2][i];
	}
	INa[0][j] = -2*sum1;
	INa[1][j] = -2*sum2;
	INab[0][j] = 4*sum3;
	INab[1][j] = 4*sum4;
	INab[2][j] = 4*sum5;
      }     

      Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
      Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
      Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
	
      for (i=0; i<(v+6); i++) {
	Aaa.x += INab[0][i]*X0[i][0];
	Aaa.y += INab[0][i]*X0[i][1];
	Aaa.z += INab[0][i]*X0[i][2];
	
	Abb.x += INab[1][i]*X0[i][0];
	Abb.y += INab[1][i]*X0[i][1];
	Abb.z += INab[1][i]*X0[i][2];
	
	Aab.x += INab[2][i]*X0[i][0];
	Aab.y += INab[2][i]*X0[i][1];
	Aab.z += INab[2][i]*X0[i][2];
      }
	     
      ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
      ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
      
      for (i=0; i<(v+6); i++) {
	ndx21.x += INa[0][i]*X0[i][0];
	ndx21.y += INa[0][i]*X0[i][1];
	ndx21.z += INa[0][i]*X0[i][2];
	
	ndx31.x += INa[1][i]*X0[i][0];
	ndx31.y += INa[1][i]*X0[i][1];
	ndx31.z += INa[1][i]*X0[i][2];
      }

      nn=UNIT(CROSS(ndx21,ndx31));
      ndX21.x = ibm->G1[ec*dof]; ndX21.y = ibm->G1[ec*dof+1]; ndX21.z = ibm->G1[ec*dof+2];
      ndX31.x = ibm->G2[ec*dof]; ndX31.y = ibm->G2[ec*dof+1]; ndX31.z = ibm->G2[ec*dof+2];

      dx21.x = X0[1][0] - X0[0][0]; dx21.y = X0[1][1] - X0[0][1]; dx21.z = X0[1][2] - X0[0][2];
      dx31.x = X0[v][0] - X0[0][0]; dx31.y = X0[v][1] - X0[0][1]; dx31.z = X0[v][2] - X0[0][2];

      nA0 = 0.5*SIZE(CROSS(ndX21, ndX31));
      nA = 0.5*SIZE(CROSS(ndx21, ndx31));     

      k[0] = DOT(Aaa,nn);
      k[1] = DOT(Abb,nn);
      k[2] = DOT(Aab,nn);
	
      gc1=CROSS(ndx31, nn);
      gc1=AMULT(1./DOT(ndx21, gc1), gc1);
      
      gc2=CROSS(nn, ndx21);
      gc2=AMULT(1./DOT(ndx31, gc2), gc2); 
      
      ibm->g1[ec*dof] = ndx21.x; ibm->g1[ec*dof+1] = ndx21.y; ibm->g1[ec*dof+2] = ndx21.z;
      ibm->g2[ec*dof] = ndx31.x; ibm->g2[ec*dof+1] = ndx31.y; ibm->g2[ec*dof+2] = ndx31.z;
      ibm->g3[ec*dof] = nn.x; ibm->g3[ec*dof+1] = nn.y; ibm->g3[ec*dof+2] = nn.z;
      //---------------------Compute E and S for membrane ---------------------------------------
      PetscReal  g[3], Em[3], Sm[3];
      
      g[0] = DOT(ndx21,ndx21);
      g[1] = DOT(ndx31,ndx31);
      g[2] = DOT(ndx21,ndx31);
      
      Em[0] = 0.5*(g[0] - ibm->G[dof*ec]);
      Em[1] = 0.5*(g[1] - ibm->G[dof*ec+1]);
      Em[2] = g[2] - ibm->G[dof*ec+2];
 
      if (ConstitutiveLawNonLinear) {
	MembraneNonLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
	//StressLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
      } else if (manufactured) {
	StressManufactured(ec, X1, X2, X3, Em, Sm, 1, ibm);
      } else {
        //PetscPrintf(PETSC_COMM_SELF, "Check b SL!\n");
	StressLinear(ec, X1, X2, X3, Em, Sm, 1, ibm);
      }

      CalcCurvStressStrainxyz(ec, k, Em, Sm, 1, 0, fem);
      
      //---------------------Compute E and S for bending ---------------------------------------
      Eb[0] = ibm->kve0[ec*dof]-k[0];
      Eb[1] = ibm->kve0[ec*dof+1]-k[1];
      Eb[2] = 2*(ibm->kve0[ec*dof+2]-k[2]);

      if (ConstitutiveLawNonLinear) {
	BendingNonLinear(ec, X1, X2, X3, Eb, S, 1, ibm);
	//StressLinear(ec, X1, X2, X3, Eb, S, 1, ibm);
      } else if (manufactured) {
	StressManufactured(ec, X1, X2, X3, Eb, S, 1, ibm);
      } else {
	StressLinear(ec, X1, X2, X3, Eb, S, 1, ibm);
      } 
      CalcCurvStressStrainxyz(ec, k, Eb, S, 1, 1, fem);
      //-----------------------------------Forming membrane matrix------------------------------
      PetscReal  Bm[3][3*(v+6)];
      for (i=0; i<(v+6); i++) {
	
      	Bm[0][3*i] = INa[0][i]*ndx21.x;   Bm[0][3*i+1] = INa[0][i]*ndx21.y;   Bm[0][3*i+2] = INa[0][i]*ndx21.z;
      	Bm[1][3*i] = INa[1][i]*ndx31.x;   Bm[1][3*i+1] = INa[1][i]*ndx31.y;   Bm[1][3*i+2] = INa[1][i]*ndx31.z;
      	Bm[2][3*i] = INa[0][i]*ndx31.x + INa[1][i]*ndx21.x;   Bm[2][3*i+1] = INa[0][i]*ndx31.y + INa[1][i]*ndx21.y;   Bm[2][3*i+2] = INa[0][i]*ndx31.z + INa[1][i]*ndx21.z;
      }
      //-----------------------------------Forming bending matrix------------------------------------------
      PetscReal  B4[3][3*(v+6)];
      
      for (i=0; i<(v+6); i++) {
      	ng1 = AMULT(INa[0][i], gc1);   ng2 = AMULT(INa[1][i], gc2);
      	ng = PLUS(ng1, ng2);
      	b1 = DOT(ng, Aaa);   b2 = DOT(ng, Abb);   b3 = DOT(ng, Aab);
	
      	B4[0][3*i] = -(INab[0][i] + b1)*nn.x;   B4[0][3*i+1] = -(INab[0][i] + b1)*nn.y;   B4[0][3*i+2] = -(INab[0][i] + b1)*nn.z;
      	B4[1][3*i] = -(INab[1][i] + b2)*nn.x;   B4[1][3*i+1] = -(INab[1][i] + b2)*nn.y;   B4[1][3*i+2] = -(INab[1][i] + b2)*nn.z;
      	B4[2][3*i] = -2*(INab[2][i] + b3)*nn.x;   B4[2][3*i+1] = -2*(INab[2][i] + b3)*nn.y;   B4[2][3*i+2] = -2*(INab[2][i] + b3)*nn.z;
      }
             
      /* //--------------------------------------calculating membrane force----------------------------- */
     
      PetscReal  _Fm[42];

      for (i=0; i<3*(v+6); i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
      	  sum += Bm[m][i]*Sm[m]*A0*h0;
      	}
      	_Fm[i] = sum;  	
      }
         
      //--------------------------------------calculating bending force-----------------------------
      for (i=0; i<3*(v+6); i++){
      	sum = 0.0;
      	for (m=0; m<3; m++){
      	  sum += B4[m][i]*S[m]*A0*pow(h0,3.)/12.;
      	}
      	_Fb[i] = sum + _Fm[i];
      }    
    
      //---------------------------------------calculating internal energy---------------------------
      PetscInt  n1e=ibm->nv1[ec], n2e=ibm->nv2[ec], n3e=ibm->nv3[ec];
      sum = 0.;
      for (i=0; i<3; i++) {
	sum += 0.5*(A0*h0*Em[i]*Sm[i] + A0*pow(h0,3.)/12.*Eb[i]*S[i]);
	fem->FC[dof*ec+i] = Sm[i]*A0*h0 + A0*pow(h0,2.)/2.*S[i];
      }
      fem->IE[n1e] += sum/3.;  fem->IE[n2e] += sum/3.;  fem->IE[n3e] += sum/3.;
      //--------------------------free----------------------   
      for (i=0; i<(v+6); i++) {
	free(X0[i]);
      }
     
      free(X0);
    }     
  }
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode InitGhost(FE *fem) {

  IBMNodes  *ibm=fem->ibm;
  PetscInt  ec;
 
  PetscMalloc(ibm->n_ghosts*sizeof(PetscInt), &(ibm->belmtsedge));
  PetscMalloc(ibm->n_ghosts*sizeof(PetscInt), &(ibm->belmts));
  PetscMalloc(ibm->n_ghosts*sizeof(PetscInt), &(ibm->edgefrontnodes));
  PetscMalloc(ibm->n_ghosts*sizeof(PetscInt), &(ibm->edgefrontnodesI));
  
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4x));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5x));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6x));
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4y));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5y));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6y));
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4z));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5z));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6z));
  
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4x0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5x0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6x0));
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4y0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5y0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6y0));
  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p4z0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p5z0));  PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->p6z0));

  if (ibm->n_edge){Ghost(ibm);}

  if (curvature==1) {
    PatchLoc(ibm);
    GhostLoc(fem);

    for (ec=0; ec<ibm->n_elmt; ec++) {
      ibm->p4x0[ec] = ibm->p4x[ec];  ibm->p4y0[ec] = ibm->p4y[ec];  ibm->p4z0[ec] = ibm->p4z[ec];
      ibm->p5x0[ec] = ibm->p5x[ec];  ibm->p5y0[ec] = ibm->p5y[ec];  ibm->p5z0[ec] = ibm->p5z[ec];
      ibm->p6x0[ec] = ibm->p6x[ec];  ibm->p6y0[ec] = ibm->p6y[ec];  ibm->p6z0[ec] = ibm->p6z[ec];
    }
  }
  if (curvature==6) {
    IrrVer(ibm);
    Patch(ibm);
  }

  Kve0(fem);
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode PatchLoc(IBMNodes *ibm) {

  PetscInt  n4e, n5e, n6e, ec;

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n4e = ibm->nv4[ec];  n5e = ibm->nv5[ec];  n6e = ibm->nv6[ec];
    
    if(n4e!=1000000){ //So its patch is availabel (not ghost node)
      ibm->p4x[ec] = ibm->x_bp[n4e];  ibm->p4y[ec] = ibm->y_bp[n4e];  ibm->p4z[ec] = ibm->z_bp[n4e];
    }
    if(n5e!=1000000){ //So its patch is availabel (not ghost node)
      ibm->p5x[ec] = ibm->x_bp[n5e];  ibm->p5y[ec] = ibm->y_bp[n5e];  ibm->p5z[ec] = ibm->z_bp[n5e];
    }
    if(n6e!=1000000){  //So its patch is availabel (not ghost node)
      ibm->p6x[ec] = ibm->x_bp[n6e];  ibm->p6y[ec] = ibm->y_bp[n6e];  ibm->p6z[ec] = ibm->z_bp[n6e];
    }
    
  }

  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode Ghost(IBMNodes *ibm) {

  PetscInt  n4e, n5e, n6e, nv, side, II, ec, gc=0;

  for (ec=0; ec<ibm->n_elmt; ec++) {
    n4e = ibm->nv4[ec];  n5e = ibm->nv5[ec];  n6e = ibm->nv6[ec];
    
    side = 0;
    if(n4e==1000000) {II = 1;  nv = ibm->nv1[ec];  side += 1;}
    if(n5e==1000000) {II = 2;  nv = ibm->nv2[ec];  side += 1;}
    if(n6e==1000000) {II = 3;  nv = ibm->nv3[ec];  side += 1;}
    
    if(side>1) {
      PetscPrintf(PETSC_COMM_SELF, "element %d has %d boundary sides\n", ec, side);
      PetscPrintf(PETSC_COMM_SELF, "increase serach criteria || check the mesh\n");
    }
    
    if(side==1) {
      ibm->belmts[gc] = ec;  ibm->edgefrontnodes[gc] = nv;  ibm->edgefrontnodesI[gc] = II;
      gc += 1;
    }//if on boundary
    
  }//elements

  //add corresponding edge for each boundary element
  PetscInt  i, start, end, edge, edge_n, be, bn, n1e, n2e, n3e, sum;
  //choose an boundary element
  for (ec=0; ec<ibm->n_ghosts; ec++) {
    be = ibm->belmts[ec];
    n1e = ibm->nv1[be];  n2e = ibm->nv2[be];  n3e = ibm->nv3[be];
    
    //for each edge
    for (edge_n=0; edge_n<ibm->n_edge; edge_n++){

      //compute start&end
      start = 0; end = 0;
      for (edge=0; edge<edge_n+1; edge++) {
  	end += ibm->n_bnodes[edge];
      }
      start = end-ibm->n_bnodes[edge_n];

      //nodes on each edge
      sum = 0;
      for (i=start; i<end; i++) {
  	bn = ibm->bnodes[i];
  	if(bn==n1e ||  bn==n2e || bn==n3e) {sum += 1;}
      }//node on a edge
      
      //if element and edge share two nodes
      if(sum==2) {ibm->belmtsedge[ec] = edge_n;}
      
    }//edges
    
  }//elements on edge

  return(0);
}

//---------------------------------------------------------------------------------------   
PetscErrorCode Kve0(FE *fem) {
  
  IBMNodes       *ibm=fem->ibm;
  PetscReal      sum;
  PetscInt       i, j, m, p, q, ec, n1e, n2e, n3e;
  struct Cmpnts  dX21, dX31, N, Gc1, Gc2;
  struct Cmpnts  A4, A5, A6;
  struct Cmpnts  X1, X2, X3, X4, X5, X6;
  PetscReal      Ze[3], k0[3], kcur[3];
  PetscReal      T[3][3], Tinv[3][3];
  PetscReal      Csi4, Eta4, Z4, Csi5, Eta5, Z5, Csi6, Eta6, Z6;
  //Gc=contravariant
  
  for (ec=0; ec<ibm->n_elmt; ec++) {

    n1e=ibm->nv1[ec];  n2e=ibm->nv2[ec];  n3e=ibm->nv3[ec];
    //initial location
    X1.x = ibm->x_bp0[n1e];  X1.y = ibm->y_bp0[n1e];  X1.z = ibm->z_bp0[n1e];
    X2.x = ibm->x_bp0[n2e];  X2.y = ibm->y_bp0[n2e];  X2.z = ibm->z_bp0[n2e];
    X3.x = ibm->x_bp0[n3e];  X3.y = ibm->y_bp0[n3e];  X3.z = ibm->z_bp0[n3e];
    X4.x = ibm->p4x0[ec];  X4.y = ibm->p4y0[ec];  X4.z = ibm->p4z0[ec];
    X5.x = ibm->p5x0[ec];  X5.y = ibm->p5y0[ec];  X5.z = ibm->p5z0[ec];
    X6.x = ibm->p6x0[ec];  X6.y = ibm->p6y0[ec];  X6.z = ibm->p6z0[ec];
    
    dX21 = MINUS(X2, X1);  dX31 = MINUS(X3, X1); //dX21:G1 , dX31:G2
    A4 = MINUS(X4, X2);   A5 = MINUS(X5, X3);  A6 = MINUS(X6, X1);
    
    N.x = ibm->Nf_x[ec];  N.y = ibm->Nf_y[ec];  N.z = ibm->Nf_z[ec];
    
    Gc1 = CROSS(dX31, N);
    Gc1 = AMULT(1./DOT(dX21, Gc1), Gc1);
    
    Gc2 = CROSS(N, dX21);
    Gc2 = AMULT(1./DOT(dX31, Gc2), Gc2);
              
    Csi4 = 1.+DOT(A4, Gc1);  Eta4 = DOT(A4, Gc2);    Z4 = DOT(A4, N);
    Csi5 = DOT(A5, Gc1);     Eta5 = 1.+DOT(A5, Gc2); Z5 = DOT(A5, N);
    Csi6 = DOT(A6, Gc1);     Eta6 = DOT(A6, Gc2);    Z6 = DOT(A6, N);
    
    T[0][0] = pow(Csi4, 2.)-Csi4;   T[0][1] = pow(Eta4, 2.)-Eta4;  T[0][2] = Csi4*Eta4;
    T[1][0] = pow(Csi5, 2.)-Csi5;   T[1][1] = pow(Eta5, 2.)-Eta5;  T[1][2] = Csi5*Eta5;
    T[2][0] = pow(Csi6, 2.)-Csi6;   T[2][1] = pow(Eta6, 2.)-Eta6;  T[2][2] = Csi6*Eta6;
    
    INV(T, Tinv);
    
    if(curvature == 1) {
      Ze[0] = Z4;  Ze[1] = Z5;  Ze[2] = Z6;
      
      for (i=0; i<3; i++){
	sum = 0.0;
	for (m=0; m<3; m++){
	  sum += 2*Tinv[i][m]*Ze[m];
	}
	k0[i] = sum;
      }

      CalcCurvStressStrainxyz(ec, k0, k0, k0, 0, 1, fem);

      ibm->kve0[ec*dof] = k0[0];
      ibm->kve0[ec*dof+1] = k0[1];
      ibm->kve0[ec*dof+2] = k0[2];
      
      ibm->kve[ec*dof] = k0[0];
      ibm->kve[ec*dof+1] = k0[1];
      ibm->kve[ec*dof+2] = k0[2];

    } else if (curvature==6) {

      PetscReal      Nab[3][12], x[12], y[12], z[12], Na[2][12];
      struct Cmpnts  Aaa, Abb, Aab, nx1, nx2, nx3, ndx21, ndx31, nn;
      PetscInt       v=ibm->val[ec], node;
     
      Na[0][0] = -0.0247;  Na[0][1] = -0.0309;  Na[0][2] = 0.;  Na[0][3] = -0.4815;  Na[0][4] =  -0.1852;  Na[0][5] =  0.0247; // First derivatives are calculated at element center
      Na[0][6] = 0.4815;  Na[0][7] = 0.;  Na[0][8] = -0.0062;  Na[0][9] = 0.0309;  Na[0][10] =  0.1852;  Na[0][11] =  0.0062;
      
      Na[1][0] = -0.0309;  Na[1][1] = -0.0247;  Na[1][2] = -0.1852;  Na[1][3] = -0.4815;  Na[1][4] =  0.;  Na[1][5] =  -0.0062;
      Na[1][6] = 0.;  Na[1][7] = 0.4815;  Na[1][8] = 0.0247;  Na[1][9] = 0.0062;  Na[1][10] =  0.1852;  Na[1][11] =  0.0309;

      Nab[0][0] = 0.1111;  Nab[0][1] = 0.2222;  Nab[0][2] = -0.2222;  Nab[0][3] = -0.2222;  Nab[0][4] =  0.4444;  Nab[0][5] =  0.1111; //second derivatives are calculated at element center
      Nab[0][6] = -0.2222;  Nab[0][7] = -0.8889;  Nab[0][8] = 0.;  Nab[0][9] = 0.2222;  Nab[0][10] =  0.4444;  Nab[0][11] =  0.;
      
      Nab[1][0] = 0.2222;  Nab[1][1] = 0.1111;  Nab[1][2] = 0.4444;  Nab[1][3] = -0.2222;  Nab[1][4] =  -0.2222;  Nab[1][5] =  0.;
      Nab[1][6] = -0.8889;  Nab[1][7] = -0.2222;  Nab[1][8] = 0.1111;  Nab[1][9] = 0.;  Nab[1][10] =  0.4444;  Nab[1][11] =  0.2222;
      
      Nab[2][0] = 0.1667;  Nab[2][1] = 0.1667;  Nab[2][2] = -0.1111;  Nab[2][3] = 0.2222;  Nab[2][4] =  -0.1111;  Nab[2][5] =  -0.0556;
      Nab[2][6] = -0.4444;  Nab[2][7] = -0.4444;  Nab[2][8] = -0.0556;  Nab[2][9] = 0.0556;  Nab[2][10] =  0.5556;  Nab[2][11] =  0.0556;

      if (ibm->ire[ec]==0) {

	for (i=0; i<12; i++) {

	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    x[i] = ibm->x_bp[node];  y[i] = ibm->y_bp[node];  z[i] = ibm->z_bp[node];
	  }
	}
		
		
	for (i=0; i<12; i++) {
	  if (ibm->patch[16*ec+i]==1000000) {
		PetscPrintf(PETSC_COMM_SELF, "Ghost node missing!\n");
		PetscPrintf(PETSC_COMM_SELF, "Element index is%d \n", ec);
		PetscPrintf(PETSC_COMM_SELF, "\n");
	}
	}
	
	Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;

	for (i=0; i<12; i++) {
	  Aaa.x += Nab[0][i]*x[i];
	  Aaa.y += Nab[0][i]*y[i];
	  Aaa.z += Nab[0][i]*z[i];

	  Abb.x += Nab[1][i]*x[i];
	  Abb.y += Nab[1][i]*y[i];
	  Abb.z += Nab[1][i]*z[i];

	  Aab.x += Nab[2][i]*x[i];
	  Aab.y += Nab[2][i]*y[i];
	  Aab.z += Nab[2][i]*z[i];
	}
       
	ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
	ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
	
	for (i=0; i<12; i++) {
	  ndx21.x += Na[0][i]*x[i];
	  ndx21.y += Na[0][i]*y[i];
	  ndx21.z += Na[0][i]*z[i];
	  
	  ndx31.x += Na[1][i]*x[i];
	  ndx31.y += Na[1][i]*y[i];
	  ndx31.z += Na[1][i]*z[i];
	}
	
	nn=UNIT(CROSS(ndx21, ndx31));

	k0[0] = DOT(Aaa, nn);
	k0[1] = DOT(Abb, nn);
	k0[2] = DOT(Aab, nn);
	
	ibm->G1[dof*ec] = ndx21.x; ibm->G1[dof*ec+1] = ndx21.y; ibm->G1[dof*ec+2] = ndx21.z;
	ibm->G2[dof*ec] = ndx31.x; ibm->G2[dof*ec+1] = ndx31.y; ibm->G2[dof*ec+2] = ndx31.z;

	ibm->G[dof*ec] = DOT(ndx21, ndx21);
	ibm->G[dof*ec+1] = DOT(ndx31, ndx31);
	ibm->G[dof*ec+2] = DOT(ndx21, ndx31);

	ibm->kve0[ec*dof] = k0[0];
	ibm->kve0[ec*dof+1] = k0[1];
	ibm->kve0[ec*dof+2] = k0[2];

	CalcCurvStressStrainxyz(ec, k0, k0, k0, 1, 1, fem);	
	
      }else if (ibm->ire[ec]==1) {
	
	for (i=0; i<(v+6); i++) {
	  if (ibm->patch[16*ec+i]==1000000) {
		PetscPrintf(PETSC_COMM_SELF, "Ghost node missing!\n");
		PetscPrintf(PETSC_COMM_SELF, "Element index is%d \n", ec);
                PetscPrintf(PETSC_COMM_SELF, "\n");	
	}
	}

	PetscReal w;
	PetscReal **X0 = (PetscReal **)malloc((v+6) * sizeof(PetscReal *));
	for (i=0; i<(v+6); i++) {
	  X0[i] = (PetscReal *)malloc(3 * sizeof(PetscReal));
	}	
			
	for (i=0; i<(v+6); i++) {
	  for (j=0; j<3; j++) {
	    X0[i][j] = 0.;
	  }
	}
	
	for (i=0; i<(v+6); i++) {
	  if (ibm->patch[16*ec+i]!=1000000) {
	    node = ibm->patch[16*ec+i];
	    X0[i][0] = ibm->x_bp[node];  X0[i][1] = ibm->y_bp[node];  X0[i][2] = ibm->z_bp[node];
	  }
	}

	w = (1./v)*(0.625 - pow(0.375 + 0.25*PetscCosReal(2*PETSC_PI/v) , 2.));
	             			
	PetscReal B3[12][v+6], B2[12][v+12], B1[v+12][v+6];	
	//-----B2=dU2/dU1 picking matrix
	for(i=0; i<12; i++) {
	  for (j=0; j<v+12; j++) {
	    B2[i][j] = 0.;
	  }
	}
	
	B2[0][v+9] = 1.;  
	B2[1][v+6] = 1.;  
	B2[2][v+4] = 1.;  
	B2[3][v+1] = 1.;  
	B2[4][v+2] = 1.;  
	B2[5][v+5] = 1.;  
	B2[6][v] = 1.;      
	B2[7][1] = 1.;     
	B2[8][v+3] = 1.; 
	B2[9][v-1] = 1.; 
	B2[10][0] = 1.;     
	B2[11][2] = 1.;     
	
	//------B1=dU1/dU0 Loop's mask
	for(i=0; i<(v+12); i++) {
	  for (j=0; j<(v+6); j++) {
	    B1[i][j] = 0.;
	  }
	}
		
	B1[0][0] = 1 - v*w;  for(j=0; j<v; j++) {B1[0][1+j] = w;}
	B1[1][0] = 0.375;  B1[1][1] = 0.375;  B1[1][2] = 0.125;  B1[1][v] = 0.125;
	B1[2][0] = 0.375;  B1[2][1] = 0.125;  B1[2][2] = 0.375;  B1[2][3] = 0.125;
        if (v>5) {B1[v-4][0] = 0.375;  B1[v-4][v-5] = 0.125;  B1[v-4][v-4] = 0.375;  B1[v-4][v-3] = 0.125;}
        if (v>4) {B1[v-3][0] = 0.375;  B1[v-3][v-4] = 0.125;  B1[v-3][v-3] = 0.375;  B1[v-3][v-2] = 0.125;}
        B1[v-2][0] = 0.375;  B1[v-2][v-3] = 0.125;  B1[v-2][v-2] = 0.375;  B1[v-2][v-1] = 0.125;
	B1[v-1][0] = 0.375;  B1[v-1][v-2] = 0.125;  B1[v-1][v-1] = 0.375;  B1[v-1][v] = 0.125;
	B1[v][0] = 0.375;  B1[v][1] = 0.125;  B1[v][v-1] = 0.125;  B1[v][v] = 0.375;
	B1[v+1][0] = 0.125;  B1[v+1][1] = 0.375;  B1[v+1][v] = 0.375;  B1[v+1][v+1] = 0.125;
	B1[v+2][0] = 0.0625;  B1[v+2][1] = 0.625;  B1[v+2][2] = 0.0625;  B1[v+2][v] = 0.0625;  B1[v+2][v+1] = 0.0625;  B1[v+2][v+2] = 0.0625;  B1[v+2][v+3] = 0.0625;
	B1[v+3][0] = 0.125;  B1[v+3][1] = 0.375;  B1[v+3][2] = 0.375;  B1[v+3][v+3] = 0.125;
	B1[v+4][0] = 0.0625;  B1[v+4][1] = 0.0625;  B1[v+4][v-1] = 0.0625;  B1[v+4][v] = 0.625;  B1[v+4][v+1] = 0.0625;  B1[v+4][v+4] = 0.0625;  B1[v+4][v+5] = 0.0625;
	B1[v+5][0] = 0.125;  B1[v+5][v-1] = 0.375;  B1[v+5][v] = 0.375;  B1[v+5][v+5] = 0.125;
	B1[v+6][1] = 0.375;  B1[v+6][v] = 0.125;  B1[v+6][v+1] = 0.375;  B1[v+6][v+2] =  0.125;
	B1[v+7][1] = 0.375;  B1[v+7][v+1] =  0.125;  B1[v+7][v+2] =  0.375;  B1[v+7][v+3] = 0.125;
	B1[v+8][1] = 0.375;  B1[v+8][2] = 0.125;  B1[v+8][v+2] = 0.125;  B1[v+8][v+3] = 0.375;
	B1[v+9][1] = 0.125;  B1[v+9][v] = 0.375;  B1[v+9][v+1] = 0.375;  B1[v+9][v+4] = 0.125;
	B1[v+10][v] = 0.375;  B1[v+10][v+1] = 0.125;  B1[v+10][v+4] = 0.375;  B1[v+10][v+5] = 0.125;
	B1[v+11][v-1] = 0.125;  B1[v+11][v] = 0.375;  B1[v+11][v+4] = 0.125;  B1[v+11][v+5] = 0.375;
	
	for (i=0; i<12; i++) {
	  for (j=0; j<(v+6); j++) {
	    B3[i][j] = 0.;
	    for (m=0; m<(v+12); m++) {
	      B3[i][j] += B2[i][m]*B1[m][j];
	    }
	  }
	}

	PetscReal INa[2][v+6], INab[3][v+6];
	PetscReal sum1, sum2, sum3, sum4, sum5;

	for (j=0; j<(v+6); j++) {
	  sum1 = 0.0;  sum2 = 0.0;  sum3 = 0.0;  sum4 = 0.0;  sum5 = 0.0;
	  for (i=0; i<12; i++) {
	    sum1 += B3[i][j]*Na[0][i];
	    sum2 += B3[i][j]*Na[1][i];
	    sum3 += B3[i][j]*Nab[0][i];
	    sum4 += B3[i][j]*Nab[1][i];
	    sum5 += B3[i][j]*Nab[2][i];
	  }
	  INa[0][j] = -2*sum1;
	  INa[1][j] = -2*sum2;
	  INab[0][j] = 4*sum3;
	  INab[1][j] = 4*sum4;
	  INab[2][j] = 4*sum5;	  
	}
      
	Aaa.x = 0.;  Aaa.y = 0.;  Aaa.z = 0.;
	Abb.x = 0.;  Abb.y = 0.;  Abb.z = 0.;
	Aab.x = 0.;  Aab.y = 0.;  Aab.z = 0.;
	
	for (i=0; i<(v+6); i++) {
	  Aaa.x += INab[0][i]*X0[i][0];
	  Aaa.y += INab[0][i]*X0[i][1];
	  Aaa.z += INab[0][i]*X0[i][2];
	  
	  Abb.x += INab[1][i]*X0[i][0];
	  Abb.y += INab[1][i]*X0[i][1];
	  Abb.z += INab[1][i]*X0[i][2];
	  
	  Aab.x += INab[2][i]*X0[i][0];
	  Aab.y += INab[2][i]*X0[i][1];
	  Aab.z += INab[2][i]*X0[i][2];
	}
	
	ndx21.x = 0.;  ndx21.y = 0.;  ndx21.z = 0.;
	ndx31.x = 0.;  ndx31.y = 0.;  ndx31.z = 0.;
	
	for (i=0; i<(v+6); i++) {
	  ndx21.x += INa[0][i]*X0[i][0];
	  ndx21.y += INa[0][i]*X0[i][1];
	  ndx21.z += INa[0][i]*X0[i][2];
	
	  ndx31.x += INa[1][i]*X0[i][0];
	  ndx31.y += INa[1][i]*X0[i][1];
	  ndx31.z += INa[1][i]*X0[i][2];
	}
	
	nn=UNIT(CROSS(ndx21, ndx31));
	      	
	k0[0] = DOT(Aaa,nn);
	k0[1] = DOT(Abb,nn);
	k0[2] = DOT(Aab,nn);
	
	ibm->G1[dof*ec] = ndx21.x; ibm->G1[dof*ec+1] = ndx21.y; ibm->G1[dof*ec+2] = ndx21.z;
	ibm->G2[dof*ec] = ndx31.x; ibm->G2[dof*ec+1] = ndx31.y; ibm->G2[dof*ec+2] = ndx31.z;

	ibm->G[dof*ec] = DOT(ndx21, ndx21);
	ibm->G[dof*ec+1] = DOT(ndx31, ndx31);
	ibm->G[dof*ec+2] = DOT(ndx21, ndx31);

	ibm->kve0[ec*dof] = k0[0];
	ibm->kve0[ec*dof+1] = k0[1];
	ibm->kve0[ec*dof+2] = k0[2];

	CalcCurvStressStrainxyz(ec, k0, k0, k0, 1, 1, fem);
	
	for (i=0; i<(v+6); i++) {
	  free(X0[i]);
	}
	free(X0); 
	
      }     
    }
  }//elements  
  
  return(0);
}


//---------------------------------------------------------------------------
PetscErrorCode IrrVer(IBMNodes *ibm) {
  
  PetscInt  nc, ec, count, elmt[10], i, bcount, *irr;
  
  PetscMalloc(ibm->n_v*sizeof(PetscInt), &irr); //saves irregular node
  
  for (ec=0; ec<ibm->n_elmt; ec++) {
    ibm->ire[ec] = 0;
    ibm->irv[ec] = 0;
    ibm->val[ec] = 6; 
  }
  
  for (nc=0; nc<ibm->n_v; nc++) {
    irr[nc] = 0;
    count = 0;
    bcount = 0;
    
    for (ec=0; ec<ibm->n_elmt; ec++) {
      if (nc==ibm->nv1[ec] || nc==ibm->nv2[ec] || nc==ibm->nv3[ec]) {
	elmt[count] = ec;
	count++;
      }
    }
    
    for (i=0; i<ibm->sum_n_bnodes; i++) {
      if (nc==ibm->bnodes[i]) {
      	bcount = count + 3;
	break; // for corners
      }
    }
    
    if (bcount==0) {
      if (count!=6) {
	irr[nc] = 1;
    	for (i=0; i<count; i++) {
    	  ibm->ire[elmt[i]] = 1;
    	  ibm->val[elmt[i]] = count;
	}
      }
    } else {
      if (bcount!=6) {
	irr[nc] = 1;
    	for (i=0; i<count; i++) {
    	  ibm->ire[elmt[i]] = 1;
    	  ibm->val[elmt[i]] = bcount;
	}
      }
    }  
  }
  
  for (ec=0; ec<ibm->n_elmt; ec++) { //detects which element vertex is irregular
    for (nc=0; nc<ibm->n_v; nc++) {
      if (irr[nc]==1) { 
	if (nc==ibm->nv1[ec]) {ibm->irv[ec] = 1;}
	if (nc==ibm->nv2[ec]) {ibm->irv[ec] = 2;}
	if (nc==ibm->nv3[ec]) {ibm->irv[ec] = 3;}
      }
    }
  }
  
  PetscFree(irr);
  
  return(0);
}

//---------------------------------------------------------------------------
PetscErrorCode Patch(IBMNodes *ibm) {

  PetscInt  ec, *p, i, m, n1p, n2p, n3p, k;
  
  for (ec=0; ec<ibm->n_elmt; ec++) {

    PetscInt  v=ibm->val[ec];
    PetscMalloc((v+6)*sizeof(PetscInt), &p);

    for (i=0; i<(v+6); i++) {p[i] = 1000000;} //to form BC later

    if (ibm->ire[ec]==0) { //regular patch

      p[3] = ibm->nv1[ec];  p[6] = ibm->nv2[ec];  p[7] = ibm->nv3[ec];
      
      for (i=0; i<ibm->n_elmt+2*ibm->n_ghosts; i++) { //find element common neighbor nodes
	n1p = ibm->nv1[i];  n2p = ibm->nv2[i];  n3p = ibm->nv3[i];
	
	m = 0;
	if(p[3]==n1p || p[3]==n2p || p[3]==n3p) {m++;} 
	if(p[6]==n1p || p[6]==n2p || p[6]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[3] && n1p!=p[6] && i!=ec) {
	    p[2] = n1p;
	  } else if (n2p!=p[3] && n2p!=p[6] && i!=ec) {
	    p[2] = n2p;
	  } else if (n3p!=p[3] && n3p!=p[6] && i!=ec) {
	    p[2] = n3p;
	  }
	}
	
	m = 0;
	if(p[3]==n1p || p[3]==n2p || p[3]==n3p) {m++;} 
	if(p[7]==n1p || p[7]==n2p || p[7]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[3] && n1p!=p[7] && i!=ec) {
	    p[4] = n1p;
	  } else if (n2p!=p[3] && n2p!=p[7] && i!=ec) {
	    p[4] = n2p;
	  } else if (n3p!=p[3] && n3p!=p[7] && i!=ec) {
	    p[4] = n3p;
	  }
	}
	
	m = 0;
	if(p[6]==n1p || p[6]==n2p || p[6]==n3p) {m++;} 
	if(p[7]==n1p || p[7]==n2p || p[7]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[6] && n1p!=p[7] && i!=ec) {
	    p[10] = n1p;
	  } else if (n2p!=p[6] && n2p!=p[7] && i!=ec) {
	    p[10] = n2p;
	  } else if (n3p!=p[6] && n3p!=p[7] && i!=ec) {
	    p[10] = n3p;
	  }
	}
      } //common neighbors
      
      for (i=0; i<ibm->n_elmt+2*ibm->n_ghosts; i++) { //find other neighbors
	n1p = ibm->nv1[i];  n2p = ibm->nv2[i];  n3p = ibm->nv3[i];
	
	m = 0;
	if(p[3]==n1p || p[3]==n2p || p[3]==n3p) {m++;} 
	if(p[2]==n1p || p[2]==n2p || p[2]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[3] && n1p!=p[2] && n1p!=p[6]) {
	    p[0] = n1p;
	  } else if (n2p!=p[3] && n2p!=p[2] && n2p!=p[6]) {
	    p[0] = n2p;
	  } else if (n3p!=p[3] && n3p!=p[2] && n3p!=p[6]) {
	    p[0] = n3p;
	  }
	}
	
	m = 0;
	if(p[3]==n1p || p[3]==n2p || p[3]==n3p) {m++;} 
	if(p[4]==n1p || p[4]==n2p || p[4]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[3] && n1p!=p[4] && n1p!=p[7]) {
	    p[1] = n1p;
	  } else if (n2p!=p[3] && n2p!=p[4] && n2p!=p[7]) {
	    p[1] = n2p;
	  } else if (n3p!=p[3] && n3p!=p[4] && n3p!=p[7]) {
	    p[1] = n3p;
	  }
	}    
	
	m = 0;
	if(p[6]==n1p || p[6]==n2p || p[6]==n3p) {m++;} 
	if(p[2]==n1p || p[2]==n2p || p[2]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[6] && n1p!=p[2] && n1p!=p[3]) {
	    p[5] = n1p;
	  } else if (n2p!=p[6] && n2p!=p[2] && n2p!=p[3]) {
	    p[5] = n2p;
	  } else if (n3p!=p[6] && n3p!=p[2] && n3p!=p[3]) {
	    p[5] = n3p;
	  }
	}
	
	m = 0;
	if(p[6]==n1p || p[6]==n2p || p[6]==n3p) {m++;} 
	if(p[10]==n1p || p[10]==n2p || p[10]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[6] && n1p!=p[10] && n1p!=p[7]) {
	    p[9] = n1p;
	  } else if (n2p!=p[6] && n2p!=p[10] && n2p!=p[7]) {
	    p[9] = n2p;
	  } else if (n3p!=p[6] && n3p!=p[10] && n3p!=p[7]) {
	    p[9] = n3p;
	  }
	}
	
	m = 0;
	if(p[7]==n1p || p[7]==n2p || p[7]==n3p) {m++;} 
	if(p[10]==n1p || p[10]==n2p || p[10]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[7] && n1p!=p[10] && n1p!=p[6]) {
	    p[11] = n1p;
	  } else if (n2p!=p[7] && n2p!=p[10] && n2p!=p[6]) {
	    p[11] = n2p;
	  } else if (n3p!=p[7] && n3p!=p[10] && n3p!=p[6]) {
	    p[11] = n3p;
	  }
	}
	
	m = 0;
	if(p[7]==n1p || p[7]==n2p || p[7]==n3p) {m++;} 
	if(p[4]==n1p || p[4]==n2p || p[4]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[7] && n1p!=p[4] && n1p!=p[3]) {
	    p[8] = n1p;
	  } else if (n2p!=p[7] && n2p!=p[4] && n2p!=p[3]) {
	    p[8] = n2p;
	  } else if (n3p!=p[7] && n3p!=p[4] && n3p!=p[3]) {
	    p[8] = n3p;
	  }
	}
      } //for other neighbors      

    } else if (ibm->ire[ec]==1) { //for irregular elements

      if (ibm->irv[ec]==1) {
	p[0] = ibm->nv1[ec];  p[1] = ibm->nv2[ec];  p[v] = ibm->nv3[ec];
      } else if (ibm->irv[ec]==2) {
	p[0] = ibm->nv2[ec];  p[1] = ibm->nv3[ec];  p[v] = ibm->nv1[ec];
      } else if (ibm->irv[ec]==3) {
	p[0] = ibm->nv3[ec];  p[1] = ibm->nv1[ec];  p[v] = ibm->nv2[ec];
      }

      for (i=0; i<ibm->n_elmt+2*ibm->n_ghosts; i++) {   //find element common neighbor nodes
	n1p = ibm->nv1[i];  n2p = ibm->nv2[i];  n3p = ibm->nv3[i];

	m = 0;
	if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	if(p[1]==n1p || p[1]==n2p || p[1]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[0] && n1p!=p[1] && i!=ec) {
	    p[2] = n1p;
	  } else if (n2p!=p[0] && n2p!=p[1] && i!=ec) {
	    p[2] = n2p;
	  } else if (n3p!=p[0] && n3p!=p[1] && i!=ec) {
	    p[2] = n3p;
	  }
	}

	m = 0;
	if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	if(p[v]==n1p || p[v]==n2p || p[v]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[0] && n1p!=p[v] && i!=ec) {
	    p[v-1] = n1p;
	  } else if (n2p!=p[0] && n2p!=p[v] && i!=ec) {
	    p[v-1] = n2p;
	  } else if (n3p!=p[0] && n3p!=p[v] && i!=ec) {
	    p[v-1] = n3p;
	  }
	}

	m = 0;
	if(p[1]==n1p || p[1]==n2p || p[1]==n3p) {m++;} 
	if(p[v]==n1p || p[v]==n2p || p[v]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[1] && n1p!=p[v] && i!=ec) {
	    p[v+1] = n1p;
	  } else if (n2p!=p[1] && n2p!=p[v] && i!=ec) {
	    p[v+1] = n2p;
	  } else if (n3p!=p[1] && n3p!=p[v] && i!=ec) {
	    p[v+1] = n3p;
	  }
	}
      } //for common neighbors

      for (i=0; i<ibm->n_elmt+2*ibm->n_ghosts; i++) {   //find element other neighbor nodes
	n1p = ibm->nv1[i];  n2p = ibm->nv2[i];  n3p = ibm->nv3[i];
	
	m = 0;
	if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	if(p[2]==n1p || p[2]==n2p || p[2]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[0] && n1p!=p[2] && n1p!=p[1]) {
	    p[3] = n1p;
	  } else if (n2p!=p[0] && n2p!=p[2] && n2p!=p[1]) {
	    p[3] = n2p;
	  } else if (n3p!=p[0] && n3p!=p[2] && n3p!=p[1]) {
	    p[3] = n3p;
	  }
	}

	m = 0;
	if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	if(p[v-1]==n1p || p[v-1]==n2p || p[v-1]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[0] && n1p!=p[v-1] && n1p!=p[v]) {
	    p[v-2] = n1p;
	  } else if (n2p!=p[0] && n2p!=p[v-1] && n2p!=p[v]) {
	    p[v-2] = n2p;
	  } else if (n3p!=p[0] && n3p!=p[v-1] && n3p!=p[v]) {
	    p[v-2] = n3p;
	  }
	}

	m = 0;
	if(p[1]==n1p || p[1]==n2p || p[1]==n3p) {m++;} 
	if(p[2]==n1p || p[2]==n2p || p[2]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[1] && n1p!=p[2] && n1p!=p[0]) {
	    p[v+3] = n1p;
	  } else if (n2p!=p[1] && n2p!=p[2] && n2p!=p[0]) {
	    p[v+3] = n2p;
	  } else if (n3p!=p[1] && n3p!=p[2] && n3p!=p[0]) {
	    p[v+3] = n3p;
	  }
	}

	m = 0;
	if(p[1]==n1p || p[1]==n2p || p[1]==n3p) {m++;} 
	if(p[v+1]==n1p || p[v+1]==n2p || p[v+1]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[1] && n1p!=p[v+1] && n1p!=p[v]) {
	    p[v+2] = n1p;
	  } else if (n2p!=p[1] && n2p!=p[v+1] && n2p!=p[v]) {
	    p[v+2] = n2p;
	  } else if (n3p!=p[1] && n3p!=p[v+1] && n3p!=p[v]) {
	    p[v+2] = n3p;
	  }
	}

	m = 0;
	if(p[v]==n1p || p[v]==n2p || p[v]==n3p) {m++;} 
	if(p[v+1]==n1p || p[v+1]==n2p || p[v+1]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[v] && n1p!=p[v+1] && n1p!=p[1]) {
	    p[v+4] = n1p;
	  } else if (n2p!=p[v] && n2p!=p[v+1] && n2p!=p[1]) {
	    p[v+4] = n2p;
	  } else if (n3p!=p[v] && n3p!=p[v+1] && n3p!=p[1]) {
	    p[v+4] = n3p;
	  }
	}

	m = 0;
	if(p[v]==n1p || p[v]==n2p || p[v]==n3p) {m++;} 
	if(p[v-1]==n1p || p[v-1]==n2p || p[v-1]==n3p) {m++;} 
	
	if (m==2) {
	  if (n1p!=p[v] && n1p!=p[v-1] && n1p!=p[0]) {
	    p[v+5] = n1p;
	  } else if (n2p!=p[v] && n2p!=p[v-1] && n2p!=p[0]) {
	    p[v+5] = n2p;
	  } else if (n3p!=p[v] && n3p!=p[v-1] && n3p!=p[0]) {
	    p[v+5] = n3p;
	  }
	}

      }

      if (ibm->val[ec]>6) { //for nodes with extra valence
	for (k=0; k<(v-6); k++) {
	  for (i=0; i<ibm->n_elmt+2*ibm->n_ghosts; i++) { 
	    n1p = ibm->nv1[i];  n2p = ibm->nv2[i];  n3p = ibm->nv3[i];

	    m = 0;
	    if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	    if(p[3+k]==n1p || p[3+k]==n2p || p[3+k]==n3p) {m++;} 
	    
	    if (m==2) {
	      if (n1p!=p[0] && n1p!=p[3+k] && n1p!=p[2+k]) {
		p[k+4] = n1p;
	      } else if (n2p!=p[0] && n2p!=p[3+k] && n2p!=p[2+k]) {
		p[k+4] = n2p;
	      } else if (n3p!=p[0] && n3p!=p[3+k] && n3p!=p[2+k]) {
		p[k+4] = n3p;
	      }
	    }


	    m = 0;
	    if(p[0]==n1p || p[0]==n2p || p[0]==n3p) {m++;} 
	    if(p[v-2-k]==n1p || p[v-2-k]==n2p || p[v-2-k]==n3p) {m++;} 

	    if (m==2) {
	      if (n1p!=p[0] && n1p!=p[v-2-k] && n1p!=p[v-1-k]) {
		p[v-3-k] = n1p;
	      } else if (n2p!=p[0] && n2p!=p[v-2-k] && n2p!=p[v-1-k]) {
		p[v-3-k] = n2p;
	      } else if (n3p!=p[0] && n3p!=p[v-2-k] && n3p!=p[v-1-k]) {
		p[v-3-k] = n3p;
	      }
	    }
	    
	  }
	}
	
      }

    }
    
    for (i=0; i<(v+6); i++) {ibm->patch[16*ec+i] = p[i];}
    PetscFree(p);
  } //elements
  
  return(0);
}

//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode GlobalGhostInit(IBMNodes *ibm) {
  
  PetscInt  i, ec, nv1, nv2, nv3, catch, edge_n, edge, start, end, count=0;
  
  for (edge_n=0; edge_n<ibm->n_edge; edge_n++){
    //compute start&end
    start = 0;  end = 0;
    for (edge=0; edge<edge_n+1; edge++) {
      end += ibm->n_bnodes[edge];
    }
    start = end - ibm->n_bnodes[edge_n];

    for (i=start; i<end-1; i++) {
      
      nv1 = ibm->bnodes[i+1];
      nv2 = ibm->bnodes[i];
      
      for (ec=0; ec<ibm->n_elmt; ec++) {
      	catch = 0;
      	if (nv1==ibm->nv1[ec] || nv1==ibm->nv2[ec] || nv1==ibm->nv3[ec]) {catch++;}
      	if (nv2==ibm->nv1[ec] || nv2==ibm->nv2[ec] || nv2==ibm->nv3[ec]) {catch++;}
      	if (catch==2) {
      	  if (ibm->nv1[ec]!=nv1 && ibm->nv1[ec]!=nv2) {
      	    nv3 = ibm->nv1[ec];
      	  } else if (ibm->nv2[ec]!=nv1 && ibm->nv2[ec]!=nv2) {
      	    nv3 = ibm->nv2[ec];
      	  } else{
      	    nv3 = ibm->nv3[ec];
      	  }
      	}	
      }

      ibm->x_bp[ibm->n_v+count] = ibm->x_bp[nv2] + ibm->x_bp[nv1] - ibm->x_bp[nv3];
      ibm->y_bp[ibm->n_v+count] = ibm->y_bp[nv2] + ibm->y_bp[nv1] - ibm->y_bp[nv3];
      ibm->z_bp[ibm->n_v+count] = ibm->z_bp[nv2] + ibm->z_bp[nv1] - ibm->z_bp[nv3];

      ibm->x_bp0[ibm->n_v+count] = ibm->x_bp[nv2] + ibm->x_bp[nv1] - ibm->x_bp[nv3];
      ibm->y_bp0[ibm->n_v+count] = ibm->y_bp[nv2] + ibm->y_bp[nv1] - ibm->y_bp[nv3];
      ibm->z_bp0[ibm->n_v+count] = ibm->z_bp[nv2] + ibm->z_bp[nv1] - ibm->z_bp[nv3];
      
      if (i!=0) { //plate (1 continuous edge)
	//if (i!=start) { //cylinder (two edges)
	ibm->nv2[ibm->n_elmt+2*count-1] = nv2;
      	ibm->nv1[ibm->n_elmt+2*count-1] = ibm->n_v + count;
      	ibm->nv3[ibm->n_elmt+2*count-1] = ibm->n_v + count - 1;
      }

      ibm->nv2[ibm->n_elmt+2*count] = nv2;
      ibm->nv1[ibm->n_elmt+2*count] = nv1;
      ibm->nv3[ibm->n_elmt+2*count] = ibm->n_v + count;
      if (i==ibm->sum_n_bnodes-2) { //plate
	//if (i==end-2) { //cylinder
	//ibm->nv1[ibm->n_elmt+2*count+1] = ibm->n_v + start - edge_n; //cylinder
	ibm->nv2[ibm->n_elmt+2*count+1] = nv1;
	ibm->nv1[ibm->n_elmt+2*count+1] = ibm->n_v; //plate
	ibm->nv3[ibm->n_elmt+2*count+1] = ibm->n_v + count;
      }
      
      count++;     
    }
  }

  return(0);
}


//------------------------------------------------------------------------------------------------------------ 
PetscErrorCode GlobalGhost(IBMNodes *ibm) {

  PetscInt i, ec, nv1, nv2, nv3, catch, edge_n, edge, start, end, count=0;

  for (edge_n=0; edge_n<ibm->n_edge; edge_n++){
    //compute start&end
    start = 0; end = 0;
    for (edge=0; edge<edge_n+1; edge++) {
      end += ibm->n_bnodes[edge];
    }
    start = end - ibm->n_bnodes[edge_n];

    for (i=start; i<end-1; i++) {
      
      nv1 = ibm->bnodes[i+1];
      nv2 = ibm->bnodes[i];
      
      for (ec=0; ec<ibm->n_elmt; ec++) {
      	catch = 0;
      	if (nv1==ibm->nv1[ec] || nv1==ibm->nv2[ec] || nv1==ibm->nv3[ec]) {catch++;}
      	if (nv2==ibm->nv1[ec] || nv2==ibm->nv2[ec] || nv2==ibm->nv3[ec]) {catch++;}
      	if (catch==2) {
      	  if (ibm->nv1[ec]!=nv1 && ibm->nv1[ec]!=nv2) {
      	    nv3 = ibm->nv1[ec];
      	  } else if (ibm->nv2[ec]!=nv1 && ibm->nv2[ec]!=nv2) {
      	    nv3 = ibm->nv2[ec];
      	  } else{
      	    nv3 = ibm->nv3[ec];
      	  }
      	}	
      }

      ibm->x_bp[ibm->n_v+count] = ibm->x_bp[nv2] + ibm->x_bp[nv1] - ibm->x_bp[nv3];
      ibm->y_bp[ibm->n_v+count] = ibm->y_bp[nv2] + ibm->y_bp[nv1] - ibm->y_bp[nv3];
      ibm->z_bp[ibm->n_v+count] = ibm->z_bp[nv2] + ibm->z_bp[nv1] - ibm->z_bp[nv3];     
      
      count++;      
    }
  }

  return(0);
}
