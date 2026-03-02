#include  "variables.h"

extern PetscInt   dof, nbody, contact, rho, h0;
extern PetscReal  dt;
extern struct Cmpnts  MINUS(struct Cmpnts v1, struct Cmpnts v2);
extern struct Cmpnts  CROSS(struct Cmpnts v1, struct Cmpnts v2);
extern PetscReal  SIZE(struct Cmpnts v1);
extern struct Cmpnts  UNIT(struct Cmpnts v1);


void initlist(List *ilist) {
  ilist->head = PETSC_NULL;
}

//-----------------------------------------------------------------------------------------------------------
void insertnode(List *ilist, PetscInt Node) {

  node  *new;
  node  *current;
  current = ilist->head;
  PetscBool  Exist=PETSC_FALSE;

  while(current) {
    if (Node==current->Node) {
      Exist = PETSC_TRUE;
    }
    if (Exist) break;
    current = current->next;
  }
  if (!Exist) {
    PetscMalloc(sizeof(node), &new);
    new->next = ilist->head;
    new->Node = Node;
    ilist->head = new;
  }
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode Fcontact(FE *fem) {

  /* PetscReal  *xda1, *xdb1, *xda2, *xdb2, *xxda, *xxdb; */
  /* IBMNodes   *ibm=fem->ibm; */
  /* PetscInt   nv; */

  /* PetscMalloc(dof*(ibm[0].n_v)*sizeof(PetscReal), &xda1); */
  /* PetscMalloc(dof*(ibm[0].n_v)*sizeof(PetscReal), &xda2); */
  /* PetscMalloc(dof*(ibm[1].n_v)*sizeof(PetscReal), &xdb1); */
  /* PetscMalloc(dof*(ibm[1].n_v)*sizeof(PetscReal), &xdb2); */

  /* for (nv=0; nv <ibm[0].n_v; nv++) { */
  /*   xda1[dof*nv] = 0.; xda1[dof*nv+1] = 0.; xda1[dof*nv+2] = 0.; */
  /*   xda2[dof*nv] = 0.; xda2[dof*nv+1] = 0.; xda2[dof*nv+2] = 0.; */
  /* }  */
  /* for (nv=0; nv <ibm[1].n_v; nv++) { */
  /*   xdb1[dof*nv] = 0.; xdb1[dof*nv+1] = 0.; xdb1[dof*nv+2] = 0.; */
  /*   xdb2[dof*nv] = 0.; xdb2[dof*nv+1] = 0.; xdb2[dof*nv+2] = 0.; */
  /* }   */

  //sphere-sphere
  // Fcontactij(&fem[1], &fem[0], 1, 0);
  //  Fcontactij(&fem[0], &fem[1], 0, 1);

  //Cloth
  //Fcontactij(&fem[0], &fem[0], 0 , 0);

  //BHV
  Fcontactij(&fem[0], &fem[1], 0, 1);
  Fcontactij(&fem[1], &fem[2], 1, 2);
  Fcontactij(&fem[2], &fem[0], 2, 0);

  Fcontactij(&fem[1], &fem[0], 1, 0);
  Fcontactij(&fem[2], &fem[1], 2, 1);
  Fcontactij(&fem[0], &fem[2], 0, 2);

  /* VecGetArray(fem[0].xd, &xxda); */
  /* VecGetArray(fem[1].xd, &xxdb); */

  /* for (nv=0; nv <ibm[0].n_v; nv++) { */
  /*   if (ibm[0].contact[nv]) */
  /*     xxda[dof*nv] = xda1[dof*nv]; xxda[dof*nv+1] = xda1[dof*nv+1]; xxda[dof*nv+2] = xda1[dof*nv+2]; */
  /* } */

  /* for (nv=0; nv <ibm[1].n_v; nv++) { */
  /*   if (ibm[1].contact[nv]) */
  /*     xxdb[dof*nv] = xdb1[dof*nv]; xxdb[dof*nv+1] = xdb1[dof*nv+1]; xxdb[dof*nv+2] = xdb1[dof*nv+2]; */
  /* } */

  /* VecRestoreArray(fem[0].xd, &xxda); */
  /* VecRestoreArray(fem[1].xd, &xxdb); */
  /* PetscFree(xda1); PetscFree(xda2); PetscFree(xdb1); PetscFree(xdb2); */

  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode Fcontactij(FE *fem1, FE *fem2, PetscInt body1, PetscInt body2) {
  
  IBMNodes       *ibm1=fem1->ibm, *ibm2=fem2->ibm;
  PetscInt	 i, j, k, nv, ncx=15, ncy=15, ncz=15;
  PetscReal	 xbp_min, ybp_min, zbp_min, xbp_max, ybp_max, zbp_max, dcx, dcy, dcz;
  PetscReal	 *x_bp = ibm2->x_bp, *y_bp = ibm2->y_bp, *z_bp = ibm2->z_bp;
  PetscInt 	 n_v = ibm2->n_v, ln_v;
  struct Cmpnts  pnv;
  List           *cell_trg;
  PetscReal	 xv_min, yv_min, zv_min, xv_max, yv_max, zv_max;
  PetscInt	 iv_min, iv_max, jv_min, jv_max, kv_min, kv_max, n1e, n2e, n3e, ic, jc, kc;

  // Bounding Box for ibm2
  xbp_min = 1.e23;  xbp_max = -1.e23;
  ybp_min = 1.e23;  ybp_max = -1.e23;
  zbp_min = 1.e23;  zbp_max = -1.e23;
  
  for(i=0; i<n_v; i++) {
    xbp_min = PetscMin(xbp_min, x_bp[i]);
    xbp_max = PetscMax(xbp_max, x_bp[i]);
    
    ybp_min = PetscMin(ybp_min, y_bp[i]);
    ybp_max = PetscMax(ybp_max, y_bp[i]);
    
    zbp_min = PetscMin(zbp_min, z_bp[i]);
    zbp_max = PetscMax(zbp_max, z_bp[i]);
  }
 
  xbp_min -= 0.001;  xbp_max += 0.001;
  ybp_min -= 0.001;  ybp_max += 0.001;
  zbp_min -= 0.001;  zbp_max += 0.001;
 
  //Control cell for ibm2
  dcx = (xbp_max - xbp_min)/(ncx - 1.);
  dcy = (ybp_max - ybp_min)/(ncy - 1.);
  dcz = (zbp_max - zbp_min)/(ncz - 1.);

  PetscMalloc(ncz*ncy*ncx*sizeof(List), &cell_trg);
 
  for (k=0; k<ncz; k++) {
    for (j=0; j<ncy; j++) {
      for (i=0; i<ncx; i++) {
  	initlist(&cell_trg[k*ncx*ncy + j*ncx + i]);
      }
    }
  }
  
  for (ln_v=0; ln_v<ibm2->n_elmt; ln_v++) {

    n1e = ibm2->nv1[ln_v];  n2e = ibm2->nv2[ln_v];  n3e = ibm2->nv3[ln_v];

    xv_min = PetscMin(PetscMin(x_bp[n1e], x_bp[n2e]), x_bp[n3e]);
    xv_max = PetscMax(PetscMax(x_bp[n1e], x_bp[n2e]), x_bp[n3e]);

    yv_min = PetscMin(PetscMin(y_bp[n1e], y_bp[n2e]), y_bp[n3e]);
    yv_max = PetscMax(PetscMax(y_bp[n1e], y_bp[n2e]), y_bp[n3e]);

    zv_min = PetscMin(PetscMin(z_bp[n1e], z_bp[n2e]), z_bp[n3e]);
    zv_max = PetscMax(PetscMax(z_bp[n1e], z_bp[n2e]), z_bp[n3e]);
    
    iv_min = floor((xv_min - xbp_min)/dcx);
    iv_max = floor((xv_max - xbp_min)/dcx) + 1;

    jv_min = floor((yv_min - ybp_min)/dcy);
    jv_max = floor((yv_max - ybp_min)/dcy) + 1;

    kv_min = floor((zv_min - zbp_min)/dcz);
    kv_max = floor((zv_max - zbp_min)/dcz) + 1;

    iv_min = (iv_min<0) ? 0:iv_min;
    iv_max = (iv_max>ncx) ? ncx:iv_max;

    jv_min = (jv_min<0) ? 0:jv_min;
    jv_max = (jv_max>ncx) ? ncy:jv_max;

    kv_min = (kv_min<0) ? 0:kv_min;
    kv_max = (kv_max>ncz) ? ncz:kv_max;
   
    // Insert IBM node information into a list
    for (k=kv_min; k<kv_max; k++) {
      for (j=jv_min; j<jv_max; j++) {
  	for (i=iv_min; i<iv_max; i++) {
  	  insertnode(&(cell_trg[k *ncx*ncy + j*ncx +i]), ln_v);
  	}
      }
    }
  }
  
  // search if ibm1 intersects ibm2 and find contact forces for fem1
  PetscReal  *xx1, *xx2, *xxd1, *xxd2, dist, dir[3], *xxda, *xxdb; //BHV 0.0002 0.1 sphere
  PetscReal  *xxdd1, *xxdd2, *xxn1, *xxn2, xxdn2[dof*ibm2->n_v]; // Update acceleration Iman 10/13/22
  PetscInt   elmt2;
  PetscReal  uan, Ub[3], ubn, uann, ubnn, Ubn[3], e=0.0, ma=1., mb=1., ub1n, ub2n, ub3n, c=0.5; //1.5 cloth 0.5 others
  PetscReal      M1, M2, M3;
  PetscReal      Gama=0.5, Beta=0.25;
  M1=1./(Beta*pow(dt,2));  M2=1./(Beta*dt);  M3=(1./(2*Beta))-1.;
  PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-e_coefficient_restitution", &e, PETSC_NULL);
  //  PetscPrintf(PETSC_COMM_WORLD, "e coefficient of restituion=%le\n", e);


  VecGetArray(fem1->x, &xx1); 
  VecGetArray(fem2->x, &xx2);
  VecGetArray(fem1->xd, &xxd1);
  VecGetArray(fem2->xd, &xxd2);
  VecGetArray(fem1->xdd, &xxdd1);
  VecGetArray(fem2->xdd, &xxdd2);
  VecGetArray(fem1->xn, &xxn1); 
  VecGetArray(fem2->xn, &xxn2);
 
  for (nv=0; nv < (dof*ibm2->n_v); nv++) {
    xxdn2[nv]=xxd2[nv];
  }

  for (nv=0; nv <ibm1->n_v; nv++) {
    pnv.x = ibm1->x_bp[nv];
    pnv.y = ibm1->y_bp[nv];
    pnv.z = ibm1->z_bp[nv];

    if (pnv.x > xbp_min && pnv.x < xbp_max &&
    	pnv.y > ybp_min && pnv.y < ybp_max &&
    	pnv.z > zbp_min && pnv.z < zbp_max) { // if in bounding box
      
      ic = floor((pnv.x - xbp_min)/dcx);
      jc = floor((pnv.y - ybp_min)/dcy);
      kc = floor((pnv.z - zbp_min)/dcz);

      // find the closest triangle in control cells
      nearestcellFEM(pnv, ibm2, &dist, &elmt2, dir, ic, jc, kc, ncx, ncy, ncz, cell_trg, body1, body2, nv);
      
      if (elmt2>-1) {      
        
	if (dist<0) {

	  n1e = ibm2->nv1[elmt2];  n2e = ibm2->nv2[elmt2];  n3e = ibm2->nv3[elmt2]; 

	  PetscPrintf(PETSC_COMM_WORLD, "nv %d elmt %d nv123 %d %d %d contact %d %d %d %d acc_a %f dist %f\n",nv, elmt2, n1e, n2e, n3e, ibm1->contact[nv],ibm2->contact[n1e],ibm2->contact[n2e],ibm2->contact[n3e],xxdd1[dof*nv], dist);

	  if ((ibm1->contact[nv]==0) && (ibm2->contact[n1e]==0 || ibm2->contact[n2e]==0 || ibm2->contact[n3e]==0)) {

	    ibm1->contact[nv] += 1;
	    
	    xx1[dof*nv  ] += c*dist*dir[0];
	    xx1[dof*nv+1] += c*dist*dir[1];
	    xx1[dof*nv+2] += c*dist*dir[2];
	    ibm1->x_bp[nv] = xx1[dof*nv];
	    ibm1->y_bp[nv] = xx1[dof*nv+1];
	    ibm1->z_bp[nv] = xx1[dof*nv+2];
	     
	    //	    ibm2->contact[n1e] += 1;  ibm2->contact[n2e] += 1;  ibm2->contact[n3e] += 1;
	 
	    ma = ibm1->m[nv];
	    // mb = ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e];//(ibm2->rho)*(ibm2->dA0[elmt2])*(ibm2->h0);//   
	    mb=0.; Ub[0]=0.;Ub[1]=0.;Ub[2]=0.;
	    if (!ibm2->contact[n1e]) { 
	      xx2[dof*n1e  ] -= c*dist*dir[0];
	      xx2[dof*n1e+1] -= c*dist*dir[1];
	      xx2[dof*n1e+2] -= c*dist*dir[2];
	      mb += ibm2->m[n1e];
	      Ub[0] += xxd2[dof*n1e  ]*ibm2->m[n1e];
	      Ub[1] += xxd2[dof*n1e+1]*ibm2->m[n1e];
	      Ub[2] += xxd2[dof*n1e+2]*ibm2->m[n1e]; 
	    }
	    if (!ibm2->contact[n2e]) {
	      xx2[dof*n2e  ] -= c*dist*dir[0];  
	      xx2[dof*n2e+1] -= c*dist*dir[1];  
	      xx2[dof*n2e+2] -= c*dist*dir[2];
	      mb += ibm2->m[n2e];  
	      Ub[0] +=  xxd2[dof*n2e  ]*ibm2->m[n2e];
	      Ub[1] +=  xxd2[dof*n2e+1]*ibm2->m[n2e];
	      Ub[2] +=  xxd2[dof*n2e+2]*ibm2->m[n2e]; 
	    }
	    if (!ibm2->contact[n3e]) {
	      xx2[dof*n3e  ] -= c*dist*dir[0];
	      xx2[dof*n3e+1] -= c*dist*dir[1];
	      xx2[dof*n3e+2] -= c*dist*dir[2];
	      mb += ibm2->m[n3e];
	      Ub[0] += xxd2[dof*n3e  ]*ibm2->m[n3e];
	      Ub[1] += xxd2[dof*n3e+1]*ibm2->m[n3e];
	      Ub[2] += xxd2[dof*n3e+2]*ibm2->m[n3e];
	    }
	    ibm2->x_bp[n1e] = xx2[dof*n1e  ];  ibm2->x_bp[n2e] = xx2[dof*n2e  ];  ibm2->x_bp[n3e] = xx2[dof*n3e  ];
	    ibm2->y_bp[n1e] = xx2[dof*n1e+1];  ibm2->y_bp[n2e] = xx2[dof*n2e+1];  ibm2->y_bp[n3e] = xx2[dof*n3e+1];
	    ibm2->z_bp[n1e] = xx2[dof*n1e+2];  ibm2->z_bp[n2e] = xx2[dof*n2e+2];  ibm2->z_bp[n3e] = xx2[dof*n3e+2];
	    
	 
	    uan = xxd1[dof*nv]*dir[0] + xxd1[dof*nv+1]*dir[1] + xxd1[dof*nv+2]*dir[2];
	    
	    // Ub = (m1 u1 + m2 u2 + m3 u3)/(m1+m2+m3)
	    Ub[0] = Ub[0]/mb;//(xxd2[dof*n1e  ]*ibm2->m[n1e] + xxd2[dof*n2e  ]*ibm2->m[n2e] + xxd2[dof*n3e  ]*ibm2->m[n3e])/mb;//(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]);
	    Ub[1] = Ub[1]/mb;//(xxd2[dof*n1e+1]*ibm2->m[n1e] + xxd2[dof*n2e+1]*ibm2->m[n2e] + xxd2[dof*n3e+1]*ibm2->m[n3e])/mb;//(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]);
	    Ub[2] = Ub[2]/mb;//(xxd2[dof*n1e+2]*ibm2->m[n1e] + xxd2[dof*n2e+2]*ibm2->m[n2e] + xxd2[dof*n3e+2]*ibm2->m[n3e])/mb;//(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]);
	    

	    /* Ub[0] = (xxdn2[dof*n1e  ]*ibm2->m[n1e] + xxdn2[dof*n2e  ]*ibm2->m[n2e] + xxdn2[dof*n3e  ]*ibm2->m[n3e])/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]); */
	    /* Ub[1] = (xxdn2[dof*n1e+1]*ibm2->m[n1e] + xxdn2[dof*n2e+1]*ibm2->m[n2e] + xxdn2[dof*n3e+1]*ibm2->m[n3e])/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]); */
	    /* Ub[2] = (xxdn2[dof*n1e+2]*ibm2->m[n1e] + xxdn2[dof*n2e+2]*ibm2->m[n2e] + xxdn2[dof*n3e+2]*ibm2->m[n3e])/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]); */
	    
	    ubn = Ub[0]*dir[0] + Ub[1]*dir[1] + Ub[2]*dir[2];
	    ub1n = xxd2[n1e*dof]*dir[0] + xxd2[n1e*dof+1]*dir[1] + xxd2[n1e*dof+2]*dir[2];
	    ub2n = xxd2[n2e*dof]*dir[0] + xxd2[n2e*dof+1]*dir[1] + xxd2[n2e*dof+2]*dir[2];
	    ub3n = xxd2[n3e*dof]*dir[0] + xxd2[n3e*dof+1]*dir[1] + xxd2[n3e*dof+2]*dir[2];
	    
	    if ((uan-ubn)>0) { //reverse velocities only if relative velocity pushes the object into each other
	    /* ma = ibm1->m[nv]; */
	    /* mb = ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e]; */

	    uann = 1./(ma+mb) * ((ma-e*mb)*uan +  mb*(1+e)*ubn);
	    ubnn = 1./(ma+mb) * ( ma*(1+e)*uan + (mb-e*ma)*ubn);
	      
	      /* uann = -uan; */
	      /* ubnn = -ubn; */

	   
	    // velocity of point A
	    xxd1[dof*nv  ] += (uann - uan)*dir[0];
	    xxd1[dof*nv+1] += (uann - uan)*dir[1];
	    xxd1[dof*nv+2] += (uann - uan)*dir[2];

	    // move based on velocity of A x=xn+c*ua*dt
	    /* xx1[dof*nv  ] += uann*dt*dir[0]; */
	    /* xx1[dof*nv+1] += uann*dt*dir[1]; */
	    /* xx1[dof*nv+2] += uann*dt*dir[2]; */
	    /* ibm1->x_bp[nv] = xx1[dof*nv]; */
	    /* ibm1->y_bp[nv] = xx1[dof*nv+1]; */
	    /* ibm1->z_bp[nv] = xx1[dof*nv+2]; */

	    // Update acceleration--- test, Iman 10/13/22
	    /* xxdd1[dof*nv  ] -= (M1*c*dist)*dir[0]/M3;//(M1*c*dist-M2*(uann - uan))*dir[0]/M3; */
	    /* xxdd1[dof*nv+1] -= (M1*c*dist)*dir[1]/M3;//(M1*c*dist-M2*(uann - uan))*dir[1]/M3; //(uann - uan)/dt*dir[1]; */
	    /* xxdd1[dof*nv+2] -= (M1*c*dist)*dir[2]/M3;//(M1*c*dist-M2*(uann - uan))*dir[2]/M3; //(uann - uan)/dt*dir[2]; */
	    /* xxdd1[dof*nv  ] = 0.;//- M2*xxd1[nv*dof  ] + M1*(xx1[nv*dof  ] - xxn1[nv*dof  ]) ;//- M3*xxdd1[nv*dof  ]; */
	    /* xxdd1[dof*nv+1] = 0.;//- M2*xxd1[nv*dof+1] + M1*(xx1[nv*dof+1] - xxn1[nv*dof+1]) ;//- M3*xxdd1[nv*dof+1]; */
	    /* xxdd1[dof*nv+2] = 0.;//- M2*xxd1[nv*dof+2] + M1*(xx1[nv*dof+2] - xxn1[nv*dof+2]) ;//- M3*xxdd1[nv*dof+2]; */
	    
	    PetscPrintf(PETSC_COMM_WORLD, "nv %d elmt %d nv123 %d %d %d contact %d %d %d %d ua %f %f ub %f %f dMVa %f dMVb %f dKEa %f dKEb %f Acc_a %f\n",nv, elmt2, n1e, n2e, n3e, ibm1->contact[nv],ibm2->contact[n1e],ibm2->contact[n2e],ibm2->contact[n3e], uan, uann, ubn, ubnn, ma*(uann-uan), mb*(ubnn-ubn), 0.5*ma*(uann*uann-uan*uan), 0.5*mb*(ubnn*ubnn-ubn*ubn), xxdd1[dof*nv]);

	    // velocity of point B - 3 nodes of the element
	    if (!ibm2->contact[n1e])  {
	      xxd2[dof*n1e  ] += (ubnn - ub1n)*dir[0];
	      xxd2[dof*n1e+1] += (ubnn - ub1n)*dir[1];
	      xxd2[dof*n1e+2] += (ubnn - ub1n)*dir[2];

	    /*   xx2[dof*n1e  ] += ubnn*dt*dir[0]; */
	    /*   xx2[dof*n1e+1] += ubnn*dt*dir[1]; */
	    /*   xx2[dof*n1e+2] += ubnn*dt*dir[2]; */

	    /*   xxdd2[dof*n1e  ] = 0.;//- M2*xxd2[n1e*dof  ] + M1*(xx2[n1e*dof  ] - xxn2[n1e*dof  ]) - M3*xxdd2[n1e*dof  ]; */
	    /* xxdd2[dof*n1e+1] = 0.;//- M2*xxd2[n1e*dof+1] + M1*(xx2[n1e*dof+1] - xxn2[n1e*dof+1]) - M3*xxdd2[n1e*dof+1]; */
	    /* xxdd2[dof*n1e+2] = 0.;//- M2*xxd2[n1e*dof+2] + M1*(xx2[n1e*dof+2] - xxn2[n1e*dof+2]) - M3*xxdd2[n1e*dof+2]; */
	    }
	    if (!ibm2->contact[n2e]) { 
	      xxd2[dof*n2e  ] += (ubnn - ub2n)*dir[0];
	      xxd2[dof*n2e+1] += (ubnn - ub2n)*dir[1];
	      xxd2[dof*n2e+2] += (ubnn - ub2n)*dir[2];

	      /* xx2[dof*n2e  ] += ubnn*dt*dir[0]; */
	      /* xx2[dof*n2e+1] += ubnn*dt*dir[1]; */
	      /* xx2[dof*n2e+2] += ubnn*dt*dir[2]; */

	      /* xxdd2[dof*n2e  ] = 0.;//-M2*xxd2[n2e*dof  ]+M1*(xx2[n2e*dof  ]-xxn2[n2e*dof  ])-M3*xxdd2[n2e*dof  ]; */
	      /* xxdd2[dof*n2e+1] = 0.;//-M2*xxd2[n2e*dof+1]+M1*(xx2[n2e*dof+1]-xxn2[n2e*dof+1])-M3*xxdd2[n2e*dof+1]; */
	      /* xxdd2[dof*n2e+2] = 0.;//-M2*xxd2[n2e*dof+2]+M1*(xx2[n2e*dof+2]-xxn2[n2e*dof+2])-M3*xxdd2[n2e*dof+2]; */
	    }
	    if (!ibm2->contact[n3e]) { 
	      xxd2[dof*n3e  ] += (ubnn - ub3n)*dir[0];
	      xxd2[dof*n3e+1] += (ubnn - ub3n)*dir[1];
	      xxd2[dof*n3e+2] += (ubnn - ub3n)*dir[2];

	      /* xx2[dof*n3e  ] += ubnn*dt*dir[0]; */
	      /* xx2[dof*n3e+1] += ubnn*dt*dir[1]; */
	      /* xx2[dof*n3e+2] += ubnn*dt*dir[2]; */

	    /* xxdd2[dof*n3e  ] = 0.;//- M2*xxd2[n3e*dof  ]+M1*(xx2[n3e*dof  ] - xxn2[n3e*dof  ]) - M3*xxdd2[n3e*dof  ]; */
	    /* xxdd2[dof*n3e+1] = 0.;//- M2*xxd2[n3e*dof+1]+M1*(xx2[n3e*dof+1] - xxn2[n3e*dof+1]) - M3*xxdd2[n3e*dof+1]; */
	    /* xxdd2[dof*n3e+2] = 0.;//- M2*xxd2[n3e*dof+2]+M1*(xx2[n3e*dof+2] - xxn2[n3e*dof+2]) - M3*xxdd2[n3e*dof+2]; */
	    }

	    ibm2->x_bp[n1e] = xx2[dof*n1e  ];  ibm2->x_bp[n2e] = xx2[dof*n2e  ];  ibm2->x_bp[n3e] = xx2[dof*n3e  ];
	    ibm2->y_bp[n1e] = xx2[dof*n1e+1];  ibm2->y_bp[n2e] = xx2[dof*n2e+1];  ibm2->y_bp[n3e] = xx2[dof*n3e+1];
	    ibm2->z_bp[n1e] = xx2[dof*n1e+2];  ibm2->z_bp[n2e] = xx2[dof*n2e+2];  ibm2->z_bp[n3e] = xx2[dof*n3e+2];

	    /* xxd2[dof*n1e  ] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[0]; */
	    /* xxd2[dof*n1e+1] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[1]; */
	    /* xxd2[dof*n1e+2] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[2]; */
	    /* xxd2[dof*n2e  ] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[0]; */
	    /* xxd2[dof*n2e+1] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[1]; */
	    /* xxd2[dof*n2e+2] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[2]; */
	    /* xxd2[dof*n3e  ] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[0]; */
	    /* xxd2[dof*n3e+1] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[1]; */
	    /* xxd2[dof*n3e+2] += mb/(ibm2->m[n1e] + ibm2->m[n2e] + ibm2->m[n3e])*(ubnn - ubn)*dir[2]; */


	    // Update acceleration--- test, Iman 10/13/22
	    /* xxdd2[dof*n1e  ] -= (-M1*c*dist-M2*(ubnn - ub1n))*dir[0]/M3;   */
	    /* xxdd2[dof*n1e+1] -= (-M1*c*dist-M2*(ubnn - ub1n))*dir[1]/M3; //(ubnn - ub1n)/dt*dir[1];   */
	    /* xxdd2[dof*n1e+2] -= (-M1*c*dist-M2*(ubnn - ub1n))*dir[2]/M3; //(ubnn - ub1n)/dt*dir[2]; */
	    /* xxdd2[dof*n2e  ] -= (-M1*c*dist-M2*(ubnn - ub2n))*dir[0]/M3; //(ubnn - ub2n)/dt*dir[0];   */
	    /* xxdd2[dof*n2e+1] -= (-M1*c*dist-M2*(ubnn - ub2n))*dir[1]/M3; //(ubnn - ub2n)/dt*dir[1];   */
	    /* xxdd2[dof*n2e+2] -= (-M1*c*dist-M2*(ubnn - ub2n))*dir[2]/M3; //(ubnn - ub2n)/dt*dir[2]; */
	    /* xxdd2[dof*n3e  ] -= (-M1*c*dist-M2*(ubnn - ub3n))*dir[0]/M3; //(ubnn - ub3n)/dt*dir[0];   */
	    /* xxdd2[dof*n3e+1] -= (-M1*c*dist-M2*(ubnn - ub3n))*dir[1]/M3; //(ubnn - ub3n)/dt*dir[1];   */
	    /* xxdd2[dof*n3e+2] -= (-M1*c*dist-M2*(ubnn - ub3n))*dir[2]/M3; //(ubnn - ub3n)/dt*dir[2]; */
	    
	    /* xxdd2[dof*n1e  ] -= (-M1*c*dist)*dir[0]/M3;   */
	    /* xxdd2[dof*n1e+1] -= (-M1*c*dist)*dir[1]/M3; //(ubnn - ub1n)/dt*dir[1];   */
	    /* xxdd2[dof*n1e+2] -= (-M1*c*dist)*dir[2]/M3; //(ubnn - ub1n)/dt*dir[2]; */
	    /* xxdd2[dof*n2e  ] -= (-M1*c*dist)*dir[0]/M3; //(ubnn - ub2n)/dt*dir[0];   */
	    /* xxdd2[dof*n2e+1] -= (-M1*c*dist)*dir[1]/M3; //(ubnn - ub2n)/dt*dir[1];   */
	    /* xxdd2[dof*n2e+2] -= (-M1*c*dist)*dir[2]/M3; //(ubnn - ub2n)/dt*dir[2]; */
	    /* xxdd2[dof*n3e  ] -= (-M1*c*dist)*dir[0]/M3; //(ubnn - ub3n)/dt*dir[0];   */
	    /* xxdd2[dof*n3e+1] -= (-M1*c*dist)*dir[1]/M3; //(ubnn - ub3n)/dt*dir[1];   */
	    /* xxdd2[dof*n3e+2] -= (-M1*c*dist)*dir[2]/M3; //(ubnn - ub3n)/dt*dir[2]; */







	  } //if ub-ua>0

	  ibm2->contact[n1e] += 1;  ibm2->contact[n2e] += 1;  ibm2->contact[n3e] += 1;

	  } // if contact==0
	}	  	
      }
    } // if in bounding box  
  } //nv

  VecRestoreArray(fem1->x, &xx1);
  VecRestoreArray(fem2->x, &xx2);
  VecRestoreArray(fem1->xd, &xxd1);
  VecRestoreArray(fem2->xd, &xxd2);
  VecRestoreArray(fem1->xdd, &xxdd1);
  VecRestoreArray(fem2->xdd, &xxdd2);
  VecRestoreArray(fem1->xn, &xxn1);
  VecRestoreArray(fem2->xn, &xxn2);
  //
  if (nbody==3)  EdgeFix(1, fem1);
  //
  return(0);
}

//------------------------------------------------------------------------------------------------------------
/* PetscErrorCode Fcontactij3(FE *fem1, FE *fem2, PetscInt ii, PetscInt jj) { */
  
/*   IBMNodes       *ibm1=fem1->ibm, *ibm2=fem2->ibm; */
/*   PetscInt	 i, j, k, nv, ncx=15, ncy=15, ncz=15; */
/*   PetscReal	 xbp_min, ybp_min, zbp_min, xbp_max, ybp_max, zbp_max, dcx, dcy, dcz; */
/*   PetscReal	 *x_bp = ibm2->x_bp, *y_bp = ibm2->y_bp, *z_bp = ibm2->z_bp; */
/*   PetscInt 	 n_v = ibm2->n_v, ln_v; */
/*   struct Cmpnts  pnv; */
/*   List           *cell_trg; */
/*   PetscReal	 xv_min, yv_min, zv_min, xv_max, yv_max, zv_max; */
/*   PetscInt	 iv_min, iv_max, jv_min, jv_max, kv_min, kv_max, n1e, n2e, n3e, ic, jc, kc; */

/*   // Bounding Box for ibm2 */
/*   xbp_min = 1.e23;  xbp_max = -1.e23; */
/*   ybp_min = 1.e23;  ybp_max = -1.e23; */
/*   zbp_min = 1.e23;  zbp_max = -1.e23; */
  
/*   for(i=0; i<n_v; i++) { */
/*     xbp_min = PetscMin(xbp_min, x_bp[i]); */
/*     xbp_max = PetscMax(xbp_max, x_bp[i]); */
    
/*     ybp_min = PetscMin(ybp_min, y_bp[i]); */
/*     ybp_max = PetscMax(ybp_max, y_bp[i]); */
    
/*     zbp_min = PetscMin(zbp_min, z_bp[i]); */
/*     zbp_max = PetscMax(zbp_max, z_bp[i]); */
/*   } */
 
/*   xbp_min -= 0.001;  xbp_max += 0.001; */
/*   ybp_min -= 0.001;  ybp_max += 0.001; */
/*   zbp_min -= 0.001;  zbp_max += 0.001; */
 
/*   //Control cell for ibm2 */
/*   dcx = (xbp_max - xbp_min)/(ncx - 1.); */
/*   dcy = (ybp_max - ybp_min)/(ncy - 1.); */
/*   dcz = (zbp_max - zbp_min)/(ncz - 1.); */

/*   PetscMalloc(ncz*ncy*ncx*sizeof(List), &cell_trg); */
 
/*   for (k=0; k<ncz; k++) { */
/*     for (j=0; j<ncy; j++) { */
/*       for (i=0; i<ncx; i++) { */
/*   	initlist(&cell_trg[k*ncx*ncy + j*ncx + i]); */
/*       } */
/*     } */
/*   } */
  
/*   for (ln_v=0; ln_v<ibm2->n_elmt; ln_v++) { */

/*     n1e = ibm2->nv1[ln_v];  n2e = ibm2->nv2[ln_v];  n3e = ibm2->nv3[ln_v]; */

/*     xv_min = PetscMin(PetscMin(x_bp[n1e], x_bp[n2e]), x_bp[n3e]); */
/*     xv_max = PetscMax(PetscMax(x_bp[n1e], x_bp[n2e]), x_bp[n3e]); */

/*     yv_min = PetscMin(PetscMin(y_bp[n1e], y_bp[n2e]), y_bp[n3e]); */
/*     yv_max = PetscMax(PetscMax(y_bp[n1e], y_bp[n2e]), y_bp[n3e]); */

/*     zv_min = PetscMin(PetscMin(z_bp[n1e], z_bp[n2e]), z_bp[n3e]); */
/*     zv_max = PetscMax(PetscMax(z_bp[n1e], z_bp[n2e]), z_bp[n3e]); */
    
/*     iv_min = floor((xv_min - xbp_min)/dcx); */
/*     iv_max = floor((xv_max - xbp_min)/dcx) + 1; */

/*     jv_min = floor((yv_min - ybp_min)/dcy); */
/*     jv_max = floor((yv_max - ybp_min)/dcy) + 1; */

/*     kv_min = floor((zv_min - zbp_min)/dcz); */
/*     kv_max = floor((zv_max - zbp_min)/dcz) + 1; */

/*     iv_min = (iv_min<0) ? 0:iv_min; */
/*     iv_max = (iv_max>ncx) ? ncx:iv_max; */

/*     jv_min = (jv_min<0) ? 0:jv_min; */
/*     jv_max = (jv_max>ncx) ? ncy:jv_max; */

/*     kv_min = (kv_min<0) ? 0:kv_min; */
/*     kv_max = (kv_max>ncz) ? ncz:kv_max; */
   
/*     // Insert IBM node information into a list */
/*     for (k=kv_min; k<kv_max; k++) { */
/*       for (j=jv_min; j<jv_max; j++) { */
/*   	for (i=iv_min; i<iv_max; i++) { */
/*   	  insertnode(&(cell_trg[k *ncx*ncy + j*ncx +i]), ln_v); */
/*   	} */
/*       } */
/*     } */
/*   } */
  
/*   // search if ibm1 intersects ibm2 and find contact forces for fem1 */
/*   PetscReal  *xx1, *xx2, dist, dmin=0.0001, dir[3], c=0.05; //0.1 */
/*   PetscInt   elmt2; */
  
/*   VecGetArray(fem1->x, &xx1);   */
/*   VecGetArray(fem2->x, &xx2); */
  
/*   for (nv=0; nv <ibm1->n_v; nv++) { */
/*     pnv.x = ibm1->x_bp[nv]; */
/*     pnv.y = ibm1->y_bp[nv]; */
/*     pnv.z = ibm1->z_bp[nv]; */

/*     if (pnv.x > xbp_min && pnv.x < xbp_max && */
/*     	pnv.y > ybp_min && pnv.y < ybp_max && */
/*     	pnv.z > zbp_min && pnv.z < zbp_max) { // if in bounding box */
      
/*       ic = floor((pnv.x - xbp_min)/dcx); */
/*       jc = floor((pnv.y - ybp_min)/dcy); */
/*       kc = floor((pnv.z - zbp_min)/dcz); */

/*       // find the closest triangle in control cells */
/*       nearestcellFEM(pnv, ibm2, &dist, &elmt2, dir, ic, jc, kc, ncx, ncy, ncz, cell_trg, body1 ,body2, nv); */
      
/*       if (elmt2>0) {       */
 
/* 	if (PetscAbsReal(dist)<dmin) { */
	  
/* 	  ibm1->contact[nv] = 1; */
/* 	  n1e = ibm2->nv1[elmt2];  n2e = ibm2->nv2[elmt2];  n3e = ibm2->nv3[elmt2]; */
/* 	  ibm2->contact[n1e] = 1;  ibm2->contact[n2e] = 1;  ibm2->contact[n3e] = 1; */
	  
/* 	  if (dist>0.) { */
	    
/* 	    xx1[dof*nv] = xx1[dof*nv] + c*(dmin - dist)*dir[0]; */
/* 	    xx1[dof*nv+1] = xx1[dof*nv+1] + c*(dmin - dist)*dir[1]; */
/* 	    xx1[dof*nv+2] = xx1[dof*nv+2] + c*(dmin - dist)*dir[2]; */
/* 	    ibm1->x_bp[nv] = xx1[dof*nv]; */
/* 	    ibm1->y_bp[nv] = xx1[dof*nv+1]; */
/* 	    ibm1->z_bp[nv] = xx1[dof*nv+2]; */

/* 	    xx2[dof*n1e] = xx2[dof*n1e] - c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n1e+1] = xx2[dof*n1e+1] - c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n1e+2] = xx2[dof*n1e+2] - c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n1e] = xx2[dof*n1e]; */
/* 	    ibm2->y_bp[n1e] = xx2[dof*n1e+1]; */
/* 	    ibm2->z_bp[n1e] = xx2[dof*n1e+2]; */

/* 	    xx2[dof*n2e] = xx2[dof*n2e] - c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n2e+1] = xx2[dof*n2e+1] - c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n2e+2] = xx2[dof*n2e+2] - c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n2e] = xx2[dof*n2e]; */
/* 	    ibm2->y_bp[n2e] = xx2[dof*n2e+1]; */
/* 	    ibm2->z_bp[n2e] = xx2[dof*n2e+2]; */

/* 	    xx2[dof*n3e] = xx2[dof*n3e] - c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n3e+1] = xx2[dof*n3e+1] - c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n3e+2] = xx2[dof*n3e+2] - c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n3e] = xx2[dof*n3e]; */
/* 	    ibm2->y_bp[n3e] = xx2[dof*n3e+1]; */
/* 	    ibm2->z_bp[n3e] = xx2[dof*n3e+2]; */
	    
/* 	  } else { */
	    
/* 	    xx1[dof*nv] = xx1[dof*nv] - c*(dmin - dist)*dir[0]; */
/* 	    xx1[dof*nv+1] = xx1[dof*nv+1] - c*(dmin - dist)*dir[1]; */
/* 	    xx1[dof*nv+2] = xx1[dof*nv+2] - c*(dmin - dist)*dir[2]; */
/* 	    ibm1->x_bp[nv] = xx1[dof*nv]; */
/* 	    ibm1->y_bp[nv] = xx1[dof*nv+1]; */
/* 	    ibm1->z_bp[nv] = xx1[dof*nv+2]; */

/* 	    xx2[dof*n1e] = xx2[dof*n1e] + c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n1e+1] = xx2[dof*n1e+1] + c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n1e+2] = xx2[dof*n1e+2] + c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n1e] = xx2[dof*n1e]; */
/* 	    ibm2->y_bp[n1e] = xx2[dof*n1e+1]; */
/* 	    ibm2->z_bp[n1e] = xx2[dof*n1e+2]; */

/* 	    xx2[dof*n2e] = xx2[dof*n2e] + c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n2e+1] = xx2[dof*n2e+1] + c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n2e+2] = xx2[dof*n2e+2] + c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n2e] = xx2[dof*n2e]; */
/* 	    ibm2->y_bp[n2e] = xx2[dof*n2e+1]; */
/* 	    ibm2->z_bp[n2e] = xx2[dof*n2e+2]; */

/* 	    xx2[dof*n3e] = xx2[dof*n3e] + c*(dmin - dist)*dir[0]; */
/* 	    xx2[dof*n3e+1] = xx2[dof*n3e+1] + c*(dmin - dist)*dir[1]; */
/* 	    xx2[dof*n3e+2] = xx2[dof*n3e+2] + c*(dmin - dist)*dir[2]; */
/* 	    ibm2->x_bp[n3e] = xx2[dof*n3e]; */
/* 	    ibm2->y_bp[n3e] = xx2[dof*n3e+1]; */
/* 	    ibm2->z_bp[n3e] = xx2[dof*n3e+2]; */
/* 	  }	   */
/* 	} // if dist<dmin */
/*       } */
/*     } // if in bounding box   */
/*   } //nv */

/*   VecRestoreArray(fem1->x, &xx1); */
/*   VecRestoreArray(fem2->x, &xx2); */
  
/*   EdgeFix(1, fem1); */
/*   EdgeFix(1, fem2); */
  
/*   return(0); */
/* } */

//------------------------------------------------------------------------------------------------------------
PetscErrorCode nearestcellFEM(struct Cmpnts p, IBMNodes *ibm, PetscReal *dmin, PetscInt *cell_min, PetscReal dir[3],
			      PetscInt ic, PetscInt jc, PetscInt kc, PetscInt ncx, PetscInt ncy, PetscInt ncz, List *cell_trg, PetscInt body1, PetscInt body2, PetscInt np) {

  PetscInt       *nv1=ibm->nv1, *nv2=ibm->nv2, *nv3=ibm->nv3;
  PetscReal      *nf_x=ibm->nf_x, *nf_y=ibm->nf_y, *nf_z=ibm->nf_z;
  PetscReal      *x_bp=ibm->x_bp, *y_bp=ibm->y_bp, *z_bp=ibm->z_bp;
  PetscInt       n_elmt=ibm->n_elmt;  
  PetscInt       ln_v;
  struct Cmpnts  p1, p2, p3, pc;
  PetscReal      tf;
  PetscInt       n1e, n2e, n3e;
  PetscReal      nfx, nfy, nfz;
  struct Cmpnts  pj; //projection point
  struct Cmpnts  pmin, po, direct;
  PetscReal      d, d_center;
  node           *current;
  PetscInt       i, j, k, im, jm, km;

  *dmin = 1.e10;
  *cell_min = -100;
  km = ncz;
  jm = ncy;
  im = ncx;
  //additionally search one before and one after the original control cell
  if (kc<ncz-2)  km = kc + 2;
  if (jc<ncy-2)  jm = jc + 2;
  if (ic<ncx-2)  im = ic + 2;

  if (kc>0) kc--;
  if (jc>0) jc--;
  if (ic>0) ic--;

  for (k=kc; k<km; k++) {
    for (j=jc; j<jm; j++) {
      for (i=ic; i<im; i++) {
  	current = cell_trg[k*ncx*ncy+j*ncx+i].head;

  	while (current) {
  	  ln_v = current->Node;
	  n1e = nv1[ln_v];  n2e = nv2[ln_v];  n3e = nv3[ln_v];

	  if (body1==body2 && (np==n1e || np==n2e || np==n3e))  { //for self-contact
	    current = current->next;
	    continue;
	  }

	  BoundingSphere(ibm, ln_v);

	  d_center = SIZE(MINUS(p, ibm->qvec[ln_v]));
	  
	  //if (PetscAbsReal(d_center - ibm->radvec[ln_v])<PetscAbsReal(*dmin)) {
	    if (d_center < ibm->radvec[ln_v]) {
	    //n1e = nv1[ln_v];  n2e = nv2[ln_v];  n3e = nv3[ln_v];
	    nfx = nf_x[ln_v];  nfy = nf_y[ln_v];  nfz = nf_z[ln_v];
	    
	    p1.x = x_bp[n1e];  p1.y = y_bp[n1e];  p1.z = z_bp[n1e];
	    p2.x = x_bp[n2e];  p2.y = y_bp[n2e];  p2.z = z_bp[n2e];
	    p3.x = x_bp[n3e];  p3.y = y_bp[n3e];  p3.z = z_bp[n3e];
	    
	    tf = ((p.x - x_bp[n1e])*nfx +
		  (p.y - y_bp[n1e])*nfy +
		  (p.z - z_bp[n1e])*nfz);
	    
	    pj.x = p.x - tf*nfx;
	    pj.y = p.y - tf*nfy;
	    pj.z = p.z - tf*nfz;
	    
	    if (ISPointInTriangle(pj, p1, p2, p3, nfx, nfy, nfz) == 1) { /* The projected point is inside the  triangle */
	      
	      if (PetscAbsReal(tf)<PetscAbsReal(*dmin)) {
		*dmin = tf;	 
		pmin.x = pj.x;
		pmin.y = pj.y;
		pmin.z = pj.z;
		*cell_min = ln_v;	  	  
	      }	      
	    }
	  } // in the bounding sphere
	  current = current->next;
	} // while current
      }
    }
  }

  direct = UNIT(MINUS(p, pmin));  
  dir[0] = direct.x;  dir[1] = direct.y;  dir[2] = direct.z;

  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode BoundingSphere(IBMNodes *ibm, PetscInt ln_v) {

  PetscInt       *nv1 = ibm->nv1, *nv2 = ibm->nv2, *nv3 = ibm->nv3;
  PetscReal      *x_bp = ibm->x_bp, *y_bp = ibm->y_bp, *z_bp = ibm->z_bp;
  PetscInt       n_elmt = ibm->n_elmt;
  //PetscInt       ln_v;
  struct Cmpnts  p1, p2, p3;
  PetscInt       n1e, n2e, n3e;
  struct Cmpnts  pa, pb, pc, pu, pv, pf, pd, pt;
  PetscReal      l12, l23, l31;
  PetscReal      gama, lamda;

  n1e = nv1[ln_v];  n2e = nv2[ln_v];  n3e = nv3[ln_v];
  
  p1.x = x_bp[n1e];  p1.y = y_bp[n1e];  p1.z = z_bp[n1e];
  p2.x = x_bp[n2e];  p2.y = y_bp[n2e];  p2.z = z_bp[n2e];
  p3.x = x_bp[n3e];  p3.y = y_bp[n3e];  p3.z = z_bp[n3e];
  
  l12 = SIZE(MINUS(p1, p2));  l23 = SIZE(MINUS(p2, p3));  l31 = SIZE(MINUS(p3, p1));
  
  /* Find the longest edge and assign the corresponding two vertices
     to pa and pb */
  if (l12>l23) {
    if (l12>l31) {
      pa = p1;  pb = p2;  pc = p3;
    }
    else {
      pa = p3;  pb = p1;  pc = p2;
    }
  }
  else {
    if (l31<l23) {
      pa = p2;  pb = p3;  pc = p1;
    }
    else {
      pa = p3;  pb = p1;  pc = p2;
    }
  }
  
  pf.x = 0.5*(pa.x + pb.x);
  pf.y = 0.5*(pa.y + pb.y);
  pf.z = 0.5*(pa.z + pb.z);
  
  // u = a - f; v = c - f;
  pu = MINUS(pa, pf);
  pv = MINUS(pc, pf);
  
  // d = (u X v) X u;
  pt = CROSS(pu, pv);
  pd = CROSS(pt, pu);
  
  // gama = (v^2 - u^2) / (2 d \dot (v - u)); // this is the correct form of point_pair pdf
  gama = -(SIZE(pu)*SIZE(pu) - SIZE(pv)*SIZE(pv));
  
  pt = MINUS(pv, pu);
  lamda = 2*(pd.x*pt.x + pd.y*pt.y + pd.z*pt.z);
  
  gama /= lamda;
  
  if (gama<0) {
    lamda = 0;
  }
  else {
    lamda = gama;
  }
  
  ibm->qvec[ln_v].x = pf.x + lamda*pd.x;
  ibm->qvec[ln_v].y = pf.y + lamda*pd.y;
  ibm->qvec[ln_v].z = pf.z + lamda*pd.z;

  ibm->radvec[ln_v] = SIZE(MINUS(ibm->qvec[ln_v], pa));   
 
  return 0;
}

//------------------------------------------------------------------------------------------------------------
PetscInt ISPointInTriangle(struct Cmpnts p, struct Cmpnts p1, struct Cmpnts p2, struct Cmpnts p3,
			   PetscReal nfx, PetscReal nfy, PetscReal nfz) {

  PetscInt  flag;
  Cpt2D	    pj, pj1, pj2, pj3;
  if (fabs(nfz)>=fabs(nfx) && fabs(nfz)>=fabs(nfy)) {
    pj.x =  p.x;  pj.y = p.y;
    pj1.x = p1.x;  pj1.y = p1.y;
    pj2.x = p2.x;  pj2.y = p2.y;
    pj3.x = p3.x;  pj3.y = p3.y;
  }
  else if (fabs(nfx)>=fabs(nfy) && fabs(nfx)>=fabs(nfz)) {
    pj.x = p.z;  pj.y = p.y;
    pj1.x = p1.z;  pj1.y = p1.y;
    pj2.x = p2.z;  pj2.y = p2.y;
    pj3.x = p3.z;  pj3.y = p3.y;
  }
  else {
    pj.x = p.x;  pj.y = p.z;
    pj1.x = p1.x;  pj1.y = p1.z;
    pj2.x = p2.x;  pj2.y = p2.z;
    pj3.x = p3.x;  pj3.y = p3.z;
  }
  flag = ISInsideTriangle2D(pj, pj1, pj2, pj3);
  
  return(flag);
}

//------------------------------------------------------------------------------------------------------------
PetscErrorCode Sign_Dis_P_Line(struct Cmpnts p, struct Cmpnts p1, struct Cmpnts p2, PetscReal nfx, PetscReal nfy, PetscReal nfz, struct Cmpnts *po, PetscReal *d) {

  PetscReal  dmin;
  PetscReal  dx21, dy21, dz21, dx31, dy31, dz31, t;

  *d = 1.e6;
  
  dx21 = p2.x - p1.x;  dy21 = p2.y - p1.y;  dz21 = p2.z - p1.z;
  dx31 = p.x  - p1.x;  dy31 = p.y  - p1.y;  dz31 = p.z  - p1.z;
  
  t = (dx31*dx21 + dy31*dy21 + dz31*dz21)/(dx21*dx21 + dy21*dy21 +
						   dz21*dz21);
  if (t<0) { // The closet point is p1
    po->x = p1.x;  po->y = p1.y;  po->z = p1.z;
    *d = sqrt(dx31*dx31 + dy31*dy31 + dz31*dz31);
    
  } else if (t>1) { // The closet point is p2
    po->x = p2.x;  po->y = p2.y;  po->z = p2.z;
    *d = sqrt((p.x - po->x)*(p.x - po->x)+(p.y - po->y)*(p.y - po->y) +
  	      (p.z - po->z)*(p.z - po->z));
  
  } else { // The closet point lies between p1 & p2
    po->x = p1.x + t*dx21;  po->y = p1.y + t*dy21;  po->z = p1.z + t*dz21;
    *d = sqrt((p.x - po->x)*(p.x - po->x) + (p.y - po->y)*(p.y - po->y) +
  	      (p.z - po->z)*(p.z - po->z));
  }

  if (nfx*(p.x-po->x)+nfy*(p.y-po->y)+nfz*(p.z-po->z)<0) {*d = -*d;}

  return(0);
}

//------------------------------------------------------------------------------------------------------------
PetscInt ISInsideTriangle2D(Cpt2D p, Cpt2D pa, Cpt2D pb, Cpt2D pc) {

  // Check if point p and p3 is located on the same side of line p1p2
  PetscInt  ls;
  
  ls = ISSameSide2D(p, pa, pb, pc);
  //  if (flagprin) PetscPrintf(PETSC_COMM_WORLD, "aaa, %d\n", ls);
  if (ls < 0) {
    return (ls);
  }
  ls = ISSameSide2D(p, pb, pc, pa);
  //  if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "bbb, %d\n", ls);
  if (ls < 0) {
    return (ls);
  }
  ls = ISSameSide2D(p, pc, pa, pb);
  //  if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "ccc, %d\n", ls);
  if (ls <0) {
    return(ls);
  }

  return (ls);
}

//------------------------------------------------------------------------------------------------------------
PetscInt ISSameSide2D(Cpt2D p, Cpt2D p1, Cpt2D p2, Cpt2D p3) {
  /* Check whether 2D point p is located on the same side of line p1p2
     with point p3. Returns:
     -1	different side
     1	same side (including the case when p is located
     right on the line)
     If p and p3 is located on the same side to line p1p2, then
     the (p-p1) X (p2-p1) and (p3-p1) X (p2-p1) should have the same sign
  */
  PetscReal  t1, t2, t3;
  PetscReal  epsilon=1.e-10;
  PetscReal  A, B, C;

  A = p2.y - p1.y;
  B = -(p2.x - p1.x);
  C = (p2.x - p1.x) * p1.y - (p2.y - p1.y) * p1.x;
  
  t3 = fabs(A * p.x + B * p.y + C) / sqrt(A*A + B*B);
  
  /*   if (t3<1.e-3) return(1); */
  if (t3 < 1.e-3) {
    t1 = A * p.x + B * p.y + C;
    t2 = A * p3.x + B * p3.y + C;
    //    if (flagprint) PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", t1, t2, t3, A, B, C);
  }
  else {
    t1 = (p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x);
    t2 = (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x);
  }
  
  //!!!!!!!!!!!!1 Change t1, t2 & lt !!!!!!!
  t1 = (p.x - p1.x) * (p2.y - p1.y) - (p.y - p1.y) * (p2.x - p1.x);
  t2 = (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x);
  PetscReal  lt;
  lt = sqrt((p1.x - p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
  //  if(flagprint) PetscPrintf(PETSC_COMM_WORLD, "%le %le %le %le %le %le\n", p1.x, p2.x, p3.x, p1.y, p2.y, p3.y);
  //if (fabs(t1) < epsilon) { // Point is located along the line of p1p2
  if (fabs(t1/lt) < epsilon) { // Point is located along the line of p1p2
    return(1);
  }
  // End of change !!!!!!!!!!!!!1
  
  if (t1 > 0) {
    if (t2 > 0) return (1); // same side
    else return(-1);  // not
  }
  else {
    if (t2 < 0) return(1); // same side
    else return(-1);
  }
}

//-------------------------------------------------------
/* PetscErrorCode Fcontact2(FE *fem) { */
/*   PetscInt i,j; */
/*   PetscReal sumFcnt,sumFext; */
/*   for (i=0;i<nbody;i++) { */
    
/*     VecSet(fem[i].Fcnt,0.0); */
/*     for (j=0;j<nbody;j++) { */
/*       if (i!=j) { */
/* 	Fcontactij2(&fem[i],&fem[j]); */
	
/* 	sumFcnt=0.0; sumFext=0.0; */
/* 	VecSum(fem[i].Fcnt, &sumFcnt); VecSum(fem[i].Fext,&sumFext); */
/* 	PetscPrintf(PETSC_COMM_WORLD, "Contact of leaflet %d and %d Fcnt=%le , Fext=%le\n",i,j,sumFcnt,sumFext);    */
/*       } */
/*     } */
/*   } */

/*   return(0); */
/* } */

/* //--------------------------------------------------------- */
/* PetscErrorCode Fcontactij2(FE *fem1, FE *fem2) { */
/* /\* fem1 the first leaflet, fem2 second leaflet *\/ */

/*   PetscInt	i, j, k, nv; */
/*   IBMNodes       *ibm1=fem1->ibm, *ibm2=fem2->ibm; */
/*   PetscInt	ncx = 15, ncy = 15, ncz = 15; //control cells number */
/*   List          *cell_trg; */
/*   PetscReal	xbp_min, ybp_min, zbp_min, xbp_max, ybp_max, zbp_max, dx,dy,dz; */
/*   PetscReal	*x_bp = ibm2->x_bp, *y_bp = ibm2->y_bp, *z_bp = ibm2->z_bp; */
  
/*   PetscInt 	ln_v, n_v = ibm2->n_v; */
/*   PetscReal	dcx, dcy, dcz; */
/*   PetscInt	n1e, n2e, n3e; */
/*   PetscReal	xv_min, yv_min, zv_min, xv_max, yv_max, zv_max; */
/*   PetscInt	iv_min, iv_max, jv_min, jv_max, kv_min, kv_max; */
/*   PetscInt	ic, jc, kc; */
  
/*   struct Cmpnts       pnv; */
  
/*   // Bounding Box for fem2 */
/*   xbp_min = 1.e23; xbp_max = -1.e23; */
/*   ybp_min = 1.e23; ybp_max = -1.e23; */
/*   zbp_min = 1.e23; zbp_max = -1.e23; */
  
/*   for(i=0; i<n_v; i++) { */
/*     xbp_min = PetscMin(xbp_min, x_bp[i]); */
/*     xbp_max = PetscMax(xbp_max, x_bp[i]); */
    
/*     ybp_min = PetscMin(ybp_min, y_bp[i]); */
/*     ybp_max = PetscMax(ybp_max, y_bp[i]); */
    
/*     zbp_min = PetscMin(zbp_min, z_bp[i]); */
/*     zbp_max = PetscMax(zbp_max, z_bp[i]); */
/*   } */
  
/*   xbp_min -=0.05; xbp_max +=0.05; */
/*   ybp_min -=0.05; ybp_max +=0.05; */
/*   zbp_min -=0.05; zbp_max +=0.05; */
  
/*   // Control cells for fem2 */
/*   dcx = (xbp_max - xbp_min) / (ncx - 1.); */
/*   dcy = (ybp_max - ybp_min) / (ncy - 1.); */
/*   dcz = (zbp_max - zbp_min) / (ncz - 1.); */
  
/*   PetscMalloc(ncz * ncy * ncx * sizeof(List), &cell_trg); */
  
/*   for (k=0; k<ncz; k++) { */
/*     for (j=0; j<ncy; j++) { */
/*       for (i=0; i<ncx; i++) { */
/* 	initlist(&cell_trg[k*ncx*ncy + j*ncx + i]); */
/*       } */
/*     } */
/*   } */
  
/*   for (ln_v=0; ln_v<ibm2->n_elmt; ln_v++) { */

/*     n1e = ibm2->nv1[ln_v];  n2e = ibm2->nv2[ln_v];  n3e = ibm2->nv3[ln_v]; */

/*     xv_min = PetscMin(PetscMin(x_bp[n1e], x_bp[n2e]), x_bp[n3e]); */
/*     xv_max = PetscMax(PetscMax(x_bp[n1e], x_bp[n2e]), x_bp[n3e]); */

/*     yv_min = PetscMin(PetscMin(y_bp[n1e], y_bp[n2e]), y_bp[n3e]); */
/*     yv_max = PetscMax(PetscMax(y_bp[n1e], y_bp[n2e]), y_bp[n3e]); */

/*     zv_min = PetscMin(PetscMin(z_bp[n1e], z_bp[n2e]), z_bp[n3e]); */
/*     zv_max = PetscMax(PetscMax(z_bp[n1e], z_bp[n2e]), z_bp[n3e]); */
    
/*     iv_min = floor((xv_min - xbp_min)/dcx); */
/*     iv_max = floor((xv_max - xbp_min)/dcx) + 1; */

/*     jv_min = floor((yv_min - ybp_min)/dcy); */
/*     jv_max = floor((yv_max - ybp_min)/dcy) + 1; */

/*     kv_min = floor((zv_min - zbp_min)/dcz); */
/*     kv_max = floor((zv_max - zbp_min)/dcz) + 1; */

/*     iv_min = (iv_min<0) ? 0:iv_min; */
/*     iv_max = (iv_max>ncx) ? ncx:iv_max; */

/*     jv_min = (jv_min<0) ? 0:jv_min; */
/*     jv_max = (jv_max>ncx) ? ncy:jv_max; */

/*     kv_min = (kv_min<0) ? 0:kv_min; */
/*     kv_max = (kv_max>ncz) ? ncz:kv_max; */
   
/*     // Insert IBM node information into a list */
/*     for (k=kv_min; k<kv_max; k++) { */
/*       for (j=jv_min; j<jv_max; j++) { */
/*   	for (i=iv_min; i<iv_max; i++) { */
/*   	  insertnode(&(cell_trg[k *ncx*ncy + j*ncx +i]), ln_v); */
/*   	} */
/*       } */
/*     } */
/*   } */

/*   // search if fem1 intersects fem2 and find contact forces for fem1 */
/*   PetscReal *FFcnt, *FFext,fext, d, dist, kf=10.,nfx,nfy,nfz, dmin=0.002, dir[3];//kf=10.0 , dmin=0.029  */
  
/*   PetscInt  elmt2,method; */
/*   VecGetArray(fem1->Fcnt, &FFcnt); */
/*   VecGetArray(fem1->Fext, &FFext); */
 
/*   for (nv=0; nv <ibm1->n_v; nv++) { */
/*     pnv.x = ibm1->x_bp[nv]; */
/*     pnv.y = ibm1->y_bp[nv]; */
/*     pnv.z = ibm1->z_bp[nv]; */
    
/*     if (pnv.x > xbp_min && pnv.x < xbp_max && */
/* 	pnv.y > ybp_min && pnv.y < ybp_max && */
/* 	pnv.z > zbp_min && pnv.z < zbp_max) { // if in bounding box   */
    
/*       ic = floor((pnv.x - xbp_min)/dcx); */
/*       jc = floor((pnv.y - ybp_min)/dcy); */
/*       kc = floor((pnv.z - zbp_min)/dcz); */
/*       // find the closest triangle in the contorl cell */
     
/*       nearestcellFEM(pnv, ibm2, &dist, &elmt2, dir, ic, jc, kc, ncx, ncy, ncz, cell_trg, body1, body2, nv); */
/*       if (elmt2>0) { */
/* 	nfx = ibm2->nf_x[elmt2]; */
/* 	nfy = ibm2->nf_y[elmt2];       */
/* 	nfz = ibm2->nf_z[elmt2]; */
/* 	fext = sqrt(FFext[nv*dof  ]*FFext[nv*dof  ] + */
/* 		    FFext[nv*dof+1]*FFext[nv*dof+1] + */
/* 		    FFext[nv*dof+2]*FFext[nv*dof+2] ); */
/* 	if (fext<1e-8) fext=1e-8; */
        
/* 	if (dist < dmin || ibm1->contact[nv]) { //1.25 dmin// add contact force */
	 
/* 	  ibm1->contact[nv] = 1; */
/* 	  if (dist<0.) { */
/* 	    FFcnt[nv*dof  ]-=(fext - kf*dist)*nfx; */
/* 	    FFcnt[nv*dof+1]-=(fext - kf*dist)*nfy; */
/* 	    FFcnt[nv*dof+2]-=(fext - kf*dist)*nfz; */
	  
/* 	  }else { */
/* 	    FFcnt[nv*dof  ]-=(fext*exp(-kf*dist/fext))*nfx; */
/* 	    FFcnt[nv*dof+1]-=(fext*exp(-kf*dist/fext))*nfy; */
/* 	    FFcnt[nv*dof+2]-=(fext*exp(-kf*dist/fext))*nfz; */
/* 	  }	  	   */
	  
/* 	} // if dist<dmin */
/*       } */
/*   } // if in bounding box */
    
    
/*   } //nv */
/*   VecRestoreArray(fem1->Fcnt, &FFcnt); */
/*   VecRestoreArray(fem1->Fext, &FFext); */
/*   EdgeFix(1, fem1); */
  
/*   return(0); */
/* } */
