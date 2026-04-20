#include <math.h>
#include <petscviewer.h>
#include "variables.h"
#include "manufactured_active_strain.h"
#include <math.h>
#include <petscsys.h>
#include <string.h>
#include <sys/stat.h>

// PetscInt  newmark = 0;
PetscReal manufactured_gamma0 = 0.0, manufactured_T = 1.0;

extern const PetscInt dof;
extern PetscInt       ti, manufactured, manufactured_fexternal_export, prescribed_force_field;
extern PetscInt       tistart, tisteps;
extern PetscReal      dt;
extern char           in_dir[256];
extern char           out_dir[256];

extern PetscErrorCode FDynamic(FE *fem);
extern PetscErrorCode xAccVel(FE *fem);
extern PetscErrorCode Output(FE *fem, PetscInt ti, PetscInt ibi, const char *out_dir);

PetscErrorCode GetManufacturedActiveStrainOptions(void)
{
  PetscFunctionBeginUser;

  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-manufactured_fexternal_export",
                     &manufactured_fexternal_export, PETSC_NULL);
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-prescribed_force_field",
                     &prescribed_force_field, PETSC_NULL);

  PetscFunctionReturn(0);
}


static PetscErrorCode ManufacturedForceVectorsOut_(FE *fem, PetscInt ti, PetscInt ibi)
{
  PetscFunctionBeginUser;

  PetscViewer viewer;
  char        filen[256];
  const char *dir = (out_dir && strlen(out_dir) > 0) ? out_dir : ".";

  mkdir(dir, 0777);

  snprintf(filen, sizeof(filen), "%s/fext%1.1d_%5.5d.dat", dir, ibi, ti);
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filen, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(fem->Fext, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}


static PetscErrorCode ManufacturedExactLocationsSet_(FE *fem, PetscInt step)
{
  PetscFunctionBeginUser;

  IBMNodes        *ibm = fem->ibm;
  PetscReal       *xx = NULL, *xxn = NULL, *xxd = NULL, *xxdd = NULL;
  PetscReal       *dxx = NULL;

  PetscReal        t = step * dt;
  const PetscReal  tnm1 = t - dt;

  PetscReal gamma, lambda, dgamma, dlambda, ddgamma, ddlambda;
  PetscReal gamman, lambdan;

  PetscCheck(manufactured_T > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "manufactured_T must be positive");
  PetscCheck(dt > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "dt must be positive");

  gamma     = manufactured_gamma0 * PetscPowReal(PetscSinReal(PETSC_PI * t / manufactured_T), 2.0);
  lambda    = 1.0 - gamma;
  dgamma    = manufactured_gamma0 * (PETSC_PI / manufactured_T) * PetscSinReal(2.0 * PETSC_PI * t / manufactured_T);
  dlambda   = -dgamma;
  ddgamma   = manufactured_gamma0 * (2.0 * PETSC_PI * PETSC_PI / (manufactured_T * manufactured_T))
            * PetscCosReal(2.0 * PETSC_PI * t / manufactured_T);
  ddlambda  = -ddgamma;

  gamman    = manufactured_gamma0 * PetscPowReal(PetscSinReal(PETSC_PI * tnm1 / manufactured_T), 2.0);
  lambdan   = 1.0 - gamman;

  PetscCall(VecGetArray(fem->x, &xx));
  PetscCall(VecGetArray(fem->xn, &xxn));
  PetscCall(VecGetArray(fem->dx, &dxx));

  for (PetscInt nv = 0; nv < ibm->n_v + ibm->n_ghosts; nv++) {
    const PetscReal X = ibm->x_bp0[nv];
    const PetscReal Y = ibm->y_bp0[nv];
    const PetscReal Z = ibm->z_bp0[nv];

    xx[nv*dof  ] = lambda * X;
    xx[nv*dof+1] = Y;
    xx[nv*dof+2] = Z;

    xxn[nv*dof  ] = lambdan * X;
    xxn[nv*dof+1] = Y;
    xxn[nv*dof+2] = Z;

    dxx[nv*dof  ] = (lambda - lambdan) * X;
    dxx[nv*dof+1] = 0.0;
    dxx[nv*dof+2] = 0.0;

    // if (nv == 79) {
    //   // PetscPrintf(PETSC_COMM_SELF, "nv=%d: X=%f, Y=%f, Z=%f, x=%f, xn=%f, dx=%f, lambda=%f, lambdan=%f, gamman=%f, gamma=%f, t=%f, tnm1=%f, dt=%f \n",
    //   //             nv, X, Y, Z, xx[nv*dof], xx[nv*dof+1], xx[nv*dof+2],
    //   //             xxn[nv*dof], xxn[nv*dof+1], xxn[nv*dof+2],
    //   //             dxx[nv*dof], dxx[nv*dof+1], dxx[nv*dof+2], lambda, lambdan, gamman, gamma, t, tnm1, dt);
    //   // PetscPrintf(PETSC_COMM_SELF, "dttti79 %f dxx %f \n", dt, dxx[nv*dof  ]);
    // }

    ibm->x_bp[nv] = xx[nv*dof  ];
    ibm->y_bp[nv] = Y;
    ibm->z_bp[nv] = Z;
  }

  PetscCall(VecRestoreArray(fem->dx, &dxx));
  PetscCall(VecRestoreArray(fem->xn, &xxn));
  PetscCall(VecRestoreArray(fem->x, &xx));


  PetscFunctionReturn(0);
}


static PetscErrorCode ManufacturedExactVelocityAccelerationSet_(FE *fem, PetscInt step)
{
  PetscFunctionBeginUser;

  IBMNodes  *ibm = fem->ibm;
  PetscReal *xxd = NULL, *xxdd = NULL;
  PetscReal  dgamma, dlambda, ddgamma, ddlambda;
  PetscReal        t = step * dt;

  PetscCheck(manufactured_T > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "manufactured_T must be positive");

  dgamma   = manufactured_gamma0 * (PETSC_PI / manufactured_T) * PetscSinReal(2.0 * PETSC_PI * t / manufactured_T);
  dlambda  = -dgamma;
  ddgamma  = manufactured_gamma0 * (2.0 * PETSC_PI * PETSC_PI / (manufactured_T * manufactured_T))
           * PetscCosReal(2.0 * PETSC_PI * t / manufactured_T);
  ddlambda = -ddgamma;

  PetscCall(VecGetArray(fem->xd, &xxd));
  PetscCall(VecGetArray(fem->xdd, &xxdd));

  for (PetscInt nv = 0; nv < ibm->n_v + ibm->n_ghosts; nv++) {
    const PetscReal X = ibm->x_bp0[nv];

    xxd[nv*dof  ] = dlambda * X;
    xxd[nv*dof+1] = 0.0;
    xxd[nv*dof+2] = 0.0;

    xxdd[nv*dof  ] = ddlambda * X;
    xxdd[nv*dof+1] = 0.0;
    xxdd[nv*dof+2] = 0.0;
  }

  PetscCall(VecRestoreArray(fem->xdd, &xxdd));
  PetscCall(VecRestoreArray(fem->xd, &xxd));

  PetscFunctionReturn(0);
}

PetscErrorCode GenerateManufacturedFExternalAndKinematics(FE *fem, PetscInt ibi, PetscInt step)
{
  PetscFunctionBeginUser;

  PetscCall(ManufacturedExactLocationsSet_(fem, step));
  PetscCall(ManufacturedExactVelocityAccelerationSet_(fem, step - 1));

  PetscCall(VecSet(fem->Fint, 0.0));
  PetscCall(VecSet(fem->Fext, 0.0));
  PetscCall(VecSet(fem->Fdyn, 0.0));

  // Newmark method needs previous time step velocity and acceleration 
  PetscCall(FDynamic(fem));
  PetscCall(VecCopy(fem->Fdyn, fem->Fext));
  
  if (manufactured_fexternal_export) {
    PetscCall(ManufacturedForceVectorsOut_(fem, step, ibi));
  }

  PetscCall(ManufacturedExactVelocityAccelerationSet_(fem, step));
  PetscCall(Output(fem, step, ibi, out_dir));

  PetscFunctionReturn(0);
}


PetscErrorCode GenerateManufacturedFExternalAndKinematicsAllTimeSteps(FE *fem, PetscInt ibi)
{
  PetscFunctionBeginUser;

  for (PetscInt step = tistart; step <= tistart + tisteps; step++) {

    GenerateManufacturedFExternalAndKinematics(fem, ibi, step);

  }

  PetscFunctionReturn(0);
}

PetscErrorCode ManufacturedKinematicsAndFExternalSetInitial(FE *fem, PetscInt ibi, PetscInt step)
{
  PetscFunctionBeginUser;

  GenerateManufacturedFExternalAndKinematics(fem, ibi, step);

  PetscFunctionReturn(0);
}
