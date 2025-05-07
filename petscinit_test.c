#include <petsc.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) {
        PetscPrintf(PETSC_COMM_WORLD, "Error initializing PETSc\n");
        return ierr;
    }

    PetscPrintf(PETSC_COMM_WORLD, "PETSc initialized successfully!\n");

    ierr = PetscFinalize();
    return ierr;
}
