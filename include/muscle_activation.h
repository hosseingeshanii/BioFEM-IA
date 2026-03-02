#ifndef MUSCLE_ACTIVATION_H
#define MUSCLE_ACTIVATION_H

#include "variables.h"

/**
 * @file muscle_activation.h
 * @brief Declarations for muscle activation module using active-strain and active-stress approaches.
 *
 * References:
 *  - Nitti, A., et al. (2021). A curvilinear isogeometric framework for the purely elastic response of shell structures
 *    undergoing active strain patterns. *Computers & Structures*, 248, 106533.
 *  - Torre, M., et al. (2024). An efficient active-stress electromechanical isogeometric formulation for Kirchhoff–Love shells.
 *    *Computers & Structures*, 276, 106879.
 *
 * @author Hossein Geshani
 * @date 2025-09-22
 */


/**
 * @brief Update muscle activation parameters from command-line options.
 *
 * Reads PETSc options (e.g. -muscle_act_gamma, -muscle_act_a_1, -muscle_act_a_2)
 * and updates the corresponding fields in the FE user context.
 *
 * @param[in,out] fem Finite element structure containing UserCtx with activation parameters.
 * @return PetscErrorCode 0 on success.
 */
PetscErrorCode update_user_act_params(FE *fem);

/**
 * @brief Step 1: Compute the active part of the deformation gradient tensor (Fa_ij) 
 *        for a given shell element.
 *
 * @param ec Index of the shell element
 * @return PetscErrorCode Standard PETSc error code
 */
PetscErrorCode compute_active_Fa_element(FE *fem, PetscInt ec);

/**
 * @brief Updates all active Fa values for each element.
 *
 * Calls compute_active_Fa_element for every element in the FE structure.
 *
 * @param fem Pointer to the FE structure.
 * @return PetscErrorCode Returns 0 on success.
 */
PetscErrorCode update_act_Fa(FE *fem);


/**
 * @brief Compute intermediate deformed basis vectors for a given element.
 *
 * Uses covariant basis vectors and the active deformation gradient to
 * update the deformed covariant basis vectors g_e for element ec.
 *
 * @param[in,out] fem Finite element structure with geometry and activation data.
 * @param[in]     ec  Element index.
 * @return PetscErrorCode 0 on success.
 */
PetscErrorCode compute_intermediate_state(FE *fem, PetscInt ec);



/**
 * @brief Update intermediate deformed basis vectors for all elements.
 *
 * Calls compute_intermediate_state() for each element in the mesh.
 *
 * @param[in,out] fem Finite element structure with geometry and activation data.
 * @return PetscErrorCode 0 on success.
 */
PetscErrorCode update_intmd_state_cov_basis(FE *fem);


/**
 * @brief Step 4: Update elastic strain and stress for a shell element
 *
 * @param ec Index of the shell element
 * @return PetscErrorCode Standard PETSc error code
 */
PetscErrorCode update_elastic_strain_stress(PetscInt ec);

/**
 * @brief Step 5: Update internal nodal forces on the patch corresponding to a shell element
 *
 * @param ec Index of the shell element
 * @return PetscErrorCode Standard PETSc error code
 */
PetscErrorCode update_internal_nodal_forces(PetscInt ec);

/**
 * @brief Step 6: Assemble nodal internal forces (looping through all elements)
 *
 * @return PetscErrorCode Standard PETSc error code
 */
PetscErrorCode assemble_nodal_internal_forces(void);

PetscErrorCode calculate_cov_basis(Vec x_guess, Vec F, void *ctx);

PetscErrorCode form_interm_state_func_grad(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx);

PetscErrorCode MyTaoMonitor(Tao tao, void *ctx);

PetscErrorCode find_intmd_coords(FE *fem);

PetscErrorCode compute_elastic_stress_after_act(FE* fem, PetscInt ec);


#endif // MUSCLE_ACTIVATION_H
