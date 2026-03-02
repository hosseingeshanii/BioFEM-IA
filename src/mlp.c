#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include  "variables.h"

extern PetscInt dof, INPUT_DIM, OUTPUT_UNITS, EPOCHS;
extern PetscReal  LEARNING_RATE;
extern ActivationFunc activation;
// Hyperparameters
extern int hidden_layers;
extern int hidden_units[];  // Number of units in each hidden layer


float compute_E(float input[INPUT_DIM]);

// Activation functions
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}
float tanh_activation(float x) {
    return tanh(x);
}
float relu(float x) {
    return fmaxf(0, x); // Using fmaxf for max of two floats
}



float *epsilon_array;

float ***weights;
float **biases;

void initialize_network() {
    weights = (float ***)malloc((hidden_layers + 1) * sizeof(float **));
    biases = (float **)malloc((hidden_layers + 1) * sizeof(float *));
    
    // Allocate memory for weights and biases
    weights[0] = (float **)malloc(INPUT_DIM * sizeof(float *));
    for (int i = 0; i < INPUT_DIM; ++i) {
        weights[0][i] = (float *)malloc(hidden_units[0] * sizeof(float));
        for (int j = 0; j < hidden_units[0]; ++j) {
            weights[0][i][j] = ((float) rand() / (float) RAND_MAX) - 0.5f;
        }
    }
    biases[0] = (float *)malloc(hidden_units[0] * sizeof(float));
    for (int i = 0; i < hidden_units[0]; ++i) {
        biases[0][i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
    }

    for (int l = 1; l < hidden_layers; ++l) {
        weights[l] = (float **)malloc(hidden_units[l-1] * sizeof(float *));
        for (int i = 0; i < hidden_units[l-1]; ++i) {
            weights[l][i] = (float *)malloc(hidden_units[l] * sizeof(float));
            for (int j = 0; j < hidden_units[l]; ++j) {
                weights[l][i][j] = ((float) rand() / (float) RAND_MAX) - 0.5f;
            }
        }
        biases[l] = (float *)malloc(hidden_units[l] * sizeof(float));
        for (int i = 0; i < hidden_units[l]; ++i) {
            biases[l][i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
        }
    }

    weights[hidden_layers] = (float **)malloc(hidden_units[hidden_layers-1] * sizeof(float *));
    for (int i = 0; i < hidden_units[hidden_layers-1]; ++i) {
        weights[hidden_layers][i] = (float *)malloc(OUTPUT_UNITS * sizeof(float));
        for (int j = 0; j < OUTPUT_UNITS; ++j) {
            weights[hidden_layers][i][j] = ((float) rand() / (float) RAND_MAX) - 0.5f;
        }
    }
    biases[hidden_layers] = (float *)malloc(OUTPUT_UNITS * sizeof(float));
    for (int i = 0; i < OUTPUT_UNITS; ++i) {
        biases[hidden_layers][i] = ((float) rand() / (float) RAND_MAX) - 0.5f;
    }
}


float forward_pass(float input[INPUT_DIM]) {
    float *hidden[hidden_layers+1];
    for (int l = 0; l < hidden_layers; ++l) {
        hidden[l] = (float *)malloc(hidden_units[l] * sizeof(float));
    }

    // Input to first hidden layer
    for (int i = 0; i < hidden_units[0]; ++i) {
        hidden[0][i] = biases[0][i];
        for (int j = 0; j < INPUT_DIM; ++j) {
            hidden[0][i] += input[j] * weights[0][j][i];
        }
        hidden[0][i] = activation(hidden[0][i]);
    }

    // Hidden layers to hidden layers
    for (int l = 1; l < hidden_layers; ++l) {
        for (int i = 0; i < hidden_units[l]; ++i) {
            hidden[l][i] = biases[l][i];
            for (int j = 0; j < hidden_units[l-1]; ++j) {
                hidden[l][i] += hidden[l-1][j] * weights[l][j][i];
            }
            hidden[l][i] = activation(hidden[l][i]);
        }
    }

    // Last hidden layer to output
    float output = biases[hidden_layers][0];
    for (int i = 0; i < hidden_units[hidden_layers-1]; ++i) {
        output += hidden[hidden_layers-1][i] * weights[hidden_layers][i][0];
    }

    for (int l = 0; l < hidden_layers; ++l) {
        free(hidden[l]);
    }

    return output;
}


float compute_E(float input[INPUT_DIM]) {
    return forward_pass(input);
}


float compute_R(float *epsilon_array, int index, int dof_index, int ibi, FE* fem) {
    PetscScalar *RRes;
    printf("Check index, %d\n", index);
    ResidualCalc(epsilon_array, ibi, &fem[ibi]);
    VecGetArray(fem[ibi].Res, &RRes);
    if (index==40){
        for (int i=0; i<ibm->n_v; i++){  
            printf("node=%d, Res : %f,%f,%f \n", i, RRes[i*dof+0], RRes[i*dof+1], RRes[i*dof+2]);
        }
    }
    double point_res=(double)RRes[index*dof+dof_index];
    printf("Check point_res, %f\n", RRes[index*dof+dof_index]);
    VecRestoreArray(fem[ibi].Res, &RRes); 

    return point_res;
}


void backward_pass(float input[INPUT_DIM], int example_index, int dof_index, int ibi, FE* fem) {
    float *hidden[hidden_layers+1];
    float *deltas[hidden_layers+1];
    for (int l = 0; l < hidden_layers; ++l) {
        hidden[l] = (float *)malloc(hidden_units[l] * sizeof(float));
        deltas[l] = (float *)malloc(hidden_units[l] * sizeof(float));
    }
    
    hidden[hidden_layers] = (float *)malloc(OUTPUT_UNITS * sizeof(float));
    deltas[hidden_layers] = (float *)malloc(OUTPUT_UNITS * sizeof(float));

    // Forward pass to calculate output and hidden layer activations
    for (int i = 0; i < hidden_units[0]; ++i) {
        hidden[0][i] = biases[0][i];
        for (int j = 0; j < INPUT_DIM; ++j) {
            hidden[0][i] += input[j] * weights[0][j][i];
        }
        hidden[0][i] = activation(hidden[0][i]);
    }

    for (int l = 1; l < hidden_layers; ++l) {
        for (int i = 0; i < hidden_units[l]; ++i) {
            hidden[l][i] = biases[l][i];
            for (int j = 0; j < hidden_units[l-1]; ++j) {
                hidden[l][i] += hidden[l-1][j] * weights[l][j][i];
            }
            hidden[l][i] = activation(hidden[l][i]);
        }
    }

    float output = biases[hidden_layers][0];
    for (int i = 0; i < hidden_units[hidden_layers-1]; ++i) {
        output += hidden[hidden_layers-1][i] * weights[hidden_layers][i][0];
    }

    float epsilon = 1e-5;
    epsilon_array[example_index] = epsilon;
    //printf("Check before R_plus\n");  
    float R_plus = compute_R(epsilon_array, example_index, dof_index, ibi, &fem[ibi]);
    printf("Check after R_plus, %f\n", R_plus);
    epsilon_array[example_index] = -epsilon;
    float R_minus = compute_R(epsilon_array, example_index, dof_index, ibi, &fem[ibi]);
    printf("Check after R_minus, %f\n", R_minus);
    epsilon_array[example_index] = 0;  // Reset epsilon
    
    float dR_dE = (R_plus - R_minus) / (2 * epsilon);

    float target_R = 0.0;  // Example target value
    //float loss = (compute_R(epsilon_array, example_index, dof_index, ibi, &fem[ibi]) - target_R) * (compute_R(epsilon_array, example_index, dof_index, ibi, &fem[ibi]) - target_R);
    
    float output_delta = 2 * (compute_R(epsilon_array, example_index, dof_index, ibi, &fem[ibi]) - target_R) * dR_dE;
    

    // Backpropagate through layers
    for (int i = 0; i < OUTPUT_UNITS; ++i) {
        biases[hidden_layers][i] -= LEARNING_RATE * output_delta;
        for (int j = 0; j < hidden_units[hidden_layers-1]; ++j) {
            weights[hidden_layers][j][i] -= LEARNING_RATE * hidden[hidden_layers-1][j] * output_delta;
        }
    }
    
    for (int l = hidden_layers - 1; l >= 0; --l) {
    
        for (int i = 0; i < hidden_units[l]; ++i) {
            float delta_sum = 0;
            for (int j = 0; j < hidden_units[l+1]; ++j) {
                //printf("Check weights[l+1][i][j] %f \n", weights[l+1][i][j]);                
                delta_sum += weights[l+1][i][j] * deltas[l+1][j];
            }
            
            deltas[l][i] = delta_sum * activation(hidden[l][i]) * (1 - activation(hidden[l][i]));
            biases[l][i] -= LEARNING_RATE * deltas[l][i];
            for (int j = 0; j < (l == 0 ? INPUT_DIM : hidden_units[l-1]); ++j) {
                weights[l][j][i] -= LEARNING_RATE * (l == 0 ? input[j] : hidden[l-1][j]) * deltas[l][i];
            }
        }
    }
    for (int l = 0; l < hidden_layers; ++l) {
        free(hidden[l]);
        free(deltas[l]);
    }
}

void free_network() {
    for (int i = 0; i < INPUT_DIM; ++i) {
        free(weights[0][i]);
    }
    free(weights[0]);
    free(biases[0]);

    for (int l = 1; l < hidden_layers; ++l) {
        for (int i = 0; i < hidden_units[l-1]; ++i) {
            free(weights[l][i]);
        }
        free(weights[l]);
        free(biases[l]);
    }

    for (int i = 0; i < hidden_units[hidden_layers-1]; ++i) {
        free(weights[hidden_layers][i]);
    }
    free(weights[hidden_layers]);
    free(biases[hidden_layers]);
}


