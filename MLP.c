#include "MLP.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

// Funzione di creazione dell'MLP
MLP* create_mlp(int input_neurons, int num_layers, int *layer_sizes, int *activations, int loss_function) {
    static MLP mlp;  // dichiarazione statica per evitare malloc
    mlp.num_layers = num_layers;
    mlp.loss_function = loss_function;

    for (int i = 0; i < num_layers; i++) {
        mlp.layers[i].input_size = (i == 0) ? input_neurons : layer_sizes[i - 1];
        mlp.layers[i].output_size = layer_sizes[i];
        mlp.layers[i].activation_function = activations[i];

        // Initialize weights and biases
        for (int j = 0; j < mlp.layers[i].output_size; j++) {
            mlp.layers[i].biases[j] = 0.0f;
            for (int k = 0; k < mlp.layers[i].input_size; k++) {
                mlp.layers[i].weights[j][k] = ((float)rand() / RAND_MAX) - 0.5f; // Random initialization
            }
        }
    }

    return &mlp;
}

// Funzione per selezionare la funzione di attivazione
float activate(int function_id, float x) {
    switch (function_id) {
        case 0: return sigmoid(x);
        case 1: return reLu(x);
        case 2: return leakyReLu(x);
        case 3: return heavySide(x);
        case 4: return linear(x);
        default: return x;
    }
}

// Funzione per selezionare la derivata della funzione di attivazione
float activate_derivative(int function_id, float x) {
    switch (function_id) {
        case 0: return sigmoid_derivative(x);
        case 1: return reLu_derivative(x);
        case 2: return leakyReLu(x);
        case 3: return heavySide(x); // Aggiungi la derivata di heavySide se necessario
        case 4: return linear_derivative();
        default: return 1.0;
    }
}

// Propagazione in avanti
int forward(MLP *mlp, float *input, int input_size) {
    float current_input[MAX_NEURONS];
    float next_input[MAX_NEURONS];

    // Copia l'input iniziale in current_input
    for (int i = 0; i < input_size; i++) {
        current_input[i] = input[i];
    }

    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = &mlp->layers[i];
        for (int j = 0; j < layer->output_size; j++) {
            float sum = layer->biases[j];
            for (int k = 0; k < layer->input_size; k++) {
                sum += layer->weights[j][k] * current_input[k];
            }
            next_input[j] = activate(layer->activation_function, sum);
        }
        // Copia next_input in current_input per il prossimo layer
        for (int j = 0; j < layer->output_size; j++) {
            current_input[j] = next_input[j];
        }
    }

    // Copia l'output finale nel layer di output
    for (int i = 0; i < mlp->layers[mlp->num_layers - 1].output_size; i++) {
        mlp->layers[mlp->num_layers - 1].output[i] = current_input[i];
    }

    return 0;
}

// Calcola l'errore di output
void calculate_output_error(MLP *mlp, float *predicted, float *true_value, int loss_function) {
    Layer *output_layer = &mlp->layers[mlp->num_layers - 1];
    for (int i = 0; i < output_layer->output_size; i++) {
        float error = predicted[i] - true_value[i];
        if (loss_function == 0) { // MEAN_SQUARED_ERROR
            output_layer->errors[i] = error * activate_derivative(output_layer->activation_function, predicted[i]);
        } else if (loss_function == 1) { // BINARY_CROSS_ENTROPY
            output_layer->errors[i] = (predicted[i] - true_value[i]) * activate_derivative(output_layer->activation_function, predicted[i]);
        }
    }
}

// Backpropagation
void backpropagate(MLP *mlp, float *input, float *true_value, float learning_rate) {
    calculate_output_error(mlp, mlp->layers[mlp->num_layers - 1].output, true_value, mlp->loss_function);

    for (int i = mlp->num_layers - 1; i >= 0; i--) {
        Layer *layer = &mlp->layers[i];
        float prev_output[MAX_NEURONS];

        // Copia l'output del layer precedente in prev_output
        if (i == 0) {
            for (int j = 0; j < layer->input_size; j++) {
                prev_output[j] = input[j];
            }
        } else {
            for (int j = 0; j < layer->input_size; j++) {
                prev_output[j] = mlp->layers[i - 1].output[j];
            }
        }

        for (int j = 0; j < layer->output_size; j++) {
            for (int k = 0; k < layer->input_size; k++) {
                layer->weights[j][k] -= learning_rate * layer->errors[j] * prev_output[k];
            }
            layer->biases[j] -= learning_rate * layer->errors[j];
        }

        if (i > 0) {
            Layer *prev_layer = &mlp->layers[i - 1];
            for (int j = 0; j < prev_layer->output_size; j++) {
                float error_sum = 0.0f;
                for (int k = 0; k < layer->output_size; k++) {
                    error_sum += layer->weights[k][j] * layer->errors[k];
                }
                prev_layer->errors[j] = error_sum * activate_derivative(prev_layer->activation_function, prev_layer->output[j]);
            }
        }
    }
}

// Funzione di addestramento
void train(MLP *mlp, float features[MAX_SAMPLES][MAX_FEATURES], float labels[MAX_SAMPLES], int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < MAX_SAMPLES; i++) {
            forward(mlp, features[i], MAX_FEATURES);
            backpropagate(mlp, features[i], &labels[i], learning_rate);
        }
    }
}