#ifndef MLP_H
#define MLP_H

#define MAX_LAYERS 5               // Numero massimo di layer
#define MAX_NEURONS 128            // Numero massimo di neuroni per layer
#define MAX_SAMPLES 1000           // Numero massimo di campioni
#define MAX_FEATURES 4             // Numero massimo di features per campione
#define LEARNING_RATE 0.01

#include "utils.h"

/*------------------------ Strutture di dati ------------------------*/

typedef struct {
    int input_size;              // Numero di neuroni in ingresso al layer
    int output_size;             // Numero di neuroni in uscita dal layer
    float weights[MAX_NEURONS][MAX_NEURONS];  // Pesi del layer (max neuroni)
    float biases[MAX_NEURONS];   // Bias del layer (max neuroni)
    float output[MAX_NEURONS];   // Output del layer (max neuroni)
    float errors[MAX_NEURONS];   // Errori del layer (max neuroni)
    int activation_function;     // Identificatore della funzione di attivazione
} Layer;

typedef struct {
    int num_layers;              // Numero di layer nel MLP
    Layer layers[MAX_LAYERS];    // Array dei layer
    int loss_function;           // Funzione di perdita (usiamo int invece di enum)
} MLP;

/*------------------------ Funzioni per la gestione del MLP ------------------------*/

MLP* create_mlp(int input_neurons, int num_layers, int *number_of_neurons, int *activations, int loss_function);

void free_mlp(MLP *mlp);

int forward(MLP *mlp, float *input, int input_size);

void calculate_output_error(MLP *mlp, float *predicted, float *true_value, int loss_function);

void backpropagate(MLP *mlp, float *input, float *true_value, float learning_rate);

void train(MLP *mlp, float features[MAX_SAMPLES][MAX_FEATURES], float labels[MAX_SAMPLES], int epochs, float learning_rate);

void print_weights(MLP *mlp);

void print_layer_output(MLP *mlp, int layer_index);

#endif // MLP_H